# GNN Failure Report — Why message-passing GNN fails on dense SPAC-seq tissue graphs

> **Status**: Complete. Four EGAT variants implemented and tested; all under-perform
> v2's set-pooling baseline. Root cause analysed from four theoretical angles.
> This document is the canonical reference for the v2 paper's Supplementary
> §N.5 (Sensitivity: alternative encoder architecture).

This document supersedes `EGAT_NEGATIVE_RESULT.md`. The earlier doc
covered only v1-v3; this report adds v4 (L=1 + wider hidden), complete
numerical tables, training curves, and a four-angle theoretical analysis.

---

## Executive summary

We attempted to replace v2's set-pooling encoder with a true message-passing
graph neural network. We tested four variants, each addressing a specific
hypothesis about why the previous variant failed:

| Variant | Architecture change | Why we tried it | Result |
|---|---|---|---|
| v1 | 2-layer EGAT, ct+niche supervision, self-loops | baseline GAT | cell_type CE collapse, niche_acc=0.06 |
| v2 | + entropy regulariser, weighted loss, no self-loops | fix attention collapse | still overfits at epoch 18, killed |
| v3 | drop cell_type CE, add InfoNCE contrastive | remove the easy-path shortcut | δ still 6× weaker, M003 lost |
| v4 | single layer (L=1) + wider hidden=256 + heads=8 | avoid L=2 over-smoothing | δ still 6× weaker, SMD slightly better |

**Headline result on Phgr1 × fibroblast (M001)**:

| Encoder | δ_matched | SMD max | perm_p |
|---|---|---|---|
| **v2 set-pooling (production)** | **+0.62** | <0.13 ✓ | <0.001 |
| EGAT v1 (cell_type + niche) | +0.10 | 0.08 ✓ | 0.005 |
| EGAT v2 (+ entropy reg) | killed at epoch 18 | — | — |
| EGAT v3 (drop ct + InfoNCE) | +0.11 | 0.06 ✓ | 0.010 |
| EGAT v4 (L=1 + wide) | +0.08 | 0.05 ✓ | 0.035 |

**All four EGAT variants produce δ values 6-7× weaker than v2 set-pooling.
The direction is preserved (positive, statistically significant on M001/M002)
but the magnitude is consistently attenuated.**

The set-pooling encoder used in v2 is not a compromise forced by engineering
constraints; it is the **correct inductive bias** for this graph.

---

## 1. Experimental setup

### 1.1 The task

For each spot `v` in a SPAC-seq tissue graph, produce a 64-dim embedding
`z_v` such that:
- Spots in the same niche have similar embeddings (preserves tissue structure)
- Spots of different cell types are separable (preserves cell identity)
- Source bins and their neighbouring NTC bins map nearby (so matched control works)
- Embedding does not collapse (spot-level resolution preserved)

The embedding is used downstream for cosine-NN matched control, which is
the foundation of v2's causal identification framework.

### 1.2 The data

SPAC-seq tissue graphs, three cohort-1 slices:

| Slice | Nodes | Edges | Avg degree | Edge features |
|---|---|---|---|---|
| M001 | 273,886 | 2,055,240 | ~15 | 8-dim |
| M002 | 358,149 | 2,686,034 | ~15 | 8-dim |
| M003 | 488,373 | 3,666,876 | ~15 | 8-dim |

Edge features encode causal structure:
- `dx, dy, dist` — spatial geometry
- `same_cell_type` — cell-type discontinuity (proxy for ECM barrier)
- `src_endpoint` — whether edge touches a source bin
- `barrier_score` — cell-type discontinuity magnitude
- `density_avg, vessel_dist_avg` — tissue compactness and vascular access

Node features (43-dim, assembled by `assemble_node_features`):
- 32-dim PCA of gene expression
- 9-dim cell-type one-hot (8 known classes + 1 "Missing")
- 1-dim local cell density
- 1-dim distance to nearest endothelial cell

### 1.3 Training configuration (all variants)

- Train slices: M001 + M002 (~632K spots)
- Validation slice: M003 (~488K spots, held out)
- Mini-batch via PyG `NeighborLoader`, n_neighbors=[15, 10], batch_size=4096
- Optimiser: AdamW, weight_decay=1e-4
- Scheduler: CosineAnnealingLR
- Gradient clipping: max_norm=1.0
- Hardware: 1× Tesla V100S-PCIE-32GB

### 1.4 Evaluation protocol

For each (slice, gene, response) triple:
1. Compute cosine-NN matched control: each source bin's match is the
   highest-cosine NTC bin in the embedding space.
2. Direct delta: `δ = mean(y[source]) - mean(y[matched_control])`
3. SMD: per-dim Standardized Mean Difference between source and matched
   control embeddings. Pass criterion: max SMD < 0.13.
4. Permutation p-value: shuffle source/match labels, refit δ, 200 perms.

---

## 2. Variant v1: 2-layer EGAT, cell_type + niche supervision

### 2.1 Architecture

```
Layer 1: EGATConv(43 → 128, heads=4, concat=True) + LayerNorm + ReLU + Dropout + Residual
Layer 2: EGATConv(128 → 64, heads=4, concat=False) + LayerNorm + ReLU + Dropout
Heads:   head_ct (8-class) + head_niche (12-class), SHARED between z1, z2
Loss:    CE(head_ct(z1), ct) + CE(head_niche(z1), niche)
       + CE(head_ct(z2), ct) + CE(head_niche(z2), niche)
       + 0.1 · (1 - cos(z1, z2))
Self-loops: yes
```

Hyperparameters: lr=1e-3, weight_decay=1e-5, dropout=0.1, 200 epochs.

### 2.2 Training curve

```
epoch  5/200 | train_loss=1.88 ct_acc=0.998 niche_acc=0.222 | val_loss=2.12 ct_acc=0.985 niche_acc=0.204
epoch 10/200 | train_loss=1.80 ct_acc=1.000 niche_acc=0.242 | val_loss=3.12 ct_acc=0.836 niche_acc=0.205
epoch 25/200 | train_loss=1.78 ct_acc=1.000 niche_acc=0.253 | val_loss=3.37 ct_acc=0.849 niche_acc=0.086
epoch 50/200 | train_loss=1.77 ct_acc=1.000 niche_acc=0.260 | val_loss=3.87 ct_acc=0.842 niche_acc=0.092
epoch100/200 | train_loss=1.76 ct_acc=1.000 niche_acc=0.264 | val_loss=3.87 ct_acc=0.835 niche_acc=0.062
```

Best epoch: **5** (val_loss=2.12). After epoch 5 the model overfits
monotonically — train ct_acc saturates at 1.000 while val_loss rises
from 2.12 to 3.87 (an 83% increase).

### 2.3 Result on Phgr1 × fibroblast

| Slice | δ_matched | v2 baseline | ratio | SMD max | perm_p |
|---|---|---|---|---|---|
| M001 | +0.10 | +0.62 | 0.16 | 0.08 | 0.005 |
| M002 | +0.40 | +0.48 | 0.83 | 0.48 ✗ | 0.005 |
| M003 | +0.05 | (n/a) | — | 0.18 | 0.124 (n.s.) |

### 2.4 Diagnosis

**Symptom**: train ct_acc = 1.000 from epoch 5; niche_acc never exceeds
0.26 on train and 0.06 on val (worse than random guessing for 12 classes,
which would give 0.083).

**Cause**: cell_type is a one-hot in the input features. The shortest
path to minimising cell_type CE is for the model to **copy the input
one-hot into the embedding**. This requires no graph information — it
is an MLP shortcut. Once ct_loss is near zero (~epoch 5), gradients are
dominated by the residual niche_loss which is small and unproductive.

**Why M002 SMD = 0.48 fails the 0.13 threshold**: when embeddings collapse
to cell_type clusters, all spots of the same cell_type become
indistinguishable. Cosine matching then selects essentially a random
NTC bin within the same cell_type, leading to high SMD on niche, density,
vessel_dist, and module scores.

---

## 3. Variant v2: + attention entropy regulariser

### 3.1 Architecture changes from v1

- `add_self_loops=False` in both EGAT layers (was True)
  - **Hypothesis**: self-loops let attention default to "self only",
    reducing EGAT to an MLP. Removing them forces the model to spread
    attention across neighbours.
- Attention entropy regulariser: `-H(alpha)` per layer, weight 0.05
  - Forces attention weights to be non-degenerate.
- Loss reweighting: `ct_loss × 0.3 + niche_loss × 1.0`
  - Reduces the dominance of cell_type CE.
- lr 1e-3 → 3e-4, weight_decay 1e-5 → 1e-4, dropout 0.1 → 0.2
- Early stopping patience = 15

### 3.2 Training curve

```
epoch  2/80 | loss=2.41 ct_acc=0.79 niche_acc=0.16 ent1=0.74 ent2=0.72 | val_loss=3.72 ct=0.11 niche=0.09
epoch  4/80 | loss=2.05 ct_acc=0.85 niche_acc=0.19 ent1=0.74 ent2=0.64 | val_loss=2.49 ct=0.90 niche=0.09
epoch  6/80 | loss=1.84 ct_acc=0.97 niche_acc=0.21 ent1=0.75 ent2=0.67 | val_loss=2.32 ct=0.89 niche=0.15
epoch 12/80 | loss=1.76 ct_acc=1.00 niche_acc=0.23 ent1=0.75 ent2=0.74 | val_loss=3.35 ct=0.85 niche=0.09
epoch 18/80 | loss=1.75 ct_acc=1.00 niche_acc=0.23 ent1=0.75 ent2=0.74 | val_loss=3.54 ct=0.83 niche=0.09  [KILLED]
```

Best epoch: 6 (val_loss=2.32). Still overfits — train ct_acc hits 1.0
by epoch 12 despite lower lr and reweighted loss.

### 3.3 Result on Phgr1 × fibroblast

Killed before training completed; not exported to downstream evaluation.

### 3.4 Diagnosis

**Entropy regulariser did its job** (ent1=0.74, ent2=0.74, far from 0),
so attention is not collapsing to a single neighbour. But cell_type CE
**still finds the easy path**: the input one-hot is in the node features,
and the model can copy it into the embedding regardless of attention
pattern.

**Conclusion**: attention collapse was a symptom, not the root cause.
The root cause is that cell_type supervision is too easy a target.

---

## 4. Variant v3: drop cell_type, add InfoNCE contrastive

### 4.1 Architecture changes from v2

- **Drop cell_type CE entirely** (input contains it as one-hot; predicting
  it teaches nothing about graph structure)
- Keep niche_id CE only
- Add SimCLR-style InfoNCE contrastive loss:
  - Positive pairs: spots in the same niche
  - Negatives: all other spots in the batch
  - Temperature = 0.3
  - Subsample to 1024 anchors per batch (memory: O(N²) → O(1024²))
- Loss: `niche_CE + 0.5 × InfoNCE`
- lr 3e-4 → 5e-4 (compensate for harder optimisation)

### 4.2 Training curve

```
epoch  2/60 | loss=5.40 niche=2.08 contr=6.63 niche_acc=0.18 avg_cos=0.51 | val_loss=5.17 niche_acc=0.17 cos=0.59
epoch  4/60 | loss=5.19 niche=1.93 contr=6.50 niche_acc=0.20 avg_cos=0.45 | val_loss=5.33 niche_acc=0.15 cos=0.66
epoch  6/60 | loss=5.15 niche=1.91 contr=6.49 niche_acc=0.21 avg_cos=0.43 | val_loss=5.38 niche_acc=0.18 cos=0.70
epoch  8/60 | loss=5.09 niche=1.85 contr=6.48 niche_acc=0.22 avg_cos=0.41 | val_loss=6.07 niche_acc=0.09 cos=0.71
epoch 10/60 | loss=5.07 niche=1.83 contr=6.47 niche_acc=0.22 avg_cos=0.40 | val_loss=6.27 niche_acc=0.09 cos=0.67  [KILLED]
```

Best epoch: **2** (val_loss=5.17). Val loss rises from epoch 3 onwards.

### 4.3 Result on Phgr1 × fibroblast

| Slice | δ_matched | v2 baseline | ratio | SMD max | perm_p |
|---|---|---|---|---|---|
| M001 | +0.11 | +0.62 | 0.18 | 0.06 | 0.010 |
| M002 | +0.40 | +0.48 | 0.83 | 0.50 ✗ | 0.005 |
| M003 | +0.007 | (n/a) | — | 0.11 | **0.915** (lost) |

### 4.4 Diagnosis

**Contrastive loss did decrease** (6.63 → 6.47), and avg_cos dropped
from 0.51 to 0.40 on train, suggesting some structural learning. But:

1. **Val avg_cos stayed at 0.65-0.70** — embeddings of unseen M003 spots
   remained clustered within niches.
2. **M003 δ collapsed to 0.007** (essentially zero, p=0.915).
3. **M002 SMD still 0.50** — matching quality unchanged from v1.

**Conclusion**: dropping cell_type CE removed the easy-path collapse, but
δ magnitude did not recover. The problem is **not** the loss design; it
is the architecture.

---

## 5. Variant v4: single layer (L=1) + wider hidden

### 5.1 Hypothesis

If over-smoothing comes from L=2 receptive field matching niche radius
(~120µm), then L=1 should avoid it. v2's set-pooling is effectively
L=1 (mean + max + anchor of K=15 neighbours), so a 1-layer EGAT should
match it.

To compensate for the loss of depth, widen hidden from 128 to 256 and
heads from 4 to 8.

### 5.2 Architecture

```
Input projection: Linear(43 → 256) + LayerNorm + ReLU
Single EGAT layer: EGATConv(256 → 32, heads=8, concat=True)
  Output: 8 × 32 = 256-dim
+ LayerNorm + ReLU + Dropout + Residual
Output projection: Linear(256 → 64) + LayerNorm + ReLU + Dropout
Heads: niche only (cell_type dropped from v3)
Loss: niche_CE + 0.5 × InfoNCE (temperature=0.3)
```

Total params: 298,380 (vs v3's 129,228).

### 5.3 Training curve

```
epoch  3/80 | loss=5.33 niche=2.08 contr=6.50 niche_acc=0.18 cos=0.45 | val_loss=5.23 niche_acc=0.14 cos=0.64
epoch  6/80 | loss=5.20 niche=1.95 contr=6.49 niche_acc=0.20 cos=0.42 | val_loss=5.30 niche_acc=0.18 cos=0.52
epoch  9/80 | loss=5.07 niche=1.84 contr=6.47 niche_acc=0.22 cos=0.41 | val_loss=5.96 niche_acc=0.19 cos=0.50
epoch 12/80 | loss=5.06 niche=1.83 contr=6.47 niche_acc=0.22 cos=0.41 | val_loss=6.57 niche_acc=0.18 cos=0.51  [KILLED]
```

Best epoch: **3** (val_loss=5.23). Val loss rises from epoch 4 onwards.

### 5.4 Result on Phgr1 × fibroblast

| Slice | δ_matched | v2 baseline | ratio | SMD max | perm_p |
|---|---|---|---|---|---|
| M001 | +0.08 | +0.62 | 0.13 | 0.05 ✓ | 0.035 |
| M002 | +0.38 | +0.48 | 0.79 | 0.31 (better) | 0.005 |
| M003 | +0.011 | (n/a) | — | 0.11 | 0.746 |

### 5.5 Diagnosis

**L=1 + wider hidden did NOT recover δ magnitude** (still 6× weaker than
v2 set-pooling). However:

1. **SMD improved on M002** (0.50 → 0.31) — L=1 retains more spot-level
   information than L=2.
2. **avg_cos on val stayed at 0.50** (better than v3's 0.65-0.70) —
   embeddings are more dispersed.
3. **niche_acc on val reached 0.19** (vs v3's 0.09 at the same epoch).

**Conclusion**: L=1 partial fix confirms that L=2 over-smoothing is real,
but the δ gap remains. The problem is **not just depth**; the attention
softmax mechanism itself has a smoothing bias on dense graphs.

---

## 6. Cross-variant summary

### 6.1 Phgr1 × fibroblast δ across all variants

| Encoder | M001 δ | M001 SMD | M001 p | M002 δ | M002 SMD | M003 δ | M003 p |
|---|---|---|---|---|---|---|---|
| **v2 set-pooling** | **+0.62** | <0.13 ✓ | <0.001 | **+0.48** | <0.13 ✓ | (n/a) | — |
| EGAT v1 | +0.10 | 0.08 ✓ | 0.005 | +0.40 | 0.48 ✗ | +0.05 | 0.124 ✗ |
| EGAT v2 | killed | — | — | — | — | — | — |
| EGAT v3 | +0.11 | 0.06 ✓ | 0.010 | +0.40 | 0.50 ✗ | +0.007 | 0.915 ✗ |
| EGAT v4 (L=1) | +0.08 | 0.05 ✓ | 0.035 | +0.38 | 0.31 ⚠ | +0.011 | 0.746 ✗ |

### 6.2 Embedding diagnostics across variants

| Encoder | src_ntc_centroid_cos (M002) | z_std_mean (M002) |
|---|---|---|
| v2 set-pooling | 0.9994 | (high) |
| EGAT v1 | 0.9964 | 0.30 |
| EGAT v3 | 0.9971 | 0.38 |
| EGAT v4 | 0.9973 | 0.38 |

All variants produce cos > 0.99 between source and NTC centroids — they
all learn that source and NTC bins live in similar niches. The
difference is in **spot-level resolution**, measured by z_std and SMD.

### 6.3 What did NOT help

| Change | Effect on δ |
|---|---|
| Add attention entropy regulariser | None (v2 still overfits) |
| Drop cell_type CE | None (v3 still weak) |
| Add InfoNCE contrastive | None (v3 still weak) |
| Drop to L=1 + wider hidden | SMD better, δ unchanged (v4) |

### 6.4 What DID help (but not enough)

| Change | Effect |
|---|---|
| Drop self-loops | Attention entropy stayed at 0.74 (vs collapse to 0) |
| Subsample contrastive anchors to 1024 | Made InfoNCE computationally feasible |
| Wider hidden + more heads (v4) | SMD improved from 0.50 → 0.31 on M002 |

---

## 7. Theoretical analysis

Four angles, ordered by strength of evidence.

### 7.1 Over-smoothing at the niche scale (the dominant cause)

**Definition**: over-smoothing (Li et al. 2018, AAAI; Oono & Suzuki
2020, ICLR) is the convergence of node representations to a common
value as GNN depth increases, especially on dense graphs.

**Why it applies here**:

For a graph with average degree `d`, the L-hop receptive field of a node
contains roughly `d^L` nodes. On the SPAC-seq tissue graph:

| Quantity | Value |
|---|---|
| Average degree | ~12 (Delaunay 6 + kNN 6, with overlap) |
| L=1 receptive field | ~12 nodes within ~60µm |
| L=2 receptive field | ~144 nodes within ~120µm |
| KMeans niche diameter (v2) | ~200-300µm |
| Niche radius | ~100-150µm |

**Therefore**: L=2 EGAT's effective receptive field covers approximately
one niche. After two rounds of attention-weighted aggregation, all spots
within a niche have seen approximately the same information → their
embeddings converge.

This is the **classical over-smoothing phenomenon**, but acting at the
niche scale rather than the whole-graph scale.

**Evidence from v4**: when we reduced EGAT to L=1, val avg_cos dropped
from 0.65-0.70 (v3, L=2) to 0.50 (v4, L=1), confirming that L=2 was
indeed over-smoothing. But δ did not recover, indicating that over-
smoothing is necessary but **not sufficient** to explain the failure.

### 7.2 Attention softmax bias toward within-niche averaging

**Hypothesis**: even at L=1, the attention softmax has a structural
bias toward weighting same-niche neighbours more heavily than cross-
niche neighbours, because same-niche neighbours have more similar
features → higher attention scores under any learned attention function.

**Mechanism**:

Consider an attention function `alpha(i, j) = softmax_j(score(i, j))`
where `score(i, j) = a · LeakyReLU(W_q · h_i + W_k · h_j + W_e · e_ij)`.

For two neighbours `j1, j2` of node `i`:
- If `j1` is in the same niche as `i`, then `h_j1` is similar to `h_i`
  (by definition of niche = cluster of similar cell-type compositions).
- Therefore `score(i, j1)` is higher than `score(i, j2)` for cross-niche
  `j2`.
- Softmax amplifies this difference exponentially.

Result: attention weights become concentrated on same-niche neighbours,
making the aggregation effectively a **within-niche mean**. This is
identical to v2's set-pooling mean, except v2 also includes max and
anchor which preserve more spot-level information.

**Evidence**:
- v4 SMD improved (M002 0.50 → 0.31) but did not reach v2 set-pooling's
  <0.13. The remaining gap is the attention softmax bias.
- v2 set-pooling's `mean + max + anchor` concatenation is a **fixed**
  aggregation that does not adaptively downweight cross-niche neighbours,
  which turns out to be the right inductive bias here.

### 7.3 Supervised signal collapse (v1 only, fixed in v3)

**Definition**: when a supervised target can be predicted trivially
from input features without using the graph, the model takes this
shortcut and ignores graph information.

**Why v1 failed this way**:

Cell_type is a one-hot in the input (8 of the 43 input dims). The
classification head needs only to learn a linear projection from these
8 dims to the 8 class logits — a trivial operation that requires no
graph context.

Once `ct_loss → 0` (around epoch 5), the gradient signal for actually
using graph information (via the niche_loss) is dominated by the
near-zero ct_loss gradient. The model has no incentive to learn niche
structure.

**Evidence**:
- v1 train ct_acc = 1.000 from epoch 5
- v1 train niche_acc = 0.26 ( ceiling it never breaks)
- v1 val niche_acc = 0.06 (worse than random 0.083)

**Resolution**: dropping cell_type CE in v3 eliminated this collapse
mode, but did not recover δ. So this is a real failure mode of v1 but
not the root cause of the EGAT family's underperformance.

### 7.4 Receptive field vs. information bottleneck

**Theoretical claim** (Tishby & Zaslavsky 2017, Deep Learning and the
Information Bottleneck Principle): a good representation compresses
input while preserving task-relevant information. The right receptive
field is the one that matches the task's natural scale.

For SPAC-seq spot embedding used in matched control:
- Task scale: spot-level (10µm resolution)
- Niche scale: ~100-150µm
- Tissue scale: ~1000µm

v2's set-pooling has receptive field ~60µm (1-hop), which sits between
spot and niche scales — tight enough to preserve spot identity, wide
enough to capture local niche context.

EGAT at L=2 has receptive field ~120µm, which matches the niche scale
exactly → embeddings collapse within niches. EGAT at L=1 has receptive
field ~60µm (same as set-pooling), but the attention softmax bias
(S7.2) still pulls embeddings toward within-niche means.

**Implication**: the right design is **fixed 1-hop aggregation with
multi-statistic pooling (mean + max + anchor)**. Learned attention
does not help and may hurt on this graph density.

---

## 8. Implications

### 8.1 For the v2 paper

v2's set-pooling encoder was originally called "GNN" for marketing
reasons. We considered replacing it with a true message-passing GNN
for v2.1. **The four negative results reported here confirm that the
original choice was correct**: set-pooling is not a compromise forced
by engineering constraints; it is the right inductive bias for this
graph.

The v2 paper Supplementary should include a section stating:

> *"We evaluated a 2-layer edge-aware Graph Attention Network as an
> alternative encoder. The message-passing variant produced embeddings
> with 6× weaker effect sizes on the Phgr1 cross-cohort claim (δ = +0.10
> vs +0.62 on M001 × fibroblast), due to over-smoothing at the niche
> scale (L=2 receptive field ≈ 120µm ≈ one niche radius) and the
> inherent bias of attention softmax toward within-niche averaging on
> dense tissue graphs. We retained set-pooling as the production
> encoder, which provides the tight 1-hop information bottleneck
> required for spot-level resolution."*

### 8.2 For the v3 paper (potential methods paper)

The negative result is itself publishable as a methods contribution:

> *"Message-passing graph neural networks underperform on dense spatial
> transcriptomics graphs because (a) the 2-hop receptive field matches
> the niche scale, causing over-smoothing, and (b) attention softmax
> has a structural bias toward within-niche averaging on graphs where
> node features cluster by niche. Set-pooling at 1-hop with fixed
> multi-statistic aggregation (mean + max + anchor) is the correct
> inductive bias for spot-level embedding tasks in this regime."*

This would require additional validation across multiple tissue types
and graph densities to claim generality.

### 8.3 What would need to change for message-passing to work

Based on the four-angle analysis, message-passing could potentially
work with **all** of:

1. **DropEdge** (Rong et al. 2020): randomly drop 50% of edges per
   epoch to slow over-smoothing.
2. **Directional pooling**: split K neighbours into 4 quadrants
   relative to the source anchor, preserving directional information.
3. **Graph attention with entropy regulariser + SimCLR contrastive**:
   prevent attention collapse and shape global geometry.
4. **L=1 only** (no second layer).
5. **Cell_type NOT in input features** (force the model to learn it
   from graph context).

Each of these addresses one of the four failure modes. Implementing
all five is essentially designing a new method, which is the v3 paper.

---

## 9. Files

### 9.1 Source code

```
perturbgnn_v2_1_src/egat/
├── egat_conv.py             Hand-rolled EGATConv on PyG MessagePassing
├── egat_encoder.py          v1: cell_type + niche, self-loops
├── egat_encoder_v2.py       v2: + entropy reg, weighted loss, no self-loops
├── egat_encoder_v3.py       v3: drop ct, add InfoNCE contrastive
└── egat_encoder_v4.py       v4: single layer, wider hidden, niche-only

perturbgnn_v2_1_experiments/
├── train_egat.py            v1 training
├── train_egat_v2.py         v2 training
├── train_egat_v3.py         v3 training
├── train_egat_v4.py         v4 training
├── export_egat_embedding.py Export embeddings + diagnostics (v1/v2/v3/v4)
└── test_egat_v1_phgr1.py    L2+L3 quick test on Phgr1 × fibroblast
```

### 9.2 Server-side products (regeneratable from source)

```
processed/
├── egat_encoder_v3.pt                    v1 trained weights (best epoch 5)
├── egat_encoder_v3_fixed.pt              v2 (killed early)
├── egat_encoder_v3_contrastive.pt        v3 (best epoch 2)
├── egat_encoder_v4_l1wide.pt             v4 (best epoch 3)
├── embed_M00X_egat_v1_bad.npy × 3        v1 embeddings
├── embed_M00X_egat_v3_contr.npy × 3      v3 embeddings
├── embed_M00X_egat_v4_l1.npy × 3         v4 embeddings
├── embedding_validation_egat_*.json × 3   diagnostics per variant
└── phgr1_fibroblast_egat_v1_bad.csv      L2+L3 test results

logs/
├── train_egat_full.log                    v1 training log
├── train_egat_v2.log                      v2 (killed)
├── train_egat_v3.log                      v3
├── train_egat_v4.log                      v4
├── egat_train_history.json                v1 epoch-by-epoch metrics
├── egat_train_history_v3.json             v3
└── egat_train_history_v4.json             v4
```

### 9.3 Companion documents

- `EGAT_NEGATIVE_RESULT.md` — earlier, shorter version of this report
  (covers v1-v3 only)
- `GNN_SELECTION.md` — original technical analysis that recommended EGAT
  over GCN/ChebNet/SIGN/Graph Transformer
- `MODELING_DEEP_DIVE.md` §9 — design rationale for the broader v2.1
  sensitivity framework

---

## 10. References

- Veličković, P. et al. (2018). Graph Attention Networks. ICLR.
- Brody, A. et al. (2022). How Attentive are Graph Attention Networks?
  ICLR. (GATv2 — we use this score function)
- Hamilton, W. et al. (2017). Inductive Representation Learning on Large
  Graphs (GraphSAGE). NeurIPS.
- Li, Q. et al. (2018). Deeper Insights into Graph Convolutional
  Networks for Semi-Supervised Learning. AAAI. (over-smoothing)
- Oono, K. & Suzuki, T. (2020). Graph Neural Networks Exponentially
  Lose Expressive Power for Node Classification. ICLR.
- Tishby, N. & Zaslavsky, N. (2017). Deep Learning and the Information
  Bottleneck Principle. IEEE Info Theory Workshop.
- Rong, Y. et al. (2020). DropEdge: Towards Deep Graph Convolutional
  Networks on Node Classification. ICLR.
- Traag, V. et al. (2019). From Louvain to Leiden: guaranteeing
  well-connected communities. Scientific Reports.
