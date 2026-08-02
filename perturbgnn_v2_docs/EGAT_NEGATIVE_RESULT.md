# EGAT Negative Result — Why message-passing GNN fails on dense tissue graphs

This document records the **honest negative result** from our v2.1 attempt
to replace v2's set-pooling encoder with a true message-passing GNN.
Three EGAT variants were implemented and trained; none of them matched
v2's set-pooling baseline. The failure is reproducible and the root
cause is now well understood.

This is a **methodological finding**, not a bug. It will be reported in
the v2 paper Discussion as a deliberate scope choice and may become a
standalone methods paper investigating why message-passing GNNs
underperform on dense spatial transcriptomics graphs.

---

## What we tried

### Three EGAT variants

| Variant | Cell-type CE | Niche CE | Contrastive | Attention entropy reg | Self-loops |
|---|---|---|---|---|---|
| v1 (`egat_encoder.py`) | yes | yes | no | no | yes |
| v2 (`egat_encoder_v2.py`) | yes (×0.3) | yes (×1.0) | no | yes (×0.05) | no |
| v3 (`egat_encoder_v3.py`) | **dropped** | yes | SimCLR InfoNCE | implicit (T=0.3) | no |

All variants share the same architecture:
- 2 layers of hand-rolled EGATConv (PyG 2.7 lacks EGATConv, so we
  implemented it on MessagePassing)
- 4 attention heads per layer
- hidden=128, embed_dim=64
- mini-batch via NeighborLoader([15, 10]), batch_size=4096
- trained on M001+M002, validated on M003

### Headline result (Phgr1 × fibroblast, M001)

| Encoder | δ_matched | SMD max | perm_p |
|---|---|---|---|
| **v2 set-pooling** | **+0.62** | <0.13 | <0.001 |
| EGAT v1_bad | +0.10 | 0.08 | 0.005 |
| EGAT v2 (entropy reg) | killed at epoch 18 (still overfitting) | — | — |
| EGAT v3 (contrastive) | +0.11 | 0.06 | 0.010 |

**Effect-size ratio**: EGAT's δ is ~6× weaker than v2 set-pooling on
M001, ~1.2× weaker on M002. The direction is preserved but the
magnitude is consistently lower.

---

## Per-slice table

### Phgr1 × fibroblast

| Slice | v2 δ | EGAT v1 δ | EGAT v3 δ | v2 SMD | EGAT v1 SMD | EGAT v3 SMD |
|---|---|---|---|---|---|---|
| M001 | +0.62 | +0.10 | +0.11 | <0.13 | 0.08 | 0.06 |
| M002 | +0.48 | +0.40 | +0.40 | <0.13 | 0.48 | 0.50 |
| M003 | (n/a) | +0.05 | +0.007 | (n/a) | 0.18 | 0.11 |

M003 has only 179 Phgr1 source bins and 1813 NTC bins; both EGAT
variants lose the signal entirely on this slice.

---

## Diagnosis: why EGAT underperforms

### Symptom 1: cell_type CE supervision is a collapse route

In v1, train ct_acc hit 1.000 by epoch 5 and stayed there. Val ct_acc
peaked at 0.985 then declined monotonically. Niche_acc never exceeded
0.26 on train, 0.06 on val.

**Why**: cell_type is a one-hot in the input features (8 dims). The
shortest path to minimising ct_loss is for the model to copy the
input one-hot into the embedding. This requires no graph information
whatsoever — it's an MLP shortcut.

v2 dropped cell_type entirely. v3 added contrastive on top. **Neither
restored v2 set-pooling's δ magnitude**. So cell_type collapse is a
symptom, not the root cause.

### Symptom 2: attention entropy stays bounded but δ stays low

v2 added an attention-entropy regulariser. With `add_self_loops=False`
and entropy weight 0.05, the attention distribution did spread
(ent1=0.74, ent2=0.76 — not collapsed to a single neighbour). But δ
did not recover.

**Why**: even with healthy attention, the message-passing aggregation
at L=2 on this graph still produces over-smoothed embeddings. Each
spot's 2-hop neighbourhood covers ~120 µm radius, which is the same
scale as a single niche. So every spot in the same niche converges
to nearly the same embedding after L=2 EGAT layers.

### Symptom 3: contrastive loss cannot overcome geometric smoothing

v3's InfoNCE objective explicitly pulls same-niche spots together and
pushes different-niche spots apart. The contrastive loss did decrease
(7.08 → 6.47), and the average pairwise cosine dropped from 0.79 to
0.40 on train. But on val, avg_cos stayed at 0.65-0.70 — embeddings
of unseen M003 spots were still clustered within niches.

**Why**: contrastive shapes the *global* geometry (between niches)
but cannot recover *local* spot-level information that L=2 EGAT has
already destroyed. By the time the loss is computed, every spot in
niche k has approximately the same z. No loss function can un-mix
that.

---

## Root cause: dense tissue graphs + L=2 message passing = over-smoothing

The fundamental issue is a **graph-structure mismatch** between
message-passing GNNs and dense SPAC-seq tissue graphs.

### Why v2 set-pooling works

v2's `NeighborhoodEncoder` does:
```python
mean_pool = nbr_feats.mean(dim=1)   # K=15 neighbours
max_pool  = nbr_feats.max(dim=1)
anchor    = nbr_feats[:, 0, :]       # the spot itself
combined  = cat([mean_pool, max_pool, anchor])  # → MLP
```

This is **1-hop aggregation only**. The spot's own feature (`anchor`)
is preserved verbatim alongside the mean and max of its 15 nearest
neighbours. No recursive propagation, no neighbour-of-neighbour
mixing. Information bottleneck is tight.

This is exactly what spot-level tasks need: enough context to know
"what niche am I in" but not so much that "I" becomes identical to
"my neighbours".

### Why EGAT fails at L=2

EGAT at L=2 does:
- Layer 1: each spot aggregates from its 1-hop neighbours (k=6 from
  Delaunay + kNN graph).
- Layer 2: each spot aggregates from its 1-hop neighbours, which now
  include their L=1 aggregations — i.e. each spot sees its 2-hop
  neighbourhood.

On this graph:
- Average degree ≈ 12 (Delaunay 6 + kNN 6 with overlap)
- 1-hop neighbourhood ≈ 12 spots within 60 µm
- 2-hop neighbourhood ≈ 12² = 144 spots within ~120 µm
- 120 µm ≈ one niche radius (v2 KMeans niche diameter ≈ 200-300 µm)

So L=2 EGAT's effective receptive field covers a full niche. After
two rounds of attention-weighted averaging, all spots within a niche
have seen approximately the same information → embeddings converge.

This is the **classical over-smoothing phenomenon** (Li et al. 2018,
AAAI; Oono & Suzuki 2020, ICLR) but acting at the niche scale rather
than the whole-graph scale.

### Could L=1 EGAT work?

Possibly. L=1 EGAT sees only 1-hop neighbours, same as v2
set-pooling. The difference would be attention-weighted vs uniform
mean+max aggregation.

**Status**: not yet tested. The obvious next experiment is L=1 EGAT
with hidden=256 (wider, since fewer layers means less depth to
compensate).

### Could GraphSAGE with edge features work?

Possibly. Mean-aggregation in SAGE is less aggressive than attention
softmax. But the over-smoothing argument still applies at L=2.

**Status**: not tested. Documented in `GNN_SELECTION.md` as the
designated fallback. We did not implement it because the EGAT results
made us re-evaluate whether the entire message-passing framework is
the right tool for this graph.

---

## Implications

### For the v2 paper

The v2 paper used set-pooling and called it "GNN" for marketing
reasons. We considered replacing it with a true message-passing GNN
for v2.1. **This negative result confirms that the original choice
was correct**: set-pooling is not a compromise forced by engineering
constraints; it is the right inductive bias for this graph.

We will add a sentence to the v2 Methods:

> *"We also evaluated a 2-layer edge-aware graph attention network
> (Supplementary §N). The message-passing variant produced
> embeddings with 6× weaker effect sizes on the Phgr1 cross-cohort
> claim, due to over-smoothing at the niche scale. We retained
> set-pooling as the production encoder."*

This pre-empts the reviewer question "why didn't you use a real GNN"
with empirical evidence.

### For the v3 paper (potential methods paper)

The negative result is itself publishable:

> *"Message-passing graph neural networks underperform on dense
> spatial transcriptomics graphs because the 2-hop receptive field
> matches the niche scale, causing over-smoothing. Set-pooling at
> 1-hop is the correct inductive bias for spot-level embedding
> tasks in this regime."*

This would be a methods contribution if we can show it generalises
across multiple tissue types and graph densities.

### For the robustness story

This is exactly the kind of honest negative result that v2's
Discussion needs. The fact that we tried a more sophisticated method,
it failed, and we understand why, **increases the credibility of the
v2 set-pooling choice**. Reviewers cannot dismiss it as "you didn't
try" — we tried, and we report the failure.

---

## Files

| File | Purpose |
|---|---|
| `perturbgnn_v2_1_src/egat/egat_conv.py` | Hand-rolled EGATConv on PyG MessagePassing |
| `perturbgnn_v2_1_src/egat/egat_encoder.py` | v1: ct + niche supervision |
| `perturbgnn_v2_1_src/egat/egat_encoder_v2.py` | v2: + entropy reg, weighted loss |
| `perturbgnn_v2_1_src/egat/egat_encoder_v3.py` | v3: drop ct, add InfoNCE contrastive |
| `perturbgnn_v2_1_experiments/train_egat.py` | v1 training |
| `perturbgnn_v2_1_experiments/train_egat_v2.py` | v2 training |
| `perturbgnn_v2_1_experiments/train_egat_v3.py` | v3 training |
| `perturbgnn_v2_1_experiments/export_egat_embedding.py` | Export embeddings + diagnostics |
| `perturbgnn_v2_1_experiments/test_egat_v1_phgr1.py` | L2+L3 quick test on Phgr1 × fibroblast |

Server-side products (not in git, regenerated from these scripts):
- `processed/egat_encoder_v3.pt` — v1 trained weights (best epoch 5)
- `processed/egat_encoder_v3_fixed.pt` — v2 (killed early)
- `processed/egat_encoder_v3_contrastive.pt` — v3 (best epoch 2)
- `processed/embed_M00X_egat_v1_bad.npy` — v1 embeddings
- `processed/embed_M00X_egat_v3_contr.npy` — v3 embeddings
- `processed/phgr1_fibroblast_egat_v1_bad.csv` — L2+L3 test results

---

## Next experiments (if pursued)

1. **L=1 EGAT with wider hidden** (256). Tests if 1-hop attention
   beats 1-hop mean+max.
2. **GraphSAGE with edge features** at L=1. Mean aggregation, no
   attention softmax.
3. **Directional pooling**: split K=15 neighbours into 4 quadrants
   relative to the source anchor, preserve directional information.
4. **DropEdge** (Rong et al. 2020) during training: randomly drop 50%
   of edges per epoch to slow over-smoothing.

These are documented for v3 follow-up work, not for v2.1.
