# PerturbGNN v2 — Modelling deep dive

This document is the **modelling-layer reference** for v2. It has two parts:

1. **§1-§3 — The twelve modelling problems** reviewers will attack, grouped
   by severity (theoretical → design → implementation). Each problem is
   stated with the specific assumption it violates, the mathematical reason
   v2's claim becomes invalid under that attack, and what v2 currently does
   (or does not) say about it.

2. **§4 — v2.1 fixes**, five sensitivity analyses that quantitatively
   answer each attack. Each fix specifies the estimator, the maths, the
   parameters, and the pass/fail acceptance criteria.

This document is **not** the v2 paper. It is the internal record of "we
know exactly which modelling choices are load-bearing, and here is the
sensitivity evidence for each".

---

# Part I — The twelve modelling problems

## §1 Theoretical-layer problems (hardest to defend)

### P1. Endogeneity of X in the Durbin model

**The v2 model.**
```
y = ρWy + Xβ + WXθ + ε
```
where `y` is the per-spot module score, `X` is the source-vs-control
indicator, `W` is the spatial weights matrix, and `(β, θ, ρ)` are fitted by
maximum likelihood.

**The OLS consistency assumption.** For the MLE of `(β, θ, ρ)` to be
consistent we need `E[Xε] = 0` — X must be **exogenous**. In SPAC-seq:

- Guide delivery is not random. Clones expand radially from a transduced
  progenitor, so `X = 1` is spatially autocorrelated at the clone radius
  (~50-200 µm).
- The injection site itself biases local inflammation and vascular
  permeability — both affect y.
- Therefore `E[Xε] ≠ 0` and `(β̂, θ̂, ρ̂)` are inconsistent.

**What DiD does and does not fix.** Layer 4's DiD subtracts the
pre/post-change difference in NTC bins from the same difference in source
bins. This addresses **parallel-trends** violation (a different confounder
on the time axis), not spatial endogeneity of X.

**The correct fix (deferred to v2.1 Fix 1).** Replace the Durbin-reduced
point estimate τ̂ with a cross-fitted AIPW estimate that explicitly models
the propensity `ê(X) = P(T=1|X)` and the outcome surfaces
`μ̂₁(X), μ̂₀(X)`. Neyman orthogonality protects τ̂ from first-order bias
in either nuisance model.

### P2. Rosenbaum Γ* systematically optimistic under embedding-based matching

**The v2 setup.** Layer 4 reports Rosenbaum Γ* as the "how much hidden
confounding would it take to flip the sign" sensitivity. The matched
controls are chosen by nearest-neighbour in the v2 embedding.

**The circularity.** Γ* bounds assume that, conditional on **observed**
covariates, treatment and control are exchangeable up to a hidden
odds-ratio Γ. v2's "observed covariates" are 64 embedding dimensions. So:

- Γ* measures unobserved confounding **conditional on the embedding**.
- But the embedding is itself a learned summary; it may not include the
  true confounders (chromatin accessibility, single-cell metabolic state).
- The bound is therefore optimistic by an unknown amount.

**The classical reference.** Lechner & Steinmayr (2014, JoE 162)
"Estimating the Sensitivity of Average Treatment Effects to Unobserved
Confounders" document this exact trap in the context of propensity-score
matching with learned representations.

**Mitigation (deferred).** v2.1 Fix 4 reports Γ* under two matching
schemes (embedding NN and raw-marker NN) and bounds the divergence.
True fix would need negative-control outcomes (Lipsitch et al. 2010,
Epidemiology) — not available in v2's data.

### P3. "Cross-cohort replication" lacks a hierarchical statistical framework

**The v2 setup.** Per-cohort p-values are combined via Fisher's method
within cohort 1 and cohort 2, then combined across cohorts via Fisher again.
"Replication" = both cohorts' combined p < 0.05 + direction consistent.

**The hidden assumption.** Fisher combination tests the global null
`H₀: cohort1 = 0 OR cohort2 = 0`. Rejecting this global null does not
quantify **how similar** the two cohorts' effects are. v2 reports
"direction consistent" but not the magnitude of cohort-to-cohort
heterogeneity.

**The proper framework.** A two-level hierarchical Bayesian model:

```
δ_global ~ Normal(μ, τ²)
δ_cohort[c] ~ Normal(δ_global, σ²_cohort[c])
δ_slice[s in c] ~ Normal(δ_cohort[c], σ²_slice[s])
```

`τ²` (between-cohort variance) is the **correct** "replication robustness"
quantity. v2 implicitly assumes `τ² = 0`, which is a modelling choice,
not a data finding.

**Mitigation (deferred).** This is left for the paper revision; v2.1 does
not re-implement the hierarchical model. TheDirection-consistency
criterion is reported as a proxy.

---

## §2 Design-layer problems (architecture-level, arguable)

### P4. Shared classification head makes cos(z₁, z₂) self-fulfilling

**The v2 architecture** (`bt_encoder.py`):

```python
self.head_ct     = ClassificationHead(...)  # shared by z1 and z2
self.head_niche  = ClassificationHead(...)  # shared by z1 and z2
```

Both views pass through the same head, so training drives z₁ and z₂
towards the same classification decision boundary. cos(z₁, z₂) = 0.9997
is therefore partly an **architectural constraint**, not pure learned
alignment.

**The proper probe.** Train `head_ct` and `head_niche` on z₁ only
(80% random split); evaluate zero-shot on z₂ (held-out 20%). Compare
with same-view accuracy (z₁ → z₁ upper bound) and a random-pairing
baseline (z₂ → permuted z₁ lower bound).

**v2.1 Fix 3** implements this probe transfer. Pass criterion:
z₁→z₂ zero-shot accuracy ≥ 70% of z₁→z₁ same-view accuracy.

### P5. Set-pooling loses directional information in the neighbourhood

**The v2 neighbourhood encoder:**

```python
mean_pool = nbr_feats.mean(dim=1)
max_pool  = nbr_feats.max(dim=1).values
anchor    = nbr_feats[:, 0, :]
combined  = cat([mean_pool, max_pool, anchor])
```

This is **set pooling** — it treats the K=15 neighbours as an unordered
set. But SPAC-seq neighbourhoods have strong directional structure:

- A spot next to a vessel vs. far from a vessel responds differently to
  the same source.
- A spot with macrophages on one side and fibroblasts on the other has
  anisotropic response that mean-pooling collapses.

**The proper alternatives** (none implemented in v2.1):

- **Directional binning.** Pool neighbours into 4 quadrants (anterior /
  posterior / left / right) relative to the source anchor, then concat.
  Adds 4× memory but preserves directional information.
- **Attention pooling (GAT-style).** Learn per-edge attention weights in
  the K=15 neighbourhood. Equivalent to a 1-layer GAT but constrained to
  1-hop, which preserves the spot-level resolution that v2 needs.
- **Spherical harmonics.** Encode neighbour directions as Y_lm
  coefficients for rotation-equivariance. Most principled but heaviest
  to implement.

**Why v2.1 does not fix this.** It is a research-grade change requiring
re-training and re-evaluation. Documented in `LIMITATIONS_V3.md`.

### P6. SLX, Durbin, and resistance distance are not theoretically unified

**v2 has three spatial modules** (`spatial/{slx,durbin,resistance}.py`),
each with a different definition of "neighbour":

| Module | W definition | Use in v2 |
|---|---|---|
| SLX | Binary KNN(k=6) adjacency | Robustness check |
| Durbin | Distance-decay weights (Gaussian kernel, σ=40µm) | Primary δ |
| Resistance | Path-sampled graph circuit resistance | Alternative δ |

These are **three different models of spatial dependence**, but v2 mixes
them: Durbin δ is the headline, resistance is an alternative measure, SLX
is a sanity check. There is no unified theoretical framework that says
when each is appropriate, or what the relationship between them is.

**Reviewer attack.** "This is methodological eclecticism, not
methodological innovation. Pick one framework and develop it."

**Mitigation (deferred).** A unified "spatial Durbin with
resistance-weighted W" is possible but requires deriving the score
function and asymptotic variance. Left for v3.

---

## §3 Implementation-layer problems (most fixable, most embarrassing if missed)

### P7. Matched control pool is contaminated by other sources' NCA effects

**The v2 setup.** For each Phgr1 source bin, v2 finds the nearest NTC bin
in embedding space (same slice) as the matched control.

**The contamination.** An NTC bin sitting within another (non-Phgr1)
source's spatial neighbourhood is itself affected by that source's NCA
effect. Using it as control for Phgr1 biases the estimated δ towards zero
(if the other source has the same response direction) or away from zero
(if opposite).

**The fix (v2.1 Fix 4).** Spatial buffer: exclude any NTC bin within
`b = 80 µm` (~ one decay length, see `phgr1_per_slice.durbin_lambda`) of
**any** source bin of **any** gene. Re-run v2's matcher on the filtered
pool.

**Pass criterion.** Phgr1 δ sign holds across all 5 responses; matched
control pool size drops by no more than 50% (else matching is
underpowered).

### P8. Permutation test swap target is ambiguous

**The v2 setup.** `causal.py` runs permutation tests to attach a p-value
to the Durbin δ. But "permutation" can mean:

- **Label permutation.** Shuffle guide labels across spots, preserving
  spatial structure. Tests "does the guide identity matter?".
- **Spatial permutation.** Shuffle spot positions, preserving guide
  structure. Tests "does spatial layout matter?".
- **Block permutation.** Permute within spatial blocks (e.g. 200×200 µm
  tiles) to preserve spatial autocorrelation. Most conservative.

v2's source code (need to verify in `causal.py`) most likely uses label
permutation, which is the standard but is **less conservative** than block
permutation. Reviewers will ask why block permutation was not used.

**Mitigation (deferred).** Adding block permutation is straightforward
but adds ~10-100× compute. Document the choice in the paper Methods.

### P9. `niche_id` from KMeans creates a circular dependency

**The v2 setup.** `raw_h5.py` runs `KMeans(k=12)` on per-spot cell-type
composition to derive `niche_id`. The encoder (Layer 1) is then
**supervised** on `niche_id`. The matching (Layer 2) uses `niche_id` as a
covariate.

**The circularity.** The encoder is trained to reproduce KMeans's decision
boundary. Matching balances on a quantity derived from that same KMeans.
Both downstream steps therefore inherit KMeans's modelling assumptions
without acknowledging it.

**The fix (v2.1 Fix 2).** Replace KMeans with **Leiden community
detection** on the kNN(cell-type composition) graph at three resolutions
(0.3, 0.6, 1.0 → ~8, 14, 26 niches). Re-run the encoder + matching + Durbin
scan for each resolution.

**Pass criterion.** Phgr1 δ sign consistent across all three Leiden
resolutions for ≥ 4/5 responses. The K=12 KMeans result must fall within
the resolution range (no cherry-pick).

### P10. Reduced-form δ has no propagated uncertainty from ρ̂

**The v2 setup.** Durbin's reduced-form effect:
```
δ = β + θ(I - ρW)⁻¹
```
`(I - ρW)⁻¹` is a Neumann series `Σ_{k=0}^∞ (ρW)^k`, dependent on the
estimate ρ̂. But v2's scan output reports `durbin_delta` and `durbin_p`
without propagating the standard error of ρ̂.

**The consequence.** If ρ̂ has a wide CI that crosses 0 or approaches 1
(spatial unit root), the reduced-form δ's CI can be much wider than
reported, or even unbounded.

**Mitigation (deferred).** Delta-method or parametric bootstrap on the
full `(β, θ, ρ)` joint distribution. Documented as a limitation.

---

## §4 Hidden deeper assumption

### P11. NCA effects assumed to be local + monotonically decaying

**The v2 assumption.** All three spatial models (SLX, Durbin, resistance)
encode the assumption that NCA effects are **local** and **monotonically
decay with distance**.

**The biological counter-examples.**

- **Vascular long-range signalling.** Cytokines entering the bloodstream
  can affect distant regions; effects are non-monotonic, may be
  step-functional at vascular branch points.
- **Neuronal-circuit NCA.** Effects propagate along axon pathways, jumping
  between connected regions; not radial.
- **Immune cell migration.** T-cell chemotaxis creates attractor patterns
  with central peaks but possible peripheral rebounds (ring-shaped
  response).

**The risk.** If Phgr1's true effect is non-monotonic, v2's decay-model
fit will have low R² → small δ → non-significant q → false negative.

**Mitigation (deferred).** Add a "no-spatial-structure" null model (pure
source vs control, ignoring distance) as a lower bound. If the null model
fits as well as the decay model, the decay assumption is untestable from
the data. Document in Discussion.

### P12. Cross-cohort batch effects not explicitly modelled

**The v2 setup.** Cohort 1 (lung metastasis) and cohort 2 (subQ
multisection) are different tissue microenvironments. v2's Fisher
combination implicitly assumes their module-score noise structures are
comparable.

**The reality.**

- Lung tissue has high vascular density, alveolar structure.
- SubQ tissue is fat-rich, fibroblast-dominated.
- NTC baselines for the same module score differ between cohorts (this
  is visible in v1's vs-NTC failure modes).

**The proper fix.** Cohort-specific z-scoring of every module score
before combining, OR a compositional model with explicit
`cohort × niche` interaction terms.

**Mitigation (deferred).** A cohort-standardized version of the scan is
straightforward to add. Documented in `LIMITATIONS_V3.md`.

---

# Part II — The five v2.1 fixes

Each fix below is **not** a replacement of v2's primary estimator. Each is
a sensitivity analysis: re-run the Phgr1 claim under a stress-test and
report whether it survives.

## Fix 1 — AIPW doubly robust estimator (attacks P1, P10)

### Estimator

Augmented inverse-probability-weighting with cross-fitting
(Chernozhukov et al. 2018, EJ):

```
μ̂₁(X) = XGBoost regression of y on X within T=1
μ̂₀(X) = XGBoost regression of y on X within T=0
ê(X)   = XGBoost classification of T on X (propensity)

ψᵢ = Tᵢ·(yᵢ - μ̂₁(Xᵢ)) / ê(Xᵢ)
   - (1-Tᵢ)·(yᵢ - μ̂₀(Xᵢ)) / (1-ê(Xᵢ))
   + μ̂₁(Xᵢ) - μ̂₀(Xᵢ)

τ̂_AIPW = (1/n) Σ ψᵢ
SE(τ̂) = sqrt(Var(ψ) / n)         # influence-function SE
95% CI: τ̂ ± 1.96 · SE
```

### X features (covariates)

| Feature | Source | Why |
|---|---|---|
| v2 embedding (64d) | embed_M00X.npy | Pre-trained spatial summary |
| cell_type (8d one-hot) | adata.obs | Known confounder |
| niche_id (12d one-hot) | adata.obs | Tissue context |
| density_local | graph_build | Tissue compactness |
| vessel_distance | graph_build | Vascular access |
| module scores (8d) | adata.obs | Co-responses (excluded: y itself) |

Total covariate dimension: 64 + 8 + 12 + 1 + 1 + 7 = **93**.

### T (treatment indicator)

`T = 1` for source bins (guide-positive for the gene being tested).
`T = 0` for matched controls (v2's existing match).

For Phgr1, n ≈ 1.1 M spots, of which ~50 K are Phgr1-source and ~50 K
are matched controls. Sub-sampling controls 1:1 to balance classes.

### Cross-fitting

5-fold KFold (sklearn `KFold(shuffle=True)`). For each fold:
1. Train μ̂₁, μ̂₀, ê on training 80%.
2. Predict on held-out 20%.
3. Concatenate predictions across folds.

This avoids the "use the same data to fit nuisance and estimate τ"
optimism. Cross-fitting is the standard modern practice.

### Hyperparameters

```python
outcome_params = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0,
    objective="reg:squarederror", n_jobs=8, verbosity=0,
)
propensity_params = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0,
    eval_metric="logloss", n_jobs=8, verbosity=0,
)
```

Max depth 3-4 is conservative; deeper trees over-fit on this sample size.

### Acceptance criteria

- AIPW τ̂ for Phgr1 × {5 responses} reported with 95% CI from
  influence-function SE.
- Sign of AIPW τ̂ agrees with Durbin δ for ≥ 4/5 responses.
- Magnitude within ±50% of Durbin δ. If AIPW τ̂ is much smaller, report
  as evidence that Durbin over-estimates.

### Pseudo-code

```python
import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np

def aipw_crossfit(X, t, y, n_splits=5, seed=7):
    X, t, y = np.asarray(X, f32), np.asarray(t, i8), np.asarray(y, f32)
    n = len(y)
    mu1_hat = np.zeros(n, f32); mu0_hat = np.zeros(n, f32)
    e_hat   = np.zeros(n, f32)

    kf = KFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        pos_tr = tr[t[tr] == 1]; neg_tr = tr[t[tr] == 0]
        m1 = xgb.XGBRegressor(**outcome_params).fit(X[pos_tr], y[pos_tr])
        m0 = xgb.XGBRegressor(**outcome_params).fit(X[neg_tr], y[neg_tr])
        e_model = xgb.XGBClassifier(**propensity_params).fit(X[tr], t[tr])
        mu1_hat[te] = m1.predict(X[te])
        mu0_hat[te] = m0.predict(X[te])
        e_hat[te]   = e_model.predict_proba(X[te])[:, 1]

    e_hat = np.clip(e_hat, 0.01, 0.99)
    psi = (t * (y - mu1_hat) / e_hat
           - (1-t) * (y - mu0_hat) / (1-e_hat)
           + mu1_hat - mu0_hat)
    tau = psi.mean()
    se  = psi.std(ddof=1) / np.sqrt(n)
    return tau, se, tau - 1.96*se, tau + 1.96*se
```

### File

`perturbgnn_v2_1_src/causal/aipw.py` — implemented, awaiting server run.

---

## Fix 2 — Leiden niches (attacks P9)

### Estimator

Leiden community detection (Traag et al. 2019, Sci Rep) on the
kNN(cell-type composition) graph at three resolutions.

### Steps

1. For each spot, compute cell-type composition in its 40µm neighbourhood
   (8-dim vector, sum to 1).
2. Build kNN(k=20) graph on these vectors (cosine distance).
3. Run Leiden with `resolution_parameter ∈ {0.3, 0.6, 1.0}`.
4. Expected community counts: ~8, ~14, ~26 (lower resolution → fewer).
5. For each Leiden niche partition:
   a. Re-train the encoder with this partition as supervision.
   b. Re-run matching + Durbin scan for Phgr1.
   c. Record δ, q, n_replicated_slices per response × resolution.

### Hyperparameters

```python
import leidenalg as la
from igraph import Graph

# Build graph from sklearn kNN
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=20, metric="cosine").fit(comp_matrix)
G = Graph.Weighted_Adjacency(nn.kneighbors_graph().toarray().tolist())

for res in [0.3, 0.6, 1.0]:
    part = la.find_partition(G, la.RBConfigurationVertexPartition,
                             resolution_parameter=res, seed=7)
    niche_labels = np.array(part.membership)
```

### Acceptance criteria

- Phgr1 δ sign consistent across all three resolutions for ≥ 4/5 responses.
- The KMeans(k=12) baseline falls within the resolution range (no
  cherry-pick — i.e. KMeans δ is between Leiden res 0.6 and 1.0).

### File

`perturbgnn_v2_1_src/data/niche_robust.py` — to be implemented.

---

## Fix 3 — Probe transfer alignment (attacks P4)

### Estimator

Re-use the existing v2 encoder checkpoint. No re-training needed.

### Steps

1. Load `bt_encoder_v3.pt` and the per-slice AnnData.
2. Compute z₁ and z₂ for every spot (forward pass only).
3. Random 80/20 split.
4. Train `head_ct` and `head_niche` on z₁ in the train set.
5. Evaluate zero-shot accuracy on:
   - **Upper bound**: z₁(test) → head → accuracy.
   - **Cross-modal**: z₂(test) → head → accuracy (the real measurement).
   - **Lower bound**: z₂(permuted test) → head → accuracy.
6. Report three accuracies × two heads (cell_type, niche) per slice.

### Acceptance criteria

- Cross-modal z₁→z₂ accuracy ≥ 70% of same-view z₁→z₁ accuracy.
- Reported alongside v2's cos(z₁, z₂) = 0.9997 as a complementary metric.
- If z₁→z₂ collapses (e.g. < 30% of same-view), report as a v2
  architectural limitation.

### File

`perturbgnn_v2_1_src/embedding/probe_encoder.py` — to be implemented.

---

## Fix 4 — Spatial buffer in matching (attacks P7)

### Estimator

Same as v2's matcher, with one extra filter on the candidate control pool.

### Steps

1. For each slice, identify all source bins (any gene).
2. Build a cKDTree on source bin coordinates.
3. For each NTC bin, query distance to nearest source bin.
4. If distance < `b = 80 µm`, mark as "buffered out".
5. Run v2's existing matcher on the filtered pool.
6. Compute new SMD; verify still < 0.13.
7. Re-run Phgr1 cross-cohort scan with the new matched controls.

### Hyperparameters

- `b = 80 µm` (one decay length, from `phgr1_per_slice.durbin_lambda`).
- Sensitivity: also try `b ∈ {40, 60, 100, 120} µm`.

### Acceptance criteria

- Phgr1 δ sign holds across all 5 responses at `b = 80 µm`.
- Matched control pool size drops by no more than 50%.
- SMD on buffered match remains < 0.13.

### Pseudo-code

```python
from scipy.spatial import cKDTree

def buffered_match(adata, source_mask, ntc_mask, buffer_um=80.0):
    xy = adata.obsm["spatial"]
    src_xy = xy[source_mask]
    ntc_xy = xy[ntc_mask]

    src_tree = cKDTree(src_xy)
    dist_to_src, _ = src_tree.query(ntc_xy, k=1)

    keep = dist_to_src >= buffer_um   # filter
    print(f"NTC pool: {len(ntc_xy)} → {keep.sum()} after buffer")
    return ntc_mask & np.concatenate([keep, [False]*(len(xy)-len(keep))])
```

### File

`perturbgnn_v2_1_src/matching/match_buffered.py` — to be implemented.

---

## Fix 5 — Real message-passing GNN baseline (attacks P5, addresses "you should have used a GNN")

### Estimator

A 2-layer GraphSAGE with mean aggregation, trained with the same
supervision targets (cell_type + niche) on the same tissue graph as v2's
Layer 1.

### Architecture

```
Input per node v:
  h_v⁽⁰⁾ = [PCA(32) ⊕ CT-onehot(8) ⊕ density ⊕ vessel_dist]  (42d)

SAGE layer 1:
  h_v⁽¹⁾ = ReLU(W₁ · CONCAT(h_v⁽⁰⁾, MEAN_{u∈N(v)} h_u⁽⁰⁾))

SAGE layer 2:
  h_v⁽²⁾ = ReLU(W₂ · CONCAT(h_v⁽¹⁾, MEAN_{u∈N(v)} h_u⁽¹⁾))

Output:
  z_v = h_v⁽²⁾  (64d, after a final linear projection)

Heads (same as v2):
  head_ct(z_v), head_niche(z_v)
```

L=2 means each spot's embedding sees 2-hop neighbourhood (~120µm radius).
Beyond L=2 risks over-smoothing on this dense graph.

### Implementation note

Use PyG `SAGEConv` to avoid re-implementing sampling:

```python
import torch_geometric.nn as gnn

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim=42, hidden=128, embed_dim=64):
        super().__init__()
        self.sage1 = gnn.SAGEConv(in_dim, hidden)
        self.sage2 = gnn.SAGEConv(hidden, embed_dim)
        self.head_ct    = ClassificationHead(embed_dim, 8)
        self.head_niche = ClassificationHead(embed_dim, 12)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
        h = self.sage2(x=h, edge_index=edge_index).relu()
        return h, self.head_ct(h), self.head_niche(h)
```

Mini-batch training via PyG `NeighborLoader` (sample 15 neighbours per
hop, batch size 4096).

### Acceptance criteria (decision rule)

| Outcome | Action |
|---|---|
| SAGE SMD ≤ v2 SMD AND SAGE δ within ±20% of v2 δ | Set-pooling validated; report as justification in Methods. |
| SAGE SMD much better OR SAGE δ much larger | Add SAGE to v2 paper as improved variant; re-run cross-cohort scan. |
| SAGE training unstable / OOM | Report as evidence that message-passing is impractical at this scale. |

This is the one fix that could **upgrade** v2. Either outcome is
publishable.

### File

`perturbgnn_v2_1_src/baseline/real_gnn.py` — to be implemented.

---

# Part III — Integration and reporting

## Output files

Each fix produces one file in `tutorials/data_v2/sensitivity/`:

| File | Fix | Content |
|---|---|---|
| `aipw_phgr1.csv` | 1 | τ̂, SE, CI, comparison with Durbin δ, per response × slice |
| `niche_leiden_sensitivity.csv` | 2 | δ per response × Leiden resolution × slice |
| `probe_transfer.json` | 3 | z₁→z₁, z₁→z₂, baseline accuracies per head per slice |
| `matching_buffer_sensitivity.csv` | 4 | δ with/without 80µm buffer, by buffer radius |
| `sage_vs_setpooling.csv` | 5 | SMD, δ, training stability for SAGE vs set-pooling |

## Master notebook integration

`SINGLE_MASTER_perturbgnn_v2.ipynb` gets a new §15 "v2.1 sensitivity"
section that reads the five files above and renders a single "robustness
table":

| Response | v2 δ | AIPW τ̂ | Leiden δ | Buffered δ | SAGE δ |
|---|---|---|---|---|---|
| fibroblast | +0.43 | TBD | TBD | TBD | TBD |
| macrophage | +0.41 | TBD | TBD | TBD | TBD |
| ... | | | | | |

If all fixes pass, the paper Discussion gains:

> *Every modelling choice in our pipeline was stress-tested in §N. The
> Phgr1 result survived all five attacks — AIPW doubly robust estimation,
> Leiden alternative niches, probe-transfer alignment, spatial-buffered
> matching, and a GraphSAGE message-passing baseline — indicating
> robustness rather than dependence on any single architectural decision.*

If any fix fails, we either fix it before submission or report it
honestly. **We do not bury failures.**

---

# Part IV — What v2.1 does **not** fix (deferred to v3)

| Problem | Why deferred |
|---|---|
| P2 Rosenbaum Γ* circularity (true fix needs negative-control outcomes) | Not available in v2's data; needs new experimental design |
| P3 Hierarchical Bayesian meta-analysis for cross-cohort | 1-2 weeks additional modelling; left for paper revision |
| P5 Directional pooling in encoder | Research-grade change; needs separate ablation study |
| P6 Unified spatial Durbin with resistance W | Requires deriving new score function and asymptotic variance |
| P8 Block permutation test | 10-100× compute; document the label-permutation choice |
| P10 Neumann-series CI on ρ̂ | Closed form unknown; bootstrap expensive |
| P11 Non-monotonic decay alternatives | Requires new model family; left for v3 |
| P12 Cohort-specific z-scoring | Straightforward but deferred to paper revision |

These are documented in `LIMITATIONS_V3.md` (to be written).

---

**Status**: §1-§4 (twelve problems) and Fix 1-5 specifications complete.
Fix 1 code implemented and awaiting server run. Fix 2-5 to be implemented
once server access is restored.
