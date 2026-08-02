# PerturbGNN v2 — Design Document

**Status**: Phase 0 (foundation)
**Last updated**: 2026-08-01
**Owner**: Xu Zhenghao
**Predecessor**: perturbgnn/ (v1, BLNK-focused, AI-managed) — declared toy, not authoritative

---

## 0. Why v2 exists

v1's four-layer failure (declared by user 2026-07-31):

1. **BLNK causal chain is broken** — single guide, partial correlation collapses
   after B-cell control, no rescue experiment.
2. **vs-NTC screen framework is wrong** — NTC is not a true null; z-score
   doesn't address spatial autocorrelation; FDR unit is mis-specified.
3. **SPAC-seq dataset cannot support propagation-radius claims** under any
   method (1 guide/gene/slice, non-overlapping panel, pseudo-replicates).
4. **GNN is decorative** — only used for cross-slice transfer; per-bin
   counterfactual loses to concentric ring on radial tasks (v1 Result 6).

v2 does not patch v1. v2 rebuilds the modeling stack from a different
scientific question.

---

## 1. Scientific question

> **In a spatial CRISPR screen with sparse perturbation and snapshot data,
> how can we estimate a single gene's effect on surrounding cells' phenotype
> as a 2D spatial function, while controlling for spot-level confounding
> through a GNN-learned background?**

Three sub-questions:
- **(b) fallback**: detectable perturbation effect of gene G on response Y
- **(c) target**: non-cell-autonomous propagation, i.e. effect on cells
  *not themselves perturbed* but spatially exposed to perturbation
- **embedding-matched null**: replace NTC / far-ring with a learned
  neighborhood-matched background

The framing is **NOT** "what is the propagation radius R". That framing
requires assumptions (radial decay, single length-scale, single clone shape)
the data cannot support. The v2 framing is "what is the effect surface
Δ_G(x,y) and how confident are we".

---

## 2. Design constraints (locked)

| Constraint | Choice | Implication |
|---|---|---|
| Target claim | (b) + attempt (c) | gene-level effect with causal diagnostics |
| Background | α+β | GNN embedding → matched control in embedding space |
| Framework | III (full rebuild) | vs-NTC + concentric ring + exponential all discarded |
| Relation to v1 | A (scrap) | v1's 97-gene scan results declared void |

---

## 3. Five-layer pipeline

```
Layer 0  Data + tissue graph                 (Phase 0)
   │     SPAC-seq H5 → AnnData + section graph
   │     edge features for resistance distance
   ▼
Layer 1  GNN embedding (Barlow Twins)        (Phase 1)
   │     spot representation e_i ∈ R^d
   │     encodes neighborhood type, NOT raw expression
   ▼
Layer 2  Embedding-matched control           (Phase 2)
   │     for each analysis spot, find matched controls
   │     outside 200μm of any source, in same embedding cluster
   ▼
Layer 3  Spatial network model               (Phase 3)
   │     multi-ring SLX + resistance distance + spatial Durbin
   │     replaces concentric ring + exponential decay
   ▼
Layer 4  Causal inference + genome scan      (Phase 4)
         DiD, guide-IV, Rosenbaum sensitivity
         per-gene Δ_G^Y with clone-bootstrap CI
```

---

## 4. Layer 0 — Data + tissue graph

### Inputs
- Raw SPAC-seq H5 (M001/M002/M003), 1.1M bins × 19,059 genes × 1,520 guides
- Official cell-type annotation CSV
- Official clone annotation CSV
- Niche annotation (computed: cell-type composition clustering)

### Outputs
- `AnnData` per slice with:
  - `obsm['spatial']` — x_um, y_um
  - `obs['cell_type']`, `obs['niche_id']`, `obs['is_source']`, `obs['is_ntc']`,
    `obs['target_gene']`, `obs['guide_total']`, `obs['top_fraction']`
  - `obsm['X_pca']` — 32-dim PCA of normalized expression
  - `obsm['module_scores']` — 7 module scores per bin (malignant/cd8/mac/fib/endo/hyp/ifn)
  - `layers['guide_count']` — per-bin guide UMI total
  - `var` — gene metadata
  - `uns['tissue_graph']` — networkx Graph + PyG Data
- Edge features (per edge):
  - `[dx, dy, dist, same_cell_type, src_endpoint]` (v1, retained)
  - `cell_density_local` — local cell density (proxy for tissue compactness)
  - `barrier_score` — derived from cell-type discontinuity (proxy for ECM/necrosis barriers)
  - `vessel_proximity` — distance to nearest endothelial cell (proxy for vessel)

### Design decisions
- **`target_gene_filter` removed** from loader — v2 scans all genes, not one
- **`response` field removed** from loader — response is decided at analysis time
- **Lazy gene expression** — full 19,059-gene matrix stays in H5 backing, loaded on demand per gene in Layer 4
- **Thread cap** — mandatory `threadpool_ctx(4)` around all dense ops

### File layout
```
src/perturbgnn_v2/data/
├── __init__.py
├── raw_h5.py          ← adapted from v1 (remove filter, remove response)
├── graph_build.py     ← adapted from v1 (add resistance-distance edge features)
└── niche.py           ← new: cell-type composition → niche label
```

---

## 5. Layer 1 — Barlow Twins GNN embedding

### Why not SpaLP / v1 E(n)-GNN
- **SpaLP**: AE + MSE loss — learns to reconstruct expression, source spots
  contaminated into embedding
- **v1 E(n)-GNN**: trained for cell-type prediction, not representation
- **Barlow Twins**: cross-modal, learns neighborhood *type* not expression

### Architecture
```
Two views per spot i (K=15 neighbors):
  view_1 = neighborhood expression PCs (K × 32)
  view_2 = neighborhood cell-type composition (K × |cell_types|)

Encoder (shared weights):
  GNN encoder (3 layers SAGEConv) → mean-pool over K neighbors → spot embedding e_i ∈ R^64

Loss (Barlow Twins):
  C = Z1^T Z2 / N      (64 × 64 cross-correlation matrix)
  L_BT = Σ_i (1 - C_ii)² + λ Σ_{i≠j} C_ij²
  (λ = 0.005, forces off-diagonal to 0 → decorrelated, expressive embedding)
```

### Training
- Batch: 8192 spots per mini-batch, sampled across all 3 slices
- Optimizer: Adam lr=1e-3, cosine schedule, 200 epochs
- Held-out: 1 slice for embedding quality validation (cell-type separability)
- Compute: 1× V100S, estimated 15 min/slice pair

### Validation
- Cell-type separability: silhouette score ≥ 0.3
- Niche clustering: adjusted rand index ≥ 0.5 vs manual niche annotation
- Robustness: embedding stability under spot-position perturbation ±5μm
- Sanity: source spot embedding should NOT cluster separately from
  matched-control spot embedding (if it does, perturbation is leaking into
  the embedding — bad)

### Outputs
- `obsm['X_bt']` — 64-dim Barlow Twins embedding per spot
- Checkpoint: `processed/bt_encoder.pt`

### File layout
```
src/perturbgnn_v2/embedding/
├── __init__.py
├── bt_encoder.py      ← model definition
├── train_bt.py        ← training loop
└── validate_bt.py     ← embedding quality checks
```

---

## 6. Layer 2 — Embedding-matched control

### Algorithm
```
For target gene G in slice s:
    source_bins(G, s) = {spot | guide_total≥2 ∧ top_fraction≥0.7
                              ∧ target_gene=G}
    analysis_bins(G, s) = {spot | dist(spot, source_bins) ≤ 200μm}
                          \ source_bins  # exclude source itself

    For each spot i in analysis_bins:
        candidates = {spot | same slice, same cell_type,
                            dist(spot, source_bins) > 400μm,
                            guide_total < 2}    # unexposed proxy
        kNN_pool = top-50 nearest in obsm['X_bt'] (cosine)
        C(i) = sample 5-10 from kNN_pool   # diversity sampling

    Δ_i = Y_i - mean(Y_c for c in C(i))
```

### Validation
- On NTC source spots: matched Δ should center at 0 (sanity check vs v1's +4.5% floor)
- Covariate balance: post-match SMD on cell_density, niche_id, X_pca[0:8] all < 0.1
- Effective sample size: ≥ 30 unique control spots per analysis spot

### Outputs
- `analysis_bins_matched.parquet` — per-spot matched control IDs and Δ
- `match_balance_report.csv` — covariate balance diagnostics

### File layout
```
src/perturbgnn_v2/matching/
├── __init__.py
├── match.py           ← embedding kNN matching
└── balance.py         ← covariate balance diagnostics
```

---

## 7. Layer 3 — Spatial network models

Three complementary models. Used jointly, not interchangeably.

### Model A — Multi-ring hierarchical SLX (baseline)
```
logit(E[Y_ijks]) = α + β·T_ij + Σ_k θ_k·(W_k T)_ij
                   + γ·X_ij + u_animal + u_section + u_clone
```
- 5 ring weights: 0–20, 20–40, 40–80, 80–120, 120–200 μm
- Random effects: clone nested in section nested in animal
- Output: θ_1...θ_5 per (gene × response) with likelihood-ratio test
- Implementation: statsmodels.MixedLM + custom W_k construction

### Model B — Resistance-weighted SLX (key innovation)
```
Same form, but W_k replaced by:
    W_k(i,j) = 1 if R(i,j) ∈ [ring_k_low, ring_k_high]
    where R(i,j) = effective resistance on tissue graph
                   with edge weights encoding barriers
```
Edge weights:
```
w(i,j) = base_cost
       + α_1 · barrier_score(i,j)
       + α_2 · density_discontinuity(i,j)
       - α_3 · vessel_proximity(i,j)
```
- Implementation: `pot` package or NetworkX `effective_resistance` (slow, fallback to Laplacian pseudo-inverse approximation)

### Model C — Multi-source spatial Durbin (causal propagation)
```
exposure_field_i = Σ_c M_c · K(d_res(i, c))
                 over all source clones c within 500μm
    M_c = clone_size_normalized × guide_posterior_mean
    K = Gaussian kernel, λ learned per (gene × response)

Y_i = α + β·T_i + δ·exposure_field_i + γ·X_ij + ε_i
```
- Output: δ per (gene × response) — the propagation slope
- This model handles multi-source overlap (the thing rings structurally cannot)

### Benchmark plan (synthetic data with known ground truth)
- 4 sources, isotropic λ=60μm decay → all 3 models should recover
- 4 sources with anisotropic barrier → only B and C should recover
- 2 overlapping sources with different amplitudes → only C attributes correctly

### File layout
```
src/perturbgnn_v2/spatial/
├── __init__.py
├── slx.py             ← Model A
├── resistance.py      ← Model B (compute R, build W_k)
├── durbin.py          ← Model C
└── benchmark.py       ← synthetic-data benchmark
```

---

## 8. Layer 4 — Causal inference + genome scan

### Per (gene G, response Y, slice s): four diagnostics

**(b1) OLS estimate** (naive baseline)
```
Δ_OLS = β from Y ~ T + covariates, no spatial structure
```

**(b2) SLX estimate** (spatially-aware)
```
Δ_SLX = Σ_k θ_k from Layer 3 Model A
95% CI: clone-level bootstrap (resample clones, refit, 1000 reps)
```

**(c1) Spatial DiD** (controls neighborhood gradient)
```
Δ_DiD = [Y_perturbed_near - Y_perturbed_far]
      - [Y_matched_control_near - Y_matched_control_far]
```
where matched_control comes from Layer 2.

**(c2) Guide-IV estimate** (tests dose-response)
```
Stage 1: T_hat = π_0 + π_1 · guide_UMI + covariates
Stage 2: Y = α + β · T_hat + covariates
Compare β_IV to β_OLS; if close, perturbation is the mechanism
```

### Sensitivity analysis (Rosenbaum)
- Compute Γ* = smallest Γ that flips significance
- Report Γ* per (gene × response); Γ* ≥ 1.5 = robust to moderate hidden confounding

### Multiple testing
- **Test unit**: (gene × response) pairs that have ≥ 2 slices
- **Correction**: Benjamini-Hochberg across all tested pairs
- **Power calibration**: synthetic injection at known effect sizes → minimum detectable effect at 80% power

### Outputs
- `genome_scan_v2.csv` — per (gene × response × slice): all 4 diagnostics + CI + Γ* + q-value
- `gene_ranking.csv` — ranked by combined evidence score
- `power_calibration.csv` — minimum detectable effect at 80% power per response

### File layout
```
src/perturbgnn_v2/causal/
├── __init__.py
├── did.py             ← spatial DiD
├── iv.py              ← guide-IV 2SLS
├── sensitivity.py     ← Rosenbaum Γ
├── scan.py            ← genome-wide scan orchestrator
└── bootstrap.py       ← clone-level bootstrap CI
```

---

## 9. Relation to v1

| Asset | Reuse? | Action |
|---|---|---|
| H5 loader (`raw_h5.py`) | ✅ partial | Adapt: remove `target_gene_filter`, remove `response` hardcode |
| Graph builder (`graph_build.py`) | ✅ partial | Adapt: add barrier/density/vessel edge features |
| Thread limiter (`_thread_limits.py`) | ✅ full | Symlinked |
| E(n)-GNN (`equivariant.py`) | ❌ | Discard — replaced by Barlow Twins encoder |
| vs-NTC screen (`s1_genome_scan.py`) | ❌ | Discard — replaced by Layer 4 |
| Concentric ring Δ (`r1-r4`) | ❌ | Discard — replaced by Layer 3 |
| BLNK clinical validation (`v1-v10`) | ⚠ hold | Reuse only if v2 independently identifies BLNK; otherwise void |
| TCGA / IMvigor / melanoma cohorts | ✅ infra | Reuse query code, replace target gene list |

---

## 10. Phase milestones and acceptance gates

| Phase | Duration | Acceptance gate |
|---|---|---|
| **P0** Data foundation | 1 wk | Slice graph builds for M001/M002/M003; ≥1 edge feature beyond v1; DESIGN.md signed off |
| **P1** Barlow Twins GNN | 2-3 wk | Silhouette ≥0.3, source vs matched-control embedding overlap ≥0.7 |
| **P2** Matched control | 1-2 wk | NTC matched Δ within ±0.005 of 0; covariate balance SMD<0.1 |
| **P3** Spatial models | 3-4 wk | All 3 models recover synthetic ground truth within 10% bias |
| **P4** Causal + scan | 2-3 wk | Genome scan completes; ≥1 gene passes Γ*≥1.5 |
| **P5** Paper | 2-3 wk | 5 figures + draft + venue selected |

---

## 11. Known limitations (carried from data)

These will remain in v2's limitations section regardless of method quality:

1. **Single guide per gene per slice** — off-target cannot be excluded
2. **3 slices from same cohort** — not independent biological replicates
3. **Non-overlapping gene panel across slices** — only 7/120 genes in all 3 slices
4. **No time course** — causal direction inferred, not measured
5. **Spot-level (8μm bin)** — coarser than single-cell, contact signaling diluted

v2's contribution is to give honest confidence intervals under these constraints, not to remove them.

---

## 12. Open questions deferred to implementation

- [ ] Should we run Barlow Twins per-slice or jointly across slices? (joint better for cross-slice transfer, per-slice safer for batch effects)
- [ ] Resistance distance: exact (`pot` package) vs Laplacian pseudo-inverse approximation (10× faster, slightly biased)
- [ ] Whether to include M002 in primary analysis (only 8 NTC bins — Layer 2 matching will struggle)
- [ ] Rosenbaum sensitivity: which covariate set to use for the bias model

These will be resolved at the start of each Phase with a 1-paragraph decision note in `docs/`.
