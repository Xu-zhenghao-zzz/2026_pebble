# PerturbGNN v2 — Pipeline layer contracts

This document specifies the **exact input and output contract** of every
layer in the v2 pipeline. It is the canonical reference for "what data
shape goes in, what comes out, what file holds it, what fields are
mandatory". Anything not listed here is implementation detail and can be
refactored without notice.

Layer numbering: L0 (raw) → L1 (encoder) → L2 (matching) → L3 (spatial
model) → L4 (causal diagnostics) → L5 (cross-cohort meta-analysis) →
L6 (orthogonal validation). v2.1 fixes hook into specific layers and are
noted at the bottom.

---

## L0 — Raw data → AnnData + tissue graph

### Input

Public SPAC-seq matrices under
`$PERTURBGNN_DATA_ROOT/SPAC/processed/extracted/`:

| File | Shape | Content |
|---|---|---|
| `Transcriptome/<slice>/filtered_gene_bc_matrix.h5` | sparse (N, G) | per-bin gene expression |
| `Perturbation/<slice>/filtered_guide_bc_matrix.h5` | sparse (N, P) | per-bin guide UMI counts |
| `obs/<slice>.csv` | (N, k) | bin metadata: spatial coords, cell_type, niche |
| `obsm/<slice>_pca.npy` | (N, 32) | pre-computed PCA of expression |

Cohorts:
- **Cohort 1** — lung metastasis: `M001`, `M002`, `M003`. 1 guide per gene.
- **Cohort 2** — multiple-section subQ: `subQ-1` … `subQ-5`. 2 guides per gene.
- **Cohort 3** — spatiotemporal T cell: `Day7_rep1`, `Day10_rep1`. Auxiliary only.

### Internal processing

`raw_h5.py` / `raw_h5_cohort2.py`:
1. Load sparse expression, compute per-bin module scores (8 scores:
   `ifn_response, hypoxia, fibroblast, macrophage, cd8_like,
   endothelial, malignant, nk_like`).
2. Identify `source_mask` (guide-positive for any gene) and
   `ntc_mask` (non-targeting control guide).
3. Run `KMeans(k=12)` on cell-type composition vectors → `niche_id`.
4. Add per-bin `cell_type_idx` (0-7, 8 classes).

`graph_build.py` (per slice):
- Build node set from bin coordinates `xy`.
- Build edges = Delaunay triangulation ∪ kNN(k=6, max_dist=60µm).
- Compute per-node auxiliary features: `density_local`
  (neighbours within 40µm), `vessel_distance` (to nearest
  endothelial-like spot).
- Compute per-edge features (8-dim): `[dx, dy, dist, same_cell_type,
  src_endpoint, barrier_score, density_avg, vessel_dist_avg]`.

### Output

Per slice, written to `processed/`:

| File | Shape / type | Content |
|---|---|---|
| `<slice>_v2.h5ad` | AnnData | full per-slice data (X, obs, obsm, obsp) |
| `tissue_graph_<slice>_v2.pt` | PyG `Data` | node_features, edge_index, edge_features |

`<slice>_v2.h5ad.obs` columns (mandatory):
```
array_row, array_col              # bin grid coords
x_um, y_um                        # spatial coords in µm
cell_type (str)                   # 8-class label
cell_type_idx (int8)              # 0-7
niche_id (int8)                   # 0-11 (KMeans k=12)
guide (str)                       # sgRNA identity or "NTC"
is_source (bool)                  # guide-positive for any target gene
is_ntc (bool)                     # non-targeting guide
module_score_<name> (float32) × 8 # 8 module scores per bin
```

`<slice>_v2.h5ad.obsm`:
```
spatial (float32, N×2)            # µm coords (== x_um, y_um)
X_pca   (float32, N×32)           # PCA of expression
```

`tissue_graph_<slice>_v2.pt` PyG `Data` fields:
```
x          (N, 42) float32   # [PCA32, CT-onehot8, density, vessel_dist]
edge_index (2, E) int64      # PyG-format edge list, undirected
edge_attr  (E, 8)  float32   # 8-dim edge features
```


---

## L1 — GNN encoder → 64-dim spot embedding

### Input

- `adata` from L0 (per slice): `obsm['X_pca']`, `obsm['spatial']`,
  `obs['cell_type_idx']`, `obs['niche_id']`.
- Hyperparameters: `K=15` (neighbourhood size), `embed_dim=64`,
  `n_layers=3`, `hidden=128`, `dropout=0.1`, `alpha_align=0.1`.

### Internal processing (`bt_encoder.py` + `train_bt.py`)

For each spot `v`:
1. Find K=15 nearest spatial neighbours via `cKDTree(xy)` (excluding self).
2. Gather `nbr_pca ∈ ℝ^{15×32}` and `nbr_ct ∈ ℝ^{15×8}` (one-hot cell type
   per neighbour).
3. View 1 (PCA): `NeighborhoodEncoder` pools via
   `[mean_pool, max_pool, anchor]` (3×32 = 96-dim) → 3-layer MLP
   (96→128→128→64) → `z1 ∈ ℝ^64`.
4. View 2 (CT): same architecture, input 3×8 = 24-dim → `z2 ∈ ℝ^64`.
5. Shared classification heads `head_ct`, `head_niche` predict
   cell_type (8-class) and niche_id (12-class) from BOTH z1 and z2.
6. Loss: `L_cls(z1) + L_cls(z2) + 0.1 · (1 - cos(z1, z2))`.

Train 200 epochs (M001+M002 train, M003 val), batch=4096, Adam.

### Output

| File | Shape | Content |
|---|---|---|
| `bt_encoder_v3.pt` | torch state_dict | trained encoder + heads |
| `embed_<slice>.npy` | (N, 64) float32 | per-spot test-time embedding `0.5·(z1+z2)` |
| `embedding_validation.json` | dict | per-slice diagnostics: `n_spots, n_source, n_ntc, silhouette_celltype, niche_ari, niche_nmi, src_ntc_centroid_cos` |

### Contract

For each slice, `embed_<slice>.npy[i]` MUST correspond to row `i` of
`adata.obs` (same ordering). Downstream layers depend on this index
alignment.


---

## L2 — Embedding-matched control → per-source counterfactual

### Input

- `adata` from L0 (per slice, full obs + obsm).
- `embed_<slice>.npy` from L1 (per-spot 64-dim).
- For each target gene `g`: source mask = `adata.obs['guide'] == sg<g>`.

### Internal processing (`matching/match.py`)

For each source bin `s` of gene `g`:
1. Candidate pool = all NTC bins in the same slice.
2. Compute cosine similarity between `embed[s]` and every NTC embedding.
3. Pick the top-K (default K=1) closest NTC bins → matched control `c(s)`.
4. Record `(s, c(s), cos_sim)` pair.

Covariate balance check:
- Sample 500 random pairs, compute SMD on each of the 64 embedding dims
  + 8 module scores + cell_type + niche_id.
- Pass criterion: `max_SMD < 0.13`.

### Output

| File | Shape | Content |
|---|---|---|
| `matches_<slice>_<gene>.csv` | (n_source, 3) | columns: `source_idx, control_idx, cos_sim` |
| `matched_control_validation.csv` | (n_pairs, d) | SMD table per covariate per gene |

### Contract

For every source bin index `s` in the file, `adata.obs.iloc[s]` MUST be a
source bin of gene `g`, and `adata.obs.iloc[control_idx]` MUST be an NTC
bin in the same slice. Cross-slice matches are forbidden.


---

## L3 — Spatial Durbin model → per-source causal effect

### Input

- `adata` from L0 (per slice).
- `embed_<slice>.npy` from L1 (used as additional covariates).
- `matches_<slice>_<gene>.csv` from L2 (source + matched control pairs).
- `tissue_graph_<slice>_v2.pt` from L0 (for the W matrix).
- Hyperparameters: `n_neighbors=6`, `max_dist_um=60`, `lambda_grid=
  [30, 60, 90, 120, 150]` (decay length search), `min_n_clones=10`.

### Internal processing (`spatial/durbin.py`)

For each (slice, gene, response) triple:
1. Build spatial weights matrix `W` from kNN edges, weighted by Gaussian
   kernel `w_ij = exp(-d_ij² / (2σ²))` with `σ = 40µm`.
2. Row-normalize W.
3. Define `y = module_score_<response>`, `X = source_indicator` (1 for
   source bins, 0 for matched controls).
4. Fit spatial Durbin MLE:
   ```
   y = ρWy + Xβ + WXθ + ε
   ε ~ N(0, σ²I)
   ```
   Optimise `(ρ, β, θ, σ²)` via `scipy.optimize.minimize` (L-BFGS-B).
5. Reduced-form effect:
   ```
   δ = β + θ · (I - ρW)⁻¹
   ```
   Compute `(I - ρW)⁻¹` via Neumann series `Σ_{k=0}^{100} (ρW)^k`.
6. Decay length `λ` from the row-sum of `(I - ρW)⁻¹` vs distance.
7. Permutation p-value: shuffle X labels across bins (preserving spatial
   structure), refit δ, repeat 1000 times. `p = (#{|δ_perm| ≥ |δ_obs|} + 1) / 1001`.

### Output

| File | Shape | Content |
|---|---|---|
| `genome_scan_v2_<slice>.csv` | (~120 × n_responses, 11) | per (slice, gene, response) row |
| `cohort2_<slice>_scan.csv` | (~120 × n_responses, 10) | per (slice, gene, response) row, cohort 2 schema |

Schema (cohort 1, 11 cols):
```
slice, gene, response, durbin_delta, durbin_p, durbin_lambda,
durbin_beta_self, durbin_r2, n_clones, did_delta, did_p, did_n_pert_near
```

Schema (cohort 2, 10 cols):
```
slice, gene, response, n_sources, durbin_delta, durbin_p, durbin_lambda,
n_clones, did_delta, did_p
```


---

## L4 — Causal diagnostics → robustness gates

### Input

- L3 scan CSVs (per slice).
- `adata` from L0 (for DiD pre/post windows and IV instrument).
- `matches_<slice>_<gene>.csv` from L2.

### Internal processing (`causal/causal.py`)

For each (slice, gene, response) triple that passed L3's permutation
p < 0.05:

**DiD (difference-in-differences).**
1. Define "near" = within 40µm of source; "far" = 40-120µm from source.
2. Δ_source = mean(y[near, source]) - mean(y[far, source]).
3. Δ_ntc = mean(y[near, NTC]) - mean(y[far, NTC]).
4. DiD = Δ_source - Δ_ntc.
5. DiD p-value via label permutation within the slice.

**IV (instrumental variable).**
1. Z = guide assignment (instrument). T = source_indicator.
2. Stage 1: T ~ Z + covariates → T̂.
3. Stage 2: y ~ T̂ + covariates → β_IV.
4. Compare sign and significance of β_IV with OLS β.

**Rosenbaum Γ\*.**
1. For each Γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, recompute the Wilcoxon
   signed-rank p-value under hidden confounding at odds ratio Γ.
2. Γ* = the smallest Γ at which the p-value crosses 0.05.

### Output

| File | Shape | Content |
|---|---|---|
| `phgr1_full_diagnostics.json` | nested dict | per (slice, response): did_p, iv_p, gamma_star, n_source, n_control |
| Per-row additions to L3 scan CSVs: `did_delta, did_p, did_n_pert_near` |

### Contract

A (slice, gene, response) passes L4 if **all four** hold:
- `durbin_p < 0.05`
- `did_p < 0.05`
- `iv_p < 0.05` AND `sign(β_IV) == sign(β_OLS)`
- `gamma_star ≥ 2`


---

## L5 — Cross-cohort meta-analysis → reportable claim

### Input

- L3+L4 scan CSVs from all 8 slices (cohort 1: M001, M002, M003; cohort 2:
  subQ-1 … subQ-5).
- Per (gene, response) → list of (slice, durbin_p, durbin_delta) tuples.

### Internal processing (`experiments/merge_cross_cohort.py`)

For each (gene, response) pair appearing in ≥ 3 slices:
1. Split per-slice p-values by cohort.
2. Within-cohort Fisher combination:
   ```
   p_cohort = -2 · Σ log(p_slice)
   ```
   Under H₀ (all p uniform), this is χ²-distributed with df = 2·n_slices.
3. Cross-cohort Fisher combination of the two cohort p-values.
4. Benjamini-Hochberg FDR correction across all (gene, response) pairs →
   combined_q.
5. Direction consistency: check `sign(durbin_delta)` is identical across
   all slices.

### Reportable gate

A (gene, response) pair is reported if:
1. `direction_consistent == True`.
2. `combined_q < 0.05` in BOTH cohort 1 AND cohort 2.
3. Passed L4 (all four diagnostics) in the cohort-1 slices where it
   appears.

### Output

| File | Shape | Content |
|---|---|---|
| `cross_cohort_combined.csv` | (~600, 11) | every (gene, response) pair with cohort p, combined p, combined q |
| `cross_cohort_replicated.csv` | (n_passed, 11) | only pairs that pass the reportable gate |

Schema (11 cols):
```
gene, response, n_slices, n_cohorts, combined_p, direction_consistent,
mean_delta, min_p, slices, cohorts, combined_q
```

Current values: `cross_cohort_replicated.csv` has 7 rows (Phgr1 × 5
responses + Rab8a × malignant + Bcam × malignant + Blnk × fibroblast).


---

## L6 — Orthogonal human validation

### Input

- Public TCGA LUAD expression + clinical (cBioPortal `luad_tcga_gdc`,
  n = 518).
- Public ICB cohort data: IMvigor210, Liu melanoma, lung ICB, bladder ICB.

### Internal processing

`experiments/phgr1_tcga.py`:
1. For each marker gene m in {CD8A, CD8B, GZMB, CD68, STAT1, IRF1, DCN,
   PDGFRA, EPCAM, KRT8, ...}:
   - Spearman ρ(PHGR1, m) across n=518 patients.
   - Bonferroni-corrected p.

`experiments/phgr1_survival.py`:
1. Dichotomise at PHGR1 Q1 vs Q4.
2. Log-rank test → p = 0.0017.
3. Kaplan-Meier medians: Q1 20.0mo vs Q4 22.7mo.
4. Cox proportional hazards:
   - Univariate: HR = 0.587, p = 0.007.
   - Multivariate (AGE, SEX): HR = 0.556, p = 0.004.

`experiments/phgr1_icb.py`:
1. For each ICB cohort, dichotomise at PHGR1 median, compare response
   rates (Fisher's exact), compare survival (Cox).
2. Result: NOT significant in any of 4 cohorts (negative finding).

### Output

| File | Content |
|---|---|
| `phgr1_tcga_validation.json` | n_samples, correlations{marker: {rho, p, group, n}}, high_vs_low{marker: {high_mean, low_mean, u_stat, p}} |
| `phgr1_survival.json` | n_patients, n_events, median_os_low/high, logrank_p, cox_uni_HR, cox_multi_summary |
| `phgr1_melanoma_icb.csv` | per-cohort ICB response statistics |


---

## Inter-layer dependencies

```
L0 (raw)
 │
 ├──→ L1 (encoder) ──── embed_<slice>.npy ───┐
 │                                           │
 ├──→ L2 (matching) ←────────────────────────┤
 │      │                                    │
 │      └──→ matches_<slice>_<gene>.csv      │
 │                                           │
 ├──→ L3 (Durbin) ←─── embed + matches ──────┤
 │      │                                    │
 │      └──→ genome_scan_v2_<slice>.csv      │
 │                                           │
 ├──→ L4 (causal) ←─── scan + matches ───────┘
 │      │
 │      └──→ phgr1_full_diagnostics.json
 │
 ├──→ L5 (cross-cohort) ←─── all 8 slice scans
 │      │
 │      └──→ cross_cohort_replicated.csv  ← headline output
 │
 └──→ L6 (TCGA + ICB) ←─── external data
        │
        └──→ phgr1_tcga_validation.json, phgr1_survival.json
```

Each layer's output is **versioned and immutable**: re-running L3 must
not change L0 or L1 outputs. If L3 logic changes, L3 outputs get a new
filename suffix (e.g. `_v3`).


---

## v2.1 fix hooks

The five v2.1 fixes do not add new layers; they re-run existing layers
under alternative assumptions and write parallel output files.

| Fix | Hooks into | Replaces / parallel | New output |
|---|---|---|---|
| Fix 1 AIPW | L3+L4 | parallel to Durbin δ | `sensitivity/aipw_phgr1.csv` |
| Fix 2 Leiden niches | L0 (niche_id derivation) | parallel to KMeans k=12 | `sensitivity/niche_leiden_sensitivity.csv` |
| Fix 3 Probe transfer | L1 (encoder evaluation) | parallel to cos(z1,z2) | `sensitivity/probe_transfer.json` |
| Fix 4 Matching buffer | L2 | parallel to default matcher | `sensitivity/matching_buffer_sensitivity.csv` |
| Fix 5 GraphSAGE baseline | L1 | parallel to set-pooling encoder | `sensitivity/sage_vs_setpooling.csv` |

v2's primary outputs (`cross_cohort_replicated.csv`, `phgr1_tcga_validation.json`,
`phgr1_survival.json`) are unchanged. v2.1's sensitivity files are read by
`SINGLE_MASTER_perturbgnn_v2.ipynb` §15 to produce the robustness table.
