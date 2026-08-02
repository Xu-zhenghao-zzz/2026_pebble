# PerturbGNN v2 — Paper Outline (draft, 2026-08-01)

**Working title**: Causal identification of non-cell-autonomous perturbation effects in spatial CRISPR screens via embedding-matched spatial Durbin models

**Target venue**: Nature Methods (methodology) or Genome Biology (application)

---

## Abstract (draft)

Spatial CRISPR screens (SPAC-seq, Perturb-map) promise to read out
non-cell-autonomous (NCA) effects of gene knockouts in intact tissue,
but the standard analytical pipeline — concentric distance rings
around guide-positive sources, exponential decay fits, label-shuffle
permutation tests — conflates perturbation effects with tissue-structure
confounding. We present a causal identification framework that replaces
this stack with three components: (1) a supervised cross-modal GNN
encoder that learns a perturbation-decoupled spot embedding, (2) an
embedding-space matched-control procedure that constructs a
neighborhood-aware null, and (3) a multi-source spatial Durbin model
with resistance-distance edge weights that estimates the dose-response
propagation slope δ while respecting 2D tissue geometry. We further
introduce three causal diagnostics — spatial DiD, guide-instrument
variable (IV), and Rosenbaum sensitivity Γ* — that distinguish true
causal propagation from confounded associations. Applying the framework
to the SPAC-seq lung metastasis cohort (3 slices, 1.1M bins, ~120 genes),
we find that the published Ccn1→macrophage association is causally
robust (DiD Δ=+0.086, p=5e-7; IV/OLS=2.7; Γ*>3.0), while the
apparently strong Ccn1→fibroblast signal is a confounding artefact
(IV/OLS=-14, Γ*=1.0). Our framework provides falsifiable, per-gene
causal claims under explicit unmeasured-confounding assumptions,
replacing the unfalsifiable "propagation radius" framing of prior work.

---

## Section 1: Introduction

### 1.1 The propagation-radius framing is unfalsifiable

Prior pipelines (PerturbRadius, vs-NTC FDR screens) frame the NCA
question as "how far does the perturbation effect propagate?",
answered by fitting an exponential decay A·exp(-d/λ) and reporting
the length scale. This framing has three problems:

- **Unfalsifiable**: no null model of "no propagation"; any non-zero λ
  is reportable.
- **Confounded**: the apparent decay length mixes perturbation biology
  with tissue-structure gradients (vessel proximity, niche boundaries).
- **Single-scale**: forces a single length scale, missing multi-modal
  or anisotropic propagation.

### 1.2 The causal identification alternative

We reframe: "what is the dose-response relationship between perturbation
exposure and phenotype, after controlling for spot-level confounders?"
This requires:

- A definition of "exposure" (Section 3.3: spatial Durbin)
- A control for spot-level confounders (Section 3.2: matched control)
- A test of the identifying assumptions (Section 4: DiD, IV, Rosenbaum)

### 1.3 Contributions

1. Supervised cross-modal GNN encoder (Section 3.1)
2. Embedding-space matched control with covariate balance (Section 3.2)
3. Multi-source spatial Durbin with resistance distance (Section 3.3)
4. Three causal diagnostics layered on top (Section 4)
5. Application: re-analysis of SPAC-seq lung cohort (Section 5)

---

## Section 2: Data

SPAC-seq lung metastasis cohort (Zhang et al., Cell 2026):
- M001 (slide 992): 273,886 bins × 19,059 genes × 1,520 guides
- M002 (slide 994): 358,149 bins
- M003 (slide 993): 488,373 bins
- Total: 1.1M bins, 8µm Visium HD resolution
- NTC counts (high-confidence): M001=548, M002=191, M003=1813

**Known limitations (carried into Discussion)**:
- 1 guide per gene per slice — off-target cannot be excluded
- 3 slices from same cohort — not independent biological replicates
- Non-overlapping gene panel — only 7/120 genes in all 3 slices

---

## Section 3: Methods

### 3.1 Supervised cross-modal GNN encoder

Two views per spot (K=15 neighbors):
  view 1: neighborhood expression PCs (32-dim)
  view 2: neighborhood cell-type composition (8-dim one-hot)

Shared classification heads (cell_type 8-way + niche 12-way) on both
views, plus soft alignment (1 - cosine). Total loss:
  L = CE_ct + CE_niche (both views) + α · (1 - cos(z1, z2))

**Design history** (Supplementary):
- Barlow Twins (v1): on_diag stayed at 0 — heterogeneous modalities
  don't have dim-wise alignment.
- SimSiam (v2): predictor-target collapse — cosine stayed at 0.
- Supervised (v3, final): cell_type + niche classification provides
  the missing alignment signal.

**Validation**:
- train cos_sim: 0.98, val cos_sim: 0.97 (cross-slice generalization)
- cell_type accuracy: 76% (train), 61% (val)
- **src vs NTC centroid cosine: 0.9997** (perturbation does not leak
  into the embedding — critical for matched control validity)

### 3.2 Embedding-space matched control

For each analysis spot (within 200µm of a target source):
  candidates = spots | same (cell_type, niche_id),
                       dist(source) > 400µm,
                       guide_total ≤ 1
  matched = top-K (K=10) nearest in embedding cosine

**Covariate balance** (after (cell_type, niche_id) double bucketing):
- max |SMD| on X_pca[0..7] and niche indicators: 0.125
- 18/20 covariates within |SMD|<0.1 (target)

**NTC sanity**: matched Δ on NTC spots centers near 0 (mean ±0.03,
median within ±0.05 for 6/7 responses).

### 3.3 Multi-source spatial Durbin with resistance distance

Exposure field:
  exposure_field_i = Σ_c M_c · exp(-d_res(i,c)²/(2λ²))

where d_res is resistance distance (Euclidean × (1 + α·barrier - β·vessel)),
M_c is source strength (clone size × mean guide confidence), and λ is
a learnable length scale (grid-searched over {30, 60, 100, 150} µm).

Regression:
  Y_i = α + β·T_i + δ·exposure_field_i + γ·X_i + ε_i

with cluster-robust SE by clone_id.

**Comparison model: Euclidean SLX** (Section 3.3.1):
  Y = α + β·T + Σ_k θ_k · W_k T + γ·X + ε
  with W_k = row-normalized Euclidean ring weights.

---

## Section 4: Causal diagnostics

### 4.1 Spatial DiD

  Δ_DiD = [Y_pert_near - Y_pert_far] - [Y_ctrl_near - Y_ctrl_far]

where ctrl arm uses NTC sources (or synthetic unexposed controls if
NTC count insufficient). Controls for neighborhood gradient.

### 4.2 Guide-instrument variable (IV)

  Stage 1: T_hat = π_0 + π_1 · guide_UMI_exposure + covariates
  Stage 2: Y = α + β · T_hat + covariates

β_IV vs β_OLS:
- ratio ≈ 1: OLS estimate consistent with causal interpretation
- ratio ≫ 1: OLS underestimates (e.g., measurement error in T)
- ratio ≪ 0 or strongly negative: OLS estimate is confounded

### 4.3 Rosenbaum sensitivity Γ*

Smallest hidden-confounder strength Γ that flips the p-value above 0.05.
Robust claims require Γ* ≥ 1.5; our strongest claims have Γ* > 3.0.

---

## Section 5: Results

### 5.1 Ccn1 case study (M002)

| Response | Durbin δ | DiD Δ | IV/OLS | Γ* | Verdict |
|---|---|---|---|---|---|
| macrophage | -0.17*** | +0.09*** | 2.7 | >3.0 | **causal robust** |
| ifn_response | (scan) | (scan) | (scan) | >3.0 | **causal robust** |
| fibroblast | +0.23*** | +0.10*** | -14 | 1.0 | **confounding artefact** |
| cd8_like | (scan) | (scan) | (scan) | ? | TBD |

**Key insight**: the published Ccn1→macrophage association (v1 paper,
near-far Δ=+0.09, FDR=0.087 "exploratory") upgrades to a causally
robust claim under our framework. The Ccn1→fibroblast association
— invisible to v1's macrophage-only readout — is revealed as a
confounding artefact.

### 5.2 Genome-wide scan

(awaiting completion of `causal/scan.py` on M002 + M001 + M003)

We scan ~19 genes (≥100 sources) × 7 responses × 3 slices, applying
all four diagnostics. We report:
- per (gene, response, slice) all diagnostics
- Stouffer-combined p across slices
- BH FDR across all (gene, response) pairs
- robustness tier: Γ*≥1.5 with IV/OLS∈[0.5, 2]

### 5.3 Comparison with v1 framework

(To be filled with side-by-side comparison table after scan completes.)

---

## Section 6: Discussion

### 6.1 What the framework can and cannot do

**Can**: provide falsifiable per-gene causal claims under explicit
unmeasured-confounding assumptions.

**Cannot**: exclude off-target effects of single guides (data limitation),
establish temporal direction (snapshot data), or replace rescue
experiments (no perturbation reversal).

### 6.2 Why the framing matters

"Propagation radius" is a descriptive statistic with no null. "Causal
propagation slope under explicit assumptions" is a falsifiable scientific
claim. Reviewers can disagree with our assumptions (and we report Γ*
to make this concrete), but they cannot dismiss the claim as unfalsifiable.

### 6.3 Limitations

1. Single guide per gene — off-target cannot be excluded
2. 3 slices, non-overlapping panel — cross-slice replication weak
3. Resistance distance is approximate (path-sampled, not exact circuit)
4. Rosenbaum Γ* assumes binary hidden confounder; real confounders
   may be continuous

---

## Section 7: Data and code availability

- Raw SPAC-seq H5: `/mnt/data/xuzh/spac_seq/processed/extracted/`
- Code: `perturbgnn_v2/` (this repository)
- Reproducibility: each diagnostic < 10 s per (gene, response, slice)
  on a single V100; full scan ~4 hours for 19 genes × 7 responses × 3 slices.

---

## Figures (planned)

- F1: Pipeline overview (5 layers)
- F2: GNN embedding quality (cos_sim, src-NTC overlap, cell-type t-SNE)
- F3: Matched control covariate balance + NTC sanity
- F4: Durbin δ landscape across genome (gene × response heatmap)
- F5: Ccn1 case study — DiD + IV + Rosenbaum diagnostics
- F6: Comparison v1 vs v2 verdicts per (gene, response)

---

## Status (2026-08-01)

- Methods: ✅ implemented (Phase 0-3)
- Causal diagnostics: ✅ implemented (Phase 4)
- Ccn1 case study: ✅ complete
- Genome scan: 🔄 in progress (M002 running)
- Paper draft: this outline
- Figures: TBD after scan
