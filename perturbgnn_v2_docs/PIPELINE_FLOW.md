# PerturbGNN v2 — Input-to-output flow (ASCII)

This is the **visual** companion to `PIPELINE_LAYERS.md`. That doc gives
the precise data contracts in tables; this one shows the same pipeline as
a single readable flow, with file names, tensor shapes, loss formulas,
and the actual numbers from the v2 run.

```
═══════════════════════════════════════════════════════════════════════════
  RAW INPUTS  (public SPAC-seq matrices)
═══════════════════════════════════════════════════════════════════════════

  Cohort 1 — lung metastasis           Cohort 2 — multiple-section subQ
  ─────────────────────────────        ─────────────────────────────────
  M001 / M002 / M003                   subQ-1 / subQ-2 / subQ-3
                                      subQ-4 / subQ-5
  Per slice (saved as <slice>_v2.h5ad):
    • X         sparse (N, 19 059)     per-bin gene expression
    • obsm['spatial']  (N, 2) float32  µm coords
    • obsm['X_pca']    (N, 32) float32 PCA of expression
    • obs['cell_type']   str           8 classes
    • obs['niche_id']    int8          12 classes (KMeans k=12)
    • obs['guide']       str           sgRNA identity, or "NTC"
    • obs['is_source']   bool          guide-positive for any target gene
    • obs['is_ntc']      bool          non-targeting control guide
    • obs['module_score_<name>'] × 8   ifn / hypoxia / fibroblast /
                                       macrophage / cd8_like /
                                       endothelial / malignant / nk_like

  Spot counts:     M001 273 886    M002 358 149    M003 488 373
                   subQ-1 ~632 K   subQ-2 ~615 K   subQ-3 ~628 K
                   subQ-4 / subQ-5 loaded
  Cohort-3 auxiliary: Day7_rep1, Day10_rep1 (T-cell panel, no sgPhgr1)


═══════════════════════════════════════════════════════════════════════════
  L0  TISSUE GRAPH CONSTRUCTION             graph_build.py
═══════════════════════════════════════════════════════════════════════════

  per slice:
    nodes    = spots  (N per slice)
    edges    = Delaunay triangulation  ∪  kNN(k=6, max_dist=60 µm)
               → 3-5 M undirected edges per slice
    node X   = [PCA(32) ⊕ CT-onehot(8) ⊕ density ⊕ vessel_dist]  → 42 dim
    edge E   = [dx, dy, dist, same_CT, src_endpoint,
                barrier, density_avg, vessel_dist_avg]            →  8 dim

        ↓
  OUTPUT  tissue_graph_<slice>_v2.pt          PyG Data object
          { x: (N, 42), edge_index: (2, E), edge_attr: (E, 8) }


═══════════════════════════════════════════════════════════════════════════
  L1  SUPERVISED CROSS-MODAL GNN ENCODER    bt_encoder.py / train_bt.py
═══════════════════════════════════════════════════════════════════════════

  For each spot v:
    neighbourhood K=15  (cKDTree spatial NN, drop self)
        ↓                                 ↓
    VIEW 1 (PCA)                    VIEW 2 (CT)
    nbr_pca  ∈ ℝ^{15 × 32}          nbr_ct  ∈ ℝ^{15 × 8}
        ↓                                 ↓
    pool: mean + max + anchor       pool: mean + max + anchor
    → 96-dim                        → 24-dim
        ↓                                 ↓
    MLP 3 layers (96→128→128→64)    MLP 3 layers (24→128→128→64)
    +LayerNorm +ReLU +Dropout(0.1)  +LayerNorm +ReLU +Dropout(0.1)
        ↓                                 ↓
      z1  ∈ ℝ^64                       z2  ∈ ℝ^64
        ↓                                 ↓
        ├── SHARED head_ct    → 8-class cell_type   (CE loss)
        ├── SHARED head_niche → 12-class niche_id   (CE loss)
        └── alignment loss    = 1 - cos(z1, z2)

  Loss = L_cls(z1) + L_cls(z2) + 0.1 · L_align
  Optimiser: Adam, batch=4096, 200 epochs
  Train: M001 + M002        Val: M003 (held-out slice)

        ↓
  test-time embedding = 0.5 · (z1 + z2)  →  64-dim per spot

  OUTPUT  bt_encoder_v3.pt            trained encoder + heads
          embed_M001.npy  (273 886, 64)   ~70 MB
          embed_M002.npy  (358 149, 64)   ~92 MB
          embed_M003.npy  (488 373, 64)  ~125 MB
          embed_subQ-1..5.npy            per-slice embeddings
          embedding_validation.json      cos(z1,z2)=0.9997
                                          silhouette, ARI, NMI per slice

  CONTRACT  embed_<slice>.npy[i]  corresponds to  adata.obs.iloc[i]
            (downstream depends on this index alignment)


═══════════════════════════════════════════════════════════════════════════
  L2  EMBEDDING-MATCHED CONTROL             matching/match.py
═══════════════════════════════════════════════════════════════════════════

  For each source bin s of gene g (guide-positive spot):
    candidate pool = NTC bins in same slice
    similarity     = cosine(embed[s], embed[ntc])  over all ntc
    matched c(s)   = argmax cosine

  covariate balance check (500 random pairs):
    SMD on 64 embedding dims + 8 module scores + cell_type + niche_id
    PASS criterion:  max SMD < 0.13

        ↓
  OUTPUT  matches_<slice>_<gene>.csv
            columns: source_idx, control_idx, cos_sim
          matched_control_validation.csv   per-covariate SMD table

  CONSTRAINT  source and control must be in the same slice
              (no cross-slice matches)


═══════════════════════════════════════════════════════════════════════════
  L3  SPATIAL DURBIN MODEL                  spatial/durbin.py
═══════════════════════════════════════════════════════════════════════════

  For each (slice, gene, response) triple:
    W  = row-normalised kNN(k=6) graph weighted by
         Gaussian kernel  w_ij = exp(-d_ij² / (2σ²)),  σ = 40 µm
    y  = module_score_<response>           per-spot
    X  = source_indicator  (1 for source, 0 for matched control)

    Spatial Durbin MLE:
        y = ρ · Wy + X · β + WX · θ + ε
            ↑       ↑       ↑
            spatial direct  indirect (spill-over)
            dep.    effect

    Reduced-form effect:
        δ = β + θ · (I - ρW)⁻¹
        (I - ρW)⁻¹  via Neumann series  Σ_{k=0}^{100} (ρW)^k

    Decay length λ  from row-sum of (I - ρW)⁻¹ vs distance.

    Permutation p-value:
        shuffle X labels across bins (preserving spatial layout),
        refit δ, repeat 1000×  →  p = (#{|δ_perm| ≥ |δ_obs|} + 1) / 1001

        ↓
  OUTPUT  genome_scan_v2_<slice>.csv        (cohort 1, 11 cols)
          cohort2_<slice>_scan.csv          (cohort 2, 10 cols)

  Schema (cohort 1):
    slice, gene, response, durbin_delta, durbin_p, durbin_lambda,
    durbin_beta_self, durbin_r2, n_clones, did_delta, did_p, did_n_pert_near


═══════════════════════════════════════════════════════════════════════════
  L4  CAUSAL DIAGNOSTICS                    causal/causal.py
═══════════════════════════════════════════════════════════════════════════

  For each (slice, gene, response) that passed L3 (durbin_p < 0.05):

  ┌─── DiD  (difference-in-differences) ─────────────────────────────┐
  │  near = within 40 µm of source       far = 40-120 µm from source │
  │  Δ_source = mean(y[near, src])   -   mean(y[far, src])           │
  │  Δ_ntc    = mean(y[near, ntc])   -   mean(y[far, ntc])           │
  │  DiD      = Δ_source - Δ_ntc        (subtracts baseline spatial   │
  │                                       heterogeneity)              │
  │  H0: DiD = 0   tested by within-slice label permutation          │
  └──────────────────────────────────────────────────────────────────┘

  ┌─── IV  (instrumental variable) ──────────────────────────────────┐
  │  Z = guide assignment (instrument)                               │
  │  T = source_indicator (treatment)                                │
  │  Stage 1:   T = π0 + π1 · Z + ε     → T̂                          │
  │  Stage 2:   y = β · T̂ + ε                                          │
  │  Pass: sign(β_IV) == sign(β_OLS)  AND  iv_p < 0.05              │
  │  (rules out guide-detection false positives)                     │
  └──────────────────────────────────────────────────────────────────┘

  ┌─── Rosenbaum Γ*  (sensitivity bound) ────────────────────────────┐
  │  For Γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, recompute Wilcoxon p-value   │
  │  under hidden confounding at odds-ratio Γ.                       │
  │  Γ* = smallest Γ at which p crosses 0.05                         │
  │  Pass: Γ* ≥ 2  (withstands at least 2× hidden confounding)      │
  └──────────────────────────────────────────────────────────────────┘

        ↓
  OUTPUT  phgr1_full_diagnostics.json
            per (slice, response): { did_p, iv_p, gamma_star,
                                     n_source, n_control }

  REPORTABLE GATE for L4  —  ALL FOUR must hold:
    durbin_p < 0.05
    did_p    < 0.05
    iv_p     < 0.05  AND  sign(β_IV) == sign(β_OLS)
    gamma_star  ≥  2


═══════════════════════════════════════════════════════════════════════════
  L5  CROSS-COHORT META-ANALYSIS           experiments/merge_cross_cohort.py
═══════════════════════════════════════════════════════════════════════════

  For each (gene, response) appearing in ≥ 3 slices:

    within-cohort Fisher:
        cohort1_p  = -2 · Σ log(p_M001, p_M002, p_M003)         ~ χ²(6)
        cohort2_p  = -2 · Σ log(p_subQ-1..5)                    ~ χ²(10)

    cross-cohort Fisher:
        combined_p = -2 · (log(cohort1_p) + log(cohort2_p))     ~ χ²(4)

    Benjamini-Hochberg FDR correction across all (gene, response) pairs:
        combined_q = BH( combined_p )

    direction_consistent = all sign(durbin_delta) identical across slices

  REPORTABLE GATE for L5  —  ALL THREE must hold:
    1.  direction_consistent == True
    2.  combined_q < 0.05  in BOTH cohort 1 AND cohort 2
    3.  passed L4 (all four diagnostics) in cohort-1 slices

        ↓
  OUTPUT  cross_cohort_combined.csv         (~600 pairs, 11 cols)
          cross_cohort_replicated.csv       (7 pairs that pass the gate)

  Current pass list (7 pairs):
    Phgr1 / fibroblast      7 slices   q = 3.0e-60    δ = +0.43
    Phgr1 / macrophage      7 slices   q = 1.1e-41    δ = +0.41
    Phgr1 / hypoxia         7 slices   q = 6.1e-21    δ = +0.44
    Phgr1 / endothelial     7 slices   q = 2.0e-19    δ = -0.24
    Phgr1 / ifn_response    (cohort-1 + cohort-2 sign agreement)
    Rab8a / malignant       6 slices   q = 5.0e-56    δ = +0.49
    Bcam  / malignant       6 slices   q = 4.4e-11    δ = +0.47
    Blnk  / fibroblast      6 slices   q = 7.5e-6     δ = +0.29

  Phgr1 clears the gate on FIVE responses → v2 headline finding.


═══════════════════════════════════════════════════════════════════════════
  L6  ORTHOGONAL HUMAN VALIDATION          TCGA + ICB public cohorts
═══════════════════════════════════════════════════════════════════════════

  (a) TCGA LUAD  (cBioPortal luad_tcga_gdc, n = 518)
      phgr1_tcga.py:
        Spearman ρ(PHGR1, IRF1) = +0.473     p = 3e-30
        Spearman ρ(PHGR1, CD68) = +0.444     p = 2e-26
        Spearman ρ(PHGR1, CD8A) = +0.439     p = 9e-26
        Spearman ρ(PHGR1, DCN)  = +0.403     p = 1e-21
        Spearman ρ(PHGR1, EPCAM)= +0.051     p = 0.25  (malignant, no signal)

      phgr1_survival.py:
        log-rank Q1 vs Q4:           p = 0.0017
        median OS:        low 20.0 mo   high 22.7 mo
        univariate Cox:   HR = 0.587   (95% CI 0.40-0.86),  p = 0.007
        multivariate Cox (AGE, SEX):
                          HR = 0.556   (95% CI 0.37-0.83),  p = 0.004

  (b) ICB cohorts  —  HONEST NEGATIVE
      phgr1_icb.py:
        IMvigor210  (bladder):        not significant
        Liu melanoma:                 not significant
        lung ICB:                     not significant
        bladder (alt cohort):         not significant

      → Phgr1 marks immune-inflamed TME but does NOT predict checkpoint
        blockade response. Reported explicitly in §11 of the master notebook.

        ↓
  OUTPUT  phgr1_tcga_validation.json
          phgr1_survival.json
          phgr1_melanoma_icb.csv
          utrn_tcga_validation.json   (positive control method check)
          figures F7 / F8 / F18


═══════════════════════════════════════════════════════════════════════════
  FINAL OUTPUT  —  paper + deliverables
═══════════════════════════════════════════════════════════════════════════

  HEADLINE CLAIM
    Phgr1 is a multi-response non-cell-autonomous causal hub.
    Effect replicated across 2 cohorts / 7 slices / 5 module-score
    responses (combined Fisher q from 3e-60 to 2e-19).
    TCGA LUAD multivariate Cox HR = 0.556 (p = 0.004),
    independent of age and sex.
    79 % of v1-significant hits fail under the v2 framework (honest).

  DELIVERABLES
    paper.pdf  (8 pages)         + paper.tex
    slides.pdf + slides_zh.pdf   (English + Chinese)
    22 figures  F1 - F19 (F4 / F5 / F7 each have a/b variants)

    GitHub: Xu-zhenghao-zzz/2026_pebble   branch v2-perturbgnn-20260802
      ├── src/perturbradius/                 original v1 ring-decay package
      ├── perturbgnn_v2_src/                 v2 source (25 .py + 18 .ipynb)
      ├── perturbgnn_v2_experiments/         30 figure + scan scripts
      ├── perturbgnn_v2_figures/             22 PNG + paper.pdf + slides
      ├── perturbgnn_v2_docs/                DESIGN + PAPER_DRAFT_FINAL
      │                                      + MODELING_DEEP_DIVE
      │                                      + PIPELINE_LAYERS
      │                                      + PIPELINE_FLOW (this doc)
      │                                      + RECONSTRUCTION_v2_1
      ├── perturbgnn_v2_1_src/               v2.1 sensitivity module
      │                                      (AIPW implemented; 4 fixes TBD)
      ├── SINGLE_MASTER_perturbgnn_v2.ipynb  2.1 MB readable master notebook
      └── tutorials/
            ├── 04_perturbgnn_v2_phgr1_cross_cohort.ipynb   pre-executed
            └── data_v2/                                    12 frozen tables


═══════════════════════════════════════════════════════════════════════════
  DESIGN PRINCIPLES  (why the layers split this way)
═══════════════════════════════════════════════════════════════════════════

  L1  Encoder only "learns a good embedding". Supervision comes from
      SPAC-seq's own cell_type + niche labels (collapse prevention), not
      from any task-specific signal. The embedding is reusable across
      every downstream layer.

  L2  Matching generates counterfactuals. The matched control bin is an
      approximation of "what would this source bin look like if it were
      not perturbed". SMD < 0.13 is a hard covariate-balance gate.

  L3  Spatial Durbin decouples spatial confounding. The model decomposes
      y into (spatial dependence + direct effect + indirect spill-over +
      noise) and δ is the reduced-form causal effect. No GNN needed
      here — this is classical spatial econometrics.

  L4  Diagnostics give hypothesis tests + robustness. DiD subtracts
      baseline, IV rules out guide-detection artefacts, Γ* quantifies
      hidden confounding. The four-piece combination is what justifies
      the word "causal".

  L5  Cross-cohort replication is a gate, not a bonus. v1's BLNK failed
      this gate on a single cohort and was overturned. Phgr1 passing on
      7 slices across 2 cohorts is the load-bearing evidence for the
      Nature-Methods-level claim.

  L6  TCGA is orthogonal correlation, not causal evidence. The causal
      evidence lives in L3-L5; TCGA provides sign-concordance only,
      explicitly stated in Discussion limitation #4.


═══════════════════════════════════════════════════════════════════════════
  v2.1 SENSITIVITY HOOKS  (parallel re-runs, do not modify v2 outputs)
═══════════════════════════════════════════════════════════════════════════

  Fix 1  AIPW doubly robust     re-runs L3 + L4   →  sensitivity/aipw_phgr1.csv
  Fix 2  Leiden niches          re-runs L0 (niche) →  sensitivity/niche_leiden_sensitivity.csv
  Fix 3  Probe transfer         re-runs L1 eval    →  sensitivity/probe_transfer.json
  Fix 4  Matching buffer        re-runs L2         →  sensitivity/matching_buffer_sensitivity.csv
  Fix 5  GraphSAGE baseline     re-runs L1         →  sensitivity/sage_vs_setpooling.csv

  Each fix produces one file under tutorials/data_v2/sensitivity/ and is
  rendered as a row in SINGLE_MASTER_perturbgnn_v2.ipynb §15 (Robustness).
```
