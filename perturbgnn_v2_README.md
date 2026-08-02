# PerturbGNN v2 — Embedding-matched causal propagation framework

**Paper**: *Causal identification of non-cell-autonomous perturbation effects
in spatial CRISPR screens via embedding-matched spatial Durbin models.*

This directory holds the **v2 module** of the `2026_pebble` repository. It is a
sibling of the original `src/perturbradius/` ring-decay package. The two are
intentionally separate: `perturbradius` measures a distance-decay length scale;
`perturbgnn_v2` replaces that framing with causal identification (DiD + IV +
Rosenbaum Γ*) plus embedding-matched controls and a cross-cohort replication
gate. They share no API.

- Pre-executed walkthrough: [`tutorials/04_perturbgnn_v2_phgr1_cross_cohort.ipynb`](tutorials/04_perturbgnn_v2_phgr1_cross_cohort.ipynb)
- Frozen small tables for the walkthrough: [`tutorials/data_v2/`](tutorials/data_v2/)
- Source code: [`perturbgnn_v2_src/`](perturbgnn_v2_src/)
- Figure scripts: [`perturbgnn_v2_experiments/`](perturbgnn_v2_experiments/)
- Figures (22 PNGs): [`perturbgnn_v2_figures/`](perturbgnn_v2_figures/)
- Design and paper drafts: [`perturbgnn_v2_docs/`](perturbgnn_v2_docs/)

Large inputs (`.h5ad` / `.npy` / `.pt`) are deliberately not versioned. They
sit on the analysis server under `/mnt/data/xuzh/spac_seq/processed/` and are
reconstructed by the Phase 0 loader from public raw matrices. See **Data
deposition** below.

---

## Key result

**Phgr1** is a multi-response non-cell-autonomous causal hub in SPAC-seq,
replicated across **2 cohorts / 7 slices / 5 responses** (combined
Fisher q < 1e-19, min q = 3e-60 for fibroblast). TCGA LUAD (n=516) confirms:
multivariate Cox HR = 0.556 (p = 0.004), independent of age and sex.

The v1 paper's BLNK finding does **not** replicate under the v2 causal
framework. **79% of v1-significant hits are v2-negative** (false positives).

Phgr1 does **not** predict ICB response in four independent cohorts — reported
as an honest negative result, not buried.

---

## Pipeline (5 layers)

```
Layer 0  Data + Tissue Graph
Layer 1  Supervised Cross-Modal GNN Encoder     → 64-dim spot embedding (cos = 0.9997)
Layer 2  Embedding-Space Matched Control         → covariate-balanced null (SMD < 0.13)
Layer 3  Spatial Network Models                  → Durbin δ + resistance distance
Layer 4  Causal Diagnostics                      → DiD + IV + Rosenbaum Γ*
```

Each layer is independently novel; the causal claims depend on all five.

---

## Data

| Cohort | Slices | Total bins | NTC bins | Guides / gene | Top hit |
|---|---|---|---|---|---|
| 1 (lung metastasis) | M001, M002, M003 | 1.1 M | 950 | 1 | Phgr1 (M001, M002) |
| 2 (multiple-section) | subQ-1 … subQ-5 | 3.06 M | 11.6 K | 2 | Phgr1, Bcam, Rab8a |
| **Total** | **8 slices** | **4.16 M** | **12.6 K** | — | — |

A third cohort (spatiotemporal T-cell, Day 7 / Day 10, rep 1) is loaded as a
temporal-trend auxiliary (`tutorials/data_v2/cohort3_timecourse.csv`). It does
not contain sgPhgr1 in its panel and is therefore not part of the
cross-cohort Phgr1 claim; see `perturbgnn_v2_docs/COHORT3_LIMITATIONS.md`.

---

## Quick start

The pipeline assumes the public SPAC-seq raw matrices have been downloaded
into a single root directory. Set the root before running:

```bash
export PERTURBGNN_DATA_ROOT=/path/to/spac_seq/processed/extracted
conda activate ppo-gnn   # torch 2.10 + PyG 2.7
cd perturbgnn_v2_src
export PYTHONPATH=.

# Phase 0–1: data + embedding (~40 min on 1× V100)
python -m perturbgnn_v2.data.raw_h5 $PERTURBGNN_DATA_ROOT
python -m perturbgnn_v2.embedding.train_bt --train M001,M002 --val M003 --epochs 100

# Phase 3–4: causal scan (~30 min / slice)
python -m perturbgnn_v2.causal.scan --slices M002 --fast
python ../perturbgnn_v2_experiments/scan_cohort2.py subQ-1

# Cross-cohort merge + headline figures
python ../perturbgnn_v2_experiments/merge_cross_cohort.py
python ../perturbgnn_v2_experiments/F9_cross_cohort_heatmap.py
python ../perturbgnn_v2_experiments/F11_phgr1_triple_evidence.py
```

If you only want to read results, the pre-computed small tables in
`tutorials/data_v2/` are sufficient — no environment setup is needed.

---

## Cross-cohort replicated hits (7 pairs, combined q < 0.05)

| Gene | Response | Slices | δ | combined q |
|---|---|---|---|---|
| **Phgr1** | fibroblast | 7 | +0.43 | 3.0e-60 |
| **Phgr1** | macrophage | 7 | +0.41 | 1.1e-41 |
| **Phgr1** | hypoxia | 7 | +0.44 | 6.1e-21 |
| **Phgr1** | endothelial | 7 | -0.24 | 2.0e-19 |
| Rab8a | malignant | 6 | +0.49 | 5.0e-56 |
| Bcam | malignant | 6 | +0.47 | 4.4e-11 |
| Blnk | fibroblast | 6 | +0.29 | 7.5e-6 |

Reproduced from `tutorials/data_v2/cross_cohort_replicated.csv`.

---

## Honest negative results

- 79% of v1-significant hits fail under the v2 causal framework
  (`perturbgnn_v2_figures/F17_v1_vs_v2.png`).
- Phgr1 does not predict ICB response in IMvigor210, melanoma, lung, and
  bladder cohorts (`perturbgnn_v2_figures/F18_phgr1_icb.png`).
- Bcam has a large effect size (δ = +0.47) but an opposite-sign Cox HR —
  included as a counter-example showing the framework does not chase δ.

These are reported explicitly, not omitted.

---

## Comparison with v1 / `perturbradius`

| Aspect | `perturbradius` (v1) | `perturbgnn_v2` |
|---|---|---|
| Framing | propagation radius | causal identification |
| Statistical framework | label-shuffle permutation | DiD + IV + Rosenbaum Γ* |
| Confound control | NTC noise floor | embedding-matched control |
| Spatial model | 1D concentric ring | 2D resistance-distance Durbin |
| Replication | single cohort, 3 slices | cross-cohort, 2 cohorts / 8 slices |
| Headline finding | BLNK (single slice) | **Phgr1** (7 slices, 2 cohorts) |
| TCGA Cox HR | 0.744 (BLNK) | **0.556 (Phgr1)** |
| Reportable gate | ring monotonicity + R² ≥ 0.5 | cross-cohort + DiD + IV + Γ* |

The two modules coexist because v1's ring-decay length scale remains a useful
**descriptive** diagnostic; v2 is what is reported when a causal claim is
required.

---

## Limitations

See `perturbgnn_v2_docs/COHORT3_LIMITATIONS.md` and the Discussion section of
`perturbgnn_v2_docs/PAPER_DRAFT_FINAL.md`. In short:

1. Cohort 1 has one guide per gene (off-target cannot be fully excluded);
   cohort 2 has two guides per gene, which mitigates this for Phgr1.
2. No rescue experiments yet (JAK-inhibitor protocol in
   `perturbgnn_v2_docs/BIOLOGY_BACKLOG.md`).
3. Resistance distance is a path-sampled approximation, not exact circuit
   resistance.
4. TCGA is correlative; the causal evidence lives in the SPAC-seq framework.

---

## Data deposition

- Raw SPAC-seq matrices: public portal `spac.pku-genomics.org` (Zhang et al.,
  *Cell* 2026, lung metastasis; multisection; spatiotemporal T cell).
- TCGA LUAD: cBioPortal `luad_tcga_gdc` (n = 516).
- This repository: code + figure scripts + small frozen tables. Raw
  expression matrices and per-spot embeddings are not versioned (multi-GB);
  they regenerate from the public raw inputs via Phase 0 above.

---

## Citation

```bibtex
@software{perturbgnn_v2_2026,
  title  = {PerturbGNN v2: Embedding-matched causal propagation framework},
  author = {Xu Zhenghao},
  year   = {2026},
  url    = {https://github.com/Xu-zhenghao-zzz/2026_pebble/tree/v2-perturbgnn-20260802}
}
```

PerturbGNN v2 is released under the MIT License, matching the parent
repository.
