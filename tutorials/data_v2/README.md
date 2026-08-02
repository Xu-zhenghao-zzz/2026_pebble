# Frozen PerturbGNN v2 result tables

These small tables and JSONs are compact derivatives of the eight-slice
PerturbGNN v2 analysis on the SPAC-seq lung metastasis + multisection +
spatiotemporal cohorts. Raw data were treated as read-only. The files retain
only fields used by the v2 tutorial figures and tables and are small enough
to version with the package.

`04_perturbgnn_v2_phgr1_cross_cohort.ipynb` reads only this directory; it
does not require the multi-GB `.h5ad` / `.npy` inputs.

## Files

| File | Rows | Source script (in `perturbgnn_v2_experiments/`) | Purpose |
|---|---|---|---|
| `cross_cohort_replicated.csv` | 7 | `merge_cross_cohort.py` | Headline table — genes that replicate across cohort 1 + cohort 2 with combined q < 0.05 |
| `cross_cohort_combined.csv` | ~600 | `scan_cohort2.py` + `merge_cross_cohort.py` | Per-gene, per-response, per-slice scan outputs that fed the Fisher combination |
| `genome_scan_v2_replicated.csv` | 7 | `merge_3slices.py` | Within-cohort-1 replication across M001/M002/M003 |
| `phgr1_tcga_validation.json` | — | `phgr1_tcga.py` | TCGA LUAD (n=516) Spearman correlations of PHGR1 vs IRF1, CD8A, CD68, DCN |
| `phgr1_survival.json` | — | `phgr1_survival.py` | Univariate + multivariate Cox, log-rank, Q1 vs Q4 |
| `phgr1_full_diagnostics.json` | — | `phgr1_case_study.py` | Per-slice effect, DiD, IV, Rosenbaum Γ* for Phgr1 |
| `utrn_tcga_validation.json` | — | `utrn_tcga.py` | Utrn positive-control TCGA correlations |
| `embedding_validation.json` | — | `train_bt.py` + `validate_bt.py` | Cross-modal encoder quality (cosine, retrieval) |
| `synthetic_benchmark.csv` | — | `synthetic_benchmark.py` | Power calibration on synthetic ground truth |
| `cohort3_timecourse.csv` | ~150 | `cohort3_timecourse.py` | Day 7 / Day 10 cohort 3 trends (auxiliary, no Phgr1 in panel) |
| `v1_vsntc_subQ-1.csv` | — | `v1_vs_v2_compare.py` | v1 vs-NTC scan on subQ-1, for the v1-vs-v2 honesty check |

## Provenance

These tables are analysis derivatives. They do not contain raw expression
matrices or patient-identifying information. To rebuild them end-to-end,
re-run the Phase 0–4 pipeline documented in
[`../perturbgnn_v2_README.md`](../perturbgnn_v2_README.md) against the public
SPAC-seq raw matrices.
