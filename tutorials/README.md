# Tutorials

The tutorials are ordered from a synthetic smoke test to three real-data case
studies:

1. [`quickstart.ipynb`](quickstart.ipynb) — synthetic end-to-end API example
   with a known generating length scale.
2. [`01_real_subq_rab8a.ipynb`](01_real_subq_rab8a.ipynb) — Rab8a-associated
   hypoxic, IFN-low malignant state in the five-section subcutaneous SPAC
   cohort.
3. [`02_real_lung_ccn1.ipynb`](02_real_lung_ccn1.ipynb) — Ccn1-associated
   near-field macrophage enrichment in the annotated lung cohort.
4. [`03_real_spatiotemporal_gata3.ipynb`](03_real_spatiotemporal_gata3.ipynb)
   — a Gata3 example that illustrates why the best-covered signal still fails
   the formal radius evidence gate.

All notebooks are committed with their tables and figures embedded. The
real-data notebooks use compact, frozen derived tables in [`data/`](data/) so
that they remain viewable and executable without downloading tens of
gigabytes. They reproduce candidate-level results, not the upstream raw-matrix
processing.

