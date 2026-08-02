# Tutorials

The tutorials are ordered from a synthetic smoke test to three real-data case
studies:

1. [`quickstart.ipynb`](quickstart.ipynb) — synthetic end-to-end API example
   with a known generating length scale.
2. [`01_real_subq_rab8a.ipynb`](01_real_subq_rab8a.ipynb) — raw sparse RNA and
   guide H5 to Rab8a source components, whole-section maps, local hypoxia/IFN
   fields and distance rings in three subcutaneous sections.
3. [`02_real_lung_ccn1.ipynb`](02_real_lung_ccn1.ipynb) — Ccn1-associated
   near-field macrophage enrichment rebuilt from raw M002 H5 axes and official
   annotation/clone CSVs.
4. [`03_real_spatiotemporal_gata3.ipynb`](03_real_spatiotemporal_gata3.ipynb)
   — raw Day-7 RNA/guide H5 and cell-type JSON to T/NK source patches, native-unit
   distance rings and an NTC-referenced Gata3 candidate signal.
5. [`04_perturbgnn_v2_phgr1_cross_cohort.ipynb`](04_perturbgnn_v2_phgr1_cross_cohort.ipynb)
   — v2 causal-identification module: Phgr1 cross-cohort replication across
   7 slices and 5 module-score responses, TCGA LUAD survival, and the v1-vs-v2
   honest-negative comparison. Reads only frozen small tables in
   [`data_v2/`](data_v2/); no multi-GB inputs required.

Tutorials 1–4 use the `perturbradius` ring-decay package. Tutorial 5 uses the
separate `perturbgnn_v2` module documented in
[`../perturbgnn_v2_README.md`](../perturbgnn_v2_README.md). The two modules
share no API; tutorial 5 is fully self-contained on the frozen tables.

All notebooks are committed with tables and figures embedded. Tutorials 2–4
start from the downloaded source matrices and annotations and do not read
[`data/`](data/), which is retained only as a compact audit snapshot of the
earlier result-level tutorials. Tutorial 5 reads only [`data_v2/`](data_v2/),
which versions the small result tables of the v2 analysis. Raw files are never
modified and guide-negative bins are never automatically treated as
unperturbed controls.

Set the data root before execution if the files are stored elsewhere:

```bash
export PERTURBRADIUS_DATA_ROOT=/path/to/downloaded/data
jupyter lab tutorials/
```

The repository CI executes the portable synthetic tutorial and validates that
the pre-executed raw tutorials retain embedded outputs. It cannot re-download
the private multi-gigabyte source matrices on a public runner.
