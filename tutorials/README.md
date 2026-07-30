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

All notebooks are committed with tables and figures embedded. The three real
notebooks start from the downloaded source matrices and annotations and do not
read [`data/`](data/), which is retained only as a compact audit snapshot of the
earlier result-level tutorials. Raw files are never modified and guide-negative
bins are never automatically treated as unperturbed controls.

Set the data root before execution if the files are stored elsewhere:

```bash
export PERTURBRADIUS_DATA_ROOT=/path/to/downloaded/data
jupyter lab tutorials/
```

The repository CI executes the portable synthetic tutorial and validates that
the pre-executed raw tutorials retain embedded outputs. It cannot re-download
the private multi-gigabyte source matrices on a public runner.
