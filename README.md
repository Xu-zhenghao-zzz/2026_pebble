# PerturbRadius

PerturbRadius is a small, CPU-first Python package for clone-centric spatial
perturbation-scale analysis. It calls guide-labelled source components, measures
distance from their boundaries, builds physical distance rings, estimates an
NTC-adjusted response curve and fits a diagnostic exponential decay.

The package deliberately separates a numerical fit from a reportable biological
radius. A radius is rejected when essential evidence such as independent
guides, NTC components, biological replicates or a monotonic response is
missing.

## Installation

```bash
git clone https://github.com/Xu-zhenghao-zzz/2026_pebble.git
cd 2026_pebble
python -m pip install -e .
```

Python 3.10 or newer is required.

## Input table

The public API accepts a `pandas.DataFrame` with one row per spatial bin or
cell. Required columns are:

| Column | Meaning |
|---|---|
| `section` | Tissue section identifier |
| `x_um`, `y_um` | Spatial coordinates in micrometres |
| `target_gene` | Perturbation target for source rows |
| `guide` | Guide identity for source rows |
| `is_source` | Whether the row is a guide-confident source |
| `is_ntc` | Whether the source carries a non-targeting guide |
| response column | Numeric phenotype, for example a pathway score |

A biological-replicate column such as `mouse` is optional for descriptive
analysis but required by the formal radius evidence gate.

Guide non-detection must not be interpreted as proof that a row is unperturbed.
Source `component` is an algorithmic spatial unit and is not automatically an
independent biological clone.

## Quick start

```python
import perturbradius as pr

result = pr.analyze_radius(
    data,
    perturbation="GeneA",
    response_col="response",
    ring_edges=(0, 20, 40, 80, 120, 180),
    eps_um=15,
    min_samples=4,
    min_source_bins=15,
    bin_size_um=10,
    replicate_col="mouse",
)

print(result.effect_table)
print(result.model_fit)
print(result.radius_decision)

ax = pr.plot_distance_response(result)
```

The synthetic API example is
[`tutorials/quickstart.ipynb`](tutorials/quickstart.ipynb). Three additional
pre-executed real-data case studies cover subQ Rab8a, lung Ccn1 and
spatiotemporal Gata3. See the [`tutorials` index](tutorials/README.md). All
notebooks embed their QC maps, result tables and interpretation directly in
the cells.

## Public API

- `validate_data`
- `call_components`
- `build_distance_rings`
- `estimate_distance_response`
- `fit_exponential`
- `assess_radius_support`
- `analyze_radius`
- `plot_distance_response`

The high-level `analyze_radius` workflow excludes all source rows from the
target pool and excludes multi-source target rows from the default effect
estimate.

## Interpretation

`result.model_fit` is always a diagnostic fit. A numerical length scale should
be treated as biologically reportable only when
`result.radius_decision.reportable` is `True`. The current gate checks:

- at least two independent guides;
- at least two NTC source components;
- at least two biological replicates;
- complete NTC support in the distance rings;
- a successful monotonic exponential fit;
- model `R² >= 0.5`.

These checks do not replace experimental rescue, orthogonal validation,
tissue-edge auditing or prospective study design.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

PerturbRadius is released under the MIT License.
