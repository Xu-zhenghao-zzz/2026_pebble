# Fix 4 (Matching Buffer) — Infeasible on SPAC-seq

## TL;DR

The 80µm spatial buffer proposed in MODELING_DEEP_DIVE.md Fix 4 to
address NTC pool contamination is **infeasible on SPAC-seq data**.
The source bin density is so high that filtering out NTC bins within
80µm of any source leaves only 0.4-3.7% of the original NTC pool,
making matched control impossible on M002 and M003.

This is itself a methodological finding: the contamination concern is
real, but the proposed mitigation is incompatible with the data.

---

## What we did

For each slice, exclude any NTC bin within `b = 80 µm` of any source
bin of any gene. Then re-run v2's cosine-NN matching on the filtered
pool.

Code: `perturbgnn_v2_1_experiments/run_matching_buffer.py`
Output: `tutorials/data_v2/sensitivity/matching_buffer_sensitivity.csv`

## Results

| Slice | NTC original | NTC after 80µm buffer | Retention |
|---|---|---|---|
| M001 | 548 | 16 | **2.9%** |
| M002 | 191 | 7 | 3.7% (skip — too few) |
| M003 | 1813 | 8 | 0.4% (skip — too few) |

Only M001 produced numerical results, and even there the SMD on all
covariates is 1.29 (far above the 0.13 threshold) because matching 1471
Phgr1 source bins against 16 NTC bins is structurally under-powered.

## Why this happens

SPAC-seq transduces a high-multiplicity guide library into the tissue.
The result is that source bins (any gene, any guide) cover essentially
the entire imaged region:

- M001: 81,921 source bins out of 273,886 total (29.9%)
- M002: 61,442 source bins out of 358,149 (17.2%)
- M003: 106,226 source bins out of 488,373 (21.8%)

With sources covering 17-30% of all bins at average inter-bin spacing
of ~10 µm, the 80 µm buffer (8-hop radius) excludes essentially all
non-source bins that are within reach of any clone.

## Two interpretations

### Interpretation A: buffer reveals v2's bias

On M001, the 16 surviving NTC bins still produce:
- fibroblast δ = +0.40 (vs v2 baseline +0.62, **65% retained**)
- macrophage δ = +0.29
- hypoxia δ = +0.73
- ifn_response δ = +0.29

Direction concordant with v2 in 5/7 responses. If we trust these 16
NTC bins (questionable given SMD=1.29), then ~65% of v2's δ survives
the most aggressive buffer possible. This is roughly consistent with
the AIPW result (Fix 1) which estimated Durbin overestimates by 2-4×.

### Interpretation B: buffer is the wrong tool

The acceptance criteria from MODELING_DEEP_DIVE.md Fix 4 explicitly
required "matched control pool size drops by no more than 50%". The
actual drop is 97-99.6%, **far beyond the 50% threshold**.

This means the buffer is not a viable sensitivity analysis for this
data — it destroys the matched-control framework rather than stress-
testing it. The reviewer question "is your NTC pool contaminated by
other sources' NCA effects" cannot be answered by buffer exclusion
on SPAC-seq.

## Implications for the paper

### What v2 can claim

v2's Discussion should **acknowledge** the contamination risk but
**not** claim the buffer analysis as evidence:

> *"The NTC pool in SPAC-seq is structurally constrained: source
> bins cover ~25% of all bins, so any NTC bin sits within the NCA
> range of multiple non-target sources. We tested an 80µm exclusion
> buffer around all source bins, but this reduced the NTC pool to
> 0.4-3.7% of its original size, making matched control infeasible.
> The AIPW doubly robust estimator (Supplementary §N) provides an
> alternative robustness check that does not depend on buffer
> exclusion."*

### What v2 cannot claim

v2 cannot claim "buffered matching confirms δ survives" because:
1. SMD = 1.29 means the buffered match is unbalanced
2. M002/M003 cannot run at all
3. The 16 surviving M001 NTC bins may themselves be a biased subset
   (they are the few bins that happen to be far from any clone,
   which may correlate with being in a specific niche type)

### Combined with AIPW (Fix 1)

The honest summary:
- AIPW (Fix 1) shows ~30% of v2's δ magnitude survives doubly robust
  estimation
- Buffer (Fix 4) is infeasible on this data
- v2's NTC contamination remains a stated limitation, mitigated by
  AIPW but not eliminated

## Alternative sensitivity analyses (not pursued)

If buffer exclusion is infeasible, the contamination concern can be
addressed by:

1. **Per-source-gene buffer**: exclude NTC bins within 80µm of the
   **target gene's** sources only (not all sources). This is a weaker
   filter but may retain enough NTC bins.
2. **Distance-decay weighting**: weight each NTC bin by exp(-d/τ)
   where d is distance to nearest source and τ is the decay length.
   Bins close to sources contribute less to the matched control mean.
3. **Synthetic controls from non-source bins**: instead of using NTC
   bins, use bins that are not source for any gene. With 70-80% of
   bins being non-source, this pool is large.

These are documented as future work; not implemented in v2.1.

## Files

- `perturbgnn_v2_1_experiments/run_matching_buffer.py` — buffer analysis runner
- `tutorials/data_v2/sensitivity/matching_buffer_sensitivity.csv` — full numerical results (only M001 has data)
