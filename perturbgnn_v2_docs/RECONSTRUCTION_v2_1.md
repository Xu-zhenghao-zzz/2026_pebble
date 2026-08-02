# PerturbGNN v2.1 — Reconstruction plan

This is a **modelling-layer reconstruction** of v2. v2.1 does not replace v2
— v2 remains the canonical submission. v2.1 is a parallel sensitivity module
that quantifies how much of v2's headline result depends on each modelling
choice that a hostile reviewer would attack.

The goal of v2.1 is **not to find a stronger Phgr1 effect**. It is to show
that the v2 result survives each of the five attacks below. If the result
holds, the v2 paper gains a "robustness" section that pre-empts Reviewer 2.
If the result breaks, we have found the load-bearing assumption and can
either fix it before submission or report it honestly.

---

## Scope decision (locked, 2026-08-02)

- **Not** a full v3 rewrite. No proximal causal inference, no causal
  representation learning, no Neumann-series confidence intervals on ρ.
  Those are documented in `LIMITATIONS_V3.md` (future work).
- v2.1 sits in its own module `perturbgnn_v2_1/` and imports from
  `perturbgnn_v2/` for data loaders and helpers. v2's source code is
  untouched.
- Each fix below is a **sensitivity analysis**, not a replacement of v2's
  primary estimator. v2's Durbin δ remains the headline number.

---

## The five fixes

### Fix 1 — AIPW doubly robust estimator (attacks: endogeneity of X in Durbin)

**Why.** v2's Durbin model `y = ρWy + Xβ + WXθ + ε` assumes X (source
indicator) is exogenous. In SPAC-seq, guide delivery has spatial structure
(clone expansion, injection site), so `E[Xε] ≠ 0` and OLS estimates of β
and θ are inconsistent. DiD (v2's Layer 4) addresses parallel-trends
violation but does not address this spatial endogeneity.

**Method.** Augmented inverse-probability-weighting (AIPW, Robins et al.
1994; recent ML extension Chernozhukov et al. 2018, EJ):

```
μ̂₁(X) = XGBoost regression of y on X within T=1
μ̂₀(X) = XGBoost regression of y on X within T=0
ê(X)   = XGBoost classification of T on X (propensity)

τ̂_AIPW = (1/n) Σ [
    T·(y - μ̂₁(X)) / ê(X)
  - (1-T)·(y - μ̂₀(X)) / (1 - ê(X))
  + μ̂₁(X) - μ̂₀(X)
]
```

With cross-fitting (5-fold), the estimator is doubly robust: consistent if
**either** the outcome model **or** the propensity model is correct, and
root-n asymptotic normal under both. Neyman orthogonality means small
errors in μ̂ or ê do not propagate to first-order bias in τ̂.

**X features**: v2's 64-dim embedding + cell-type + density + vessel_dist +
niche_id + module scores (excluded: the response itself to avoid leakage).

**T**: source-vs-matched-control indicator (v2's existing match).

**Acceptance criteria**:
- AIPW τ̂ for Phgr1 × {5 responses} reported with 95% CI from
  cross-fitted influence-function bootstrap.
- Sign of AIPW τ̂ agrees with Durbin δ for ≥ 4 / 5 responses.
- Magnitude within ±50% of Durbin δ. If AIPW τ̂ is much smaller, that is
  evidence that Durbin over-estimates and we report it as such.

**File**: `perturbgnn_v2_1/causal/aipw.py`.

---

### Fix 2 — Niche label circularity (attacks: KMeans niche → encoder → match uses niche)

**Why.** v2 derives `niche_id` from KMeans(k=12) on cell-type composition,
then uses `niche_id` as (a) Layer 1 supervision target and (b) a matching
covariate. The encoder is therefore trained to reproduce KMeans's decision
boundary, and matching balances on a quantity derived from the same KMeans.
This is a circular dependency.

**Method.** Replace KMeans with **Leiden community detection** on the
kNN(cell-type composition) graph at three resolutions
{0.3, 0.6, 1.0}, giving 8 / 14 / 26 niche partitions.

For each partition:
1. Re-train the encoder with niche as supervision.
2. Re-run matching + Durbin scan for Phgr1.
3. Report δ and q per response × resolution.

**Acceptance criteria**:
- Phgr1 δ sign is consistent across all three Leiden resolutions for ≥ 4/5
  responses.
- The K=12 KMeans result from v2 is within the range of Leiden-resolution
  variation (no special-case cherry-pick).

**File**: `perturbgnn_v2_1/data/niche_robust.py`.

---

### Fix 3 — Real alignment quality via probe transfer (attacks: shared classification head)

**Why.** v2's `SupervisedCrossModalEncoder` makes z1 (PCA view) and z2
(cell-type view) **share a single classification head**. cos(z1,z2)=0.9997
is therefore a consequence of architectural constraint, not a measurement
of learned cross-modal alignment.

**Method.** Re-use the existing trained encoder (no re-training needed):

1. Train `head_ct` and `head_niche` on **z1 only** (random 80% of spots).
2. Evaluate zero-shot accuracy of these heads on **z2** (held-out 20%).
3. Compare with same-view accuracy (z1 → z1) for an upper bound, and with
   a random-pairing baseline (z2 → permuted z1) for a lower bound.

**Acceptance criteria**:
- z1 → z2 zero-shot accuracy is at least 70% of z1 → z1 same-view accuracy.
- The reported cos(z1,z2) on the trained encoder is supplemented with
  "probe transfer accuracy" as a second alignment metric.

If z1 → z2 zero-shot accuracy collapses (say, 20% of same-view), the
"two views are aligned" claim is shown to be architectural and we report
it as a v2 limitation.

**File**: `perturbgnn_v2_1/embedding/probe_encoder.py`.

---

### Fix 4 — Spatial buffer in matching (attacks: NTC pool contamination)

**Why.** v2's matched control for a source bin is the nearest NTC bin in
embedding space within the same slice. But an NTC bin sitting within the
spatial neighbourhood of another (non-Phgr1) source bin is itself
contaminated by that source's NCA effect, biasing the matched control.

**Method.** Define an exclusion buffer:
- For each candidate NTC control bin, compute distance to the nearest
  source bin of **any** gene.
- Exclude the candidate if that distance is < 80 µm (~ one decay length).
- Re-run v2's existing matcher on the filtered pool.
- Re-run the Phgr1 cross-cohort scan.

**Acceptance criteria**:
- Phgr1 δ sign holds across all 5 responses after buffer.
- The matched control pool size drops by no more than 50% (else the buffer
  is too aggressive and the matching becomes under-powered).
- Report SMD before/after to confirm covariate balance is still met.

**File**: `perturbgnn_v2_1/matching/match_buffered.py`.

---

### Fix 5 — Real message-passing GNN baseline (attacks: "you should have used a GNN")

**Why.** v2's "GNN" is set-pooling on K=15 neighbours — no message passing.
Reviewer 2 will ask whether a real GraphSAGE / GAT would do better. We need
to show that for v2's specific task (matched-control quality + δ recovery)
set-pooling is sufficient, with empirical evidence rather than assertion.

**Method.** Implement a 2-layer GraphSAGE with mean-aggregation on the same
tissue graph as v2's Layer 1:

```
SAGE layer:
  h_v^(l+1) = ReLU(W · CONCAT(h_v^(l), MEAN_{u∈N(v)} h_u^(l)))

Same supervision targets (cell_type + niche).
Same test-time embedding (last-layer output).
Same downstream: matching + Durbin + cross-cohort scan.
```

Train on M001 + M002, validate on M003 (same split as v2).

**Acceptance criteria**:
- SMD on the SAGE-derived match is reported alongside v2's set-pooling SMD.
- Phgr1 δ from SAGE-derived match is reported alongside v2's δ.
- Decision rule: if SAGE δ is within ±20% of v2 δ and SMD is no better,
  the v2 set-pooling choice is validated. If SAGE is substantially better,
  we add it to the v2 paper as an improved variant.

This is the one fix that could **upgrade** v2 rather than just stress-test
it. Either outcome (SAGE wins → use it; SAGE doesn't → justify set-pooling)
is publishable.

**File**: `perturbgnn_v2_1/baseline/real_gnn.py`.

---

## Integration and reporting

Each fix produces a **single CSV or JSON** in `tutorials/data_v2/sensitivity/`:

- `aipw_phgr1.csv` — τ̂, CI, comparison with Durbin δ
- `niche_leiden_sensitivity.csv` — δ per response × resolution
- `probe_transfer.json` — z1→z1, z1→z2, baseline accuracies
- `matching_buffer_sensitivity.csv` — δ with/without 80µm buffer
- `sage_vs_setpooling.csv` — SMD + δ comparison

The master notebook `SINGLE_MASTER_perturbgnn_v2.ipynb` gets a new
**§15 v2.1 sensitivity** section that reads these files and produces a
single "robustness table" for the paper.

If all five fixes pass acceptance, the paper Discussion gains the sentence:
*"Every modelling choice in our pipeline was stress-tested in §N; the
Phgr1 result survived all five attacks, indicating robustness rather than
dependence on any single architectural decision."*

If any fix fails, we either fix it before submission or report it as a
limitation. We do not bury failures.

---

## Time estimate

| Fix | Estimated time | Bottleneck |
|---|---|---|
| Fix 1 (AIPW) | 1-2 days | XGBoost + cross-fitting on 1.1M spots |
| Fix 2 (Leiden niches) | 1 day | 3 resolutions × re-train encoder |
| Fix 3 (probe transfer) | 0.5 day | No re-training; only inference |
| Fix 4 (matching buffer) | 0.5 day | kNN update on coordinates |
| Fix 5 (SAGE baseline) | 2-3 days | PyG SAGE train + tune |
| Integration + notebook | 1 day | Master notebook §15 + push |

**Total: 6-8 working days.** This is sensitive to (a) GPU availability on
the analysis server (currently co-tenanted with the 2608 PPO runs) and
(b) whether AIPW/SAGE need more than one tuning pass.

---

## What v2.1 does **not** do

The following are documented limitations and are deferred to a real v3
(if ever):

- **Proximal causal inference** (Tchetgen Tchetgen 2020) for unobserved
  confounding — requires negative-control proxies we do not have.
- **Causal representation learning** (Schölkopf 2021) — needs structural
  causal model assumptions we cannot justify.
- **Neumann-series confidence intervals on ρ** — closed form unknown;
  bootstrap is expensive but possible (future work).
- **Spatial Arellano-Bond GMM** — panel-data instrument, requires
  time-series SPAC-seq we do not have at sufficient granularity.
- **Effect-size consistency across cohorts** (ICC, prediction-powered
  inference) — left to paper revision, not v2.1.

These remain in `LIMITATIONS_V3.md` (to be written).

---

**Status**: planning complete, fixes 1-5 to be implemented in order.
