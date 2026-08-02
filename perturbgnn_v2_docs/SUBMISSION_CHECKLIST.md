# Submission Checklist — PerturbGNN v2

**Target venue**: Nature Methods (primary) / Genome Biology (backup)

---

## 1. Cover letter key points

- **Novelty**: First causal identification framework for SPAC-seq NCA effects,
  replacing the unfalsifiable "propagation radius" framing with DiD + IV +
  Rosenbaum Γ* + cross-cohort replication.
- **Methodological innovation**: Resistance-distance Durbin model + supervised
  cross-modal GNN encoder (perturbation-decoupled embedding, cosine=0.9997).
- **Biological discovery**: Phgr1 — a multi-response NCA hub missed by v1
  vs-NTC framework. Cross-cohort (2 cohorts, 6-7 slices) replicated with
  q < 1e-21. TCGA LUAD (n=516) confirms: Cox HR=0.556, p=0.004.
- **Honesty**: 79% of v1-significant hits are v2-negative (false positives).
  Phgr1 does NOT predict ICB response (negative result reported).
- **Reproducibility**: Full pipeline < 3 hours on 1× V100. Code on GitHub.

## 2. Why Nature Methods

- **Methodological depth**: 5-layer pipeline (GNN + matching + Durbin +
  causal diagnostics + cross-cohort). Each layer is independently novel.
- **Synthetic benchmark** (F19): power calibration, not just empirical claims.
- **v1 vs v2 comparison** (F17): quantifies false positive rate reduction.
- **Cross-cohort replication**: 2 SPAC-seq cohorts, 8 slices — sets a new
  standard for SPAC-seq NCA claims.

## 3. Predicted reviewer questions

### Q1: "Single guide per gene — off-target?"
**A**: Cohort 2 has 2 guides/gene. Phgr1 replicates in both cohort 1
(1 guide) and cohort 2 (2 guides), making off-target unlikely.
Limitation #1 in Discussion.

### Q2: "Why not rescue experiments?"
**A**: Future work. BIOLOGY_BACKLOG.md contains detailed protocol.
Mouse→human TCGA concordance (5 compartments) is strong circumstantial
evidence even without rescue.

### Q3: "TCGA is correlative, not causal."
**A**: Correct — stated in Limitation #4. The SPAC-seq causal framework
(DiD + IV + Γ*) is the causal evidence; TCGA is orthogonal validation.

### Q4: "Why does Phgr1 not predict ICB response?"
**A**: Phgr1 may mark immune-inflamed TME (good prognosis) without
being the rate-limiting target for checkpoint blockade. Reported as
honest negative result (Section 4.x).

### Q5: "Resistance distance is approximate."
**A**: Limitation #3. Path-sampled approximation, not exact circuit
resistance. Sensitivity analysis on α_b/α_v parameters recommended
(future work).

### Q6: "Bcam has opposite Cox HR — why include it?"
**A**: Bcam is a counter-example showing that large effect size ≠
good prognosis. Demonstrates the framework doesn't just chase δ.

## 4. Figure checklist

| Figure | Content | Status |
|---|---|---|
| F1 | Pipeline overview | ✅ |
| F2 | GNN embedding quality | ✅ |
| F3 | Matched control balance | ✅ |
| F4 | Durbin landscape (3 slices) | ✅ |
| F5 | Phgr1 case study | ✅ |
| F6 | v2 causal volcano | ✅ |
| F7 | Phgr1 TCGA correlations | ✅ |
| F8 | Phgr1 survival | ✅ |
| F9 | Cross-cohort heatmap | ✅ |
| F10 | Phgr1 direction consistency | ✅ |
| F11 | Phgr1 triple evidence | ✅ |
| F12 | Gene comparison | ✅ |
| F13 | Cohort overview | ✅ |
| F14 | v1 vs v2 head-to-head | ✅ |
| F15 | Phgr1 decay curves | ✅ |
| F16 | Cohort 3 timecourse | ✅ |
| F17 | v1 vs v2 scatter | ✅ |
| F18 | Phgr1 ICB (negative) | ✅ |
| F19 | Synthetic benchmark | ✅ |

**Paper PDF** uses 10 of these (F1, F2, F5, F7, F8, F9, F11, F13, F17, F19).
Supplementary can include the rest.

## 5. Data deposition

- [x] Raw SPAC-seq: public (spac.pku-genomics.org)
- [x] TCGA: cBioPortal (luad_tcga_gdc)
- [ ] GitHub repo (pending upload)
- [ ] Zenodo DOI (after GitHub)
- [ ] CodeOcean capsule (optional, for Nature Methods)

## 6. Pre-submission timeline

| Step | Time |
|---|---|
| GitHub upload + README polish | 2 hours |
| Cover letter draft | 1 day |
| PI/advisor review | 1 week |
| Format for Nature Methods | 2 days |
| Submit | — |
