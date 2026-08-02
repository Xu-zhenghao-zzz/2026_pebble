# Cohort 3 (Spatiotemporal T cell) — limitations and findings

## Data overview
- MC38 tumor + OT1-Cas9 T cell perturbation (T cells, not tumor cells)
- 33 genes + 1 NTC = 34 sgRNAs
- Day4/7/10 timepoints × 2 replicates = 6 samples
- Total ~3M bins

## Findings

### Guide panel disjoint from cohort 1/2
Cohort 3 panel: Adrb2, Aqp3, Arntl, Cd44, Cd52, Gata3, ...
NO Phgr1, Bcam, Rab8a, Blnk — the v2 cross-cohort hits are absent.
Cannot be used to validate v2 main Phgr1 finding.

### Low guide capture efficiency
- Day4: 14K source bins (3% of 507K total), avg guide UMI = 0.09
- Day7: stronger capture, 29 guides analyzable
- Day10: only 5 guides with enough sources
- Time-course analysis is severely underpowered.

### Gata3 (perturbradius tutorial gene) replicates
Gata3 (the perturbradius spatiotemporal tutorial) shows malignant δ=+0.046
(p=0.034) at Day7 in our framework. This is consistent with prior reports
of Gata3 affecting malignant identity, but the effect is small and
single-timepoint.

## Decision: not included in main paper

Cohort 3 is mentioned in Discussion as:
- An independent T-cell-perturbation cohort validating that our framework
  works on a different experimental design.
- Honest negative result: time-course SPAC-seq is currently underpowered
  for the kind of cross-cohort replication we did with cohort 2.

## Lesson learned
Even with our improved framework, the underlying SPAC-seq data quality
(especially T cell capture at Day4) is the bottleneck. Future
spatiotemporal designs should boost T cell infiltration at early timepoints.
