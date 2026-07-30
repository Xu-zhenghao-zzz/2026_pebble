# Frozen real-data tutorial tables

These TSV files are compact derivatives of the three local SPAC analyses. Raw
data were treated as read-only. The files retain only fields used by the
tutorial figures and are small enough to version with the package.

## Selection policy

Candidates were selected using existing, precomputed evidence rather than
visual appeal:

- coverage across sections or biological replicates;
- clone-level directional consistency;
- within-section permutation;
- structural matching where available;
- source/boundary sensitivity;
- external-cohort support where available;
- explicit failure of any formal-radius gate.

The selected cases are:

- **subQ Rab8a**: three evaluable sections and eight ring-complete clones;
  hypoxia increases in 8/8 clones, IFN response decreases in 7/8, and
  malignant-like fraction increases in 7/8. Only `sgRab8a_1` is represented,
  so this is a reproducible spatial-state association, not a gene-level causal
  radius.
- **lung Ccn1**: five M002 territories with macrophage enrichment under raw
  and structurally matched comparisons and 5/5 core/hybrid direction
  agreement. It remains a one-section, one-guide result.
- **spatiotemporal Gata3**: Day7 `fraction_NK` has the smallest empirical P
  value in the V2 main screen, but global FDR and matching balance fail. No
  radius is reported.

## Local provenance

The frozen files were derived from:

- `perturb_radius_pilot/processed/clone_summary_spatial.tsv`
- `perturb_radius_pilot/results/pilot_gene_coverage.tsv`
- `perturb_radius_anno/manuscript/source_data/Rab8a_subQ_malignant_only_*`
- `perturb_radius_anno/results/lung_subq_gene_summary.tsv`
- `perturb_radius_anno/results/lung_subq_robustness.tsv`
- `perturb_radius_anno/manuscript/source_data/Figures2_4_candidate_source_coordinates.tsv`
- `perturb_radius_anno/manuscript/source_data/Figure4_Ccn1_macrophage_radial_matrix.tsv`
- `perturb_radius_anno/manuscript/source_data/Figure4_boundary_sensitivity.tsv`
- `perturb_radius_anno/results/consensus_evidence.tsv`
- `perturb_radius_spatiotemporal/processed/v2/supported_patch_anchors.parquet`
- `perturb_radius_spatiotemporal/results/v2_supported_guide_coverage.tsv`
- `perturb_radius_spatiotemporal/results/v2_matched_distance_curves.tsv`
- `perturb_radius_spatiotemporal/results/v2_radius_gate.tsv`
- `perturb_radius_spatiotemporal/results/v2_robust_screen.tsv`

The tables are analysis derivatives and do not contain raw expression matrices
or patient-identifying information.
