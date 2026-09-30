# Final Paper Figure / Table Plan

The frozen computational package is complete. This directory is the
source package for the manuscript figures and tables.

## Main-paper figures

### Figure 1 — P1 held-out consequence prediction
Import:
`figure_sources/F1_P1_heldout_model_comparison.csv`

Recommended Prism graph:
grouped dot/bar plot of held-out MAE (and a separate RMSE panel)
across the prespecified candidate models. Include the no-signal
baseline. Do not label a single candidate as universally optimal.

### Figure 2 — P8 distance-matched control across six cells
Import:
`figure_sources/F2_P8_distance_matched.csv`

Recommended Prism graph:
point estimates with 95% t-based CI, one row per algorithm/environment
cell. Keep the zero reference line. This is the principal cross-cell
control figure.

### Figure 3 — P9 cross-cell reliability estimator
Import:
`figure_sources/F3_P9_reliability_summary.csv`

Recommended Prism graph:
primary 3-feature estimator: Spearman(R,C10) by cell with mean ± SD
across held-out policy seeds. A companion graph can show MAE/RMSE.

### Figure 4 — P9 risk-coverage
Import:
`figure_sources/F4_P9_risk_coverage_nonzero_shift.csv`

Recommended Prism graph:
risk/consequence versus coverage, using the frozen summary values.
The CSV is authoritative; do not manually recreate values.

## Supplementary figures

### Figure S1 — CQL primary replication
Import:
`figure_sources/F5_CQL_primary_replication.csv`

This summarizes P5/P6/P7 seed-level slopes. Use error bars only as the
stored 95% t-based CIs and do not compare raw magnitudes across
environments as standardized effect sizes.

### Figure S2 — P4 baseline comparison
Import:
`figure_sources/F6_P4_baselines.csv`

Descriptive comparison of support/OOD, critic uncertainty, action
disagreement, and combined formulations.

### Figure S3 — P10 controlled-shift detectability
Import:
`figure_sources/F7_P10_controlled_shift_ood_sanity.csv`

Label explicitly:
"Controlled observation-shift detectability sanity check"
and not "generic OOD detection".

### Figure S4 — CQL Hopper dose response
Import:
`figure_sources/F8_P5_CQL_Hopper_dose_response.csv`

Useful for showing the monotonic controlled-shift dose response.

## Main-paper tables

Table 1: `tables/T1_experiment_matrix.csv`
Table 2: `tables/T2_P1_heldout_predictive_comparison.csv`
Table 3: `tables/T3_P8_distance_matched_control.csv`
Table 4: `tables/T4_P9_reliability_estimator.csv`

## Supplementary tables

Table S1: `tables/T5_P4_baseline_comparison.csv`
Table S2: `tables/T6_P10_controlled_shift_ood_sanity.csv`
Table S3: `tables/T7_CQL_primary_replication.csv`
Table S4: `tables/T8_P5_CQL_Hopper_dose_response.csv`

## Statistical guardrails

- Independent replication unit is policy seed within cell.
- P9 is leave-one-policy-seed-out within each cell.
- P8 and P3 are control/association analyses, not causal identification.
- P4 is descriptive and does not establish a universal best baseline.
- P10 tests the frozen controlled perturbation family only.
- Exact sign-flip p-values and t-based confidence intervals are distinct
  inferential procedures.
- Do not pool IQL and CQL policy seeds for primary inference.
- Do not rank raw slope magnitudes across environments as if they were
  standardized effect sizes.
