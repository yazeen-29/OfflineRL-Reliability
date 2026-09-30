# P8 Distance-Matched Control Generalization Evidence Summary

## Primary design

- Six algorithm/environment cells
- H=10
- Nonzero sigma only
- Ten support-distance strata
- Lowest/highest action-disagreement quartiles
- Five policy seeds per cell
- Primary matched pairs: 3600

## Primary cell results

### CQL_HalfCheetah
- Mean seed-level matched ΔC10: 0.00600836
- SD across seeds: 0.00383173
- 95% t-based CI: [0.00125063, 0.01076609]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5
- Matched pairs: 600

### CQL_Hopper
- Mean seed-level matched ΔC10: 0.12036227
- SD across seeds: 0.13608070
- 95% t-based CI: [-0.04860423, 0.28932878]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5
- Matched pairs: 600

### CQL_Walker2d
- Mean seed-level matched ΔC10: 0.15157432
- SD across seeds: 0.11113450
- 95% t-based CI: [0.01358261, 0.28956603]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5
- Matched pairs: 600

### IQL_HalfCheetah
- Mean seed-level matched ΔC10: 0.04603574
- SD across seeds: 0.01898410
- 95% t-based CI: [0.02246387, 0.06960761]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5
- Matched pairs: 600

### IQL_Hopper
- Mean seed-level matched ΔC10: 0.07630156
- SD across seeds: 0.01827561
- 95% t-based CI: [0.05360939, 0.09899372]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5
- Matched pairs: 600

### IQL_Walker2d
- Mean seed-level matched ΔC10: 0.05080012
- SD across seeds: 0.01729527
- 95% t-based CI: [0.02932521, 0.07227504]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5
- Matched pairs: 600

## Cross-cell synthesis

- Cells with positive effects in all five seeds: 6/6
- Cross-cell synthesis is descriptive; no pooled pair-level inference is performed.

## Interpretation boundary

A positive cell-level matched effect supports an association between higher action disagreement and larger downstream consequence after matching on support distance and sigma.
It does not establish causality or complete control of all possible confounding.
Cross-cell replication does not imply directly comparable effect magnitudes across environments or algorithms.

## Sensitivity

The 5-stratum and 20-stratum specifications are reported separately from the primary 10-stratum result.
