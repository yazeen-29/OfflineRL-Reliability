# P3 Distance-Matched Control Evidence Summary

## Primary design

- Horizon: H=10
- Population: nonzero observation shifts
- Independent unit: policy seed
- Support-distance strata: 10
- Low/high action-disagreement groups: lowest/highest quartile
- Total matched pairs: 600
- Pairs per policy seed: 120

## Primary matched effect

- Mean seed-level matched ΔC10: 0.07630156
- SD across seeds: 0.01827561
- 95% t-based CI: [0.05360939, 0.09899372]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.06250
- Positive seed effects: 5/5

## Balance

### Policy seed 0
- Matched pairs: 120
- Mean absolute support-distance gap: 0.10790453
- Median absolute support-distance gap: 0.09134635
- Mean action-disagreement gap: 0.06583091
- Minimum action-disagreement gap: 0.00277921
- Mean paired ΔC10: 0.09527171

### Policy seed 1
- Matched pairs: 120
- Mean absolute support-distance gap: 0.12198067
- Median absolute support-distance gap: 0.08489920
- Mean action-disagreement gap: 0.06369698
- Minimum action-disagreement gap: 0.00273743
- Mean paired ΔC10: 0.07345202

### Policy seed 2
- Matched pairs: 120
- Mean absolute support-distance gap: 0.14593678
- Median absolute support-distance gap: 0.05353702
- Mean action-disagreement gap: 0.04048290
- Minimum action-disagreement gap: 0.00209715
- Mean paired ΔC10: 0.09457146

### Policy seed 3
- Matched pairs: 120
- Mean absolute support-distance gap: 0.07330353
- Median absolute support-distance gap: 0.02639025
- Mean action-disagreement gap: 0.03936916
- Minimum action-disagreement gap: 0.00157826
- Mean paired ΔC10: 0.06373562

### Policy seed 4
- Matched pairs: 120
- Mean absolute support-distance gap: 0.10011056
- Median absolute support-distance gap: 0.08613811
- Mean action-disagreement gap: 0.04160226
- Minimum action-disagreement gap: 0.00096498
- Mean paired ΔC10: 0.05447697

## Interpretation boundary

A positive matched effect supports an association between higher action disagreement and larger downstream consequence after matching on support distance and sigma.
It does not establish causality or complete control of all possible confounding.

The five policy seeds are the independent inferential units; matched pairs are not treated as independent policy replicates.

The 5-stratum sensitivity analysis is reported separately. The 20-stratum specification is mechanically infeasible under the frozen minimum-group-size requirement when each support-distance cell contains five observations.
