# P4 OOD / Uncertainty Baseline Evidence Summary

- Horizon: H=10
- Population: nonzero observation shifts
- Independent unit: policy seed
- Policy seeds: 0, 1, 2, 3, 4

## Baselines

- B1: support / OOD only
- B2: critic uncertainty only
- B3: support + critic
- B4: action disagreement only
- B5: action + support + critic

## Seed-level summary

### Support / OOD only
- Mean Spearman(R,C10): 0.204992 (SD 0.170415)

### Critic uncertainty only
- Mean Spearman(R,C10): 0.146727 (SD 0.135224)

### Support + critic
- Mean Spearman(R,C10): -0.137288 (SD 0.134806)
- Mean MAE: 0.062687 (SD 0.012420)
- Mean RMSE: 0.102068 (SD 0.038047)

### Action disagreement only
- Mean Spearman(R,C10): -0.827457 (SD 0.028900)

### Action + support + critic
- Mean Spearman(R,C10): -0.821250 (SD 0.036254)
- Mean MAE: 0.035427 (SD 0.012430)
- Mean RMSE: 0.077053 (SD 0.044084)

## Interpretation boundary

P4 compares predefined OOD, uncertainty, action-disagreement, and combined formulations.
The analysis is descriptive across the five held-out policy seeds and does not declare a universal or uniquely optimal baseline.
It does not establish causal faithfulness or universal OOD detection capability.

Repository commit: 6c3079b3755f3d9c3eff86c8761898b69ab0407c
