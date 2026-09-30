# P9 Cross-Cell Reliability Estimator Evidence Summary

- Horizon: H=10
- Independent unit: policy seed within cell
- Cells: IQL Hopper, IQL HalfCheetah, IQL Walker2d, CQL Hopper, CQL HalfCheetah, CQL Walker2d
- Policy seeds: 0, 1, 2, 3, 4
- Evaluation: leave-one-policy-seed-out within each cell
- Primary estimator: StandardScaler + Ridge(alpha=1.0)
- Primary features: action disagreement, support distance, twin-critic disagreement
- Sensitivity estimator: action disagreement only
- Reliability score: R(x) = 1 - F_train(predicted C10)
- Coverage: 10, 20, 30, 50, 70, 90, 100%
- Reliability deciles: 10 positional deciles

## Cell-level descriptive summaries

| Cell | Model | Held-out seeds | Mean MAE | SD MAE | Mean RMSE | SD RMSE | Mean Spearman(R,C10) | SD Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CQL_HalfCheetah | action_only_sensitivity | 5 | 0.00225612 | 0.00063070 | 0.00452962 | 0.00237355 | -0.88544951 | 0.01138123 |
| CQL_HalfCheetah | primary_3feature | 5 | 0.00345380 | 0.00308242 | 0.00551313 | 0.00321889 | -0.80994679 | 0.17585699 |
| CQL_Hopper | action_only_sensitivity | 5 | 0.10431215 | 0.08699251 | 0.23663412 | 0.20679368 | -0.85710296 | 0.04510262 |
| CQL_Hopper | primary_3feature | 5 | 0.11960784 | 0.08209587 | 0.24920269 | 0.19979071 | -0.72134672 | 0.14730664 |
| CQL_Walker2d | action_only_sensitivity | 5 | 0.10571252 | 0.04621019 | 0.23278863 | 0.13068170 | -0.81169068 | 0.02173053 |
| CQL_Walker2d | primary_3feature | 5 | 0.11094986 | 0.04366192 | 0.23742253 | 0.12582324 | -0.67125600 | 0.09260012 |
| IQL_HalfCheetah | action_only_sensitivity | 5 | 0.01324609 | 0.00349358 | 0.02605273 | 0.01008532 | -0.92962835 | 0.02271448 |
| IQL_HalfCheetah | primary_3feature | 5 | 0.01560264 | 0.00510957 | 0.02728061 | 0.01059489 | -0.92095965 | 0.02353139 |
| IQL_Hopper | action_only_sensitivity | 5 | 0.02918984 | 0.01066546 | 0.07073313 | 0.04099769 | -0.89055780 | 0.01827266 |
| IQL_Hopper | primary_3feature | 5 | 0.03076186 | 0.01072027 | 0.07120240 | 0.04085565 | -0.86382374 | 0.03881198 |
| IQL_Walker2d | action_only_sensitivity | 5 | 0.03224789 | 0.01453352 | 0.08633468 | 0.04364014 | -0.79718001 | 0.06556410 |
| IQL_Walker2d | primary_3feature | 5 | 0.03477988 | 0.01280128 | 0.08639413 | 0.04167095 | -0.71228175 | 0.08072136 |

## Decile monotonicity

The diagnostic counts below describe how often observed C10 is non-decreasing across the reliability deciles within held-out policy seeds.

- CQL_HalfCheetah / action_only_sensitivity: 5/5 held-out seeds monotonic; 0 total adjacent-decile violations.
- CQL_HalfCheetah / primary_3feature: 4/5 held-out seeds monotonic; 4 total adjacent-decile violations.
- CQL_Hopper / action_only_sensitivity: 5/5 held-out seeds monotonic; 0 total adjacent-decile violations.
- CQL_Hopper / primary_3feature: 2/5 held-out seeds monotonic; 5 total adjacent-decile violations.
- CQL_Walker2d / action_only_sensitivity: 5/5 held-out seeds monotonic; 0 total adjacent-decile violations.
- CQL_Walker2d / primary_3feature: 1/5 held-out seeds monotonic; 6 total adjacent-decile violations.
- IQL_HalfCheetah / action_only_sensitivity: 5/5 held-out seeds monotonic; 0 total adjacent-decile violations.
- IQL_HalfCheetah / primary_3feature: 3/5 held-out seeds monotonic; 2 total adjacent-decile violations.
- IQL_Hopper / action_only_sensitivity: 5/5 held-out seeds monotonic; 0 total adjacent-decile violations.
- IQL_Hopper / primary_3feature: 3/5 held-out seeds monotonic; 2 total adjacent-decile violations.
- IQL_Walker2d / action_only_sensitivity: 5/5 held-out seeds monotonic; 0 total adjacent-decile violations.
- IQL_Walker2d / primary_3feature: 3/5 held-out seeds monotonic; 2 total adjacent-decile violations.

## Interpretation boundary

P9 evaluates cross-cell replication of the frozen continuous consequence estimator.
The six cells are summarized descriptively and are not treated as a pooled inferential population.
The analysis does not establish causal faithfulness, introduce a binary failure threshold, or select a universally best estimator.

Repository commit: d36e04d5c5e8ebbb7238f741dfa4942c76f7b3c6
