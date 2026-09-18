# P2 Reliability Score / Estimator Evidence Summary

- Horizon: H=10
- Independent unit: policy seed
- Policy seeds: 0, 1, 2, 3, 4
- Evaluation: leave-one-policy-seed-out
- Primary estimator: StandardScaler + Ridge(alpha=1.0)
- Primary features: action disagreement, support distance, twin-critic disagreement
- Sensitivity estimator: action disagreement only
- Reliability score: R(x) = 1 - F_train(predicted C10)

## Seed-level predictive metrics

### action_only_sensitivity
- MAE: 0.02918984 (SD 0.01066546)
- RMSE: 0.07073313 (SD 0.04099769)
- Spearman(R, C10): -0.89055780 (SD 0.01827266)

### primary_3feature
- MAE: 0.03076186 (SD 0.01072027)
- RMSE: 0.07120240 (SD 0.04085565)
- Spearman(R, C10): -0.86382374 (SD 0.03881198)

## Interpretation boundary

These analyses evaluate predictive and operational properties of the frozen continuous consequence task.
They do not establish causal faithfulness, a binary failure threshold, or a uniquely optimal estimator.

## Monotonicity qualification

Across held-out seeds, the pooled decile-average observed C10 increases monotonically from high- to low-reliability deciles for both estimators.
The action-only sensitivity has zero adjacent-decile violations on all five held-out seeds.
The primary three-feature estimator has one adjacent-decile violation for held-out seeds 2 and 4.

Repository commit: a5c716f49891eeb29f1fe83d6a9edb14b2d91cfc
