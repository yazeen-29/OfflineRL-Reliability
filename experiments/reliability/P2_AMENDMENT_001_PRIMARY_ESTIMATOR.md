# P2 Amendment 001 — Primary Estimator Specification

Date: 2026-09-19

## Purpose

Freeze the primary consequence estimator before full P2 evaluation.

## Primary estimator

Predict C10 using a deterministic Ridge regression model with:

- features:
  - action disagreement
  - support distance
  - twin-critic disagreement
- preprocessing:
  - StandardScaler
- estimator:
  - sklearn Ridge
- regularization:
  - alpha = 1.0
- intercept:
  - enabled
- stochasticity:
  - none

The scaler and Ridge model are fitted using training policy seeds only.

## Score construction

For each held-out policy seed:

1. fit preprocessing and Ridge parameters using the four training
   policy seeds;
2. predict C10 for held-out states;
3. compute the empirical CDF of the training-fold predicted C10 values;
4. define

   R(x) = 1 - F_train(yhat(x))

where yhat(x) is predicted C10.

Larger R denotes lower predicted consequence.

The held-out seed must not influence score scaling, model fitting,
or any estimator parameter.

## Secondary sensitivity estimator

Repeat the same procedure using action disagreement alone.

This is a sensitivity/reference analysis and is not used to choose the
primary estimator post hoc.

## Hyperparameter selection

No hyperparameter tuning is performed.

alpha = 1.0 is fixed a priori.

## Reliability evaluation

Evaluate the primary reliability score using:

- held-out MAE and RMSE for C10 prediction;
- Spearman correlation between R and observed C10;
- observed C10 across reliability deciles;
- reliability-score risk-coverage;
- held-out policy-seed summaries.

No binary consequence threshold is introduced.

## Independent unit

Policy seed remains the independent evaluation unit.

States within a held-out seed are prediction observations but are not
treated as independent policy-level replicates.

## Reproducibility

Record the repository commit, source-data hashes, Python/package
versions, feature columns, estimator parameters, and policy-seed split.
