# P9 Cross-Cell Reliability Estimator Replication Protocol

## Objective

Evaluate whether the frozen P2 continuous reliability estimator
remains predictive and operationally useful across the six frozen
algorithm/environment cells.

## Primary outcome

C10 = absolute downstream return consequence at horizon H=10.

## Independent unit

Policy seed within each algorithm/environment cell.

Each cell contains five policy seeds.

## Frozen cells

1. IQL Hopper
2. IQL HalfCheetah
3. IQL Walker2d
4. CQL Hopper
5. CQL HalfCheetah
6. CQL Walker2d

## Prediction task

Estimate continuous downstream consequence C10 from diagnostic
features available at the queried state.

Primary diagnostic features:

- action disagreement
- support distance
- twin-critic disagreement

Sensitivity feature set:

- action disagreement alone

## Cross-policy-seed evaluation

For each cell independently, perform leave-one-policy-seed-out
evaluation.

For each held-out policy seed:

1. fit all preprocessing parameters using the remaining four
   policy seeds;
2. fit the primary estimator using only those four policy seeds;
3. predict C10 for the held-out policy seed;
4. compute the empirical CDF of training-fold predicted C10 values;
5. construct the reliability score using only the training-fold
   prediction distribution;
6. evaluate the held-out predictions and reliability score.

No held-out observations may determine model parameters, score scaling,
bin boundaries, or calibration parameters.

## Primary estimator

Preprocessing:

- StandardScaler

Estimator:

- sklearn Ridge
- alpha = 1.0
- intercept = enabled
- stochasticity = none

No hyperparameter tuning is performed.

The estimator specification is inherited directly from the frozen
P2 primary estimator.

## Reliability score

Let yhat(x) be the fitted prediction of C10.

Let F_train be the empirical CDF of training-fold predicted C10.

Define:

R(x) = 1 - F_train(yhat(x))

Larger R denotes lower predicted consequence.

The held-out policy seed must not influence score scaling or estimator
parameters.

## Primary evaluation

For every cell report:

- held-out MAE
- held-out RMSE
- Spearman correlation between R and observed C10
- reliability-decile observed C10
- risk-coverage
- seed-level summaries

Risk-coverage and decile handling must mirror the frozen P2 analysis.

## Sensitivity analysis

Repeat the same evaluation using action disagreement alone.

This is a sensitivity/reference analysis and is not used for post-hoc
selection of the primary estimator.

## Cross-cell synthesis

Summarize the six cells descriptively.

Report cell-level and seed-level results.

Cross-cell synthesis must not treat the 30 held-out seeds as a single
pooled inferential population.

Do not rank algorithm/environment cells.

Do not declare a universal best cell or estimator.

## Binary outcomes

Do not introduce AUROC, AUPRC, Brier score, binary calibration,
or thresholded adverse-consequence metrics.

## Exclusions

No outcome-dependent exclusion is permitted.

Record all exclusions and reasons if any occur.

## Reproducibility

Record:

- repository commit
- source-data hashes
- policy-seed split
- feature specification
- estimator specification
- Python/package versions
- protocol hash
