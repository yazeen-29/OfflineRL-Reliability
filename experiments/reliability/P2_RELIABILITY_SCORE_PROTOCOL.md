# P2 Reliability Score / Estimator Protocol

## Objective

Construct and evaluate a continuous 0–1 reliability score for
reference-based explanations under observation distribution shift.

## Frozen input

Use only the frozen P1 consequence dataset:

data_frozen/P1/

No P1 records may be changed, regenerated, resampled, or excluded
except under the already-frozen P1 exclusion rules.

## Primary outcome

C10 = absolute downstream return consequence at horizon H=10.

## Independent unit

Policy seed.

The five P1 policy seeds are the independent evaluation units.

## Prediction task

Estimate continuous downstream consequence C10 from observable
diagnostic features available at the queried state.

Candidate diagnostic inputs:

A. action disagreement
B. support distance
C. twin-critic disagreement

The estimator must be trained using training policy seeds only and
evaluated on held-out policy seeds.

## Cross-policy-seed evaluation

Use the existing five policy seeds with held-out-seed evaluation.

For each held-out seed:

1. fit all estimator parameters using the remaining four policy seeds;
2. generate predictions for the held-out seed;
3. transform predicted consequence into the reliability score using
   only the training-fold prediction distribution;
4. evaluate the held-out predictions and reliability score.

No held-out observations may determine model parameters, score scaling,
bin boundaries, or calibration parameters.

## Reliability score

Let yhat(x) be the fitted prediction of C10.

Let F_train be the empirical CDF of training-fold predicted C10 values.

Define:

R(x) = 1 - F_train(yhat(x))

Thus larger R corresponds to lower predicted consequence.

The score is bounded to [0,1] using the empirical CDF definition.

## Primary P2 evaluation

Continuous prediction:

- MAE
- RMSE

Reliability ordering:

- Spearman rank correlation between R and observed C10
- monotonicity of observed C10 across reliability quantiles

Operational stratification:

- observed C10 versus reliability coverage
- risk-coverage curve
- reliability-decile consequence summary

## Descriptive uncertainty

Report results by held-out policy seed.

Use seed-level summaries as the independent observations.

No state-level p-values will be treated as independent-policy evidence.

## Model specification

The estimator family must be fixed before evaluating the full
five-seed results.

Feature transformations, regularization, scaling, and any
hyperparameters must be determined using training seeds only.

## Model comparison

Model variants may be compared descriptively.

Do not declare a unique "best" estimator unless a pre-specified
selection rule identifies one using held-out predictive performance.

## Binary outcomes

Do not compute AUROC, AUPRC, Brier score, binary calibration, or
thresholded adverse-consequence metrics because P1 did not freeze a
binary consequence threshold.

## Exclusions

No outcome-dependent exclusion is permitted.

Record all exclusions and reasons if any occur.

## Reproducibility

Record:

- repository commit
- source-data hashes
- random seeds
- policy-seed split
- feature specification
- estimator specification
- software/environment versions
