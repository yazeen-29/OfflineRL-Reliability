# P1 Evidence Summary

## Experiment

Publication-scale P1 decision-consequence experiment.

Environment:
Hopper-medium-v0

Algorithm:
IQL

Policy seeds:
0, 1, 2, 3, 4

Decision states:
100 per policy seed

Total frozen records:
14,000

Gaussian observation shifts:
0.00, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30

Horizons:
1, 5, 10, 20

Primary horizon:
10

## Primary outcome

C10 = |Delta_J(10)|

Policy seed is the independent replication unit.

## Held-out predictive evaluation

Leave-one-policy-seed-out evaluation was used.

Each held-out seed contains 700 H=10 observations.

Training uses the other four policy seeds.

## No-signal baseline

The baseline predicts the training-fold mean C10.

Mean MAE:
0.0590687904

Mean RMSE:
0.0974069880

## Candidate models

A:
Action disagreement only

B:
Action disagreement + support distance

C:
Action disagreement + twin-critic disagreement

D:
Action disagreement + support distance + twin-critic disagreement

E:
Action disagreement + support distance
+ action-disagreement × support-distance

## Primary H=10 results

All five candidate models improved over the no-signal baseline
for all five held-out policy seeds in both MAE and RMSE.

### Seed-level paired inference

Differences are defined as:

model error - baseline error

Therefore negative values indicate improvement.

All five models:

MAE:
5/5 held-out seeds improved

RMSE:
5/5 held-out seeds improved

Exact one-sided sign-flip p:
0.03125

Exact two-sided sign-flip p:
0.06250

## Interpretation

The P1 results support cross-policy-seed predictive generalization
of the measured reference/instability-related signals for
counterfactual consequence magnitude.

The results do not establish a uniquely optimal feature combination.
No numerical post-hoc model-selection rule was imposed.

## Binary prediction

AUROC, AUPRC, calibration error, and Brier score were not computed.

Reason:
the frozen protocol does not define an adverse-consequence
threshold. Introducing such a threshold after observing the
predictive results would violate the prespecified analysis rule.

## Exclusions

The frozen JSON records contain successful retained observations.
Attempt-level failed collection events are not represented in the
frozen record files, so attempt-level exclusion counts cannot be
reconstructed from these files alone.

## Provenance

Frozen dataset:
d24357a

Held-out predictive analysis:
08d66cc

Seed-level paired inference:
1fb915c

Frozen policy-seed collection protocol:
55e03b2

## Claim boundary

Supported:
- cross-policy-seed predictive generalization
- improvement over the no-signal baseline
- consistent direction across the five held-out seeds

Not established:
- a unique optimal predictor
- causal faithfulness of the explanation
- binary adverse-consequence discrimination
- generalization beyond the evaluated IQL/Hopper setting
