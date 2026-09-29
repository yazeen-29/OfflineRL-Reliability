# P6 CQL Walker2d Evidence Summary

## Experiment

Environment:
Walker2d-v5

Task:
mujoco/walker2d/medium-v0

Algorithm:
CQL

Policy seeds:
0, 1, 2, 3, 4

Decision states:
100 per policy seed

Total records:
14,000

Primary horizon:
H = 10

Primary sigma levels:
0.01, 0.025, 0.05, 0.10, 0.20, 0.30

## Primary outcome

C10 = absolute_consequence = |delta_J|

delta_J = clean_return - shifted_return

## Primary predictor

action_disagreement

## Primary estimand

For each decision state, regress:

    C10 = intercept + beta_state * action_disagreement

across the six nonzero sigma levels.

For each policy seed, average the valid state-level slopes to obtain
one seed-level slope.

Policy seed is the independent replication unit.

## Primary result

Seed-level slopes:

seed 0: 2.9867688248
seed 1: 1.0884269469
seed 2: 1.5167711656
seed 3: 0.7383647970
seed 4: 2.1579701809

Cross-seed:

mean = 1.6976603830
SD = 0.8941224741
95% t-based CI = [0.5874606970, 2.8078600691]

positive seed-level slopes = 5/5

Exact one-sided sign-flip p = 0.03125
Exact two-sided sign-flip p = 0.06250

## Interpretation boundary

The result supports reproduction of the evaluated
action-disagreement/downstream-consequence relationship for the
CQL Walker2d setting.

The result does not by itself establish generalization to all
offline-RL algorithms, datasets, environments, or configurations.

The two-sided exact sign-flip p-value is reported alongside the
t-based confidence interval because they are distinct inferential
procedures.

Cross-environment slope magnitudes are not treated as directly
comparable because consequence/reward scales differ across tasks.
