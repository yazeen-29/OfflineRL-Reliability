# IQL Walker2d Evidence Summary

## Experiment status

Final 5-seed IQL Walker2d environment-generalization consequence experiment.

Task:
- Minari: mujoco/walker2d/medium-v0
- Environment: Walker2d-v5
- Algorithm: IQL
- Training: 100,000 steps
- Policy seeds: 0, 1, 2, 3, 4

Final consequence dataset:
- 2,800 records per seed
- 14,000 records total
- 100 states per seed
- 7 sigma levels
- 4 horizons
- H10 primary

## Primary result

The primary estimator regresses absolute_consequence on
action_disagreement across the six nonzero sigma levels for each state,
then averages the 100 state-level slopes to obtain one seed-level slope.

Seed-level slopes:

- seed 0: 0.4756666535
- seed 1: 1.0404983321
- seed 2: 1.6445173039
- seed 3: 1.1195700083
- seed 4: 1.3694530834

Cross-seed:

- mean slope: 1.1299410763
- SD: 0.4354128154
- 95% t CI: [0.5893046590, 1.6705774936]
- positive seeds: 5/5
- exact one-sided 2^5 sign-flip p: 0.03125
- exact two-sided 2^5 sign-flip p: 0.06250

The five policy seeds are the inferential units.

## State-level heterogeneity

Positive state-level slope fractions:

- seed 0: 0.99
- seed 1: 0.98
- seed 2: 1.00
- seed 3: 0.98
- seed 4: 0.96

State-level slope SD:

- seed 0: 1.3601943375
- seed 1: 0.8635519046
- seed 2: 2.0857289763
- seed 3: 3.1406012602
- seed 4: 1.2172005233

The state-level slopes therefore show substantial within-seed heterogeneity.

## Secondary diagnostics

### Dose response

For each seed, mean action disagreement and mean absolute consequence
increase across the six nonzero sigma levels.

The secondary output reports perfectly monotone rank ordering across
the six dose levels for every seed.

These are descriptive dose-response summaries and are not treated as
additional independent policy replications.

### Support-conditioned relationships

Support-to-action slopes are negative for all five seeds.

Support-to-consequence slopes are heterogeneous across seeds:

- seed 0: +0.0346448461
- seed 1: -0.0046261614
- seed 2: -0.0280658161
- seed 3: +0.0106870228
- seed 4: -0.0051507266

Therefore support distance does not show a uniform directional relationship
with downstream consequence across the five policy seeds in this secondary
analysis.

## Interpretation boundary

The primary result supports the evaluated action-disagreement /
downstream-consequence relationship for independently trained IQL policies
in the Walker2d-medium setting under the frozen Gaussian observation
corruption protocol.

It should not be interpreted as pooled evidence across different algorithms
or environments.

IQL Hopper, IQL HalfCheetah, CQL Hopper, and CQL Walker2d remain separate
experimental families/settings for primary inference.

## Reproducibility

Frozen consequence protocol:
experiments/reliability/IQL_WALKER2D_CONSEQUENCE_PROTOCOL.md

Frozen analysis protocol:
experiments/reliability/IQL_WALKER2D_ANALYSIS_PROTOCOL.md

Raw-data hash manifest:
experiments/reliability/IQL_WALKER2D_FINAL_CONSEQUENCE_DATA.sha256

Analysis-output hash manifest:
experiments/reliability/IQL_WALKER2D_ANALYSIS_OUTPUTS.sha256

Training checkpoints are preserved separately in the local IQL Walker2d
backup archive.

