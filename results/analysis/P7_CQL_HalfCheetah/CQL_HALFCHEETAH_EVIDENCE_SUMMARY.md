# P7 CQL HalfCheetah Evidence Summary

## Experiment

- Algorithm: CQL
- Task: `mujoco/halfcheetah/medium-v0`
- Environment: `HalfCheetah-v5`
- Policy seeds: 0, 1, 2, 3, 4
- Final collection: 14,000 records
- Decision states: 100 per seed
- Sigma levels: 0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30
- Horizons: 1, 5, 10, 20
- Primary horizon: H=10
- Primary condition: sigma > 0

## Primary estimator

For each seed and decision state, regress absolute downstream
consequence at H=10 on action disagreement across the six nonzero
sigma levels.

Average the state-level slopes within each seed.

The five seed-level slopes are the independent replication units.

## Primary results

Seed-level slopes:

- seed 0: 0.3172364608
- seed 1: 0.2927790658
- seed 2: 0.2010072467
- seed 3: 0.2261899772
- seed 4: 0.2521626830

Cross-seed summary:

- mean: 0.2578750867
- SD: 0.0474673294
- 95% CI: [0.1989366127, 0.3168135607]
- positive seeds: 5/5
- exact one-sided sign-flip p: 0.03125
- exact two-sided sign-flip p: 0.06250

State-level positive slope fractions:

- seed 0: 1.00
- seed 1: 0.98
- seed 2: 0.95
- seed 3: 1.00
- seed 4: 1.00

## Secondary findings

Across all five seeds, the dose-response was monotonic across the six
nonzero sigma levels for both mean action disagreement and mean absolute
consequence.

Each seed had Spearman rho = 1.0 and Kendall tau approximately 1.0.

Support-distance relationships were heterogeneous:

- support -> action disagreement varied around zero with low R-squared
- support -> consequence varied across seeds, including both positive
  and negative slopes

These are secondary descriptive analyses.

## Data integrity

- Final records: 14,000
- Final audit: PASS
- Five seeds: PASS
- 100 states per seed: PASS
- 2,800 records per seed: PASS
- Primary records per seed: 600
- Archive SHA-256:
  `f6f886bd35903a56898ee6eb9829e28491c2ec7260ab185e7e8f679cda6a5d5d`

## Interpretation boundary

The result supports the evaluated action-disagreement/downstream-
consequence relationship for this CQL HalfCheetah experiment under the
frozen protocol.

It does not establish generalization to all offline-RL algorithms,
datasets, environments, or CQL configurations.

## Status

P7 CQL HalfCheetah complete.
