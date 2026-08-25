# HalfCheetah-medium-v0 Cross-Environment Replication Protocol

## Environment

Task:
mujoco/halfcheetah/medium-v0

Dataset:
Minari offline dataset

Observation dimension:
17

Action dimension:
6

Episodes:
1000

Episode length:
1000

## Algorithm

Algorithm:
IQL

Training steps:
100000

Policy seeds:
0, 1, 2, 3, 4

## Evaluation protocol

Episode-level reference/query split:
90% reference, 10% query

Queries:
1000 held-out observations per policy seed

Observation standardization:
Reference-dataset-only z-score

Primary reference method:
Nearest neighbor in standardized observation space

Action disagreement metric:
RMS action disagreement

Gaussian observation-shift levels:
0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30

Gaussian repeats:
5 per query and level

## Replication rule

This protocol reproduces the frozen Hopper Gaussian methodology.
Only the environment changes.

No hyperparameter tuning is permitted after inspecting
cross-seed evaluation results.

## Primary research question

Does the relationship between distributional distance and
explanation disagreement replicate in a second offline RL
environment?
