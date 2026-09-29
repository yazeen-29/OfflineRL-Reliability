# CQL HalfCheetah — Generalization Training Protocol

Status:
FROZEN BEFORE TRAINING

## Objective

Test whether the CQL reliability/consequence relationship established
in the Hopper setting replicates in the HalfCheetah environment.

## Algorithm

CQL

## Task

mujoco/halfcheetah/medium-v0

Environment:

HalfCheetah-v5

## Training configuration

Training steps:

    100,000

Policy seeds:

    0, 1, 2, 3, 4

Device:

    CUDA when available in the training runtime.

Each seed is trained independently.

No post-hoc seed selection or removal is permitted.

## CQL configuration

Use the repository CQL implementation exactly as committed.

Configuration:

    CQLConfig
    StandardObservationScaler
    MinMaxActionScaler
    initial_alpha = 1.0
    alpha_learning_rate = 1e-4

Other CQLConfig parameters remain at the repository/d3rlpy
defaults and are not tuned post hoc.

## Implementation

Use the repository's existing training entry point.

No architecture, optimizer, or regularization changes are permitted
for the purpose of improving the observed reliability result.

## Checkpoint provenance

For every seed record:

- repository commit
- git status
- Python version
- PyTorch version
- d3rlpy version
- Minari version
- task
- environment
- algorithm
- training steps
- seed
- device
- checkpoint path
- checkpoint SHA-256
- verification result

## Acceptance criteria

A seed is usable only if:

- training exits successfully;
- complete checkpoint exists;
- checkpoint reload succeeds;
- existing verification path succeeds;
- provenance metadata is saved.

## Reproducibility boundary

This experiment tests algorithm/environment generalization of CQL.

The five policy seeds remain the independent replication units.

No seed is selected or discarded based on observed performance,
reliability, consequence, or evaluation outcome.
