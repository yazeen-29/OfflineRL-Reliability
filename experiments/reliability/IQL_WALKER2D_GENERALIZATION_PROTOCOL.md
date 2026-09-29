# IQL Walker2d — Generalization Training Protocol

Status:
FROZEN BEFORE TRAINING

## Objective

Test whether the established IQL reliability/consequence relationship
replicates in the Walker2d environment.

## Algorithm

IQL

## Task

mujoco/walker2d/medium-v0

Environment:

Walker2d-v5

## Training configuration

Training steps:

    100,000

Policy seeds:

    0, 1, 2, 3, 4

Device:

    CUDA when available in the training runtime.

Each seed is trained independently.

No post-hoc seed selection or removal is permitted.

## Implementation

Use the repository's existing training entry point and IQL
implementation exactly as committed.

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

This experiment tests environment generalization of IQL.

The five policy seeds remain the independent replication units.

No seed is selected or discarded based on observed performance,
reliability, consequence, or evaluation outcome.
