# P5 CQL Generalization Protocol

## Objective

Test whether the reliability/consequence relationship established for
IQL generalizes to a distinct offline-RL policy family: Conservative
Q-Learning (CQL).

## Policy family

Algorithm:

    CQL

Implementation:

    repository src/utils/policy_io.py::build_cql

The implementation is used exactly as committed. No CQL architecture,
optimizer, or regularization parameter is manually changed for P5.

## Environment and dataset

Task:

    mujoco/hopper/medium-v0

Dataset loading:

    d3rlpy.datasets.get_minari("mujoco/hopper/medium-v0")

The existing repository training entry point is used.

## Training configuration

Training steps:

    100,000

Policy seeds:

    0, 1, 2, 3, 4

Device:

    CUDA when available in the training runtime.

Training is performed independently for each policy seed.

## CQL configuration

Repository implementation:

    CQLConfig

Observation scaler:

    StandardObservationScaler

Action scaler:

    MinMaxActionScaler

Initial CQL alpha:

    1.0

CQL alpha learning rate:

    1e-4

Other CQLConfig parameters remain at the d3rlpy/repository defaults
and are not tuned post hoc.

## Independence

Policy seed is the independent unit.

The five CQL policies are not pooled with the five IQL policies for
primary policy-family inference.

IQL and CQL are treated as separate policy families.

## Checkpoint provenance

Each policy must be:

1. trained from the specified seed;
2. saved as a complete d3rlpy checkpoint;
3. reloaded from that checkpoint;
4. verified using the existing repository verification path.

Record:

- repository commit;
- git status;
- d3rlpy version;
- Python version;
- PyTorch version;
- training seed;
- task;
- training steps;
- CQL configuration;
- checkpoint SHA-256;
- verification return;
- output paths.

## Acceptance criteria

A CQL seed is considered usable for P5 only if:

- training exits successfully;
- a complete checkpoint exists;
- checkpoint reload succeeds;
- the existing verification path succeeds;
- provenance metadata is saved.

No seed may be selected or removed using its observed reliability,
consequence, or evaluation performance.

## P5 consequence collection

After all five CQL policies pass verification, collect a new CQL-specific
counterfactual consequence dataset.

Do not reuse the frozen IQL consequence observations as though they
were CQL outcomes.

The consequence collection should reproduce the P1 sampling and
counterfactual protocol structure:

- same environment task;
- same evaluation state-sampling procedure;
- same state-sampling seed;
- same observation-noise seed;
- same sigma levels;
- same horizons;
- same horizon-10 primary outcome;
- same support-reference construction where applicable.

The CQL policy acts on the same type of observed states, but the
resulting clean/shifted actions and downstream outcomes are newly
generated from the CQL policies.

## Primary P5 outcome

At H=10:

    C10 = absolute_consequence

## Primary generalization question

Evaluate whether the relationship between action disagreement and
downstream consequence observed for IQL is also present for CQL.

Secondary evaluation may reuse the frozen P2 reliability-score
specification after the CQL consequence dataset is frozen.

## Statistical convention

Policy seed remains the independent unit.

Individual counterfactual states are not treated as independent
policy-level replicates for cross-policy inference.

## Interpretation boundary

A positive CQL result supports policy-family generalization from IQL
to CQL for the evaluated Hopper setting.

It does not establish generalization to all offline-RL algorithms,
datasets, environments, or training configurations.

A null result must be reported as-is and must not trigger post-hoc
CQL hyperparameter tuning for the purpose of improving the reliability
result.
