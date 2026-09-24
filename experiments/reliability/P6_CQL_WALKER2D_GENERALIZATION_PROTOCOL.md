# P6 CQL Walker2d Environment Generalization Protocol

## Status

Pre-training specification for evaluating the frozen CQL reliability/consequence
relationship on the Walker2d-medium offline RL setting.

## Objective

Evaluate whether the action-disagreement to downstream-consequence relationship
observed for CQL on Hopper is reproduced on a distinct MuJoCo environment:
Walker2d-medium.

This is an environment-generalization extension of the frozen CQL policy-family
study. It does not modify the Hopper CQL analysis.

## Policy family

Algorithm:

    CQL

Implementation:

    repository src/utils/policy_io.py::build_cql

The committed CQL implementation is used exactly as specified for the frozen
CQL Hopper study.

No architecture, optimizer, or regularization parameter is changed for the
purpose of this experiment.

## Environment and dataset

Task:

    mujoco/walker2d/medium-v0

Environment:

    Walker2d-v5

Dataset loading:

    d3rlpy.datasets.get_minari("mujoco/walker2d/medium-v0")

Observed dataset signature during smoke validation:

    observation shape = (17,)
    action shape      = (6,)
    action space      = continuous
    dataset episodes  = 1044

## Training configuration

Training steps:

    100,000

Policy seeds:

    0, 1, 2, 3, 4

Device:

    CUDA when available in the training runtime.

Training is independent across policy seeds.

## CQL configuration

Observation scaler:

    StandardObservationScaler

Action scaler:

    MinMaxActionScaler

Initial CQL alpha:

    1.0

CQL alpha learning rate:

    1e-4

All other CQLConfig parameters remain at the repository/d3rlpy defaults
used in the frozen Hopper CQL study.

No post-hoc hyperparameter tuning is permitted.

## Checkpoint acceptance

Each Walker2d CQL seed is usable only if:

1. training exits successfully;
2. a complete d3rlpy .d3 checkpoint exists;
3. the checkpoint can be reloaded with d3rlpy;
4. in-memory and reloaded deterministic verification agree;
5. provenance metadata is saved;
6. checkpoint SHA-256 is recorded.

The five policy seeds are the independent policy-level units.

No seed may be selected or removed based on observed reliability,
consequence, or evaluation performance.

## Checkpoint preservation

Each completed seed must be backed up before the next long training run.

The canonical checkpoint and its SHA-256 hash must be recorded locally.

Loss of a Kaggle/runtime filesystem must not invalidate a completed seed.

## Consequence collection

After all five Walker2d CQL checkpoints pass verification, collect a new
Walker2d-specific consequence dataset.

Do not reuse Hopper consequence observations as Walker2d outcomes.

The Walker2d collector should preserve the structural choices of the frozen
P1/P5 consequence methodology where they remain well-defined:

- same perturbation sigma levels;
- same horizons;
- H=10 as the primary horizon;
- same state/noise seed convention;
- same support-reference construction where applicable;
- newly generated clean and shifted actions;
- newly generated downstream consequences.

Any Walker2d-specific state/environment handling must be documented before
the consequence collection is frozen.

## Primary consequence outcome

At horizon H=10:

    C10 = absolute_consequence

## Primary predictor

    action_disagreement

## Primary question

Determine whether the positive action-disagreement to downstream-consequence
relationship evaluated under CQL on Hopper is also reproduced under CQL on
Walker2d-medium.

## Statistical convention

Policy seed remains the independent unit.

Individual states, sigma levels, and counterfactual records are not treated
as independent policy-level replications.

The primary analysis must be specified and frozen before inspecting the
Walker2d consequence results.

## Comparison with previous experiments

Walker2d CQL is a separate environment/policy configuration.

CQL and IQL remain separate policy families for inference.

Raw regression coefficients across different environments are descriptive
unless their scales and estimands are demonstrably comparable.

No pooled CQL+IQL primary inference is permitted.

## Interpretation boundary

A positive Walker2d CQL result supports reproduction of the evaluated
relationship for the Walker2d-medium CQL setting.

It does not establish generalization to all environments, datasets,
offline-RL algorithms, CQL configurations, or training procedures.

A null or mixed result must be reported as observed and must not trigger
post-hoc tuning intended to improve the result.
