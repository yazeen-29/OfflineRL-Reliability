# P1 IQL Training Specification V1

Status:
FROZEN BEFORE ADDITIONAL POLICY TRAINING

Purpose:
Define the reproducible training configuration for newly trained
IQL policies used in the publication-scale P1 predictive study.

## Policy

Algorithm:
IQL

Environment dataset:
mujoco/hopper/medium-v0

Training steps:
100000

Policy seeds:
5, 6, 7, 8, 9

Existing verified reference policies:
0, 1, 2, 3, 4

## Reproducibility requirement

For seeds 5–9, the training configuration must remain identical across
seeds. The policy seed is the only intentional experimental factor.

The final training implementation must use the repository's frozen
training configuration and must record:

- Git commit
- Git working-tree status
- Python version
- PyTorch version
- NumPy version
- d3rlpy version
- device
- CUDA version
- dataset identifier
- dataset episode count
- algorithm
- training steps
- policy seed
- checkpoint path
- checkpoint SHA-256
- checkpoint size

## Checkpoint verification

Every newly trained policy must:

1. save a complete d3rlpy checkpoint;
2. reload the checkpoint;
3. run the predefined deterministic verification evaluation;
4. compare in-memory and reloaded outputs;
5. record the numerical difference;
6. receive an explicit PASS/FAIL status.

A policy with failed checkpoint/reload verification must not enter
the publication-scale predictive analysis.

## Independence

The policy seeds are treated as independent policy-training
replications.

No P1 consequence result may be used to modify training
hyperparameters or choose a policy seed.

## Environment and software provenance

Existing seeds 0–4 are retained as prior verified policies. Their
historical metadata may contain dirty Git working-tree states and
different execution environments; those facts are retained as
provenance rather than rewritten.

Seeds 5–9 must be generated from the clean repository state associated
with this frozen specification.

## Analysis separation

Training data, policy fitting, and checkpoint generation are completed
before publication-scale P1 consequence outcomes are analyzed.

No P1 outcome is used for model selection during training.
