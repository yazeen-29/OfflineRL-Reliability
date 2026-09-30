# P10 Amendment 001 — Interpretation Boundary

## Purpose

Clarify the interpretation of the completed P10 standalone OOD
analysis after auditing the frozen diagnostic distributions.

## Finding

Across all six algorithm/environment cells:

- sigma=0 observations have exactly zero action disagreement;
- sigma>0 observations have strictly positive action disagreement;
- support distance is unchanged across sigma levels;
- twin-critic disagreement is unchanged across sigma levels.

Therefore, the observed AUROC=1.0 for action disagreement reflects
deterministic separation of the specific controlled perturbation family
used in the frozen consequence datasets.

## Interpretation

P10 is retained as a controlled perturbation detectability sanity
check.

It is not treated as evidence of generic or universal OOD detection.

The P10 AUROC/AP results must not be presented as a deployment-level
OOD benchmark.

## Paper use

P10 may be reported as a supplemental sanity check showing that action
disagreement responds detectably to the controlled perturbation family.

The core paper claims should instead rely on:

- P1 continuous consequence prediction;
- P2 reliability estimation;
- P3 distance-matched control;
- P4 predefined OOD/uncertainty baseline comparison;
- P6/P7 cross-algorithm/environment consequence replication;
- P8 six-cell distance-matched generalization;
- P9 six-cell reliability-estimator generalization.
