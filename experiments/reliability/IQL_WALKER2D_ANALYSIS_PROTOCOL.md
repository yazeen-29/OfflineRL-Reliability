# IQL Walker2d Consequence Analysis Protocol

Status: FROZEN BEFORE ANALYSIS

## 1. Primary endpoint

The primary endpoint is the H10 decision consequence.

For each independently trained policy seed:

1. For each of the 100 sampled states, use the six nonzero sigma levels:
   0.01, 0.025, 0.05, 0.10, 0.20, 0.30.
2. Regress absolute_consequence on action_disagreement across those
   six nonzero sigma levels.
3. Retain the resulting state-level slope.
4. Average the 100 state-level slopes to obtain one seed-level slope.

The five seed-level slopes are the inferential units.

## 2. Primary inference

Do not pool policy seeds for the primary inference.

Report:

- seed-level slopes;
- mean seed-level slope;
- SD;
- 95% t-based confidence interval;
- number of positive seed-level slopes;
- exact one-sided 2^5 sign-flip p-value;
- exact two-sided 2^5 sign-flip p-value.

The exact sign-flip test treats the five observed seed-level slopes as the
fixed magnitudes and enumerates all 2^5 sign assignments.

## 3. Secondary diagnostics

Report:

- support distance -> action disagreement;
- support distance -> absolute consequence;
- dose-response rank correlations;
- state-level slope distribution;
- per-seed mean, median and trimmed state-level slope summaries;
- per-seed fraction of positive state-level slopes.

These are descriptive/secondary analyses and do not replace the primary
seed-level inference.

## 4. Heterogeneity

Retain all five policy seeds.

Do not select, remove, or downweight a seed based on its observed reliability,
consequence magnitude, or slope.

Report heterogeneous seed-level and state-level behavior rather than replacing
it with a pooled estimator.

## 5. Scope

This analysis applies only to:

- Algorithm: IQL
- Environment: Walker2d-v5
- Dataset: mujoco/walker2d/medium-v0
- Training: 100,000 steps
- Seeds: 0,1,2,3,4

The primary analysis must not pool these records with:

- IQL Hopper,
- IQL HalfCheetah,
- CQL Hopper,
- CQL Walker2d,
- or future experiments.

## 6. Input files

results/reliability/IQL_Walker2d/raw/seed0.json
results/reliability/IQL_Walker2d/raw/seed1.json
results/reliability/IQL_Walker2d/raw/seed2.json
results/reliability/IQL_Walker2d/raw/seed3.json
results/reliability/IQL_Walker2d/raw/seed4.json

## 7. Required outputs

results/analysis/IQL_Walker2d/

- IQL_WALKER2D_PRIMARY_ANALYSIS.json
- IQL_WALKER2D_SEED_SUMMARY.csv
- IQL_WALKER2D_STATE_SLOPES.csv
- IQL_WALKER2D_SECONDARY_ANALYSIS.json

## 8. Reproducibility

The analysis must record:

- input file hashes;
- Git commit;
- analysis protocol;
- primary horizon;
- sigma levels;
- number of states;
- number of policy seeds.

No post-hoc modification of the primary estimator is permitted after seeing
the five seed-level results.
