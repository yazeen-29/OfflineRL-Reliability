# P5 CQL Generalization Analysis Protocol

Status:
Pre-analysis specification for the frozen P5 CQL Hopper consequence dataset.

## Objective

Determine whether the action-disagreement to downstream-consequence
relationship established in the decision-consequence study is reproduced
for Conservative Q-Learning (CQL).

## Frozen data

Input:

    results/reliability/P5_CQL/raw/seed0.json
    results/reliability/P5_CQL/raw/seed1.json
    results/reliability/P5_CQL/raw/seed2.json
    results/reliability/P5_CQL/raw/seed3.json
    results/reliability/P5_CQL/raw/seed4.json

Policy seeds:

    0, 1, 2, 3, 4

Task:

    mujoco/hopper/medium-v0

Algorithm:

    CQL

## Primary subset

Primary horizon:

    H = 10

Sigma condition:

    sigma > 0

The sigma=0 condition is excluded from the primary slope analysis because
it is the identity condition and produces no action-instability variation.

Nonzero sigma levels:

    0.01, 0.025, 0.05, 0.10, 0.20, 0.30

Each policy seed contributes:

    100 decision states
    x 6 nonzero sigma levels
    = 600 primary observations

## Primary outcome

C10:

    absolute_consequence

defined by the frozen P5 consequence dataset.

## Primary predictor

    action_disagreement

## Primary estimand

For each policy seed and each decision state, fit:

    C10 = intercept + beta_state * action_disagreement

across the six nonzero sigma levels.

The policy-level slope is:

    beta_seed = mean(beta_state)

across the 100 decision states for that policy seed.

The five policy-level slopes are the independent replication units.

## Primary cross-seed inference

Report:

    mean of five seed-level slopes
    sample standard deviation
    two-sided 95% t-based confidence interval
    exact one-sided seed-level sign-flip p-value
    exact two-sided seed-level sign-flip p-value
    number of positive seed-level slopes / 5

The exact sign-flip test enumerates all 2^5 = 32 sign assignments.

No individual state, sigma level, or consequence record is treated as an
independent policy replication.

## Secondary dose-response summaries

For each seed, report descriptive dose-level means for:

    action_disagreement
    absolute_consequence
    support_distance

Spearman and Kendall dose-response summaries may be reported descriptively
over the six nonzero sigma levels.

These summaries are not treated as additional independent policy
replications.

## Secondary support-conditioned analyses

At H=10 and sigma>0, collapse each decision state across its six nonzero
sigma levels by averaging:

    action_disagreement
    absolute_consequence

For each policy seed, report descriptive regressions of:

    mean action_disagreement ~ support_distance
    mean absolute_consequence ~ support_distance

The five seed-level estimates may be summarized using the same seed-level
CI and exact sign-flip convention, but these are secondary analyses.

## Reliability-score analysis

After the primary P5 consequence analysis is frozen, the frozen P2
reliability-score specification may be evaluated as a secondary CQL
generalization analysis.

No CQL-specific hyperparameter tuning or model selection is permitted
based on P5 consequence results.

## Comparison with IQL

CQL and IQL are analyzed as separate policy families.

Primary policy-family inference must not pool CQL and IQL policy seeds.

Comparison between IQL and CQL is descriptive and evidence-based.

## Interpretation boundary

A positive CQL result supports reproduction of the evaluated relationship
for the Hopper CQL setting.

It does not establish generalization to all offline-RL algorithms,
datasets, environments, or CQL configurations.
