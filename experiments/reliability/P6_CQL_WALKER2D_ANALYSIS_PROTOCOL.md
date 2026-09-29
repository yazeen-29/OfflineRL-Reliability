# P6 CQL Walker2d — Consequence Analysis Protocol

Status:
FROZEN BEFORE FINAL ANALYSIS

## Objective

Determine whether the action-disagreement to downstream-consequence
relationship established in the IQL decision-consequence study is
reproduced for CQL on Walker2d.

## Frozen data

Input:

    results/reliability/P6_CQL_Walker2d/raw/seed0.json
    results/reliability/P6_CQL_Walker2d/raw/seed1.json
    results/reliability/P6_CQL_Walker2d/raw/seed2.json
    results/reliability/P6_CQL_Walker2d/raw/seed3.json
    results/reliability/P6_CQL_Walker2d/raw/seed4.json

Policy seeds:

    0, 1, 2, 3, 4

Task:

    mujoco/walker2d/medium-v0

Environment:

    Walker2d-v5

Algorithm:

    CQL

## Primary subset

Primary horizon:

    H = 10

Primary sigma condition:

    sigma > 0

Nonzero sigma levels:

    0.01, 0.025, 0.05, 0.10, 0.20, 0.30

The sigma=0 condition is excluded from the primary slope analysis
because it is the identity condition.

Each policy seed contributes:

    100 decision states
    x 6 nonzero sigma levels
    = 600 primary observations

## Primary outcome

C10:

    absolute_consequence

where:

    absolute_consequence = |delta_J|

and:

    delta_J = clean_return - shifted_return

## Primary predictor

    action_disagreement

defined as:

    ||a_clean - a_shifted||_2 / sqrt(d_a)

## Primary estimand

For each policy seed and each decision state, fit:

    C10 = intercept + beta_state * action_disagreement

across the six nonzero sigma levels.

For each policy seed:

    beta_seed = mean(beta_state)

across the 100 decision states.

The five beta_seed values are the independent replication units.

## Primary cross-seed inference

Report:

    mean of five seed-level slopes
    sample standard deviation
    two-sided 95% t-based confidence interval
    exact one-sided seed-level sign-flip p-value
    exact two-sided seed-level sign-flip p-value
    positive seed-level slopes / 5

The sign-flip test enumerates all 2^5 = 32 sign assignments.

No individual state, sigma level, or raw consequence record is
treated as an independent policy replication.

## Secondary dose-response analysis

For each seed, report descriptive means across the six nonzero
sigma levels for:

    action_disagreement
    absolute_consequence
    support_distance

Spearman and Kendall dose-response summaries may be reported
descriptively.

These are secondary summaries and are not treated as additional
independent policy replications.

## Secondary support-conditioned analysis

At H=10 and sigma>0, collapse each decision state across the six
nonzero sigma levels.

Report descriptive regressions of:

    mean action_disagreement ~ support_distance

and:

    mean absolute_consequence ~ support_distance

The seed-level estimates may be summarized using the same seed-level
confidence-interval and exact sign-flip convention as a secondary
analysis.

## Critic disagreement

Twin-critic disagreement is treated as a recorded secondary signal.

It is not the primary predictor.

No post-hoc predictor selection is permitted based on observed P6
results.

## Comparison with prior experiments

P6 is analyzed as a separate CQL Walker2d policy family/environment
experiment.

Do not pool P6 seed-level estimates with:

    IQL Hopper
    IQL HalfCheetah
    CQL Hopper

for primary inference.

Cross-experiment comparisons are descriptive and evidence-based.

## Interpretation boundary

A positive result supports reproduction of the evaluated
action-disagreement/consequence relationship for the CQL Walker2d
setting.

It does not establish generalization to all offline-RL algorithms,
datasets, environments, or training configurations.

A null or negative result must be reported as observed.

No post-hoc CQL tuning, state selection, sigma selection, or predictor
selection is permitted to improve the P6 result.

## Reproducibility

Record:

    P6 consequence protocol SHA256
    P6 analysis protocol SHA256
    P6 collector SHA256
    repository commit
    five checkpoint SHA256 values
    five raw-result SHA256 values
    canonical final archive SHA256
