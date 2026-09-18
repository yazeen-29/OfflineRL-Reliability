# P1 Protocol Amendment 006

Date:
2026-09-18

Status:
Approved before publication-scale P1 consequence collection.

## Purpose

Freeze the final decision-state sampling rule and record the
five-seed shortfall before publication-scale data collection.

## Policy seeds

The publication-scale P1 consequence analysis will use the five
existing verified IQL policy seeds:

    0, 1, 2, 3, 4

No additional IQL policy training will be performed for P1.

The existing policies are retained as previously verified policy
artifacts and are not retrained or modified based on P1 outcomes.

## Seed shortfall

The original protocol targeted ten policy seeds with a minimum
acceptable set of eight. Due to the study's fixed compute/scope
constraint, the publication-scale P1 collection will proceed with
the five existing verified policy seeds.

This shortfall is recorded before final P1 data collection.
The analysis will use all five available policy seeds without
post-hoc seed selection.

## Final decision-state sampling

For each policy seed:

    20 clean-policy evaluation episodes

From each episode:

    randomly sample 5 eligible decision states without replacement

Eligibility remains:

    non-terminal, valid simulator states from clean evaluation
    trajectories.

Thus:

    20 episodes × 5 states = 100 decision states per policy seed.

Across the five policy seeds:

    5 × 100 = 500 decision states.

Episode-level separation is preserved throughout collection.

## Sampling randomness

The state-sampling seed is fixed before final data collection and
must be recorded in every final dataset manifest.

The same sampling rule is applied identically across policy seeds.

## Intervention grid

Gaussian observation-shift levels remain:

    0.00
    0.01
    0.025
    0.05
    0.10
    0.20
    0.30

Horizons remain:

    1
    5
    10
    20

Primary horizon:

    10

## Expected raw observation count

For each policy seed:

    100 states × 7 sigma levels × 4 horizons
    = 2,800 state-condition records.

Across five policy seeds:

    5 × 2,800
    = 14,000 records.

## Analysis unit

Policy seed remains the independent unit for cross-seed inference.

Decision states, dose levels, and individual rollout records are
not treated as independent policy replications.

## Separation from engineering gate

The previously completed 20-state seed-0 engineering gate remains
diagnostic/engineering evidence only.

It is not substituted for the 100-state final seed-0 collection.

## No outcome-driven modification

No final P1 sampling rule, policy seed, hyperparameter, or collection
condition may be changed based on inspection of the 20-state
engineering-gate outcomes.

## Primary outcome

    C10 = |Delta_J(10)|

Secondary outcome:

    signed Delta_J(10)

Secondary operational outcomes:

    branch termination/failure indicators.

## Model evaluation

Candidate predictive models remain those specified in the previously
approved P1 amendments.

Final model selection remains based on held-out policy-seed predictive
evaluation and not on in-sample seed-0 results.
