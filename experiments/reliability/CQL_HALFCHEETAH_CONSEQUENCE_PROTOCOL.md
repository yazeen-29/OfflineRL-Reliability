# CQL HalfCheetah — Consequence Collection Protocol

## Status

FROZEN BEFORE CONSEQUENCE COLLECTION

## Objective

Test whether the action-disagreement to downstream-consequence
relationship evaluated for CQL in prior environments is reproduced
in HalfCheetah.

This is a separate algorithm/environment evaluation cell.

## Algorithm

CQL

## Task

`mujoco/halfcheetah/medium-v0`

## Environment

`HalfCheetah-v5`

## Policy seeds

0, 1, 2, 3, 4

The five policy seeds are the independent replication units.

No seed may be selected, removed, or replaced based on training,
performance, reliability, consequence, or evaluation results.

## Frozen training boundary

The consequence experiment uses the already completed 100,000-step
CQL HalfCheetah checkpoints.

No additional training is permitted for this experiment.

## Reference dataset

Use the Minari dataset associated with:

`mujoco/halfcheetah/medium-v0`

Reference fraction:

`0.90`

The reference episodes are shuffled with:

`STATE_SAMPLING_SEED = 20260828`

Reference observations are standardized using the frozen reference
mean and standard deviation, then indexed for nearest-support-distance
calculation.

## Decision-state sampling

For each policy seed:

- Engineering gate: 4 episodes × 5 states per episode = 20 states.
- Final collection: 20 episodes × 5 states per episode = 100 states.

State sampling uses:

`STATE_SAMPLING_SEED = 20260828`

Each selected state preserves its simulator state for exact
counterfactual restoration.

## Observation corruption

Gaussian observation corruption is applied using:

`NOISE_SEED = 20260829`

The frozen sigma levels are:

- 0.00
- 0.01
- 0.025
- 0.05
- 0.10
- 0.20
- 0.30

For each decision state, one standard-normal noise vector is sampled
and scaled by sigma and the frozen reference observation standard
deviation.

The sigma=0 condition is retained in the raw dataset as the identity
condition.

## Downstream horizons

The frozen horizons are:

- H = 1
- H = 5
- H = 10
- H = 20

Primary horizon:

`H = 10`

## Counterfactual construction

For each selected decision state:

1. Restore the clean and shifted environments to the identical saved
   simulator state.
2. Verify that the restored states agree.
3. Evaluate the clean observation and shifted observation with the same
   trained CQL policy.
4. Execute the clean and shifted counterfactual branches under matched
   horizon-specific seeds.
5. Calculate the clean and shifted cumulative returns.
6. Calculate:

   `delta_J = clean_return - shifted_return`

7. Calculate:

   `absolute_consequence = abs(delta_J)`

## Primary predictor

`action_disagreement`

defined as the Euclidean action difference normalized by the square root
of the action dimension.

## Primary outcome

`absolute_consequence` at:

`H = 10`

## Primary raw-data subset

Only records satisfying:

- horizon = 10
- sigma > 0

enter the primary slope analysis.

The six nonzero sigma levels are therefore:

0.01, 0.025, 0.05, 0.10, 0.20, 0.30

Each seed contributes:

100 states × 6 sigma levels = 600 primary observations.

All five seeds therefore contribute:

5 × 600 = 3,000 primary observations.

## Full raw collection size

The complete dataset contains:

5 seeds × 100 states × 7 sigma levels × 4 horizons

= 14,000 records.

Expected records per seed:

2,800

## Primary estimand

For each policy seed and each decision state, fit:

`C10 = intercept + beta_state * action_disagreement`

across the six nonzero sigma levels.

For each policy seed:

`beta_seed = mean(beta_state)`

across the 100 decision states.

The five beta_seed values are the independent replication units.

## Primary cross-seed inference

Report:

- mean of the five seed-level slopes
- sample standard deviation
- two-sided 95% t-based confidence interval
- exact one-sided seed-level sign-flip p-value
- exact two-sided seed-level sign-flip p-value
- number of positive seed-level slopes / 5

The exact sign-flip test enumerates all:

`2^5 = 32`

possible sign assignments.

No state, sigma level, or individual consequence record is treated as
an independent policy replication.

## Secondary descriptive analyses

For each seed, report dose-level descriptive summaries for:

- action_disagreement
- absolute_consequence
- support_distance

Spearman and Kendall dose-response summaries may be reported over the
six nonzero sigma levels.

At H=10 and sigma>0, secondary support-conditioned regressions may
describe:

- mean action_disagreement ~ support_distance
- mean absolute_consequence ~ support_distance

These are secondary and do not replace the primary estimand.

## Required provenance

Every record must preserve, where applicable:

- policy seed
- state id
- source episode id
- source step
- checkpoint path
- repository commit
- state sampling seed
- noise seed
- sigma
- horizon
- standardized noise
- clean action
- shifted action
- action disagreement
- support distance
- nearest reference index
- twin critic values/disagreement
- clean return
- shifted return
- delta_J
- absolute consequence
- relative consequence
- clean/shifted trajectory lengths
- termination and truncation indicators

## Engineering gate

Before final collection, seed 0 must successfully produce:

4 episodes × 5 states × 7 sigma levels × 4 horizons

= 560 records.

The gate must confirm:

- checkpoint loads successfully
- 20 decision states are collected
- exactly 7 sigma levels are represented
- exactly 4 horizons are represented
- 560 records are produced
- all expected provenance fields are present
- numerical outputs are finite
- sigma=0 produces zero action disagreement
- restored clean and shifted simulator states validate successfully

A failed engineering gate must be resolved before final collection.

## Final collection acceptance

Final collection is accepted only when:

- all five seeds complete
- each seed produces exactly 2,800 records
- total records = 14,000
- all expected sigma levels are present
- all expected horizons are present
- no duplicate state/sigma/horizon combinations exist within a seed
- all required provenance fields are present
- numerical outputs are finite

## Reproducibility boundary

The consequence protocol is frozen before observing the HalfCheetah
consequence result.

No post-hoc changes to:

- policy seeds
- state sampling
- sigma levels
- horizons
- primary horizon
- primary outcome
- primary predictor
- seed-level aggregation
- sign-flip inference

are permitted.

## Analysis boundary

CQL HalfCheetah is analyzed as a separate policy/environment cell.

CQL and IQL policy seeds are not pooled for the primary inference.

Any comparison with other algorithms or environments is descriptive
and evidence-based.

## Software

Use the repository's existing counterfactual, critic-disagreement,
support-distance, and CQL consequence machinery without changing the
scientific estimator for this experiment.

## Collector

`src/reliability/p7_cql_halfcheetah_consequence_full.py`
