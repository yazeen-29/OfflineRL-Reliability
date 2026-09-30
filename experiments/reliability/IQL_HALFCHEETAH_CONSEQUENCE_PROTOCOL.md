# IQL HalfCheetah Decision-Consequence Collection Protocol

## Status

FROZEN BEFORE CONSEQUENCE COLLECTION

This protocol defines the consequence collection procedure for the
IQL HalfCheetah generalization cell.

## Objective

Collect the same frozen observation-shift consequence measurements used
in the established reliability pipeline for:

- Algorithm: IQL
- Dataset: mujoco/halfcheetah/medium-v0
- Environment: HalfCheetah-v5
- Policy seeds: 0, 1, 2, 3, 4

The collected data will later support the distance-matched control
generalization analysis.

## Policy checkpoints

Use only the following previously trained checkpoints:

- seed 0:
  checkpoints/iql_100k_halfcheetah/iql_seed0/iql_mujoco_halfcheetah_medium-v0_seed0.d3
- seed 1:
  checkpoints/iql_100k_halfcheetah/iql_seed1/iql_mujoco_halfcheetah_medium-v0_seed1.d3
- seed 2:
  checkpoints/iql_100k_halfcheetah/iql_seed2/iql_mujoco_halfcheetah_medium-v0_seed2.d3
- seed 3:
  checkpoints/iql_100k_halfcheetah/iql_seed3/iql_mujoco_halfcheetah_medium-v0_seed3.d3
- seed 4:
  checkpoints/iql_100k_halfcheetah/iql_seed4/iql_mujoco_halfcheetah_medium-v0_seed4.d3

No policy retraining is performed.

## State sampling

For every policy seed:

- 20 source episodes
- 5 sampled states per episode
- 100 final states per seed
- state sampling seed: 20260828
- episode reset seeds: 10000 + episode_id

States are sampled from clean policy rollouts.

## Observation reference set

Use the Minari dataset:

mujoco/halfcheetah/medium-v0

Use:

- reference fraction: 0.90
- state/reference split seed: 20260828
- reference-only standardization
- nearest-neighbor support distance in standardized observation space

## Observation perturbation

For every selected state, construct Gaussian observation shifts using
one standardized Gaussian noise vector shared across sigma levels.

Sigma levels:

[0.00, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]

Noise seed:

20260829

The shifted observation is:

shifted_obs = clean_obs + standard_normal * sigma * reference_std

## Counterfactual horizons

Evaluate:

[1, 5, 10, 20]

Primary horizon:

H = 10

## Recorded quantities

Every record must include, at minimum:

- policy_seed
- state_id
- source_episode_id
- source_step
- sigma
- horizon
- action_disagreement
- support_distance
- absolute_consequence
- critic_disagreement
- twin_critic_disagreement
- clean_return
- shifted_return
- delta_J
- provenance fields

Absolute consequence is:

absolute_consequence = |clean_return - shifted_return|

Action disagreement is the normalized Euclidean action difference used
by the existing frozen consequence collector.

## Primary analysis population

The later distance-matched analysis will use:

- H = 10
- sigma > 0
- policy seeds 0-4

Sigma = 0 is retained in the raw collection for completeness but is not
part of the primary matched analysis.

## Expected collection size

Per seed:

100 states × 7 sigma levels × 4 horizons = 2800. 

Correction: each selected source state is collected once, therefore:

100 states × 7 sigma levels × 4 horizons = 2800 records per seed.

Across five policy seeds:

14,000 records.

## Integrity requirements

For every seed:

- exactly 100 states
- exactly 2800 records
- exactly 7 sigma levels
- exactly 4 horizons
- no duplicate state × sigma × horizon cells
- sigma = 0 action disagreement must be zero
- all required numeric quantities must be finite
- source provenance must be complete
- final status must be `final_collection`

## Reproducibility

Record:

- repository commit
- checkpoint path
- source-data metadata
- state sampling seed
- noise seed
- reference fraction
- sigma levels
- horizons
- environment/software versions
- raw-data SHA256 hashes

## Interpretation boundary

This collection does not establish causal effects.

The subsequent matched analysis tests association between action
disagreement and downstream consequence after controlling the matching
variables specified by its frozen analysis protocol.
