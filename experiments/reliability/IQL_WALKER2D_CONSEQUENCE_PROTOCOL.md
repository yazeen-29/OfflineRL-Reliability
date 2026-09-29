# IQL Walker2d Consequence Collection Protocol

Status: FROZEN BEFORE DATA COLLECTION

## 1. Objective

Collect the decision-consequence dataset for the independently trained
IQL Walker2d policies under Gaussian observation corruption.

This experiment is an environment-generalization replication of the
established IQL consequence methodology.

## 2. Task

- Minari dataset: mujoco/walker2d/medium-v0
- Environment: Walker2d-v5
- Algorithm: IQL
- Training: 100,000 steps
- Policy seeds: 0, 1, 2, 3, 4

## 3. Policy checkpoints

- checkpoints/iql_100k_walker2d/iql_seed0/iql_mujoco_walker2d_medium-v0_seed0.d3
- checkpoints/iql_100k_walker2d/iql_seed1/iql_mujoco_walker2d_medium-v0_seed1.d3
- checkpoints/iql_100k_walker2d/iql_seed2/iql_mujoco_walker2d_medium-v0_seed2.d3
- checkpoints/iql_100k_walker2d/iql_seed3/iql_mujoco_walker2d_medium-v0_seed3.d3
- checkpoints/iql_100k_walker2d/iql_seed4/iql_mujoco_walker2d_medium-v0_seed4.d3

## 4. Reference support set

- Reference fraction: 0.90
- Episode ordering/shuffle seed: 20260828
- Standardization is fit only on the 90% reference subset.
- Nearest-neighbor support distance is computed in standardized observation
  space using that reference subset.

## 5. Evaluation state collection

For each policy seed:

- 20 clean evaluation episodes.
- 5 eligible states sampled from each episode.
- 100 states per policy seed.
- State sampling seed: 20260828.
- States must have finite observations and finite policy actions.

The evaluation trajectory is generated using the frozen policy.

## 6. Observation corruption

For every selected state:

- Draw one standard-normal noise vector.
- Reuse that same vector across all sigma levels for that state.
- Sigma levels:

  [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]

- Corrupted observation:

  shifted_obs = clean_obs + standard_normal * sigma * reference_std

The clean observation corresponds to sigma = 0.

## 7. Action disagreement

For each state and sigma:

- Predict the clean action from clean_obs.
- Predict the shifted action from shifted_obs.
- Compute normalized action disagreement:

  ||clean_action - shifted_action||_2 / sqrt(action_dimension)

## 8. Decision consequence

For every state, sigma and horizon:

- Horizons: 1, 5, 10, 20
- Restore the exact same simulator state for both branches.
- Validate simulator-state equality before branching.
- The clean branch uses the clean observation for the first action.
- The shifted branch uses the corrupted observation for the first action.
- After the first action, each branch continues using its own simulator
  observations and the frozen policy.
- No parameter updates occur during collection.

Primary horizon: H = 10.

## 9. Recorded quantities

Each record must include, at minimum:

- policy_seed
- state_id
- source_episode_id
- source_step
- checkpoint
- git_commit
- state_sampling_seed
- noise_seed
- sigma
- horizon
- standardized_noise
- clean_action
- shifted_action
- action_disagreement
- support_distance
- nearest_reference_index
- q1_clean
- q2_clean
- critic_disagreement
- twin_critic_disagreement
- clean_return
- shifted_return
- delta_J
- absolute_consequence
- relative_consequence
- clean_steps
- shifted_steps
- clean_terminated
- shifted_terminated
- clean_truncated
- shifted_truncated

## 10. Expected data volume

Per policy seed:

20 episodes × 5 states = 100 states

100 states × 7 sigma levels × 4 horizons = 2,800 records.

Across five independently trained policy seeds:

5 × 2,800 = 14,000 records.

## 11. Primary analysis

The primary endpoint is H10.

For each policy seed and each selected state:

- regress absolute_consequence on action_disagreement across the six
  nonzero sigma levels;
- obtain a state-level slope;
- average the 100 state-level slopes to obtain the policy-seed-level
  primary slope.

Primary inference:

- retain policy seeds as the inferential unit;
- do not pool policy seeds for the primary test;
- use the exact 2^5 sign-flip test over the five seed-level slopes;
- report the seed-level mean, SD, 95% t interval, exact one-sided and
  two-sided sign-flip p-values, and number of positive seed-level slopes.

## 12. Secondary diagnostics

Report, where supported:

- support-distance relationships,
- support -> action disagreement,
- support -> absolute consequence,
- dose-response rank correlations,
- state-level slope distribution,
- per-seed robustness summaries.

## 13. Reproducibility requirements

The collector must record:

- frozen checkpoint path,
- current Git commit,
- dataset identity,
- state sampling seed,
- noise seed,
- sigma levels,
- horizons,
- reference fraction.

No policy seed may be selected or excluded based on observed reliability or
consequence results.

No pooling with P5 CQL Hopper or P6 CQL Walker2d is permitted for the
primary inference.

## 14. Acceptance criteria

A final seed file is accepted only if:

- exactly 2,800 records are present;
- exactly 100 state IDs are represented;
- all seven sigma levels are represented;
- all four horizons are represented;
- sigma = 0 action disagreement is zero within numerical tolerance;
- no duplicate state/sigma/horizon combinations exist;
- all recorded numeric quantities are finite;
- provenance fields are present;
- the collection status is explicitly final_collection.
