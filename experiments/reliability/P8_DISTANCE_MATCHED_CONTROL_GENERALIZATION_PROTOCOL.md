# P8 Distance-Matched Control Generalization Protocol

## Status

FROZEN BEFORE ANALYSIS

## Objective

Test whether the association between higher action disagreement and
larger downstream consequence remains after matching observations on
support distance and observation-shift magnitude across the complete
2 × 3 Offline RL generalization matrix.

## Generalization matrix

Six frozen algorithm/environment cells are analyzed:

1. IQL / Hopper-v5
2. IQL / HalfCheetah-v5
3. IQL / Walker2d-v5
4. CQL / Hopper-v5
5. CQL / HalfCheetah-v5
6. CQL / Walker2d-v5

## Frozen source data

IQL / Hopper:
data_frozen/P1/

IQL / HalfCheetah:
results/reliability/IQL_HalfCheetah/raw/

IQL / Walker2d:
results/reliability/IQL_Walker2d/raw/

CQL / Hopper:
results/reliability/P5_CQL/raw/

CQL / HalfCheetah:
results/reliability/P7_CQL_HalfCheetah/raw/

CQL / Walker2d:
results/reliability/P6_CQL_Walker2d/raw/

For every cell, policy seeds 0, 1, 2, 3, and 4 are required.

The source JSON files are not modified, regenerated, resampled, or
selected using observed consequence.

## Primary analysis population

For every cell:

- horizon H = 10
- sigma > 0
- policy seeds 0 through 4

Sigma = 0 is excluded because it does not provide a high-versus-low
action-disagreement contrast.

## Independent unit

Within each algorithm/environment cell, policy seed is the independent
inferential unit.

Matched pairs are not treated as independent policy replicates.

Cross-cell results are synthesized descriptively. No pooled
pair-level hypothesis test is used as evidence of generalization.

## Matching structure

Matching is performed independently within every:

policy_seed × sigma

combination.

Within each combination:

1. Sort observations by support_distance.
2. Divide observations into 10 approximately equal-frequency
   support-distance strata using deterministic rank ordering.
3. Identify the lowest quartile of action_disagreement as LOW.
4. Identify the highest quartile of action_disagreement as HIGH.
5. Require at least two observations in each group for a stratum to
   contribute matched pairs.
6. Sort LOW and HIGH groups by support_distance.
7. Pair equal-order observations without replacement.
8. The number of pairs is the smaller group size.

Matching uses only:

- support_distance
- sigma
- policy_seed
- action_disagreement
- deterministic state_id tie-breaking

Observed consequence is never used to form matches.

## Primary outcome

For each matched pair:

pair_delta_C10 =
    C10_high_action_disagreement
    -
    C10_low_action_disagreement

where C10 is absolute_consequence at H = 10.

For each policy seed, compute the mean pair_delta_C10 over all eligible
matched pairs.

The five seed-level means are the primary independent observations for
each algorithm/environment cell.

## Primary inference

For every cell report:

- mean seed-level matched effect
- SD across five seeds
- descriptive 95% t-based CI
- positive seed-effect count
- exact one-sided sign-flip p-value for mean matched effect > 0
- exact two-sided sign-flip p-value

The one-sided and two-sided values are both reported.

## Cross-cell synthesis

The six cell-level results are reported without ranking.

Report descriptively:

- cell-level mean effects
- number of positive seed effects
- number of cells with all five seed effects positive
- total number of matched pairs
- matching-balance diagnostics

No cell is declared a winner.

## Balance diagnostics

For every cell and policy seed report:

- eligible support-distance strata
- total matched pairs
- mean absolute support-distance difference
- median absolute support-distance difference
- mean action-disagreement difference
- median action-disagreement difference
- minimum action-disagreement difference
- maximum action-disagreement difference
- exact sigma matching
- excluded strata and exclusion reasons

## Sensitivity analyses

Repeat the matching using:

- 5 support-distance strata
- 20 support-distance strata

The 5-stratum analysis is a sensitivity analysis.

The 20-stratum configuration is expected to be mechanically infeasible
when each sigma cell contains 100 observations, because the minimum
quartile group size requirement is two observations.

If infeasible, report the feasibility result rather than replacing the
specification.

## Required source integrity checks

Before analysis verify every cell has:

- 5 policy seeds
- 100 states per seed
- 2800 records per seed
- 7 sigma levels
- 4 horizons
- 2800 unique state × sigma × horizon cells
- exactly 400 records per sigma level
- exactly 700 records per horizon
- 600 H=10, sigma>0 primary records
- required variables finite

Required primary variables:

- policy_seed
- state_id
- source_episode_id
- source_step
- sigma
- horizon
- support_distance
- action_disagreement
- absolute_consequence

## Provenance

Record:

- repository commit
- protocol hash
- analyzer hash
- Python version
- NumPy version
- pandas version
- SciPy version
- exact source paths
- SHA256 hash of every source seed file
- algorithm/environment metadata
- matching parameters

## Interpretation boundary

A positive matched effect supports an association between higher action
disagreement and larger downstream consequence after matching on
support distance and sigma.

It does not establish causality, eliminate all confounding, or prove
that action disagreement is the only mechanism responsible for the
observed consequence.

Replication across the six cells is evidence of cross-cell robustness
under this protocol, but effect magnitudes are not assumed to be
directly calibrated across environments or algorithms.
