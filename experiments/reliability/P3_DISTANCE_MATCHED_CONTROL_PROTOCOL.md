# P3 Distance-Matched Control Protocol

## Objective

Test whether higher action disagreement is associated with greater
downstream consequence after matching observations on support distance
and observation-shift magnitude.

## Frozen input

Use only the frozen P1 dataset:

data_frozen/P1/

No P1 records may be changed, regenerated, resampled, or selected using
the observed consequence.

## Primary analysis population

- environment: Hopper-medium-v0
- horizon: H=10
- nonzero observation shifts only: sigma > 0
- policy seeds: 0, 1, 2, 3, 4

The sigma=0 observations are excluded because their action disagreement
and consequence are zero by construction and therefore cannot provide
a high-vs-low action-disagreement contrast.

## Independent unit

Policy seed.

All inferential comparisons are performed on seed-level matched effects.
Individual matched pairs are not treated as independent policy-level
replicates for hypothesis testing.

## Matching structure

Matching is performed independently within each:

    policy_seed × sigma

combination.

Within each combination:

1. Sort observations by support_distance.
2. Divide them into 10 approximately equal-frequency
   support-distance strata using deterministic rank ordering.
3. Within each support-distance stratum, identify:
   - LOW action-disagreement group: lowest quartile
   - HIGH action-disagreement group: highest quartile
4. Require at least two observations in each group for a stratum to
   contribute matched pairs.
5. Match HIGH and LOW observations by sorting each group by
   support_distance and pairing equal-order observations without
   replacement.
6. The number of pairs in a stratum is the smaller of the two group
   sizes.

This matching uses only support_distance, sigma, policy_seed, and
action_disagreement. The downstream consequence is not used to form
matches.

## Primary matched outcome

For each matched pair:

    pair_delta_C10 =
        C10_high_action_disagreement
        -
        C10_low_action_disagreement

where C10 is absolute_consequence at H=10.

For each policy seed, compute the mean pair_delta_C10 across all
eligible matched pairs.

The seed-level mean is the primary P3 effect estimate.

## Balance diagnostics

Report for every policy seed:

- number of eligible sigma × support-distance strata;
- number of matched pairs;
- mean and median absolute support-distance difference;
- mean and median action-disagreement difference;
- exact sigma matching;
- high/low action-disagreement distributions.

## Inferential analysis

Use the five policy-seed-level mean matched effects as the independent
observations.

Report:

- mean seed-level matched effect;
- descriptive 95% t-based CI;
- number of positive seed-level effects;
- exact one-sided sign-flip p-value for
  mean matched effect > 0.

No pair-level hypothesis test is treated as evidence across independent
policy seeds.

## Sensitivity analyses

Where feasible, additionally report:

- median pair_delta_C10 within each policy seed;
- results restricted to strata with a larger action-disagreement
  separation;
- sensitivity to 5 and 20 support-distance strata.

These are sensitivity analyses and do not replace the primary
10-stratum specification.

## Exclusions

Strata may be excluded only because they fail the pre-specified
minimum group-size requirement.

No exclusion may depend on observed consequence.

All excluded strata and reasons must be reported.

## Interpretation boundary

A positive matched effect supports an association between higher action
disagreement and larger consequence after matching on support distance
and sigma.

It does not establish causal effects or complete control of all
confounding.

## Reproducibility

Record:

- repository commit;
- source-data hashes;
- software/environment versions;
- policy seeds;
- matching parameters;
- number of support-distance strata;
- minimum group size;
- exact source columns.
