# P6 CQL Walker2d — Decision-Consequence Collection Protocol

## Status

Protocol status: FROZEN BEFORE FULL DATA COLLECTION

## Objective

Evaluate the decision-consequence relationship for five frozen CQL
policies in the Walker2d-medium offline-RL setting using the same
counterfactual collection structure established for the frozen P1 and
P5 studies.

## Environment and dataset

Task:

    mujoco/walker2d/medium-v0

Environment:

    Walker2d-v5

Algorithm:

    CQL

The five CQL policies are the frozen P6 training policies corresponding
to policy seeds 0, 1, 2, 3, and 4.

## Policy seeds

Policy seeds:

    0, 1, 2, 3, 4

Policy seed is the policy-level independent replication unit.

No policy seed may be selected or removed using observed consequence or
reliability results.

## Decision-state sampling

For each policy seed:

    20 clean evaluation episodes
    5 eligible decision states sampled without replacement per episode

Therefore:

    100 decision states per policy seed

Terminal and invalid states are excluded from the primary collection.

State sampling seed:

    20260828

## Reference set

Reference fraction:

    0.90

The support-reference construction follows the established
reference-standardized observation-space procedure.

## Observation shift

Primary perturbation:

    Gaussian observation shift

Sigma levels:

    0.00
    0.01
    0.025
    0.05
    0.10
    0.20
    0.30

Noise seed:

    20260829

Noise is generated in reference-standardized observation coordinates
and applied only to the policy observation.

## Counterfactual intervention

For each decision state:

1. Capture the exact simulator state.
2. Restore that state in independent clean and shifted simulator
   branches.
3. Construct the clean observation.
4. Construct the shifted observation using the specified Gaussian
   perturbation.
5. Evaluate the frozen CQL policy on both observations.
6. Execute the clean and shifted actions from the same underlying
   simulator state.
7. Continue both branches for the specified downstream horizon.
8. Record rewards, termination/truncation, returns, action disagreement,
   support distance, critic disagreement, and consequence measures.

The true simulator state at the intervention point must be identical
across the clean and shifted branches.

## Horizons

Horizons:

    H = 1
    H = 5
    H = 10
    H = 20

Primary horizon:

    H = 10

The primary horizon is fixed before full collection.

## Outcomes

For each horizon H:

    J_clean(H) = cumulative clean-branch reward
    J_shifted(H) = cumulative shifted-branch reward

Signed consequence:

    Delta_J(H) = J_clean(H) - J_shifted(H)

Absolute consequence magnitude:

    C(H) = |Delta_J(H)|

Relative consequence:

    C_rel(H) =
        |Delta_J(H)| /
        (1e-8 + 0.5 * (|J_clean(H)| + |J_shifted(H)|))

## Predictors recorded

Support distance:

    nearest-neighbor Euclidean distance in the frozen
    reference-standardized observation space.

Action disagreement:

    Delta_a =
        ||a_clean - a_shifted||_2 / sqrt(d_a)

Twin-critic disagreement:

    U_twin = |Q1 - Q2|

Twin-critic disagreement is recorded as a diagnostic uncertainty
quantity and is not treated as a calibrated epistemic uncertainty
estimator.

## Dataset size

Per policy seed:

    100 states × 7 sigma levels × 4 horizons
    = 2,800 records

Across five policy seeds:

    14,000 records

## Required record provenance

Each record must preserve, where applicable:

- policy seed;
- state identifier;
- source episode and source step;
- checkpoint path;
- repository commit;
- state sampling seed;
- noise seed;
- sigma;
- horizon;
- standardized perturbation;
- clean action;
- shifted action;
- action disagreement;
- support distance;
- nearest reference index;
- clean critic values;
- critic disagreement;
- clean return;
- shifted return;
- signed consequence;
- absolute consequence;
- relative consequence;
- termination/truncation indicators.

## Acceptance criteria

A final seed collection is usable only if:

- the correct frozen Walker2d CQL checkpoint is loaded;
- the collection completes without fatal errors;
- all 100 decision states are represented;
- all seven sigma levels are represented;
- all four horizons are represented;
- exactly 2,800 records are produced;
- required fields are present and finite where numeric;
- policy seed provenance is consistent;
- repository/protocol provenance is recorded.

## Statistical boundary

Policy seed remains the independent unit for cross-seed inference.

Individual decision states, sigma levels, and consequence records are
not treated as independent policy-level replications.

The primary analysis specification will be frozen separately before
analysis of the completed P6 consequence dataset.

## Reproducibility boundary

The P6 consequence dataset is newly generated from the five frozen
Walker2d CQL policies.

The frozen IQL P1 consequence observations are not reused as CQL
outcomes.

The P5 CQL methodology is used as the methodological reference, but
Walker2d results are treated as a separate environment extension.

