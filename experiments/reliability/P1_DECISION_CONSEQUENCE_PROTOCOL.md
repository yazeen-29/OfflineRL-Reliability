# P1 — Decision-Consequence Experiment Protocol

## Status

Protocol status: FROZEN BEFORE FULL DATA COLLECTION

Purpose:
Test whether observation distribution shift at a fixed underlying
simulator state produces local policy instability that predicts an
actual downstream decision consequence.

---

## 1. Research question

At the same underlying simulator state, does a controlled change in the
observation supplied to a frozen offline-RL policy produce action
instability that predicts downstream performance degradation?

---

## 2. Core causal structure

The physical simulator state is held fixed at the intervention point.

Two observation branches are constructed:

    same true simulator state
            |
        +---+---+
        |       |
      clean   shifted
       obs      obs
        |        |
        v        v
      clean   shifted
      action   action
        |        |
        +---+----+
            |
            v
      downstream consequence

The shifted branch differs only in the observation supplied to the policy.

---

## 3. Primary hypotheses

### H1 — support distance

Greater distance between the query observation and its nearest
held-out reference is associated with larger downstream return
degradation.

    support distance ↑  ->  return degradation ↑

### H2 — local action instability

Greater disagreement between the clean-policy action and shifted-policy
action is associated with larger downstream return degradation.

    action disagreement ↑  ->  return degradation ↑

### H3 — incremental predictive value

A model using both support distance and action disagreement predicts
downstream degradation better than either signal individually.

Primary comparison:

    Model A:
        degradation ~ support_distance

    Model B:
        degradation ~ action_disagreement

    Model C:
        degradation ~ support_distance + action_disagreement

### H4 — uncertainty comparison

Twin-critic disagreement provides an additional baseline.

    U_twin(s,a) = |Q1(s,a) - Q2(s,a)|

The combined model is evaluated against support and instability alone.

---

## 4. Experimental unit and hierarchy

Primary independent replication unit:
    independently trained policy seed

Nested observations:
    decision states / episodes within a policy seed

Individual timesteps must NOT be treated as independent policy
replications.

---

## 5. Environment

Primary initial environment:
    mujoco/hopper/medium-v0

Simulator:
    Hopper-v5

Observation dimension:
    11

Action dimension:
    3

Maximum episode length:
    1000 steps

---

## 6. Policy

Algorithm:
    IQL

Training:
    100,000 offline training steps

P1 policy seeds:
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Existing verified policies:
    seeds 0–4

New policies for the predictive study:
    seeds 5–9

Primary target:
    10 policy seeds.

Minimum acceptable:
    8 policy seeds.

If fewer than 10 seeds are available because of compute limitations,
the shortfall must be recorded before full-data collection and the
analysis must use the achieved number without post-hoc selection.

The original five-seed Gaussian results remain frozen and are not
recomputed solely for P1.

---

## 7. Decision-state sampling

Decision states are obtained from clean-policy evaluation trajectories.

The sampling rule must be fixed before outcome inspection.

Eligible states:
    non-terminal, valid simulator states from clean evaluation
    trajectories.

Terminal and invalid states are excluded from the primary analysis.

Sampling is uniform over the eligible decision-state pool unless a later
protocol amendment explicitly states otherwise.

---

## 8. Counterfactual intervention

For each decision state:

    1. Capture exact MuJoCo qpos and qvel.
    2. Restore the state in independent simulator instances.
    3. Construct the clean observation.
    4. Construct a shifted observation by adding controlled observation
       perturbation.
    5. Evaluate clean and shifted policy actions.
    6. Execute the two actions from copies of the same true state.
    7. Continue each branch for a fixed downstream horizon H = 10 environment steps.
    8. Record rewards, termination, and return.

Primary horizon:
    H = 10 steps

Sensitivity horizons:
    H = 1, 5, and 20 steps

The primary horizon is fixed before full-data collection and is not
selected using the observed effect size or predictive performance.

The true simulator state at the intervention point must be identical
across branches.

---

## 9. Observation shift

Primary perturbation:
    Gaussian observation shift

Shift levels:

    sigma ∈ {
        0.00,
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.30
    }

Noise is sampled in reference-standardized observation coordinates.

The perturbation is applied to the policy observation only.

---

## 10. Primary downstream outcome

For horizon H:

    J_clean(H)
        = cumulative reward over H clean steps

    J_shifted(H)
        = cumulative reward over H shifted steps

Primary continuous outcome:

    Delta_J(H) = J_clean(H) - J_shifted(H)

Positive Delta_J means that the shifted observation caused worse
downstream performance.

---

## 11. Secondary outcome

A binary adverse-consequence label may be evaluated as a secondary
analysis.

The primary P1 outcome is continuous downstream return degradation
Delta_J(H). The binary label is not used to define the primary success
criterion.

If a binary adverse-consequence threshold is introduced, it must be
specified and justified before the corresponding full-data analysis
and must not be selected by maximizing AUROC, AUPRC, calibration, or
another post-hoc performance metric.

---

## 12. Primary predictors

### Predictor 1 — support distance

Nearest-neighbor Euclidean distance in the frozen
reference-standardized observation space.

### Predictor 2 — action disagreement

    Delta_a =
        ||a_clean - a_shifted||_2 / sqrt(d_a)

where d_a is the action dimension.

### Predictor 3 — twin-critic disagreement

For a specified state/action pair:

    U_twin = |Q1 - Q2|

This is treated as a critic-disagreement baseline, not as a fully
calibrated epistemic uncertainty estimator.

---

## 13. Statistical hierarchy

For each policy seed:

    decision-level effects
        ↓
    seed-level summary

Primary cross-seed inference uses policy seeds as the independent
replication units.

Within-seed decision states are not counted as independent policy
replications.

---

## 14. Predictive evaluation

The reliability model must be evaluated on held-out policy seeds.

Preferred design:

    train/calibration seeds
        ↓
    fit score/model
        ↓
    held-out policy seed
        ↓
    evaluate prediction

Primary evaluation targets:

    continuous:
        return degradation

    binary:
        adverse consequence

Primary metrics:

    AUROC
    AUPRC
    MAE / RMSE where appropriate
    calibration error
    Brier score where a probabilistic outcome is used

Operational evaluation:

    risk-coverage analysis

---

## 15. Required model comparisons

The following models must be evaluated using identical held-out splits:

    A. support distance only

    B. action disagreement only

    C. support distance + action disagreement

    D. twin-critic disagreement only

    E. support distance + twin-critic disagreement

    F. support distance + action disagreement + twin-critic disagreement

The primary proposed model is C.

Model F is an extended model used to assess whether the proposed
signals add information beyond the twin-critic baseline.

---

## 16. Reproducibility

Every P1 run must record:

    policy seed
    dataset/task
    checkpoint
    state sampling seed
    perturbation seed
    shift level
    decision-state identifier
    horizon
    simulator configuration
    observation standardization statistics
    reference construction
    code version / Git commit

---

## 17. Exclusion rules

A decision point is excluded from the primary analysis only when:

    - simulator state restoration fails verification;
    - observation dimensions are invalid;
    - policy action contains non-finite values;
    - simulator execution is invalid;
    - required outcome values are missing.

Exclusion counts must be reported.

No exclusion rule may depend on whether a result supports the hypothesis.

---

## 18. Amendments

Any change to:

    - sampling
    - shift levels
    - horizon
    - outcome definition
    - model specification
    - exclusion rule

after full-data collection begins must be recorded as a protocol amendment
with a reason and must not silently replace the frozen primary analysis.

---

## 19. Relationship to existing frozen evidence

Existing frozen studies remain unchanged:

    Hopper Gaussian replication
    HalfCheetah Gaussian replication
    Structured directional shift
    Uniform-random reference control

P1 is a new consequence-oriented experiment.

It is not permitted to overwrite or retroactively modify the existing
frozen result artifacts.

---

## 19A. Protocol Freeze Rules

Before full P1 data collection begins, the following are fixed:

    Primary horizon:
        H = 10 steps

    Sensitivity horizons:
        H = 1, 5, 20 steps

    Policy-seed target:
        10

    Minimum policy-seed count:
        8

    Primary outcome:
        continuous 10-step downstream return degradation

No primary parameter may be selected or changed using the observed
direction or magnitude of P1 results.

Any later change requires an explicit protocol amendment and separate
annotation of exploratory analyses.

## 20. Success criteria

P1 is scientifically successful if it demonstrates that:

    1. support distance is associated with downstream degradation;
    2. action disagreement is associated with downstream degradation;
    3. the combined support + instability model provides meaningful
       predictive improvement over either signal alone;
    4. the result survives held-out policy-seed evaluation;
    5. the result is not fully explained by twin-critic disagreement.

Failure to meet any of these conditions must be reported honestly.

A null or weak result is not grounds for changing the protocol.

---

## 21. Next-stage generalization

After P1 is validated:

    CQL pilot
    additional environment (target: Walker2d)
    improved distance-matched reference control
    expanded uncertainty/OOD baselines

These are separate stages and must not be mixed into the primary P1
analysis retrospectively.
