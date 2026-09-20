# P4 OOD / Uncertainty Baseline Protocol

## Objective

Evaluate whether action-disagreement-based reliability provides
information beyond generic support/OOD and critic-uncertainty signals.

## Frozen input

Use only:

data_frozen/P1/

No new RL training and no modification of P1 records.

## Primary evaluation population

- environment: Hopper-medium-v0
- horizon: H=10
- nonzero observation shifts only: sigma > 0
- policy seeds: 0, 1, 2, 3, 4

The sigma=0 records are excluded from the primary P4 evaluation.

## Independent unit

Policy seed.

Held-out policy seed is the independent evaluation unit.

## Baseline score definitions

All reliability scores are monotone transforms of the specified
diagnostic signal(s), with parameters determined without using the
held-out seed outcome.

### B1: Support/OOD

Raw signal:

    support_distance

Higher support distance corresponds to lower reliability.

### B2: Critic uncertainty

Raw signal:

    twin_critic_disagreement

Higher critic disagreement corresponds to lower reliability.

### B3: Support + critic

Features:

    support_distance
    twin_critic_disagreement

Fit a deterministic StandardScaler + Ridge(alpha=1.0) consequence
predictor using training policy seeds only.

Convert held-out predictions to reliability using the training-fold
empirical CDF:

    R(x) = 1 - F_train(predicted_C10)

### B4: Action disagreement

Raw signal:

    action_disagreement

Higher action disagreement corresponds to lower reliability.

### B5: Primary full formulation

Features:

    action_disagreement
    support_distance
    twin_critic_disagreement

Use the frozen P2 primary estimator:

    StandardScaler + Ridge(alpha=1.0)

## Evaluation protocol

For every held-out policy seed:

1. use the remaining four policy seeds for any fitted estimator;
2. generate held-out predictions where applicable;
3. construct the reliability score without using held-out outcomes;
4. evaluate observed H=10 consequence on the held-out seed.

Raw-score baselines B1, B2, and B4 do not require fitted parameters.
Their reliability scores are generated from training-fold empirical
distributions.

## Primary outcome

Observed absolute downstream consequence:

    C10 = absolute_consequence

## Primary P4 metrics

### Ordering

- Spearman correlation between reliability score and observed C10.

Higher reliability should correspond to lower observed consequence.

### Reliability stratification

- observed C10 across ten reliability deciles.

### Operational risk-coverage

Use coverage:

    10%, 20%, 30%, 50%, 70%, 90%, 100%

and evaluate mean observed C10 among the highest-reliability states
retained at each coverage.

The main operational comparison is descriptive across the five
held-out policy seeds.

## Baseline comparison

Report all five formulations side-by-side.

Do not declare a single best baseline or overall winner based on the
observed results.

The principal scientific question is whether action disagreement
provides a distinct reliability signal relative to support/OOD and
critic-uncertainty baselines.

## Seed-level inference

Where a formal comparison is performed, use the five held-out
policy-seed-level paired differences as the independent observations.

Use exact sign-flip inference.

No state-level observation count is treated as the independent sample
size for policy-level inference.

## Binary outcomes

Do not introduce binary failure thresholds or AUROC/AUPRC/Brier
analysis.

## Exclusions

No outcome-dependent exclusion is allowed.

Record any exclusions and their reasons.

## Reproducibility

Record:

- repository commit;
- source-data hashes;
- protocol hash;
- policy-seed split;
- features;
- estimator specification;
- software versions;
- random/stochasticity assumptions.

## Interpretation boundary

P4 can establish comparative predictive/ordering behavior among the
specified diagnostic baselines.

It does not establish causal faithfulness, universal OOD detection,
or that action disagreement is the uniquely optimal reliability signal.
