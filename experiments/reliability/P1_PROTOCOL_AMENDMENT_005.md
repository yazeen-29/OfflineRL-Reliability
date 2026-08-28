# P1 Protocol Amendment 005

Date:
2026-08-28

Status:
Approved before additional-policy training.

Purpose:
Define the transition from seed-0 diagnostic analysis to the
publication-scale multi-seed predictive study.

Seed-0 diagnostic interpretation:

The diagnostic analysis indicates that local action disagreement
is strongly associated with counterfactual consequence magnitude,
while support distance and twin-critic disagreement provide little
incremental in-sample explanatory value in the current seed-0 data.

This diagnostic observation does NOT select the final predictive
model.

Final model selection is deferred until held-out policy-seed
evaluation.

Publication-scale candidate models:

    A:
        action disagreement

    B:
        action disagreement + support distance

    C:
        action disagreement + twin-critic disagreement

    D:
        action disagreement + support distance
        + twin-critic disagreement

    E:
        action disagreement + support distance
        + action-disagreement × support-distance interaction

All candidate models will be evaluated using identical held-out
policy-seed splits.

The final method will be selected using pre-specified predictive
evaluation criteria and not by optimizing performance on the
existing seed-0 diagnostic dataset.

Primary continuous outcome:
    C10 = |Delta_J(10)|

Secondary outcome:
    signed Delta_J(10)

Secondary operational outcomes:
    branch termination/failure indicators.

The five original Gaussian replication results remain frozen and
are not recomputed for model selection.

No seed-0 diagnostic result is automatically promoted to the
canonical final-results registry.
