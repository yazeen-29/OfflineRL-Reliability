# P1 Protocol Amendment 004

Date:
2026-08-28

Status:
Approved before P1.3 analysis.

Purpose:
Pre-specify the clustered seed-0 analysis of the diagnostic
multi-dose predictor dataset.

Primary diagnostic endpoint:
    C10 = |Delta_J(10)|

Primary predictor:
    action disagreement

Primary incremental question:
    Does support distance explain additional variation in C10
    beyond action disagreement?

Diagnostic model hierarchy:

    M1:
        C10 ~ action_disagreement

    M2:
        C10 ~ action_disagreement + support_distance

    M3:
        C10 ~ action_disagreement + twin_critic_disagreement

    M4:
        C10 ~ action_disagreement
             + support_distance
             + twin_critic_disagreement

Interaction model:

    M5:
        C10 ~ action_disagreement
             + support_distance
             + action_disagreement:support_distance

The interaction model tests whether support distance moderates
the relationship between local action instability and downstream
counterfactual consequence magnitude.

The diagnostic analysis must account for repeated observations
within decision states and episodes.

No individual timestep or dose-level record is treated as an
independent policy replication.

Primary robustness approach:
    episode-cluster bootstrap.

Secondary robustness approach:
    state-cluster bootstrap.

Dose level is treated as a repeated intervention condition rather
than an independent observation.

All models are diagnostic and in-sample. Their results are not
publication-grade predictive estimates.

No model is selected based on the resulting effect size or p-value.
All prespecified models are reported.

The final predictive study must use held-out policy seeds.
