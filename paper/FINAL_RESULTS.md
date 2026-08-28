# FINAL EXPERIMENTAL RESULTS

## Project

**Offline RL Reliability under Observation Distribution Shift**

Status: `IN PROGRESS — FINAL RESULTS REGISTRY`

This file is the single human-readable index of the canonical
results used for the final paper.

Only results explicitly marked `FROZEN` are eligible for manuscript
claims.

---

## 1. Primary Gaussian Observation-Shift Study

### Hopper

Status: `FROZEN`

Canonical analysis:
`results/analysis/...`

Mean seed-level slope:
`0.1342720710`

95% CI:
`[0.1223567020, 0.1461874400]`

Positive query-slope fraction:
`0.9970`

Exact one-sided sign-flip p:
`0.03125`

Exact two-sided sign-flip p:
`0.06250`

---

### HalfCheetah

Status: `FROZEN`

Canonical analysis:
`results/analysis/...`

Mean seed-level slope:
`0.0651419255`

95% CI:
`[0.0584082882, 0.0718755627]`

Positive query-slope fraction:
`0.9252`

Exact one-sided sign-flip p:
`0.03125`

Exact two-sided sign-flip p:
`0.06250`

---

## 2. Cross-Environment Replication

Status: `FROZEN`

Hopper seed-level slopes:
`5/5 positive`

HalfCheetah seed-level slopes:
`5/5 positive`

Combined:
`10/10 positive`

Canonical synthesis:
`results/analysis/cross_environment_gaussian_synthesis.json`

---

## 3. Structured Directional Shift

Status: `FROZEN`

Environment:
`Hopper`

Mean explanation dose-response slope:
`0.2699650813`

95% CI:
`[0.2325682855, 0.3073618770]`

Exact one-sided sign-flip p:
`0.03125`

Exact two-sided sign-flip p:
`0.06250`

Mean nearest-neighbor distance slope:
`2.0280882524`

Mean policy-action-change slope:
`0.4316813072`

Canonical analysis:
`results/analysis/iql_100k_multiseed_structured_analysis.json`

---

## 4. Uniform Random-Reference Control

Status: `FROZEN`

Nearest-reference mean slope:
`0.2768094429`

Uniform-random-reference mean slope:
`0.0370033350`

Nearest-minus-random difference:
`0.2398061079`

95% CI:
`[0.2164962713, 0.2631159445]`

All five seed-level differences positive:
`5/5`

Exact one-sided sign-flip p:
`0.03125`

Exact two-sided sign-flip p:
`0.06250`

Canonical analysis:
`results/analysis/iql_100k_multiseed_random_reference_control_v2.json`

---

## 5. Distance-Matched Reference Control

Status: `PENDING FINAL VALIDATION`

The existing strict matching protocol was found to have insufficient
feasibility for the original ±10% criterion.

Diagnostic result:
only `8%` of the original diagnostic queries satisfied the strict
criterion under the smoke protocol.

A redesigned rank/stratum-matched control is required before this
experiment becomes canonical evidence.

Existing implementation:
`src/explanations/distance_matched_random_reference_diagnostic.py`

---

## 6. Decision-Consequence Experiment

Status: `NOT YET RUN`

Purpose:

Test whether support distance and local action instability predict
an actual downstream performance consequence when the policy receives
a perturbed observation at a fixed underlying simulator state.

Primary outcome:
`H-step downstream return degradation`

Primary predictors:
`support distance`
`action disagreement`

Secondary predictor:
`twin-critic disagreement`

---

## 7. Reliability Score

Status: `NOT YET RUN`

Candidate models:

1. support distance only
2. action disagreement only
3. support + action disagreement
4. support + uncertainty
5. support + action disagreement + uncertainty

Validation:
held-out policy seeds

Primary question:
Does the combined score predict consequential degradation better than
individual baselines?

---

## 8. CQL Generalization

Status: `NOT YET RUN`

Protocol:
initial low-cost pilot before full study.

Purpose:
determine whether the core phenomenon is specific to IQL.

---

## 9. Walker2d Generalization

Status: `NOT YET RUN`

Purpose:
test whether the core relationship extends beyond Hopper and
HalfCheetah.

---

## 10. Final Figures

Only files inside:

`paper/figures/final/`

are canonical paper figures.

Current canonical figure families:

- Gaussian cross-environment response
- Cross-seed replication
- Distance-to-disagreement relationship
- Structured robustness

---

## 11. Final Tables

Only files inside:

`paper/tables/final/`

are canonical paper tables.

---

## 12. Statistical Convention

Primary independent replication unit:

`independently trained policy seed`

Query observations and perturbation repeats are not treated as
independent policy replications.

Exact sign-flip tests are interpreted as directional tests at the
seed level.

Raw cross-environment slope magnitudes are not interpreted as
standardized effect sizes.

---

## 13. Superseded / Diagnostic Artifacts

Older analyses, figures, smoke tests, and intermediate outputs remain
in their original locations for provenance.

They must not be cited as final results unless explicitly promoted
into the canonical registry.

---

## FINAL STATUS

Current experimental foundation:
`FROZEN`

New reliability study:
`IN PROGRESS`

Final paper:
`NOT YET FROZEN`
