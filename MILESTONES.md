# Project Milestones

## M0 — Clean Restart
Status: COMPLETE

- Fresh Git repository created.
- Previous project preserved as backup.

## M1 — Reproducible Environment
Status: COMPLETE

- Python 3.10
- MuJoCo 3.2.3
- Gymnasium 1.0.0
- d3rlpy 2.8.1
- Minari 0.5.3
- imageio 2.37.4

## M2 — Dataset Verification
Status: COMPLETE

- Dataset: mujoco/hopper/medium-v0
- Episodes: 1327
- Environment: Hopper-v5

## M3 — IQL Training Pipeline
Status: COMPLETE

- IQL construction verified.
- 100-step CPU smoke training completed.

## M4 — Checkpoint Creation
Status: COMPLETE

- Complete d3rlpy checkpoint creation verified.
- Checkpoint size: approximately 1.61 MB.

## M5 — Initial Checkpoint Reload
Status: COMPLETE — FAILED TEST RECORDED

- Initial implementation used save_model() with a .d3 extension.
- Reload failed with:
  RuntimeError: Invalid magic number; corrupt file?
- Root cause identified as incorrect save/load pairing.

## M6 — Verified Persistent Checkpoint
Status: COMPLETE

- Corrected to policy.save().
- Corrected to d3rlpy.load_learnable().
- Fresh IQL reconstructed successfully.
- In-memory return: 8.730
- Reloaded return: 8.730
- Absolute difference: 0.000
- Verification: CONSISTENT
- Experiment ID: EXP-IQL-IQL-S0-20260821T145206Z

## M7 — Kaggle GPU Training
Status: PENDING

## M8 — CQL Training
Status: PENDING

## M9 — Explanation Experiments
Status: PENDING

## M10 — Distribution Shift Experiments
Status: PENDING

## M11 — Reliability Evaluation
Status: PENDING

## M12 — Final Results
Status: PENDING
## ✅ M4 — IQL 100k checkpoint independently verified on Arch

The exact Kaggle-trained checkpoint was downloaded and loaded on the
local rlxai2 environment.

Kaggle return: 5.541286887524365
Arch return:   5.541287391475207
Difference:    5.039508428339445e-07

Status: CHECKPOINT REPRODUCED LOCALLY

## ✅ M5 — Nearest-neighbor explanation baseline

Experiment:
IQL-100K-nearest-neighbor-explanation-baseline

Reference policy:
checkpoints/iql_100k/iql_mujoco_hopper_medium-v0_seed0.d3

Dataset:
mujoco/hopper/medium-v0

Episodes:
1327

Dataset observations:
999404

Observation dimension:
11

Queries:
100

Nearest-other-state distance:
- Mean: 0.14202243366439135
- Std: 0.0747119539300029
- Median: 0.12882323611384422
- Min: 0.0029248404721406787
- Max: 2.212756981226611

Status:
EXPLANATION BASELINE COMPLETE

# FINAL RESEARCH STATUS — 2026-08-27

## M7 — Five-seed IQL Gaussian replication
Status: COMPLETE

### Hopper
- 5 independently trained IQL policy seeds
- 1,000 queries per seed
- 35,000 records per seed
- 175,000 records total
- Mean seed-level slope: 0.1342720710
- 95% CI: [0.1223567020, 0.1461874400]
- Positive query-level slope fraction: 0.9970
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.0625

### HalfCheetah
- 5 independently trained IQL policy seeds
- 1,000 queries per seed
- 35,000 records per seed
- 175,000 records total
- Mean seed-level slope: 0.0651419255
- 95% CI: [0.0584082882, 0.0718755627]
- Positive query-level slope fraction: 0.9252
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.0625

## M8 — Cross-environment replication
Status: COMPLETE

- Hopper and HalfCheetah analyzed using the same five-seed replication structure.
- All 10 policy-seed slopes were positive.
- Cross-environment synthesis validated against both source analyses.
- Raw slope magnitudes are treated as descriptive rather than standardized cross-environment effect sizes.

## M9 — Structured directional robustness
Status: COMPLETE

- Environment: mujoco/hopper/medium-v0
- Policy seeds: 0, 1, 2, 3, 4
- Queries per seed: 1,000
- Records per seed: 7,000
- Gaussian-equivalent displacement levels:
  0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30
- Mean explanation dose-response slope: 0.2699650813
- 95% CI: [0.2325682855, 0.3073618770]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.0625
- Mean nearest-neighbor distance slope: 2.0280882524
- Mean policy-action-change slope: 0.4316813072

## M10 — Random-reference control
Status: COMPLETE

- Environment: mujoco/hopper/medium-v0
- Policy seeds: 0, 1, 2, 3, 4
- Queries per seed: 1,000
- Records per seed: 35,000
- Nearest-reference mean slope: 0.2768094429
- Uniform-random-reference mean slope: 0.0370033350
- Nearest-minus-random mean difference: 0.2398061079
- 95% CI: [0.2164962713, 0.2631159445]
- Exact one-sided sign-flip p: 0.03125
- Exact two-sided sign-flip p: 0.0625

## M11 — Final analysis and audit
Status: COMPLETE

- Cross-environment Gaussian synthesis validated.
- Five-seed HalfCheetah raw data validated.
- Hopper analysis re-audited against the existing result.
- Random-reference v2 validated as the canonical control analysis.
- Publication figures 4–6 validated.
- Master experiment audit passed.

## M12 — Final paper evidence package
Status: COMPLETE

- Cross-environment Gaussian results table generated.
- Structured robustness table generated.
- Master evidence summary generated.
- Publication figures generated in PDF/PNG/SVG.
- Experimental results frozen for manuscript preparation.
