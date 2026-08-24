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
