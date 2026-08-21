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