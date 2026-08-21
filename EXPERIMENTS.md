# Experiment Registry

| ID | Experiment | Status | Key Result |
|---|---|---|---|
| EXP-001 | Environment verification | PASS | Dependencies functional |
| EXP-002 | Minari Hopper loading | PASS | 1327 episodes |
| EXP-003 | IQL 100-step training | PASS | Training completed |
| EXP-004 | Initial checkpoint save | PASS | 1.61 MB checkpoint |
| EXP-005 | Initial checkpoint reload | FAIL | Invalid magic number |
| EXP-006 | Corrected checkpoint save/reload | PASS | In-memory/reloaded return difference = 0.000 |

## EXP-006 Details

- Algorithm: IQL
- Task: mujoco/hopper/medium-v0
- Seed: 0
- Training steps: 100
- Device: CPU
- Dataset episodes: 1327
- Checkpoint: complete d3rlpy .d3 format
- In-memory return: 8.730
- Reloaded return: 8.730
- Absolute difference: 0.000
- Verification: CONSISTENT
- Experiment metadata:
  results/verification/EXP-IQL-IQL-S0-20260821T145206Z_metadata.json
- Experiment results:
  results/verification/EXP-IQL-IQL-S0-20260821T145206Z_results.json
- Experiment log:
  logs/verification/EXP-IQL-IQL-S0-20260821T145206Z.log