# Kaggle P1 Run Guide

This is the execution guide for the Option-B P1 study. It uses only the
existing IQL checkpoints for policy seeds 0--4. It does not train IQL seeds
5--9 and it does not modify the frozen protocol.

## 1. Prepare the Kaggle notebook

Use a GPU notebook, attach the dataset/input containing this repository and
the five checkpoint files, and enable internet only for package/dataset
installation if required. In the first cell:

```bash
%cd /kaggle/working
!git clone https://github.com/yazeen-29/OfflineRL-Reliability.git
%cd /kaggle/working/OfflineRL-Reliability
!pip install -q "d3rlpy==2.8.1" "gymnasium[mujoco]" minari scikit-learn
```

If the repository or checkpoints are supplied as a Kaggle Dataset rather than
cloned, copy them into `/kaggle/working/OfflineRL-Reliability` and keep the
relative paths unchanged. The collector expects the checkpoint paths listed in
`results/reliability/P1_decision_consequence/metadata/manifest.json`.

## 2. Verify the environment and checkpoints

```bash
%cd /kaggle/working/OfflineRL-Reliability
!python - <<'PY'
import d3rlpy
from pathlib import Path

paths = [
    Path("checkpoints/iql_100k/iql_mujoco_hopper_medium-v0_seed0.d3"),
    Path("checkpoints/iql_replications/iql_seed1/iql_mujoco_hopper_medium-v0_seed1.d3"),
    Path("checkpoints/iql_replications/iql_seed2/iql_mujoco_hopper_medium-v0_seed2.d3"),
    Path("checkpoints/iql_replications/iql_seed3/iql_mujoco_hopper_medium-v0_seed3.d3"),
    Path("checkpoints/iql_replications/iql_seed4/iql_mujoco_hopper_medium-v0_seed4.d3"),
]
for path in paths:
    assert path.exists(), path
    policy = d3rlpy.load_learnable(str(path), device="cpu")
    print("OK", path, type(policy).__name__)
PY
```

Do not continue if any checkpoint is missing or fails to load.

## 3. Run the engineering gate

This uses 4 clean episodes x 5 states = 20 states, all seven sigma levels,
and horizons 1, 5, 10, and 20. It produces 560 records.

```bash
!python src/reliability/p1_consequence_full.py \
  --policy-seed 0 \
  --n-episodes 4 \
  --states-per-episode 5 \
  --status engineering_gate
!python src/reliability/validate_p1_output.py \
  results/reliability/P1_decision_consequence/raw/seed0.json
```

The structural audit must pass before any other seed is run. Download or
commit `seed0.json` as diagnostic evidence and retain the Kaggle cell output.

## 4. Run final P1 collection

After the gate passes, run each seed separately with the publication-scale
sampling rule: 20 episodes x 5 states = 100 states per policy seed.

```bash
for seed in 0 1 2 3 4; do
  python src/reliability/p1_consequence_full.py \
    --policy-seed "$seed" \
    --n-episodes 20 \
    --states-per-episode 5 \
    --status final_collection
  python src/reliability/validate_p1_output.py \
    "results/reliability/P1_decision_consequence/raw/seed${seed}.json"
done
```

Each final file should contain 2,800 records: 100 states x 7 sigma levels x
4 horizons. If a seed fails, preserve the partial log and rerun only that
seed after diagnosing the failure.

## 5. Package results

Download these files from Kaggle without changing their contents:

```text
results/reliability/P1_decision_consequence/raw/seed0.json
results/reliability/P1_decision_consequence/raw/seed1.json
results/reliability/P1_decision_consequence/raw/seed2.json
results/reliability/P1_decision_consequence/raw/seed3.json
results/reliability/P1_decision_consequence/raw/seed4.json
```

Do not run P2, CQL, Walker2d, or alter the paper claims until the five raw
files pass the repository-side integrity audit and clustered analysis.
