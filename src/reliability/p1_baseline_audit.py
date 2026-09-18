#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data_frozen" / "P1"
RESULTS_DIR = ROOT / "results" / "analysis" / "P1"

SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10


def load_data():
    frames = []

    for seed in SEEDS:
        with open(DATA_DIR / f"seed{seed}.json") as f:
            payload = json.load(f)

        rows = []

        for r in payload["records"]:
            if int(r["horizon"]) != HORIZON:
                continue

            rows.append(
                {
                    "policy_seed": int(r["policy_seed"]),
                    "state_id": int(r["state_id"]),
                    "sigma": float(r["sigma"]),
                    "absolute_consequence": float(
                        r["absolute_consequence"]
                    ),
                }
            )

        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)


def main():
    data = load_data()

    rows = []

    for test_seed in SEEDS:
        train = data[data["policy_seed"] != test_seed]
        test = data[data["policy_seed"] == test_seed]

        baseline_value = float(
            train["absolute_consequence"].mean()
        )

        prediction = np.full(
            len(test),
            baseline_value,
            dtype=float,
        )

        y = test["absolute_consequence"].to_numpy(
            dtype=float
        )

        mae = mean_absolute_error(y, prediction)
        rmse = np.sqrt(
            mean_squared_error(y, prediction)
        )

        rows.append(
            {
                "horizon": HORIZON,
                "test_seed": test_seed,
                "n_train_records": len(train),
                "n_test_records": len(test),
                "training_mean_C10": baseline_value,
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )

    result = pd.DataFrame(rows)

    out = (
        result
        .agg(
            {
                "mae": ["mean", "std"],
                "rmse": ["mean", "std"],
            }
        )
    )

    out_dir = RESULTS_DIR / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        out_dir / "no_signal_baseline.csv",
        index=False,
    )

    with open(
        out_dir / "no_signal_baseline_summary.json",
        "w",
    ) as f:
        json.dump(
            {
                "horizon": HORIZON,
                "independent_unit": "policy_seed",
                "baseline": (
                    "training-fold mean C10"
                ),
                "n_heldout_seeds": len(SEEDS),
                "mean_mae": float(result["mae"].mean()),
                "sd_mae": float(result["mae"].std()),
                "mean_rmse": float(result["rmse"].mean()),
                "sd_rmse": float(result["rmse"].std()),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("P1 NO-SIGNAL BASELINE")
    print("=" * 80)
    print(result.to_string(index=False))
    print()
    print(
        f"Mean MAE  : {result['mae'].mean():.6f}"
    )
    print(
        f"Mean RMSE : {result['rmse'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
