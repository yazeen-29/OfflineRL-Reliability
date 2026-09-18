#!/usr/bin/env python3

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "analysis" / "P1"

baseline = pd.read_csv(
    RESULTS / "statistics" / "no_signal_baseline.csv"
)

models = pd.read_csv(
    RESULTS / "statistics" / "heldout_fold_metrics.csv"
)

baseline = baseline.rename(
    columns={
        "mae": "baseline_mae",
        "rmse": "baseline_rmse",
    }
)

models = models[models["horizon"] == 10].copy()

merged = models.merge(
    baseline[
        [
            "test_seed",
            "baseline_mae",
            "baseline_rmse",
        ]
    ],
    on="test_seed",
    how="left",
)

merged["delta_mae"] = (
    merged["mae"] - merged["baseline_mae"]
)

merged["delta_rmse"] = (
    merged["rmse"] - merged["baseline_rmse"]
)

merged["mae_reduction_pct"] = (
    100.0
    * (merged["baseline_mae"] - merged["mae"])
    / merged["baseline_mae"]
)

merged["rmse_reduction_pct"] = (
    100.0
    * (merged["baseline_rmse"] - merged["rmse"])
    / merged["baseline_rmse"]
)

summary = (
    merged
    .groupby(
        ["model", "model_label"],
        as_index=False,
    )
    .agg(
        heldout_seeds=("test_seed", "nunique"),
        mean_delta_mae=("delta_mae", "mean"),
        sd_delta_mae=("delta_mae", "std"),
        min_delta_mae=("delta_mae", "min"),
        max_delta_mae=("delta_mae", "max"),
        seeds_improved_mae=(
            "delta_mae",
            lambda x: int((x < 0).sum()),
        ),
        mean_delta_rmse=("delta_rmse", "mean"),
        sd_delta_rmse=("delta_rmse", "std"),
        min_delta_rmse=("delta_rmse", "min"),
        max_delta_rmse=("delta_rmse", "max"),
        seeds_improved_rmse=(
            "delta_rmse",
            lambda x: int((x < 0).sum()),
        ),
        mean_mae_reduction_pct=(
            "mae_reduction_pct",
            "mean",
        ),
        mean_rmse_reduction_pct=(
            "rmse_reduction_pct",
            "mean",
        ),
    )
)

out_dir = RESULTS / "statistics"
out_dir.mkdir(parents=True, exist_ok=True)

merged.to_csv(
    out_dir / "baseline_improvement_by_seed.csv",
    index=False,
)

summary.to_csv(
    out_dir / "baseline_improvement_summary.csv",
    index=False,
)

metadata = {
    "horizon": 10,
    "comparison": "held-out model versus no-signal training-fold mean",
    "independent_unit": "policy_seed",
    "n_policy_seeds": 5,
    "interpretation": (
        "Negative delta values indicate lower prediction error "
        "than the no-signal baseline."
    ),
}

(out_dir / "baseline_improvement_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("=" * 80)
print("P1 BASELINE IMPROVEMENT AUDIT")
print("=" * 80)
print()
print(summary.to_string(index=False))
print()
print("Per-seed results:")
print(merged.to_string(index=False))
