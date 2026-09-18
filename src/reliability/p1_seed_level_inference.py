#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "analysis" / "P1"

SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10


def exact_sign_flip_p(values: np.ndarray, alternative: str) -> float:
    """
    Exact policy-seed sign-flip test.

    H0: mean difference is zero and signs are exchangeable.
    Negative values mean the model improves over baseline.

    alternative='less':
        H1: mean difference < 0

    alternative='two-sided':
        H1: mean difference != 0
    """
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())

    null_means = []

    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        null_means.append(
            float(
                np.mean(values * np.asarray(signs))
            )
        )

    null_means = np.asarray(null_means)

    if alternative == "less":
        return float(
            np.mean(null_means <= observed + 1e-15)
        )

    if alternative == "two-sided":
        return float(
            np.mean(
                np.abs(null_means)
                >= abs(observed) - 1e-15
            )
        )

    raise ValueError(f"Unknown alternative: {alternative}")


def summarize_metric(
    differences: np.ndarray,
    baseline: np.ndarray,
    model: np.ndarray,
) -> dict:
    differences = np.asarray(differences, dtype=float)

    n = len(differences)
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))

    t_critical = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    standard_error = (
        sd_diff / np.sqrt(n)
    )

    ci_low = (
        mean_diff
        - t_critical * standard_error
    )

    ci_high = (
        mean_diff
        + t_critical * standard_error
    )

    improvement_count = int(
        np.sum(differences < 0)
    )

    baseline_mean = float(
        np.mean(baseline)
    )

    model_mean = float(
        np.mean(model)
    )

    aggregate_reduction_pct = (
        100.0
        * (baseline_mean - model_mean)
        / baseline_mean
    )

    return {
        "n_policy_seeds": n,
        "mean_difference": mean_diff,
        "sd_difference": sd_diff,
        "t_based_95ci_low": float(ci_low),
        "t_based_95ci_high": float(ci_high),
        "exact_one_sided_signflip_p": (
            exact_sign_flip_p(
                differences,
                "less",
            )
        ),
        "exact_two_sided_signflip_p": (
            exact_sign_flip_p(
                differences,
                "two-sided",
            )
        ),
        "seeds_improved": improvement_count,
        "seeds_total": n,
        "baseline_mean": baseline_mean,
        "model_mean": model_mean,
        "aggregate_reduction_percent": (
            float(aggregate_reduction_pct)
        ),
    }


def main() -> None:
    baseline_path = (
        RESULTS
        / "statistics"
        / "no_signal_baseline.csv"
    )

    model_path = (
        RESULTS
        / "statistics"
        / "heldout_fold_metrics.csv"
    )

    baseline = pd.read_csv(
        baseline_path
    )

    models = pd.read_csv(
        model_path
    )

    baseline = baseline[
        baseline["horizon"] == HORIZON
    ].copy()

    models = models[
        models["horizon"] == HORIZON
    ].copy()

    rows = []
    per_seed_rows = []

    for model_name in sorted(
        models["model"].unique()
    ):
        model_data = models[
            models["model"] == model_name
        ].copy()

        merged = model_data.merge(
            baseline[
                [
                    "test_seed",
                    "mae",
                    "rmse",
                ]
            ].rename(
                columns={
                    "mae": "baseline_mae",
                    "rmse": "baseline_rmse",
                }
            ),
            on="test_seed",
            how="inner",
            validate="one_to_one",
        )

        merged = merged.sort_values(
            "test_seed"
        )

        if merged["test_seed"].tolist() != SEEDS:
            raise ValueError(
                f"{model_name}: missing held-out seed"
            )

        delta_mae = (
            merged["mae"].to_numpy()
            - merged["baseline_mae"].to_numpy()
        )

        delta_rmse = (
            merged["rmse"].to_numpy()
            - merged["baseline_rmse"].to_numpy()
        )

        mae_stats = summarize_metric(
            delta_mae,
            merged["baseline_mae"].to_numpy(),
            merged["mae"].to_numpy(),
        )

        rmse_stats = summarize_metric(
            delta_rmse,
            merged["baseline_rmse"].to_numpy(),
            merged["rmse"].to_numpy(),
        )

        rows.append(
            {
                "model": model_name,
                "model_label": merged[
                    "model_label"
                ].iloc[0],

                "mae_mean_difference": (
                    mae_stats["mean_difference"]
                ),
                "mae_sd_difference": (
                    mae_stats["sd_difference"]
                ),
                "mae_95ci_low": (
                    mae_stats["t_based_95ci_low"]
                ),
                "mae_95ci_high": (
                    mae_stats["t_based_95ci_high"]
                ),
                "mae_exact_one_sided_p": (
                    mae_stats[
                        "exact_one_sided_signflip_p"
                    ]
                ),
                "mae_exact_two_sided_p": (
                    mae_stats[
                        "exact_two_sided_signflip_p"
                    ]
                ),
                "mae_seeds_improved": (
                    mae_stats["seeds_improved"]
                ),
                "mae_aggregate_reduction_percent": (
                    mae_stats[
                        "aggregate_reduction_percent"
                    ]
                ),

                "rmse_mean_difference": (
                    rmse_stats["mean_difference"]
                ),
                "rmse_sd_difference": (
                    rmse_stats["sd_difference"]
                ),
                "rmse_95ci_low": (
                    rmse_stats["t_based_95ci_low"]
                ),
                "rmse_95ci_high": (
                    rmse_stats["t_based_95ci_high"]
                ),
                "rmse_exact_one_sided_p": (
                    rmse_stats[
                        "exact_one_sided_signflip_p"
                    ]
                ),
                "rmse_exact_two_sided_p": (
                    rmse_stats[
                        "exact_two_sided_signflip_p"
                    ]
                ),
                "rmse_seeds_improved": (
                    rmse_stats["seeds_improved"]
                ),
                "rmse_aggregate_reduction_percent": (
                    rmse_stats[
                        "aggregate_reduction_percent"
                    ]
                ),
            }
        )

        for _, row in merged.iterrows():
            per_seed_rows.append(
                {
                    "model": model_name,
                    "model_label": row[
                        "model_label"
                    ],
                    "test_seed": int(
                        row["test_seed"]
                    ),
                    "baseline_mae": float(
                        row["baseline_mae"]
                    ),
                    "model_mae": float(
                        row["mae"]
                    ),
                    "delta_mae": float(
                        row["mae"]
                        - row["baseline_mae"]
                    ),
                    "baseline_rmse": float(
                        row["baseline_rmse"]
                    ),
                    "model_rmse": float(
                        row["rmse"]
                    ),
                    "delta_rmse": float(
                        row["rmse"]
                        - row["baseline_rmse"]
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    per_seed = pd.DataFrame(
        per_seed_rows
    )

    out_dir = (
        RESULTS
        / "statistics"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        out_dir
        / "seed_level_inference.csv",
        index=False,
    )

    per_seed.to_csv(
        out_dir
        / "seed_level_inference_by_seed.csv",
        index=False,
    )

    metadata = {
        "analysis": (
            "Exact paired seed-level inference "
            "for held-out model vs no-signal baseline"
        ),
        "horizon": HORIZON,
        "independent_unit": "policy_seed",
        "policy_seeds": SEEDS,
        "test": (
            "exact sign-flip across policy seeds"
        ),
        "direction": (
            "negative difference = model has lower error "
            "than baseline"
        ),
        "confidence_interval": (
            "two-sided t-based 95% CI over five policy seeds; "
            "descriptive uncertainty interval"
        ),
        "model_selection": (
            "no selection or ranking rule imposed"
        ),
    }

    (
        out_dir
        / "seed_level_inference_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("=" * 80)
    print("P1 SEED-LEVEL INFERENCE")
    print("=" * 80)
    print()
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
