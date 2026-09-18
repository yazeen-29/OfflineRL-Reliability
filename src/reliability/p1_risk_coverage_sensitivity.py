#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "analysis" / "P1"

SEEDS = [0, 1, 2, 3, 4]
PRIMARY_HORIZON = 10


def compute_curve(
    predictions: pd.DataFrame,
    exclude_sigma_zero: bool,
) -> pd.DataFrame:

    data = predictions[
        predictions["horizon"] == PRIMARY_HORIZON
    ].copy()

    if exclude_sigma_zero:
        data = data[data["sigma"] > 0.0].copy()

    rows = []

    for model in sorted(data["model"].unique()):

        model_data = data[
            data["model"] == model
        ]

        for seed in SEEDS:

            fold = model_data[
                model_data["test_seed"] == seed
            ].copy()

            fold = fold.sort_values(
                "prediction",
                kind="mergesort",
            )

            n = len(fold)

            for coverage_percent in range(10, 101, 10):

                coverage_target = coverage_percent / 100.0

                # Use integer percentage arithmetic so that
                # 30% of 600 is exactly 180 rather than being
                # interpreted as 180.00000000000003.
                keep_n = max(
                    1,
                    int(np.ceil(
                        coverage_percent * n / 100
                    )),
                )

                retained = fold.iloc[:keep_n]

                rows.append(
                    {
                        "model": model,
                        "model_label": retained[
                            "model_label"
                        ].iloc[0],
                        "test_seed": seed,
                        "coverage_target": float(
                            coverage_target
                        ),
                        "actual_coverage": float(
                            keep_n / n
                        ),
                        "retained_records": int(
                            keep_n
                        ),
                        "mean_observed_C10": float(
                            retained[
                                "absolute_consequence"
                            ].mean()
                        ),
                    }
                )

    result = pd.DataFrame(rows)

    return (
        result
        .groupby(
            [
                "model",
                "model_label",
                "coverage_target",
            ],
            as_index=False,
        )
        .agg(
            n_heldout_seeds=(
                "test_seed",
                "nunique",
            ),
            mean_actual_coverage=(
                "actual_coverage",
                "mean",
            ),
            mean_observed_C10=(
                "mean_observed_C10",
                "mean",
            ),
            sd_observed_C10=(
                "mean_observed_C10",
                "std",
            ),
        )
    )


def main():

    predictions = pd.read_csv(
        RESULTS
        / "models"
        / "heldout_predictions.csv"
    )

    all_shift = compute_curve(
        predictions,
        exclude_sigma_zero=False,
    )

    nonzero_shift = compute_curve(
        predictions,
        exclude_sigma_zero=True,
    )

    out_dir = (
        RESULTS
        / "statistics"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_shift.to_csv(
        out_dir
        / "risk_coverage_all_shifts_sensitivity.csv",
        index=False,
    )

    nonzero_shift.to_csv(
        out_dir
        / "risk_coverage_nonzero_shift.csv",
        index=False,
    )

    print("=" * 80)
    print("P1 RISK-COVERAGE SENSITIVITY")
    print("=" * 80)

    print(
        "\nAll shifts:"
    )

    print(
        all_shift[
            [
                "model",
                "coverage_target",
                "mean_observed_C10",
            ]
        ].to_string(index=False)
    )

    print(
        "\nNonzero shifts only (sigma > 0):"
    )

    print(
        nonzero_shift[
            [
                "model",
                "coverage_target",
                "mean_observed_C10",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
