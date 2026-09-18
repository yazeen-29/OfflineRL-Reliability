#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data_frozen" / "P1"
OUT_DIR = ROOT / "results" / "analysis" / "P1"

SEEDS = [0, 1, 2, 3, 4]
SIGMAS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]
HORIZONS = [1, 5, 10, 20]
PRIMARY_HORIZON = 10

EXPECTED_STATES_PER_SEED = 100
EXPECTED_RECORDS_PER_SEED = 2800

SCALAR_REQUIRED = [
    "policy_seed",
    "state_id",
    "source_episode_id",
    "source_step",
    "sigma",
    "horizon",
    "action_disagreement",
    "support_distance",
    "nearest_reference_index",
    "q1_clean",
    "q2_clean",
    "critic_disagreement",
    "twin_critic_disagreement",
    "clean_return",
    "shifted_return",
    "delta_J",
    "absolute_consequence",
    "relative_consequence",
    "clean_steps",
    "shifted_steps",
]

ARRAY_REQUIRED = [
    "standardized_noise",
    "clean_action",
    "shifted_action",
]

# Publication-scale candidates from Amendment 005.
MODEL_SPECS = {
    "A_action_only": [
        "action_disagreement",
    ],
    "B_action_plus_support": [
        "action_disagreement",
        "support_distance",
    ],
    "C_action_plus_critic": [
        "action_disagreement",
        "twin_critic_disagreement",
    ],
    "D_action_plus_support_plus_critic": [
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
    ],
    "E_action_plus_support_interaction": [
        "action_disagreement",
        "support_distance",
        "action_support_interaction",
    ],
}

MODEL_LABELS = {
    "A_action_only": "Action disagreement only",
    "B_action_plus_support": "Action disagreement + support distance",
    "C_action_plus_critic": "Action disagreement + twin-critic disagreement",
    "D_action_plus_support_plus_critic": (
        "Action disagreement + support distance + twin-critic disagreement"
    ),
    "E_action_plus_support_interaction": (
        "Action disagreement + support distance + "
        "action-disagreement × support-distance"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    manifest_path = DATA_DIR / "FREEZE_MANIFEST.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    if manifest["policy_seeds"] != SEEDS:
        raise ValueError(
            f"Manifest seeds {manifest['policy_seeds']} != expected {SEEDS}"
        )

    if manifest["states_per_seed"] != EXPECTED_STATES_PER_SEED:
        raise ValueError("Unexpected states_per_seed")

    if manifest["total_records"] != 14000:
        raise ValueError("Unexpected total record count")

    if manifest["primary_horizon"] != PRIMARY_HORIZON:
        raise ValueError("Unexpected primary horizon")

    frames = []
    audit = []

    for seed in SEEDS:
        name = f"seed{seed}.json"
        path = DATA_DIR / name

        if not path.exists():
            raise FileNotFoundError(path)

        meta = manifest["files"][name]

        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)

        if actual_size != meta["size_bytes"]:
            raise ValueError(f"{name}: size mismatch")

        if actual_sha != meta["sha256"]:
            raise ValueError(f"{name}: SHA mismatch")

        data = json.loads(path.read_text())

        if data["policy_seed"] != seed:
            raise ValueError(f"{name}: policy_seed mismatch")

        records = data["records"]

        if len(records) != EXPECTED_RECORDS_PER_SEED:
            raise ValueError(
                f"{name}: expected {EXPECTED_RECORDS_PER_SEED} records, "
                f"got {len(records)}"
            )

        if data["n_states"] != EXPECTED_STATES_PER_SEED:
            raise ValueError(f"{name}: unexpected n_states")

        local_rows = []

        for rec in records:
            missing = [
                key for key in SCALAR_REQUIRED + ARRAY_REQUIRED
                if key not in rec
            ]

            if missing:
                raise ValueError(
                    f"{name}: missing fields {missing}"
                )

            for key in ARRAY_REQUIRED:
                arr = np.asarray(rec[key], dtype=float)

                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"{name}: non-finite {key}"
                    )

            numeric_values = [
                rec[key] for key in SCALAR_REQUIRED
                if key not in {
                    "policy_seed",
                    "state_id",
                    "source_episode_id",
                    "source_step",
                    "horizon",
                    "nearest_reference_index",
                    "clean_steps",
                    "shifted_steps",
                }
            ]

            if not all(
                np.isfinite(float(value))
                for value in numeric_values
            ):
                raise ValueError(
                    f"{name}: non-finite scalar value"
                )

            if len(rec["standardized_noise"]) != 11:
                raise ValueError(
                    f"{name}: standardized_noise must have length 11"
                )

            # Internal consistency check.
            if not np.isclose(
                abs(float(rec["delta_J"])),
                float(rec["absolute_consequence"]),
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"{name}: absolute_consequence inconsistent with delta_J"
                )

            row = {
                key: rec[key]
                for key in SCALAR_REQUIRED
            }

            row["clean_terminated"] = bool(rec["clean_terminated"])
            row["shifted_terminated"] = bool(rec["shifted_terminated"])
            row["clean_truncated"] = bool(rec["clean_truncated"])
            row["shifted_truncated"] = bool(rec["shifted_truncated"])

            row["checkpoint"] = rec["checkpoint"]
            row["git_commit"] = rec["git_commit"]
            row["state_sampling_seed"] = rec["state_sampling_seed"]
            row["noise_seed"] = rec["noise_seed"]

            local_rows.append(row)

        frame = pd.DataFrame(local_rows)

        expected_sigma = set(round(x, 10) for x in SIGMAS)
        observed_sigma = set(
            round(float(x), 10)
            for x in frame["sigma"].unique()
        )

        if expected_sigma != observed_sigma:
            raise ValueError(
                f"{name}: sigma coverage mismatch"
            )

        if set(frame["horizon"].unique()) != set(HORIZONS):
            raise ValueError(
                f"{name}: horizon coverage mismatch"
            )

        cell_counts = (
            frame.groupby(
                ["state_id", "sigma", "horizon"],
                dropna=False,
            )
            .size()
        )

        if len(cell_counts) != EXPECTED_RECORDS_PER_SEED:
            raise ValueError(
                f"{name}: duplicate or missing state/sigma/horizon cells"
            )

        if not np.all(cell_counts.values == 1):
            raise ValueError(
                f"{name}: non-unique state/sigma/horizon cells"
            )

        duplicate_records = frame.duplicated(
            subset=["policy_seed", "state_id", "sigma", "horizon"]
        ).sum()

        if duplicate_records:
            raise ValueError(
                f"{name}: duplicate primary cells={duplicate_records}"
            )

        frames.append(frame)

        audit.append(
            {
                "seed": seed,
                "records": len(frame),
                "states": frame["state_id"].nunique(),
                "sha256": actual_sha,
                "size_bytes": actual_size,
                "duplicate_cells": int(duplicate_records),
                "invalid_records_in_frozen_file": 0,
            }
        )

    all_data = pd.concat(frames, ignore_index=True)

    if len(all_data) != 14000:
        raise ValueError(
            f"Expected 14000 records, got {len(all_data)}"
        )

    expected_keys = {
        (seed, state, sigma, horizon)
        for seed in SEEDS
        for state in range(EXPECTED_STATES_PER_SEED)
        for sigma in SIGMAS
        for horizon in HORIZONS
    }

    observed_keys = {
        (
            int(row.policy_seed),
            int(row.state_id),
            round(float(row.sigma), 10),
            int(row.horizon),
        )
        for row in all_data.itertuples()
    }

    if observed_keys != expected_keys:
        raise ValueError("Global frozen-cell coverage mismatch")

    audit_df = pd.DataFrame(audit)

    return all_data, {
        "manifest": manifest,
        "seed_audit": audit_df,
    }


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["action_support_interaction"] = (
        out["action_disagreement"]
        * out["support_distance"]
    )
    return out


def design_matrix(
    frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    features = MODEL_SPECS[model_name]
    return frame[features].astype(float)


def fit_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
) -> tuple[np.ndarray, LinearRegression]:
    x_train = design_matrix(train, model_name)
    y_train = train["absolute_consequence"].astype(float).to_numpy()

    x_test = design_matrix(test, model_name)

    model = LinearRegression()
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    return prediction, model


def primary_and_sensitivity_metrics(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    prediction_rows = []

    for horizon in HORIZONS:
        horizon_data = data[data["horizon"] == horizon].copy()

        for test_seed in SEEDS:
            train = horizon_data[
                horizon_data["policy_seed"] != test_seed
            ].copy()

            test = horizon_data[
                horizon_data["policy_seed"] == test_seed
            ].copy()

            for model_name in MODEL_SPECS:
                prediction, _ = fit_one_fold(
                    train=train,
                    test=test,
                    model_name=model_name,
                )

                y_true = test["absolute_consequence"].to_numpy(
                    dtype=float
                )

                mae = mean_absolute_error(
                    y_true,
                    prediction,
                )

                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_true,
                            prediction,
                        )
                    )
                )

                fold_rows.append(
                    {
                        "horizon": horizon,
                        "primary_horizon": horizon == PRIMARY_HORIZON,
                        "test_seed": test_seed,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "n_train_records": len(train),
                        "n_test_records": len(test),
                        "mae": float(mae),
                        "rmse": rmse,
                    }
                )

                temp = test[
                    [
                        "policy_seed",
                        "state_id",
                        "sigma",
                        "horizon",
                        "absolute_consequence",
                        "action_disagreement",
                        "support_distance",
                        "twin_critic_disagreement",
                    ]
                ].copy()

                temp["model"] = model_name
                temp["model_label"] = MODEL_LABELS[model_name]
                temp["prediction"] = prediction
                temp["prediction_error"] = (
                    temp["prediction"]
                    - temp["absolute_consequence"]
                )
                temp["test_seed"] = test_seed

                prediction_rows.append(temp)

    return (
        pd.DataFrame(fold_rows),
        pd.concat(prediction_rows, ignore_index=True),
    )


def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        fold_metrics
        .groupby(
            ["horizon", "primary_horizon", "model", "model_label"],
            as_index=False,
        )
        .agg(
            n_heldout_seeds=("test_seed", "nunique"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
        )
    )


def make_descriptive_summaries(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_sigma = (
        data
        .groupby(
            ["policy_seed", "horizon", "sigma"],
            as_index=False,
        )
        .agg(
            n_records=("state_id", "size"),
            n_states=("state_id", "nunique"),
            mean_C=("absolute_consequence", "mean"),
            median_C=("absolute_consequence", "median"),
            sd_C=("absolute_consequence", "std"),
            mean_delta_J=("delta_J", "mean"),
            mean_action_disagreement=(
                "action_disagreement",
                "mean",
            ),
            mean_support_distance=(
                "support_distance",
                "mean",
            ),
            mean_twin_critic_disagreement=(
                "twin_critic_disagreement",
                "mean",
            ),
            clean_termination_rate=(
                "clean_terminated",
                "mean",
            ),
            shifted_termination_rate=(
                "shifted_terminated",
                "mean",
            ),
            clean_truncation_rate=(
                "clean_truncated",
                "mean",
            ),
            shifted_truncation_rate=(
                "shifted_truncated",
                "mean",
            ),
        )
    )

    seed_level = (
        data[data["horizon"] == PRIMARY_HORIZON]
        .groupby("policy_seed", as_index=False)
        .agg(
            n_records=("state_id", "size"),
            n_states=("state_id", "nunique"),
            mean_C10=("absolute_consequence", "mean"),
            median_C10=("absolute_consequence", "median"),
            sd_C10=("absolute_consequence", "std"),
            mean_delta_J10=("delta_J", "mean"),
            mean_action_disagreement=(
                "action_disagreement",
                "mean",
            ),
            mean_support_distance=(
                "support_distance",
                "mean",
            ),
            mean_twin_critic_disagreement=(
                "twin_critic_disagreement",
                "mean",
            ),
            shifted_termination_rate=(
                "shifted_terminated",
                "mean",
            ),
        )
    )

    return by_sigma, seed_level


def make_risk_coverage(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    coverage_grid = np.arange(
        0.1,
        1.0001,
        0.1,
    )

    primary = predictions[
        predictions["horizon"] == PRIMARY_HORIZON
    ].copy()

    for model_name in MODEL_SPECS:
        model_predictions = primary[
            primary["model"] == model_name
        ]

        for test_seed in SEEDS:
            fold = model_predictions[
                model_predictions["test_seed"] == test_seed
            ].copy()

            fold = fold.sort_values(
                "prediction",
                kind="mergesort",
            )

            n = len(fold)

            for coverage in coverage_grid:
                coverage_percent = int(round(coverage * 100))

                # Integer percentage arithmetic avoids floating-point
                # ceil() errors such as 0.3 * 700 -> 211.
                keep_n = max(
                    1,
                    int(np.ceil(
                        coverage_percent * n / 100
                    )),
                )

                retained = fold.iloc[:keep_n]

                rows.append(
                    {
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "test_seed": test_seed,
                        "coverage": float(keep_n / n),
                        "retained_records": keep_n,
                        "observed_mean_C10": float(
                            retained["absolute_consequence"].mean()
                        ),
                        "mean_predicted_C10": float(
                            retained["prediction"].mean()
                        ),
                    }
                )

    rc = pd.DataFrame(rows)

    return (
        rc
        .groupby(
            ["model", "model_label", "coverage"],
            as_index=False,
        )
        .agg(
            n_heldout_seeds=("test_seed", "nunique"),
            mean_observed_risk=("observed_mean_C10", "mean"),
            sd_observed_risk=("observed_mean_C10", "std"),
        )
    )


def write_metadata(
    audit_payload: dict,
    fold_metrics: pd.DataFrame,
) -> None:
    metadata = {
        "analysis": "P1 final held-out policy-seed predictive analysis",
        "analysis_git_commit": git_commit(),
        "data_directory": str(DATA_DIR),
        "policy_seeds": SEEDS,
        "primary_horizon": PRIMARY_HORIZON,
        "all_horizons": HORIZONS,
        "sigma_levels": SIGMAS,
        "primary_outcome": "absolute_consequence = |delta_J|",
        "primary_predictive_metrics": [
            "MAE",
            "RMSE",
        ],
        "binary_metrics": {
            "status": "not_computed",
            "reason": (
                "The frozen protocol does not define a binary "
                "adverse-consequence threshold."
            ),
            "metrics_not_computed": [
                "AUROC",
                "AUPRC",
                "calibration_error",
                "Brier_score",
            ],
        },
        "risk_coverage": {
            "status": "computed",
            "risk_definition": (
                "mean observed C10 among retained low-predicted-risk "
                "records"
            ),
            "coverage_grid": [
                round(float(x), 2)
                for x in np.arange(0.1, 1.0001, 0.1)
            ],
        },
        "model_selection": {
            "status": "descriptive_comparison_only",
            "selection_rule": (
                "No post-hoc winner rule is imposed because the "
                "protocol specifies metrics and held-out evaluation "
                "but does not specify a numerical tie-break rule."
            ),
        },
        "seed_audit": audit_payload["seed_audit"].to_dict(
            orient="records"
        ),
        "fold_count": int(len(fold_metrics)),
        "attempt_level_exclusions": (
            "Unavailable from frozen JSON because failed attempts "
            "are not encoded as records."
        ),
    }

    (OUT_DIR / "statistics" / "analysis_run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def write_binary_status() -> None:
    payload = {
        "status": "not_computed",
        "reason": (
            "No protocol-defined binary adverse-consequence "
            "threshold exists in the frozen P1 specification."
        ),
        "required_before_binary_analysis": [
            "pre-specify threshold",
            "justify threshold independently of observed predictive metrics",
        ],
        "metrics_deferred": [
            "AUROC",
            "AUPRC",
            "calibration_error",
            "Brier_score",
        ],
    }

    (
        OUT_DIR / "statistics" / "binary_metrics_status.json"
    ).write_text(
        json.dumps(payload, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data, audit_payload = load_and_validate()

    data = add_model_features(data)

    by_sigma, seed_level = make_descriptive_summaries(data)

    by_sigma.to_csv(
        OUT_DIR / "summaries" / "sigma_horizon_summary.csv",
        index=False,
    )

    seed_level.to_csv(
        OUT_DIR / "summaries" / "seed_level_summary.csv",
        index=False,
    )

    audit_payload["seed_audit"].to_csv(
        OUT_DIR / "summaries" / "freeze_audit.csv",
        index=False,
    )

    fold_metrics, predictions = (
        primary_and_sensitivity_metrics(data)
    )

    fold_metrics.to_csv(
        OUT_DIR / "statistics" / "heldout_fold_metrics.csv",
        index=False,
    )

    summary_metrics = summarize_fold_metrics(
        fold_metrics
    )

    summary_metrics.to_csv(
        OUT_DIR / "tables" / "heldout_model_comparison.csv",
        index=False,
    )

    predictions.to_csv(
        OUT_DIR / "models" / "heldout_predictions.csv",
        index=False,
    )

    risk_coverage = make_risk_coverage(
        predictions
    )

    risk_coverage.to_csv(
        OUT_DIR / "tables" / "risk_coverage.csv",
        index=False,
    )

    write_metadata(
        audit_payload,
        fold_metrics,
    )

    write_binary_status()

    print("=" * 80)
    print("P1 FINAL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Data records : {len(data)}")
    print(f"Seeds        : {SEEDS}")
    print(f"Primary H    : {PRIMARY_HORIZON}")
    print()
    print("Held-out model comparison:")
    print(
        summary_metrics[
            summary_metrics["primary_horizon"]
        ][
            [
                "model",
                "mean_mae",
                "sd_mae",
                "mean_rmse",
                "sd_rmse",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Binary AUROC/AUPRC/Brier/calibration: "
        "NOT COMPUTED — no frozen binary threshold."
    )
    print()
    print(
        f"Outputs written to: {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
