from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# P9 configuration
#
# P9 inherits the frozen P2 estimator and operational evaluation
# exactly. Only the frozen source cell changes.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "results" / "analysis" / "P9"

PRED_DIR = OUT_DIR / "predictions"
STATS_DIR = OUT_DIR / "statistics"
RISK_DIR = OUT_DIR / "risk_coverage"
DECILE_DIR = OUT_DIR / "reliability_deciles"
PROV_DIR = OUT_DIR / "provenance"

POLICY_SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10
RIDGE_ALPHA = 1.0

COVERAGE_PERCENT = [10, 20, 30, 50, 70, 90, 100]
N_DECILES = 10

MODEL_SPECS = {
    "primary_3feature": [
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
    ],
    "action_only_sensitivity": [
        "action_disagreement",
    ],
}


# ============================================================
# Frozen source cells
# ============================================================

CELL_SPECS = {
    "IQL_Hopper": (
        ROOT / "data_frozen" / "P1"
    ),
    "IQL_HalfCheetah": (
        ROOT / "results" / "reliability"
        / "IQL_HalfCheetah" / "raw"
    ),
    "IQL_Walker2d": (
        ROOT / "results" / "reliability"
        / "IQL_Walker2d" / "raw"
    ),
    "CQL_Hopper": (
        ROOT / "results" / "reliability"
        / "P5_CQL" / "raw"
    ),
    "CQL_HalfCheetah": (
        ROOT / "results" / "reliability"
        / "P7_CQL_HalfCheetah" / "raw"
    ),
    "CQL_Walker2d": (
        ROOT / "results" / "reliability"
        / "P6_CQL_Walker2d" / "raw"
    ),
}


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def git_command(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


# ============================================================
# P2-identical model / scoring helpers
# ============================================================

def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def empirical_reliability_score(
    train_predictions: np.ndarray,
    heldout_predictions: np.ndarray,
) -> np.ndarray:
    """
    Exact P2 definition:

        R(x) = 1 - F_train(yhat(x))

    F_train is the empirical CDF of training-fold predictions,
    using right-continuous <= semantics.

    Larger R => lower predicted consequence => higher reliability.
    """

    train_sorted = np.sort(
        np.asarray(
            train_predictions,
            dtype=float,
        )
    )

    ranks = np.searchsorted(
        train_sorted,
        np.asarray(
            heldout_predictions,
            dtype=float,
        ),
        side="right",
    )

    cdf = ranks / len(train_sorted)

    reliability = 1.0 - cdf

    return np.clip(
        reliability,
        0.0,
        1.0,
    )


def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    P2-compatible Spearman calculation.
    """

    x = pd.Series(
        np.asarray(
            x,
            dtype=float,
        )
    )

    y = pd.Series(
        np.asarray(
            y,
            dtype=float,
        )
    )

    if x.nunique(dropna=False) <= 1:
        return float("nan")

    if y.nunique(dropna=False) <= 1:
        return float("nan")

    return float(
        x.corr(
            y,
            method="spearman",
        )
    )


# ============================================================
# Input loading
# ============================================================

def load_h10_records(
    cell_name: str,
    data_dir: Path,
) -> pd.DataFrame:

    rows: list[dict] = []

    required = {
        "policy_seed",
        "state_id",
        "source_episode_id",
        "source_step",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
        "absolute_consequence",
    }

    numeric_columns = [
        "policy_seed",
        "state_id",
        "source_episode_id",
        "source_step",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
        "absolute_consequence",
    ]

    for seed in POLICY_SEEDS:

        path = data_dir / f"seed{seed}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"{cell_name}: missing source file {path}"
            )

        with path.open() as f:
            payload = json.load(f)

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{cell_name}: {path}: "
                f"expected policy_seed={seed}, "
                f"got {payload['policy_seed']}"
            )

        records = payload["records"]

        if len(records) != 2800:
            raise ValueError(
                f"{cell_name}: {path}: "
                f"expected 2800 records, "
                f"got {len(records)}"
            )

        for record in records:
            if int(record["horizon"]) == HORIZON:
                rows.append(record)

    df = pd.DataFrame(rows)

    expected_total = 3500

    if len(df) != expected_total:
        raise ValueError(
            f"{cell_name}: expected {expected_total} "
            f"H=10 records, got {len(df)}"
        )

    counts = (
        df.groupby("policy_seed")
        .size()
        .to_dict()
    )

    for seed in POLICY_SEEDS:

        if counts.get(seed, 0) != 700:
            raise ValueError(
                f"{cell_name}: seed {seed}: "
                f"expected 700 H=10 records, "
                f"got {counts.get(seed, 0)}"
            )

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{cell_name}: missing required columns: "
            f"{missing}"
        )

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="raise",
        ).to_numpy()

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"{cell_name}: "
                f"non-finite values found in {column}"
            )

    return (
        df.sort_values(
            [
                "policy_seed",
                "state_id",
                "sigma",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Exact P2 risk-coverage implementation
# ============================================================

def risk_curve_for_seed(
    df: pd.DataFrame,
    cell_name: str,
    model_name: str,
    nonzero_only: bool,
) -> pd.DataFrame:

    rows: list[dict] = []

    working = df.copy()

    if nonzero_only:
        working = working[
            working["sigma"] > 0
        ].copy()

    for seed, seed_df in working.groupby(
        "policy_seed"
    ):

        seed_df = (
            seed_df.sort_values(
                [
                    "reliability_score",
                    "state_id",
                    "sigma",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        n = len(seed_df)

        for coverage_percent in COVERAGE_PERCENT:

            keep_n = max(
                1,
                int(
                    math.ceil(
                        coverage_percent
                        * n
                        / 100
                    )
                ),
            )

            retained = seed_df.iloc[:keep_n]

            rows.append(
                {
                    "cell": cell_name,
                    "model": model_name,
                    "policy_seed": int(seed),
                    "coverage_percent": coverage_percent,
                    "coverage": coverage_percent / 100.0,
                    "n_total": n,
                    "n_retained": keep_n,
                    "mean_observed_C10": float(
                        retained[
                            "absolute_consequence"
                        ].mean()
                    ),
                    "median_observed_C10": float(
                        retained[
                            "absolute_consequence"
                        ].median()
                    ),
                    "mean_reliability_score": float(
                        retained[
                            "reliability_score"
                        ].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Exact P2 decile implementation
# ============================================================

def deciles_for_seed(
    df: pd.DataFrame,
    cell_name: str,
    model_name: str,
) -> pd.DataFrame:

    rows: list[dict] = []

    for seed, seed_df in df.groupby(
        "policy_seed"
    ):

        seed_df = (
            seed_df.sort_values(
                [
                    "reliability_score",
                    "state_id",
                    "sigma",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        index_chunks = np.array_split(
            np.arange(len(seed_df)),
            N_DECILES,
        )

        for decile_index, indices in enumerate(
            index_chunks,
            start=1,
        ):

            chunk = seed_df.iloc[indices]

            rows.append(
                {
                    "cell": cell_name,
                    "model": model_name,
                    "policy_seed": int(seed),
                    "reliability_decile": (
                        decile_index
                    ),
                    "n": len(chunk),
                    "mean_reliability_score": float(
                        chunk[
                            "reliability_score"
                        ].mean()
                    ),
                    "median_reliability_score": float(
                        chunk[
                            "reliability_score"
                        ].median()
                    ),
                    "mean_observed_C10": float(
                        chunk[
                            "absolute_consequence"
                        ].mean()
                    ),
                    "median_observed_C10": float(
                        chunk[
                            "absolute_consequence"
                        ].median()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:

    for directory in [
        OUT_DIR,
        PRED_DIR,
        STATS_DIR,
        RISK_DIR,
        DECILE_DIR,
        PROV_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    all_prediction_rows: list[dict] = []
    seed_metric_rows: list[dict] = []

    all_decile_tables: list[pd.DataFrame] = []
    all_risk_tables: list[pd.DataFrame] = []
    all_risk_nonzero_tables: list[pd.DataFrame] = []

    source_hashes: dict[str, dict] = {}

    # --------------------------------------------------------
    # Process each frozen cell independently
    # --------------------------------------------------------

    for cell_name, data_dir in CELL_SPECS.items():

        print()
        print("=" * 90)
        print(f"CELL: {cell_name}")
        print(f"SOURCE: {data_dir.relative_to(ROOT)}")
        print("=" * 90)

        # Record source hashes before analysis.
        source_hashes[cell_name] = {}

        for seed in POLICY_SEEDS:

            path = (
                data_dir
                / f"seed{seed}.json"
            )

            if not path.exists():
                raise FileNotFoundError(path)

            source_hashes[cell_name][
                str(seed)
            ] = {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }

        df = load_h10_records(
            cell_name,
            data_dir,
        )

        print(
            f"H=10 records: {len(df)}"
        )

        # ----------------------------------------------------
        # Both P2 model specifications
        # ----------------------------------------------------

        for model_name, features in MODEL_SPECS.items():

            print()
            print("-" * 90)
            print(
                f"{cell_name} / {model_name}"
            )
            print(
                "features:",
                features,
            )
            print("-" * 90)

            model_predictions_for_risk: list[
                pd.DataFrame
            ] = []

            # ----------------------------------------------
            # Leave-one-policy-seed-out
            # ----------------------------------------------

            for test_seed in POLICY_SEEDS:

                train = df[
                    df["policy_seed"]
                    != test_seed
                ].copy()

                test = df[
                    df["policy_seed"]
                    == test_seed
                ].copy()

                if len(train) != 2800:
                    raise ValueError(
                        f"{cell_name}, "
                        f"{model_name}, "
                        f"test seed {test_seed}: "
                        f"expected 2800 training "
                        f"records, got {len(train)}"
                    )

                if len(test) != 700:
                    raise ValueError(
                        f"{cell_name}, "
                        f"{model_name}, "
                        f"test seed {test_seed}: "
                        f"expected 700 held-out "
                        f"records, got {len(test)}"
                    )

                model = build_model()

                X_train = train[
                    features
                ].to_numpy(dtype=float)

                y_train = train[
                    "absolute_consequence"
                ].to_numpy(dtype=float)

                X_test = test[
                    features
                ].to_numpy(dtype=float)

                y_test = test[
                    "absolute_consequence"
                ].to_numpy(dtype=float)

                # P2-exact fitting order.
                model.fit(
                    X_train,
                    y_train,
                )

                train_pred = model.predict(
                    X_train
                )

                test_pred = model.predict(
                    X_test
                )

                reliability = (
                    empirical_reliability_score(
                        train_predictions=train_pred,
                        heldout_predictions=test_pred,
                    )
                )

                mae = mean_absolute_error(
                    y_test,
                    test_pred,
                )

                rmse = math.sqrt(
                    mean_squared_error(
                        y_test,
                        test_pred,
                    )
                )

                rho = spearman_rho(
                    reliability,
                    y_test,
                )

                seed_metric_rows.append(
                    {
                        "cell": cell_name,
                        "model": model_name,
                        "test_seed": int(
                            test_seed
                        ),
                        "n_train_records": len(
                            train
                        ),
                        "n_test_records": len(
                            test
                        ),
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "spearman_reliability_vs_C10": rho,
                        "train_prediction_min": float(
                            np.min(train_pred)
                        ),
                        "train_prediction_max": float(
                            np.max(train_pred)
                        ),
                        "test_prediction_min": float(
                            np.min(test_pred)
                        ),
                        "test_prediction_max": float(
                            np.max(test_pred)
                        ),
                        "reliability_min": float(
                            np.min(reliability)
                        ),
                        "reliability_max": float(
                            np.max(reliability)
                        ),
                    }
                )

                test_out = test.copy()

                test_out["cell"] = cell_name
                test_out["model"] = model_name
                test_out["prediction_C10"] = (
                    test_pred
                )
                test_out["reliability_score"] = (
                    reliability
                )
                test_out["test_seed"] = int(
                    test_seed
                )

                cols = [
                    "cell",
                    "model",
                    "test_seed",
                    "policy_seed",
                    "state_id",
                    "source_episode_id",
                    "source_step",
                    "sigma",
                    "horizon",
                    "action_disagreement",
                    "support_distance",
                    "twin_critic_disagreement",
                    "absolute_consequence",
                    "prediction_C10",
                    "reliability_score",
                ]

                all_prediction_rows.extend(
                    test_out[
                        cols
                    ].to_dict(
                        "records"
                    )
                )

                model_predictions_for_risk.append(
                    test_out
                )

                print(
                    f"test_seed={test_seed}: "
                    f"MAE={mae:.6f}, "
                    f"RMSE={rmse:.6f}, "
                    f"Spearman={rho:.6f}"
                )

            model_pred_df = pd.concat(
                model_predictions_for_risk,
                ignore_index=True,
            )

            # ----------------------------------------------
            # Deciles
            # Exact P2 behavior: all H=10 records.
            # ----------------------------------------------

            deciles = deciles_for_seed(
                model_pred_df,
                cell_name,
                model_name,
            )

            # ----------------------------------------------
            # Risk coverage
            # Exact P2 behavior.
            # ----------------------------------------------

            risk_all = risk_curve_for_seed(
                model_pred_df,
                cell_name,
                model_name,
                nonzero_only=False,
            )

            risk_nonzero = risk_curve_for_seed(
                model_pred_df,
                cell_name,
                model_name,
                nonzero_only=True,
            )

            all_decile_tables.append(
                deciles
            )

            all_risk_tables.append(
                risk_all
            )

            all_risk_nonzero_tables.append(
                risk_nonzero
            )

    # --------------------------------------------------------
    # Save held-out predictions
    # --------------------------------------------------------

    prediction_df = (
        pd.DataFrame(
            all_prediction_rows
        )
        .sort_values(
            [
                "cell",
                "model",
                "test_seed",
                "state_id",
                "sigma",
            ]
        )
        .reset_index(drop=True)
    )

    prediction_path = (
        PRED_DIR
        / "P9_heldout_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # Seed-level statistics
    # --------------------------------------------------------

    seed_metrics = (
        pd.DataFrame(
            seed_metric_rows
        )
        .sort_values(
            [
                "cell",
                "model",
                "test_seed",
            ]
        )
        .reset_index(drop=True)
    )

    seed_metrics_path = (
        STATS_DIR
        / "P9_seed_level_metrics.csv"
    )

    seed_metrics.to_csv(
        seed_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Cell/model summary
    # Descriptive only.
    # --------------------------------------------------------

    summary_rows: list[dict] = []

    for (
        cell_name,
        model_name,
    ), group in seed_metrics.groupby(
        ["cell", "model"],
        sort=True,
    ):

        summary_rows.append(
            {
                "cell": cell_name,
                "model": model_name,
                "n_heldout_policy_seeds": len(
                    group
                ),
                "mean_mae": float(
                    group["mae"].mean()
                ),
                "sd_mae": float(
                    group["mae"].std()
                ),
                "mean_rmse": float(
                    group["rmse"].mean()
                ),
                "sd_rmse": float(
                    group["rmse"].std()
                ),
                "mean_spearman": float(
                    group[
                        "spearman_reliability_vs_C10"
                    ].mean()
                ),
                "sd_spearman": float(
                    group[
                        "spearman_reliability_vs_C10"
                    ].std()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "cell",
            "model",
        ]
    )

    summary_path = (
        STATS_DIR
        / "P9_cell_model_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Deciles
    # --------------------------------------------------------

    deciles = pd.concat(
        all_decile_tables,
        ignore_index=True,
    )

    deciles_path = (
        DECILE_DIR
        / "P9_reliability_deciles_by_seed.csv"
    )

    deciles.to_csv(
        deciles_path,
        index=False,
    )

    decile_summary = (
        deciles
        .groupby(
            [
                "cell",
                "model",
                "reliability_decile",
            ],
            as_index=False,
        )
        .agg(
            mean_reliability_score=(
                "mean_reliability_score",
                "mean",
            ),
            sd_reliability_score=(
                "mean_reliability_score",
                "std",
            ),
            mean_observed_C10=(
                "mean_observed_C10",
                "mean",
            ),
            sd_observed_C10=(
                "mean_observed_C10",
                "std",
            ),
            n_policy_seeds=(
                "policy_seed",
                "nunique",
            ),
        )
    )

    decile_summary_path = (
        DECILE_DIR
        / "P9_reliability_deciles_summary.csv"
    )

    decile_summary.to_csv(
        decile_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Risk coverage
    # --------------------------------------------------------

    risk_all = pd.concat(
        all_risk_tables,
        ignore_index=True,
    )

    risk_nonzero = pd.concat(
        all_risk_nonzero_tables,
        ignore_index=True,
    )

    risk_all_path = (
        RISK_DIR
        / "P9_risk_coverage_all_shifts_by_seed.csv"
    )

    risk_nonzero_path = (
        RISK_DIR
        / "P9_risk_coverage_nonzero_shift_by_seed.csv"
    )

    risk_all.to_csv(
        risk_all_path,
        index=False,
    )

    risk_nonzero.to_csv(
        risk_nonzero_path,
        index=False,
    )

    risk_summary = (
        risk_nonzero
        .groupby(
            [
                "cell",
                "model",
                "coverage_percent",
            ],
            as_index=False,
        )
        .agg(
            mean_observed_C10=(
                "mean_observed_C10",
                "mean",
            ),
            sd_observed_C10=(
                "mean_observed_C10",
                "std",
            ),
            mean_retained_reliability=(
                "mean_reliability_score",
                "mean",
            ),
            n_policy_seeds=(
                "policy_seed",
                "nunique",
            ),
        )
    )

    risk_summary_path = (
        RISK_DIR
        / "P9_risk_coverage_nonzero_shift_summary.csv"
    )

    risk_summary.to_csv(
        risk_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Monotonicity diagnostics
    # Exact P2 logic, extended with cell.
    # --------------------------------------------------------

    monotonic_rows: list[dict] = []

    for (
        cell_name,
        model_name,
    ), group in deciles.groupby(
        ["cell", "model"]
    ):

        for seed, seed_df in group.groupby(
            "policy_seed"
        ):

            ordered = (
                seed_df.sort_values(
                    "reliability_decile"
                )[
                    "mean_observed_C10"
                ]
                .to_numpy()
            )

            violations = int(
                np.sum(
                    np.diff(ordered) < 0
                )
            )

            monotonic_rows.append(
                {
                    "cell": cell_name,
                    "model": model_name,
                    "policy_seed": int(seed),
                    "decile_adjacent_violations": violations,
                    "max_possible_adjacent_violations": (
                        N_DECILES - 1
                    ),
                    "monotonic_non_decreasing": (
                        violations == 0
                    ),
                }
            )

    monotonicity = (
        pd.DataFrame(
            monotonic_rows
        )
        .sort_values(
            [
                "cell",
                "model",
                "policy_seed",
            ]
        )
    )

    monotonicity_path = (
        STATS_DIR
        / "P9_decile_monotonicity_by_seed.csv"
    )

    monotonicity.to_csv(
        monotonicity_path,
        index=False,
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance = {
        "experiment": (
            "P9 cross-cell reliability "
            "estimator replication"
        ),
        "repository_commit": git_command(
            "rev-parse",
            "HEAD",
        ),
        "repository_status": git_command(
            "status",
            "--porcelain",
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__(
            "sklearn"
        ).__version__,
        "horizon": HORIZON,
        "policy_seeds": POLICY_SEEDS,
        "coverage_percent": COVERAGE_PERCENT,
        "n_deciles": N_DECILES,
        "cells": {
            name: str(
                path.relative_to(ROOT)
            )
            for name, path
            in CELL_SPECS.items()
        },
        "primary_features": MODEL_SPECS[
            "primary_3feature"
        ],
        "sensitivity_features": MODEL_SPECS[
            "action_only_sensitivity"
        ],
        "estimator": {
            "preprocessing": (
                "StandardScaler"
            ),
            "model": "Ridge",
            "alpha": RIDGE_ALPHA,
            "intercept": True,
            "stochasticity": "none",
        },
        "reliability_score": (
            "R(x) = 1 - F_train(predicted_C10)"
        ),
        "cross_policy_evaluation": (
            "leave-one-policy-seed-out"
        ),
        "p2_inheritance": {
            "base_protocol_sha256": (
                sha256_file(
                    ROOT
                    / "experiments"
                    / "reliability"
                    / "P2_RELIABILITY_SCORE_PROTOCOL.md"
                )
            ),
            "amendment_001_sha256": (
                sha256_file(
                    ROOT
                    / "experiments"
                    / "reliability"
                    / "P2_AMENDMENT_001_PRIMARY_ESTIMATOR.md"
                )
            ),
        },
        "p9_protocol_sha256": (
            sha256_file(
                ROOT
                / "experiments"
                / "reliability"
                / "P9_CROSS_CELL_RELIABILITY_ESTIMATOR_PROTOCOL.md"
            )
        ),
        "source_hashes": source_hashes,
        "source_records": int(
            len(prediction_df)
        ),
        "records_per_cell": {
            cell: 3500
            for cell in CELL_SPECS
        },
        "heldout_prediction_records": int(
            len(prediction_df)
        ),
    }

    provenance_path = (
        PROV_DIR
        / "P9_provenance.json"
    )

    with provenance_path.open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Human-readable evidence summary
    # --------------------------------------------------------

    summary_lines = [
        "# P9 Cross-Cell Reliability Estimator Evidence Summary",
        "",
        "- Horizon: H=10",
        "- Independent unit: policy seed within cell",
        "- Cells: IQL Hopper, IQL HalfCheetah, IQL Walker2d, "
        "CQL Hopper, CQL HalfCheetah, CQL Walker2d",
        "- Policy seeds: 0, 1, 2, 3, 4",
        "- Evaluation: leave-one-policy-seed-out within each cell",
        "- Primary estimator: StandardScaler + Ridge(alpha=1.0)",
        "- Primary features: action disagreement, support distance, "
        "twin-critic disagreement",
        "- Sensitivity estimator: action disagreement only",
        "- Reliability score: R(x) = 1 - F_train(predicted C10)",
        "- Coverage: 10, 20, 30, 50, 70, 90, 100%",
        "- Reliability deciles: 10 positional deciles",
        "",
        "## Cell-level descriptive summaries",
        "",
        "| Cell | Model | Held-out seeds | Mean MAE | SD MAE | "
        "Mean RMSE | SD RMSE | Mean Spearman(R,C10) | SD Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary.to_dict(
        "records"
    ):

        summary_lines.append(
            "| "
            f"{row['cell']} | "
            f"{row['model']} | "
            f"{row['n_heldout_policy_seeds']} | "
            f"{row['mean_mae']:.8f} | "
            f"{row['sd_mae']:.8f} | "
            f"{row['mean_rmse']:.8f} | "
            f"{row['sd_rmse']:.8f} | "
            f"{row['mean_spearman']:.8f} | "
            f"{row['sd_spearman']:.8f} |"
        )

    # Descriptive monotonicity counts.
    summary_lines.extend(
        [
            "",
            "## Decile monotonicity",
            "",
            "The diagnostic counts below describe how often observed "
            "C10 is non-decreasing across the reliability deciles "
            "within held-out policy seeds.",
            "",
        ]
    )

    mono_group = (
        monotonicity
        .groupby(
            ["cell", "model"],
            as_index=False,
        )
        .agg(
            n_seed_evaluations=(
                "policy_seed",
                "nunique",
            ),
            n_monotonic=(
                "monotonic_non_decreasing",
                "sum",
            ),
            total_adjacent_violations=(
                "decile_adjacent_violations",
                "sum",
            ),
        )
    )

    for row in mono_group.to_dict(
        "records"
    ):

        summary_lines.append(
            "- "
            f"{row['cell']} / {row['model']}: "
            f"{int(row['n_monotonic'])}/"
            f"{int(row['n_seed_evaluations'])} "
            "held-out seeds monotonic; "
            f"{int(row['total_adjacent_violations'])} "
            "total adjacent-decile violations."
        )

    summary_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "P9 evaluates cross-cell replication of the frozen "
            "continuous consequence estimator.",
            "The six cells are summarized descriptively and are not "
            "treated as a pooled inferential population.",
            "The analysis does not establish causal faithfulness, "
            "introduce a binary failure threshold, or select a "
            "universally best estimator.",
            "",
            f"Repository commit: "
            f"{provenance['repository_commit']}",
        ]
    )

    evidence_path = (
        OUT_DIR
        / "P9_EVIDENCE_SUMMARY.md"
    )

    evidence_path.write_text(
        "\n".join(summary_lines)
        + "\n"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("P9 CROSS-CELL RELIABILITY ANALYSIS COMPLETE")
    print("=" * 90)
    print()
    print(
        "Cells:",
        len(CELL_SPECS),
    )
    print(
        "H=10 records per cell:",
        3500,
    )
    print(
        "Held-out prediction records:",
        len(prediction_df),
    )
    print()
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print("Outputs:")
    print(
        " ",
        prediction_path,
    )
    print(
        " ",
        seed_metrics_path,
    )
    print(
        " ",
        summary_path,
    )
    print(
        " ",
        decile_summary_path,
    )
    print(
        " ",
        risk_summary_path,
    )
    print(
        " ",
        monotonicity_path,
    )
    print(
        " ",
        provenance_path,
    )
    print(
        " ",
        evidence_path,
    )
    print()


if __name__ == "__main__":
    main()
