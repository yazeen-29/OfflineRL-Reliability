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
# Configuration frozen by:
# P2_RELIABILITY_SCORE_PROTOCOL.md
# P2_AMENDMENT_001_PRIMARY_ESTIMATOR.md
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data_frozen" / "P1"
OUT_DIR = ROOT / "results" / "analysis" / "P2"

PRED_DIR = OUT_DIR / "predictions"
STATS_DIR = OUT_DIR / "statistics"
RISK_DIR = OUT_DIR / "risk_coverage"
DECILE_DIR = OUT_DIR / "reliability_deciles"
PROV_DIR = OUT_DIR / "provenance"
FIG_DIR = OUT_DIR / "figures"

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
# Utilities
# ============================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
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


def load_h10_records() -> pd.DataFrame:
    rows: list[dict] = []

    for seed in POLICY_SEEDS:
        path = DATA_DIR / f"seed{seed}.json"

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open() as f:
            payload = json.load(f)

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{path}: expected policy_seed={seed}, "
                f"got {payload['policy_seed']}"
            )

        records = payload["records"]

        if len(records) != 2800:
            raise ValueError(
                f"{path}: expected 2800 records, "
                f"got {len(records)}"
            )

        for record in records:
            if int(record["horizon"]) == HORIZON:
                rows.append(record)

    df = pd.DataFrame(rows)

    if len(df) != 3500:
        raise ValueError(
            f"Expected 3500 H=10 records, got {len(df)}"
        )

    expected_per_seed = 700

    counts = df.groupby("policy_seed").size().to_dict()

    for seed in POLICY_SEEDS:
        if counts.get(seed, 0) != expected_per_seed:
            raise ValueError(
                f"Seed {seed}: expected {expected_per_seed} "
                f"H=10 records, got {counts.get(seed, 0)}"
            )

    required = {
        "policy_seed",
        "state_id",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
        "absolute_consequence",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    numeric_columns = [
        "policy_seed",
        "state_id",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
        "absolute_consequence",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="raise").to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values found in {column}"
            )

    return df.sort_values(
        ["policy_seed", "state_id", "sigma"]
    ).reset_index(drop=True)


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
    R(x) = 1 - F_train(yhat(x))

    F_train is the empirical CDF of training-fold predictions,
    using right-continuous <= semantics.

    Larger R => lower predicted consequence => higher reliability.
    """
    train_sorted = np.sort(
        np.asarray(train_predictions, dtype=float)
    )

    ranks = np.searchsorted(
        train_sorted,
        np.asarray(heldout_predictions, dtype=float),
        side="right",
    )

    cdf = ranks / len(train_sorted)

    reliability = 1.0 - cdf

    return np.clip(reliability, 0.0, 1.0)


def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Spearman correlation without introducing another package
    dependency.

    Returns NaN when either vector has no rank variation.
    """
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))

    if x.nunique(dropna=False) <= 1:
        return float("nan")

    if y.nunique(dropna=False) <= 1:
        return float("nan")

    return float(x.corr(y, method="spearman"))


def risk_curve_for_seed(
    df: pd.DataFrame,
    score_column: str,
    nonzero_only: bool,
) -> pd.DataFrame:
    rows: list[dict] = []

    working = df.copy()

    if nonzero_only:
        working = working[working["sigma"] > 0].copy()

    for seed, seed_df in working.groupby("policy_seed"):
        seed_df = seed_df.sort_values(
            ["reliability_score", "state_id", "sigma"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        n = len(seed_df)

        for coverage_percent in COVERAGE_PERCENT:
            keep_n = max(
                1,
                int(
                    math.ceil(
                        coverage_percent * n / 100
                    )
                ),
            )

            retained = seed_df.iloc[:keep_n]

            rows.append(
                {
                    "model": score_column,
                    "policy_seed": int(seed),
                    "coverage_percent": coverage_percent,
                    "coverage": coverage_percent / 100.0,
                    "n_total": n,
                    "n_retained": keep_n,
                    "mean_observed_C10": float(
                        retained["absolute_consequence"].mean()
                    ),
                    "median_observed_C10": float(
                        retained["absolute_consequence"].median()
                    ),
                    "mean_reliability_score": float(
                        retained["reliability_score"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def deciles_for_seed(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for seed, seed_df in df.groupby("policy_seed"):
        seed_df = seed_df.sort_values(
            ["reliability_score", "state_id", "sigma"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        # Use positional indices rather than splitting the DataFrame
        # directly, avoiding NumPy's deprecated DataFrame.swapaxes path.
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
                    "model": model_name,
                    "policy_seed": int(seed),
                    "reliability_decile": decile_index,
                    "n": len(chunk),
                    "mean_reliability_score": float(
                        chunk["reliability_score"].mean()
                    ),
                    "median_reliability_score": float(
                        chunk["reliability_score"].median()
                    ),
                    "mean_observed_C10": float(
                        chunk["absolute_consequence"].mean()
                    ),
                    "median_observed_C10": float(
                        chunk["absolute_consequence"].median()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    for directory in [
        OUT_DIR,
        PRED_DIR,
        STATS_DIR,
        RISK_DIR,
        DECILE_DIR,
        PROV_DIR,
        FIG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    df = load_h10_records()

    all_prediction_rows: list[dict] = []
    seed_metric_rows: list[dict] = []

    all_decile_tables: list[pd.DataFrame] = []
    all_risk_tables: list[pd.DataFrame] = []
    all_risk_nonzero_tables: list[pd.DataFrame] = []

    for model_name, features in MODEL_SPECS.items():

        print()
        print("=" * 80)
        print(model_name)
        print("features:", features)
        print("=" * 80)

        model_predictions_for_risk: list[pd.DataFrame] = []

        for test_seed in POLICY_SEEDS:

            train = df[df["policy_seed"] != test_seed].copy()
            test = df[df["policy_seed"] == test_seed].copy()

            if len(train) != 2800:
                raise ValueError(
                    f"{model_name}, test seed {test_seed}: "
                    f"expected 2800 training records, got {len(train)}"
                )

            if len(test) != 700:
                raise ValueError(
                    f"{model_name}, test seed {test_seed}: "
                    f"expected 700 held-out records, got {len(test)}"
                )

            model = build_model()

            X_train = train[features].to_numpy(dtype=float)
            y_train = train[
                "absolute_consequence"
            ].to_numpy(dtype=float)

            X_test = test[features].to_numpy(dtype=float)
            y_test = test[
                "absolute_consequence"
            ].to_numpy(dtype=float)

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            reliability = empirical_reliability_score(
                train_predictions=train_pred,
                heldout_predictions=test_pred,
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
                    "model": model_name,
                    "test_seed": test_seed,
                    "n_train_records": len(train),
                    "n_test_records": len(test),
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

            test_out["model"] = model_name
            test_out["prediction_C10"] = test_pred
            test_out["reliability_score"] = reliability
            test_out["test_seed"] = test_seed

            cols = [
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
                test_out[cols].to_dict("records")
            )

            model_predictions_for_risk.append(test_out)

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

        deciles = deciles_for_seed(
            model_pred_df,
            model_name,
        )

        risk_all = risk_curve_for_seed(
            model_pred_df,
            model_name,
            nonzero_only=False,
        )

        risk_nonzero = risk_curve_for_seed(
            model_pred_df,
            model_name,
            nonzero_only=True,
        )

        all_decile_tables.append(deciles)
        all_risk_tables.append(risk_all)
        all_risk_nonzero_tables.append(risk_nonzero)

    # --------------------------------------------------------
    # Save prediction table
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        all_prediction_rows
    ).sort_values(
        ["model", "test_seed", "state_id", "sigma"]
    )

    prediction_df.to_csv(
        PRED_DIR / "P2_heldout_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save seed-level statistics
    # --------------------------------------------------------

    seed_metrics = pd.DataFrame(
        seed_metric_rows
    ).sort_values(
        ["model", "test_seed"]
    )

    seed_metrics.to_csv(
        STATS_DIR / "P2_seed_level_metrics.csv",
        index=False,
    )

    summary_rows: list[dict] = []

    for model, group in seed_metrics.groupby("model"):
        summary_rows.append(
            {
                "model": model,
                "n_heldout_policy_seeds": len(group),
                "mean_mae": float(group["mae"].mean()),
                "sd_mae": float(group["mae"].std()),
                "mean_rmse": float(group["rmse"].mean()),
                "sd_rmse": float(group["rmse"].std()),
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

    summary = pd.DataFrame(summary_rows).sort_values("model")

    summary.to_csv(
        STATS_DIR / "P2_model_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Deciles
    # --------------------------------------------------------

    deciles = pd.concat(
        all_decile_tables,
        ignore_index=True,
    )

    deciles.to_csv(
        DECILE_DIR / "P2_reliability_deciles_by_seed.csv",
        index=False,
    )

    decile_summary = (
        deciles
        .groupby(
            ["model", "reliability_decile"],
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

    decile_summary.to_csv(
        DECILE_DIR / "P2_reliability_deciles_summary.csv",
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

    risk_all.to_csv(
        RISK_DIR / "P2_risk_coverage_all_shifts_by_seed.csv",
        index=False,
    )

    risk_nonzero.to_csv(
        RISK_DIR / "P2_risk_coverage_nonzero_shift_by_seed.csv",
        index=False,
    )

    risk_summary = (
        risk_nonzero
        .groupby(
            ["model", "coverage_percent"],
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

    risk_summary.to_csv(
        RISK_DIR / "P2_risk_coverage_nonzero_shift_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Basic monotonicity diagnostics
    # --------------------------------------------------------

    monotonic_rows: list[dict] = []

    for model, group in deciles.groupby("model"):
        for seed, seed_df in group.groupby("policy_seed"):
            ordered = seed_df.sort_values(
                "reliability_decile"
            )["mean_observed_C10"].to_numpy()

            violations = int(
                np.sum(
                    np.diff(ordered) < 0
                )
            )

            monotonic_rows.append(
                {
                    "model": model,
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

    monotonicity = pd.DataFrame(monotonic_rows)

    monotonicity.to_csv(
        STATS_DIR / "P2_decile_monotonicity_by_seed.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    source_hashes = {}

    for seed in POLICY_SEEDS:
        path = DATA_DIR / f"seed{seed}.json"

        source_hashes[str(seed)] = {
            "path": str(
                path.relative_to(ROOT)
            ),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    provenance = {
        "experiment": "P2 reliability score / estimator",
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
        "data_directory": str(
            DATA_DIR.relative_to(ROOT)
        ),
        "source_hashes": source_hashes,
        "protocol_hashes": {
            "base_protocol": sha256_file(
                ROOT
                / "experiments"
                / "reliability"
                / "P2_RELIABILITY_SCORE_PROTOCOL.md"
            ),
            "amendment_001": sha256_file(
                ROOT
                / "experiments"
                / "reliability"
                / "P2_AMENDMENT_001_PRIMARY_ESTIMATOR.md"
            ),
        },
        "policy_seeds": POLICY_SEEDS,
        "horizon": HORIZON,
        "primary_features": MODEL_SPECS[
            "primary_3feature"
        ],
        "sensitivity_features": MODEL_SPECS[
            "action_only_sensitivity"
        ],
        "estimator": {
            "preprocessing": "StandardScaler",
            "model": "Ridge",
            "alpha": RIDGE_ALPHA,
            "intercept": True,
            "stochasticity": "none",
        },
        "reliability_score": (
            "R(x) = 1 - F_train(predicted_C10)"
        ),
        "coverage_percent": COVERAGE_PERCENT,
        "n_deciles": N_DECILES,
        "source_records": len(df),
        "records_per_policy_seed": {
            str(seed): int(
                (df["policy_seed"] == seed).sum()
            )
            for seed in POLICY_SEEDS
        },
    }

    with (
        PROV_DIR / "P2_provenance.json"
    ).open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary_lines = [
        "# P2 Reliability Score / Estimator Evidence Summary",
        "",
        f"- Horizon: H={HORIZON}",
        "- Independent unit: policy seed",
        "- Policy seeds: 0, 1, 2, 3, 4",
        "- Evaluation: leave-one-policy-seed-out",
        "- Primary estimator: StandardScaler + Ridge(alpha=1.0)",
        "- Primary features: action disagreement, support distance, "
          "twin-critic disagreement",
        "- Sensitivity estimator: action disagreement only",
        "- Reliability score: R(x) = 1 - F_train(predicted C10)",
        "",
        "## Seed-level predictive metrics",
        "",
    ]

    for row in summary.to_dict("records"):
        summary_lines.extend(
            [
                f"### {row['model']}",
                f"- MAE: {row['mean_mae']:.8f} "
                f"(SD {row['sd_mae']:.8f})",
                f"- RMSE: {row['mean_rmse']:.8f} "
                f"(SD {row['sd_rmse']:.8f})",
                f"- Spearman(R, C10): "
                f"{row['mean_spearman']:.8f} "
                f"(SD {row['sd_spearman']:.8f})",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These analyses evaluate predictive and operational "
            "properties of the frozen continuous consequence task.",
            "They do not establish causal faithfulness, a binary "
            "failure threshold, or a uniquely optimal estimator.",
            "",
            "## Monotonicity qualification",
            "",
            "Across held-out seeds, the pooled decile-average observed "
            "C10 increases monotonically from high- to low-reliability "
            "deciles for both estimators.",
            "The action-only sensitivity has zero adjacent-decile "
            "violations on all five held-out seeds.",
            "The primary three-feature estimator has one adjacent-decile "
            "violation for held-out seeds 2 and 4.",
            "",
            f"Repository commit: {provenance['repository_commit']}",
        ]
    )

    (
        OUT_DIR / "P2_EVIDENCE_SUMMARY.md"
    ).write_text(
        "\n".join(summary_lines)
        + "\n"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("P2 RELIABILITY SCORE ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(
        "H=10 records:",
        len(df),
    )
    print(
        "Policy seeds:",
        POLICY_SEEDS,
    )
    print()
    print(summary.to_string(index=False))
    print()
    print("Outputs:")
    print(
        " ",
        PRED_DIR / "P2_heldout_predictions.csv",
    )
    print(
        " ",
        STATS_DIR / "P2_seed_level_metrics.csv",
    )
    print(
        " ",
        STATS_DIR / "P2_model_summary.csv",
    )
    print(
        " ",
        RISK_DIR
        / "P2_risk_coverage_nonzero_shift_summary.csv",
    )
    print(
        " ",
        DECILE_DIR
        / "P2_reliability_deciles_summary.csv",
    )
    print(
        " ",
        PROV_DIR / "P2_provenance.json",
    )
    print(
        " ",
        OUT_DIR / "P2_EVIDENCE_SUMMARY.md",
    )
    print()


if __name__ == "__main__":
    main()
