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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data_frozen" / "P1"
OUT_DIR = ROOT / "results" / "analysis" / "P4"

PRED_DIR = OUT_DIR / "predictions"
STAT_DIR = OUT_DIR / "statistics"
DECILE_DIR = OUT_DIR / "reliability_deciles"
RISK_DIR = OUT_DIR / "risk_coverage"
PROV_DIR = OUT_DIR / "provenance"

for d in [
    OUT_DIR,
    PRED_DIR,
    STAT_DIR,
    DECILE_DIR,
    RISK_DIR,
    PROV_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


POLICY_SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10
COVERAGE_PERCENT = [10, 20, 30, 50, 70, 90, 100]
N_DECILES = 10
RIDGE_ALPHA = 1.0


MODEL_SPECS = {
    "B1_support_only": {
        "kind": "raw_score",
        "features": ["support_distance"],
    },
    "B2_critic_only": {
        "kind": "raw_score",
        "features": ["twin_critic_disagreement"],
    },
    "B3_support_critic": {
        "kind": "ridge",
        "features": [
            "support_distance",
            "twin_critic_disagreement",
        ],
    },
    "B4_action_only": {
        "kind": "raw_score",
        "features": ["action_disagreement"],
    },
    "B5_full": {
        "kind": "ridge",
        "features": [
            "action_disagreement",
            "support_distance",
            "twin_critic_disagreement",
        ],
    },
}

MODEL_LABELS = {
    "B1_support_only": "Support / OOD only",
    "B2_critic_only": "Critic uncertainty only",
    "B3_support_critic": "Support + critic",
    "B4_action_only": "Action disagreement only",
    "B5_full": "Action + support + critic",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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


def load_h10_nonzero() -> pd.DataFrame:
    rows = []

    for seed in POLICY_SEEDS:
        path = DATA_DIR / f"seed{seed}.json"

        with path.open() as f:
            payload = json.load(f)

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{path}: expected policy_seed={seed}"
            )

        for record in payload["records"]:
            if int(record["horizon"]) != HORIZON:
                continue

            if float(record["sigma"]) <= 0:
                continue

            rows.append(
                {
                    "policy_seed": int(
                        record["policy_seed"]
                    ),
                    "state_id": int(record["state_id"]),
                    "source_episode_id": int(
                        record["source_episode_id"]
                    ),
                    "source_step": int(
                        record["source_step"]
                    ),
                    "sigma": float(record["sigma"]),
                    "action_disagreement": float(
                        record["action_disagreement"]
                    ),
                    "support_distance": float(
                        record["support_distance"]
                    ),
                    "twin_critic_disagreement": float(
                        record[
                            "twin_critic_disagreement"
                        ]
                    ),
                    "absolute_consequence": float(
                        record[
                            "absolute_consequence"
                        ]
                    ),
                }
            )

    df = pd.DataFrame(rows)

    if len(df) != 3000:
        raise ValueError(
            f"Expected 3000 H=10 nonzero-shift records; "
            f"got {len(df)}"
        )

    counts = (
        df.groupby("policy_seed")
        .size()
        .to_dict()
    )

    for seed in POLICY_SEEDS:
        if counts.get(seed, 0) != 600:
            raise ValueError(
                f"Seed {seed}: expected 600 records; "
                f"got {counts.get(seed, 0)}"
            )

    required = {
        "policy_seed",
        "state_id",
        "sigma",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
        "absolute_consequence",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    for col in required:
        values = pd.to_numeric(
            df[col],
            errors="raise",
        ).to_numpy()

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values in {col}"
            )

    return df.sort_values(
        ["policy_seed", "state_id", "sigma"]
    ).reset_index(drop=True)


def build_ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def empirical_reliability(
    train_score: np.ndarray,
    heldout_score: np.ndarray,
) -> np.ndarray:
    """
    R = 1 - F_train(score)

    For all P4 diagnostic signals and predicted C10, larger raw
    signal / prediction means lower reliability.
    """
    train_sorted = np.sort(
        np.asarray(train_score, dtype=float)
    )

    ranks = np.searchsorted(
        train_sorted,
        np.asarray(heldout_score, dtype=float),
        side="right",
    )

    cdf = ranks / len(train_sorted)

    return np.clip(
        1.0 - cdf,
        0.0,
        1.0,
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)

    if sx.nunique() <= 1:
        return float("nan")

    if sy.nunique() <= 1:
        return float("nan")

    return float(
        sx.corr(sy, method="spearman")
    )


def make_deciles(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    rows = []

    for seed, group in df.groupby(
        "policy_seed",
        sort=True,
    ):
        g = (
            group
            .sort_values(
                [
                    "reliability_score",
                    "state_id",
                    "sigma",
                ],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        index_chunks = np.array_split(
            np.arange(len(g)),
            N_DECILES,
        )

        for decile, indices in enumerate(
            index_chunks,
            start=1,
        ):
            chunk = g.iloc[indices]

            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "policy_seed": int(seed),
                    "reliability_decile": decile,
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


def make_risk_coverage(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    rows = []

    for seed, group in df.groupby(
        "policy_seed",
        sort=True,
    ):
        g = (
            group
            .sort_values(
                [
                    "reliability_score",
                    "state_id",
                    "sigma",
                ],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        n = len(g)

        for coverage_percent in COVERAGE_PERCENT:
            keep_n = max(
                1,
                int(
                    math.ceil(
                        coverage_percent * n / 100
                    )
                ),
            )

            retained = g.iloc[:keep_n]

            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "policy_seed": int(seed),
                    "coverage_percent":
                        coverage_percent,
                    "coverage":
                        coverage_percent / 100.0,
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


def main() -> None:
    df = load_h10_nonzero()

    all_predictions = []
    seed_metrics = []
    all_deciles = []
    all_risk = []

    for model_name, spec in MODEL_SPECS.items():
        kind = spec["kind"]
        features = spec["features"]

        print()
        print("=" * 80)
        print(model_name)
        print("kind:", kind)
        print("features:", features)
        print("=" * 80)

        for test_seed in POLICY_SEEDS:
            train = df[
                df["policy_seed"] != test_seed
            ].copy()

            test = df[
                df["policy_seed"] == test_seed
            ].copy()

            if len(train) != 2400:
                raise ValueError(
                    f"{model_name}, seed {test_seed}: "
                    f"expected 2400 train rows; got {len(train)}"
                )

            if len(test) != 600:
                raise ValueError(
                    f"{model_name}, seed {test_seed}: "
                    f"expected 600 test rows; got {len(test)}"
                )

            y_test = test[
                "absolute_consequence"
            ].to_numpy(float)

            if kind == "raw_score":
                train_raw = train[
                    features[0]
                ].to_numpy(float)

                test_raw = test[
                    features[0]
                ].to_numpy(float)

                score = test_raw

                reliability = empirical_reliability(
                    train_score=train_raw,
                    heldout_score=test_raw,
                )

                prediction = np.full(
                    len(test),
                    np.nan,
                    dtype=float,
                )

                mae = np.nan
                rmse = np.nan

            elif kind == "ridge":
                model = build_ridge()

                X_train = train[
                    features
                ].to_numpy(float)

                y_train = train[
                    "absolute_consequence"
                ].to_numpy(float)

                X_test = test[
                    features
                ].to_numpy(float)

                model.fit(
                    X_train,
                    y_train,
                )

                train_prediction = model.predict(
                    X_train
                )

                prediction = model.predict(
                    X_test
                )

                reliability = empirical_reliability(
                    train_score=train_prediction,
                    heldout_score=prediction,
                )

                mae = float(
                    np.mean(
                        np.abs(
                            y_test
                            - prediction
                        )
                    )
                )

                rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                y_test
                                - prediction
                            ) ** 2
                        )
                    )
                )

            else:
                raise ValueError(
                    f"Unknown model kind: {kind}"
                )

            rho = spearman(
                reliability,
                y_test,
            )

            seed_metrics.append(
                {
                    "model": model_name,
                    "model_label":
                        MODEL_LABELS[model_name],
                    "test_seed": test_seed,
                    "kind": kind,
                    "n_train_records": len(train),
                    "n_test_records": len(test),
                    "spearman_reliability_vs_C10":
                        rho,
                    "mae": mae,
                    "rmse": rmse,
                    "reliability_min":
                        float(reliability.min()),
                    "reliability_max":
                        float(reliability.max()),
                    "reliability_mean":
                        float(reliability.mean()),
                    "reliability_std":
                        float(reliability.std()),
                }
            )

            out = test.copy()

            out["model"] = model_name
            out["model_label"] = MODEL_LABELS[
                model_name
            ]
            out["test_seed"] = test_seed
            out["prediction_C10"] = prediction
            out["reliability_score"] = reliability

            all_predictions.extend(
                out[
                    [
                        "model",
                        "model_label",
                        "test_seed",
                        "policy_seed",
                        "state_id",
                        "source_episode_id",
                        "source_step",
                        "sigma",
                        "action_disagreement",
                        "support_distance",
                        "twin_critic_disagreement",
                        "absolute_consequence",
                        "prediction_C10",
                        "reliability_score",
                    ]
                ].to_dict("records")
            )

            print(
                f"test_seed={test_seed}: "
                f"Spearman={rho:.6f}"
                + (
                    f", MAE={mae:.6f}, RMSE={rmse:.6f}"
                    if not np.isnan(mae)
                    else ""
                )
            )

        model_pred = pd.DataFrame(
            [
                r for r in all_predictions
                if r["model"] == model_name
            ]
        )

        all_deciles.append(
            make_deciles(
                model_pred,
                model_name,
            )
        )

        all_risk.append(
            make_risk_coverage(
                model_pred,
                model_name,
            )
        )

    predictions = pd.DataFrame(
        all_predictions
    ).sort_values(
        [
            "model",
            "test_seed",
            "state_id",
            "sigma",
        ]
    )

    metrics = pd.DataFrame(
        seed_metrics
    ).sort_values(
        ["model", "test_seed"]
    )

    deciles = pd.concat(
        all_deciles,
        ignore_index=True,
    )

    risk = pd.concat(
        all_risk,
        ignore_index=True,
    )

    predictions.to_csv(
        PRED_DIR / "P4_heldout_scores.csv",
        index=False,
    )

    metrics.to_csv(
        STAT_DIR / "P4_seed_level_metrics.csv",
        index=False,
    )

    deciles.to_csv(
        DECILE_DIR
        / "P4_reliability_deciles_by_seed.csv",
        index=False,
    )

    risk.to_csv(
        RISK_DIR
        / "P4_risk_coverage_nonzero_by_seed.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    summary = (
        metrics
        .groupby(
            [
                "model",
                "model_label",
                "kind",
            ],
            as_index=False,
        )
        .agg(
            n_heldout_policy_seeds=(
                "test_seed",
                "nunique",
            ),
            mean_spearman=(
                "spearman_reliability_vs_C10",
                "mean",
            ),
            sd_spearman=(
                "spearman_reliability_vs_C10",
                "std",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            sd_mae=(
                "mae",
                "std",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            sd_rmse=(
                "rmse",
                "std",
            ),
        )
    )

    summary.to_csv(
        STAT_DIR / "P4_model_summary.csv",
        index=False,
    )

    # Decile pooled summary
    decile_summary = (
        deciles
        .groupby(
            [
                "model",
                "model_label",
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

    decile_summary.to_csv(
        DECILE_DIR
        / "P4_reliability_deciles_summary.csv",
        index=False,
    )

    # Risk-coverage pooled summary
    risk_summary = (
        risk
        .groupby(
            [
                "model",
                "model_label",
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

    risk_summary.to_csv(
        RISK_DIR
        / "P4_risk_coverage_nonzero_summary.csv",
        index=False,
    )

    # Direction consistency by seed.
    direction = (
        metrics
        .groupby("model")
        .agg(
            n_negative_spearman=(
                "spearman_reliability_vs_C10",
                lambda x: int((x < 0).sum()),
            ),
            n_policy_seeds=(
                "test_seed",
                "nunique",
            ),
        )
        .reset_index()
    )

    direction.to_csv(
        STAT_DIR
        / "P4_direction_consistency.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    source_hashes = {
        str(seed): sha256_file(
            DATA_DIR / f"seed{seed}.json"
        )
        for seed in POLICY_SEEDS
    }

    provenance = {
        "experiment":
            "P4 OOD / uncertainty baselines",
        "repository_commit":
            git_command(
                "rev-parse",
                "HEAD",
            ),
        "repository_status":
            git_command(
                "status",
                "--porcelain",
            ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn":
            __import__("sklearn").__version__,
        "source_data":
            "data_frozen/P1",
        "source_hashes":
            source_hashes,
        "protocol_hash":
            sha256_file(
                ROOT
                / "experiments"
                / "reliability"
                / "P4_OOD_UNCERTAINTY_BASELINES_PROTOCOL.md"
            ),
        "policy_seeds": POLICY_SEEDS,
        "horizon": HORIZON,
        "sigma_condition": "sigma > 0",
        "coverage_percent":
            COVERAGE_PERCENT,
        "n_deciles": N_DECILES,
        "ridge_alpha": RIDGE_ALPHA,
        "models": MODEL_SPECS,
        "model_labels": MODEL_LABELS,
        "source_records": len(df),
        "records_per_policy_seed": {
            str(seed): int(
                (df["policy_seed"] == seed).sum()
            )
            for seed in POLICY_SEEDS
        },
    }

    with (
        PROV_DIR / "P4_provenance.json"
    ).open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    lines = [
        "# P4 OOD / Uncertainty Baseline Evidence Summary",
        "",
        f"- Horizon: H={HORIZON}",
        "- Population: nonzero observation shifts",
        "- Independent unit: policy seed",
        "- Policy seeds: 0, 1, 2, 3, 4",
        "",
        "## Baselines",
        "",
        "- B1: support / OOD only",
        "- B2: critic uncertainty only",
        "- B3: support + critic",
        "- B4: action disagreement only",
        "- B5: action + support + critic",
        "",
        "## Seed-level summary",
        "",
    ]

    for row in summary.to_dict("records"):
        lines.extend(
            [
                f"### {row['model_label']}",
                f"- Mean Spearman(R,C10): "
                f"{row['mean_spearman']:.6f} "
                f"(SD {row['sd_spearman']:.6f})",
            ]
        )

        if not pd.isna(row["mean_mae"]):
            lines.extend(
                [
                    f"- Mean MAE: "
                    f"{row['mean_mae']:.6f} "
                    f"(SD {row['sd_mae']:.6f})",
                    f"- Mean RMSE: "
                    f"{row['mean_rmse']:.6f} "
                    f"(SD {row['sd_rmse']:.6f})",
                ]
            )

        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "P4 compares predefined OOD, uncertainty, action-disagreement, "
            "and combined formulations.",
            "The analysis is descriptive across the five held-out policy "
            "seeds and does not declare a universal or uniquely optimal "
            "baseline.",
            "It does not establish causal faithfulness or universal OOD "
            "detection capability.",
            "",
            f"Repository commit: "
            f"{provenance['repository_commit']}",
        ]
    )

    (
        OUT_DIR / "P4_EVIDENCE_SUMMARY.md"
    ).write_text(
        "\n".join(lines) + "\n"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("P4 OOD / UNCERTAINTY BASELINE ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(
        "H=10, sigma>0 records:",
        len(df),
    )
    print()
    print(
        summary.to_string(index=False)
    )
    print()
    print("Outputs:")
    print(
        " ",
        PRED_DIR / "P4_heldout_scores.csv",
    )
    print(
        " ",
        STAT_DIR / "P4_seed_level_metrics.csv",
    )
    print(
        " ",
        STAT_DIR / "P4_model_summary.csv",
    )
    print(
        " ",
        DECILE_DIR
        / "P4_reliability_deciles_summary.csv",
    )
    print(
        " ",
        RISK_DIR
        / "P4_risk_coverage_nonzero_summary.csv",
    )
    print(
        " ",
        PROV_DIR / "P4_provenance.json",
    )
    print(
        " ",
        OUT_DIR / "P4_EVIDENCE_SUMMARY.md",
    )
    print()


if __name__ == "__main__":
    main()
