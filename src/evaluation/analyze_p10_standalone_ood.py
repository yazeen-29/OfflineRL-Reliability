from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "results" / "analysis" / "P10"

STATS_DIR = OUT_DIR / "statistics"
SUMMARY_DIR = OUT_DIR / "summaries"
PROV_DIR = OUT_DIR / "provenance"

POLICY_SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10
SIGMAS = [0.00, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]

SCORE_SPECS = {
    "action_disagreement": "Action disagreement",
    "support_distance": "Support distance",
    "twin_critic_disagreement": "Twin-critic disagreement",
}

CELL_SPECS = {
    "IQL_Hopper": ROOT / "data_frozen" / "P1",
    "IQL_HalfCheetah": (
        ROOT
        / "results"
        / "reliability"
        / "IQL_HalfCheetah"
        / "raw"
    ),
    "IQL_Walker2d": (
        ROOT
        / "results"
        / "reliability"
        / "IQL_Walker2d"
        / "raw"
    ),
    "CQL_Hopper": (
        ROOT
        / "results"
        / "reliability"
        / "P5_CQL"
        / "raw"
    ),
    "CQL_HalfCheetah": (
        ROOT
        / "results"
        / "reliability"
        / "P7_CQL_HalfCheetah"
        / "raw"
    ),
    "CQL_Walker2d": (
        ROOT
        / "results"
        / "reliability"
        / "P6_CQL_Walker2d"
        / "raw"
    ),
}


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


def load_h10_cell(
    cell_name: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    hashes: dict = {}

    required = {
        "policy_seed",
        "state_id",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
    }

    for seed in POLICY_SEEDS:
        path = data_dir / f"seed{seed}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"{cell_name}: missing {path}"
            )

        hashes[str(seed)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

        with path.open() as f:
            payload = json.load(f)

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{cell_name}: seed mismatch in {path}"
            )

        records = payload["records"]

        if len(records) != 2800:
            raise ValueError(
                f"{cell_name}: seed {seed}: "
                f"expected 2800 records, "
                f"got {len(records)}"
            )

        for record in records:
            if int(record["horizon"]) == HORIZON:
                rows.append(record)

    df = pd.DataFrame(rows)

    if len(df) != 3500:
        raise ValueError(
            f"{cell_name}: expected 3500 H=10 records, "
            f"got {len(df)}"
        )

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{cell_name}: missing columns {missing}"
        )

    for column in [
        "policy_seed",
        "state_id",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "twin_critic_disagreement",
    ]:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        ).to_numpy()

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{cell_name}: non-finite {column}"
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
                f"expected 700 H=10 records"
            )

    sigma_counts = (
        df.groupby(
            ["policy_seed", "sigma"]
        )
        .size()
    )

    for seed in POLICY_SEEDS:
        for sigma in SIGMAS:
            n = int(
                sigma_counts.get(
                    (seed, sigma),
                    0,
                )
            )

            if n != 100:
                raise ValueError(
                    f"{cell_name}: seed {seed}, "
                    f"sigma {sigma}: expected 100, got {n}"
                )

    return (
        df.sort_values(
            [
                "policy_seed",
                "state_id",
                "sigma",
            ]
        ).reset_index(drop=True),
        hashes,
    )


def safe_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    if len(np.unique(score)) < 2:
        return float("nan"), float("nan")

    return (
        float(roc_auc_score(y_true, score)),
        float(average_precision_score(y_true, score)),
    )


def main() -> None:
    for directory in [
        OUT_DIR,
        STATS_DIR,
        SUMMARY_DIR,
        PROV_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    seed_rows: list[dict] = []
    severity_rows: list[dict] = []
    source_hashes: dict = {}

    # --------------------------------------------------------
    # Load and evaluate every cell
    # --------------------------------------------------------

    for cell_name, data_dir in CELL_SPECS.items():

        print()
        print("=" * 90)
        print(f"CELL: {cell_name}")
        print(
            f"SOURCE: {data_dir.relative_to(ROOT)}"
        )
        print("=" * 90)

        df, hashes = load_h10_cell(
            cell_name,
            data_dir,
        )

        source_hashes[cell_name] = hashes

        # ----------------------------------------------------
        # Overall OOD: sigma > 0 versus sigma = 0
        # ----------------------------------------------------

        for score_column, score_label in SCORE_SPECS.items():

            for seed in POLICY_SEEDS:

                seed_df = df[
                    df["policy_seed"] == seed
                ].copy()

                y = (
                    seed_df["sigma"]
                    .to_numpy()
                    > 0
                ).astype(int)

                score = seed_df[
                    score_column
                ].to_numpy(
                    dtype=float
                )

                auroc, ap = safe_metrics(
                    y,
                    score,
                )

                prevalence = float(
                    np.mean(y)
                )

                seed_rows.append(
                    {
                        "cell": cell_name,
                        "score": score_column,
                        "score_label": score_label,
                        "policy_seed": int(seed),
                        "task": "any_ood",
                        "id_sigma": 0.0,
                        "ood_sigma": "all_nonzero",
                        "n_total": len(seed_df),
                        "n_id": int(
                            np.sum(y == 0)
                        ),
                        "n_ood": int(
                            np.sum(y == 1)
                        ),
                        "ood_prevalence": prevalence,
                        "auroc": auroc,
                        "average_precision": ap,
                    }
                )

                print(
                    f"{score_column:28s} "
                    f"seed={seed}: "
                    f"AUROC={auroc:.6f}, "
                    f"AP={ap:.6f}"
                )

            # ------------------------------------------------
            # Sigma-specific detection
            # ------------------------------------------------

            for sigma in SIGMAS[1:]:

                for seed in POLICY_SEEDS:

                    seed_df = df[
                        df["policy_seed"] == seed
                    ].copy()

                    task_df = seed_df[
                        (seed_df["sigma"] == 0)
                        | (
                            seed_df["sigma"]
                            == sigma
                        )
                    ].copy()

                    y = (
                        task_df["sigma"]
                        .to_numpy()
                        == sigma
                    ).astype(int)

                    score = task_df[
                        score_column
                    ].to_numpy(
                        dtype=float
                    )

                    auroc, ap = safe_metrics(
                        y,
                        score,
                    )

                    severity_rows.append(
                        {
                            "cell": cell_name,
                            "score": score_column,
                            "score_label": score_label,
                            "policy_seed": int(seed),
                            "sigma": float(sigma),
                            "n_total": len(task_df),
                            "n_id": int(
                                np.sum(y == 0)
                            ),
                            "n_ood": int(
                                np.sum(y == 1)
                            ),
                            "auroc": auroc,
                            "average_precision": ap,
                        }
                    )

    seed_metrics = pd.DataFrame(
        seed_rows
    ).sort_values(
        [
            "cell",
            "score",
            "policy_seed",
        ]
    )

    severity_metrics = pd.DataFrame(
        severity_rows
    ).sort_values(
        [
            "cell",
            "score",
            "sigma",
            "policy_seed",
        ]
    )

    seed_metrics.to_csv(
        STATS_DIR / "P10_any_ood_seed_metrics.csv",
        index=False,
    )

    severity_metrics.to_csv(
        STATS_DIR / "P10_sigma_specific_seed_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Cell summaries
    # --------------------------------------------------------

    any_ood_summary = (
        seed_metrics
        .groupby(
            [
                "cell",
                "score",
            ],
            as_index=False,
        )
        .agg(
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            mean_average_precision=(
                "average_precision",
                "mean",
            ),
            sd_average_precision=(
                "average_precision",
                "std",
            ),
            n_policy_seeds=(
                "policy_seed",
                "nunique",
            ),
        )
    )

    any_ood_summary.to_csv(
        SUMMARY_DIR / "P10_any_ood_cell_summary.csv",
        index=False,
    )

    severity_summary = (
        severity_metrics
        .groupby(
            [
                "cell",
                "score",
                "sigma",
            ],
            as_index=False,
        )
        .agg(
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            mean_average_precision=(
                "average_precision",
                "mean",
            ),
            sd_average_precision=(
                "average_precision",
                "std",
            ),
            n_policy_seeds=(
                "policy_seed",
                "nunique",
            ),
        )
    )

    severity_summary.to_csv(
        SUMMARY_DIR / "P10_sigma_specific_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Direction / consistency
    # --------------------------------------------------------

    direction_rows: list[dict] = []

    for (
        cell_name,
        score,
    ), group in seed_metrics.groupby(
        ["cell", "score"]
    ):
        direction_rows.append(
            {
                "cell": cell_name,
                "score": score,
                "n_policy_seeds": len(group),
                "n_auroc_above_0_5": int(
                    np.sum(
                        group["auroc"] > 0.5
                    )
                ),
                "n_auroc_below_0_5": int(
                    np.sum(
                        group["auroc"] < 0.5
                    )
                ),
                "n_auroc_equal_0_5": int(
                    np.sum(
                        group["auroc"] == 0.5
                    )
                ),
            }
        )

    direction = pd.DataFrame(
        direction_rows
    ).sort_values(
        ["cell", "score"]
    )

    direction.to_csv(
        STATS_DIR / "P10_direction_consistency.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance = {
        "experiment": "P10 standalone OOD detection",
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
        "sigma_levels": SIGMAS,
        "ood_definition": (
            "sigma > 0 versus sigma = 0"
        ),
        "scores": SCORE_SPECS,
        "cells": {
            name: str(
                path.relative_to(ROOT)
            )
            for name, path
            in CELL_SPECS.items()
        },
        "protocol_sha256": sha256_file(
            ROOT
            / "experiments"
            / "reliability"
            / "P10_STANDALONE_OOD_DETECTION_PROTOCOL.md"
        ),
        "source_hashes": source_hashes,
    }

    with (
        PROV_DIR
        / "P10_provenance.json"
    ).open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    lines = [
        "# P10 Standalone OOD Detection Evidence Summary",
        "",
        "- H=10 only",
        "- ID: sigma=0",
        "- OOD: sigma>0",
        "- Independent unit: policy seed",
        "- Cells: 6",
        "- Policy seeds per cell: 5",
        "",
        "## Any-shift OOD detection",
        "",
        "| Cell | Score | Mean AUROC | SD AUROC | Mean AP | SD AP | Seeds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in any_ood_summary.to_dict(
        "records"
    ):
        lines.append(
            "| "
            f"{row['cell']} | "
            f"{row['score']} | "
            f"{row['mean_auroc']:.6f} | "
            f"{row['sd_auroc']:.6f} | "
            f"{row['mean_average_precision']:.6f} | "
            f"{row['sd_average_precision']:.6f} | "
            f"{int(row['n_policy_seeds'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These results evaluate detection of the controlled "
            "observation-shift parameter used by the frozen experiments.",
            "They do not establish detection of arbitrary real-world "
            "OOD states or universal OOD detection capability.",
            "The OOD label is generated by sigma and does not use C10.",
        ]
    )

    (
        OUT_DIR
        / "P10_EVIDENCE_SUMMARY.md"
    ).write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print("=" * 90)
    print("P10 STANDALONE OOD DETECTION COMPLETE")
    print("=" * 90)
    print()
    print(any_ood_summary.to_string(index=False))
    print()
    print(
        "Outputs:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
