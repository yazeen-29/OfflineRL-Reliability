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
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "P8_DISTANCE_MATCHED"

PAIR_DIR = OUT / "matched_pairs"
STAT_DIR = OUT / "statistics"
SENS_DIR = OUT / "sensitivity"
PROV_DIR = OUT / "provenance"

for d in [PAIR_DIR, STAT_DIR, SENS_DIR, PROV_DIR]:
    d.mkdir(parents=True, exist_ok=True)


SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10
PRIMARY_STRATA = 10
SENSITIVITY_STRATA = [5, 20]
MIN_GROUP_SIZE = 2
LOW_HIGH_FRACTION = 0.25
SIGMA_VALUES = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3]

CELLS = {
    "IQL_Hopper": {
        "algorithm": "IQL",
        "environment": "Hopper-v5",
        "task": "mujoco/hopper/medium-v0",
        "directory": ROOT / "data_frozen" / "P1",
    },
    "IQL_HalfCheetah": {
        "algorithm": "IQL",
        "environment": "HalfCheetah-v5",
        "task": "mujoco/halfcheetah/medium-v0",
        "directory": ROOT / "results" / "reliability" / "IQL_HalfCheetah" / "raw",
    },
    "IQL_Walker2d": {
        "algorithm": "IQL",
        "environment": "Walker2d-v5",
        "task": "mujoco/walker2d/medium-v0",
        "directory": ROOT / "results" / "reliability" / "IQL_Walker2d" / "raw",
    },
    "CQL_Hopper": {
        "algorithm": "CQL",
        "environment": "Hopper-v5",
        "task": "mujoco/hopper/medium-v0",
        "directory": ROOT / "results" / "reliability" / "P5_CQL" / "raw",
    },
    "CQL_HalfCheetah": {
        "algorithm": "CQL",
        "environment": "HalfCheetah-v5",
        "task": "mujoco/halfcheetah/medium-v0",
        "directory": ROOT / "results" / "reliability" / "P7_CQL_HalfCheetah" / "raw",
    },
    "CQL_Walker2d": {
        "algorithm": "CQL",
        "environment": "Walker2d-v5",
        "task": "mujoco/walker2d/medium-v0",
        "directory": ROOT / "results" / "reliability" / "P6_CQL_Walker2d" / "raw",
    },
}


REQUIRED_COLUMNS = {
    "policy_seed",
    "state_id",
    "source_episode_id",
    "source_step",
    "sigma",
    "horizon",
    "support_distance",
    "action_disagreement",
    "absolute_consequence",
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


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean_obs = float(values.mean())
    count_ge = 0
    count_abs_ge = 0
    total = 2 ** n

    for mask in range(total):
        signs = np.ones(n)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1.0

        m = float(np.mean(values * signs))

        if m >= mean_obs - 1e-15:
            count_ge += 1

        if abs(m) >= abs(mean_obs) - 1e-15:
            count_abs_ge += 1

    return count_ge / total, count_abs_ge / total


def validate_cell_payload(
    cell_name: str,
    cell: dict,
) -> list[dict]:
    rows = []

    for seed in SEEDS:
        path = cell["directory"] / f"seed{seed}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"{cell_name}: missing source file {path}"
            )

        payload = json.loads(path.read_text())

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{cell_name} seed {seed}: wrong policy_seed"
            )

        if payload.get("algorithm") != cell["algorithm"]:
            raise ValueError(
                f"{cell_name} seed {seed}: wrong algorithm"
            )

        if payload.get("environment") != cell["environment"]:
            raise ValueError(
                f"{cell_name} seed {seed}: wrong environment"
            )

        if payload.get("task") != cell["task"]:
            raise ValueError(
                f"{cell_name} seed {seed}: wrong task"
            )

        if payload.get("status") != "final_collection":
            raise ValueError(
                f"{cell_name} seed {seed}: status is not final_collection"
            )

        if int(payload.get("n_states", -1)) != 100:
            raise ValueError(
                f"{cell_name} seed {seed}: expected 100 states"
            )

        records = payload["records"]

        if len(records) != 2800:
            raise ValueError(
                f"{cell_name} seed {seed}: expected 2800 records"
            )

        missing = REQUIRED_COLUMNS - set(records[0])
        if missing:
            raise ValueError(
                f"{cell_name} seed {seed}: missing {sorted(missing)}"
            )

        sigmas = sorted({float(r["sigma"]) for r in records})
        horizons = sorted({int(r["horizon"]) for r in records})

        if sigmas != [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3]:
            raise ValueError(
                f"{cell_name} seed {seed}: wrong sigma grid"
            )

        if horizons != [1, 5, 10, 20]:
            raise ValueError(
                f"{cell_name} seed {seed}: wrong horizon grid"
            )

        grid = {
            (
                int(r["state_id"]),
                float(r["sigma"]),
                int(r["horizon"]),
            )
            for r in records
        }

        if len(grid) != 2800:
            raise ValueError(
                f"{cell_name} seed {seed}: duplicate grid cells"
            )

        primary = [
            r
            for r in records
            if int(r["horizon"]) == HORIZON
            and float(r["sigma"]) > 0
        ]

        if len(primary) != 600:
            raise ValueError(
                f"{cell_name} seed {seed}: expected 600 primary records"
            )

        for col in REQUIRED_COLUMNS:
            values = np.asarray(
                [r[col] for r in primary],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{cell_name} seed {seed}: non-finite {col}"
                )

        for r in primary:
            row = {k: r[k] for k in REQUIRED_COLUMNS}
            row["cell"] = cell_name
            row["algorithm"] = cell["algorithm"]
            row["environment"] = cell["environment"]
            rows.append(row)

    return rows


def assign_support_strata(
    cell: pd.DataFrame,
    n_strata: int,
) -> pd.DataFrame:
    g = (
        cell.sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    n = len(g)

    g["support_stratum"] = (
        np.floor(np.arange(n) * n_strata / n).astype(int) + 1
    )

    return g


def match_stratum(stratum: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = len(stratum)

    q = int(math.floor(LOW_HIGH_FRACTION * n))

    if q < MIN_GROUP_SIZE:
        return (
            pd.DataFrame(),
            {
                "feasible": False,
                "reason": "insufficient_quartile_group_size",
                "n_total": n,
                "n_low": q,
                "n_high": q,
                "n_pairs": 0,
            },
        )

    ordered = (
        stratum.sort_values(
            ["action_disagreement", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    low = (
        ordered.iloc[:q]
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    high = (
        ordered.iloc[-q:]
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    n_pairs = min(len(low), len(high))

    rows = []

    for pair_index in range(n_pairs):
        lo = low.iloc[pair_index]
        hi = high.iloc[pair_index]

        rows.append(
            {
                "cell": lo["cell"],
                "algorithm": lo["algorithm"],
                "environment": lo["environment"],
                "policy_seed": int(lo["policy_seed"]),
                "sigma": float(lo["sigma"]),
                "support_stratum": int(lo["support_stratum"]),
                "low_state_id": int(lo["state_id"]),
                "high_state_id": int(hi["state_id"]),
                "low_source_episode_id": int(lo["source_episode_id"]),
                "high_source_episode_id": int(hi["source_episode_id"]),
                "low_source_step": int(lo["source_step"]),
                "high_source_step": int(hi["source_step"]),
                "low_support_distance": float(lo["support_distance"]),
                "high_support_distance": float(hi["support_distance"]),
                "support_distance_abs_diff": abs(
                    float(hi["support_distance"])
                    - float(lo["support_distance"])
                ),
                "low_action_disagreement": float(
                    lo["action_disagreement"]
                ),
                "high_action_disagreement": float(
                    hi["action_disagreement"]
                ),
                "action_disagreement_diff": (
                    float(hi["action_disagreement"])
                    - float(lo["action_disagreement"])
                ),
                "low_C10": float(lo["absolute_consequence"]),
                "high_C10": float(hi["absolute_consequence"]),
                "pair_delta_C10": (
                    float(hi["absolute_consequence"])
                    - float(lo["absolute_consequence"])
                ),
                "pair_index": pair_index,
            }
        )

    pairs = pd.DataFrame(rows)

    audit = {
        "feasible": True,
        "reason": "",
        "n_total": n,
        "n_low": len(low),
        "n_high": len(high),
        "n_pairs": n_pairs,
        "support_abs_diff_mean": float(
            np.abs(
                high["support_distance"].to_numpy()[:n_pairs]
                - low["support_distance"].to_numpy()[:n_pairs]
            ).mean()
        ),
        "support_abs_diff_median": float(
            np.median(
                np.abs(
                    high["support_distance"].to_numpy()[:n_pairs]
                    - low["support_distance"].to_numpy()[:n_pairs]
                )
            )
        ),
        "action_diff_mean": float(
            (
                high["action_disagreement"].to_numpy()[:n_pairs]
                - low["action_disagreement"].to_numpy()[:n_pairs]
            ).mean()
        ),
        "action_diff_median": float(
            np.median(
                high["action_disagreement"].to_numpy()[:n_pairs]
                - low["action_disagreement"].to_numpy()[:n_pairs]
            )
        ),
        "action_diff_min": float(
            (
                high["action_disagreement"].to_numpy()[:n_pairs]
                - low["action_disagreement"].to_numpy()[:n_pairs]
            ).min()
        ),
        "action_diff_max": float(
            (
                high["action_disagreement"].to_numpy()[:n_pairs]
                - low["action_disagreement"].to_numpy()[:n_pairs]
            ).max()
        ),
    }

    return pairs, audit


def run_matching(
    df: pd.DataFrame,
    n_strata: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames = []
    audit_rows = []

    for (
        cell_name,
        seed,
        sigma,
    ), group in df.groupby(
        ["cell", "policy_seed", "sigma"],
        sort=True,
    ):
        grouped = assign_support_strata(group, n_strata)

        for stratum_id, stratum in grouped.groupby(
            "support_stratum",
            sort=True,
        ):
            pairs, audit = match_stratum(stratum)

            audit_rows.append(
                {
                    "cell": cell_name,
                    "policy_seed": int(seed),
                    "sigma": float(sigma),
                    "support_stratum": int(stratum_id),
                    **audit,
                }
            )

            if not pairs.empty:
                pair_frames.append(pairs)

    pairs_df = (
        pd.concat(pair_frames, ignore_index=True)
        if pair_frames
        else pd.DataFrame()
    )

    audit_df = pd.DataFrame(audit_rows)

    return pairs_df, audit_df


def summarize_pairs(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []

    for (
        cell_name,
        seed,
    ), group in pairs.groupby(
        ["cell", "policy_seed"],
        sort=True,
    ):
        x = group["pair_delta_C10"].to_numpy(float)

        seed_rows.append(
            {
                "cell": cell_name,
                "policy_seed": int(seed),
                "n_pairs": len(group),
                "mean_pair_delta_C10": float(x.mean()),
                "median_pair_delta_C10": float(np.median(x)),
                "mean_support_abs_diff": float(
                    group["support_distance_abs_diff"].mean()
                ),
                "median_support_abs_diff": float(
                    group["support_distance_abs_diff"].median()
                ),
                "mean_action_disagreement_diff": float(
                    group["action_disagreement_diff"].mean()
                ),
                "median_action_disagreement_diff": float(
                    group["action_disagreement_diff"].median()
                ),
                "min_action_disagreement_diff": float(
                    group["action_disagreement_diff"].min()
                ),
                "max_action_disagreement_diff": float(
                    group["action_disagreement_diff"].max()
                ),
            }
        )

    seed_df = pd.DataFrame(seed_rows)

    cell_rows = []

    for cell_name, group in seed_df.groupby(
        "cell",
        sort=True,
    ):
        x = group["mean_pair_delta_C10"].to_numpy(float)
        mean_effect = float(x.mean())
        sd = float(x.std(ddof=1))
        n = len(x)

        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        half_width = tcrit * sd / math.sqrt(n)
        one_sided, two_sided = exact_sign_flip(x)

        cell_rows.append(
            {
                "cell": cell_name,
                "algorithm": group["cell"].iloc[0].split("_")[0],
                "n_policy_seeds": n,
                "mean_seed_effect": mean_effect,
                "sd_seed_effect": sd,
                "ci95_t_low": mean_effect - half_width,
                "ci95_t_high": mean_effect + half_width,
                "exact_sign_flip_one_sided_p": one_sided,
                "exact_sign_flip_two_sided_p": two_sided,
                "n_positive_seed_effects": int((x > 0).sum()),
                "n_nonnegative_seed_effects": int((x >= 0).sum()),
                "minimum_seed_effect": float(x.min()),
                "maximum_seed_effect": float(x.max()),
                "total_pairs": int(group["n_pairs"].sum()),
            }
        )

    cell_df = pd.DataFrame(cell_rows)

    return seed_df, cell_df


def run_sensitivity(
    df: pd.DataFrame,
    n_strata: int,
    name: str,
) -> dict:
    pairs, audit = run_matching(df, n_strata)

    audit.to_csv(
        SENS_DIR / f"P8_{name}_stratum_audit.csv",
        index=False,
    )

    feasible = bool(audit["feasible"].all()) if len(audit) else False

    result = {
        "n_strata": n_strata,
        "feasible": feasible,
        "n_total_strata": int(len(audit)),
        "n_feasible_strata": int(audit["feasible"].sum())
        if len(audit)
        else 0,
    }

    if feasible and not pairs.empty:
        seed_df, cell_df = summarize_pairs(pairs)

        pairs.to_csv(
            SENS_DIR / f"P8_{name}_matched_pairs.csv",
            index=False,
        )
        seed_df.to_csv(
            SENS_DIR / f"P8_{name}_seed_effects.csv",
            index=False,
        )
        cell_df.to_csv(
            SENS_DIR / f"P8_{name}_cell_summary.csv",
            index=False,
        )

        result["total_pairs"] = int(len(pairs))
        result["cell_summary"] = cell_df.to_dict("records")
    else:
        result["reason"] = "minimum group size requirement not met"

    return result


def main() -> None:
    all_rows = []
    source_hashes = {}
    source_metadata = {}

    print("=" * 80)
    print("P8 DISTANCE-MATCHED CONTROL GENERALIZATION")
    print("=" * 80)

    for cell_name, cell in CELLS.items():
        print(f"\nLoading {cell_name} ...")

        cell_rows = validate_cell_payload(cell_name, cell)
        all_rows.extend(cell_rows)

        source_hashes[cell_name] = {}
        source_metadata[cell_name] = {}

        for seed in SEEDS:
            path = cell["directory"] / f"seed{seed}.json"
            source_hashes[cell_name][str(seed)] = sha256_file(path)

            payload = json.loads(path.read_text())
            source_metadata[cell_name][str(seed)] = {
                "task": payload.get("task"),
                "environment": payload.get("environment"),
                "algorithm": payload.get("algorithm"),
                "status": payload.get("status"),
                "n_states": payload.get("n_states"),
                "records": len(payload.get("records", [])),
                "git_commit_in_payload": sorted(
                    {
                        r.get("git_commit")
                        for r in payload["records"]
                    }
                ),
            }

    df = pd.DataFrame(all_rows)

    expected_total = 6 * 5 * 600
    if len(df) != expected_total:
        raise ValueError(
            f"Expected {expected_total} primary records, got {len(df)}"
        )

    print(f"\nPrimary records loaded: {len(df)}")

    pairs, audit = run_matching(
        df,
        PRIMARY_STRATA,
    )

    expected_pairs = 6 * 5 * 120

    if len(pairs) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} primary pairs, got {len(pairs)}"
        )

    if not audit["feasible"].all():
        raise ValueError("Primary P8 matching contains infeasible strata.")

    if not (pairs["action_disagreement_diff"] > 0).all():
        raise ValueError(
            "Primary P8 matching produced non-positive "
            "action-disagreement differences."
        )

    seed_df, cell_df = summarize_pairs(pairs)

    pairs.to_csv(
        PAIR_DIR / "P8_PRIMARY_MATCHED_PAIRS.csv",
        index=False,
    )

    audit.to_csv(
        STAT_DIR / "P8_PRIMARY_STRATUM_AUDIT.csv",
        index=False,
    )

    seed_df.to_csv(
        STAT_DIR / "P8_PRIMARY_SEED_EFFECTS.csv",
        index=False,
    )

    cell_df.to_csv(
        STAT_DIR / "P8_PRIMARY_CELL_SUMMARY.csv",
        index=False,
    )

    with (
        STAT_DIR / "P8_PRIMARY_CELL_SUMMARY.json"
    ).open("w") as f:
        json.dump(
            cell_df.to_dict("records"),
            f,
            indent=2,
        )

    sensitivity = {}

    sensitivity["5_strata"] = run_sensitivity(
        df,
        5,
        "5STRATA",
    )

    sensitivity["20_strata"] = run_sensitivity(
        df,
        20,
        "20STRATA",
    )

    protocol = (
        ROOT
        / "experiments"
        / "reliability"
        / "P8_DISTANCE_MATCHED_CONTROL_GENERALIZATION_PROTOCOL.md"
    )

    provenance = {
        "experiment": "P8 distance-matched control generalization",
        "repository_commit": git_command("rev-parse", "HEAD"),
        "repository_status": git_command("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": stats.__version__
        if hasattr(stats, "__version__")
        else __import__("scipy").__version__,
        "protocol_hash": sha256_file(protocol),
        "primary": {
            "horizon": HORIZON,
            "sigma_condition": "sigma > 0",
            "support_distance_strata": PRIMARY_STRATA,
            "low_high_fraction": LOW_HIGH_FRACTION,
            "minimum_group_size": MIN_GROUP_SIZE,
            "cells": list(CELLS),
            "primary_records": int(len(df)),
            "primary_pairs": int(len(pairs)),
        },
        "source_paths": {
            name: str(cell["directory"])
            for name, cell in CELLS.items()
        },
        "source_hashes": source_hashes,
        "source_metadata": source_metadata,
        "sensitivity": sensitivity,
    }

    with (
        PROV_DIR / "P8_PROVENANCE.json"
    ).open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    lines = [
        "# P8 Distance-Matched Control Generalization Evidence Summary",
        "",
        "## Primary design",
        "",
        "- Six algorithm/environment cells",
        "- H=10",
        "- Nonzero sigma only",
        "- Ten support-distance strata",
        "- Lowest/highest action-disagreement quartiles",
        "- Five policy seeds per cell",
        f"- Primary matched pairs: {len(pairs)}",
        "",
        "## Primary cell results",
        "",
    ]

    for row in cell_df.to_dict("records"):
        lines.extend(
            [
                f"### {row['cell']}",
                f"- Mean seed-level matched ΔC10: "
                f"{row['mean_seed_effect']:.8f}",
                f"- SD across seeds: "
                f"{row['sd_seed_effect']:.8f}",
                f"- 95% t-based CI: "
                f"[{row['ci95_t_low']:.8f}, "
                f"{row['ci95_t_high']:.8f}]",
                f"- Exact one-sided sign-flip p: "
                f"{row['exact_sign_flip_one_sided_p']:.5f}",
                f"- Exact two-sided sign-flip p: "
                f"{row['exact_sign_flip_two_sided_p']:.5f}",
                f"- Positive seed effects: "
                f"{row['n_positive_seed_effects']}/"
                f"{row['n_policy_seeds']}",
                f"- Matched pairs: {row['total_pairs']}",
                "",
            ]
        )

    all_positive = int(
        (
            cell_df["n_positive_seed_effects"]
            == cell_df["n_policy_seeds"]
        ).sum()
    )

    lines.extend(
        [
            "## Cross-cell synthesis",
            "",
            f"- Cells with positive effects in all five seeds: "
            f"{all_positive}/6",
            "- Cross-cell synthesis is descriptive; no pooled "
            "pair-level inference is performed.",
            "",
            "## Interpretation boundary",
            "",
            "A positive cell-level matched effect supports an association "
            "between higher action disagreement and larger downstream "
            "consequence after matching on support distance and sigma.",
            "It does not establish causality or complete control of all "
            "possible confounding.",
            "Cross-cell replication does not imply directly comparable "
            "effect magnitudes across environments or algorithms.",
            "",
            "## Sensitivity",
            "",
            "The 5-stratum and 20-stratum specifications are reported "
            "separately from the primary 10-stratum result.",
        ]
    )

    (OUT / "P8_EVIDENCE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n"
    )

    print("\n" + "=" * 80)
    print("P8 ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Primary records: {len(df)}")
    print(f"Primary matched pairs: {len(pairs)}")
    print("\nCell-level primary effects:")
    print(
        cell_df[
            [
                "cell",
                "mean_seed_effect",
                "ci95_t_low",
                "ci95_t_high",
                "exact_sign_flip_one_sided_p",
                "n_positive_seed_effects",
                "total_pairs",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
