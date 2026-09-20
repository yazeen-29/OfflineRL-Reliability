from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data_frozen" / "P1"
OUT_DIR = ROOT / "results" / "analysis" / "P3_feasibility"

OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10
N_STRATA = 10
MIN_GROUP_SIZE = 2
LOW_HIGH_FRACTION = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data() -> pd.DataFrame:
    rows = []

    for seed in POLICY_SEEDS:
        path = DATA_DIR / f"seed{seed}.json"

        with path.open() as f:
            payload = json.load(f)

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{path}: wrong policy_seed"
            )

        for record in payload["records"]:
            if (
                int(record["horizon"]) == HORIZON
                and float(record["sigma"]) > 0.0
            ):
                rows.append(
                    {
                        "policy_seed": int(record["policy_seed"]),
                        "state_id": int(record["state_id"]),
                        "sigma": float(record["sigma"]),
                        "support_distance": float(
                            record["support_distance"]
                        ),
                        "action_disagreement": float(
                            record["action_disagreement"]
                        ),
                    }
                )

    df = pd.DataFrame(rows)

    required = {
        "policy_seed",
        "state_id",
        "sigma",
        "support_distance",
        "action_disagreement",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    for col in [
        "policy_seed",
        "state_id",
        "sigma",
        "support_distance",
        "action_disagreement",
    ]:
        values = df[col].to_numpy(float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values in {col}"
            )

    return df


def assign_support_strata(group: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic approximately equal-frequency ranking by support
    distance.

    Ties are broken by state_id.
    """
    g = (
        group
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    n = len(g)

    g["support_stratum"] = (
        np.floor(
            np.arange(n) * N_STRATA / n
        )
        .astype(int)
        + 1
    )

    return g


def match_one_stratum(
    stratum: pd.DataFrame,
) -> dict:
    """
    Identify low/high action-disagreement quartiles and pair them by
    ordered support distance.

    No consequence field is used.
    """
    n = len(stratum)

    q = int(np.floor(
        LOW_HIGH_FRACTION * n
    ))

    if q < MIN_GROUP_SIZE:
        return {
            "feasible": False,
            "reason": "insufficient_quartile_group_size",
            "n_total": n,
            "n_low": q,
            "n_high": q,
            "n_pairs": 0,
            "support_abs_diff_mean": np.nan,
            "support_abs_diff_median": np.nan,
            "action_diff_mean": np.nan,
            "action_diff_median": np.nan,
        }

    ordered_action = (
        stratum
        .sort_values(
            ["action_disagreement", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    low = (
        ordered_action
        .iloc[:q]
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    high = (
        ordered_action
        .iloc[-q:]
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    n_pairs = min(len(low), len(high))

    support_diff = (
        high["support_distance"].to_numpy()
        - low["support_distance"].to_numpy()
    )

    action_diff = (
        high["action_disagreement"].to_numpy()
        - low["action_disagreement"].to_numpy()
    )

    return {
        "feasible": True,
        "reason": "",
        "n_total": n,
        "n_low": len(low),
        "n_high": len(high),
        "n_pairs": n_pairs,
        "support_abs_diff_mean": float(
            np.abs(support_diff[:n_pairs]).mean()
        ),
        "support_abs_diff_median": float(
            np.median(
                np.abs(support_diff[:n_pairs])
            )
        ),
        "action_diff_mean": float(
            action_diff[:n_pairs].mean()
        ),
        "action_diff_median": float(
            np.median(
                action_diff[:n_pairs]
            )
        ),
        "action_diff_min": float(
            action_diff[:n_pairs].min()
        ),
        "action_diff_max": float(
            action_diff[:n_pairs].max()
        ),
    }


def main() -> None:
    df = load_data()

    strata_rows = []

    for (seed, sigma), cell in df.groupby(
        ["policy_seed", "sigma"],
        sort=True,
    ):
        cell = assign_support_strata(cell)

        for stratum_id, stratum in cell.groupby(
            "support_stratum",
            sort=True,
        ):
            result = match_one_stratum(stratum)

            strata_rows.append(
                {
                    "policy_seed": int(seed),
                    "sigma": float(sigma),
                    "support_stratum": int(stratum_id),
                    **result,
                }
            )

    strata = pd.DataFrame(strata_rows)

    strata.to_csv(
        OUT_DIR / "P3_matching_strata_audit.csv",
        index=False,
    )

    summary_rows = []

    for seed, seed_df in strata.groupby(
        "policy_seed",
        sort=True,
    ):
        eligible = seed_df[
            seed_df["feasible"]
        ]

        summary_rows.append(
            {
                "policy_seed": int(seed),
                "sigma_cells": seed_df["sigma"].nunique(),
                "total_strata": len(seed_df),
                "feasible_strata": len(eligible),
                "infeasible_strata": int(
                    (~seed_df["feasible"]).sum()
                ),
                "total_pairs": int(
                    eligible["n_pairs"].sum()
                ),
                "mean_abs_support_distance_gap": float(
                    eligible[
                        "support_abs_diff_mean"
                    ].mean()
                ),
                "median_abs_support_distance_gap": float(
                    eligible[
                        "support_abs_diff_median"
                    ].median()
                ),
                "mean_action_disagreement_gap": float(
                    eligible["action_diff_mean"].mean()
                ),
                "median_action_disagreement_gap": float(
                    eligible["action_diff_median"].median()
                ),
                "min_action_disagreement_gap": float(
                    eligible["action_diff_min"].min()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUT_DIR / "P3_matching_feasibility_by_seed.csv",
        index=False,
    )

    overall = {
        "horizon": HORIZON,
        "sigma_condition": "sigma > 0",
        "policy_seeds": POLICY_SEEDS,
        "support_distance_strata": N_STRATA,
        "minimum_quartile_group_size": MIN_GROUP_SIZE,
        "low_high_fraction": LOW_HIGH_FRACTION,
        "source_records": int(len(df)),
        "records_per_policy_seed": {
            str(seed): int(
                (df["policy_seed"] == seed).sum()
            )
            for seed in POLICY_SEEDS
        },
        "total_strata": int(len(strata)),
        "feasible_strata": int(
            strata["feasible"].sum()
        ),
        "infeasible_strata": int(
            (~strata["feasible"]).sum()
        ),
        "total_pairs": int(
            strata.loc[
                strata["feasible"],
                "n_pairs",
            ].sum()
        ),
        "all_strata_feasible": bool(
            strata["feasible"].all()
        ),
        "all_action_gaps_positive": bool(
            (
                strata.loc[
                    strata["feasible"],
                    "action_diff_min",
                ]
                > 0
            ).all()
        ),
        "source_hashes": {
            str(seed): sha256_file(
                DATA_DIR / f"seed{seed}.json"
            )
            for seed in POLICY_SEEDS
        },
    }

    with (
        OUT_DIR / "P3_matching_feasibility_summary.json"
    ).open("w") as f:
        json.dump(
            overall,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("P3 MATCHING FEASIBILITY AUDIT COMPLETE")
    print("=" * 80)
    print()
    print(
        "H=10, sigma>0 records:",
        len(df),
    )
    print(
        "Total strata:",
        len(strata),
    )
    print(
        "Feasible strata:",
        int(strata["feasible"].sum()),
    )
    print(
        "Infeasible strata:",
        int((~strata["feasible"]).sum()),
    )
    print(
        "Total matched pairs:",
        int(
            strata.loc[
                strata["feasible"],
                "n_pairs",
            ].sum()
        ),
    )
    print()
    print(summary.to_string(index=False))
    print()
    print(
        "All strata feasible:",
        bool(strata["feasible"].all()),
    )
    print(
        "All action gaps positive:",
        bool(
            (
                strata.loc[
                    strata["feasible"],
                    "action_diff_min",
                ]
                > 0
            ).all()
        ),
    )
    print()


if __name__ == "__main__":
    main()
