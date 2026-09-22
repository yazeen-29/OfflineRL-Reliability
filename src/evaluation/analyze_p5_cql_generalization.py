#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, kendalltau, t


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "results" / "reliability" / "P5_CQL" / "raw"
OUT_DIR = ROOT / "results" / "analysis" / "P5_CQL"

SEEDS = [0, 1, 2, 3, 4]
PRIMARY_HORIZON = 10

ALL_SIGMAS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]
PRIMARY_SIGMAS = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30]

EXPECTED_STATES = 100
EXPECTED_RECORDS_PER_SEED = 2800
EXPECTED_PRIMARY_RECORDS_PER_SEED = 600


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def exact_sign_flip(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) != 5:
        raise ValueError(
            f"Expected exactly 5 independent policy-seed effects, "
            f"got {len(values)}"
        )

    observed = float(np.mean(values))

    null_means = []

    for signs in itertools.product([-1.0, 1.0], repeat=5):
        signs = np.asarray(signs, dtype=float)
        null_means.append(float(np.mean(values * signs)))

    null_means = np.asarray(null_means, dtype=float)

    one_sided = float(
        np.mean(null_means >= observed - 1e-12)
    )

    two_sided = float(
        np.mean(
            np.abs(null_means)
            >= abs(observed) - 1e-12
        )
    )

    return {
        "n_seeds": 5,
        "observed_mean": observed,
        "one_sided_p": one_sided,
        "two_sided_p": two_sided,
        "num_exact_sign_assignments": 32,
    }


def mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)

    n = len(values)

    if n == 0:
        raise ValueError("Cannot compute CI for empty values.")

    mean = float(np.mean(values))

    if n > 1:
        sd = float(np.std(values, ddof=1))
        sem = sd / np.sqrt(n)
        critical = float(t.ppf(0.975, df=n - 1))
        margin = critical * sem
    else:
        sd = 0.0
        margin = 0.0

    return {
        "n": int(n),
        "mean": mean,
        "std": sd,
        "ci95_low": float(mean - margin),
        "ci95_high": float(mean + margin),
    }


def load_seed(seed: int) -> pd.DataFrame:
    path = DATA_DIR / f"seed{seed}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())

    if data["algorithm"] != "CQL":
        raise ValueError(f"seed{seed}: algorithm is not CQL")

    if int(data["policy_seed"]) != seed:
        raise ValueError(f"seed{seed}: policy seed mismatch")

    records = data["records"]

    if len(records) != EXPECTED_RECORDS_PER_SEED:
        raise ValueError(
            f"seed{seed}: expected {EXPECTED_RECORDS_PER_SEED} records, "
            f"got {len(records)}"
        )

    rows = []

    required = [
        "policy_seed",
        "state_id",
        "sigma",
        "horizon",
        "action_disagreement",
        "support_distance",
        "absolute_consequence",
    ]

    for record in records:
        missing = [x for x in required if x not in record]

        if missing:
            raise ValueError(
                f"seed{seed}: missing fields {missing}"
            )

        rows.append(
            {
                "policy_seed": int(record["policy_seed"]),
                "state_id": int(record["state_id"]),
                "sigma": float(record["sigma"]),
                "horizon": int(record["horizon"]),
                "action_disagreement": float(
                    record["action_disagreement"]
                ),
                "support_distance": float(
                    record["support_distance"]
                ),
                "absolute_consequence": float(
                    record["absolute_consequence"]
                ),
            }
        )

    frame = pd.DataFrame(rows)

    if frame["state_id"].nunique() != EXPECTED_STATES:
        raise ValueError(
            f"seed{seed}: expected {EXPECTED_STATES} states, "
            f"got {frame['state_id'].nunique()}"
        )

    observed_sigmas = sorted(frame["sigma"].unique())

    if not np.allclose(observed_sigmas, ALL_SIGMAS):
        raise ValueError(
            f"seed{seed}: unexpected sigma levels: {observed_sigmas}"
        )

    if set(frame["horizon"].unique()) != {1, 5, 10, 20}:
        raise ValueError(
            f"seed{seed}: unexpected horizons"
        )

    # Every state × sigma × horizon cell must occur exactly once.
    cell_counts = (
        frame.groupby(
            ["state_id", "sigma", "horizon"],
            dropna=False,
        )
        .size()
    )

    if len(cell_counts) != EXPECTED_RECORDS_PER_SEED:
        raise ValueError(
            f"seed{seed}: missing or duplicate cells"
        )

    if not np.all(cell_counts.values == 1):
        raise ValueError(
            f"seed{seed}: non-unique cells"
        )

    numeric_columns = [
        "action_disagreement",
        "support_distance",
        "absolute_consequence",
    ]

    for column in numeric_columns:
        if not np.all(np.isfinite(frame[column].to_numpy())):
            raise ValueError(
                f"seed{seed}: non-finite values in {column}"
            )

    return frame


def primary_seed_analysis(
    seed: int,
    frame: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    subset = frame[
        (frame["horizon"] == PRIMARY_HORIZON)
        & (frame["sigma"] > 0.0)
    ].copy()

    if len(subset) != EXPECTED_PRIMARY_RECORDS_PER_SEED:
        raise ValueError(
            f"seed{seed}: expected "
            f"{EXPECTED_PRIMARY_RECORDS_PER_SEED} primary records, "
            f"got {len(subset)}"
        )

    state_slopes = []

    for state_id, group in subset.groupby("state_id"):
        group = group.sort_values("sigma")

        if len(group) != len(PRIMARY_SIGMAS):
            raise ValueError(
                f"seed{seed}, state {state_id}: expected "
                f"{len(PRIMARY_SIGMAS)} nonzero sigma observations"
            )

        x = group["action_disagreement"].to_numpy(dtype=float)
        y = group["absolute_consequence"].to_numpy(dtype=float)

        if np.allclose(x, x[0]):
            slope = np.nan
        else:
            fit = linregress(x, y)
            slope = float(fit.slope)

        state_slopes.append(
            {
                "policy_seed": seed,
                "state_id": int(state_id),
                "slope": slope,
            }
        )

    state_df = pd.DataFrame(state_slopes)

    valid = state_df[np.isfinite(state_df["slope"])].copy()

    if len(valid) == 0:
        raise ValueError(f"seed{seed}: no finite state slopes")

    seed_slope = float(valid["slope"].mean())

    result = {
        "seed": seed,
        "primary_records": len(subset),
        "states": int(subset["state_id"].nunique()),
        "finite_state_slopes": int(len(valid)),
        "mean_state_slope": seed_slope,
        "state_slope_sd": float(
            valid["slope"].std(ddof=1)
        )
        if len(valid) > 1
        else 0.0,
        "positive_state_slope_fraction": float(
            np.mean(valid["slope"].to_numpy() > 0)
        ),
    }

    return result, state_df


def secondary_seed_summaries(
    seed: int,
    frame: pd.DataFrame,
) -> dict:
    subset = frame[
        (frame["horizon"] == PRIMARY_HORIZON)
        & (frame["sigma"] > 0.0)
    ].copy()

    state_means = (
        subset.groupby("state_id", as_index=False)
        .agg(
            action_disagreement=(
                "action_disagreement",
                "mean",
            ),
            absolute_consequence=(
                "absolute_consequence",
                "mean",
            ),
            support_distance=(
                "support_distance",
                "first",
            ),
        )
    )

    def regression(y_name: str, x_name: str) -> dict:
        x = state_means[x_name].to_numpy(dtype=float)
        y = state_means[y_name].to_numpy(dtype=float)

        fit = linregress(x, y)

        return {
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "rvalue": float(fit.rvalue),
            "r_squared": float(fit.rvalue ** 2),
            "p_value": float(fit.pvalue),
        }

    dose_rows = []

    for sigma in PRIMARY_SIGMAS:
        cell = subset[np.isclose(subset["sigma"], sigma)]

        action_mean = float(
            cell["action_disagreement"].mean()
        )

        consequence_mean = float(
            cell["absolute_consequence"].mean()
        )

        dose_rows.append(
            {
                "seed": seed,
                "sigma": sigma,
                "action_disagreement_mean": action_mean,
                "absolute_consequence_mean": consequence_mean,
            }
        )

    dose_df = pd.DataFrame(dose_rows)

    spearman_rho, spearman_p = spearmanr(
        dose_df["action_disagreement_mean"],
        dose_df["absolute_consequence_mean"],
    )

    kendall_tau, kendall_p = kendalltau(
        dose_df["action_disagreement_mean"],
        dose_df["absolute_consequence_mean"],
    )

    return {
        "seed": seed,
        "support_to_action": regression(
            "action_disagreement",
            "support_distance",
        ),
        "support_to_consequence": regression(
            "absolute_consequence",
            "support_distance",
        ),
        "dose_response": {
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_tau),
            "kendall_p": float(kendall_p),
            "levels": dose_rows,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = {}
    primary_results = []
    state_slope_frames = []
    secondary_results = []

    for seed in SEEDS:
        frame = load_seed(seed)
        all_frames[seed] = frame

        primary, state_df = primary_seed_analysis(
            seed,
            frame,
        )

        primary_results.append(primary)

        state_slope_frames.append(state_df)

        secondary_results.append(
            secondary_seed_summaries(
                seed,
                frame,
            )
        )

        print(
            f"Seed {seed}: "
            f"mean state slope = "
            f"{primary['mean_state_slope']:.8f}, "
            f"positive state slope fraction = "
            f"{primary['positive_state_slope_fraction']:.6f}"
        )

    seed_slopes = np.asarray(
        [
            row["mean_state_slope"]
            for row in primary_results
        ],
        dtype=float,
    )

    primary_summary = mean_ci(seed_slopes)
    sign_flip = exact_sign_flip(seed_slopes)

    output = {
        "experiment": "P5_CQL_Hopper_final_consequence",
        "analysis": "P5_CQL_generalization_primary",
        "git_commit": current_commit(),
        "policy_seeds": SEEDS,
        "primary_horizon": PRIMARY_HORIZON,
        "primary_sigmas": PRIMARY_SIGMAS,
        "outcome": "absolute_consequence",
        "predictor": "action_disagreement",
        "independent_unit": "policy_seed",
        "state_level_slope_summary": primary_results,
        "cross_seed_primary": {
            **primary_summary,
            **sign_flip,
            "positive_seed_fraction": float(
                np.mean(seed_slopes > 0)
            ),
            "seed_level_slope_values": seed_slopes.tolist(),
        },
        "secondary": secondary_results,
    }

    json_path = OUT_DIR / "P5_CQL_PRIMARY_ANALYSIS.json"
    json_path.write_text(
        json.dumps(output, indent=2)
    )

    seed_csv = OUT_DIR / "P5_CQL_SEED_SUMMARY.csv"

    pd.DataFrame(primary_results).to_csv(
        seed_csv,
        index=False,
    )

    state_csv = OUT_DIR / "P5_CQL_STATE_SLOPES.csv"

    pd.concat(
        state_slope_frames,
        ignore_index=True,
    ).to_csv(
        state_csv,
        index=False,
    )

    secondary_json = OUT_DIR / "P5_CQL_SECONDARY_ANALYSIS.json"

    secondary_json.write_text(
        json.dumps(
            {
                "experiment": output["experiment"],
                "git_commit": current_commit(),
                "secondary": secondary_results,
            },
            indent=2,
        )
    )

    print("\n" + "=" * 80)
    print("P5 CQL PRIMARY GENERALIZATION ANALYSIS")
    print("=" * 80)

    print("\nSeed-level slopes:")
    for seed, slope in zip(SEEDS, seed_slopes):
        print(f"  seed {seed}: {slope:.10f}")

    print("\nCross-seed summary:")
    print(f"  mean slope: {primary_summary['mean']:.10f}")
    print(f"  SD:         {primary_summary['std']:.10f}")
    print(
        "  95% CI:     "
        f"[{primary_summary['ci95_low']:.10f}, "
        f"{primary_summary['ci95_high']:.10f}]"
    )

    print("\nExact sign-flip:")
    print(f"  one-sided p: {sign_flip['one_sided_p']:.5f}")
    print(f"  two-sided p: {sign_flip['two_sided_p']:.5f}")
    print(
        "  positive seeds: "
        f"{int(np.sum(seed_slopes > 0))}/5"
    )

    print("\nOutputs:")
    print(" ", json_path)
    print(" ", seed_csv)
    print(" ", state_csv)
    print(" ", secondary_json)


if __name__ == "__main__":
    main()
