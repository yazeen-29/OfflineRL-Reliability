from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HOPPER_PATH = Path(
    "results/analysis/multiseed/"
    "iql_100k_multiseed_gaussian_analysis_v2.json"
)

HALFCHEETAH_PATH = Path(
    "results/analysis/halfcheetah/"
    "iql_100k_multiseed_gaussian_analysis.json"
)

EXPECTED_SEEDS = [0, 1, 2, 3, 4]


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing analysis file: {path}"
        )

    with path.open("r") as f:
        return json.load(f)


def summarize(name: str, data: dict) -> dict:
    seeds = [
        int(x)
        for x in data["policy_seeds"]
    ]

    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"{name}: expected policy seeds "
            f"{EXPECTED_SEEDS}, got {seeds}"
        )

    cross = data["cross_seed"]

    slopes = np.asarray(
        cross["seed_level_slope_values"],
        dtype=float,
    )

    positive = np.asarray(
        cross["seed_level_positive_fraction_values"],
        dtype=float,
    )

    if len(slopes) != 5:
        raise RuntimeError(
            f"{name}: expected 5 seed slopes, "
            f"got {len(slopes)}"
        )

    if len(positive) != 5:
        raise RuntimeError(
            f"{name}: expected 5 positive-fraction values, "
            f"got {len(positive)}"
        )

    if not np.isfinite(slopes).all():
        raise RuntimeError(
            f"{name}: non-finite slopes found."
        )

    if not np.isfinite(positive).all():
        raise RuntimeError(
            f"{name}: non-finite positive fractions found."
        )

    mean_slope = float(
        cross["mean_query_slope"]["mean"]
    )

    slope_sd = float(
        cross["mean_query_slope"]["std"]
    )

    slope_ci_low = float(
        cross["mean_query_slope"]["ci95_low"]
    )

    slope_ci_high = float(
        cross["mean_query_slope"]["ci95_high"]
    )

    positive_mean = float(
        cross["mean_positive_slope_fraction"]["mean"]
    )

    positive_ci_low = float(
        cross["mean_positive_slope_fraction"]["ci95_low"]
    )

    positive_ci_high = float(
        cross["mean_positive_slope_fraction"]["ci95_high"]
    )

    sign_flip = cross["exact_sign_flip_test"]

    return {
        "environment": name,
        "task": data["task"],
        "policy_seeds": seeds,
        "records_per_seed": int(
            data["records_per_seed"]
        ),
        "queries_per_seed": int(
            data["queries_per_seed"]
        ),
        "repeats_per_query_level": int(
            data["noise_repeats_per_query_level"]
        ),
        "noise_levels": data["noise_levels"],
        "seed_level_slopes": slopes.tolist(),
        "mean_slope": mean_slope,
        "slope_sd": slope_sd,
        "slope_ci95": {
            "low": slope_ci_low,
            "high": slope_ci_high,
        },
        "seed_positive_slope_fractions": (
            positive.tolist()
        ),
        "mean_positive_slope_fraction": (
            positive_mean
        ),
        "positive_slope_fraction_ci95": {
            "low": positive_ci_low,
            "high": positive_ci_high,
        },
        "exact_sign_flip_test": sign_flip,
        "all_seed_slopes_positive": bool(
            np.all(slopes > 0)
        ),
    }


def main() -> None:
    print("=" * 80)
    print("CROSS-ENVIRONMENT GAUSSIAN SYNTHESIS")
    print("=" * 80)

    hopper = summarize(
        "Hopper",
        load(HOPPER_PATH),
    )

    halfcheetah = summarize(
        "HalfCheetah",
        load(HALFCHEETAH_PATH),
    )

    environments = {
        "Hopper": hopper,
        "HalfCheetah": halfcheetah,
    }

    all_slopes = np.concatenate(
        [
            np.asarray(
                hopper["seed_level_slopes"],
                dtype=float,
            ),
            np.asarray(
                halfcheetah["seed_level_slopes"],
                dtype=float,
            ),
        ]
    )

    positive_seed_count = int(
        np.sum(all_slopes > 0)
    )

    total_seed_count = int(
        len(all_slopes)
    )

    output = {
        "analysis": (
            "Cross-environment synthesis of Gaussian "
            "observation-shift explanation reliability"
        ),
        "environments": [
            "Hopper",
            "HalfCheetah",
        ],
        "primary_question": (
            "Does the positive relationship between "
            "nearest-neighbor distributional distance "
            "and explanation-action disagreement "
            "replicate across offline RL environments?"
        ),
        "independent_replication_unit": (
            "independently trained IQL policy seed"
        ),
        "environment_summaries": environments,
        "replication_summary": {
            "total_policy_seeds": total_seed_count,
            "positive_seed_slopes": positive_seed_count,
            "positive_seed_slope_fraction": (
                float(
                    positive_seed_count
                    / total_seed_count
                )
            ),
            "all_policy_seed_slopes_positive": bool(
                positive_seed_count
                == total_seed_count
            ),
        },
        "interpretation_guardrail": (
            "Raw slope magnitudes are reported descriptively. "
            "They are not treated as directly standardized "
            "cross-environment effect sizes. The primary "
            "cross-environment conclusion concerns replication "
            "of the positive relationship and consistency "
            "across independently trained policy seeds."
        ),
        "statistical_unit_note": (
            "Within each environment, policy seeds are the "
            "independent replication units for cross-seed "
            "inference. Query-level observations estimate "
            "the seed-level effect and are not treated as "
            "independent policy replications."
        ),
    }

    output_path = Path(
        "results/analysis/cross_environment/"
        "iql_cross_environment_gaussian_analysis.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("Hopper")
    print(
        "  Mean slope:",
        hopper["mean_slope"],
    )
    print(
        "  95% CI:",
        (
            hopper["slope_ci95"]["low"],
            hopper["slope_ci95"]["high"],
        ),
    )
    print(
        "  Positive-slope fraction:",
        hopper["mean_positive_slope_fraction"],
    )
    print(
        "  One-sided p:",
        hopper["exact_sign_flip_test"][
            "one_sided_p"
        ],
    )
    print(
        "  Two-sided p:",
        hopper["exact_sign_flip_test"][
            "two_sided_p"
        ],
    )

    print()
    print("HalfCheetah")
    print(
        "  Mean slope:",
        halfcheetah["mean_slope"],
    )
    print(
        "  95% CI:",
        (
            halfcheetah["slope_ci95"]["low"],
            halfcheetah["slope_ci95"]["high"],
        ),
    )
    print(
        "  Positive-slope fraction:",
        halfcheetah[
            "mean_positive_slope_fraction"
        ],
    )
    print(
        "  One-sided p:",
        halfcheetah["exact_sign_flip_test"][
            "one_sided_p"
        ],
    )
    print(
        "  Two-sided p:",
        halfcheetah["exact_sign_flip_test"][
            "two_sided_p"
        ],
    )

    print()
    print(
        "All 10 policy-seed slopes positive:",
        positive_seed_count
        == total_seed_count,
    )

    print(
        "Combined positive-seed fraction:",
        f"{positive_seed_count / total_seed_count:.4f}",
    )

    print()
    print("Saved:", output_path)
    print("=" * 80)
    print("✅ CROSS-ENVIRONMENT ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
