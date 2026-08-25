from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import linregress, t


NOISE_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)

EXPECTED_QUERY_COUNT = 1000
EXPECTED_REPEAT_COUNT = 5
EXPECTED_RECORD_COUNT = (
    EXPECTED_QUERY_COUNT
    * len(NOISE_LEVELS)
    * EXPECTED_REPEAT_COUNT
)

POLICY_SEEDS = [0, 1, 2, 3, 4]


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def mean_ci(values: np.ndarray) -> dict:
    """
    Mean, sample SD, and two-sided 95% t-based CI.

    The CI is across the independent policy seeds when used
    at the cross-seed level.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot compute CI for empty values."
        )

    n = len(values)

    mean = float(
        np.mean(values)
    )

    if n > 1:
        sd = float(
            np.std(
                values,
                ddof=1,
            )
        )

        sem = sd / np.sqrt(n)

        critical = float(
            t.ppf(
                0.975,
                df=n - 1,
            )
        )

        margin = critical * sem

    else:
        sd = 0.0
        margin = 0.0

    return {
        "n": int(n),
        "mean": mean,
        "std": sd,
        "ci95_low": float(
            mean - margin
        ),
        "ci95_high": float(
            mean + margin
        ),
    }


def exact_sign_flip_test(
    values: np.ndarray,
) -> dict:
    """
    Exact one- and two-sided sign-flip permutation test
    at the independent policy-seed level.

    Null:
        mean effect = 0

    Alternative:
        mean effect > 0

    With five policy seeds there are only 2^5 = 32
    possible sign assignments, so the exact null distribution
    can be enumerated completely.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            "No finite values supplied."
        )

    observed_mean = float(
        np.mean(values)
    )

    null_means = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(values),
    ):
        signs = np.asarray(
            signs,
            dtype=float,
        )

        null_mean = float(
            np.mean(
                values * signs
            )
        )

        null_means.append(
            null_mean
        )

    null_means = np.asarray(
        null_means,
        dtype=float,
    )

    one_sided_p = float(
        np.mean(
            null_means >= (
                observed_mean - 1e-12
            )
        )
    )

    two_sided_p = float(
        np.mean(
            np.abs(null_means)
            >= (
                abs(observed_mean) - 1e-12
            )
        )
    )

    return {
        "n_seeds": int(
            len(values)
        ),
        "observed_mean": observed_mean,
        "one_sided_p": one_sided_p,
        "two_sided_p": two_sided_p,
        "num_exact_sign_assignments": int(
            2 ** len(values)
        ),
    }


def query_level_arrays(
    records: list[dict],
):
    """
    Convert raw repeated records into query x noise-level
    matrices by averaging exactly five perturbation repeats
    within each query/noise cell.
    """
    required_fields = {
        "query_id",
        "repeat_id",
        "noise_level",
        "nearest_neighbor_distance",
        "explanation_action_disagreement",
    }

    if not records:
        raise RuntimeError(
            "No records found."
        )

    missing = (
        required_fields
        - set(records[0].keys())
    )

    if missing:
        raise RuntimeError(
            f"Missing required record fields: {sorted(missing)}"
        )

    query_ids = sorted(
        {
            int(r["query_id"])
            for r in records
        }
    )

    noise_levels = sorted(
        {
            float(r["noise_level"])
            for r in records
        }
    )

    repeat_ids = sorted(
        {
            int(r["repeat_id"])
            for r in records
        }
    )

    if len(query_ids) != EXPECTED_QUERY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERY_COUNT} queries, "
            f"got {len(query_ids)}."
        )

    if not np.allclose(
        noise_levels,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            "Unexpected noise levels.\n"
            f"Expected: {NOISE_LEVELS.tolist()}\n"
            f"Observed: {noise_levels}"
        )

    if repeat_ids != list(
        range(EXPECTED_REPEAT_COUNT)
    ):
        raise RuntimeError(
            "Unexpected repeat IDs.\n"
            f"Expected: {list(range(EXPECTED_REPEAT_COUNT))}\n"
            f"Observed: {repeat_ids}"
        )

    if len(records) != EXPECTED_RECORD_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORD_COUNT} records, "
            f"got {len(records)}."
        )

    q_index = {
        q: i
        for i, q in enumerate(
            query_ids
        )
    }

    level_index = {
        level: i
        for i, level in enumerate(
            noise_levels
        )
    }

    nn = np.full(
        (
            len(query_ids),
            len(noise_levels),
        ),
        np.nan,
        dtype=float,
    )

    explanation = np.full(
        (
            len(query_ids),
            len(noise_levels),
        ),
        np.nan,
        dtype=float,
    )

    counts = np.zeros(
        (
            len(query_ids),
            len(noise_levels),
        ),
        dtype=int,
    )

    for record in records:
        q = int(
            record["query_id"]
        )

        level = float(
            record["noise_level"]
        )

        i = q_index[q]
        j = level_index[level]

        if np.isnan(
            nn[i, j]
        ):
            nn[i, j] = 0.0
            explanation[i, j] = 0.0

        nn[i, j] += float(
            record[
                "nearest_neighbor_distance"
            ]
        )

        explanation[i, j] += float(
            record[
                "explanation_action_disagreement"
            ]
        )

        counts[i, j] += 1

    if not np.all(
        counts == EXPECTED_REPEAT_COUNT
    ):
        bad_cells = np.argwhere(
            counts
            != EXPECTED_REPEAT_COUNT
        )

        raise RuntimeError(
            "Not every query/noise-level cell contains "
            f"exactly {EXPECTED_REPEAT_COUNT} repeats. "
            f"Bad cells: {bad_cells[:10].tolist()}"
        )

    valid = counts > 0

    nn[valid] /= counts[valid]
    explanation[valid] /= counts[valid]

    return (
        np.asarray(
            query_ids,
            dtype=int,
        ),
        np.asarray(
            noise_levels,
            dtype=float,
        ),
        nn,
        explanation,
    )


def per_query_slopes(
    nn: np.ndarray,
    explanation: np.ndarray,
) -> np.ndarray:
    """
    For each query, regress:

        explanation disagreement
        ~ nearest-neighbor distance

    across the seven Gaussian noise levels.
    """
    slopes = []

    for i in range(
        nn.shape[0]
    ):
        x = nn[i]
        y = explanation[i]

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if np.sum(valid) < 3:
            slopes.append(
                np.nan
            )
            continue

        fit = linregress(
            x[valid],
            y[valid],
        )

        slopes.append(
            float(
                fit.slope
            )
        )

    return np.asarray(
        slopes,
        dtype=float,
    )


def analyze_seed(
    seed: int,
    path: Path,
) -> dict:
    """
    Fully validate and analyze one policy seed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing seed {seed} result: {path}"
        )

    data = load_json(
        path
    )

    records = data.get(
        "records",
        [],
    )

    (
        query_ids,
        noise_levels,
        nn,
        explanation,
    ) = query_level_arrays(
        records
    )

    slopes = per_query_slopes(
        nn,
        explanation,
    )

    valid_slopes = slopes[
        np.isfinite(slopes)
    ]

    if len(valid_slopes) == 0:
        raise RuntimeError(
            f"Seed {seed}: no valid query-level slopes."
        )

    positive_fraction = float(
        np.mean(
            valid_slopes > 0
        )
    )

    seed_slope_summary = mean_ci(
        valid_slopes
    )

    clean_nn = float(
        np.mean(
            nn[:, 0]
        )
    )

    clean_explanation = float(
        np.mean(
            explanation[:, 0]
        )
    )

    noise_level_means = {}

    for j, level in enumerate(
        noise_levels
    ):
        noise_level_means[
            str(float(level))
        ] = {
            "nn_distance": float(
                np.mean(
                    nn[:, j]
                )
            ),
            "explanation_disagreement": float(
                np.mean(
                    explanation[:, j]
                )
            ),
        }

    return {
        "seed": int(seed),
        "source_file": str(path),
        "num_queries": int(
            len(query_ids)
        ),
        "num_levels": int(
            len(noise_levels)
        ),
        "num_repeats": int(
            EXPECTED_REPEAT_COUNT
        ),
        "record_count": int(
            len(records)
        ),
        "valid_query_slope_count": int(
            len(valid_slopes)
        ),
        "positive_slope_fraction": (
            positive_fraction
        ),
        "query_slope_summary": (
            seed_slope_summary
        ),
        "clean_nn_distance": (
            clean_nn
        ),
        "clean_explanation_disagreement": (
            clean_explanation
        ),
        "noise_level_means": (
            noise_level_means
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-seed analysis of the IQL Gaussian "
            "observation-shift explanation reliability study."
        )
    )

    parser.add_argument(
        "--seed0",
        default=(
            "results/shifts/"
            "iql_100k_gaussian_observation_noise.json"
        ),
    )

    parser.add_argument(
        "--multiseed_dir",
        default=(
            "results/shifts/multiseed"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/analysis/multiseed/"
            "iql_100k_multiseed_gaussian_analysis_v2.json"
        ),
    )

    args = parser.parse_args()

    source_files = {
        0: Path(
            args.seed0
        ),
        1: Path(
            args.multiseed_dir
        )
        / "iql_seed1_gaussian_observation_noise.json",
        2: Path(
            args.multiseed_dir
        )
        / "iql_seed2_gaussian_observation_noise.json",
        3: Path(
            args.multiseed_dir
        )
        / "iql_seed3_gaussian_observation_noise.json",
        4: Path(
            args.multiseed_dir
        )
        / "iql_seed4_gaussian_observation_noise.json",
    }

    print("=" * 80)
    print(
        "PAPER-GRADE CROSS-SEED IQL GAUSSIAN SHIFT ANALYSIS"
    )
    print("=" * 80)

    seed_results = {}

    for seed in POLICY_SEEDS:
        result = analyze_seed(
            seed,
            source_files[seed],
        )

        seed_results[
            str(seed)
        ] = result

        print()
        print(
            f"Seed {seed}"
        )
        print(
            "  Records:",
            result[
                "record_count"
            ],
        )
        print(
            "  Queries:",
            result[
                "num_queries"
            ],
        )
        print(
            "  Repeats:",
            result[
                "num_repeats"
            ],
        )
        print(
            "  Positive slope fraction:",
            result[
                "positive_slope_fraction"
            ],
        )
        print(
            "  Mean query slope:",
            result[
                "query_slope_summary"
            ][
                "mean"
            ],
        )
        print(
            "  Query-level CI95:",
            (
                result[
                    "query_slope_summary"
                ][
                    "ci95_low"
                ],
                result[
                    "query_slope_summary"
                ][
                    "ci95_high"
                ],
            ),
        )

    # ---------------------------------------------------------
    # Policy-seed aggregation
    # ---------------------------------------------------------

    seed_slopes = np.asarray(
        [
            seed_results[str(seed)][
                "query_slope_summary"
            ][
                "mean"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    seed_positive_fractions = np.asarray(
        [
            seed_results[str(seed)][
                "positive_slope_fraction"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    cross_seed_slope_summary = mean_ci(
        seed_slopes
    )

    cross_seed_positive_fraction_summary = (
        mean_ci(
            seed_positive_fractions
        )
    )

    exact_slope_test = (
        exact_sign_flip_test(
            seed_slopes
        )
    )

    # ---------------------------------------------------------
    # Cross-seed dose-response summaries
    # ---------------------------------------------------------

    cross_seed_noise_levels = {}

    for level in NOISE_LEVELS:
        level_key = str(
            float(level)
        )

        nn_values = np.asarray(
            [
                seed_results[str(seed)][
                    "noise_level_means"
                ][
                    level_key
                ][
                    "nn_distance"
                ]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        explanation_values = np.asarray(
            [
                seed_results[str(seed)][
                    "noise_level_means"
                ][
                    level_key
                ][
                    "explanation_disagreement"
                ]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        cross_seed_noise_levels[
            level_key
        ] = {
            "nn_distance": mean_ci(
                nn_values
            ),
            "explanation_disagreement": (
                mean_ci(
                    explanation_values
                )
            ),
            "seed_values": {
                "nn_distance": (
                    nn_values.tolist()
                ),
                "explanation_disagreement": (
                    explanation_values.tolist()
                ),
            },
        }

    # ---------------------------------------------------------
    # Reproducibility metadata
    # ---------------------------------------------------------

    analysis = {
        "experiment": (
            "IQL-100K-cross-seed-gaussian-shift-v2"
        ),
        "task": (
            "mujoco/hopper/medium-v0"
        ),
        "policy_seeds": (
            POLICY_SEEDS
        ),
        "independent_replication_unit": (
            "independently trained IQL policy seed"
        ),
        "queries_per_seed": (
            EXPECTED_QUERY_COUNT
        ),
        "noise_repeats_per_query_level": (
            EXPECTED_REPEAT_COUNT
        ),
        "records_per_seed": (
            EXPECTED_RECORD_COUNT
        ),
        "noise_levels": (
            NOISE_LEVELS.tolist()
        ),
        "statistical_unit_note": (
            "Each policy seed is treated as an independent "
            "replication. The 1,000 queries within a seed "
            "are used to estimate that seed's effect and "
            "are not treated as independent policy "
            "replications in the cross-seed inference."
        ),
        "per_seed": seed_results,
        "cross_seed": {
            "seed_level_slope_values": (
                seed_slopes.tolist()
            ),
            "mean_query_slope": (
                cross_seed_slope_summary
            ),
            "seed_level_positive_fraction_values": (
                seed_positive_fractions.tolist()
            ),
            "mean_positive_slope_fraction": (
                cross_seed_positive_fraction_summary
            ),
            "exact_sign_flip_test": (
                exact_slope_test
            ),
            "noise_level_results": (
                cross_seed_noise_levels
            ),
        },
        "interpretation_note": (
            "The slope summarizes the relationship between "
            "nearest-neighbor distance and explanation-action "
            "disagreement across the predefined Gaussian "
            "observation-shift levels. The cross-seed summaries "
            "describe replication across five independently "
            "trained policies."
        ),
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            analysis,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "CROSS-SEED PRIMARY RESULT"
    )
    print("=" * 80)

    print(
        "Policy-seed mean slope:",
        cross_seed_slope_summary[
            "mean"
        ],
    )

    print(
        "Policy-seed slope SD:",
        cross_seed_slope_summary[
            "std"
        ],
    )

    print(
        "Policy-seed slope 95% CI:",
        (
            cross_seed_slope_summary[
                "ci95_low"
            ],
            cross_seed_slope_summary[
                "ci95_high"
            ],
        ),
    )

    print(
        "Mean positive-slope fraction:",
        cross_seed_positive_fraction_summary[
            "mean"
        ],
    )

    print(
        "Positive-slope fraction 95% CI:",
        (
            cross_seed_positive_fraction_summary[
                "ci95_low"
            ],
            cross_seed_positive_fraction_summary[
                "ci95_high"
            ],
        ),
    )

    print()
    print(
        "EXACT POLICY-SEED SIGN-FLIP TEST"
    )

    print(
        "Observed mean slope:",
        exact_slope_test[
            "observed_mean"
        ],
    )

    print(
        "One-sided p:",
        exact_slope_test[
            "one_sided_p"
        ],
    )

    print(
        "Two-sided p:",
        exact_slope_test[
            "two_sided_p"
        ],
    )

    print(
        "Exact sign assignments:",
        exact_slope_test[
            "num_exact_sign_assignments"
        ],
    )

    print()
    print(
        "OUTPUT VALIDATION"
    )

    print(
        "Policy seeds:",
        POLICY_SEEDS,
    )

    print(
        "Queries/seed:",
        EXPECTED_QUERY_COUNT,
    )

    print(
        "Repeats/query/level:",
        EXPECTED_REPEAT_COUNT,
    )

    print(
        "Records/seed:",
        EXPECTED_RECORD_COUNT,
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()