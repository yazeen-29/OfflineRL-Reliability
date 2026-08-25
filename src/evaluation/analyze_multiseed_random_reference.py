from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import t, wilcoxon


# ============================================================
# LOCKED STUDY CONFIGURATION
# ============================================================

POLICY_SEEDS = [0, 1, 2, 3, 4]

NOISE_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)

EXPECTED_QUERIES = 1000
EXPECTED_REPEATS = 5
EXPECTED_RECORDS = (
    EXPECTED_QUERIES
    * EXPECTED_REPEATS
    * len(NOISE_LEVELS)
)


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


# ============================================================
# VALIDATION / AGGREGATION
# ============================================================

def build_query_level_matrix(
    records: list[dict],
    value_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate one experiment and average the five repeats
    within each query/noise-level cell.

    Returns:
        query_ids
        noise_levels
        query x noise matrix
    """

    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} records, "
            f"got {len(records)}."
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

    if len(query_ids) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries, "
            f"got {len(query_ids)}."
        )

    if not np.allclose(
        noise_levels,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            "Noise-level mismatch.\n"
            f"Expected: {NOISE_LEVELS.tolist()}\n"
            f"Observed: {noise_levels}"
        )

    if repeat_ids != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"Expected repeat IDs [0,1,2,3,4], "
            f"got {repeat_ids}."
        )

    q_index = {
        q: i
        for i, q in enumerate(query_ids)
    }

    level_index = {
        level: i
        for i, level in enumerate(noise_levels)
    }

    matrix = np.zeros(
        (
            len(query_ids),
            len(noise_levels),
        ),
        dtype=float,
    )

    counts = np.zeros_like(
        matrix,
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

        try:
            value = float(
                record[value_key]
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Missing required field "
                f"'{value_key}'."
            ) from exc

        matrix[i, j] += value
        counts[i, j] += 1

    if not np.all(
        counts == EXPECTED_REPEATS
    ):
        bad = np.argwhere(
            counts != EXPECTED_REPEATS
        )

        raise RuntimeError(
            "Not every query/noise-level cell "
            "contains exactly five repeats. "
            f"Example bad cells: {bad[:10].tolist()}"
        )

    matrix /= counts

    return (
        np.asarray(
            query_ids,
            dtype=int,
        ),
        np.asarray(
            noise_levels,
            dtype=float,
        ),
        matrix,
    )


# ============================================================
# POLICY-SEED SUMMARY
# ============================================================

def mean_ci(
    values: np.ndarray,
) -> dict:
    """
    Mean, sample SD, and two-sided 95% t CI.

    This function is used for independent policy seeds.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    if n == 0:
        raise ValueError(
            "Cannot summarize empty values."
        )

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

        sem = (
            sd
            / np.sqrt(n)
        )

        critical = float(
            t.ppf(
                0.975,
                df=n - 1,
            )
        )

        margin = (
            critical
            * sem
        )

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


# ============================================================
# EXACT FIVE-SEED SIGN-FLIP TEST
# ============================================================

def exact_sign_flip_test(
    values: np.ndarray,
) -> dict:
    """
    Exact sign-flip permutation test across the five
    independently trained policy seeds.

    Null:
        mean seed-level effect = 0

    Directional alternative:
        mean seed-level effect > 0
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot run sign-flip test on empty values."
        )

    observed = float(
        np.mean(values)
    )

    null_means = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(values),
    ):
        signs_array = np.asarray(
            signs,
            dtype=float,
        )

        null_means.append(
            float(
                np.mean(
                    values
                    * signs_array
                )
            )
        )

    null_means = np.asarray(
        null_means,
        dtype=float,
    )

    one_sided_p = float(
        np.mean(
            null_means
            >= observed - 1e-12
        )
    )

    two_sided_p = float(
        np.mean(
            np.abs(null_means)
            >= abs(observed) - 1e-12
        )
    )

    return {
        "observed_mean": observed,
        "one_sided_p": one_sided_p,
        "two_sided_p": two_sided_p,
        "n_seeds": int(
            len(values)
        ),
        "exact_assignments": int(
            2 ** len(values)
        ),
        "alternative": (
            "mean seed-level "
            "Nearest - Random slope difference > 0"
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final cross-seed comparison of "
            "nearest-neighbor and uniform-random "
            "reference explanations."
        )
    )

    parser.add_argument(
        "--nearest_dir",
        default="results/shifts",
    )

    parser.add_argument(
        "--random_dir",
        default=(
            "results/shifts/random_reference"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/analysis/"
            "iql_100k_multiseed_random_reference_control_v2.json"
        ),
    )

    args = parser.parse_args()

    nearest_dir = Path(
        args.nearest_dir
    )

    random_dir = Path(
        args.random_dir
    )

    print("=" * 80)
    print(
        "FINAL CROSS-SEED RANDOM-REFERENCE CONTROL ANALYSIS"
    )
    print("=" * 80)

    per_seed = {}

    # ========================================================
    # LOAD AND VALIDATE ALL FIVE POLICY SEEDS
    # ========================================================

    for seed in POLICY_SEEDS:

        if seed == 0:
            nearest_path = (
                nearest_dir
                / "iql_100k_gaussian_observation_noise.json"
            )
        else:
            nearest_path = (
                nearest_dir
                / "multiseed"
                / (
                    f"iql_seed{seed}_"
                    "gaussian_observation_noise.json"
                )
            )

        random_path = (
            random_dir
            / (
                f"iql_seed{seed}_"
                "random_reference.json"
            )
        )

        if not nearest_path.exists():
            raise FileNotFoundError(
                f"Missing nearest-neighbor result "
                f"for seed {seed}: {nearest_path}"
            )

        if not random_path.exists():
            raise FileNotFoundError(
                f"Missing uniform-random result "
                f"for seed {seed}: {random_path}"
            )

        print()
        print(
            f"Loading seed {seed}"
        )

        nearest_data = load_json(
            nearest_path
        )

        random_data = load_json(
            random_path
        )

        if nearest_data.get(
            "task"
        ) != random_data.get(
            "task"
        ):
            raise RuntimeError(
                f"Seed {seed}: task mismatch."
            )

        # ----------------------------------------------------
        # Primary nearest-neighbor result
        # ----------------------------------------------------

        (
            nearest_queries,
            nearest_levels,
            nearest_matrix,
        ) = build_query_level_matrix(
            nearest_data["records"],
            "explanation_action_disagreement",
        )

        # ----------------------------------------------------
        # Uniform-random result
        # ----------------------------------------------------

        (
            random_queries,
            random_levels,
            random_matrix,
        ) = build_query_level_matrix(
            random_data["records"],
            "random_reference_action_disagreement",
        )

        if not np.array_equal(
            nearest_queries,
            random_queries,
        ):
            raise RuntimeError(
                f"Seed {seed}: query IDs differ "
                "between NN and random-reference results."
            )

        if not np.allclose(
            nearest_levels,
            random_levels,
        ):
            raise RuntimeError(
                f"Seed {seed}: noise levels differ."
            )

        # ----------------------------------------------------
        # Per-noise-level comparison
        # ----------------------------------------------------

        level_results = []

        nearest_noise_means = []
        random_noise_means = []
        nearest_minus_random_means = []

        for j, level in enumerate(
            nearest_levels
        ):

            nearest = nearest_matrix[
                :,
                j,
            ]

            random = random_matrix[
                :,
                j,
            ]

            # THIS is the important directional quantity:
            #
            # positive =
            # NN degradation > random-reference degradation
            #
            nearest_minus_random = (
                nearest - random
            )

            # Supportive query-level paired test.
            #
            # This is NOT the main policy-level inference.
            if np.allclose(
                nearest_minus_random,
                0.0,
            ):
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = (
                    wilcoxon(
                        nearest_minus_random,
                        alternative="greater",
                        zero_method="wilcox",
                        method="auto",
                    )
                )

            nearest_mean = float(
                np.mean(nearest)
            )

            random_mean = float(
                np.mean(random)
            )

            difference_mean = float(
                np.mean(
                    nearest_minus_random
                )
            )

            nearest_noise_means.append(
                nearest_mean
            )

            random_noise_means.append(
                random_mean
            )

            nearest_minus_random_means.append(
                difference_mean
            )

            level_results.append(
                {
                    "noise_level": float(
                        level
                    ),
                    "nearest_neighbor_mean": (
                        nearest_mean
                    ),
                    "uniform_random_mean": (
                        random_mean
                    ),
                    "nearest_minus_random_mean": (
                        difference_mean
                    ),
                    "paired_query_test": {
                        "test": (
                            "Wilcoxon signed-rank"
                        ),
                        "alternative": (
                            "Nearest > Random"
                        ),
                        "statistic": float(
                            statistic
                        ),
                        "p": float(
                            p_value
                        ),
                    },
                }
            )

        # ----------------------------------------------------
        # Per-seed dose-response slopes
        # ----------------------------------------------------

        nearest_noise_means = np.asarray(
            nearest_noise_means,
            dtype=float,
        )

        random_noise_means = np.asarray(
            random_noise_means,
            dtype=float,
        )

        nearest_minus_random_means = (
            np.asarray(
                nearest_minus_random_means,
                dtype=float,
            )
        )

        nearest_slope = float(
            np.polyfit(
                nearest_levels,
                nearest_noise_means,
                1,
            )[0]
        )

        random_slope = float(
            np.polyfit(
                nearest_levels,
                random_noise_means,
                1,
            )[0]
        )

        slope_difference = float(
            nearest_slope
            - random_slope
        )

        per_seed[str(seed)] = {
            "seed": int(seed),
            "nearest_source": str(
                nearest_path
            ),
            "random_source": str(
                random_path
            ),
            "nearest_noise_means": (
                nearest_noise_means.tolist()
            ),
            "random_noise_means": (
                random_noise_means.tolist()
            ),
            "nearest_minus_random_means": (
                nearest_minus_random_means.tolist()
            ),
            "nearest_noise_slope": (
                nearest_slope
            ),
            "random_noise_slope": (
                random_slope
            ),
            "nearest_minus_random_noise_slope": (
                slope_difference
            ),
            "level_results": (
                level_results
            ),
        }

        print(
            "  NN noise slope:",
            nearest_slope,
        )

        print(
            "  Random noise slope:",
            random_slope,
        )

        print(
            "  NN - Random slope:",
            slope_difference,
        )

    # ========================================================
    # POLICY-SEED AGGREGATION
    # ========================================================

    nearest_slopes = np.asarray(
        [
            per_seed[str(seed)][
                "nearest_noise_slope"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    random_slopes = np.asarray(
        [
            per_seed[str(seed)][
                "random_noise_slope"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    nearest_minus_random_slopes = (
        nearest_slopes
        - random_slopes
    )

    nearest_summary = mean_ci(
        nearest_slopes
    )

    random_summary = mean_ci(
        random_slopes
    )

    slope_difference_summary = mean_ci(
        nearest_minus_random_slopes
    )

    exact_directional_test = (
        exact_sign_flip_test(
            nearest_minus_random_slopes
        )
    )

    # ========================================================
    # CROSS-SEED LEVEL SUMMARIES
    # ========================================================

    cross_seed_levels = []

    for j, level in enumerate(
        NOISE_LEVELS
    ):

        nearest_values = np.asarray(
            [
                per_seed[str(seed)][
                    "nearest_noise_means"
                ][j]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        random_values = np.asarray(
            [
                per_seed[str(seed)][
                    "random_noise_means"
                ][j]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        nearest_minus_random_values = (
            nearest_values
            - random_values
        )

        cross_seed_levels.append(
            {
                "noise_level": float(
                    level
                ),
                "nearest_neighbor": mean_ci(
                    nearest_values
                ),
                "uniform_random": mean_ci(
                    random_values
                ),
                "nearest_minus_random": mean_ci(
                    nearest_minus_random_values
                ),
                "seed_values": {
                    "nearest_neighbor": (
                        nearest_values.tolist()
                    ),
                    "uniform_random": (
                        random_values.tolist()
                    ),
                    "nearest_minus_random": (
                        nearest_minus_random_values.tolist()
                    ),
                },
            }
        )

    # ========================================================
    # FINAL JSON
    # ========================================================

    analysis = {
        "experiment": (
            "IQL-100K-cross-seed-"
            "uniform-random-reference-control-v2"
        ),
        "task": (
            "mujoco/hopper/medium-v0"
        ),
        "policy_seeds": (
            POLICY_SEEDS
        ),
        "queries_per_seed": (
            EXPECTED_QUERIES
        ),
        "repeats_per_query_level": (
            EXPECTED_REPEATS
        ),
        "records_per_seed": (
            EXPECTED_RECORDS
        ),
        "noise_levels": (
            NOISE_LEVELS.tolist()
        ),
        "statistical_unit": (
            "independently trained policy seed"
        ),
        "primary_directional_quantity": (
            "Nearest-neighbor dose-response slope "
            "minus uniform-random-reference dose-response slope"
        ),
        "per_seed": per_seed,
        "cross_seed": {
            "nearest_neighbor_noise_slope": (
                nearest_summary
            ),
            "uniform_random_noise_slope": (
                random_summary
            ),
            "nearest_minus_random_noise_slope": (
                slope_difference_summary
            ),
            "exact_sign_flip_test": (
                exact_directional_test
            ),
            "level_results": (
                cross_seed_levels
            ),
        },
        "interpretation_note": (
            "Positive nearest-minus-random slope difference "
            "means that explanation disagreement increases "
            "more strongly with Gaussian shift magnitude "
            "for nearest-neighbor references than for "
            "uniformly sampled references."
        ),
        "inference_note": (
            "The exact sign-flip test is performed across "
            "the five independently trained policy seeds. "
            "The query-level Wilcoxon tests are supportive "
            "paired analyses within each policy and are not "
            "treated as independent policy replications."
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

    # ========================================================
    # FINAL CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL CROSS-SEED CONTROL RESULT"
    )
    print("=" * 80)

    print(
        "NN mean noise slope:",
        nearest_summary["mean"],
    )

    print(
        "NN 95% CI:",
        (
            nearest_summary["ci95_low"],
            nearest_summary["ci95_high"],
        ),
    )

    print(
        "Random mean noise slope:",
        random_summary["mean"],
    )

    print(
        "Random 95% CI:",
        (
            random_summary["ci95_low"],
            random_summary["ci95_high"],
        ),
    )

    print(
        "NN - Random slope difference:",
        slope_difference_summary["mean"],
    )

    print(
        "Difference 95% CI:",
        (
            slope_difference_summary["ci95_low"],
            slope_difference_summary["ci95_high"],
        ),
    )

    print(
        "Exact one-sided sign-flip p:",
        exact_directional_test[
            "one_sided_p"
        ],
    )

    print(
        "Exact two-sided sign-flip p:",
        exact_directional_test[
            "two_sided_p"
        ],
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    print("=" * 80)


if __name__ == "__main__":
    main()