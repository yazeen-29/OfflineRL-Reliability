from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import t


POLICY_SEEDS = [0, 1, 2, 3, 4]

SIGMA_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)

EXPECTED_QUERIES = 1000
EXPECTED_RECORDS = (
    EXPECTED_QUERIES
    * len(SIGMA_LEVELS)
)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def validate_and_aggregate(
    data: dict,
):
    records = data["records"]

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

    levels = sorted(
        {
            float(r["gaussian_equivalent_sigma"])
            for r in records
        }
    )

    if len(query_ids) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries, "
            f"got {len(query_ids)}."
        )

    if not np.allclose(
        levels,
        SIGMA_LEVELS,
    ):
        raise RuntimeError(
            f"Unexpected sigma levels: {levels}"
        )

    q_index = {
        q: i
        for i, q in enumerate(query_ids)
    }

    level_index = {
        level: i
        for i, level in enumerate(levels)
    }

    metrics = {
        "nearest_neighbor_distance": np.zeros(
            (
                len(query_ids),
                len(levels),
            ),
            dtype=float,
        ),
        "policy_action_change": np.zeros(
            (
                len(query_ids),
                len(levels),
            ),
            dtype=float,
        ),
        "explanation_action_disagreement": np.zeros(
            (
                len(query_ids),
                len(levels),
            ),
            dtype=float,
        ),
        "standardized_displacement": np.zeros(
            (
                len(query_ids),
                len(levels),
            ),
            dtype=float,
        ),
    }

    counts = np.zeros(
        (
            len(query_ids),
            len(levels),
        ),
        dtype=int,
    )

    for record in records:
        q = int(
            record["query_id"]
        )

        level = float(
            record[
                "gaussian_equivalent_sigma"
            ]
        )

        i = q_index[q]
        j = level_index[level]

        metrics[
            "nearest_neighbor_distance"
        ][i, j] += float(
            record[
                "nearest_neighbor_distance"
            ]
        )

        metrics[
            "policy_action_change"
        ][i, j] += float(
            record[
                "policy_action_change"
            ]
        )

        metrics[
            "explanation_action_disagreement"
        ][i, j] += float(
            record[
                "explanation_action_disagreement"
            ]
        )

        metrics[
            "standardized_displacement"
        ][i, j] += float(
            record[
                "actual_standardized_displacement"
            ]
        )

        counts[i, j] += 1

    if not np.all(
        counts == 1
    ):
        raise RuntimeError(
            "Structured experiment should have "
            "exactly one deterministic observation "
            "per query/level cell."
        )

    return (
        np.asarray(
            query_ids,
            dtype=int,
        ),
        np.asarray(
            levels,
            dtype=float,
        ),
        metrics,
    )


def mean_ci(
    values: np.ndarray,
):
    values = np.asarray(
        values,
        dtype=float,
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


def exact_sign_flip_test(
    values: np.ndarray,
):
    """
    Exact directional sign-flip test across
    independently trained policy seeds.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    observed = float(
        np.mean(values)
    )

    null_means = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(values),
    ):
        signs = np.asarray(
            signs,
            dtype=float,
        )

        null_means.append(
            float(
                np.mean(
                    values * signs
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
            "mean policy-seed slope > 0"
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cross-seed analysis of the structured "
            "directional observation-shift study."
        )
    )

    parser.add_argument(
        "--input_dir",
        default="results/shifts/structured",
    )

    parser.add_argument(
        "--output",
        default=(
            "results/analysis/"
            "iql_100k_multiseed_structured_analysis.json"
        ),
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    print("=" * 80)
    print(
        "CROSS-SEED STRUCTURED SHIFT ANALYSIS"
    )
    print("=" * 80)

    per_seed = {}

    # ========================================================
    # LOAD EACH POLICY SEED
    # ========================================================

    for seed in POLICY_SEEDS:

        path = (
            input_dir
            / (
                f"iql_seed{seed}_"
                "structured_directional.json"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing structured result for "
                f"seed {seed}: {path}"
            )

        data = load_json(
            path
        )

        (
            queries,
            levels,
            metrics,
        ) = validate_and_aggregate(
            data
        )

        # ----------------------------------------------------
        # Per-seed summaries
        # ----------------------------------------------------

        explanation_means = np.mean(
            metrics[
                "explanation_action_disagreement"
            ],
            axis=0,
        )

        nn_means = np.mean(
            metrics[
                "nearest_neighbor_distance"
            ],
            axis=0,
        )

        action_means = np.mean(
            metrics[
                "policy_action_change"
            ],
            axis=0,
        )

        displacement_means = np.mean(
            metrics[
                "standardized_displacement"
            ],
            axis=0,
        )

        # ----------------------------------------------------
        # Dose-response slope
        # ----------------------------------------------------

        explanation_slope = float(
            np.polyfit(
                levels,
                explanation_means,
                1,
            )[0]
        )

        distance_slope = float(
            np.polyfit(
                levels,
                nn_means,
                1,
            )[0]
        )

        action_slope = float(
            np.polyfit(
                levels,
                action_means,
                1,
            )[0]
        )

        # ----------------------------------------------------
        # Positivity across adjacent levels
        # ----------------------------------------------------

        explanation_differences = np.diff(
            explanation_means
        )

        positive_fraction = float(
            np.mean(
                explanation_differences > 0
            )
        )

        per_seed[str(seed)] = {
            "seed": int(seed),
            "num_queries": int(
                len(queries)
            ),
            "sigma_levels": levels.tolist(),
            "explanation_means": (
                explanation_means.tolist()
            ),
            "nearest_neighbor_means": (
                nn_means.tolist()
            ),
            "policy_action_change_means": (
                action_means.tolist()
            ),
            "standardized_displacement_means": (
                displacement_means.tolist()
            ),
            "explanation_dose_response_slope": (
                explanation_slope
            ),
            "nearest_neighbor_distance_slope": (
                distance_slope
            ),
            "policy_action_change_slope": (
                action_slope
            ),
            "positive_adjacent_explanation_fraction": (
                positive_fraction
            ),
        }

        print()
        print(
            f"Seed {seed}"
        )

        print(
            "  Explanation slope:",
            explanation_slope,
        )

        print(
            "  NN-distance slope:",
            distance_slope,
        )

        print(
            "  Policy-action slope:",
            action_slope,
        )

        print(
            "  Positive adjacent fraction:",
            positive_fraction,
        )

    # ========================================================
    # CROSS-SEED AGGREGATION
    # ========================================================

    explanation_slopes = np.asarray(
        [
            per_seed[str(seed)][
                "explanation_dose_response_slope"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    nn_slopes = np.asarray(
        [
            per_seed[str(seed)][
                "nearest_neighbor_distance_slope"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    action_slopes = np.asarray(
        [
            per_seed[str(seed)][
                "policy_action_change_slope"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    positive_fractions = np.asarray(
        [
            per_seed[str(seed)][
                "positive_adjacent_explanation_fraction"
            ]
            for seed in POLICY_SEEDS
        ],
        dtype=float,
    )

    explanation_summary = mean_ci(
        explanation_slopes
    )

    nn_summary = mean_ci(
        nn_slopes
    )

    action_summary = mean_ci(
        action_slopes
    )

    positive_fraction_summary = mean_ci(
        positive_fractions
    )

    explanation_sign_test = (
        exact_sign_flip_test(
            explanation_slopes
        )
    )

    # ========================================================
    # CROSS-SEED LEVEL-SPECIFIC SUMMARIES
    # ========================================================

    cross_seed_levels = []

    for j, level in enumerate(
        SIGMA_LEVELS
    ):

        explanation_values = np.asarray(
            [
                per_seed[str(seed)][
                    "explanation_means"
                ][j]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        nn_values = np.asarray(
            [
                per_seed[str(seed)][
                    "nearest_neighbor_means"
                ][j]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        action_values = np.asarray(
            [
                per_seed[str(seed)][
                    "policy_action_change_means"
                ][j]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        displacement_values = np.asarray(
            [
                per_seed[str(seed)][
                    "standardized_displacement_means"
                ][j]
                for seed in POLICY_SEEDS
            ],
            dtype=float,
        )

        cross_seed_levels.append(
            {
                "gaussian_equivalent_sigma": float(
                    level
                ),
                "explanation_disagreement": (
                    mean_ci(
                        explanation_values
                    )
                ),
                "nearest_neighbor_distance": (
                    mean_ci(
                        nn_values
                    )
                ),
                "policy_action_change": (
                    mean_ci(
                        action_values
                    )
                ),
                "standardized_displacement": (
                    mean_ci(
                        displacement_values
                    )
                ),
                "seed_values": {
                    "explanation_disagreement": (
                        explanation_values.tolist()
                    ),
                    "nearest_neighbor_distance": (
                        nn_values.tolist()
                    ),
                    "policy_action_change": (
                        action_values.tolist()
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
            "structured-directional-shift"
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
        "records_per_seed": (
            EXPECTED_RECORDS
        ),
        "sigma_levels": (
            SIGMA_LEVELS.tolist()
        ),
        "statistical_unit": (
            "independently trained policy seed"
        ),
        "directional_shift": {
            "definition": (
                "Unit L2 direction with all standardized "
                "observation components positive."
            ),
            "direction_norm": 1.0,
            "mapping": (
                "structured_delta = "
                "sigma * sqrt(observation_dim)"
            ),
            "observation_dimension": 11,
        },
        "per_seed": per_seed,
        "cross_seed": {
            "explanation_dose_response_slope": (
                explanation_summary
            ),
            "nearest_neighbor_distance_slope": (
                nn_summary
            ),
            "policy_action_change_slope": (
                action_summary
            ),
            "positive_adjacent_explanation_fraction": (
                positive_fraction_summary
            ),
            "exact_sign_flip_test": (
                explanation_sign_test
            ),
            "level_results": (
                cross_seed_levels
            ),
        },
        "interpretation_note": (
            "The structured-shift experiment changes only "
            "the observation perturbation mechanism relative "
            "to the frozen Gaussian study. The held-out query "
            "set, reference split, reference-only standardization, "
            "nearest-neighbor explanation metric, and frozen "
            "policy remain matched by construction."
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
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CROSS-SEED STRUCTURED SHIFT RESULT"
    )
    print("=" * 80)

    print(
        "Mean explanation slope:",
        explanation_summary[
            "mean"
        ],
    )

    print(
        "Explanation 95% CI:",
        (
            explanation_summary[
                "ci95_low"
            ],
            explanation_summary[
                "ci95_high"
            ],
        ),
    )

    print(
        "Mean NN-distance slope:",
        nn_summary[
            "mean"
        ],
    )

    print(
        "Mean policy-action slope:",
        action_summary[
            "mean"
        ],
    )

    print(
        "Mean positive adjacent fraction:",
        positive_fraction_summary[
            "mean"
        ],
    )

    print(
        "Exact one-sided sign-flip p:",
        explanation_sign_test[
            "one_sided_p"
        ],
    )

    print(
        "Exact two-sided sign-flip p:",
        explanation_sign_test[
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