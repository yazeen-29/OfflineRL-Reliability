from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon


NOISE_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3],
    dtype=float,
)

EXPECTED_QUERIES = 1000
EXPECTED_REPEATS = 5
EXPECTED_RECORDS = (
    EXPECTED_QUERIES
    * len(NOISE_LEVELS)
    * EXPECTED_REPEATS
)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def build_matrix(
    records: list[dict],
    value_key: str,
):
    """
    Build query x noise-level values by averaging the five
    repeats within each query/noise cell.
    """
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} records, "
            f"got {len(records)}"
        )

    query_ids = sorted(
        {int(r["query_id"]) for r in records}
    )

    noise_levels = sorted(
        {float(r["noise_level"]) for r in records}
    )

    repeat_ids = sorted(
        {int(r["repeat_id"]) for r in records}
    )

    if len(query_ids) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_QUERIES} queries, "
            f"got {len(query_ids)}"
        )

    if not np.allclose(
        noise_levels,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            f"Unexpected noise levels: {noise_levels}"
        )

    if repeat_ids != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"Unexpected repeat IDs: {repeat_ids}"
        )

    q_index = {
        q: i for i, q in enumerate(query_ids)
    }

    l_index = {
        level: i
        for i, level in enumerate(noise_levels)
    }

    values = np.full(
        (
            len(query_ids),
            len(noise_levels),
        ),
        np.nan,
        dtype=float,
    )

    counts = np.zeros_like(
        values,
        dtype=int,
    )

    for record in records:
        q = int(record["query_id"])
        level = float(record["noise_level"])

        i = q_index[q]
        j = l_index[level]

        if np.isnan(values[i, j]):
            values[i, j] = 0.0

        values[i, j] += float(
            record[value_key]
        )

        counts[i, j] += 1

    if not np.all(
        counts == EXPECTED_REPEATS
    ):
        raise RuntimeError(
            "Not every query/noise-level cell "
            "contains exactly 5 repeats."
        )

    values /= counts

    return (
        np.asarray(query_ids),
        np.asarray(noise_levels),
        values,
    )


def paired_wilcoxon(
    nearest: np.ndarray,
    random: np.ndarray,
):
    """
    Paired comparison across queries.

    Positive difference means:
        random_reference > nearest_neighbor
    """
    diff = random - nearest

    valid = np.isfinite(diff)

    diff = diff[valid]

    if len(diff) == 0:
        return {
            "n_queries": 0,
            "mean_difference": None,
            "median_difference": None,
            "wilcoxon_statistic": None,
            "wilcoxon_p": None,
        }

    if np.allclose(
        diff,
        0.0,
    ):
        return {
            "n_queries": int(len(diff)),
            "mean_difference": 0.0,
            "median_difference": 0.0,
            "wilcoxon_statistic": 0.0,
            "wilcoxon_p": 1.0,
        }

    statistic, p_value = wilcoxon(
        diff,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )

    return {
        "n_queries": int(len(diff)),
        "mean_difference": float(
            np.mean(diff)
        ),
        "median_difference": float(
            np.median(diff)
        ),
        "wilcoxon_statistic": float(
            statistic
        ),
        "wilcoxon_p": float(
            p_value
        ),
    }


def mean_ci_across_queries(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 1234,
):
    """
    Descriptive bootstrap 95% CI for query-level mean.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    rng = np.random.default_rng(seed)
    n = len(values)

    boot = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )
        boot[i] = np.mean(sample)

    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(
            np.percentile(
                boot,
                2.5,
            )
        ),
        "ci95_high": float(
            np.percentile(
                boot,
                97.5,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired analysis of nearest-neighbor and "
            "uniform-random reference explanations."
        )
    )

    parser.add_argument(
        "--nearest",
        default=(
            "results/shifts/"
            "iql_100k_gaussian_observation_noise.json"
        ),
    )

    parser.add_argument(
        "--random",
        default=(
            "results/shifts/random_reference/"
            "iql_seed0_random_reference.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/analysis/"
            "iql_seed0_random_reference_control.json"
        ),
    )

    args = parser.parse_args()

    nearest_path = Path(args.nearest)
    random_path = Path(args.random)
    output_path = Path(args.output)

    print("=" * 80)
    print(
        "NEAREST-NEIGHBOR VS UNIFORM-RANDOM REFERENCE CONTROL"
    )
    print("=" * 80)

    nearest_data = load_json(
        nearest_path
    )

    random_data = load_json(
        random_path
    )

    nearest_records = nearest_data["records"]
    random_records = random_data["records"]

    # ---------------------------------------------------------
    # Build matched query x level matrices
    # ---------------------------------------------------------

    (
        nearest_queries,
        nearest_levels,
        nearest_disagreement,
    ) = build_matrix(
        nearest_records,
        "explanation_action_disagreement",
    )

    (
        random_queries,
        random_levels,
        random_disagreement,
    ) = build_matrix(
        random_records,
        "random_reference_action_disagreement",
    )

    if not np.array_equal(
        nearest_queries,
        random_queries,
    ):
        raise RuntimeError(
            "Nearest and random analyses do not use "
            "the same query IDs."
        )

    if not np.allclose(
        nearest_levels,
        random_levels,
    ):
        raise RuntimeError(
            "Nearest and random analyses do not use "
            "the same noise levels."
        )

    # ---------------------------------------------------------
    # Per-level paired analysis
    # ---------------------------------------------------------

    level_results = []

    for j, level in enumerate(
        nearest_levels
    ):

        nearest = nearest_disagreement[
            :, j
        ]

        random = random_disagreement[
            :, j
        ]

        difference = (
            random - nearest
        )

        paired = paired_wilcoxon(
            nearest,
            random,
        )

        result = {
            "noise_level": float(level),
            "nearest_neighbor": (
                mean_ci_across_queries(
                    nearest,
                    seed=1000 + j,
                )
            ),
            "uniform_random": (
                mean_ci_across_queries(
                    random,
                    seed=2000 + j,
                )
            ),
            "random_minus_nearest": (
                mean_ci_across_queries(
                    difference,
                    seed=3000 + j,
                )
            ),
            "paired_test": paired,
        }

        level_results.append(
            result
        )

        print()
        print(
            f"Noise = {level:.3f}"
        )

        print(
            "  NN mean:",
            result[
                "nearest_neighbor"
            ]["mean"],
        )

        print(
            "  Random mean:",
            result[
                "uniform_random"
            ]["mean"],
        )

        print(
            "  Random - NN:",
            result[
                "random_minus_nearest"
            ]["mean"],
        )

        print(
            "  Paired Wilcoxon p:",
            paired[
                "wilcoxon_p"
            ],
        )

    # ---------------------------------------------------------
    # Overall relationship with NN distance
    # ---------------------------------------------------------

    flat_nn_distance = np.asarray(
        [
            float(
                r[
                    "nearest_neighbor_distance"
                ]
            )
            for r in nearest_records
        ],
        dtype=float,
    )

    flat_nn_disagreement = (
        np.asarray(
            [
                float(
                    r[
                        "explanation_action_disagreement"
                    ]
                )
                for r in nearest_records
            ],
            dtype=float,
        )
    )

    flat_random_disagreement = (
        np.asarray(
            [
                float(
                    r[
                        "random_reference_action_disagreement"
                    ]
                )
                for r in random_records
            ],
            dtype=float,
        )
    )

    valid_nn = (
        np.isfinite(
            flat_nn_distance
        )
        & np.isfinite(
            flat_nn_disagreement
        )
    )

    valid_random = (
        np.isfinite(
            flat_nn_distance
        )
        & np.isfinite(
            flat_random_disagreement
        )
    )

    nn_pearson = pearsonr(
        flat_nn_distance[valid_nn],
        flat_nn_disagreement[valid_nn],
    )

    nn_spearman = spearmanr(
        flat_nn_distance[valid_nn],
        flat_nn_disagreement[valid_nn],
    )

    random_pearson = pearsonr(
        flat_nn_distance[valid_random],
        flat_random_disagreement[valid_random],
    )

    random_spearman = spearmanr(
        flat_nn_distance[valid_random],
        flat_random_disagreement[valid_random],
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    analysis = {
        "experiment": (
            "IQL-100K-seed0-random-reference-control"
        ),
        "task": nearest_data["task"],
        "seed": nearest_data["seed"],
        "analysis_unit": (
            "paired query-level means after averaging "
            "five perturbation repeats within each "
            "query/noise-level cell"
        ),
        "queries": EXPECTED_QUERIES,
        "noise_levels": [
            float(x)
            for x in nearest_levels
        ],
        "nearest_source": str(
            nearest_path
        ),
        "random_source": str(
            random_path
        ),
        "level_results": level_results,
        "distance_relationships": {
            "nearest_neighbor_explanation": {
                "pearson_r": float(
                    nn_pearson.statistic
                ),
                "pearson_p": float(
                    nn_pearson.pvalue
                ),
                "spearman_rho": float(
                    nn_spearman.statistic
                ),
                "spearman_p": float(
                    nn_spearman.pvalue
                ),
            },
            "uniform_random_explanation": {
                "pearson_r": float(
                    random_pearson.statistic
                ),
                "pearson_p": float(
                    random_pearson.pvalue
                ),
                "spearman_rho": float(
                    random_spearman.statistic
                ),
                "spearman_p": float(
                    random_spearman.pvalue
                ),
            },
        },
    }

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

    print()
    print("=" * 80)
    print(
        "RANDOM-REFERENCE CONTROL ANALYSIS COMPLETE"
    )
    print("=" * 80)
    print(
        "Saved:",
        output_path,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()