from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import (
    pearsonr,
    spearmanr,
    wilcoxon,
    linregress,
)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def bootstrap_ci(
    values: np.ndarray,
    seed: int,
    n_bootstrap: int = 5000,
    statistic=np.mean,
):
    """
    Nonparametric bootstrap 95% confidence interval.

    The observations supplied here must already represent the
    independent experimental unit being analyzed.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
        }

    rng = np.random.default_rng(seed)
    n = len(values)

    estimates = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )
        estimates[i] = statistic(sample)

    return {
        "estimate": float(statistic(values)),
        "ci95_low": float(
            np.percentile(estimates, 2.5)
        ),
        "ci95_high": float(
            np.percentile(estimates, 97.5)
        ),
        "n": int(n),
    }


def get_records(data):
    records = data.get("records", [])

    if not records:
        raise RuntimeError(
            "No individual records found in JSON."
        )

    required = {
        "query_id",
        "repeat_id",
        "noise_level",
        "nearest_neighbor_distance",
        "policy_action_change",
        "explanation_action_disagreement",
    }

    missing = required - set(records[0].keys())

    if missing:
        raise RuntimeError(
            f"Records are missing required fields: {missing}"
        )

    return records


def build_query_level_arrays(records):
    """
    Convert repeated records into query x noise-level matrices.

    Five perturbation repeats are averaged within each
    query/noise-level cell.

    Independent statistical unit:
        query_id
    """

    noise_levels = sorted(
        {
            float(r["noise_level"])
            for r in records
        }
    )

    query_ids = sorted(
        {
            int(r["query_id"])
            for r in records
        }
    )

    q_index = {
        q: i
        for i, q in enumerate(query_ids)
    }

    l_index = {
        level: i
        for i, level in enumerate(noise_levels)
    }

    n_queries = len(query_ids)
    n_levels = len(noise_levels)

    nn = np.full(
        (n_queries, n_levels),
        np.nan,
    )

    action = np.full(
        (n_queries, n_levels),
        np.nan,
    )

    explanation = np.full(
        (n_queries, n_levels),
        np.nan,
    )

    counts = np.zeros(
        (n_queries, n_levels),
        dtype=int,
    )

    for r in records:
        q = int(r["query_id"])
        level = float(r["noise_level"])

        i = q_index[q]
        j = l_index[level]

        if np.isnan(nn[i, j]):
            nn[i, j] = 0.0
            action[i, j] = 0.0
            explanation[i, j] = 0.0

        nn[i, j] += float(
            r["nearest_neighbor_distance"]
        )

        action[i, j] += float(
            r["policy_action_change"]
        )

        explanation[i, j] += float(
            r["explanation_action_disagreement"]
        )

        counts[i, j] += 1

    valid = counts > 0

    nn[valid] /= counts[valid]
    action[valid] /= counts[valid]
    explanation[valid] /= counts[valid]

    return (
        np.asarray(query_ids, dtype=int),
        np.asarray(noise_levels, dtype=float),
        nn,
        action,
        explanation,
        counts,
    )


def paired_wilcoxon(
    clean: np.ndarray,
    shifted: np.ndarray,
):
    """
    Paired one-sided Wilcoxon signed-rank test.

    Independent unit = query.

    Tests:
        shifted > clean

    Returns a numerical p-value. Very small values can be reported
    in scientific notation by downstream code.
    """
    difference = np.asarray(
        shifted - clean,
        dtype=float,
    )

    difference = difference[
        np.isfinite(difference)
    ]

    if len(difference) == 0:
        return {
            "n_queries": 0,
            "mean_difference": None,
            "median_difference": None,
            "wilcoxon_statistic": None,
            "wilcoxon_p": None,
        }

    if np.allclose(
        difference,
        0.0,
    ):
        return {
            "n_queries": int(len(difference)),
            "mean_difference": 0.0,
            "median_difference": 0.0,
            "wilcoxon_statistic": 0.0,
            "wilcoxon_p": 1.0,
        }

    statistic, p_value = wilcoxon(
        difference,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )

    return {
        "n_queries": int(len(difference)),
        "mean_difference": float(
            np.mean(difference)
        ),
        "median_difference": float(
            np.median(difference)
        ),
        "wilcoxon_statistic": float(
            statistic
        ),
        "wilcoxon_p": float(
            p_value
        ),
    }


def holm_bonferroni(
    p_values: np.ndarray,
):
    """
    Holm-Bonferroni family-wise error correction.

    Input:
        raw p-values in original test order

    Output:
        adjusted p-values in the same order
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    if len(p_values) == 0:
        return np.array(
            [],
            dtype=float,
        )

    order = np.argsort(
        p_values
    )

    sorted_p = p_values[
        order
    ]

    m = len(
        sorted_p
    )

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for i, p in enumerate(
        sorted_p
    ):
        adjusted_value = (
            m - i
        ) * p

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0,
        )

    adjusted = np.empty_like(
        adjusted_sorted
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


def per_query_slopes(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    For each query, fit:

        explanation_disagreement
            ~
        nearest_neighbor_distance

    across the seven noise levels.

    Returns:
        one slope per query
    """
    slopes = []

    for i in range(
        x.shape[0]
    ):
        xi = x[i]
        yi = y[i]

        valid = (
            np.isfinite(xi)
            & np.isfinite(yi)
        )

        if np.sum(valid) < 3:
            slopes.append(
                np.nan
            )
            continue

        fit = linregress(
            xi[valid],
            yi[valid],
        )

        slopes.append(
            fit.slope
        )

    return np.asarray(
        slopes,
        dtype=float,
    )


def permutation_p_for_slopes(
    slopes: np.ndarray,
    seed: int,
    n_permutations: int = 10000,
):
    """
    Optional robustness test for whether the mean per-query slope
    is greater than zero.

    Null:
        slopes are exchangeable around zero.

    This is a sign-flip permutation test.
    """
    slopes = np.asarray(
        slopes,
        dtype=float,
    )

    slopes = slopes[
        np.isfinite(slopes)
    ]

    if len(slopes) == 0:
        return {
            "n_queries": 0,
            "observed_mean": None,
            "permutation_p": None,
        }

    rng = np.random.default_rng(
        seed
    )

    observed = float(
        np.mean(slopes)
    )

    greater_or_equal = 0

    for _ in range(
        n_permutations
    ):
        signs = rng.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=len(slopes),
        )

        permuted_mean = float(
            np.mean(
                slopes * signs
            )
        )

        if (
            permuted_mean
            >= observed
        ):
            greater_or_equal += 1

    p_value = (
        greater_or_equal + 1
    ) / (
        n_permutations + 1
    )

    return {
        "n_queries": int(
            len(slopes)
        ),
        "observed_mean": observed,
        "permutation_p": float(
            p_value
        ),
        "n_permutations": int(
            n_permutations
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paper-grade query-level statistical "
            "analysis of explanation degradation "
            "under Gaussian observation shift."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "results/shifts/"
            "iql_100k_gaussian_observation_noise.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/analysis/"
            "iql_100k_gaussian_shift_analysis_v3.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "PAPER-GRADE QUERY-LEVEL GAUSSIAN SHIFT ANALYSIS"
    )
    print("=" * 78)

    data = load_json(
        args.input
    )

    records = get_records(
        data
    )

    (
        query_ids,
        noise_levels,
        nn,
        action,
        explanation,
        counts,
    ) = build_query_level_arrays(
        records
    )

    print(
        f"Queries: {len(query_ids)}"
    )

    print(
        "Noise levels:",
        noise_levels,
    )

    print(
        "Repeats per query/level:",
        np.unique(counts),
    )

    if not np.array_equal(
        np.unique(counts),
        np.array([5]),
    ):
        raise RuntimeError(
            "Expected exactly five perturbation "
            "repeats for every query/noise-level cell."
        )

    # ------------------------------------------------------------
    # Identify clean level.
    # ------------------------------------------------------------

    clean_matches = np.where(
        np.isclose(
            noise_levels,
            0.0,
        )
    )[0]

    if len(clean_matches) != 1:
        raise RuntimeError(
            "Exactly one noise level of 0.0 is required."
        )

    clean_idx = int(
        clean_matches[0]
    )

    clean_nn = nn[
        :,
        clean_idx,
    ]

    clean_action = action[
        :,
        clean_idx,
    ]

    clean_explanation = explanation[
        :,
        clean_idx,
    ]

    print(
        "\nClean baseline:"
    )

    print(
        "Mean NN distance:",
        np.mean(clean_nn),
    )

    print(
        "Mean explanation disagreement:",
        np.mean(
            clean_explanation
        ),
    )

    # ------------------------------------------------------------
    # Level-wise summaries and paired tests.
    # ------------------------------------------------------------

    level_results = []

    raw_explanation_p = []
    explanation_test_indices = []

    for j, level in enumerate(
        noise_levels
    ):
        current_nn = nn[
            :,
            j,
        ]

        current_action = action[
            :,
            j,
        ]

        current_explanation = explanation[
            :,
            j,
        ]

        nn_difference = (
            current_nn
            - clean_nn
        )

        action_difference = (
            current_action
            - clean_action
        )

        explanation_difference = (
            current_explanation
            - clean_explanation
        )

        paired_nn = paired_wilcoxon(
            clean_nn,
            current_nn,
        )

        paired_action = paired_wilcoxon(
            clean_action,
            current_action,
        )

        paired_explanation = paired_wilcoxon(
            clean_explanation,
            current_explanation,
        )

        result = {
            "noise_level": float(
                level
            ),
            "n_queries": int(
                len(query_ids)
            ),
            "nn_distance": bootstrap_ci(
                current_nn,
                seed=1000 + j,
            ),
            "policy_action_change": bootstrap_ci(
                current_action,
                seed=2000 + j,
            ),
            "explanation_disagreement": bootstrap_ci(
                current_explanation,
                seed=3000 + j,
            ),
            "paired_clean_vs_current": {
                "nn_distance": paired_nn,
                "policy_action_change": paired_action,
                "explanation_disagreement": (
                    paired_explanation
                ),
            },
            "mean_change_from_clean": {
                "nn_distance": float(
                    np.mean(
                        nn_difference
                    )
                ),
                "policy_action_change": float(
                    np.mean(
                        action_difference
                    )
                ),
                "explanation_disagreement": float(
                    np.mean(
                        explanation_difference
                    )
                ),
            },
        }

        level_results.append(
            result
        )

        if level > 0.0:
            raw_explanation_p.append(
                paired_explanation[
                    "wilcoxon_p"
                ]
            )

            explanation_test_indices.append(
                len(level_results) - 1
            )

    # ------------------------------------------------------------
    # Holm correction for six clean-vs-shifted
    # explanation tests.
    # ------------------------------------------------------------

    raw_explanation_p = np.asarray(
        raw_explanation_p,
        dtype=float,
    )

    adjusted_explanation_p = holm_bonferroni(
        raw_explanation_p
    )

    for idx, adjusted_p in zip(
        explanation_test_indices,
        adjusted_explanation_p,
    ):
        level_results[
            idx
        ][
            "paired_clean_vs_current"
        ][
            "explanation_disagreement"
        ][
            "holm_adjusted_p"
        ] = float(
            adjusted_p
        )

    # Clean level has no actual comparison.
    level_results[
        clean_idx
    ][
        "paired_clean_vs_current"
    ][
        "explanation_disagreement"
    ][
        "holm_adjusted_p"
    ] = 1.0

    # ------------------------------------------------------------
    # PRIMARY ANALYSIS
    #
    # One slope per query:
    #
    # explanation disagreement
    #      ~
    # nearest-neighbor distance
    #
    # The query is the independent unit.
    # ------------------------------------------------------------

    query_slopes = per_query_slopes(
        nn,
        explanation,
    )

    valid_slopes = query_slopes[
        np.isfinite(
            query_slopes
        )
    ]

    positive_fraction = float(
        np.mean(
            valid_slopes > 0
        )
    )

    slope_summary = bootstrap_ci(
        valid_slopes,
        seed=7000,
        statistic=np.mean,
    )

    slope_wilcoxon = wilcoxon(
        valid_slopes,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )

    permutation = permutation_p_for_slopes(
        valid_slopes,
        seed=8000,
        n_permutations=10000,
    )

    primary_slope_analysis = {
        "independent_unit": (
            "query"
        ),
        "n_queries": int(
            len(valid_slopes)
        ),
        "positive_slope_fraction": (
            positive_fraction
        ),
        "mean_query_slope": (
            slope_summary
        ),
        "wilcoxon_test": {
            "statistic": float(
                slope_wilcoxon.statistic
            ),
            "p": float(
                slope_wilcoxon.pvalue
            ),
            "alternative": (
                "greater_than_zero"
            ),
        },
        "sign_flip_permutation_test": (
            permutation
        ),
    }

    # ------------------------------------------------------------
    # DESCRIPTIVE query-level association.
    #
    # This uses 7 points per query, but the repeated observations
    # are NOT treated as independent for the primary inference.
    #
    # We retain these correlations as descriptive summaries only.
    # ------------------------------------------------------------

    flat_nn = nn.reshape(
        -1
    )

    flat_explanation = explanation.reshape(
        -1
    )

    valid = (
        np.isfinite(flat_nn)
        & np.isfinite(
            flat_explanation
        )
    )

    flat_nn = flat_nn[
        valid
    ]

    flat_explanation = (
        flat_explanation[
            valid
        ]
    )

    pearson_r, _ = pearsonr(
        flat_nn,
        flat_explanation,
    )

    spearman_rho, _ = spearmanr(
        flat_nn,
        flat_explanation,
    )

    descriptive_regression = linregress(
        flat_nn,
        flat_explanation,
    )

    descriptive_relationship = {
        "interpretation": (
            "Descriptive only: repeated query-level "
            "observations across noise levels are "
            "correlated within query, so these "
            "p-values must not be interpreted as "
            "independent-sample inference."
        ),
        "n_query_level_cells": int(
            len(flat_nn)
        ),
        "pearson_r": float(
            pearson_r
        ),
        "spearman_rho": float(
            spearman_rho
        ),
        "linear_regression": {
            "slope": float(
                descriptive_regression.slope
            ),
            "intercept": float(
                descriptive_regression.intercept
            ),
            "r_squared": float(
                descriptive_regression.rvalue ** 2
            ),
        },
    }

    # ------------------------------------------------------------
    # Dose-response summaries.
    # ------------------------------------------------------------

    level_noise = noise_levels.astype(
        float
    )

    level_nn = np.array(
        [
            r[
                "nn_distance"
            ][
                "estimate"
            ]
            for r in level_results
        ]
    )

    level_action = np.array(
        [
            r[
                "policy_action_change"
            ][
                "estimate"
            ]
            for r in level_results
        ]
    )

    level_explanation = np.array(
        [
            r[
                "explanation_disagreement"
            ][
                "estimate"
            ]
            for r in level_results
        ]
    )

    dose_response = {
        "noise_vs_nn_distance_pearson_r": float(
            pearsonr(
                level_noise,
                level_nn,
            )[0]
        ),
        "noise_vs_policy_action_change_pearson_r": float(
            pearsonr(
                level_noise,
                level_action,
            )[0]
        ),
        "noise_vs_explanation_disagreement_pearson_r": float(
            pearsonr(
                level_noise,
                level_explanation,
            )[0]
        ),
    }

    # ------------------------------------------------------------
    # Relative changes from clean.
    # ------------------------------------------------------------

    clean_mean_nn = float(
        np.mean(
            clean_nn
        )
    )

    clean_mean_action = float(
        np.mean(
            clean_action
        )
    )

    clean_mean_explanation = float(
        np.mean(
            clean_explanation
        )
    )

    relative_changes = []

    for j, level in enumerate(
        noise_levels
    ):
        current_nn = float(
            np.mean(
                nn[
                    :,
                    j,
                ]
            )
        )

        current_action = float(
            np.mean(
                action[
                    :,
                    j,
                ]
            )
        )

        current_explanation = float(
            np.mean(
                explanation[
                    :,
                    j,
                ]
            )
        )

        relative_changes.append(
            {
                "noise_level": float(
                    level
                ),
                "nn_relative_increase": (
                    (
                        current_nn
                        - clean_mean_nn
                    )
                    / clean_mean_nn
                    if clean_mean_nn != 0
                    else None
                ),
                "policy_action_relative_change": (
                    (
                        current_action
                        - clean_mean_action
                    )
                    / clean_mean_action
                    if clean_mean_action != 0
                    else None
                ),
                "explanation_relative_increase": (
                    (
                        current_explanation
                        - clean_mean_explanation
                    )
                    / clean_mean_explanation
                    if clean_mean_explanation != 0
                    else None
                ),
            }
        )

    # ------------------------------------------------------------
    # Final result object.
    # ------------------------------------------------------------

    output = {
        "analysis_version": (
            "v3-paper-grade"
        ),
        "source_experiment": data[
            "experiment"
        ],
        "task": data[
            "task"
        ],
        "seed": data[
            "seed"
        ],
        "checkpoint": data[
            "checkpoint"
        ],
        "query_count": int(
            len(query_ids)
        ),
        "noise_levels": [
            float(x)
            for x in noise_levels
        ],
        "noise_repeat_count": 5,
        "independent_unit": (
            "query_id"
        ),
        "repeat_handling": (
            "five perturbation repeats averaged "
            "within each query/noise-level cell"
        ),
        "primary_hypothesis": {
            "statement": (
                "Queries with stronger movement away "
                "from the reference data distribution "
                "will show greater disagreement between "
                "the shifted-policy action and the "
                "nearest-neighbor explanation action."
            )
        },
        "primary_query_slope_analysis": (
            primary_slope_analysis
        ),
        "descriptive_relationship": (
            descriptive_relationship
        ),
        "dose_response": (
            dose_response
        ),
        "clean_baseline": {
            "mean_nn_distance": clean_mean_nn,
            "mean_policy_action_change": (
                clean_mean_action
            ),
            "mean_explanation_disagreement": (
                clean_mean_explanation
            ),
        },
        "holm_corrected_level_results": (
            level_results
        ),
        "relative_changes": (
            relative_changes
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
            output,
            indent=2,
        )
    )

    print(
        "\n" + "=" * 78
    )
    print(
        "PRIMARY QUERY-SLOPE RESULT"
    )
    print(
        "Positive slope fraction:",
        positive_fraction,
    )
    print(
        "Mean slope:",
        slope_summary,
    )
    print(
        "Wilcoxon p:",
        slope_wilcoxon.pvalue,
    )
    print(
        "Sign-flip permutation p:",
        permutation[
            "permutation_p"
        ],
    )

    print(
        "\nDESCRIPTIVE RELATIONSHIP"
    )
    print(
        "Pearson r:",
        pearson_r,
    )
    print(
        "Spearman rho:",
        spearman_rho,
    )

    print(
        "\nHOLM-CORRECTED EXPLANATION TESTS"
    )

    for result in level_results:
        level = result[
            "noise_level"
        ]

        adjusted_p = result[
            "paired_clean_vs_current"
        ][
            "explanation_disagreement"
        ][
            "holm_adjusted_p"
        ]

        print(
            f"noise={level:.3f} "
            f"Holm-p={adjusted_p:.6e}"
        )

    print(
        "\nSaved:",
        output_path,
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()