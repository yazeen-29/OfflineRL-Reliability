from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import linregress, t


NOISE_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def query_level_arrays(records: list[dict]):
    query_ids = sorted(
        {int(r["query_id"]) for r in records}
    )

    noise_levels = sorted(
        {float(r["noise_level"]) for r in records}
    )

    if not np.allclose(
        noise_levels,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            f"Unexpected noise levels: {noise_levels}"
        )

    q_index = {
        q: i for i, q in enumerate(query_ids)
    }

    l_index = {
        level: i
        for i, level in enumerate(noise_levels)
    }

    nn = np.full(
        (len(query_ids), len(noise_levels)),
        np.nan,
    )

    explanation = np.full(
        (len(query_ids), len(noise_levels)),
        np.nan,
    )

    counts = np.zeros(
        (len(query_ids), len(noise_levels)),
        dtype=int,
    )

    for r in records:
        q = int(r["query_id"])
        level = float(r["noise_level"])

        i = q_index[q]
        j = l_index[level]

        if np.isnan(nn[i, j]):
            nn[i, j] = 0.0
            explanation[i, j] = 0.0

        nn[i, j] += float(
            r["nearest_neighbor_distance"]
        )

        explanation[i, j] += float(
            r["explanation_action_disagreement"]
        )

        counts[i, j] += 1

    valid = counts > 0

    nn[valid] /= counts[valid]
    explanation[valid] /= counts[valid]

    if not np.all(counts == 5):
        raise RuntimeError(
            "Expected exactly 5 repeats per query/noise cell."
        )

    if len(query_ids) != 1000:
        raise RuntimeError(
            f"Expected 1000 queries, got {len(query_ids)}."
        )

    return (
        np.asarray(query_ids),
        np.asarray(noise_levels),
        nn,
        explanation,
    )


def per_query_slopes(
    nn: np.ndarray,
    explanation: np.ndarray,
) -> np.ndarray:
    slopes = []

    for i in range(nn.shape[0]):
        x = nn[i]
        y = explanation[i]

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if np.sum(valid) < 3:
            slopes.append(np.nan)
            continue

        fit = linregress(
            x[valid],
            y[valid],
        )

        slopes.append(
            fit.slope
        )

    return np.asarray(
        slopes,
        dtype=float,
    )


def mean_ci(values: np.ndarray):
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)
    mean = float(np.mean(values))
    sd = (
        float(np.std(values, ddof=1))
        if n > 1
        else 0.0
    )

    if n > 1:
        sem = sd / np.sqrt(n)
        critical = float(
            t.ppf(
                0.975,
                df=n - 1,
            )
        )
        margin = critical * sem
    else:
        margin = 0.0

    return {
        "n": int(n),
        "mean": mean,
        "std": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def main():
    parser = argparse.ArgumentParser()

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
            "iql_100k_multiseed_gaussian_analysis.json"
        ),
    )

    args = parser.parse_args()

    sources = {
        0: Path(args.seed0),
        1: Path(args.multiseed_dir)
        / "iql_seed1_gaussian_observation_noise.json",
        2: Path(args.multiseed_dir)
        / "iql_seed2_gaussian_observation_noise.json",
        3: Path(args.multiseed_dir)
        / "iql_seed3_gaussian_observation_noise.json",
        4: Path(args.multiseed_dir)
        / "iql_seed4_gaussian_observation_noise.json",
    }

    print("=" * 80)
    print(
        "CROSS-SEED IQL GAUSSIAN SHIFT ANALYSIS"
    )
    print("=" * 80)

    seed_results = {}

    for seed, path in sources.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing seed {seed} result: {path}"
            )

        data = load_json(path)

        records = data.get(
            "records",
            [],
        )

        if len(records) != 35000:
            raise RuntimeError(
                f"Seed {seed}: expected "
                f"35000 records, got {len(records)}"
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

        positive_fraction = float(
            np.mean(
                valid_slopes > 0
            )
        )

        seed_slope = mean_ci(
            valid_slopes
        )

        seed_results[str(seed)] = {
            "seed": int(seed),
            "num_queries": int(
                len(query_ids)
            ),
            "num_levels": int(
                len(noise_levels)
            ),
            "num_repeats": 5,
            "positive_slope_fraction": (
                positive_fraction
            ),
            "query_slope": seed_slope,
            "clean_nn_distance": float(
                np.mean(
                    nn[:, 0]
                )
            ),
            "clean_explanation_disagreement": float(
                np.mean(
                    explanation[:, 0]
                )
            ),
            "noise_level_means": {
                str(level): {
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
                for j, level in enumerate(
                    noise_levels
                )
            },
        }

        print()
        print(
            f"Seed {seed}"
        )
        print(
            "  Positive slope fraction:",
            positive_fraction,
        )
        print(
            "  Mean query slope:",
            seed_slope["mean"],
        )
        print(
            "  CI95:",
            (
                seed_slope["ci95_low"],
                seed_slope["ci95_high"],
            ),
        )

    # ---------------------------------------------------------
    # Across-policy-seed aggregation
    # ---------------------------------------------------------

    seed_ids = [
        0,
        1,
        2,
        3,
        4,
    ]

    slopes_across_seeds = np.array(
        [
            seed_results[str(seed)][
                "query_slope"
            ]["mean"]
            for seed in seed_ids
        ],
        dtype=float,
    )

    positive_fractions = np.array(
        [
            seed_results[str(seed)][
                "positive_slope_fraction"
            ]
            for seed in seed_ids
        ],
        dtype=float,
    )

    cross_seed_slope = mean_ci(
        slopes_across_seeds
    )

    cross_seed_positive_fraction = (
        mean_ci(
            positive_fractions
        )
    )

    # ---------------------------------------------------------
    # Cross-seed noise-level means
    # ---------------------------------------------------------

    pooled_level_results = {}

    for level in NOISE_LEVELS:
        values_nn = np.array(
            [
                seed_results[str(seed)][
                    "noise_level_means"
                ][str(level)][
                    "nn_distance"
                ]
                for seed in seed_ids
            ],
            dtype=float,
        )

        values_explanation = np.array(
            [
                seed_results[str(seed)][
                    "noise_level_means"
                ][str(level)][
                    "explanation_disagreement"
                ]
                for seed in seed_ids
            ],
            dtype=float,
        )

        pooled_level_results[
            str(level)
        ] = {
            "nn_distance": mean_ci(
                values_nn
            ),
            "explanation_disagreement": mean_ci(
                values_explanation
            ),
        }

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    analysis = {
        "experiment": (
            "IQL-100K-cross-seed-gaussian-shift"
        ),
        "task": (
            "mujoco/hopper/medium-v0"
        ),
        "policy_seeds": seed_ids,
        "independent_replication_unit": (
            "trained IQL policy seed"
        ),
        "queries_per_seed": 1000,
        "noise_repeats_per_query_level": 5,
        "noise_levels": (
            NOISE_LEVELS.tolist()
        ),
        "statistical_note": (
            "Cross-seed inference is performed "
            "over the five independently trained "
            "policy seeds. Query-level observations "
            "are used to estimate each seed-level "
            "effect and are not treated as "
            "independent policy replications."
        ),
        "per_seed": seed_results,
        "cross_seed": {
            "mean_query_slope": (
                cross_seed_slope
            ),
            "positive_slope_fraction": (
                cross_seed_positive_fraction
            ),
            "seed_level_slope_values": (
                slopes_across_seeds.tolist()
            ),
            "seed_level_positive_fraction_values": (
                positive_fractions.tolist()
            ),
            "noise_level_results": (
                pooled_level_results
            ),
        },
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

    print()
    print("=" * 80)
    print(
        "CROSS-SEED PRIMARY RESULT"
    )
    print("=" * 80)

    print(
        "Mean policy-seed slope:",
        cross_seed_slope,
    )

    print(
        "Mean positive-slope fraction:",
        cross_seed_positive_fraction,
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()