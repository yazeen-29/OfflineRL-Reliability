from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr


HOPPER = Path(
    "results/analysis/multiseed/"
    "iql_100k_multiseed_gaussian_analysis_v2.json"
)

HALFCHEETAH = Path(
    "results/analysis/halfcheetah/"
    "iql_100k_multiseed_gaussian_analysis.json"
)

OUTPUT = Path(
    "results/analysis/"
    "cross_environment_gaussian_synthesis.json"
)


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing analysis file: {path}"
        )

    return json.loads(
        path.read_text()
    )


def finite(values):
    return np.asarray(
        values,
        dtype=float,
    )[
        lambda x: np.isfinite(x)
    ]


def environment_summary(
    name: str,
    data: dict,
) -> dict:

    cross = data["cross_seed"]

    slopes = np.asarray(
        cross["seed_level_slope_values"],
        dtype=float,
    )

    positive = np.asarray(
        cross[
            "seed_level_positive_fraction_values"
        ],
        dtype=float,
    )

    levels = np.asarray(
        data["noise_levels"],
        dtype=float,
    )

    dose = cross["noise_level_results"]

    per_seed_endpoint = []

    per_seed_relative = []

    per_seed_spearman = []

    per_seed_kendall = []

    per_seed_monotonic_nn = []

    per_seed_monotonic_explanation = []

    for seed in data["policy_seeds"]:

        seed_data = data["per_seed"][str(seed)]

        level_means = seed_data[
            "noise_level_means"
        ]

        nn = np.asarray(
            [
                level_means[str(float(level))]
                ["nn_distance"]
                for level in levels
            ],
            dtype=float,
        )

        explanation = np.asarray(
            [
                level_means[str(float(level))]
                ["explanation_disagreement"]
                for level in levels
            ],
            dtype=float,
        )

        clean = float(
            explanation[0]
        )

        high = float(
            explanation[-1]
        )

        endpoint_delta = (
            high - clean
        )

        relative_change = (
            endpoint_delta / clean
            if clean != 0
            else np.nan
        )

        rho, rho_p = spearmanr(
            nn,
            explanation,
        )

        tau, tau_p = kendalltau(
            nn,
            explanation,
        )

        nn_adjacent = (
            np.diff(nn) > 0
        )

        exp_adjacent = (
            np.diff(explanation) > 0
        )

        per_seed_endpoint.append(
            endpoint_delta
        )

        per_seed_relative.append(
            relative_change
        )

        per_seed_spearman.append(
            {
                "rho": float(rho),
                "p": float(rho_p),
            }
        )

        per_seed_kendall.append(
            {
                "tau": float(tau),
                "p": float(tau_p),
            }
        )

        per_seed_monotonic_nn.append(
            float(
                np.mean(nn_adjacent)
            )
        )

        per_seed_monotonic_explanation.append(
            float(
                np.mean(exp_adjacent)
            )
        )

    return {
        "environment": name,
        "task": data["task"],
        "policy_seeds": data["policy_seeds"],
        "queries_per_seed": data[
            "queries_per_seed"
        ],
        "records_per_seed": data[
            "records_per_seed"
        ],
        "total_records": int(
            data["records_per_seed"]
            * len(data["policy_seeds"])
        ),
        "primary_mean_slope": cross[
            "mean_query_slope"
        ],
        "seed_level_slopes": slopes.tolist(),
        "positive_query_slope_fraction": cross[
            "mean_positive_slope_fraction"
        ],
        "exact_sign_flip_test": cross[
            "exact_sign_flip_test"
        ],
        "endpoint_explanation_change": {
            "seed_values": (
                np.asarray(
                    per_seed_endpoint,
                    dtype=float,
                )
                .tolist()
            ),
            "mean": float(
                np.mean(
                    per_seed_endpoint
                )
            ),
            "std": float(
                np.std(
                    per_seed_endpoint,
                    ddof=1,
                )
            ),
        },
        "relative_explanation_change": {
            "seed_values": (
                np.asarray(
                    per_seed_relative,
                    dtype=float,
                )
                .tolist()
            ),
            "mean": float(
                np.mean(
                    per_seed_relative
                )
            ),
            "std": float(
                np.std(
                    per_seed_relative,
                    ddof=1,
                )
            ),
        },
        "dose_response_sensitivity": {
            "spearman": per_seed_spearman,
            "kendall": per_seed_kendall,
            "mean_spearman_rho": float(
                np.mean(
                    [
                        x["rho"]
                        for x in per_seed_spearman
                    ]
                )
            ),
            "mean_kendall_tau": float(
                np.mean(
                    [
                        x["tau"]
                        for x in per_seed_kendall
                    ]
                )
            ),
            "mean_adjacent_increase_fraction_nn": (
                float(
                    np.mean(
                        per_seed_monotonic_nn
                    )
                )
            ),
            "mean_adjacent_increase_fraction_explanation": (
                float(
                    np.mean(
                        per_seed_monotonic_explanation
                    )
                )
            ),
        },
    }


def main():

    hopper = load(HOPPER)
    halfcheetah = load(HALFCHEETAH)

    hopper_summary = environment_summary(
        "Hopper",
        hopper,
    )

    halfcheetah_summary = environment_summary(
        "HalfCheetah",
        halfcheetah,
    )

    # Environment-level comparison is descriptive only.
    # Raw slope magnitudes are not treated as directly
    # comparable effect sizes because the environments have
    # different action spaces and disagreement scales.
    slope_ratio = (
        halfcheetah_summary[
            "primary_mean_slope"
        ]["mean"]
        /
        hopper_summary[
            "primary_mean_slope"
        ]["mean"]
    )

    synthesis = {
        "experiment": (
            "cross-environment-gaussian-synthesis"
        ),
        "environments": [
            hopper_summary,
            halfcheetah_summary,
        ],
        "descriptive_comparison": {
            "hopper_mean_slope": (
                hopper_summary[
                    "primary_mean_slope"
                ]["mean"]
            ),
            "halfcheetah_mean_slope": (
                halfcheetah_summary[
                    "primary_mean_slope"
                ]["mean"]
            ),
            "halfcheetah_to_hopper_raw_slope_ratio": (
                float(slope_ratio)
            ),
            "all_seed_slopes_positive": {
                "hopper": bool(
                    np.all(
                        np.asarray(
                            hopper_summary[
                                "seed_level_slopes"
                            ]
                        )
                        > 0
                    )
                ),
                "halfcheetah": bool(
                    np.all(
                        np.asarray(
                            halfcheetah_summary[
                                "seed_level_slopes"
                            ]
                        )
                        > 0
                    )
                ),
            },
        },
        "interpretation_guardrail": (
            "The primary cross-environment claim is replication "
            "of a positive distance-to-explanation-disagreement "
            "relationship and its consistency across independently "
            "trained policy seeds. Raw slope magnitudes are not "
            "interpreted as directly comparable effect sizes "
            "between environments."
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            synthesis,
            indent=2,
        )
    )

    print("=" * 80)
    print("CROSS-ENVIRONMENT GAUSSIAN SYNTHESIS")
    print("=" * 80)

    for summary in [
        hopper_summary,
        halfcheetah_summary,
    ]:
        primary = summary[
            "primary_mean_slope"
        ]

        print()
        print(summary["environment"])
        print(
            "  Mean slope:",
            primary["mean"],
        )
        print(
            "  95% CI:",
            (
                primary["ci95_low"],
                primary["ci95_high"],
            ),
        )
        print(
            "  Mean positive query-slope fraction:",
            summary[
                "positive_query_slope_fraction"
            ]["mean"],
        )
        print(
            "  Mean Spearman rho:",
            summary[
                "dose_response_sensitivity"
            ]["mean_spearman_rho"],
        )
        print(
            "  Mean Kendall tau:",
            summary[
                "dose_response_sensitivity"
            ]["mean_kendall_tau"],
        )
        print(
            "  Mean adjacent NN-increase fraction:",
            summary[
                "dose_response_sensitivity"
            ][
                "mean_adjacent_increase_fraction_nn"
            ],
        )
        print(
            "  Mean adjacent explanation-increase fraction:",
            summary[
                "dose_response_sensitivity"
            ][
                "mean_adjacent_increase_fraction_explanation"
            ],
        )
        print(
            "  Mean endpoint explanation change:",
            summary[
                "endpoint_explanation_change"
            ]["mean"],
        )
        print(
            "  Mean relative endpoint change:",
            summary[
                "relative_explanation_change"
            ]["mean"],
        )

    print()
    print("=" * 80)
    print("CROSS-ENVIRONMENT DESCRIPTIVE RESULT")
    print("=" * 80)

    print(
        "All Hopper seed slopes positive:",
        synthesis[
            "descriptive_comparison"
        ][
            "all_seed_slopes_positive"
        ]["hopper"],
    )

    print(
        "All HalfCheetah seed slopes positive:",
        synthesis[
            "descriptive_comparison"
        ][
            "all_seed_slopes_positive"
        ]["halfcheetah"],
    )

    print(
        "Raw slope ratio (HalfCheetah/Hopper):",
        slope_ratio,
    )

    print()
    print("Saved:", OUTPUT)
    print("✅ CROSS-ENVIRONMENT SYNTHESIS COMPLETE")


if __name__ == "__main__":
    main()
