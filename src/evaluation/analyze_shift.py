from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress


def load_results(path: str):
    with open(path, "r") as f:
        return json.load(f)


def extract_level_arrays(data):
    levels = data["noise_level_summaries"]

    noise = np.array(
        [x["noise_level"] for x in levels],
        dtype=float,
    )

    nn_distance = np.array(
        [
            x["nearest_neighbor_distance"]["mean"]
            for x in levels
        ],
        dtype=float,
    )

    action_change = np.array(
        [
            x["policy_action_change"]["mean"]
            for x in levels
        ],
        dtype=float,
    )

    explanation_disagreement = np.array(
        [
            x["explanation_action_disagreement"]["mean"]
            for x in levels
        ],
        dtype=float,
    )

    return (
        noise,
        nn_distance,
        action_change,
        explanation_disagreement,
    )


def correlation_report(x, y, label):
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    return {
        "label": label,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def regression_report(x, y, label):
    fit = linregress(x, y)

    return {
        "label": label,
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r_value": float(fit.rvalue),
        "r_squared": float(fit.rvalue ** 2),
        "p_value": float(fit.pvalue),
        "stderr": float(fit.stderr),
    }


def main():
    parser = argparse.ArgumentParser()

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
            "iql_100k_gaussian_shift_analysis.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print("GAUSSIAN SHIFT STATISTICAL ANALYSIS")
    print("=" * 72)

    data = load_results(args.input)

    (
        noise,
        nn_distance,
        action_change,
        explanation_disagreement,
    ) = extract_level_arrays(data)

    print("\nNoise levels:")
    print(noise)

    print("\nNN distances:")
    print(nn_distance)

    print("\nPolicy action changes:")
    print(action_change)

    print("\nExplanation disagreement:")
    print(explanation_disagreement)

    # ------------------------------------------------------------
    # Primary research relationship
    # OOD distance -> explanation disagreement
    # ------------------------------------------------------------
    primary_corr = correlation_report(
        nn_distance,
        explanation_disagreement,
        "OOD distance vs explanation disagreement",
    )

    primary_regression = regression_report(
        nn_distance,
        explanation_disagreement,
        "OOD distance vs explanation disagreement",
    )

    # ------------------------------------------------------------
    # Secondary relationships
    # ------------------------------------------------------------
    shift_vs_explanation = correlation_report(
        noise,
        explanation_disagreement,
        "noise level vs explanation disagreement",
    )

    shift_vs_policy = correlation_report(
        noise,
        action_change,
        "noise level vs policy action change",
    )

    ood_vs_policy = correlation_report(
        nn_distance,
        action_change,
        "OOD distance vs policy action change",
    )

    # ------------------------------------------------------------
    # Relative degradation from clean state
    # ------------------------------------------------------------
    clean_explanation = float(
        explanation_disagreement[0]
    )

    clean_nn = float(
        nn_distance[0]
    )

    clean_action_change = float(
        action_change[0]
    )

    rows = []

    for i in range(len(noise)):
        explanation_relative_increase = (
            (
                explanation_disagreement[i]
                - clean_explanation
            )
            / clean_explanation
            if clean_explanation != 0
            else None
        )

        nn_relative_increase = (
            (
                nn_distance[i]
                - clean_nn
            )
            / clean_nn
            if clean_nn != 0
            else None
        )

        rows.append(
            {
                "noise_level": float(noise[i]),
                "nn_distance": float(nn_distance[i]),
                "policy_action_change": float(
                    action_change[i]
                ),
                "explanation_disagreement": float(
                    explanation_disagreement[i]
                ),
                "nn_relative_increase": (
                    float(nn_relative_increase)
                    if nn_relative_increase is not None
                    else None
                ),
                "explanation_relative_increase": (
                    float(
                        explanation_relative_increase
                    )
                    if explanation_relative_increase
                    is not None
                    else None
                ),
            }
        )

    analysis = {
        "source_experiment": data["experiment"],
        "task": data["task"],
        "seed": data["seed"],
        "checkpoint": data["checkpoint"],
        "query_observations": data[
            "query_observations"
        ],
        "noise_repeats": data[
            "noise_repeats"
        ],
        "primary_result": {
            "correlation": primary_corr,
            "regression": primary_regression,
        },
        "secondary_results": {
            "noise_vs_explanation": shift_vs_explanation,
            "noise_vs_policy_change": shift_vs_policy,
            "ood_vs_policy_change": ood_vs_policy,
        },
        "clean_baseline": {
            "nn_distance": clean_nn,
            "policy_action_change": clean_action_change,
            "explanation_disagreement": (
                clean_explanation
            ),
        },
        "level_results": rows,
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

    print("\n" + "=" * 72)
    print("PRIMARY RESULT")
    print("=" * 72)

    print(
        "Pearson r:",
        primary_corr["pearson_r"],
    )

    print(
        "Pearson p:",
        primary_corr["pearson_p"],
    )

    print(
        "Spearman rho:",
        primary_corr["spearman_rho"],
    )

    print(
        "Spearman p:",
        primary_corr["spearman_p"],
    )

    print(
        "Regression slope:",
        primary_regression["slope"],
    )

    print(
        "R²:",
        primary_regression["r_squared"],
    )

    print(
        "Regression p:",
        primary_regression["p_value"],
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()