from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    n_bootstrap: int = 2000,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    estimates = np.empty(
        n_bootstrap,
        dtype=float,
    )

    n = len(values)

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )
        estimates[i] = np.mean(sample)

    return (
        float(np.mean(values)),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def query_level_matrices(records):
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

    q_index = {
        q: i
        for i, q in enumerate(query_ids)
    }

    l_index = {
        level: i
        for i, level in enumerate(noise_levels)
    }

    n_q = len(query_ids)
    n_l = len(noise_levels)

    nn = np.full(
        (n_q, n_l),
        np.nan,
    )

    explanation = np.full(
        (n_q, n_l),
        np.nan,
    )

    counts = np.zeros(
        (n_q, n_l),
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

    return (
        np.asarray(query_ids),
        np.asarray(noise_levels),
        nn,
        explanation,
        counts,
    )


def per_query_slopes(nn, explanation):
    slopes = []

    for i in range(nn.shape[0]):
        x = nn[i]
        y = explanation[i]

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if np.sum(valid) < 3:
            continue

        coeff = np.polyfit(
            x[valid],
            y[valid],
            1,
        )

        slopes.append(coeff[0])

    return np.asarray(
        slopes,
        dtype=float,
    )


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
        "--analysis",
        default=(
            "results/analysis/"
            "iql_100k_gaussian_shift_analysis_v3.json"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="paper/figures",
    )

    args = parser.parse_args()

    data = load_json(args.input)
    analysis = load_json(args.analysis)

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    levels = analysis[
        "holm_corrected_level_results"
    ]

    noise = np.array(
        [
            r["noise_level"]
            for r in levels
        ],
        dtype=float,
    )

    nn_mean = np.array(
        [
            r["nn_distance"]["estimate"]
            for r in levels
        ],
        dtype=float,
    )

    nn_low = np.array(
        [
            r["nn_distance"]["ci95_low"]
            for r in levels
        ],
        dtype=float,
    )

    nn_high = np.array(
        [
            r["nn_distance"]["ci95_high"]
            for r in levels
        ],
        dtype=float,
    )

    action_mean = np.array(
        [
            r["policy_action_change"]["estimate"]
            for r in levels
        ],
        dtype=float,
    )

    action_low = np.array(
        [
            r["policy_action_change"]["ci95_low"]
            for r in levels
        ],
        dtype=float,
    )

    action_high = np.array(
        [
            r["policy_action_change"]["ci95_high"]
            for r in levels
        ],
        dtype=float,
    )

    explanation_mean = np.array(
        [
            r["explanation_disagreement"]["estimate"]
            for r in levels
        ],
        dtype=float,
    )

    explanation_low = np.array(
        [
            r["explanation_disagreement"]["ci95_low"]
            for r in levels
        ],
        dtype=float,
    )

    explanation_high = np.array(
        [
            r["explanation_disagreement"]["ci95_high"]
            for r in levels
        ],
        dtype=float,
    )

    # ---------------------------------------------------------
    # FIGURE 1
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        noise,
        nn_mean,
        marker="o",
        linewidth=2,
    )

    plt.fill_between(
        noise,
        nn_low,
        nn_high,
        alpha=0.20,
    )

    plt.xlabel(
        "Normalized observation noise level"
    )

    plt.ylabel(
        "Nearest-neighbor distance\n"
        "(standardized space)"
    )

    plt.tight_layout()

    plt.savefig(
        out / "figure1_noise_vs_ood_distance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        out / "figure1_noise_vs_ood_distance.pdf",
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # FIGURE 2
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        noise,
        action_mean,
        marker="o",
        linewidth=2,
    )

    plt.fill_between(
        noise,
        action_low,
        action_high,
        alpha=0.20,
    )

    plt.xlabel(
        "Normalized observation noise level"
    )

    plt.ylabel(
        "Policy action change (RMS)"
    )

    plt.tight_layout()

    plt.savefig(
        out / "figure2_noise_vs_policy_change.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        out / "figure2_noise_vs_policy_change.pdf",
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # FIGURE 3
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        noise,
        explanation_mean,
        marker="o",
        linewidth=2,
    )

    plt.fill_between(
        noise,
        explanation_low,
        explanation_high,
        alpha=0.20,
    )

    plt.xlabel(
        "Normalized observation noise level"
    )

    plt.ylabel(
        "Explanation action disagreement (RMS)"
    )

    plt.tight_layout()

    plt.savefig(
        out / "figure3_noise_vs_explanation_disagreement.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        out / "figure3_noise_vs_explanation_disagreement.pdf",
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # BUILD QUERY-LEVEL MATRICES
    # ---------------------------------------------------------

    (
        query_ids,
        noise_levels,
        nn_matrix,
        explanation_matrix,
        counts,
    ) = query_level_matrices(
        data["records"]
    )

    if not np.array_equal(
        np.unique(counts),
        np.array([5]),
    ):
        raise RuntimeError(
            "Expected exactly five repeats per "
            "query/noise-level cell."
        )

    # ---------------------------------------------------------
    # FIGURE 4
    #
    # All query-level observations summarized into bins.
    # ---------------------------------------------------------

    clean_idx = np.where(
        np.isclose(
            noise_levels,
            0.0,
        )
    )[0]

    if len(clean_idx) != 1:
        raise RuntimeError(
            "Expected exactly one clean noise level."
        )

    # Flatten query x level data.
    x = nn_matrix.reshape(-1)
    y = explanation_matrix.reshape(-1)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    # Distance bins based on quantiles.
    n_bins = 10

    edges = np.quantile(
        x,
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        ),
    )

    edges = np.unique(edges)

    bin_x = []
    bin_y = []
    bin_low = []
    bin_high = []

    for i in range(
        len(edges) - 1
    ):
        if i == len(edges) - 2:
            mask = (
                (x >= edges[i])
                & (x <= edges[i + 1])
            )
        else:
            mask = (
                (x >= edges[i])
                & (x < edges[i + 1])
            )

        if np.sum(mask) < 20:
            continue

        x_bin = x[mask]
        y_bin = y[mask]

        _, low, high = bootstrap_mean_ci(
            y_bin,
            seed=9000 + i,
        )

        bin_x.append(
            float(np.mean(x_bin))
        )

        bin_y.append(
            float(np.mean(y_bin))
        )

        bin_low.append(low)
        bin_high.append(high)

    bin_x = np.asarray(bin_x)
    bin_y = np.asarray(bin_y)
    bin_low = np.asarray(bin_low)
    bin_high = np.asarray(bin_high)

    primary = analysis[
        "primary_query_slope_analysis"
    ]

    slope = primary[
        "mean_query_slope"
    ]["estimate"]

    slope_low = primary[
        "mean_query_slope"
    ]["ci95_low"]

    slope_high = primary[
        "mean_query_slope"
    ]["ci95_high"]

    positive_fraction = (
        primary[
            "positive_slope_fraction"
        ]
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        bin_x,
        bin_y,
        marker="o",
        linewidth=2,
    )

    plt.fill_between(
        bin_x,
        bin_low,
        bin_high,
        alpha=0.20,
    )

    plt.xlabel(
        "Nearest-neighbor distance\n"
        "(standardized space)"
    )

    plt.ylabel(
        "Explanation action disagreement (RMS)"
    )

    annotation = (
        f"N = {len(query_ids):,} queries\n"
        f"Positive slopes = "
        f"{100 * positive_fraction:.1f}%\n"
        f"Mean query slope = {slope:.3f}\n"
        f"95% bootstrap CI = "
        f"[{slope_low:.3f}, {slope_high:.3f}]"
    )

    plt.text(
        0.03,
        0.97,
        annotation,
        transform=plt.gca().transAxes,
        va="top",
    )

    plt.tight_layout()

    plt.savefig(
        out / "figure4_ood_vs_explanation_disagreement.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        out / "figure4_ood_vs_explanation_disagreement.pdf",
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # FIGURE 5
    # ---------------------------------------------------------

    slopes = per_query_slopes(
        nn_matrix,
        explanation_matrix,
    )

    slopes = slopes[
        np.isfinite(slopes)
    ]

    positive_fraction = float(
        np.mean(
            slopes > 0
        )
    )

    plt.figure(figsize=(7, 5))

    plt.hist(
        slopes,
        bins=35,
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel(
        "Per-query degradation slope"
    )

    plt.ylabel(
        "Number of queries"
    )

    annotation = (
        f"N = {len(slopes):,}\n"
        f"Positive slopes = "
        f"{100 * positive_fraction:.1f}%\n"
        f"Mean slope = {slope:.3f}\n"
        f"95% CI = "
        f"[{slope_low:.3f}, {slope_high:.3f}]"
    )

    plt.text(
        0.97,
        0.97,
        annotation,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
    )

    plt.tight_layout()

    plt.savefig(
        out / "figure5_per_query_slopes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        out / "figure5_per_query_slopes.pdf",
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "source_shift_results": args.input,
        "source_analysis": args.analysis,
        "query_count": int(
            len(query_ids)
        ),
        "noise_levels": [
            float(x)
            for x in noise_levels
        ],
        "figures": [
            "figure1_noise_vs_ood_distance",
            "figure2_noise_vs_policy_change",
            "figure3_noise_vs_explanation_disagreement",
            "figure4_ood_vs_explanation_disagreement",
            "figure5_per_query_slopes",
        ],
    }

    (
        out / "figure_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "Generated figures in:",
        out,
    )


if __name__ == "__main__":
    main()