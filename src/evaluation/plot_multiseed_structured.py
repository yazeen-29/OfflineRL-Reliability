from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


SEEDS = [0, 1, 2, 3, 4]

SIGMA_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)

EXPECTED_QUERIES = 1000
EXPECTED_STRUCTURED_RECORDS = 7000
EXPECTED_GAUSSIAN_RECORDS = 35000


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("r") as f:
        return json.load(f)


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# STATISTICS
# ============================================================

def mean_ci(values: np.ndarray):
    """
    Mean and two-sided 95% t CI across independently
    trained policy seeds.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    n = values.shape[0]

    mean = np.mean(
        values,
        axis=0,
    )

    if n < 2:
        return mean, mean, mean

    std = np.std(
        values,
        axis=0,
        ddof=1,
    )

    sem = std / np.sqrt(n)

    critical = t.ppf(
        0.975,
        df=n - 1,
    )

    margin = critical * sem

    return (
        mean,
        mean - margin,
        mean + margin,
    )


# ============================================================
# STRUCTURED ANALYSIS
# ============================================================

def load_structured_curves(
    analysis: dict,
):
    per_seed = analysis["per_seed"]

    explanation_by_seed = np.vstack(
        [
            np.asarray(
                per_seed[str(seed)][
                    "explanation_means"
                ],
                dtype=float,
            )
            for seed in SEEDS
        ]
    )

    nn_by_seed = np.vstack(
        [
            np.asarray(
                per_seed[str(seed)][
                    "nearest_neighbor_means"
                ],
                dtype=float,
            )
            for seed in SEEDS
        ]
    )

    action_by_seed = np.vstack(
        [
            np.asarray(
                per_seed[str(seed)][
                    "policy_action_change_means"
                ],
                dtype=float,
            )
            for seed in SEEDS
        ]
    )

    displacement_by_seed = np.vstack(
        [
            np.asarray(
                per_seed[str(seed)][
                    "standardized_displacement_means"
                ],
                dtype=float,
            )
            for seed in SEEDS
        ]
    )

    return (
        explanation_by_seed,
        nn_by_seed,
        action_by_seed,
        displacement_by_seed,
    )


# ============================================================
# GAUSSIAN RAW DATA
# ============================================================

def gaussian_result_path(
    seed: int,
) -> Path:
    if seed == 0:
        return Path(
            "results/shifts/"
            "iql_100k_gaussian_observation_noise.json"
        )

    return Path(
        "results/shifts/multiseed/"
        f"iql_seed{seed}_"
        "gaussian_observation_noise.json"
    )


def aggregate_gaussian_seed(
    data: dict,
):
    """
    Aggregate raw Gaussian results into seven query-level
    dose-response points.

    Five repeats are averaged within each query/noise cell,
    then the 1000 query means are averaged.
    """
    records = data["records"]

    if len(records) != EXPECTED_GAUSSIAN_RECORDS:
        raise RuntimeError(
            "Unexpected Gaussian record count: "
            f"expected {EXPECTED_GAUSSIAN_RECORDS}, "
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
            f"Unexpected Gaussian query count: "
            f"{len(query_ids)}"
        )

    if not np.allclose(
        levels,
        SIGMA_LEVELS,
    ):
        raise RuntimeError(
            "Gaussian noise-level mismatch.\n"
            f"Expected: {SIGMA_LEVELS.tolist()}\n"
            f"Observed: {levels}"
        )

    if repeat_ids != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            "Gaussian repeat IDs are not [0,1,2,3,4]."
        )

    q_index = {
        q: i
        for i, q in enumerate(query_ids)
    }

    level_index = {
        level: i
        for i, level in enumerate(levels)
    }

    nn = np.zeros(
        (
            len(query_ids),
            len(levels),
        ),
        dtype=float,
    )

    explanation = np.zeros_like(nn)

    counts = np.zeros_like(
        nn,
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
        counts == 5
    ):
        raise RuntimeError(
            "Gaussian query/noise cells do not all "
            "contain exactly five repeats."
        )

    nn /= counts
    explanation /= counts

    return (
        np.mean(nn, axis=0),
        np.mean(explanation, axis=0),
    )


def load_gaussian_cross_seed_means():
    nn_by_seed = []
    explanation_by_seed = []

    for seed in SEEDS:
        path = gaussian_result_path(seed)

        print(
            f"Loading Gaussian seed {seed}: {path}"
        )

        data = load_json(path)

        nn_mean, explanation_mean = (
            aggregate_gaussian_seed(data)
        )

        nn_by_seed.append(nn_mean)
        explanation_by_seed.append(
            explanation_mean
        )

    nn_by_seed = np.vstack(
        nn_by_seed
    )

    explanation_by_seed = np.vstack(
        explanation_by_seed
    )

    return (
        mean_ci(nn_by_seed),
        mean_ci(explanation_by_seed),
        nn_by_seed,
        explanation_by_seed,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final publication-quality figures for the "
            "five-seed structured observation-shift study."
        )
    )

    parser.add_argument(
        "--analysis",
        default=(
            "results/analysis/"
            "iql_100k_multiseed_structured_analysis.json"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "paper/figures/structured"
        ),
    )

    args = parser.parse_args()

    analysis_path = Path(
        args.analysis
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "FINAL PAPER-QUALITY STRUCTURED FIGURE GENERATION"
    )
    print("=" * 80)

    analysis = load_json(
        analysis_path
    )

    (
        explanation_by_seed,
        nn_by_seed,
        action_by_seed,
        displacement_by_seed,
    ) = load_structured_curves(
        analysis
    )

    # --------------------------------------------------------
    # Cross-seed structured summaries
    # --------------------------------------------------------

    (
        structured_explanation_mean,
        structured_explanation_low,
        structured_explanation_high,
    ) = mean_ci(
        explanation_by_seed
    )

    (
        structured_nn_mean,
        structured_nn_low,
        structured_nn_high,
    ) = mean_ci(
        nn_by_seed
    )

    (
        structured_action_mean,
        structured_action_low,
        structured_action_high,
    ) = mean_ci(
        action_by_seed
    )

    (
        structured_displacement_mean,
        structured_displacement_low,
        structured_displacement_high,
    ) = mean_ci(
        displacement_by_seed
    )

    slope_summary = (
        analysis["cross_seed"][
            "explanation_dose_response_slope"
        ]
    )

    sign_test = (
        analysis["cross_seed"][
            "exact_sign_flip_test"
        ]
    )

    # ========================================================
    # FIGURE 1
    # Main structured relationship
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.4, 5.6)
    )

    for i, seed in enumerate(SEEDS):
        order = np.argsort(
            nn_by_seed[i]
        )

        ax.plot(
            nn_by_seed[i][order],
            explanation_by_seed[i][order],
            marker="o",
            markersize=4.2,
            linewidth=1.25,
            alpha=0.48,
            label=f"Seed {seed}",
        )

    order = np.argsort(
        structured_nn_mean
    )

    ax.fill_between(
        structured_nn_mean[order],
        structured_explanation_low[order],
        structured_explanation_high[order],
        alpha=0.16,
        label="95% CI",
    )

    ax.plot(
        structured_nn_mean[order],
        structured_explanation_mean[order],
        marker="o",
        markersize=5.5,
        linewidth=2.8,
        label="Across-seed mean",
    )

    ax.set_xlabel(
        "Nearest-neighbor distance\n"
        "(standardized observation space)"
    )

    ax.set_ylabel(
        "Explanation action disagreement (RMS)"
    )

    ax.set_title(
        "Structured directional shift and\n"
        "explanation disagreement",
        fontsize=13,
        pad=10,
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax.text(
        0.98,
        0.035,
        (
            f"5 independently trained IQL policies\n"
            f"Mean slope = "
            f"{slope_summary['mean']:.3f}\n"
            f"95% CI = "
            f"[{slope_summary['ci95_low']:.3f}, "
            f"{slope_summary['ci95_high']:.3f}]\n"
            f"One-sided exact p = "
            f"{sign_test['one_sided_p']:.5f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
    )

    ax.legend(
        frameon=False,
        fontsize=8.3,
        ncol=2,
        loc="upper left",
        handlelength=2.2,
        columnspacing=1.0,
    )

    ax.margins(
        x=0.025,
        y=0.08,
    )

    save_figure(
        fig,
        output_dir,
        "figure1_structured_nn_vs_explanation",
    )

    # ========================================================
    # FIGURE 2
    # Three-panel structured dose response
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.8),
        sharex=True,
    )

    panels = [
        (
            structured_nn_mean,
            structured_nn_low,
            structured_nn_high,
            "Distributional distance",
            "Nearest-neighbor distance",
            "o",
        ),
        (
            structured_action_mean,
            structured_action_low,
            structured_action_high,
            "Policy response",
            "Policy action change (RMS)",
            "s",
        ),
        (
            structured_explanation_mean,
            structured_explanation_low,
            structured_explanation_high,
            "Explanation response",
            "Explanation action disagreement (RMS)",
            "^",
        ),
    ]

    for panel_index, (
        ax,
        (
            mean,
            low,
            high,
            title,
            ylabel,
            marker,
        ),
    ) in enumerate(
        zip(axes, panels)
    ):

        ax.plot(
            SIGMA_LEVELS,
            mean,
            marker=marker,
            markersize=5.0,
            linewidth=2.3,
        )

        ax.fill_between(
            SIGMA_LEVELS,
            low,
            high,
            alpha=0.16,
        )

        panel_label = chr(
            ord("a") + panel_index
        )

        ax.text(
            0.015,
            0.90,
            f"({panel_label}) {title}",
            transform=ax.transAxes,
            fontsize=10.8,
            fontweight="bold",
            va="top",
        )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            alpha=0.18,
            linewidth=0.7,
        )

        ax.margins(
            x=0.025,
            y=0.12,
        )

    axes[-1].set_xlabel(
        "Standardized directional shift magnitude"
    )

    axes[-1].set_xticks(
        [0.0, 0.05, 0.10, 0.20, 0.30]
    )

    axes[-1].set_xticklabels(
        [
            "0",
            "0.05",
            "0.10",
            "0.20",
            "0.30",
        ]
    )

    fig.suptitle(
        "Response to structured directional observation shift",
        fontsize=13,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.96,
        top=0.94,
        bottom=0.08,
        hspace=0.30,
    )

    save_figure(
        fig,
        output_dir,
        "figure2_structured_dose_response",
    )

    # ========================================================
    # FIGURE 3
    # Gaussian vs structured robustness
    # ========================================================

    print()
    print(
        "Loading frozen Gaussian raw results..."
    )

    (
        (
            gaussian_nn_mean,
            gaussian_nn_low,
            gaussian_nn_high,
        ),
        (
            gaussian_explanation_mean,
            gaussian_explanation_low,
            gaussian_explanation_high,
        ),
        gaussian_nn_by_seed,
        gaussian_explanation_by_seed,
    ) = load_gaussian_cross_seed_means()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.7),
        sharey=True,
    )

    # --------------------------------------------------------
    # Gaussian panel
    # --------------------------------------------------------

    ax = axes[0]

    for i, seed in enumerate(SEEDS):
        order = np.argsort(
            gaussian_nn_by_seed[i]
        )

        ax.plot(
            gaussian_nn_by_seed[i][order],
            gaussian_explanation_by_seed[i][order],
            marker="o",
            markersize=3.3,
            linewidth=1.0,
            alpha=0.30,
        )

    order = np.argsort(
        gaussian_nn_mean
    )

    ax.fill_between(
        gaussian_nn_mean[order],
        gaussian_explanation_low[order],
        gaussian_explanation_high[order],
        alpha=0.16,
    )

    ax.plot(
        gaussian_nn_mean[order],
        gaussian_explanation_mean[order],
        marker="o",
        markersize=5.2,
        linewidth=2.5,
        label="Across-seed mean",
    )

    ax.set_title(
        "(a) Gaussian observation shift",
        fontsize=11.5,
        pad=8,
    )

    ax.set_xlabel(
        "Nearest-neighbor distance\n"
        "(standardized observation space)"
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax.text(
        0.97,
        0.05,
        "5 policy seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )

    # --------------------------------------------------------
    # Structured panel
    # --------------------------------------------------------

    ax = axes[1]

    for i, seed in enumerate(SEEDS):
        order = np.argsort(
            nn_by_seed[i]
        )

        ax.plot(
            nn_by_seed[i][order],
            explanation_by_seed[i][order],
            marker="o",
            markersize=3.3,
            linewidth=1.0,
            alpha=0.30,
        )

    order = np.argsort(
        structured_nn_mean
    )

    ax.fill_between(
        structured_nn_mean[order],
        structured_explanation_low[order],
        structured_explanation_high[order],
        alpha=0.16,
    )

    ax.plot(
        structured_nn_mean[order],
        structured_explanation_mean[order],
        marker="o",
        markersize=5.2,
        linewidth=2.5,
        label="Across-seed mean",
    )

    ax.set_title(
        "(b) Structured directional shift",
        fontsize=11.5,
        pad=8,
    )

    ax.set_xlabel(
        "Nearest-neighbor distance\n"
        "(standardized observation space)"
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax.text(
        0.97,
        0.05,
        "5 policy seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )

    axes[0].set_ylabel(
        "Explanation action disagreement (RMS)"
    )

    # --------------------------------------------------------
    # Shared y-range for direct visual comparison
    # --------------------------------------------------------

    x_min = min(
        np.min(gaussian_nn_mean),
        np.min(structured_nn_mean),
    )

    x_max = max(
        np.max(gaussian_nn_mean),
        np.max(structured_nn_mean),
    )

    y_min = min(
        np.min(gaussian_explanation_low),
        np.min(structured_explanation_low),
    )

    y_max = max(
        np.max(gaussian_explanation_high),
        np.max(structured_explanation_high),
    )

    x_margin = 0.03 * (
        x_max - x_min
    )

    y_margin = 0.05 * (
        y_max - y_min
    )

    for ax in axes:
        ax.set_xlim(
            x_min - x_margin,
            x_max + x_margin,
        )

        ax.set_ylim(
            y_min - y_margin,
            y_max + y_margin,
        )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D(
            [0],
            [0],
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Across-seed mean",
        ),
        Patch(
            alpha=0.16,
            label="95% CI",
        ),
        Line2D(
            [0],
            [0],
            linewidth=1.0,
            alpha=0.30,
            marker="o",
            markersize=3,
            label="Individual policy seeds",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.suptitle(
        "Robustness across observation-shift mechanisms",
        fontsize=13,
        y=0.99,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.84,
        bottom=0.25,
        wspace=0.08,
    )

    save_figure(
        fig,
        output_dir,
        "figure3_gaussian_vs_structured",
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "analysis_source": str(
            analysis_path
        ),
        "gaussian_sources": {
            str(seed): str(
                gaussian_result_path(seed)
            )
            for seed in SEEDS
        },
        "policy_seeds": SEEDS,
        "sigma_levels": (
            SIGMA_LEVELS.tolist()
        ),
        "figures": [
            "figure1_structured_nn_vs_explanation.png",
            "figure1_structured_nn_vs_explanation.pdf",
            "figure2_structured_dose_response.png",
            "figure2_structured_dose_response.pdf",
            "figure3_gaussian_vs_structured.png",
            "figure3_gaussian_vs_structured.pdf",
        ],
        "structured_statistics": {
            "mean_explanation_slope": float(
                slope_summary["mean"]
            ),
            "ci95_low": float(
                slope_summary["ci95_low"]
            ),
            "ci95_high": float(
                slope_summary["ci95_high"]
            ),
            "exact_one_sided_p": float(
                sign_test["one_sided_p"]
            ),
            "exact_two_sided_p": float(
                sign_test["two_sided_p"]
            ),
        },
        "figure2_axis_note": (
            "The x-axis is standardized directional "
            "shift magnitude. Shift magnitudes are calibrated "
            "so their standardized L2 displacement matches "
            "the expected magnitude of the corresponding "
            "Gaussian perturbation."
        ),
        "figure3_note": (
            "The Gaussian and structured panels are shown "
            "as a robustness comparison. The panels use "
            "matched axes and should not be interpreted as "
            "identical causal effect-size scales."
        ),
    }

    metadata_path = (
        output_dir
        / "structured_figure_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL PAPER-QUALITY STRUCTURED FIGURES GENERATED"
    )
    print("=" * 80)

    print(
        "Output:",
        output_dir,
    )

    print(
        "Mean explanation slope:",
        slope_summary["mean"],
    )

    print(
        "95% CI:",
        (
            slope_summary["ci95_low"],
            slope_summary["ci95_high"],
        ),
    )

    print(
        "Exact one-sided p:",
        sign_test["one_sided_p"],
    )

    print()
    print(
        "Generated:"
    )

    print(
        "  Figure 1: structured relationship"
    )

    print(
        "  Figure 2: structured dose response"
    )

    print(
        "  Figure 3: Gaussian vs structured robustness"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()