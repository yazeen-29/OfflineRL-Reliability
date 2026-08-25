from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


# ============================================================
# LOCKED EXPERIMENT CONFIGURATION
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
    * len(NOISE_LEVELS)
    * EXPECTED_REPEATS
)


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
) -> None:
    """Save both raster and vector versions."""
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
# SOURCE PATHS
# ============================================================

def raw_seed_path(
    seed: int,
    seed0_path: Path,
    multiseed_dir: Path,
) -> Path:
    if seed == 0:
        return seed0_path

    return (
        multiseed_dir
        / f"iql_seed{seed}_gaussian_observation_noise.json"
    )


# ============================================================
# RAW DATA AGGREGATION
# ============================================================

def aggregate_raw_seed(
    data: dict,
) -> dict:
    """
    Validate one raw Gaussian-shift experiment and compute
    query-averaged dose-response metrics for that policy seed.

    Five repeats are first averaged within each
    query/noise-level cell; query means are then averaged
    to obtain one policy-seed dose-response curve.
    """

    records = data.get(
        "records",
        [],
    )

    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            "Unexpected record count: "
            f"expected {EXPECTED_RECORDS}, "
            f"got {len(records)}"
        )

    query_ids = sorted(
        {
            int(r["query_id"])
            for r in records
        }
    )

    if len(query_ids) != EXPECTED_QUERIES:
        raise RuntimeError(
            "Unexpected query count: "
            f"expected {EXPECTED_QUERIES}, "
            f"got {len(query_ids)}"
        )

    noise_levels = sorted(
        {
            float(r["noise_level"])
            for r in records
        }
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

    repeat_ids = sorted(
        {
            int(r["repeat_id"])
            for r in records
        }
    )

    if repeat_ids != list(
        range(EXPECTED_REPEATS)
    ):
        raise RuntimeError(
            "Unexpected repeat IDs.\n"
            f"Expected: {list(range(EXPECTED_REPEATS))}\n"
            f"Observed: {repeat_ids}"
        )

    q_index = {
        q: i
        for i, q in enumerate(query_ids)
    }

    level_index = {
        level: i
        for i, level in enumerate(noise_levels)
    }

    nn = np.full(
        (
            len(query_ids),
            len(noise_levels),
        ),
        np.nan,
        dtype=float,
    )

    action = np.full(
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

        if np.isnan(nn[i, j]):
            nn[i, j] = 0.0
            action[i, j] = 0.0
            explanation[i, j] = 0.0

        nn[i, j] += float(
            record[
                "nearest_neighbor_distance"
            ]
        )

        action[i, j] += float(
            record[
                "policy_action_change"
            ]
        )

        explanation[i, j] += float(
            record[
                "explanation_action_disagreement"
            ]
        )

        counts[i, j] += 1

    if not np.all(
        counts == EXPECTED_REPEATS
    ):
        bad_cells = np.argwhere(
            counts != EXPECTED_REPEATS
        )

        raise RuntimeError(
            "Not every query/noise-level cell contains "
            f"exactly {EXPECTED_REPEATS} repeats. "
            f"Example bad cells: {bad_cells[:10].tolist()}"
        )

    valid = counts > 0

    nn[valid] /= counts[valid]
    action[valid] /= counts[valid]
    explanation[valid] /= counts[valid]

    return {
        "noise_levels": np.asarray(
            noise_levels,
            dtype=float,
        ),
        "nn": np.mean(
            nn,
            axis=0,
        ),
        "action": np.mean(
            action,
            axis=0,
        ),
        "explanation": np.mean(
            explanation,
            axis=0,
        ),
    }


# ============================================================
# CROSS-SEED CONFIDENCE INTERVAL
# ============================================================

def mean_ci_across_seeds(
    values: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Mean and two-sided 95% t-based CI across the five
    independently trained policy seeds.
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
        return (
            mean,
            mean,
            mean,
        )

    std = np.std(
        values,
        axis=0,
        ddof=1,
    )

    sem = (
        std
        / np.sqrt(n)
    )

    critical = t.ppf(
        0.975,
        df=n - 1,
    )

    margin = (
        critical
        * sem
    )

    return (
        mean,
        mean - margin,
        mean + margin,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality figures for "
            "the five-seed IQL Gaussian observation-shift study."
        )
    )

    parser.add_argument(
        "--analysis",
        default=(
            "results/analysis/multiseed/"
            "iql_100k_multiseed_gaussian_analysis_v2.json"
        ),
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
        "--output_dir",
        default=(
            "paper/figures/multiseed"
        ),
    )

    args = parser.parse_args()

    analysis_path = Path(
        args.analysis
    )

    seed0_path = Path(
        args.seed0
    )

    multiseed_dir = Path(
        args.multiseed_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {analysis_path}"
        )

    print("=" * 80)
    print(
        "PAPER-QUALITY MULTI-SEED GAUSSIAN FIGURE GENERATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load frozen statistical analysis
    # --------------------------------------------------------

    analysis = load_json(
        analysis_path
    )

    cross_seed = analysis[
        "cross_seed"
    ]

    seed_slopes = np.asarray(
        cross_seed[
            "seed_level_slope_values"
        ],
        dtype=float,
    )

    seed_positive = np.asarray(
        cross_seed[
            "seed_level_positive_fraction_values"
        ],
        dtype=float,
    )

    mean_slope = float(
        cross_seed[
            "mean_query_slope"
        ]["mean"]
    )

    slope_ci_low = float(
        cross_seed[
            "mean_query_slope"
        ]["ci95_low"]
    )

    slope_ci_high = float(
        cross_seed[
            "mean_query_slope"
        ]["ci95_high"]
    )

    sign_flip = cross_seed[
        "exact_sign_flip_test"
    ]

    # --------------------------------------------------------
    # Load all five raw experiments
    # --------------------------------------------------------

    seed_curves = {}

    for seed in POLICY_SEEDS:

        path = raw_seed_path(
            seed,
            seed0_path,
            multiseed_dir,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw result for seed {seed}: {path}"
            )

        print(
            f"Loading seed {seed}: {path}"
        )

        data = load_json(
            path
        )

        seed_curves[seed] = (
            aggregate_raw_seed(
                data
            )
        )

    # --------------------------------------------------------
    # Cross-seed arrays
    # --------------------------------------------------------

    nn_by_seed = np.vstack(
        [
            seed_curves[seed]["nn"]
            for seed in POLICY_SEEDS
        ]
    )

    action_by_seed = np.vstack(
        [
            seed_curves[seed]["action"]
            for seed in POLICY_SEEDS
        ]
    )

    explanation_by_seed = np.vstack(
        [
            seed_curves[seed]["explanation"]
            for seed in POLICY_SEEDS
        ]
    )

    noise = seed_curves[0][
        "noise_levels"
    ]

    (
        nn_mean,
        nn_low,
        nn_high,
    ) = mean_ci_across_seeds(
        nn_by_seed
    )

    (
        action_mean,
        action_low,
        action_high,
    ) = mean_ci_across_seeds(
        action_by_seed
    )

    (
        explanation_mean,
        explanation_low,
        explanation_high,
    ) = mean_ci_across_seeds(
        explanation_by_seed
    )

    # ========================================================
    # FIGURE 1
    # Main cross-seed relationship
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.4, 5.6)
    )

    for seed_idx, seed in enumerate(
        POLICY_SEEDS
    ):
        x = nn_by_seed[
            seed_idx
        ]

        y = explanation_by_seed[
            seed_idx
        ]

        order = np.argsort(
            x
        )

        ax.plot(
            x[order],
            y[order],
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            alpha=0.50,
            label=f"Seed {seed}",
        )

    order = np.argsort(
        nn_mean
    )

    x_sorted = nn_mean[
        order
    ]

    y_sorted = explanation_mean[
        order
    ]

    low_sorted = explanation_low[
        order
    ]

    high_sorted = explanation_high[
        order
    ]

    ax.fill_between(
        x_sorted,
        low_sorted,
        high_sorted,
        alpha=0.18,
        label="95% CI",
    )

    ax.plot(
        x_sorted,
        y_sorted,
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
        "Explanation disagreement increases with "
        "nearest-neighbor distance"
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=2,
        loc="upper left",
    )

    annotation = (
        f"5 independently trained IQL policies\n"
        f"Mean slope = {mean_slope:.3f}\n"
        f"95% CI = "
        f"[{slope_ci_low:.3f}, "
        f"{slope_ci_high:.3f}]"
    )

    ax.text(
        0.98,
        0.03,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    save_figure(
        fig,
        output_dir,
        "figure1_cross_seed_ood_vs_explanation",
    )

    # ========================================================
    # FIGURE 2
    # Replication across policy seeds
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.4, 5.2)
    )

    # Seed points occupy x = 0..4.
    seed_x = np.arange(
        len(POLICY_SEEDS),
        dtype=float,
    )

    # Aggregate result occupies a separate x position.
    aggregate_x = 5.25

    ax.scatter(
        seed_x,
        seed_slopes,
        s=70,
        zorder=4,
        label="Policy-seed slope",
    )

    ax.axhline(
        mean_slope,
        linestyle="--",
        linewidth=2.0,
        label="Cross-seed mean",
    )

    ax.errorbar(
        aggregate_x,
        mean_slope,
        yerr=np.array(
            [
                mean_slope - slope_ci_low,
                slope_ci_high - mean_slope,
            ]
        ).reshape(2, 1),
        fmt="o",
        markersize=8,
        capsize=7,
        linewidth=2.0,
        label="Aggregate mean ± 95% CI",
        zorder=5,
    )

    ax.set_xticks(
        np.append(
            seed_x,
            aggregate_x,
        ),
        [
            "Seed 0",
            "Seed 1",
            "Seed 2",
            "Seed 3",
            "Seed 4",
            "Aggregate",
        ],
    )

    ax.set_ylabel(
        "Per-seed mean query-level degradation slope"
    )

    ax.set_xlabel(
        "Independent policy seed"
    )

    ax.set_title(
        "Replication across independently trained IQL policies"
    )

    y_min = float(
        np.min(
            seed_slopes
        )
    )

    y_max = float(
        np.max(
            seed_slopes
        )
    )

    y_range = max(
        y_max - y_min,
        0.005,
    )

    ax.set_ylim(
        y_min - 0.20 * y_range,
        y_max + 0.90 * y_range,
    )

    ax.text(
        0.02,
        0.97,
        (
            f"Mean = {mean_slope:.3f}\n"
            f"95% CI = "
            f"[{slope_ci_low:.3f}, "
            f"{slope_ci_high:.3f}]\n"
            f"Exact one-sided sign-flip p = "
            f"{sign_flip['one_sided_p']:.5f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.7,
    )

    # Avoid excessive empty space between Seed 4 and Aggregate.
    ax.set_xlim(
        -0.5,
        5.85,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8.5,
    )

    save_figure(
        fig,
        output_dir,
        "figure2_cross_seed_slopes",
    )

    # ========================================================
    # FIGURE 3
    # Three-panel dose-response
    # ========================================================

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.4, 9.0),
        sharex=True,
    )

    # --------------------------------------------------------
    # Panel A: NN distance
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        noise,
        nn_mean,
        marker="o",
        markersize=5,
        linewidth=2.4,
    )

    ax.fill_between(
        noise,
        nn_low,
        nn_high,
        alpha=0.18,
    )

    ax.set_ylabel(
        "NN distance"
    )

    ax.set_title(
        "Distributional shift and policy/explanation response",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    # --------------------------------------------------------
    # Panel B: Policy action change
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        noise,
        action_mean,
        marker="s",
        markersize=5,
        linewidth=2.4,
    )

    ax.fill_between(
        noise,
        action_low,
        action_high,
        alpha=0.18,
    )

    ax.set_ylabel(
        "Policy action change\n"
        "(RMS)"
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    # --------------------------------------------------------
    # Panel C: Explanation disagreement
    # --------------------------------------------------------

    ax = axes[2]

    ax.plot(
        noise,
        explanation_mean,
        marker="^",
        markersize=5,
        linewidth=2.4,
    )

    ax.fill_between(
        noise,
        explanation_low,
        explanation_high,
        alpha=0.18,
    )

    ax.set_ylabel(
        "Explanation action\n"
        "disagreement (RMS)"
    )

    ax.set_xlabel(
        "Gaussian observation-noise magnitude"
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    # --------------------------------------------------------
    # Clean readable x-axis
    #
    # All seven experimental levels are still plotted.
    # Only the displayed major labels are simplified.
    # --------------------------------------------------------

    display_ticks = np.array(
        [0.00, 0.05, 0.10, 0.20, 0.30],
        dtype=float,
    )

    axes[2].set_xticks(
        display_ticks
    )

    axes[2].set_xticklabels(
        [
            "0",
            "0.05",
            "0.10",
            "0.20",
            "0.30",
        ]
    )

    fig.subplots_adjust(
        hspace=0.28
    )

    save_figure(
        fig,
        output_dir,
        "figure3_cross_seed_dose_response",
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "analysis_source": str(
            analysis_path
        ),
        "raw_seed_sources": {
            str(seed): str(
                raw_seed_path(
                    seed,
                    seed0_path,
                    multiseed_dir,
                )
            )
            for seed in POLICY_SEEDS
        },
        "policy_seeds": POLICY_SEEDS,
        "queries_per_seed": EXPECTED_QUERIES,
        "repeats_per_query_level": EXPECTED_REPEATS,
        "records_per_seed": EXPECTED_RECORDS,
        "noise_levels": [
            float(x)
            for x in noise
        ],
        "figure_files": [
            "figure1_cross_seed_ood_vs_explanation.png",
            "figure1_cross_seed_ood_vs_explanation.pdf",
            "figure2_cross_seed_slopes.png",
            "figure2_cross_seed_slopes.pdf",
            "figure3_cross_seed_dose_response.png",
            "figure3_cross_seed_dose_response.pdf",
        ],
        "cross_seed_statistics": {
            "mean_slope": mean_slope,
            "slope_ci95_low": slope_ci_low,
            "slope_ci95_high": slope_ci_high,
            "exact_one_sided_sign_flip_p": float(
                sign_flip["one_sided_p"]
            ),
            "exact_two_sided_sign_flip_p": float(
                sign_flip["two_sided_p"]
            ),
        },
        "plotting_convention": {
            "individual_seed_curves": (
                "shown for replication transparency"
            ),
            "aggregate_uncertainty": (
                "95% t-based CI across the five "
                "independent policy seeds"
            ),
            "figure3": (
                "three panels use independent y-axes "
                "because the metrics have different "
                "numerical scales"
            ),
        },
    }

    metadata_path = (
        output_dir
        / "multiseed_figure_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print(
        "FINAL PAPER-QUALITY MULTI-SEED FIGURES GENERATED"
    )
    print("=" * 80)

    print(
        "Output:",
        output_dir,
    )

    print(
        "Mean slope:",
        mean_slope,
    )

    print(
        "95% CI:",
        (
            slope_ci_low,
            slope_ci_high,
        ),
    )

    print(
        "Exact one-sided sign-flip p:",
        sign_flip["one_sided_p"],
    )

    print(
        "Exact two-sided sign-flip p:",
        sign_flip["two_sided_p"],
    )

    print()
    print(
        "Generated:"
    )

    print(
        "  Figure 1: cross-seed relationship"
    )

    print(
        "  Figure 2: policy-seed replication"
    )

    print(
        "  Figure 3: three-panel dose response"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()