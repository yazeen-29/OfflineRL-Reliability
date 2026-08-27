from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# PUBLICATION SETTINGS
# =============================================================================

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,

        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,

        "legend.fontsize": 8.5,

        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "grid.linewidth": 0.6,
        "grid.alpha": 0.20,

        "legend.frameon": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)


# =============================================================================
# INPUTS
# =============================================================================

HOPPER = Path(
    "results/analysis/multiseed/"
    "iql_100k_multiseed_gaussian_analysis_v2.json"
)

HALFCHEETAH = Path(
    "results/analysis/halfcheetah/"
    "iql_100k_multiseed_gaussian_analysis.json"
)

SYNTHESIS = Path(
    "results/analysis/"
    "cross_environment_gaussian_synthesis.json"
)

OUTPUT_DIR = Path(
    "paper/figures/cross_environment"
)

NOISE_LEVELS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )

    return json.loads(
        path.read_text()
    )


def extract_environment(
    data: dict,
    name: str,
) -> dict:

    levels = np.asarray(
        data["noise_levels"],
        dtype=float,
    )

    if not np.allclose(
        levels,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            f"{name}: unexpected noise levels: "
            f"{levels.tolist()}"
        )

    cross = data["cross_seed"]

    dose = cross[
        "noise_level_results"
    ]

    nn_mean = []
    nn_low = []
    nn_high = []

    exp_mean = []
    exp_low = []
    exp_high = []

    for level in levels:

        key = str(float(level))

        nn = dose[key][
            "nn_distance"
        ]

        exp = dose[key][
            "explanation_disagreement"
        ]

        nn_mean.append(
            nn["mean"]
        )
        nn_low.append(
            nn["ci95_low"]
        )
        nn_high.append(
            nn["ci95_high"]
        )

        exp_mean.append(
            exp["mean"]
        )
        exp_low.append(
            exp["ci95_low"]
        )
        exp_high.append(
            exp["ci95_high"]
        )

    slope = cross[
        "mean_query_slope"
    ]

    positive = cross[
        "mean_positive_slope_fraction"
    ]

    sign_flip = cross[
        "exact_sign_flip_test"
    ]

    return {
        "name": name,

        "levels": levels,

        "nn_mean": np.asarray(
            nn_mean,
            dtype=float,
        ),
        "nn_low": np.asarray(
            nn_low,
            dtype=float,
        ),
        "nn_high": np.asarray(
            nn_high,
            dtype=float,
        ),

        "exp_mean": np.asarray(
            exp_mean,
            dtype=float,
        ),
        "exp_low": np.asarray(
            exp_low,
            dtype=float,
        ),
        "exp_high": np.asarray(
            exp_high,
            dtype=float,
        ),

        "slope_mean": float(
            slope["mean"]
        ),
        "slope_low": float(
            slope["ci95_low"]
        ),
        "slope_high": float(
            slope["ci95_high"]
        ),

        "seed_slopes": np.asarray(
            cross[
                "seed_level_slope_values"
            ],
            dtype=float,
        ),

        "positive_fraction": float(
            positive["mean"]
        ),

        "one_sided_p": float(
            sign_flip["one_sided_p"]
        ),

        "two_sided_p": float(
            sign_flip["two_sided_p"]
        ),
    }


def save_figure(
    fig: plt.Figure,
    stem: Path,
) -> None:

    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.045,
        facecolor="white",
    )

    plt.close(fig)


def configure_x_axis(ax):
    ax.set_xlim(
        -0.005,
        0.305,
    )

    ax.set_xticks(
        [
            0.00,
            0.05,
            0.10,
            0.20,
            0.30,
        ]
    )

    ax.set_xticklabels(
        [
            "0",
            "0.05",
            "0.10",
            "0.20",
            "0.30",
        ]
    )


def panel_label(
    ax,
    label: str,
):
    ax.text(
        0.025,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )


# =============================================================================
# FIGURE 4
# =============================================================================


def make_figure4(
    hopper: dict,
    halfcheetah: dict,
):

    # Distinguish environments using both color and marker shape.
    hopper_color = "#1f77b4"
    halfcheetah_color = "#d62728"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.8, 5.65),
        sharex="col",
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.18,
        top=0.84,
        wspace=0.23,
        hspace=0.27,
    )

    environments = [
        (hopper, hopper_color, "o"),
        (halfcheetah, halfcheetah_color, "s"),
    ]

    # -------------------------------------------------------------------------
    # TOP ROW — DISTRIBUTIONAL DISTANCE
    # -------------------------------------------------------------------------

    for col, (env, color, marker) in enumerate(
        environments
    ):
        ax = axes[0, col]

        x = env["levels"]

        ax.fill_between(
            x,
            env["nn_low"],
            env["nn_high"],
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            env["nn_mean"],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )

        ax.set_title(
            env["name"],
            fontsize=11,
            fontweight="bold",
            pad=7,
        )

        ax.set_ylabel(
            "Nearest-neighbor distance"
        )

        ax.grid(
            True,
            axis="both",
        )

        ax.set_axisbelow(True)

        configure_x_axis(ax)

        panel_label(
            ax,
            "(a)" if col == 0 else "(b)",
        )

    # -------------------------------------------------------------------------
    # BOTTOM ROW — EXPLANATION DISAGREEMENT
    # -------------------------------------------------------------------------

    for col, (env, color, marker) in enumerate(
        environments
    ):
        ax = axes[1, col]

        x = env["levels"]

        ax.fill_between(
            x,
            env["exp_low"],
            env["exp_high"],
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            env["exp_mean"],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )

        ax.set_ylabel(
            "Explanation action disagreement\n(RMS)"
        )

        ax.set_xlabel(
            "Gaussian observation-shift magnitude, σ"
        )

        ax.grid(
            True,
            axis="both",
        )

        ax.set_axisbelow(True)

        configure_x_axis(ax)

        panel_label(
            ax,
            "(c)" if col == 0 else "(d)",
        )

    # -------------------------------------------------------------------------
    # MAIN TITLE
    # -------------------------------------------------------------------------

    fig.text(
        0.5,
        0.965,
        "Response to Gaussian observation shift",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.918,
        "Across-seed response across seven predefined shift magnitudes",
        ha="center",
        va="top",
        fontsize=9.3,
    )

    # -------------------------------------------------------------------------
    # SHARED LEGEND
    # -------------------------------------------------------------------------

    mean_handle = plt.Line2D(
        [0],
        [0],
        color="#333333",
        linewidth=2.0,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="#333333",
        markeredgewidth=0.9,
        label="Across-seed mean",
    )

    ci_handle = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#777777",
        alpha=0.15,
        edgecolor="none",
        label="95% CI across policy seeds",
    )

    fig.legend(
        handles=[
            mean_handle,
            ci_handle,
        ],
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.055,
        ),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=2.0,
    )

    save_figure(
        fig,
        OUTPUT_DIR
        / "figure4_cross_environment_response",
    )


# =============================================================================
# FIGURE 5
# =============================================================================

def make_figure5(
    hopper: dict,
    halfcheetah: dict,
):

    fig, ax = plt.subplots(
        figsize=(6.25, 4.75),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.17,
        top=0.85,
    )

    environments = [
        hopper,
        halfcheetah,
    ]

    # Keep environments visually distinct.
    colors = [
        "#1f77b4",
        "#d62728",
    ]

    x_centers = np.array(
        [
            0.0,
            1.0,
        ]
    )

    jitter = np.array(
        [
            -0.075,
            -0.0375,
            0.0,
            0.0375,
            0.075,
        ]
    )

    for i, (env, color) in enumerate(
        zip(
            environments,
            colors,
        )
    ):

        slopes = env[
            "seed_slopes"
        ]

        if len(slopes) != 5:
            raise RuntimeError(
                f"{env['name']}: expected 5 seeds."
            )

        # -------------------------------------------------------------
        # Individual policy seeds
        # -------------------------------------------------------------

        ax.scatter(
            x_centers[i] + jitter,
            slopes,
            s=48,
            color=color,
            alpha=0.84,
            edgecolor="white",
            linewidth=0.75,
            zorder=4,
            label=(
                "Individual policy seeds"
                if i == 0
                else None
            ),
        )

        # -------------------------------------------------------------
        # Mean + 95% CI
        # -------------------------------------------------------------

        mean = env[
            "slope_mean"
        ]

        low = env[
            "slope_low"
        ]

        high = env[
            "slope_high"
        ]

        ax.errorbar(
            [x_centers[i]],
            [mean],
            yerr=[
                [mean - low],
                [high - mean],
            ],
            fmt="o",
            color="#222222",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="#222222",
            markeredgewidth=1.6,
            capsize=5,
            capthick=1.25,
            elinewidth=1.35,
            zorder=6,
            label=(
                "Across-seed mean ± 95% CI"
                if i == 0
                else None
            ),
        )

        # -------------------------------------------------------------
        # 5/5 annotation
        # -------------------------------------------------------------

        ax.text(
            x_centers[i],
            high + 0.008,
            "5/5 seeds positive",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )

        # -------------------------------------------------------------
        # Numerical mean + CI
        # -------------------------------------------------------------

        annotation = (
            f"{mean:.3f} "
            f"[{low:.3f}, {high:.3f}]"
        )

        vertical_offset = (
            -0.020
            if i == 0
            else -0.018
        )

        ax.text(
            x_centers[i],
            mean + vertical_offset,
            annotation,
            ha="center",
            va="top",
            fontsize=8.2,
        )

    # -------------------------------------------------------------------------
    # ZERO LINE
    # -------------------------------------------------------------------------

    ax.axhline(
        0.0,
        color="#666666",
        linestyle="--",
        linewidth=0.85,
        alpha=0.65,
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # AXES
    # -------------------------------------------------------------------------

    ax.set_xticks(
        x_centers
    )

    ax.set_xticklabels(
        [
            "Hopper",
            "HalfCheetah",
        ]
    )

    ax.set_xlim(
        -0.28,
        1.28,
    )

    ax.set_ylim(
        -0.006,
        0.165,
    )

    ax.set_ylabel(
        "Seed-level distance–disagreement slope"
    )

    ax.grid(
        True,
        axis="y",
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # TITLE
    # -------------------------------------------------------------------------

    ax.set_title(
        "Replication across independently trained policies",
        fontsize=12,
        fontweight="bold",
        pad=11,
    )

    # -------------------------------------------------------------------------
    # LEGEND
    # -------------------------------------------------------------------------

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.7,
        borderaxespad=0.3,
    )

    # -------------------------------------------------------------------------
    # STATISTICAL FOOTNOTE
    # -------------------------------------------------------------------------

    fig.text(
        0.5,
        0.035,
        "Points: individual IQL policy seeds   "
        "○: across-seed mean   "
        "error bars: two-sided 95% CI across five seeds",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )

    save_figure(
        fig,
        OUTPUT_DIR
        / "figure5_cross_environment_seed_slopes",
    )


# =============================================================================
# METADATA
# =============================================================================

def write_metadata(
    hopper: dict,
    halfcheetah: dict,
):

    metadata = {
        "figure4": {
            "title": (
                "Response to Gaussian observation shift"
            ),
            "panels": {
                "a": (
                    "Hopper nearest-neighbor distance"
                ),
                "b": (
                    "HalfCheetah nearest-neighbor distance"
                ),
                "c": (
                    "Hopper explanation action disagreement"
                ),
                "d": (
                    "HalfCheetah explanation action disagreement"
                ),
            },
            "uncertainty": (
                "95% CI across the five independently trained "
                "policy seeds."
            ),
            "x_axis": (
                "Gaussian observation-shift magnitude, sigma"
            ),
        },

        "figure5": {
            "title": (
                "Replication across independently trained policies"
            ),
            "individual_points": (
                "Five independently trained IQL policy seeds "
                "per environment."
            ),
            "mean_marker": (
                "Across-seed mean."
            ),
            "error_bars": (
                "Two-sided 95% t-based CI across the five "
                "independent policy seeds."
            ),
            "hopper": {
                "mean": hopper["slope_mean"],
                "ci95": [
                    hopper["slope_low"],
                    hopper["slope_high"],
                ],
                "seed_slopes": (
                    hopper["seed_slopes"].tolist()
                ),
                "positive_query_slope_fraction": (
                    hopper["positive_fraction"]
                ),
            },
            "halfcheetah": {
                "mean": halfcheetah["slope_mean"],
                "ci95": [
                    halfcheetah["slope_low"],
                    halfcheetah["slope_high"],
                ],
                "seed_slopes": (
                    halfcheetah["seed_slopes"].tolist()
                ),
                "positive_query_slope_fraction": (
                    halfcheetah["positive_fraction"]
                ),
            },
        },

        "interpretation_guardrail": (
            "The primary cross-environment claim is replication "
            "of a positive distance-to-explanation-disagreement "
            "relationship. Raw slope magnitudes are not treated "
            "as directly comparable effect sizes across environments."
        ),
    }

    path = (
        OUTPUT_DIR
        / "cross_environment_publication_figure_metadata.json"
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hopper = extract_environment(
        load_json(HOPPER),
        "Hopper",
    )

    halfcheetah = extract_environment(
        load_json(HALFCHEETAH),
        "HalfCheetah",
    )

    # Ensure the synthesis artifact agrees with the two
    # source analysis files.
    synthesis = load_json(
        SYNTHESIS
    )

    hopper_synthesis = synthesis[
        "environments"
    ][0]

    halfcheetah_synthesis = synthesis[
        "environments"
    ][1]

    if not np.isclose(
        hopper_synthesis[
            "primary_mean_slope"
        ]["mean"],
        hopper["slope_mean"],
    ):
        raise RuntimeError(
            "Hopper synthesis mismatch."
        )

    if not np.isclose(
        halfcheetah_synthesis[
            "primary_mean_slope"
        ]["mean"],
        halfcheetah["slope_mean"],
    ):
        raise RuntimeError(
            "HalfCheetah synthesis mismatch."
        )

    make_figure4(
        hopper,
        halfcheetah,
    )

    make_figure5(
        hopper,
        halfcheetah,
    )

    write_metadata(
        hopper,
        halfcheetah,
    )

    print("=" * 80)
    print("FINAL PUBLICATION FIGURES")
    print("=" * 80)

    print()
    print(
        f"Hopper: "
        f"{hopper['slope_mean']:.6f} "
        f"[{hopper['slope_low']:.6f}, "
        f"{hopper['slope_high']:.6f}]"
    )

    print(
        f"HalfCheetah: "
        f"{halfcheetah['slope_mean']:.6f} "
        f"[{halfcheetah['slope_low']:.6f}, "
        f"{halfcheetah['slope_high']:.6f}]"
    )

    print()
    print("Generated:")

    for name in [
        "figure4_cross_environment_response",
        "figure5_cross_environment_seed_slopes",
    ]:
        print(
            " ",
            OUTPUT_DIR / f"{name}.pdf",
        )
        print(
            " ",
            OUTPUT_DIR / f"{name}.png",
        )
        print(
            " ",
            OUTPUT_DIR / f"{name}.svg",
        )

    print()
    print("✅ CAMERA-READY FIGURES GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
