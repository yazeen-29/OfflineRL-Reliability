from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HOPPER_ANALYSIS = Path(
    "results/analysis/multiseed/"
    "iql_100k_multiseed_gaussian_analysis_v2.json"
)

HALFCHEETAH_ANALYSIS = Path(
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

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
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


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )

    return json.loads(path.read_text())


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


def extract_environment(
    name: str,
    data: dict,
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

    dose = cross.get(
        "noise_level_results"
    )

    if dose is None:
        raise RuntimeError(
            f"{name}: missing cross_seed "
            "'noise_level_results'."
        )

    distance = []
    disagreement = []

    disagreement_low = []
    disagreement_high = []

    for level in levels:
        key = str(float(level))

        if key not in dose:
            raise RuntimeError(
                f"{name}: missing noise level {key}"
            )

        nn = dose[key]["nn_distance"]
        exp = dose[key]["explanation_disagreement"]

        distance.append(
            float(nn["mean"])
        )

        disagreement.append(
            float(exp["mean"])
        )

        disagreement_low.append(
            float(exp["ci95_low"])
        )

        disagreement_high.append(
            float(exp["ci95_high"])
        )

    slope = cross["mean_query_slope"]
    positive = cross["mean_positive_slope_fraction"]
    sign_flip = cross["exact_sign_flip_test"]

    return {
        "name": name,
        "levels": levels,
        "distance": np.asarray(
            distance,
            dtype=float,
        ),
        "disagreement": np.asarray(
            disagreement,
            dtype=float,
        ),
        "disagreement_low": np.asarray(
            disagreement_low,
            dtype=float,
        ),
        "disagreement_high": np.asarray(
            disagreement_high,
            dtype=float,
        ),
        "slope_mean": float(slope["mean"]),
        "slope_low": float(slope["ci95_low"]),
        "slope_high": float(slope["ci95_high"]),
        "positive_fraction": float(positive["mean"]),
        "one_sided_p": float(sign_flip["one_sided_p"]),
        "two_sided_p": float(sign_flip["two_sided_p"]),
    }


def audit_synthesis(
    synthesis: dict,
    environments: list[dict],
) -> None:

    synthesis_envs = {
        item["environment"]: item
        for item in synthesis["environments"]
    }

    for env in environments:
        name = env["name"]

        if name not in synthesis_envs:
            raise RuntimeError(
                f"Synthesis missing environment: {name}"
            )

        source_mean = env["slope_mean"]

        synthesis_mean = (
            synthesis_envs[name]
            ["primary_mean_slope"]
            ["mean"]
        )

        if not np.isclose(
            source_mean,
            synthesis_mean,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{name}: source analysis and synthesis "
                "mean slopes disagree."
            )


def make_figure(
    environments: list[dict],
) -> None:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.45, 3.45),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.21,
        top=0.78,
        wspace=0.27,
    )

    markers = {
        "Hopper": "o",
        "HalfCheetah": "s",
    }

    linestyles = {
        "Hopper": "-",
        "HalfCheetah": "--",
    }

    for ax, env in zip(
        axes,
        environments,
    ):
        x = env["distance"]
        y = env["disagreement"]

        yerr_low = (
            y
            - env["disagreement_low"]
        )

        yerr_high = (
            env["disagreement_high"]
            - y
        )

        # The connecting line follows the predefined
        # Gaussian-dose ordering and is descriptive only.
        ax.plot(
            x,
            y,
            linestyle=linestyles[env["name"]],
            linewidth=1.55,
            alpha=0.72,
            zorder=2,
        )

        # Vertical error bars show the across-seed
        # uncertainty in explanation disagreement.
        # Horizontal uncertainty is deliberately omitted
        # here to avoid clutter; the NN-distance values
        # are the horizontal coordinates.
        ax.errorbar(
            x,
            y,
            yerr=[
                yerr_low,
                yerr_high,
            ],
            fmt=markers[env["name"]],
            linestyle="none",
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.05,
            ecolor="black",
            capsize=2.8,
            capthick=0.8,
            elinewidth=0.8,
            zorder=4,
        )

        ax.set_title(
            env["name"],
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        ax.set_xlabel(
            "Nearest-neighbor distance"
        )

        ax.set_ylabel(
            "Explanation action disagreement (RMS)"
        )

        ax.grid(
            True,
            axis="both",
        )

        ax.set_axisbelow(True)

        xmin = float(
            np.min(x)
        )
        xmax = float(
            np.max(x)
        )

        ymin = float(
            np.min(
                env["disagreement_low"]
            )
        )
        ymax = float(
            np.max(
                env["disagreement_high"]
            )
        )

        xspan = max(
            xmax - xmin,
            1e-12,
        )
        yspan = max(
            ymax - ymin,
            1e-12,
        )

        ax.set_xlim(
            xmin - 0.08 * xspan,
            xmax + 0.08 * xspan,
        )

        ax.set_ylim(
            max(
                0.0,
                ymin - 0.10 * yspan,
            ),
            ymax + 0.10 * yspan,
        )

    fig.text(
        0.5,
        0.955,
        "Distributional distance tracks explanation disagreement",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.900,
        "Across-seed means for seven predefined Gaussian shift levels",
        ha="center",
        va="top",
        fontsize=9.0,
    )

    fig.text(
        0.5,
        0.035,
        "Markers show across-seed means; error bars show two-sided 95% CI across five independent policy seeds",
        ha="center",
        va="bottom",
        fontsize=7.6,
    )

    stem = (
        OUTPUT_DIR
        / "figure6_cross_environment_distance_vs_disagreement"
    )

    save_figure(
        fig,
        stem,
    )


def write_metadata(
    environments: list[dict],
) -> None:

    metadata = {
        "figure": "Figure 6",
        "title": (
            "Distributional distance tracks explanation disagreement"
        ),
        "purpose": (
            "Direct visualization of the distance-to-"
            "explanation-disagreement relationship in "
            "Hopper and HalfCheetah."
        ),
        "point_definition": (
            "Across-seed mean at each predefined Gaussian "
            "observation-shift level."
        ),
        "uncertainty": (
            "Two-sided 95% CI across five independently "
            "trained policy seeds for explanation disagreement."
        ),
        "labels": (
            "All seven Gaussian shift levels are plotted. "
            "Dose labels are omitted from the figure to "
            "avoid overlap and visual crowding; exact levels "
            "are defined in the experiment protocol."
        ),
        "statistical_unit": (
            "Policy seed is the independent replication unit."
        ),
        "guardrail": (
            "The figure is descriptive across predefined "
            "Gaussian dose levels. Primary inferential claims "
            "use independent policy-seed replication. Raw "
            "slope magnitudes are not interpreted as directly "
            "standardized cross-environment effect sizes."
        ),
        "environments": {},
    }

    for env in environments:
        metadata["environments"][
            env["name"]
        ] = {
            "mean_slope": env["slope_mean"],
            "ci95": [
                env["slope_low"],
                env["slope_high"],
            ],
            "positive_query_slope_fraction": (
                env["positive_fraction"]
            ),
            "one_sided_exact_sign_flip_p": (
                env["one_sided_p"]
            ),
            "two_sided_exact_sign_flip_p": (
                env["two_sided_p"]
            ),
        }

    metadata_path = (
        OUTPUT_DIR
        / "figure6_cross_environment_distance_vs_disagreement_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


def main() -> None:

    hopper = extract_environment(
        "Hopper",
        load_json(HOPPER_ANALYSIS),
    )

    halfcheetah = extract_environment(
        "HalfCheetah",
        load_json(HALFCHEETAH_ANALYSIS),
    )

    environments = [
        hopper,
        halfcheetah,
    ]

    synthesis = load_json(
        SYNTHESIS
    )

    audit_synthesis(
        synthesis,
        environments,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_figure(
        environments
    )

    write_metadata(
        environments
    )

    print("=" * 80)
    print("FIGURE 6 — FINAL CROSS-ENVIRONMENT RELATIONSHIP")
    print("=" * 80)

    for env in environments:
        print()
        print(env["name"])
        print(
            "  Mean slope:",
            f"{env['slope_mean']:.6f}",
        )
        print(
            "  95% CI:",
            (
                f"{env['slope_low']:.6f}",
                f"{env['slope_high']:.6f}",
            ),
        )

    print()
    print(
        "PDF:",
        OUTPUT_DIR
        / "figure6_cross_environment_distance_vs_disagreement.pdf",
    )
    print(
        "PNG:",
        OUTPUT_DIR
        / "figure6_cross_environment_distance_vs_disagreement.png",
    )
    print(
        "SVG:",
        OUTPUT_DIR
        / "figure6_cross_environment_distance_vs_disagreement.svg",
    )
    print(
        "Metadata:",
        OUTPUT_DIR
        / "figure6_cross_environment_distance_vs_disagreement_metadata.json",
    )

    print()
    print("✅ FINAL FIGURE 6 GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
