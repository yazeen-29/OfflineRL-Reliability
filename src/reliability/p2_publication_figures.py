from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results" / "analysis" / "P2"

DECILE_DIR = RESULTS / "reliability_deciles"
RISK_DIR = RESULTS / "risk_coverage"
STATS_DIR = RESULTS / "statistics"

FIG_DIR = RESULTS / "final_figures"
TABLE_DIR = RESULTS / "final_tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


MODEL_ORDER = [
    "primary_3feature",
    "action_only_sensitivity",
]

MODEL_LABELS = {
    "primary_3feature":
        "Action + support + critic",
    "action_only_sensitivity":
        "Action disagreement only",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIG_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIG_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Input data
# ============================================================

deciles = pd.read_csv(
    DECILE_DIR / "P2_reliability_deciles_summary.csv"
)

risk = pd.read_csv(
    RISK_DIR
    / "P2_risk_coverage_nonzero_shift_summary.csv"
)

metrics = pd.read_csv(
    STATS_DIR / "P2_model_summary.csv"
)


# ============================================================
# Figure 1
# Reliability deciles vs observed consequence
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 5.2))
fig.subplots_adjust(right=0.77)

for model in MODEL_ORDER:
    d = (
        deciles[
            deciles["model"] == model
        ]
        .sort_values("reliability_decile")
    )

    x = d["reliability_decile"].to_numpy(float)
    y = d["mean_observed_C10"].to_numpy(float)
    yerr = d["sd_observed_C10"].to_numpy(float)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        capsize=3,
        label=MODEL_LABELS[model],
    )

ax.set_xlabel(
    "Reliability decile (high reliability → low reliability)"
)
ax.set_ylabel(
    r"Mean observed $C_{10}$ ± SD across policy seeds"
)
ax.set_title(
    "Observed consequence across reliability deciles"
)

ax.set_xticks(range(1, 11))
ax.set_xlim(0.7, 10.3)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    fontsize=7.5,
)

ax.text(
    0.02,
    0.97,
    "Higher score denotes higher predicted reliability",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)

save_figure(
    fig,
    "fig1_reliability_deciles",
)


# ============================================================
# Figure 2
# Nonzero-shift risk coverage
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 5.2))
fig.subplots_adjust(right=0.77)

for model in MODEL_ORDER:
    d = (
        risk[
            risk["model"] == model
        ]
        .sort_values("coverage_percent")
    )

    x = d["coverage_percent"].to_numpy(float)
    y = d["mean_observed_C10"].to_numpy(float)
    yerr = d["sd_observed_C10"].to_numpy(float)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        capsize=3,
        label=MODEL_LABELS[model],
    )

ax.set_xlabel("Coverage of nonzero-shift states (%)")
ax.set_ylabel(
    r"Mean observed $C_{10}$ ± SD across policy seeds"
)
ax.set_title(
    r"Reliability-score risk–coverage ($\sigma>0$)"
)

ax.set_xticks(
    [10, 20, 30, 50, 70, 90, 100]
)
ax.set_xlim(5, 102)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    fontsize=7.5,
)

ax.text(
    0.02,
    0.97,
    "Lower coverage retains the highest-reliability states",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)

save_figure(
    fig,
    "fig2_risk_coverage_nonzero",
)


# ============================================================
# Publication table
# ============================================================

table = metrics.copy()

table["Model"] = table["model"].map(MODEL_LABELS)

table["MAE"] = (
    table["mean_mae"].map(lambda x: f"{x:.6f}")
    + " ± "
    + table["sd_mae"].map(lambda x: f"{x:.6f}")
)

table["RMSE"] = (
    table["mean_rmse"].map(lambda x: f"{x:.6f}")
    + " ± "
    + table["sd_rmse"].map(lambda x: f"{x:.6f}")
)

table["Spearman(R vs C10)"] = (
    table["mean_spearman"].map(lambda x: f"{x:.6f}")
    + " ± "
    + table["sd_spearman"].map(lambda x: f"{x:.6f}")
)

table = table[
    [
        "Model",
        "MAE",
        "RMSE",
        "Spearman(R vs C10)",
    ]
]

table.to_csv(
    TABLE_DIR / "P2_model_summary_publication.csv",
    index=False,
)

latex_table = table.to_latex(
    index=False,
    escape=False,
)

(
    TABLE_DIR / "P2_model_summary_publication.tex"
).write_text(latex_table)


# ============================================================
# Figure metadata
# ============================================================

metadata = {
    "experiment": "P2 reliability score / estimator",
    "horizon": 10,
    "independent_unit": "policy seed",
    "policy_seeds": [0, 1, 2, 3, 4],
    "primary_model": "primary_3feature",
    "sensitivity_model": "action_only_sensitivity",
    "figure_1": {
        "description":
            "Observed H=10 consequence across reliability deciles",
        "source":
            "results/analysis/P2/reliability_deciles/"
            "P2_reliability_deciles_summary.csv",
    },
    "figure_2": {
        "description":
            "Nonzero-shift risk-coverage using reliability score",
        "source":
            "results/analysis/P2/risk_coverage/"
            "P2_risk_coverage_nonzero_shift_summary.csv",
    },
    "table": {
        "description":
            "Held-out policy-seed predictive and ordering metrics",
        "source":
            "results/analysis/P2/statistics/"
            "P2_model_summary.csv",
    },
}

(
    FIG_DIR / "figure_metadata.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
)

print("=" * 80)
print("P2 PUBLICATION FIGURES + TABLE COMPLETE")
print("=" * 80)
print()
print("Figures:")
print(" ", FIG_DIR / "fig1_reliability_deciles.pdf")
print(" ", FIG_DIR / "fig1_reliability_deciles.png")
print(" ", FIG_DIR / "fig2_risk_coverage_nonzero.pdf")
print(" ", FIG_DIR / "fig2_risk_coverage_nonzero.png")
print()
print("Tables:")
print(
    " ",
    TABLE_DIR / "P2_model_summary_publication.csv",
)
print(
    " ",
    TABLE_DIR / "P2_model_summary_publication.tex",
)
print()
