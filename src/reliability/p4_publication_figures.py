from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results" / "analysis" / "P4"

STAT_DIR = RESULTS / "statistics"
DECILE_DIR = RESULTS / "reliability_deciles"
RISK_DIR = RESULTS / "risk_coverage"

FIG_DIR = RESULTS / "final_figures"
TABLE_DIR = RESULTS / "final_tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


MODEL_ORDER = [
    "B1_support_only",
    "B2_critic_only",
    "B3_support_critic",
    "B4_action_only",
    "B5_full",
]

MODEL_LABELS = {
    "B1_support_only": "Support / OOD",
    "B2_critic_only": "Critic uncertainty",
    "B3_support_critic": "Support + critic",
    "B4_action_only": "Action disagreement",
    "B5_full": "Action + support + critic",
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


metrics = pd.read_csv(
    STAT_DIR / "P4_seed_level_metrics.csv"
)

summary = pd.read_csv(
    STAT_DIR / "P4_model_summary.csv"
)

deciles = pd.read_csv(
    DECILE_DIR / "P4_reliability_deciles_summary.csv"
)

risk = pd.read_csv(
    RISK_DIR / "P4_risk_coverage_nonzero_summary.csv"
)


# ============================================================
# Figure 1: seed-level Spearman comparison
# ============================================================

fig, ax = plt.subplots(figsize=(8.0, 5.4))
fig.subplots_adjust(bottom=0.22, right=0.78)

rng = np.random.default_rng(20260920)

for i, model in enumerate(MODEL_ORDER):
    d = metrics[
        metrics["model"] == model
    ].sort_values("test_seed")

    values = d[
        "spearman_reliability_vs_C10"
    ].to_numpy(float)

    jitter = rng.uniform(
        -0.08,
        0.08,
        size=len(values),
    )

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        s=34,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
    )

    mean = float(values.mean())
    sd = float(values.std(ddof=1))

    ax.errorbar(
        i,
        mean,
        yerr=sd,
        fmt="o",
        color="black",
        markersize=7,
        capsize=5,
        linewidth=1.6,
        zorder=4,
    )

ax.axhline(
    0.0,
    color="0.4",
    linestyle="--",
    linewidth=1.0,
)

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    rotation=22,
    ha="right",
)

ax.set_ylabel(
    r"Spearman correlation: reliability score vs. $C_{10}$"
)
ax.set_title(
    "Held-out reliability ordering across policy seeds"
)

save_figure(
    fig,
    "fig1_seed_level_spearman",
)


# ============================================================
# Figure 2: reliability deciles
# ============================================================

fig, ax = plt.subplots(figsize=(8.0, 5.4))
fig.subplots_adjust(right=0.76)

for model in MODEL_ORDER:
    d = (
        deciles[
            deciles["model"] == model
        ]
        .sort_values("reliability_decile")
    )

    ax.errorbar(
        d["reliability_decile"],
        d["mean_observed_C10"],
        yerr=d["sd_observed_C10"],
        marker="o",
        markersize=3.8,
        linewidth=1.3,
        capsize=2.5,
        label=MODEL_LABELS[model],
    )

ax.set_xlabel(
    "Reliability decile "
    "(high reliability → low reliability)"
)
ax.set_ylabel(
    r"Mean observed $C_{10}$ ± SD"
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
    fontsize=7.0,
)

save_figure(
    fig,
    "fig2_reliability_deciles",
)


# ============================================================
# Figure 3: nonzero-shift risk coverage
# ============================================================

fig, ax = plt.subplots(figsize=(8.0, 5.4))
fig.subplots_adjust(right=0.76)

for model in MODEL_ORDER:
    d = (
        risk[
            risk["model"] == model
        ]
        .sort_values("coverage_percent")
    )

    ax.errorbar(
        d["coverage_percent"],
        d["mean_observed_C10"],
        yerr=d["sd_observed_C10"],
        marker="o",
        markersize=3.8,
        linewidth=1.3,
        capsize=2.5,
        label=MODEL_LABELS[model],
    )

ax.set_xlabel(
    "Coverage of nonzero-shift states (%)"
)
ax.set_ylabel(
    r"Mean observed $C_{10}$ ± SD"
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
    fontsize=7.0,
)

save_figure(
    fig,
    "fig3_risk_coverage_nonzero",
)


# ============================================================
# Compact publication table
# ============================================================

table_rows = []

for _, row in summary.iterrows():
    model = row["model"]

    table_rows.append(
        {
            "Model":
                MODEL_LABELS[model],
            "Mean Spearman ± SD":
                f"{row['mean_spearman']:.4f} ± "
                f"{row['sd_spearman']:.4f}",
            "Mean MAE ± SD":
                (
                    "—"
                    if pd.isna(row["mean_mae"])
                    else
                    f"{row['mean_mae']:.4f} ± "
                    f"{row['sd_mae']:.4f}"
                ),
            "Mean RMSE ± SD":
                (
                    "—"
                    if pd.isna(row["mean_rmse"])
                    else
                    f"{row['mean_rmse']:.4f} ± "
                    f"{row['sd_rmse']:.4f}"
                ),
        }
    )

table = pd.DataFrame(table_rows)

table.to_csv(
    TABLE_DIR / "P4_model_summary_publication.csv",
    index=False,
)

(
    TABLE_DIR / "P4_model_summary_publication.tex"
).write_text(
    table.to_latex(
        index=False,
        escape=False,
    )
)


metadata = {
    "experiment": "P4 OOD / uncertainty baselines",
    "horizon": 10,
    "population": "nonzero observation shifts",
    "independent_unit": "policy seed",
    "policy_seeds": [0, 1, 2, 3, 4],
    "models": MODEL_LABELS,
    "figures": {
        "fig1_seed_level_spearman":
            "Five held-out policy-seed Spearman correlations",
        "fig2_reliability_deciles":
            "Observed C10 across reliability deciles",
        "fig3_risk_coverage_nonzero":
            "Observed C10 under nonzero-shift risk coverage",
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
print("P4 PUBLICATION FIGURES + TABLE COMPLETE")
print("=" * 80)
print()
print("Figures:")
for name in [
    "fig1_seed_level_spearman",
    "fig2_reliability_deciles",
    "fig3_risk_coverage_nonzero",
]:
    print(" ", FIG_DIR / f"{name}.pdf")
    print(" ", FIG_DIR / f"{name}.png")

print()
print("Tables:")
print(
    " ",
    TABLE_DIR / "P4_model_summary_publication.csv",
)
print(
    " ",
    TABLE_DIR / "P4_model_summary_publication.tex",
)
print()
