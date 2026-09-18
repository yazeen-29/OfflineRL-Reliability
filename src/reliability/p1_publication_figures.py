#!/usr/bin/env python3
"""
P1 publication figures + paper-ready table.

Inputs are existing frozen-analysis outputs only.
This script does NOT touch data_frozen/P1 and does NOT retrain anything.

Outputs:
  results/analysis/P1/final_figures_v2/
    fig1_signal_relationship.png
    fig2_paired_mae_vs_baseline.png
    fig3_risk_coverage_nonzero.png
    fig4_horizon_sensitivity.png

  results/analysis/P1/final_tables/
    P1_primary_model_comparison.csv
    P1_primary_model_comparison.tex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "analysis" / "P1"

FIG_OUT = RESULTS / "final_figures_v2"
TABLE_OUT = RESULTS / "final_tables"

FIG_OUT.mkdir(parents=True, exist_ok=True)
TABLE_OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10

MODEL_ORDER = [
    "A_action_only",
    "B_action_plus_support",
    "C_action_plus_critic",
    "D_action_plus_support_plus_critic",
    "E_action_plus_support_interaction",
]

MODEL_LABELS = {
    "A_action_only": "Action",
    "B_action_plus_support": "Action + Support",
    "C_action_plus_critic": "Action + Critic",
    "D_action_plus_support_plus_critic": "Action + Support + Critic",
    "E_action_plus_support_interaction": "Action + Support + Interaction",
}

LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "P"]


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Load existing analysis outputs
# ---------------------------------------------------------------------
pred = pd.read_csv(
    RESULTS / "models" / "heldout_predictions.csv"
)

fold = pd.read_csv(
    RESULTS / "statistics" / "heldout_fold_metrics.csv"
)

baseline = pd.read_csv(
    RESULTS / "statistics" / "no_signal_baseline.csv"
)

inference = pd.read_csv(
    RESULTS / "statistics" / "seed_level_inference.csv"
)

model_comparison = pd.read_csv(
    RESULTS / "tables" / "heldout_model_comparison.csv"
)

improvement = pd.read_csv(
    RESULTS / "statistics" / "baseline_improvement_by_seed.csv"
)

risk = pd.read_csv(
    RESULTS / "statistics" / "risk_coverage_nonzero_shift.csv"
)

# ---------------------------------------------------------------------
# Figure 1
# One underlying dataset only, not five duplicated model clouds.
#
# Scatter = 3500 unique H=10 observations.
# Trend = quantile-binned median C10 with IQR ribbon.
# ---------------------------------------------------------------------
h10 = pred[pred["horizon"] == HORIZON].copy()

unique_cols = [
    "policy_seed",
    "state_id",
    "sigma",
    "horizon",
]

h10 = h10.drop_duplicates(
    subset=unique_cols,
    keep="first",
).copy()

if len(h10) != 3500:
    raise RuntimeError(
        f"Expected 3500 unique H=10 observations, got {len(h10)}"
    )

x = h10["action_disagreement"].to_numpy(float)
y = h10["absolute_consequence"].to_numpy(float)

fig, ax = plt.subplots(figsize=(7.2, 5.2))

ax.scatter(
    x,
    y,
    s=9,
    alpha=0.16,
    linewidths=0,
    color="0.35",
)

# Quantile bins give a stable descriptive trend without pretending
# that the observations are independent inferential units.
quantiles = np.linspace(0, 1, 21)
edges = np.quantile(x, quantiles)
edges[0] -= 1e-12
edges[-1] += 1e-12

bin_x = []
bin_med = []
bin_q25 = []
bin_q75 = []

for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (x >= lo) & (x < hi)
    if mask.sum() < 5:
        continue

    vals_x = x[mask]
    vals_y = y[mask]

    bin_x.append(np.median(vals_x))
    bin_med.append(np.median(vals_y))
    bin_q25.append(np.quantile(vals_y, 0.25))
    bin_q75.append(np.quantile(vals_y, 0.75))

bin_x = np.asarray(bin_x)
bin_med = np.asarray(bin_med)
bin_q25 = np.asarray(bin_q25)
bin_q75 = np.asarray(bin_q75)

ax.plot(
    bin_x,
    bin_med,
    linewidth=2.2,
    color="black",
    label="Binned median",
)

ax.fill_between(
    bin_x,
    bin_q25,
    bin_q75,
    color="0.75",
    alpha=0.35,
    linewidth=0,
    label="Interquartile range",
)

ax.set_xlabel("Action disagreement")
ax.set_ylabel(r"$C_{10}=|\Delta J(10)|$")
ax.set_title(
    "Counterfactual consequence vs. action disagreement"
)

ax.legend(frameon=True, fontsize=8)

ax.text(
    0.99,
    0.02,
    "n = 3,500 H=10 observations",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)

save_fig(
    fig,
    FIG_OUT / "fig1_signal_relationship.png",
)


# ---------------------------------------------------------------------
# Figure 2
# Paired policy-seed inference:
# ΔMAE = model MAE - no-signal baseline MAE.
# Negative = improvement.
# Shows five individual seed points + 95% t CI on the mean.
# ---------------------------------------------------------------------
imp = improvement[
    improvement["horizon"] == HORIZON
].copy()

inf = inference.copy()

fig, ax = plt.subplots(figsize=(8.0, 5.3))

xpos = np.arange(len(MODEL_ORDER))

rng = np.random.default_rng(20260918)

for i, model in enumerate(MODEL_ORDER):
    d = imp[imp["model"] == model].sort_values("test_seed")
    s = inf[inf["model"] == model].iloc[0]

    seed_values = d["delta_mae"].to_numpy(float)

    # Deterministic, tiny horizontal jitter solely for readability.
    jitter = rng.uniform(-0.08, 0.08, size=len(seed_values))

    ax.scatter(
        np.full(len(seed_values), i) + jitter,
        seed_values,
        s=36,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
    )

    mean_delta = float(s["mae_mean_difference"])
    ci_low = float(s["mae_95ci_low"])
    ci_high = float(s["mae_95ci_high"])

    ax.errorbar(
        i,
        mean_delta,
        yerr=np.array(
            [[mean_delta - ci_low], [ci_high - mean_delta]]
        ),
        fmt="o",
        color="black",
        markersize=7,
        capsize=5,
        linewidth=1.8,
        zorder=4,
    )

ax.axhline(
    0.0,
    color="0.35",
    linestyle="--",
    linewidth=1.2,
)

ax.set_xticks(xpos)
ax.set_xticklabels(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    rotation=22,
    ha="right",
)

ax.set_ylabel(
    r"$\Delta$MAE  (model $-$ baseline; lower = smaller error)"
)
ax.set_title(
    "Held-out consequence prediction: paired policy-seed improvement"
)

save_fig(
    fig,
    FIG_OUT / "fig2_paired_mae_vs_baseline.png",
)


# ---------------------------------------------------------------------
# Figure 3
# Nonzero shifts only.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
fig.subplots_adjust(right=0.76)

for model, ls, marker in zip(
    MODEL_ORDER,
    LINE_STYLES,
    MARKERS,
):
    d = risk[risk["model"] == model].sort_values(
        "coverage_target"
    )

    ax.plot(
        d["coverage_target"],
        d["mean_observed_C10"],
        linestyle=ls,
        marker=marker,
        markersize=4.5,
        linewidth=1.6,
        label=MODEL_LABELS[model],
    )

ax.set_xlabel("Coverage")
ax.set_ylabel(r"Mean observed $C_{10}$")
ax.set_title(
    r"Risk–coverage under nonzero observation shifts ($\sigma>0$)"
)

ax.set_xlim(0.05, 1.02)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    fontsize=7.2,
)

save_fig(
    fig,
    FIG_OUT / "fig3_risk_coverage_nonzero.png",
)


# ---------------------------------------------------------------------
# Figure 4
# Horizon sensitivity with seed-to-seed SD.
# H=10 explicitly marked as the primary horizon.
# ---------------------------------------------------------------------
hsummary = (
    fold
    .groupby(
        ["horizon", "model", "model_label"],
        as_index=False,
    )
    .agg(
        mean_mae=("mae", "mean"),
        sd_mae=("mae", "std"),
    )
)

fig, ax = plt.subplots(figsize=(7.6, 5.3))
fig.subplots_adjust(right=0.76)

for model, ls, marker in zip(
    MODEL_ORDER,
    LINE_STYLES,
    MARKERS,
):
    d = hsummary[
        hsummary["model"] == model
    ].sort_values("horizon")

    ax.errorbar(
        d["horizon"],
        d["mean_mae"],
        yerr=d["sd_mae"],
        linestyle=ls,
        marker=marker,
        markersize=4.5,
        linewidth=1.5,
        capsize=3,
        label=MODEL_LABELS[model],
    )

ax.axvline(
    HORIZON,
    color="0.45",
    linestyle="--",
    linewidth=1.0,
)

ymin, ymax = ax.get_ylim()

ax.text(
    HORIZON + 0.25,
    ymax * 0.97,
    "H=10 primary",
    fontsize=8,
    color="0.25",
    va="top",
)

ax.set_xlabel("Prediction horizon")
ax.set_ylabel("Mean held-out MAE ± SD")
ax.set_title(
    "Held-out consequence prediction across downstream horizons"
)
ax.set_xticks([1, 5, 10, 20])
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    fontsize=7.2,
)

save_fig(
    fig,
    FIG_OUT / "fig4_horizon_sensitivity.png",
)


# ---------------------------------------------------------------------
# Paper-ready primary table.

#
# Uses H=10 and the already-computed paired seed inference.
# ---------------------------------------------------------------------
primary_fold = fold[
    fold["horizon"] == HORIZON
].copy()

baseline_mae = float(
    baseline[
        baseline["horizon"] == HORIZON
    ]["mae"].mean()
)

baseline_rmse = float(
    baseline[
        baseline["horizon"] == HORIZON
    ]["rmse"].mean()
)

rows = []

for model in MODEL_ORDER:
    f = model_comparison[
        (model_comparison["horizon"] == HORIZON)
        & (model_comparison["model"] == model)
    ].iloc[0]

    s = inference[
        inference["model"] == model
    ].iloc[0]

    rows.append(
        {
            "Model": MODEL_LABELS[model],
            "MAE": float(f["mean_mae"]),
            "RMSE": float(f["mean_rmse"]),
            "Delta_MAE": float(s["mae_mean_difference"]),
            "MAE_95CI_low": float(s["mae_95ci_low"]),
            "MAE_95CI_high": float(s["mae_95ci_high"]),
            "MAE_exact_one_sided_p": float(
                s["mae_exact_one_sided_p"]
            ),
            "Delta_RMSE": float(s["rmse_mean_difference"]),
            "RMSE_95CI_low": float(s["rmse_95ci_low"]),
            "RMSE_95CI_high": float(s["rmse_95ci_high"]),
            "RMSE_exact_one_sided_p": float(
                s["rmse_exact_one_sided_p"]
            ),
            "Seeds_improved_MAE": int(
                s["mae_seeds_improved"]
            ),
            "Seeds_improved_RMSE": int(
                s["rmse_seeds_improved"]
            ),
        }
    )

table = pd.DataFrame(rows)

table.to_csv(
    TABLE_OUT / "P1_primary_model_comparison.csv",
    index=False,
)

# Compact IEEE-style LaTeX table.
latex = table[
    [
        "Model",
        "MAE",
        "RMSE",
        "Delta_MAE",
        "MAE_95CI_low",
        "MAE_95CI_high",
        "MAE_exact_one_sided_p",
        "Seeds_improved_MAE",
    ]
].copy()

latex = latex.rename(
    columns={
        "Model": "Predictor",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "Delta_MAE": "$\\Delta$MAE",
        "MAE_95CI_low": "CI low",
        "MAE_95CI_high": "CI high",
        "MAE_exact_one_sided_p": "$p$",
        "Seeds_improved_MAE": "5-seed improvements",
    }
)

latex["MAE"] = latex["MAE"].map(lambda x: f"{x:.4f}")
latex["RMSE"] = latex["RMSE"].map(lambda x: f"{x:.4f}")
latex["$\\Delta$MAE"] = latex["$\\Delta$MAE"].map(
    lambda x: f"{x:.4f}"
)
latex["CI low"] = latex["CI low"].map(
    lambda x: f"{x:.4f}"
)
latex["CI high"] = latex["CI high"].map(
    lambda x: f"{x:.4f}"
)
latex["$p$"] = latex["$p$"].map(
    lambda x: f"{x:.5f}"
)

latex_text = latex.to_latex(
    index=False,
    escape=False,
    caption=(
        "Primary H=10 held-out consequence prediction. "
        f"No-signal baseline MAE={baseline_mae:.4f}, "
        f"RMSE={baseline_rmse:.4f}. "
        "Negative $\\Delta$MAE indicates lower error than baseline. "
        "Inference uses policy seed as the independent unit."
    ),
    label="tab:p1_primary_prediction",
)

(TABLE_OUT / "P1_primary_model_comparison.tex").write_text(
    latex_text
)

# ---------------------------------------------------------------------
# Machine-readable figure metadata.
# ---------------------------------------------------------------------
metadata = {
    "figures": [
        {
            "file": "fig1_signal_relationship",
            "n_unique_H10_observations": int(len(h10)),
            "description": (
                "Single underlying H=10 dataset with quantile-binned "
                "median and IQR; no duplicate model clouds."
            ),
        },
        {
            "file": "fig2_paired_mae_vs_baseline",
            "independent_unit": "policy_seed",
            "n_seeds": 5,
            "description": (
                "Paired held-out MAE differences with individual seed "
                "points and t-based 95% CI."
            ),
        },
        {
            "file": "fig3_risk_coverage_nonzero",
            "sigma_filter": "> 0",
            "description": (
                "Descriptive risk-coverage curves for nonzero shifts."
            ),
        },
        {
            "file": "fig4_horizon_sensitivity",
            "horizons": [1, 5, 10, 20],
            "primary_horizon": 10,
            "description": (
                "Held-out MAE across downstream horizons with seed SD."
            ),
        },
    ],
    "baseline": {
        "H10_mean_MAE": baseline_mae,
        "H10_mean_RMSE": baseline_rmse,
    },
}

import json

(FIG_OUT / "figure_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("=" * 80)
print("P1 PUBLICATION FIGURES + PAPER TABLE COMPLETE")
print("=" * 80)
print()
print("Figures:")
for p in sorted(FIG_OUT.glob("*")):
    print(" ", p)
print()
print("Tables:")
for p in sorted(TABLE_OUT.glob("P1_primary_model_comparison.*")):
    print(" ", p)
