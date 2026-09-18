#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "analysis" / "P1"
OUT = RESULTS / "final_figures"

OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Figure 1: consequence vs action disagreement
# ---------------------------------------------------------------------

pred = pd.read_csv(
    RESULTS / "models" / "heldout_predictions.csv"
)

h10 = pred[pred["horizon"] == 10].copy()

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

for model in sorted(h10["model"].unique()):
    d = h10[h10["model"] == model]

    ax.scatter(
        d["action_disagreement"],
        d["absolute_consequence"],
        s=8,
        alpha=0.15,
        label=model,
    )

ax.set_xlabel("Action disagreement")
ax.set_ylabel(r"$C_{10}=|\Delta J(10)|$")
ax.set_title("Counterfactual consequence vs. action disagreement")
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(
    OUT / "fig1_consequence_vs_action_disagreement.png",
    dpi=300,
)
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: held-out MAE vs no-signal baseline
# ---------------------------------------------------------------------

fold = pd.read_csv(
    RESULTS / "statistics" / "heldout_fold_metrics.csv"
)

base = pd.read_csv(
    RESULTS / "statistics" / "no_signal_baseline.csv"
)

fold = fold[fold["horizon"] == 10].copy()
base = base[base["horizon"] == 10].copy()

base_mean_mae = base["mae"].mean()

model_summary = (
    fold
    .groupby(["model", "model_label"], as_index=False)
    .agg(
        mean_mae=("mae", "mean"),
        sd_mae=("mae", "std"),
    )
)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

x = np.arange(len(model_summary))

ax.axhline(
    base_mean_mae,
    linestyle="--",
    label=f"No-signal baseline ({base_mean_mae:.4f})",
)

ax.errorbar(
    x,
    model_summary["mean_mae"],
    yerr=model_summary["sd_mae"],
    fmt="o",
    capsize=4,
)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "Action",
        "Action+Support",
        "Action+Critic",
        "Action+Support+Critic",
        "Action+Support+Interaction",
    ],
    rotation=25,
    ha="right",
)

ax.set_ylabel("Held-out MAE")
ax.set_title("Five-seed held-out consequence prediction")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(
    OUT / "fig2_heldout_mae_baseline.png",
    dpi=300,
)
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: risk coverage
# ---------------------------------------------------------------------

rc_all = pd.read_csv(
    RESULTS
    / "final_tables"
    / "risk_coverage_all_shifts.csv"
)

rc_nonzero = pd.read_csv(
    RESULTS
    / "final_tables"
    / "risk_coverage_nonzero_shift.csv"
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for model in sorted(rc_nonzero["model"].unique()):
    d = rc_nonzero[
        rc_nonzero["model"] == model
    ]

    ax.plot(
        d["coverage_target"],
        d["mean_observed_C10"],
        marker="o",
        linewidth=1.5,
        markersize=3,
        label=model,
    )

ax.set_xlabel("Coverage")
ax.set_ylabel(r"Mean observed $C_{10}$")
ax.set_title(
    "Risk–coverage sensitivity: nonzero shifts only"
)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(
    OUT / "fig3_risk_coverage_nonzero_shift.png",
    dpi=300,
)
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 4: horizon sensitivity
# ---------------------------------------------------------------------

summary = pd.read_csv(
    RESULTS / "tables" / "heldout_model_comparison.csv"
)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

for model in summary["model"].unique():
    d = summary[
        summary["model"] == model
    ].sort_values("horizon")

    ax.plot(
        d["horizon"],
        d["mean_mae"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=model,
    )

ax.set_xlabel("Prediction horizon")
ax.set_ylabel("Mean held-out MAE")
ax.set_title("Prediction error across downstream horizons")
ax.set_xticks([1, 5, 10, 20])
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(
    OUT / "fig4_horizon_sensitivity.png",
    dpi=300,
)
plt.close(fig)


print("=" * 80)
print("P1 FINAL FIGURES GENERATED")
print("=" * 80)

for path in sorted(OUT.glob("*.png")):
    print(path)
