
#!/usr/bin/env python3
"""
Final paper tables + Prism-ready figure sources for OfflineRL-Reliability.

IMPORTANT:
    Run this from the repository checkout:
        python src/evaluation/build_final_paper_assets.py

Before running:
    cp /mnt/data/build_final_paper_assets.py \
       src/evaluation/build_final_paper_assets.py

This script NEVER trains or recollects RL data. It consumes frozen P1-P10
analysis artifacts already present in the repository.

Outputs:
    results/paper_assets/
        tables/          CSV + Markdown paper tables
        figure_sources/  CSV files for GraphPad Prism
        figures/         PNG/SVG QC previews
        PAPER_FIGURE_TABLE_PLAN.md
        PAPER_ASSET_MANIFEST.csv
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_assets"
TABLES = OUT / "tables"
FIGSRC = OUT / "figure_sources"
FIGURES = OUT / "figures"

CELLS = [
    "IQL_Hopper",
    "IQL_HalfCheetah",
    "IQL_Walker2d",
    "CQL_Hopper",
    "CQL_HalfCheetah",
    "CQL_Walker2d",
]

SUMMARY_PATHS = {
    "P1": ROOT / "results/analysis/P1/P1_EVIDENCE_SUMMARY.md",
    "P2": ROOT / "results/analysis/P2/P2_EVIDENCE_SUMMARY.md",
    "P3": ROOT / "results/analysis/P3/P3_EVIDENCE_SUMMARY.md",
    "P4": ROOT / "results/analysis/P4/P4_EVIDENCE_SUMMARY.md",
    "P5": ROOT / "results/analysis/P5_CQL/P5_EVIDENCE_SUMMARY.md",
    "P6": ROOT / "results/analysis/P6_CQL_Walker2d/P6_EVIDENCE_SUMMARY.md",
    "P7": ROOT / "results/analysis/P7_CQL_HalfCheetah/CQL_HALFCHEETAH_EVIDENCE_SUMMARY.md",
    "P8": ROOT / "results/analysis/P8_DISTANCE_MATCHED/P8_EVIDENCE_SUMMARY.md",
    "P9": ROOT / "results/analysis/P9/P9_EVIDENCE_SUMMARY.md",
    "P10": ROOT / "results/analysis/P10/P10_EVIDENCE_SUMMARY.md",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    return path


def txt(path: Path) -> str:
    return require(path).read_text(encoding="utf-8")


def extract_float(pattern: str, text: str, label: str) -> float:
    m = re.search(pattern, text, flags=re.I | re.M)
    if not m:
        raise ValueError(f"Could not parse {label}. Pattern={pattern!r}")
    return float(m.group(1))


def extract_pair(pattern: str, text: str, label: str) -> Tuple[float, float]:
    m = re.search(pattern, text, flags=re.I | re.M)
    if not m:
        raise ValueError(f"Could not parse {label}. Pattern={pattern!r}")
    return float(m.group(1)), float(m.group(2))


def save_df(df: pd.DataFrame, stem: str, title: str) -> None:
    csv_path = TABLES / f"{stem}.csv"
    md_path = TABLES / f"{stem}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n{df.to_markdown(index=False)}\n", encoding="utf-8")


def parse_p4(summary: str) -> pd.DataFrame:
    sections = {
        "B1_support_only": "Support / OOD only",
        "B2_critic_only": "Critic uncertainty only",
        "B3_support_plus_critic": "Support + critic",
        "B4_action_only": "Action disagreement only",
        "B5_action_support_critic": "Action + support + critic",
    }
    rows = []
    for key, heading in sections.items():
        m = re.search(
            rf"###\s+{re.escape(heading)}(.*?)(?=\n###\s+|\n##\s+|\Z)",
            summary,
            flags=re.I | re.S,
        )
        if not m:
            raise ValueError(f"P4 section not found: {heading}")
        sec = m.group(1)
        rho = extract_float(
            r"Mean Spearman\(R,C10\):\s*([0-9.eE+-]+)",
            sec,
            f"P4 {key} Spearman",
        )
        rho_sd = extract_float(
            r"\(SD\s*([0-9.eE+-]+)\)",
            sec,
            f"P4 {key} Spearman SD",
        )
        mae = np.nan
        rmse = np.nan
        if "Mean MAE:" in sec:
            mae = extract_float(r"Mean MAE:\s*([0-9.eE+-]+)", sec, f"P4 {key} MAE")
            rmse = extract_float(r"Mean RMSE:\s*([0-9.eE+-]+)", sec, f"P4 {key} RMSE")
        rows.append(
            {
                "baseline": key,
                "mean_spearman_R_C10": rho,
                "sd_spearman_R_C10": rho_sd,
                "mean_MAE": mae,
                "mean_RMSE": rmse,
            }
        )
    return pd.DataFrame(rows)


def parse_p8(summary: str) -> pd.DataFrame:
    blocks = re.split(r"^###\s+", summary, flags=re.M)
    rows = []
    for block in blocks:
        if not block.strip():
            continue
        cell = block.splitlines()[0].strip()
        if cell not in CELLS:
            continue
        mean = extract_float(
            r"Mean seed-level matched ΔC10:\s*([0-9.eE+-]+)",
            block,
            f"P8 {cell} mean",
        )
        sd = extract_float(r"SD across seeds:\s*([0-9.eE+-]+)", block, f"P8 {cell} SD")
        lo, hi = extract_pair(
            r"95% t-based CI:\s*\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]",
            block,
            f"P8 {cell} CI",
        )
        one = extract_float(
            r"Exact one-sided sign-flip p:\s*([0-9.eE+-]+)",
            block,
            f"P8 {cell} one-sided p",
        )
        two = extract_float(
            r"Exact two-sided sign-flip p:\s*([0-9.eE+-]+)",
            block,
            f"P8 {cell} two-sided p",
        )
        pos = re.search(r"Positive seed effects:\s*(\d+)/(\d+)", block, flags=re.I)
        if not pos:
            raise ValueError(f"P8 positive-seed count missing for {cell}")
        pairs = int(extract_float(r"Matched pairs:\s*(\d+)", block, f"P8 {cell} matched pairs"))
        rows.append(
            {
                "cell": cell,
                "mean_matched_delta_C10": mean,
                "sd": sd,
                "ci_low": lo,
                "ci_high": hi,
                "error_minus": mean - lo,
                "error_plus": hi - mean,
                "positive_seed_fraction": int(pos.group(1)) / int(pos.group(2)),
                "one_sided_p": one,
                "two_sided_p": two,
                "matched_pairs": pairs,
            }
        )
    got = {r["cell"] for r in rows}
    if got != set(CELLS):
        raise ValueError(f"P8 did not parse all six cells. Parsed={sorted(got)}")
    return pd.DataFrame(rows)


def parse_cql_primary(summary: str, cell: str, style: str) -> dict:
    """
    Parse the frozen CQL P5/P6/P7 seed-level primary slopes.

    P5 stores the five slopes in a Markdown table:
        | Seed | Records | States | Finite slopes | Mean state slope | ... |

    P6/P7 store the five slopes as prose lines:
        seed 0: ...
        seed 1: ...

    No scientific values are generated here; values are extracted from the
    already frozen evidence summaries.
    """

    seed_values = {}

    # ---------------------------------------------------------------
    # P5: Markdown-table format
    # ---------------------------------------------------------------
    if style == "P5":
        # P5 contains several Markdown tables with a "Seed" column.
        # Scope extraction strictly to the frozen primary seed-level table
        # so dose-response rows cannot overwrite the primary slopes.
        section_match = re.search(
            r"##\s+Primary seed-level results(.*?)(?=\n##\s+|\Z)",
            summary,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not section_match:
            raise ValueError(
                f"{cell}: P5 primary seed-level results section not found"
            )

        primary_section = section_match.group(1)

        row_pattern = re.compile(
            r"^\|\s*([0-4])\s*\|"
            r"\s*600\s*\|"
            r"\s*100\s*\|"
            r"\s*100\s*\|"
            r"\s*([0-9.eE+-]+)\s*\|",
            flags=re.MULTILINE,
        )

        for m in row_pattern.finditer(primary_section):
            seed = int(m.group(1))
            value = float(m.group(2))
            seed_values[seed] = value

    # ---------------------------------------------------------------
    # P6/P7: prose seed lines
    # ---------------------------------------------------------------
    else:
        for seed in range(5):
            m = re.search(
                rf"^\s*-?\s*seed\s+{seed}\s*:\s*([0-9.eE+-]+)\s*$",
                summary,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if m:
                seed_values[seed] = float(m.group(1))

    missing_seeds = [s for s in range(5) if s not in seed_values]
    if missing_seeds:
        raise ValueError(
            f"{cell}: missing seed slope(s): {missing_seeds}"
        )

    seed_lines = [seed_values[s] for s in range(5)]

    # ---------------------------------------------------------------
    # Parse frozen cross-seed summary
    # ---------------------------------------------------------------
    if style == "P5":
        mean = extract_float(
            r"Mean seed-level slope:\s*`?([0-9.eE+-]+)",
            summary,
            f"{cell} mean",
        )
        lo, hi = extract_pair(
            r"95% t-based CI:\s*`?\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]",
            summary,
            f"{cell} CI",
        )
        one = extract_float(
            r"Exact one-sided sign-flip p:\s*`?([0-9.eE+-]+)",
            summary,
            f"{cell} p1",
        )
        two = extract_float(
            r"Exact two-sided sign-flip p:\s*`?([0-9.eE+-]+)",
            summary,
            f"{cell} p2",
        )

    elif style == "P6":
        mean = extract_float(
            r"mean\s*=\s*([0-9.eE+-]+)",
            summary,
            f"{cell} mean",
        )
        lo, hi = extract_pair(
            r"95% t-based CI\s*=\s*\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]",
            summary,
            f"{cell} CI",
        )
        one = extract_float(
            r"Exact one-sided sign-flip p\s*=\s*([0-9.eE+-]+)",
            summary,
            f"{cell} p1",
        )
        two = extract_float(
            r"Exact two-sided sign-flip p\s*=\s*([0-9.eE+-]+)",
            summary,
            f"{cell} p2",
        )

    elif style == "P7":
        mean = extract_float(
            r"- mean:\s*([0-9.eE+-]+)",
            summary,
            f"{cell} mean",
        )
        lo, hi = extract_pair(
            r"- 95% CI:\s*\[([0-9.eE+-]+),\s*([0-9.eE+-]+)\]",
            summary,
            f"{cell} CI",
        )
        one = extract_float(
            r"- exact one-sided sign-flip p:\s*([0-9.eE+-]+)",
            summary,
            f"{cell} p1",
        )
        two = extract_float(
            r"- exact two-sided sign-flip p:\s*([0-9.eE+-]+)",
            summary,
            f"{cell} p2",
        )

    else:
        raise ValueError(f"Unsupported CQL summary style: {style}")

    # ---------------------------------------------------------------
    # Integrity cross-check:
    # recompute the mean from the five frozen seed-level values.
    # ---------------------------------------------------------------
    recomputed_mean = float(np.mean(seed_lines))
    if not np.isclose(recomputed_mean, mean, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"{cell}: stored mean slope {mean:.12g} does not match "
            f"recomputed seed mean {recomputed_mean:.12g}"
        )

    positive_count = int(sum(v > 0 for v in seed_lines))
    if positive_count != 5:
        raise ValueError(
            f"{cell}: expected all five seed slopes positive in the "
            f"frozen primary replication, found {positive_count}/5"
        )

    if not (0.0 <= one <= 1.0 and 0.0 <= two <= 1.0):
        raise ValueError(
            f"{cell}: invalid sign-flip p-values: one-sided={one}, two-sided={two}"
        )

    return {
        "cell": cell,
        "seed0": seed_lines[0],
        "seed1": seed_lines[1],
        "seed2": seed_lines[2],
        "seed3": seed_lines[3],
        "seed4": seed_lines[4],
        "mean_slope": mean,
        "ci_low": lo,
        "ci_high": hi,
        "error_minus": mean - lo,
        "error_plus": hi - mean,
        "positive_seed_count": positive_count,
        "one_sided_p": one,
        "two_sided_p": two,
    }


def parse_p5_dose(summary: str) -> pd.DataFrame:
    m = re.search(r"## Dose-response values(.*?)(?=\n##\s+|\Z)", summary, flags=re.S)
    if not m:
        raise ValueError("P5 dose-response section not found")
    rows = []
    for line in m.group(1).splitlines():
        if not line.startswith("|") or "Seed" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "seed": int(parts[0]),
                "sigma": float(parts[1]),
                "mean_action_disagreement": float(parts[2]),
                "mean_absolute_consequence": float(parts[3]),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != 30:
        raise ValueError(f"P5 dose-response expected 30 rows, found {len(df)}")
    return df


def parse_p1_predictive_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparison = require(
        ROOT / "results/analysis/P1/tables/heldout_model_comparison.csv"
    )
    risk = require(
        ROOT / "results/analysis/P1/final_tables/risk_coverage_nonzero_shift.csv"
    )
    return pd.read_csv(comparison), pd.read_csv(risk)


def parse_p9() -> pd.DataFrame:
    p = require(ROOT / "results/analysis/P9/statistics/P9_cell_model_summary.csv")
    df = pd.read_csv(p)

    # These are the actual frozen P9 column names.
    # Keep them unchanged at the source; normalize only inside the
    # paper-asset builder for a stable downstream schema.
    needed = {
        "cell",
        "model",
        "n_heldout_policy_seeds",
        "mean_mae",
        "sd_mae",
        "mean_rmse",
        "sd_rmse",
        "mean_spearman",
        "sd_spearman",
    }

    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"P9 model summary missing columns: {sorted(missing)}"
        )

    df = df.rename(
        columns={
            "n_heldout_policy_seeds": "n_heldout_seeds",
            "mean_spearman": "mean_spearman_reliability_c10",
            "sd_spearman": "sd_spearman_reliability_c10",
        }
    )

    return df


def parse_p10() -> pd.DataFrame:
    p = require(ROOT / "results/analysis/P10/summaries/P10_any_ood_cell_summary.csv")
    df = pd.read_csv(p)
    needed = {
        "cell",
        "score",
        "mean_auroc",
        "sd_auroc",
        "mean_average_precision",
        "sd_average_precision",
        "n_policy_seeds",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"P10 summary missing columns: {sorted(missing)}")
    return df


def build_plan() -> str:
    return dedent(
        """
        # Final Paper Figure / Table Plan

        The frozen computational package is complete. This directory is the
        source package for the manuscript figures and tables.

        ## Main-paper figures

        ### Figure 1 — P1 held-out consequence prediction
        Import:
        `figure_sources/F1_P1_heldout_model_comparison.csv`

        Recommended Prism graph:
        grouped dot/bar plot of held-out MAE (and a separate RMSE panel)
        across the prespecified candidate models. Include the no-signal
        baseline. Do not label a single candidate as universally optimal.

        ### Figure 2 — P8 distance-matched control across six cells
        Import:
        `figure_sources/F2_P8_distance_matched.csv`

        Recommended Prism graph:
        point estimates with 95% t-based CI, one row per algorithm/environment
        cell. Keep the zero reference line. This is the principal cross-cell
        control figure.

        ### Figure 3 — P9 cross-cell reliability estimator
        Import:
        `figure_sources/F3_P9_reliability_summary.csv`

        Recommended Prism graph:
        primary 3-feature estimator: Spearman(R,C10) by cell with mean ± SD
        across held-out policy seeds. A companion graph can show MAE/RMSE.

        ### Figure 4 — P9 risk-coverage
        Import:
        `figure_sources/F4_P9_risk_coverage_nonzero_shift.csv`

        Recommended Prism graph:
        risk/consequence versus coverage, using the frozen summary values.
        The CSV is authoritative; do not manually recreate values.

        ## Supplementary figures

        ### Figure S1 — CQL primary replication
        Import:
        `figure_sources/F5_CQL_primary_replication.csv`

        This summarizes P5/P6/P7 seed-level slopes. Use error bars only as the
        stored 95% t-based CIs and do not compare raw magnitudes across
        environments as standardized effect sizes.

        ### Figure S2 — P4 baseline comparison
        Import:
        `figure_sources/F6_P4_baselines.csv`

        Descriptive comparison of support/OOD, critic uncertainty, action
        disagreement, and combined formulations.

        ### Figure S3 — P10 controlled-shift detectability
        Import:
        `figure_sources/F7_P10_controlled_shift_ood_sanity.csv`

        Label explicitly:
        "Controlled observation-shift detectability sanity check"
        and not "generic OOD detection".

        ### Figure S4 — CQL Hopper dose response
        Import:
        `figure_sources/F8_P5_CQL_Hopper_dose_response.csv`

        Useful for showing the monotonic controlled-shift dose response.

        ## Main-paper tables

        Table 1: `tables/T1_experiment_matrix.csv`
        Table 2: `tables/T2_P1_heldout_predictive_comparison.csv`
        Table 3: `tables/T3_P8_distance_matched_control.csv`
        Table 4: `tables/T4_P9_reliability_estimator.csv`

        ## Supplementary tables

        Table S1: `tables/T5_P4_baseline_comparison.csv`
        Table S2: `tables/T6_P10_controlled_shift_ood_sanity.csv`
        Table S3: `tables/T7_CQL_primary_replication.csv`
        Table S4: `tables/T8_P5_CQL_Hopper_dose_response.csv`

        ## Statistical guardrails

        - Independent replication unit is policy seed within cell.
        - P9 is leave-one-policy-seed-out within each cell.
        - P8 and P3 are control/association analyses, not causal identification.
        - P4 is descriptive and does not establish a universal best baseline.
        - P10 tests the frozen controlled perturbation family only.
        - Exact sign-flip p-values and t-based confidence intervals are distinct
          inferential procedures.
        - Do not pool IQL and CQL policy seeds for primary inference.
        - Do not rank raw slope magnitudes across environments as if they were
          standardized effect sizes.
        """
    ).strip() + "\n"


def main() -> None:
    for p in (TABLES, FIGSRC, FIGURES):
        p.mkdir(parents=True, exist_ok=True)

    summaries = {k: txt(v) for k, v in SUMMARY_PATHS.items()}

    # ------------------------------ T1 ---------------------------------
    t1 = pd.DataFrame(
        [
            ["IQL", "Hopper", "P1", "P3", "P2/P9", "P10"],
            ["IQL", "HalfCheetah", "consequence collection + P8/P9", "P8", "P9", "P10"],
            ["IQL", "Walker2d", "IQL consequence analysis", "P8", "P9", "P10"],
            ["CQL", "Hopper", "P5", "P8", "P9", "P10"],
            ["CQL", "HalfCheetah", "P7", "P8", "P9", "P10"],
            ["CQL", "Walker2d", "P6", "P8", "P9", "P10"],
        ],
        columns=[
            "algorithm",
            "environment",
            "primary_consequence_experiment",
            "distance_matched",
            "reliability_estimator",
            "controlled_shift_sanity",
        ],
    )
    save_df(t1, "T1_experiment_matrix", "Table 1 — Experimental matrix")

    # ------------------------------ T2 ---------------------------------
    p1_comparison, p1_risk = parse_p1_predictive_sources()
    p1_comparison.to_csv(FIGSRC / "F1_P1_heldout_model_comparison.csv", index=False)
    p1_risk.to_csv(FIGSRC / "F4_P1_risk_coverage_nonzero_shift.csv", index=False)

    # Preserve P1's actual source table rather than rewriting scientific values.
    p1_table = p1_comparison.copy()
    save_df(
        p1_table,
        "T2_P1_heldout_predictive_comparison",
        "Table 2 — P1 held-out predictive comparison",
    )

    # ------------------------------ T3 ---------------------------------
    p8 = parse_p8(summaries["P8"]).sort_values("cell")
    p8.to_csv(FIGSRC / "F2_P8_distance_matched.csv", index=False)
    save_df(
        p8,
        "T3_P8_distance_matched_control",
        "Table 3 — P8 distance-matched control",
    )

    # ------------------------------ T4 ---------------------------------
    p9 = parse_p9()
    p9_primary = p9[p9["model"] == "primary_3feature"].copy().sort_values("cell")
    p9_primary.to_csv(FIGSRC / "F3_P9_reliability_summary.csv", index=False)

    p9_all = p9.copy().sort_values(["cell", "model"])
    save_df(
        p9_all,
        "T4_P9_reliability_estimator",
        "Table 4 — P9 cross-cell reliability estimator",
    )

    p9_risk = require(
        ROOT / "results/analysis/P9/risk_coverage/P9_risk_coverage_nonzero_shift_summary.csv"
    )
    p9_risk_df = pd.read_csv(p9_risk)
    p9_risk_df.to_csv(FIGSRC / "F4_P9_risk_coverage_nonzero_shift.csv", index=False)

    # ------------------------------ T5 ---------------------------------
    p4 = parse_p4(summaries["P4"])
    p4.to_csv(FIGSRC / "F6_P4_baselines.csv", index=False)
    save_df(p4, "T5_P4_baseline_comparison", "Table S1 — P4 baseline comparison")

    # ------------------------------ T6 ---------------------------------
    p10 = parse_p10().sort_values(["cell", "score"])
    p10.to_csv(FIGSRC / "F7_P10_controlled_shift_ood_sanity.csv", index=False)
    save_df(
        p10,
        "T6_P10_controlled_shift_ood_sanity",
        "Table S2 — P10 controlled-shift detectability sanity check",
    )

    # ------------------------------ T7 ---------------------------------
    cql = pd.DataFrame(
        [
            parse_cql_primary(summaries["P5"], "CQL_Hopper", "P5"),
            parse_cql_primary(summaries["P7"], "CQL_HalfCheetah", "P7"),
            parse_cql_primary(summaries["P6"], "CQL_Walker2d", "P6"),
        ]
    ).sort_values("cell")
    cql.to_csv(FIGSRC / "F5_CQL_primary_replication.csv", index=False)
    save_df(
        cql,
        "T7_CQL_primary_replication",
        "Table S3 — CQL primary consequence replication",
    )

    # ------------------------------ T8 ---------------------------------
    p5_dose = parse_p5_dose(summaries["P5"]).sort_values(["seed", "sigma"])
    p5_dose.to_csv(FIGSRC / "F8_P5_CQL_Hopper_dose_response.csv", index=False)
    save_df(
        p5_dose,
        "T8_P5_CQL_Hopper_dose_response",
        "Table S4 — P5 CQL Hopper dose response",
    )

    # -------------------------- QC figures ------------------------------
    # Figure 2: P8 distance-matched
    plot = p8
    y = np.arange(len(plot))
    plt.figure(figsize=(8, 4.8))
    plt.errorbar(
        plot["mean_matched_delta_C10"],
        y,
        xerr=[plot["error_minus"], plot["error_plus"]],
        fmt="o",
        capsize=4,
    )
    plt.yticks(y, plot["cell"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Matched ΔC10")
    plt.ylabel("Algorithm / environment")
    plt.title("P8 distance-matched control")
    plt.tight_layout()
    plt.savefig(FIGURES / "F2_P8_distance_matched.png", dpi=300)
    plt.savefig(FIGURES / "F2_P8_distance_matched.svg")
    plt.close()

    # Figure 3: P9 primary Spearman
    plot = p9_primary
    y = np.arange(len(plot))
    plt.figure(figsize=(8, 4.8))
    plt.errorbar(
        plot["mean_spearman_reliability_c10"],
        y,
        xerr=plot["sd_spearman_reliability_c10"],
        fmt="o",
        capsize=4,
    )
    plt.yticks(y, plot["cell"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Spearman(R, C10), mean ± SD")
    plt.ylabel("Algorithm / environment")
    plt.title("P9 cross-cell reliability estimator")
    plt.tight_layout()
    plt.savefig(FIGURES / "F3_P9_reliability_spearman.png", dpi=300)
    plt.savefig(FIGURES / "F3_P9_reliability_spearman.svg")
    plt.close()

    # Figure 5: CQL primary replication
    plot = cql
    y = np.arange(len(plot))
    plt.figure(figsize=(8, 4.2))
    plt.errorbar(
        plot["mean_slope"],
        y,
        xerr=[plot["error_minus"], plot["error_plus"]],
        fmt="o",
        capsize=4,
    )
    plt.yticks(y, plot["cell"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Mean state-level slope")
    plt.ylabel("CQL cell")
    plt.title("CQL consequence replication")
    plt.tight_layout()
    plt.savefig(FIGURES / "F5_CQL_primary_replication.png", dpi=300)
    plt.savefig(FIGURES / "F5_CQL_primary_replication.svg")
    plt.close()

    # Figure 7: P10
    pivot = p10.pivot(index="cell", columns="score", values="mean_auroc").loc[CELLS]
    plt.figure(figsize=(8.5, 4.8))
    for score in pivot.columns:
        plt.plot(pivot.index, pivot[score], marker="o", label=score)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(-0.02, 1.02)
    plt.ylabel("Mean AUROC")
    plt.xlabel("Algorithm / environment")
    plt.title("P10 controlled-shift detectability sanity check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "F7_P10_controlled_shift_ood_sanity.png", dpi=300)
    plt.savefig(FIGURES / "F7_P10_controlled_shift_ood_sanity.svg")
    plt.close()

    # Figure 8: P5 dose response
    agg = (
        p5_dose.groupby("sigma", as_index=False)[
            ["mean_action_disagreement", "mean_absolute_consequence"]
        ]
        .mean()
        .sort_values("sigma")
    )
    plt.figure(figsize=(7.4, 4.8))
    plt.plot(
        agg["sigma"],
        agg["mean_action_disagreement"],
        marker="o",
        label="Action disagreement",
    )
    plt.plot(
        agg["sigma"],
        agg["mean_absolute_consequence"],
        marker="s",
        label="Absolute consequence",
    )
    plt.xlabel("Observation-shift sigma")
    plt.ylabel("Mean across five policy seeds")
    plt.title("P5 CQL Hopper dose response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "F8_P5_CQL_Hopper_dose_response.png", dpi=300)
    plt.savefig(FIGURES / "F8_P5_CQL_Hopper_dose_response.svg")
    plt.close()

    # --------------------------- plan + manifest ------------------------
    (OUT / "PAPER_FIGURE_TABLE_PLAN.md").write_text(build_plan(), encoding="utf-8")

    manifest = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "PAPER_ASSET_MANIFEST.csv":
            manifest.append(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(manifest).to_csv(OUT / "PAPER_ASSET_MANIFEST.csv", index=False)

    print("=" * 80)
    print("FINAL PAPER ASSET PACKAGE BUILT")
    print("=" * 80)
    print(f"Output: {OUT}")
    print()
    print("Tables:", len(list(TABLES.glob("*.csv"))))
    print("Prism-ready sources:", len(list(FIGSRC.glob("*.csv"))))
    print("QC figures:", len(list(FIGURES.glob("*.png"))))
    print()
    print("Validation: PASS")
    print("No RL training or consequence collection was performed.")
    print("CSV files are the authoritative data sources for Prism.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "No partial scientific conclusion should be used. Fix the missing "
            "artifact/schema and rerun.",
            file=sys.stderr,
        )
        raise
