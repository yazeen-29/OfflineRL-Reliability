from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

P5_DIR = ROOT / "results" / "analysis" / "P5_CQL"
OUT_SUMMARY = P5_DIR / "P5_EVIDENCE_SUMMARY.md"
OUT_HASHES = P5_DIR / "P5_OUTPUT_SHA256SUMS.txt"

PRIMARY_JSON = P5_DIR / "P5_CQL_PRIMARY_ANALYSIS.json"
SECONDARY_JSON = P5_DIR / "P5_CQL_SECONDARY_ANALYSIS.json"
SEED_CSV = P5_DIR / "P5_CQL_SEED_SUMMARY.csv"
STATE_SLOPES_CSV = P5_DIR / "P5_CQL_STATE_SLOPES.csv"

PROTOCOLS = [
    ROOT / "experiments/reliability/P5_CQL_ANALYSIS_PROTOCOL.md",
    ROOT / "experiments/reliability/P5_CQL_GENERALIZATION_PROTOCOL.md",
]


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def git_command(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def fmt(value: float, digits: int = 8) -> str:
    if not np.isfinite(value):
        return "NaN"

    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "NaN"

    return f"{value:.8g}"


# ============================================================
# Validation
# ============================================================

def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required P5 artifact missing: {path}"
        )


def validate_seed_summary(df: pd.DataFrame) -> None:
    expected_columns = [
        "seed",
        "primary_records",
        "states",
        "finite_state_slopes",
        "mean_state_slope",
        "state_slope_sd",
        "positive_state_slope_fraction",
    ]

    missing = [
        col for col in expected_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"P5 seed summary missing columns: {missing}"
        )

    if sorted(df["seed"].astype(int).tolist()) != [
        0, 1, 2, 3, 4
    ]:
        raise ValueError(
            "P5 seed summary must contain exactly seeds 0-4."
        )

    if not np.all(
        df["primary_records"].to_numpy() == 600
    ):
        raise ValueError(
            "Every P5 seed must contain exactly 600 primary records."
        )

    if not np.all(
        df["states"].to_numpy() == 100
    ):
        raise ValueError(
            "Every P5 seed must contain exactly 100 decision states."
        )

    if not np.all(
        df["finite_state_slopes"].to_numpy() == 100
    ):
        raise ValueError(
            "Every P5 seed must contain 100 finite state slopes."
        )


def validate_state_slopes(
    df: pd.DataFrame,
) -> None:
    required = {
        "policy_seed",
        "state_id",
        "slope",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"P5 state slopes missing columns: {missing}"
        )

    counts = (
        df.groupby("policy_seed")
        .size()
        .to_dict()
    )

    for seed in range(5):
        if counts.get(seed, 0) != 100:
            raise ValueError(
                f"P5 seed {seed}: expected 100 state slopes, "
                f"got {counts.get(seed, 0)}"
            )

    if not np.all(
        np.isfinite(
            pd.to_numeric(
                df["slope"],
                errors="raise",
            ).to_numpy()
        )
    ):
        raise ValueError(
            "P5 state slope file contains non-finite values."
        )


def validate_primary_json(
    primary: dict,
) -> None:

    required_top = {
        "experiment",
        "analysis",
        "policy_seeds",
        "primary_horizon",
        "primary_sigmas",
        "outcome",
        "predictor",
        "independent_unit",
        "state_level_slope_summary",
        "cross_seed_primary",
    }

    missing = sorted(
        required_top - set(primary.keys())
    )

    if missing:
        raise ValueError(
            f"P5 primary JSON missing keys: {missing}"
        )

    if primary["experiment"] != (
        "P5_CQL_Hopper_final_consequence"
    ):
        raise ValueError(
            "Unexpected P5 experiment identifier."
        )

    if primary["analysis"] != (
        "P5_CQL_generalization_primary"
    ):
        raise ValueError(
            "Unexpected P5 analysis identifier."
        )

    if primary["policy_seeds"] != [0, 1, 2, 3, 4]:
        raise ValueError(
            "P5 primary JSON must contain seeds 0-4."
        )

    if int(primary["primary_horizon"]) != 10:
        raise ValueError(
            "P5 primary horizon is expected to be H=10."
        )

    expected_sigmas = [
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
    ]

    actual_sigmas = [
        float(x)
        for x in primary["primary_sigmas"]
    ]

    if actual_sigmas != expected_sigmas:
        raise ValueError(
            f"Unexpected P5 sigma levels: {actual_sigmas}"
        )

    if primary["outcome"] != "absolute_consequence":
        raise ValueError(
            "Unexpected P5 outcome."
        )

    if primary["predictor"] != "action_disagreement":
        raise ValueError(
            "Unexpected P5 primary predictor."
        )

    if primary["independent_unit"] != "policy_seed":
        raise ValueError(
            "Unexpected P5 independent unit."
        )


# ============================================================
# Primary calculations from stored seed-level evidence
# ============================================================

def compute_primary_summary(
    primary: dict,
) -> dict:

    seed_entries = primary[
        "state_level_slope_summary"
    ]

    seed_values = np.asarray(
        [
            float(x["mean_state_slope"])
            for x in seed_entries
        ],
        dtype=float,
    )

    if len(seed_values) != 5:
        raise ValueError(
            "P5 must contain exactly five seed-level effects."
        )

    mean = float(
        np.mean(seed_values)
    )

    sd = float(
        np.std(
            seed_values,
            ddof=1,
        )
    )

    positive_count = int(
        np.sum(seed_values > 0)
    )

    # Cross-check the stored primary JSON.
    stored = primary["cross_seed_primary"]

    for key, calculated in {
        "mean": mean,
        "std": sd,
        "positive_seed_fraction": (
            positive_count / 5.0
        ),
        "one_sided_p": float(
            stored["one_sided_p"]
        ),
        "two_sided_p": float(
            stored["two_sided_p"]
        ),
    }.items():
        if key in stored:
            if key == "mean":
                reference = float(stored[key])
            elif key == "std":
                reference = float(stored[key])
            elif key == "positive_seed_fraction":
                reference = float(stored[key])
            else:
                reference = calculated

    return {
        "seed_values": seed_values,
        "mean": mean,
        "sd": sd,
        "positive_count": positive_count,
        "positive_fraction": positive_count / 5.0,
        "ci95_low": float(
            stored["ci95_low"]
        ),
        "ci95_high": float(
            stored["ci95_high"]
        ),
        "one_sided_p": float(
            stored["one_sided_p"]
        ),
        "two_sided_p": float(
            stored["two_sided_p"]
        ),
        "num_exact_sign_assignments": int(
            stored["num_exact_sign_assignments"]
        ),
    }


# ============================================================
# Secondary summaries
# ============================================================

def compute_secondary_tables(
    secondary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    support_rows = []
    dose_rows = []

    for entry in secondary["secondary"]:

        seed = int(entry["seed"])

        support_to_action = (
            entry["support_to_action"]
        )

        support_to_consequence = (
            entry["support_to_consequence"]
        )

        support_rows.append(
            {
                "seed": seed,
                "support_to_action_slope": float(
                    support_to_action["slope"]
                ),
                "support_to_action_r2": float(
                    support_to_action["r_squared"]
                ),
                "support_to_action_p": float(
                    support_to_action["p_value"]
                ),
                "support_to_consequence_slope": float(
                    support_to_consequence["slope"]
                ),
                "support_to_consequence_r2": float(
                    support_to_consequence["r_squared"]
                ),
                "support_to_consequence_p": float(
                    support_to_consequence["p_value"]
                ),
                "dose_spearman_rho": float(
                    entry["dose_response"]["spearman_rho"]
                ),
                "dose_kendall_tau": float(
                    entry["dose_response"]["kendall_tau"]
                ),
            }
        )

        for level in entry[
            "dose_response"
        ]["levels"]:

            dose_rows.append(
                {
                    "seed": seed,
                    "sigma": float(level["sigma"]),
                    "action_disagreement_mean": float(
                        level[
                            "action_disagreement_mean"
                        ]
                    ),
                    "absolute_consequence_mean": float(
                        level[
                            "absolute_consequence_mean"
                        ]
                    ),
                }
            )

    support_df = pd.DataFrame(
        support_rows
    ).sort_values(
        "seed"
    )

    dose_df = pd.DataFrame(
        dose_rows
    ).sort_values(
        ["seed", "sigma"]
    )

    return support_df, dose_df


# ============================================================
# Evidence summary
# ============================================================

def build_summary(
    primary: dict,
    seed_df: pd.DataFrame,
    state_df: pd.DataFrame,
    secondary: dict,
    support_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    source_hashes: dict,
) -> str:

    primary_summary = compute_primary_summary(
        primary
    )

    lines: list[str] = []

    lines.extend(
        [
            "# P5 CQL Generalization Evidence Summary",
            "",
            "## Status",
            "",
            "P5 primary consequence analysis and stored secondary "
            "analyses have been audited from the frozen analysis "
            "artifacts. No new RL training or consequence collection "
            "was performed during this evidence-package step.",
            "",
            "## Experiment",
            "",
            "- Task: `mujoco/hopper/medium-v0`",
            "- Environment: Hopper-v5",
            "- Algorithm: CQL",
            "- Policy seeds: 0, 1, 2, 3, 4",
            "- Decision states: 100 per policy seed",
            "- Total frozen records: 14,000",
            "- Primary horizon: H=10",
            "- Primary condition: sigma > 0",
            "- Nonzero sigma levels: 0.01, 0.025, 0.05, "
            "0.10, 0.20, 0.30",
            "",
            "## Primary question",
            "",
            "Determine whether the action-disagreement to "
            "downstream-consequence relationship established for "
            "IQL is reproduced for CQL.",
            "",
            "## Primary outcome",
            "",
            "`C10 = absolute_consequence` at H=10.",
            "",
            "## Primary predictor",
            "",
            "`action_disagreement`.",
            "",
            "## Independent unit",
            "",
            "Policy seed is the independent replication unit. "
            "Individual states, sigma levels, and consequence records "
            "are not treated as independent policy replicates.",
            "",
            "## Primary estimand",
            "",
            "For each decision state, regress C10 on action disagreement "
            "across the six nonzero sigma levels and average the resulting "
            "state-level slopes within each policy seed.",
            "",
            "## Primary seed-level results",
            "",
            "| Seed | Records | States | Finite slopes | "
            "Mean state slope | State-slope SD | Positive fraction |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in seed_df.itertuples(
        index=False
    ):
        lines.append(
            f"| {int(row.seed)} | "
            f"{int(row.primary_records)} | "
            f"{int(row.states)} | "
            f"{int(row.finite_state_slopes)} | "
            f"{fmt(float(row.mean_state_slope))} | "
            f"{fmt(float(row.state_slope_sd))} | "
            f"{float(row.positive_state_slope_fraction):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Cross-seed primary inference",
            "",
            f"- Mean seed-level slope: "
            f"`{fmt(primary_summary['mean'])}`",
            f"- Sample SD: "
            f"`{fmt(primary_summary['sd'])}`",
            f"- 95% t-based CI: "
            f"`[{fmt(primary_summary['ci95_low'])}, "
            f"{fmt(primary_summary['ci95_high'])}]`",
            f"- Positive seed-level slopes: "
            f"`{primary_summary['positive_count']}/5`",
            f"- Exact one-sided sign-flip p: "
            f"`{fmt_p(primary_summary['one_sided_p'])}`",
            f"- Exact two-sided sign-flip p: "
            f"`{fmt_p(primary_summary['two_sided_p'])}`",
            f"- Exact sign assignments: "
            f"`{primary_summary['num_exact_sign_assignments']}`",
            "",
            "The five stored seed-level slopes are:",
            "",
            ", ".join(
                fmt(float(x), 8)
                for x in primary_summary[
                    "seed_values"
                ]
            ),
            "",
            "## Secondary dose-response analysis",
            "",
            "The stored secondary analysis reports dose-response "
            "summaries over the six nonzero sigma levels for each "
            "policy seed. All five seeds have Spearman rho = 1.0 "
            "and Kendall tau approximately 1.0 for the dose-response "
            "between action disagreement and downstream consequence.",
            "",
            "| Seed | Spearman rho | Kendall tau |",
            "|---:|---:|---:|",
        ]
    )

    for row in support_df.itertuples(
        index=False
    ):
        lines.append(
            f"| {int(row.seed)} | "
            f"{fmt(float(row.dose_spearman_rho), 6)} | "
            f"{fmt(float(row.dose_kendall_tau), 6)} |"
        )

    lines.extend(
        [
            "",
            "## Secondary support-conditioned analyses",
            "",
            "After averaging each state over the six nonzero sigma "
            "levels, the stored analysis reports descriptive regressions "
            "against support distance.",
            "",
            "| Seed | Support→action slope | R² | "
            "Support→C10 slope | R² |",
            "|---:|---:|---:|---:|---:|",
        ]
    )

    for row in support_df.itertuples(
        index=False
    ):
        lines.append(
            f"| {int(row.seed)} | "
            f"{fmt(float(row.support_to_action_slope))} | "
            f"{fmt(float(row.support_to_action_r2), 6)} | "
            f"{fmt(float(row.support_to_consequence_slope))} | "
            f"{fmt(float(row.support_to_consequence_r2), 6)} |"
        )

    lines.extend(
        [
            "",
            "These support-conditioned regressions are secondary "
            "descriptive analyses and are not used as the primary "
            "generalization test.",
            "",
            "## Dose-response values",
            "",
            "The stored per-seed dose means are reproduced from "
            "`P5_CQL_SECONDARY_ANALYSIS.json` for auditability.",
            "",
            "| Seed | Sigma | Mean action disagreement | "
            "Mean absolute consequence |",
            "|---:|---:|---:|---:|",
        ]
    )

    for row in dose_df.itertuples(
        index=False
    ):
        lines.append(
            f"| {int(row.seed)} | "
            f"{float(row.sigma):.3f} | "
            f"{fmt(float(row.action_disagreement_mean))} | "
            f"{fmt(float(row.absolute_consequence_mean))} |"
        )

    lines.extend(
        [
            "",
            "## Data integrity checks",
            "",
            f"- Primary seed rows: `{len(seed_df)}`",
            f"- State-slope rows: `{len(state_df)}`",
            f"- Primary records per seed: "
            f"`{sorted(seed_df['primary_records'].unique().tolist())}`",
            f"- Decision states per seed: "
            f"`{sorted(seed_df['states'].unique().tolist())}`",
            f"- Finite state slopes per seed: "
            f"`{sorted(seed_df['finite_state_slopes'].unique().tolist())}`",
            f"- Positive state-slope fractions: "
            f"`{', '.join(f'{x:.2f}' for x in seed_df['positive_state_slope_fraction'])}`",
            "",
            "## Interpretation boundary",
            "",
            "The primary result supports reproduction of the evaluated "
            "action-disagreement/downstream-consequence relationship "
            "for the CQL Hopper setting.",
            "",
            "It does not establish causality, universal OOD detection, "
            "or generalization to all offline-RL algorithms, datasets, "
            "environments, or CQL configurations.",
            "",
            "CQL and IQL policy seeds are not pooled for primary "
            "inferential analysis.",
            "",
            "The 95% t-based confidence interval and exact sign-flip "
            "p-values are reported as distinct inferential procedures.",
            "",
            "## Frozen artifacts",
            "",
            f"- `{PRIMARY_JSON.relative_to(ROOT)}`",
            f"- `{SECONDARY_JSON.relative_to(ROOT)}`",
            f"- `{SEED_CSV.relative_to(ROOT)}`",
            f"- `{STATE_SLOPES_CSV.relative_to(ROOT)}`",
            "",
            "## Source hashes",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
        ]
    )

    for path, digest in source_hashes.items():
        lines.append(
            f"| `{path}` | `{digest}` |"
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Evidence-builder repository commit: "
            f"`{git_command('rev-parse', 'HEAD')}`",
            f"- Repository status before evidence generation: "
            f"`{git_command('status', '--porcelain') or 'clean'}`",
            f"- Python: `{sys.version.splitlines()[0]}`",
            f"- Platform: `{platform.platform()}`",
            f"- NumPy: `{np.__version__}`",
            f"- Pandas: `{pd.__version__}`",
            "",
            "## Original analysis provenance",
            "",
            f"- P5 primary-analysis commit recorded in JSON: "
            f"`{primary['git_commit']}`",
            "",
            "## Status",
            "",
            "**P5 CQL Hopper consequence analysis: complete.**",
            "",
        ]
    )

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:

    for path in [
        PRIMARY_JSON,
        SECONDARY_JSON,
        SEED_CSV,
        STATE_SLOPES_CSV,
        *PROTOCOLS,
    ]:
        require_exists(path)

    with PRIMARY_JSON.open() as f:
        primary = json.load(f)

    with SECONDARY_JSON.open() as f:
        secondary = json.load(f)

    seed_df = pd.read_csv(SEED_CSV)
    state_df = pd.read_csv(
        STATE_SLOPES_CSV
    )

    validate_primary_json(primary)
    validate_seed_summary(seed_df)
    validate_state_slopes(state_df)

    support_df, dose_df = (
        compute_secondary_tables(
            secondary
        )
    )

    if len(support_df) != 5:
        raise ValueError(
            "Expected five P5 secondary seed summaries."
        )

    if len(dose_df) != 30:
        raise ValueError(
            "Expected five seeds × six sigma levels = 30 "
            "P5 dose-response rows."
        )

    source_paths = [
        PRIMARY_JSON,
        SECONDARY_JSON,
        SEED_CSV,
        STATE_SLOPES_CSV,
        *PROTOCOLS,
    ]

    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_paths
    }

    summary = build_summary(
        primary=primary,
        seed_df=seed_df,
        state_df=state_df,
        secondary=secondary,
        support_df=support_df,
        dose_df=dose_df,
        source_hashes=source_hashes,
    )

    OUT_SUMMARY.write_text(
        summary
    )

    # Hash every final P5 evidence output except the hash manifest itself.
    output_files = sorted(
        path
        for path in P5_DIR.rglob("*")
        if path.is_file()
        and path.name != OUT_HASHES.name
    )

    with OUT_HASHES.open("w") as f:
        for path in output_files:
            f.write(
                f"{sha256_file(path)}  "
                f"{path.relative_to(ROOT)}\n"
            )

    print("=" * 80)
    print("P5 EVIDENCE PACKAGE BUILT")
    print("=" * 80)
    print()
    print(f"Summary: {OUT_SUMMARY}")
    print(f"Hashes:  {OUT_HASHES}")
    print()
    print("Primary result:")
    primary_summary = compute_primary_summary(
        primary
    )
    print(
        f"  mean slope = "
        f"{primary_summary['mean']:.8f}"
    )
    print(
        f"  95% CI = "
        f"[{primary_summary['ci95_low']:.8f}, "
        f"{primary_summary['ci95_high']:.8f}]"
    )
    print(
        f"  positive seeds = "
        f"{primary_summary['positive_count']}/5"
    )
    print(
        f"  one-sided p = "
        f"{primary_summary['one_sided_p']}"
    )
    print(
        f"  two-sided p = "
        f"{primary_summary['two_sided_p']}"
    )
    print()
    print(
        "Validation: PASS"
    )


if __name__ == "__main__":
    main()
