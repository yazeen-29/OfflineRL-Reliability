from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data_frozen" / "P1"
OUT_DIR = ROOT / "results" / "analysis" / "P3"

PAIR_DIR = OUT_DIR / "matched_pairs"
STAT_DIR = OUT_DIR / "statistics"
SENS_DIR = OUT_DIR / "sensitivity"
PROV_DIR = OUT_DIR / "provenance"

for d in [PAIR_DIR, STAT_DIR, SENS_DIR, PROV_DIR]:
    d.mkdir(parents=True, exist_ok=True)


POLICY_SEEDS = [0, 1, 2, 3, 4]
HORIZON = 10
SIGMA_NONZERO = True

PRIMARY_STRATA = 10
SENSITIVITY_STRATA = [5, 20]

MIN_GROUP_SIZE = 2
LOW_HIGH_FRACTION = 0.25


REQUIRED_COLUMNS = {
    "policy_seed",
    "state_id",
    "source_episode_id",
    "source_step",
    "sigma",
    "horizon",
    "support_distance",
    "action_disagreement",
    "absolute_consequence",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def git_command(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def load_data() -> pd.DataFrame:
    rows = []

    for seed in POLICY_SEEDS:
        path = DATA_DIR / f"seed{seed}.json"

        with path.open() as f:
            payload = json.load(f)

        if int(payload["policy_seed"]) != seed:
            raise ValueError(
                f"{path}: expected policy_seed={seed}"
            )

        for record in payload["records"]:
            if int(record["horizon"]) != HORIZON:
                continue

            if float(record["sigma"]) <= 0:
                continue

            row = {
                k: record[k]
                for k in REQUIRED_COLUMNS
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    expected = 3000

    if len(df) != expected:
        raise ValueError(
            f"Expected {expected} H=10 nonzero-shift records, "
            f"got {len(df)}"
        )

    for column in [
        "policy_seed",
        "state_id",
        "source_episode_id",
        "source_step",
        "sigma",
        "horizon",
        "support_distance",
        "action_disagreement",
        "absolute_consequence",
    ]:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        ).to_numpy()

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values in {column}"
            )

    return df.sort_values(
        [
            "policy_seed",
            "sigma",
            "state_id",
        ]
    ).reset_index(drop=True)


def assign_support_strata(
    cell: pd.DataFrame,
    n_strata: int,
) -> pd.DataFrame:
    g = (
        cell
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    n = len(g)

    g["support_stratum"] = (
        np.floor(
            np.arange(n) * n_strata / n
        )
        .astype(int)
        + 1
    )

    return g


def match_stratum(
    stratum: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    n = len(stratum)

    q = int(
        math.floor(
            LOW_HIGH_FRACTION * n
        )
    )

    if q < MIN_GROUP_SIZE:
        return (
            pd.DataFrame(),
            {
                "feasible": False,
                "reason":
                    "insufficient_quartile_group_size",
                "n_total": n,
                "n_low": q,
                "n_high": q,
                "n_pairs": 0,
            },
        )

    ordered = (
        stratum
        .sort_values(
            ["action_disagreement", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    low = (
        ordered
        .iloc[:q]
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    high = (
        ordered
        .iloc[-q:]
        .sort_values(
            ["support_distance", "state_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    n_pairs = min(
        len(low),
        len(high),
    )

    rows = []

    for pair_index in range(n_pairs):
        lo = low.iloc[pair_index]
        hi = high.iloc[pair_index]

        rows.append(
            {
                "low_policy_seed":
                    int(lo["policy_seed"]),
                "high_policy_seed":
                    int(hi["policy_seed"]),
                "low_state_id":
                    int(lo["state_id"]),
                "high_state_id":
                    int(hi["state_id"]),
                "low_source_episode_id":
                    int(lo["source_episode_id"]),
                "high_source_episode_id":
                    int(hi["source_episode_id"]),
                "low_source_step":
                    int(lo["source_step"]),
                "high_source_step":
                    int(hi["source_step"]),
                "sigma":
                    float(lo["sigma"]),
                "support_stratum":
                    int(lo["support_stratum"]),
                "low_support_distance":
                    float(lo["support_distance"]),
                "high_support_distance":
                    float(hi["support_distance"]),
                "support_distance_abs_diff":
                    abs(
                        float(
                            hi["support_distance"]
                        )
                        - float(
                            lo["support_distance"]
                        )
                    ),
                "low_action_disagreement":
                    float(
                        lo["action_disagreement"]
                    ),
                "high_action_disagreement":
                    float(
                        hi["action_disagreement"]
                    ),
                "action_disagreement_diff":
                    (
                        float(
                            hi["action_disagreement"]
                        )
                        - float(
                            lo["action_disagreement"]
                        )
                    ),
                "low_C10":
                    float(
                        lo["absolute_consequence"]
                    ),
                "high_C10":
                    float(
                        hi["absolute_consequence"]
                    ),
                "pair_delta_C10":
                    (
                        float(
                            hi["absolute_consequence"]
                        )
                        - float(
                            lo["absolute_consequence"]
                        )
                    ),
                "pair_index":
                    pair_index,
            }
        )

    pairs = pd.DataFrame(rows)

    return (
        pairs,
        {
            "feasible": True,
            "reason": "",
            "n_total": n,
            "n_low": len(low),
            "n_high": len(high),
            "n_pairs": n_pairs,
        },
    )


def run_matching(
    df: pd.DataFrame,
    n_strata: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    pair_tables = []
    stratum_rows = []

    for (seed, sigma), cell in df.groupby(
        ["policy_seed", "sigma"],
        sort=True,
    ):
        cell = assign_support_strata(
            cell,
            n_strata=n_strata,
        )

        for stratum_id, stratum in cell.groupby(
            "support_stratum",
            sort=True,
        ):
            pairs, audit = match_stratum(
                stratum
            )

            stratum_rows.append(
                {
                    "policy_seed": int(seed),
                    "sigma": float(sigma),
                    "support_stratum":
                        int(stratum_id),
                    **audit,
                }
            )

            if not pairs.empty:
                pairs["policy_seed"] = int(seed)
                pairs["n_strata"] = n_strata
                pair_tables.append(pairs)

    if pair_tables:
        pairs_all = pd.concat(
            pair_tables,
            ignore_index=True,
        )
    else:
        pairs_all = pd.DataFrame()

    strata = pd.DataFrame(
        stratum_rows
    )

    return pairs_all, strata


def exact_sign_flip(
    seed_effects: np.ndarray,
) -> tuple[float, float]:
    observed = float(
        np.mean(seed_effects)
    )

    values = np.asarray(
        seed_effects,
        dtype=float,
    )

    n = len(values)

    signed_means = []

    for mask in range(2 ** n):
        signs = np.array(
            [
                1.0 if (
                    mask & (1 << i)
                ) else -1.0
                for i in range(n)
            ]
        )

        signed_means.append(
            float(
                np.mean(
                    values * signs
                )
            )
        )

    signed_means = np.asarray(
        signed_means
    )

    one_sided = float(
        np.mean(
            signed_means >= observed - 1e-15
        )
    )

    two_sided = float(
        np.mean(
            np.abs(signed_means)
            >= abs(observed) - 1e-15
        )
    )

    return one_sided, two_sided


def summarize_effects(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for seed, group in pairs.groupby(
        "policy_seed",
        sort=True,
    ):
        rows.append(
            {
                "policy_seed": int(seed),
                "n_pairs": len(group),
                "mean_pair_delta_C10":
                    float(
                        group["pair_delta_C10"].mean()
                    ),
                "median_pair_delta_C10":
                    float(
                        group["pair_delta_C10"].median()
                    ),
                "mean_support_abs_diff":
                    float(
                        group[
                            "support_distance_abs_diff"
                        ].mean()
                    ),
                "median_support_abs_diff":
                    float(
                        group[
                            "support_distance_abs_diff"
                        ].median()
                    ),
                "mean_action_disagreement_diff":
                    float(
                        group[
                            "action_disagreement_diff"
                        ].mean()
                    ),
                "median_action_disagreement_diff":
                    float(
                        group[
                            "action_disagreement_diff"
                        ].median()
                    ),
                "min_action_disagreement_diff":
                    float(
                        group[
                            "action_disagreement_diff"
                        ].min()
                    ),
                "max_action_disagreement_diff":
                    float(
                        group[
                            "action_disagreement_diff"
                        ].max()
                    ),
            }
        )

    effects = pd.DataFrame(rows)

    x = effects[
        "mean_pair_delta_C10"
    ].to_numpy(float)

    mean_effect = float(
        x.mean()
    )

    sd_effect = float(
        x.std(ddof=1)
    )

    n = len(x)

    t_critical = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    half_width = (
        t_critical
        * sd_effect
        / math.sqrt(n)
    )

    ci_low = mean_effect - half_width
    ci_high = mean_effect + half_width

    one_sided, two_sided = exact_sign_flip(
        x
    )

    summary = {
        "n_policy_seeds": n,
        "mean_seed_effect": mean_effect,
        "sd_seed_effect": sd_effect,
        "ci95_t_low": ci_low,
        "ci95_t_high": ci_high,
        "exact_sign_flip_one_sided_p":
            one_sided,
        "exact_sign_flip_two_sided_p":
            two_sided,
        "n_positive_seed_effects":
            int((x > 0).sum()),
        "n_nonnegative_seed_effects":
            int((x >= 0).sum()),
        "minimum_seed_effect":
            float(x.min()),
        "maximum_seed_effect":
            float(x.max()),
        "total_pairs":
            int(
                effects["n_pairs"].sum()
            ),
    }

    return effects, summary


def main() -> None:
    df = load_data()

    all_seed_summaries = []
    sensitivity_summaries = {}

    # --------------------------------------------------------
    # Primary analysis
    # --------------------------------------------------------

    pairs, strata = run_matching(
        df,
        n_strata=PRIMARY_STRATA,
    )

    if len(pairs) != 600:
        raise ValueError(
            f"Primary analysis expected 600 pairs, "
            f"got {len(pairs)}"
        )

    if not strata["feasible"].all():
        raise ValueError(
            "Primary protocol has infeasible strata."
        )

    if not (
        pairs["action_disagreement_diff"] > 0
    ).all():
        raise ValueError(
            "Primary matching produced non-positive "
            "action-disagreement differences."
        )

    pairs.to_csv(
        PAIR_DIR / "P3_primary_matched_pairs.csv",
        index=False,
    )

    strata.to_csv(
        STAT_DIR / "P3_primary_stratum_audit.csv",
        index=False,
    )

    effects, summary = summarize_effects(
        pairs
    )

    effects.to_csv(
        STAT_DIR / "P3_primary_seed_effects.csv",
        index=False,
    )

    all_seed_summaries = effects.copy()

    with (
        STAT_DIR
        / "P3_primary_effect_summary.json"
    ).open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Sensitivity: 5 support-distance strata
    # --------------------------------------------------------

    pairs5, strata5 = run_matching(
        df,
        n_strata=5,
    )

    if not strata5["feasible"].all():
        raise ValueError(
            "5-stratum sensitivity unexpectedly "
            "contains infeasible strata."
        )

    pairs5.to_csv(
        SENS_DIR / "P3_5strata_matched_pairs.csv",
        index=False,
    )

    effects5, summary5 = summarize_effects(
        pairs5
    )

    effects5.to_csv(
        SENS_DIR / "P3_5strata_seed_effects.csv",
        index=False,
    )

    with (
        SENS_DIR
        / "P3_5strata_effect_summary.json"
    ).open("w") as f:
        json.dump(
            summary5,
            f,
            indent=2,
        )

    sensitivity_summaries["5_strata"] = summary5

    # --------------------------------------------------------
    # Sensitivity: 20 support-distance strata
    # --------------------------------------------------------

    pairs20, strata20 = run_matching(
        df,
        n_strata=20,
    )

    feasible20 = strata20[
        "feasible"
    ].all()

    strata20.to_csv(
        SENS_DIR
        / "P3_20strata_feasibility.csv",
        index=False,
    )

    if feasible20:
        effects20, summary20 = summarize_effects(
            pairs20
        )

        effects20.to_csv(
            SENS_DIR
            / "P3_20strata_seed_effects.csv",
            index=False,
        )

        with (
            SENS_DIR
            / "P3_20strata_effect_summary.json"
        ).open("w") as f:
            json.dump(
                summary20,
                f,
                indent=2,
            )

        sensitivity_summaries["20_strata"] = summary20
    else:
        sensitivity_summaries[
            "20_strata"
        ] = {
            "feasible": False,
            "reason":
                "minimum group size requirement not met",
            "feasible_strata": int(
                strata20["feasible"].sum()
            ),
            "total_strata": len(strata20),
        }

    # --------------------------------------------------------
    # Seed-level direction
    # --------------------------------------------------------

    positive = int(
        (
            all_seed_summaries[
                "mean_pair_delta_C10"
            ] > 0
        ).sum()
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    source_hashes = {
        str(seed): sha256_file(
            DATA_DIR / f"seed{seed}.json"
        )
        for seed in POLICY_SEEDS
    }

    provenance = {
        "experiment":
            "P3 distance-matched control",
        "repository_commit":
            git_command(
                "rev-parse",
                "HEAD",
            ),
        "repository_status":
            git_command(
                "status",
                "--porcelain",
            ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__(
            "scipy"
        ).__version__,
        "source_data":
            "data_frozen/P1",
        "source_hashes":
            source_hashes,
        "protocol_hash":
            sha256_file(
                ROOT
                / "experiments"
                / "reliability"
                / "P3_DISTANCE_MATCHED_CONTROL_PROTOCOL.md"
            ),
        "policy_seeds":
            POLICY_SEEDS,
        "horizon":
            HORIZON,
        "sigma_condition":
            "sigma > 0",
        "primary_matching": {
            "support_distance_strata":
                PRIMARY_STRATA,
            "low_high_fraction":
                LOW_HIGH_FRACTION,
            "minimum_group_size":
                MIN_GROUP_SIZE,
            "total_pairs":
                int(len(pairs)),
        },
        "sensitivity":
            sensitivity_summaries,
    }

    with (
        PROV_DIR
        / "P3_provenance.json"
    ).open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    mean_effect = summary[
        "mean_seed_effect"
    ]

    lines = [
        "# P3 Distance-Matched Control Evidence Summary",
        "",
        "## Primary design",
        "",
        f"- Horizon: H={HORIZON}",
        "- Population: nonzero observation shifts",
        "- Independent unit: policy seed",
        "- Support-distance strata: 10",
        "- Low/high action-disagreement groups: lowest/highest quartile",
        "- Total matched pairs: 600",
        "- Pairs per policy seed: 120",
        "",
        "## Primary matched effect",
        "",
        f"- Mean seed-level matched ΔC10: {mean_effect:.8f}",
        f"- SD across seeds: {summary['sd_seed_effect']:.8f}",
        f"- 95% t-based CI: "
        f"[{summary['ci95_t_low']:.8f}, "
        f"{summary['ci95_t_high']:.8f}]",
        f"- Exact one-sided sign-flip p: "
        f"{summary['exact_sign_flip_one_sided_p']:.5f}",
        f"- Exact two-sided sign-flip p: "
        f"{summary['exact_sign_flip_two_sided_p']:.5f}",
        f"- Positive seed effects: "
        f"{summary['n_positive_seed_effects']}/"
        f"{summary['n_policy_seeds']}",
        "",
        "## Balance",
        "",
    ]

    for row in effects.to_dict(
        orient="records"
    ):
        lines.extend(
            [
                f"### Policy seed {row['policy_seed']}",
                f"- Matched pairs: {row['n_pairs']}",
                f"- Mean absolute support-distance gap: "
                f"{row['mean_support_abs_diff']:.8f}",
                f"- Median absolute support-distance gap: "
                f"{row['median_support_abs_diff']:.8f}",
                f"- Mean action-disagreement gap: "
                f"{row['mean_action_disagreement_diff']:.8f}",
                f"- Minimum action-disagreement gap: "
                f"{row['min_action_disagreement_diff']:.8f}",
                f"- Mean paired ΔC10: "
                f"{row['mean_pair_delta_C10']:.8f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "A positive matched effect supports an association between "
            "higher action disagreement and larger downstream consequence "
            "after matching on support distance and sigma.",
            "It does not establish causality or complete control of all "
            "possible confounding.",
            "",
            "The five policy seeds are the independent inferential units; "
            "matched pairs are not treated as independent policy replicates.",
            "",
            "The 5-stratum sensitivity analysis is reported separately. "
            "The 20-stratum specification is mechanically infeasible under "
            "the frozen minimum-group-size requirement when each "
            "support-distance cell contains five observations.",
        ]
    )

    (
        OUT_DIR / "P3_EVIDENCE_SUMMARY.md"
    ).write_text(
        "\n".join(lines) + "\n"
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("P3 DISTANCE-MATCHED CONTROL ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(
        "H=10, sigma>0 records:",
        len(df),
    )
    print(
        "Primary matched pairs:",
        len(pairs),
    )
    print()
    print(
        "Seed-level primary effects:"
    )
    print(
        all_seed_summaries[
            [
                "policy_seed",
                "n_pairs",
                "mean_pair_delta_C10",
                "median_pair_delta_C10",
                "mean_support_abs_diff",
                "mean_action_disagreement_diff",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Primary mean seed effect:",
        f"{summary['mean_seed_effect']:.8f}",
    )
    print(
        "95% CI:",
        f"[{summary['ci95_t_low']:.8f}, "
        f"{summary['ci95_t_high']:.8f}]",
    )
    print(
        "Exact one-sided sign-flip p:",
        f"{summary['exact_sign_flip_one_sided_p']:.5f}",
    )
    print(
        "Positive seed effects:",
        f"{summary['n_positive_seed_effects']}/"
        f"{summary['n_policy_seeds']}",
    )
    print()
    print(
        "5-stratum sensitivity mean effect:",
        f"{summary5['mean_seed_effect']:.8f}",
    )
    print(
        "20-stratum feasible:",
        feasible20,
    )
    print()


if __name__ == "__main__":
    main()
