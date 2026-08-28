from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


INPUT = Path(
    "results/diagnostics/reliability/"
    "p1_multidose_predictor_seed0.json"
)

OUTPUT = Path(
    "results/diagnostics/reliability/"
    "p1_clustered_analysis_seed0.json"
)

SEED = 20260830

N_BOOTSTRAP = 5000


def r2_score(
    y: np.ndarray,
    prediction: np.ndarray,
) -> float:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    sse = np.sum(
        (y - prediction) ** 2
    )

    centered = (
        y - np.mean(y)
    )

    sst = np.sum(
        centered ** 2
    )

    if sst <= 0:
        return float("nan")

    return float(
        1.0 - sse / sst
    )


def fit_r2(
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    model = LinearRegression()
    model.fit(X, y)
    return r2_score(
        y,
        model.predict(X),
    )


def fit_coefficients(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    model = LinearRegression()
    model.fit(X, y)

    return np.concatenate(
        [
            np.asarray(
                [model.intercept_],
                dtype=float,
            ),
            np.asarray(
                model.coef_,
                dtype=float,
            ),
        ]
    )


def bootstrap_clusters(
    records: list[dict],
    cluster_key: str,
    model_builder,
    n_bootstrap: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    clusters = np.asarray(
        sorted(
            {
                int(r[cluster_key])
                for r in records
            }
        ),
        dtype=int,
    )

    cluster_to_rows = {
        cluster: [
            r
            for r in records
            if int(r[cluster_key]) == int(cluster)
        ]
        for cluster in clusters
    }

    estimates = []

    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )

        sampled_rows = []

        for cluster in sampled_clusters:
            sampled_rows.extend(
                cluster_to_rows[int(cluster)]
            )

        X, y = model_builder(
            sampled_rows
        )

        estimates.append(
            fit_r2(
                X,
                y,
            )
        )

    return np.asarray(
        estimates,
        dtype=float,
    )


def model_builder(
    names: list[str],
):
    def build(
        records: list[dict],
    ):
        y = np.asarray(
            [
                r["abs_delta_J10"]
                for r in records
            ],
            dtype=float,
        )

        columns = []

        for name in names:
            if name == "action":
                columns.append(
                    np.asarray(
                        [
                            r["action_disagreement"]
                            for r in records
                        ],
                        dtype=float,
                    )
                )

            elif name == "support":
                columns.append(
                    np.asarray(
                        [
                            r["support_distance"]
                            for r in records
                        ],
                        dtype=float,
                    )
                )

            elif name == "critic":
                columns.append(
                    np.asarray(
                        [
                            r[
                                "twin_critic_disagreement"
                            ]
                            for r in records
                        ],
                        dtype=float,
                    )
                )

            elif name == "interaction":
                action = np.asarray(
                    [
                        r["action_disagreement"]
                        for r in records
                    ],
                    dtype=float,
                )

                support = np.asarray(
                    [
                        r["support_distance"]
                        for r in records
                    ],
                    dtype=float,
                )

                # Center the variables before forming the interaction
                # to reduce unnecessary numerical collinearity.
                action_c = (
                    action - np.mean(action)
                )

                support_c = (
                    support - np.mean(support)
                )

                columns.append(
                    action_c * support_c
                )

            else:
                raise ValueError(
                    f"Unknown predictor: {name}"
                )

        X = np.column_stack(
            columns
        )

        return X, y

    return build


def main():
    print("=" * 80)
    print("P1.3 — CLUSTERED SEED-0 ANALYSIS")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    data = json.loads(
        INPUT.read_text()
    )

    # Exclude sigma=0 because it is the identity condition and
    # contains no action-instability variation.
    records = [
        r
        for r in data["records"]
        if float(r["sigma"]) > 0.0
    ]

    print(
        "Records:",
        len(records),
    )

    print(
        "States:",
        len(
            {
                int(r["state_id"])
                for r in records
            }
        ),
    )

    print(
        "Episodes:",
        len(
            {
                int(r["episode_id"])
                for r in records
            }
        ),
    )

    models = {
        "M1_action": [
            "action",
        ],
        "M2_action_support": [
            "action",
            "support",
        ],
        "M3_action_critic": [
            "action",
            "critic",
        ],
        "M4_all": [
            "action",
            "support",
            "critic",
        ],
        "M5_action_support_interaction": [
            "action",
            "support",
            "interaction",
        ],
    }

    results = {}

    for model_name, predictor_names in models.items():
        builder = model_builder(
            predictor_names
        )

        X, y = builder(
            records
        )

        coefficients = fit_coefficients(
            X,
            y,
        )

        observed_r2 = fit_r2(
            X,
            y,
        )

        print(
            f"\n{model_name}"
        )

        print(
            "  predictors:",
            predictor_names,
        )

        print(
            "  in-sample R2:",
            observed_r2,
        )

        print(
            "  coefficients:",
            coefficients.tolist(),
        )

        print(
            "  episode bootstrap..."
        )

        episode_bootstrap = (
            bootstrap_clusters(
                records,
                "episode_id",
                builder,
                N_BOOTSTRAP,
                SEED,
            )
        )

        print(
            "  state bootstrap..."
        )

        state_bootstrap = (
            bootstrap_clusters(
                records,
                "state_id",
                builder,
                N_BOOTSTRAP,
                SEED + 1,
            )
        )

        episode_bootstrap = (
            episode_bootstrap[
                np.isfinite(
                    episode_bootstrap
                )
            ]
        )

        state_bootstrap = (
            state_bootstrap[
                np.isfinite(
                    state_bootstrap
                )
            ]
        )

        results[model_name] = {
            "predictors": predictor_names,
            "in_sample_r2": float(
                observed_r2
            ),
            "coefficients": coefficients.tolist(),
            "episode_bootstrap": {
                "n": int(
                    len(
                        episode_bootstrap
                    )
                ),
                "median_r2": float(
                    np.median(
                        episode_bootstrap
                    )
                ),
                "ci95_low": float(
                    np.percentile(
                        episode_bootstrap,
                        2.5,
                    )
                ),
                "ci95_high": float(
                    np.percentile(
                        episode_bootstrap,
                        97.5,
                    )
                ),
            },
            "state_bootstrap": {
                "n": int(
                    len(
                        state_bootstrap
                    )
                ),
                "median_r2": float(
                    np.median(
                        state_bootstrap
                    )
                ),
                "ci95_low": float(
                    np.percentile(
                        state_bootstrap,
                        2.5,
                    )
                ),
                "ci95_high": float(
                    np.percentile(
                        state_bootstrap,
                        97.5,
                    )
                ),
            },
        }

    # ----------------------------------------------------------
    # Incremental R2 diagnostics.
    # ----------------------------------------------------------

    r2_m1 = results[
        "M1_action"
    ]["in_sample_r2"]

    r2_m2 = results[
        "M2_action_support"
    ]["in_sample_r2"]

    r2_m3 = results[
        "M3_action_critic"
    ]["in_sample_r2"]

    r2_m4 = results[
        "M4_all"
    ]["in_sample_r2"]

    r2_m5 = results[
        "M5_action_support_interaction"
    ]["in_sample_r2"]

    incremental = {
        "support_over_action": float(
            r2_m2 - r2_m1
        ),
        "critic_over_action": float(
            r2_m3 - r2_m1
        ),
        "all_over_action": float(
            r2_m4 - r2_m1
        ),
        "interaction_over_action_support": float(
            r2_m5 - r2_m2
        ),
    }

    print("\n" + "=" * 80)
    print("INCREMENTAL R2")
    print("=" * 80)

    for key, value in incremental.items():
        print(
            f"{key:38s}: {value:+.8f}"
        )

    output = {
        "experiment": (
            "P1.3-clustered-seed0-analysis"
        ),
        "status": "diagnostic_only",
        "endpoint": "abs_delta_J10",
        "records": len(records),
        "states": len(
            {
                int(r["state_id"])
                for r in records
            }
        ),
        "episodes": len(
            {
                int(r["episode_id"])
                for r in records
            }
        ),
        "bootstrap_replicates": N_BOOTSTRAP,
        "seed": SEED,
        "analysis_protocol": (
            "experiments/reliability/"
            "P1_PROTOCOL_AMENDMENT_004.md"
        ),
        "models": results,
        "incremental_r2": incremental,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("\nSaved:", OUTPUT)

    print(
        "\n⚠️ DIAGNOSTIC ONLY — NOT PAPER EVIDENCE"
    )


if __name__ == "__main__":
    main()
