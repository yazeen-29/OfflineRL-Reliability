from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import d3rlpy
import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_dataset(task: str):
    dataset, env = d3rlpy.datasets.get_minari(task)
    return dataset, env


def split_episodes(
    dataset,
    seed: int,
    reference_fraction: float = 0.90,
):
    """
    Split at the episode level so query states cannot leak into
    the explanation reference database.
    """
    episodes = list(dataset.episodes)

    if len(episodes) < 20:
        raise RuntimeError(
            "Dataset is too small for a reliable episode-level split."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    split = int(len(indices) * reference_fraction)

    reference_indices = indices[:split]
    query_indices = indices[split:]

    reference_episodes = [
        episodes[int(i)] for i in reference_indices
    ]

    query_episodes = [
        episodes[int(i)] for i in query_indices
    ]

    return reference_episodes, query_episodes


def episodes_to_observations(episodes) -> np.ndarray:
    observations = []

    for episode in episodes:
        obs = np.asarray(
            episode.observations,
            dtype=np.float64,
        )

        if obs.ndim != 2:
            raise ValueError(
                f"Expected 2D observations, got {obs.shape}"
            )

        observations.append(obs)

    if not observations:
        raise RuntimeError(
            "No observations found."
        )

    return np.concatenate(
        observations,
        axis=0,
    )


def sample_query_observations(
    episodes,
    num_queries: int,
    seed: int,
) -> np.ndarray:
    """
    Sample uniformly over held-out observations.

    The query observations are kept separate from the reference
    database.
    """
    all_observations = episodes_to_observations(
        episodes
    )

    if num_queries > len(all_observations):
        num_queries = len(all_observations)

    rng = np.random.default_rng(seed)

    query_indices = rng.choice(
        len(all_observations),
        size=num_queries,
        replace=False,
    )

    return all_observations[query_indices]


def fit_standardization(
    observations: np.ndarray,
):
    mean = np.mean(
        observations,
        axis=0,
    )

    std = np.std(
        observations,
        axis=0,
    )

    # Prevent division by zero.
    std = np.maximum(
        std,
        1e-8,
    )

    return mean, std


def standardize(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
):
    return (
        observations - mean
    ) / std


def build_neighbor_index(
    standardized_reference: np.ndarray,
):
    """
    Euclidean nearest-neighbor search in standardized feature space.
    """
    index = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    index.fit(
        standardized_reference
    )

    return index


def nearest_neighbor(
    standardized_query: np.ndarray,
    index: NearestNeighbors,
):
    distances, indices = index.kneighbors(
        standardized_query.reshape(1, -1),
        n_neighbors=1,
    )

    return (
        float(distances[0, 0]),
        int(indices[0, 0]),
    )


def policy_action(
    policy,
    observation: np.ndarray,
):
    action = policy.predict(
        np.asarray(
            [observation],
            dtype=np.float64,
        )
    )[0]

    return np.asarray(
        action,
        dtype=np.float64,
    )


def action_disagreement(
    action_a: np.ndarray,
    action_b: np.ndarray,
):
    """
    RMS-style action disagreement.

    Hopper has 3 continuous actions, so this is normalized by
    sqrt(action dimension).
    """
    return float(
        np.linalg.norm(
            action_a - action_b
        )
        / np.sqrt(len(action_a))
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    num_bootstrap: int = 2000,
):
    """
    Bootstrap 95% confidence interval for the mean.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        num_bootstrap,
        dtype=np.float64,
    )

    n = len(values)

    for i in range(num_bootstrap):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    return {
        "mean": float(
            np.mean(values)
        ),
        "ci95_low": float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        "ci95_high": float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Research-grade Gaussian observation-shift "
            "experiment for explanation reliability."
        )
    )

    parser.add_argument(
        "--task",
        default="mujoco/hopper/medium-v0",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/iql_100k/"
            "iql_mujoco_hopper_medium-v0_seed0.d3"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_queries",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--noise_levels",
        nargs="+",
        type=float,
        default=[
            0.00,
            0.01,
            0.025,
            0.05,
            0.10,
            0.20,
            0.30,
        ],
    )

    parser.add_argument(
        "--noise_repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--reference_fraction",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/shifts/"
            "iql_100k_gaussian_observation_noise.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 76)
    print(
        "RESEARCH-GRADE GAUSSIAN OBSERVATION-SHIFT "
        "EXPERIMENT"
    )
    print("=" * 76)
    print(f"Task              : {args.task}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Seed              : {args.seed}")
    print(f"Queries            : {args.num_queries}")
    print(f"Noise repeats      : {args.noise_repeats}")
    print(f"Noise levels       : {args.noise_levels}")
    print(
        f"Reference fraction : "
        f"{args.reference_fraction}"
    )
    print("=" * 76)

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    # ------------------------------------------------------------
    # 1. Frozen policy.
    # ------------------------------------------------------------
    print(
        "\n[1/7] Loading frozen IQL policy..."
    )

    policy = d3rlpy.load_learnable(
        args.checkpoint,
        device="cpu",
    )

    print(
        f"[OK] Policy: "
        f"{type(policy).__name__}"
    )

    # ------------------------------------------------------------
    # 2. Dataset.
    # ------------------------------------------------------------
    print(
        "\n[2/7] Loading offline dataset..."
    )

    dataset, _ = load_dataset(
        args.task
    )

    print(
        f"[OK] Episodes: "
        f"{dataset.size()}"
    )

    # ------------------------------------------------------------
    # 3. Episode split.
    # ------------------------------------------------------------
    print(
        "\n[3/7] Creating episode-level split..."
    )

    reference_episodes, query_episodes = (
        split_episodes(
            dataset,
            seed=args.seed,
            reference_fraction=(
                args.reference_fraction
            ),
        )
    )

    reference_observations = (
        episodes_to_observations(
            reference_episodes
        )
    )

    query_observations = (
        sample_query_observations(
            query_episodes,
            num_queries=args.num_queries,
            seed=args.seed + 12345,
        )
    )

    print(
        f"[OK] Reference episodes: "
        f"{len(reference_episodes)}"
    )

    print(
        f"[OK] Query episodes: "
        f"{len(query_episodes)}"
    )

    print(
        f"[OK] Reference observations: "
        f"{len(reference_observations)}"
    )

    print(
        f"[OK] Query observations: "
        f"{len(query_observations)}"
    )

    # ------------------------------------------------------------
    # 4. Standardization and nearest-neighbor index.
    # ------------------------------------------------------------
    print(
        "\n[4/7] Standardizing observations..."
    )

    mean, std = fit_standardization(
        reference_observations
    )

    standardized_reference = standardize(
        reference_observations,
        mean,
        std,
    )

    print(
        "[OK] Feature standard deviations:"
    )

    print(std)

    print(
        "\nBuilding nearest-neighbor index..."
    )

    neighbor_index = build_neighbor_index(
        standardized_reference
    )

    print(
        "[OK] Nearest-neighbor index built"
    )

    # ------------------------------------------------------------
    # 5. Clean baseline.
    # ------------------------------------------------------------
    print(
        "\n[5/7] Computing clean baseline..."
    )

    clean_results = []

    for query_id, clean_obs in enumerate(
        query_observations
    ):
        clean_action = policy_action(
            policy,
            clean_obs,
        )

        standardized_clean = standardize(
            clean_obs,
            mean,
            std,
        )

        nn_distance, nn_index = nearest_neighbor(
            standardized_clean,
            neighbor_index,
        )

        nn_obs = reference_observations[
            nn_index
        ]

        nn_action = policy_action(
            policy,
            nn_obs,
        )

        clean_fidelity = action_disagreement(
            clean_action,
            nn_action,
        )

        clean_results.append(
            {
                "query_id": int(query_id),
                "nearest_neighbor_distance": (
                    nn_distance
                ),
                "explanation_action_disagreement": (
                    clean_fidelity
                ),
            }
        )

    clean_distance = np.asarray(
        [
            x["nearest_neighbor_distance"]
            for x in clean_results
        ]
    )

    clean_fidelity = np.asarray(
        [
            x[
                "explanation_action_disagreement"
            ]
            for x in clean_results
        ]
    )

    print(
        "[OK] Clean mean NN distance:",
        np.mean(clean_distance),
    )

    print(
        "[OK] Clean mean action disagreement:",
        np.mean(clean_fidelity),
    )

    # ------------------------------------------------------------
    # 6. Controlled shifts.
    # ------------------------------------------------------------
    print(
        "\n[6/7] Running controlled Gaussian shifts..."
    )

    level_summaries = []
    all_records = []

    for level_id, noise_level in enumerate(
        args.noise_levels
    ):
        print(
            f"\nNoise level = "
            f"{noise_level:.4f}"
        )

        nn_distances = []
        action_changes = []
        explanation_disagreements = []
        noise_norms = []

        for query_id, clean_obs in enumerate(
            query_observations
        ):
            clean_action = policy_action(
                policy,
                clean_obs,
            )

            for repeat_id in range(
                args.noise_repeats
            ):
                noise_seed = (
                    args.seed
                    * 1_000_000
                    + level_id
                    * 100_000
                    + query_id
                    * 10
                    + repeat_id
                )

                rng = np.random.default_rng(
                    noise_seed
                )

                standardized_noise = rng.normal(
                    loc=0.0,
                    scale=noise_level,
                    size=len(clean_obs),
                )

                shifted_obs = (
                    clean_obs
                    + standardized_noise * std
                )

                shifted_action = policy_action(
                    policy,
                    shifted_obs,
                )

                action_change = (
                    action_disagreement(
                        shifted_action,
                        clean_action,
                    )
                )

                standardized_shifted = (
                    standardize(
                        shifted_obs,
                        mean,
                        std,
                    )
                )

                nn_distance, nn_index = (
                    nearest_neighbor(
                        standardized_shifted,
                        neighbor_index,
                    )
                )

                nn_obs = reference_observations[
                    nn_index
                ]

                nn_action = policy_action(
                    policy,
                    nn_obs,
                )

                explanation_fidelity = (
                    action_disagreement(
                        shifted_action,
                        nn_action,
                    )
                )

                noise_norm = float(
                    np.linalg.norm(
                        standardized_noise
                    )
                    / np.sqrt(len(clean_obs))
                )

                nn_distances.append(
                    nn_distance
                )

                action_changes.append(
                    action_change
                )

                explanation_disagreements.append(
                    explanation_fidelity
                )

                noise_norms.append(
                    noise_norm
                )

                all_records.append(
                    {
                        "query_id": int(
                            query_id
                        ),
                        "repeat_id": int(
                            repeat_id
                        ),
                        "noise_level": float(
                            noise_level
                        ),
                        "noise_norm_standardized": (
                            noise_norm
                        ),
                        "nearest_neighbor_distance": (
                            nn_distance
                        ),
                        "policy_action_change": (
                            action_change
                        ),
                        "explanation_action_disagreement": (
                            explanation_fidelity
                        ),
                    }
                )

        nn_distances = np.asarray(
            nn_distances
        )

        action_changes = np.asarray(
            action_changes
        )

        explanation_disagreements = (
            np.asarray(
                explanation_disagreements
            )
        )

        noise_norms = np.asarray(
            noise_norms
        )

        summary = {
            "noise_level": float(
                noise_level
            ),
            "num_samples": int(
                len(nn_distances)
            ),
            "mean_standardized_noise_norm": (
                bootstrap_mean_ci(
                    noise_norms,
                    seed=args.seed + level_id,
                )
            ),
            "nearest_neighbor_distance": (
                bootstrap_mean_ci(
                    nn_distances,
                    seed=args.seed
                    + 100
                    + level_id,
                )
            ),
            "policy_action_change": (
                bootstrap_mean_ci(
                    action_changes,
                    seed=args.seed
                    + 200
                    + level_id,
                )
            ),
            "explanation_action_disagreement": (
                bootstrap_mean_ci(
                    explanation_disagreements,
                    seed=args.seed
                    + 300
                    + level_id,
                )
            ),
        }

        level_summaries.append(
            summary
        )

        print(
            "  NN distance:",
            summary[
                "nearest_neighbor_distance"
            ]["mean"],
        )

        print(
            "  Policy action change:",
            summary[
                "policy_action_change"
            ]["mean"],
        )

        print(
            "  Explanation disagreement:",
            summary[
                "explanation_action_disagreement"
            ]["mean"],
        )

    # ------------------------------------------------------------
    # 7. Save complete experiment.
    # ------------------------------------------------------------
    print(
        "\n[7/7] Saving experiment results..."
    )

    output = {
        "experiment": (
            "IQL-100K-research-grade-"
            "gaussian-observation-shift"
        ),
        "task": args.task,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "policy_type": type(policy).__name__,
        "dataset_episodes": int(
            dataset.size()
        ),
        "reference_episodes": int(
            len(reference_episodes)
        ),
        "query_episodes": int(
            len(query_episodes)
        ),
        "reference_observations": int(
            len(reference_observations)
        ),
        "query_observations": int(
            len(query_observations)
        ),
        "observation_dim": int(
            reference_observations.shape[1]
        ),
        "reference_fraction": float(
            args.reference_fraction
        ),
        "num_queries_requested": int(
            args.num_queries
        ),
        "noise_repeats": int(
            args.noise_repeats
        ),
        "noise_levels": [
            float(x)
            for x in args.noise_levels
        ],
        "standardization": {
            "method": "reference-dataset-z-score",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "clean_baseline": {
            "nearest_neighbor_distance": (
                bootstrap_mean_ci(
                    clean_distance,
                    seed=args.seed + 500,
                )
            ),
            "explanation_action_disagreement": (
                bootstrap_mean_ci(
                    clean_fidelity,
                    seed=args.seed + 501,
                )
            ),
        },
        "noise_level_summaries": (
            level_summaries
        ),
        "records": all_records,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print(
        "\n" + "=" * 76
    )
    print(
        "RESEARCH-GRADE SHIFT EXPERIMENT COMPLETE"
    )
    print(
        "=" * 76
    )
    print(
        f"Saved: {output_path}"
    )
    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()