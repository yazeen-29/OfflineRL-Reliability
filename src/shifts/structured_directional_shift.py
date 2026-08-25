from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import d3rlpy
import numpy as np
from sklearn.neighbors import NearestNeighbors


# ============================================================
# LOCKED EXPERIMENT CONFIGURATION
# ============================================================

GAUSSIAN_SIGMAS = np.array(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30],
    dtype=float,
)

DEFAULT_NUM_QUERIES = 100
DEFAULT_NOISE_REPEATS = 2
DEFAULT_REFERENCE_FRACTION = 0.90

# Hopper observation dimension is 11.
#
# The directional vector is:
#
#     d = [1,1,...,1] / sqrt(11)
#
# Therefore ||d||_2 = 1.
#
# Structured displacement magnitude delta is chosen so that
# delta ~= expected standardized L2 magnitude of Gaussian
# observation noise:
#
#     delta = sigma * sqrt(observation_dim)
#
# The actual dimension is checked after loading the dataset.
#
STRUCTURED_DIRECTION_NAME = (
    "all_features_positive_unit_vector"
)


# ============================================================
# DATASET
# ============================================================

def load_dataset(task: str):
    dataset, env = d3rlpy.datasets.get_minari(task)
    return dataset, env


def split_episodes(
    dataset,
    seed: int,
    reference_fraction: float,
):
    """
    Identical episode-level split used by the frozen
    Gaussian observation-shift experiment.
    """
    episodes = list(dataset.episodes)

    if len(episodes) < 20:
        raise RuntimeError(
            "Dataset is too small for reliable "
            "episode-level splitting."
        )

    rng = np.random.default_rng(seed)

    indices = np.arange(
        len(episodes)
    )

    rng.shuffle(indices)

    split = int(
        len(indices)
        * reference_fraction
    )

    reference_indices = indices[
        :split
    ]

    query_indices = indices[
        split:
    ]

    reference_episodes = [
        episodes[int(i)]
        for i in reference_indices
    ]

    query_episodes = [
        episodes[int(i)]
        for i in query_indices
    ]

    return (
        reference_episodes,
        query_episodes,
    )


def episodes_to_observations(
    episodes,
) -> np.ndarray:
    observations = []

    for episode in episodes:
        obs = np.asarray(
            episode.observations,
            dtype=np.float64,
        )

        if obs.ndim != 2:
            raise ValueError(
                f"Expected 2D observations, "
                f"got {obs.shape}"
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
    Identical held-out query sampling used by the
    frozen Gaussian experiment.
    """
    all_observations = (
        episodes_to_observations(
            episodes
        )
    )

    if num_queries > len(
        all_observations
    ):
        num_queries = len(
            all_observations
        )

    rng = np.random.default_rng(
        seed
    )

    query_indices = rng.choice(
        len(all_observations),
        size=num_queries,
        replace=False,
    )

    return all_observations[
        query_indices
    ]


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_standardization(
    observations: np.ndarray,
):
    """
    Reference-dataset-only standardization.
    """
    mean = np.mean(
        observations,
        axis=0,
    )

    std = np.std(
        observations,
        axis=0,
    )

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


def unstandardize(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
):
    return (
        observations * std
    ) + mean


# ============================================================
# NEAREST-NEIGHBOR INDEX
# ============================================================

def build_neighbor_index(
    standardized_reference: np.ndarray,
):
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


def nearest_neighbor_distance(
    standardized_query: np.ndarray,
    index: NearestNeighbors,
):
    distances, indices = (
        index.kneighbors(
            standardized_query.reshape(
                1,
                -1,
            ),
            n_neighbors=1,
        )
    )

    return (
        float(
            distances[0, 0]
        ),
        int(
            indices[0, 0]
        ),
    )


# ============================================================
# POLICY / METRICS
# ============================================================

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
    Same RMS-style action disagreement used by the
    frozen Gaussian study.
    """
    return float(
        np.linalg.norm(
            action_a - action_b
        )
        / np.sqrt(
            len(action_a)
        )
    )


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    num_bootstrap: int = 2000,
):
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

    n = len(values)

    bootstrap_means = np.empty(
        num_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        num_bootstrap
    ):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = (
            np.mean(sample)
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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Controlled deterministic directional "
            "observation-shift experiment for IQL."
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
        default=DEFAULT_NUM_QUERIES,
    )

    parser.add_argument(
        "--noise_repeats",
        type=int,
        default=DEFAULT_NOISE_REPEATS,
    )

    parser.add_argument(
        "--reference_fraction",
        type=float,
        default=DEFAULT_REFERENCE_FRACTION,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/shifts/structured/"
            "iql_seed0_structured_directional_smoke.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "STRUCTURED DIRECTIONAL OBSERVATION-SHIFT EXPERIMENT"
    )
    print("=" * 80)

    print(
        f"Task              : {args.task}"
    )

    print(
        f"Checkpoint        : {args.checkpoint}"
    )

    print(
        f"Seed              : {args.seed}"
    )

    print(
        f"Queries            : {args.num_queries}"
    )

    print(
        f"Repeats            : {args.noise_repeats}"
    )

    print(
        f"Reference fraction : {args.reference_fraction}"
    )

    print(
        f"Direction          : "
        f"{STRUCTURED_DIRECTION_NAME}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 1. Frozen policy
    # --------------------------------------------------------

    print(
        "\n[1/8] Loading frozen IQL policy..."
    )

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    policy = d3rlpy.load_learnable(
        args.checkpoint,
        device="cpu",
    )

    print(
        f"[OK] Policy: "
        f"{type(policy).__name__}"
    )

    # --------------------------------------------------------
    # 2. Dataset
    # --------------------------------------------------------

    print(
        "\n[2/8] Loading offline dataset..."
    )

    dataset, _ = load_dataset(
        args.task
    )

    print(
        f"[OK] Episodes: "
        f"{dataset.size()}"
    )

    # --------------------------------------------------------
    # 3. Exact 90/10 episode split + held-out queries
    # --------------------------------------------------------

    print(
        "\n[3/8] Creating episode-level split..."
    )

    (
        reference_episodes,
        query_episodes,
    ) = split_episodes(
        dataset,
        seed=args.seed,
        reference_fraction=(
            args.reference_fraction
        ),
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

    # --------------------------------------------------------
    # 4. Exact reference-only standardization
    # --------------------------------------------------------

    print(
        "\n[4/8] Standardizing observations..."
    )

    mean, std = fit_standardization(
        reference_observations
    )

    standardized_reference = (
        standardize(
            reference_observations,
            mean,
            std,
        )
    )

    observation_dim = (
        standardized_reference.shape[1]
    )

    print(
        "[OK] Observation dimension:",
        observation_dim,
    )

    # --------------------------------------------------------
    # Locked direction:
    #
    # d = [1,...,1] / sqrt(D)
    #
    # so ||d||_2 = 1
    # --------------------------------------------------------

    direction = np.ones(
        observation_dim,
        dtype=np.float64,
    )

    direction /= np.linalg.norm(
        direction
    )

    print(
        "[OK] Direction norm:",
        np.linalg.norm(
            direction
        ),
    )

    # Equivalent standardized displacement for the
    # seven Gaussian sigma levels.
    structured_magnitudes = (
        GAUSSIAN_SIGMAS
        * np.sqrt(
            observation_dim
        )
    )

    print(
        "[OK] Gaussian-equivalent levels:"
    )

    for sigma, delta in zip(
        GAUSSIAN_SIGMAS,
        structured_magnitudes,
    ):
        print(
            f"    sigma={sigma:.3f} "
            f"-> delta={delta:.6f}"
        )

    # --------------------------------------------------------
    # 5. Nearest-neighbor index
    # --------------------------------------------------------

    print(
        "\n[5/8] Building nearest-neighbor index..."
    )

    neighbor_index = (
        build_neighbor_index(
            standardized_reference
        )
    )

    print(
        "[OK] Index built"
    )

    # --------------------------------------------------------
    # 6. Clean baseline
    # --------------------------------------------------------

    print(
        "\n[6/8] Computing clean baseline..."
    )

    clean_distances = []
    clean_disagreements = []

    for query_id, clean_obs in enumerate(
        query_observations
    ):

        clean_action = policy_action(
            policy,
            clean_obs,
        )

        standardized_clean = (
            standardize(
                clean_obs,
                mean,
                std,
            )
        )

        distance, neighbor_id = (
            nearest_neighbor_distance(
                standardized_clean,
                neighbor_index,
            )
        )

        reference_action = policy_action(
            policy,
            reference_observations[
                neighbor_id
            ],
        )

        disagreement = (
            action_disagreement(
                clean_action,
                reference_action,
            )
        )

        clean_distances.append(
            distance
        )

        clean_disagreements.append(
            disagreement
        )

    print(
        "[OK] Clean mean NN distance:",
        np.mean(
            clean_distances
        ),
    )

    print(
        "[OK] Clean mean explanation disagreement:",
        np.mean(
            clean_disagreements
        ),
    )

    # --------------------------------------------------------
    # 7. Structured directional shifts
    # --------------------------------------------------------

    print(
        "\n[7/8] Running structured directional shifts..."
    )

    level_summaries = []
    all_records = []

    for level_id, (
        gaussian_sigma,
        delta,
    ) in enumerate(
        zip(
            GAUSSIAN_SIGMAS,
            structured_magnitudes,
        )
    ):

        print(
            f"\nGaussian-equivalent sigma = "
            f"{gaussian_sigma:.4f}"
        )

        print(
            f"Structured magnitude delta = "
            f"{delta:.6f}"
        )

        distances = []
        action_changes = []
        explanation_disagreements = []
        displacement_norms = []

        for query_id, clean_obs in enumerate(
            query_observations
        ):

            clean_action = policy_action(
                policy,
                clean_obs,
            )

            standardized_clean = (
                standardize(
                    clean_obs,
                    mean,
                    std,
                )
            )

            # ------------------------------------------------
            # Deterministic structured shift.
            #
            # Every query at a given level receives exactly
            # the same standardized displacement vector.
            # ------------------------------------------------

            shifted_standardized = (
                standardized_clean
                + delta * direction
            )

            shifted_obs = (
                unstandardize(
                    shifted_standardized,
                    mean,
                    std,
                )
            )

            # Numerical safety check.
            actual_displacement = np.linalg.norm(
                shifted_standardized
                - standardized_clean
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

            nn_distance, neighbor_id = (
                nearest_neighbor_distance(
                    shifted_standardized,
                    neighbor_index,
                )
            )

            explanation_reference_action = (
                policy_action(
                    policy,
                    reference_observations[
                        neighbor_id
                    ],
                )
            )

            explanation_disagreement = (
                action_disagreement(
                    shifted_action,
                    explanation_reference_action,
                )
            )

            distances.append(
                nn_distance
            )

            action_changes.append(
                action_change
            )

            explanation_disagreements.append(
                explanation_disagreement
            )

            displacement_norms.append(
                actual_displacement
            )

            all_records.append(
                {
                    "query_id": int(
                        query_id
                    ),
                    "repeat_id": 0,
                    "level_id": int(
                        level_id
                    ),
                    "gaussian_equivalent_sigma": float(
                        gaussian_sigma
                    ),
                    "structured_shift_magnitude": float(
                        delta
                    ),
                    "actual_standardized_displacement": float(
                        actual_displacement
                    ),
                    "nearest_neighbor_distance": float(
                        nn_distance
                    ),
                    "policy_action_change": float(
                        action_change
                    ),
                    "explanation_action_disagreement": float(
                        explanation_disagreement
                    ),
                    "nearest_neighbor_index": int(
                        neighbor_id
                    ),
                }
            )

        distances = np.asarray(
            distances,
            dtype=np.float64,
        )

        action_changes = np.asarray(
            action_changes,
            dtype=np.float64,
        )

        explanation_disagreements = (
            np.asarray(
                explanation_disagreements,
                dtype=np.float64,
            )
        )

        displacement_norms = np.asarray(
            displacement_norms,
            dtype=np.float64,
        )

        summary = {
            "level_id": int(
                level_id
            ),
            "gaussian_equivalent_sigma": float(
                gaussian_sigma
            ),
            "structured_shift_magnitude": float(
                delta
            ),
            "num_queries": int(
                len(distances)
            ),
            "standardized_displacement": (
                bootstrap_mean_ci(
                    displacement_norms,
                    seed=(
                        args.seed
                        + level_id
                    ),
                )
            ),
            "nearest_neighbor_distance": (
                bootstrap_mean_ci(
                    distances,
                    seed=(
                        args.seed
                        + 100
                        + level_id
                    ),
                )
            ),
            "policy_action_change": (
                bootstrap_mean_ci(
                    action_changes,
                    seed=(
                        args.seed
                        + 200
                        + level_id
                    ),
                )
            ),
            "explanation_action_disagreement": (
                bootstrap_mean_ci(
                    explanation_disagreements,
                    seed=(
                        args.seed
                        + 300
                        + level_id
                    ),
                )
            ),
        }

        level_summaries.append(
            summary
        )

        print(
            "  Standardized displacement:",
            summary[
                "standardized_displacement"
            ]["mean"],
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

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    print(
        "\n[8/8] Saving experiment results..."
    )

    output = {
        "experiment": (
            "IQL-100K-structured-directional-"
            "observation-shift"
        ),
        "task": args.task,
        "seed": int(
            args.seed
        ),
        "checkpoint": args.checkpoint,
        "policy_type": type(
            policy
        ).__name__,
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
            observation_dim
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
        "direction": {
            "name": (
                STRUCTURED_DIRECTION_NAME
            ),
            "components": (
                direction.tolist()
            ),
            "norm_l2": float(
                np.linalg.norm(
                    direction
                )
            ),
        },
        "standardization": {
            "method": (
                "reference-dataset-z-score"
            ),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "level_mapping": {
            "gaussian_sigma_values": [
                float(x)
                for x in GAUSSIAN_SIGMAS
            ],
            "structured_delta_values": [
                float(x)
                for x in structured_magnitudes
            ],
            "mapping_definition": (
                "delta = sigma * sqrt(observation_dim), "
                "so the structured displacement norm equals "
                "the expected standardized L2 magnitude of "
                "an isotropic Gaussian perturbation."
            ),
        },
        "clean_baseline": {
            "nearest_neighbor_distance": (
                bootstrap_mean_ci(
                    np.asarray(
                        clean_distances,
                        dtype=np.float64,
                    ),
                    seed=args.seed + 900,
                )
            ),
            "explanation_action_disagreement": (
                bootstrap_mean_ci(
                    np.asarray(
                        clean_disagreements,
                        dtype=np.float64,
                    ),
                    seed=args.seed + 901,
                )
            ),
        },
        "level_summaries": (
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

    print()
    print("=" * 80)
    print(
        "STRUCTURED DIRECTIONAL SHIFT COMPLETE"
    )
    print("=" * 80)
    print(
        f"Saved: {output_path}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()