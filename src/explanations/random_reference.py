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

DEFAULT_NOISE_LEVELS = [
    0.00,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
]

DEFAULT_NUM_QUERIES = 100
DEFAULT_NOISE_REPEATS = 2
DEFAULT_REFERENCE_FRACTION = 0.90


# ============================================================
# DATASET
# ============================================================

def load_dataset(task: str):
    dataset, env = d3rlpy.datasets.get_minari(task)
    return dataset, env


def split_episodes(
    dataset,
    seed: int,
    reference_fraction: float = DEFAULT_REFERENCE_FRACTION,
):
    """
    Identical episode-level split protocol to the frozen
    Gaussian nearest-neighbor experiment.

    Query episodes are completely disjoint from the
    reference episodes.
    """
    episodes = list(dataset.episodes)

    if len(episodes) < 20:
        raise RuntimeError(
            "Dataset is too small for a reliable "
            "episode-level split."
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

        observations.append(
            obs
        )

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
    Identical query sampling protocol to the
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


# ============================================================
# REFERENCE-DISTANCE UTILITY
# ============================================================

def build_neighbor_index(
    standardized_reference: np.ndarray,
):
    """
    Used only to measure the distance from a shifted query
    to its nearest reference state.

    The actual explanatory reference in this control is RANDOM.
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


def nearest_reference_distance(
    standardized_query: np.ndarray,
    index: NearestNeighbors,
):
    distances, indices = (
        index.kneighbors(
            standardized_query.reshape(
                1, -1
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
    Exactly the same RMS-style action disagreement
    used by the frozen Gaussian experiment.
    """
    return float(
        np.linalg.norm(
            action_a - action_b
        )
        / np.sqrt(
            len(action_a)
        )
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    num_bootstrap: int = 2000,
):
    """
    Bootstrap 95% CI for a mean.
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
            "Random-reference control for the IQL "
            "Gaussian observation-shift explanation study."
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
        "--noise_levels",
        nargs="+",
        type=float,
        default=DEFAULT_NOISE_LEVELS,
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
            "results/shifts/random_reference/"
            "iql_seed0_random_reference_smoke.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "RANDOM-REFERENCE CONTROL EXPERIMENT"
    )
    print("=" * 78)
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
        f"Noise repeats      : {args.noise_repeats}"
    )
    print(
        f"Noise levels       : {args.noise_levels}"
    )
    print(
        f"Reference fraction : "
        f"{args.reference_fraction}"
    )
    print("=" * 78)

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    # --------------------------------------------------------
    # 1. Frozen policy
    # --------------------------------------------------------

    print(
        "\n[1/8] Loading frozen IQL policy..."
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
    # 3. EXACT SAME episode split
    # --------------------------------------------------------

    print(
        "\n[3/8] Creating episode-level split..."
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

    # --------------------------------------------------------
    # 4. EXACT SAME standardization
    # --------------------------------------------------------

    print(
        "\n[4/8] Standardizing observations..."
    )

    mean, std = fit_standardization(
        reference_observations
    )

    standardized_reference = standardize(
        reference_observations,
        mean,
        std,
    )

    # Build nearest-neighbor index ONLY to obtain
    # the query-to-dataset distance metric.
    neighbor_index = build_neighbor_index(
        standardized_reference
    )

    print(
        "[OK] Reference standardization fitted"
    )

    # --------------------------------------------------------
    # 5. Random-reference assignment
    # --------------------------------------------------------

    print(
        "\n[5/8] Creating deterministic random-reference assignments..."
    )

    # One random reference is selected for each
    # query/repeat pair and held fixed across noise levels.
    #
    # This is deliberate:
    # the reference-selection rule itself is random,
    # but the reference does not change simply because
    # the observation is perturbed.
    #
    # This prevents the control from introducing an
    # unnecessary extra source of randomness across noise levels.

    random_reference_indices = np.zeros(
        (
            len(query_observations),
            args.noise_repeats,
        ),
        dtype=np.int64,
    )

    for query_id in range(
        len(query_observations)
    ):
        for repeat_id in range(
            args.noise_repeats
        ):

            reference_seed = (
                args.seed * 1_000_000
                + 900_000
                + query_id * 10
                + repeat_id
            )

            rng = np.random.default_rng(
                reference_seed
            )

            random_reference_indices[
                query_id,
                repeat_id,
            ] = rng.integers(
                low=0,
                high=len(
                    reference_observations
                ),
            )

    print(
        "[OK] Random references assigned"
    )

    # --------------------------------------------------------
    # 6. Clean baseline
    # --------------------------------------------------------

    print(
        "\n[6/8] Computing clean random-reference baseline..."
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

        standardized_clean = standardize(
            clean_obs,
            mean,
            std,
        )

        nearest_distance, _ = (
            nearest_reference_distance(
                standardized_clean,
                neighbor_index,
            )
        )

        clean_distances.append(
            nearest_distance
        )

        for repeat_id in range(
            args.noise_repeats
        ):

            random_ref_index = int(
                random_reference_indices[
                    query_id,
                    repeat_id,
                ]
            )

            random_reference_obs = (
                reference_observations[
                    random_ref_index
                ]
            )

            random_reference_action = (
                policy_action(
                    policy,
                    random_reference_obs,
                )
            )

            disagreement = (
                action_disagreement(
                    clean_action,
                    random_reference_action,
                )
            )

            clean_disagreements.append(
                disagreement
            )

    print(
        "[OK] Mean nearest-neighbor distance:",
        np.mean(
            clean_distances
        ),
    )

    print(
        "[OK] Mean random-reference disagreement:",
        np.mean(
            clean_disagreements
        ),
    )

    # --------------------------------------------------------
    # 7. Controlled Gaussian shifts
    # --------------------------------------------------------

    print(
        "\n[7/8] Running controlled Gaussian shifts..."
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

        distances = []
        action_changes = []
        random_disagreements = []
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

                # EXACT SAME Gaussian perturbation
                # seed formula as the frozen experiment.
                noise_seed = (
                    args.seed
                    * 1_000_000
                    + level_id
                    * 100_000
                    + query_id * 10
                    + repeat_id
                )

                rng = np.random.default_rng(
                    noise_seed
                )

                standardized_noise = (
                    rng.normal(
                        loc=0.0,
                        scale=noise_level,
                        size=len(clean_obs),
                    )
                )

                shifted_obs = (
                    clean_obs
                    + standardized_noise
                    * std
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

                # Distributional distance remains the
                # SAME metric as in the primary experiment.
                nn_distance, _ = (
                    nearest_reference_distance(
                        standardized_shifted,
                        neighbor_index,
                    )
                )

                # RANDOM reference is selected independently
                # of the observation perturbation.
                random_ref_index = int(
                    random_reference_indices[
                        query_id,
                        repeat_id,
                    ]
                )

                random_reference_obs = (
                    reference_observations[
                        random_ref_index
                    ]
                )

                random_reference_action = (
                    policy_action(
                        policy,
                        random_reference_obs,
                    )
                )

                random_disagreement = (
                    action_disagreement(
                        shifted_action,
                        random_reference_action,
                    )
                )

                noise_norm = float(
                    np.linalg.norm(
                        standardized_noise
                    )
                    / np.sqrt(
                        len(clean_obs)
                    )
                )

                distances.append(
                    nn_distance
                )

                action_changes.append(
                    action_change
                )

                random_disagreements.append(
                    random_disagreement
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
                        "random_reference_index": int(
                            random_ref_index
                        ),
                        "random_reference_action_disagreement": (
                            random_disagreement
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

        random_disagreements = np.asarray(
            random_disagreements,
            dtype=np.float64,
        )

        noise_norms = np.asarray(
            noise_norms,
            dtype=np.float64,
        )

        summary = {
            "noise_level": float(
                noise_level
            ),
            "num_samples": int(
                len(distances)
            ),
            "mean_standardized_noise_norm": (
                bootstrap_mean_ci(
                    noise_norms,
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
            "random_reference_action_disagreement": (
                bootstrap_mean_ci(
                    random_disagreements,
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
            "  Random-reference disagreement:",
            summary[
                "random_reference_action_disagreement"
            ]["mean"],
        )

    # --------------------------------------------------------
    # 8. Save complete control experiment
    # --------------------------------------------------------

    print(
        "\n[8/8] Saving random-reference control..."
    )

    output = {
        "experiment": (
            "IQL-100K-random-reference-"
            "gaussian-observation-shift-control"
        ),
        "control_definition": (
            "Reference observation is sampled uniformly "
            "from the held-out reference observation pool "
            "using a deterministic seed. The selected "
            "reference is fixed for each query/repeat pair "
            "across all noise levels."
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
            "method": (
                "reference-dataset-z-score"
            ),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "random_reference_sampling": {
            "method": "uniform-over-reference-observations",
            "replacement": True,
            "fixed_per_query_repeat_across_noise": True,
            "seed_offset": 900_000,
        },
        "clean_baseline": {
            "nearest_neighbor_distance": (
                bootstrap_mean_ci(
                    np.asarray(
                        clean_distances,
                        dtype=np.float64,
                    ),
                    seed=(
                        args.seed
                        + 500
                    ),
                )
            ),
            "random_reference_action_disagreement": (
                bootstrap_mean_ci(
                    np.asarray(
                        clean_disagreements,
                        dtype=np.float64,
                    ),
                    seed=(
                        args.seed
                        + 501
                    ),
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
        "\n" + "=" * 78
    )
    print(
        "RANDOM-REFERENCE CONTROL COMPLETE"
    )
    print(
        "=" * 78
    )
    print(
        f"Saved: {output_path}"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()