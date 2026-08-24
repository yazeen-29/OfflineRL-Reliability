from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

import d3rlpy


def load_hopper_dataset(task: str):
    dataset, env = d3rlpy.datasets.get_minari(task)
    return dataset, env


def extract_observations(dataset) -> np.ndarray:
    """
    Collect all observations from all episodes into one matrix.

    Shape:
        (N, observation_dim)
    """
    observations = []

    for episode in dataset.episodes:
        obs = np.asarray(episode.observations, dtype=np.float64)

        if obs.ndim != 2:
            raise ValueError(
                f"Expected episode observations to be 2D, got shape {obs.shape}"
            )

        observations.append(obs)

    if not observations:
        raise RuntimeError("Dataset contains no observations.")

    return np.concatenate(observations, axis=0)


def fit_neighbor_index(observations: np.ndarray) -> NearestNeighbors:
    """
    Fit a standard Euclidean nearest-neighbor index.

    We deliberately fit this only on dataset observations.
    """
    if observations.ndim != 2:
        raise ValueError(
            f"Expected observations with shape (N, D), got {observations.shape}"
        )

    index = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )
    index.fit(observations)

    return index


def explain_observation(
    observation: np.ndarray,
    observations: np.ndarray,
    neighbor_index: NearestNeighbors,
) -> dict:
    """
    Find the closest dataset state to a query observation.
    """
    query = np.asarray(observation, dtype=np.float64).reshape(1, -1)

    distance, index = neighbor_index.kneighbors(query, n_neighbors=1)

    neighbor_id = int(index[0, 0])
    neighbor_distance = float(distance[0, 0])

    return {
        "query_observation": query[0].tolist(),
        "nearest_neighbor_index": neighbor_id,
        "nearest_neighbor_distance": neighbor_distance,
        "nearest_neighbor_observation": observations[neighbor_id].tolist(),
    }


def evaluate_dataset_coverage(
    observations: np.ndarray,
    neighbor_index: NearestNeighbors,
) -> dict:
    """
    Leave-one-out style dataset coverage check.

    For each state, ask for its two nearest points and use the second
    one as the nearest *other* dataset state.
    """
    if len(observations) < 2:
        raise ValueError("Need at least two observations.")

    distances, _ = neighbor_index.kneighbors(
        observations,
        n_neighbors=2,
    )

    # Column 0 is the point itself; column 1 is the nearest other point.
    other_distances = distances[:, 1]

    return {
        "num_observations": int(len(observations)),
        "observation_dim": int(observations.shape[1]),
        "mean_nearest_other_distance": float(np.mean(other_distances)),
        "std_nearest_other_distance": float(np.std(other_distances)),
        "median_nearest_other_distance": float(np.median(other_distances)),
        "min_nearest_other_distance": float(np.min(other_distances)),
        "max_nearest_other_distance": float(np.max(other_distances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor explanation baseline."
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
        default=100,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/explanations/"
            "iql_100k_nearest_neighbor_baseline.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NEAREST-NEIGHBOR EXPLANATION BASELINE")
    print("=" * 70)
    print(f"Task       : {args.task}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Seed       : {args.seed}")
    print(f"Queries    : {args.num_queries}")
    print("=" * 70)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    # ------------------------------------------------------------------
    # 1. Verify the frozen policy can be loaded.
    # ------------------------------------------------------------------
    print("\n[1/5] Loading frozen IQL checkpoint...")
    policy = d3rlpy.load_learnable(
        args.checkpoint,
        device="cpu",
    )
    print(f"[OK] Policy loaded: {type(policy).__name__}")

    # ------------------------------------------------------------------
    # 2. Load dataset.
    # ------------------------------------------------------------------
    print("\n[2/5] Loading Minari dataset...")
    dataset, env = load_hopper_dataset(args.task)
    print(f"[OK] Dataset episodes: {dataset.size()}")

    # ------------------------------------------------------------------
    # 3. Build explanation index.
    # ------------------------------------------------------------------
    print("\n[3/5] Extracting dataset observations...")
    observations = extract_observations(dataset)

    print(f"[OK] Observations: {observations.shape}")

    print("Building nearest-neighbor index...")
    neighbor_index = fit_neighbor_index(observations)
    print("[OK] Index built")

    # ------------------------------------------------------------------
    # 4. Coverage baseline.
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating dataset coverage...")
    coverage = evaluate_dataset_coverage(
        observations,
        neighbor_index,
    )

    print(
        "Mean nearest-other distance:",
        coverage["mean_nearest_other_distance"],
    )
    print(
        "Std nearest-other distance:",
        coverage["std_nearest_other_distance"],
    )

    # ------------------------------------------------------------------
    # 5. Query the frozen policy on deterministic initial states.
    # ------------------------------------------------------------------
    print("\n[5/5] Generating explanation examples...")

    rng = np.random.default_rng(args.seed)

    query_records = []

    episode_count = 0

    while len(query_records) < args.num_queries:
        episode_count += 1

        obs, _ = env.reset(seed=args.seed + episode_count)

        action = policy.predict(
            np.asarray([obs], dtype=np.float64)
        )[0]

        explanation = explain_observation(
            obs,
            observations,
            neighbor_index,
        )

        explanation["policy_action"] = np.asarray(
            action
        ).tolist()

        explanation["query_id"] = len(query_records)

        query_records.append(explanation)

    # Deterministic shuffle only for presentation order.
    rng.shuffle(query_records)

    result = {
        "experiment": "IQL-100K-nearest-neighbor-explanation-baseline",
        "task": args.task,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "policy_type": type(policy).__name__,
        "dataset_episodes": int(dataset.size()),
        "dataset_observations": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "coverage": coverage,
        "num_queries": len(query_records),
        "queries": query_records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result, indent=2)
    )

    print("\n" + "=" * 70)
    print("EXPLANATION BASELINE COMPLETE")
    print("=" * 70)
    print(f"Saved: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()