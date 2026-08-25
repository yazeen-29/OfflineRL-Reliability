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

# Fixed candidate-set size.
#
# The nearest neighbor itself is excluded.
# We search among the next 127 nearby reference states.
K_NEIGHBORS = 128

# Among the best distance-matched candidates, randomly select
# one of the top few candidates. This preserves a controlled
# element of randomization without sacrificing distance matching.
TOP_MATCH_CANDIDATES = 8

# Maximum relative distance error considered a close match.
#
# This is NOT used as a hidden fallback.
# It is reported as a diagnostic criterion.
MATCH_TOLERANCE = 0.10


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
    Identical episode-level split to the frozen Gaussian study.
    """
    episodes = list(dataset.episodes)

    if len(episodes) < 20:
        raise RuntimeError(
            "Dataset is too small for a reliable "
            "episode-level split."
        )

    rng = np.random.default_rng(seed)

    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    split = int(
        len(indices) * reference_fraction
    )

    reference_indices = indices[:split]
    query_indices = indices[split:]

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
    Identical held-out query sampling protocol.
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
# NEAREST-NEIGHBOR INDEX
# ============================================================

def build_neighbor_index(
    standardized_reference: np.ndarray,
):
    """
    Fixed local candidate set.

    K=128 is deliberately fixed before looking at results.
    """
    n_neighbors = min(
        K_NEIGHBORS,
        len(standardized_reference),
    )

    index = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    index.fit(
        standardized_reference
    )

    return index


# ============================================================
# DISTANCE-MATCHED ALTERNATIVE REFERENCE
# ============================================================

def select_distance_matched_reference(
    standardized_query: np.ndarray,
    index: NearestNeighbors,
    rng: np.random.Generator,
):
    """
    Select an alternative reference observation whose distance
    is as close as possible to the true nearest-neighbor distance.

    Procedure:

        1. Retrieve K nearest reference observations.
        2. Identify the true nearest neighbor.
        3. Exclude that observation.
        4. Compute each alternative's relative distance error:

               |d_candidate / d_nn - 1|

        5. Keep the best TOP_MATCH_CANDIDATES alternatives.
        6. Randomly choose one of those candidates.

    No broad tolerance fallback is used.

    The actual distance error is always recorded.
    """

    query = standardized_query.reshape(
        1,
        -1,
    )

    distances, indices = index.kneighbors(
        query,
        n_neighbors=K_NEIGHBORS,
    )

    distances = distances[0]
    indices = indices[0]

    nearest_distance = float(
        distances[0]
    )

    nearest_index = int(
        indices[0]
    )

    candidate_distances = distances[1:]
    candidate_indices = indices[1:]

    if len(candidate_indices) == 0:
        raise RuntimeError(
            "No alternative reference observations available."
        )

    if nearest_distance > 0.0:
        relative_errors = np.abs(
            candidate_distances
            / nearest_distance
            - 1.0
        )
    else:
        # Exact query/reference duplicate.
        # In this rare case, absolute distance is used.
        relative_errors = np.abs(
            candidate_distances
        )

    order = np.argsort(
        relative_errors
    )

    top_count = min(
        TOP_MATCH_CANDIDATES,
        len(order),
    )

    best_positions = order[
        :top_count
    ]

    chosen_position = int(
        rng.choice(
            best_positions
        )
    )

    matched_index = int(
        candidate_indices[
            chosen_position
        ]
    )

    matched_distance = float(
        candidate_distances[
            chosen_position
        ]
    )

    if nearest_distance > 0.0:
        distance_ratio = float(
            matched_distance
            / nearest_distance
        )
    else:
        distance_ratio = 1.0

    relative_error = float(
        abs(
            distance_ratio - 1.0
        )
    )

    return {
        "nearest_neighbor_index": nearest_index,
        "nearest_neighbor_distance": nearest_distance,
        "matched_reference_index": matched_index,
        "matched_reference_distance": matched_distance,
        "matched_distance_ratio": distance_ratio,
        "relative_distance_error": relative_error,
        "candidate_pool_size": int(
            top_count
        ),
        "within_ten_percent": bool(
            relative_error <= MATCH_TOLERANCE
        ),
    }


# ============================================================
# POLICY / METRIC
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
    Same RMS-style action disagreement as the frozen
    Gaussian experiment.
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

    rng = np.random.default_rng(seed)

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
            "Distance-matched random-reference control "
            "for the IQL Gaussian observation-shift study."
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
            "iql_seed0_distance_matched_smoke.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "DISTANCE-MATCHED RANDOM-REFERENCE CONTROL"
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
    print(
        f"K nearest candidates: {K_NEIGHBORS}"
    )
    print(
        f"Random top candidates: {TOP_MATCH_CANDIDATES}"
    )
    print(
        f"10% matching criterion: <= {MATCH_TOLERANCE:.0%}"
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
        "\n[1/9] Loading frozen IQL policy..."
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
        "\n[2/9] Loading offline dataset..."
    )

    dataset, _ = load_dataset(
        args.task
    )

    print(
        f"[OK] Episodes: "
        f"{dataset.size()}"
    )

    # --------------------------------------------------------
    # 3. Same episode split and query sampling
    # --------------------------------------------------------

    print(
        "\n[3/9] Creating episode-level split..."
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
    # 4. Same reference-only standardization
    # --------------------------------------------------------

    print(
        "\n[4/9] Standardizing observations..."
    )

    mean, std = fit_standardization(
        reference_observations
    )

    standardized_reference = standardize(
        reference_observations,
        mean,
        std,
    )

    neighbor_index = build_neighbor_index(
        standardized_reference
    )

    print(
        "[OK] Reference standardization fitted"
    )

    # --------------------------------------------------------
    # 5. Matching diagnostic
    # --------------------------------------------------------

    print(
        "\n[5/9] Evaluating distance-matching quality..."
    )

    diagnostic_rng = np.random.default_rng(
        args.seed + 777_777
    )

    diagnostic_ratios = []
    diagnostic_errors = []
    diagnostic_within_tolerance = []
    diagnostic_count = min(
        100,
        len(query_observations),
    )

    for query_id in range(
        diagnostic_count
    ):

        clean_obs = query_observations[
            query_id
        ]

        standardized_clean = standardize(
            clean_obs,
            mean,
            std,
        )

        match = (
            select_distance_matched_reference(
                standardized_clean,
                neighbor_index,
                diagnostic_rng,
            )
        )

        diagnostic_ratios.append(
            match[
                "matched_distance_ratio"
            ]
        )

        diagnostic_errors.append(
            match[
                "relative_distance_error"
            ]
        )

        diagnostic_within_tolerance.append(
            match[
                "within_ten_percent"
            ]
        )

    diagnostic_ratios = np.asarray(
        diagnostic_ratios,
        dtype=np.float64,
    )

    diagnostic_errors = np.asarray(
        diagnostic_errors,
        dtype=np.float64,
    )

    diagnostic_within_tolerance = np.asarray(
        diagnostic_within_tolerance,
        dtype=bool,
    )

    diagnostic_match_fraction = float(
        np.mean(
            diagnostic_within_tolerance
        )
    )

    print(
        "[OK] Diagnostic queries:",
        diagnostic_count,
    )

    print(
        "[OK] Mean matched/NN ratio:",
        np.mean(
            diagnostic_ratios
        ),
    )

    print(
        "[OK] Median matched/NN ratio:",
        np.median(
            diagnostic_ratios
        ),
    )

    print(
        "[OK] Mean relative distance error:",
        np.mean(
            diagnostic_errors
        ),
    )

    print(
        "[OK] Fraction within ±10%:",
        diagnostic_match_fraction,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do not proceed to a full run automatically if the
    # diagnostic is poor.
    # --------------------------------------------------------

    if (
        diagnostic_match_fraction < 0.80
    ):
        raise RuntimeError(
            "\n"
            "DISTANCE-MATCHING QUALITY TOO LOW.\n"
            f"Only {diagnostic_match_fraction:.1%} "
            "of diagnostic queries have a matched reference "
            "within ±10% of the nearest-neighbor distance.\n"
            "\n"
            "Do NOT run the full experiment yet. "
            "The reference pool does not provide sufficiently "
            "close alternative states under this fixed protocol."
        )

    # --------------------------------------------------------
    # 6. Clean baseline
    # --------------------------------------------------------

    print(
        "\n[6/9] Computing clean distance-matched baseline..."
    )

    clean_nn_distances = []
    clean_matched_distances = []
    clean_disagreements = []
    clean_relative_errors = []
    clean_within_tolerance = []

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

        for repeat_id in range(
            args.noise_repeats
        ):

            match_rng = np.random.default_rng(
                args.seed * 1_000_000
                + 800_000
                + query_id * 10
                + repeat_id
            )

            match = (
                select_distance_matched_reference(
                    standardized_clean,
                    neighbor_index,
                    match_rng,
                )
            )

            matched_index = int(
                match[
                    "matched_reference_index"
                ]
            )

            matched_obs = (
                reference_observations[
                    matched_index
                ]
            )

            matched_action = policy_action(
                policy,
                matched_obs,
            )

            disagreement = (
                action_disagreement(
                    clean_action,
                    matched_action,
                )
            )

            clean_nn_distances.append(
                match[
                    "nearest_neighbor_distance"
                ]
            )

            clean_matched_distances.append(
                match[
                    "matched_reference_distance"
                ]
            )

            clean_disagreements.append(
                disagreement
            )

            clean_relative_errors.append(
                match[
                    "relative_distance_error"
                ]
            )

            clean_within_tolerance.append(
                match[
                    "within_ten_percent"
                ]
            )

    print(
        "[OK] Mean NN distance:",
        np.mean(
            clean_nn_distances
        ),
    )

    print(
        "[OK] Mean matched distance:",
        np.mean(
            clean_matched_distances
        ),
    )

    print(
        "[OK] Mean relative distance error:",
        np.mean(
            clean_relative_errors
        ),
    )

    print(
        "[OK] Clean matched-reference disagreement:",
        np.mean(
            clean_disagreements
        ),
    )

    print(
        "[OK] Clean within ±10%:",
        np.mean(
            clean_within_tolerance
        ),
    )

    # --------------------------------------------------------
    # 7. Controlled Gaussian shifts
    # --------------------------------------------------------

    print(
        "\n[7/9] Running controlled Gaussian shifts..."
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
        matched_distances = []
        action_changes = []
        matched_disagreements = []
        relative_errors = []
        distance_ratios = []
        noise_norms = []
        within_tolerance = []

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

                # EXACT Gaussian seed formula from
                # the frozen primary experiment.
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

                standardized_noise = rng.normal(
                    loc=0.0,
                    scale=noise_level,
                    size=len(clean_obs),
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

                standardized_shifted = standardize(
                    shifted_obs,
                    mean,
                    std,
                )

                match_rng = np.random.default_rng(
                    args.seed * 1_000_000
                    + 2_000_000
                    + level_id * 100_000
                    + query_id * 10
                    + repeat_id
                )

                match = (
                    select_distance_matched_reference(
                        standardized_shifted,
                        neighbor_index,
                        match_rng,
                    )
                )

                matched_index = int(
                    match[
                        "matched_reference_index"
                    ]
                )

                matched_reference_obs = (
                    reference_observations[
                        matched_index
                    ]
                )

                matched_reference_action = (
                    policy_action(
                        policy,
                        matched_reference_obs,
                    )
                )

                explanation_disagreement = (
                    action_disagreement(
                        shifted_action,
                        matched_reference_action,
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

                nn_distance = float(
                    match[
                        "nearest_neighbor_distance"
                    ]
                )

                matched_distance = float(
                    match[
                        "matched_reference_distance"
                    ]
                )

                relative_error = float(
                    match[
                        "relative_distance_error"
                    ]
                )

                distance_ratio = float(
                    match[
                        "matched_distance_ratio"
                    ]
                )

                within_tolerance = bool(
                    match[
                        "within_ten_percent"
                    ]
                )

                nn_distances.append(
                    nn_distance
                )

                matched_distances.append(
                    matched_distance
                )

                action_changes.append(
                    action_change
                )

                matched_disagreements.append(
                    explanation_disagreement
                )

                relative_errors.append(
                    relative_error
                )

                distance_ratios.append(
                    distance_ratio
                )

                noise_norms.append(
                    noise_norm
                )

                within_tolerance.append(
                    within_tolerance
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
                        "matched_reference_index": (
                            matched_index
                        ),
                        "matched_reference_distance": (
                            matched_distance
                        ),
                        "matched_distance_ratio": (
                            distance_ratio
                        ),
                        "relative_distance_error": (
                            relative_error
                        ),
                        "within_ten_percent": (
                            within_tolerance
                        ),
                        "candidate_pool_size": (
                            int(
                                match[
                                    "candidate_pool_size"
                                ]
                            )
                        ),
                        "policy_action_change": (
                            action_change
                        ),
                        "matched_reference_action_disagreement": (
                            explanation_disagreement
                        ),
                    }
                )

        nn_distances = np.asarray(
            nn_distances,
            dtype=np.float64,
        )

        matched_distances = np.asarray(
            matched_distances,
            dtype=np.float64,
        )

        action_changes = np.asarray(
            action_changes,
            dtype=np.float64,
        )

        matched_disagreements = np.asarray(
            matched_disagreements,
            dtype=np.float64,
        )

        relative_errors = np.asarray(
            relative_errors,
            dtype=np.float64,
        )

        distance_ratios = np.asarray(
            distance_ratios,
            dtype=np.float64,
        )

        noise_norms = np.asarray(
            noise_norms,
            dtype=np.float64,
        )

        within_tolerance = np.asarray(
            within_tolerance,
            dtype=bool,
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
                    seed=(
                        args.seed
                        + level_id
                    ),
                )
            ),
            "nearest_neighbor_distance": (
                bootstrap_mean_ci(
                    nn_distances,
                    seed=(
                        args.seed
                        + 100
                        + level_id
                    ),
                )
            ),
            "matched_reference_distance": (
                bootstrap_mean_ci(
                    matched_distances,
                    seed=(
                        args.seed
                        + 150
                        + level_id
                    ),
                )
            ),
            "relative_distance_error": (
                bootstrap_mean_ci(
                    relative_errors,
                    seed=(
                        args.seed
                        + 175
                        + level_id
                    ),
                )
            ),
            "matched_distance_ratio": (
                bootstrap_mean_ci(
                    distance_ratios,
                    seed=(
                        args.seed
                        + 180
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
            "matched_reference_action_disagreement": (
                bootstrap_mean_ci(
                    matched_disagreements,
                    seed=(
                        args.seed
                        + 300
                        + level_id
                    ),
                )
            ),
            "fraction_within_ten_percent": float(
                np.mean(
                    within_tolerance
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
            "  Matched-reference distance:",
            summary[
                "matched_reference_distance"
            ]["mean"],
        )

        print(
            "  Mean relative distance error:",
            summary[
                "relative_distance_error"
            ]["mean"],
        )

        print(
            "  Mean matched/NN ratio:",
            summary[
                "matched_distance_ratio"
            ]["mean"],
        )

        print(
            "  Policy action change:",
            summary[
                "policy_action_change"
            ]["mean"],
        )

        print(
            "  Matched-reference disagreement:",
            summary[
                "matched_reference_action_disagreement"
            ]["mean"],
        )

        print(
            "  Fraction within ±10%:",
            summary[
                "fraction_within_ten_percent"
            ],
        )

    # --------------------------------------------------------
    # 8. Final matching-quality validation
    # --------------------------------------------------------

    print(
        "\n[8/9] Validating matching quality..."
    )

    all_ratios = np.asarray(
        [
            record[
                "matched_distance_ratio"
            ]
            for record in all_records
        ],
        dtype=np.float64,
    )

    all_relative_errors = np.asarray(
        [
            record[
                "relative_distance_error"
            ]
            for record in all_records
        ],
        dtype=np.float64,
    )

    all_within_tolerance = np.asarray(
        [
            record[
                "within_ten_percent"
            ]
            for record in all_records
        ],
        dtype=bool,
    )

    overall_match_fraction = float(
        np.mean(
            all_within_tolerance
        )
    )

    overall_mean_ratio = float(
        np.mean(
            all_ratios
        )
    )

    overall_mean_error = float(
        np.mean(
            all_relative_errors
        )
    )

    print(
        "[OK] Overall fraction within ±10%:",
        overall_match_fraction,
    )

    print(
        "[OK] Overall mean matched/NN ratio:",
        overall_mean_ratio,
    )

    print(
        "[OK] Overall mean relative error:",
        overall_mean_error,
    )

    # --------------------------------------------------------
    # Do not silently declare a successful control if the
    # matching quality is poor.
    # --------------------------------------------------------

    if overall_match_fraction < 0.80:
        raise RuntimeError(
            "\n"
            "MATCHING QUALITY INSUFFICIENT.\n"
            f"Only {overall_match_fraction:.1%} "
            "of records fall within ±10% of the "
            "nearest-neighbor distance.\n"
            "\n"
            "The control should NOT be treated as a "
            "distance-matched reference experiment."
        )

    # --------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------

    print(
        "\n[9/9] Saving experiment results..."
    )

    output = {
        "experiment": (
            "IQL-100K-distance-matched-random-reference-"
            "gaussian-observation-shift-control"
        ),
        "control_definition": (
            "For each shifted query, retrieve a fixed set of "
            "nearest reference observations, exclude the true "
            "nearest neighbor, and randomly sample among the "
            "best distance-matched alternatives according to "
            "absolute relative error with respect to the "
            "nearest-neighbor distance."
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
        "matching_protocol": {
            "k_neighbors": int(
                K_NEIGHBORS
            ),
            "top_match_candidates": int(
                TOP_MATCH_CANDIDATES
            ),
            "tolerance": float(
                MATCH_TOLERANCE
            ),
            "exclude_true_nearest_neighbor": True,
            "no_hidden_distance_fallback": True,
        },
        "standardization": {
            "method": (
                "reference-dataset-z-score"
            ),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "matching_quality": {
            "fraction_within_ten_percent": (
                overall_match_fraction
            ),
            "mean_matched_to_nn_ratio": (
                overall_mean_ratio
            ),
            "mean_relative_distance_error": (
                overall_mean_error
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

    print()
    print("=" * 78)
    print(
        "DISTANCE-MATCHED CONTROL COMPLETE"
    )
    print("=" * 78)

    print(
        f"Saved: {output_path}"
    )

    print(
        "Overall within ±10%:",
        overall_match_fraction,
    )

    print(
        "Mean matched/NN ratio:",
        overall_mean_ratio,
    )

    print(
        "Mean relative distance error:",
        overall_mean_error,
    )

    print("=" * 78)


if __name__ == "__main__":
    main()