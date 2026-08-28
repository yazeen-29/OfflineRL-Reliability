from __future__ import annotations

import json
from pathlib import Path

import d3rlpy
import gymnasium as gym
import numpy as np

from src.reliability.counterfactual import (
    capture_state,
    get_observation,
    make_shifted_observation,
    restore_state,
    run_counterfactual_pair,
    validate_state_pair,
)

from src.reliability.support_distance import (
    build_reference_index,
    fit_reference_standardization,
    nearest_support_distance,
)

from src.reliability.critic_disagreement import (
    twin_critic_values,
    twin_critic_disagreement,
)


TASK = "mujoco/hopper/medium-v0"
ENV_NAME = "Hopper-v5"

CHECKPOINT = Path(
    "checkpoints/iql_100k/"
    "iql_mujoco_hopper_medium-v0_seed0.d3"
)

OUTPUT = Path(
    "results/diagnostics/reliability/"
    "p1_predictor_completeness_seed0.json"
)

REFERENCE_FRACTION = 0.90

N_EPISODES = 20
STATES_PER_EPISODE = 5
N_STATES = N_EPISODES * STATES_PER_EPISODE

SIGMA = 0.10
HORIZON = 10

STATE_SAMPLING_SEED = 20260828
NOISE_SEED = 20260829

EPSILON = 1e-8


def load_reference() -> tuple[np.ndarray, np.ndarray, object]:
    """
    Reproduce the frozen Gaussian reference construction:
    episode-level 90/10 split, reference-only standardization,
    Euclidean nearest-neighbor index.
    """
    dataset, _ = d3rlpy.datasets.get_minari(
        TASK
    )

    episodes = list(
        dataset.episodes
    )

    rng = np.random.default_rng(
        STATE_SAMPLING_SEED
    )

    indices = np.arange(
        len(episodes)
    )

    rng.shuffle(indices)

    split = int(
        len(indices)
        * REFERENCE_FRACTION
    )

    reference_episodes = [
        episodes[int(i)]
        for i in indices[:split]
    ]

    reference_observations = np.concatenate(
        [
            np.asarray(
                episode.observations,
                dtype=np.float64,
            )
            for episode in reference_episodes
        ],
        axis=0,
    )

    mean, std = fit_reference_standardization(
        reference_observations
    )

    standardized_reference = (
        (
            reference_observations
            - mean
        )
        / std
    )

    reference_index = build_reference_index(
        standardized_reference
    )

    return (
        mean,
        std,
        reference_index,
    )


def collect_episode_balanced_states(
    policy,
):
    """
    Collect clean-policy states from exactly 20 episodes,
    then sample five states from each episode.

    This is the diagnostic pilot sampling rule specified in
    P1 Protocol Amendment 002.
    """
    rng = np.random.default_rng(
        STATE_SAMPLING_SEED
    )

    selected = []

    for episode_id in range(
        N_EPISODES
    ):
        env = gym.make(
            ENV_NAME
        )

        episode_states = []

        try:
            env.reset(
                seed=10000 + episode_id
            )

            for step in range(
                1000
            ):
                observation = get_observation(
                    env
                )

                if not np.all(
                    np.isfinite(observation)
                ):
                    break

                state = capture_state(
                    env
                )

                action = np.asarray(
                    policy.predict(
                        observation.reshape(
                            1,
                            -1
                        )
                    )[0],
                    dtype=np.float32,
                )

                if not np.all(
                    np.isfinite(action)
                ):
                    break

                episode_states.append(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "state": state,
                        "observation": observation.copy(),
                    }
                )

                _, _, terminated, truncated, _ = (
                    env.step(action)
                )

                if terminated or truncated:
                    break

        finally:
            env.close()

        if len(episode_states) < STATES_PER_EPISODE:
            raise RuntimeError(
                f"Episode {episode_id} yielded only "
                f"{len(episode_states)} eligible states."
            )

        indices = rng.choice(
            len(episode_states),
            size=STATES_PER_EPISODE,
            replace=False,
        )

        selected.extend(
            episode_states[int(i)]
            for i in indices
        )

    if len(selected) != N_STATES:
        raise RuntimeError(
            f"Expected {N_STATES} states, "
            f"got {len(selected)}."
        )

    return selected


def action_disagreement(
    clean_action,
    shifted_action,
) -> float:
    clean_action = np.asarray(
        clean_action,
        dtype=np.float64,
    )

    shifted_action = np.asarray(
        shifted_action,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            clean_action
            - shifted_action
        )
        / np.sqrt(
            len(clean_action)
        )
    )


def relative_consequence(
    signed_consequence: float,
    clean_return: float,
    shifted_return: float,
) -> float:
    denominator = (
        EPSILON
        + 0.5
        * (
            abs(clean_return)
            + abs(shifted_return)
        )
    )

    return float(
        abs(signed_consequence)
        / denominator
    )


def main():
    print("=" * 80)
    print(
        "P1.2b — 100-STATE PREDICTOR-COMPLETENESS PILOT"
    )
    print("=" * 80)
    print("Status: DIAGNOSTIC ONLY")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    policy = d3rlpy.load_learnable(
        str(CHECKPOINT),
        device="cpu",
    )

    mean, std, reference_index = (
        load_reference()
    )

    print(
        "Reference standardization dimension:",
        len(mean),
    )

    states = collect_episode_balanced_states(
        policy
    )

    print(
        "Episodes sampled:",
        N_EPISODES,
    )

    print(
        "States per episode:",
        STATES_PER_EPISODE,
    )

    print(
        "Total states:",
        len(states),
    )

    noise_rng = np.random.default_rng(
        NOISE_SEED
    )

    records = []

    for i, record in enumerate(states):
        source_state = record["state"]

        clean_env = gym.make(
            ENV_NAME
        )

        shifted_env = gym.make(
            ENV_NAME
        )

        try:
            clean_env.reset(
                seed=20000 + i
            )

            shifted_env.reset(
                seed=30000 + i
            )

            restore_state(
                clean_env,
                source_state,
            )

            restore_state(
                shifted_env,
                source_state,
            )

            validate_state_pair(
                clean_env,
                shifted_env,
            )

            clean_observation = get_observation(
                clean_env
            )

            shifted_observation, standardized_noise = (
                make_shifted_observation(
                    clean_observation,
                    std,
                    SIGMA,
                    noise_rng,
                )
            )

            support_distance, nearest_index = (
                nearest_support_distance(
                    clean_observation,
                    mean,
                    std,
                    reference_index,
                )
            )

            clean_action = np.asarray(
                policy.predict(
                    clean_observation.reshape(
                        1,
                        -1,
                    )
                )[0],
                dtype=np.float64,
            )

            shifted_action = np.asarray(
                policy.predict(
                    shifted_observation.reshape(
                        1,
                        -1,
                    )
                )[0],
                dtype=np.float64,
            )

            delta_action = action_disagreement(
                clean_action,
                shifted_action,
            )

            q1, q2 = twin_critic_values(
                policy,
                clean_observation,
                clean_action,
            )

            critic_gap = twin_critic_disagreement(
                q1,
                q2,
            )

            clean_outcome, shifted_outcome = (
                run_counterfactual_pair(
                    env_clean=clean_env,
                    env_shifted=shifted_env,
                    policy=policy,
                    clean_observation=clean_observation,
                    shifted_observation=shifted_observation,
                    horizon=HORIZON,
                )
            )

            clean_return = (
                clean_outcome.cumulative_return
            )

            shifted_return = (
                shifted_outcome.cumulative_return
            )

            signed_consequence = (
                clean_return
                - shifted_return
            )

            absolute_consequence = abs(
                signed_consequence
            )

            normalized_consequence = (
                relative_consequence(
                    signed_consequence,
                    clean_return,
                    shifted_return,
                )
            )

            records.append(
                {
                    "state_id": i,
                    "episode_id": int(
                        record["episode_id"]
                    ),
                    "decision_step": int(
                        record["step"]
                    ),
                    "sigma": SIGMA,
                    "support_distance": float(
                        support_distance
                    ),
                    "nearest_reference_index": int(
                        nearest_index
                    ),
                    "clean_observation": (
                        clean_observation.tolist()
                    ),
                    "shifted_observation": (
                        shifted_observation.tolist()
                    ),
                    "standardized_noise": (
                        standardized_noise.tolist()
                    ),
                    "clean_action": (
                        clean_action.tolist()
                    ),
                    "shifted_action": (
                        shifted_action.tolist()
                    ),
                    "action_disagreement": float(
                        delta_action
                    ),
                    "q1_clean": float(
                        q1
                    ),
                    "q2_clean": float(
                        q2
                    ),
                    "twin_critic_disagreement": float(
                        critic_gap
                    ),
                    "clean_return_h10": float(
                        clean_return
                    ),
                    "shifted_return_h10": float(
                        shifted_return
                    ),
                    "delta_J10": float(
                        signed_consequence
                    ),
                    "abs_delta_J10": float(
                        absolute_consequence
                    ),
                    "relative_consequence_h10": float(
                        normalized_consequence
                    ),
                    "clean_steps": int(
                        len(
                            clean_outcome.rewards
                        )
                    ),
                    "shifted_steps": int(
                        len(
                            shifted_outcome.rewards
                        )
                    ),
                    "clean_terminated": bool(
                        np.any(
                            clean_outcome.terminated
                        )
                    ),
                    "shifted_terminated": bool(
                        np.any(
                            shifted_outcome.terminated
                        )
                    ),
                    "clean_truncated": bool(
                        np.any(
                            clean_outcome.truncated
                        )
                    ),
                    "shifted_truncated": bool(
                        np.any(
                            shifted_outcome.truncated
                        )
                    ),
                }
            )

        finally:
            clean_env.close()
            shifted_env.close()

    # ----------------------------------------------------------
    # Basic validation.
    # ----------------------------------------------------------

    required = {
        "support_distance",
        "action_disagreement",
        "twin_critic_disagreement",
        "delta_J10",
        "abs_delta_J10",
        "relative_consequence_h10",
    }

    for record in records:
        missing = (
            required
            - set(record)
        )

        if missing:
            raise RuntimeError(
                f"State {record['state_id']} "
                f"missing fields: {missing}"
            )

    # ----------------------------------------------------------
    # Summary.
    # ----------------------------------------------------------

    def arr(key):
        return np.asarray(
            [
                record[key]
                for record in records
            ],
            dtype=np.float64,
        )

    summary = {}

    for key in [
        "support_distance",
        "action_disagreement",
        "twin_critic_disagreement",
        "delta_J10",
        "abs_delta_J10",
        "relative_consequence_h10",
    ]:
        values = arr(key)

        summary[key] = {
            "n": int(len(values)),
            "mean": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
            "std": float(
                np.std(values)
            ),
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
            "finite_fraction": float(
                np.mean(
                    np.isfinite(values)
                )
            ),
        }

    summary["execution"] = {
        "clean_full_horizon_fraction": float(
            np.mean(
                [
                    r["clean_steps"] == HORIZON
                    for r in records
                ]
            )
        ),
        "shifted_full_horizon_fraction": float(
            np.mean(
                [
                    r["shifted_steps"] == HORIZON
                    for r in records
                ]
            )
        ),
        "clean_termination_fraction": float(
            np.mean(
                [
                    r["clean_terminated"]
                    for r in records
                ]
            )
        ),
        "shifted_termination_fraction": float(
            np.mean(
                [
                    r["shifted_terminated"]
                    for r in records
                ]
            )
        ),
    }

    output = {
        "experiment": (
            "P1.2b-predictor-completeness-pilot"
        ),
        "status": "diagnostic_only",
        "task": TASK,
        "environment": ENV_NAME,
        "checkpoint": str(
            CHECKPOINT
        ),
        "policy_seed": 0,
        "n_episodes": N_EPISODES,
        "states_per_episode": STATES_PER_EPISODE,
        "n_states": N_STATES,
        "sigma": SIGMA,
        "horizon": HORIZON,
        "state_sampling_seed": STATE_SAMPLING_SEED,
        "noise_seed": NOISE_SEED,
        "reference_fraction": REFERENCE_FRACTION,
        "epsilon": EPSILON,
        "protocol": (
            "experiments/reliability/"
            "P1_DECISION_CONSEQUENCE_PROTOCOL.md"
        ),
        "summary": summary,
        "records": records,
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

    print("\n" + "=" * 80)
    print(
        "P1.2b PREDICTOR-COMPLETENESS PILOT COMPLETE"
    )
    print("=" * 80)

    for key, result in summary.items():
        if key == "execution":
            continue

        print(
            f"{key:32s} "
            f"mean={result['mean']:.8f} "
            f"median={result['median']:.8f}"
        )

    print(
        "\nSaved:",
        OUTPUT,
    )

    print(
        "\n⚠️ DIAGNOSTIC ONLY — NOT PAPER EVIDENCE"
    )


if __name__ == "__main__":
    main()
