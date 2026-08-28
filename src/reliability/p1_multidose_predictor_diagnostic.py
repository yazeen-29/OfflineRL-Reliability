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
    "p1_multidose_predictor_seed0.json"
)

REFERENCE_FRACTION = 0.90

N_EPISODES = 20
STATES_PER_EPISODE = 5
N_STATES = N_EPISODES * STATES_PER_EPISODE

SIGMA_LEVELS = [
    0.00,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
]

HORIZON = 10

STATE_SAMPLING_SEED = 20260828
NOISE_SEED = 20260829

EPSILON = 1e-8


def load_reference():
    dataset, _ = d3rlpy.datasets.get_minari(TASK)

    episodes = list(dataset.episodes)

    rng = np.random.default_rng(
        STATE_SAMPLING_SEED
    )

    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    split = int(
        len(indices) * REFERENCE_FRACTION
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
        (reference_observations - mean)
        / std
    )

    reference_index = build_reference_index(
        standardized_reference
    )

    return mean, std, reference_index


def collect_balanced_states(policy):
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

        eligible = []

        try:
            env.reset(
                seed=10000 + episode_id
            )

            for step in range(1000):
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

                eligible.append(
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

        if len(eligible) < STATES_PER_EPISODE:
            raise RuntimeError(
                f"Episode {episode_id} yielded "
                f"{len(eligible)} states."
            )

        chosen = rng.choice(
            len(eligible),
            size=STATES_PER_EPISODE,
            replace=False,
        )

        selected.extend(
            eligible[int(i)]
            for i in chosen
        )

    if len(selected) != N_STATES:
        raise RuntimeError(
            f"Expected {N_STATES} states; "
            f"got {len(selected)}."
        )

    return selected


def action_disagreement(
    clean_action,
    shifted_action,
):
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
            clean_action - shifted_action
        )
        / np.sqrt(
            len(clean_action)
        )
    )


def relative_consequence(
    signed_consequence,
    clean_return,
    shifted_return,
):
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
    print("P1.2c — MULTI-DOSE PREDICTOR DIAGNOSTIC")
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

    mean, std, reference_index = load_reference()

    states = collect_balanced_states(
        policy
    )

    print(
        "Episodes:",
        N_EPISODES,
    )
    print(
        "States/episode:",
        STATES_PER_EPISODE,
    )
    print(
        "Total states:",
        len(states),
    )

    # Same noise stream, state ordering, and state set are reused
    # across all sigma values.
    noise_rng = np.random.default_rng(
        NOISE_SEED
    )

    records = []

    for state_id, state_record in enumerate(states):
        source_state = state_record["state"]

        clean_env = gym.make(
            ENV_NAME
        )

        try:
            clean_env.reset(
                seed=20000 + state_id
            )

            restore_state(
                clean_env,
                source_state,
            )

            clean_observation = get_observation(
                clean_env
            )

            clean_action = np.asarray(
                policy.predict(
                    clean_observation.reshape(
                        1,
                        -1
                    )
                )[0],
                dtype=np.float64,
            )

            support_distance, nearest_index = (
                nearest_support_distance(
                    clean_observation,
                    mean,
                    std,
                    reference_index,
                )
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

        finally:
            clean_env.close()

        # One fixed Gaussian draw in standardized coordinates is
        # generated per state and then scaled for each sigma.
        standard_normal = noise_rng.normal(
            loc=0.0,
            scale=1.0,
            size=clean_observation.shape,
        )

        for sigma in SIGMA_LEVELS:
            shifted_observation = (
                clean_observation
                + (
                    standard_normal
                    * float(sigma)
                    * std
                )
            )

            shifted_env = gym.make(
                ENV_NAME
            )

            try:
                shifted_env.reset(
                    seed=30000
                    + state_id
                )

                restore_state(
                    shifted_env,
                    source_state,
                )

                validate_state_pair(
                    shifted_env,
                    shifted_env,
                )

                # Fresh clean environment is required because the clean
                # branch is consumed inside run_counterfactual_pair.
                clean_branch_env = gym.make(
                    ENV_NAME
                )

                try:
                    clean_branch_env.reset(
                        seed=40000
                        + state_id
                    )

                    restore_state(
                        clean_branch_env,
                        source_state,
                    )

                    validate_state_pair(
                        clean_branch_env,
                        shifted_env,
                    )

                    clean_outcome, shifted_outcome = (
                        run_counterfactual_pair(
                            env_clean=clean_branch_env,
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

                    signed = (
                        clean_return
                        - shifted_return
                    )

                    disagreement = (
                        action_disagreement(
                            clean_action,
                            shifted_outcome.actions[0],
                        )
                    )

                    records.append(
                        {
                            "state_id": state_id,
                            "episode_id": int(
                                state_record["episode_id"]
                            ),
                            "decision_step": int(
                                state_record["step"]
                            ),
                            "sigma": float(sigma),
                            "support_distance": float(
                                support_distance
                            ),
                            "nearest_reference_index": int(
                                nearest_index
                            ),
                            "action_disagreement": float(
                                disagreement
                            ),
                            "q1_clean": float(q1),
                            "q2_clean": float(q2),
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
                                signed
                            ),
                            "abs_delta_J10": float(
                                abs(signed)
                            ),
                            "relative_consequence_h10": (
                                relative_consequence(
                                    signed,
                                    clean_return,
                                    shifted_return,
                                )
                            ),
                            "clean_steps": int(
                                len(clean_outcome.rewards)
                            ),
                            "shifted_steps": int(
                                len(shifted_outcome.rewards)
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
                    clean_branch_env.close()

            finally:
                shifted_env.close()

    # ----------------------------------------------------------
    # Summary by sigma.
    # ----------------------------------------------------------

    summary = {}

    for sigma in SIGMA_LEVELS:
        sigma_records = [
            r for r in records
            if np.isclose(
                r["sigma"],
                sigma,
            )
        ]

        degradation = np.asarray(
            [
                r["delta_J10"]
                for r in sigma_records
            ],
            dtype=float,
        )

        absolute = np.asarray(
            [
                r["abs_delta_J10"]
                for r in sigma_records
            ],
            dtype=float,
        )

        disagreement = np.asarray(
            [
                r["action_disagreement"]
                for r in sigma_records
            ],
            dtype=float,
        )

        summary[str(sigma)] = {
            "n_states": int(
                len(sigma_records)
            ),
            "mean_delta_J10": float(
                np.mean(degradation)
            ),
            "median_delta_J10": float(
                np.median(degradation)
            ),
            "mean_abs_delta_J10": float(
                np.mean(absolute)
            ),
            "median_abs_delta_J10": float(
                np.median(absolute)
            ),
            "positive_delta_fraction": float(
                np.mean(
                    degradation > 0
                )
            ),
            "mean_action_disagreement": float(
                np.mean(disagreement)
            ),
            "median_action_disagreement": float(
                np.median(disagreement)
            ),
            "clean_full_horizon_fraction": float(
                np.mean(
                    [
                        r["clean_steps"] == HORIZON
                        for r in sigma_records
                    ]
                )
            ),
            "shifted_full_horizon_fraction": float(
                np.mean(
                    [
                        r["shifted_steps"] == HORIZON
                        for r in sigma_records
                    ]
                )
            ),
            "shifted_termination_fraction": float(
                np.mean(
                    [
                        r["shifted_terminated"]
                        for r in sigma_records
                    ]
                )
            ),
        }

    output = {
        "experiment": (
            "P1.2c-multidose-predictor-diagnostic"
        ),
        "status": "diagnostic_only",
        "task": TASK,
        "environment": ENV_NAME,
        "checkpoint": str(CHECKPOINT),
        "policy_seed": 0,
        "n_episodes": N_EPISODES,
        "states_per_episode": STATES_PER_EPISODE,
        "n_states": N_STATES,
        "sigma_levels": SIGMA_LEVELS,
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
    print("P1.2c COMPLETE")
    print("=" * 80)

    for sigma in SIGMA_LEVELS:
        s = summary[str(sigma)]

        print(
            f"sigma={sigma:>5.3f} "
            f"mean ΔJ10={s['mean_delta_J10']:+.8f} "
            f"mean |ΔJ10|={s['mean_abs_delta_J10']:.8f} "
            f"mean Δa={s['mean_action_disagreement']:.8f}"
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
