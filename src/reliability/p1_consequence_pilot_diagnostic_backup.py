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


TASK = "mujoco/hopper/medium-v0"
ENV_NAME = "Hopper-v5"

CHECKPOINT = Path(
    "checkpoints/iql_100k/"
    "iql_mujoco_hopper_medium-v0_seed0.d3"
)

OUTPUT = Path(
    "results/diagnostics/reliability/"
    "p1_consequence_pilot_seed0.json"
)

# Diagnostic-only pilot settings.
N_STATES = 20
SIGMA = 0.10
HORIZONS = [1, 5, 10, 20]

REFERENCE_FRACTION = 0.90

STATE_SAMPLING_SEED = 20260828
NOISE_SEED = 20260829


def load_reference_std() -> np.ndarray:
    """Compute reference-only feature standard deviations."""
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

    observations = []

    for episode in reference_episodes:
        obs = np.asarray(
            episode.observations,
            dtype=np.float64,
        )

        if obs.ndim != 2:
            raise RuntimeError(
                f"Unexpected observation shape: {obs.shape}"
            )

        observations.append(obs)

    reference_observations = np.concatenate(
        observations,
        axis=0,
    )

    std = np.std(
        reference_observations,
        axis=0,
    )

    return np.maximum(
        std,
        1e-8,
    )


def collect_clean_states(policy):
    """
    Collect a clean-policy decision-state pool.

    This is a pilot sampler only. It is intentionally separate from
    the final publication-scale P1 sampler.
    """
    records = []

    for episode_id in range(5):
        env = gym.make(ENV_NAME)

        try:
            obs, _ = env.reset(
                seed=1000 + episode_id
            )

            for step in range(1000):
                state = capture_state(env)
                clean_obs = get_observation(env)

                if not np.all(
                    np.isfinite(clean_obs)
                ):
                    break

                action = np.asarray(
                    policy.predict(
                        clean_obs.reshape(1, -1)
                    )[0],
                    dtype=np.float32,
                )

                if not np.all(
                    np.isfinite(action)
                ):
                    break

                records.append(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "state": state,
                        "observation": clean_obs.copy(),
                    }
                )

                _, _, terminated, truncated, _ = env.step(
                    action
                )

                if terminated or truncated:
                    break

        finally:
            env.close()

    if len(records) < N_STATES:
        raise RuntimeError(
            f"Only {len(records)} valid decision states collected; "
            f"need at least {N_STATES}."
        )

    rng = np.random.default_rng(
        STATE_SAMPLING_SEED
    )

    selected = rng.choice(
        len(records),
        size=N_STATES,
        replace=False,
    )

    return [
        records[int(i)]
        for i in selected
    ]


def action_disagreement(
    clean_action: np.ndarray,
    shifted_action: np.ndarray,
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
            clean_action - shifted_action
        )
        / np.sqrt(
            len(clean_action)
        )
    )


def main():
    print("=" * 80)
    print("P1.2 — DECISION-CONSEQUENCE PILOT")
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

    print(
        "Policy:",
        type(policy).__name__,
    )
    print(
        "Checkpoint:",
        CHECKPOINT,
    )
    print(
        "States:",
        N_STATES,
    )
    print(
        "Sigma:",
        SIGMA,
    )
    print(
        "Horizons:",
        HORIZONS,
    )

    reference_std = load_reference_std()

    print(
        "Reference std shape:",
        reference_std.shape,
    )

    states = collect_clean_states(
        policy
    )

    print(
        "Selected decision states:",
        len(states),
    )

    noise_rng = np.random.default_rng(
        NOISE_SEED
    )

    results = []

    for state_id, record in enumerate(states):
        state = record["state"]

        clean_env = gym.make(ENV_NAME)
        shifted_env = gym.make(ENV_NAME)

        try:
            clean_env.reset(
                seed=4000 + state_id
            )
            shifted_env.reset(
                seed=5000 + state_id
            )

            restore_state(
                clean_env,
                state,
            )
            restore_state(
                shifted_env,
                state,
            )

            validate_state_pair(
                clean_env,
                shifted_env,
            )

            clean_obs = get_observation(
                clean_env
            )

            shifted_obs, standardized_noise = (
                make_shifted_observation(
                    clean_obs,
                    reference_std,
                    SIGMA,
                    noise_rng,
                )
            )

            clean_action = np.asarray(
                policy.predict(
                    clean_obs.reshape(1, -1)
                )[0],
                dtype=np.float64,
            )

            shifted_action = np.asarray(
                policy.predict(
                    shifted_obs.reshape(1, -1)
                )[0],
                dtype=np.float64,
            )

            disagreement = action_disagreement(
                clean_action,
                shifted_action,
            )

            state_result = {
                "state_id": state_id,
                "source_episode_id": int(
                    record["episode_id"]
                ),
                "source_step": int(
                    record["step"]
                ),
                "sigma": SIGMA,
                "standardized_noise": (
                    standardized_noise.tolist()
                ),
                "clean_action": clean_action.tolist(),
                "shifted_action": shifted_action.tolist(),
                "action_disagreement": disagreement,
                "horizons": {},
            }

            for horizon in HORIZONS:
                env_clean_h = gym.make(ENV_NAME)
                env_shifted_h = gym.make(ENV_NAME)

                try:
                    env_clean_h.reset(
                        seed=6000
                        + state_id * 100
                        + horizon
                    )

                    env_shifted_h.reset(
                        seed=7000
                        + state_id * 100
                        + horizon
                    )

                    restore_state(
                        env_clean_h,
                        state,
                    )
                    restore_state(
                        env_shifted_h,
                        state,
                    )

                    validate_state_pair(
                        env_clean_h,
                        env_shifted_h,
                    )

                    clean_outcome, shifted_outcome = (
                        run_counterfactual_pair(
                            env_clean=env_clean_h,
                            env_shifted=env_shifted_h,
                            policy=policy,
                            clean_observation=clean_obs,
                            shifted_observation=shifted_obs,
                            horizon=horizon,
                        )
                    )

                    clean_return = (
                        clean_outcome.cumulative_return
                    )

                    shifted_return = (
                        shifted_outcome.cumulative_return
                    )

                    state_result["horizons"][
                        str(horizon)
                    ] = {
                        "clean_return": float(
                            clean_return
                        ),
                        "shifted_return": float(
                            shifted_return
                        ),
                        "return_degradation": float(
                            clean_return
                            - shifted_return
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

                finally:
                    env_clean_h.close()
                    env_shifted_h.close()

            results.append(
                state_result
            )

        finally:
            clean_env.close()
            shifted_env.close()

    summary = {}

    for horizon in HORIZONS:
        degradation = np.asarray(
            [
                r["horizons"][
                    str(horizon)
                ]["return_degradation"]
                for r in results
            ],
            dtype=np.float64,
        )

        disagreement = np.asarray(
            [
                r["action_disagreement"]
                for r in results
            ],
            dtype=np.float64,
        )

        summary[str(horizon)] = {
            "n_states": int(
                len(degradation)
            ),
            "mean_return_degradation": float(
                np.mean(degradation)
            ),
            "median_return_degradation": float(
                np.median(degradation)
            ),
            "min_return_degradation": float(
                np.min(degradation)
            ),
            "max_return_degradation": float(
                np.max(degradation)
            ),
            "positive_degradation_fraction": float(
                np.mean(degradation > 0.0)
            ),
            "mean_action_disagreement": float(
                np.mean(disagreement)
            ),
            "median_action_disagreement": float(
                np.median(disagreement)
            ),
        }

    output = {
        "experiment": (
            "P1.2-decision-consequence-pilot"
        ),
        "status": "diagnostic_only",
        "task": TASK,
        "environment": ENV_NAME,
        "checkpoint": str(CHECKPOINT),
        "policy_seed": 0,
        "n_states": N_STATES,
        "sigma": SIGMA,
        "horizons": HORIZONS,
        "state_sampling_seed": STATE_SAMPLING_SEED,
        "noise_seed": NOISE_SEED,
        "reference_fraction": REFERENCE_FRACTION,
        "protocol": (
            "experiments/reliability/"
            "P1_DECISION_CONSEQUENCE_PROTOCOL.md"
        ),
        "summary": summary,
        "records": results,
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
    print("P1.2 PILOT COMPLETE")
    print("=" * 80)

    for horizon in HORIZONS:
        s = summary[str(horizon)]

        print(
            f"H={horizon:2d}: "
            f"mean ΔJ={s['mean_return_degradation']:.8f}, "
            f"median ΔJ={s['median_return_degradation']:.8f}, "
            f"positive fraction="
            f"{s['positive_degradation_fraction']:.3f}, "
            f"mean action disagreement="
            f"{s['mean_action_disagreement']:.8f}"
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
