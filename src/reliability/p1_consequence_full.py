"""Full P1 decision-consequence collector and engineering gate."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import d3rlpy
import gymnasium as gym
import numpy as np

from src.reliability.counterfactual import capture_state, get_observation, restore_state, run_counterfactual_pair, validate_state_pair
from src.reliability.critic_disagreement import twin_critic_disagreement, twin_critic_values
from src.reliability.support_distance import build_reference_index, fit_reference_standardization, nearest_support_distance

TASK = "mujoco/hopper/medium-v0"
ENV_NAME = "Hopper-v5"
CHECKPOINTS = {
    0: Path("checkpoints/iql_100k/iql_mujoco_hopper_medium-v0_seed0.d3"),
    1: Path("checkpoints/iql_replications/iql_seed1/iql_mujoco_hopper_medium-v0_seed1.d3"),
    2: Path("checkpoints/iql_replications/iql_seed2/iql_mujoco_hopper_medium-v0_seed2.d3"),
    3: Path("checkpoints/iql_replications/iql_seed3/iql_mujoco_hopper_medium-v0_seed3.d3"),
    4: Path("checkpoints/iql_replications/iql_seed4/iql_mujoco_hopper_medium-v0_seed4.d3"),
}
SIGMA_LEVELS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30]
HORIZONS = [1, 5, 10, 20]
REFERENCE_FRACTION = 0.90
N_EPISODES = 4
STATES_PER_EPISODE = 5
STATE_SAMPLING_SEED = 20260828
NOISE_SEED = 20260829
OUTPUT_DIR = Path("results/reliability/P1_decision_consequence/raw")
PROTOCOL = "experiments/reliability/P1_DECISION_CONSEQUENCE_PROTOCOL.md"


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _reference():
    dataset, _ = d3rlpy.datasets.get_minari(TASK)
    episodes = list(dataset.episodes)
    rng = np.random.default_rng(STATE_SAMPLING_SEED)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)
    split = int(len(indices) * REFERENCE_FRACTION)
    observations = np.concatenate([np.asarray(episodes[int(i)].observations, dtype=np.float64) for i in indices[:split]], axis=0)
    mean, std = fit_reference_standardization(observations)
    return mean, std, build_reference_index((observations - mean) / std)


def _states(policy):
    rng = np.random.default_rng(STATE_SAMPLING_SEED)
    selected = []
    for episode_id in range(N_EPISODES):
        env = gym.make(ENV_NAME)
        eligible = []
        try:
            env.reset(seed=10000 + episode_id)
            for step in range(1000):
                observation = get_observation(env)
                if not np.all(np.isfinite(observation)):
                    break
                state = capture_state(env)
                action = np.asarray(policy.predict(observation.reshape(1, -1))[0], dtype=np.float32)
                if not np.all(np.isfinite(action)):
                    break
                eligible.append({"episode_id": episode_id, "step": step, "state": state})
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        if len(eligible) < STATES_PER_EPISODE:
            raise RuntimeError(f"Episode {episode_id} has too few valid states")
        selected.extend(eligible[int(i)] for i in rng.choice(len(eligible), STATES_PER_EPISODE, replace=False))
    return selected


def _action_gap(clean, shifted):
    return float(np.linalg.norm(np.asarray(clean) - np.asarray(shifted)) / np.sqrt(len(clean)))


def collect_seed(policy_seed, checkpoint):
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    policy = d3rlpy.load_learnable(str(checkpoint), device="cpu")
    mean, std, reference_index = _reference()
    states = _states(policy)
    noise_rng = np.random.default_rng(NOISE_SEED)
    records = []
    for state_id, source in enumerate(states):
        clean_env = gym.make(ENV_NAME)
        try:
            clean_env.reset(seed=20000 + state_id)
            restore_state(clean_env, source["state"])
            clean_obs = get_observation(clean_env)
            clean_action = np.asarray(policy.predict(clean_obs.reshape(1, -1))[0], dtype=np.float64)
            support, nearest = nearest_support_distance(clean_obs, mean, std, reference_index)
            q1, q2 = twin_critic_values(policy, clean_obs, clean_action)
            critic_gap = twin_critic_disagreement(q1, q2)
        finally:
            clean_env.close()
        standard_normal = noise_rng.normal(size=clean_obs.shape)
        for sigma in SIGMA_LEVELS:
            shifted_obs = clean_obs + standard_normal * sigma * std
            shifted_action = np.asarray(policy.predict(shifted_obs.reshape(1, -1))[0], dtype=np.float64)
            for horizon in HORIZONS:
                clean_branch = gym.make(ENV_NAME)
                shifted_branch = gym.make(ENV_NAME)
                try:
                    clean_branch.reset(seed=40000 + state_id)
                    shifted_branch.reset(seed=30000 + state_id)
                    restore_state(clean_branch, source["state"])
                    restore_state(shifted_branch, source["state"])
                    validate_state_pair(clean_branch, shifted_branch)
                    clean_out, shifted_out = run_counterfactual_pair(clean_branch, shifted_branch, policy, clean_obs, shifted_obs, horizon)
                    clean_return = float(clean_out.cumulative_return)
                    shifted_return = float(shifted_out.cumulative_return)
                    delta = clean_return - shifted_return
                    records.append({
                        "policy_seed": policy_seed, "state_id": state_id,
                        "source_episode_id": int(source["episode_id"]), "source_step": int(source["step"]),
                        "sigma": float(sigma), "horizon": horizon,
                        "standardized_noise": (standard_normal * sigma).tolist(),
                        "clean_action": clean_action.tolist(), "shifted_action": shifted_action.tolist(),
                        "action_disagreement": _action_gap(clean_action, shifted_action),
                        "support_distance": float(support), "nearest_reference_index": int(nearest),
                        "q1_clean": float(q1), "q2_clean": float(q2),
                        "critic_disagreement": float(critic_gap), "twin_critic_disagreement": float(critic_gap),
                        "clean_return": clean_return, "shifted_return": shifted_return, "delta_J": float(delta),
                        "absolute_consequence": float(abs(delta)),
                        "relative_consequence": float(abs(delta) / (1e-8 + 0.5 * (abs(clean_return) + abs(shifted_return)))),
                        "clean_steps": len(clean_out.rewards), "shifted_steps": len(shifted_out.rewards),
                        "clean_terminated": bool(np.any(clean_out.terminated)), "shifted_terminated": bool(np.any(shifted_out.terminated)),
                        "clean_truncated": bool(np.any(clean_out.truncated)), "shifted_truncated": bool(np.any(shifted_out.truncated)),
                    })
                finally:
                    clean_branch.close()
                    shifted_branch.close()
    output = {"experiment": "P1_decision_consequence_full", "status": "engineering_gate", "task": TASK, "environment": ENV_NAME,
              "policy_seed": policy_seed, "checkpoint": str(checkpoint), "sigma_levels": SIGMA_LEVELS, "horizons": HORIZONS,
              "primary_horizon": 10, "n_episodes": N_EPISODES, "states_per_episode": STATES_PER_EPISODE, "n_states": len(states),
              "state_sampling_seed": STATE_SAMPLING_SEED, "noise_seed": NOISE_SEED, "reference_fraction": REFERENCE_FRACTION,
              "protocol": PROTOCOL, "git_commit": _git_commit(), "records": records}
    output_path = OUTPUT_DIR / f"seed{policy_seed}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"saved {output_path} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-seed", type=int, choices=sorted(CHECKPOINTS), default=0)
    args = parser.parse_args()
    collect_seed(args.policy_seed, CHECKPOINTS[args.policy_seed])


if __name__ == "__main__":
    main()
