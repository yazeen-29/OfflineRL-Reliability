from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class SimulatorState:
    """Exact MuJoCo state at a decision point."""
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class BranchOutcome:
    """Observed outcome of one counterfactual branch."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def cumulative_return(self) -> float:
        return float(np.sum(self.rewards))


def capture_state(env: gym.Env) -> SimulatorState:
    """Capture an exact MuJoCo state without retaining mutable views."""
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "data"):
        raise TypeError(
            "Environment does not expose MuJoCo data."
        )

    return SimulatorState(
        qpos=np.array(
            unwrapped.data.qpos,
            dtype=np.float64,
            copy=True,
        ),
        qvel=np.array(
            unwrapped.data.qvel,
            dtype=np.float64,
            copy=True,
        ),
    )


def restore_state(
    env: gym.Env,
    state: SimulatorState,
) -> None:
    """Restore an exact MuJoCo qpos/qvel state."""
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "set_state"):
        raise TypeError(
            "Environment does not expose set_state(qpos, qvel)."
        )

    unwrapped.set_state(
        np.array(
            state.qpos,
            dtype=np.float64,
            copy=True,
        ),
        np.array(
            state.qvel,
            dtype=np.float64,
            copy=True,
        ),
    )


def get_observation(env: gym.Env) -> np.ndarray:
    """Read the environment observation without changing simulator state."""
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "_get_obs"):
        raise TypeError(
            "Environment does not expose _get_obs()."
        )

    return np.asarray(
        unwrapped._get_obs(),
        dtype=np.float64,
    ).copy()


def make_shifted_observation(
    clean_observation: np.ndarray,
    reference_std: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample Gaussian noise in reference-standardized coordinates.

    Returns:
        shifted_observation,
        standardized_noise
    """
    clean_observation = np.asarray(
        clean_observation,
        dtype=np.float64,
    )

    reference_std = np.asarray(
        reference_std,
        dtype=np.float64,
    )

    if clean_observation.ndim != 1:
        raise ValueError(
            "clean_observation must be one-dimensional."
        )

    if reference_std.shape != clean_observation.shape:
        raise ValueError(
            "reference_std must match observation shape."
        )

    if not np.all(np.isfinite(clean_observation)):
        raise ValueError(
            "clean_observation contains non-finite values."
        )

    if not np.all(np.isfinite(reference_std)):
        raise ValueError(
            "reference_std contains non-finite values."
        )

    if np.any(reference_std <= 0.0):
        raise ValueError(
            "reference_std must be strictly positive."
        )

    if sigma < 0.0:
        raise ValueError(
            "sigma must be non-negative."
        )

    standardized_noise = rng.normal(
        loc=0.0,
        scale=float(sigma),
        size=clean_observation.shape,
    )

    shifted_observation = (
        clean_observation
        + standardized_noise * reference_std
    )

    return (
        shifted_observation,
        standardized_noise,
    )


def validate_state_pair(
    env_a: gym.Env,
    env_b: gym.Env,
    atol: float = 1e-12,
) -> None:
    """Assert that two MuJoCo environments have identical physical state."""
    qpos_a = np.asarray(
        env_a.unwrapped.data.qpos,
        dtype=np.float64,
    )
    qpos_b = np.asarray(
        env_b.unwrapped.data.qpos,
        dtype=np.float64,
    )

    qvel_a = np.asarray(
        env_a.unwrapped.data.qvel,
        dtype=np.float64,
    )
    qvel_b = np.asarray(
        env_b.unwrapped.data.qvel,
        dtype=np.float64,
    )

    if not np.allclose(
        qpos_a,
        qpos_b,
        atol=atol,
        rtol=0.0,
    ):
        raise AssertionError(
            "MuJoCo qpos differs between counterfactual branches."
        )

    if not np.allclose(
        qvel_a,
        qvel_b,
        atol=atol,
        rtol=0.0,
    ):
        raise AssertionError(
            "MuJoCo qvel differs between counterfactual branches."
        )


def policy_action(
    policy,
    observation: np.ndarray,
) -> np.ndarray:
    """Evaluate a deterministic policy action."""
    observation = np.asarray(
        observation,
        dtype=np.float64,
    )

    if observation.ndim != 1:
        raise ValueError(
            "observation must be one-dimensional."
        )

    action = policy.predict(
        observation.reshape(1, -1)
    )[0]

    action = np.asarray(
        action,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(action)):
        raise ValueError(
            "Policy produced a non-finite action."
        )

    return action


def execute_one_action(
    env: gym.Env,
    action: np.ndarray,
) -> Tuple[np.ndarray, float, bool, bool]:
    """Execute exactly one action."""
    action = np.asarray(
        action,
        dtype=np.float32,
    )

    next_obs, reward, terminated, truncated, _ = env.step(
        action
    )

    return (
        np.asarray(
            next_obs,
            dtype=np.float64,
        ).copy(),
        float(reward),
        bool(terminated),
        bool(truncated),
    )


def continue_clean_policy(
    env: gym.Env,
    policy,
    first_observation: np.ndarray,
    horizon: int,
) -> BranchOutcome:
    """
    Continue a branch using normal simulator observations after the
    intervention step.

    This function is intended for steps AFTER the first counterfactual
    action has already been executed.
    """
    if horizon < 0:
        raise ValueError(
            "horizon must be non-negative."
        )

    observations = []
    actions = []
    rewards = []
    terminated = []
    truncated = []

    current_observation = np.asarray(
        first_observation,
        dtype=np.float64,
    ).copy()

    for _ in range(horizon):
        action = policy_action(
            policy,
            current_observation,
        )

        observations.append(
            current_observation.copy()
        )

        actions.append(
            action.copy()
        )

        next_observation, reward, done, trunc = (
            execute_one_action(
                env,
                action,
            )
        )

        rewards.append(reward)
        terminated.append(done)
        truncated.append(trunc)

        current_observation = next_observation

        if done or trunc:
            break

    return BranchOutcome(
        observations=np.asarray(
            observations,
            dtype=np.float64,
        ),
        actions=np.asarray(
            actions,
            dtype=np.float64,
        ),
        rewards=np.asarray(
            rewards,
            dtype=np.float64,
        ),
        terminated=np.asarray(
            terminated,
            dtype=bool,
        ),
        truncated=np.asarray(
            truncated,
            dtype=bool,
        ),
    )


def run_counterfactual_pair(
    env_clean: gym.Env,
    env_shifted: gym.Env,
    policy,
    clean_observation: np.ndarray,
    shifted_observation: np.ndarray,
    horizon: int,
) -> Tuple[BranchOutcome, BranchOutcome]:
    """
    Execute the paired counterfactual intervention.

    At step 0:
        clean branch receives clean_observation
        shifted branch receives shifted_observation

    After step 0:
        both branches receive their own normal simulator observations.

    Thus the observation perturbation occurs only at the intervention
    decision. Subsequent divergence is an outcome of that intervention.
    """
    if horizon < 1:
        raise ValueError(
            "horizon must be >= 1."
        )

    clean_observation = np.asarray(
        clean_observation,
        dtype=np.float64,
    )

    shifted_observation = np.asarray(
        shifted_observation,
        dtype=np.float64,
    )

    if clean_observation.shape != shifted_observation.shape:
        raise ValueError(
            "Clean and shifted observations must have identical shapes."
        )

    validate_state_pair(
        env_clean,
        env_shifted,
    )

    clean_action = policy_action(
        policy,
        clean_observation,
    )

    shifted_action = policy_action(
        policy,
        shifted_observation,
    )

    # First counterfactual step.
    clean_next_obs, clean_reward, clean_done, clean_trunc = (
        execute_one_action(
            env_clean,
            clean_action,
        )
    )

    shifted_next_obs, shifted_reward, shifted_done, shifted_trunc = (
        execute_one_action(
            env_shifted,
            shifted_action,
        )
    )

    clean_observations = [clean_observation.copy()]
    shifted_observations = [shifted_observation.copy()]

    clean_actions = [clean_action.copy()]
    shifted_actions = [shifted_action.copy()]

    clean_rewards = [clean_reward]
    shifted_rewards = [shifted_reward]

    clean_terminated = [clean_done]
    shifted_terminated = [shifted_done]

    clean_truncated = [clean_trunc]
    shifted_truncated = [shifted_trunc]

    # Continue both branches normally.
    remaining = horizon - 1

    if remaining > 0 and not (clean_done or clean_trunc):
        clean_tail = continue_clean_policy(
            env_clean,
            policy,
            clean_next_obs,
            remaining,
        )

        if len(clean_tail.observations):
            clean_observations.extend(
                clean_tail.observations
            )
            clean_actions.extend(
                clean_tail.actions
            )
            clean_rewards.extend(
                clean_tail.rewards
            )
            clean_terminated.extend(
                clean_tail.terminated
            )
            clean_truncated.extend(
                clean_tail.truncated
            )

    if remaining > 0 and not (shifted_done or shifted_trunc):
        shifted_tail = continue_clean_policy(
            env_shifted,
            policy,
            shifted_next_obs,
            remaining,
        )

        if len(shifted_tail.observations):
            shifted_observations.extend(
                shifted_tail.observations
            )
            shifted_actions.extend(
                shifted_tail.actions
            )
            shifted_rewards.extend(
                shifted_tail.rewards
            )
            shifted_terminated.extend(
                shifted_tail.terminated
            )
            shifted_truncated.extend(
                shifted_tail.truncated
            )

    clean_outcome = BranchOutcome(
        observations=np.asarray(
            clean_observations,
            dtype=np.float64,
        ),
        actions=np.asarray(
            clean_actions,
            dtype=np.float64,
        ),
        rewards=np.asarray(
            clean_rewards,
            dtype=np.float64,
        ),
        terminated=np.asarray(
            clean_terminated,
            dtype=bool,
        ),
        truncated=np.asarray(
            clean_truncated,
            dtype=bool,
        ),
    )

    shifted_outcome = BranchOutcome(
        observations=np.asarray(
            shifted_observations,
            dtype=np.float64,
        ),
        actions=np.asarray(
            shifted_actions,
            dtype=np.float64,
        ),
        rewards=np.asarray(
            shifted_rewards,
            dtype=np.float64,
        ),
        terminated=np.asarray(
            shifted_terminated,
            dtype=bool,
        ),
        truncated=np.asarray(
            shifted_truncated,
            dtype=bool,
        ),
    )

    return clean_outcome, shifted_outcome
