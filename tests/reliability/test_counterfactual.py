from __future__ import annotations

import numpy as np
import gymnasium as gym
import d3rlpy

from src.reliability.counterfactual import (
    capture_state,
    get_observation,
    restore_state,
    run_counterfactual_pair,
    validate_state_pair,
)


CHECKPOINT = (
    "checkpoints/iql_100k/"
    "iql_mujoco_hopper_medium-v0_seed0.d3"
)


def main():
    print("=" * 80)
    print("P1.1 — COUNTERFACTUAL PAIR UNIT TEST")
    print("=" * 80)

    policy = d3rlpy.load_learnable(
        CHECKPOINT,
        device="cpu",
    )

    env_source = gym.make("Hopper-v5")
    env_clean = gym.make("Hopper-v5")
    env_shifted = gym.make("Hopper-v5")

    try:
        # ----------------------------------------------------------
        # Capture one exact physical state.
        # ----------------------------------------------------------
        env_source.reset(seed=123)

        state = capture_state(
            env_source
        )

        # ----------------------------------------------------------
        # Restore exactly the same physical state.
        # ----------------------------------------------------------
        env_clean.reset(seed=456)
        env_shifted.reset(seed=789)

        restore_state(
            env_clean,
            state,
        )

        restore_state(
            env_shifted,
            state,
        )

        validate_state_pair(
            env_clean,
            env_shifted,
        )

        print("✅ Initial physical states identical")

        # ----------------------------------------------------------
        # Clean observation.
        # ----------------------------------------------------------
        clean_obs = get_observation(
            env_clean
        )

        shifted_obs = clean_obs.copy()

        # Deterministic test perturbation.
        shifted_obs[0] += 0.25

        if np.allclose(
            clean_obs,
            shifted_obs,
            atol=1e-12,
            rtol=0.0,
        ):
            raise AssertionError(
                "Test perturbation failed to change observation."
            )

        print("✅ Clean and shifted observations differ")

        # ----------------------------------------------------------
        # Physical simulator state must still match BEFORE
        # intervention.
        # ----------------------------------------------------------
        validate_state_pair(
            env_clean,
            env_shifted,
        )

        print(
            "✅ Physical states remain identical before intervention"
        )

        # ----------------------------------------------------------
        # Run paired counterfactual.
        # ----------------------------------------------------------
        clean_outcome, shifted_outcome = (
            run_counterfactual_pair(
                env_clean=env_clean,
                env_shifted=env_shifted,
                policy=policy,
                clean_observation=clean_obs,
                shifted_observation=shifted_obs,
                horizon=10,
            )
        )

        # ----------------------------------------------------------
        # Basic output validation.
        # ----------------------------------------------------------
        if len(clean_outcome.rewards) < 1:
            raise AssertionError(
                "Clean branch produced no reward."
            )

        if len(shifted_outcome.rewards) < 1:
            raise AssertionError(
                "Shifted branch produced no reward."
            )

        if len(clean_outcome.rewards) > 10:
            raise AssertionError(
                "Clean branch exceeded H=10."
            )

        if len(shifted_outcome.rewards) > 10:
            raise AssertionError(
                "Shifted branch exceeded H=10."
            )

        print(
            "✅ Both counterfactual branches executed"
        )

        # ----------------------------------------------------------
        # First-step action comparison.
        # ----------------------------------------------------------
        clean_action = clean_outcome.actions[0]
        shifted_action = shifted_outcome.actions[0]

        action_disagreement = float(
            np.linalg.norm(
                clean_action - shifted_action
            )
            / np.sqrt(len(clean_action))
        )

        print(
            "\nFirst-step clean action:",
            clean_action,
        )

        print(
            "First-step shifted action:",
            shifted_action,
        )

        print(
            "First-step action disagreement:",
            action_disagreement,
        )

        # This is diagnostic, not a hard scientific criterion.
        if action_disagreement == 0.0:
            print(
                "⚠️ Deterministic perturbation produced zero action "
                "difference for this state."
            )
        else:
            print(
                "✅ Perturbed observation changed the policy action"
            )

        # ----------------------------------------------------------
        # Outcome comparison.
        # ----------------------------------------------------------
        clean_return = (
            clean_outcome.cumulative_return
        )

        shifted_return = (
            shifted_outcome.cumulative_return
        )

        degradation = (
            clean_return - shifted_return
        )

        print(
            "\nClean H-step return:",
            clean_return,
        )

        print(
            "Shifted H-step return:",
            shifted_return,
        )

        print(
            "H-step return degradation:",
            degradation,
        )

        # ----------------------------------------------------------
        # Final state diagnostics.
        # ----------------------------------------------------------
        print(
            "\nClean branch steps:",
            len(clean_outcome.rewards),
        )

        print(
            "Shifted branch steps:",
            len(shifted_outcome.rewards),
        )

        print("\n" + "=" * 80)
        print(
            "✅ P1.1 COUNTERFACTUAL UNIT TEST PASSED"
        )
        print("=" * 80)

    finally:
        env_source.close()
        env_clean.close()
        env_shifted.close()


if __name__ == "__main__":
    main()
