from __future__ import annotations

import numpy as np
import torch


def _to_numpy_float32(value) -> np.ndarray:
    """Convert array-like data to a finite float32 NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = np.asarray(
        value,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(value)):
        raise ValueError(
            "Input contains non-finite values."
        )

    return value


def _scale_observation(
    policy,
    observation: np.ndarray,
) -> np.ndarray:
    """Apply the loaded d3rlpy observation scaler."""
    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(1, -1)

    scaler = policy.observation_scaler

    if scaler is None:
        return observation

    # StandardObservationScaler in this environment stores NumPy
    # statistics, so its normal transform path is valid here.
    try:
        transformed = scaler.transform(
            observation
        )
    except TypeError as exc:
        # Defensive fallback for d3rlpy objects whose internal
        # statistics have become tensors.
        mean = getattr(
            scaler,
            "mean",
            None,
        )
        std = getattr(
            scaler,
            "std",
            None,
        )
        eps = getattr(
            scaler,
            "eps",
            1e-6,
        )

        if mean is None or std is None:
            raise exc

        if isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().numpy()

        if isinstance(std, torch.Tensor):
            std = std.detach().cpu().numpy()

        transformed = (
            observation - np.asarray(
                mean,
                dtype=np.float32,
            )
        ) / (
            np.asarray(
                std,
                dtype=np.float32,
            )
            + float(eps)
        )

    return _to_numpy_float32(
        transformed
    )


def _scale_action(
    policy,
    action: np.ndarray,
) -> np.ndarray:
    """
    Apply the loaded d3rlpy action scaler.

    MinMaxActionScaler operates on NumPy arrays in the public API.
    """
    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(1, -1)

    scaler = policy.action_scaler

    if scaler is None:
        return action

    try:
        transformed = scaler.transform(
            action
        )
    except TypeError:
        # Defensive fallback for tensor-backed scaler implementations.
        transformed = scaler.transform(
            torch.as_tensor(
                action,
                dtype=torch.float32,
            )
        )

    return _to_numpy_float32(
        transformed
    )


def twin_critic_values(
    policy,
    observation: np.ndarray,
    action: np.ndarray,
) -> tuple[float, float]:
    """
    Return the two IQL critic values for one observation/action pair.

    This is a twin-critic disagreement baseline, not a calibrated
    epistemic uncertainty estimator.
    """
    observation = np.asarray(
        observation,
        dtype=np.float64,
    )

    action = np.asarray(
        action,
        dtype=np.float64,
    )

    if observation.ndim != 1:
        raise ValueError(
            "observation must be one-dimensional."
        )

    if action.ndim != 1:
        raise ValueError(
            "action must be one-dimensional."
        )

    if not np.all(
        np.isfinite(observation)
    ):
        raise ValueError(
            "observation contains non-finite values."
        )

    if not np.all(
        np.isfinite(action)
    ):
        raise ValueError(
            "action contains non-finite values."
        )

    impl = policy._impl
    critics = impl.q_function

    if len(critics) != 2:
        raise RuntimeError(
            "Expected exactly two IQL critics; "
            f"found {len(critics)}."
        )

    obs_scaled = _scale_observation(
        policy,
        observation,
    )

    action_scaled = _scale_action(
        policy,
        action,
    )

    # Match the loaded critic's actual device.
    try:
        device = next(
            critics[0].parameters()
        ).device
    except StopIteration:
        device = torch.device("cpu")

    obs_tensor = torch.as_tensor(
        obs_scaled,
        dtype=torch.float32,
        device=device,
    )

    action_tensor = torch.as_tensor(
        action_scaled,
        dtype=torch.float32,
        device=device,
    )

    values = []

    with torch.no_grad():
        for critic in critics:
            output = critic(
                obs_tensor,
                action_tensor,
            )

            # d3rlpy 2.8.1 ContinuousMeanQFunction returns a
            # QFunctionOutput. In this implementation the field is
            # expected to be q_value, but keep value as a fallback.
            if hasattr(
                output,
                "q_value",
            ):
                raw_value = output.q_value

            elif hasattr(
                output,
                "value",
            ):
                raw_value = output.value

            else:
                raise RuntimeError(
                    "Unsupported QFunctionOutput structure: "
                    f"{type(output)}"
                )

            if isinstance(
                raw_value,
                torch.Tensor,
            ):
                raw_value = (
                    raw_value.detach()
                    .cpu()
                    .numpy()
                )

            value = float(
                np.asarray(
                    raw_value,
                    dtype=np.float64,
                ).reshape(-1)[0]
            )

            if not np.isfinite(value):
                raise ValueError(
                    "Critic returned a non-finite Q-value."
                )

            values.append(
                value
            )

    return (
        values[0],
        values[1],
    )


def twin_critic_disagreement(
    q1: float,
    q2: float,
) -> float:
    """
    Absolute difference between the two IQL critic estimates.
    """
    q1 = float(q1)
    q2 = float(q2)

    if not np.isfinite(q1):
        raise ValueError(
            "q1 must be finite."
        )

    if not np.isfinite(q2):
        raise ValueError(
            "q2 must be finite."
        )

    return float(
        abs(q1 - q2)
    )
