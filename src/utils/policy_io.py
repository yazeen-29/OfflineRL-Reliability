import d3rlpy


def build_iql(
    observation_scaler=True,
    action_scaler=True,
    device="cpu",
):
    return d3rlpy.algos.IQLConfig(
        observation_scaler=(
            d3rlpy.preprocessing.StandardObservationScaler()
            if observation_scaler
            else None
        ),
        action_scaler=(
            d3rlpy.preprocessing.MinMaxActionScaler()
            if action_scaler
            else None
        ),
    ).create(device=device)


def build_cql(
    observation_scaler=True,
    action_scaler=True,
    device="cpu",
    initial_alpha=1.0,
    alpha_learning_rate=1e-4,
):
    return d3rlpy.algos.CQLConfig(
        observation_scaler=(
            d3rlpy.preprocessing.StandardObservationScaler()
            if observation_scaler
            else None
        ),
        action_scaler=(
            d3rlpy.preprocessing.MinMaxActionScaler()
            if action_scaler
            else None
        ),
        initial_alpha=initial_alpha,
        alpha_learning_rate=alpha_learning_rate,
    ).create(device=device)


def load_policy(checkpoint_path, device="cpu"):
    """
    Load a complete d3rlpy .d3 checkpoint.

    The checkpoint must have been created with policy.save().
    """
    return d3rlpy.load_learnable(
        checkpoint_path,
        device=device,
    )