import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import d3rlpy


# ---------------------------------------------------------------------
# Make src/ importable when running:
# python src/training/train_and_verify.py
# ---------------------------------------------------------------------

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from utils.policy_io import (
    build_iql,
    build_cql,
    load_policy,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_git_commit():
    """Return the current Git commit if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def get_git_status():
    """Return whether the repository has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )

        return "CLEAN" if not result.stdout.strip() else "DIRTY"

    except Exception:
        return "UNKNOWN"


def create_experiment_id(algo, seed):
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    return f"EXP-IQL-{algo.upper()}-S{seed}-{timestamp}"


def save_json(path, data):
    """Save JSON metadata/results."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )


def get_environment_metadata():
    """Collect reproducibility information."""
    metadata = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "python_version": platform.python_version(),

        "platform": platform.platform(),

        "machine": platform.machine(),

        "processor": platform.processor(),

        "pytorch_version": torch.__version__,

        "numpy_version": np.__version__,

        "d3rlpy_version": d3rlpy.__version__,

        "cuda_available": torch.cuda.is_available(),

        "cuda_version": torch.version.cuda,

    }

    if torch.cuda.is_available():
        metadata["gpu_count"] = torch.cuda.device_count()

        metadata["gpu_name"] = (
            torch.cuda.get_device_name(0)
        )
    else:
        metadata["gpu_count"] = 0
        metadata["gpu_name"] = None

    return metadata


# ---------------------------------------------------------------------
# Policy creation
# ---------------------------------------------------------------------

def make_policy(algo, device):
    """Create the requested offline RL algorithm."""

    if algo == "iql":
        return build_iql(
            device=device
        )

    if algo == "cql":
        return build_cql(
            device=device
        )

    raise ValueError(
        f"Unsupported algorithm: {algo}"
    )


# ---------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------

def verify(
    policy,
    env,
    seed,
    label,
    max_steps=1200,
):
    """
    Run one episode and return its total reward.
    """

    observation, _ = env.reset(
        seed=seed
    )

    total_reward = 0.0

    for step in range(max_steps):

        action = policy.predict(
            np.asarray([observation])
        )[0]

        observation, reward, terminated, truncated, _ = (
            env.step(action)
        )

        total_reward += float(reward)

        if terminated or truncated:

            print(
                f"[{label}] "
                f"Episode ended at step "
                f"{step + 1}, "
                f"return={total_reward:.3f}"
            )

            return total_reward

    print(
        f"[{label}] "
        f"Survived {max_steps} steps, "
        f"return={total_reward:.3f}"
    )

    return total_reward


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train, save, reload and verify "
            "an offline RL policy."
        )
    )

    parser.add_argument(
        "--algo",
        choices=["iql", "cql"],
        default="iql",
    )

    parser.add_argument(
        "--task",
        default="mujoco/hopper/medium-v0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--out_dir",
        default="/kaggle/working/checkpoints",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Experiment identity
    # -----------------------------------------------------------------

    experiment_id = create_experiment_id(
        args.algo,
        args.seed,
    )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    results_dir = os.path.join(
        "results",
        "verification",
    )

    logs_dir = os.path.join(
        "logs",
        "verification",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    os.makedirs(
        logs_dir,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print("=" * 70)
    print("OFFLINE RL TRAIN + SAVE + VERIFY")
    print("=" * 70)

    print(f"Experiment: {experiment_id}")
    print(f"Algorithm : {args.algo}")
    print(f"Task      : {args.task}")
    print(f"Seed      : {args.seed}")
    print(f"Steps     : {args.n_steps}")
    print(f"Device    : {args.device}")
    print(f"Output    : {args.out_dir}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Environment metadata
    # -----------------------------------------------------------------

    metadata = get_environment_metadata()

    metadata.update(
        {
            "experiment_id": experiment_id,
            "algorithm": args.algo,
            "task": args.task,
            "seed": args.seed,
            "training_steps": args.n_steps,
            "device": args.device,
            "git_commit": get_git_commit(),
            "git_status": get_git_status(),
        }
    )

    metadata_path = os.path.join(
        results_dir,
        f"{experiment_id}_metadata.json",
    )

    save_json(
        metadata_path,
        metadata,
    )

    print(
        f"[OK] Metadata saved: "
        f"{metadata_path}"
    )

    # -----------------------------------------------------------------
    # Seed
    # -----------------------------------------------------------------

    d3rlpy.seed(
        args.seed
    )

    print(
        "[OK] Random seed configured"
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    print(
        "\nLoading Minari dataset..."
    )

    dataset, env = (
        d3rlpy.datasets.get_minari(
            args.task
        )
    )

    print(
        f"[OK] Dataset loaded: "
        f"{dataset.size()} episodes"
    )

    print(
        f"[OK] Environment: {env}"
    )

    metadata[
        "dataset_episodes"
    ] = dataset.size()

    metadata[
        "environment"
    ] = str(env)

    save_json(
        metadata_path,
        metadata,
    )

    # -----------------------------------------------------------------
    # Create policy
    # -----------------------------------------------------------------

    print(
        f"\nCreating "
        f"{args.algo.upper()} policy..."
    )

    policy = make_policy(
        args.algo,
        args.device,
    )

    print(
        f"[OK] Policy created: "
        f"{type(policy).__name__}"
    )

    # -----------------------------------------------------------------
    # Evaluator
    # -----------------------------------------------------------------

    evaluator = (
        d3rlpy.metrics.EnvironmentEvaluator(
            env,
            n_trials=10,
        )
    )

    print(
        "[OK] Environment evaluator created"
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    experiment_name = (
        f"{args.algo}_"
        f"{args.task.replace('/', '_')}_"
        f"seed{args.seed}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 70
    )

    policy.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=1000,
        save_interval=10,
        experiment_name=experiment_name,
        evaluators={
            "environment": evaluator
        },
        show_progress=True,
    )

    print(
        "\n[OK] Training completed"
    )

    # -----------------------------------------------------------------
    # Save COMPLETE d3rlpy checkpoint
    # -----------------------------------------------------------------

    checkpoint_filename = (
        f"{args.algo}_"
        f"{args.task.replace('/', '_')}_"
        f"seed{args.seed}.d3"
    )

    checkpoint_path = os.path.join(
        args.out_dir,
        checkpoint_filename,
    )

    print(
        "\nSaving COMPLETE d3rlpy model to:"
    )

    print(
        checkpoint_path
    )

    # IMPORTANT:
    # save() = complete model + serialized config
    # save_model() = parameters only
    policy.save(
        checkpoint_path
    )

    # -----------------------------------------------------------------
    # Verify checkpoint file
    # -----------------------------------------------------------------

    if not os.path.isfile(
        checkpoint_path
    ):
        raise RuntimeError(
            "CHECKPOINT SAVE FAILED"
        )

    checkpoint_size = (
        os.path.getsize(
            checkpoint_path
        )
    )

    if checkpoint_size <= 0:
        raise RuntimeError(
            "CHECKPOINT EXISTS BUT IS EMPTY"
        )

    print(
        f"[OK] Complete checkpoint exists "
        f"({checkpoint_size / 1024 / 1024:.2f} MB)"
    )

    metadata[
        "checkpoint_path"
    ] = checkpoint_path

    metadata[
        "checkpoint_size_bytes"
    ] = checkpoint_size

    metadata[
        "checkpoint_format"
    ] = "d3rlpy_full_model"

    save_json(
        metadata_path,
        metadata,
    )

    # -----------------------------------------------------------------
    # In-memory verification
    # -----------------------------------------------------------------

    print(
        "\nRunning in-memory verification..."
    )

    reward_memory = verify(
        policy,
        env,
        args.seed,
        "IN-MEMORY",
    )

    # -----------------------------------------------------------------
    # Reload COMPLETE checkpoint
    # -----------------------------------------------------------------

    print(
        "\nLoading saved checkpoint..."
    )

    # IMPORTANT:
    # load_learnable reconstructs the complete
    # algorithm from the .d3 file.
    policy_reloaded = load_policy(
        checkpoint_path,
        device=args.device,
    )

    print(
        f"[OK] Checkpoint reloaded as "
        f"{type(policy_reloaded).__name__}"
    )

    # -----------------------------------------------------------------
    # Reloaded verification
    # -----------------------------------------------------------------

    print(
        "\nRunning reloaded-policy verification..."
    )

    reward_reloaded = verify(
        policy_reloaded,
        env,
        args.seed,
        "RELOADED",
    )

    # -----------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------

    difference = abs(
        reward_memory -
        reward_reloaded
    )

    tolerance = 50.0

    consistent = (
        difference < tolerance
    )

    # -----------------------------------------------------------------
    # Final results
    # -----------------------------------------------------------------

    results = {
        "experiment_id": experiment_id,
        "algorithm": args.algo,
        "task": args.task,
        "seed": args.seed,
        "training_steps": args.n_steps,
        "device": args.device,
        "dataset_episodes": dataset.size(),
        "checkpoint_path": checkpoint_path,
        "checkpoint_size_bytes": checkpoint_size,
        "in_memory_return": reward_memory,
        "reloaded_return": reward_reloaded,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "verification": (
            "CONSISTENT"
            if consistent
            else "MISMATCH"
        ),
        "git_commit": get_git_commit(),
        "git_status": get_git_status(),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    results_path = os.path.join(
        results_dir,
        f"{experiment_id}_results.json",
    )

    save_json(
        results_path,
        results,
    )

    # -----------------------------------------------------------------
    # Human-readable log
    # -----------------------------------------------------------------

    log_path = os.path.join(
        logs_dir,
        f"{experiment_id}.log",
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log:

        log.write(
            "OFFLINE RL CHECKPOINT VERIFICATION\n"
        )

        log.write(
            "=" * 60 + "\n"
        )

        for key, value in results.items():

            log.write(
                f"{key}: {value}\n"
            )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CHECKPOINT VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        f"In-memory return : "
        f"{reward_memory:.3f}"
    )

    print(
        f"Reloaded return  : "
        f"{reward_reloaded:.3f}"
    )

    print(
        f"Absolute difference: "
        f"{difference:.3f}"
    )

    print(
        f"Tolerance: "
        f"{tolerance:.3f}"
    )

    print(
        f"\nResults saved: "
        f"{results_path}"
    )

    print(
        f"Log saved: "
        f"{log_path}"
    )

    if consistent:

        print(
            "\nVERIFICATION: CONSISTENT"
        )

        print(
            "The complete d3rlpy checkpoint "
            "was successfully saved, reloaded, "
            "and executed."
        )

        print(
            "\nPHASE 2 COMPLETE"
        )

    else:

        print(
            "\nVERIFICATION: MISMATCH"
        )

        print(
            "DO NOT TRUST THIS CHECKPOINT."
        )

        sys.exit(1)

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()