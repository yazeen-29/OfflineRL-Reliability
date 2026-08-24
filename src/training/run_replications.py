from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def save_status(
    path: Path,
    status: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            status,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run verified IQL training sequentially "
            "for multiple random seeds."
        )
    )

    parser.add_argument(
        "--task",
        default="mujoco/hopper/medium-v0",
    )

    parser.add_argument(
        "--algo",
        default="iql",
        choices=["iql", "cql"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--base_out_dir",
        default="/kaggle/working/OfflineRL-Reliability/checkpoints",
    )

    parser.add_argument(
        "--status_file",
        default=(
            "/kaggle/working/OfflineRL-Reliability/"
            "results/replications/iql_replication_status.json"
        ),
    )

    parser.add_argument(
        "--console_log_dir",
        default=(
            "/kaggle/working/OfflineRL-Reliability/"
            "logs/replications"
        ),
    )

    args = parser.parse_args()

    repo_root = Path.cwd()

    train_script = (
        repo_root
        / "src"
        / "training"
        / "train_and_verify.py"
    )

    if not train_script.exists():
        raise FileNotFoundError(
            f"Training script not found: {train_script}"
        )

    base_out_dir = Path(
        args.base_out_dir
    )

    status_path = Path(
        args.status_file
    )

    console_log_dir = Path(
        args.console_log_dir
    )

    base_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    console_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = {
        "experiment": "IQL-100K-multi-seed-replication",
        "task": args.task,
        "algorithm": args.algo,
        "training_steps": args.n_steps,
        "device": args.device,
        "seeds_requested": args.seeds,
        "started_utc": utc_timestamp(),
        "finished_utc": None,
        "overall_status": "RUNNING",
        "seeds": {},
    }

    for seed in args.seeds:
        status["seeds"][str(seed)] = {
            "status": "PENDING",
            "started_utc": None,
            "finished_utc": None,
            "return_code": None,
            "checkpoint_dir": str(
                base_out_dir
                / f"{args.algo}_seed{seed}"
            ),
            "console_log": str(
                console_log_dir
                / f"{args.algo}_seed{seed}.log"
            ),
        }

    save_status(
        status_path,
        status,
    )

    print("=" * 80)
    print(
        "MULTI-SEED IQL REPLICATION RUNNER"
    )
    print("=" * 80)
    print(f"Task         : {args.task}")
    print(f"Algorithm    : {args.algo}")
    print(f"Steps/seed   : {args.n_steps}")
    print(f"Device       : {args.device}")
    print(f"Seeds        : {args.seeds}")
    print(f"Status file  : {status_path}")
    print("=" * 80)

    for seed in args.seeds:
        print()
        print("=" * 80)
        print(
            f"STARTING SEED {seed}"
        )
        print("=" * 80)

        seed_out_dir = (
            base_out_dir
            / f"{args.algo}_seed{seed}"
        )

        seed_log_path = (
            console_log_dir
            / f"{args.algo}_seed{seed}.log"
        )

        seed_out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        status["seeds"][str(seed)][
            "status"
        ] = "RUNNING"

        status["seeds"][str(seed)][
            "started_utc"
        ] = utc_timestamp()

        save_status(
            status_path,
            status,
        )

        command = [
            sys.executable,
            str(train_script),
            "--algo",
            args.algo,
            "--task",
            args.task,
            "--seed",
            str(seed),
            "--n_steps",
            str(args.n_steps),
            "--device",
            args.device,
            "--out_dir",
            str(seed_out_dir),
        ]

        print(
            "Command:",
            " ".join(command),
        )

        with open(
            seed_log_path,
            "w",
        ) as log_file:

            log_file.write(
                "COMMAND\n"
                + " ".join(command)
                + "\n\n"
            )

            log_file.flush()

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(
                    line,
                    end="",
                )
                log_file.write(
                    line
                )
                log_file.flush()

            return_code = process.wait()

        status["seeds"][str(seed)][
            "return_code"
        ] = int(return_code)

        status["seeds"][str(seed)][
            "finished_utc"
        ] = utc_timestamp()

        if return_code != 0:
            status["seeds"][str(seed)][
                "status"
            ] = "FAILED"

            status["overall_status"] = (
                "FAILED"
            )

            status["finished_utc"] = (
                utc_timestamp()
            )

            save_status(
                status_path,
                status,
            )

            print()
            print("=" * 80)
            print(
                f"SEED {seed} FAILED"
            )
            print(
                f"Return code: {return_code}"
            )
            print(
                "Replication run stopped."
            )
            print("=" * 80)

            raise SystemExit(
                return_code
            )

        status["seeds"][str(seed)][
            "status"
        ] = "COMPLETE"

        save_status(
            status_path,
            status,
        )

        print()
        print("=" * 80)
        print(
            f"SEED {seed} COMPLETE"
        )
        print("=" * 80)

    status["overall_status"] = (
        "COMPLETE"
    )

    status["finished_utc"] = (
        utc_timestamp()
    )

    save_status(
        status_path,
        status,
    )

    print()
    print("=" * 80)
    print(
        "ALL REPLICATIONS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()