"""Dependency-light structural audit for one P1 raw output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SIGMAS = {0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3}
HORIZONS = {1, 5, 10, 20}
REQUIRED = {
    "policy_seed", "state_id", "source_episode_id", "source_step", "sigma",
    "standardized_noise", "clean_action", "shifted_action", "action_disagreement",
    "support_distance", "critic_disagreement", "clean_return", "shifted_return",
    "delta_J", "absolute_consequence", "relative_consequence", "clean_steps",
    "shifted_steps", "clean_terminated", "shifted_terminated", "clean_truncated",
    "shifted_truncated", "checkpoint", "git_commit", "state_sampling_seed", "noise_seed",
}


def audit(path: Path) -> None:
    data = json.loads(path.read_text())
    records = data["records"]
    actual_sigmas = {round(float(r["sigma"]), 3) for r in records}
    actual_horizons = {int(r["horizon"]) for r in records}
    states = {int(r["state_id"]) for r in records}
    missing = sorted(REQUIRED - set(records[0])) if records else sorted(REQUIRED)
    assert actual_sigmas == {round(x, 3) for x in SIGMAS}, (actual_sigmas, SIGMAS)
    assert actual_horizons == HORIZONS, (actual_horizons, HORIZONS)
    assert missing == [], missing
    assert len(states) == int(data["n_states"]), (len(states), data["n_states"])
    assert all(r["policy_seed"] == data["policy_seed"] for r in records)
    assert all(len(r["standardized_noise"]) == 11 for r in records)
    assert all(all(isinstance(r[k], (int, float, bool, list)) for k in REQUIRED) for r in records)
    print(f"PASS {path}: records={len(records)}, states={len(states)}, sigmas={sorted(actual_sigmas)}, horizons={sorted(actual_horizons)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    audit(parser.parse_args().path)
