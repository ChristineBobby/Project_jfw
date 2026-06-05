from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("data/splits/split_v1.json")
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"


def build_split_v1() -> dict[str, Any]:
    return {
        "version": "split_v1",
        "dataset": DATASET_ID,
        "unit": "episode",
        "candidate_train": list(range(0, 40)),
        "val": list(range(40, 45)),
        "test": list(range(45, 50)),
        "notes": (
            "Episode-level split. Selectors, normalizers, and training subsets must use "
            "candidate_train only. Validation is for early stopping/model selection. "
            "Test is reserved for final reporting."
        ),
    }


def validate_split(split: dict[str, Any], total_episodes: int) -> None:
    required = ["candidate_train", "val", "test"]
    missing = [key for key in required if key not in split]
    if missing:
        raise ValueError(f"Split missing required keys: {missing}")

    groups = {key: split[key] for key in required}
    for name, episodes in groups.items():
        if not all(isinstance(ep, int) for ep in episodes):
            raise ValueError(f"{name} contains non-integer episode ids")
        if len(episodes) != len(set(episodes)):
            raise ValueError(f"{name} contains duplicate episode ids")
        out_of_range = [ep for ep in episodes if ep < 0 or ep >= total_episodes]
        if out_of_range:
            raise ValueError(f"{name} contains out-of-range episode ids: {out_of_range}")

    all_episodes = [ep for episodes in groups.values() for ep in episodes]
    if len(all_episodes) != len(set(all_episodes)):
        raise ValueError("Split groups overlap")

    expected = list(range(total_episodes))
    observed = sorted(all_episodes)
    if observed != expected:
        raise ValueError(f"Split does not cover all episodes 0..{total_episodes - 1}")


def write_split(path: Path, split: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed episode-level split for Project_jfw.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-episodes", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_split_v1()
    validate_split(split, total_episodes=args.total_episodes)
    write_split(args.output, split)
    print(f"wrote: {args.output}")
    print(f"candidate_train: {len(split['candidate_train'])} episodes")
    print(f"val: {len(split['val'])} episodes")
    print(f"test: {len(split['test'])} episodes")


if __name__ == "__main__":
    main()
