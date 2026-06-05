from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def build_random_episode_coreset(candidate_train: list[int], budget: int, seed: int) -> list[int]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(candidate_train):
        raise ValueError("budget cannot exceed candidate_train size")
    rng = random.Random(seed)
    return sorted(rng.sample(candidate_train, budget))


def write_coreset_json(path: Path, seed: int, selected_episodes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "method": "random_episode",
        "unit": "episode",
        "seed": int(seed),
        "budget_episodes": len(selected_episodes),
        "selected_episodes": [int(ep) for ep in selected_episodes],
        "notes": "Random 10% episode baseline sampled from candidate_train only.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
