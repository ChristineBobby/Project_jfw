from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vla_coreset.selection.scores import robust_normalize


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (values / norms).astype(np.float32)


def build_episode_embeddings(features: np.ndarray, actions: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
    features = np.asarray(features, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if len(index) != len(features) or len(index) != len(actions):
        raise ValueError("features, actions, and index must have the same number of rows")
    if "episode_index" not in index.columns:
        raise ValueError("index is missing required column: episode_index")
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected actions [N,>=7], got {actions.shape}")

    rows: list[np.ndarray] = []
    episodes: list[int] = []
    for episode, positions in index.groupby("episode_index", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=np.int64)
        image_mean = features[idx].mean(axis=0)
        action_mean = actions[idx, :7].mean(axis=0)
        action_std = actions[idx, :7].std(axis=0)
        rows.append(np.concatenate([image_mean, action_mean, action_std]).astype(np.float32))
        episodes.append(int(episode))

    matrix = _l2_normalize_rows(np.vstack(rows))
    return pd.DataFrame(matrix, index=episodes)


def _pairwise_min_distance(candidate_embeddings: np.ndarray, selected_embeddings: np.ndarray) -> np.ndarray:
    if len(selected_embeddings) == 0:
        return np.ones(len(candidate_embeddings), dtype=np.float32)
    distances = np.linalg.norm(candidate_embeddings[:, None, :] - selected_embeddings[None, :, :], axis=2)
    return distances.min(axis=1).astype(np.float32)


def select_diverse_episodes(
    episode_scores: pd.DataFrame,
    embeddings: pd.DataFrame,
    budget: int,
    saliency_weight: float = 0.65,
    diversity_weight: float = 0.35,
) -> tuple[list[int], pd.DataFrame]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(episode_scores):
        raise ValueError("budget cannot exceed number of candidate episodes")
    if saliency_weight < 0 or diversity_weight < 0:
        raise ValueError("weights must be non-negative")
    if saliency_weight + diversity_weight <= 0:
        raise ValueError("at least one weight must be positive")
    required = {"episode_index", "episode_score"}
    missing = required.difference(episode_scores.columns)
    if missing:
        raise ValueError(f"episode_scores is missing required columns: {sorted(missing)}")

    scores = episode_scores[["episode_index", "episode_score"]].copy()
    scores["episode_index"] = scores["episode_index"].astype(int)
    scores = scores.sort_values("episode_index").reset_index(drop=True)
    missing_embeddings = sorted(set(scores["episode_index"]) - set(embeddings.index.astype(int)))
    if missing_embeddings:
        raise ValueError(f"embeddings missing episodes: {missing_embeddings}")

    saliency = robust_normalize(scores["episode_score"].to_numpy(dtype=np.float32), lower=0.0, upper=100.0)
    episode_ids = scores["episode_index"].to_numpy(dtype=np.int64)
    embedding_matrix = embeddings.loc[episode_ids].to_numpy(dtype=np.float32)

    selected: list[int] = []
    trace_rows: list[dict[str, float | int]] = []
    available = np.ones(len(episode_ids), dtype=bool)
    for order in range(1, budget + 1):
        selected_indices = [int(np.flatnonzero(episode_ids == episode)[0]) for episode in selected]
        selected_embeddings = embedding_matrix[selected_indices] if selected_indices else np.empty((0, embedding_matrix.shape[1]))
        raw_diversity = _pairwise_min_distance(embedding_matrix, selected_embeddings)
        diversity = np.ones_like(raw_diversity) if not selected else robust_normalize(raw_diversity, lower=0.0, upper=100.0)
        final = saliency_weight * saliency + diversity_weight * diversity
        final = np.where(available, final, -np.inf)
        chosen_idx = int(np.argmax(final))
        chosen_episode = int(episode_ids[chosen_idx])
        selected.append(chosen_episode)
        available[chosen_idx] = False
        trace_rows.append(
            {
                "selection_order": order,
                "episode_index": chosen_episode,
                "final_score": float(final[chosen_idx]),
                "saliency_score": float(saliency[chosen_idx]),
                "diversity_score": float(diversity[chosen_idx]),
                "raw_diversity_distance": float(raw_diversity[chosen_idx]),
                "source_episode_score": float(scores.loc[chosen_idx, "episode_score"]),
            }
        )

    return selected, pd.DataFrame(trace_rows)


def write_pc_ras_coreset_json(
    path: Path,
    selected_episodes: list[int],
    trace: pd.DataFrame,
    saliency_weight: float,
    diversity_weight: float,
    score_file: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_records: list[dict[str, Any]] = []
    for row in trace.to_dict(orient="records"):
        trace_records.append(
            {
                "selection_order": int(row["selection_order"]),
                "episode_index": int(row["episode_index"]),
                "final_score": float(row["final_score"]),
                "saliency_score": float(row["saliency_score"]),
                "diversity_score": float(row["diversity_score"]),
                "raw_diversity_distance": float(row.get("raw_diversity_distance", 0.0)),
                "source_episode_score": float(row.get("source_episode_score", 0.0)),
            }
        )
    payload: dict[str, Any] = {
        "method": "pc_ras_diversity_greedy",
        "unit": "episode",
        "budget_episodes": len(selected_episodes),
        "selected_episodes": [int(ep) for ep in selected_episodes],
        "weights": {
            "saliency": float(saliency_weight),
            "diversity": float(diversity_weight),
        },
        "source_score_file": score_file,
        "selection_trace": trace_records,
        "notes": "PC-RAS episode coreset selected from candidate_train with diversity-aware greedy.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
