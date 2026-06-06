from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vla_coreset.selection.episode_selector import select_diverse_episodes
from vla_coreset.selection.scores import robust_normalize


ABLATION_VARIANTS = [
    "action_delta_only",
    "visual_delta_only",
    "pc_only",
    "pc_ras_no_coverage",
    "coverage_only",
    "pc_ras_full",
]


def _score_column_for_variant(variant: str) -> tuple[str, ...]:
    if variant == "action_delta_only":
        return ("action_delta",)
    if variant == "visual_delta_only":
        return ("visual_delta",)
    if variant == "pc_only":
        return ("pc_score",)
    if variant == "pc_ras_no_coverage":
        return ("pc_score", "ras_score")
    if variant in {"coverage_only", "pc_ras_full"}:
        return ("pc_score", "ras_score")
    raise ValueError(f"Unknown ablation variant: {variant}")


def build_ablation_episode_scores(
    frame_scores: pd.DataFrame,
    variant: str,
    top_fraction: float = 0.2,
) -> pd.DataFrame:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    required = {"episode_index", *_score_column_for_variant(variant)}
    missing = required.difference(frame_scores.columns)
    if missing:
        raise ValueError(f"frame_scores is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for episode, group in frame_scores.groupby("episode_index", sort=True):
        if variant == "pc_ras_no_coverage":
            raw = 0.55 * group["pc_score"].to_numpy(dtype=np.float32) + 0.45 * group["ras_score"].to_numpy(dtype=np.float32)
        elif variant == "pc_ras_full":
            raw = 0.55 * group["pc_score"].to_numpy(dtype=np.float32) + 0.45 * group["ras_score"].to_numpy(dtype=np.float32)
        elif variant == "coverage_only":
            raw = np.ones(len(group), dtype=np.float32)
        else:
            column = _score_column_for_variant(variant)[0]
            raw = group[column].to_numpy(dtype=np.float32)
        top_count = max(1, int(np.ceil(len(raw) * top_fraction)))
        top_values = np.sort(raw)[-top_count:]
        rows.append(
            {
                "method": variant,
                "episode_index": int(episode),
                "episode_score": float(top_values.mean()),
                "frames": int(len(group)),
                "top_frames": int(top_count),
            }
        )
    out = pd.DataFrame(rows)
    out["episode_score"] = robust_normalize(out["episode_score"].to_numpy(dtype=np.float32), lower=0.0, upper=100.0)
    return out.sort_values("episode_index").reset_index(drop=True)


def select_ablation_coreset(
    variant: str,
    episode_scores: pd.DataFrame,
    embeddings: pd.DataFrame,
    budget: int,
) -> tuple[list[int], pd.DataFrame]:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    if variant == "coverage_only":
        # A tiny uniform saliency epsilon makes the first selected episode deterministic.
        scores = episode_scores.copy()
        scores["episode_score"] = 1.0
        selected, trace = select_diverse_episodes(scores, embeddings, budget, saliency_weight=0.0, diversity_weight=1.0)
    elif variant == "pc_ras_full":
        selected, trace = select_diverse_episodes(episode_scores, embeddings, budget, saliency_weight=0.65, diversity_weight=0.35)
    else:
        selected, trace = select_diverse_episodes(episode_scores, embeddings, budget, saliency_weight=1.0, diversity_weight=0.0)
    trace.insert(0, "method", variant)
    return selected, trace


def write_ablation_coreset_json(
    path: Path,
    variant: str,
    selected_episodes: list[int],
    trace: pd.DataFrame,
    source_frame_score_file: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in trace.to_dict(orient="records"):
        records.append(
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
        "method": f"ablation_{variant}",
        "variant": variant,
        "unit": "episode",
        "budget_episodes": len(selected_episodes),
        "selected_episodes": [int(ep) for ep in selected_episodes],
        "source_frame_score_file": source_frame_score_file,
        "selection_trace": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate_ablation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"variant", "seed", "test_original_mse", "test_normalized_mse", "selected_episodes"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics is missing required columns: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    for variant, group in metrics.groupby("variant", sort=True):
        rows.append(
            {
                "variant": str(variant),
                "selected_episodes": str(group["selected_episodes"].iloc[0]),
                "num_seeds": int(group["seed"].nunique()),
                "test_original_mse_mean": float(group["test_original_mse"].mean()),
                "test_original_mse_std": float(group["test_original_mse"].std()),
                "test_original_mse_min": float(group["test_original_mse"].min()),
                "test_original_mse_max": float(group["test_original_mse"].max()),
                "test_normalized_mse_mean": float(group["test_normalized_mse"].mean()),
                "test_normalized_mse_std": float(group["test_normalized_mse"].std()),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)
