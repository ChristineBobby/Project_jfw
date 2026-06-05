#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vla_coreset.training.mlp_baseline import load_feature_artifacts
from vla_coreset.selection.scores import aggregate_episode_scores, compute_frame_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PC-RAS frame and episode scores.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/selection"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--split-name", default="candidate_train")
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--phase-bins", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    features, actions, _, index = load_feature_artifacts(args.feature_dir)
    split_mask = (index["split"] == args.split_name).to_numpy()
    if not split_mask.any():
        raise ValueError(f"No rows found for split {args.split_name!r}")

    frame_scores = compute_frame_scores(
        features=features[split_mask],
        actions=actions[split_mask],
        index=index.loc[split_mask].reset_index(drop=True),
        phase_bins=args.phase_bins,
    )
    episode_scores = aggregate_episode_scores(frame_scores, top_fraction=args.top_fraction)
    episode_scores = episode_scores.sort_values("episode_score", ascending=False).reset_index(drop=True)
    episode_scores.insert(0, "rank", range(1, len(episode_scores) + 1))

    frame_path = args.cache_dir / "pc_ras_frame_scores.parquet"
    episode_path = args.table_dir / "pc_ras_episode_scores.csv"
    summary_path = args.table_dir / "pc_ras_frame_scores_summary.csv"

    frame_scores.to_parquet(frame_path, index=False)
    episode_scores.to_csv(episode_path, index=False)
    summary = frame_scores[
        [
            "visual_delta",
            "action_delta",
            "action_jerk",
            "gripper_delta",
            "pc_score",
            "ras_score",
            "frame_score",
        ]
    ].agg(["mean", "std", "min", "max"])
    summary.to_csv(summary_path)

    print(f"frame_scores: {frame_scores.shape} -> {frame_path}")
    print(f"episode_scores: {episode_scores.shape} -> {episode_path}")
    print(f"summary: {summary.shape} -> {summary_path}")
    print(episode_scores.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
