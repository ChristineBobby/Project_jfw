#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vla_coreset.selection.episode_selector import (
    build_episode_embeddings,
    select_diverse_episodes,
    write_pc_ras_coreset_json,
)
from vla_coreset.training.mlp_baseline import load_feature_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select PC-RAS episode coreset with diversity-aware greedy.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--episode-score-file", type=Path, default=Path("results/tables/pc_ras_episode_scores.csv"))
    parser.add_argument("--coreset-dir", type=Path, default=Path("data/coresets"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--split-name", default="candidate_train")
    parser.add_argument("--saliency-weight", type=float, default=0.65)
    parser.add_argument("--diversity-weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.coreset_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    features, actions, _, index = load_feature_artifacts(args.feature_dir)
    split_mask = (index["split"] == args.split_name).to_numpy()
    if not split_mask.any():
        raise ValueError(f"No rows found for split {args.split_name!r}")

    episode_scores = pd.read_csv(args.episode_score_file)
    train_episodes = set(index.loc[split_mask, "episode_index"].astype(int).unique().tolist())
    episode_scores = episode_scores[episode_scores["episode_index"].astype(int).isin(train_episodes)].copy()
    if len(episode_scores) != len(train_episodes):
        raise ValueError("episode score file does not cover every candidate_train episode")

    embeddings = build_episode_embeddings(
        features=features[split_mask],
        actions=actions[split_mask],
        index=index.loc[split_mask].reset_index(drop=True),
    )
    selected, trace = select_diverse_episodes(
        episode_scores=episode_scores,
        embeddings=embeddings,
        budget=args.budget,
        saliency_weight=args.saliency_weight,
        diversity_weight=args.diversity_weight,
    )

    coreset_path = args.coreset_dir / f"pc_ras_episode_top{args.budget}.json"
    trace_path = args.table_dir / "pc_ras_episode_selection_trace.csv"
    write_pc_ras_coreset_json(
        path=coreset_path,
        selected_episodes=selected,
        trace=trace,
        saliency_weight=args.saliency_weight,
        diversity_weight=args.diversity_weight,
        score_file=str(args.episode_score_file),
    )
    trace.to_csv(trace_path, index=False)

    print(f"selected_episodes: {selected}")
    print(f"coreset: {coreset_path}")
    print(f"trace: {trace_path}")
    print(trace.to_string(index=False))


if __name__ == "__main__":
    main()
