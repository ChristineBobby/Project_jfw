#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vla_coreset.training.mlp_baseline import load_feature_artifacts
from vla_coreset.visualization.report_figures import FigureInputs, make_report_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-quality figures for PC-RAS experiments.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--random-summary", type=Path, default=Path("results/tables/random10_clip_summary.csv"))
    parser.add_argument("--pc-ras-summary", type=Path, default=Path("results/tables/pc_ras_clip_summary.csv"))
    parser.add_argument("--episode-scores", type=Path, default=Path("results/tables/pc_ras_episode_scores.csv"))
    parser.add_argument("--selection-trace", type=Path, default=Path("results/tables/pc_ras_episode_selection_trace.csv"))
    parser.add_argument("--pc-ras-coreset", type=Path, default=Path("data/coresets/pc_ras_episode_top5.json"))
    parser.add_argument("--random-coreset", type=Path, default=Path("data/coresets/random_episode_seed2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def _load_selected(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(ep) for ep in payload["selected_episodes"]]


def build_coverage_points(feature_dir: Path, random_selected: list[int], pc_ras_selected: list[int]) -> pd.DataFrame:
    features, actions, _, index = load_feature_artifacts(feature_dir)
    mask = (index["split"] == "candidate_train").to_numpy()
    index = index.loc[mask].reset_index(drop=True)
    features = features[mask]
    actions = actions[mask]

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    for episode, positions in index.groupby("episode_index", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=np.int64)
        vector = np.concatenate(
            [
                features[idx].mean(axis=0),
                actions[idx, :7].mean(axis=0),
                actions[idx, :7].std(axis=0),
            ]
        )
        vectors.append(vector.astype(np.float32))
        ep = int(episode)
        if ep in pc_ras_selected:
            group = "PC-RAS"
        elif ep in random_selected:
            group = "Random"
        else:
            group = "Candidate"
        rows.append({"episode_index": ep, "group": group})

    scaled = StandardScaler().fit_transform(np.vstack(vectors))
    coords = PCA(n_components=2, random_state=0).fit_transform(scaled)
    points = pd.DataFrame(rows)
    points["x"] = coords[:, 0]
    points["y"] = coords[:, 1]
    return points


def main() -> None:
    args = parse_args()
    random_selected = _load_selected(args.random_coreset)
    pc_ras_selected = _load_selected(args.pc_ras_coreset)
    coverage_points = build_coverage_points(args.feature_dir, random_selected, pc_ras_selected)
    inputs = FigureInputs(
        random_summary=pd.read_csv(args.random_summary),
        pc_ras_summary=pd.read_csv(args.pc_ras_summary),
        episode_scores=pd.read_csv(args.episode_scores),
        selection_trace=pd.read_csv(args.selection_trace),
        coverage_points=coverage_points,
        output_dir=args.output_dir,
    )
    written = make_report_figures(inputs)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
