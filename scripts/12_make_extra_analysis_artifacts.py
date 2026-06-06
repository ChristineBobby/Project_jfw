#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_coreset.analysis.artifacts import (
    build_selected_episode_explanations,
    compute_phase_mse_summary,
    make_action_curve_figure,
    make_phase_mse_figure,
    make_score_timeline_figure,
)
from vla_coreset.data.inspect_dataset import DEFAULT_REPO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate extra analysis tables and figures.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cache/lerobot/aloha_sim_transfer_cube_human"))
    parser.add_argument("--pc-ras-coreset", type=Path, default=Path("data/coresets/pc_ras_episode_top5.json"))
    parser.add_argument("--random-coreset", type=Path, default=Path("data/coresets/random_episode_seed2.json"))
    parser.add_argument("--frame-scores", type=Path, default=Path("data/cache/selection/pc_ras_frame_scores.parquet"))
    parser.add_argument("--episode-scores", type=Path, default=Path("results/tables/pc_ras_episode_scores.csv"))
    parser.add_argument("--selection-trace", type=Path, default=Path("results/tables/pc_ras_episode_selection_trace.csv"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--timeline-episodes", type=int, nargs="+", default=[36, 20, 6])
    parser.add_argument("--phase-bins", type=int, default=5)
    return parser.parse_args()


def _load_selected(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(ep) for ep in payload["selected_episodes"]]


def _load_prediction_runs(prediction_dir: Path, stem: str, seeds: list[int]) -> list[dict]:
    runs = []
    for seed in seeds:
        path = prediction_dir / f"{stem}_seed{seed}.npz"
        arrays = np.load(path)
        runs.append(
            {
                "seed": seed,
                "test_pred": arrays["test_pred"],
                "test_target": arrays["test_target"],
            }
        )
    return runs


def _build_action_rows(index: pd.DataFrame, pc_ras_selected: list[int], random_selected: list[int]) -> pd.DataFrame:
    keep = index[index["episode_index"].isin(set(pc_ras_selected + random_selected))].copy()
    groups = []
    for episode in keep["episode_index"].astype(int):
        if episode in pc_ras_selected:
            groups.append("PC-RAS")
        elif episode in random_selected:
            groups.append("Random")
        else:
            groups.append("Other")
    keep["group"] = groups
    rename = {f"action_left_{i}": f"action_{i}" for i in range(7)}
    return keep.rename(columns=rename)


def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def make_keyframe_figure(
    dataset_root: Path,
    frame_scores: pd.DataFrame,
    index: pd.DataFrame,
    selected_episodes: list[int],
    output_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
) -> list[Path]:
    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)
    rows = []
    for episode in selected_episodes:
        top = frame_scores[frame_scores["episode_index"] == int(episode)].nlargest(1, "frame_score").iloc[0]
        match = index[(index["episode_index"] == int(episode)) & (index["frame_index"] == int(top["frame_index"]))].iloc[0]
        rows.append(
            {
                "episode_index": int(episode),
                "frame_index": int(top["frame_index"]),
                "row_index": int(match["row_index"]),
                "frame_score": float(top["frame_score"]),
                "pc_score": float(top["pc_score"]),
                "ras_score": float(top["ras_score"]),
                "phase": float(top["phase"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 3.0), squeeze=False)
    for ax, row in zip(axes.flat, rows, strict=True):
        sample = dataset[row["row_index"]]
        ax.imshow(_tensor_image_to_numpy(sample["observation.images.top"]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"ep {row['episode_index']} | f {row['frame_index']}\n"
            f"score {row['frame_score']:.2f}, phase {row['phase']:.2f}",
            fontsize=8,
            labelpad=6,
        )
    fig.suptitle("Representative PC-RAS key frames", fontsize=11)
    fig.tight_layout()
    paths = [output_dir / "pc_ras_key_frames.png", output_dir / "pc_ras_key_frames.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(args.feature_dir / "index.parquet")
    frame_scores = pd.read_parquet(args.frame_scores)
    episode_scores = pd.read_csv(args.episode_scores)
    selection_trace = pd.read_csv(args.selection_trace)
    pc_ras_selected = _load_selected(args.pc_ras_coreset)
    random_selected = _load_selected(args.random_coreset)

    explanations = build_selected_episode_explanations(frame_scores, episode_scores, selection_trace, pc_ras_selected)
    explanation_path = args.table_dir / "pc_ras_selected_episode_explanations.csv"
    explanations.to_csv(explanation_path, index=False)
    print(explanation_path)

    predictions = {
        "Random-10%": _load_prediction_runs(args.prediction_dir, "random10_clip", [0, 1, 2, 3, 4]),
        "PC-RAS": _load_prediction_runs(args.prediction_dir, "pc_ras_clip", [0, 1, 2, 3, 4]),
    }
    phase_summary = compute_phase_mse_summary(index, predictions, phase_bins=args.phase_bins)
    phase_path = args.table_dir / "phase_mse_summary.csv"
    phase_summary.to_csv(phase_path, index=False)
    print(phase_path)
    for path in make_phase_mse_figure(phase_summary, args.figure_dir):
        print(path)

    low_score_episode = int(episode_scores.sort_values("episode_score").iloc[0]["episode_index"])
    for episode in list(dict.fromkeys(args.timeline_episodes + [low_score_episode])):
        for path in make_score_timeline_figure(frame_scores, episode, args.figure_dir):
            print(path)

    action_rows = _build_action_rows(index, pc_ras_selected, random_selected)
    for path in make_action_curve_figure(action_rows, args.figure_dir):
        print(path)

    for path in make_keyframe_figure(args.dataset_root, frame_scores, index, pc_ras_selected, args.figure_dir):
        print(path)


if __name__ == "__main__":
    main()
