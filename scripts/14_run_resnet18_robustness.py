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
from tqdm import tqdm

from vla_coreset.selection.random_baseline import build_random_episode_coreset, write_coreset_json
from vla_coreset.training.mlp_baseline import (
    load_feature_artifacts,
    make_feature_matrix,
    train_mlp_baseline,
)
from vla_coreset.visualization.report_figures import COLORS, set_paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Random-10% and PC-RAS baselines on ResNet18 features.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/resnet18_top_left7"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/split_v1.json"))
    parser.add_argument("--pc-ras-coreset", type=Path, default=Path("data/coresets/pc_ras_episode_top5.json"))
    parser.add_argument("--coreset-dir", type=Path, default=Path("data/coresets"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Use fewer epochs for quick pipeline checks.")
    return parser.parse_args()


def _load_selected(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [int(ep) for ep in payload["selected_episodes"]]
    if not selected:
        raise ValueError(f"No selected episodes in {path}")
    return selected


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.copy()
    return summary


def _save_comparison_figure(random_summary: pd.DataFrame, pc_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    set_paper_style()
    rows = [
        ("ResNet18 Random-10%", random_summary["test_original_mse"].mean(), random_summary["test_original_mse"].std()),
        ("ResNet18 PC-RAS", pc_summary["test_original_mse"].mean(), pc_summary["test_original_mse"].std()),
    ]
    labels = [row[0] for row in rows]
    means = np.asarray([row[1] for row in rows], dtype=np.float32)
    stds = np.asarray([0.0 if pd.isna(row[2]) else row[2] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(4.9, 3.1), constrained_layout=True)
    x = np.arange(len(rows))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=0.52,
        color=[COLORS["random"], COLORS["pc_ras"]],
        edgecolor="#2F2F2F",
        linewidth=0.8,
        capsize=4,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test MSE, original action space")
    ax.set_title("ResNet18 feature robustness")
    ax.set_ylim(0.0, float((means + stds).max() * 1.25))
    label_offset = float((means + stds).max()) * 0.035
    for bar, value, err in zip(bars, means, stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + err + label_offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "resnet18_mse_comparison.png", output_dir / "resnet18_mse_comparison.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.coreset_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    candidate_train = [int(ep) for ep in split["candidate_train"]]
    pc_ras_selected = _load_selected(args.pc_ras_coreset)
    features, actions, text_features, index = load_feature_artifacts(args.feature_dir)
    x = make_feature_matrix(features, text_features)
    max_epochs = min(args.max_epochs, 5) if args.smoke else args.max_epochs
    patience = min(args.patience, 3) if args.smoke else args.patience

    random_metrics: list[dict] = []
    pc_metrics: list[dict] = []
    for method in tqdm(["resnet18_random10", "resnet18_pc_ras"], desc="resnet18 methods"):
        for seed in tqdm(args.seeds, desc=method, leave=False):
            if method == "resnet18_random10":
                selected = build_random_episode_coreset(candidate_train, budget=args.budget, seed=seed)
                coreset_path = args.coreset_dir / f"resnet18_random_episode_seed{seed}.json"
                write_coreset_json(coreset_path, seed=seed, selected_episodes=selected)
            else:
                selected = pc_ras_selected
                coreset_path = args.pc_ras_coreset
            result = train_mlp_baseline(
                x=x,
                actions=actions,
                index=index,
                selected_episodes=selected,
                seed=seed,
                device_name=args.device,
                batch_size=args.batch_size,
                max_epochs=max_epochs,
                patience=patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                progress=True,
            )
            metrics = {
                "method": method,
                "coreset_file": str(coreset_path),
                **result["metrics"],
                "selected_episodes": " ".join(str(ep) for ep in selected),
            }
            if method == "resnet18_random10":
                random_metrics.append(metrics)
            else:
                pc_metrics.append(metrics)
            pd.DataFrame([metrics]).to_csv(args.table_dir / f"{method}_seed{seed}_metrics.csv", index=False)
            np.savez_compressed(
                args.prediction_dir / f"{method}_seed{seed}.npz",
                val_pred=result["val_pred"],
                test_pred=result["test_pred"],
                val_target=result["val_target"],
                test_target=result["test_target"],
            )

    random_summary = _summary(pd.DataFrame(random_metrics))
    pc_summary = _summary(pd.DataFrame(pc_metrics))
    random_summary.to_csv(args.table_dir / "resnet18_random10_summary.csv", index=False)
    pc_summary.to_csv(args.table_dir / "resnet18_pc_ras_summary.csv", index=False)
    for path in _save_comparison_figure(random_summary, pc_summary, args.figure_dir):
        print(path)
    print("random")
    print(random_summary[["test_original_mse", "test_normalized_mse"]].agg(["mean", "std", "min", "max"]).to_string())
    print("pc_ras")
    print(pc_summary[["test_original_mse", "test_normalized_mse"]].agg(["mean", "std", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
