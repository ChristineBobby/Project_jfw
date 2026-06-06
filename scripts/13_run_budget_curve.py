#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vla_coreset.analysis.artifacts import make_budget_curve_figure
from vla_coreset.selection.episode_selector import build_episode_embeddings, select_diverse_episodes
from vla_coreset.selection.random_baseline import build_random_episode_coreset, write_coreset_json
from vla_coreset.selection.scores import aggregate_episode_scores
from vla_coreset.training.mlp_baseline import (
    load_feature_artifacts,
    make_feature_matrix,
    train_mlp_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Random vs PC-RAS episode budget curve.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/split_v1.json"))
    parser.add_argument("--frame-score-file", type=Path, default=Path("data/cache/selection/pc_ras_frame_scores.parquet"))
    parser.add_argument("--coreset-dir", type=Path, default=Path("data/coresets"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 5, 10, 20], help="Candidate-train budget percentages.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Use fewer epochs for quick pipeline checks.")
    return parser.parse_args()


def _budget_to_episode_count(candidate_count: int, percent: int) -> int:
    if percent <= 0:
        raise ValueError("budget percentages must be positive")
    return max(1, int(round(candidate_count * percent / 100.0)))


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, budget_percent, budget_episodes), group in metrics.groupby(
        ["method", "budget_percent", "budget_episodes"], sort=True
    ):
        rows.append(
            {
                "method": method,
                "budget_percent": int(budget_percent),
                "budget_episodes": int(budget_episodes),
                "num_seeds": int(group["seed"].nunique()),
                "test_original_mse_mean": float(group["test_original_mse"].mean()),
                "test_original_mse_std": float(group["test_original_mse"].std()),
                "test_normalized_mse_mean": float(group["test_normalized_mse"].mean()),
                "test_normalized_mse_std": float(group["test_normalized_mse"].std()),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "budget_percent"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.coreset_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    candidate_train = [int(ep) for ep in split["candidate_train"]]
    features, actions, text_features, index = load_feature_artifacts(args.feature_dir)
    x = make_feature_matrix(features, text_features)
    train_mask = (index["split"] == "candidate_train").to_numpy()
    embeddings = build_episode_embeddings(
        features=features[train_mask],
        actions=actions[train_mask],
        index=index.loc[train_mask].reset_index(drop=True),
    )
    episode_scores = aggregate_episode_scores(pd.read_parquet(args.frame_score_file))
    max_epochs = min(args.max_epochs, 5) if args.smoke else args.max_epochs
    patience = min(args.patience, 3) if args.smoke else args.patience

    all_metrics: list[dict] = []
    for budget_percent in tqdm(args.budgets, desc="budget percentages"):
        budget_episodes = _budget_to_episode_count(len(candidate_train), int(budget_percent))
        pc_selected, pc_trace = select_diverse_episodes(
            episode_scores,
            embeddings,
            budget=budget_episodes,
            saliency_weight=0.65,
            diversity_weight=0.35,
        )
        pc_trace.to_csv(args.table_dir / f"budget_pc_ras_{budget_percent}pct_selection_trace.csv", index=False)
        write_coreset_json(
            args.coreset_dir / f"budget_pc_ras_{budget_percent}pct_top{budget_episodes}.json",
            seed=0,
            selected_episodes=pc_selected,
        )

        for method in ["Random", "PC-RAS"]:
            for seed in tqdm(args.seeds, desc=f"{method} {budget_percent}%", leave=False):
                if method == "Random":
                    selected = build_random_episode_coreset(candidate_train, budget=budget_episodes, seed=seed)
                    coreset_path = args.coreset_dir / f"budget_random_{budget_percent}pct_seed{seed}.json"
                    write_coreset_json(coreset_path, seed=seed, selected_episodes=selected)
                else:
                    selected = pc_selected
                    coreset_path = args.coreset_dir / f"budget_pc_ras_{budget_percent}pct_top{budget_episodes}.json"

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
                    "budget_percent": int(budget_percent),
                    "budget_episodes": int(budget_episodes),
                    "coreset_file": str(coreset_path),
                    **result["metrics"],
                    "selected_episodes": " ".join(str(ep) for ep in selected),
                }
                all_metrics.append(metrics)
                slug = method.lower().replace("-", "").replace(" ", "_")
                pd.DataFrame([metrics]).to_csv(
                    args.table_dir / f"budget_{slug}_{budget_percent}pct_seed{seed}_metrics.csv",
                    index=False,
                )
                np.savez_compressed(
                    args.prediction_dir / f"budget_{slug}_{budget_percent}pct_seed{seed}.npz",
                    val_pred=result["val_pred"],
                    test_pred=result["test_pred"],
                    val_target=result["val_target"],
                    test_target=result["test_target"],
                )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = args.table_dir / "budget_curve_metrics.csv"
    summary_path = args.table_dir / "budget_curve_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary = _summarize(metrics_df)
    summary.to_csv(summary_path, index=False)
    print(metrics_path)
    print(summary_path)
    for path in make_budget_curve_figure(summary, args.figure_dir):
        print(path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
