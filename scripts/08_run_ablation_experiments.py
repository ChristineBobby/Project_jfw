#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vla_coreset.selection.ablation import (
    ABLATION_VARIANTS,
    aggregate_ablation_summary,
    build_ablation_episode_scores,
    select_ablation_coreset,
    write_ablation_coreset_json,
)
from vla_coreset.selection.episode_selector import build_episode_embeddings
from vla_coreset.training.mlp_baseline import (
    load_feature_artifacts,
    make_feature_matrix,
    train_mlp_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PC-RAS selector ablation experiments.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--frame-score-file", type=Path, default=Path("data/cache/selection/pc_ras_frame_scores.parquet"))
    parser.add_argument("--coreset-dir", type=Path, default=Path("data/coresets"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--variants", nargs="+", default=ABLATION_VARIANTS, choices=ABLATION_VARIANTS)
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


def main() -> None:
    args = parse_args()
    args.coreset_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    features, actions, text_features, index = load_feature_artifacts(args.feature_dir)
    x = make_feature_matrix(features, text_features)
    frame_scores = pd.read_parquet(args.frame_score_file)
    train_mask = (index["split"] == "candidate_train").to_numpy()
    embeddings = build_episode_embeddings(
        features=features[train_mask],
        actions=actions[train_mask],
        index=index.loc[train_mask].reset_index(drop=True),
    )
    max_epochs = min(args.max_epochs, 5) if args.smoke else args.max_epochs
    patience = min(args.patience, 3) if args.smoke else args.patience

    all_metrics: list[dict] = []
    for variant in tqdm(args.variants, desc="ablation variants"):
        episode_scores = build_ablation_episode_scores(frame_scores, variant)
        selected, trace = select_ablation_coreset(variant, episode_scores, embeddings, budget=args.budget)
        coreset_path = args.coreset_dir / f"ablation_{variant}_top{args.budget}.json"
        trace_path = args.table_dir / f"ablation_{variant}_selection_trace.csv"
        write_ablation_coreset_json(
            path=coreset_path,
            variant=variant,
            selected_episodes=selected,
            trace=trace,
            source_frame_score_file=str(args.frame_score_file),
        )
        trace.to_csv(trace_path, index=False)

        for seed in tqdm(args.seeds, desc=f"{variant} seeds", leave=False):
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
                "method": f"ablation_{variant}",
                "variant": variant,
                "coreset_file": str(coreset_path),
                **result["metrics"],
                "selected_episodes": " ".join(str(ep) for ep in selected),
            }
            all_metrics.append(metrics)
            pd.DataFrame([metrics]).to_csv(args.table_dir / f"ablation_{variant}_seed{seed}_metrics.csv", index=False)
            np.savez_compressed(
                args.prediction_dir / f"ablation_{variant}_seed{seed}.npz",
                val_pred=result["val_pred"],
                test_pred=result["test_pred"],
                val_target=result["val_target"],
                test_target=result["test_target"],
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = args.table_dir / "ablation_metrics.csv"
    summary_path = args.table_dir / "ablation_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary = aggregate_ablation_summary(metrics_df)
    summary.to_csv(summary_path, index=False)
    print(f"metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
