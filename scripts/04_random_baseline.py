#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vla_coreset.selection.random_baseline import build_random_episode_coreset, write_coreset_json
from vla_coreset.training.mlp_baseline import (
    load_feature_artifacts,
    make_feature_matrix,
    train_mlp_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Random-10% CLIP MLP baseline.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/split_v1.json"))
    parser.add_argument("--coreset-dir", type=Path, default=Path("data/coresets"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
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


def main() -> None:
    args = parse_args()
    args.coreset_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    features, actions, text_features, index = load_feature_artifacts(args.feature_dir)
    x = make_feature_matrix(features, text_features)
    max_epochs = min(args.max_epochs, 5) if args.smoke else args.max_epochs
    patience = min(args.patience, 3) if args.smoke else args.patience

    all_metrics: list[dict] = []
    for seed in tqdm(args.seeds, desc="random baseline seeds"):
        selected = build_random_episode_coreset(split["candidate_train"], budget=args.budget, seed=seed)
        coreset_path = args.coreset_dir / f"random_episode_seed{seed}.json"
        write_coreset_json(coreset_path, seed=seed, selected_episodes=selected)
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
        metrics = result["metrics"]
        all_metrics.append(metrics)
        pd.DataFrame([metrics]).to_csv(args.table_dir / f"random10_clip_seed{seed}_metrics.csv", index=False)
        np.savez_compressed(
            args.prediction_dir / f"random10_clip_seed{seed}.npz",
            val_pred=result["val_pred"],
            test_pred=result["test_pred"],
            val_target=result["val_target"],
            test_target=result["test_target"],
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.table_dir / "random10_clip_summary.csv", index=False)
    summary = metrics_df[["test_original_mse", "test_normalized_mse"]].agg(["mean", "std", "min", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
