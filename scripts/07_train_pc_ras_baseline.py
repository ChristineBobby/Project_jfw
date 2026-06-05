#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vla_coreset.training.mlp_baseline import (
    load_feature_artifacts,
    make_feature_matrix,
    train_mlp_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP MLP on PC-RAS episode coreset.")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/clip_vit_b32_top_left7"))
    parser.add_argument("--coreset-file", type=Path, default=Path("data/coresets/pc_ras_episode_top5.json"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Use fewer epochs for quick pipeline checks.")
    return parser.parse_args()


def _load_pc_ras_coreset(path: Path) -> tuple[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [int(ep) for ep in payload["selected_episodes"]]
    method = str(payload.get("method", "pc_ras_diversity_greedy"))
    if not selected:
        raise ValueError("PC-RAS coreset has no selected episodes")
    return method, selected


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    method, selected = _load_pc_ras_coreset(args.coreset_file)
    features, actions, text_features, index = load_feature_artifacts(args.feature_dir)
    x = make_feature_matrix(features, text_features)
    max_epochs = min(args.max_epochs, 5) if args.smoke else args.max_epochs
    patience = min(args.patience, 3) if args.smoke else args.patience

    all_metrics: list[dict] = []
    for seed in tqdm(args.seeds, desc="pc-ras baseline seeds"):
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
            "coreset_file": str(args.coreset_file),
            **result["metrics"],
            "selected_episodes": " ".join(str(ep) for ep in selected),
        }
        all_metrics.append(metrics)
        pd.DataFrame([metrics]).to_csv(args.table_dir / f"pc_ras_clip_seed{seed}_metrics.csv", index=False)
        np.savez_compressed(
            args.prediction_dir / f"pc_ras_clip_seed{seed}.npz",
            val_pred=result["val_pred"],
            test_pred=result["test_pred"],
            val_target=result["val_target"],
            test_target=result["test_target"],
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.table_dir / "pc_ras_clip_summary.csv", index=False)
    summary = metrics_df[["test_original_mse", "test_normalized_mse"]].agg(["mean", "std", "min", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
