from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from vla_coreset.features.extract_clip import (
    DATASET_ROOT,
    IMAGE_KEY,
    REPO_ID,
    SPLIT_FILE,
    build_index_frame,
    choose_device,
    ensure_output_available,
    episode_to_split,
    left7_action,
    load_split,
    make_sample_order,
    FeatureRecord,
)


OUTPUT_DIR = Path("data/features/resnet18_top_left7")


def save_resnet_feature_artifacts(
    output_dir: Path,
    features: np.ndarray,
    actions_left7: np.ndarray,
    text_features: np.ndarray,
    index: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", features.astype(np.float32, copy=False))
    np.save(output_dir / "actions_left7.npy", actions_left7.astype(np.float32, copy=False))
    np.save(output_dir / "text_features.npy", text_features.astype(np.float32, copy=False))
    index.to_parquet(output_dir / "index.parquet", index=False)


def _make_feature_extractor(device: torch.device) -> tuple[torch.nn.Module, Any]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, weights.transforms()


def extract_resnet18_features(
    repo_id: str,
    dataset_root: Path,
    split_file: Path,
    output_dir: Path,
    batch_size: int,
    device_name: str,
    max_frames: int | None,
    stride: int,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_available(output_dir, overwrite=overwrite)
    split = load_split(split_file)
    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)
    indices = make_sample_order(len(dataset), max_frames=max_frames, stride=stride)
    if not indices:
        raise ValueError("No dataset indices selected for feature extraction")

    device = choose_device(device_name)
    model, preprocess = _make_feature_extractor(device)
    feature_batches: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    records: list[FeatureRecord] = []
    image_batch: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not image_batch:
            return
        batch = torch.stack(image_batch).to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = F.normalize(model(batch), dim=-1)
        feature_batches.append(encoded.detach().cpu().numpy().astype(np.float32))
        image_batch.clear()

    for row_index, dataset_index in enumerate(tqdm(indices, desc="resnet18 feature frames")):
        sample = dataset[dataset_index]
        episode_index = int(sample["episode_index"])
        action_left = left7_action(sample["action"])
        action_rows.append(action_left)
        records.append(
            FeatureRecord(
                row_index=row_index,
                episode_index=episode_index,
                frame_index=int(sample["frame_index"]),
                timestamp=float(sample["timestamp"]),
                split=episode_to_split(episode_index, split),
                action_left=action_left,
            )
        )
        image_batch.append(preprocess(sample[IMAGE_KEY]))
        if len(image_batch) >= batch_size:
            flush_batch()
    flush_batch()

    features = np.concatenate(feature_batches, axis=0)
    actions_left7 = np.stack(action_rows).astype(np.float32)
    text_features = np.zeros((1, features.shape[1]), dtype=np.float32)
    index = build_index_frame(records)
    save_resnet_feature_artifacts(output_dir, features, actions_left7, text_features, index)
    return {
        "rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions_left7.shape[1]),
        "text_feature_shape": list(text_features.shape),
        "output_dir": str(output_dir),
        "device": str(device),
        "model_name": "resnet18",
        "pretrained": "IMAGENET1K_V1",
        "stride": stride,
        "max_frames": max_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen ResNet18 image features for ALOHA.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_resnet18_features(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_frames=args.max_frames,
        stride=args.stride,
        overwrite=args.overwrite,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
