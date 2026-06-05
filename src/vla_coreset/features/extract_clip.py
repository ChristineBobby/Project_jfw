from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
DATASET_ROOT = Path("data/cache/lerobot/aloha_sim_transfer_cube_human")
SPLIT_FILE = Path("data/splits/split_v1.json")
OUTPUT_DIR = Path("data/features/clip_vit_b32_top_left7")
IMAGE_KEY = "observation.images.top"
TASK_TEXT = "transfer the cube"


@dataclass(frozen=True)
class FeatureRecord:
    row_index: int
    episode_index: int
    frame_index: int
    timestamp: float
    split: str
    action_left: np.ndarray


def load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def episode_to_split(episode_index: int, split: dict[str, Any]) -> str:
    for name in ("candidate_train", "val", "test"):
        if episode_index in set(split[name]):
            return name
    raise ValueError(f"Episode {episode_index} is not present in split file")


def left7_action(action: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.shape[0] < 7:
        raise ValueError(f"Expected action with at least 7 dims, got shape {action.shape}")
    return action[:7].astype(np.float32, copy=True)


def build_index_frame(records: list[FeatureRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.action_left.shape != (7,):
            raise ValueError(f"Expected left action shape (7,), got {record.action_left.shape}")
        row = {
            "row_index": int(record.row_index),
            "episode_index": int(record.episode_index),
            "frame_index": int(record.frame_index),
            "timestamp": float(record.timestamp),
            "split": record.split,
        }
        row.update({f"action_left_{i}": float(record.action_left[i]) for i in range(7)})
        rows.append(row)
    return pd.DataFrame(rows)


def save_feature_artifacts(
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


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = (array * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    outputs = ["features.npy", "actions_left7.npy", "text_features.npy", "index.parquet"]
    existing = [output_dir / name for name in outputs if (output_dir / name).exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; pass --overwrite to replace: {existing_text}")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_sample_order(dataset_len: int, max_frames: int | None, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = list(range(0, dataset_len, stride))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        indices = indices[:max_frames]
    return indices


def _encode_text(model: torch.nn.Module, tokenizer: Any, text: str, device: torch.device) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features.detach().cpu().numpy().astype(np.float32)


def extract_clip_features(
    repo_id: str,
    dataset_root: Path,
    split_file: Path,
    output_dir: Path,
    model_name: str,
    pretrained: str,
    task_text: str,
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
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    text_features = _encode_text(model, tokenizer, task_text, device)

    feature_batches: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    records: list[FeatureRecord] = []
    image_batch: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not image_batch:
            return
        batch = torch.stack(image_batch).to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = model.encode_image(batch)
            encoded = F.normalize(encoded, dim=-1)
        feature_batches.append(encoded.detach().cpu().numpy().astype(np.float32))
        image_batch.clear()

    for row_index, dataset_index in enumerate(indices):
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
        image_batch.append(preprocess(tensor_to_pil(sample[IMAGE_KEY])))
        if len(image_batch) >= batch_size:
            flush_batch()
    flush_batch()

    features = np.concatenate(feature_batches, axis=0)
    actions_left7 = np.stack(action_rows).astype(np.float32)
    index = build_index_frame(records)
    save_feature_artifacts(output_dir, features, actions_left7, text_features, index)

    return {
        "rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions_left7.shape[1]),
        "text_feature_shape": list(text_features.shape),
        "output_dir": str(output_dir),
        "device": str(device),
        "model_name": model_name,
        "pretrained": pretrained,
        "task_text": task_text,
        "stride": stride,
        "max_frames": max_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen CLIP image/text features for ALOHA.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--task-text", default=TASK_TEXT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_clip_features(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        task_text=args.task_text,
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
