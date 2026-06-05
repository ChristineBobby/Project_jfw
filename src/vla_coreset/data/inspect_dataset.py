from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
DEFAULT_DATASET_ROOT = Path("data/cache/lerobot/aloha_sim_transfer_cube_human")
DEFAULT_TABLE_DIR = Path("results/tables")
DEFAULT_FIGURE_DIR = Path("results/figures")


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    episode_dir = dataset_root / "meta" / "episodes"
    files = sorted(episode_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files under {episode_dir}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def load_task_texts(dataset_root: Path) -> list[str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks metadata: {tasks_path}")
    tasks = pd.read_parquet(tasks_path)
    return [str(index) for index in tasks.index.tolist()]


def episode_length_stats(episodes: pd.DataFrame) -> dict[str, int]:
    required = {"episode_index", "length"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"Episode metadata missing required columns: {sorted(missing)}")

    return {
        "episode_count": int(episodes["episode_index"].nunique()),
        "min_episode_index": int(episodes["episode_index"].min()),
        "max_episode_index": int(episodes["episode_index"].max()),
        "min_frames_per_episode": int(episodes["length"].min()),
        "max_frames_per_episode": int(episodes["length"].max()),
    }


def _shape_text(feature: dict[str, Any]) -> str:
    return "x".join(str(dim) for dim in feature.get("shape", []))


def _feature_dim(feature: dict[str, Any]) -> str:
    shape = feature.get("shape", [])
    if not shape:
        return ""
    if len(shape) == 1:
        return str(shape[0])
    return _shape_text(feature)


def build_summary_rows(
    info: dict[str, Any],
    episode_stats: dict[str, int],
    task_texts: list[str],
) -> list[dict[str, str]]:
    features = info["features"]
    image_key = "observation.images.top"
    rows = [
        {"metric": "repo_id", "value": DEFAULT_REPO_ID},
        {"metric": "codebase_version", "value": str(info.get("codebase_version", ""))},
        {"metric": "robot_type", "value": str(info.get("robot_type", ""))},
        {"metric": "total_episodes", "value": str(info["total_episodes"])},
        {"metric": "total_frames", "value": str(info["total_frames"])},
        {"metric": "total_tasks", "value": str(info["total_tasks"])},
        {"metric": "fps", "value": str(info["fps"])},
        {"metric": "episode_count_from_meta", "value": str(episode_stats["episode_count"])},
        {"metric": "episode_index_min", "value": str(episode_stats["min_episode_index"])},
        {"metric": "episode_index_max", "value": str(episode_stats["max_episode_index"])},
        {"metric": "frames_per_episode_min", "value": str(episode_stats["min_frames_per_episode"])},
        {"metric": "frames_per_episode_max", "value": str(episode_stats["max_frames_per_episode"])},
        {"metric": "image_key", "value": image_key},
        {"metric": "image_dtype", "value": str(features[image_key].get("dtype", ""))},
        {"metric": "image_shape", "value": _shape_text(features[image_key])},
        {"metric": "state_dim", "value": _feature_dim(features["observation.state"])},
        {"metric": "action_dim", "value": _feature_dim(features["action"])},
        {"metric": "state_dtype", "value": str(features["observation.state"].get("dtype", ""))},
        {"metric": "action_dtype", "value": str(features["action"].get("dtype", ""))},
    ]
    rows.extend({"metric": f"task_{i}", "value": task} for i, task in enumerate(task_texts))
    return rows


def motor_names_from_info(info: dict[str, Any]) -> list[str]:
    names = info["features"]["action"].get("names", {}).get("motors", [])
    return [str(name) for name in names]


def action_stats_from_episodes(episodes: pd.DataFrame) -> dict[str, list[float]]:
    required = ["stats/action/min", "stats/action/max"]
    missing = [column for column in required if column not in episodes.columns]
    if missing:
        raise ValueError(f"Episode metadata missing action stats columns: {missing}")

    per_episode_min = torch.from_numpy(np.stack(episodes["stats/action/min"].to_numpy())).float()
    per_episode_max = torch.from_numpy(np.stack(episodes["stats/action/max"].to_numpy())).float()
    action_min = per_episode_min.min(dim=0).values
    action_max = per_episode_max.max(dim=0).values
    return {
        "stats/action/min": [float(value) for value in action_min.tolist()],
        "stats/action/max": [float(value) for value in action_max.tolist()],
    }


def action_range_rows(stats: dict[str, list[float]], motor_names: list[str]) -> list[dict[str, str]]:
    mins = stats["stats/action/min"]
    maxs = stats["stats/action/max"]
    if len(mins) != len(maxs):
        raise ValueError("Action min/max lengths differ")
    if len(motor_names) != len(mins):
        raise ValueError(f"Expected {len(mins)} motor names, got {len(motor_names)}")

    return [
        {"motor": motor, "min": f"{minimum:.6f}", "max": f"{maximum:.6f}"}
        for motor, minimum, maximum in zip(motor_names, mins, maxs, strict=True)
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tensor_image_to_numpy(image: torch.Tensor) -> Any:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    image = image.detach().cpu().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def save_sample_figure(
    dataset: LeRobotDataset,
    output_path: Path,
    sample_indices: list[int],
    image_key: str = "observation.images.top",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(sample_indices))
    rows = (len(sample_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")

    for axis, index in zip(axes.flat, sample_indices, strict=False):
        sample = dataset[index]
        axis.imshow(_tensor_image_to_numpy(sample[image_key]))
        episode = int(sample["episode_index"])
        frame = int(sample["frame_index"])
        axis.set_title(f"episode {episode}, frame {frame}", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def sample_indices_for_episodes(episodes: pd.DataFrame, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if episodes.empty:
        raise ValueError("episodes metadata is empty")

    selected = episodes.sort_values("episode_index").head(count)
    indices: list[int] = []
    for _, row in selected.iterrows():
        start = int(row["dataset_from_index"])
        length = int(row["length"])
        indices.append(start + length // 2)
    return indices


def inspect_dataset(
    repo_id: str,
    dataset_root: Path,
    table_dir: Path,
    figure_dir: Path,
    sample_count: int,
) -> dict[str, Path]:
    info = load_info(dataset_root)
    episodes = load_episode_metadata(dataset_root)
    tasks = load_task_texts(dataset_root)
    ep_stats = episode_length_stats(episodes)
    summary_rows = build_summary_rows(info, ep_stats, tasks)
    action_rows = action_range_rows(action_stats_from_episodes(episodes), motor_names_from_info(info))

    summary_csv = table_dir / "dataset_summary.csv"
    action_csv = table_dir / "action_ranges.csv"
    summary_json = table_dir / "dataset_summary.json"
    figure_path = figure_dir / "dataset_samples.png"

    write_csv(summary_csv, summary_rows, ["metric", "value"])
    write_csv(action_csv, action_rows, ["motor", "min", "max"])
    write_json(
        summary_json,
        {
            "summary": {row["metric"]: row["value"] for row in summary_rows},
            "action_ranges": action_rows,
        },
    )

    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)
    save_sample_figure(dataset, figure_path, sample_indices_for_episodes(episodes, sample_count))

    return {
        "summary_csv": summary_csv,
        "action_csv": action_csv,
        "summary_json": summary_json,
        "sample_figure": figure_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the cached ALOHA LeRobot dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = inspect_dataset(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        sample_count=args.sample_count,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
