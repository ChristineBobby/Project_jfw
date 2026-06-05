from __future__ import annotations

import math

import numpy as np
import pandas as pd


def robust_normalize(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.percentile(values, lower))
    hi = float(np.percentile(values, upper))
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _episode_delta(values: np.ndarray, episodes: np.ndarray, order: np.ndarray = 1) -> np.ndarray:
    deltas = np.zeros(len(values), dtype=np.float32)
    for episode in np.unique(episodes):
        positions = np.flatnonzero(episodes == episode)
        if len(positions) <= order:
            continue
        episode_values = values[positions]
        if order == 1:
            diff = episode_values[1:] - episode_values[:-1]
            deltas[positions[1:]] = np.linalg.norm(diff, axis=1)
        elif order == 2:
            diff = episode_values[2:] - 2.0 * episode_values[1:-1] + episode_values[:-2]
            deltas[positions[2:]] = np.linalg.norm(diff, axis=1)
        else:
            raise ValueError("order must be 1 or 2")
    return deltas


def _episode_abs_delta(values: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    deltas = np.zeros(len(values), dtype=np.float32)
    for episode in np.unique(episodes):
        positions = np.flatnonzero(episodes == episode)
        if len(positions) <= 1:
            continue
        episode_values = values[positions]
        deltas[positions[1:]] = np.abs(episode_values[1:] - episode_values[:-1])
    return deltas


def compute_frame_scores(
    features: np.ndarray,
    actions: np.ndarray,
    index: pd.DataFrame,
    phase_bins: int = 5,
) -> pd.DataFrame:
    features = np.asarray(features, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected features [N,D], got {features.shape}")
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected actions [N,>=7], got {actions.shape}")
    if len(index) != len(features) or len(index) != len(actions):
        raise ValueError("features, actions, and index must have the same number of rows")
    required = {"episode_index", "frame_index", "split"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"index is missing required columns: {sorted(missing)}")

    out = index[["episode_index", "frame_index", "split"]].copy().reset_index(drop=True)
    episodes = out["episode_index"].to_numpy()
    frame_index = out["frame_index"].to_numpy(dtype=np.float32)
    max_frame = out.groupby("episode_index")["frame_index"].transform("max").to_numpy(dtype=np.float32)
    phase = np.divide(frame_index, np.maximum(max_frame, 1.0), out=np.zeros_like(frame_index), where=max_frame > 0)
    phase_bin = np.minimum((phase * phase_bins).astype(np.int64), phase_bins - 1)

    visual_delta = _episode_delta(features, episodes, order=1)
    action_delta = _episode_delta(actions[:, :7], episodes, order=1)
    action_jerk = _episode_delta(actions[:, :7], episodes, order=2)
    gripper_delta = _episode_abs_delta(actions[:, 6], episodes)
    phase_balance_base = np.ones(len(out), dtype=np.float32) / float(phase_bins)

    visual_score = robust_normalize(visual_delta)
    action_delta_score = robust_normalize(action_delta)
    action_jerk_score = robust_normalize(action_jerk)
    gripper_score = robust_normalize(gripper_delta)
    pc_score = visual_score + 0.7 * action_delta_score + 0.5 * action_jerk_score
    ras_score = gripper_score + 0.7 * action_delta_score + 0.3 * phase_balance_base
    frame_score = 0.55 * pc_score + 0.45 * ras_score

    out["phase"] = phase.astype(np.float32)
    out["phase_bin"] = phase_bin
    out["visual_delta"] = visual_delta
    out["action_delta"] = action_delta
    out["action_jerk"] = action_jerk
    out["gripper_delta"] = gripper_delta
    out["visual_score"] = visual_score
    out["action_delta_score"] = action_delta_score
    out["action_jerk_score"] = action_jerk_score
    out["gripper_score"] = gripper_score
    out["pc_score"] = pc_score.astype(np.float32)
    out["ras_score"] = ras_score.astype(np.float32)
    out["frame_score"] = frame_score.astype(np.float32)
    return out


def aggregate_episode_scores(frame_scores: pd.DataFrame, top_fraction: float = 0.2) -> pd.DataFrame:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    required = {
        "episode_index",
        "frame_score",
        "pc_score",
        "ras_score",
        "action_delta",
        "visual_delta",
        "phase_bin",
    }
    missing = required.difference(frame_scores.columns)
    if missing:
        raise ValueError(f"frame_scores is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | int]] = []
    for episode, group in frame_scores.groupby("episode_index", sort=True):
        top_count = max(1, int(math.ceil(len(group) * top_fraction)))
        top = group.nlargest(top_count, "frame_score")
        rows.append(
            {
                "episode_index": int(episode),
                "frames": int(len(group)),
                "top_frames": int(top_count),
                "episode_score": float(top["frame_score"].mean()),
                "episode_pc_score": float(top["pc_score"].mean()),
                "episode_ras_score": float(top["ras_score"].mean()),
                "mean_frame_score": float(group["frame_score"].mean()),
                "mean_action_delta": float(group["action_delta"].mean()),
                "mean_visual_delta": float(group["visual_delta"].mean()),
                "phase_coverage": int(group["phase_bin"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
