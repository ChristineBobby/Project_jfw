from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vla_coreset.visualization.report_figures import COLORS, set_paper_style


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def build_selected_episode_explanations(
    frame_scores: pd.DataFrame,
    episode_scores: pd.DataFrame,
    selection_trace: pd.DataFrame,
    selected_episodes: list[int],
    top_k_frames: int = 5,
) -> pd.DataFrame:
    required_frames = {"episode_index", "frame_index", "phase_bin", "frame_score", "visual_delta", "action_delta", "gripper_delta"}
    required_episode = {
        "episode_index",
        "episode_score",
        "episode_pc_score",
        "episode_ras_score",
        "phase_coverage",
        "mean_visual_delta",
        "mean_action_delta",
    }
    required_trace = {"episode_index", "selection_order", "final_score", "diversity_score", "raw_diversity_distance"}
    for name, frame, required in [
        ("frame_scores", frame_scores, required_frames),
        ("episode_scores", episode_scores, required_episode),
        ("selection_trace", selection_trace, required_trace),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    episode_lookup = episode_scores.set_index("episode_index")
    trace_lookup = selection_trace.set_index("episode_index")
    rows: list[dict[str, Any]] = []
    for episode in selected_episodes:
        ep = int(episode)
        frames = frame_scores[frame_scores["episode_index"] == ep].sort_values("frame_score", ascending=False)
        if frames.empty:
            raise ValueError(f"No frame scores for selected episode {ep}")
        ep_row = episode_lookup.loc[ep]
        trace_row = trace_lookup.loc[ep]
        top = frames.head(top_k_frames)
        reason_parts = ["high PC-RAS score"]
        if float(trace_row["diversity_score"]) >= 0.75:
            reason_parts.append("strong coverage contribution")
        if float(ep_row["episode_ras_score"]) >= float(ep_row["episode_pc_score"]) * 0.55:
            reason_parts.append("clear robot-action saliency")
        rows.append(
            {
                "episode_index": ep,
                "selection_order": int(trace_row["selection_order"]),
                "episode_score": float(ep_row["episode_score"]),
                "episode_pc_score": float(ep_row["episode_pc_score"]),
                "episode_ras_score": float(ep_row["episode_ras_score"]),
                "final_selection_score": float(trace_row["final_score"]),
                "diversity_score": float(trace_row["diversity_score"]),
                "diversity_distance": float(trace_row["raw_diversity_distance"]),
                "phase_coverage": int(ep_row["phase_coverage"]),
                "mean_visual_delta": float(ep_row["mean_visual_delta"]),
                "mean_action_delta": float(ep_row["mean_action_delta"]),
                "mean_gripper_delta": float(frames["gripper_delta"].mean()),
                "top_frame_indices": " ".join(str(int(v)) for v in top["frame_index"].tolist()),
                "selected_reason": "; ".join(reason_parts),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_order").reset_index(drop=True)


def _phase_for_test_rows(index: pd.DataFrame, phase_bins: int) -> tuple[np.ndarray, list[str]]:
    test_index = index[index["split"] == "test"].copy().reset_index(drop=True)
    max_frame = test_index.groupby("episode_index")["frame_index"].transform("max").to_numpy(dtype=np.float32)
    frame_index = test_index["frame_index"].to_numpy(dtype=np.float32)
    phase = np.divide(frame_index, np.maximum(max_frame, 1.0), out=np.zeros_like(frame_index), where=max_frame > 0)
    bins = np.minimum((phase * phase_bins).astype(np.int64), phase_bins - 1)
    labels = [f"{int(i * 100 / phase_bins)}-{int((i + 1) * 100 / phase_bins)}%" for i in range(phase_bins)]
    return bins, labels


def compute_phase_mse_summary(
    index: pd.DataFrame,
    predictions_by_method: dict[str, list[dict[str, Any]]],
    phase_bins: int = 5,
) -> pd.DataFrame:
    if phase_bins <= 0:
        raise ValueError("phase_bins must be positive")
    bins, labels = _phase_for_test_rows(index, phase_bins)
    rows: list[dict[str, float | int | str]] = []
    for method, runs in predictions_by_method.items():
        for run in runs:
            pred = np.asarray(run["test_pred"], dtype=np.float32)
            target = np.asarray(run["test_target"], dtype=np.float32)
            if pred.shape != target.shape:
                raise ValueError(f"{method} seed {run.get('seed')} prediction/target shape mismatch")
            if len(pred) != len(bins):
                raise ValueError(f"{method} seed {run.get('seed')} test rows mismatch: {len(pred)} vs {len(bins)}")
            per_frame_mse = ((pred - target) ** 2).mean(axis=1)
            for phase_bin in range(phase_bins):
                mask = bins == phase_bin
                rows.append(
                    {
                        "method": method,
                        "seed": int(run.get("seed", -1)),
                        "phase_bin": phase_bin,
                        "phase_label": labels[phase_bin],
                        "num_frames": int(mask.sum()),
                        "mse": float(per_frame_mse[mask].mean()),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["method", "phase_bin", "phase_label"], as_index=False)
        .agg(num_frames=("num_frames", "max"), mse_mean=("mse", "mean"), mse_std=("mse", "std"))
        .sort_values(["method", "phase_bin"])
        .reset_index(drop=True)
    )
    summary["mse_std"] = summary["mse_std"].fillna(0.0)
    return summary


def make_score_timeline_figure(frame_scores: pd.DataFrame, episode_index: int, output_dir: Path) -> list[Path]:
    required = {"episode_index", "frame_index", "phase", "frame_score", "pc_score", "ras_score"}
    missing = required.difference(frame_scores.columns)
    if missing:
        raise ValueError(f"frame_scores is missing required columns: {sorted(missing)}")
    subset = frame_scores[frame_scores["episode_index"] == int(episode_index)].sort_values("frame_index")
    if subset.empty:
        raise ValueError(f"No frame scores for episode {episode_index}")

    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.75), constrained_layout=True)
    x = subset["frame_index"].to_numpy()
    for start, end in [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]:
        lo = subset["frame_index"].min() + start * (subset["frame_index"].max() - subset["frame_index"].min())
        hi = subset["frame_index"].min() + end * (subset["frame_index"].max() - subset["frame_index"].min())
        ax.axvspan(lo, hi, color="#EEF1F5", alpha=0.65, linewidth=0)
    ax.plot(x, subset["frame_score"], color=COLORS["final"], linewidth=1.8, label="PC-RAS")
    ax.plot(x, subset["pc_score"], color=COLORS["saliency"], linewidth=1.2, label="PC")
    ax.plot(x, subset["ras_score"], color=COLORS["diversity"], linewidth=1.2, label="RAS")
    ax.set_title(f"Score timeline, episode {int(episode_index)}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", ncol=3, frameon=True, borderpad=0.35)
    return _save_figure(fig, output_dir, f"score_timeline_episode_{int(episode_index)}")


def make_action_curve_figure(action_rows: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"episode_index", "frame_index", "group", "action_0", "action_1", "action_2", "action_6"}
    missing = required.difference(action_rows.columns)
    if missing:
        raise ValueError(f"action_rows is missing required columns: {sorted(missing)}")

    set_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.7), constrained_layout=True, sharex=False)
    action_cols = [("action_0", "a0"), ("action_1", "a1"), ("action_2", "a2"), ("action_6", "gripper")]
    colors = {"Random": COLORS["random"], "PC-RAS": COLORS["pc_ras"]}
    for ax, (column, label) in zip(axes.flat, action_cols, strict=True):
        for group, group_frame in action_rows.groupby("group", sort=False):
            per_frame = group_frame.groupby("frame_index", as_index=False)[column].mean().sort_values("frame_index")
            ax.plot(per_frame["frame_index"], per_frame[column], color=colors.get(str(group), "#555555"), linewidth=1.6, label=str(group))
        ax.set_title(label)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Action value")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    return _save_figure(fig, output_dir, "selected_episode_action_curves")


def make_phase_mse_figure(phase_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"method", "phase_bin", "phase_label", "mse_mean", "mse_std"}
    missing = required.difference(phase_summary.columns)
    if missing:
        raise ValueError(f"phase_summary is missing required columns: {sorted(missing)}")

    set_paper_style()
    methods = list(dict.fromkeys(phase_summary["method"].tolist()))
    labels = (
        phase_summary[["phase_bin", "phase_label"]]
        .drop_duplicates()
        .sort_values("phase_bin")["phase_label"]
        .tolist()
    )
    x = np.arange(len(labels))
    width = 0.34 if len(methods) == 2 else 0.25
    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    for i, method in enumerate(methods):
        subset = phase_summary[phase_summary["method"] == method].sort_values("phase_bin")
        offset = (i - (len(methods) - 1) / 2.0) * width
        color = COLORS["pc_ras"] if "PC" in method else COLORS["random"]
        ax.bar(
            x + offset,
            subset["mse_mean"],
            yerr=subset["mse_std"],
            width=width,
            color=color,
            edgecolor="#2F2F2F",
            linewidth=0.6,
            capsize=2.5,
            label=method,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Episode phase")
    ax.set_ylabel("Test MSE")
    ax.set_title("Phase-aware test error")
    ax.legend(loc="upper right", frameon=True)
    return _save_figure(fig, output_dir, "phase_mse_bar_chart")


def make_budget_curve_figure(budget_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"method", "budget_percent", "test_original_mse_mean", "test_original_mse_std"}
    missing = required.difference(budget_summary.columns)
    if missing:
        raise ValueError(f"budget_summary is missing required columns: {sorted(missing)}")

    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.7, 3.4), constrained_layout=True)
    for method, subset in budget_summary.groupby("method", sort=False):
        subset = subset.sort_values("budget_percent")
        color = COLORS["pc_ras"] if "PC" in str(method) else COLORS["random"]
        ax.errorbar(
            subset["budget_percent"],
            subset["test_original_mse_mean"],
            yerr=subset["test_original_mse_std"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=str(method),
        )
    ax.set_xlabel("Candidate-train budget (%)")
    ax.set_ylabel("Test MSE, original action space")
    ax.set_title("Budget curve")
    ax.legend(loc="upper right", frameon=True)
    return _save_figure(fig, output_dir, "budget_curve")
