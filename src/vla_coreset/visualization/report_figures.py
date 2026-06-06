from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTION_LABELS = [f"a{i}" for i in range(7)]
COLORS = {
    "random": "#5B8DB8",
    "pc_ras": "#D95F4D",
    "candidate": "#B8BEC8",
    "final": "#2D4F7C",
    "saliency": "#D95F4D",
    "diversity": "#4C9A6A",
    "ablation_baseline": "#8C94A3",
    "ablation_component": "#5B8DB8",
    "ablation_full": "#D95F4D",
}


@dataclass(frozen=True)
class FigureInputs:
    random_summary: pd.DataFrame
    pc_ras_summary: pd.DataFrame
    episode_scores: pd.DataFrame
    selection_trace: pd.DataFrame
    coverage_points: pd.DataFrame
    output_dir: Path


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_method_summary(random_summary: pd.DataFrame, pc_ras_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in [("Random-10%", random_summary), ("PC-RAS", pc_ras_summary)]:
        rows.append(
            {
                "method": method,
                "test_original_mse_mean": float(frame["test_original_mse"].mean()),
                "test_original_mse_std": float(frame["test_original_mse"].std()),
                "test_normalized_mse_mean": float(frame["test_normalized_mse"].mean()),
                "test_normalized_mse_std": float(frame["test_normalized_mse"].std()),
            }
        )
    out = pd.DataFrame(rows)
    random_mean = out.loc[out["method"] == "Random-10%", "test_original_mse_mean"].iloc[0]
    pc_mean = out.loc[out["method"] == "PC-RAS", "test_original_mse_mean"].iloc[0]
    out.attrs["relative_improvement"] = float((random_mean - pc_mean) / random_mean)
    return out


def per_dimension_mse_table(random_summary: pd.DataFrame, pc_ras_summary: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for method, frame in [("Random-10%", random_summary), ("PC-RAS", pc_ras_summary)]:
        values = []
        for i in range(7):
            col = f"test_original_action_{i}_mse"
            if col not in frame.columns:
                raise ValueError(f"Missing per-dimension column: {col}")
            values.append(float(frame[col].mean()))
        rows[method] = values
    return pd.DataFrame.from_dict(rows, orient="index", columns=ACTION_LABELS)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def make_ablation_figure(ablation_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"variant", "test_original_mse_mean", "test_original_mse_std"}
    missing = required.difference(ablation_summary.columns)
    if missing:
        raise ValueError(f"Missing ablation summary columns: {sorted(missing)}")

    set_paper_style()
    labels = {
        "action_delta_only": "ActionDelta-only",
        "visual_delta_only": "VisualDelta-only",
        "pc_only": "PC-only",
        "pc_ras_no_coverage": "PC+RAS",
        "coverage_only": "Coverage-only",
        "pc_ras_full": "PC+RAS+Coverage",
    }
    order = [
        "visual_delta_only",
        "action_delta_only",
        "coverage_only",
        "pc_only",
        "pc_ras_no_coverage",
        "pc_ras_full",
    ]
    summary = ablation_summary.set_index("variant").loc[order].reset_index()
    y = np.arange(len(summary))
    means = summary["test_original_mse_mean"].astype(float).to_numpy()
    stds = summary["test_original_mse_std"].astype(float).fillna(0.0).to_numpy()
    row_colors = [
        COLORS["ablation_full"] if variant == "pc_ras_full" else COLORS["ablation_component"]
        for variant in summary["variant"]
    ]
    row_colors[0] = COLORS["ablation_baseline"]
    row_colors[1] = COLORS["ablation_baseline"]
    row_colors[2] = COLORS["ablation_baseline"]

    fig, ax = plt.subplots(figsize=(6.7, 3.55), constrained_layout=True)
    ax.errorbar(
        means,
        y,
        xerr=stds,
        fmt="none",
        ecolor="#31343A",
        elinewidth=1.15,
        capsize=3,
        capthick=1.0,
        zorder=2,
    )
    ax.scatter(means, y, s=52, c=row_colors, edgecolor="#2F2F2F", linewidth=0.55, zorder=3)
    ax.set_yticks(y, [labels[variant] for variant in summary["variant"]])
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE, original action space")
    ax.set_title("Ablation of PC-RAS selection components")
    ax.grid(axis="x", alpha=0.26)
    ax.grid(axis="y", visible=False)
    xmin = max(0.0, float((means - stds).min()) * 0.88)
    xmax = float((means + stds).max()) * 1.12
    ax.set_xlim(xmin, xmax)
    value_offset = (xmax - xmin) * 0.018
    for xpos, ypos, err in zip(means, y, stds):
        ax.text(
            xpos + err + value_offset,
            ypos,
            f"{xpos:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#2A2D33",
        )
    return _save_figure(fig, output_dir, "ablation_mse_comparison")


def _plot_main_mse(inputs: FigureInputs) -> list[Path]:
    summary = build_method_summary(inputs.random_summary, inputs.pc_ras_summary)
    improvement = summary.attrs["relative_improvement"] * 100.0
    fig, ax = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)
    x = np.arange(len(summary))
    means = summary["test_original_mse_mean"].to_numpy()
    stds = summary["test_original_mse_std"].to_numpy()
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
    ax.set_xticks(x, summary["method"].tolist())
    ax.set_ylabel("Test MSE, original action space")
    ax.set_title("PC-RAS improves 10% episode training")
    ax.set_ylim(0.0, float((means + stds).max() * 1.32))
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + float(stds.max()) * 0.18,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.text(
        0.5,
        0.94,
        f"{improvement:.1f}% lower mean MSE",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#2D4F7C",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D4DC", "alpha": 0.95},
    )
    return _save_figure(fig, inputs.output_dir, "main_mse_comparison")


def _plot_per_dimension_heatmap(inputs: FigureInputs) -> list[Path]:
    table = per_dimension_mse_table(inputs.random_summary, inputs.pc_ras_summary)
    fig, ax = plt.subplots(figsize=(5.8, 2.45), constrained_layout=True)
    image = ax.imshow(table.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(table.index)), table.index.tolist())
    ax.set_xticks(np.arange(len(table.columns)), table.columns.tolist())
    ax.set_xlabel("Action dimension")
    ax.set_title("Per-dimension test MSE")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Original-space MSE")
    return _save_figure(fig, inputs.output_dir, "per_dimension_mse_heatmap")


def _plot_episode_ranking(inputs: FigureInputs) -> list[Path]:
    scores = inputs.episode_scores.sort_values("rank").reset_index(drop=True)
    selected = set(inputs.selection_trace["episode_index"].astype(int).tolist())
    colors = [COLORS["pc_ras"] if int(ep) in selected else COLORS["candidate"] for ep in scores["episode_index"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.3), constrained_layout=True)
    x = np.arange(len(scores))
    y = scores["episode_score"].to_numpy()
    ax.vlines(x, y.min() * 0.98, y, color=colors, linewidth=1.3, alpha=0.9)
    ax.scatter(x, y, c=colors, s=35, edgecolor="#333333", linewidth=0.4, zorder=3)
    ax.set_xlabel("Candidate episodes ranked by PC-RAS score")
    ax.set_ylabel("Episode score")
    ax.set_title("PC-RAS episode ranking")
    ax.set_xticks(x[::4], scores["episode_index"].astype(int).iloc[::4].tolist())
    ax.margins(x=0.02)
    for _, row in scores[scores["episode_index"].astype(int).isin(selected)].iterrows():
        xpos = int(row["rank"]) - 1
        ax.annotate(
            str(int(row["episode_index"])),
            xy=(xpos, row["episode_score"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2A2A2A",
        )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["candidate"], markeredgecolor="#333333", label="Candidate"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["pc_ras"], markeredgecolor="#333333", label="Selected"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, borderpad=0.5)
    return _save_figure(fig, inputs.output_dir, "episode_score_ranking")


def _plot_selection_trace(inputs: FigureInputs) -> list[Path]:
    trace = inputs.selection_trace.sort_values("selection_order").reset_index(drop=True)
    x = trace["selection_order"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    ax.plot(x, trace["final_score"], marker="o", color=COLORS["final"], linewidth=2.0, label="Final")
    ax.plot(x, trace["saliency_score"], marker="s", color=COLORS["saliency"], linewidth=1.6, label="Saliency")
    ax.plot(x, trace["diversity_score"], marker="^", color=COLORS["diversity"], linewidth=1.6, label="Diversity")
    ax.set_xlabel("Greedy selection step")
    ax.set_ylabel("Normalized score")
    ax.set_title("Selection trace balances saliency and coverage")
    ax.set_xticks(x, [f"{int(step)}\n(ep {int(ep)})" for step, ep in zip(x, trace["episode_index"])])
    ax.set_ylim(0.0, 1.08)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D4DC",
        handlelength=2.0,
    )
    return _save_figure(fig, inputs.output_dir, "selection_trace")


def _plot_coverage_pca(inputs: FigureInputs) -> list[Path]:
    points = inputs.coverage_points.copy()
    fig, ax = plt.subplots(figsize=(5.7, 4.2), constrained_layout=True)
    for group, color, size, alpha, zorder in [
        ("Candidate", COLORS["candidate"], 38, 0.7, 1),
        ("Random", COLORS["random"], 75, 0.92, 2),
        ("PC-RAS", COLORS["pc_ras"], 95, 0.98, 3),
    ]:
        subset = points[points["group"] == group]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            s=size,
            c=color,
            label=group,
            alpha=alpha,
            edgecolor="#2F2F2F" if group != "Candidate" else "none",
            linewidth=0.6,
            zorder=zorder,
        )
    offsets = [(-28, 8), (10, -12), (-18, 10), (14, 8), (12, -14)]
    for offset, (_, row) in zip(offsets, points[points["group"] == "PC-RAS"].iterrows()):
        ax.annotate(
            str(int(row["episode_index"])),
            xy=(row["x"], row["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color="#2F2F2F",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Episode-level coverage in CLIP-action space")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D4DC",
        borderpad=0.45,
    )
    return _save_figure(fig, inputs.output_dir, "coverage_pca")


def make_report_figures(inputs: FigureInputs) -> list[Path]:
    set_paper_style()
    written: list[Path] = []
    written.extend(_plot_main_mse(inputs))
    written.extend(_plot_per_dimension_heatmap(inputs))
    written.extend(_plot_episode_ranking(inputs))
    written.extend(_plot_selection_trace(inputs))
    written.extend(_plot_coverage_pca(inputs))
    return written
