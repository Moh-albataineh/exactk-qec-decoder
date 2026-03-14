#!/usr/bin/env python3
"""Generate publication-ready bar charts for ExactK paper figures.

This script reads canonical all_results.json artifacts and produces:
- Figure 3: d=5 primary validation
- Figure 4: d=7 OOD transfer
- Figure 5: d=7 holdout validation

Expected JSON structure:
  {
    "Control_47000": [ {"epoch": 1, "G1_aligned": ...}, ... ],
    "ExactK_Tuned_47000": [...],
    ...
  }

Usage:
  python plot_figures.py \
      --day69 path/to/day69/all_results.json \
      --day70 path/to/day70/all_results.json \
      --day75 path/to/day75/all_results.json \
      --outdir paper_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Publication-friendly defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Control": "#4C566A",
    "ExactK": "#2E8B57",
    "Gated": "#C28E0E",
    "EarlyCutoff": "#B65E2E",
}

D5_SEEDS = [47000, 49000, 49200, 50000, 51000, 52000, 53000, 54000, 55000, 56000]
HOLDOUT_SEEDS = list(range(60000, 60010))


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def active_epoch_metric(records: Sequence[dict], metric: str = "G1_aligned", epoch_min: int = 6) -> float:
    vals = [float(r.get(metric, r.get("G1_raw_probe", 0.0))) for r in records if int(r.get("epoch", 0)) >= epoch_min]
    if not vals:
        raise ValueError(f"No active epochs >= {epoch_min} with metric {metric}")
    return float(np.median(vals))


def extract_arm(data: dict, arm_prefix: str, seeds: Iterable[int], metric: str = "G1_aligned", epoch_min: int = 6) -> List[float]:
    out: List[float] = []
    for seed in seeds:
        key = f"{arm_prefix}_{seed}"
        if key not in data:
            raise KeyError(f"Missing key: {key}")
        out.append(active_epoch_metric(data[key], metric=metric, epoch_min=epoch_min))
    return out


def seed_labels(seeds: Sequence[int]) -> List[str]:
    # If all seeds share the same thousands-prefix, use full numbers
    # (avoids ambiguous labels like "60k" for holdout seeds 60000-60009)
    prefixes = set(s // 1000 for s in seeds)
    if len(prefixes) == 1:
        return [str(s) for s in seeds]

    labels = []
    for s in seeds:
        if s >= 10000:
            prefix = s // 1000
            remainder = s % 1000
            if remainder == 0:
                labels.append(f"{prefix}k")
            elif remainder == 200:
                labels.append(f"{prefix}.2k")
            else:
                labels.append(str(s))
        else:
            labels.append(str(s))
    return labels


def plot_grouped_bars(
    seeds: Sequence[int],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    outfile: Path,
    add_threshold: bool = True,
    threshold_y: float = 0.015,
    annotate_seed: int | None = None,
) -> None:
    if len(series) != len(labels):
        raise ValueError("series and labels length mismatch")

    x = np.arange(len(seeds))
    n = len(series)
    width = 0.78 / n
    offsets = [(i - (n - 1) / 2.0) * width for i in range(n)]

    fig, ax = plt.subplots(figsize=(11, 6))
    max_val = 0.0
    for vals, label, offset in zip(series, labels, offsets):
        color = COLORS.get(label.replace("ExactK (", "").replace(")", ""), None)
        if color is None and label.startswith("ExactK"):
            color = COLORS["ExactK"]
        if color is None:
            color = "#777777"
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.6)
        max_val = max(max_val, max(vals))
        # highlight median in legend text only via table/caption, not extra lines.

    if add_threshold:
        ax.axhline(
            y=threshold_y,
            color="#C44E52",
            linestyle="--",
            linewidth=1.5,
            label=f"Historical organic-clean heuristic ({threshold_y:.3f})",
            zorder=0,
        )

    ax.set_ylabel(r"Epoch-median $G_1$ (lower is better)")
    ax.set_xlabel("Seed")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels(seeds), rotation=45, ha="right")
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True)

    if annotate_seed is not None and annotate_seed in seeds and len(series) >= 2:
        idx = list(seeds).index(annotate_seed)
        y = series[1][idx]
        ax.annotate(
            "Adversarial seed",
            xy=(x[idx] + offsets[1], y),
            xytext=(x[idx] + offsets[1], y + max_val * 0.12),
            ha="center",
            arrowprops={"arrowstyle": "->", "lw": 1.2},
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars_broken_axis(
    seeds: Sequence[int],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    outfile: Path,
    break_low: float = 0.08,
    break_high: float = 0.16,
    add_threshold: bool = True,
    threshold_y: float = 0.015,
) -> None:
    """Grouped bar chart with a broken y-axis to reduce outlier compression.

    The y-axis is split into two regions:
      - lower panel: 0 to break_low  (most seeds live here)
      - upper panel: break_high to max  (outliers)
    """
    if len(series) != len(labels):
        raise ValueError("series and labels length mismatch")

    all_vals = [v for s in series for v in s]
    max_val = max(all_vals)

    x = np.arange(len(seeds))
    n = len(series)
    width = 0.78 / n
    offsets = [(i - (n - 1) / 2.0) * width for i in range(n)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7),
        gridspec_kw={"height_ratios": [1, 2.5], "hspace": 0.06},
    )

    for ax in (ax_top, ax_bot):
        for vals, label, offset in zip(series, labels, offsets):
            color = COLORS.get(label.replace("ExactK (", "").replace(")", ""), None)
            if color is None and label.startswith("ExactK"):
                color = COLORS["ExactK"]
            if color is None:
                color = "#777777"
            ax.bar(x + offset, vals, width, label=label, color=color,
                   edgecolor="black", linewidth=0.6)
        if add_threshold:
            ax.axhline(y=threshold_y, color="#C44E52", linestyle="--",
                       linewidth=1.5, zorder=0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Set axis limits for broke region
    ax_top.set_ylim(break_high, max_val * 1.12)
    ax_bot.set_ylim(0, break_low)

    # Hide spines at the break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Diagonal break markers
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels and title
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(seed_labels(seeds), rotation=45, ha="right")
    ax_bot.set_xlabel("Seed")
    fig.text(0.02, 0.5, r"Epoch-median $G_1$ (lower is better)",
             va="center", rotation="vertical", fontsize=12)
    ax_top.set_title(title)

    # Legend on top panel only, with threshold label
    handles, leg_labels = ax_bot.get_legend_handles_labels()
    if add_threshold:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#C44E52", linestyle="--",
                              linewidth=1.5))
        leg_labels.append(f"Historical organic-clean heuristic ({threshold_y:.3f})")
    ax_top.legend(handles, leg_labels, frameon=True, loc="upper right")

    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ExactK paper figures from canonical JSON artifacts.")
    parser.add_argument("--day69", required=True, help="Path to Day 69 all_results.json")
    parser.add_argument("--day70", required=True, help="Path to Day 70 all_results.json")
    parser.add_argument("--day75", required=True, help="Path to Day 75 all_results.json")
    parser.add_argument("--outdir", default="paper_figures", help="Output directory for figures")
    parser.add_argument("--metric", default="G1_aligned", help="Metric field to aggregate (default: G1_aligned)")
    parser.add_argument("--active-epoch-min", type=int, default=6, help="Minimum epoch included in epoch-median aggregation")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 3: d=5 primary validation
    day69 = load_json(args.day69)
    d5_control = extract_arm(day69, "Control", D5_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    d5_exactk = extract_arm(day69, "ExactK_Tuned", D5_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    d5_gated = extract_arm(day69, "ExactK_Tuned_Gated", D5_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    plot_grouped_bars(
        D5_SEEDS,
        [d5_control, d5_exactk, d5_gated],
        ["Control", "ExactK", "ExactK (Gated)"],
        title=r"$d=5$ primary validation",
        outfile=outdir / "figure_3_d5",
        annotate_seed=49200,
    )

    # Figure 4: d=7 transfer / OOD
    day70 = load_json(args.day70)
    d7_control = extract_arm(day70, "Control", D5_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    d7_exactk = extract_arm(day70, "ExactK_Tuned_Prod", D5_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    d7_cutoff = extract_arm(day70, "ExactK_EarlyCutoff", D5_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    plot_grouped_bars_broken_axis(
        D5_SEEDS,
        [d7_control, d7_exactk, d7_cutoff],
        ["Control", "ExactK", "ExactK (EarlyCutoff)"],
        title=r"$d=7$ transfer evaluation",
        outfile=outdir / "figure_4_d7_broken_axis",
    )

    # Figure 5: holdout validation
    day75 = load_json(args.day75)
    h_control = extract_arm(day75, "Control", HOLDOUT_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    h_exactk = extract_arm(day75, "ExactK_Tuned_Prod", HOLDOUT_SEEDS, metric=args.metric, epoch_min=args.active_epoch_min)
    plot_grouped_bars(
        HOLDOUT_SEEDS,
        [h_control, h_exactk],
        ["Control", "ExactK"],
        title=r"Holdout evaluation on unseen $d=7$ seeds",
        outfile=outdir / "figure_5_holdout",
    )

    print(f"Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
