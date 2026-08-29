#!/usr/bin/env python3
"""Plot register pressure, theoretical occupancy, and measured performance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGS_PER_SM = 65536
MAX_WARPS_PER_SM = 48
WARP_SIZE = 32
MAX_THREADS_PER_SM = MAX_WARPS_PER_SM * WARP_SIZE
MAX_BLOCKS_PER_SM = 24  # RTX 4060 Laptop (AD107) deviceQuery 实测值
REG_ALLOC_GRANULARITY_PER_WARP = 256
BLOCK_SIZES = [128, 256, 512, 1024]


def effective_regs_per_thread(regs: int) -> int:
    regs_per_warp = regs * WARP_SIZE
    rounded = math.ceil(regs_per_warp / REG_ALLOC_GRANULARITY_PER_WARP)
    return (rounded * REG_ALLOC_GRANULARITY_PER_WARP) // WARP_SIZE


def analytical_occupancy(regs_per_thread: int, block_size: int) -> tuple[int, float]:
    eff_regs = effective_regs_per_thread(regs_per_thread)
    blocks_by_regs = REGS_PER_SM // (eff_regs * block_size)
    blocks_by_warps = MAX_WARPS_PER_SM // (block_size // WARP_SIZE)
    blocks_by_threads = MAX_THREADS_PER_SM // block_size
    active_blocks = min(
        blocks_by_regs, blocks_by_warps, blocks_by_threads, MAX_BLOCKS_PER_SM
    )
    occupancy = active_blocks * block_size / MAX_THREADS_PER_SM
    return active_blocks, occupancy


def staircase_dataframe() -> pd.DataFrame:
    rows = []
    for block_size in BLOCK_SIZES:
        for regs in range(8, 257, 8):
            active_blocks, occ = analytical_occupancy(regs, block_size)
            rows.append(
                {
                    "block_size": block_size,
                    "regs_per_thread": regs,
                    "effective_regs_per_thread": effective_regs_per_thread(regs),
                    "active_blocks_per_sm_model": active_blocks,
                    "occupancy_model": occ,
                }
            )
    return pd.DataFrame(rows)


def mark_occupancy_cliffs(ax: plt.Axes, model: pd.DataFrame) -> None:
    cliff_regs = set()
    for block_size, group in model.groupby("block_size"):
        group = group.sort_values("regs_per_thread")
        drops = group["occupancy_model"].diff() < 0
        cliff_regs.update(group.loc[drops, "regs_per_thread"].astype(int).tolist())

    for x in sorted(cliff_regs):
        ax.axvspan(x - 2.5, x + 2.5, color="0.7", alpha=0.10, lw=0)
    if cliff_regs:
        ax.text(
            0.985,
            0.08,
            "shaded bands: occupancy cliffs",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=9,
            color="0.25",
        )


def prepare_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    keep = [
        "experiment",
        "reg_tmp_size",
        "launch_bounds_min_blocks",
        "block_size",
        "regs_per_thread",
        "active_blocks_per_sm",
        "theoretical_occupancy",
        "avg_ms",
        "throughput_gel_s",
        "grid_blocks",
        "elements",
        "iters",
        "repeats",
        "warmup",
    ]
    summary = df[keep].copy()
    summary = summary.sort_values(
        ["experiment", "block_size", "reg_tmp_size", "launch_bounds_min_blocks"]
    )
    summary.to_csv(outdir / "occupancy_summary.csv", index=False)
    return summary


def plot(input_csv: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    prepare_summary(df, outdir)

    sweep = df[df["experiment"] == "sweep"].copy()
    launch_bounds = df[df["experiment"] == "launch_bounds"].copy()
    model = staircase_dataframe()

    colors = {
        128: "#0072B2",
        256: "#D55E00",
        512: "#009E73",
        1024: "#CC79A7",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Register Pressure vs Occupancy (RTX 4060 Laptop, sm_89)",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    mark_occupancy_cliffs(ax, model)
    for block_size in BLOCK_SIZES:
        m = model[model["block_size"] == block_size]
        s = sweep[sweep["block_size"] == block_size].sort_values("regs_per_thread")
        ax.step(
            m["regs_per_thread"],
            m["occupancy_model"],
            where="post",
            color=colors[block_size],
            lw=2,
            label=f"{block_size} threads/block model",
        )
        ax.scatter(
            s["regs_per_thread"],
            s["theoretical_occupancy"],
            color=colors[block_size],
            edgecolor="white",
            s=38,
            zorder=4,
            label=f"{block_size} measured API points",
        )
    ax.set_xlabel("Registers per thread")
    ax.set_ylabel("Theoretical occupancy")
    ax.set_xlim(8, 256)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)

    ax = axes[0, 1]
    for block_size in BLOCK_SIZES:
        s = sweep[sweep["block_size"] == block_size].sort_values("regs_per_thread")
        ax.plot(
            s["regs_per_thread"],
            s["avg_ms"],
            marker="o",
            lw=2,
            color=colors[block_size],
            label=f"{block_size} threads/block",
        )
    ax.set_xlabel("Registers per thread")
    ax.set_ylabel("Average runtime per launch (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for block_size in BLOCK_SIZES:
        s = sweep[sweep["block_size"] == block_size].sort_values("regs_per_thread")
        ax.plot(
            s["regs_per_thread"],
            s["throughput_gel_s"],
            marker="s",
            lw=2,
            color=colors[block_size],
            label=f"{block_size} threads/block",
        )
    ax.set_xlabel("Registers per thread")
    ax.set_ylabel("Throughput (G elements/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    if not launch_bounds.empty:
        lb = launch_bounds.sort_values("launch_bounds_min_blocks")
        x = lb["launch_bounds_min_blocks"].astype(int)
        ax2 = ax.twinx()
        ax.plot(
            x,
            lb["avg_ms"],
            marker="o",
            lw=2,
            color="#0072B2",
            label="runtime",
        )
        ax2.plot(
            x,
            lb["regs_per_thread"],
            marker="^",
            lw=2,
            color="#D55E00",
            label="regs/thread",
        )
        ax2.plot(
            x,
            lb["theoretical_occupancy"],
            marker="s",
            lw=2,
            color="#009E73",
            label="occupancy",
        )
        ax.set_xlabel("__launch_bounds__(256, N): N")
        ax.set_ylabel("Average runtime per launch (ms)", color="#0072B2")
        ax2.set_ylabel("Regs/thread and occupancy", color="0.25")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=9, loc="best")
    else:
        for block_size in BLOCK_SIZES:
            s = sweep[sweep["block_size"] == block_size]
            ax.scatter(
                s["theoretical_occupancy"],
                s["throughput_gel_s"],
                color=colors[block_size],
                s=42,
                alpha=0.85,
                label=f"{block_size} threads/block",
            )
        ax.set_xlabel("Theoretical occupancy")
        ax.set_ylabel("Throughput (G elements/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for path in [outdir / "occupancy_analysis.png", outdir / "occupancy_analysis.svg"]:
        if path.suffix == ".png":
            fig.savefig(path, dpi=180, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    args = parser.parse_args()
    plot(args.input, args.outdir)


if __name__ == "__main__":
    main()
