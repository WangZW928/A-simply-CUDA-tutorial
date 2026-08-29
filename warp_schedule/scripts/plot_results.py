#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_runtime(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for bpsm in sorted(df["blocks_per_sm"].unique()):
        sub = df[df["blocks_per_sm"] == bpsm].sort_values("warps_per_block")
        plt.plot(
            sub["warps_per_block"],
            sub["avg_ms"],
            marker="o",
            linewidth=1.8,
            label=f"blocks/SM={bpsm}",
        )
    plt.xlabel("Warps per block")
    plt.ylabel("Kernel time (ms)")
    plt.title("Runtime vs warps per block")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_throughput_line(df: pd.DataFrame, output_path: Path) -> None:
    eff = df.assign(warps_per_ms=df["total_warps"] / df["avg_ms"])

    plt.figure(figsize=(10, 6))
    for bpsm in sorted(eff["blocks_per_sm"].unique()):
        sub = eff[eff["blocks_per_sm"] == bpsm].sort_values("warps_per_block")
        plt.plot(
            sub["warps_per_block"],
            sub["warps_per_ms"],
            marker="o",
            linewidth=1.8,
            label=f"blocks/SM={bpsm}",
        )

    plt.xlabel("Warps per block")
    plt.ylabel("Throughput (total warps / ms)")
    plt.title("Throughput vs warps per block")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_normalized_throughput_line(df: pd.DataFrame, output_path: Path) -> None:
    eff = df.assign(warps_per_ms=df["total_warps"] / df["avg_ms"])
    peak = eff["warps_per_ms"].max()
    if peak <= 0:
        raise ValueError("Invalid throughput values: max throughput must be > 0")

    eff = eff.assign(norm_throughput_pct=eff["warps_per_ms"] / peak * 100.0)

    plt.figure(figsize=(10, 6))
    for bpsm in sorted(eff["blocks_per_sm"].unique()):
        sub = eff[eff["blocks_per_sm"] == bpsm].sort_values("warps_per_block")
        plt.plot(
            sub["warps_per_block"],
            sub["norm_throughput_pct"],
            marker="o",
            linewidth=1.8,
            label=f"blocks/SM={bpsm}",
        )

    plt.xlabel("Warps per block")
    plt.ylabel("Normalized throughput (% of best config)")
    plt.title("Normalized throughput vs warps per block")
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_efficiency_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    work_metric = df["total_warps"] / df["avg_ms"]
    eff = df.assign(warps_per_ms=work_metric)

    pivot = (
        eff.pivot_table(
            index="warps_per_block",
            columns="blocks_per_sm",
            values="warps_per_ms",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    ax.set_xlabel("Blocks per SM")
    ax.set_ylabel("Warps per block")
    ax.set_title("Throughput heatmap (total warps / ms)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Warps per ms")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CUDA warp benchmark CSV")
    parser.add_argument("--input", default="results/warp_benchmark.csv", help="Input CSV file")
    parser.add_argument("--outdir", default="results", help="Directory for figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("CSV is empty")

    plot_runtime(df, outdir / "runtime_vs_warps.png")
    plot_throughput_line(df, outdir / "throughput_vs_warps.png")
    plot_normalized_throughput_line(df, outdir / "normalized_throughput_vs_warps.png")
    plot_efficiency_heatmap(df, outdir / "throughput_heatmap.png")

    summary = (
        df.assign(warps_per_ms=df["total_warps"] / df["avg_ms"])
        .sort_values("warps_per_ms", ascending=False)
        .head(10)
    )
    summary.to_csv(outdir / "top10_configs.csv", index=False)

    print(f"Saved figures to: {outdir.resolve()}")
    print(f"Top-10 config table: {(outdir / 'top10_configs.csv').resolve()}")


if __name__ == "__main__":
    main()
