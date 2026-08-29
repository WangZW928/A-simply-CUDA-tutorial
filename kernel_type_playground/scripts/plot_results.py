#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_block_sweep(df: pd.DataFrame, outdir: Path) -> None:
    sub = df[df["experiment"] == "block_size_sweep"].copy()
    if sub.empty:
        return
    plt.figure(figsize=(9.5, 5.8))
    for kernel in sorted(sub["kernel_type"].unique()):
        kdf = sub[sub["kernel_type"] == kernel].sort_values("block_size")
        plt.plot(kdf["block_size"], kdf["mean_ms"], marker="o", linewidth=2, label=kernel)
    plt.xlabel("Block size")
    plt.ylabel("Mean time (ms)")
    plt.title("Kernel type sensitivity to block size")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "block_size_sweep.png", dpi=180)
    plt.close()


def plot_occupancy_sweep(df: pd.DataFrame, outdir: Path) -> None:
    sub = df[df["experiment"] == "occupancy_sweep"].copy()
    if sub.empty:
        return
    sub["blocks_per_sm"] = sub["blocks"] / sub["sm_count"]
    plt.figure(figsize=(9.5, 5.8))
    for kernel in sorted(sub["kernel_type"].unique()):
        kdf = sub[sub["kernel_type"] == kernel].sort_values("blocks_per_sm")
        plt.plot(kdf["blocks_per_sm"], kdf["throughput_units_per_ms"], marker="o", linewidth=2, label=kernel)
    plt.xlabel("Launched blocks per SM")
    plt.ylabel("Throughput units / ms")
    plt.title("Kernel type sensitivity to launch occupancy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "occupancy_sweep.png", dpi=180)
    plt.close()


def plot_mode_compare(df: pd.DataFrame, outdir: Path, experiment: str, output: str) -> None:
    sub = df[df["experiment"] == experiment].copy()
    if sub.empty:
        return
    labels = sub["kernel_type"] + "\n" + sub["mode"]
    plt.figure(figsize=(8.5, 5.2))
    plt.bar(labels, sub["mean_ms"], color="#4c78a8")
    plt.ylabel("Mean time (ms)")
    plt.title(experiment.replace("_", " ").title())
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CUDA kernel type playground results")
    parser.add_argument("--input", default="results/kernel_type_benchmark.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("CSV is empty")

    plot_block_sweep(df, outdir)
    plot_occupancy_sweep(df, outdir)
    plot_mode_compare(df, outdir, "stream_compare", "stream_compare.png")
    plot_mode_compare(df, outdir, "graph_compare", "graph_compare.png")
    df.sort_values(["experiment", "kernel_type", "mean_ms"]).to_csv(outdir / "summary.csv", index=False)
    print(f"Saved plots and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
