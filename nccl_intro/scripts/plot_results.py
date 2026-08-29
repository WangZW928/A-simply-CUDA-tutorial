#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_bandwidth(df: pd.DataFrame, outdir: Path, column: str, filename: str, ylabel: str) -> None:
    plt.figure(figsize=(8.8, 5.2))
    for (mode, num_gpus), group in df.groupby(["mode", "num_gpus"]):
        group = group.sort_values("message_bytes")
        label = f"{mode}, {num_gpus} GPUs"
        plt.plot(
            group["message_bytes"],
            group[column],
            marker="o",
            linewidth=2.0,
            label=label,
        )

    plt.xscale("log", base=2)
    plt.xlabel("Message size per rank (bytes)")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Message Size")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NCCL benchmark results")
    parser.add_argument("--input", default="results/nccl_allreduce.csv", help="Input CSV")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty")

    required = {
        "message_bytes",
        "elements",
        "num_gpus",
        "mode",
        "mean_ms",
        "algbw_gb_s",
        "busbw_gb_s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    plot_bandwidth(df, outdir, "algbw_gb_s", "allreduce_algbw.png", "Algorithm bandwidth (GB/s)")
    plot_bandwidth(df, outdir, "busbw_gb_s", "allreduce_busbw.png", "Bus bandwidth (GB/s)")

    summary = df.sort_values(["mode", "num_gpus", "message_bytes"]).copy()
    summary.to_csv(outdir / "nccl_summary.csv", index=False)
    print(f"Saved plots and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
