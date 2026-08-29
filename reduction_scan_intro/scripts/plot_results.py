#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CUDA reduction/scan benchmark results")
    parser.add_argument("--input", default="results/reduce_scan_benchmark.csv")
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

    labels = df["operation"] + "\n" + df["variant"]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, df["mean_ms"], color=["#4c78a8", "#54a24b", "#f58518"][: len(df)])
    plt.ylabel("Mean kernel time (ms)")
    plt.title("Reduction and scan runtime")
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "runtime_compare.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, df["effective_gb_s"], color=["#4c78a8", "#54a24b", "#f58518"][: len(df)])
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title("Reduction and scan effective bandwidth")
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "bandwidth_compare.png", dpi=180)
    plt.close()

    df.sort_values(["operation", "mean_ms"]).to_csv(outdir / "summary.csv", index=False)
    print(f"Saved plots and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
