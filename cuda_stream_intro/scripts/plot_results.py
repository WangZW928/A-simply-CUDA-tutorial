#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CUDA stream benchmark results")
    parser.add_argument("--input", default="results/stream_benchmark.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty")

    order = ["default", "two_streams"]
    sub = df.set_index("mode").loc[order].reset_index()

    plt.figure(figsize=(6.8, 4.8))
    plt.bar(sub["mode"], sub["mean_ms"], color=["#4c78a8", "#f58518"])
    plt.ylabel("Mean time (ms)")
    plt.title("Default Stream vs Two Explicit Streams")
    plt.tight_layout()
    plt.savefig(outdir / "stream_vs_default.png", dpi=180)
    plt.close()

    speedup = sub[sub["mode"] == "default"]["mean_ms"].iloc[0] / sub[sub["mode"] == "two_streams"]["mean_ms"].iloc[0]
    (outdir / "summary.txt").write_text(f"speedup(default/two_streams)={speedup:.6f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
