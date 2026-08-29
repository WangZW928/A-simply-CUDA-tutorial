#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CUDA Graph benchmark result")
    parser.add_argument("--input", default="results/graph_benchmark.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty")

    order = ["normal", "graph"]
    sub = df.set_index("mode").loc[order].reset_index()

    plt.figure(figsize=(6.5, 4.8))
    plt.bar(sub["mode"], sub["mean_ms"], color=["#4c78a8", "#f58518"])
    plt.ylabel("Mean latency per iteration (ms)")
    plt.title("Normal Launch vs CUDA Graph Replay")
    plt.tight_layout()
    plt.savefig(outdir / "graph_vs_normal.png", dpi=180)
    plt.close()

    speedup = sub[sub["mode"] == "normal"]["mean_ms"].iloc[0] / sub[sub["mode"] == "graph"]["mean_ms"].iloc[0]
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"speedup(normal/graph)={speedup:.6f}\n")

    print(f"Saved plot: {(outdir / 'graph_vs_normal.png').resolve()}")
    print(f"Saved summary: {(outdir / 'summary.txt').resolve()}")


if __name__ == "__main__":
    main()
