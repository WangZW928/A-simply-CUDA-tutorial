#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stream benchmark CSV used by Nsight Systems")
    parser.add_argument("--input", default="results/stream_profile_input.csv")
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

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(df["mode"], df["mean_ms"], color=["#4c78a8", "#f58518"])
    plt.ylabel("Mean end-to-end time (ms)")
    plt.title("Stream benchmark timing for Nsight Systems profile")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "stream_profile_timing.png", dpi=180)
    plt.close()

    summary = df[["mode", "mean_ms"]].copy()
    if set(summary["mode"]) >= {"default", "two_streams"}:
        base = summary.loc[summary["mode"] == "default", "mean_ms"].iloc[0]
        overlap = summary.loc[summary["mode"] == "two_streams", "mean_ms"].iloc[0]
        summary["speedup_vs_two_streams"] = base / summary["mean_ms"]
        (outdir / "summary.txt").write_text(
            f"default_ms={base:.6f}\ntwo_streams_ms={overlap:.6f}\nspeedup={base / overlap:.3f}\n",
            encoding="utf-8",
        )
    summary.to_csv(outdir / "stream_profile_summary.csv", index=False)
    print(f"Saved plot and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
