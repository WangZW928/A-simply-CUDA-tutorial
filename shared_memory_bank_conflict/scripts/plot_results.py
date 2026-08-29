#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot shared memory bank conflict benchmark results")
    parser.add_argument("--input", default="results/bank_conflict.csv", help="Input CSV")
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
        "stride",
        "estimated_conflict_degree",
        "mean_ms",
        "shared_loads_per_ms",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values("stride")

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(df["stride"], df["mean_ms"], marker="o", linewidth=2.0)
    plt.xscale("log", base=2)
    plt.xlabel("Shared memory stride")
    plt.ylabel("Kernel time (ms)")
    plt.title("Runtime vs Shared Memory Stride")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_stride.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(df["stride"], df["shared_loads_per_ms"], marker="o", linewidth=2.0)
    plt.xscale("log", base=2)
    plt.xlabel("Shared memory stride")
    plt.ylabel("Shared loads / ms")
    plt.title("Throughput vs Shared Memory Stride")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "throughput_vs_stride.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.bar(df["stride"].astype(str), df["estimated_conflict_degree"])
    plt.xlabel("Shared memory stride")
    plt.ylabel("Estimated conflict degree")
    plt.title("Estimated Bank Conflict Degree")
    plt.tight_layout()
    plt.savefig(outdir / "estimated_conflict_degree.png", dpi=180)
    plt.close()

    base = df[df["stride"] == 1]["shared_loads_per_ms"].iloc[0]
    df["relative_to_stride1"] = df["shared_loads_per_ms"] / base
    df.to_csv(outdir / "bank_conflict_summary.csv", index=False)

    print(f"Saved plots and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
