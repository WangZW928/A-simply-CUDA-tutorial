#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_stride(df: pd.DataFrame, outdir: Path) -> None:
    stride = df[df["experiment"] == "stride_sweep"].sort_values("param_value")
    if stride.empty:
        raise ValueError("No stride_sweep rows found")

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(stride["param_value"], stride["requested_bandwidth_gb_s"], marker="o", linewidth=2.0)
    plt.xscale("log", base=2)
    plt.xlabel("Stride")
    plt.ylabel("Requested bandwidth (GB/s)")
    plt.title("Bandwidth vs Stride")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "bandwidth_vs_stride.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(stride["param_value"], stride["mean_ms"], marker="o", linewidth=2.0)
    plt.xscale("log", base=2)
    plt.xlabel("Stride")
    plt.ylabel("Kernel time (ms)")
    plt.title("Runtime vs Stride")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "runtime_vs_stride.png", dpi=180)
    plt.close()


def plot_offset(df: pd.DataFrame, outdir: Path) -> None:
    offset = df[df["experiment"] == "offset_sweep"].sort_values("param_value")
    if offset.empty:
        raise ValueError("No offset_sweep rows found")

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(offset["param_value"], offset["requested_bandwidth_gb_s"], marker="o", linewidth=2.0)
    plt.xlabel("Offset (float elements)")
    plt.ylabel("Requested bandwidth (GB/s)")
    plt.title("Bandwidth vs Contiguous Offset")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "bandwidth_vs_offset.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CUDA memory coalescing benchmark results")
    parser.add_argument("--input", default="results/memory_coalescing.csv", help="Input CSV")
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
        "experiment",
        "param_value",
        "mean_ms",
        "requested_bandwidth_gb_s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    plot_stride(df, outdir)
    plot_offset(df, outdir)

    stride = df[df["experiment"] == "stride_sweep"].copy()
    if not stride.empty:
        base = stride[stride["param_value"] == 1]["requested_bandwidth_gb_s"].iloc[0]
        stride["relative_to_stride1"] = stride["requested_bandwidth_gb_s"] / base
        stride.to_csv(outdir / "stride_summary.csv", index=False)

    print(f"Saved plots and summaries to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
