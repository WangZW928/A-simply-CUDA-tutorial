#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_bars(df: pd.DataFrame, outdir: Path) -> None:
    names = df["kernel"].tolist()

    plt.figure(figsize=(7, 4.8))
    plt.bar(names, df["regs_per_thread"], color=["#4c78a8", "#f58518"])
    plt.ylabel("Registers per thread")
    plt.title("Register Pressure Comparison")
    plt.tight_layout()
    plt.savefig(outdir / "registers_per_thread.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.bar(names, df["theoretical_occupancy"] * 100.0, color=["#4c78a8", "#f58518"])
    plt.ylabel("Theoretical occupancy (%)")
    plt.title("Occupancy Comparison")
    plt.tight_layout()
    plt.savefig(outdir / "occupancy_compare.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    plt.bar(names, df["avg_ms"], yerr=df["std_ms"], capsize=6, color=["#4c78a8", "#f58518"])
    plt.ylabel("Kernel time (ms)")
    plt.title("Runtime Comparison")
    plt.tight_layout()
    plt.savefig(outdir / "runtime_compare.png", dpi=180)
    plt.close()


def build_summary(df: pd.DataFrame, outdir: Path) -> None:
    low = df[df["kernel"] == "low_reg"].iloc[0]
    high = df[df["kernel"] == "high_reg"].iloc[0]

    summary = pd.DataFrame(
        {
            "metric": [
                "regs_per_thread",
                "theoretical_occupancy",
                "avg_ms",
            ],
            "low_reg": [
                low["regs_per_thread"],
                low["theoretical_occupancy"],
                low["avg_ms"],
            ],
            "high_reg": [
                high["regs_per_thread"],
                high["theoretical_occupancy"],
                high["avg_ms"],
            ],
        }
    )
    summary["high_vs_low_ratio"] = summary["high_reg"] / summary["low_reg"]
    summary.to_csv(outdir / "summary_compare.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot register/occupancy benchmark results")
    parser.add_argument("--input", default="results/reg_occ_benchmark.csv", help="Input CSV")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("CSV is empty")

    required = {"kernel", "regs_per_thread", "theoretical_occupancy", "avg_ms", "std_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    plot_bars(df, outdir)
    build_summary(df, outdir)

    print(f"Saved figures and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
