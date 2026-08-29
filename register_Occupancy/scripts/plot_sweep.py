#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_line(df: pd.DataFrame, x: str, y: str, ylabel: str, title: str, output: Path) -> None:
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(df[x], df[y], marker="o", linewidth=2.0)
    plt.xlabel("Registers per thread")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot register occupancy sweep results")
    parser.add_argument("--input", default="results/reg_occ_sweep.csv", help="Sweep CSV path")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Sweep CSV is empty")

    required = {
        "kernel",
        "high_reg_tmp_size",
        "regs_per_thread",
        "theoretical_occupancy",
        "avg_ms",
        "elems_per_ms",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 只画高寄存器版本，展示随寄存器压力变化的趋势
    high = df[df["kernel"] == "high_reg"].copy()
    if high.empty:
        raise ValueError("No high_reg rows found in sweep CSV")

    high = high.sort_values(["regs_per_thread", "high_reg_tmp_size"])

    plot_line(
        high,
        x="regs_per_thread",
        y="theoretical_occupancy",
        ylabel="Theoretical occupancy",
        title="Occupancy vs Registers per Thread",
        output=outdir / "sweep_occupancy_vs_regs.png",
    )

    plot_line(
        high,
        x="regs_per_thread",
        y="avg_ms",
        ylabel="Kernel time (ms)",
        title="Runtime vs Registers per Thread",
        output=outdir / "sweep_runtime_vs_regs.png",
    )

    plot_line(
        high,
        x="regs_per_thread",
        y="elems_per_ms",
        ylabel="Throughput (elements/ms)",
        title="Throughput vs Registers per Thread",
        output=outdir / "sweep_throughput_vs_regs.png",
    )

    summary_cols = [
        "high_reg_tmp_size",
        "regs_per_thread",
        "theoretical_occupancy",
        "avg_ms",
        "std_ms",
        "elems_per_ms",
    ]
    high[summary_cols].to_csv(outdir / "sweep_summary.csv", index=False)

    print(f"Saved sweep figures and summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
