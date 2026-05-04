"""Dense-output overhead chart — DoPri5 ± dense output on Lorenz.

Comparing the same DoPri5 integration with and without dense output
enabled. The dense-output coefficients add a one-time per-step cost
proportional to the order of the embedded interpolant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIGURES_DIR))
from lib import style  # noqa: E402
from lib.captioning import Provenance, save_with_provenance  # noqa: E402
from perf._data import load_group  # noqa: E402

style.apply()

REPO_ROOT = FIGURES_DIR.parent.parent
OUT = REPO_ROOT / "website" / "book" / "src" / "assets" / "figures" / "perf_dense_output_overhead.svg"


def main() -> None:
    df = load_group("dense_output").copy()
    df["mean_us"] = df["mean_ns"] / 1_000.0
    df["err_low_us"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000.0
    df["err_high_us"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000.0

    label_map = {
        "dopri5_no_dense": "DoPri5 (no dense output)",
        "dopri5_dense": "DoPri5 (dense output enabled)",
    }
    df["label"] = df["bench"].map(label_map).fillna(df["bench"])
    df = df.sort_values("bench")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars = ax.bar(
        df["label"],
        df["mean_us"],
        yerr=[df["err_low_us"], df["err_high_us"]],
        capsize=4,
        color=[style.WONG[0], style.WONG[1]],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_ylabel("Wall-clock runtime (µs)")
    ax.set_title("Dense-output overhead on Lorenz")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Annotate each bar with absolute time and relative overhead.
    if len(df) == 2 and "dopri5_no_dense" in df["bench"].values:
        baseline = float(df.loc[df["bench"] == "dopri5_no_dense", "mean_us"].iloc[0])
        for rect, _, row in zip(bars, df["bench"], df.itertuples(index=False)):
            ratio = (rect.get_height() / baseline) - 1.0 if baseline > 0 else 0.0
            offset = "" if row.bench == "dopri5_no_dense" else f" (+{ratio*100:.1f}%)"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{rect.get_height():.0f} µs{offset}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    prov = Provenance(
        problem="Lorenz system, y0=[1,1,1], t in [0, 10]",
        solver="DoPri5 with and without dense output (5th-order interpolant)",
        tol="rtol = 1e-6, atol = 1e-9",
        repro_script="numra-bench/benches/solvers.rs::bench_dense_output",
        extra={
            "measurement": "20.0 s per variant",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
