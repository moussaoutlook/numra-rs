"""Dimension-scaling chart — DoPri5 vs Radau5 on coupled linear ODE.

For non-stiff dynamics, an explicit method's per-step cost grows
linearly with state dimension. An implicit method's per-step cost
also grows because it solves a dense linear system at every Newton
iterate — for dense Jacobians that scales O(n^3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIGURES_DIR))
from lib import style  # noqa: E402
from lib.captioning import Provenance, save_with_provenance  # noqa: E402
from perf._data import load_group  # noqa: E402

style.apply()

REPO_ROOT = FIGURES_DIR.parent.parent
OUT = REPO_ROOT / "website" / "book" / "src" / "assets" / "figures" / "perf_dimension_scaling.svg"

SOLVERS = [
    ("dopri5", "DoPri5 (explicit)", style.WONG[0]),
    ("radau5", "Radau5 (implicit)", style.WONG[1]),
]


def main() -> None:
    df = load_group("dimension_scaling").copy()
    df["n"] = df["param"].astype(int)
    df["mean_us"] = df["mean_ns"] / 1_000.0
    df["err_low_us"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000.0
    df["err_high_us"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000.0
    df = df.sort_values(["bench", "n"])

    fig, ax = plt.subplots()
    for bench_id, label, color in SOLVERS:
        sub = df[df["bench"] == bench_id]
        if sub.empty:
            continue
        ax.errorbar(
            sub["n"],
            sub["mean_us"],
            yerr=[sub["err_low_us"], sub["err_high_us"]],
            fmt="o-",
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.8,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"State dimension $n$")
    ax.set_ylabel("Wall-clock runtime (µs)")
    ax.set_title("Dimension scaling on a coupled linear system")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    notes = []
    for bench_id, label, _ in SOLVERS:
        sub = df[df["bench"] == bench_id]
        if len(sub) < 2:
            continue
        log_n = np.log10(sub["n"].to_numpy())
        log_t = np.log10(sub["mean_us"].to_numpy())
        slope, _ = np.polyfit(log_n, log_t, 1)
        notes.append(f"{label}: slope ≈ {slope:.2f}")
    if notes:
        ax.text(
            0.02,
            0.98,
            "\n".join(notes),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            color="#5C6370",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=4),
        )

    prov = Provenance(
        problem="Coupled linear ODE: dy_i/dt = -alpha y_i + beta (y_{i-1} + y_{i+1}), "
        "alpha=2, beta=0.5, t in [0, 5]",
        solver="DoPri5 (explicit) and Radau5 (implicit)",
        tol="rtol = 1e-6, atol = 1e-9",
        repro_script="numra-bench/benches/solvers.rs::bench_dimension_scaling",
        extra={
            "n_values": "2, 10, 50, 200",
            "measurement": "20.0 s per (solver, n)",
            "samples": "20 (sample_size override)",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
