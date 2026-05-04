"""Tolerance-vs-runtime chart — DoPri5/Tsit5/Vern6 on Lorenz.

Each adaptive RK pair has a different efficiency curve: at slack
tolerances the cheaper schemes win on wall-clock; at tight tolerances
the higher-order schemes amortise their per-step cost. This chart
makes the crossover visible.
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
OUT = REPO_ROOT / "website" / "book" / "src" / "assets" / "figures" / "perf_tolerance_runtime.svg"

SOLVERS = [
    ("dopri5", "DoPri5 (5/4)", style.WONG[0]),
    ("tsit5", "Tsit5 (5/4)", style.WONG[1]),
    ("vern6", "Vern6 (6/5)", style.WONG[2]),
]


def main() -> None:
    df = load_group("tolerance_scaling").copy()
    if "param" not in df.columns:
        raise SystemExit("expected 'param' column from parameterised bench")
    df["rtol"] = df["param"].astype(float)
    df["mean_us"] = df["mean_ns"] / 1_000.0
    df["err_low_us"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000.0
    df["err_high_us"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000.0
    df = df.sort_values(["bench", "rtol"], ascending=[True, False])

    fig, ax = plt.subplots()
    for bench_id, label, color in SOLVERS:
        sub = df[df["bench"] == bench_id]
        if sub.empty:
            continue
        ax.errorbar(
            sub["rtol"],
            sub["mean_us"],
            yerr=[sub["err_low_us"], sub["err_high_us"]],
            fmt="o-",
            color=color,
            capsize=3,
            markersize=5,
            linewidth=1.6,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Requested relative tolerance $r_\mathrm{tol}$")
    ax.set_ylabel("Wall-clock runtime (µs)")
    ax.set_title("Tolerance vs runtime — three adaptive RK pairs on Lorenz")
    ax.legend(loc="upper right")

    prov = Provenance(
        problem="Lorenz system (sigma=10, rho=28, beta=8/3), y0=[1,1,1], t in [0, 20]",
        solver="DoPri5 / Tsit5 / Vern6 (adaptive explicit RK with PI controller)",
        tol="rtol from 1e-3 to 1e-9, atol = rtol * 1e-3",
        repro_script="numra-bench/benches/solvers.rs::bench_tolerance_scaling",
        extra={
            "warmup": "3.0 s per point",
            "measurement": "20.0 s per point",
            "n_points_per_solver": str(len(df) // len(SOLVERS) if len(SOLVERS) else 0),
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
