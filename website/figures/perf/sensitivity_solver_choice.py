"""Sensitivity-by-solver-choice chart on Robertson stiff kinetics.

The Robertson rate constants span eight orders of magnitude. Implicit
solvers (Radau5, BDF) handle this comfortably and integrate to t=40.
Explicit solvers (DoPri5, Tsit5) cannot — they're held to t=1 just so
the bench finishes, and the chart annotates the asymmetry instead of
hiding it.
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
OUT = (
    REPO_ROOT
    / "website"
    / "book"
    / "src"
    / "assets"
    / "figures"
    / "perf_sensitivity_solver_choice.svg"
)

# (bench_id, label, color, t_final, is_implicit)
SOLVERS = [
    ("radau5", "Radau5", style.WONG[5], 40.0, True),
    ("bdf", "BDF (NDF)", style.WONG[3], 40.0, True),
    ("dopri5", "DoPri5", style.WONG[1], 1.0, False),
    ("tsit5", "Tsit5", style.WONG[6], 1.0, False),
]


def main() -> None:
    df = load_group("sensitivity_solver_choice").copy()
    df["mean_us"] = df["mean_ns"] / 1_000.0
    df["err_low_us"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000.0
    df["err_high_us"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000.0
    df = df.set_index("bench")

    means = [df.loc[bid, "mean_us"] for bid, *_ in SOLVERS]
    err_low = [df.loc[bid, "err_low_us"] for bid, *_ in SOLVERS]
    err_high = [df.loc[bid, "err_high_us"] for bid, *_ in SOLVERS]
    labels = [lbl for _, lbl, *_ in SOLVERS]
    colors = [c for _, _, c, *_ in SOLVERS]
    t_finals = [t for *_, t, _ in SOLVERS]
    is_implicit = [imp for *_, imp in SOLVERS]

    x = np.arange(len(SOLVERS))
    fig, ax = plt.subplots()
    bars = ax.bar(
        x,
        means,
        yerr=[err_low, err_high],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )

    # Hatch the explicit-solver bars so the t_final asymmetry is loud.
    for bar, imp in zip(bars, is_implicit):
        if not imp:
            bar.set_hatch("//")

    for i, (mean, t_final) in enumerate(zip(means, t_finals)):
        ax.text(
            x[i],
            mean,
            f"{mean:.0f} µs\n(t_f={t_final:g})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock runtime per solve (µs)")
    ax.set_title("Sensitivity-solver choice on Robertson kinetics")
    ax.set_ylim(0, max(means) * 1.30)
    ax.grid(True, axis="y", alpha=0.3)

    # Inset note explaining the t_final asymmetry.
    ax.text(
        0.5,
        -0.18,
        r"Hatched bars: explicit solvers held to $t_f=1$; "
        r"implicit solvers integrate to $t_f=40$. "
        r"Comparison is on cost-per-call, not work-equivalent.",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        transform=ax.transAxes,
    )

    prov = Provenance(
        problem=(
            "Robertson stiff kinetics (k1=0.04, k2=3e7, k3=1e4), "
            "N=3 states, N_s=3 params, y0=[1, 0, 0]"
        ),
        solver="Radau5 / BDF (NDF) / DoPri5 / Tsit5 — see chart for t_final",
        tol="rtol=1e-6, atol=1e-9",
        repro_script="numra-bench/benches/sensitivity.rs::bench_sensitivity_solver_choice",
        extra={
            "warmup": "3.0 s per point",
            "measurement": "20.0 s per point",
            "implicit_t_final": "40.0",
            "explicit_t_final": "1.0",
            "jacobian_mode": "analytical (J_y row-major + J_p column-major, both flagged)",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
