"""Stiffness-handling chart — Radau5/BDF/ESDIRK54 on Van der Pol.

Wall-clock per integration vs the stiffness parameter μ. As μ grows
the ratio of fast-to-slow timescales diverges; the curves show how
much each implicit scheme amortises that across reused Jacobians and
factorisations.
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
OUT = REPO_ROOT / "website" / "book" / "src" / "assets" / "figures" / "perf_stiffness_handling.svg"

SOLVERS = [
    ("radau5", "Radau5 (5th-order Gauss)", style.WONG[0]),
    ("bdf", "BDF (variable order, 1–5)", style.WONG[1]),
    ("esdirk54", "ESDIRK54 (5th-order SDIRK)", style.WONG[2]),
]


def main() -> None:
    df = load_group("van_der_pol_stiff").copy()
    df["mu"] = df["param"].astype(float)
    df["mean_ms"] = df["mean_ns"] / 1_000_000.0
    df["err_low_ms"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000_000.0
    df["err_high_ms"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000_000.0
    df = df.sort_values(["bench", "mu"])

    fig, ax = plt.subplots()
    for bench_id, label, color in SOLVERS:
        sub = df[df["bench"] == bench_id]
        if sub.empty:
            continue
        ax.errorbar(
            sub["mu"],
            sub["mean_ms"],
            yerr=[sub["err_low_ms"], sub["err_high_ms"]],
            fmt="o-",
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.8,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Stiffness parameter $\mu$")
    ax.set_ylabel("Wall-clock per integration (ms)")
    ax.set_title(r"Stiff Van der Pol: $x'' - \mu(1-x^2)x' + x = 0$")
    ax.set_xticks([10.0, 100.0, 1000.0])
    ax.set_xticklabels([r"$10$", r"$100$", r"$1000$"])
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Power-law fits per solver — visible on a log-log plot if the
    # implicit schemes scale linearly with μ (because tf = 2μ here).
    notes = []
    for bench_id, label, _ in SOLVERS:
        sub = df[df["bench"] == bench_id]
        if len(sub) < 2:
            continue
        log_mu = np.log10(sub["mu"].to_numpy())
        log_t = np.log10(sub["mean_ms"].to_numpy())
        slope, _ = np.polyfit(log_mu, log_t, 1)
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
        problem="Van der Pol oscillator x'' - μ(1-x^2)x' + x = 0, integrated to t = 2μ",
        solver="Radau5, BDF, ESDIRK54 (implicit, with internal Jacobian + Newton)",
        tol="rtol = 1e-4, atol = 1e-6",
        repro_script="numra-bench/benches/solvers.rs::bench_stiff_solvers",
        extra={
            "mu_values": "10, 100, 1000",
            "measurement": "20.0 s per (solver, μ)",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
