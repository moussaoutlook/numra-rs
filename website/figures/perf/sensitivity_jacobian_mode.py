"""Sensitivity-vs-Jacobian-mode chart.

Implementations of `ParametricOdeSystem` choose how Jacobians are
supplied: the trait-default forward FD, an analytical state Jacobian,
or analytical state + parameter Jacobians together. On stiff problems
driven by `Radau5`, where the augmented Jacobian appears inside Newton
iteration, this choice changes wall-clock measurably. The bar chart
quantifies that.
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
    / "perf_sensitivity_jacobian_mode.svg"
)

MODES = [
    ("fd", "Forward FD\n(default)", style.WONG[6]),
    ("analytical_y", r"Analytical $J_y$", style.WONG[5]),
    ("analytical_yp", r"Analytical $J_y$ + $J_p$", style.WONG[3]),
]


def main() -> None:
    df = load_group("sensitivity_jacobian_mode").copy()
    df["mean_ms"] = df["mean_ns"] / 1_000_000.0
    df["err_low_ms"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000_000.0
    df["err_high_ms"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000_000.0
    df = df.set_index("param")

    means = [df.loc[mode_key, "mean_ms"] for mode_key, _, _ in MODES]
    err_low = [df.loc[mode_key, "err_low_ms"] for mode_key, _, _ in MODES]
    err_high = [df.loc[mode_key, "err_high_ms"] for mode_key, _, _ in MODES]
    labels = [lbl for _, lbl, _ in MODES]
    colors = [c for _, _, c in MODES]

    x = np.arange(len(MODES))
    fig, ax = plt.subplots()
    ax.bar(
        x,
        means,
        yerr=[err_low, err_high],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )

    fd_mean = df.loc["fd", "mean_ms"]
    for i, mean in enumerate(means):
        speedup = fd_mean / mean
        ax.text(
            x[i],
            mean,
            f"{mean:.2f} ms\n({speedup:.2f}× FD)" if i > 0 else f"{mean:.2f} ms\n(baseline)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock runtime per solve (ms)")
    ax.set_title("Sensitivity cost vs Jacobian mode — Lotka–Volterra (Radau5)")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    prov = Provenance(
        problem=(
            "Lotka–Volterra (alpha=1.0, beta=0.1, delta=0.075, gamma=1.5), "
            "N=2 states, N_s=4 params, y0=[10, 5], t in [0, 5]"
        ),
        solver="Radau5 (5th-order implicit Runge–Kutta on the augmented system)",
        tol="rtol=1e-8, atol=1e-10",
        repro_script="numra-bench/benches/sensitivity.rs::bench_sensitivity_jacobian_mode",
        extra={
            "warmup": "3.0 s per point",
            "measurement": "20.0 s per point",
            "fd_mode": "trait-default forward FD on both J_y and J_p",
            "analytical_y_mode": "analytical J_y, FD J_p",
            "analytical_yp_mode": "analytical J_y and J_p, both flags set",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
