"""Sensitivity-vs-parameter-count chart.

Forward sensitivity costs scale linearly with `n_params` in the
augmented-system formulation: each parameter adds one length-N column
to the integrated state. This chart pins down the constant: how many
nanoseconds per parameter on top of the baseline state-only solve.
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
OUT = (
    REPO_ROOT
    / "website"
    / "book"
    / "src"
    / "assets"
    / "figures"
    / "perf_sensitivity_param_scaling.svg"
)


def main() -> None:
    df = load_group("sensitivity_param_scaling").copy()
    df["n_params"] = df["param"].astype(int)
    df["mean_us"] = df["mean_ns"] / 1_000.0
    df["err_low_us"] = (df["mean_ns"] - df["ci_low_ns"]) / 1_000.0
    df["err_high_us"] = (df["ci_high_ns"] - df["mean_ns"]) / 1_000.0
    df = df.sort_values("n_params")

    fig, ax = plt.subplots()
    ax.errorbar(
        df["n_params"],
        df["mean_us"],
        yerr=[df["err_low_us"], df["err_high_us"]],
        fmt="o-",
        color=style.WONG[5],
        capsize=3,
        markersize=6,
        linewidth=1.8,
        label="DoPri5 augmented solve",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel(r"Number of parameters $N_s$")
    ax.set_ylabel("Wall-clock runtime per solve (µs)")
    ax.set_title("Sensitivity cost vs parameter count — 2-state oscillator")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    prov = Provenance(
        problem=(
            "Damped harmonic oscillator with N_s additive sinusoidal forcings, "
            "N=2 states, t in [0, 10], y0=[1, 0]"
        ),
        solver="DoPri5 (adaptive explicit RK with PI controller) on the augmented system",
        tol="rtol=1e-8, atol=1e-10",
        repro_script="numra-bench/benches/sensitivity.rs::bench_sensitivity_param_scaling",
        extra={
            "warmup": "3.0 s per point",
            "measurement": "20.0 s per point",
            "n_points": str(len(df)),
            "jacobian_mode": "analytical (J_y row-major + J_p column-major, both flagged)",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
