"""
Home-page hero benchmark chart.

Plots wall-clock runtime vs requested tolerance for DoPri5 on the Lorenz
system integrated to t=20. Data comes from `cargo bench -p numra-bench --
tolerance_scaling`, which writes Criterion's `estimates.json` files into
`target/criterion/tolerance_scaling/dopri5/<tol>/new/estimates.json`.

The chart is log-log, with the 95% confidence interval drawn as a vertical
error bar on each point. Provenance caption lands at the bottom of the SVG;
a sidecar `.provenance.txt` carries the full hardware/toolchain fingerprint.
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
from lib.data import load_criterion_group  # noqa: E402

style.apply()

REPO_ROOT = FIGURES_DIR.parent.parent
CRITERION_ROOT = REPO_ROOT / "target" / "criterion"
OUT = REPO_ROOT / "website" / "site" / "public" / "figures" / "home_tolerance_runtime.svg"


def main() -> None:
    df = load_criterion_group(CRITERION_ROOT, "tolerance_scaling/dopri5")
    if df.empty:
        raise SystemExit(
            "No Criterion data found. Run "
            "`cargo bench -p numra-bench -- tolerance_scaling/` "
            "from the repo root first."
        )

    # Criterion bench id is "1e-3", "1e-4", ...; convert to a numeric tolerance.
    df = df.copy()
    df["rtol"] = df["bench"].str.replace("1e-", "1e-", regex=False).astype(float)
    df["mean_us"] = df["mean_ns"] / 1_000.0
    df["err_low_us"] = (df["mean_ns"] - df["ci_lower_ns"]) / 1_000.0
    df["err_high_us"] = (df["ci_upper_ns"] - df["mean_ns"]) / 1_000.0
    df = df.sort_values("rtol", ascending=False)  # x ascending toward stricter tol

    fig, ax = plt.subplots()
    ax.errorbar(
        df["rtol"],
        df["mean_us"],
        yerr=[df["err_low_us"], df["err_high_us"]],
        fmt="o-",
        color=style.ACCENT,
        capsize=3,
        markersize=5,
        linewidth=1.6,
        label=r"DoPri5 on Lorenz, $t \in [0, 20]$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # stricter tolerance on the right
    ax.set_xlabel(r"Requested relative tolerance $r_\mathrm{tol}$")
    ax.set_ylabel("Wall-clock runtime (µs)")
    ax.set_title("Runtime vs tolerance — adaptive step pays off")
    ax.legend(loc="upper right")

    # Annotate each point with its actual runtime, just left of the marker.
    for _, row in df.iterrows():
        if row["mean_us"] >= 100.0:
            label = f"{row['mean_us']:.0f} µs"
        else:
            label = f"{row['mean_us']:.1f} µs"
        ax.annotate(
            label,
            xy=(row["rtol"], row["mean_us"]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8,
            color=style.ACCENT,
        )

    # Reference power-law slope: log10(t) ~ k * log10(1/rtol). Fit and annotate.
    log_x = np.log10(1.0 / df["rtol"].to_numpy())
    log_y = np.log10(df["mean_us"].to_numpy())
    slope, _ = np.polyfit(log_x, log_y, 1)
    ax.text(
        0.04,
        0.06,
        rf"empirical slope: $\log t \approx {slope:.2f} \cdot \log(1/r_\mathrm{{tol}})$",
        transform=ax.transAxes,
        fontsize=9,
        color="#5C6370",
    )

    prov = Provenance(
        problem="Lorenz system (sigma=10, rho=28, beta=8/3), y0=[1,1,1], t in [0, 20]",
        solver="DoPri5 (5(4) explicit RK with PI step controller)",
        tol="rtol from 1e-3 to 1e-9, atol = rtol * 1e-3",
        repro_script="numra-bench/benches/solvers.rs::bench_tolerance_scaling",
        extra={
            "samples_per_point": "100 (Criterion default)",
            "warmup": "3.0 s per point",
            "n_points": str(len(df)),
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
