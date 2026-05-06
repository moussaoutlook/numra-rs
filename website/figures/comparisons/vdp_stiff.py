"""Numra (Radau5) vs SciPy (Radau) on a stiff Van der Pol oscillator.

Per-SPEC §17 honest comparison: one scoped problem class, one chart, the
caption explicitly names where Numra is slower than SciPy. Both sides
run the same algorithm (3-stage Radau IIA) so the comparison isolates
runtime overhead rather than algorithm choice.

Pipeline:
    1. Compute a high-precision reference final state with SciPy Radau at
       rtol=atol=1e-12. This is treated as ground truth.
    2. For each tolerance in {1e-4, 1e-6, 1e-8}:
         - Drive the Rust binary `compare_vdp` (built from
           numra-bench/src/bin/compare_vdp.rs) to time Numra Radau5.
         - Run SciPy Radau in-process and time it the same way (warm-up
           run discarded, median of N reps).
         - Compute "correct digits" as -log10(relative error vs reference).
    3. Render two panels:
         (a) Wall-clock vs tolerance (log-log, bars per library)
         (b) Wall-clock per correct digit (bars per library) — a normalised
             "cost of accuracy" view.

The whole script writes one SVG + sidecar provenance.txt.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

THIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIGURES_DIR))
from lib import style  # noqa: E402
from lib.captioning import Provenance, save_with_provenance  # noqa: E402

style.apply()

REPO_ROOT = FIGURES_DIR.parent.parent
OUT = REPO_ROOT / "website" / "book" / "src" / "assets" / "figures" / "comparison_vdp_stiff.svg"
DATA_OUT = THIS_DIR / "data" / "vdp_stiff.json"
RUST_BIN = REPO_ROOT / "target" / "release" / "compare_vdp"

MU = 10.0
TF = 2.0 * MU
Y0 = np.array([2.0, 0.0])
TOLERANCES = [1e-4, 1e-6, 1e-8]
REPS = 5


def vdp_rhs(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], MU * (1.0 - y[0] ** 2) * y[1] - y[0]])


def vdp_jac(t: float, y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0],
            [-2.0 * MU * y[0] * y[1] - 1.0, MU * (1.0 - y[0] ** 2)],
        ]
    )


def reference_state() -> np.ndarray:
    """Ground-truth final state via SciPy Radau at rtol=atol=1e-12."""
    sol = solve_ivp(
        vdp_rhs,
        (0.0, TF),
        Y0,
        method="Radau",
        jac=vdp_jac,
        rtol=1e-12,
        atol=1e-14,
        dense_output=False,
    )
    if not sol.success:
        raise SystemExit(f"reference solve failed: {sol.message}")
    return sol.y[:, -1]


def time_scipy(rtol: float, atol: float, reps: int) -> dict:
    """Median-of-N wall-clock of SciPy Radau, plus final state."""
    # Warm-up.
    _ = solve_ivp(vdp_rhs, (0.0, TF), Y0, method="Radau", jac=vdp_jac,
                  rtol=rtol, atol=atol)

    samples_ns: list[int] = []
    final = None
    nfev = njev = nlu = 0
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        sol = solve_ivp(
            vdp_rhs,
            (0.0, TF),
            Y0,
            method="Radau",
            jac=vdp_jac,
            rtol=rtol,
            atol=atol,
        )
        elapsed = time.perf_counter_ns() - t0
        if not sol.success:
            raise SystemExit(f"scipy Radau failed at rtol={rtol}: {sol.message}")
        samples_ns.append(elapsed)
        final = sol.y[:, -1]
        nfev = int(sol.nfev)
        njev = int(getattr(sol, "njev", 0) or 0)
        nlu = int(getattr(sol, "nlu", 0) or 0)

    samples_ns.sort()
    return {
        "library": "scipy-radau",
        "rtol": rtol,
        "atol": atol,
        "reps": reps,
        "median_ns": samples_ns[len(samples_ns) // 2],
        "mean_ns": float(np.mean(samples_ns)),
        "min_ns": samples_ns[0],
        "max_ns": samples_ns[-1],
        "final_x": float(final[0]),
        "final_xprime": float(final[1]),
        "n_eval": nfev,
        "n_jac": njev,
        "n_lu": nlu,
    }


def time_numra(solver: str, rtol: float, atol: float, reps: int) -> dict:
    if not RUST_BIN.exists():
        raise SystemExit(
            f"compare_vdp binary not found at {RUST_BIN}. "
            "Build it first:\n"
            "  cargo build -p numra-bench --release --bin compare_vdp"
        )
    res = subprocess.run(
        [
            str(RUST_BIN),
            "--solver", solver,
            "--mu", str(MU),
            "--rtol", f"{rtol:e}",
            "--atol", f"{atol:e}",
            "--reps", str(reps),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(res.stdout.strip())


def correct_digits(measured: np.ndarray, reference: np.ndarray) -> float:
    """-log10(relative L2 error). Capped at 14 (double-precision floor)."""
    err = np.linalg.norm(measured - reference) / max(np.linalg.norm(reference), 1e-30)
    if err <= 0.0 or not np.isfinite(err):
        return 14.0
    return min(-np.log10(err), 14.0)


def main() -> None:
    print("Computing reference final state (SciPy Radau, rtol=1e-12) …", flush=True)
    ref = reference_state()
    print(f"  reference x_final     = {ref[0]:.15e}")
    print(f"  reference xprime_final = {ref[1]:.15e}")

    rows: list[dict] = []
    for rtol in TOLERANCES:
        atol = rtol * 1e-3
        print(f"\n— rtol = {rtol:.0e}, atol = {atol:.0e}")

        print("  Numra Radau5 …", flush=True)
        nu = time_numra("radau5", rtol, atol, REPS)
        nu_final = np.array([nu["final_x"], nu["final_xprime"]])
        nu["correct_digits"] = correct_digits(nu_final, ref)
        rows.append(nu)
        print(f"    median = {nu['median_ns']/1e6:.2f} ms, "
              f"correct_digits = {nu['correct_digits']:.2f}")

        print("  SciPy Radau …", flush=True)
        sp = time_scipy(rtol, atol, REPS)
        sp_final = np.array([sp["final_x"], sp["final_xprime"]])
        sp["correct_digits"] = correct_digits(sp_final, ref)
        rows.append(sp)
        print(f"    median = {sp['median_ns']/1e6:.2f} ms, "
              f"correct_digits = {sp['correct_digits']:.2f}")

    # Persist raw timings.
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.write_text(json.dumps(
        {
            "problem": "van_der_pol",
            "mu": MU,
            "tf": TF,
            "y0": Y0.tolist(),
            "reference_final": ref.tolist(),
            "reps": REPS,
            "samples": rows,
        },
        indent=2,
    ))
    print(f"\nWrote raw timings → {DATA_OUT.relative_to(REPO_ROOT)}")

    # ---- Render figure -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.3))

    n_groups = len(TOLERANCES)
    bar_w = 0.36
    x = np.arange(n_groups)

    numra_color = style.WONG[5]   # blue
    scipy_color = style.WONG[6]   # vermilion

    numra_rows = [r for r in rows if r["library"] == "numra-radau5"]
    scipy_rows = [r for r in rows if r["library"] == "scipy-radau"]

    nu_ms = [r["median_ns"] / 1e6 for r in numra_rows]
    sp_ms = [r["median_ns"] / 1e6 for r in scipy_rows]

    ax1.bar(x - bar_w / 2, nu_ms, bar_w, color=numra_color, label="Numra Radau5")
    ax1.bar(x + bar_w / 2, sp_ms, bar_w, color=scipy_color, label="SciPy Radau (Fortran-backed)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$10^{{{int(np.log10(r))}}}$" for r in TOLERANCES])
    ax1.set_xlabel("rtol (atol = rtol·10$^{-3}$)")
    ax1.set_ylabel("Wall-clock (ms, median of 5)")
    ax1.set_yscale("log")
    ax1.set_title("Raw runtime")
    ax1.legend(loc="upper left")

    nu_per_dig = [
        (r["median_ns"] / 1e6) / max(r["correct_digits"], 1e-3) for r in numra_rows
    ]
    sp_per_dig = [
        (r["median_ns"] / 1e6) / max(r["correct_digits"], 1e-3) for r in scipy_rows
    ]

    ax2.bar(x - bar_w / 2, nu_per_dig, bar_w, color=numra_color)
    ax2.bar(x + bar_w / 2, sp_per_dig, bar_w, color=scipy_color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$10^{{{int(np.log10(r))}}}$" for r in TOLERANCES])
    ax2.set_xlabel("rtol")
    ax2.set_ylabel("ms per correct digit")
    ax2.set_yscale("log")
    ax2.set_title("Cost-per-accuracy (lower is better)")

    fig.suptitle(
        rf"Numra vs SciPy on Van der Pol ($\mu = {int(MU)}$), $t \in [0, {int(TF)}]$",
        fontsize=11,
        y=0.99,
    )

    prov = Provenance(
        problem=f"Van der Pol oscillator, mu={MU:g}, t in [0, {TF:g}]",
        solver="Numra Radau5  vs  SciPy solve_ivp(method='Radau')",
        tol="rtol ∈ {1e-4, 1e-6, 1e-8}, atol = rtol·1e-3",
        repro_script="website/figures/comparisons/vdp_stiff.py",
        extra={
            "numra_bench_bin": "numra-bench/src/bin/compare_vdp.rs",
            "scipy_method": "Radau (analytical Jacobian provided)",
            "reference": "SciPy Radau, rtol=atol=1e-12",
            "reps_per_point": str(REPS),
        },
    )
    save_with_provenance(fig, OUT, prov)
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
