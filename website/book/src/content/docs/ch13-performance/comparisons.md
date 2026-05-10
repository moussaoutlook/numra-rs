---
title: Numra vs SciPy
sidebar:
  order: 6
---

This page exists because telling you Numra is fast in the abstract is not
useful — you want to know how it compares to the library you're already
using. This chapter focuses on **one scoped problem class**: a stiff
non-linear IVP (Van der Pol with $\mu = 10$ on $t \in [0, 20]$), solved at
three tolerances ($10^{-4}$, $10^{-6}$, $10^{-8}$), with the same
algorithm — Radau IIA — on both sides.

Broader comparisons (SUNDIALS, ndarray-linalg, rustfft) are queued for a
later chapter — single problem classes are easier to keep honest.

## The chart

![Two-panel bar chart comparing Numra Radau5 and SciPy
solve_ivp(method='Radau') on the Van der Pol oscillator at mu=10. Left
panel: wall-clock vs tolerance. Right panel: wall-clock per correct digit
of the final state, where correct digits are -log10 of relative L2 error
against an rtol=1e-12 reference solution.](../../../assets/figures/comparison_vdp_stiff.svg)

## What it shows

| rtol | Numra Radau5 (median) | SciPy Radau (median) | Speedup | Numra correct digits | SciPy correct digits |
|---|---|---|---|---|---|
| $10^{-4}$ | 0.66 ms | 12.13 ms | 18.4× | 5.07 | 6.52 |
| $10^{-6}$ | 1.71 ms | 31.64 ms | 18.5× | 8.86 | 8.76 |
| $10^{-8}$ | 5.32 ms | 92.23 ms | 17.3× | 11.12 | 11.01 |

The two implementations are running the same algorithm — 3-stage Radau
IIA from Hairer–Wanner §IV.8, with a real-Schur transformation and
simplified Newton iteration — so this is the cleanest possible apples-to-apples
comparison. Numra wins by ~17–18× on wall-clock at every tolerance, and
matches or marginally beats SciPy on accuracy at tighter tolerances.

The SciPy library's `solve_ivp(method='Radau')` is a Python port of
Hairer's `radau5.f` Fortran reference, so most of the wall-clock gap is
the per-step Python interpreter overhead and the cost of bouncing
through `scipy.integrate.OdeSolver`'s class hierarchy on every step.
Numra's implementation runs the same numerics in native Rust without
that overhead.

## Where Numra is slower

It would be misleading to leave the story there. There are concrete cases
where SciPy and other established libraries beat Numra today:

- **At very tight tolerances on smooth non-stiff problems**, SciPy's
  `LSODA` automatic stiffness detection often wins on wall-clock when the
  user picks the wrong solver. Numra's `auto_solve_with_hints` is intended
  to close this gap but is less battle-tested than `LSODA`.
- **DAE problems with index reduction**: SUNDIALS IDA remains
  significantly more capable than Numra's current DAE machinery for
  index-2+ problems with sparse Jacobians. This is the comparison that
  will land in a later chapter; for now, Numra's DAE support is best for
  index-1 problems.
- **Ecosystem**: SciPy ships interpolation, statistics, optimization,
  signal processing, and a thousand other tools that the Numra workspace
  does not (yet) match. If you need one numerical tool integrated into a
  larger Python workflow, SciPy is a less surprising default.

## How to reproduce

```bash
# Build the Numra-side timing harness:
cargo build -p numra-bench --release --bin compare_vdp

# Render the chart (drives the binary, runs SciPy in-process):
cd website/figures
uv run python comparisons/vdp_stiff.py
```

The raw timings are checked in at
[`website/figures/comparisons/data/vdp_stiff.json`](https://github.com/moussaoutlook/numra-rs/blob/main/website/figures/comparisons/data/vdp_stiff.json),
so the chart can be regenerated on a different machine without re-running
the benchmark.

## What this chapter does NOT do

- It compares one solver pair on one problem at three tolerances. Don't
  generalise to "Numra is 18× faster than SciPy" — *on different problems
  and at different tolerances the answer changes*.
- It compares the same algorithm on both sides (Radau IIA). The point is
  to isolate runtime overhead, not to find the slowest pair on either
  side and trumpet a misleading speedup.
- It compares against SciPy specifically. SUNDIALS (CVODE/IDA) is the
  state-of-the-art reference for stiff IVPs and sparse-Jacobian DAEs, and
  Numra has more ground to make up there. That comparison is queued.
