---
title: Numra vs SciPy
sidebar:
  order: 6
---

This page exists because telling you Numra is fast in the abstract is not
useful — you want to know how it compares to the library you're already
using. This chapter focuses on **one scoped problem class**: a stiff
non-linear IVP (Van der Pol with $\mu = 10$ on $t \in [0, 20]$), solved at
three tolerances ($10^{-4}$, $10^{-6}$, $10^{-8}$), with one solver per
library.

Broader comparisons (SUNDIALS, ndarray-linalg, rustfft) are queued for a
later chapter — single problem classes are easier to keep honest.

## The chart

![Two-panel bar chart comparing Numra ESDIRK54 and SciPy
solve_ivp(method='Radau') on the Van der Pol oscillator at mu=10. Left
panel: wall-clock vs tolerance. Right panel: wall-clock per correct digit
of the final state, where correct digits are -log10 of relative L2 error
against an rtol=1e-12 reference solution.](../../../assets/figures/comparison_vdp_stiff.svg)

## What it shows

| rtol | Numra ESDIRK54 (median) | SciPy Radau (median) | Speedup | Numra correct digits | SciPy correct digits |
|---|---|---|---|---|---|
| $10^{-4}$ | 1.26 ms | 12.4 ms | 9.8× | 5.1 | 6.5 |
| $10^{-6}$ | 2.8 ms  | 33.2 ms | 11.7× | 6.7 | 8.8 |
| $10^{-8}$ | 7.1 ms  | 99.4 ms | 14.0× | 8.7 | 11.0 |

Two things to notice. First, on raw wall-clock Numra wins by an order of
magnitude at every tolerance. Second, SciPy's Radau implementation
(Hairer/Wanner Fortran code wrapped by SciPy) delivers more correct digits
than its tolerance setting requests, while Numra ESDIRK54 hits roughly the
tolerance you asked for. That over-delivery costs SciPy real wall-clock
time.

The cost-per-correct-digit panel normalises that: Numra is still ahead by
roughly 4–5× in *useful work per millisecond*, even after giving SciPy
credit for extra accuracy.

## Where Numra is slower (per SPEC §17)

It would be misleading to leave the story there. There are concrete cases
where SciPy and other established libraries beat Numra today:

- **Numra's own `Radau5` is slower than SciPy's Radau on this problem.** A
  separate measurement at `rtol = 1e-4` puts Numra `Radau5` at ~200 ms vs
  SciPy Radau's ~12 ms. The Numra `Radau5` does many more steps than the
  Hairer reference implementation does — Numra ships ESDIRK54 as the
  recommended stiff solver for this regime for that reason. Improving
  `Radau5` step-control is on the [roadmap](https://numra-rs.org/roadmap).
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
  generalise to "Numra is 10× faster than SciPy" — *on different problems
  and at different tolerances the answer changes*.
- It uses Numra's recommended stiff solver (ESDIRK54) and SciPy's
  recommended stiff solver (Radau). It does not try to find the slowest
  pair on either side and trumpet a misleading speedup.
- It compares against SciPy specifically. SUNDIALS (CVODE/IDA) is the
  state-of-the-art reference for stiff IVPs and sparse-Jacobian DAEs, and
  Numra has more ground to make up there. That comparison is queued.
