---
title: Tolerance vs runtime
sidebar:
  order: 2
---

For a fixed problem, *runtime is a smooth function of accuracy*. As
you tighten `rtol`, the adaptive controller takes more, smaller steps
and the wall-clock cost rises. The interesting question is *how
quickly* — and that depends on the order of the embedded RK pair.

## The chart

![Tolerance versus wall-clock runtime for DoPri5, Tsit5, and Vern6 on
the Lorenz system. All three are adaptive explicit RK pairs; the
higher-order Vern6 amortises its per-step cost at tight tolerances.](../../../assets/figures/perf_tolerance_runtime.svg)

## What it shows

- **At slack tolerances (`rtol = 1e-3`)** the lower-order schemes
  (DoPri5, Tsit5) win on wall-clock — they take roughly the same
  number of steps as Vern6 but each step is cheaper.
- **At tight tolerances (`rtol = 1e-9`)** the higher-order Vern6
  pulls ahead. A 6th-order pair needs *fewer* steps to hit the same
  accuracy, and once the per-step overhead is amortised over enough
  function evaluations the order advantage dominates.
- **The slope on a log-log plot** is roughly `0.1–0.2` for all three
  schemes — meaning a 6-decade tightening of tolerance costs about
  one decade of runtime. That's the adaptive controller earning its
  keep.

## How to read this for your problem

The crossover between Tsit5 and Vern6 sits between `1e-6` and `1e-7`
on Lorenz. For *your* problem the crossover may be elsewhere:

- **Smoother right-hand sides** push the crossover toward looser
  tolerances (the higher-order scheme wins more often).
- **Stiffer or oscillatory dynamics** push it the other way and may
  push you toward an implicit solver entirely — see the
  [stiffness-handling](./stiffness-handling/) page.
- **Very expensive RHS evaluations** also favour higher-order schemes
  because each saved step is a saved RHS evaluation.

The Numra `auto_solve` heuristic uses these crossovers as defaults
when you don't pick a solver explicitly.
