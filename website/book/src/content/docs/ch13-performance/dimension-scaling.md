---
title: Dimension scaling
sidebar:
  order: 4
---

State dimension is the second axis (after stiffness) along which
solver choice matters. Per-step cost grows differently for explicit
and implicit schemes, and the crossover where one beats the other
moves with `n`.

## The chart

![Wall-clock runtime versus state dimension for DoPri5 (explicit) and
Radau5 (implicit) on a coupled linear ODE. The explicit scheme scales
roughly linearly in n; the implicit scheme pays for a dense linear
solve every Newton iterate, so it scales faster.](../../../assets/figures/perf_dimension_scaling.svg)

## What it shows

The benchmark integrates a tridiagonally-coupled linear ODE,

```text
dy_i/dt = -α y_i + β (y_{i-1} + y_{i+1}),     α = 2,  β = 0.5
```

over `t ∈ [0, 5]`, at `n ∈ {2, 10, 50, 200}`.

- **DoPri5** has slope ≈ `1.0` on a log-log plot of runtime versus
  `n` — each RHS evaluation does `O(n)` work, and the number of steps
  doesn't depend on `n` here, so the total scales linearly.
- **Radau5** has a steeper slope. Each step solves a dense linear
  system inside Newton, which is `O(n³)` for an unstructured
  Jacobian. The `n = 200` data point is deliberately omitted for
  Radau5 — at that size a single Criterion sample takes minutes,
  which says everything you need to know about the cost-per-step
  cliff and nothing useful about the slope you'd actually deploy.

## When does this matter?

The two regimes:

- **Non-stiff problems with large `n`** — stay explicit. Even a
  high-order RK is cheaper than factoring a 200×200 dense matrix
  every step.
- **Stiff problems with structured Jacobians** — the implicit-solver
  cost in this chart is *worst case* (dense Jacobian). For problems
  where the Jacobian is banded, sparse, or block-structured, the
  cost-per-step drops dramatically. See the [linear algebra](../../ch06-linear-algebra/sparse-matrices/)
  chapter for what Numra exposes there.

If you're unsure: pass `Hints::dimension(n)` to `auto_solve_with_hints`
and the heuristic will pick an appropriate scheme.

## What's not in the chart

This is a non-stiff problem. For *stiff* high-dimensional problems,
the calculus shifts — the implicit solver may still win by a wide
margin because its step size doesn't shrink with stiffness, even if
each step is more expensive. The PDE method-of-lines bench (separate
file `bench_pde_mol.rs`) explores that regime; see the appendix's
solver-reference for the underlying tradeoffs.
