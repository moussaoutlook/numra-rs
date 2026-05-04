---
title: Dense-output overhead
sidebar:
  order: 5
---

Dense output gives you a continuous interpolant of the solution
between discrete time steps — useful for event detection, smooth
plotting, and downstream operations like quadrature. It's not free,
but it's cheaper than you'd guess.

## The chart

![Bar chart comparing wall-clock runtime of DoPri5 with and without
dense output enabled, on the Lorenz system over t ∈ [0, 10]. Dense
output adds a small fixed per-step cost for the embedded interpolant
coefficients.](../../../assets/figures/perf_dense_output_overhead.svg)

## What it shows

For `DoPri5` on Lorenz over `t ∈ [0, 10]` at `rtol = 1e-6`:

- The baseline (no dense output) reports just the `(t, y)` pairs at
  the controller's chosen step locations.
- Enabling dense output adds the cost of computing the embedded
  5th-order interpolant coefficients at each accepted step.
- On Lorenz at this tolerance the overhead lands around 25–30% of
  total runtime. That number is problem-specific: when the integrator
  is dominated by RHS evaluations (stiff problems, large state, or
  expensive RHS) the relative cost of forming the interpolant
  shrinks; on a cheap RHS like Lorenz the interpolant work is a
  bigger slice of the total.

## When to enable it

Enable dense output when you need any of:

- **Event detection** — locating where a continuous predicate
  changes sign between steps. The bouncing-ball example
  (`numra/examples/bouncing_ball.rs`) leans on this.
- **Plotting** — sampling the solution on a fine, regular grid
  for visualization, without forcing the solver to take fine steps.
- **Downstream quadrature** — passing the trajectory to
  `numra::integrate::gauss_kronrod_quadrature` or any other
  integrator that needs a callable.

If you only need the discrete `(t, y)` output, skip it — there's no
sense paying a couple of percent for state you'll never look at.

## A note on continuous output

For solvers that don't ship a native dense-output interpolant
(e.g. some implicit BDF variants), Numra falls back to a cubic
spline through the discrete steps. That spline is *less accurate*
than the embedded interpolant — the per-step error is `O(h⁴)`
instead of matching the underlying scheme's order. Watch the
caption on each per-solver chart for which kind of dense output
you're getting.
