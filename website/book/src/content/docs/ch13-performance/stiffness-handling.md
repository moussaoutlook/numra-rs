---
title: Stiffness handling
sidebar:
  order: 3
---

When a problem develops sharp transitions between fast and slow
dynamics, an explicit solver's stability footprint shrinks until each
step shrinks with it. An implicit solver's per-step cost is higher —
it has to factor a Jacobian and solve a Newton iteration — but its
stability is unconditional, so the step size stays sane.

## The chart

![Wall-clock runtime versus the Van der Pol stiffness parameter μ for
Radau5, BDF, and ESDIRK54. All three are implicit; their slopes on a
log-log plot expose how each scheme amortises Jacobian reuse and
factorisation across the integration.](../../../assets/figures/perf_stiffness_handling.svg)

## What it shows

- The integration interval here is `t ∈ [0, 2μ]`. As `μ` grows the
  problem gets *both* longer and stiffer — runtime should grow at
  least linearly in `μ` purely from interval length, plus more from
  step-count growth.
- **Radau5** dominates at every `μ` in the swept range. The 5th-order
  L-stable Radau IIA scheme combined with the rewritten step
  controller (Hairer–Wanner §IV.8 + Gustafsson predictive controller)
  takes large, well-sized strides through the slow phase and adapts
  cleanly through the transients.
- **ESDIRK54** sits a small constant factor behind Radau5. Its
  singly-diagonally-implicit structure lets each Newton iterate factor
  the same Jacobian block, which is a useful win on systems where
  factorisation dominates.
- **BDF** is roughly flat in `μ` here — its per-step cost is small but
  the variable-order controller lingers at low order on the slow
  phase, so accumulated step counts dominate.

## How to choose

The pragmatic rule of thumb:

| Stiffness scale                    | Recommended starting solver |
|------------------------------------|-----------------------------|
| Stiff IVPs and DAEs (default)      | `Radau5`                    |
| Smooth coefficients, prefer SDIRK  | `ESDIRK54`                  |
| Long-horizon, low-order acceptable | `BDF`                       |

When in doubt, use `auto_solve_with_hints` and pass `Stiffness::High`
— it will pick from the same list using these crossover points as
defaults.

## What's not in the chart

This bench fixes tolerances at `rtol = 1e-4, atol = 1e-6`. Tightening
tolerances changes the *absolute* numbers but not the slopes — the
ranking is robust across a couple of decades of tolerance.

For the explicit-vs-implicit crossover (when does it stop making
sense to push DoPri5 through a stiff problem at all?), see the
[dimension-scaling](./dimension-scaling/) page.
