---
title: "Method Cards"
---

These cards are practical starting points for choosing Numra methods. They complement the solver tables with failure modes, tolerance guidance, and traceability to `docs/audit/inventory.yaml`.

## ODE Explicit Runge-Kutta

**Inventory ids:** `ode.dopri5`, `ode.tsit5`, `ode.verner`  
**Rustdoc:** `numra-ode` explicit solvers

- **Use when:** The system is non-stiff, smooth enough for adaptive explicit steps, and RHS evaluations are cheap compared with linear solves.
- **Typical tolerances:** Start with `rtol = 1e-6`, `atol = 1e-9`; use Verner methods for tighter tolerances around `1e-9` to `1e-12`.
- **Avoid when:** Step sizes collapse, high-frequency transients force excessive rejections, or the model contains DAEs/algebraic constraints.
- **Failure modes:** Missed stiffness, overly small absolute tolerances near zero-valued components, and event functions that oscillate faster than the step controller can resolve.

## ODE Implicit And BDF

**Inventory ids:** `ode.esdirk`, `ode.radau5`, `ode.bdf`, `ode.auto`  
**Rustdoc:** `numra-ode` implicit solvers

- **Use when:** The system is stiff, has fast decaying modes, or comes from a semi-discretized PDE or DAE.
- **Typical tolerances:** Start with `rtol = 1e-6`, `atol = 1e-8` for exploratory work; tighten once scaling and residual behavior are understood.
- **Avoid when:** The problem is small and clearly non-stiff; explicit methods are often simpler and cheaper.
- **Failure modes:** Singular or ill-conditioned Jacobians, inconsistent DAE initial values, poor component scaling, and hidden constraints that need index reduction.

## ODE Events, Dense Output, And DAE Diagnostics

**Inventory ids:** `ode.events`, `ode.dae`  
**Rustdoc:** `numra-ode` events and index reduction modules

- **Use when:** You need terminal conditions, crossing times, sampled output, consistent initialization, or structural DAE diagnostics.
- **Typical tolerances:** Match event tolerances to solution tolerances; use smaller `h_max` when roots are clustered or near the initial time.
- **Avoid when:** The event function is discontinuous or encodes a broad inequality instead of a sign-changing root.
- **Failure modes:** Multiple roots inside one accepted step, tangential roots with no sign change, and algebraic constraints that are only approximately satisfied.

## SDE Solvers

**Inventory ids:** `sde.euler_maruyama`, `sde.milstein`, `sde.sra`  
**Rustdoc:** `numra-sde`

- **Use when:** The model includes stochastic forcing and you can validate output statistically rather than path-by-path.
- **Typical settings:** Use fixed seeds for reproducibility; choose `dt` by convergence studies, often starting near `1e-3` to `1e-2` for scalar examples.
- **Avoid when:** You need deterministic exact trajectories, or the quantity of interest is too sensitive for a single path.
- **Failure modes:** Bands that are too tight for Monte Carlo variance, negative states in multiplicative-noise models without positivity-preserving transformations, and interpreting strong-order behavior as distributional proof.

## DDE, FDE, IDE, PDE, And SPDE

**Inventory ids:** `dde.method_of_steps`, `fde.l1_caputo`, `ide.volterra_prony`, `pde.mol`, `spde.heat`  
**Rustdoc:** equation-family crates

- **Use when:** The equation type has delay, memory, spatial coupling, fractional order, or stochastic spatial forcing that cannot be represented as a plain ODE without losing the model structure.
- **Typical settings:** Start with conservative step sizes and small grids; compare against limiting ODE/PDE cases where available.
- **Avoid when:** A simpler ODE or deterministic PDE formulation captures the behavior you need.
- **Failure modes:** History interpolation artifacts in DDEs, fractional-memory cost growth, spatial discretization dominating time-stepping error, and stochastic forcing variance mistaken for deterministic instability.

## Optimization

**Inventory ids:** `optim.overview`, `optim.lp_qp_milp`  
**Rustdoc:** `numra-optim`

- **Use when:** The problem class is explicit: smooth unconstrained, bound-constrained, constrained nonlinear, LP/QP/MILP, global, multi-objective, or robust/stochastic optimization.
- **Typical tolerances:** Use default tolerances first; for LP/QP/MILP, add small golden problems with known optima before trusting larger models.
- **Avoid when:** Objective or constraint scaling is unknown, gradients are inconsistent, or integer formulations are not bounded tightly enough for branch-and-bound.
- **Failure modes:** Local minima, infeasible constraints masked by loose tolerances, poorly scaled gradients, and non-unique optima used as brittle golden vectors.

## Analysis Utilities

**Inventory ids:** `integrate.quadrature`, `interp.curves`, `autodiff.gradients`, `special.functions`, `fit.curves`, `fft.spectral`, `dsp.signal`, `stats.overview`, `core.traits`, `linalg.matrix`, `nonlinear.newton`, `ocp.overview`  
**Rustdoc:** corresponding crate modules

- **Use when:** You need building blocks that feed solvers: interpolation, quadrature, transforms, statistics, fitting, gradients, nonlinear solves, or matrix factorizations.
- **Typical settings:** Validate each building block independently on a closed-form or synthetic case before composing it into a larger workflow.
- **Avoid when:** The problem’s conditioning or sampling assumptions are unknown; measure those first.
- **Failure modes:** Aliasing in FFT/signal workflows, extrapolation in interpolation, ill-conditioned least squares, finite-difference sensitivity to scale, and statistical tests applied outside their assumptions.
