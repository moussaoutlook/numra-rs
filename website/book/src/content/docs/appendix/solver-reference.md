---
title: "Solver Reference Table"
---

A reference table for **ODE, SDE, and DDE** solver types exported by Numra. For optimization, quadrature, interpolation, and other numerical APIs, see the main chapters and the inventory file `docs/audit/public-api-method-inventory.md` in the Numra repository.

## Explicit Runge-Kutta Methods (Non-Stiff)

| Solver | Order | Embedded | Stages | FSAL | Import |
|--------|-------|----------|--------|------|--------|
| `DoPri5` | 5 | 4 | 7 | No | `numra::ode::DoPri5` |
| `Tsit5` | 5 | 4 | 7 | Yes | `numra::ode::Tsit5` |
| `Vern6` | 6 | 5 | 9 | No | `numra::ode::Vern6` |
| `Vern7` | 7 | 6 | 10 | No | `numra::ode::Vern7` |
| `Vern8` | 8 | 7 | 13 | No | `numra::ode::Vern8` |

### When to use each

- **DoPri5**: General-purpose default. The "workhorse" for non-stiff problems.
  Good for rtol ~ 1e-6 to 1e-8.

- **Tsit5**: Slightly more efficient than DoPri5 thanks to the FSAL property
  (the last stage of step n is the first stage of step n+1, saving one RHS
  call per step). Preferred when many steps are needed.

- **Vern6**: When you need 6th-order accuracy. Good for rtol ~ 1e-8 to 1e-10.

- **Vern7**: High-accuracy problems. Good for rtol ~ 1e-10 to 1e-12.

- **Vern8**: Maximum accuracy for non-stiff problems. Most efficient at
  very tight tolerances (rtol < 1e-12).

## Implicit Runge-Kutta Methods (Stiff)

| Solver | Order | Stages | L-stable | A-stable | Import |
|--------|-------|--------|----------|----------|--------|
| `ESDIRK32` | 2(1) | 3 | Yes | Yes | `numra::ode::Esdirk32` |
| `ESDIRK43` | 3(2) | 4 | Yes | Yes | `numra::ode::Esdirk43` |
| `ESDIRK54` | 4(3) | 6 | Yes | Yes | `numra::ode::Esdirk54` |
| `Radau5` | 5 | 3 | Yes | Yes | `numra::ode::Radau5` |

### When to use each

- **ESDIRK32**: Mildly stiff problems where 2nd-order accuracy suffices.

- **ESDIRK43**: Moderate stiffness with 3rd-order accuracy.

- **ESDIRK54**: Stiff problems requiring 4th-order accuracy.

- **Radau5**: The gold standard for severely stiff problems and DAEs.
  Handles stiffness ratios exceeding 10^11 (Robertson problem).

## Multistep Methods

| Solver | Order | Type | Import |
|--------|-------|------|--------|
| `Bdf` | 1-5 (variable) | BDF | `numra::ode::Bdf` |

## Automatic Solver

| Solver | Strategy | Import |
|--------|----------|--------|
| `Auto` | Starts explicit, switches to implicit if stiff | `numra::ode::Auto` |

## SDE Solvers

| Solver | Strong order | Weak order | Import |
|--------|-------------|------------|--------|
| `EulerMaruyama` | 0.5 | 1.0 | `numra::sde::EulerMaruyama` |
| `Milstein` | 1.0 | 1.0 | `numra::sde::Milstein` |
| `Sra1` | 1.0–1.5 (adaptive; additive noise) | — | `numra::sde::Sra1` |
| `Sra2` | — | 2.0 (adaptive) | `numra::sde::Sra2` |

These types implement the literature’s strong-order-1.5 / weak-order-2 **SRA**-style adaptive schemes (see `numra-sde` and algorithm references). Older drafts incorrectly used a different spelling; the public Rust types are **`Sra1`** and **`Sra2`** only.

## DDE Solvers

| Solver | Order | Import |
|--------|-------|--------|
| `MethodOfSteps` | 5(4) embedded RK internally | `numra::dde::MethodOfSteps` |

The DDE integrator is exposed as **`MethodOfSteps`** (method of steps with an embedded Dormand–Prince pair). Do not use non-exported legacy type names from older drafts.

For decision-oriented guidance beyond these tables, see the [method cards](method-cards).

## Common Usage Pattern

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0, 10.0,
    vec![1.0],
);

let options = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10);

let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0], &options).unwrap();
```

## SolverOptions Reference

| Option | Default | Builder method | Description |
|--------|---------|----------------|-------------|
| rtol | 1e-6 | `.rtol(v)` | Relative tolerance |
| atol | 1e-9 | `.atol(v)` | Absolute tolerance |
| h0 | Auto | `.h0(v)` | Initial step size |
| h_max | Infinity | `.h_max(v)` | Maximum step size |
| h_min | 1e-14 | `.h_min(v)` | Minimum step size |
| max_steps | 100,000 | `.max_steps(n)` | Maximum number of steps |
| output times | None | see below | Output at specific times |
| dense_output | false | `.dense()` | Enable dense output |
| events | [] | `.event(box)` | Event functions |

To specify output times, use the `t_eval` builder method with a `Vec<S>`.

## SolverResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `t` | `Vec<S>` | Time points |
| `y` | `Vec<S>` | Solution (row-major: `y[i*dim + j]`) |
| `dim` | `usize` | System dimension |
| `stats` | `SolverStats` | Performance statistics |
| `success` | `bool` | Whether integration succeeded |
| `message` | `String` | Error message if failed |
| `events` | `Vec<Event<S>>` | Detected events |

### SolverResult Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `y_final()` | `Option<Vec<S>>` | Final state vector |
| `t_final()` | `Option<S>` | Final time |
| `y_at(i)` | `&[S]` | State at time index i |
| `component(j)` | `Vec<S>` | j-th variable over all times |
| `iter()` | Iterator | `(t, &[S])` pairs |
| `n_steps()` | `usize` | Number of time steps |

### SolverStats Fields

| Field | Description |
|-------|-------------|
| `n_accept` | Accepted steps |
| `n_reject` | Rejected steps |
| `n_jac` | Jacobian computations |
| `n_lu` | LU decompositions |
