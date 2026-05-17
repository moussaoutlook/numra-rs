# numra-ode

**ODE and DAE solvers for the [Numra](https://numra-rs.org/) workspace — with forward sensitivity, event detection, and automatic stiffness handling.**

[![Crates.io](https://img.shields.io/crates/v/numra-ode.svg)](https://crates.io/crates/numra-ode)
[![docs.rs](https://docs.rs/numra-ode/badge.svg)](https://docs.rs/numra-ode)

Solves initial-value problems `dy/dt = f(t, y), y(t0) = y0` with explicit Runge-Kutta, implicit Runge-Kutta, and multistep methods. Includes forward sensitivity analysis, DAE consistent-initialization, structural index reduction, event handling, and parameter-uncertainty propagation.

## Example

```rust
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};

// Lorenz system
let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
        dydt[0] = sigma * (y[1] - y[0]);
        dydt[1] = y[0] * (rho - y[2]) - y[1];
        dydt[2] = y[0] * y[1] - beta * y[2];
    },
    0.0, 20.0, vec![1.0, 1.0, 1.0],
);

let opts = SolverOptions::default().rtol(1e-8);
let result = DoPri5::solve(&problem, 0.0, 20.0, &[1.0, 1.0, 1.0], &opts).unwrap();
```

## What's in this crate

**Solver families:**

| Family | Solvers |
|---|---|
| Explicit Runge-Kutta (non-stiff) | `DoPri5`, `Tsit5`, `Vern6`, `Vern7`, `Vern8` |
| Implicit Runge-Kutta (stiff)     | `Radau5`, `Esdirk32`, `Esdirk43`, `Esdirk54` |
| Multistep                        | `Bdf` (variable order 1–5) |
| Automatic selection              | `Auto`, `auto_solve`, `auto_solve_with_hints` |

**Additional capabilities:**

- **Forward parameter sensitivity**: `ParametricOdeSystem`, `solve_forward_sensitivity`, `SensitivityResult` — `S(t) = ∂y(t)/∂p` via the variational equations.
- **Initial-condition sensitivity (state-transition matrix)**: `solve_initial_condition_sensitivity`, `StateTransitionResult` — `Φ(t) = ∂y(t)/∂y₀` over any `Solver`; the same variational machinery seeded as `S(t₀) = I`. Foundation for multiple-shooting / boundary-value methods, Lyapunov-exponent and Floquet stability analysis, periodic-orbit and monodromy computations.
- **Autodiff-derived `J_y`** (cargo feature `autodiff`): `AutodiffJacobianSystem` — an `OdeSystem` adapter that obtains `J_y = ∂f/∂y` via forward-mode automatic differentiation through `numra-autodiff::Dual<S>`. Round-off-exact Jacobian, zero allocation per step. Opt-in by cargo feature so callers who don't need it pay zero in compile time and binary size.
- **DAEs**: `DaeProblem`, `compute_consistent_initial`, structural index reduction (`reduce_dae_problem`, `analyze_dae_index`)
- **Events**: zero-crossing detection with state resets (`bouncing_ball`-style)
- **Uncertainty**: `UncertainParam`, `solve_with_uncertainty`, `solve_monte_carlo`
- **Diagnostics**: `SolverStats` on every result (accepted/rejected steps, function/Jacobian/LU evaluations)

## Composes with

ODE trajectories from this crate flow into:

- [`numra-interp`](https://docs.rs/numra-interp) for cubic-spline / Akima resampling of the trajectory onto arbitrary points
- [`numra-integrate`](https://docs.rs/numra-integrate) for quadrature of functions defined on the trajectory
- [`numra-fft`](https://docs.rs/numra-fft) for spectral analysis of the solution
- [`numra-signal`](https://docs.rs/numra-signal) for peak detection and filtering on the solution
- [`numra-ocp`](https://docs.rs/numra-ocp) for parameter estimation of ODE models against measured data

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for verified end-to-end examples.

## Install

```toml
[dependencies]
numra-ode = "0.1"
```

Or pull in the whole Numra workspace via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-ode>
- **Book**: [Solving ODEs](https://book.numra-rs.org/ch02-solving-odes/) · [Stiff systems](https://book.numra-rs.org/ch03-stiff-systems/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-ode>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
