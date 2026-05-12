# numra

**Composable numerical methods for Rust — differential equations, optimization, sensitivity analysis, uncertainty quantification, linear algebra, and more in one workspace.**

[![Crates.io](https://img.shields.io/crates/v/numra.svg)](https://crates.io/crates/numra)
[![docs.rs](https://docs.rs/numra/badge.svg)](https://docs.rs/numra)
[![CI](https://github.com/moussaoutlook/numra-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/moussaoutlook/numra-rs/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-numra--rs.org-blue)](https://numra-rs.org/)
[![Book](https://img.shields.io/badge/book-book.numra--rs.org-blue)](https://book.numra-rs.org/)
[![MSRV](https://img.shields.io/badge/MSRV-1.83-blue)](https://github.com/moussaoutlook/numra-rs/blob/main/Cargo.toml)

`numra` is the umbrella crate of the [Numra](https://numra-rs.org/) workspace: twenty native-Rust crates covering ordinary, stochastic, delay, fractional, integro-, and partial differential equations, plus forward sensitivity analysis, uncertainty quantification, optimization, automatic differentiation, linear algebra, statistics, quadrature, interpolation, special functions, FFT, signal processing, and curve fitting — under one unified facade with shared `Scalar` and `Vector` abstractions and a single error model. Dense linear algebra builds on [faer](https://github.com/sarah-ek/faer-rs); the rest is native Rust with no FFI to C or Fortran.

## Install

```toml
[dependencies]
numra = "0.1"
```

To depend on a single subsystem instead of the whole workspace, use the corresponding member crate directly (e.g. `numra-ode = "0.1"`).

## Quick start

```rust
use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0,
    2.0,
    vec![1.0],
);
let opts = SolverOptions::default().rtol(1e-8);
let result = DoPri5::solve(&problem, 0.0, 2.0, &[1.0], &opts).expect("solve");

let y_end = result.y_final().expect("trajectory")[0];
assert!((y_end - (-2.0_f64).exp()).abs() < 1e-5);
```

## Composability

Numra is designed so that subsystems compose without glue: ODE trajectories flow into interpolation, interpolation into quadrature, autodiff Jacobians into optimization, optimization into curve fitting, PDE state vectors into statistics — all sharing one `NumraError` type and the same `Scalar`/`Vector` traits. End-to-end workflows are verified in `numra/tests/interop_workflows.rs`:

| Workflow | Test |
|---|---|
| ODE → cubic spline → quadrature | `workflow_ode_interp_integrate` |
| ODE → FFT → signal processing → peak detection | `workflow_ode_fft_signal_peaks` |
| Parameter estimation → sensitivity → uncertainty | `workflow_param_est_sensitivity_uncertainty` |
| Autodiff → optimization → curve fitting | `workflow_autodiff_optim_fit` |
| PDE method-of-lines → statistics | `workflow_pde_statistics` |
| Sampling → Monte Carlo ODE | `workflow_stats_monte_carlo_ode` |

```bash
cargo test -p numra --test interop_workflows
```

Narrative patterns and worked examples live in the book chapter [Composing solvers](https://book.numra-rs.org/ch12-advanced-topics/composing-solvers/).

## Subsystems

| Subsystem | Module path | Member crate |
|---|---|---|
| Core (Scalar, Vector, Signal, errors, uncertainty) | `numra::{scalar, vector, signal, error, uncertainty}` | [`numra-core`](https://docs.rs/numra-core) |
| ODE / DAE | `numra::ode` | [`numra-ode`](https://docs.rs/numra-ode) |
| SDE | `numra::sde` | [`numra-sde`](https://docs.rs/numra-sde) |
| DDE | `numra::dde` | [`numra-dde`](https://docs.rs/numra-dde) |
| FDE (fractional) | `numra::fde` | [`numra-fde`](https://docs.rs/numra-fde) |
| IDE (Volterra) | `numra::ide` | [`numra-ide`](https://docs.rs/numra-ide) |
| PDE (Method of Lines) | `numra::pde` | [`numra-pde`](https://docs.rs/numra-pde) |
| SPDE | `numra::spde` | [`numra-spde`](https://docs.rs/numra-spde) |
| Linear algebra | `numra::linalg` | [`numra-linalg`](https://docs.rs/numra-linalg) |
| Nonlinear solvers | `numra::nonlinear` | [`numra-nonlinear`](https://docs.rs/numra-nonlinear) |
| Optimization | `numra::optim` | [`numra-optim`](https://docs.rs/numra-optim) |
| Optimal control / parameter estimation | `numra::ocp` | [`numra-ocp`](https://docs.rs/numra-ocp) |
| Quadrature | `numra::integrate` | [`numra-integrate`](https://docs.rs/numra-integrate) |
| Interpolation | `numra::interp` | [`numra-interp`](https://docs.rs/numra-interp) |
| Special functions | `numra::special` | [`numra-special`](https://docs.rs/numra-special) |
| FFT and spectral analysis | `numra::fft` | [`numra-fft`](https://docs.rs/numra-fft) |
| Statistics | `numra::stats` | [`numra-stats`](https://docs.rs/numra-stats) |
| Curve fitting | `numra::fit` | [`numra-fit`](https://docs.rs/numra-fit) |
| Automatic differentiation | `numra::autodiff` | [`numra-autodiff`](https://docs.rs/numra-autodiff) |
| Digital signal processing | `numra::dsp` | [`numra-signal`](https://docs.rs/numra-signal) |

## Examples

Fifteen runnable examples ship with the crate; four span the breadth:

```bash
cargo run --example lorenz                  # ODE + sensitivity + uncertainty
cargo run --example gbm_monte_carlo         # SDE Monte Carlo (geometric Brownian motion)
cargo run --example heat_equation           # 1D PDE via Method of Lines
cargo run --example robertson_sensitivity   # Stiff ODE + forward sensitivity
```

Full list: see `[[example]]` entries in [`numra/Cargo.toml`](https://github.com/moussaoutlook/numra-rs/blob/main/numra/Cargo.toml) or the [examples gallery](https://numra-rs.org/examples/).

## Documentation

- **Book**: <https://book.numra-rs.org/> — narrative chapters per subsystem with executable examples
- **Website**: <https://numra-rs.org/>
- **API reference**: <https://docs.rs/numra>
- **Source**: <https://github.com/moussaoutlook/numra-rs>
- **Changelog**: <https://github.com/moussaoutlook/numra-rs/blob/main/CHANGELOG.md>

## License

Numra is licensed under the **Numra Academic & Research License (Non-Commercial)**.

- **Academic and research use:** free as in freedom.
- **Commercial use:** requires a separate license. Contact `contact@spectralautomata.com` to discuss terms.

See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE) for the full text.
