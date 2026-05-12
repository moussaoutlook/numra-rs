# numra-dde

**Delay differential equation solvers for the [Numra](https://numra-rs.org/) workspace — method of steps with embedded Runge-Kutta and Hermite history interpolation.**

[![Crates.io](https://img.shields.io/crates/v/numra-dde.svg)](https://crates.io/crates/numra-dde)
[![docs.rs](https://docs.rs/numra-dde/badge.svg)](https://docs.rs/numra-dde)

Solves DDEs `y'(t) = f(t, y(t), y(t − τ₁), y(t − τ₂), …)` with constant or state-dependent delays. The method of steps integrates between delay breakpoints with an embedded RK solver and reconstructs `y(t − τ)` from a Hermite interpolant of the recorded trajectory.

## Example

```rust
use numra_dde::{DdeOptions, DdeSolver, DdeSystem, MethodOfSteps};

// Mackey-Glass: y'(t) = β · y(t-τ) / (1 + y(t-τ)^n) - γ · y(t)
struct MackeyGlass { beta: f64, gamma: f64, n: f64, tau: f64 }

impl DdeSystem<f64> for MackeyGlass {
    fn dim(&self) -> usize { 1 }
    fn delays(&self) -> Vec<f64> { vec![self.tau] }
    fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        let y_tau = y_delayed[0][0];
        dydt[0] = self.beta * y_tau / (1.0 + y_tau.powf(self.n)) - self.gamma * y[0];
    }
}

let mg = MackeyGlass { beta: 2.0, gamma: 1.0, n: 9.65, tau: 2.0 };
let history = |_t: f64| vec![0.5];
let opts = DdeOptions::default();
let _ = MethodOfSteps::solve(&mg, 0.0, 100.0, &history, &opts);
```

## What's in this crate

- **`MethodOfSteps`** — embedded-RK method-of-steps solver
- **`DdeSystem` trait** — user-defined RHS with one or more delays, optional state-dependent
- **`History` / `HistoryFunction` / `HermiteInterpolator`** — initial-history representation and continuous extension
- **`DdeOptions` / `DdeResult` / `DdeStats`** — tolerance / step-control configuration and diagnostics

## Composes with

- [`numra-ode`](https://docs.rs/numra-ode) — the RK steppers that drive between-delay integration
- [`numra-stats`](https://docs.rs/numra-stats) — descriptive statistics on chaotic attractors (e.g. range, percentile)
- [`numra-fft`](https://docs.rs/numra-fft) — spectral analysis of long-time DDE solutions
- [`numra-interp`](https://docs.rs/numra-interp) — post-hoc resampling of the trajectory onto fixed grids

See [cross-formalism tests](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/composition_tests.rs) for the DDE-approximates-ODE consistency check.

## Install

```toml
[dependencies]
numra-dde = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-dde>
- **Book**: [Delay DEs](https://book.numra-rs.org/ch04-beyond-odes/delay-des/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-dde>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
