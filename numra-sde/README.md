# numra-sde

**Stochastic differential equation solvers for the [Numra](https://numra-rs.org/) workspace — Euler-Maruyama, Milstein, adaptive SRA1 / SRA2, with built-in Monte Carlo ensemble runner.**

[![Crates.io](https://img.shields.io/crates/v/numra-sde.svg)](https://crates.io/crates/numra-sde)
[![docs.rs](https://docs.rs/numra-sde/badge.svg)](https://docs.rs/numra-sde)

Solves SDEs `dX = f(t, X) dt + g(t, X) dW` with strong-order 0.5 / 1.0 fixed-step and adaptive higher-order methods. The ensemble runner parallelizes Monte Carlo realizations via Rayon and ships with running-statistics primitives.

## Example

```rust
use numra_sde::{EulerMaruyama, SdeOptions, SdeSolver, SdeSystem};

// Geometric Brownian Motion: dS = μS dt + σS dW
struct Gbm { mu: f64, sigma: f64 }

impl SdeSystem<f64> for Gbm {
    fn dim(&self) -> usize { 1 }
    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) { f[0] = self.mu * x[0]; }
    fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) { g[0] = self.sigma * x[0]; }
}

let gbm = Gbm { mu: 0.05, sigma: 0.2 };
let opts = SdeOptions::default().dt(0.01);
let result = EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &opts, None);
assert!(result.is_ok());
```

## What's in this crate

**Solvers:**

- **`EulerMaruyama`** — strong order 0.5, fixed step
- **`Milstein`** — strong order 1.0 (Itô)
- **`Sra1`** — adaptive strong order 1.5 for additive-noise SDEs
- **`Sra2`** — adaptive weak order 2.0

**Ensemble & noise:**

- **`EnsembleRunner` / `EnsembleResult`** — Rayon-parallel Monte Carlo
- **`WienerProcess` / `WienerIncrement` / `create_wiener`** — reproducible Brownian increments
- **`RunningStats`, `EnsembleStats`, `mean`, `variance`, `std`, `median`, `percentile`, `Percentiles`** — online and offline summaries

## Composes with

- [`numra-spde`](https://docs.rs/numra-spde) — SDE time-stepping inside the Method-of-Lines for stochastic PDEs
- [`numra-stats`](https://docs.rs/numra-stats) — descriptive statistics on ensemble trajectories
- [`numra-interp`](https://docs.rs/numra-interp) — resampling sample paths onto common time grids
- [`numra-signal`](https://docs.rs/numra-signal) — filtering and peak detection on noisy realizations

See [cross-formalism tests](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/composition_tests.rs) for SDE reproducibility and composition checks.

## Install

```toml
[dependencies]
numra-sde = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-sde>
- **Book**: [Stochastic DEs](https://book.numra-rs.org/ch04-beyond-odes/stochastic-des/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-sde>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
