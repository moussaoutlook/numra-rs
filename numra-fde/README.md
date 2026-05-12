# numra-fde

**Fractional differential equation solvers for the [Numra](https://numra-rs.org/) workspace — L1 scheme for Caputo derivatives, with Mittag-Leffler utilities for analytical comparison.**

[![Crates.io](https://img.shields.io/crates/v/numra-fde.svg)](https://crates.io/crates/numra-fde)
[![docs.rs](https://docs.rs/numra-fde/badge.svg)](https://docs.rs/numra-fde)

Solves time-fractional ODEs `D^α y(t) = f(t, y)` for α ∈ (0, 1] using the Caputo derivative. Reduces to the ordinary derivative at α = 1; preserves the "memory" property for α < 1 (the right-hand side depends on the full history of y).

## Example

```rust
use numra_fde::{FdeOptions, FdeSolver, FdeSystem, L1Solver};

// Fractional relaxation: D^α y = -λy, y(0) = 1
struct FractionalRelaxation { lambda: f64 }

impl FdeSystem<f64> for FractionalRelaxation {
    fn dim(&self) -> usize { 1 }
    fn alpha(&self) -> f64 { 0.5 }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.lambda * y[0];
    }
}

let opts = FdeOptions::default().dt(0.01);
let _ = L1Solver::solve(&FractionalRelaxation { lambda: 1.0 }, 0.0, 1.0, &[1.0], &opts);
```

## What's in this crate

- **`L1Solver`** — L1 scheme for the Caputo derivative, accuracy `O(dt^{2-α})`
- **`FdeSystem` trait** — user-defined RHS plus fractional order `α`
- **`FdeOptions` / `FdeResult`** — configuration and diagnostics
- **`caputo_weights`** — quadrature weights for direct use
- **`mittag_leffler`, `mittag_leffler_1`** — closed-form Mittag-Leffler function for analytical reference (the FDE analogue of `exp` for fractional relaxation)
- **`gamma`** — gamma function used internally and re-exported

## Composes with

- [`numra-interp`](https://docs.rs/numra-interp) for post-hoc resampling of the trajectory onto fixed grids
- [`numra-stats`](https://docs.rs/numra-stats) for descriptive statistics on fractional-decay trajectories
- [`numra-fft`](https://docs.rs/numra-fft) for spectral analysis of long-memory dynamics
- [`numra-special`](https://docs.rs/numra-special) for cross-checking against alternative Mittag-Leffler implementations

See [cross-formalism tests](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/composition_tests.rs) for the ODE-approximates-FDE consistency check at α = 1.

## Install

```toml
[dependencies]
numra-fde = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-fde>
- **Book**: [Fractional DEs](https://book.numra-rs.org/ch04-beyond-odes/fractional-des/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-fde>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
