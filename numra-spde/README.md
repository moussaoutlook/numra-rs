# numra-spde

**Stochastic partial differential equation solvers for the [Numra](https://numra-rs.org/) workspace — Method of Lines with SDE time stepping, white and colored noise sources.**

[![Crates.io](https://img.shields.io/crates/v/numra-spde.svg)](https://crates.io/crates/numra-spde)
[![docs.rs](https://docs.rs/numra-spde/badge.svg)](https://docs.rs/numra-spde)

Solves SPDEs `∂u/∂t = L[u] + σ(u) ξ(x,t)` by discretizing the spatial operator `L` with finite differences (sharing `numra-pde`'s discretization machinery) and time-stepping the resulting SDE system with `numra-sde`'s steppers. Supports additive and multiplicative noise, ensemble runs, and adaptive time control.

## Example

```rust
use numra_pde::Grid1D;
use numra_spde::{SpdeSystem};

// Stochastic heat equation: ∂T/∂t = α ∂²T/∂x² + σ ξ(x,t)
struct StochasticHeat { alpha: f64, sigma: f64 }

impl SpdeSystem<f64> for StochasticHeat {
    fn dim(&self) -> usize { 1 }
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let n = u.len();
        for i in 0..n {
            let l = if i == 0 { 0.0 } else { u[i - 1] };
            let r = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (l - 2.0 * u[i] + r) / (dx * dx);
        }
    }
    fn diffusion(&self, _t: f64, u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for s in sigma.iter_mut() { *s = self.sigma; }
    }
}
```

## What's in this crate

- **`SpdeSystem` trait** — drift, diffusion, optional noise-correlation hooks
- **`MolSdeSolver`** — Method-of-Lines stepper composing `numra-pde` discretizations with `numra-sde` time integrators
- **`SpdeEnsemble`** — Rayon-parallel Monte Carlo over realizations
- **`SpdeOptions` / `SpdeResult` / `SpdeStats`** — configuration and diagnostics
- **`SpdeMethod`** — selectable underlying SDE stepper
- **Noise sources**: `WhiteNoise`, `ColoredNoise`, `ColoredNoiseGenerator`, `SpaceTimeNoise`, `NoiseCorrelation`

## Composes with

- [`numra-pde`](https://docs.rs/numra-pde) — provides spatial grids and finite-difference operators
- [`numra-sde`](https://docs.rs/numra-sde) — provides the time-stepping schemes underneath MOL
- [`numra-stats`](https://docs.rs/numra-stats) — ensemble mean / std / percentile across realizations
- [`numra-interp`](https://docs.rs/numra-interp) — post-hoc resampling of fields onto common grids

See [cross-formalism tests](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/composition_tests.rs) for the SPDE = PDE + SDE composition check.

## Install

```toml
[dependencies]
numra-spde = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-spde>
- **Book**: [Stochastic PDEs](https://book.numra-rs.org/ch04-beyond-odes/stochastic-pdes/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-spde>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
