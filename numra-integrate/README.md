# numra-integrate

**Numerical integration for the [Numra](https://numra-rs.org/) workspace — adaptive Gauss-Kronrod, fixed Gauss-Legendre / Laguerre / Hermite, composite rules, and double integrals.**

[![Crates.io](https://img.shields.io/crates/v/numra-integrate.svg)](https://crates.io/crates/numra-integrate)
[![docs.rs](https://docs.rs/numra-integrate/badge.svg)](https://docs.rs/numra-integrate)

Quadrature for one- and two-dimensional integrals over finite, semi-infinite, and infinite intervals.

## Example

```rust
use numra_integrate::{quad, QuadOptions};

// ∫₀^π sin(x) dx = 2
let r = quad(
    |x: f64| x.sin(),
    0.0,
    std::f64::consts::PI,
    &QuadOptions::default(),
).unwrap();

assert!((r.value - 2.0).abs() < 1e-8);
```

## What's in this crate

- **Adaptive**: `quad` (Gauss-Kronrod G7K15 with adaptive subdivision), `QuadOptions`, `QuadResult`
- **Fixed Gaussian**: `gauss_legendre` (finite intervals), `gauss_laguerre` (semi-infinite, weight `e^{-x}`), `gauss_hermite` (infinite, weight `e^{-x²}`)
- **Composite rules**: `trapezoid`, `trapezoid_nonuniform`, `simpson`, `romberg`, `cumulative_trapezoid`
- **Multi-dimensional**: `dblquad` (iterated 1D quadrature for double integrals)
- **`IntegrationError`** — quadrature-failure reporting

## Composes with

- [`numra-ode`](https://docs.rs/numra-ode) — quadrature of functions defined on trajectories
- [`numra-interp`](https://docs.rs/numra-interp) — quadrature of interpolants between sample points
- [`numra-stats`](https://docs.rs/numra-stats) — definite integrals of distribution PDFs and integral-form expectations
- [`numra-ide`](https://docs.rs/numra-ide) — independently uses quadrature to evaluate Volterra memory kernels

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified ODE → cubic spline → quadrature workflow.

## Install

```toml
[dependencies]
numra-integrate = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-integrate>
- **Book**: [Numerical integration](https://book.numra-rs.org/ch07-calculus-and-analysis/numerical-integration/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-integrate>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
