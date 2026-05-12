# numra-interp

**Interpolation for the [Numra](https://numra-rs.org/) workspace — linear, cubic spline, PCHIP, Akima, and barycentric Lagrange under a common `Interpolant` trait.**

[![Crates.io](https://img.shields.io/crates/v/numra-interp.svg)](https://crates.io/crates/numra-interp)
[![docs.rs](https://docs.rs/numra-interp/badge.svg)](https://docs.rs/numra-interp)

1D interpolation methods chosen for the property they preserve: PCHIP for monotonicity, Akima for robustness to outliers, cubic spline for smoothness, barycentric Lagrange for stable polynomial interpolation over Chebyshev nodes.

## Example

```rust
use numra_interp::{interp1d, Interp1dMethod, Interpolant};

let x = vec![0.0, 1.0, 2.0, 3.0];
let y = vec![1.0, 0.5, 0.25, 0.125];

// PCHIP preserves monotonic data — the interpolant won't overshoot
let pchip = interp1d(&x, &y, Interp1dMethod::Pchip).unwrap();
let mid = pchip.interpolate(1.5);

assert!(mid < 0.5 && mid > 0.25);
```

## What's in this crate

- **`Linear`** — piecewise linear
- **`CubicSpline`** — natural, clamped, and not-a-knot boundary variants
- **`Pchip`** — piecewise cubic Hermite (monotonicity-preserving)
- **`Akima`** — piecewise cubic, tolerant of outliers
- **`BarycentricLagrange`** — stable polynomial interpolation
- **`Interpolant` trait** — `interpolate`, `interpolate_vec`, optional `derivative` and `integrate`
- **`interp1d`** — convenience constructor by `Interp1dMethod`

## Composes with

- [`numra-ode`](https://docs.rs/numra-ode) — resampling ODE trajectories onto arbitrary points
- [`numra-sde`](https://docs.rs/numra-sde) — resampling sample paths onto common grids
- [`numra-integrate`](https://docs.rs/numra-integrate) — quadrature against an interpolant
- [`numra-fit`](https://docs.rs/numra-fit) — comparing parametric fits against non-parametric interpolants

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified ODE → cubic spline → quadrature workflow.

## Install

```toml
[dependencies]
numra-interp = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-interp>
- **Book**: [Interpolation](https://book.numra-rs.org/ch07-calculus-and-analysis/interpolation/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-interp>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
