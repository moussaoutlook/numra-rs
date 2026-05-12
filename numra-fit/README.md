# numra-fit

**Curve fitting for the [Numra](https://numra-rs.org/) workspace — nonlinear least squares (Levenberg-Marquardt), weighted fits, polynomial fit and evaluation.**

[![Crates.io](https://img.shields.io/crates/v/numra-fit.svg)](https://crates.io/crates/numra-fit)
[![docs.rs](https://docs.rs/numra-fit/badge.svg)](https://docs.rs/numra-fit)

Fit arbitrary `y = f(x, params)` models against data using `numra-optim`'s Levenberg-Marquardt under the hood, with optional analytical Jacobians (paired naturally with `numra-autodiff`) and weighted residuals. Polynomial fits use `numra-linalg`'s SVD for stability on ill-conditioned Vandermonde matrices.

## Example

```rust
use numra_fit::curve_fit;

// Fit y = a · exp(-b · x) to noisy decay data
let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![5.0, 3.03, 1.84, 1.11, 0.67, 0.41];

let r = curve_fit(
    |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
    &x, &y,
    &[4.0, 0.3],
    None,
).unwrap();

assert!((r.params[0] - 5.0).abs() < 0.1);
assert!((r.params[1] - 0.5).abs() < 0.1);
```

## What's in this crate

- **`curve_fit`** — nonlinear least squares for `y = f(x, params)`
- **`curve_fit_weighted`** — weighted residuals for heteroscedastic data
- **`curve_fit_with_jacobian`** — supply an analytical Jacobian for faster, more accurate fits
- **`polyfit`** — degree-`n` polynomial regression
- **`polyval`** — evaluate a polynomial at given points
- **`FitOptions` / `FitResult`** — configuration and post-fit diagnostics (parameters, R², residuals)
- **`FitError`** — fit-failure reporting

## Composes with

- [`numra-optim`](https://docs.rs/numra-optim) — Levenberg-Marquardt backend
- [`numra-autodiff`](https://docs.rs/numra-autodiff) — analytical Jacobians for residual functions
- [`numra-linalg`](https://docs.rs/numra-linalg) — SVD-based polynomial regression
- [`numra-stats`](https://docs.rs/numra-stats) — post-fit residual statistics and goodness-of-fit checks
- [`numra-ocp`](https://docs.rs/numra-ocp) — alternative path for fitting ODE-model parameters when the residual involves an ODE integration

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified autodiff → optim → curve-fit workflow.

## Install

```toml
[dependencies]
numra-fit = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-fit>
- **Book**: [Curve fitting](https://book.numra-rs.org/ch07-calculus-and-analysis/curve-fitting/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-fit>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
