# numra-nonlinear

**Nonlinear solvers for the [Numra](https://numra-rs.org/) workspace — Newton-Raphson with Wolfe line-search globalization, analytical or finite-difference Jacobians.**

[![Crates.io](https://img.shields.io/crates/v/numra-nonlinear.svg)](https://crates.io/crates/numra-nonlinear)
[![docs.rs](https://docs.rs/numra-nonlinear/badge.svg)](https://docs.rs/numra-nonlinear)

Iterative methods for systems of nonlinear equations `F(x) = 0`. The same machinery underlies every implicit ODE stage and every gradient-based optimizer's inner solve in the workspace.

## Example

```rust
use numra_nonlinear::{Newton, NonlinearSystem};

// Solve x² = 2 by finding a root of F(x) = x² − 2.
struct SquareRoot;

impl NonlinearSystem<f64> for SquareRoot {
    fn dim(&self) -> usize { 1 }
    fn eval
        (&self, x: &[f64], f: &mut [f64]) {
        f[0] = x[0] * x[0] - 2.0;
    }
}

let solver = Newton::default_solver();
let r = solver.solve(&SquareRoot, &[1.5]).unwrap();
assert!((r.x[0] - std::f64::consts::SQRT_2).abs() < 1e-10);
```

## What's in this crate

- **`Newton`** — Newton-Raphson with optional Wolfe line search
- **`newton_solve` / `NewtonOptions` / `NewtonResult`** — function form with full control over tolerances, max-iterations, and FD Jacobian step
- **`NonlinearSystem` trait** — user-defined `F(x)` and optional analytical Jacobian
- **`wolfe_line_search` / `WolfeOptions`** — standalone strong-Wolfe line search for use outside the Newton solver

## Composes with

- [`numra-linalg`](https://docs.rs/numra-linalg) for the linear-system solve at each Newton step
- [`numra-ode`](https://docs.rs/numra-ode) for the implicit-stage and BDF nonlinear correction
- [`numra-optim`](https://docs.rs/numra-optim) for SQP and augmented-Lagrangian inner solves

## Install

```toml
[dependencies]
numra-nonlinear = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-nonlinear>
- **Book**: [Implicit methods](https://book.numra-rs.org/ch02-solving-odes/implicit-methods/) — where the Newton machinery is documented in its primary use case
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-nonlinear>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
