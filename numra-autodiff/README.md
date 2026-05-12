# numra-autodiff

**Automatic differentiation for the [Numra](https://numra-rs.org/) workspace — forward-mode dual numbers and reverse-mode tape, with a bridge module that plugs AD-derived gradients and Jacobians into `numra-optim` and `numra-fit`.**

[![Crates.io](https://img.shields.io/crates/v/numra-autodiff.svg)](https://crates.io/crates/numra-autodiff)
[![docs.rs](https://docs.rs/numra-autodiff/badge.svg)](https://docs.rs/numra-autodiff)

Forward-mode uses `Dual<S>` numbers and costs `O(n)` passes for `n` inputs (best when outputs ≫ inputs — Jacobians of small systems, ODE sensitivity). Reverse-mode uses a tape and costs `O(m)` passes for `m` outputs (best when inputs ≫ outputs — gradients of scalar objectives, the standard optimization case).

## Example

```rust
use numra_autodiff::{gradient, Dual};

// f(x, y) = x² + y², gradient at (3, 4) should be (6, 8)
let grad = gradient(
    |x: &[Dual<f64>]| x[0] * x[0] + x[1] * x[1],
    &[3.0, 4.0],
);
assert!((grad[0] - 6.0).abs() < 1e-12);
assert!((grad[1] - 8.0).abs() < 1e-12);
```

For reverse-mode:

```rust
use numra_autodiff::reverse::grad;

let g = grad(|x| x[0].clone() * x[0].clone() + x[1].clone() * x[1].clone(), &[3.0, 4.0]);
assert!((g[0] - 6.0).abs() < 1e-12);
assert!((g[1] - 8.0).abs() < 1e-12);
```

## What's in this crate

- **`Dual<S>`** — forward-mode dual numbers
- **`gradient`, `jacobian`** — forward-mode gradient and Jacobian of closures
- **`reverse::Var`, `reverse::grad`** — reverse-mode tape-based variables and gradient
- **`tape`** — low-level tape for custom reverse-mode pipelines
- **`bridge::gradient_closure`, `bridge::model_jacobian_closure`** — adapters that expose AD-derived gradients / Jacobians in the calling-convention used by `numra-optim`'s `OptimProblem::gradient` and `numra-fit`'s `curve_fit_with_jacobian`

## Composes with

- [`numra-optim`](https://docs.rs/numra-optim) — supplies gradients for BFGS, L-BFGS, Levenberg-Marquardt, SQP via `bridge::gradient_closure`
- [`numra-fit`](https://docs.rs/numra-fit) — analytical Jacobians for nonlinear least-squares residuals via `bridge::model_jacobian_closure`
- [`numra-ocp`](https://docs.rs/numra-ocp) — gradients for parameter-estimation and optimal-control objectives

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified autodiff → optimization → curve-fitting workflow.

## Install

```toml
[dependencies]
numra-autodiff = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-autodiff>
- **Book**: [Automatic differentiation](https://book.numra-rs.org/ch07-calculus-and-analysis/automatic-differentiation/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-autodiff>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
