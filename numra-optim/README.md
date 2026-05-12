# numra-optim

**Optimization for the [Numra](https://numra-rs.org/) workspace — unconstrained, bound-constrained, general constrained, global, multi-objective, and LP / MILP solvers behind a single `OptimProblem` builder with automatic algorithm selection.**

[![Crates.io](https://img.shields.io/crates/v/numra-optim.svg)](https://crates.io/crates/numra-optim)
[![docs.rs](https://docs.rs/numra-optim/badge.svg)](https://docs.rs/numra-optim)

Ten-plus algorithms unified under one declarative API. Hand a problem to `OptimProblem`, optionally with a gradient, and the auto-selector picks BFGS / L-BFGS / L-BFGS-B / SQP / augmented Lagrangian / Nelder-Mead / CMA-ES / NSGA-II based on the constraint structure. For known patterns (nonlinear least squares, linear programs, mixed-integer linear programs), call the specialized solvers directly.

## Example

```rust
use numra_optim::OptimProblem;

// Minimize Rosenbrock: f(x, y) = (1-x)² + 100(y-x²)²
let r = OptimProblem::new(2)
    .x0(&[0.0, 0.0])
    .objective(|x: &[f64]| {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    })
    .solve()
    .unwrap();

assert!((r.x[0] - 1.0).abs() < 0.01);
assert!((r.x[1] - 1.0).abs() < 0.01);
```

## What's in this crate

| Family | Solvers |
|---|---|
| Unconstrained (quasi-Newton) | `Bfgs`, `Lbfgs`, `bfgs_minimize`, `lbfgs_minimize` |
| Bound-constrained            | `Lbfgsb`, `lbfgsb_minimize` |
| Nonlinear least squares      | `lm_minimize` (Levenberg-Marquardt) |
| Constrained (general)        | `sqp_minimize`, `augmented_lagrangian_minimize` |
| Derivative-free              | `nelder_mead`, `powell` |
| Global                       | `de_minimize` (Differential Evolution), `cmaes_minimize` |
| Multi-objective              | `nsga2_optimize` |
| Quadratic / Linear           | `active_set_qp_solve`, `simplex_solve`, `milp_solve` |
| Declarative builder          | `OptimProblem` (auto solver choice via `ProblemHint`) |

**Extras:** `RobustProblem` / `StochasticProblem` for uncertainty-aware optimization; `compute_param_sensitivity` for post-solve parameter-sensitivity analysis.

## Composes with

- [`numra-autodiff`](https://docs.rs/numra-autodiff) — supplies analytical gradients / Jacobians
- [`numra-nonlinear`](https://docs.rs/numra-nonlinear) — Newton inner solves inside SQP and augmented Lagrangian
- [`numra-linalg`](https://docs.rs/numra-linalg) — Hessian factorizations and QP subproblems
- [`numra-fit`](https://docs.rs/numra-fit) — Levenberg-Marquardt as the curve-fitting backend
- [`numra-ocp`](https://docs.rs/numra-ocp) — outer optimizer for parameter estimation, shooting, and collocation

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified autodiff → optim → fit workflow.

## Install

```toml
[dependencies]
numra-optim = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-optim>
- **Book**: [Unconstrained](https://book.numra-rs.org/ch05-optimization/unconstrained/) · [Constrained](https://book.numra-rs.org/ch05-optimization/constrained/) · [Global](https://book.numra-rs.org/ch05-optimization/global-optimization/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-optim>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).
