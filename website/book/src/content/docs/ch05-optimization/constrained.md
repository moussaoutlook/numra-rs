---
title: "Constrained Optimization"
---

This chapter covers optimization problems with constraints:

$$
\min_{x \in \mathbb{R}^n} f(x) \quad \text{subject to} \quad h_i(x) = 0, \quad g_j(x) \le 0, \quad l_k \le x_k \le u_k.
$$

Numra provides three constrained solvers: the **Augmented Lagrangian** method for general nonlinear constraints, **SQP** (Sequential Quadratic Programming) for high-accuracy constrained problems, and **L-BFGS-B** for bound-constrained optimization.

## Augmented Lagrangian Method

The Augmented Lagrangian approach converts a constrained problem into a sequence of unconstrained (or bound-constrained) subproblems. For equality constraints $ h_j(x) = 0 $ and inequality constraints $ g_i(x) \le 0 $, the augmented Lagrangian is:

$$
\mathcal{L}_A(x; \lambda, \mu, \sigma) = f(x) + \sum_j \Bigl[\lambda_j h_j(x) + \frac{\sigma}{2} h_j(x)^2 \Bigr]
+ \sum_i \frac{\sigma}{2} \max\!\Bigl(0,\; g_i(x) + \frac{\mu_i}{\sigma}\Bigr)^{\!2}
$$

At each outer iteration, the method solves an unconstrained subproblem (using L-BFGS or L-BFGS-B if bounds are present), then updates the multiplier estimates $ \lambda_j \leftarrow \lambda_j + \sigma h_j(x^*) $ and $ \mu_i \leftarrow \max(0, \mu_i + \sigma g_i(x^*)) $, and increases the penalty parameter $ \sigma \leftarrow \sigma \cdot \sigma_\text{factor} $.

### Declarative Builder API

The easiest way to solve constrained problems is through `OptimProblem`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

// Minimize x0 + x1 subject to x0^2 + x1^2 = 1
let result = OptimProblem::new(2)
    .x0(&[1.0, 0.0])
    .objective(|x: &[f64]| x[0] + x[1])
    .gradient(|_x: &[f64], g: &mut [f64]| { g[0] = 1.0; g[1] = 1.0; })
    .constraint_eq_with_grad(
        |x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0,
        |x: &[f64], g: &mut [f64]| { g[0] = 2.0 * x[0]; g[1] = 2.0 * x[1]; },
    )
    .solve()
    .unwrap();

assert!(result.converged);
let expected = -1.0 / 2.0_f64.sqrt();
assert!((result.x[0] - expected).abs() < 1e-3);
assert!((result.x[1] - expected).abs() < 1e-3);
assert!(result.constraint_violation < 1e-5);

// Lagrange multiplier estimates are available:
println!("lambda_eq = {:?}", result.lambda_eq);
```

### Inequality Constraints

Inequality constraints are specified with `constraint_ineq`. The convention is $ g(x) \le 0 $:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

// Minimize (x0-2)^2 + (x1-2)^2 subject to x0 + x1 <= 2
// Convention: g(x) = x0 + x1 - 2 <= 0
let result = OptimProblem::new(2)
    .x0(&[0.0, 0.0])
    .objective(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
    .gradient(|x: &[f64], g: &mut [f64]| {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 2.0);
    })
    .constraint_ineq_with_grad(
        |x: &[f64]| x[0] + x[1] - 2.0,
        |_x: &[f64], g: &mut [f64]| { g[0] = 1.0; g[1] = 1.0; },
    )
    .solve()
    .unwrap();

assert!((result.x[0] - 1.0).abs() < 1e-2);
assert!((result.x[1] - 1.0).abs() < 1e-2);
```

### Mixed Constraints

Equality and inequality constraints can be freely combined:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

// Minimize x0^2 + x1^2
// subject to: x0 + x1 = 1 (equality)
//             x0 >= 0.6   (inequality: 0.6 - x0 <= 0)
let result = OptimProblem::new(2)
    .x0(&[1.0, 1.0])
    .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
    .gradient(|x: &[f64], g: &mut [f64]| { g[0] = 2.0 * x[0]; g[1] = 2.0 * x[1]; })
    .constraint_eq_with_grad(
        |x: &[f64]| x[0] + x[1] - 1.0,
        |_x: &[f64], g: &mut [f64]| { g[0] = 1.0; g[1] = 1.0; },
    )
    .constraint_ineq_with_grad(
        |x: &[f64]| 0.6 - x[0],
        |_x: &[f64], g: &mut [f64]| { g[0] = -1.0; g[1] = 0.0; },
    )
    .solve()
    .unwrap();

assert!((result.x[0] - 0.6).abs() < 5e-2);
assert!((result.x[1] - 0.4).abs() < 5e-2);
```

### Tuning the Augmented Lagrangian

Custom options control the outer loop behavior:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{OptimProblem, AugLagOptions};

let opts = AugLagOptions {
    sigma_init: 10.0,         // initial penalty parameter
    sigma_factor: 10.0,       // penalty growth factor
    sigma_max: 1e12,          // maximum penalty
    ctol: 1e-8,               // constraint violation tolerance
    max_outer_iter: 50,       // maximum outer iterations
    ..AugLagOptions::default()
};

let result = OptimProblem::new(2)
    .x0(&[1.0, 0.0])
    .objective(|x: &[f64]| x[0] + x[1])
    .constraint_eq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0)
    .aug_lag_options(opts)
    .solve()
    .unwrap();

assert!(result.constraint_violation < 1e-7);
```

| Option | Default | Description |
|--------|---------|-------------|
| `sigma_init` | 1.0 | Initial penalty parameter |
| `sigma_factor` | 10.0 | Penalty growth factor per outer iteration |
| `sigma_max` | 1e12 | Maximum penalty (prevents numerical overflow) |
| `ctol` | 1e-6 | Constraint violation convergence tolerance |
| `max_outer_iter` | 50 | Maximum outer loop iterations |

## Sequential Quadratic Programming (SQP)

The **SQP** method solves a quadratic programming (QP) subproblem at each iteration:

$$
\min_d \; \tfrac{1}{2} d^\top B_k d + \nabla f_k^\top d
\quad \text{s.t.} \quad \nabla h_i(x_k)^\top d + h_i(x_k) = 0, \quad \nabla g_j(x_k)^\top d + g_j(x_k) \le 0,
$$

where $ B_k $ is a BFGS Hessian approximation updated with **Powell's damping**. The step length is chosen by backtracking on the **L1 merit function**:

$$
\phi(x; \mu) = f(x) + \mu \sum_i |h_i(x)| + \mu \sum_j \max(0, g_j(x)).
$$

SQP includes a **feasibility restoration** phase that activates when constraint violation stagnates, performing Gauss--Newton steps toward the constraint surface.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{sqp_minimize, SqpOptions, Constraint, ConstraintKind};

let constraints = vec![
    Constraint {
        func: Box::new(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0),
        grad: Some(Box::new(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0]; g[1] = 2.0 * x[1];
        })),
        kind: ConstraintKind::Equality,
    },
];

let result = sqp_minimize(
    |x: &[f64]| x[0] + x[1],
    |_x: &[f64], g: &mut [f64]| { g[0] = 1.0; g[1] = 1.0; },
    &constraints,
    &[1.0, 0.0],
    &SqpOptions::default(),
).unwrap();

assert!(result.converged);
assert!(result.constraint_violation < 1e-3);
```

### SQP vs. Augmented Lagrangian

| Property | Augmented Lagrangian | SQP |
|----------|---------------------|-----|
| Convergence rate | Linear (outer loop) | Superlinear |
| Per-iteration cost | L-BFGS subproblem | KKT system solve |
| Feasibility | Asymptotic | Maintained via restoration |
| Robustness | High (always descends) | Moderate (may need restoration) |
| Best for | General NLP, moderate accuracy | High-accuracy NLP |

## L-BFGS-B (Bound-Constrained Optimization)

For problems with only **box constraints** $ l_i \le x_i \le u_i $, the **L-BFGS-B** algorithm combines the L-BFGS two-loop recursion with projected Armijo backtracking:

$$
x_{k+1} = P\!\bigl(x_k + \alpha_k d_k, [l, u]\bigr),
$$

where $ P(\cdot, [l, u]) $ clamps each component to its bounds. Convergence is measured by the **projected gradient norm**:

$$
\|P(x - \nabla f) - x\|_\infty < \text{gtol}.
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{lbfgsb_minimize, LbfgsBOptions};

// Minimize (x0-2)^2 + (x1-2)^2 with bounds: 0 <= x0 <= 1, 0 <= x1 <= 3
let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
let grad = |x: &[f64], g: &mut [f64]| {
    g[0] = 2.0 * (x[0] - 2.0);
    g[1] = 2.0 * (x[1] - 2.0);
};
let bounds = vec![Some((0.0, 1.0)), Some((0.0, 3.0))];
let opts = LbfgsBOptions::default().gtol(1e-10);

let result = lbfgsb_minimize(f, grad, &[0.5, 0.5], &bounds, &opts).unwrap();

assert!(result.converged);
assert!((result.x[0] - 1.0).abs() < 1e-6);  // at upper bound
assert!((result.x[1] - 2.0).abs() < 1e-6);  // interior
assert!(result.active_bounds.contains(&0));   // x0 is at its bound
```

### Active Bounds Reporting

The `result.active_bounds` field contains the indices of variables that are at their bounds at the solution. This is useful for sensitivity analysis and understanding which bounds are active constraints.

## Automatic Constraint Gradients

If you omit constraint gradients, Numra falls back to **central finite differences**:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

let result = OptimProblem::new(2)
    .x0(&[1.0, 0.0])
    .objective(|x: &[f64]| x[0] + x[1])
    // No gradient for objective -- finite differences used
    .constraint_eq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0)
    // No gradient for constraint -- finite differences used
    .solve()
    .unwrap();

assert!(result.converged);
```

Finite differences cost $ 2n $ extra function evaluations per gradient and introduce $ O(\epsilon^{2/3}) $ errors with step size $ \epsilon = 10^{-8} $. Providing analytical gradients is always preferred for accuracy and speed.

## Output Fields for Constrained Problems

The `OptimResult` includes fields specific to constrained optimization:

| Field | Type | Description |
|-------|------|-------------|
| `lambda_eq` | `Vec<S>` | Lagrange multiplier estimates for equality constraints |
| `lambda_ineq` | `Vec<S>` | Lagrange multiplier estimates for inequality constraints |
| `active_bounds` | `Vec<usize>` | Indices of variables at their bounds |
| `constraint_violation` | `S` | Maximum constraint violation at the solution |
