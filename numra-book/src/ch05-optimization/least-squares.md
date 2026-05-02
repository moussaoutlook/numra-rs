# Nonlinear Least Squares

Nonlinear least-squares problems arise in curve fitting, parameter estimation, and data assimilation. The goal is to minimize a sum of squared residuals:

\\[
\min_{x \in \mathbb{R}^n} \; \|r(x)\|^2 = \sum_{i=1}^{m} r_i(x)^2,
\\]

where \\( r: \mathbb{R}^n \to \mathbb{R}^m \\) is a vector-valued residual function with \\( m \ge n \\).

## Levenberg-Marquardt Algorithm

The **Levenberg-Marquardt (LM)** algorithm interpolates between Gauss-Newton and gradient descent using a damping parameter \\( \lambda \\). At each iteration it solves:

\\[
\bigl(J^\top J + \lambda \, \text{diag}(J^\top J)\bigr) \, \Delta x = -J^\top r,
\\]

where \\( J \in \mathbb{R}^{m \times n} \\) is the Jacobian of the residual. The adaptive damping strategy uses the **gain ratio** \\( \rho = (\|r(x_k)\|^2 - \|r(x_k + \Delta x)\|^2) / \text{predicted} \\):

- If \\( \rho > 0 \\) and the step reduces the cost: **accept**, decrease \\( \lambda \leftarrow \lambda \cdot \lambda_\text{down} \\).
- Otherwise: **reject**, increase \\( \lambda \leftarrow \lambda \cdot \lambda_\text{up} \\).

This adaptive scaling by \\( \text{diag}(J^\top J) \\) rather than the identity matrix ensures scale invariance.

## Curve Fitting: Exponential Decay

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{lm_minimize, LmOptions};

// Fit y = a * exp(b * t) to data
let t_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
let y_data: Vec<f64> = t_data.iter()
    .map(|&t| 2.0 * (-0.5 * t).exp())
    .collect();
let m = t_data.len();

let residual = |p: &[f64], r: &mut [f64]| {
    for i in 0..m {
        r[i] = p[0] * (p[1] * t_data[i]).exp() - y_data[i];
    }
};

let jacobian = |p: &[f64], jac: &mut [f64]| {
    for i in 0..m {
        let e = (p[1] * t_data[i]).exp();
        jac[i * 2]     = e;                         // dr_i/da
        jac[i * 2 + 1] = p[0] * t_data[i] * e;     // dr_i/db
    }
};

let result = lm_minimize(
    residual, jacobian, &[1.0, -1.0], m, &LmOptions::default()
).unwrap();

assert!(result.converged);
assert!((result.x[0] - 2.0).abs() < 1e-4);   // a ~ 2.0
assert!((result.x[1] + 0.5).abs() < 1e-4);    // b ~ -0.5
println!("Final cost: {:.2e}", result.f);       // ||r||^2 ~ 0
```

### Residual and Jacobian Signatures

The residual function writes into a buffer of length \\( m \\):
<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// residual: (params, output) -> ()
let residual = |p: &[f64], r: &mut [f64]| {
    r[0] = ...; // first residual
    r[1] = ...; // second residual
};
```

The Jacobian writes into a **row-major** buffer of length \\( m \times n \\):
<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// jacobian: (params, output) -> ()
// jac[i * n + j] = dr_i / dp_j
let jacobian = |p: &[f64], jac: &mut [f64]| {
    jac[0 * 2 + 0] = ...; // dr_0/dp_0
    jac[0 * 2 + 1] = ...; // dr_0/dp_1
    jac[1 * 2 + 0] = ...; // dr_1/dp_0
    // ...
};
```

## Circle Fitting

A classic geometric fitting problem: given points on or near a circle, find the center \\( (c_x, c_y) \\) and radius \\( r \\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{lm_minimize, LmOptions};
use std::f64::consts::PI;

let n_pts = 20;
let angles: Vec<f64> = (0..n_pts)
    .map(|i| 2.0 * PI * i as f64 / n_pts as f64)
    .collect();
let x_data: Vec<f64> = angles.iter().map(|a| a.cos()).collect();
let y_data: Vec<f64> = angles.iter().map(|a| a.sin()).collect();

let residual = |p: &[f64], r: &mut [f64]| {
    let (cx, cy, rad) = (p[0], p[1], p[2]);
    for i in 0..n_pts {
        let dx = x_data[i] - cx;
        let dy = y_data[i] - cy;
        r[i] = (dx * dx + dy * dy).sqrt() - rad;
    }
};

let jacobian = |p: &[f64], jac: &mut [f64]| {
    let (cx, cy) = (p[0], p[1]);
    for i in 0..n_pts {
        let dx = x_data[i] - cx;
        let dy = y_data[i] - cy;
        let dist = (dx * dx + dy * dy).sqrt();
        jac[i * 3]     = -dx / dist;   // dr_i/dcx
        jac[i * 3 + 1] = -dy / dist;   // dr_i/dcy
        jac[i * 3 + 2] = -1.0;         // dr_i/dr
    }
};

let result = lm_minimize(
    residual, jacobian, &[0.5, 0.5, 0.5], n_pts, &LmOptions::default()
).unwrap();

assert!(result.converged);
assert!(result.x[0].abs() < 1e-6);         // cx ~ 0
assert!(result.x[1].abs() < 1e-6);         // cy ~ 0
assert!((result.x[2] - 1.0).abs() < 1e-6); // r ~ 1
```

## Declarative Builder API

The `OptimProblem` builder supports least-squares via `.least_squares()` and `.jacobian()`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::OptimProblem;

let t_data = vec![0.0, 1.0, 2.0];
let y_data = vec![1.0, 3.0, 5.0];

let result = OptimProblem::new(2)
    .x0(&[0.0, 0.0])
    .least_squares(3, move |p: &[f64], r: &mut [f64]| {
        for i in 0..3 {
            r[i] = p[0] * t_data[i] + p[1] - y_data[i];
        }
    })
    .solve()
    .unwrap();

assert!(result.converged);
assert!((result.x[0] - 2.0).abs() < 1e-4);  // slope
assert!((result.x[1] - 1.0).abs() < 1e-4);  // intercept
```

When no Jacobian is provided, the auto solver uses **finite-difference Jacobians** computed via central differences (\\( 2n \\) residual evaluations per Jacobian).

## LM Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iter` | 200 | Maximum iterations |
| `gtol` | 1e-10 | Gradient norm convergence tolerance \\( \lVert J^\top r\rVert \\) |
| `xtol` | 1e-10 | Step size convergence tolerance |
| `ftol` | 1e-12 | Relative cost change tolerance |
| `lambda_init` | 1e-3 | Initial damping parameter |
| `lambda_up` | 10.0 | Damping increase factor on rejected step |
| `lambda_down` | 0.1 | Damping decrease factor on accepted step |

## Gauss-Newton vs. Levenberg-Marquardt

The **Gauss-Newton** method is the special case \\( \lambda = 0 \\):

\\[
J^\top J \, \Delta x = -J^\top r.
\\]

This converges quadratically near a zero-residual solution but can diverge when \\( J^\top J \\) is nearly singular or the residuals are large. Levenberg-Marquardt adds regularization to ensure global convergence:

| Property | Gauss-Newton | LM (\\( \lambda \\) small) | LM (\\( \lambda \\) large) |
|----------|-------------|--------------------------|--------------------------|
| Direction | Newton-like | Newton-like | Steepest descent |
| Convergence | Quadratic (near-zero residual) | Superlinear | Linear |
| Robustness | Can diverge | Global convergence | Global convergence |

Numra's LM implementation automatically adjusts \\( \lambda \\) to stay in the superlinear regime when possible.

## Convergence Diagnostics

The `result.history` field tracks convergence per iteration:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
for rec in &result.history {
    println!(
        "iter {}: cost={:.4e}, ||J^T r||={:.2e}, lambda={:.2e}",
        rec.iteration, rec.objective, rec.gradient_norm, rec.step_size
    );
}
```

A healthy LM run shows the cost decreasing monotonically (on accepted steps) and the gradient norm shrinking. The `step_size` field tracks the current damping \\( \lambda \\), which should decrease as the algorithm approaches the solution.
