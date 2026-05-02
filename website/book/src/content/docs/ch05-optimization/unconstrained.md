---
title: "Unconstrained Optimization"
---

This chapter covers algorithms for solving

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

without any constraints on the decision variables. Numra provides four unconstrained minimizers spanning the spectrum from gradient-based quasi-Newton methods to derivative-free direct search.

## BFGS Quasi-Newton Method

The **Broyden--Fletcher--Goldfarb--Shanno (BFGS)** algorithm maintains a dense approximation $ H_k \approx (\nabla^2 f)^{-1} $ to the inverse Hessian and updates it each iteration with a rank-2 correction:

$$
H_{k+1} = \Bigl(I - \rho_k s_k y_k^{\!\top}\Bigr) H_k \Bigl(I - \rho_k y_k s_k^{\!\top}\Bigr) + \rho_k s_k s_k^{\!\top},
\quad \rho_k = \frac{1}{y_k^{\!\top} s_k},
$$

where $ s_k = x_{k+1} - x_k $ and $ y_k = \nabla f_{k+1} - \nabla f_k $. A Wolfe line search determines the step length $ \alpha_k $ at each iteration.

### Functional API

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{bfgs_minimize, OptimOptions};

// Rosenbrock function: f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2
let f = |x: &[f64]| {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
};
let grad = |x: &[f64], g: &mut [f64]| {
    g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
    g[1] = 200.0 * (x[1] - x[0] * x[0]);
};

let opts = OptimOptions::default().gtol(1e-8).max_iter(2000);
let result = bfgs_minimize(f, grad, &[-1.0, 1.0], &opts).unwrap();

assert!(result.converged);
assert!((result.x[0] - 1.0).abs() < 1e-5);
assert!((result.x[1] - 1.0).abs() < 1e-5);
println!("Iterations: {}, f_evals: {}", result.iterations, result.n_feval);
```

### Struct API

The `Bfgs` struct wraps the same algorithm with an object-oriented interface:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{Bfgs, OptimOptions};

let bfgs = Bfgs::new(OptimOptions::default().gtol(1e-10));
let result = bfgs.minimize(
    |x: &[f64]| x[0] * x[0] + x[1] * x[1],
    |x: &[f64], g: &mut [f64]| { g[0] = 2.0 * x[0]; g[1] = 2.0 * x[1]; },
    &[3.0, 4.0],
).unwrap();

assert!(result.converged);
```

### Convergence Properties

BFGS converges **superlinearly** on smooth strongly convex functions. On the $ n $-dimensional Rosenbrock it typically converges in $ O(50\text{--}200) $ iterations regardless of $ n $, because the inverse Hessian approximation captures the curvature of the narrow valley.

**Storage cost:** $ O(n^2) $ for the dense inverse Hessian. For $ n > 1000 $, use L-BFGS instead.

## L-BFGS (Limited-Memory BFGS)

L-BFGS replaces the dense $ n \times n $ inverse Hessian with a compact representation using the most recent $ m $ correction pairs $ \{(s_i, y_i)\} $. The search direction is computed via the **two-loop recursion** in $ O(mn) $ time and $ O(mn) $ storage:

$$
d_k = -H_k^0 \nabla f_k + \sum_{i} \beta_i s_i,
$$

where $ H_k^0 = \gamma_k I $ with $ \gamma_k = s_{k-1}^\top y_{k-1} / y_{k-1}^\top y_{k-1} $ providing automatic scaling.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{lbfgs_minimize, LbfgsOptions};

// Large-scale quadratic: f(x) = sum(x_i^2), n = 100
let n = 100;
let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
let grad = |x: &[f64], g: &mut [f64]| {
    for i in 0..x.len() { g[i] = 2.0 * x[i]; }
};

let x0: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
let opts = LbfgsOptions::default().memory(10).gtol(1e-10);
let result = lbfgs_minimize(f, grad, &x0, &opts).unwrap();

assert!(result.converged);
assert!(result.f < 1e-10);
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `memory` | 10 | Number of correction pairs to store |
| `gtol` | 1e-8 | Gradient norm convergence tolerance |
| `max_iter` | 1000 | Maximum number of iterations |
| `ftol` | 1e-12 | Relative function change tolerance |

**When to use L-BFGS over BFGS:** Use L-BFGS when $ n \gtrsim 1000 $. For small problems ($ n < 50 $), BFGS can converge in fewer iterations because the full inverse Hessian captures curvature more accurately.

## Nelder-Mead Simplex Method

The **Nelder-Mead method** is a derivative-free optimizer that maintains a simplex of $ n+1 $ vertices and transforms it via four operations: **reflection**, **expansion**, **contraction**, and **shrink**.

Numra's implementation uses **adaptive parameters** (Gao & Han, 2012) that scale with problem dimension:

$$
\alpha = 1, \quad \gamma = 1 + \frac{2}{n}, \quad \rho = 0.75 - \frac{1}{2n}, \quad \sigma = 1 - \frac{1}{n}.
$$

These adaptive parameters significantly improve convergence in high dimensions compared to the classical fixed parameters $ (1, 2, 0.5, 0.5) $.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{nelder_mead, NelderMeadOptions};

// Beale function (non-convex, gradient not readily available)
let f = |x: &[f64]| {
    (1.5 - x[0] + x[0] * x[1]).powi(2)
    + (2.25 - x[0] + x[0] * x[1] * x[1]).powi(2)
    + (2.625 - x[0] + x[0] * x[1].powi(3)).powi(2)
};

let opts = NelderMeadOptions {
    max_iter: 50_000,
    fatol: 1e-8,
    xatol: 1e-8,
    adaptive: true,
    ..Default::default()
};
let result = nelder_mead(f, &[1.0, 1.0], &opts).unwrap();

assert!(result.converged);
assert!(result.f < 1e-4);
println!("Function evaluations: {}", result.n_feval);
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `fatol` | 1e-8 | Function value tolerance (convergence when spread < fatol) |
| `xatol` | 1e-8 | Simplex size tolerance |
| `initial_simplex_scale` | 0.05 | Scale factor for initial simplex construction |
| `adaptive` | true | Use adaptive Gao--Han parameters |
| `max_iter` | 10000 | Maximum iterations |

**Convergence:** Nelder-Mead converges **linearly** at best and can stall on degenerate problems. It uses **zero gradient evaluations** (n_geval = 0), making it ideal for black-box functions, noisy objectives, or non-differentiable problems.

## Powell's Conjugate Direction Method

**Powell's method** builds a set of conjugate directions by performing sequential **1-D line minimizations** along each direction, then replaces the direction with the largest decrease with the net displacement direction. Each line minimization uses **Brent's method** (combining parabolic interpolation with golden-section search).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{powell, PowellOptions};

let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
let opts = PowellOptions::default();
let result = powell(f, &[0.0, 0.0], &opts).unwrap();

assert!(result.converged);
assert!((result.x[0] - 2.0).abs() < 1e-4);
assert!((result.x[1] - 3.0).abs() < 1e-4);
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `ftol` | 1e-8 | Function change convergence tolerance |
| `xtol` | 1e-8 | Step size convergence tolerance |
| `max_line_search_iter` | 100 | Maximum iterations per Brent line search |
| `max_iter` | 10000 | Maximum outer iterations |

## Solver Comparison

| Solver | Gradients | Storage | Convergence | Best For |
|--------|-----------|---------|-------------|----------|
| **BFGS** | Required | $ O(n^2) $ | Superlinear | Smooth, small-to-medium scale ($ n < 1000 $) |
| **L-BFGS** | Required | $ O(mn) $ | Superlinear | Smooth, large scale ($ n \gg 1000 $) |
| **Nelder-Mead** | None | $ O(n^2) $ | Linear | Black-box, noisy, non-smooth |
| **Powell** | None | $ O(n^2) $ | Superlinear (quadratic) | Derivative-free, smooth |

## Common Options: `OptimOptions`

All gradient-based solvers share the `OptimOptions` struct:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimOptions;

let opts = OptimOptions::default()
    .max_iter(2000)
    .gtol(1e-10)     // gradient norm threshold
    .ftol(1e-14)     // relative function change threshold
    .xtol(1e-14);    // step size threshold
```

Default values: `max_iter = 1000`, `gtol = 1e-8`, `ftol = 1e-12`, `xtol = 1e-12`.

## Reading the `OptimResult`

Every solver returns an `OptimResult<S>`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let result = bfgs_minimize(f, grad, &x0, &opts).unwrap();

println!("Minimizer:    {:?}", result.x);
println!("Minimum:      {:.6e}", result.f);
println!("Gradient:     {:?}", result.grad);
println!("Converged:    {}", result.converged);
println!("Status:       {:?}", result.status);   // e.g., GradientConverged
println!("Iterations:   {}", result.iterations);
println!("f evaluations: {}", result.n_feval);
println!("g evaluations: {}", result.n_geval);
println!("Wall time:    {:.3} s", result.wall_time_secs);

// Convergence history (objective, gradient norm per iteration)
for rec in &result.history {
    println!(
        "  iter {}: f={:.4e}, ||g||={:.2e}, alpha={:.2e}",
        rec.iteration, rec.objective, rec.gradient_norm, rec.step_size
    );
}
```

### Convergence Status Variants

| `OptimStatus` | Meaning |
|---------------|---------|
| `GradientConverged` | $ \lVert \nabla f\rVert < $ gtol |
| `FunctionConverged` | $ \vert f_{k+1} - f_k\vert < $ ftol $ \cdot (1 + \vert f_k\vert) $ |
| `StepConverged` | $ \lVert s_k\rVert < $ xtol $ \cdot (1 + \lVert x_k\rVert) $ |
| `MaxIterations` | Iteration limit reached without convergence |
| `LineSearchFailed` | Wolfe line search could not find an acceptable step |
