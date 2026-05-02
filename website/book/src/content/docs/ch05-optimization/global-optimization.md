---
title: "Global Optimization"
---

Local optimizers (BFGS, L-BFGS, Nelder-Mead) converge to the nearest local minimum from their starting point. **Global optimizers** search the entire feasible region for the global minimum, making them essential for multimodal, non-convex problems.

Numra provides two population-based global optimizers: **Differential Evolution (DE)** and **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy).

## Differential Evolution (DE/rand/1/bin)

Differential Evolution maintains a population of $ N_p $ candidate solutions and evolves them through **mutation**, **crossover**, and **selection**. The specific strategy implemented is **DE/rand/1/bin**:

1. **Mutation:** For each individual $ x_i $, pick three distinct random individuals $ x_a, x_b, x_c $ and form a donor vector:
$$
v_i = x_a + F \cdot (x_b - x_c),
$$
where $ F \in [0, 2] $ is the **differential weight**.

2. **Binomial crossover:** Create a trial vector $ u_i $ where each component is taken from $ v_i $ with probability $ CR $ (crossover rate), otherwise from $ x_i $. At least one component is always from $ v_i $.

3. **Selection:** Replace $ x_i $ with $ u_i $ if $ f(u_i) \le f(x_i) $.

### Basic Usage

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{de_minimize, DEOptions};

// Rastrigin function: many local minima, global minimum at origin
let f = |x: &[f64]| {
    let n = x.len() as f64;
    10.0 * n + x.iter()
        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
};

let bounds = vec![(-5.12, 5.12); 2];
let opts = DEOptions {
    max_generations: 1000,
    ..Default::default()
};

let result = de_minimize(f, &bounds, &opts).unwrap();

assert!(result.f < 1.0);  // near global minimum f=0
println!("Best: x={:?}, f={:.6}", result.x, result.f);
println!("Generations: {}, f_evals: {}", result.iterations, result.n_feval);
```

### Rosenbrock (Non-Convex Valley)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{de_minimize, DEOptions};

let f = |x: &[f64]| {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
};

let bounds = vec![(-5.0, 10.0), (-5.0, 10.0)];
let opts = DEOptions {
    max_generations: 2000,
    ..Default::default()
};

let result = de_minimize(f, &bounds, &opts).unwrap();

assert!(result.f < 0.01);
assert!((result.x[0] - 1.0).abs() < 0.1);
assert!((result.x[1] - 1.0).abs() < 0.1);
```

### DE Options

| Option | Default | Description |
|--------|---------|-------------|
| `pop_size` | None (= 10n) | Population size |
| `max_generations` | 1000 | Maximum number of generations |
| `differential_weight` | 0.8 | Mutation scale factor $ F \in [0, 2] $ |
| `crossover_rate` | 0.9 | Crossover probability $ CR \in [0, 1] $ |
| `tol` | 1e-12 | Convergence tolerance on fitness spread |
| `seed` | 42 | Random seed for reproducibility |

**Tuning guidelines:**
- Increase `pop_size` for highly multimodal landscapes.
- Use $ F \in [0.5, 1.0] $ and $ CR \in [0.7, 1.0] $ as a starting point.
- Lower $ CR $ (e.g., 0.1--0.3) for separable functions.
- The algorithm is **deterministic** given the same seed.

## CMA-ES (Covariance Matrix Adaptation)

**CMA-ES** adapts a multivariate normal distribution $ \mathcal{N}(m_k, \sigma_k^2 C_k) $ to the landscape. At each generation it:

1. Samples $ \lambda $ offspring from the current distribution.
2. Ranks them by fitness.
3. Recombines the best $ \mu $ to update the mean $ m_{k+1} $.
4. Updates the covariance matrix $ C_{k+1} $ using **evolution paths** (cumulation) for rank-one and rank-$ \mu $ updates.
5. Updates the global step size $ \sigma_{k+1} $ via **cumulative step-size adaptation (CSA)**.

CMA-ES excels on ill-conditioned problems because the covariance adaptation learns the local curvature.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{cmaes_minimize, CmaEsOptions};

// Ill-conditioned ellipsoid: f(x) = sum(10^{6*i/(n-1)} * x_i^2)
let n = 5;
let f = |x: &[f64]| {
    x.iter().enumerate()
        .map(|(i, xi)| 10.0_f64.powf(6.0 * i as f64 / (n as f64 - 1.0)) * xi * xi)
        .sum::<f64>()
};

let bounds = vec![(-5.0, 5.0); n];
let opts = CmaEsOptions {
    max_generations: 5000,
    ..Default::default()
};

let result = cmaes_minimize(f, &[3.0; n], &bounds, &opts).unwrap();

assert!(result.f < 1e-6);
println!("Solution: {:?}", result.x);
```

### CMA-ES Options

| Option | Default | Description |
|--------|---------|-------------|
| `pop_size` | None (= 4 + floor(3*ln(n))) | Population size $ \lambda $ |
| `max_generations` | 10000 | Maximum generations |
| `sigma_init` | 0.3 | Initial step size (fraction of domain) |
| `tol` | 1e-12 | Convergence tolerance |
| `seed` | 42 | Random seed |

## DE vs. CMA-ES Comparison

| Property | DE | CMA-ES |
|----------|-----|--------|
| Population size | Large (10n) | Small (4 + 3 ln n) |
| Per-generation cost | $ O(N_p \cdot n) $ | $ O(\lambda \cdot n + n^2) $ for covariance |
| Strengths | Robust, simple tuning | Ill-conditioned, correlated |
| Weaknesses | Slow on correlated problems | $ O(n^2) $ memory for covariance |
| Bounds handling | Native (clipping) | Native (clipping) |
| Gradient-free | Yes | Yes |
| Best for | General multimodal, $ n \le 100 $ | Moderate dimension, ill-conditioned |

## Using Global Optimization via the Builder

The `OptimProblem` builder supports global optimization with `.global(true)`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

let result = OptimProblem::new(2)
    .objective(|x: &[f64]| {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    })
    .all_bounds(&[(-5.0, 5.0), (-5.0, 5.0)])
    .global(true)
    .solve()
    .unwrap();

println!("x = {:?}, f = {:.6}", result.x, result.f);
```

When `.global(true)` is set, the auto solver dispatches to Differential Evolution. Custom DE options can be provided with `.de_options(opts)`.

## Deterministic Reproducibility

Both DE and CMA-ES are **deterministic** given the same seed:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{de_minimize, DEOptions};

let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
let bounds = vec![(-5.0, 5.0); 2];

let r1 = de_minimize(f, &bounds, &DEOptions::default()).unwrap();
let r2 = de_minimize(f, &bounds, &DEOptions::default()).unwrap();

assert_eq!(r1.x, r2.x);
assert_eq!(r1.f, r2.f);
```

This is important for debugging and scientific reproducibility. Change the seed to explore different search trajectories.

## Practical Recommendations

1. **Start with DE** for general multimodal problems. It requires only bounds (no initial point) and has few tuning parameters.

2. **Switch to CMA-ES** if the problem is poorly conditioned (condition number > 100) or the variables are correlated.

3. **Refine with a local solver:** After global search, polish the result with BFGS or L-BFGS for maximum accuracy:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{de_minimize, bfgs_minimize, DEOptions, OptimOptions};

// Phase 1: Global search
let global = de_minimize(f, &bounds, &DEOptions::default()).unwrap();

// Phase 2: Local refinement
let local = bfgs_minimize(f, grad, &global.x, &OptimOptions::default()).unwrap();
```

This two-phase approach combines the global exploration of DE with the superlinear convergence of BFGS.
