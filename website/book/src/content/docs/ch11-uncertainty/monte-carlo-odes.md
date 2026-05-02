---
title: "Monte Carlo with ODEs"
---

When first-order uncertainty propagation is insufficient -- because uncertainties
are large, functions are nonlinear, or you need full probability distributions --
Monte Carlo simulation provides the most flexible alternative.

## The Approach

1. Sample input parameters from their probability distributions
2. Run the ODE solver for each sample
3. Collect output statistics (mean, variance, quantiles, histograms)

This is embarrassingly parallel and makes no linearity assumptions.

## Example: Uncertain Initial Conditions

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let n_samples = 1000;
let mut final_values = Vec::with_capacity(n_samples);

// Random number generator (seed for reproducibility)
let mut rng = rand::rngs::StdRng::seed_from_u64(42);

for _ in 0..n_samples {
    // Sample initial condition: y0 ~ Normal(1.0, 0.1²)
    let y0 = 1.0 + 0.1 * rand_distr::Normal::new(0.0, 1.0).unwrap().sample(&mut rng);

    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -0.5 * y[0];
        },
        0.0, 10.0,
        vec![y0],
    );

    let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
    let result = DoPri5::solve(&problem, 0.0, 10.0, &[y0], &options).unwrap();
    final_values.push(result.y_final().unwrap()[0]);
}

// Compute statistics
let mean: f64 = final_values.iter().sum::<f64>() / n_samples as f64;
let var: f64 = final_values.iter()
    .map(|&x| (x - mean) * (x - mean))
    .sum::<f64>() / (n_samples - 1) as f64;

println!("Mean: {:.6}, Std: {:.6}", mean, var.sqrt());
```

## Example: Uncertain Parameters

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
for _ in 0..n_samples {
    // Sample decay rate: k ~ Uniform(0.4, 0.6)
    let k = 0.4 + 0.2 * rng.gen::<f64>();

    let problem = OdeProblem::new(
        move |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -k * y[0];
        },
        0.0, 10.0,
        vec![1.0],
    );

    let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0], &options).unwrap();
    final_values.push(result.y_final().unwrap()[0]);
}
```

## Monte Carlo vs Linear Propagation

| Feature | Monte Carlo | Linear (Uncertain) |
|---------|------------|-------------------|
| **Accuracy** | Exact (in limit) | First-order approx |
| **Nonlinearity** | Handles any | Only mild |
| **Distribution** | Full distribution | Mean + variance only |
| **Cost** | N ODE solves | 1 ODE solve |
| **Correlations** | Easy to include | Assumes independence |

## Convergence

Monte Carlo error decreases as $ 1/\sqrt{N} $:

| Samples | Relative error (95% CI) |
|---------|------------------------|
| 100 | ~20% |
| 1,000 | ~6% |
| 10,000 | ~2% |
| 100,000 | ~0.6% |

For tail probabilities (rare events), you need many more samples. Variance
reduction techniques (importance sampling, control variates) can help.

## Combining with SDE Solvers

For systems with intrinsic stochastic dynamics *and* uncertain parameters,
combine Monte Carlo parameter sampling with SDE integration:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::sde::{SdeSystem, EulerMaruyama, SdeSolver, SdeOptions};

for _ in 0..n_samples {
    let mu = 0.05 + 0.02 * normal.sample(&mut rng);    // uncertain drift
    let sigma = 0.2 + 0.05 * normal.sample(&mut rng);  // uncertain volatility

    let system = SdeSystem::new(
        1,
        move |_t, y: &[f64], drift: &mut [f64]| { drift[0] = mu * y[0]; },
        move |_t, y: &[f64], diff: &mut [f64]| { diff[0] = sigma * y[0]; },
    );

    let sde_options = SdeOptions::new(0.001, rng.gen());
    let result = EulerMaruyama::solve(&system, 0.0, 1.0, &[100.0], &sde_options);
    prices.push(result.y_final()[0]);
}
```

This captures both **epistemic uncertainty** (uncertain parameters) and
**aleatoric uncertainty** (inherent stochastic noise).
