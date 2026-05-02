# Robust & Stochastic Optimization

Real-world optimization problems often involve **uncertain parameters**. Numra provides two complementary frameworks:

- **Robust optimization:** Ensures feasibility under worst-case parameter perturbations within a confidence ellipsoid.
- **Stochastic optimization:** Minimizes expected cost over a probability distribution via Sample-Average Approximation (SAA).

## Robust Optimization

### Problem Formulation

Given uncertain parameters \\( p \sim \mathcal{N}(\bar{p}, \Sigma) \\), robust optimization solves:

\\[
\min_x f(x, \bar{p}) \quad \text{subject to} \quad g_j(x, p_j^{\text{worst}}) \le 0 \;\;\forall j,
\\]

where \\( p_j^{\text{worst}} \\) is the worst-case parameter vector for constraint \\( j \\) within the \\( k\sigma \\) confidence ellipsoid. The factor \\( k = \Phi^{-1}(\alpha) \\) is determined by the confidence level \\( \alpha \\) (e.g., \\( k \approx 1.645 \\) for 95%).

For each constraint, the worst-case direction is estimated via finite differences:
\\[
p_j^{\text{worst}} = \bar{p}_j + k \cdot \sigma_j \cdot \text{sign}\!\left(\frac{\partial g}{\partial p_j}\right).
\\]

### Builder API

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::RobustProblem;

// Minimize (x - p)^2 with uncertain p = 5 +/- 1
let result = RobustProblem::<f64>::new(1)
    .x0(&[0.0])
    .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
    .gradient(|x: &[f64], p: &[f64], g: &mut [f64]| {
        g[0] = 2.0 * (x[0] - p[0]);
    })
    .param("target", 5.0, 1.0)  // name, mean, std
    .solve()
    .unwrap();

assert!((result.x[0] - 5.0).abs() < 0.1);
println!("x* = {:.3}", result.x[0]);
println!("f_nominal = {:.6}", result.f_nominal);
println!("f_worst_case = {:.6}", result.f_worst_case);
println!("x_std = {:?}", result.x_std);  // solution uncertainty
```

### Constraint Tightening Example

When a constraint depends on uncertain parameters, robust optimization tightens it to maintain feasibility:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::RobustProblem;

// Maximize x (i.e., minimize -x) subject to x <= p
// where p ~ N(10, 2^2), 95% confidence
// Nominal: x* = 10. Robust: x* ~ 10 - 1.645*2 = 6.71
let result = RobustProblem::<f64>::new(1)
    .x0(&[5.0])
    .objective(|x: &[f64], _p: &[f64]| -x[0])
    .gradient(|_x: &[f64], _p: &[f64], g: &mut [f64]| { g[0] = -1.0; })
    .constraint_ineq(|x: &[f64], p: &[f64]| x[0] - p[0])  // x - p <= 0
    .param("capacity", 10.0, 2.0)
    .confidence(0.95)
    .bounds(0, (-100.0, 100.0))
    .solve()
    .unwrap();

assert!(result.x[0] < 8.5);  // well below nominal 10
println!("Robust x* = {:.2} (nominal would be 10.0)", result.x[0]);
```

### Robust Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `x` | `Vec<S>` | Optimal decision variables |
| `f_nominal` | `S` | Objective at nominal parameters |
| `f_worst_case` | `S` | Objective at worst-case parameters |
| `x_std` | `Vec<S>` | Solution uncertainty (std dev of each \\( x_i \\)) |
| `sensitivity` | `Option<ParamSensitivity<S>>` | \\( dx^*/dp \\) sensitivity matrix |
| `converged` | `bool` | Convergence status |

### Parametric Sensitivity

The `sensitivity` field contains \\( \partial x^*_i / \partial p_j \\), computed by re-solving the problem at perturbed parameter values:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
if let Some(sens) = &result.sensitivity {
    for j in 0..sens.n_params {
        println!("Parameter '{}': dx*/dp = {:?}",
            sens.names[j], sens.column(j));
    }
}
```

## Stochastic Optimization (SAA)

### Problem Formulation

Given a parameterized objective \\( f(x, \xi) \\) with random \\( \xi \\), Sample-Average Approximation solves:

\\[
\min_x \; \frac{1}{N} \sum_{s=1}^{N} f(x, \xi_s),
\\]

where \\( \xi_1, \ldots, \xi_N \\) are i.i.d. samples from the distribution of \\( \xi \\).

### Basic Expected Value Minimization

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::StochasticProblem;

// Minimize E[(x - xi)^2] where xi ~ N(5, 1)
// Optimal: x* = E[xi] = 5
let result = StochasticProblem::new(1)
    .x0(&[0.0])
    .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
    .param_normal("xi", 5.0, 1.0)
    .n_samples(200)
    .solve()
    .unwrap();

assert!((result.x[0] - 5.0).abs() < 0.5);
println!("x* = {:.3}, f_mean = {:.4} +/- {:.4}",
    result.x[0], result.f_mean, result.f_std_error);
```

### Stochastic Parameters

Numra supports several parameter distributions:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::StochasticProblem;

let problem = StochasticProblem::new(2)
    .x0(&[0.0, 0.0])
    .objective(|x: &[f64], p: &[f64]| {
        (x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)
    })
    .param_normal("demand", 100.0, 15.0)     // Normal(mean, std)
    .param_uniform("price", 8.0, 12.0)       // Uniform(lo, hi)
    .n_samples(500)
    .seed(123);
```

For custom distributions, use `param_sampled`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::stochastic::param_sampled;
use rand::rngs::StdRng;
use rand_distr::{Distribution, LogNormal};

let dist = LogNormal::new(2.0, 0.5).unwrap();
let param = param_sampled("yield", 7.39, move |rng: &mut StdRng| {
    dist.sample(rng)
});
```

### Chance Constraints

A chance constraint enforces \\( P\{g(x, \xi) \le 0\} \ge \alpha \\). Numra approximates this via a smooth quadratic penalty:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::StochasticProblem;

// Maximize x (min -x) subject to P{x <= xi} >= 0.95
// xi ~ N(10, 2). Optimal: x ~ 10 - 1.645*2 = 6.71
let result = StochasticProblem::new(1)
    .x0(&[5.0])
    .objective(|x: &[f64], _p: &[f64]| -x[0])
    .chance_constraint(
        |x: &[f64], p: &[f64]| x[0] - p[0],  // x - xi <= 0
        0.95,
    )
    .param_normal("xi", 10.0, 2.0)
    .bounds(0, (0.0, 20.0))
    .n_samples(500)
    .max_iter(2000)
    .solve()
    .unwrap();

assert!(result.x[0] < 10.0);
println!("x* = {:.2}", result.x[0]);
println!("Chance satisfaction: {:.1}%",
    result.chance_satisfaction[0] * 100.0);
```

### CVaR Minimization

**Conditional Value at Risk (CVaR)** at confidence level \\( \alpha \\) is the expected cost in the worst \\( (1-\alpha) \\) fraction of scenarios. Numra implements the **Rockafellar--Uryasev reformulation**:

\\[
\text{CVaR}_\alpha(X) = \min_t \Bigl\{ t + \frac{1}{1-\alpha} \, \mathbb{E}\bigl[\max(0, X - t)\bigr] \Bigr\},
\\]

where an auxiliary variable \\( t \\) (the VaR estimate) is appended to the decision vector. A smooth approximation \\( \max(0, z) \approx \frac{z + \sqrt{z^2 + \epsilon}}{2} \\) ensures differentiability.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::StochasticProblem;

// Minimize CVaR_0.9 of (x - xi)^2 where xi ~ N(5, 2)
let result = StochasticProblem::new(1)
    .x0(&[3.0])
    .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
    .param_normal("xi", 5.0, 2.0)
    .n_samples(500)
    .max_iter(2000)
    .minimize_cvar(0.9)
    .solve()
    .unwrap();

// CVaR-optimal is near the mean for symmetric distributions
assert!((result.x[0] - 5.0).abs() < 1.0);
assert_eq!(result.x.len(), 1);  // auxiliary variable stripped
println!("CVaR x* = {:.3}, f_mean = {:.4}", result.x[0], result.f_mean);
```

### Stochastic Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `x` | `Vec<S>` | Optimal decision variables |
| `f_mean` | `S` | Sample-average objective value |
| `f_std_error` | `S` | Standard error of the objective estimate |
| `scenario_objectives` | `Vec<S>` | Per-scenario objective values |
| `chance_satisfaction` | `Vec<S>` | Fraction satisfied per chance constraint |
| `converged` | `bool` | Convergence status |

### Stochastic Options

| Option | Default | Description |
|--------|---------|-------------|
| `n_samples` | 100 | Number of SAA scenarios |
| `seed` | 42 | Random seed |
| `max_iter` | 1000 | Maximum optimizer iterations |

## Robust vs. Stochastic: When to Use Which

| Criterion | Robust | Stochastic (SAA) |
|-----------|--------|------------------|
| Uncertainty model | Bounded (confidence set) | Distributional (samples) |
| Objective | Nominal (worst-case constraints) | Expected value or CVaR |
| Conservatism | Can be overly conservative | Risk-aware (CVaR) or risk-neutral (EV) |
| Computational cost | One solve (tightened constraints) | One solve (larger problem with N scenarios) |
| Best for | Safety-critical constraints | Cost optimization under uncertainty |

## Combined Workflow

For complex problems, combine both approaches:

1. Use **robust optimization** for hard safety constraints.
2. Use **stochastic optimization** to optimize expected performance.
3. Use **parametric sensitivity** to understand which parameters matter most.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Step 1: Identify critical parameters via robust sensitivity
let robust_result = RobustProblem::new(n)
    .objective(obj)
    .param("p1", 10.0, 1.0)
    .param("p2", 5.0, 2.0)
    .solve()
    .unwrap();

if let Some(sens) = &robust_result.sensitivity {
    println!("Sensitivity to p1: {:?}", sens.column(0));
    println!("Sensitivity to p2: {:?}", sens.column(1));
}
```
