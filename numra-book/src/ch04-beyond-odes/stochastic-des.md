# Stochastic Differential Equations

Stochastic differential equations (SDEs) generalize ordinary differential equations by
incorporating random noise, making them essential for modeling systems subject to
uncertainty, thermal fluctuations, or market volatility. Numra provides the `numra-sde`
crate with multiple solvers, noise types, and a parallel Monte Carlo ensemble framework.

## Mathematical Background

An SDE in Ito form is written as:

\\[
dX(t) = f(t, X)\,dt + g(t, X)\,dW(t)
\\]

where:
- \\(f(t, X)\\) is the **drift** coefficient (deterministic tendency),
- \\(g(t, X)\\) is the **diffusion** coefficient (noise intensity),
- \\(W(t)\\) is a **Wiener process** (Brownian motion) satisfying \\(dW \sim \mathcal{N}(0, dt)\\).

### Ito vs. Stratonovich Interpretation

The same SDE notation can be interpreted differently:

| Property | Ito | Stratonovich |
|---|---|---|
| Chain rule | Modified (Ito's lemma) | Standard calculus chain rule |
| Martingale property | \\(\mathbb{E}[\int g\,dW] = 0\\) | Not a martingale in general |
| Numerical methods | Euler-Maruyama, Milstein | Heun-type SDE methods |
| Conversion | \\(f_{\text{Ito}} = f_{\text{Strat}} - \frac{1}{2}g \frac{\partial g}{\partial x}\\) | Add correction term |

Numra uses the **Ito interpretation** throughout. If your model is naturally expressed in
Stratonovich form, apply the drift correction before defining the system.

### Convergence Orders

SDE solvers have two notions of convergence:

- **Strong order** \\(\gamma\\): pathwise accuracy, \\(\mathbb{E}[|X_N - X(T)|] = O(\Delta t^\gamma)\\).
- **Weak order** \\(\beta\\): accuracy of expectations, \\(|\mathbb{E}[h(X_N)] - \mathbb{E}[h(X(T))]| = O(\Delta t^\beta)\\).

Strong convergence tracks individual trajectories; weak convergence suffices for computing
statistics (mean, variance, distributions).

## The SdeSystem Trait

Every SDE in Numra implements the `SdeSystem` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub trait SdeSystem<S: Scalar>: Sync {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Evaluate the drift function f(t, x).
    fn drift(&self, t: S, x: &[S], f: &mut [S]);

    /// Evaluate the diffusion function g(t, x).
    fn diffusion(&self, t: S, x: &[S], g: &mut [S]);

    /// Type of noise (default: Diagonal).
    fn noise_type(&self) -> NoiseType { NoiseType::Diagonal }

    /// Derivative of diffusion: g * dg/dx (for Milstein).
    /// Default uses finite differences.
    fn diffusion_derivative(&self, t: S, x: &[S], gdg: &mut [S]) { ... }
}
```

## Noise Types

Numra supports three noise configurations via `NoiseType`:

| Noise Type | Description | Wiener Processes | Diffusion Buffer Size |
|---|---|---|---|
| `Diagonal` | Independent noise per component | \\(m = n\\) | \\(n\\) |
| `Scalar` | Single noise source for all components | \\(m = 1\\) | \\(n\\) |
| `General` | Full noise matrix | \\(m\\) (user-specified) | \\(n \times m\\) |

**Diagonal noise** is the default and most common: each state variable has its own
independent Wiener process. **Scalar noise** models a single random forcing that drives
all components. **General noise** allows arbitrary coupling between \\(m\\) Wiener
processes and \\(n\\) state variables.

## Solvers

### Euler-Maruyama (Strong Order 0.5)

The simplest SDE solver. At each step:

\\[
X_{n+1} = X_n + f(t_n, X_n)\,\Delta t + g(t_n, X_n)\,\Delta W_n
\\]

where \\(\Delta W_n \sim \mathcal{N}(0, \Delta t)\\).

- **Strong order**: 0.5
- **Weak order**: 1.0
- **Step size**: Fixed
- **Best for**: Quick prototyping, additive noise problems

### Milstein (Strong Order 1.0)

Adds a correction term that accounts for the curvature of the diffusion:

\\[
X_{n+1} = X_n + f\,\Delta t + g\,\Delta W + \frac{1}{2} g \frac{\partial g}{\partial x} \left(\Delta W^2 - \Delta t\right)
\\]

- **Strong order**: 1.0
- **Weak order**: 1.0
- **Step size**: Fixed
- **Requires**: Diffusion derivative \\(\partial g / \partial x\\) (auto-computed via finite differences, or user-provided for efficiency)
- **Note**: For additive noise (\\(g\\) constant), the correction term vanishes, and Milstein reduces to Euler-Maruyama.

### SRA1 -- Adaptive Strong Order 1.0-1.5

A two-stage stochastic Runge-Kutta method with embedded error estimation for adaptive
step size control:

- **Strong order**: 1.5 for additive noise, 1.0 for multiplicative
- **Step size**: Adaptive (controlled by `rtol` and `atol`)
- **Best for**: Problems where step size sensitivity matters or long integration intervals

### SRA2 -- Adaptive Weak Order 2.0

A three-stage method optimized for weak convergence:

- **Weak order**: 2.0
- **Step size**: Adaptive
- **Best for**: Computing expectations, means, and distributions (Monte Carlo)

## Solver Options

Configure solvers with `SdeOptions` using the builder pattern:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_sde::SdeOptions;

let opts = SdeOptions::default()
    .dt(0.001)              // Fixed time step
    .rtol(1e-3)             // Relative tolerance (adaptive methods)
    .atol(1e-6)             // Absolute tolerance (adaptive methods)
    .dt_max(0.1)            // Maximum time step
    .seed(42)               // Random seed for reproducibility
    .save_trajectory(true); // Save full trajectory (vs. final state only)
```

| Option | Default | Description |
|---|---|---|
| `dt` | 0.01 | Fixed time step |
| `rtol` | 1e-3 | Relative tolerance |
| `atol` | 1e-6 | Absolute tolerance |
| `dt_max` | infinity | Maximum step size |
| `dt_min` | 1e-10 | Minimum step size |
| `max_steps` | 1,000,000 | Safety limit on step count |
| `seed` | None (entropy) | Random seed |
| `save_trajectory` | true | Save all time points |

## Example: Geometric Brownian Motion

Geometric Brownian Motion (GBM) is the foundational model for stock prices in the
Black-Scholes framework:

\\[
dS = \mu S\,dt + \sigma S\,dW
\\]

The exact solution is:

\\[
S(t) = S_0 \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)t + \sigma W(t)\right]
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_sde::{SdeSystem, EulerMaruyama, SdeSolver, SdeOptions};

// Define Geometric Brownian Motion
struct GBM {
    mu: f64,     // drift rate
    sigma: f64,  // volatility
}

impl SdeSystem<f64> for GBM {
    fn dim(&self) -> usize { 1 }

    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.mu * x[0];
    }

    fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma * x[0];
    }
}

let gbm = GBM { mu: 0.05, sigma: 0.2 };
let opts = SdeOptions::default().dt(0.001).seed(42);

let result = EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &opts, None)
    .expect("Solve failed");

let final_price = result.y_final().unwrap()[0];
println!("Final stock price: {:.2}", final_price);
```

### Using Milstein for Higher Accuracy

For GBM, the Milstein correction is analytically computable:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_sde::{SdeSystem, Milstein, SdeSolver, SdeOptions};

struct GBMMillstein { mu: f64, sigma: f64 }

impl SdeSystem<f64> for GBMMillstein {
    fn dim(&self) -> usize { 1 }

    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.mu * x[0];
    }

    fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma * x[0];
    }

    // Analytical: g = sigma*x, dg/dx = sigma, so g*dg/dx = sigma^2*x
    fn diffusion_derivative(&self, _t: f64, x: &[f64], gdg: &mut [f64]) {
        gdg[0] = self.sigma * self.sigma * x[0];
    }
}

let gbm = GBMMillstein { mu: 0.05, sigma: 0.2 };
let opts = SdeOptions::default().dt(0.01).seed(42);

let result = Milstein::solve(&gbm, 0.0, 1.0, &[100.0], &opts, None)
    .expect("Solve failed");
```

## Example: Ornstein-Uhlenbeck Process

The Ornstein-Uhlenbeck (OU) process models mean-reverting dynamics:

\\[
dX = \theta(\mu - X)\,dt + \sigma\,dW
\\]

This has **additive noise** (diffusion is constant), so Euler-Maruyama and Milstein are
equivalent, and SRA1 achieves strong order 1.5.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_sde::{SdeSystem, Sra1, SdeSolver, SdeOptions};

struct OrnsteinUhlenbeck {
    theta: f64,  // mean reversion rate
    mu: f64,     // long-term mean
    sigma: f64,  // volatility
}

impl SdeSystem<f64> for OrnsteinUhlenbeck {
    fn dim(&self) -> usize { 1 }

    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.theta * (self.mu - x[0]);
    }

    fn diffusion(&self, _t: f64, _x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma;
    }
}

let ou = OrnsteinUhlenbeck { theta: 1.0, mu: 0.0, sigma: 0.5 };
let opts = SdeOptions::default()
    .dt(0.01)
    .rtol(1e-4)
    .atol(1e-6)
    .seed(123);

// Use adaptive SRA1 for optimal accuracy with additive noise
let result = Sra1::solve(&ou, 0.0, 10.0, &[1.0], &opts, None)
    .expect("Solve failed");
```

## Monte Carlo Ensemble Simulation

For computing statistics (expected values, confidence intervals, probability densities),
Numra provides `EnsembleRunner` which runs many trajectories in parallel using `rayon`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_sde::{
    SdeSystem, EulerMaruyama, SdeSolver, SdeOptions,
    EnsembleRunner, EnsembleResult,
};

struct GBM { mu: f64, sigma: f64 }

impl SdeSystem<f64> for GBM {
    fn dim(&self) -> usize { 1 }
    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.mu * x[0];
    }
    fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma * x[0];
    }
}

let gbm = GBM { mu: 0.05, sigma: 0.2 };
let opts = SdeOptions::default().dt(0.001).seed(0);

// Run 10,000 trajectories in parallel
let ensemble: EnsembleResult<f64> = EnsembleRunner::run::<_, _, EulerMaruyama>(
    &gbm, 0.0, 1.0, &[100.0], &opts, 10_000
);

// Extract final prices from all successful trajectories
let finals = ensemble.successful_final_values(0);
let mean: f64 = finals.iter().sum::<f64>() / finals.len() as f64;
let variance: f64 = finals.iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f64>() / finals.len() as f64;

println!("E[S(T)] = {:.2} (theory: {:.2})", mean, 100.0 * (0.05_f64).exp());
println!("Std[S(T)] = {:.2}", variance.sqrt());
```

The ensemble automatically assigns unique seeds to each trajectory from the base seed,
ensuring both reproducibility and statistical independence.

### Custom Seeds and Sequential Execution

For fine-grained control:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Run with specific seeds
let seeds = vec![100, 200, 300, 400, 500];
let ensemble = EnsembleRunner::run_with_seeds::<_, _, EulerMaruyama>(
    &gbm, 0.0, 1.0, &[100.0], &opts, &seeds,
);

// Run sequentially (useful for debugging)
let ensemble = EnsembleRunner::run_sequential::<_, _, EulerMaruyama>(
    &gbm, 0.0, 1.0, &[100.0], &opts, 100,
);
```

## Working with Results

`SdeResult` stores solutions in row-major format, identical to ODE results:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = EulerMaruyama::solve(&system, t0, tf, &x0, &opts, None)?;

// Access final state
let x_final: Vec<f64> = result.y_final().unwrap();

// Iterate over (time, state) pairs
for (t, x) in result.iter() {
    println!("t = {:.3}, x = {:?}", t, x);
}

// Access state at index i
let x_i: &[f64] = result.y_at(5);

// Solver statistics
println!("Drift evaluations: {}", result.stats.n_drift);
println!("Accepted steps: {}", result.stats.n_accept);
println!("Rejected steps: {}", result.stats.n_reject);
```

## Choosing a Solver

| Scenario | Recommended Solver | Reason |
|---|---|---|
| Quick prototype | `EulerMaruyama` | Simplest, no derivatives needed |
| Multiplicative noise, pathwise accuracy | `Milstein` | Strong order 1.0 |
| Additive noise, high accuracy | `Sra1` | Strong order 1.5 |
| Monte Carlo (computing expectations) | `Sra2` | Weak order 2.0 |
| Adaptive step size needed | `Sra1` or `Sra2` | Built-in error control |

## Reproducibility

SDE solvers are inherently stochastic, but Numra ensures full reproducibility when
a seed is provided:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let opts = SdeOptions::default().dt(0.01).seed(42);
let r1 = EulerMaruyama::solve(&system, 0.0, 1.0, &x0, &opts, None)?;
let r2 = EulerMaruyama::solve(&system, 0.0, 1.0, &x0, &opts, None)?;

// r1 and r2 produce identical trajectories
assert_eq!(r1.y_final(), r2.y_final());
```

A seed passed directly to `solve()` overrides the one in `SdeOptions`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = EulerMaruyama::solve(&system, 0.0, 1.0, &x0, &opts, Some(99))?;
// Uses seed 99 regardless of opts.seed
```
