# Quick Start

This page walks through five common tasks to give you a feel for Numra's API.

## 1. Solve an ODE

The Lorenz system is a classic chaotic ODE:

\\[
\frac{dx}{dt} = \sigma(y - x), \quad
\frac{dy}{dt} = x(\rho - z) - y, \quad
\frac{dz}{dt} = xy - \beta z
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

fn main() {
    let sigma = 10.0;
    let rho = 28.0;
    let beta = 8.0 / 3.0;

    let problem = OdeProblem::new(
        move |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = sigma * (y[1] - y[0]);
            dydt[1] = y[0] * (rho - y[2]) - y[1];
            dydt[2] = y[0] * y[1] - beta * y[2];
        },
        0.0, 50.0, vec![1.0, 1.0, 1.0],
    );

    let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
    let result = DoPri5::solve(&problem, 0.0, 50.0, &[1.0, 1.0, 1.0], &options).unwrap();

    println!("Steps: {}", result.n_steps());
    println!("Final state: {:?}", result.y_final().unwrap());
}
```

## 2. Optimize a Function

Find the minimum of the Rosenbrock function using BFGS:

\\[
f(x, y) = (1 - x)^2 + 100(y - x^2)^2
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{bfgs_minimize, OptimOptions};

fn main() {
    let result = bfgs_minimize(
        |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        },
        &[-1.0, 1.0],  // initial guess
        &OptimOptions::default(),
    ).unwrap();

    println!("Minimum at: {:?}", result.x);
    println!("f(x*) = {:.2e}", result.fval);
    println!("Iterations: {}", result.iterations);
}
```

## 3. Compute an FFT

Analyze the frequency content of a signal:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fft::{rfft, fftfreq};

fn main() {
    let fs = 1000.0;  // 1 kHz sample rate
    let n = 1024;
    let dt = 1.0 / fs;

    // Signal: 50 Hz + 120 Hz
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
        })
        .collect();

    let spectrum = rfft(&signal);
    let freqs = fftfreq(n, dt);

    // Find peak frequencies
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
    let max_idx = magnitudes.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;

    println!("Dominant frequency: {:.1} Hz", freqs[max_idx]);
}
```

## 4. Fit a Curve to Data

Fit an exponential decay model to noisy data:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::{curve_fit, FitOptions};

fn main() {
    // Generate noisy data: y = 3.0 * exp(-0.5 * x)
    let x_data: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let y_data: Vec<f64> = x_data.iter()
        .map(|&x| 3.0 * (-0.5 * x).exp() + 0.05 * (x * 137.0).sin())
        .collect();

    // Model: y = a * exp(b * x)
    let result = curve_fit(
        |x: f64, params: &[f64]| params[0] * (params[1] * x).exp(),
        &x_data,
        &y_data,
        &[1.0, -1.0],  // initial guess
        None,
    ).unwrap();

    println!("Fitted: a = {:.4}, b = {:.4}", result.params[0], result.params[1]);
    println!("R^2 = {:.6}", result.r_squared);
}
```

## 5. Solve a Stochastic DE

Simulate geometric Brownian motion (stock price model):

\\[
dS = \mu S \, dt + \sigma S \, dW
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::sde::{SdeSystem, EulerMaruyama, SdeSolver, SdeOptions};

fn main() {
    let mu = 0.05;     // drift
    let sigma = 0.2;   // volatility

    let system = SdeSystem::new(
        1,  // dimension
        move |_t, y: &[f64], drift: &mut [f64]| {
            drift[0] = mu * y[0];
        },
        move |_t, y: &[f64], diffusion: &mut [f64]| {
            diffusion[0] = sigma * y[0];
        },
    );

    let options = SdeOptions::new(0.001, 42);  // dt = 0.001, seed = 42
    let result = EulerMaruyama::solve(&system, 0.0, 1.0, &[100.0], &options);

    println!("Initial price: {:.2}", 100.0);
    println!("Final price:   {:.2}", result.y_final()[0]);
    println!("Steps: {}", result.n_steps());
}
```

## Next Steps

- [Architecture Overview](./architecture-overview.md) -- understand how the
  crates fit together
- [Solving ODEs](../ch02-solving-odes/your-first-ode.md) -- deep dive into
  ODE solver selection and configuration
- [Solver Comparison](../ch02-solving-odes/solver-comparison.md) -- which
  solver to use for your problem
