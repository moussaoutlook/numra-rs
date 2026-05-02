---
title: "Composing Solvers"
---

Numra's modular crate architecture means you can combine solvers from different
domains to tackle complex problems. This chapter shows how to chain ODE solvers
with optimization, automatic differentiation, interpolation, and more.

## ODE + Optimization: Parameter Fitting

Fit ODE model parameters to experimental data by embedding the ODE solver
inside an optimizer:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};
use numra::fit::{curve_fit, FitOptions};

// Experimental data: time points and observed values
let t_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let y_data = vec![1.0, 0.61, 0.37, 0.22, 0.14, 0.08];

// Model: y' = -k * y, y(0) = 1 → y(t) = exp(-k*t)
// We want to find k that best fits the data.
let model = |t: f64, params: &[f64]| -> f64 {
    let k = params[0];
    let problem = OdeProblem::new(
        move |_t, y: &[f64], dydt: &mut [f64]| { dydt[0] = -k * y[0]; },
        0.0, t,
        vec![1.0],
    );
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);
    let result = DoPri5::solve(&problem, 0.0, t, &[1.0], &opts).unwrap();
    result.y_final().unwrap()[0]
};

let fit = curve_fit(model, &t_data, &y_data, &[0.5], &FitOptions::default()).unwrap();
println!("Fitted k = {:.6}", fit.params[0]);  // should be ~1.0
```

For multi-parameter or multi-state problems, use `ParamEstProblem` from
`numra::ocp` instead (see [Parameter Estimation](../ch10-optimal-control/parameter-estimation)).

## ODE + Automatic Differentiation

Compute derivatives of ODE solutions with respect to parameters using
forward-mode AD:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::autodiff::{Dual, gradient};

// Sensitivity: how does y(T) change with initial condition?
let dy_dy0 = gradient(|params| {
    // The ODE solver computes with Dual numbers for AD
    let y0 = params[0];
    let k = 0.5;
    // For a simple linear ODE, the analytical gradient is exp(-k*T)
    // For complex ODEs, you'd run the solver with Dual arithmetic
    (-k * 5.0_f64).exp() * y0  // y(T) = y0 * exp(-kT)
}, &[1.0]);

println!("dy/dy0 = {:.6}", dy_dy0[0]);  // exp(-2.5) ≈ 0.0821
```

For large-scale sensitivity analysis, the adjoint method (see
[Adjoint Methods](../ch10-optimal-control/adjoint-methods)) is more
efficient than forward-mode AD when the number of parameters exceeds the
number of outputs.

## ODE + Interpolation: Resampling Solutions

ODE solvers produce output at adaptively-chosen time points. Use interpolation
to resample onto a uniform grid for FFT or comparison:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};
use numra::interp::{CubicSpline, Interpolant};

// Solve the ODE
let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    },
    0.0, 10.0,
    vec![1.0, 0.0],
);

let opts = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &opts).unwrap();

// Extract component 0 and build a cubic spline
let t_points: Vec<f64> = result.t.clone();
let y0_points: Vec<f64> = result.component(0);

let spline = CubicSpline::new(&t_points, &y0_points);

// Resample at 1000 uniform points for FFT
let n = 1000;
let dt = 10.0 / n as f64;
let uniform: Vec<f64> = (0..n).map(|i| spline.interpolate(i as f64 * dt)).collect();
```

Alternatively, use `t_eval` in solver options for direct output at specified
times:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let times: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
let opts = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10)
    .t_eval(times);
```

## ODE + Statistics: Ensemble Analysis

Run many ODE solves with random parameters and analyze the distribution:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};
use numra::stats::{mean, std_dev, quantile};

let n_samples = 1000;
let mut finals = Vec::with_capacity(n_samples);

for i in 0..n_samples {
    let k = 0.4 + 0.2 * (i as f64 / n_samples as f64);  // sweep k in [0.4, 0.6]
    let problem = OdeProblem::new(
        move |_t, y: &[f64], dydt: &mut [f64]| { dydt[0] = -k * y[0]; },
        0.0, 5.0, vec![1.0],
    );
    let opts = SolverOptions::default();
    let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &opts).unwrap();
    finals.push(result.y_final().unwrap()[0]);
}

println!("Mean:   {:.6}", mean(&finals));
println!("Std:    {:.6}", std_dev(&finals));
println!("Median: {:.6}", quantile(&finals, 0.5));
```

## ODE + Signal Processing: Frequency Analysis

Solve an ODE and analyze its frequency content:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Tsit5, Solver, OdeProblem, SolverOptions};
use numra::fft::fft;

// Van der Pol oscillator - produces a limit cycle
let mu = 1.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 100.0,
    vec![2.0, 0.0],
);

// Use t_eval for uniform sampling (required by FFT)
let dt = 0.1;
let times: Vec<f64> = (0..1000).map(|i| i as f64 * dt).collect();
let opts = SolverOptions::default().rtol(1e-8).atol(1e-10).t_eval(times);
let result = Tsit5::solve(&problem, 0.0, 100.0, &[2.0, 0.0], &opts).unwrap();

let y0 = result.component(0);
let spectrum = fft(&y0);
// The dominant frequency should be near 1/(2π) ≈ 0.159 Hz
```

## Composition Patterns Summary

| Combination | Use Case | Key Insight |
|------------|----------|-------------|
| ODE + Optimization | Parameter fitting | Solver runs inside objective function |
| ODE + Autodiff | Sensitivity | Derivatives through the solver |
| ODE + Interpolation | Resampling | Uniform grid from adaptive output |
| ODE + Statistics | Ensemble | Distribution from many solves |
| ODE + FFT | Frequency analysis | Spectral content of dynamics |
| ODE + Uncertainty | Error bounds | Propagate parameter uncertainty |
| SDE + Statistics | Stochastic analysis | Path statistics over realizations |
