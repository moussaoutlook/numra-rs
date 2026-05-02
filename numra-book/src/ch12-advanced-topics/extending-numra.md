# Extending Numra

Numra is designed around traits that you can implement for your own types.
This chapter shows how to extend the library with custom solvers, ODE systems,
interpolants, and more.

## Custom ODE System

The simplest extension point. Implement `OdeSystem<S>` to define your problem
as a struct instead of a closure:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeSystem, DoPri5, Solver, SolverOptions};
use numra_core::Scalar;

struct Pendulum<S: Scalar> {
    length: S,
    gravity: S,
    damping: S,
}

impl<S: Scalar> OdeSystem<S> for Pendulum<S> {
    fn dim(&self) -> usize { 2 }

    fn rhs(&self, _t: S, y: &[S], dydt: &mut [S]) {
        // y[0] = angle, y[1] = angular velocity
        dydt[0] = y[1];
        dydt[1] = -(self.gravity / self.length) * y[0].sin()
                  - self.damping * y[1];
    }

    fn jacobian(&self, _t: S, y: &[S], jac: &mut [S]) {
        // Analytical Jacobian for better implicit solver performance
        let n = 2;
        jac[0*n + 0] = S::ZERO;
        jac[0*n + 1] = S::ONE;
        jac[1*n + 0] = -(self.gravity / self.length) * y[0].cos();
        jac[1*n + 1] = -self.damping;
    }

    fn is_autonomous(&self) -> bool { true }
}

// Use it with any solver
let pendulum = Pendulum {
    length: 1.0,
    gravity: 9.81,
    damping: 0.1,
};

let opts = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = DoPri5::solve(&pendulum, 0.0, 20.0, &[1.0, 0.0], &opts).unwrap();
```

### Benefits over closures

| Feature | Closure | Struct + OdeSystem |
|---------|---------|-------------------|
| **Simplicity** | Easy for quick experiments | Requires more boilerplate |
| **Jacobian** | Finite-difference only | Analytical available |
| **Reusability** | Single use | Parameterized, reusable |
| **Testing** | Hard to unit test | Easy to test RHS and Jacobian |
| **Mass matrix** | Not available | DAE support via `mass_matrix()` |

## Custom ODE System with Mass Matrix (DAE)

For differential-algebraic equations M * y' = f(t, y):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
struct PendulumDAE;

impl OdeSystem<f64> for PendulumDAE {
    fn dim(&self) -> usize { 3 }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let (x, v, lambda) = (y[0], y[1], y[2]);
        dydt[0] = v;
        dydt[1] = -lambda * x;
        dydt[2] = x * x + v * v - 1.0;  // algebraic constraint
    }

    fn has_mass_matrix(&self) -> bool { true }

    fn mass_matrix(&self, mass: &mut [f64]) {
        // M = diag(1, 1, 0) -- third equation is algebraic
        let n = 3;
        for i in 0..n*n { mass[i] = 0.0; }
        mass[0*n + 0] = 1.0;
        mass[1*n + 1] = 1.0;
        // mass[2*n + 2] = 0.0  -- algebraic row
    }
}
```

## Custom Event Function

Detect specific conditions during integration:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{EventFunction, EventDirection, EventAction};
use numra_core::Scalar;

/// Detect when a specific component crosses a threshold
struct ThresholdCrossing {
    component: usize,
    threshold: f64,
}

impl EventFunction<f64> for ThresholdCrossing {
    fn trigger(&self, _t: f64, y: &[f64]) -> f64 {
        y[self.component] - self.threshold
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Decreasing  // only when crossing downward
    }

    fn action(&self) -> EventAction {
        EventAction::Continue  // or Stop to halt integration
    }
}

// Usage
let event = ThresholdCrossing { component: 0, threshold: 0.01 };
let opts = SolverOptions::default()
    .event(Box::new(event));
```

## Custom Solver

Implement the `Solver<S>` trait to add your own integration method:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Solver, SolverResult, SolverOptions, SolverStats, OdeSystem};
use numra::ode::SolverError;
use numra_core::Scalar;

/// Simple forward Euler method (for demonstration only!)
struct ForwardEuler;

impl<S: Scalar> Solver<S> for ForwardEuler {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError> {
        let n = problem.dim();
        let h = S::from_f64(0.001);  // fixed step size
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut dydt = vec![S::ZERO; n];

        let mut t_out = vec![t];
        let mut y_out = y.clone();
        let mut stats = SolverStats::new();

        while t < tf {
            let step = if t + h > tf { tf - t } else { h };
            problem.rhs(t, &y, &mut dydt);

            for i in 0..n {
                y[i] = y[i] + step * dydt[i];
            }
            t = t + step;

            t_out.push(t);
            y_out.extend_from_slice(&y);
            stats.n_accept += 1;
        }

        Ok(SolverResult::new(t_out, y_out, n, stats))
    }
}
```

**Warning:** Forward Euler is first-order and has no error control. This is
for demonstration only. Real solvers need adaptive step-size control, error
estimation, and embedded pairs.

## Custom Interpolant

Implement the `Interpolant` trait from `numra-interp`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::Interpolant;

/// Nearest-neighbor interpolation
struct NearestNeighbor {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl NearestNeighbor {
    fn new(x: &[f64], y: &[f64]) -> Self {
        Self { x: x.to_vec(), y: y.to_vec() }
    }
}

impl Interpolant<f64> for NearestNeighbor {
    fn interpolate(&self, t: f64) -> f64 {
        // Find nearest data point
        let mut best = 0;
        let mut best_dist = (t - self.x[0]).abs();
        for i in 1..self.x.len() {
            let dist = (t - self.x[i]).abs();
            if dist < best_dist {
                best = i;
                best_dist = dist;
            }
        }
        self.y[best]
    }

    fn derivative(&self, _t: f64) -> f64 {
        0.0  // piecewise constant has zero derivative
    }

    fn integrate(&self, a: f64, b: f64) -> f64 {
        // Simple rectangle rule using nearest values
        let n = 100;
        let h = (b - a) / n as f64;
        (0..n).map(|i| h * self.interpolate(a + (i as f64 + 0.5) * h)).sum()
    }
}
```

## Custom Nonlinear System

For the Newton solver in `numra-nonlinear`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::nonlinear::{NonlinearSystem, Newton};

struct CircleIntersection;

impl NonlinearSystem<f64> for CircleIntersection {
    fn dim(&self) -> usize { 2 }

    fn residual(&self, x: &[f64], f: &mut [f64]) {
        // Two circles: x^2 + y^2 = 1 and (x-1)^2 + y^2 = 1
        f[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
        f[1] = (x[0] - 1.0) * (x[0] - 1.0) + x[1] * x[1] - 1.0;
    }

    fn jacobian(&self, x: &[f64], jac: &mut [f64]) {
        jac[0] = 2.0 * x[0];           jac[1] = 2.0 * x[1];
        jac[2] = 2.0 * (x[0] - 1.0);  jac[3] = 2.0 * x[1];
    }
}
```

## Trait Summary

| Trait | Crate | Purpose | Key methods |
|-------|-------|---------|-------------|
| `Scalar` | numra-core | Numeric type | `from_f64`, math ops |
| `Vector` | numra-core | Vector ops | `axpy`, `dot`, `norm2` |
| `Signal` | numra-core | Forcing functions | `value_at(t)` |
| `OdeSystem` | numra-ode | ODE definition | `rhs`, `jacobian`, `dim` |
| `Solver` | numra-ode | ODE solver | `solve` |
| `EventFunction` | numra-ode | Event detection | `trigger`, `direction`, `action` |
| `Interpolant` | numra-interp | Interpolation | `interpolate`, `derivative` |
| `NonlinearSystem` | numra-nonlinear | F(x) = 0 | `residual`, `jacobian` |
| `Matrix` | numra-linalg | Matrix operations | `get`, `set`, `nrows`, `ncols` |
