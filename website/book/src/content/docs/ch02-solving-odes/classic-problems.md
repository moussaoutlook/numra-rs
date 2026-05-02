---
title: "Classic Problems"
---

This section presents the standard test problems used throughout the ODE
literature. Each example includes the mathematical formulation, a working
Numra code example, expected behavior, and guidance on which solver to use.

## Exponential Decay

The simplest ODE with a known analytical solution.

$$
y' = -y, \quad y(0) = 1, \quad \text{solution: } y(t) = e^{-t}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0, 5.0,
    vec![1.0],
);

let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

let y_final = result.y_final().unwrap();
let exact = (-5.0_f64).exp();
let error = (y_final[0] - exact).abs();
println!("Numerical: {:.10}", y_final[0]);
println!("Exact:     {:.10}", exact);
println!("Error:     {:.2e}", error);
// Error should be well below 1e-8
```

**Solver:** Any explicit method works. DoPri5 or Tsit5 are natural choices.
This problem is non-stiff so implicit methods would be overkill.

**Purpose:** Verifying basic solver correctness and convergence order.

## Harmonic Oscillator

A 2D system with periodic solutions and an energy invariant.

$$
y_0' = y_1, \quad y_1' = -y_0
$$

The solution is $ y_0(t) = \cos(t) $, $ y_1(t) = -\sin(t) $ for
initial conditions $ y(0) = [1, 0] $. The energy
$ E = y_0^2 + y_1^2 $ is conserved (always equals 1).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    },
    0.0, 100.0,
    vec![1.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-10).atol(1e-12);
let result = DoPri5::solve(&problem, 0.0, 100.0, &[1.0, 0.0], &options).unwrap();

// Check energy conservation across the trajectory
let mut max_energy_error = 0.0_f64;
for (_t, y) in result.iter() {
    let energy = y[0] * y[0] + y[1] * y[1];
    let err = (energy - 1.0).abs();
    max_energy_error = max_energy_error.max(err);
}

let y_final = result.y_final().unwrap();
println!("y(100)  = [{:.10}, {:.10}]", y_final[0], y_final[1]);
println!("Exact   = [{:.10}, {:.10}]", 100.0_f64.cos(), -100.0_f64.sin());
println!("Max energy drift: {:.2e}", max_energy_error);
```

**Solver:** DoPri5 or Vern6 for moderate to high accuracy. Energy drift grows
slowly over long integrations due to dissipation in non-symplectic methods.
For very long integrations, tighter tolerances reduce the drift.

**Purpose:** Testing phase accuracy and long-term stability. Energy drift is
a sensitive indicator of integration quality.

## Lotka-Volterra (Predator-Prey)

A nonlinear 2D system with periodic orbits in phase space.

$$
\frac{dx}{dt} = \alpha x - \beta x y, \qquad
\frac{dy}{dt} = \delta x y - \gamma y
$$

The system has a conserved quantity
$ V = \delta x - \gamma \ln x + \beta y - \alpha \ln y $, so orbits are
closed curves in the $ (x, y) $ plane.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Tsit5, Solver, OdeProblem, SolverOptions};

let alpha = 1.5;
let beta = 1.0;
let delta = 1.0;
let gamma = 3.0;

let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = alpha * y[0] - beta * y[0] * y[1];
        dydt[1] = delta * y[0] * y[1] - gamma * y[1];
    },
    0.0, 30.0,
    vec![10.0, 1.0],
);

let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = Tsit5::solve(&problem, 0.0, 30.0, &[10.0, 1.0], &options).unwrap();

// Check that the conserved quantity is maintained
let v = |x: f64, y: f64| -> f64 {
    delta * x - gamma * x.ln() + beta * y - alpha * y.ln()
};

let v0 = v(10.0, 1.0);
let mut max_v_error = 0.0_f64;
for (_t, y) in result.iter() {
    let err = (v(y[0], y[1]) - v0).abs();
    max_v_error = max_v_error.max(err);
}

let y_final = result.y_final().unwrap();
println!("Final: prey = {:.4}, predator = {:.4}", y_final[0], y_final[1]);
println!("Max invariant drift: {:.2e}", max_v_error);
println!("Steps: {}", result.stats.n_accept);
```

**Solver:** Tsit5 or DoPri5. Non-stiff, moderately nonlinear. Populations
oscillate periodically -- prey booms lead to predator booms, which crash the
prey, which crashes the predators, and the cycle repeats.

**Purpose:** Testing conservation of nonlinear invariants and handling of
oscillatory solutions.

## Lorenz Attractor

The iconic chaotic system. Three coupled nonlinear ODEs with sensitive
dependence on initial conditions.

$$
\dot{x} = \sigma(y - x), \quad
\dot{y} = x(\rho - z) - y, \quad
\dot{z} = xy - \beta z
$$

With the standard parameters $ \sigma = 10 $, $ \rho = 28 $,
$ \beta = 8/3 $, the system is chaotic: nearby trajectories diverge
exponentially. The Lyapunov exponent is about 0.9, meaning trajectories
diverge by a factor of $ e $ roughly every 1.1 time units.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let sigma = 10.0;
let rho = 28.0;
let beta = 8.0 / 3.0;

let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = sigma * (y[1] - y[0]);
        dydt[1] = y[0] * (rho - y[2]) - y[1];
        dydt[2] = y[0] * y[1] - beta * y[2];
    },
    0.0, 20.0,
    vec![1.0, 1.0, 1.0],
);

let options = SolverOptions::default().rtol(1e-10).atol(1e-12);
let result = DoPri5::solve(
    &problem, 0.0, 20.0, &[1.0, 1.0, 1.0], &options
).unwrap();

let y_final = result.y_final().unwrap();
println!("Final state: [{:.6}, {:.6}, {:.6}]", y_final[0], y_final[1], y_final[2]);
println!("Steps: {}, Fn calls: {}", result.stats.n_accept, result.stats.n_eval);
```

**Sensitivity to initial conditions:** perturb the initial state by
$ 10^{-10} $ and the solutions diverge completely after a few Lyapunov
times. This is not a solver bug -- it is the nature of chaos.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// Perturbed initial conditions
let y0_perturbed = vec![1.0 + 1e-10, 1.0, 1.0];
let result2 = DoPri5::solve(
    &problem, 0.0, 20.0, &y0_perturbed, &options
).unwrap();

let y1 = result.y_final().unwrap();
let y2 = result2.y_final().unwrap();
let diff = ((y1[0]-y2[0]).powi(2) + (y1[1]-y2[1]).powi(2) + (y1[2]-y2[2]).powi(2)).sqrt();
println!("Divergence after t=20: {:.6}", diff);  // will be O(1) or larger
```

**Solver:** DoPri5 with tight tolerances. The Lorenz system is not stiff, but
high accuracy is important to delay the onset of trajectory divergence. Vern8
can push the "shadow trajectory" further.

**Purpose:** Testing adaptive stepping in chaotic regimes, where the solver
must take small steps during rapid state changes near the attractor's lobes.

## Van der Pol Oscillator

A nonlinear oscillator that transitions from non-stiff to extremely stiff as
the parameter $ \mu $ increases.

$$
y_0' = y_1, \qquad y_1' = \mu(1 - y_0^2) y_1 - y_0
$$

For small $ \mu $, the system exhibits gentle limit cycles. For large
$ \mu $, the solution develops sharp boundary layers -- fast transitions
separated by slow quasi-static phases -- making it severely stiff.

### Non-stiff regime (mu = 1)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Tsit5, Solver, OdeProblem, SolverOptions};

let mu = 1.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 20.0,
    vec![2.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = Tsit5::solve(&problem, 0.0, 20.0, &[2.0, 0.0], &options).unwrap();

let y_final = result.y_final().unwrap();
println!("mu=1: y = [{:.6}, {:.6}]", y_final[0], y_final[1]);
println!("Steps: {}, Rejected: {}", result.stats.n_accept, result.stats.n_reject);
```

### Stiff regime (mu = 1000)

With $ \mu = 1000 $, the solution has sharp transitions every half-period
(roughly $ t \approx 1000 $). Explicit methods would need millions of
steps; an implicit method handles it easily.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Radau5, Solver, OdeProblem, SolverOptions};

let mu = 1000.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 3000.0,
    vec![2.0, 0.0],
);

let options = SolverOptions::default()
    .rtol(1e-6)
    .atol(1e-8)
    .max_steps(200_000);

let result = Radau5::solve(
    &problem, 0.0, 3000.0, &[2.0, 0.0], &options
).unwrap();

let y_final = result.y_final().unwrap();
println!("mu=1000: y = [{:.6}, {:.6}]", y_final[0], y_final[1]);
println!("Steps: {}, LU: {}", result.stats.n_accept, result.stats.n_lu);
```

**Why is it stiff?** When $ |y_0| > 1 $, the $ \mu(1 - y_0^2) $ term
creates a large negative damping coefficient, forcing the solution to change on
a timescale of $ 1/\mu $. Explicit methods must resolve this fast scale
even during the slow phases, while Radau5 steps right over it.

**Solver:** Tsit5 for $ \mu \leq 10 $, Esdirk54 for $ \mu \sim 100 $,
Radau5 for $ \mu \geq 1000 $. This problem is the classic benchmark for
demonstrating when you need to switch from explicit to implicit methods.

**Purpose:** The definitive stiffness test. If your solver can handle Van der
Pol at $ \mu = 1000 $ efficiently, it can handle most stiff problems.

## Robertson Chemical Kinetics

An extremely stiff 3-species chemical system with rate constants spanning 11
orders of magnitude.

$$
y_0' = -0.04 \, y_0 + 10^4 \, y_1 \, y_2
$$
$$
y_1' = 0.04 \, y_0 - 10^4 \, y_1 \, y_2 - 3 \times 10^7 \, y_1^2
$$
$$
y_2' = 3 \times 10^7 \, y_1^2
$$

The initial conditions are $ y(0) = [1, 0, 0] $ and the system conserves
mass: $ y_0 + y_1 + y_2 = 1 $. The component $ y_1 $ reaches a
maximum of about $ 3.4 \times 10^{-5} $ and then decays, while $ y_2 $
slowly approaches 1 as $ y_0 $ decays. Integration to $ t = 10^{11} $
is typical.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Radau5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    },
    0.0, 1e5,
    vec![1.0, 0.0, 0.0],
);

let options = SolverOptions::default()
    .rtol(1e-6)
    .atol(1e-10);

let result = Radau5::solve(
    &problem, 0.0, 1e5, &[1.0, 0.0, 0.0], &options
).unwrap();

let y_final = result.y_final().unwrap();
let mass = y_final[0] + y_final[1] + y_final[2];
println!("y = [{:.6e}, {:.6e}, {:.6e}]", y_final[0], y_final[1], y_final[2]);
println!("Mass conservation: {:.2e}", (mass - 1.0).abs());
println!("Steps: {}, LU: {}", result.stats.n_accept, result.stats.n_lu);
```

**Solver:** Only `Radau5` handles this reliably. The stiffness ratio exceeds
$ 10^{11} $, and the solution spans many orders of magnitude in time.
BDF can work at lower tolerances. Explicit methods will fail completely.

**Purpose:** The acid test for stiff solvers. If a method can integrate
Robertson to $ t = 10^{11} $ while maintaining mass conservation, it is
production-ready for stiff problems.

## Problem Summary

| Problem | Dimension | Stiff? | Recommended Solver | Key Test |
|---------|-----------|--------|-------------------|----------|
| Exponential decay | 1 | No | DoPri5 | Basic correctness |
| Harmonic oscillator | 2 | No | DoPri5, Vern6 | Energy conservation |
| Lotka-Volterra | 2 | No | Tsit5 | Nonlinear invariant |
| Lorenz | 3 | No | DoPri5 | Chaos, adaptivity |
| Van der Pol (mu=1) | 2 | No | Tsit5 | Limit cycles |
| Van der Pol (mu=1000) | 2 | Yes | Radau5 | Stiffness handling |
| Robertson | 3 | Extreme | Radau5 | Extreme stiffness |
