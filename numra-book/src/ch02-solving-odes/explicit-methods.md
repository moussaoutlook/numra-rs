# Explicit Methods

Explicit Runge-Kutta methods are the workhorses for non-stiff ODEs. They advance
the solution using a weighted combination of derivative evaluations, requiring
no linear algebra -- just function calls and additions.

Numra provides five explicit methods of increasing order and accuracy.

## DoPri5 -- Dormand-Prince 5(4)

The most widely-used general-purpose ODE solver. This is the method behind
MATLAB's `ode45` and SciPy's `RK45`.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = -y[0];  // simple harmonic oscillator
    },
    0.0, 100.0, vec![1.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = DoPri5::solve(&problem, 0.0, 100.0, &[1.0, 0.0], &options).unwrap();

// DoPri5 uses the FSAL property: the last stage of step n
// is the first stage of step n+1, saving one function call per step.
println!("Function evals: {}", result.stats.n_eval);
```

**Properties:**
- Order 5 with embedded order 4 error estimator
- 7 stages (effectively 6 per step due to FSAL)
- PI step size controller for smooth step adaptation
- Free 4th-order dense output interpolant
- Event detection support via Hermite cubic interpolation

**When to use:** Your default choice for non-stiff problems. Handles most
textbook ODEs efficiently.

## Tsit5 -- Tsitouras 5(4)

An optimized alternative to DoPri5 with coefficients tuned for better
performance on typical ODE problems.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Tsit5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = (1.0 - y[0] * y[0]) * y[1] - y[0]; // Van der Pol, mu=1
    },
    0.0, 20.0, vec![2.0, 0.0],
);

let result = Tsit5::solve(&problem, 0.0, 20.0, &[2.0, 0.0],
    &SolverOptions::default()).unwrap();
```

**Properties:**
- Order 5(4), same structure as DoPri5
- FSAL property
- Optimized coefficients often give fewer function evaluations than DoPri5
- Simpler step size controller (I-controller)

**When to use:** When you want slightly better efficiency than DoPri5.
This is the default choice in Julia's DifferentialEquations.jl.

## Vern6 -- Verner's 6(5) "Efficient"

For problems requiring more than 6 digits of accuracy.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Vern6, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        // Two-body problem (Kepler orbit)
        let r3 = (y[0] * y[0] + y[1] * y[1]).powf(1.5);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -y[0] / r3;
        dydt[3] = -y[1] / r3;
    },
    0.0, 20.0, vec![1.0, 0.0, 0.0, 1.0],  // circular orbit
);

let options = SolverOptions::default().rtol(1e-10).atol(1e-12);
let result = Vern6::solve(&problem, 0.0, 20.0, &[1.0, 0.0, 0.0, 1.0], &options).unwrap();
```

**Properties:**
- Order 6 with embedded order 5 error estimator
- 10 stages (FSAL)
- Better efficiency than DoPri5 at tight tolerances (fewer steps needed)

**When to use:** Medium-to-high accuracy problems (rtol < 1e-8).

## Vern7 -- Verner's 7(6) "Efficient"

High-accuracy solver for demanding problems.

**Properties:**
- Order 7 with embedded order 6 error estimator
- 10 stages
- Excellent for tight tolerances (rtol ~ 1e-10 to 1e-12)

**When to use:** High-accuracy requirements where Vern6 isn't enough.

## Vern8 -- Verner's 8(7) "Efficient"

The highest-order explicit method in Numra.

**Properties:**
- Order 8 with embedded order 7 error estimator
- 13 stages
- Very high accuracy with minimal function evaluations at tight tolerances

**When to use:** Near-machine-precision accuracy (rtol < 1e-12). Celestial
mechanics, reference solution computation, problems where every digit matters.

## Method Comparison

| Method | Order | Stages | FSAL | Dense Output | Step Control |
|--------|-------|--------|------|-------------|--------------|
| DoPri5 | 5(4) | 7 | Yes | 4th order | PI controller |
| Tsit5 | 5(4) | 7 | Yes | -- | I controller |
| Vern6 | 6(5) | 10 | Yes | -- | I controller |
| Vern7 | 7(6) | 10 | No | -- | I controller |
| Vern8 | 8(7) | 13 | No | -- | I controller |

### Efficiency at Different Tolerances

Higher-order methods pay more per step (more stages) but take fewer steps. The
crossover points are roughly:

- **rtol > 1e-6:** DoPri5 or Tsit5 (fewest evaluations per step)
- **1e-10 < rtol < 1e-6:** Vern6 (fewer steps outweighs extra stages)
- **1e-12 < rtol < 1e-10:** Vern7 (high order really pays off)
- **rtol < 1e-12:** Vern8 (every digit of order matters)

## Step Size Controllers

### PI Controller (DoPri5)

DoPri5 uses a proportional-integral controller for smooth step size adaptation:

\\[
h_{new} = h \cdot \left(\frac{1}{\text{err}}\right)^{\beta_1} \cdot \left(\frac{\text{err}_{old}}{\text{err}}\right)^{\beta_2}
\\]

The PI controller remembers the previous error estimate, producing smoother
step size sequences with less oscillation than simpler controllers.

### I Controller (Tsit5, Verner)

The integral-only controller uses:

\\[
h_{new} = h \cdot \text{safety} \cdot \left(\frac{1}{\text{err}}\right)^{1/p}
\\]

Simpler and slightly less smooth, but effective. The safety factor (0.9) and
min/max factors (0.2, 10.0) prevent drastic step changes.
