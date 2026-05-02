# Solving ODEs

An ordinary differential equation (ODE) describes how a quantity changes over
time. Numra provides 11 ODE solvers covering everything from simple non-stiff
systems to extremely stiff problems and differential-algebraic equations.

## The Basic Pattern

Every ODE solve in Numra follows the same three steps:

1. **Define the problem** -- specify the right-hand side function, time span,
   and initial conditions
2. **Configure options** -- set tolerances, step size limits, and output
   preferences
3. **Choose a solver and solve** -- call `Solver::solve` with your choice of
   method

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

// Step 1: Define the problem
// dy/dt = -y, y(0) = 1 (exponential decay)
let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0,          // t0
    5.0,          // tf
    vec![1.0],    // y0
);

// Step 2: Configure options
let options = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10);

// Step 3: Solve
let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();
```

## Defining the Right-Hand Side

The RHS function has the signature:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
Fn(S, &[S], &mut [S])
//  t   y     dydt
```

- `t` is the current time
- `y` is the current state (length = system dimension)
- `dydt` is the output derivatives (you must fill this)

For systems with multiple components, each index corresponds to one equation:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeProblem, DoPri5, Solver, SolverOptions};

// Lotka-Volterra predator-prey model:
//   dx/dt = alpha*x - beta*x*y
//   dy/dt = delta*x*y - gamma*y
let alpha = 1.5;
let beta = 1.0;
let delta = 1.0;
let gamma = 3.0;

let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        let prey = y[0];
        let predator = y[1];
        dydt[0] = alpha * prey - beta * prey * predator;
        dydt[1] = delta * prey * predator - gamma * predator;
    },
    0.0,
    30.0,
    vec![10.0, 1.0],  // initial: 10 prey, 1 predator
);

let options = SolverOptions::default();
let result = DoPri5::solve(&problem, 0.0, 30.0, &[10.0, 1.0], &options).unwrap();

let y_final = result.y_final().unwrap();
println!("Final prey:     {:.4}", y_final[0]);
println!("Final predator: {:.4}", y_final[1]);
```

## Working with Results

`SolverResult` stores the entire trajectory:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = DoPri5::solve(&problem, 0.0, 30.0, &[10.0, 1.0], &options).unwrap();

// Number of time steps taken
println!("Steps: {}", result.n_steps());

// Final state
let y_final = result.y_final().unwrap();

// State at step i
let y_i = result.y_at(3);  // returns &[f64]

// Extract a single component across all time steps
let prey_history = result.component(0);      // Vec<f64>
let predator_history = result.component(1);  // Vec<f64>

// Iterate over (time, state) pairs
for (t, y) in result.iter() {
    println!("t = {t:.4}, prey = {:.4}, predator = {:.4}", y[0], y[1]);
}
```

### Solution Storage Format

Solutions are stored in row-major format in a flat `Vec`:

```text
result.y = [y0(t0), y1(t0), y0(t1), y1(t1), y0(t2), y1(t2), ...]
```

To access component `j` at time index `i`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let value = result.y[i * result.dim + j];
```

The `y_at(i)`, `component(j)`, and `iter()` methods handle this indexing for you.

## Solver Statistics

Every solve returns performance statistics:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
println!("Function evaluations: {}", result.stats.n_eval);
println!("Jacobian evaluations: {}", result.stats.n_jac);
println!("Accepted steps:       {}", result.stats.n_accept);
println!("Rejected steps:       {}", result.stats.n_reject);
println!("LU decompositions:    {}", result.stats.n_lu);
```

A high `n_reject` / `n_accept` ratio (> 0.3) suggests the solver is struggling
-- possibly the tolerances are too tight or the wrong solver type was chosen.
A high `n_lu` indicates the problem is being treated as stiff.

## Error Handling

Solvers return `Result<SolverResult, SolverError>`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::SolverError;

match DoPri5::solve(&problem, t0, tf, &y0, &options) {
    Ok(result) => { /* success */ }
    Err(SolverError::MaxSteps { t, steps }) => {
        println!("Hit step limit at t={t}, after {steps} steps");
    }
    Err(SolverError::StepTooSmall { t, h }) => {
        println!("Step size too small at t={t}: h={h:.2e}");
        println!("Hint: the problem may be too stiff for this solver");
    }
    Err(e) => {
        println!("Solver error: {e}");
    }
}
```

## Which Solver Should I Use?

Here's a quick decision guide:

| Situation | Recommended Solver |
|-----------|-------------------|
| Don't know, just try something | `Auto` |
| Non-stiff, moderate accuracy | `DoPri5` or `Tsit5` |
| Non-stiff, high accuracy | `Vern6`, `Vern7`, or `Vern8` |
| Stiff system | `Radau5` or `Esdirk54` |
| Very stiff, moderate accuracy | `Bdf` |
| DAE (differential-algebraic) | `Radau5` or `Bdf` |

The following sections dive deep into each solver family, their trade-offs,
and how to get the best performance from each one.
