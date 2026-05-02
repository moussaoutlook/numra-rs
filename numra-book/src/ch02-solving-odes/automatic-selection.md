# Automatic Solver Selection

Choosing the right solver requires knowing whether your problem is stiff, how
much accuracy you need, and sometimes trial and error. The `Auto` solver
handles this decision for you by analyzing the problem before (and during)
integration.

## Basic Usage

The simplest way to solve any ODE without thinking about solver choice:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Auto, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = -y[0];  // harmonic oscillator
    },
    0.0, 50.0,
    vec![1.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
let result = Auto::solve(&problem, 0.0, 50.0, &[1.0, 0.0], &options).unwrap();

let y_final = result.y_final().unwrap();
println!("cos(50) approx: {:.10}", y_final[0]);
```

There is also a convenience function:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{auto_solve, OdeProblem, SolverOptions};

let result = auto_solve(&problem, 0.0, 50.0, &[1.0, 0.0], &options).unwrap();
```

## How Stiffness Detection Works

Before selecting a solver, `Auto` evaluates the Jacobian at the initial point
using finite differences and computes a **stiffness ratio**: the ratio of the
largest to smallest absolute Jacobian element magnitudes sampled from up to 5
state components.

The classification thresholds are:

| Jacobian Ratio | Classification | Meaning |
|----------------|----------------|---------|
| < 100 | `NonStiff` | Time scales are comparable |
| 100 -- 10,000 | `ModeratelyStiff` | Some scale separation |
| > 10,000 | `VeryStiff` | Large scale separation |

If the maximum Jacobian element is below \\( 10^{-10} \\), the problem is
classified as `NonStiff` regardless of the ratio (nearly-zero dynamics).

## Selection Logic

`Auto` uses a two-dimensional decision table based on stiffness and accuracy:

| Stiffness / Accuracy | Low (rtol >= 1e-3) | Standard (1e-7 to 1e-3) | High (1e-11 to 1e-7) | Very High (< 1e-11) |
|----------------------|-------|----------|------|-----------|
| **NonStiff** | Tsit5 | Tsit5 | Vern6 | Vern8 |
| **ModeratelyStiff** | Esdirk54 | Esdirk54 | Esdirk54 | Esdirk54 |
| **VeryStiff** | Bdf | Bdf | Radau5 | Radau5 |

When stiffness is unknown (no hints, detection disabled, or ambiguous), `Auto`
uses a **try-and-fallback** strategy:

1. Attempt integration with Tsit5 (cheap, explicit)
2. If Tsit5 succeeds and the rejection ratio is acceptable (rejected < accepted), return the result
3. Otherwise, fall back to Esdirk54

## Providing Hints

If you know something about your problem, you can guide the selection with
`SolverHints`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{
    Auto, OdeProblem, SolverOptions, SolverHints, Stiffness, Accuracy,
    auto_solve_with_hints,
};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + y[1];
        dydt[1] = -y[1];
    },
    0.0, 10.0,
    vec![1.0, 1.0],
);

let options = SolverOptions::default().rtol(1e-4);

// Tell Auto the problem is very stiff -- skip detection
let hints = SolverHints::new()
    .stiffness(Stiffness::VeryStiff)
    .accuracy(Accuracy::Standard);

let result = auto_solve_with_hints(
    &problem, 0.0, 10.0, &[1.0, 1.0], &options, &hints
).unwrap();

let y_final = result.y_final().unwrap();
println!("y = [{:.6e}, {:.6e}]", y_final[0], y_final[1]);
```

### Available Hints

**Stiffness:**
- `Stiffness::NonStiff` -- skip detection, use explicit methods
- `Stiffness::ModeratelyStiff` -- use ESDIRK family
- `Stiffness::VeryStiff` -- use BDF or Radau5

**Accuracy:**
- `Accuracy::Low` -- rtol around 1e-3
- `Accuracy::Standard` -- rtol around 1e-6
- `Accuracy::High` -- rtol around 1e-10
- `Accuracy::VeryHigh` -- rtol around 1e-12 or tighter

**Other options:**
- `.implicit()` -- prefer implicit methods even for non-stiff problems (useful for conservation properties)
- `.detect_stiffness(false)` -- disable automatic detection

## Auto vs Manual Selection

The main trade-off:

| Aspect | Auto | Manual |
|--------|------|--------|
| Convenience | Just works | Must understand problem |
| Overhead | Jacobian evaluation at t0 | None |
| Optimality | Good default | Can be tuned exactly |
| Fallback cost | May waste steps on failed attempt | None |

For production code on a well-understood problem, picking the solver directly
is marginally more efficient. For exploratory work, prototyping, or when you
are unsure about stiffness, `Auto` is the right choice.

## Example: Auto Detects Stiffness

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Auto, Solver, OdeProblem, SolverOptions};

// Non-stiff: simple pendulum
let non_stiff = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = -y[0].sin();
    },
    0.0, 20.0, vec![0.5, 0.0],
);

// Stiff: fast/slow reaction
let stiff = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + 0.01 * y[1];
        dydt[1] = 0.01 * y[0] - y[1];
    },
    0.0, 1.0, vec![1.0, 1.0],
);

let opts = SolverOptions::default();

// Auto will pick Tsit5 for the pendulum (non-stiff)
let r1 = Auto::solve(&non_stiff, 0.0, 20.0, &[0.5, 0.0], &opts).unwrap();
println!("Pendulum: {} steps, {} LU decomps", r1.stats.n_accept, r1.stats.n_lu);

// Auto will pick an implicit method for the reaction system (stiff)
let r2 = Auto::solve(&stiff, 0.0, 1.0, &[1.0, 1.0], &opts).unwrap();
println!("Reaction: {} steps, {} LU decomps", r2.stats.n_accept, r2.stats.n_lu);
```

The pendulum solve will show zero LU decompositions (explicit method selected),
while the reaction system will show nonzero LU decompositions (implicit method
selected).
