---
title: "Solver Options & Configuration"
---

All ODE solvers share a common `SolverOptions` struct that controls tolerances,
step size, output behavior, and event detection.

## Tolerances

The most important settings are the relative and absolute tolerances. The solver
controls the local error at each step to satisfy:

$$
|e_i| \leq \text{atol} + \text{rtol} \cdot |y_i|
$$

This is assessed per-component using a weighted RMS norm.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let options = SolverOptions::default()
    .rtol(1e-8)    // relative tolerance (default: 1e-6)
    .atol(1e-10);  // absolute tolerance (default: 1e-9)
```

**Guidelines for choosing tolerances:**

| Accuracy needed | rtol | atol | Use case |
|----------------|------|------|----------|
| Quick survey | 1e-3 | 1e-6 | Exploring parameter space |
| Standard | 1e-6 | 1e-9 | Most engineering applications |
| High accuracy | 1e-8 | 1e-11 | Reference solutions, validation |
| Very high | 1e-10 | 1e-13 | Publication-quality, celestial mechanics |
| Near machine eps | 1e-13 | 1e-15 | Theoretical studies (use Vern7/Vern8) |

The absolute tolerance matters when solution components pass through zero. If
`atol` is too large, the solver won't control error on small-valued components.
If your components have very different scales, consider scaling your equations
so all components are O(1).

## Step Size Control

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let options = SolverOptions::default()
    .h0(1e-3)          // initial step size (default: auto-detected)
    .h_max(0.1)        // maximum step size (default: infinity)
    .h_min(1e-14)      // minimum step size (default: 1e-14)
    .max_steps(50000);  // step limit (default: 100,000)
```

### Initial Step Size (`h0`)

By default, the solver automatically selects a starting step size based on the
problem's derivatives. Setting `h0` explicitly is useful when:

- You know the characteristic time scale of your problem
- The automatic selection is too conservative or too aggressive
- You want reproducible step sequences for testing

### Maximum Step Size (`h_max`)

Limits how large a single step can be. Useful when:

- The solution has features at known time scales that might be "stepped over"
- You need sufficient time resolution in the output
- The problem has discontinuous forcing at known times

### Minimum Step Size (`h_min`)

If the step size drops below `h_min`, the solver returns a `StepTooSmall` error.
This is a safety net for detecting problems that are too stiff for the chosen
solver.

### Step Limit (`max_steps`)

Prevents runaway integration. If the solver exceeds this count, it returns a
`MaxSteps` error. Increase this for very long integrations or very tight
tolerances.

## Output Control

### Save at Specific Times (`t_eval`)

By default, the solver saves the solution at every internal step. Use `t_eval`
to get output at specific times instead:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let t_output: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
let options = SolverOptions::default()
    .t_eval(t_output);
```

The solver uses interpolation to provide the solution at the requested times
without reducing step sizes.

### Dense Output

Enable dense output for continuous interpolation between steps:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let options = SolverOptions::default().dense();
```

When enabled, the solver stores interpolation coefficients at each step. This
allows computing the solution at any time in the integration interval after
the solve completes. Currently supported by DoPri5 (4th-order polynomial
interpolant).

## Complete Example

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -0.5 * y[0];  // slow exponential decay
    },
    0.0, 100.0, vec![1.0],
);

let options = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10)
    .h_max(1.0)        // don't skip large regions
    .max_steps(10000);  // plenty of budget

let result = DoPri5::solve(&problem, 0.0, 100.0, &[1.0], &options).unwrap();

println!("Solved in {} steps", result.stats.n_accept);
println!("Final value: {:.6e}", result.y_final().unwrap()[0]);
```

## Default Values Reference

| Option | Default | Description |
|--------|---------|-------------|
| `rtol` | 1e-6 | Relative tolerance |
| `atol` | 1e-9 | Absolute tolerance |
| `h0` | None (auto) | Initial step size |
| `h_max` | infinity | Maximum step size |
| `h_min` | 1e-14 | Minimum step size |
| `max_steps` | 100,000 | Maximum number of steps |
| `t_eval` | None (all steps) | Output time points |
| `dense_output` | false | Enable dense interpolation |
| `events` | empty | Event functions |
