# Performance Tuning

This chapter provides practical guidance for getting the most out of Numra's
solvers. The right configuration can mean orders-of-magnitude speedups.

## Tolerance Selection

The most impactful performance knob is the tolerance pair `(rtol, atol)`.

**Rule of thumb:** set `rtol` to the number of correct digits you need, and
`atol` to a value below the smallest meaningful solution magnitude.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::SolverOptions;

// "Engineering" accuracy: 6 digits
let opts = SolverOptions::default().rtol(1e-6).atol(1e-9);

// "Scientific" accuracy: 10 digits
let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);

// When solution passes through zero, atol dominates:
// If y ~ 1e-8 and atol = 1e-9, you still get 1 correct digit
let opts = SolverOptions::default().rtol(1e-6).atol(1e-14);
```

| Tolerance | Typical use | Relative cost |
|-----------|-------------|---------------|
| rtol = 1e-3 | Quick exploration | 1x |
| rtol = 1e-6 | Engineering | 2-5x |
| rtol = 1e-8 | Publication quality | 5-15x |
| rtol = 1e-10 | Reference solutions | 15-50x |
| rtol = 1e-12 | Verification | 50-200x |

Tighter tolerance means smaller steps means more RHS calls.
The cost scales roughly as rtol to the power of -1/p where p is the method order.

## Solver Selection

Choosing the right solver is the second most impactful decision.

### Decision tree

1. **Is the problem stiff?**
   - No: Use an explicit method (DoPri5, Tsit5, Vern6/7/8)
   - Yes: Use an implicit method (Radau5, ESDIRK, BDF)
   - Not sure: Use `Auto`

2. **What accuracy do you need?**
   - Moderate (rtol ~ 1e-6): DoPri5 or ESDIRK54
   - High (rtol ~ 1e-10): Vern6 or Vern7
   - Very high (rtol ~ 1e-12): Vern8

3. **Is it a long integration?**
   - Energy conservation matters: Tighten tolerances
   - Many RHS calls: Consider FSAL methods (Tsit5)

### Cost comparison

| Solver | Order | Stages | RHS calls/step | Best for |
|--------|-------|--------|----------------|----------|
| Tsit5 | 5(4) | 7 (FSAL: 6) | 6 | General non-stiff |
| DoPri5 | 5(4) | 7 | 6-7 | General non-stiff |
| Vern6 | 6(5) | 9 | 8-9 | Medium-high accuracy |
| Vern7 | 7(6) | 10 | 10 | High accuracy |
| Vern8 | 8(7) | 13 | 13 | Very high accuracy |
| ESDIRK32 | 2(1) | 3 | 3 + LU | Mild stiffness |
| ESDIRK43 | 3(2) | 4 | 4 + LU | Moderate stiffness |
| ESDIRK54 | 4(3) | 6 | 6 + LU | Stiff, high accuracy |
| Radau5 | 5 | 3 (implicit) | 3*n + LU | Severe stiffness, DAEs |
| BDF | 1-5 | multistep | 1 + LU | Severe stiffness |

## Reducing RHS Calls

The right-hand-side function is called thousands to millions of times. Make
it fast:

### Avoid allocations

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// BAD: allocates every call
let problem = OdeProblem::new(
    |t, y: &[f64], dydt: &mut [f64]| {
        let temp = vec![0.0; y.len()];  // allocation!
        // ...
    },
    0.0, 10.0, vec![1.0],
);

// GOOD: pre-allocate outside, or compute in-place
let problem = OdeProblem::new(
    |t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];  // no allocation
    },
    0.0, 10.0, vec![1.0],
);
```

### Pre-compute constants

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// BAD: recomputes every call
let problem = OdeProblem::new(
    |t, y: &[f64], dydt: &mut [f64]| {
        let omega = (2.0 * std::f64::consts::PI * 60.0);  // recomputed!
        dydt[0] = omega.cos() * y[0];
    },
    0.0, 10.0, vec![1.0],
);

// GOOD: capture pre-computed value
let omega = 2.0 * std::f64::consts::PI * 60.0;
let problem = OdeProblem::new(
    move |t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = omega.cos() * y[0];
    },
    0.0, 10.0, vec![1.0],
);
```

### Provide analytical Jacobians

For implicit methods, the Jacobian J = df/dy is needed.
The default finite-difference approximation requires n extra RHS calls
per Jacobian (where n is the system dimension). Providing an analytical
Jacobian avoids this cost:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeSystem, Radau5, Solver, SolverOptions};

struct Robertson;

impl OdeSystem<f64> for Robertson {
    fn dim(&self) -> usize { 3 }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    }

    fn jacobian(&self, _t: f64, y: &[f64], jac: &mut [f64]) {
        // Row-major 3x3 Jacobian
        jac[0] = -0.04;  jac[1] = 1e4 * y[2];    jac[2] = 1e4 * y[1];
        jac[3] =  0.04;  jac[4] = -1e4*y[2] - 6e7*y[1]; jac[5] = -1e4 * y[1];
        jac[6] =  0.0;   jac[7] = 6e7 * y[1];    jac[8] = 0.0;
    }
}
```

## Diagnosing Performance Issues

### Check solver statistics

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = DoPri5::solve(&problem, 0.0, 100.0, &[1.0], &opts).unwrap();
let s = &result.stats;

println!("Accepted steps: {}", s.n_accept);
println!("Rejected steps: {}", s.n_reject);
println!("Jacobian computations: {}", s.n_jac);
println!("LU decompositions: {}", s.n_lu);
println!("Reject ratio: {:.1}%", 100.0 * s.n_reject as f64 / s.n_accept as f64);
```

### Warning signs

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| High reject ratio (> 30%) | Tolerance too tight for method order | Use higher-order method or loosen tolerance |
| Millions of steps | Wrong solver class for problem | Switch explicit to implicit |
| Many LU decompositions | Rapidly changing Jacobian | Tighten Newton tolerance or use ESDIRK |
| max_steps exceeded | Extremely stiff or singular problem | Increase max_steps, check problem formulation |
| Slow individual steps | Expensive RHS function | Profile and optimize RHS, provide Jacobian |

### Stiffness detection

If an explicit solver takes far more steps than expected, the problem may be
stiff. Try `Auto` which detects stiffness automatically:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Auto, Solver};

let result = Auto::solve(&problem, 0.0, 100.0, &y0, &opts).unwrap();
// Auto starts with Tsit5 and switches to Radau5 if stiffness is detected
```

## Step Size Control

### Initial step size

The solver estimates an initial step size automatically. Override for problems
where the auto-estimate is poor:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Force a small initial step for problems with fast initial transients
let opts = SolverOptions::default().h0(1e-6);
```

### Maximum step size

Limit the step size to capture features the error estimator might miss:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Ensure at least 100 steps over the interval
let opts = SolverOptions::default().h_max(0.1);  // for [0, 10]
```

## Memory Usage

### Dense output

Enabling dense output stores interpolation coefficients at every step, which
can use significant memory for long integrations:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// If you only need the final state, don't enable dense output:
let opts = SolverOptions::default();  // dense_output = false by default

// If you need output at specific times, use the t_eval option instead of dense:
let times = vec![1.0, 2.0, 5.0, 10.0];
let opts = SolverOptions::default().t_eval(times);
```

### Solution storage

The `SolverResult` stores the full trajectory. For very large systems
(n > 10000) over many time steps, this can be gigabytes. Use the t_eval option
to limit saved output, or extract components:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Only save output at 100 points instead of every adaptive step
let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
let opts = SolverOptions::default().t_eval(times);
```

## Quick Reference

| Goal | Configuration |
|------|--------------|
| Fastest solve | Loose tolerance + low-order method |
| Most accurate | Tight tolerance + Vern8 |
| Stiff problems | Radau5 or BDF |
| Minimize memory | Use t_eval option, avoid `.dense()` |
| Debug convergence | Check `result.stats` fields |
| Long integrations | Set `max_steps` and `h_max` |
