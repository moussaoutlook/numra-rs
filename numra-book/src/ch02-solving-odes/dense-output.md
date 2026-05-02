# Dense Output & Interpolation

By default, an ODE solver saves the solution only at the time points it
naturally steps through. These step points are chosen for accuracy and
efficiency, not for your convenience -- they are unevenly spaced and may skip
over times you care about.

**Dense output** solves this by constructing a polynomial interpolant within
each step, allowing computation of the solution at any time without
re-integrating.

## Why Dense Output Matters

Without dense output, if you want the solution at \\( t = 1.5 \\) but the
solver stepped from \\( t = 1.2 \\) to \\( t = 1.8 \\), you get no value at
1.5. You would have to either:

- Accept only the solver's natural step points, or
- Re-run the integration with a smaller maximum step size (wasteful)

Dense output gives you a continuous representation of the solution, which is
essential for:

- **Output at specific times** -- saving the solution on a uniform grid
- **Event detection** -- precisely locating zero-crossings between steps
- **Visualization** -- smooth trajectories for plotting
- **Coupling** -- providing the solution to other systems at arbitrary times

## Enabling Dense Output

Use the `.dense()` builder method on `SolverOptions`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0, 5.0,
    vec![1.0],
);

let options = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10)
    .dense();  // enable dense output storage

let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();
```

With `.dense()` enabled, the solver stores interpolation coefficients for every
accepted step. This increases memory usage but enables continuous querying of
the solution.

## The t_eval Mechanism

The most common use case is obtaining the solution at specific times. Use
`t_eval` to request output at exact time points:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];
    },
    0.0, 5.0,
    vec![1.0],
);

// Get solution at exactly these times
let t_points: Vec<f64> = (0..=50).map(|i| i as f64 * 0.1).collect();

let options = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10)
    .t_eval(t_points);

let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

// result.t now contains exactly [0.0, 0.1, 0.2, ..., 5.0]
// result.y contains the interpolated solution at each of these points
for (t, y) in result.iter() {
    let exact = (-t).exp();
    let error = (y[0] - exact).abs();
    // error should be well within tolerance
}
```

When `t_eval` is set, the solver internally enables dense output, interpolates
at each requested time, and stores only those points in the result. The solver
still takes its own adaptive steps for accuracy -- it just reports at your
requested times.

## DoPri5's 4th-Order Interpolant

DoPri5 provides a free dense output interpolant built from its 7 stage
derivatives. Given a step from \\( t_n \\) to \\( t_n + h \\), the interpolant
computes the solution at \\( t_n + \theta h \\) for \\( \theta \in [0, 1] \\):

\\[
y(t_n + \theta h) = y_n + h \theta \left[ d_0 + \theta \left( d_1 + \theta \left( d_2 + \theta \left( d_3 + \theta \, d_4 \right)\right)\right)\right]
\\]

The coefficients \\( d_0, \ldots, d_4 \\) are computed from the stage values
\\( k_1, k_3, k_4, k_5, k_7 \\) using the formulas from Hairer, Norsett, and
Wanner (*Solving ODEs I*, Section II.6). The interpolant is 4th-order
accurate -- one less than the method order of 5.

**Key properties of the DoPri5 interpolant:**

- Exact at endpoints: \\( y(\theta=0) = y_n \\), \\( y(\theta=1) = y_{n+1} \\)
- No additional function evaluations required (reuses existing stage values)
- 4th-order accuracy at interior points
- Smooth derivatives at step boundaries

### How the Interpolant is Structured

Internally, Numra represents dense output as a sequence of `DenseSegment`
values, each covering one integration step:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub struct DenseSegment<S: Scalar> {
    pub t_start: S,       // step start time
    pub t_end: S,         // step end time
    pub coeffs: Vec<S>,   // interpolation coefficients
    pub dim: usize,       // system dimension
}
```

For DoPri5, the `coeffs` vector stores 6 groups of `dim` values: the initial
state \\( y_n \\) followed by the 5 polynomial coefficients \\( d_0 \\)
through \\( d_4 \\) for each component.

A `DenseOutput` container holds all segments and provides binary search to
find the correct segment for any given time:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub struct DenseOutput<S: Scalar> {
    segments: Vec<DenseSegment<S>>,
    dim: usize,
    direction: S,  // +1 for forward, -1 for backward integration
}
```

## Dense Output and Events

Dense output is automatically enabled when event functions are registered (see
[Event Detection](./event-detection.md)). The solver uses the interpolant to
check the event function at intermediate times during bisection, achieving
precise zero-crossing location without additional RHS evaluations.

## Dense Output vs t_eval

| Feature | `.dense()` | `.t_eval(...)` |
|---------|-----------|----------------|
| **Memory** | O(steps x dim x 6) | O(n_output x dim) |
| **Query time** | Any t after solve | Only at specified times |
| **Flexibility** | Query at arbitrary times later | Must specify times upfront |
| **Use case** | Interactive exploration, event detection | Known output times |

**Recommendation:** Use `t_eval` when you know the output times in advance.
Use `.dense()` when you need the flexibility to query at unknown times after
solving (for example, adaptive plotting or multi-resolution analysis).

## Which Solvers Support Dense Output?

| Solver | Dense Output | Interpolant Order |
|--------|-------------|-------------------|
| DoPri5 | Yes | 4th order polynomial |
| Tsit5 | Step endpoints only | -- |
| Vern6/7/8 | Step endpoints only | -- |
| Radau5 | Step endpoints only | -- |
| ESDIRK | Step endpoints only | -- |
| BDF | Step endpoints only | -- |

Currently, DoPri5 is the only solver with a built-in continuous interpolant.
For other solvers, `t_eval` uses linear interpolation between step endpoints,
which may reduce accuracy for output points that fall between natural steps.

## Performance Considerations

Dense output coefficients are computed from the stage values during the step,
adding negligible computational cost. The main cost is **memory**: storing 6
values per dimension per step. For a 3D system with 1000 steps, this is
18,000 floating-point values -- trivial. For a 1000D system with 100,000
steps, this is 600 million values -- plan accordingly.

If you only need output at specific times, prefer `t_eval` over `.dense()`.
With `t_eval`, the solver interpolates on the fly and discards the
coefficients after use, keeping memory constant.
