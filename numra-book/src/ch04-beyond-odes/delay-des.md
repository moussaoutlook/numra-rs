# Delay Differential Equations

Delay differential equations (DDEs) are differential equations where the derivative at
the current time depends on the state at one or more past times. They arise naturally in
biological systems (population dynamics, neural networks), control theory (feedback
loops with transmission delay), and epidemiology (incubation periods). Numra provides the
`numra-dde` crate with a Dormand-Prince-based Method of Steps solver that supports
constant delays, state-dependent delays, and discontinuity tracking.

## Mathematical Formulation

A DDE with \\(k\\) delays takes the form:

\\[
y'(t) = f\!\left(t,\; y(t),\; y(t - \tau_1),\; y(t - \tau_2),\; \ldots,\; y(t - \tau_k)\right)
\\]

where \\(\tau_i > 0\\) are the delays. Unlike an ODE, solving a DDE requires a
**history function** \\(\varphi(t)\\) that specifies the solution for \\(t \leq t_0\\).

### Key Differences from ODEs

| Property | ODE | DDE |
|---|---|---|
| Initial condition | Point value \\(y(t_0)\\) | Function \\(\varphi(t)\\) on \\((-\infty, t_0]\\) |
| Smoothness | As smooth as \\(f\\) | Derivative discontinuities propagate |
| Stability | Determined by eigenvalues | Determined by characteristic equation \\(\det(\lambda I - A - B e^{-\lambda\tau}) = 0\\) |
| Phase space | Finite-dimensional | Infinite-dimensional |
| Chaotic behavior | Requires \\(\geq 3\\) dimensions | Possible in 1D with delay |

### Discontinuity Propagation

A central challenge in DDEs is that discontinuities in the history function propagate
through the solution. If the history \\(\varphi(t)\\) has a jump in its \\(k\\)-th
derivative at \\(t_0\\), then \\(y(t)\\) has a jump in its \\((k+1)\\)-th derivative at
\\(t_0 + \tau\\) for each delay \\(\tau\\). These propagating discontinuities continue
at \\(t_0 + 2\tau\\), \\(t_0 + 3\tau\\), and so on, with each propagation smoothing
by one derivative order.

For accurate numerical integration, the solver should step exactly to these
discontinuity points. Numra's `MethodOfSteps` optionally tracks and steps to
discontinuities when `track_discontinuities` is enabled.

## The DdeSystem Trait

Every DDE in Numra implements the `DdeSystem` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub trait DdeSystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Get the delays tau_i (constant delays).
    fn delays(&self) -> Vec<S>;

    /// Get delays at a specific state (for state-dependent delays).
    /// Default returns constant delays.
    fn delays_at(&self, _t: S, _y: &[S]) -> Vec<S> { self.delays() }

    /// Evaluate the right-hand side f(t, y(t), y(t-tau_1), ...).
    fn rhs(&self, t: S, y: &[S], y_delayed: &[&[S]], dydt: &mut [S]);

    /// Whether delays are state-dependent.
    fn has_state_dependent_delays(&self) -> bool { false }
}
```

The `y_delayed` argument is a slice of slices: `y_delayed[i]` is the state vector
\\(y(t - \tau_i)\\), and `y_delayed[i][j]` is the \\(j\\)-th component of that
delayed state.

## The Method of Steps Solver

Numra's `MethodOfSteps` solver uses an embedded Dormand-Prince 5(4) Runge-Kutta
method with Hermite interpolation for evaluating delayed states between mesh points.

### Algorithm

1. **History initialization**: Store the user-provided history function \\(\varphi(t)\\).
2. **RK stage evaluation**: At each stage of the Runge-Kutta method, evaluate
   \\(y(t - \tau_i)\\) by interpolating the stored solution history using cubic
   Hermite polynomials.
3. **Step acceptance**: Use the embedded 4th-order solution for error estimation.
   Accept or reject the step based on local error vs. tolerances.
4. **History extension**: On acceptance, add the new step (including derivative
   information) to the history for future interpolation.
5. **FSAL**: The 7th stage of DoPri5 is reused as the 1st stage of the next step.

### Discontinuity Tracking

When enabled, the solver pre-computes all discontinuity points up to a specified
propagation order and adjusts step sizes to land exactly on them:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let opts = DdeOptions::default()
    .track_discontinuities(true)
    .discontinuity_order(3);  // Track up to 3 propagation levels
```

For a single delay \\(\tau\\), this tracks discontinuities at
\\(t_0 + \tau\\), \\(t_0 + 2\tau\\), and \\(t_0 + 3\tau\\).

## Solver Options

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_dde::DdeOptions;

let opts = DdeOptions::default()
    .rtol(1e-6)                    // Relative tolerance
    .atol(1e-9)                    // Absolute tolerance
    .h0(0.001)                     // Initial step size (None = auto)
    .h_max(1.0)                    // Maximum step size
    .max_steps(100_000)            // Safety limit
    .track_discontinuities(true)   // Step to discontinuity points
    .discontinuity_order(3);       // Propagation levels to track
```

| Option | Default | Description |
|---|---|---|
| `rtol` | 1e-6 | Relative tolerance |
| `atol` | 1e-9 | Absolute tolerance |
| `h0` | auto | Initial step size |
| `h_max` | infinity | Maximum step size |
| `h_min` | 1e-14 | Minimum step size |
| `max_steps` | 100,000 | Maximum number of steps |
| `dense_output` | true | Enable dense output for history interpolation |
| `track_discontinuities` | false | Enable discontinuity tracking |
| `discontinuity_order` | 0 | Number of propagation levels |

## Example: Mackey-Glass Equation

The Mackey-Glass equation is a classic DDE from hematology that models the production
of blood cells:

\\[
y'(t) = \frac{\beta \, y(t-\tau)}{1 + y(t-\tau)^n} - \gamma \, y(t)
\\]

For \\(\tau > 17\\), the dynamics become chaotic -- one of the first demonstrations that
a single scalar DDE can produce chaos.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_dde::{DdeSystem, MethodOfSteps, DdeSolver, DdeOptions};

struct MackeyGlass {
    beta: f64,
    gamma: f64,
    n: f64,
    tau: f64,
}

impl DdeSystem<f64> for MackeyGlass {
    fn dim(&self) -> usize { 1 }

    fn delays(&self) -> Vec<f64> { vec![self.tau] }

    fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        let y_tau = y_delayed[0][0];  // y(t - tau)
        dydt[0] = self.beta * y_tau / (1.0 + y_tau.powf(self.n)) - self.gamma * y[0];
    }
}

let mg = MackeyGlass { beta: 2.0, gamma: 1.0, n: 9.65, tau: 2.0 };
let history = |_t: f64| vec![0.5];  // Constant history
let opts = DdeOptions::default()
    .rtol(1e-6)
    .atol(1e-9)
    .max_steps(50_000);

let result = MethodOfSteps::solve(&mg, 0.0, 100.0, &history, &opts)
    .expect("Solve failed");

// Solution stays bounded and positive
for (t, y) in result.iter() {
    println!("t = {:.2}, y = {:.6}", t, y[0]);
}
```

## Example: Delayed Feedback Control

A simple model of a controlled system with feedback delay:

\\[
y'(t) = -y(t) + K \cdot y(t - \tau)
\\]

The constant history \\(y(t) = 1\\) for \\(t \leq 0\\) introduces a derivative
discontinuity at \\(t = 0\\) that propagates to \\(t = \tau, 2\tau, \ldots\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_dde::{DdeSystem, MethodOfSteps, DdeSolver, DdeOptions};

struct DelayedFeedback {
    gain: f64,
    tau: f64,
}

impl DdeSystem<f64> for DelayedFeedback {
    fn dim(&self) -> usize { 1 }

    fn delays(&self) -> Vec<f64> { vec![self.tau] }

    fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        dydt[0] = -y[0] + self.gain * y_delayed[0][0];
    }
}

let system = DelayedFeedback { gain: 0.5, tau: 1.0 };
let history = |_t: f64| vec![1.0];

// Enable discontinuity tracking for better accuracy
let opts = DdeOptions::default()
    .rtol(1e-8)
    .atol(1e-10)
    .track_discontinuities(true)
    .discontinuity_order(5);

let result = MethodOfSteps::solve(&system, 0.0, 10.0, &history, &opts)
    .expect("Solve failed");

println!("Discontinuities handled: {}", result.stats.n_discontinuities);
println!("Steps accepted: {}", result.stats.n_accept);
println!("Steps rejected: {}", result.stats.n_reject);
```

### Exact Solution Verification

For the simple equation \\(y'(t) = -y(t-1)\\) with \\(y(t) = 1\\) for \\(t \leq 0\\),
the exact solution on \\([0, 1]\\) is \\(y(t) = 1 - t\\), and on \\([1, 2]\\) it is
\\(y(t) = 1 - t + (t-1)^2/2\\). This piecewise polynomial structure makes it an
excellent test problem.

## Multiple Delays

Systems with multiple delays are straightforward:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_dde::{DdeSystem, MethodOfSteps, DdeSolver, DdeOptions};

struct TwoDelays;

impl DdeSystem<f64> for TwoDelays {
    fn dim(&self) -> usize { 1 }

    fn delays(&self) -> Vec<f64> { vec![0.5, 1.0] }

    fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        // y_delayed[0] = y(t - 0.5), y_delayed[1] = y(t - 1.0)
        dydt[0] = -y[0] + 0.5 * y_delayed[0][0] + 0.3 * y_delayed[1][0];
    }
}

let system = TwoDelays;
let history = |_t: f64| vec![1.0];
let opts = DdeOptions::default();

let result = MethodOfSteps::solve(&system, 0.0, 10.0, &history, &opts)
    .expect("Solve failed");
```

With multiple delays, discontinuity tracking generates points at all combinations:
for delays \\(\tau_1 = 0.5\\) and \\(\tau_2 = 1.0\\), the first-level discontinuities
are at \\(t = 0.5\\) and \\(t = 1.0\\); the second level adds \\(t = 1.0\\) (duplicate,
removed), \\(t = 1.5\\), and \\(t = 2.0\\).

## State-Dependent Delays

Some systems have delays that depend on the current state:

\\[
y'(t) = f\!\big(t,\; y(t),\; y\!\left(t - \tau(t, y(t))\right)\big)
\\]

Implement this by overriding `delays_at` and setting `has_state_dependent_delays`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_dde::{DdeSystem, MethodOfSteps, DdeSolver, DdeOptions};

struct StateDependentDelay;

impl DdeSystem<f64> for StateDependentDelay {
    fn dim(&self) -> usize { 1 }

    fn delays(&self) -> Vec<f64> {
        vec![0.5]  // Nominal delay (used for n_delays())
    }

    fn delays_at(&self, _t: f64, y: &[f64]) -> Vec<f64> {
        // Delay depends on current state
        vec![0.5 + 0.1 * y[0]]
    }

    fn has_state_dependent_delays(&self) -> bool { true }

    fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        dydt[0] = -y_delayed[0][0];
    }
}

let system = StateDependentDelay;
let history = |_t: f64| vec![1.0];
let opts = DdeOptions::default().rtol(1e-6).atol(1e-8);

let result = MethodOfSteps::solve(&system, 0.0, 5.0, &history, &opts)
    .expect("Solve failed");
```

Note that the solver validates delays at every stage evaluation and rejects negative
delays with a clear error message.

## Multi-Dimensional Systems

DDEs with coupled components work naturally:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_dde::{DdeSystem, MethodOfSteps, DdeSolver, DdeOptions};

struct CoupledDDE;

impl DdeSystem<f64> for CoupledDDE {
    fn dim(&self) -> usize { 2 }

    fn delays(&self) -> Vec<f64> { vec![1.0] }

    fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        // Cross-coupling with delay
        dydt[0] = -y[0] + y_delayed[0][1];  // x' depends on delayed y
        dydt[1] = -y[1] + y_delayed[0][0];  // y' depends on delayed x
    }
}

let system = CoupledDDE;
let history = |_t: f64| vec![1.0, 0.5];
let opts = DdeOptions::default();

let result = MethodOfSteps::solve(&system, 0.0, 10.0, &history, &opts)
    .expect("Solve failed");

assert_eq!(result.dim, 2);
```

## Working with Results

`DdeResult` provides the same interface as other Numra result types:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = MethodOfSteps::solve(&system, t0, tf, &history, &opts)?;

// Check success
assert!(result.success);

// Final state
let y_final = result.y_final().unwrap();

// Iterate over solution
for (t, y) in result.iter() {
    println!("t = {:.4}, y = {:?}", t, y);
}

// Access state at a specific index
let y_i = result.y_at(10);

// Statistics
println!("RHS evaluations: {}", result.stats.n_eval);
println!("Accepted steps: {}", result.stats.n_accept);
println!("Rejected steps: {}", result.stats.n_reject);
println!("Discontinuities handled: {}", result.stats.n_discontinuities);
```

## Practical Considerations

### Choosing Tolerances

DDEs are generally harder to integrate than ODEs because of the history dependence.
Start with moderate tolerances and tighten as needed:

- **Exploratory**: `rtol = 1e-4`, `atol = 1e-6`
- **Production**: `rtol = 1e-6`, `atol = 1e-9`
- **High accuracy**: `rtol = 1e-8`, `atol = 1e-10` with discontinuity tracking

### When to Enable Discontinuity Tracking

Enable discontinuity tracking when:
- The history function is not smooth (e.g., constant history meets a different initial slope)
- You need high accuracy near \\(t_0 + k\tau\\)
- The solver is taking unexpectedly many steps

Disable it when:
- There are many delays with large propagation order (combinatorial explosion; capped at 1000 points internally)
- The history is smooth and compatible with the DDE at \\(t_0\\)

### Stiffness

The current `MethodOfSteps` solver uses an explicit Runge-Kutta method, so it is not
suited for stiff DDEs. If you encounter stiffness (very small accepted step sizes,
many rejections), consider reducing the integration interval or using a coarser
tolerance.
