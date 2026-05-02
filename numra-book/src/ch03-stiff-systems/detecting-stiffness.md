# Detecting Stiffness

Before you can choose the right solver, you need to know whether your problem is
stiff. Sometimes this is obvious from the physics (chemical kinetics, circuit
simulation), but often it is not. This section covers both analytical techniques
and practical diagnostic approaches, including Numra's built-in stiffness detection.

## Analytical Detection: Jacobian Eigenvalues

The most direct test is to compute the Jacobian \\( J = \partial f / \partial y \\)
at some representative point and examine its eigenvalues.

For a system \\( \mathbf{y}' = f(t, \mathbf{y}) \\), the Jacobian is the
\\( n \times n \\) matrix:

\\[
J_{ij} = \frac{\partial f_i}{\partial y_j}
\\]

**Stiffness indicators from the eigenvalue spectrum:**

| Criterion | Value | Interpretation |
|-----------|-------|----------------|
| \\( \max \vert \operatorname{Re}(\lambda)\vert / \min \vert \operatorname{Re}(\lambda)\vert \\) | \\( > 10^4 \\) | Very stiff |
| \\( \max \vert \operatorname{Re}(\lambda)\vert / \min \vert \operatorname{Re}(\lambda)\vert \\) | \\( 10^2 - 10^4 \\) | Moderately stiff |
| \\( \max \vert \operatorname{Re}(\lambda)\vert / \min \vert \operatorname{Re}(\lambda)\vert \\) | \\( < 10^2 \\) | Non-stiff |
| All \\( \operatorname{Re}(\lambda_i) < 0 \\) | Required | Stable (stiffness is meaningful) |

**Example:** For the Robertson system at \\( t = 0 \\), \\( y = (1, 0, 0) \\):

\\[
J = \begin{pmatrix}
-0.04 & 0 & 0 \\\\
0.04 & 0 & 0 \\\\
0 & 0 & 0
\end{pmatrix}
\\]

The eigenvalues are \\( \{-0.04, 0, 0\} \\). This does not look stiff at the initial
point! But at \\( t \approx 10^{-5} \\), once \\( y_2 \\) becomes nonzero, the eigenvalues
become \\( \{-3 \times 10^7, -10^4, -0.04\} \\) -- a stiffness ratio of nearly \\( 10^9 \\).

This illustrates that stiffness is often *transient* and depends on the current state.

## Numra's Automatic Detection

Numra's `Auto` solver includes a built-in stiffness detector that examines the
Jacobian at the initial point. The algorithm works by sampling Jacobian elements
via finite differences and computing the ratio of largest to smallest magnitudes.

### The Detection Algorithm

The `Auto::detect_stiffness` function:

1. Evaluates the right-hand side \\( f(t_0, y_0) \\).
2. Perturbs each of the first 5 state components to estimate Jacobian column entries.
3. Computes the ratio of the largest to smallest nonzero Jacobian element magnitudes.
4. Classifies the result:

| Ratio | Classification | Solver selected |
|-------|---------------|-----------------|
| \\( > 10^4 \\) | `VeryStiff` | BDF (standard accuracy) or Radau5 (high accuracy) |
| \\( 10^2 - 10^4 \\) | `ModeratelyStiff` | Esdirk54 |
| \\( < 10^2 \\) | `NonStiff` | Tsit5 (standard), Vern6/8 (high accuracy) |

### Using Auto with Stiffness Detection

The simplest approach is to let `Auto` handle everything:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Auto, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + y[1];
        dydt[1] = y[0] - y[1];
    },
    0.0, 10.0, vec![1.0, 1.0],
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-9);

// Auto detects stiffness and selects an appropriate solver
let result = Auto::solve(&problem, 0.0, 10.0, &[1.0, 1.0], &options)
    .expect("Auto solver should succeed");
```

### Providing Hints

If you know your problem is stiff, you can bypass the detection step and provide
hints directly using `SolverHints`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Auto, Solver, OdeProblem, SolverOptions};
use numra::ode::auto::{SolverHints, Stiffness, Accuracy};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1e6 * y[0];
    },
    0.0, 1.0, vec![1.0],
);

let options = SolverOptions::default().rtol(1e-8).atol(1e-10);

// Tell Auto exactly what kind of problem this is
let hints = SolverHints::new()
    .stiffness(Stiffness::VeryStiff)
    .accuracy(Accuracy::High);

let result = Auto::solve_with_hints(
    &problem, 0.0, 1.0, &[1.0], &options, &hints,
).expect("Should use Radau5 for very stiff + high accuracy");
```

The available stiffness classifications are:

- `Stiffness::NonStiff` -- forces explicit solver selection
- `Stiffness::ModeratelyStiff` -- selects ESDIRK54
- `Stiffness::VeryStiff` -- selects BDF or Radau5 depending on accuracy
- `Stiffness::Unknown` -- enables automatic detection (default)

And the accuracy levels:

- `Accuracy::Low` -- rtol around \\( 10^{-3} \\)
- `Accuracy::Standard` -- rtol around \\( 10^{-6} \\)
- `Accuracy::High` -- rtol around \\( 10^{-10} \\)
- `Accuracy::VeryHigh` -- rtol around \\( 10^{-12} \\) or tighter

## Manual Detection Approaches

### Approach 1: Finite-Difference Jacobian Sampling

You can compute the Jacobian yourself using the same technique Numra uses
internally. The `OdeSystem` trait provides a default `jacobian` method via
finite differences:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeProblem, OdeSystem};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + y[1];
        dydt[1] = y[0] - y[1];
    },
    0.0, 10.0, vec![1.0, 1.0],
);

let dim = 2;
let mut jac = vec![0.0; dim * dim];
problem.jacobian(0.0, &[1.0, 1.0], &mut jac);

// jac is now stored row-major:
// J = [ jac[0]  jac[1] ]   = [ -1000   1 ]
//     [ jac[2]  jac[3] ]     [   1    -1 ]
println!("Jacobian diagonal: {}, {}", jac[0], jac[3]);
// The ratio |jac[0]| / |jac[3]| = 1000 => moderately to very stiff
```

### Approach 2: Trial Run with an Explicit Solver

Run DoPri5 or Tsit5 on a short integration interval. If the solver:

- Takes an unexpectedly large number of steps, or
- Has a high rejection ratio (\\( n_{\text{reject}} / n_{\text{accept}} > 0.5 \\)), or
- Fails outright with step size collapse,

then the problem is likely stiff.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Tsit5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + y[1];
        dydt[1] = y[0] - y[1];
    },
    0.0, 1.0, vec![1.0, 1.0],
);

let options = SolverOptions::default()
    .rtol(1e-6)
    .atol(1e-9)
    .max_steps(10000);

match Tsit5::solve(&problem, 0.0, 1.0, &[1.0, 1.0], &options) {
    Ok(result) => {
        let ratio = result.stats.n_reject as f64
                  / result.stats.n_accept.max(1) as f64;
        if ratio > 0.5 || result.stats.n_accept > 5000 {
            println!("Likely stiff! Reject ratio: {:.2}, steps: {}",
                     ratio, result.stats.n_accept);
        }
    }
    Err(_) => {
        println!("Explicit solver failed -- almost certainly stiff!");
    }
}
```

This is essentially what `Auto` does in `Stiffness::Unknown` mode: it tries Tsit5
first, checks whether the rejection ratio is reasonable, and falls back to Esdirk54
if it is not.

### Approach 3: Physical Reasoning

Many problem domains are known to produce stiff systems:

| Domain | Why stiff | Typical stiffness ratio |
|--------|-----------|------------------------|
| Chemical kinetics | Fast/slow reactions | \\( 10^3 - 10^{12} \\) |
| Electrical circuits | Parasitic capacitances | \\( 10^3 - 10^9 \\) |
| Control systems | Fast actuators + slow plant | \\( 10^2 - 10^6 \\) |
| Structural mechanics | High-frequency vibrations | \\( 10^3 - 10^8 \\) |
| Combustion | Radical chemistry | \\( 10^6 - 10^{15} \\) |
| Semiconductor devices | Carrier dynamics | \\( 10^6 - 10^{12} \\) |

If your problem comes from one of these domains, start with an implicit solver.

## Diagnostic Signs During Integration

Even without advance analysis, certain runtime behaviors are strong indicators
of stiffness:

### Step Size Collapse

The solver repeatedly reduces the step size until it hits the minimum. For explicit
solvers, this is the classic symptom: the step controller demands smaller and
smaller \\( h \\) to satisfy stability, not accuracy.

### High Rejection Ratio

A healthy explicit integration has a rejection ratio below 10%. If you see 50%+
rejections, stability constraints are dominating. Check `result.stats.n_reject`
and `result.stats.n_accept`.

### Excessive Function Evaluations

For a 5th-order method integrating over \\( [0, T] \\), you might expect roughly
\\( T / h_{\text{accuracy}} \\) steps. If the actual step count is orders of magnitude
higher, stiffness is the likely culprit.

### Oscillating Step Size

The step controller alternates between large and small steps, unable to find a
stable rhythm. This often happens when the problem transitions between stiff and
non-stiff regions.

## Summary: Detection Decision Tree

```text
Is the problem from a known stiff domain?
  |
  +-- Yes --> Use an implicit solver (Radau5, BDF, ESDIRK)
  |
  +-- No / Unsure
        |
        Can you compute the Jacobian eigenvalues?
          |
          +-- Yes --> Stiffness ratio > 100? --> Stiff
          |                                 --> Non-stiff
          |
          +-- No --> Use Auto (it detects stiffness automatically)
                     or try an explicit solver first and check:
                       - Rejection ratio > 50%? --> Stiff
                       - Step count >> expected? --> Stiff
                       - Solver fails entirely?  --> Stiff
```

When in doubt, use `Auto` -- it adds negligible overhead for non-stiff problems
and automatically switches to an appropriate implicit method when stiffness is
detected.
