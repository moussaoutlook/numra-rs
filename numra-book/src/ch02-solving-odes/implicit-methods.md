# Implicit Methods

When explicit solvers struggle -- taking thousands of tiny steps, or failing
outright with "step size too small" errors -- the problem is almost certainly
**stiff**. Stiff systems contain dynamics on vastly different time scales, and
explicit methods must resolve the fastest scale even when you only care about
the slow one.

Implicit methods solve a nonlinear system at each step, allowing them to take
large stable steps regardless of the stiffness ratio. Numra provides three
families of implicit solvers.

## Stability: A-stable vs L-stable

Before diving into the solvers, two key stability concepts:

**A-stable** means the method's stability region covers the entire left half of
the complex plane. Any decaying mode stays bounded, no matter how fast it
decays. All implicit methods in Numra are at least A-stable.

**L-stable** adds the requirement that the amplification factor goes to zero as
\\[
|R(z)| \to 0 \quad \text{as} \quad z \to -\infty
\\]
This means fast-decaying transients are not just bounded but are damped out
quickly -- exactly what you want for stiff problems. Radau5 and all three ESDIRK methods are
L-stable; BDF orders 1 and 2 are L-stable.

## Radau5 -- Radau IIA (3-stage, order 5)

The gold standard for stiff ODEs and index-1 DAEs. Based on the algorithm in
Hairer and Wanner's *Solving ODEs II*, Radau5 uses the 3-stage Radau IIA
collocation formula with a real-Schur transformation to reduce the coupled
3n-by-3n stage system to one real and one complex n-by-n linear solve per
Newton iteration.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Radau5, Solver, OdeProblem, SolverOptions};

// Robertson chemical kinetics (classic stiff test problem)
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
println!("y1 = {:.6e}, y2 = {:.6e}, y3 = {:.6}", y_final[0], y_final[1], y_final[2]);
println!("LU decompositions: {}", result.stats.n_lu);
```

**Properties:**

| Property | Value |
|----------|-------|
| Order | 5 |
| Stages | 3 (fully implicit) |
| Stability | L-stable |
| DAE support | Index-1 |
| Error estimator | Embedded (Hairer ESTRAD) |
| Newton iteration | Simplified with Jacobian reuse |

**When to use:** Extremely stiff problems, problems with stiffness ratios
above 10^6, index-1 DAEs, and any situation where BDF or ESDIRK methods fail.
This is the most robust stiff solver in Numra.

## ESDIRK -- Singly Diagonally Implicit Runge-Kutta

ESDIRK methods have an explicit first stage and a shared diagonal coefficient
across all implicit stages. This means only **one** LU factorization per step
(reused across all stages), making them cheaper per step than Radau5 at the
cost of lower order for the same stage count.

### Esdirk32 (3-stage, order 2)

A simple, robust starter method. Good for problems where you need L-stability
but not high accuracy.

### Esdirk43 (4-stage, order 3)

The middle ground -- L-stable with reasonable accuracy for moderately stiff
problems.

### Esdirk54 (6-stage, order 4)

The recommended ESDIRK variant. L-stable with 4th-order accuracy.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Esdirk54, Solver, OdeProblem, SolverOptions};

// Van der Pol oscillator with moderate stiffness (mu = 100)
let mu = 100.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 2.0 * mu,
    vec![2.0, 0.0],
);

let options = SolverOptions::default()
    .rtol(1e-6)
    .atol(1e-8);

let result = Esdirk54::solve(
    &problem, 0.0, 2.0 * mu, &[2.0, 0.0], &options
).unwrap();

let y_final = result.y_final().unwrap();
println!("Steps: {}, Rejected: {}", result.stats.n_accept, result.stats.n_reject);
```

**ESDIRK comparison:**

| Method | Order | Stages | Stability | Cost per step |
|--------|-------|--------|-----------|---------------|
| Esdirk32 | 2 | 3 | L-stable | 1 LU + 2 solves |
| Esdirk43 | 3 | 4 | L-stable | 1 LU + 3 solves |
| Esdirk54 | 4 | 6 | L-stable | 1 LU + 5 solves |

**When to use:** Moderately stiff problems (stiffness ratio 10^2 to 10^5)
where you want better efficiency than Radau5. Esdirk54 is the automatic
choice for moderately stiff problems detected by the `Auto` solver.

## BDF -- Backward Differentiation Formulas (orders 1-5)

BDF methods are **multistep** methods: they use solution values from several
previous steps rather than multiple stages within a single step. This makes
each step very cheap (one LU solve), but they need a startup phase and are
sensitive to step size changes.

The k-th order BDF formula is:

\\[
\sum_{j=0}^{k} \alpha_j \, y_{n+1-j} = h \, \beta_k \, f(t_{n+1}, y_{n+1})
\\]

Numra's BDF implementation supports variable order (1 through 5) with
automatic order selection.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Bdf, Solver, OdeProblem, SolverOptions};

// Stiff linear system with widely separated eigenvalues
let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + y[1];
        dydt[1] = -y[1];
    },
    0.0, 10.0,
    vec![1.0, 1.0],
);

let options = SolverOptions::default()
    .rtol(1e-4)
    .atol(1e-6);

let result = Bdf::solve(
    &problem, 0.0, 10.0, &[1.0, 1.0], &options
).unwrap();

let y_final = result.y_final().unwrap();
println!("y = [{:.6e}, {:.6e}]", y_final[0], y_final[1]);
```

**BDF stability by order:**

| Order | Stability | L-stable? | Notes |
|-------|-----------|-----------|-------|
| 1 | A-stable | Yes | Backward Euler (very diffusive) |
| 2 | A-stable | Yes | Most commonly used order |
| 3 | A(86.0)-stable | No | Good balance of accuracy and stability |
| 4 | A(73.4)-stable | No | Higher accuracy, narrower stability |
| 5 | A(51.8)-stable | No | Highest order; avoid for very stiff problems |

BDF6 is intentionally omitted -- it is only A(17.8)-stable and rarely useful
in practice.

**When to use:** Very stiff problems at moderate accuracy (rtol around 1e-3 to
1e-6). BDF is the method behind MATLAB's `ode15s` and SUNDIALS' CVODE. Best
for problems where you want cheap steps and can tolerate the startup overhead.

## Choosing Between Implicit Methods

| Scenario | Recommended |
|----------|-------------|
| Extremely stiff (ratio > 10^6) | `Radau5` |
| Index-1 DAE | `Radau5` or `Bdf` |
| Moderately stiff, good accuracy | `Esdirk54` |
| Stiff, moderate accuracy, many steps | `Bdf` |
| Simple stiff problem, low accuracy | `Esdirk32` |
| Not sure if stiff | Use `Auto` (see next section) |

## How Implicit Methods Handle Stiffness

Consider the test equation \\( y' = \lambda y \\) with \\( \lambda \ll 0 \\).
An explicit method with step size \\( h \\) computes \\( y_{n+1} = R(h\lambda) \, y_n \\)
where \\( R \\) is a polynomial. For stability, \\( |h\lambda| \\) must stay
inside the method's stability region, forcing \\( h < 2/|\lambda| \\) for Euler.

An implicit method like backward Euler computes:

\\[
y_{n+1} = \frac{1}{1 - h\lambda} \, y_n
\\]

The amplification factor \\( 1/(1 - h\lambda) \\) is less than 1 for any
\\( h > 0 \\) when \\( \lambda < 0 \\). No step size restriction from stability
-- only accuracy limits the step size.

The cost is solving a (possibly nonlinear) system at each step, requiring
Jacobian evaluations and LU decompositions. Numra computes Jacobians
automatically via finite differences, so you never need to provide one
manually.
