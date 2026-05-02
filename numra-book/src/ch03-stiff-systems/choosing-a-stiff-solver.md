# Choosing a Stiff Solver

Numra provides three families of implicit solvers for stiff problems. Each has
different strengths, and choosing the right one can mean the difference between
a computation that takes seconds and one that takes hours (or fails entirely).

## The Three Families

### Radau5 -- Implicit Runge-Kutta (Radau IIA)

- **Order:** 5 (3-stage)
- **Stability:** L-stable
- **DAE support:** Index-1
- **Best for:** High-accuracy solutions of very stiff problems, DAEs

Radau5 is a fully implicit 3-stage Runge-Kutta method based on the Radau IIA
quadrature nodes. It uses a real-Schur transformation to decouple the \\( 3n \times 3n \\)
stage system into one \\( n \times n \\) real and one \\( 2n \times 2n \\) real-equivalent
complex system, reducing the per-step cost from \\( O((3n)^3) \\) to \\( O(n^3) \\).

The method is *stiffly accurate* (the last stage coincides with the step endpoint),
which is essential for DAE support and gives excellent error control on stiff
components.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Radau5, Solver, OdeProblem, SolverOptions};

let mu = 100.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 20.0, vec![2.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
let result = Radau5::solve(&problem, 0.0, 20.0, &[2.0, 0.0], &options)
    .expect("Radau5 handles stiff Van der Pol");
```

### ESDIRK -- Singly Diagonally Implicit Runge-Kutta

- **Variants:** Esdirk32 (order 2), Esdirk43 (order 3), Esdirk54 (order 4)
- **Stability:** L-stable (all three variants)
- **Best for:** Moderately stiff problems, problems requiring frequent Jacobian reuse

ESDIRK methods have a special structure: the first stage is explicit, and all
implicit stages share the same diagonal coefficient \\( \gamma \\). This means you
only need *one* LU factorization per step (not per stage), since the iteration
matrix \\( (I - h\gamma J) \\) is the same for every implicit stage.

| Method | Stages | Order | Embedded | Diagonal \\( \gamma \\) |
|--------|--------|-------|----------|----------------------|
| Esdirk32 | 3 | 2 | 1st order | 0.2929 |
| Esdirk43 | 4 | 3 | 2nd order | 0.4359 |
| Esdirk54 | 6 | 4 | 3rd order | 0.25 |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Esdirk54, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -100.0 * y[0] + y[1];
        dydt[1] = y[0] - y[1];
    },
    0.0, 10.0, vec![1.0, 1.0],
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = Esdirk54::solve(&problem, 0.0, 10.0, &[1.0, 1.0], &options)
    .expect("ESDIRK54 is great for moderate stiffness");
```

### BDF -- Backward Differentiation Formulas

- **Orders:** 1-5 (variable order)
- **Stability:** A-stable (orders 1-2), A(\\(\alpha\\))-stable (orders 3-5)
- **DAE support:** Index-1
- **Best for:** Long integrations of very stiff problems, problems where Jacobian evaluation is expensive

BDF methods are *multistep* methods: they use information from previous time steps
rather than multiple stages within a single step. This makes each step cheaper
(only one nonlinear solve and one LU factorization), but requires a startup phase
and is sensitive to step size changes.

The variable-order capability (BDF1 through BDF5) allows the method to adapt:
low order near discontinuities or rapid changes, high order during smooth phases.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Bdf, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    },
    0.0, 1e11, vec![1.0, 0.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-4).atol(1e-8);
let result = Bdf::solve(&problem, 0.0, 1e11, &[1.0, 0.0, 0.0], &options)
    .expect("BDF handles long-time stiff integration");
```

## Cost Analysis

The dominant cost for all implicit methods is the LU factorization of the iteration
matrix. Here is how the methods compare per step:

| Method | LU factorizations | Linear solves | Function evaluations | Memory |
|--------|-------------------|---------------|---------------------|---------|
| Radau5 | 2 (one \\(n \times n\\), one \\(2n \times 2n\\)) | \\( 3{-}7 \\) Newton iterations \\(\times\\) 2 systems | \\( 3 \times \text{Newton iters} \\) | \\( O(n^2) \\) |
| ESDIRK | 1 (\\(n \times n\\)) | \\( s{-}1 \\) stages \\(\times\\) Newton iters | \\( s \times \text{Newton iters} \\) | \\( O(n^2) \\) |
| BDF | 1 (\\(n \times n\\)) | Newton iters | Newton iters | \\( O(kn) \\) history |

where \\( s \\) is the number of stages and \\( k \\) is the BDF order.

**Key insight:** For small systems (\\( n < 100 \\)), the LU cost is negligible and
Radau5's higher order pays off with fewer steps. For large systems (\\( n > 1000 \\)),
LU is the bottleneck and BDF's single \\( n \times n \\) factorization per step wins.

## Stability Properties

### A-Stability

A method is A-stable if its stability region contains the entire left half of the
complex plane. This means it can handle any stable eigenvalue at any step size.

- **A-stable:** BDF1, BDF2, all ESDIRK, Radau5
- **Not A-stable:** BDF3, BDF4, BDF5 (A(\\(\alpha\\))-stable with decreasing angle)

### L-Stability

L-stability adds the requirement that the amplification factor goes to zero as
\\( h\lambda \to -\infty \\). This ensures that very fast, stiff modes are *completely*
damped in a single step, rather than just bounded.

- **L-stable:** Radau5, Esdirk32, Esdirk43, Esdirk54, BDF1
- **A-stable but not L-stable:** BDF2

For extremely stiff problems (stiffness ratio \\( > 10^6 \\)), L-stability is
strongly preferred. The non-L-stable methods can introduce spurious oscillations
on the fast modes.

### Stability Region Comparison

\\[
\text{Stability function } R(z) = \frac{P(z)}{Q(z)} \quad \text{where } z = h\lambda
\\]

| Method | \\( \vert R(\infty)\vert  \\) | A-stable? | L-stable? | Stability angle |
|--------|------------------|-----------|-----------|----------------|
| Radau5 | 0 | Yes | Yes | \\( 90^\circ \\) |
| Esdirk54 | 0 | Yes | Yes | \\( 90^\circ \\) |
| Esdirk43 | 0 | Yes | Yes | \\( 90^\circ \\) |
| Esdirk32 | 0 | Yes | Yes | \\( 90^\circ \\) |
| BDF1 | 0 | Yes | Yes | \\( 90^\circ \\) |
| BDF2 | \\( \neq 0 \\) | Yes | No | \\( 90^\circ \\) |
| BDF3 | -- | No | No | \\( 86^\circ \\) |
| BDF4 | -- | No | No | \\( 73^\circ \\) |
| BDF5 | -- | No | No | \\( 51^\circ \\) |

The decreasing stability angle of higher-order BDF methods means they cannot handle
eigenvalues with large imaginary parts. Problems with oscillatory stiff modes
(e.g., vibrational systems) should prefer Radau5 or ESDIRK.

## Decision Guide

### By Problem Characteristics

| Problem type | Recommended solver | Why |
|-------------|-------------------|-----|
| Moderately stiff (\\( 10^2 - 10^4 \\)) | Esdirk54 | Efficient, L-stable, one LU per step |
| Very stiff, high accuracy | Radau5 | 5th order, L-stable, excellent error control |
| Very stiff, moderate accuracy | Bdf | Low cost per step, variable order |
| DAE (index-1) | Radau5 | Stiffly accurate, best DAE support |
| Large system (\\( n > 1000 \\)) | Bdf | Cheapest LU cost per step |
| Oscillatory stiff modes | Radau5 or Esdirk54 | A-stable for imaginary eigenvalues |
| Long-time integration | Bdf | Efficient Jacobian reuse |
| Rapid transients + slow phases | Esdirk54 | Good adaptation to changing stiffness |

### By Accuracy Requirement

| Tolerance (rtol) | Recommended solver |
|------------------|--------------------|
| \\( 10^{-2} - 10^{-4} \\) | Bdf or Esdirk32 |
| \\( 10^{-4} - 10^{-8} \\) | Esdirk54 or Bdf |
| \\( 10^{-8} - 10^{-12} \\) | Radau5 |
| \\( < 10^{-12} \\) | Radau5 (or consider Vern8 if non-stiff) |

### Quick Decision Flowchart

```text
Is the system a DAE (mass matrix)?
  |
  +-- Yes --> Radau5
  |
  +-- No
        |
        Stiffness ratio > 10^4?
          |
          +-- Yes --> Need rtol < 10^-8?
          |             |
          |             +-- Yes --> Radau5
          |             +-- No  --> Bdf
          |
          +-- No (moderate stiffness)
                |
                --> Esdirk54
```

## Practical Tips

### Start Conservative

When unsure, start with Esdirk54. It handles moderate and severe stiffness well,
is L-stable, and has reasonable per-step cost. If it struggles (many rejected steps
or slow convergence), switch to Radau5 or BDF.

### Watch the Statistics

After solving, always check `result.stats`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = Radau5::solve(&problem, t0, tf, &y0, &options)?;

println!("Steps: {} accepted, {} rejected",
    result.stats.n_accept, result.stats.n_reject);
println!("Function evals: {}", result.stats.n_eval);
println!("Jacobian evals: {}", result.stats.n_jac);
println!("LU decompositions: {}", result.stats.n_lu);
```

Rules of thumb for a healthy integration:

- Rejection ratio below 20%
- Newton convergence in 2-4 iterations (Radau5 uses up to 7)
- Jacobian updated every 5-20 steps (BDF reuses aggressively)

### Tolerance Selection for Stiff Problems

Stiff solvers require careful tolerance selection:

- **rtol:** Controls the relative accuracy of all components. For stiff problems,
  \\( 10^{-4} \\) to \\( 10^{-8} \\) is typical.
- **atol:** Critical for components that pass through zero or become very small.
  Set it to the smallest meaningful value for each component.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// For the Robertson system where y2 ~ 1e-5:
let options = SolverOptions::default()
    .rtol(1e-4)   // 0.01% relative accuracy
    .atol(1e-10); // Resolve values down to 1e-10
```

### Fixed-Order BDF

If you know the problem structure, you can lock BDF to a specific order:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::Bdf;

// BDF2 only (A-stable, good for mildly stiff problems)
let solver = Bdf::fixed_order(2);

// BDF1 = Backward Euler (most robust, lowest accuracy)
let solver = Bdf::fixed_order(1);
```

## Summary

| Solver | Order | Stability | LU/step | Best niche |
|--------|-------|-----------|---------|------------|
| Radau5 | 5 | L-stable | 2 | High accuracy, DAEs, very stiff |
| Esdirk32 | 2 | L-stable | 1 | Quick-and-dirty, low accuracy |
| Esdirk43 | 3 | L-stable | 1 | Moderate accuracy |
| Esdirk54 | 4 | L-stable | 1 | General purpose stiff solver |
| Bdf | 1-5 | Varies | 1 | Long integrations, large systems |
