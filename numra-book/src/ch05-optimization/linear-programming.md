# Linear & Quadratic Programming

This chapter covers optimization with linear or quadratic objectives and linear constraints. These are structured convex problems with efficient specialized solvers.

## Linear Programming (LP)

A linear program in standard form is:

\\[
\min_{x} \; c^\top x \quad \text{subject to} \quad A_\text{ineq}\, x \le b_\text{ineq}, \quad A_\text{eq}\, x = b_\text{eq}, \quad x \ge 0.
\\]

Numra solves LPs using the **Revised Simplex method** with a **two-phase** approach:

- **Phase I** introduces artificial variables and minimizes their sum to find a basic feasible solution (or detect infeasibility).
- **Phase II** pivots to optimize the actual objective while maintaining feasibility.

### Basic Usage

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{simplex_solve, LPOptions};

// Maximize x0 + x1 subject to x0 + x1 <= 4, x0 <= 3, x1 <= 3, x >= 0
// (Convert max to min by negating the objective)
let c = vec![-1.0, -1.0];
let a_ineq = vec![
    vec![1.0, 1.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
];
let b_ineq = vec![4.0, 3.0, 3.0];

let result = simplex_solve(
    &c, &a_ineq, &b_ineq, &[], &[], &LPOptions::default()
).unwrap();

assert!(result.converged);
assert!((result.f - (-4.0)).abs() < 1e-8);  // max = 4 at (1, 3) or (3, 1)
```

### LP with Equality Constraints

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{simplex_solve, LPOptions};

// Minimize x0 + 2*x1 subject to x0 + x1 = 3, x >= 0
let c = vec![1.0, 2.0];
let a_eq = vec![vec![1.0, 1.0]];
let b_eq = vec![3.0];

let result = simplex_solve(
    &c, &[], &[], &a_eq, &b_eq, &LPOptions::default()
).unwrap();

assert!(result.converged);
assert!((result.f - 3.0).abs() < 1e-8);      // optimal: x0=3, x1=0
assert!((result.x[0] - 3.0).abs() < 1e-8);
```

### Production Planning Example

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{simplex_solve, LPOptions};

// Maximize 5*x1 + 4*x2 + 3*x3 subject to resource constraints
// 6*x1 + 4*x2 + 2*x3 <= 240  (resource 1)
// 3*x1 + 2*x2 + 5*x3 <= 270  (resource 2)
// x >= 0
let c = vec![-5.0, -4.0, -3.0]; // negate for minimization
let a_ineq = vec![
    vec![6.0, 4.0, 2.0],
    vec![3.0, 2.0, 5.0],
];
let b_ineq = vec![240.0, 270.0];

let result = simplex_solve(
    &c, &a_ineq, &b_ineq, &[], &[], &LPOptions::default()
).unwrap();

assert!(result.converged);
println!("Max profit: {:.1}", -result.f);
println!("Production: x1={:.1}, x2={:.1}, x3={:.1}",
    result.x[0], result.x[1], result.x[2]);

// Dual variables (shadow prices) available:
println!("Shadow prices: {:?}", result.lambda_ineq);
```

### LP Solver Signature

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub fn simplex_solve<S>(
    c: &[S],              // objective coefficients (length n)
    a_ineq: &[Vec<S>],    // inequality constraint rows
    b_ineq: &[S],         // inequality RHS
    a_eq: &[Vec<S>],      // equality constraint rows
    b_eq: &[S],           // equality RHS
    opts: &LPOptions<S>,  // solver options
) -> Result<OptimResult<S>, OptimError>
```

### Error Conditions

| Error | When |
|-------|------|
| `OptimError::LPInfeasible` | No feasible solution exists (Phase I objective > 0) |
| `OptimError::Unbounded` | Objective can be decreased without bound |
| `OptimError::DimensionMismatch` | Constraint row length does not match `c.len()` |

## Quadratic Programming (QP)

A convex quadratic program is:

\\[
\min_{x} \; \tfrac{1}{2} x^\top H x + c^\top x \quad \text{subject to} \quad A_\text{ineq}\, x \le b_\text{ineq}, \quad A_\text{eq}\, x = b_\text{eq}, \quad l \le x \le u,
\\]

where \\( H \\) must be **positive semi-definite** (PSD). Numra validates this requirement via Cholesky factorization and returns `QPNotPositiveSemiDefinite` if violated.

The solver uses the **primal active set method**: it maintains a working set of active constraints and iterates between solving equality-constrained QP subproblems (via KKT systems) and adding/removing constraints from the working set.

### Basic QP

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{active_set_qp_solve, QPOptions};

// Minimize 1/2 x^T H x + c^T x where H = [[2,0],[0,2]], c = [-2,-4]
// Optimal: x = [1, 2], f = -5
let h = vec![2.0, 0.0, 0.0, 2.0]; // row-major
let c = vec![-2.0, -4.0];

let result = active_set_qp_solve(
    &h, &c, 2, &[], &[], &[], &[], &[], &QPOptions::default()
).unwrap();

assert!(result.converged);
assert!((result.x[0] - 1.0).abs() < 1e-6);
assert!((result.x[1] - 2.0).abs() < 1e-6);
```

### QP with Constraints and Bounds

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{active_set_qp_solve, QPOptions};

// Minimize 1/2 x^T I x + [-3,-3]^T x with 0 <= xi <= 1
// Unconstrained min at (3,3), bounded to (1,1)
let h = vec![1.0, 0.0, 0.0, 1.0];
let c = vec![-3.0, -3.0];
let bounds: Vec<Option<(f64, f64)>> = vec![Some((0.0, 1.0)), Some((0.0, 1.0))];

let result = active_set_qp_solve(
    &h, &c, 2, &[], &[], &[], &[], &bounds, &QPOptions::default()
).unwrap();

assert!((result.x[0] - 1.0).abs() < 1e-6);
assert!((result.x[1] - 1.0).abs() < 1e-6);
```

### Markowitz Portfolio Optimization

A classic QP application: minimize portfolio variance subject to a return target and budget constraint.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::{active_set_qp_solve, QPOptions};

// Covariance matrix (3 assets)
let sigma = vec![
    0.04, 0.006, 0.002,
    0.006, 0.01, 0.004,
    0.002, 0.004, 0.0225,
];
let c = vec![0.0, 0.0, 0.0]; // no linear term

// Expected returns: [0.12, 0.10, 0.07]
// Target return >= 0.10: -mu^T x <= -0.10
let a_ineq = vec![vec![-0.12, -0.10, -0.07]];
let b_ineq = vec![-0.10];

// Budget: w1 + w2 + w3 = 1
let a_eq = vec![vec![1.0, 1.0, 1.0]];
let b_eq = vec![1.0];

// No short selling: 0 <= wi <= 1
let bounds: Vec<Option<(f64, f64)>> = vec![
    Some((0.0, 1.0)), Some((0.0, 1.0)), Some((0.0, 1.0))
];

let result = active_set_qp_solve(
    &sigma, &c, 3, &a_ineq, &b_ineq, &a_eq, &b_eq, &bounds,
    &QPOptions::default()
).unwrap();

assert!(result.converged);
let total: f64 = result.x.iter().sum();
assert!((total - 1.0).abs() < 1e-6);  // weights sum to 1

let ret = 0.12 * result.x[0] + 0.10 * result.x[1] + 0.07 * result.x[2];
println!("Portfolio: w = {:?}", result.x);
println!("Return: {:.4}, Variance: {:.6}", ret, result.f);
```

## Mixed-Integer Linear Programming (MILP)

For problems with integer or binary variables, Numra provides a **Branch-and-Bound** solver that uses LP relaxations:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::OptimProblem;

// Binary knapsack: maximize 4*x1 + 5*x2 + 3*x3
// subject to 3*x1 + 4*x2 + 2*x3 <= 7, xi in {0, 1}
let result = OptimProblem::new(3)
    .linear_objective(&[-4.0, -5.0, -3.0])  // negate for min
    .linear_constraint_ineq(&[3.0, 4.0, 2.0], 7.0)
    .binary_var(0)
    .binary_var(1)
    .binary_var(2)
    .solve()
    .unwrap();

assert!(result.converged);
println!("Selection: {:?}", result.x);
println!("Value: {:.0}", -result.f);
```

### MILP Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_nodes` | 10000 | Maximum branch-and-bound nodes |
| `tol` | 1e-6 | Integrality tolerance |
| `verbose` | false | Print search tree progress |

## Declarative Builder for LP/QP

The `OptimProblem` builder automatically dispatches to the correct solver based on the problem structure:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::optim::OptimProblem;

// LP via builder
let lp_result = OptimProblem::new(2)
    .linear_objective(&[-1.0, -1.0])
    .linear_constraint_ineq(&[1.0, 1.0], 4.0)
    .linear_constraint_ineq(&[1.0, 0.0], 3.0)
    .solve()
    .unwrap();

// QP via builder
let qp_result = OptimProblem::new(2)
    .quadratic_objective_dense(&[2.0, 0.0, 0.0, 2.0], &[-2.0, -4.0])
    .solve()
    .unwrap();
```

Auto dispatch logic:
- `ObjectiveKind::Linear` + no integer vars --> `simplex_solve`
- `ObjectiveKind::Linear` + integer/binary vars --> `milp_solve`
- `ObjectiveKind::Quadratic` --> `active_set_qp_solve`

## LP/QP Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iter` | 10000 | Maximum simplex/active-set iterations |
| `tol` | 1e-10 | Numerical tolerance for pivoting and convergence |
| `verbose` | false | Print iteration details |
