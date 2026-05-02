---
title: "Optimizer Comparison"
---

This chapter provides a unified reference for choosing the right solver from the `numra::optim` module. The table below summarizes all available algorithms, followed by a decision flowchart and error handling guide.

## Master Comparison Table

| Solver | Function | Problem Type | Gradients | Constraints | Convergence |
|--------|----------|-------------|-----------|-------------|-------------|
| **BFGS** | `bfgs_minimize` | Unconstrained | Required | None | Superlinear |
| **L-BFGS** | `lbfgs_minimize` | Unconstrained | Required | None | Superlinear |
| **Nelder-Mead** | `nelder_mead` | Unconstrained | None | None | Linear |
| **Powell** | `powell` | Unconstrained | None | None | Superlinear |
| **L-BFGS-B** | `lbfgsb_minimize` | Bound-constrained | Required | Box bounds | Superlinear |
| **Aug. Lagrangian** | `augmented_lagrangian_minimize` | General NLP | Optional | Eq + Ineq + Bounds | Linear (outer) |
| **SQP** | `sqp_minimize` | General NLP | Required | Eq + Ineq | Superlinear |
| **Levenberg-Marquardt** | `lm_minimize` | Least squares | Jacobian | None | Superlinear |
| **Simplex** | `simplex_solve` | Linear program | N/A | Linear Eq + Ineq | Finite |
| **Active Set QP** | `active_set_qp_solve` | Quadratic program | N/A | Linear + Bounds | Finite |
| **MILP** | `milp_solve` | Mixed-integer LP | N/A | Linear + Integer | Finite |
| **DE** | `de_minimize` | Global (box) | None | Box bounds | Stochastic |
| **CMA-ES** | `cmaes_minimize` | Global (box) | None | Box bounds | Stochastic |
| **NSGA-II** | `nsga2_optimize` | Multi-objective | None | Box bounds | Stochastic |
| **Robust** | `RobustProblem::solve` | Uncertain params | Optional | Eq + Ineq + Bounds | Depends on inner |
| **Stochastic** | `StochasticProblem::solve` | Random params | Optional | Det + Chance | Depends on inner |

## Decision Flowchart

Use this guide to select the appropriate solver:

```text
Is the objective linear?
  |-- Yes: Are there integer/binary variables?
  |     |-- Yes --> milp_solve (Branch-and-Bound)
  |     |-- No  --> simplex_solve (Revised Simplex)
  |
  |-- No: Is the objective quadratic with linear constraints?
        |-- Yes --> active_set_qp_solve (Active Set QP)
        |
        |-- No: Is it a least-squares problem (sum of squared residuals)?
              |-- Yes --> lm_minimize (Levenberg-Marquardt)
              |
              |-- No: Are there multiple objectives?
                    |-- Yes --> nsga2_optimize (NSGA-II)
                    |
                    |-- No: Are parameters uncertain?
                          |-- Yes: Worst-case safety?
                          |     |-- Yes --> RobustProblem (Robust Opt.)
                          |     |-- No  --> StochasticProblem (SAA/CVaR)
                          |
                          |-- No: Is the landscape multimodal/non-convex?
                                |-- Yes --> de_minimize or cmaes_minimize
                                |
                                |-- No: Are there constraints?
                                      |-- Bounds only --> lbfgsb_minimize
                                      |-- General    --> augmented_lagrangian_minimize or sqp_minimize
                                      |-- None       --> Are gradients available?
                                            |-- Yes: n < 1000? --> bfgs_minimize
                                            |        n >= 1000? --> lbfgs_minimize
                                            |-- No  --> nelder_mead or powell
```

## Automatic Solver Selection

The `OptimProblem` builder's `.solve()` method automatically dispatches to the appropriate solver:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

// Auto-selects based on problem structure:
let result = OptimProblem::new(2)
    .x0(&[1.0, 1.0])
    .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
    .gradient(|x: &[f64], g: &mut [f64]| { g[0] = 2.0 * x[0]; g[1] = 2.0 * x[1]; })
    .solve()  // dispatches to L-BFGS (unconstrained + gradient)
    .unwrap();
```

The dispatch logic considers:

| Problem Structure | Auto-Selected Solver |
|------------------|---------------------|
| Linear objective | Simplex or MILP |
| Quadratic objective | Active Set QP |
| Least-squares | Levenberg-Marquardt |
| Multi-objective | NSGA-II |
| Global flag set | Differential Evolution |
| Constraints present | Augmented Lagrangian |
| Bounds only | L-BFGS-B |
| Unconstrained + gradient | L-BFGS |
| Unconstrained, no gradient | Nelder-Mead |

### Explicit Solver Selection

Override auto-dispatch with `.solve_with()`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{OptimProblem, SolverChoice};

let result = OptimProblem::new(2)
    .x0(&[1.0, 1.0])
    .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
    .solve_with(SolverChoice::Bfgs)
    .unwrap();
```

## Error Handling Summary

All solvers return `Result<OptimResult<S>, OptimError>`. The error variants cover all failure modes:

| Error Variant | Description | Common Cause |
|--------------|-------------|--------------|
| `LineSearchFailed` | Wolfe line search failed | Bad gradient, non-smooth objective |
| `NotDescentDirection` | Search direction has positive directional derivative | Numerical issues in Hessian approximation |
| `InvalidFunctionValue` | NaN or Inf objective value | Undefined region of objective function |
| `SingularMatrix` | Linear system solve failed | Degenerate constraints, poor conditioning |
| `DimensionMismatch` | Array sizes do not match | Programming error in problem setup |
| `NoObjective` | No objective function provided | Missing `.objective()` call |
| `NoInitialPoint` | No initial guess provided | Missing `.x0()` call |
| `Infeasible` | Constraints cannot be satisfied | Over-constrained problem |
| `Unbounded` | Objective can decrease without bound | Missing constraints in LP |
| `LPInfeasible` | LP has no feasible solution | Contradictory linear constraints |
| `QPNotPositiveSemiDefinite` | QP Hessian is indefinite | Non-convex QP formulation |
| `MILPInfeasible` | MILP has no feasible integer solution | Over-constrained integer program |
| `Other(String)` | Catch-all for miscellaneous errors | Various |

### Error Handling Pattern

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{OptimProblem, OptimError};

match OptimProblem::new(2)
    .x0(&[1.0, 1.0])
    .objective(|x: &[f64]| x[0] + x[1])
    .constraint_eq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0)
    .solve()
{
    Ok(result) => {
        if result.converged {
            println!("Optimal: x={:?}, f={:.6}", result.x, result.f);
        } else {
            println!("Warning: did not converge. Status: {:?}", result.status);
            println!("Best found: x={:?}, f={:.6}", result.x, result.f);
        }
    }
    Err(OptimError::Infeasible { violation }) => {
        println!("Infeasible: max violation = {:.2e}", violation);
    }
    Err(e) => {
        println!("Optimization failed: {}", e);
    }
}
```

## Performance Tips

1. **Always provide gradients** when possible. Finite-difference gradients cost $ 2n $ extra function evaluations and are less accurate.

2. **Use L-BFGS over BFGS** for $ n > 1000 $. The $ O(n^2) $ dense Hessian in BFGS becomes prohibitive.

3. **Tighten tolerances gradually.** Start with default tolerances and only tighten `gtol`/`ftol` if the solution is insufficiently accurate.

4. **Provide a good initial point.** For local solvers, the initial point determines which local minimum is found. For constrained problems, start near the feasible region.

5. **Use the builder API** for complex problems. It handles solver selection, finite-difference fallbacks, and constraint formatting automatically.

6. **Combine global + local** for multimodal problems: use DE or CMA-ES for exploration, then BFGS for polishing.

7. **Check `result.history`** for convergence diagnostics. A stalling gradient norm or oscillating objective suggests the solver is struggling.

## Common `OptimResult` Fields

Every solver populates these fields:

| Field | Type | Always Available |
|-------|------|-----------------|
| `x` | `Vec<S>` | Yes |
| `f` | `S` | Yes |
| `grad` | `Vec<S>` | Yes (empty for derivative-free) |
| `iterations` | `usize` | Yes |
| `n_feval` | `usize` | Yes |
| `n_geval` | `usize` | Yes (0 for derivative-free) |
| `converged` | `bool` | Yes |
| `message` | `String` | Yes |
| `status` | `OptimStatus` | Yes |
| `history` | `Vec<IterationRecord<S>>` | Yes (may be empty) |
| `wall_time_secs` | `f64` | Yes |
| `lambda_eq` | `Vec<S>` | Constrained solvers |
| `lambda_ineq` | `Vec<S>` | Constrained solvers |
| `active_bounds` | `Vec<usize>` | L-BFGS-B, QP |
| `constraint_violation` | `S` | Constrained solvers |
| `pareto` | `Option<ParetoResult<S>>` | NSGA-II only |
| `sensitivity` | `Option<ParamSensitivity<S>>` | Robust only |
