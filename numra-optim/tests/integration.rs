//! Integration tests: autodiff gradients feeding into optimizers,
//! builder API, constrained optimization.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_autodiff::{gradient, Dual};
use numra_optim::{
    bfgs_minimize, lbfgs_minimize, LbfgsOptions, OptimOptions, OptimProblem, ProblemHint,
    SolverChoice,
};

// ── Existing tests ──

#[test]
fn test_autodiff_bfgs_rosenbrock() {
    fn rosenbrock_dual(x: &[Dual<f64>]) -> Dual<f64> {
        let one = Dual::constant(1.0);
        let hundred = Dual::constant(100.0);
        let a = one - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + hundred * b * b
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    let grad_fn = |x: &[f64], g: &mut [f64]| {
        let grad = gradient(rosenbrock_dual, x);
        g.copy_from_slice(&grad);
    };

    let result =
        bfgs_minimize(rosenbrock, grad_fn, &[-1.0, 1.0], &OptimOptions::default()).unwrap();
    assert!(
        result.converged,
        "BFGS+AD did not converge: {}",
        result.message
    );
    assert!((result.x[0] - 1.0).abs() < 1e-4);
    assert!((result.x[1] - 1.0).abs() < 1e-4);
}

#[test]
fn test_autodiff_lbfgs_sphere() {
    fn sphere_dual(x: &[Dual<f64>]) -> Dual<f64> {
        x.iter().fold(Dual::constant(0.0), |acc, &xi| acc + xi * xi)
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    let grad_fn = |x: &[f64], g: &mut [f64]| {
        let grad = gradient(sphere_dual, x);
        g.copy_from_slice(&grad);
    };

    let x0: Vec<f64> = (1..=50).map(|i| i as f64 * 0.1).collect();
    let result = lbfgs_minimize(sphere, grad_fn, &x0, &LbfgsOptions::default()).unwrap();
    assert!(result.converged);
    assert!(result.f < 1e-10);
}

// ── Builder API tests ──

#[test]
fn test_builder_unconstrained_solve() {
    let result = OptimProblem::new(2)
        .x0(&[5.0, 3.0])
        .objective(|x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1])
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 8.0 * x[1];
        })
        .solve()
        .unwrap();

    assert!(result.converged, "did not converge: {}", result.message);
    assert!(result.x[0].abs() < 1e-6);
    assert!(result.x[1].abs() < 1e-6);
}

#[test]
fn test_builder_with_bounds_lbfgsb() {
    // f(x) = (x0-2)^2 + (x1-2)^2
    // bounds: x0 in [0,1], x1 in [0,3]
    // min at (1, 2) with x0 at upper bound
    let result = OptimProblem::new(2)
        .x0(&[0.5, 0.5])
        .objective(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 2.0);
            g[1] = 2.0 * (x[1] - 2.0);
        })
        .all_bounds(&[(0.0, 1.0), (0.0, 3.0)])
        .solve()
        .unwrap();

    assert!(result.converged, "did not converge: {}", result.message);
    assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 1e-4, "x1={}", result.x[1]);
}

#[test]
fn test_builder_with_constraints_auglag() {
    // minimize x0 + x1 subject to x0^2 + x1^2 = 1
    let result = OptimProblem::new(2)
        .x0(&[1.0, 0.0])
        .objective(|x: &[f64]| x[0] + x[1])
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 1.0;
            g[1] = 1.0;
            let _ = x;
        })
        .constraint_eq_with_grad(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0,
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            },
        )
        .solve()
        .unwrap();

    assert!(result.converged, "did not converge: {}", result.message);
    let expected = -1.0 / 2.0_f64.sqrt();
    assert!((result.x[0] - expected).abs() < 1e-3, "x0={}", result.x[0]);
    assert!((result.x[1] - expected).abs() < 1e-3, "x1={}", result.x[1]);
    assert!(result.constraint_violation < 1e-4);
}

#[test]
fn test_builder_least_squares_lm() {
    // Fit y = a*x + b to data: (0,1), (1,3), (2,5)
    let t_data = vec![0.0, 1.0, 2.0];
    let y_data = vec![1.0, 3.0, 5.0];

    let t = t_data.clone();
    let y = y_data.clone();
    let result = OptimProblem::new(2)
        .x0(&[0.0, 0.0])
        .least_squares(3, move |p: &[f64], r: &mut [f64]| {
            for i in 0..t.len() {
                r[i] = p[0] * t[i] + p[1] - y[i];
            }
        })
        .jacobian({
            let t = t_data.clone();
            move |_p: &[f64], jac: &mut [f64]| {
                for i in 0..t.len() {
                    jac[i * 2] = t[i];
                    jac[i * 2 + 1] = 1.0;
                }
            }
        })
        .solve()
        .unwrap();

    assert!(
        result.converged,
        "LM via builder did not converge: {}",
        result.message
    );
    assert!((result.x[0] - 2.0).abs() < 1e-6, "a={}", result.x[0]);
    assert!((result.x[1] - 1.0).abs() < 1e-6, "b={}", result.x[1]);
}

#[test]
fn test_gradient_free_finite_diff_solve() {
    // No gradient provided — uses finite differences internally
    let result = OptimProblem::new(2)
        .x0(&[5.0, 3.0])
        .objective(|x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1])
        .solve()
        .unwrap();

    assert!(
        result.converged,
        "FD solve did not converge: {}",
        result.message
    );
    assert!(result.x[0].abs() < 1e-4, "x0={}", result.x[0]);
    assert!(result.x[1].abs() < 1e-4, "x1={}", result.x[1]);
}

#[test]
fn test_solve_with_explicit_bfgs() {
    let result = OptimProblem::new(2)
        .x0(&[5.0, 3.0])
        .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
        })
        .solve_with(SolverChoice::Bfgs)
        .unwrap();

    assert!(result.converged);
    assert!(result.x[0].abs() < 1e-6);
    assert!(result.x[1].abs() < 1e-6);
}

#[test]
fn test_builder_autodiff_gradient_bfgs() {
    // Use autodiff gradient via the builder API
    fn rosenbrock_dual(x: &[Dual<f64>]) -> Dual<f64> {
        let one = Dual::constant(1.0);
        let hundred = Dual::constant(100.0);
        let a = one - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + hundred * b * b
    }

    let result = OptimProblem::new(2)
        .x0(&[-1.0, 1.0])
        .objective(|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
        .gradient(move |x: &[f64], g: &mut [f64]| {
            let grad = gradient(rosenbrock_dual, x);
            g.copy_from_slice(&grad);
        })
        .solve_with(SolverChoice::Bfgs)
        .unwrap();

    assert!(result.converged, "BFGS+AD via builder: {}", result.message);
    assert!((result.x[0] - 1.0).abs() < 1e-4);
    assert!((result.x[1] - 1.0).abs() < 1e-4);
}

// ── Phase 3 integration tests ──

#[test]
fn test_convergence_history_populated() {
    // Solve Rosenbrock and verify history is populated
    let result = numra_optim::OptimProblem::new(2)
        .x0(&[-1.0, 1.0])
        .objective(|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        })
        .solve()
        .unwrap();
    assert!(result.converged, "did not converge: {}", result.message);
    assert!(!result.history.is_empty(), "history should be populated");
    assert!(
        result.wall_time_secs > 0.0,
        "wall_time_secs should be positive"
    );
}

#[test]
fn test_wall_time_all_solvers() {
    // L-BFGS (unconstrained)
    let r = numra_optim::OptimProblem::new(2)
        .x0(&[5.0, 3.0])
        .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
        })
        .solve()
        .unwrap();
    assert!(
        r.wall_time_secs > 0.0,
        "L-BFGS wall_time should be positive"
    );

    // L-BFGS-B (bounded)
    let r = numra_optim::OptimProblem::new(2)
        .x0(&[0.5, 0.5])
        .objective(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 2.0);
            g[1] = 2.0 * (x[1] - 2.0);
        })
        .all_bounds(&[(0.0, 1.0), (0.0, 3.0)])
        .solve()
        .unwrap();
    assert!(
        r.wall_time_secs > 0.0,
        "L-BFGS-B wall_time should be positive"
    );

    // LM (least squares)
    let r = numra_optim::OptimProblem::new(2)
        .x0(&[0.0, 0.0])
        .least_squares(3, |x: &[f64], res: &mut [f64]| {
            res[0] = x[0] - 1.0;
            res[1] = x[1] - 2.0;
            res[2] = x[0] + x[1] - 3.0;
        })
        .solve()
        .unwrap();
    assert!(r.wall_time_secs > 0.0, "LM wall_time should be positive");

    // Augmented Lagrangian (constrained)
    let r = numra_optim::OptimProblem::new(2)
        .x0(&[1.0, 0.0])
        .objective(|x: &[f64]| x[0] + x[1])
        .gradient(|x: &[f64], g: &mut [f64]| {
            g[0] = 1.0;
            g[1] = 1.0;
            let _ = x;
        })
        .constraint_eq_with_grad(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0,
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            },
        )
        .solve()
        .unwrap();
    assert!(
        r.wall_time_secs > 0.0,
        "AugLag wall_time should be positive"
    );
}

#[test]
fn test_maximize_constrained() {
    // maximize x0 + x1 subject to x0^2 + x1^2 <= 1
    // Maximum at (1/sqrt(2), 1/sqrt(2))
    let result = numra_optim::OptimProblem::new(2)
        .x0(&[0.5, 0.5])
        .maximize(|x: &[f64]| x[0] + x[1])
        .constraint_ineq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0)
        .solve()
        .unwrap();
    assert!(result.converged, "did not converge: {}", result.message);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!(
        (result.x[0] - expected).abs() < 5e-2,
        "x0={}, expected {}",
        result.x[0],
        expected
    );
    assert!(
        (result.x[1] - expected).abs() < 5e-2,
        "x1={}, expected {}",
        result.x[1],
        expected
    );
}

// ── Phase 4 integration tests ──

#[test]
fn test_lp_via_builder() {
    // Transport problem: minimize shipping cost
    // minimize 2*x0 + 3*x1 + x2 + 4*x3
    // subject to: x0 + x1 = 10 (supply 1)
    //             x2 + x3 = 15 (supply 2)
    //             x0 + x2 >= 8 => -x0 - x2 <= -8 (demand 1)
    //             x1 + x3 >= 12 => -x1 - x3 <= -12 (demand 2)
    let result = numra_optim::OptimProblem::new(4)
        .linear_objective(&[2.0, 3.0, 1.0, 4.0])
        .linear_constraint_eq(&[1.0, 1.0, 0.0, 0.0], 10.0)
        .linear_constraint_eq(&[0.0, 0.0, 1.0, 1.0], 15.0)
        .linear_constraint_ineq(&[-1.0, 0.0, -1.0, 0.0], -8.0)
        .linear_constraint_ineq(&[0.0, -1.0, 0.0, -1.0], -12.0)
        .solve()
        .unwrap();
    assert!(result.converged, "LP did not converge: {}", result.message);
    assert!(result.wall_time_secs > 0.0);
    let supply1: f64 = result.x[0] + result.x[1];
    assert!((supply1 - 10.0).abs() < 1e-6, "supply1={}", supply1);
    let supply2: f64 = result.x[2] + result.x[3];
    assert!((supply2 - 15.0).abs() < 1e-6, "supply2={}", supply2);
}

#[test]
fn test_qp_via_builder() {
    // Portfolio optimization: minimize risk subject to return >= target
    // 2 assets: Sigma = [[0.04, 0.01], [0.01, 0.09]], mu = [0.10, 0.15]
    // target return = 0.12
    let result = numra_optim::OptimProblem::new(2)
        .quadratic_objective_dense(&[0.04, 0.01, 0.01, 0.09], &[0.0, 0.0])
        .linear_constraint_eq(&[1.0, 1.0], 1.0)
        .linear_constraint_ineq(&[-0.10, -0.15], -0.12)
        .solve()
        .unwrap();
    assert!(result.converged, "QP did not converge: {}", result.message);
    assert!(result.wall_time_secs > 0.0);
    let sum: f64 = result.x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "sum={}", sum);
    let ret = 0.10 * result.x[0] + 0.15 * result.x[1];
    assert!(ret >= 0.12 - 1e-4, "return={}", ret);
}

#[test]
fn test_auto_selects_simplex_for_linear() {
    let p = numra_optim::OptimProblem::new(2)
        .linear_objective(&[1.0, 2.0])
        .linear_constraint_ineq(&[1.0, 1.0], 5.0);
    assert_eq!(
        numra_optim::auto::select_solver(&p),
        numra_optim::SolverChoice::Simplex,
    );
}

#[test]
fn test_auto_selects_active_set_for_quadratic() {
    let p = numra_optim::OptimProblem::new(2)
        .quadratic_objective_dense(&[2.0, 0.0, 0.0, 2.0], &[0.0, 0.0]);
    assert_eq!(
        numra_optim::auto::select_solver(&p),
        numra_optim::SolverChoice::ActiveSetQP,
    );
}

#[test]
fn test_lp_wall_time_and_history() {
    let result = numra_optim::OptimProblem::new(2)
        .linear_objective(&[-1.0, -1.0])
        .linear_constraint_ineq(&[1.0, 1.0], 4.0)
        .linear_constraint_ineq(&[1.0, 0.0], 3.0)
        .linear_constraint_ineq(&[0.0, 1.0], 3.0)
        .solve()
        .unwrap();
    assert!(result.converged);
    assert!(result.wall_time_secs > 0.0, "LP should have wall time");
}

// ── Phase 5: MILP, Global, Multi-Objective ──

#[test]
fn test_milp_knapsack_via_builder() {
    // 0-1 knapsack: maximize 6b1 + 5b2 + 4b3 s.t. 3b1 + 4b2 + 2b3 <= 7
    let result = OptimProblem::new(3)
        .linear_objective(&[-6.0, -5.0, -4.0])
        .linear_constraint_ineq(&[3.0, 4.0, 2.0], 7.0)
        .binary_var(0)
        .binary_var(1)
        .binary_var(2)
        .solve()
        .unwrap();
    assert!(
        result.converged,
        "MILP did not converge: {}",
        result.message
    );
    for &xi in &result.x {
        let xi: f64 = xi;
        assert!(xi < 1e-6 || (xi - 1.0).abs() < 1e-6, "non-binary: {}", xi);
    }
    assert!(result.f <= -10.9, "f={}", result.f);
}

#[test]
fn test_milp_integer_assignment_via_builder() {
    // minimize -x1 - 2*x2 s.t. x1+x2=3, x1,x2 integer, >=0
    let result = OptimProblem::new(2)
        .linear_objective(&[-1.0, -2.0])
        .linear_constraint_eq(&[1.0, 1.0], 3.0)
        .integer_var(0)
        .integer_var(1)
        .solve()
        .unwrap();
    assert!(result.converged);
    assert!((result.f - (-6.0_f64)).abs() < 0.1, "f={}", result.f);
}

#[test]
fn test_tiny_lp_has_unique_golden_vertex() {
    // Minimize -2x - y, equivalent to maximizing 2x + y.
    // The unique optimum is x = 2, y = 2 under x <= 2, y <= 3, x + y <= 4.
    let result = numra_optim::simplex_solve::<f64>(
        &[-2.0, -1.0],
        &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        &[2.0, 3.0, 4.0],
        &[],
        &[],
        &numra_optim::LPOptions::default(),
    )
    .unwrap();

    assert!(result.converged, "LP did not converge: {}", result.message);
    assert!((result.x[0] - 2.0).abs() < 1e-8, "x={:?}", result.x);
    assert!((result.x[1] - 2.0).abs() < 1e-8, "x={:?}", result.x);
    assert!((result.f + 6.0).abs() < 1e-8, "f={}", result.f);
}

#[test]
fn test_tiny_qp_has_unconstrained_golden_minimizer() {
    // 0.5 * x^T diag(2, 4) x + [-2, -8]^T x has optimum [1, 2].
    let result = numra_optim::active_set_qp_solve::<f64>(
        &[2.0, 0.0, 0.0, 4.0],
        &[-2.0, -8.0],
        2,
        &[],
        &[],
        &[],
        &[],
        &[],
        &numra_optim::QPOptions::default(),
    )
    .unwrap();

    assert!(result.converged, "QP did not converge: {}", result.message);
    assert!((result.x[0] - 1.0).abs() < 1e-8, "x={:?}", result.x);
    assert!((result.x[1] - 2.0).abs() < 1e-8, "x={:?}", result.x);
    assert!((result.f + 9.0).abs() < 1e-8, "f={}", result.f);
}

#[test]
fn test_tiny_milp_has_unique_binary_solution() {
    // Maximize 5b0 + 3b1 + 2b2 with capacity 5. Unique optimum: b0=1, b1=0, b2=1.
    let result = numra_optim::milp_solve::<f64>(
        &[-5.0, -3.0, -2.0],
        &[vec![4.0, 2.0, 1.0]],
        &[5.0],
        &[],
        &[],
        &[
            numra_optim::VarType::Binary,
            numra_optim::VarType::Binary,
            numra_optim::VarType::Binary,
        ],
        &[None, None, None],
        &numra_optim::MILPOptions::default(),
    )
    .unwrap();

    assert!(
        result.converged,
        "MILP did not converge: {}",
        result.message
    );
    assert!((result.x[0] - 1.0).abs() < 1e-8, "x={:?}", result.x);
    assert!(result.x[1].abs() < 1e-8, "x={:?}", result.x);
    assert!((result.x[2] - 1.0).abs() < 1e-8, "x={:?}", result.x);
    assert!((result.f + 7.0).abs() < 1e-8, "f={}", result.f);
}

#[test]
fn test_global_rosenbrock_via_builder() {
    let result = OptimProblem::new(2)
        .objective(|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
        .all_bounds(&[(-5.0, 10.0), (-5.0, 10.0)])
        .global(true)
        .solve()
        .unwrap();
    assert!(result.f < 0.1, "f={}", result.f);
    assert!((result.x[0] - 1.0).abs() < 0.5, "x0={}", result.x[0]);
}

#[test]
fn test_de_explicit_via_builder() {
    let result = OptimProblem::new(3)
        .objective(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
        .all_bounds(&[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)])
        .solve_with(SolverChoice::DifferentialEvolution)
        .unwrap();
    assert!(result.f < 0.1, "f={}", result.f);
}

#[test]
fn test_nsga2_biobj_via_builder() {
    let result = OptimProblem::new(1)
        .multi_objective(vec![
            Box::new(|x: &[f64]| x[0] * x[0]),
            Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
        ])
        .all_bounds(&[(0.0, 4.0)])
        .solve()
        .unwrap();
    let pareto = result.pareto.as_ref().expect("no Pareto result");
    assert!(
        pareto.points.len() >= 3,
        "too few points: {}",
        pareto.points.len()
    );
    // Verify non-domination
    for (i, pi) in pareto.points.iter().enumerate() {
        for (j, pj) in pareto.points.iter().enumerate() {
            if i == j {
                continue;
            }
            let dominates = pi
                .objectives
                .iter()
                .zip(&pj.objectives)
                .all(|(a, b)| a <= b)
                && pi.objectives.iter().zip(&pj.objectives).any(|(a, b)| a < b);
            assert!(!dominates, "point {} dominates {}", i, j);
        }
    }
}

#[test]
fn test_milp_wall_time() {
    let result = OptimProblem::new(2)
        .linear_objective(&[-1.0, -2.0])
        .linear_constraint_ineq(&[1.0, 1.0], 5.0)
        .integer_var(0)
        .integer_var(1)
        .solve()
        .unwrap();
    assert!(result.wall_time_secs > 0.0, "wall_time not populated");
}

// ── Phase 7: Robust, Stochastic, Sensitivity ──

#[test]
fn test_robust_structural_design() {
    use numra_optim::RobustProblem;

    // Minimize area A = b * h (2 vars: b, h)
    // Subject to: stress = P / (b * h) <= sigma_allow (constraint: P/(b*h) - sigma_allow <= 0)
    // P is uncertain: P = 10 ± 0.5 (load in kN)
    // sigma_allow = 250 (MPa, certain)
    // bounds: b in [0.01, 1], h in [0.01, 1]

    let sigma_allow = 250.0;

    // First solve nominal
    let nominal_result = numra_optim::OptimProblem::new(2)
        .x0(&[0.1, 0.1])
        .objective(|x: &[f64]| x[0] * x[1])
        .constraint_ineq(move |x: &[f64]| 10.0 / (x[0] * x[1]) - sigma_allow)
        .bounds(0, (0.01, 1.0))
        .bounds(1, (0.01, 1.0))
        .solve()
        .unwrap();

    // Now solve robust with uncertain load
    let robust_result = RobustProblem::new(2)
        .x0(&[0.1, 0.1])
        .objective(|x: &[f64], _p: &[f64]| x[0] * x[1])
        .constraint_ineq(move |x: &[f64], p: &[f64]| p[0] / (x[0] * x[1]) - sigma_allow)
        .param("P", 10.0, 0.5)
        .confidence(0.95)
        .bounds(0, (0.01, 1.0))
        .bounds(1, (0.01, 1.0))
        .solve()
        .unwrap();

    // Robust solution should be more conservative (larger area)
    let nominal_area = nominal_result.x[0] * nominal_result.x[1];
    let robust_area = robust_result.x[0] * robust_result.x[1];
    assert!(
        robust_area >= nominal_area * 0.99,
        "robust area {} should be >= nominal area {}",
        robust_area,
        nominal_area
    );
}

#[test]
fn test_stochastic_newsvendor() {
    use numra_optim::StochasticProblem;

    // min E[c_over * max(0, x - D) + c_under * max(0, D - x)]
    // D ~ Normal(100, 15), c_over = 1, c_under = 3
    // Optimal: x* = quantile(c_under/(c_over+c_under), D) = quantile(0.75, D) ~ 100 + 0.674*15 ~ 110

    let c_over = 1.0;
    let c_under = 3.0;

    let result = StochasticProblem::new(1)
        .x0(&[100.0])
        .objective(move |x: &[f64], p: &[f64]| {
            let demand = p[0];
            let overage = (x[0] - demand).max(0.0) * c_over;
            let underage = (demand - x[0]).max(0.0) * c_under;
            overage + underage
        })
        .param_normal("demand", 100.0, 15.0)
        .n_samples(500)
        .seed(42)
        .bounds(0, (0.0, 200.0))
        .solve()
        .unwrap();

    // Optimal around 110 +/- 15 (stochastic tolerance)
    assert!(
        result.x[0] > 90.0 && result.x[0] < 130.0,
        "newsvendor x* = {}, expected ~110",
        result.x[0]
    );
    assert!(result.converged);
    assert!(result.f_mean > 0.0);
    assert!(result.f_std_error > 0.0);
}

#[test]
fn test_sensitivity_with_constraints() {
    use numra_optim::compute_param_sensitivity;

    // min x^2 s.t. x >= p (equivalently: p - x <= 0)
    // When constraint is active: x* = p, so dx*/dp = 1
    let result = compute_param_sensitivity(
        |params: &[f64]| {
            let p = params[0];
            numra_optim::OptimProblem::new(1)
                .x0(&[p + 0.1]) // start slightly above p
                .objective(|x: &[f64]| x[0] * x[0])
                .gradient(|x: &[f64], g: &mut [f64]| {
                    g[0] = 2.0 * x[0];
                })
                .constraint_ineq_with_grad(
                    move |x: &[f64]| p - x[0], // p - x <= 0 means x >= p
                    move |_x: &[f64], g: &mut [f64]| {
                        g[0] = -1.0;
                    },
                )
                .bounds(0, (0.0, 100.0))
                .max_iter(1000)
        },
        &[3.0],
        &["p"],
        None,
    )
    .unwrap();

    // dx*/dp should be 1 (constraint active, x* tracks p)
    assert!(
        (result.get(0, 0) - 1.0).abs() < 0.1,
        "dx*/dp = {}, expected ~1.0",
        result.get(0, 0)
    );
}

#[test]
fn test_robust_solution_uncertainty() {
    use numra_optim::RobustProblem;

    // min (x - p)^2, unconstrained, p = 10 +/- 2
    // x* = p, so dx*/dp = 1, x_std = 1 * 2 = 2
    let result = RobustProblem::new(1)
        .x0(&[0.0])
        .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
        .param("p", 10.0, 2.0)
        .solve()
        .unwrap();

    assert!(result.converged);
    assert!((result.x[0] - 10.0).abs() < 0.5, "x* = {}", result.x[0]);
    assert!(!result.x_std.is_empty());
    assert!(
        result.x_std[0] > 0.5,
        "x_std = {}, expected ~2.0",
        result.x_std[0]
    );
    assert!(
        result.x_std[0] < 4.0,
        "x_std = {}, expected ~2.0",
        result.x_std[0]
    );
    assert!(result.wall_time_secs > 0.0);
}

// ── ProblemHint tests ──

#[test]
fn test_hint_linear_closure_solves_correctly() {
    // Linear problem: min 2*x0 + 3*x1, bounds [0,10]
    // Optimal at (0, 0) with f=0
    let result = OptimProblem::new(2)
        .x0(&[5.0, 5.0])
        .objective(|x: &[f64]| 2.0 * x[0] + 3.0 * x[1])
        .hint(ProblemHint::Linear)
        .all_bounds(&[(0.0, 10.0), (0.0, 10.0)])
        .solve()
        .unwrap();

    assert!(result.converged, "did not converge: {}", result.message);
    assert!(result.x[0].abs() < 1e-3, "x0={}", result.x[0]);
    assert!(result.x[1].abs() < 1e-3, "x1={}", result.x[1]);
    assert!(result.f.abs() < 1e-3, "f={}", result.f);
}

#[test]
fn test_hint_quadratic_closure_solves_correctly() {
    // Quadratic problem: min (x0-3)^2 + (x1-4)^2, bounds [0,10]
    // Optimal at (3, 4) with f=0
    let result = OptimProblem::new(2)
        .x0(&[0.0, 0.0])
        .objective(|x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2))
        .hint(ProblemHint::Quadratic)
        .all_bounds(&[(0.0, 10.0), (0.0, 10.0)])
        .solve()
        .unwrap();

    assert!(result.converged, "did not converge: {}", result.message);
    assert!((result.x[0] - 3.0).abs() < 1e-3, "x0={}", result.x[0]);
    assert!((result.x[1] - 4.0).abs() < 1e-3, "x1={}", result.x[1]);
}

#[test]
fn test_hint_linear_selects_lbfgsb() {
    let p = OptimProblem::new(2)
        .x0(&[1.0, 1.0])
        .objective(|x: &[f64]| x[0] + x[1])
        .hint(ProblemHint::Linear)
        .all_bounds(&[(0.0, 10.0), (0.0, 10.0)]);
    assert_eq!(numra_optim::auto::select_solver(&p), SolverChoice::LbfgsB,);
}

#[test]
fn test_hint_quadratic_constrained_selects_sqp() {
    let p = OptimProblem::new(2)
        .x0(&[1.0, 1.0])
        .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
        .hint(ProblemHint::Quadratic)
        .constraint_ineq(|x: &[f64]| x[0] + x[1] - 1.0);
    assert_eq!(numra_optim::auto::select_solver(&p), SolverChoice::Sqp,);
}

// ── Integer rounding tests ──

#[test]
fn test_integer_rounding_nlp() {
    // Nonlinear objective with integer vars: min (x0-2.7)^2 + (x1-3.3)^2
    // x0 integer, x1 continuous, bounds [0,10]
    // NLP solution ~(2.7, 3.3), after rounding x0 should be 3
    let result = OptimProblem::new(2)
        .x0(&[1.0, 1.0])
        .objective(|x: &[f64]| (x[0] - 2.7).powi(2) + (x[1] - 3.3).powi(2))
        .integer_var(0)
        .all_bounds(&[(0.0, 10.0), (0.0, 10.0)])
        .solve()
        .unwrap();

    // x0 must be integer
    assert!(
        (result.x[0] - result.x[0].round()).abs() < 1e-9,
        "x0={} should be integer",
        result.x[0]
    );
    assert!(
        (result.x[0] - 3.0).abs() < 1e-9,
        "x0={}, expected 3.0",
        result.x[0]
    );
    // x1 continuous, should be near 3.3
    assert!(
        (result.x[1] - 3.3).abs() < 0.1,
        "x1={}, expected ~3.3",
        result.x[1]
    );
}

#[test]
fn test_binary_rounding_nlp() {
    // Nonlinear objective with binary var: min (x0-0.8)^2 + (x1-0.2)^2
    // Both binary -> should round to x0=1, x1=0
    let result = OptimProblem::new(2)
        .x0(&[0.5, 0.5])
        .objective(|x: &[f64]| (x[0] - 0.8).powi(2) + (x[1] - 0.2).powi(2))
        .binary_var(0)
        .binary_var(1)
        .solve()
        .unwrap();

    assert!(
        result.x[0] == 0.0 || result.x[0] == 1.0,
        "x0={} should be binary",
        result.x[0]
    );
    assert!(
        result.x[1] == 0.0 || result.x[1] == 1.0,
        "x1={} should be binary",
        result.x[1]
    );
    assert!(
        (result.x[0] - 1.0).abs() < 1e-9,
        "x0={}, expected 1",
        result.x[0]
    );
    assert!(result.x[1].abs() < 1e-9, "x1={}, expected 0", result.x[1]);
    assert!(
        result.message.contains("integer vars rounded"),
        "message should note rounding: {}",
        result.message
    );
}
