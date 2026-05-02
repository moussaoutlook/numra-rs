//! Consistent initialization for DAEs.
//!
//! For differential-algebraic equations (DAEs), the initial conditions must satisfy
//! the algebraic constraints. This module provides functions to compute consistent
//! initial conditions for index-1 DAEs.
//!
//! # Background
//!
//! A DAE in semi-explicit form is:
//! ```text
//! y₁' = f₁(t, y₁, y₂)    (differential equations)
//! 0   = g(t, y₁, y₂)      (algebraic constraints)
//! ```
//!
//! Given initial values (y₁⁰, y₂⁰) where y₂⁰ may not satisfy the constraint g = 0,
//! we solve for y₂ such that g(t₀, y₁⁰, y₂) = 0.
//!
//! # Example
//!
//! ```rust
//! use numra_ode::{DaeProblem, compute_consistent_initial};
//!
//! // Constraint: y₂ = y₁²
//! let dae = DaeProblem::new(
//!     |_t, y: &[f64], dydt: &mut [f64]| {
//!         dydt[0] = -y[0];                // y₁' = -y₁
//!         dydt[1] = y[1] - y[0] * y[0];   // 0 = y₂ - y₁² (residual)
//!     },
//!     |mass: &mut [f64]| {
//!         mass[0] = 1.0; mass[1] = 0.0;
//!         mass[2] = 0.0; mass[3] = 0.0;   // M[1,1] = 0 (algebraic)
//!     },
//!     0.0, 1.0,
//!     vec![2.0, 0.0],  // Inconsistent: y₂ should be 4
//!     vec![1],         // Index 1 is algebraic
//! );
//!
//! let y0 = compute_consistent_initial(&dae, 0.0, &[2.0, 0.0]).unwrap();
//! assert!((y0[1] - 4.0).abs() < 1e-10);  // Now consistent
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::OdeSystem;
use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization};

/// Compute consistent initial conditions for an index-1 DAE.
///
/// Given y0 with potentially inconsistent algebraic variables,
/// solve the algebraic constraints g(t0, y) = 0 to find consistent values.
///
/// Uses default tolerance of 1e-10 and max 50 iterations.
pub fn compute_consistent_initial<S, Sys>(system: &Sys, t0: S, y0: &[S]) -> Result<Vec<S>, String>
where
    S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    Sys: OdeSystem<S>,
{
    compute_consistent_initial_tol(system, t0, y0, S::from_f64(1e-10), 50)
}

/// Compute consistent initial conditions with user-specified tolerance.
///
/// # Algorithm
///
/// Uses Newton iteration with a full Jacobian on the algebraic variable subset:
/// 1. Keep differential variables fixed
/// 2. Evaluate residual g(t₀, y) on algebraic equations
/// 3. Build full Jacobian ∂g/∂y_alg via finite differences
/// 4. Solve J_alg * Δy_alg = -g using LU factorization
/// 5. Update y_alg += Δy_alg
/// 6. Repeat until convergence
///
/// This handles coupled algebraic constraints correctly (unlike a diagonal
/// Jacobian approach which fails when constraints share variables).
pub fn compute_consistent_initial_tol<S, Sys>(
    system: &Sys,
    t0: S,
    y0: &[S],
    tol: S,
    max_iter: usize,
) -> Result<Vec<S>, String>
where
    S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    Sys: OdeSystem<S>,
{
    // If not a DAE (no singular mass matrix), y0 is already consistent
    if !system.is_singular_mass() {
        return Ok(y0.to_vec());
    }

    let alg_indices = system.algebraic_indices();
    if alg_indices.is_empty() {
        return Ok(y0.to_vec());
    }

    let dim = y0.len();
    let n_alg = alg_indices.len();
    let mut y = y0.to_vec();
    let mut f = vec![S::ZERO; dim];
    let mut f_pert = vec![S::ZERO; dim];
    let fd_eps = S::from_f64(1e-7);

    for iter in 0..max_iter {
        // Evaluate the residual f(t, y)
        system.rhs(t0, &y, &mut f);

        // Check convergence: max residual on algebraic equations
        let mut max_residual = S::ZERO;
        for &i in &alg_indices {
            max_residual = max_residual.max(f[i].abs());
        }

        if max_residual < tol {
            return Ok(y);
        }

        // Build full Jacobian on the algebraic variable subset
        // J[row, col] = ∂f[alg_indices[row]] / ∂y[alg_indices[col]]
        let mut jac_data = vec![S::ZERO; n_alg * n_alg];

        for (col, &j) in alg_indices.iter().enumerate() {
            let y_orig = y[j];
            let h = fd_eps * (S::ONE + y_orig.abs());
            y[j] = y_orig + h;
            system.rhs(t0, &y, &mut f_pert);
            y[j] = y_orig;

            for (row, &i) in alg_indices.iter().enumerate() {
                jac_data[row * n_alg + col] = (f_pert[i] - f[i]) / h;
            }
        }

        // Build RHS: -g (negative residual on algebraic equations)
        let mut rhs_alg = vec![S::ZERO; n_alg];
        for (row, &i) in alg_indices.iter().enumerate() {
            rhs_alg[row] = -f[i];
        }

        // Solve J_alg * delta = -g using LU factorization
        let jac_mat = DenseMatrix::from_row_major(n_alg, n_alg, &jac_data);
        let lu = LUFactorization::new(&jac_mat).map_err(|e| {
            format!(
                "DAE Jacobian factorization failed at iteration {}: {}. \
                 System may be index-2 or higher.",
                iter, e
            )
        })?;

        let delta = lu.solve(&rhs_alg).map_err(|e| {
            format!(
                "DAE Jacobian solve failed at iteration {}: {}. \
                 Jacobian may be singular; system may be index-2 or higher.",
                iter, e
            )
        })?;

        // Update algebraic variables
        for (idx, &j) in alg_indices.iter().enumerate() {
            y[j] = y[j] + delta[idx];
        }
    }

    Err(format!(
        "Failed to find consistent initial conditions after {} iterations \
         (residual still > {})",
        max_iter,
        tol.to_f64()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DaeProblem;

    #[test]
    fn test_dae_consistent_init() {
        // Simple algebraic constraint: y2 = y1^2
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            0.0,
            1.0,
            vec![2.0, 0.0],
            vec![1],
        );

        let y0 = compute_consistent_initial(&dae, 0.0, &[2.0, 0.0]).unwrap();

        let constraint = y0[1] - y0[0].powi(2);
        assert!(
            constraint.abs() < 1e-10,
            "Constraint not satisfied: {}",
            constraint
        );
        assert!((y0[0] - 2.0).abs() < 1e-10, "y1 should be unchanged");
        assert!(
            (y0[1] - 4.0).abs() < 1e-10,
            "y2 should be 4.0, got {}",
            y0[1]
        );
    }

    #[test]
    fn test_dae_consistent_init_already_consistent() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            0.0,
            1.0,
            vec![2.0, 4.0],
            vec![1],
        );

        let y0 = compute_consistent_initial(&dae, 0.0, &[2.0, 4.0]).unwrap();
        assert!((y0[0] - 2.0).abs() < 1e-10);
        assert!((y0[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_dae_consistent_init_ode_passthrough() {
        use crate::OdeProblem;

        let ode = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );

        let y0 = compute_consistent_initial(&ode, 0.0, &[5.0]).unwrap();
        assert!((y0[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dae_consistent_init_multiple_algebraic() {
        // y1' = -y1
        // 0 = y2 - y1^2  (algebraic)
        // 0 = y3 - y1^3  (algebraic)
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
                dydt[2] = y[2] - y[0].powi(3);
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
                mass[4] = 0.0;
                mass[5] = 0.0;
                mass[6] = 0.0;
                mass[7] = 0.0;
                mass[8] = 0.0;
            },
            0.0,
            1.0,
            vec![3.0, 0.0, 0.0],
            vec![1, 2],
        );

        let y0 = compute_consistent_initial(&dae, 0.0, &[3.0, 0.0, 0.0]).unwrap();

        assert!((y0[0] - 3.0).abs() < 1e-10, "y1 unchanged");
        assert!((y0[1] - 9.0).abs() < 1e-10, "y2 = y1^2 = 9");
        assert!((y0[2] - 27.0).abs() < 1e-10, "y3 = y1^3 = 27");
    }

    #[test]
    fn test_dae_consistent_init_nonlinear_constraint() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = 1.0;
                dydt[1] = y[1] - y[0].sin();
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            0.0,
            1.0,
            vec![1.0, 0.0],
            vec![1],
        );

        let y0 = compute_consistent_initial(&dae, 0.0, &[1.0, 0.0]).unwrap();
        let expected_y2 = 1.0_f64.sin();
        assert!((y0[0] - 1.0).abs() < 1e-10, "y1 unchanged");
        assert!(
            (y0[1] - expected_y2).abs() < 1e-10,
            "y2 = sin(y1), got {} expected {}",
            y0[1],
            expected_y2
        );
    }

    #[test]
    fn test_dae_consistent_init_with_tolerance() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            0.0,
            1.0,
            vec![2.0, 0.0],
            vec![1],
        );

        let y0 = compute_consistent_initial_tol(&dae, 0.0, &[2.0, 0.0], 1e-12, 100).unwrap();
        let constraint = y0[1] - y0[0].powi(2);
        assert!(constraint.abs() < 1e-12, "Should satisfy tighter tolerance");
    }

    #[test]
    fn test_dae_init_coupled_constraints() {
        // DAE with two coupled algebraic constraints:
        // y1' = -y1
        // 0 = y2 + y3 - 1    (constraint 1)
        // 0 = y2 - y3        (constraint 2)
        // Solution: y2 = y3 = 0.5
        // A diagonal Jacobian cannot solve this because d(g1)/d(y2) and d(g1)/d(y3) both matter.
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] + y[2] - 1.0; // g1 = y2 + y3 - 1
                dydt[2] = y[1] - y[2]; // g2 = y2 - y3
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
                mass[4] = 0.0;
                mass[5] = 0.0;
                mass[6] = 0.0;
                mass[7] = 0.0;
                mass[8] = 0.0;
            },
            0.0,
            1.0,
            vec![1.0, 0.7, 0.2],
            vec![1, 2],
        );

        let y0 = compute_consistent_initial(&dae, 0.0, &[1.0, 0.7, 0.2]).unwrap();

        assert!((y0[0] - 1.0).abs() < 1e-10, "y1 unchanged");
        assert!(
            (y0[1] - 0.5).abs() < 1e-8,
            "y2 should be 0.5, got {}",
            y0[1]
        );
        assert!(
            (y0[2] - 0.5).abs() < 1e-8,
            "y3 should be 0.5, got {}",
            y0[2]
        );

        // Verify constraints
        let g1 = y0[1] + y0[2] - 1.0;
        let g2 = y0[1] - y0[2];
        assert!(g1.abs() < 1e-10, "Constraint 1 violated: {}", g1);
        assert!(g2.abs() < 1e-10, "Constraint 2 violated: {}", g2);
    }
}
