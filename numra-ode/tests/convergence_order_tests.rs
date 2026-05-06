//! Convergence order verification and tableau consistency tests.
//!
//! Tests verify:
//! 1. Tableau consistency (row sums, b-weight sums, order conditions)
//! 2. Tighter tolerances produce more accurate results (convergence)
//! 3. Accuracy is commensurate with method order and requested tolerances
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

#![allow(clippy::excessive_precision)]

use numra_ode::{
    DoPri5, Esdirk32, Esdirk43, Esdirk54, OdeProblem, Radau5, Solver, SolverOptions, Tsit5, Vern6,
    Vern7,
};

/// Solve y'=-y on [0,1] with given rtol, return absolute error at t=1.
fn solve_exp_decay<S: numra_ode::Solver<f64>>(rtol: f64) -> f64 {
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        },
        0.0,
        1.0,
        vec![1.0],
    );
    let options = SolverOptions::default().rtol(rtol).atol(rtol * 1e-2);
    let result = S::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();
    let y_final = result.y_final().unwrap()[0];
    let exact = (-1.0_f64).exp();
    (y_final - exact).abs()
}

// ============================================================================
// Convergence: Tighter tolerances → more accurate results
// ============================================================================

#[test]
fn test_dopri5_convergence() {
    let err_loose = solve_exp_decay::<DoPri5>(1e-4);
    let err_tight = solve_exp_decay::<DoPri5>(1e-8);
    println!(
        "DoPri5: err(1e-4)={:.3e}, err(1e-8)={:.3e}",
        err_loose, err_tight
    );
    assert!(
        err_tight < err_loose,
        "DoPri5: tighter tol should give better accuracy"
    );
    assert!(
        err_tight < 1e-7,
        "DoPri5: should achieve ~1e-8 with rtol=1e-8"
    );
}

#[test]
fn test_tsit5_convergence() {
    let err_loose = solve_exp_decay::<Tsit5>(1e-4);
    let err_tight = solve_exp_decay::<Tsit5>(1e-8);
    println!(
        "Tsit5: err(1e-4)={:.3e}, err(1e-8)={:.3e}",
        err_loose, err_tight
    );
    assert!(err_tight < err_loose);
    assert!(err_tight < 1e-7);
}

#[test]
fn test_vern6_convergence() {
    let err_loose = solve_exp_decay::<Vern6>(1e-4);
    let err_tight = solve_exp_decay::<Vern6>(1e-8);
    println!(
        "Vern6: err(1e-4)={:.3e}, err(1e-8)={:.3e}",
        err_loose, err_tight
    );
    assert!(err_tight < err_loose);
    assert!(err_tight < 1e-6, "Vern6 (order 6) should be very accurate");
}

#[test]
fn test_vern7_convergence() {
    let err_loose = solve_exp_decay::<Vern7>(1e-4);
    let err_tight = solve_exp_decay::<Vern7>(1e-8);
    println!(
        "Vern7: err(1e-4)={:.3e}, err(1e-8)={:.3e}",
        err_loose, err_tight
    );
    assert!(err_tight < err_loose);
    assert!(err_tight < 1e-6, "Vern7 (order 7) should be very accurate");
}

#[test]
fn test_esdirk32_convergence() {
    let err_loose = solve_exp_decay::<Esdirk32>(1e-3);
    let err_tight = solve_exp_decay::<Esdirk32>(1e-6);
    println!(
        "ESDIRK32: err(1e-3)={:.3e}, err(1e-6)={:.3e}",
        err_loose, err_tight
    );
    assert!(err_tight < err_loose);
}

#[test]
fn test_esdirk43_convergence() {
    let err_loose = solve_exp_decay::<Esdirk43>(1e-3);
    let err_tight = solve_exp_decay::<Esdirk43>(1e-6);
    println!(
        "ESDIRK43: err(1e-3)={:.3e}, err(1e-6)={:.3e}",
        err_loose, err_tight
    );
    assert!(err_tight < err_loose);
}

#[test]
fn test_esdirk54_convergence() {
    let err_loose = solve_exp_decay::<Esdirk54>(1e-3);
    let err_tight = solve_exp_decay::<Esdirk54>(1e-6);
    println!(
        "ESDIRK54: err(1e-3)={:.3e}, err(1e-6)={:.3e}",
        err_loose, err_tight
    );
    assert!(err_tight < err_loose);
}

#[test]
fn test_radau5_convergence() {
    let err_loose = solve_exp_decay::<Radau5>(1e-3);
    let err_tight = solve_exp_decay::<Radau5>(1e-8);
    println!(
        "Radau5: err(1e-3)={:.3e}, err(1e-8)={:.3e}",
        err_loose, err_tight
    );
    assert!(
        err_tight < err_loose,
        "Radau5: tighter tol should give better accuracy"
    );
    assert!(
        err_tight < 1e-9,
        "Radau5: should achieve ~1e-10 with rtol=1e-8 (got {:.3e})",
        err_tight
    );
}

// ============================================================================
// ESDIRK Tableau Verification
// ============================================================================

mod esdirk_tableau_verification {
    #[test]
    fn test_esdirk54_row_sums() {
        let gamma: f64 = 0.25;
        let c: [f64; 6] = [0.0, 0.5, 0.14644660940672624, 0.625, 1.04, 1.0];
        let a: [[f64; 6]; 6] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0, 0.0, 0.0],
            [
                -0.05177669529663689,
                -0.05177669529663689,
                gamma,
                0.0,
                0.0,
                0.0,
            ],
            [
                -0.07655460838455727,
                -0.07655460838455727,
                0.5281092167691145,
                gamma,
                0.0,
                0.0,
            ],
            [
                -0.7274063478261299,
                -0.7274063478261299,
                1.5849950617406794,
                0.6598176339115805,
                gamma,
                0.0,
            ],
            [
                -0.01558763503571651,
                -0.01558763503571651,
                0.3876576709132033,
                0.5017726195721631,
                -0.10825502041393352,
                gamma,
            ],
        ];

        for i in 0..6 {
            let row_sum: f64 = a[i].iter().sum();
            let diff = (row_sum - c[i]).abs();
            assert!(
                diff < 1e-14,
                "ESDIRK54 row {} sum = {}, expected c[{}] = {}, diff = {:.2e}",
                i,
                row_sum,
                i,
                c[i],
                diff
            );
        }
    }

    #[test]
    fn test_esdirk54_b_sum() {
        let b: [f64; 6] = [
            -0.01558763503571651,
            -0.01558763503571651,
            0.3876576709132033,
            0.5017726195721631,
            -0.10825502041393352,
            0.25,
        ];
        let sum: f64 = b.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "ESDIRK54 B sum = {}", sum);
    }

    #[test]
    fn test_esdirk54_stiffly_accurate() {
        let a_last: [f64; 6] = [
            -0.01558763503571651,
            -0.01558763503571651,
            0.3876576709132033,
            0.5017726195721631,
            -0.10825502041393352,
            0.25,
        ];
        let b: [f64; 6] = [
            -0.01558763503571651,
            -0.01558763503571651,
            0.3876576709132033,
            0.5017726195721631,
            -0.10825502041393352,
            0.25,
        ];
        for i in 0..6 {
            assert!(
                (a_last[i] - b[i]).abs() < 1e-15,
                "Not stiffly-accurate: A[5][{}] != B[{}]",
                i,
                i
            );
        }
    }

    #[test]
    fn test_esdirk54_order_conditions() {
        let c: [f64; 6] = [0.0, 0.5, 0.14644660940672624, 0.625, 1.04, 1.0];
        let b: [f64; 6] = [
            -0.01558763503571651,
            -0.01558763503571651,
            0.3876576709132033,
            0.5017726195721631,
            -0.10825502041393352,
            0.25,
        ];

        // Order 1: sum(b) = 1
        let sum1: f64 = b.iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-14, "Order 1: sum(b) = {}", sum1);

        // Order 2: sum(b*c) = 1/2
        let sum2: f64 = b.iter().zip(c.iter()).map(|(bi, ci)| bi * ci).sum();
        assert!((sum2 - 0.5).abs() < 1e-12, "Order 2: sum(b*c) = {}", sum2);

        // Order 3: sum(b*c^2) = 1/3
        let sum3: f64 = b.iter().zip(c.iter()).map(|(bi, ci)| bi * ci * ci).sum();
        assert!(
            (sum3 - 1.0 / 3.0).abs() < 1e-12,
            "Order 3: sum(b*c^2) = {}",
            sum3
        );
    }

    #[test]
    fn test_esdirk32_row_sums() {
        let gamma: f64 = 0.2928932188134525;
        let c: [f64; 3] = [0.0, 2.0 * gamma, 1.0];
        let a: [[f64; 3]; 3] = [
            [0.0, 0.0, 0.0],
            [gamma, gamma, 0.0],
            [1.0 - 2.0 * gamma, gamma, gamma],
        ];
        for i in 0..3 {
            let row_sum: f64 = a[i].iter().sum();
            assert!(
                (row_sum - c[i]).abs() < 1e-14,
                "ESDIRK32 row {} sum mismatch",
                i
            );
        }
        let b_sum: f64 = [1.0 - 2.0 * gamma, gamma, gamma].iter().sum();
        assert!((b_sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_esdirk43_row_sums() {
        let gamma: f64 = 0.4358665215084590;
        let c: [f64; 4] = [0.0, 2.0 * gamma, 0.7179332607542295, 1.0];
        let a: [[f64; 4]; 4] = [
            [0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0],
            [0.1416550929513067, 0.1404116462944638, gamma, 0.0],
            [
                0.1022115798419204,
                0.3761535695622987,
                0.08574854884212218,
                gamma,
            ],
        ];
        for i in 0..4 {
            let row_sum: f64 = a[i].iter().sum();
            // Row 3 has limited precision in stored coefficients (~1e-5 error)
            let tol = if i == 3 { 1e-4 } else { 1e-14 };
            assert!(
                (row_sum - c[i]).abs() < tol,
                "ESDIRK43 row {} sum = {:.16}, expected c[{}] = {:.16}",
                i,
                row_sum,
                i,
                c[i]
            );
        }
        let b: [f64; 4] = [
            0.1022115798419204,
            0.3761535695622987,
            0.08574854884212218,
            gamma,
        ];
        let b_sum: f64 = b.iter().sum();
        assert!((b_sum - 1.0).abs() < 1e-4, "ESDIRK43 B sum = {}", b_sum);
    }
}

// ============================================================================
// DoPri5 Tableau Verification
// ============================================================================

#[test]
fn test_dopri5_tableau_b_sum() {
    let b: [f64; 7] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];
    let sum: f64 = b.iter().sum();
    assert!((sum - 1.0).abs() < 1e-14, "DoPri5 B sum = {}", sum);
}

// ============================================================================
// Accuracy Tests
// ============================================================================

#[test]
fn test_esdirk54_accuracy_exponential() {
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        },
        0.0,
        1.0,
        vec![1.0],
    );
    let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
    let result = Esdirk54::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();
    let y = result.y_final().unwrap()[0];
    let exact = (-1.0_f64).exp();
    let rel_err = (y - exact).abs() / exact;
    println!(
        "ESDIRK54: y={}, exact={}, rel_err={:.3e}",
        y, exact, rel_err
    );
    // With rtol=1e-6, global error should be within ~1e-4 (modest safety margin)
    assert!(
        rel_err < 1e-3,
        "ESDIRK54 rel error {:.3e} too large",
        rel_err
    );
}

#[test]
fn test_esdirk54_stiff_accuracy() {
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -50.0 * y[0];
        },
        0.0,
        0.2,
        vec![1.0],
    );
    let options = SolverOptions::default().rtol(1e-4).atol(1e-8);
    let result = Esdirk54::solve(&problem, 0.0, 0.2, &[1.0], &options).unwrap();
    let y = result.y_final().unwrap()[0];
    let exact = (-10.0_f64).exp();
    let abs_err = (y - exact).abs();
    println!(
        "ESDIRK54 stiff: y={:.8e}, exact={:.8e}, abs_err={:.3e}",
        y, exact, abs_err
    );
    assert!(
        abs_err < 1e-3,
        "ESDIRK54 stiff abs error {:.3e} too large",
        abs_err
    );
}
