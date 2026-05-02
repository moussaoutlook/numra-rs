//! BDF Algorithm Verification Tests
//!
//! These tests verify the mathematical correctness of the BDF implementation
//! by testing individual components before integration testing.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

/// Correct BDF coefficients from Wikipedia/SUNDIALS
/// Formula: α₀·y_{n+1} + α₁·y_n + α₂·y_{n-1} + ... = h·β·f(t_{n+1}, y_{n+1})
/// With normalization α₀ = 1
mod correct_coefficients {
    /// Alpha coefficients for BDF orders 1-5
    /// ALPHA[k-1][i] = coefficient for y_{n+1-i} in BDF-k
    pub const ALPHA: [[f64; 6]; 5] = [
        // BDF1: y_{n+1} - y_n = h*f
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        // BDF2: y_{n+2} - (4/3)*y_{n+1} + (1/3)*y_n = (2/3)*h*f
        [1.0, -4.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
        // BDF3: y_{n+3} - (18/11)*y_{n+2} + (9/11)*y_{n+1} - (2/11)*y_n = (6/11)*h*f
        [1.0, -18.0 / 11.0, 9.0 / 11.0, -2.0 / 11.0, 0.0, 0.0],
        // BDF4
        [
            1.0,
            -48.0 / 25.0,
            36.0 / 25.0,
            -16.0 / 25.0,
            3.0 / 25.0,
            0.0,
        ],
        // BDF5
        [
            1.0,
            -300.0 / 137.0,
            300.0 / 137.0,
            -200.0 / 137.0,
            75.0 / 137.0,
            -12.0 / 137.0,
        ],
    ];

    /// Beta coefficients (multiplier for h*f)
    /// β = 1 / (1 + 1/2 + 1/3 + ... + 1/k) for order k
    pub const BETA: [f64; 5] = [
        1.0,          // BDF1: β = 1
        2.0 / 3.0,    // BDF2: β = 2/3
        6.0 / 11.0,   // BDF3: β = 6/11
        12.0 / 25.0,  // BDF4: β = 12/25
        60.0 / 137.0, // BDF5: β = 60/137
    ];

    /// Error constants for local truncation error estimation
    /// C_{k+1} for BDF-k
    #[allow(dead_code)]
    pub const ERROR_CONST: [f64; 5] = [
        1.0 / 2.0,    // BDF1: 1/2
        2.0 / 9.0,    // BDF2: 2/9
        3.0 / 22.0,   // BDF3: 3/22
        12.0 / 125.0, // BDF4: 12/125
        10.0 / 137.0, // BDF5: 10/137
    ];
}

#[cfg(test)]
mod tests {
    use super::correct_coefficients::*;

    /// Test: BDF coefficients sum to zero (consistency condition)
    /// For a consistent method: Σ αᵢ = 0
    #[test]
    fn test_bdf_alpha_sum_to_zero() {
        for order in 1..=5 {
            let alpha = &ALPHA[order - 1];
            let sum: f64 = alpha.iter().take(order + 1).sum();
            assert!(
                sum.abs() < 1e-14,
                "BDF{} alpha sum = {} (should be 0)",
                order,
                sum
            );
        }
    }

    /// Test: BDF order condition on derivatives
    /// For order p method: Σ(-i)·αᵢ = β (first derivative consistency)
    #[test]
    fn test_bdf_first_order_condition() {
        for order in 1..=5 {
            let alpha = &ALPHA[order - 1];
            let beta = BETA[order - 1];

            // Sum of -i * alpha[i] for i = 0 to order
            let mut sum = 0.0;
            for (i, &a) in alpha.iter().enumerate().take(order + 1) {
                sum += -(i as f64) * a;
            }

            assert!(
                (sum - beta).abs() < 1e-14,
                "BDF{}: Σ(-i·αᵢ) = {} ≠ β = {}",
                order,
                sum,
                beta
            );
        }
    }

    /// Test: Single BDF1 step (backward Euler) analytical verification
    /// Problem: y' = -y, y(0) = 1
    /// BDF1: y₁ - y₀ = h·f(t₁, y₁) = h·(-y₁)
    /// => y₁ + h·y₁ = y₀ => y₁ = y₀/(1+h)
    #[test]
    fn test_bdf1_single_step_analytical() {
        let y0: f64 = 1.0;
        let h: f64 = 0.1;

        // Analytical BDF1 solution
        let y1_bdf = y0 / (1.0 + h);

        // True exponential solution
        let y1_exact = (-h).exp();

        // BDF1 should give y₁ ≈ 0.909...
        assert!(
            (y1_bdf - 1.0 / 1.1).abs() < 1e-14,
            "BDF1 step: got {}, expected {}",
            y1_bdf,
            1.0 / 1.1
        );

        // Local truncation error should be O(h²)
        let lte = (y1_bdf - y1_exact).abs();
        let expected_lte_order = h * h * 0.5; // Leading term of error
        assert!(
            lte < h * h,
            "BDF1 LTE = {} should be O(h²) ≈ {}",
            lte,
            expected_lte_order
        );
    }

    /// Test: BDF2 single step analytical verification
    /// Problem: y' = -y with known y₀, y₁
    /// BDF2: y₂ - (4/3)y₁ + (1/3)y₀ = (2/3)h·f(t₂, y₂)
    #[test]
    fn test_bdf2_single_step_analytical() {
        let h: f64 = 0.1;
        let y0: f64 = 1.0;
        let y1: f64 = (-h).exp(); // True solution at t=h

        // BDF2: y₂ - (4/3)y₁ + (1/3)y₀ = (2/3)h·(-y₂)
        // => y₂·(1 + (2/3)h) = (4/3)y₁ - (1/3)y₀
        // => y₂ = ((4/3)y₁ - (1/3)y₀) / (1 + (2/3)h)
        let rhs = (4.0 / 3.0) * y1 - (1.0 / 3.0) * y0;
        let coeff = 1.0 + (2.0 / 3.0) * h;
        let y2_bdf = rhs / coeff;

        let y2_exact = (-2.0 * h).exp();

        // Error should be O(h³) for BDF2
        let lte = (y2_bdf - y2_exact).abs();
        assert!(lte < h * h * h, "BDF2 LTE = {} should be O(h³)", lte);
    }

    /// Test: Newton iteration matrix structure
    /// The iteration matrix is M = I - γJ where γ = h·β
    /// For scalar y' = λy, J = λ, so M = 1 - h·β·λ
    #[test]
    fn test_newton_iteration_matrix() {
        let h = 0.1;
        let lambda = -1.0; // From y' = -y

        for order in 1..=5 {
            let beta = BETA[order - 1];
            let gamma = h * beta;
            let m = 1.0 - gamma * lambda;

            // For stability, need |1 - γλ| to be bounded
            // With λ = -1, we get m = 1 + γ > 1 (always well-conditioned)
            assert!(
                m > 0.0,
                "BDF{} iteration matrix = {} should be positive for λ = -1",
                order,
                m
            );
        }
    }

    /// Test: Predictor extrapolates from history
    /// Predictor y⁰_{n+1} = -Σ αᵢ·y_{n+1-i} / α₀ for i ≥ 1
    ///
    /// Note: The predictor is meant to be a starting point for Newton iteration,
    /// not a highly accurate estimate. Its purpose is to be "close enough" that
    /// Newton converges quickly.
    #[test]
    fn test_predictor_extrapolation() {
        // For exponential decay with history y_n = exp(-n·h)
        let h: f64 = 0.1;

        // BDF1 predictor test (should be exact for linear extrapolation from one point)
        // With only y_n available, the predictor just uses y_n as the starting guess
        let alpha1 = &ALPHA[0];
        let y0: f64 = 1.0; // y at t=0
        let predictor1 = -alpha1[1] * y0; // -(-1) * 1 = 1
                                          // For BDF1, the predictor is just y_n
        assert!(
            (predictor1 - y0).abs() < 1e-14,
            "BDF1 predictor should be y_n"
        );

        // BDF2 predictor test
        // With history y_n, y_{n-1}, extrapolate to y_{n+1}
        // Linear extrapolation: y_{n+1} ≈ 2*y_n - y_{n-1}
        let y_n = (-h).exp();
        let y_n_1 = 1.0; // y at t=0

        let alpha2 = &ALPHA[1];
        let predictor2 = -(alpha2[1] * y_n + alpha2[2] * y_n_1);
        // Should be: -((-4/3)*y_n + (1/3)*y_{n-1}) = (4/3)*y_n - (1/3)*y_{n-1}

        let expected_linear = (4.0 / 3.0) * y_n - (1.0 / 3.0) * y_n_1;
        assert!(
            (predictor2 - expected_linear).abs() < 1e-14,
            "BDF2 predictor computation error"
        );

        // The predictor should be a reasonable estimate
        let y_true = (-2.0 * h).exp();
        let pred_error = (predictor2 - y_true).abs();
        // Predictor error is roughly O(h²) for a second-order method
        // With h=0.1, expect error ~ h² = 0.01, but extrapolation
        // from exponential can have larger error
        assert!(
            pred_error < 0.1, // Just needs to be close enough for Newton to converge
            "BDF2 predictor error {} should be reasonable",
            pred_error
        );
    }

    /// Test: Beta values follow the formula β = 1/(1 + 1/2 + ... + 1/k)
    #[test]
    fn test_beta_formula() {
        for order in 1..=5 {
            let harmonic_sum: f64 = (1..=order).map(|i| 1.0 / i as f64).sum();
            let expected_beta = 1.0 / harmonic_sum;
            let actual_beta = BETA[order - 1];

            assert!(
                (actual_beta - expected_beta).abs() < 1e-14,
                "BDF{} β = {} ≠ expected {}",
                order,
                actual_beta,
                expected_beta
            );
        }
    }

    /// Test: Full BDF1 Newton iteration for y' = -y
    /// This traces through the exact Newton iteration to verify correctness
    #[test]
    fn test_bdf1_newton_iteration_trace() {
        // Problem: dy/dt = -y, y(0) = 1
        // After step h, true solution is exp(-h)
        let h: f64 = 0.1;
        let y0: f64 = 1.0;
        let lambda: f64 = -1.0; // From f(t,y) = -y = λy

        // BDF1 coefficients (corrected)
        let alpha0 = ALPHA[0][0]; // 1.0
        let alpha1 = ALPHA[0][1]; // -1.0
        let beta = BETA[0]; // 1.0

        // Predictor: y_pred = -α₁*y₀/α₀ = -(-1)*1/1 = 1
        let y_pred = -alpha1 * y0 / alpha0;
        assert!((y_pred - 1.0).abs() < 1e-14, "BDF1 predictor should be y0");

        // Iteration matrix: M = I - h*β*J = 1 - h*β*λ = 1 - 0.1*1*(-1) = 1.1
        // (For scalar case, J = df/dy = λ)
        let m = 1.0 - h * beta * lambda;
        assert!((m - 1.1).abs() < 1e-14, "BDF1 iteration matrix");

        // Newton iteration from predictor
        let mut y = y_pred;
        for iter in 0..5 {
            // f(t, y) = λy
            let f = lambda * y;

            // Residual: F(y) = α₀*y - h*β*f + α₁*y₀
            //                = 1*y - 0.1*1*(-y) + (-1)*1
            //                = y + 0.1*y - 1
            //                = 1.1*y - 1
            let residual = alpha0 * y - h * beta * f + alpha1 * y0;

            if residual.abs() < 1e-12 {
                break;
            }

            // Newton step: y_new = y - F(y)/M = y - residual/m
            let delta = residual / m;
            y -= delta;

            // Should converge in 1 iteration for linear problem
            if iter == 0 {
                // After first iteration, should be very close to exact
                let y_expected = 1.0 / 1.1;
                assert!(
                    (y - y_expected).abs() < 1e-12,
                    "BDF1 Newton iter 1: y={}, expected={}",
                    y,
                    y_expected
                );
            }
        }

        // Final solution should be 1/1.1
        let y_exact_bdf1 = y0 / (1.0 + h);
        assert!(
            (y - y_exact_bdf1).abs() < 1e-12,
            "BDF1 final: y={}, expected={}",
            y,
            y_exact_bdf1
        );
    }

    /// Integration test: Run actual BDF solver on exponential decay
    /// This directly tests the solver implementation
    #[test]
    fn test_bdf_solver_integration() {
        use numra_ode::{Bdf, OdeProblem, Solver, SolverOptions};

        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );

        // Use relaxed tolerances to ensure convergence
        let options = SolverOptions::default().rtol(1e-2).atol(1e-4).h0(0.01); // Start with small step

        let result = Bdf::solve(&problem, 0.0, 1.0, &[1.0], &options);

        match result {
            Ok(r) => {
                assert!(r.success, "BDF should succeed");
                let y_final = r.y_final().unwrap();
                let expected = (-1.0_f64).exp();
                // Allow 5% error for relaxed tolerances
                assert!(
                    (y_final[0] - expected).abs() < expected * 0.05,
                    "BDF result {} differs from expected {}",
                    y_final[0],
                    expected
                );
            }
            Err(e) => {
                panic!("BDF solver failed: {}", e);
            }
        }
    }
}
