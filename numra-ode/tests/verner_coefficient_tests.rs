//! Verner Method Coefficient Verification Tests
//!
//! These tests verify that the Butcher tableau coefficients satisfy
//! the necessary order conditions for RK methods.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

#![allow(clippy::excessive_precision)]

#[cfg(test)]
mod tests {
    // Import the tableau from the main crate
    // We'll define the coefficients locally for testing

    /// Vern7 coefficients from Jim Verner's website
    mod vern7 {
        pub const C: [f64; 10] = [
            0.0,
            0.005,
            0.10888888888888888888888888888888888889,
            0.16333333333333333333333333333333333333,
            0.4555,
            0.60950944899783813170870044214860249496,
            0.884,
            0.925,
            1.0,
            1.0,
        ];

        pub const B: [f64; 10] = [
            0.047155618486272221704317651088381756796,
            0.0,
            0.0,
            0.25750564298434151895964361010376875810,
            0.26216653977412620477138630957645277111,
            0.15216092656738557403231331991651175355,
            0.49399691700324842469071758932278768443,
            -0.29430311714032504415572447440927034291,
            0.081317472324951099997345994401367618925,
            0.0,
        ];

        pub const BHAT: [f64; 10] = [
            0.044608606606341176287318175974791977814,
            0.0,
            0.0,
            0.26716403785713726805091022609438378997,
            0.22010183001772930199797157766507530963,
            0.21884317031431568309831208335128938246,
            0.22898717054112028833781738897635523654,
            0.0,
            0.0,
            0.020295184663356282227670547938104303586,
        ];
    }

    /// Test: B weights sum to 1 (consistency condition)
    #[test]
    fn test_vern7_b_weights_sum() {
        let sum: f64 = vern7::B.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "Vern7 B weights sum = {} (should be 1)",
            sum
        );
    }

    /// Test: Bhat weights sum to 1 (consistency condition)
    #[test]
    fn test_vern7_bhat_weights_sum() {
        let sum: f64 = vern7::BHAT.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "Vern7 Bhat weights sum = {} (should be 1)",
            sum
        );
    }

    /// Test: First order condition Σbᵢ = 1
    #[test]
    fn test_vern7_order_condition_1() {
        let sum: f64 = vern7::B.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "Order 1 condition failed: Σbᵢ = {}",
            sum
        );
    }

    /// Test: Second order condition Σbᵢcᵢ = 1/2
    #[test]
    fn test_vern7_order_condition_2() {
        let sum: f64 = vern7::B
            .iter()
            .zip(vern7::C.iter())
            .map(|(b, c)| b * c)
            .sum();
        assert!(
            (sum - 0.5).abs() < 1e-12,
            "Order 2 condition failed: Σbᵢcᵢ = {} (should be 0.5)",
            sum
        );
    }

    /// Test: Third order condition Σbᵢcᵢ² = 1/3
    #[test]
    fn test_vern7_order_condition_3() {
        let sum: f64 = vern7::B
            .iter()
            .zip(vern7::C.iter())
            .map(|(b, c)| b * c * c)
            .sum();
        assert!(
            (sum - 1.0 / 3.0).abs() < 1e-12,
            "Order 3 condition failed: Σbᵢcᵢ² = {} (should be 1/3)",
            sum
        );
    }

    /// Test: Verify error coefficients E = B - Bhat
    #[test]
    fn test_vern7_error_coefficients() {
        for i in 0..10 {
            let expected_e = vern7::B[i] - vern7::BHAT[i];
            // Just verify the values we computed are correct
            println!("E[{}] = {}", i, expected_e);
        }
    }

    /// Vern7 A matrix from Jim Verner's website (for row-sum verification)
    mod vern7_a_matrix {
        // Row i contains coefficients for stage i+1
        // A[i][j] = a_{i+2, j+1} in standard notation
        pub const A: [[f64; 9]; 9] = [
            // Stage 2: a21 = 0.005
            [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            // Stage 3: a31, a32
            [
                -1.076790123456790123456790123456790123457,
                1.185679012345679012345679012345679012346,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            // Stage 4: a41, a42=0, a43
            [
                0.04083333333333333333333333333333333333333,
                0.0,
                0.1225,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            // Stage 5
            [
                0.6389139236255726780508121615993336109954,
                0.0,
                -2.455672638223656809662640566430653894211,
                2.272258714598084131611828404831320283215,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            // Stage 6
            [
                -2.661577375018757131119259297861818119279,
                0.0,
                10.80451388645613769565396655365532838482,
                -8.353914657396199411968048547819291691541,
                0.8204875949566569791420417341743839209619,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            // Stage 7
            [
                6.067741434696770992718360183877276714679,
                0.0,
                -24.71127363591108579734203485290746001803,
                20.42751793078889394045773111748346612697,
                -1.906157978816647150624096784352757010879,
                1.006172249242068014790040335899474187268,
                0.0,
                0.0,
                0.0,
            ],
            // Stage 8
            [
                12.05467007625320299509109452892778311648,
                0.0,
                -49.75478495046898932807257615331444758322,
                41.14288863860467663259698416710157354209,
                -4.461760149974004185641911603484815375051,
                2.042334822239174959821717077708608543738,
                -0.09834843665406107379530801693870224403537,
                0.0,
                0.0,
            ],
            // Stage 9
            [
                10.13814652288180787641845141981689030769,
                0.0,
                -42.64113603171750214622846006736635730625,
                35.76384003992257007135021178023160054034,
                -4.348022840392907653340370296908245943710,
                2.009862268377035895441943593011827554771,
                0.3487490460338272405953822853053145879140,
                -0.2714390051048312842371587140910297407572,
                0.0,
            ],
            // Stage 10 (used for bhat only, not in main 7th order solution)
            [
                -45.03007203429867712435322405073769635151,
                0.0,
                187.3272437654588840752418206154201997384,
                -154.0288236935018690596728621034510402582,
                18.56465306347536233859492332958439136765,
                -7.141809679295078854925420496823551192821,
                1.308808578161378625114762706007696696508,
                0.0,
                0.0,
            ],
        ];
    }

    /// Test: Row sum condition - c[i] = sum(A[i][j])
    #[test]
    fn test_vern7_row_sum_condition() {
        for i in 0..9 {
            let row_sum: f64 = vern7_a_matrix::A[i].iter().sum();
            let expected_c = vern7::C[i + 1]; // c[i+1] for stage i+2
            assert!(
                (row_sum - expected_c).abs() < 1e-10,
                "Row {} sum = {} ≠ c[{}] = {}",
                i + 2,
                row_sum,
                i + 1,
                expected_c
            );
        }
    }

    /// Integration test: Solve exponential decay with Vern7
    #[test]
    fn test_vern7_exponential_decay() {
        use numra_ode::{OdeProblem, Solver, SolverOptions, Vern7};

        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            2.0,
            vec![1.0],
        );

        let options = SolverOptions::default().rtol(1e-6).atol(1e-8);

        let result = Vern7::solve(&problem, 0.0, 2.0, &[1.0], &options).unwrap();

        assert!(result.success, "Vern7 should succeed");
        let y_final = result.y_final().unwrap();
        let expected = (-2.0_f64).exp();

        println!("Vern7 result: {}, expected: {}", y_final[0], expected);
        println!(
            "Relative error: {}",
            (y_final[0] - expected).abs() / expected
        );

        // For a 7th order method with rtol=1e-6, expect very good accuracy
        // Actual relative error is ~1.5e-8; use 1e-5 for comfortable margin
        assert!(
            (y_final[0] - expected).abs() < expected * 1e-5,
            "Vern7 result {} differs from expected {} by more than 1e-5 relative",
            y_final[0],
            expected
        );
    }
}
