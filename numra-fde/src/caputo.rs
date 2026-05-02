//! Caputo fractional derivative utilities.
//!
//! The Caputo derivative of order α is:
//! D^α y(t) = (1/Γ(1-α)) ∫₀ᵗ (t-s)^(-α) y'(s) ds
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Compute the L1 scheme weights for Caputo derivative discretization.
///
/// The L1 scheme approximates D^α y(tₙ) as:
/// ```text
/// D^α y(tₙ) ≈ (Δt^(-α) / Γ(2-α)) Σⱼ₌₀ⁿ⁻¹ bⱼ (y(tₙ₋ⱼ) - y(tₙ₋ⱼ₋₁))
/// ```
///
/// where bⱼ = (j+1)^(1-α) - j^(1-α).
///
/// # Arguments
/// * `n` - Number of weights to compute
/// * `alpha` - Fractional order (0 < α < 1)
///
/// # Returns
/// Vector of weights b₀, b₁, ..., bₙ₋₁
pub fn caputo_weights<S: Scalar>(n: usize, alpha: S) -> Vec<S> {
    let one_minus_alpha = S::ONE - alpha;
    let mut b = Vec::with_capacity(n);

    for j in 0..n {
        let j_plus_1 = S::from_usize(j + 1);
        // For j = 0: b_0 = 1^(1-α) - 0^(1-α) = 1 - 0 = 1
        // (Since 0^x = 0 for any x > 0, and even at α=1 we want b_0 = 1)
        // For j > 0: b_j = (j+1)^(1-α) - j^(1-α)
        let bj = if j == 0 {
            S::ONE // Always 1 for j = 0
        } else {
            let j_s = S::from_usize(j);
            j_plus_1.powf(one_minus_alpha) - j_s.powf(one_minus_alpha)
        };
        b.push(bj);
    }

    b
}

/// Compute the coefficient for L1 scheme: Δt^(-α) / Γ(2-α).
pub fn l1_coefficient<S: Scalar>(dt: S, alpha: S) -> S {
    let two_minus_alpha = S::from_f64(2.0) - alpha;
    dt.powf(-alpha) / gamma(two_minus_alpha)
}

/// Gamma function approximation using Lanczos approximation.
///
/// Accurate for positive real arguments.
pub fn gamma<S: Scalar>(z: S) -> S {
    // Lanczos approximation coefficients (g=7)
    #[allow(clippy::excessive_precision)]
    let p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let z_f = z.to_f64();
    if z_f < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        let pi = std::f64::consts::PI;
        let sin_pz = (pi * z_f).sin();
        let g1mz = gamma(S::ONE - z);
        return S::from_f64(pi) / (S::from_f64(sin_pz) * g1mz);
    }

    let z_f = z_f - 1.0;
    let mut x = p[0];
    for (i, coeff) in p.iter().enumerate().skip(1) {
        x += coeff / (z_f + i as f64);
    }

    let t = z_f + 7.5;
    let result = (2.0 * std::f64::consts::PI).sqrt() * t.powf(z_f + 0.5) * (-t).exp() * x;
    S::from_f64(result)
}

/// Mittag-Leffler function E_{α,β}(z).
///
/// The solution to D^α y = λy, y(0) = y₀ is y(t) = y₀ * E_α(λ * t^α).
///
/// E_{α,β}(z) = Σₖ₌₀^∞ z^k / Γ(αk + β)
///
/// # Arguments
/// * `z` - Argument
/// * `alpha` - First parameter α > 0
/// * `beta` - Second parameter β > 0 (default 1.0)
/// * `max_terms` - Maximum terms for series (default 100)
pub fn mittag_leffler<S: Scalar>(z: S, alpha: S, beta: S, max_terms: usize) -> S {
    let tol = S::from_f64(1e-15);
    let mut sum = S::ZERO;
    let term = S::ONE / gamma(beta);
    sum += term;

    let mut z_pow = S::ONE;
    for k in 1..max_terms {
        z_pow *= z;
        let gamma_arg = alpha * S::from_usize(k) + beta;
        let new_term = z_pow / gamma(gamma_arg);

        sum += new_term;

        // Check convergence
        if new_term.abs() < tol * sum.abs() {
            break;
        }
    }

    sum
}

/// Mittag-Leffler function E_α(z) = E_{α,1}(z).
pub fn mittag_leffler_1<S: Scalar>(z: S, alpha: S) -> S {
    mittag_leffler(z, alpha, S::ONE, 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_integers() {
        // Γ(n) = (n-1)! for positive integers
        assert!((gamma(1.0_f64) - 1.0).abs() < 1e-10);
        assert!((gamma(2.0_f64) - 1.0).abs() < 1e-10); // Γ(2) = 1! = 1
        assert!((gamma(3.0_f64) - 2.0).abs() < 1e-10); // Γ(3) = 2! = 2
        assert!((gamma(4.0_f64) - 6.0).abs() < 1e-10); // Γ(4) = 3! = 6
        assert!((gamma(5.0_f64) - 24.0).abs() < 1e-9); // Γ(5) = 4! = 24
    }

    #[test]
    fn test_gamma_half() {
        // Γ(1/2) = √π
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert!((gamma(0.5_f64) - sqrt_pi).abs() < 1e-10);
    }

    #[test]
    fn test_caputo_weights() {
        // For α = 0.5, b_j = (j+1)^0.5 - j^0.5
        let alpha = 0.5_f64;
        let b = caputo_weights(5, alpha);

        assert!((b[0] - 1.0).abs() < 1e-10); // b_0 = 1^0.5 - 0^0.5 = 1
        assert!((b[1] - (2.0_f64.sqrt() - 1.0)).abs() < 1e-10);
        assert!((b[2] - (3.0_f64.sqrt() - 2.0_f64.sqrt())).abs() < 1e-10);
    }

    #[test]
    fn test_mittag_leffler_alpha_1() {
        // E_1(z) = exp(z)
        let z = 1.0_f64;
        let ml = mittag_leffler_1(z, 1.0);
        let exp_z = z.exp();
        assert!((ml - exp_z).abs() < 1e-8);
    }

    #[test]
    fn test_mittag_leffler_alpha_2() {
        // E_2(z) = cosh(√z) for z > 0
        let z = 1.0_f64;
        let ml = mittag_leffler_1(z, 2.0);
        let cosh_sqrt_z = z.sqrt().cosh();
        assert!((ml - cosh_sqrt_z).abs() < 1e-8);
    }

    #[test]
    fn test_mittag_leffler_negative() {
        // E_{0.5}(-1) should be a finite positive number < 1
        let ml = mittag_leffler_1(-1.0_f64, 0.5);
        assert!(ml > 0.0 && ml < 1.0);
    }

    #[test]
    fn test_caputo_weights_alpha_1() {
        // When α = 1, b_j = (j+1)^0 - j^0 = 1 - 1 = 0 for j > 0
        // b_0 is always 1
        let b = caputo_weights(5, 1.0_f64);
        assert!((b[0] - 1.0).abs() < 1e-10, "b[0] should be 1, got {}", b[0]);
        for j in 1..5 {
            assert!(b[j].abs() < 1e-10, "b[{}] should be 0, got {}", j, b[j]);
        }
    }

    #[test]
    fn test_l1_coefficient() {
        // l1_coefficient(dt, alpha) = dt^(-alpha) / Gamma(2 - alpha)
        let dt = 0.1_f64;
        let alpha = 0.5_f64;
        let c = l1_coefficient(dt, alpha);

        // Manual: dt^(-0.5) = 1/sqrt(0.1) = sqrt(10)
        // Gamma(1.5) = sqrt(pi)/2
        let expected = 10.0_f64.sqrt() / (std::f64::consts::PI.sqrt() / 2.0);
        assert!(
            (c - expected).abs() < 1e-10,
            "l1_coefficient: got {}, expected {}",
            c,
            expected
        );
    }

    #[test]
    fn test_mittag_leffler_e2_negative() {
        // E_2(-z) = cos(sqrt(z)) for z > 0
        // So E_2(-4) = cos(2)
        let z = -4.0_f64;
        let ml = mittag_leffler_1(z, 2.0);
        let expected = 2.0_f64.cos();
        assert!(
            (ml - expected).abs() < 1e-6,
            "E_2(-4) = cos(2): got {}, expected {}",
            ml,
            expected
        );
    }

    #[test]
    fn test_gamma_half_integer() {
        let sqrt_pi = std::f64::consts::PI.sqrt();

        // Gamma(3/2) = sqrt(pi)/2
        let g32 = gamma(1.5_f64);
        let expected_32 = sqrt_pi / 2.0;
        assert!(
            (g32 - expected_32).abs() < 1e-10,
            "Gamma(3/2): got {}, expected {}",
            g32,
            expected_32
        );

        // Gamma(5/2) = 3*sqrt(pi)/4
        let g52 = gamma(2.5_f64);
        let expected_52 = 3.0 * sqrt_pi / 4.0;
        assert!(
            (g52 - expected_52).abs() < 1e-10,
            "Gamma(5/2): got {}, expected {}",
            g52,
            expected_52
        );
    }
}
