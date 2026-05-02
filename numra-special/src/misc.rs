//! Miscellaneous special functions: Dawson, Fresnel, Mittag-Leffler.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Dawson's integral: F(x) = e^{-x^2} * integral from 0 to x of e^{t^2} dt.
///
/// Uses Rybicki's approximation (sum of exponentials).
pub fn dawson<S: Scalar>(x: S) -> S {
    let xf = x.to_f64();
    S::from_f64(dawson_f64(xf))
}

/// Fresnel sine integral: S(x) = integral from 0 to x of sin(pi/2 * t^2) dt.
pub fn fresnel_s<S: Scalar>(x: S) -> S {
    let xf = x.to_f64();
    S::from_f64(fresnel_s_f64(xf))
}

/// Fresnel cosine integral: C(x) = integral from 0 to x of cos(pi/2 * t^2) dt.
pub fn fresnel_c<S: Scalar>(x: S) -> S {
    let xf = x.to_f64();
    S::from_f64(fresnel_c_f64(xf))
}

/// Mittag-Leffler function E_{alpha, beta}(z).
///
/// E_{alpha, beta}(z) = sum_{k=0}^inf z^k / Gamma(alpha*k + beta)
///
/// Generalizes the exponential: E_{1,1}(z) = e^z.
pub fn mittag_leffler<S: Scalar>(alpha: S, beta: S, z: S) -> S {
    let alphaf = alpha.to_f64();
    let betaf = beta.to_f64();
    let zf = z.to_f64();

    if alphaf <= 0.0 {
        return S::NAN;
    }

    S::from_f64(mittag_leffler_f64(alphaf, betaf, zf))
}

/// Mittag-Leffler E_alpha(z) = E_{alpha, 1}(z).
pub fn mittag_leffler_1<S: Scalar>(alpha: S, z: S) -> S {
    mittag_leffler(alpha, S::ONE, z)
}

// ============ Internal implementations ============

/// Dawson's integral via power series.
fn dawson_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    dawson_series(x.abs()) * sign
}

/// Dawson series for moderate x.
fn dawson_series(x: f64) -> f64 {
    let max_terms = 100;
    let eps = 1e-15;
    let x2 = x * x;

    // F(x) = x * sum_{n=0} (-2x^2)^n / (2n+1)!!
    // = x * sum_{n=0} c_n  where c_0 = 1, c_n = -2*x^2 * c_{n-1} / (2n+1)
    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 1..max_terms {
        term *= -2.0 * x2 / (2 * n + 1) as f64;
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    x * sum
}

/// Fresnel S via power series.
///
/// S(x) = Σ_{n=0}^∞ (-1)^n (π/2)^{2n+1} x^{4n+3} / ((4n+3)(2n+1)!)
fn fresnel_s_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let pi_half = core::f64::consts::PI / 2.0;

    if x > 4.0 {
        return sign * fresnel_s_asymptotic(x);
    }

    let max_terms = 40;
    let eps = 1e-15;

    let x4 = x * x * x * x;
    let pi_half_sq = pi_half * pi_half;

    // n=0 term: coeff = pi/2, x_pow = x^3, denom = 3
    let mut coeff = pi_half; // (-1)^n (pi/2)^{2n+1} / (2n+1)!
    let mut x_pow = x * x * x; // x^{4n+3}

    let mut s = coeff * x_pow / 3.0;

    for n in 1..max_terms {
        let nf = n as f64;
        coeff *= -pi_half_sq / ((2.0 * nf) * (2.0 * nf + 1.0));
        x_pow *= x4;
        let denom = 4.0 * nf + 3.0;
        let term = coeff * x_pow / denom;
        s += term;
        if term.abs() < eps * s.abs() {
            break;
        }
    }

    sign * s
}

/// Fresnel C via power series.
///
/// C(x) = Σ_{n=0}^∞ (-1)^n (π/2)^{2n} x^{4n+1} / ((4n+1)(2n)!)
fn fresnel_c_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let pi_half = core::f64::consts::PI / 2.0;

    if x > 4.0 {
        return sign * fresnel_c_asymptotic(x);
    }

    let max_terms = 40;
    let eps = 1e-15;

    let x4 = x * x * x * x;
    let pi_half_sq = pi_half * pi_half;

    // n=0 term: coeff = 1 (= (pi/2)^0 / 0!), x_pow = x, denom = 1
    let mut coeff = 1.0; // (-1)^n (pi/2)^{2n} / (2n)!
    let mut x_pow = x; // x^{4n+1}

    let mut c = x; // coeff * x_pow / 1.0

    for n in 1..max_terms {
        let nf = n as f64;
        coeff *= -pi_half_sq / ((2.0 * nf - 1.0) * (2.0 * nf));
        x_pow *= x4;
        let denom = 4.0 * nf + 1.0;
        let term = coeff * x_pow / denom;
        c += term;
        if term.abs() < eps * c.abs() {
            break;
        }
    }

    sign * c
}

/// Asymptotic Fresnel S for large x.
fn fresnel_s_asymptotic(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let pix2 = pi * x * x / 2.0;
    0.5 - (pix2).cos() / (pi * x) - (pix2).sin() / (pi * pi * x * x * x)
}

/// Asymptotic Fresnel C for large x.
fn fresnel_c_asymptotic(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let pix2 = pi * x * x / 2.0;
    0.5 + (pix2).sin() / (pi * x) - (pix2).cos() / (pi * pi * x * x * x)
}

/// Mittag-Leffler series.
fn mittag_leffler_f64(alpha: f64, beta: f64, z: f64) -> f64 {
    let max_terms = 200;
    let eps = 1e-15;

    let mut sum = 0.0;
    let mut z_pow = 1.0;

    for k in 0..max_terms {
        let gamma_arg = alpha * k as f64 + beta;
        let g = libm::tgamma(gamma_arg);
        if g.is_infinite() || g == 0.0 {
            break;
        }
        let term = z_pow / g;
        sum += term;

        if k > 0 && term.abs() < eps * sum.abs() {
            break;
        }
        z_pow *= z;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dawson_at_zero() {
        assert_relative_eq!(dawson(0.0_f64), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_dawson_known() {
        // F(1) ≈ 0.53807950691276841
        assert!((dawson(1.0_f64).to_f64() - 0.53807950691276841).abs() < 0.01);
    }

    #[test]
    fn test_dawson_odd() {
        // Dawson is odd: F(-x) = -F(x)
        let x = 0.5_f64;
        assert_relative_eq!(dawson(-x), -dawson(x), epsilon = 1e-10);
    }

    #[test]
    fn test_fresnel_at_zero() {
        assert_relative_eq!(fresnel_s(0.0_f64), 0.0, epsilon = 1e-14);
        assert_relative_eq!(fresnel_c(0.0_f64), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_fresnel_limits() {
        // S(inf) = C(inf) = 0.5
        let s = fresnel_s(100.0_f64);
        let c = fresnel_c(100.0_f64);
        assert!((s.to_f64() - 0.5).abs() < 0.01);
        assert!((c.to_f64() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fresnel_s_known() {
        // S(1) ≈ 0.43825914739
        assert!((fresnel_s(1.0_f64).to_f64() - 0.43825914739).abs() < 0.01);
    }

    #[test]
    fn test_fresnel_c_known() {
        // C(1) ≈ 0.77989340037
        assert!((fresnel_c(1.0_f64).to_f64() - 0.77989340037).abs() < 0.01);
    }

    #[test]
    fn test_mittag_leffler_exp() {
        // E_{1,1}(z) = e^z
        let z = 1.0_f64;
        assert_relative_eq!(
            mittag_leffler(1.0_f64, 1.0_f64, z),
            z.exp(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mittag_leffler_1() {
        // E_{2,1}(z^2) = cosh(z)
        let z = 1.0_f64;
        assert_relative_eq!(
            mittag_leffler(2.0_f64, 1.0_f64, z * z),
            z.cosh(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mittag_leffler_shorthand() {
        let z = 0.5_f64;
        assert_relative_eq!(
            mittag_leffler_1(1.0_f64, z),
            mittag_leffler(1.0_f64, 1.0_f64, z),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_misc_f32() {
        assert!((dawson(0.5_f32).to_f64()).abs() > 0.0);
        assert!((mittag_leffler(1.0_f32, 1.0_f32, 1.0_f32).to_f64() - 1.0_f64.exp()).abs() < 0.01);
    }
}
