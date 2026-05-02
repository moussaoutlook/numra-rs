//! Error functions: erf, erfc, erfinv, erfcinv.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Error function erf(x).
///
/// Delegates to libm via the Scalar trait.
pub fn erf<S: Scalar>(x: S) -> S {
    x.erf_fn()
}

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// More accurate than `1 - erf(x)` for large x.
pub fn erfc<S: Scalar>(x: S) -> S {
    x.erfc_fn()
}

/// Inverse error function: erfinv(p) such that erf(erfinv(p)) = p.
///
/// Uses rational approximation from J.M. Blair (1976) with Newton refinement.
/// Valid for p in (-1, 1).
pub fn erfinv<S: Scalar>(p: S) -> S {
    let pf = p.to_f64();

    if pf.is_nan() || pf <= -1.0 || pf >= 1.0 {
        if pf == -1.0 {
            return S::NEG_INFINITY;
        }
        if pf == 1.0 {
            return S::INFINITY;
        }
        return S::NAN;
    }
    if pf == 0.0 {
        return S::ZERO;
    }

    S::from_f64(erfinv_f64(pf))
}

/// Inverse complementary error function: erfcinv(q) such that erfc(erfcinv(q)) = q.
///
/// Valid for q in (0, 2).
pub fn erfcinv<S: Scalar>(q: S) -> S {
    erfinv(S::ONE - q)
}

/// Rational approximation + Newton polish for erfinv.
/// Uses the algorithm from Winitzki (2008) as initial guess, then Halley refinement.
fn erfinv_f64(p: f64) -> f64 {
    // Winitzki's initial approximation:
    // erfinv(x) ~ sgn(x) * sqrt(sqrt((2/(pi*a) + ln(1-x^2)/2)^2 - ln(1-x^2)/a)
    //            - (2/(pi*a) + ln(1-x^2)/2))
    // where a = 0.147 (tuned constant)
    let a_const = 0.147;
    let pi = core::f64::consts::PI;
    let ln1mx2 = (1.0 - p * p).ln();
    let two_over_pi_a = 2.0 / (pi * a_const);
    let half_ln = ln1mx2 / 2.0;
    let inner = two_over_pi_a + half_ln;
    let outer = inner * inner - ln1mx2 / a_const;

    let sign = if p < 0.0 { -1.0 } else { 1.0 };
    let mut x = sign * (outer.sqrt() - inner).sqrt();

    // Newton-Raphson refinement: f(x) = erf(x) - p, f'(x) = 2/sqrt(pi) * e^{-x^2}
    let two_over_sqrtpi = 2.0 / pi.sqrt();
    for _ in 0..3 {
        let err = libm::erf(x) - p;
        let deriv = two_over_sqrtpi * (-x * x).exp();
        if deriv.abs() < 1e-300 {
            break;
        }
        x -= err / deriv;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_erf_values() {
        assert_relative_eq!(erf(0.0_f64), 0.0, epsilon = 1e-14);
        assert_relative_eq!(erf(1.0_f64), 0.8427007929497149, epsilon = 1e-12);
        assert_relative_eq!(erf(-1.0_f64), -0.8427007929497149, epsilon = 1e-12);
    }

    #[test]
    fn test_erfc_values() {
        assert_relative_eq!(erfc(0.0_f64), 1.0, epsilon = 1e-14);
        assert_relative_eq!(erfc(3.0_f64), 0.000022090496998585, epsilon = 1e-12);
    }

    #[test]
    fn test_erfinv_roundtrip() {
        for &p in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, -0.5, -0.9] {
            let x = erfinv(p as f64);
            let recovered = erf(x);
            assert_relative_eq!(recovered, p, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_erfinv_boundary() {
        assert_eq!(erfinv(0.0_f64), 0.0_f64);
        assert!(erfinv(1.0_f64).to_f64().is_infinite());
        assert!(erfinv(-1.0_f64).to_f64().is_infinite());
        assert!(erfinv(1.5_f64).to_f64().is_nan());
    }

    #[test]
    fn test_erfcinv_roundtrip() {
        for &q in &[0.1, 0.5, 1.0, 1.5, 1.9] {
            let x = erfcinv(q as f64);
            let recovered = erfc(x);
            assert_relative_eq!(recovered, q, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_erf_f32() {
        assert!((erf(1.0_f32) - 0.84270079_f32).abs() < 1e-5);
    }
}
