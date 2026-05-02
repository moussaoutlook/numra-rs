//! Hypergeometric functions: confluent 1F1 and Gauss 2F1.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Confluent hypergeometric function 1F1(a; b; z) (Kummer's function M).
///
/// 1F1(a; b; z) = sum_{k=0}^inf (a)_k / (b)_k * z^k / k!
///
/// where (a)_k is the Pochhammer symbol (rising factorial).
pub fn hyp1f1<S: Scalar>(a: S, b: S, z: S) -> S {
    let af = a.to_f64();
    let bf = b.to_f64();
    let zf = z.to_f64();

    if bf <= 0.0 && bf == bf.floor() {
        // b is a non-positive integer => pole (unless a is more negative integer)
        return S::NAN;
    }

    S::from_f64(hyp1f1_f64(af, bf, zf))
}

/// Gauss hypergeometric function 2F1(a, b; c; z).
///
/// 2F1(a, b; c; z) = sum_{k=0}^inf (a)_k (b)_k / (c)_k * z^k / k!
///
/// Converges for |z| < 1. For |z| >= 1, uses analytic continuation.
pub fn hyp2f1<S: Scalar>(a: S, b: S, c: S, z: S) -> S {
    let af = a.to_f64();
    let bf = b.to_f64();
    let cf = c.to_f64();
    let zf = z.to_f64();

    S::from_f64(hyp2f1_f64(af, bf, cf, zf))
}

/// 1F1 via direct series summation.
fn hyp1f1_f64(a: f64, b: f64, z: f64) -> f64 {
    // For large negative z, use Kummer's transformation: M(a,b,z) = e^z M(b-a, b, -z)
    if z < -10.0 {
        return z.exp() * hyp1f1_series(b - a, b, -z);
    }

    hyp1f1_series(a, b, z)
}

/// Direct series for 1F1.
fn hyp1f1_series(a: f64, b: f64, z: f64) -> f64 {
    let max_terms = 300;
    let eps = 1e-15;

    let mut sum = 1.0;
    let mut term = 1.0;

    for k in 0..max_terms {
        let kf = k as f64;
        term *= (a + kf) * z / ((b + kf) * (kf + 1.0));
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum
}

/// 2F1 implementation.
fn hyp2f1_f64(a: f64, b: f64, c: f64, z: f64) -> f64 {
    // Check for polynomial termination
    if (a <= 0.0 && a == a.floor()) || (b <= 0.0 && b == b.floor()) {
        return hyp2f1_series(a, b, c, z);
    }

    if z.abs() < 0.5 {
        hyp2f1_series(a, b, c, z)
    } else if z.abs() < 1.0 {
        // Use Pfaff transformation: 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))
        let z_new = z / (z - 1.0);
        if z_new.abs() < 0.5 {
            (1.0 - z).powf(-a) * hyp2f1_series(a, c - b, c, z_new)
        } else {
            // Euler transformation: 2F1(a,b;c;z) = (1-z)^{c-a-b} 2F1(c-a,c-b;c;z)
            (1.0 - z).powf(c - a - b) * hyp2f1_series(c - a, c - b, c, z)
        }
    } else if z == 1.0 {
        // 2F1(a,b;c;1) = Gamma(c)Gamma(c-a-b) / (Gamma(c-a)Gamma(c-b))  if c > a+b
        if c > a + b {
            let num = libm::lgamma(c) + libm::lgamma(c - a - b);
            let den = libm::lgamma(c - a) + libm::lgamma(c - b);
            (num - den).exp()
        } else {
            f64::INFINITY
        }
    } else {
        // |z| > 1: use analytic continuation via 1/z
        // Only valid for certain parameter ranges; use linear transformation
        if z > 1.0 {
            // Real z > 1 is on the branch cut for generic parameters
            f64::NAN
        } else {
            // z < -1: use 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1))
            let z_new = z / (z - 1.0);
            (1.0 - z).powf(-a) * hyp2f1_f64(a, c - b, c, z_new)
        }
    }
}

/// Direct series for 2F1.
fn hyp2f1_series(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let max_terms = 500;
    let eps = 1e-14;

    let mut sum = 1.0;
    let mut term = 1.0;

    for k in 0..max_terms {
        let kf = k as f64;
        term *= (a + kf) * (b + kf) * z / ((c + kf) * (kf + 1.0));
        sum += term;

        if term.abs() < eps * sum.abs() {
            break;
        }

        // Polynomial termination
        if term == 0.0 {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hyp1f1_zero_z() {
        // 1F1(a; b; 0) = 1
        assert_relative_eq!(hyp1f1(2.0_f64, 3.0_f64, 0.0_f64), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hyp1f1_exp() {
        // 1F1(a; a; z) = e^z
        let z = 2.0_f64;
        assert_relative_eq!(hyp1f1(3.0_f64, 3.0_f64, z), z.exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_known() {
        // 1F1(1; 2; z) = (e^z - 1)/z
        let z = 1.0_f64;
        let expected = (z.exp() - 1.0) / z;
        assert_relative_eq!(hyp1f1(1.0_f64, 2.0_f64, z), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_negative_z() {
        // 1F1(1; 1; -5) = e^{-5}
        assert_relative_eq!(
            hyp1f1(1.0_f64, 1.0_f64, -5.0_f64),
            (-5.0_f64).exp(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_hyp2f1_zero() {
        // 2F1(a,b;c;0) = 1
        assert_relative_eq!(
            hyp2f1(1.0_f64, 2.0_f64, 3.0_f64, 0.0_f64),
            1.0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_hyp2f1_gauss_sum() {
        // 2F1(a,b;c;1) = Gamma(c)Gamma(c-a-b) / (Gamma(c-a)Gamma(c-b))
        // 2F1(1, 1; 3; 1) = Gamma(3)*Gamma(1) / (Gamma(2)*Gamma(2)) = 2*1/(1*1) = 2
        assert_relative_eq!(
            hyp2f1(1.0_f64, 1.0_f64, 3.0_f64, 1.0_f64),
            2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_hyp2f1_geometric() {
        // 2F1(1, 1; 1; z) = 1/(1-z) for |z| < 1
        let z = 0.3_f64;
        assert_relative_eq!(
            hyp2f1(1.0_f64, 1.0_f64, 1.0_f64, z),
            1.0 / (1.0 - z),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_hyp2f1_negative_z() {
        // 2F1(1, 1; 2; -1) = ln(2)
        assert_relative_eq!(
            hyp2f1(1.0_f64, 1.0_f64, 2.0_f64, -1.0_f64),
            2.0_f64.ln(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_hypergeometric_f32() {
        let r = hyp1f1(1.0_f32, 1.0_f32, 1.0_f32);
        assert!((r.to_f64() - 1.0_f64.exp()).abs() < 1e-4);
    }
}
