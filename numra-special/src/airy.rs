//! Airy functions Ai(x) and Bi(x).
//!
//! For small |x|: ascending power series.
//! For large |x|: asymptotic expansions.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Airy function Ai(x).
///
/// Solution to y'' - xy = 0 that decays as x -> +infinity.
pub fn airy_ai<S: Scalar>(x: S) -> S {
    S::from_f64(airy_ai_f64(x.to_f64()))
}

/// Airy function Bi(x).
///
/// Solution to y'' - xy = 0 that grows as x -> +infinity.
pub fn airy_bi<S: Scalar>(x: S) -> S {
    S::from_f64(airy_bi_f64(x.to_f64()))
}

/// Ai(x) implementation.
fn airy_ai_f64(x: f64) -> f64 {
    if x > 5.0 {
        airy_ai_asymptotic_pos(x)
    } else if x < -5.0 {
        airy_ai_asymptotic_neg(x)
    } else {
        airy_ai_series(x)
    }
}

/// Bi(x) implementation.
fn airy_bi_f64(x: f64) -> f64 {
    if x > 5.0 {
        airy_bi_asymptotic_pos(x)
    } else if x < -5.0 {
        airy_bi_asymptotic_neg(x)
    } else {
        airy_bi_series(x)
    }
}

/// Ai(x) ascending series around x=0.
///
/// Ai(x) = c1 * f(x) - c2 * g(x)
/// where f and g are the two independent series solutions.
fn airy_ai_series(x: f64) -> f64 {
    // Ai(0) = 1 / (3^{2/3} * Gamma(2/3))
    // Ai'(0) = -1 / (3^{1/3} * Gamma(1/3))
    let c1 = 0.355028053887817; // Ai(0)
    let c2 = 0.258819403792807; // -Ai'(0)

    let (f, g) = airy_series_fg(x);
    c1 * f - c2 * g
}

/// Bi(x) ascending series around x=0.
fn airy_bi_series(x: f64) -> f64 {
    // Bi(0) = 1 / (3^{1/6} * Gamma(2/3))
    // Bi'(0) = 3^{1/6} / Gamma(1/3)
    let c1 = 0.614926627446001; // Bi(0)
    let c2 = 0.448288357353826; // Bi'(0)

    let (f, g) = airy_series_fg(x);
    c1 * f + c2 * g
}

/// Compute the two independent series solutions f(x) and g(x).
///
/// f(x) = sum_{k=0} a_k x^{3k} / (3k)!  with a_0 = 1, a_k = ...
/// g(x) = sum_{k=0} b_k x^{3k+1} / (3k+1)! with b_0 = 1, b_k = ...
///
/// Recurrence: f series: f_k = x^3 * f_{k-1} / ((3k-1)(3k))
///             g series: g_k = x^3 * g_{k-1} / ((3k)(3k+1))
fn airy_series_fg(x: f64) -> (f64, f64) {
    let x3 = x * x * x;
    let max_terms = 40;
    let eps = 1e-16;

    // f(x) = 1 + x^3/(2*3) + x^6/(2*3*5*6) + ...
    let mut f_sum = 1.0;
    let mut f_term = 1.0;
    for k in 1..max_terms {
        let k3 = 3 * k;
        f_term *= x3 / ((k3 - 1) as f64 * k3 as f64);
        f_sum += f_term;
        if f_term.abs() < eps * f_sum.abs() {
            break;
        }
    }

    // g(x) = x + x^4/(3*4) + x^7/(3*4*6*7) + ...
    let mut g_sum = x;
    let mut g_term = x;
    for k in 1..max_terms {
        let k3 = 3 * k;
        g_term *= x3 / (k3 as f64 * (k3 + 1) as f64);
        g_sum += g_term;
        if g_term.abs() < eps * g_sum.abs() {
            break;
        }
    }

    (f_sum, g_sum)
}

/// Asymptotic Ai(x) for large positive x: Ai(x) ~ e^{-zeta} / (2*sqrt(pi)*x^{1/4})
fn airy_ai_asymptotic_pos(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    (-zeta).exp() / (2.0 * pi.sqrt() * x.powf(0.25))
}

/// Asymptotic Ai(x) for large negative x: Ai(-x) ~ sin(zeta + pi/4) / (sqrt(pi)*x^{1/4})
fn airy_ai_asymptotic_neg(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let xabs = x.abs();
    let zeta = 2.0 / 3.0 * xabs.powf(1.5);
    (zeta + pi / 4.0).sin() / (pi.sqrt() * xabs.powf(0.25))
}

/// Asymptotic Bi(x) for large positive x: Bi(x) ~ e^{zeta} / (sqrt(pi)*x^{1/4})
fn airy_bi_asymptotic_pos(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    zeta.exp() / (pi.sqrt() * x.powf(0.25))
}

/// Asymptotic Bi(x) for large negative x: Bi(-x) ~ cos(zeta + pi/4) / (sqrt(pi)*x^{1/4})
fn airy_bi_asymptotic_neg(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let xabs = x.abs();
    let zeta = 2.0 / 3.0 * xabs.powf(1.5);
    (zeta + pi / 4.0).cos() / (pi.sqrt() * xabs.powf(0.25))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_airy_ai_at_zero() {
        // Ai(0) ≈ 0.35502805388781723
        assert!((airy_ai(0.0_f64) - 0.35502805388781723).abs() < 1e-10);
    }

    #[test]
    fn test_airy_bi_at_zero() {
        // Bi(0) ≈ 0.61492662744600065
        assert!((airy_bi(0.0_f64) - 0.61492662744600065).abs() < 1e-10);
    }

    #[test]
    fn test_airy_ai_positive() {
        // Ai(1) ≈ 0.13529241631288141
        assert!((airy_ai(1.0_f64) - 0.13529241631288141).abs() < 1e-8);
    }

    #[test]
    fn test_airy_bi_positive() {
        // Bi(1) ≈ 1.2074235949528714
        assert!((airy_bi(1.0_f64) - 1.2074235949528714).abs() < 1e-8);
    }

    #[test]
    fn test_airy_ai_negative() {
        // Ai(-1) ≈ 0.53556088329235
        assert!((airy_ai(-1.0_f64) - 0.53556088329235).abs() < 1e-8);
    }

    #[test]
    fn test_airy_ai_large_positive() {
        // Ai(10) ≈ 1.105e-9 (exponentially small)
        let ai10 = airy_ai(10.0_f64);
        assert!(ai10.to_f64() > 0.0 && ai10.to_f64() < 1e-7);
    }

    #[test]
    fn test_airy_bi_large_positive() {
        // Bi(10) is very large
        let bi10 = airy_bi(10.0_f64);
        assert!(bi10.to_f64() > 1e8);
    }

    #[test]
    fn test_airy_f32() {
        assert!((airy_ai(0.0_f32) - 0.3550_f32).abs() < 0.01);
        assert!((airy_bi(0.0_f32) - 0.6149_f32).abs() < 0.01);
    }
}
