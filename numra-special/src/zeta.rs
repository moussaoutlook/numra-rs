//! Riemann zeta and Hurwitz zeta functions.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Riemann zeta function zeta(s) for real s.
///
/// Uses Euler-Maclaurin summation for s > 0.5,
/// and the reflection formula for s < 0.5.
pub fn zeta<S: Scalar>(s: S) -> S {
    let sf = s.to_f64();
    S::from_f64(zeta_f64(sf))
}

/// Hurwitz zeta function zeta(s, a) = sum_{k=0}^inf 1/(k+a)^s.
///
/// Requires s > 1 and a > 0. Uses Euler-Maclaurin summation.
pub fn hurwitz_zeta<S: Scalar>(s: S, a: S) -> S {
    let sf = s.to_f64();
    let af = a.to_f64();

    if af <= 0.0 || sf <= 1.0 {
        return S::NAN;
    }

    S::from_f64(euler_maclaurin_zeta(sf, af))
}

/// Riemann zeta for real s.
fn zeta_f64(s: f64) -> f64 {
    if s == 1.0 {
        return f64::INFINITY;
    }

    // Trivial zeros at negative even integers
    if s < 0.0 && s == s.floor() && (s as i64) % 2 == 0 {
        return 0.0;
    }

    // Reflection formula for s < 0.5:
    // zeta(s) = 2^s * pi^{s-1} * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
    if s < 0.5 {
        let pi = core::f64::consts::PI;
        let one_minus_s = 1.0 - s;
        let zeta_1ms = zeta_f64(one_minus_s);
        let prefactor =
            2.0_f64.powf(s) * pi.powf(s - 1.0) * (pi * s / 2.0).sin() * libm::tgamma(one_minus_s);
        return prefactor * zeta_1ms;
    }

    // For s > 0.5: Euler-Maclaurin with a=1
    euler_maclaurin_zeta(s, 1.0)
}

/// Euler-Maclaurin summation for Hurwitz zeta(s, a).
///
/// zeta(s, a) = sum_{k=0}^{N-1} 1/(k+a)^s + (N+a)^{1-s}/(s-1) + 1/2 (N+a)^{-s}
///            + sum_{j=1}^{p} B_{2j}/(2j)! * prod_{i=0}^{2j-2}(s+i) * (N+a)^{-(s+2j-1)}
fn euler_maclaurin_zeta(s: f64, a: f64) -> f64 {
    let n = 20;

    let mut sum = 0.0;
    for k in 0..n {
        sum += 1.0 / (k as f64 + a).powf(s);
    }

    let na = n as f64 + a;

    // Integral remainder
    let integral = na.powf(1.0 - s) / (s - 1.0);

    // Endpoint correction
    let boundary = 0.5 * na.powf(-s);

    // Bernoulli corrections: sum_{j=1}^{p} B_{2j}/(2j)! * s(s+1)...(s+2j-2) * (N+a)^{-(s+2j-1)}
    // B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730, B14=7/6
    let bernoulli_2j = [
        1.0 / 6.0,       // B2
        -1.0 / 30.0,     // B4
        1.0 / 42.0,      // B6
        -1.0 / 30.0,     // B8
        5.0 / 66.0,      // B10
        -691.0 / 2730.0, // B12
        7.0 / 6.0,       // B14
    ];

    let mut corr = 0.0;
    // j=1: B2/2! * s * (N+a)^{-(s+1)}
    let mut rising = s;
    let mut inv_na_pow = na.powf(-s - 1.0); // (N+a)^{-(s+2*1-1)}
    corr += bernoulli_2j[0] / 2.0 * rising * inv_na_pow;

    // j=2..7
    for j in 2..=7_usize {
        let jf = j as f64;
        // rising factorial gets two more terms: (s+2j-3)(s+2j-2)
        rising *= (s + 2.0 * jf - 3.0) * (s + 2.0 * jf - 2.0);
        // power of na decreases by 2
        inv_na_pow /= na * na;
        // B_{2j} / (2j)!
        corr += bernoulli_2j[j - 1] / factorial_2j(j) * rising * inv_na_pow;
    }

    sum + integral + boundary + corr
}

/// (2j)! for small j
fn factorial_2j(j: usize) -> f64 {
    match j {
        1 => 2.0,
        2 => 24.0,
        3 => 720.0,
        4 => 40320.0,
        5 => 3628800.0,
        6 => 479001600.0,
        7 => 87178291200.0,
        _ => {
            let n = 2 * j;
            let mut f = 1.0;
            for i in 2..=n {
                f *= i as f64;
            }
            f
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zeta_2() {
        // zeta(2) = pi^2/6
        let expected = core::f64::consts::PI * core::f64::consts::PI / 6.0;
        assert_relative_eq!(zeta(2.0_f64), expected, epsilon = 1e-8);
    }

    #[test]
    fn test_zeta_4() {
        // zeta(4) = pi^4/90
        let pi = core::f64::consts::PI;
        let expected = pi.powi(4) / 90.0;
        assert_relative_eq!(zeta(4.0_f64), expected, epsilon = 1e-8);
    }

    #[test]
    fn test_zeta_3() {
        // Apery's constant
        assert_relative_eq!(zeta(3.0_f64), 1.2020569031595942, epsilon = 1e-8);
    }

    #[test]
    fn test_zeta_negative_even() {
        assert!((zeta(-2.0_f64).to_f64()).abs() < 1e-10);
        assert!((zeta(-4.0_f64).to_f64()).abs() < 1e-10);
    }

    #[test]
    fn test_zeta_negative_one() {
        // zeta(-1) = -1/12
        assert_relative_eq!(zeta(-1.0_f64), -1.0 / 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hurwitz_zeta_riemann() {
        // hurwitz_zeta(s, 1) = zeta(s)
        assert_relative_eq!(
            hurwitz_zeta(2.0_f64, 1.0_f64),
            zeta(2.0_f64),
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_zeta_f32() {
        let pi = core::f64::consts::PI;
        let expected = pi * pi / 6.0;
        assert!((zeta(2.0_f32).to_f64() - expected).abs() < 0.001);
    }
}
