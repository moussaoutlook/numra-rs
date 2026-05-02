//! Gamma family: Gamma, Beta, Digamma, incomplete Gamma/Beta.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Gamma function via Lanczos approximation (g=7, 9 coefficients).
///
/// Uses the reflection formula for negative arguments.
pub fn gamma<S: Scalar>(z: S) -> S {
    // Delegate to libm via Scalar trait for f32/f64
    z.gamma_fn()
}

/// Natural log of the absolute value of the Gamma function.
pub fn lgamma<S: Scalar>(z: S) -> S {
    z.ln_gamma()
}

/// Digamma (psi) function: psi(x) = d/dx ln Gamma(x) = Gamma'(x)/Gamma(x).
///
/// Uses recurrence psi(x+1) = psi(x) + 1/x to shift to large x,
/// then asymptotic series (Stirling).
pub fn digamma<S: Scalar>(x: S) -> S {
    let xf = x.to_f64();

    if xf.is_nan() {
        return S::NAN;
    }

    // Reflection formula for x < 0: psi(1-x) - pi*cot(pi*x)
    if xf < 0.0 {
        let pi = core::f64::consts::PI;
        let cot = (pi * xf).cos() / (pi * xf).sin();
        let result = digamma(S::ONE - x).to_f64() - pi * cot;
        return S::from_f64(result);
    }

    // Shift to large x via recurrence psi(x+1) = psi(x) + 1/x
    let mut xf = xf;
    let mut result = 0.0;
    while xf < 8.0 {
        result -= 1.0 / xf;
        xf += 1.0;
    }

    // Asymptotic series for large x:
    // psi(x) ~ ln(x) - 1/(2x) - sum_{k=1} B_{2k} / (2k * x^{2k})
    // B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730
    result += xf.ln() - 0.5 / xf;
    let x2 = xf * xf;
    let mut xpow = x2;
    // Bernoulli coefficients B_{2k}/(2k)
    let coeffs = [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        5.0 / 660.0,
        -691.0 / 32760.0,
    ];
    for &c in &coeffs {
        result -= c / xpow;
        xpow *= x2;
    }
    S::from_f64(result)
}

/// Beta function: B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b).
pub fn beta<S: Scalar>(a: S, b: S) -> S {
    let la = lgamma(a).to_f64();
    let lb = lgamma(b).to_f64();
    let lab = lgamma(a + b).to_f64();
    S::from_f64((la + lb - lab).exp())
}

/// Regularized lower incomplete gamma P(a, x) = gamma(a, x) / Gamma(a).
///
/// Uses series expansion for x < a+1, continued fraction otherwise.
pub fn gammainc<S: Scalar>(a: S, x: S) -> S {
    let af = a.to_f64();
    let xf = x.to_f64();

    if xf < 0.0 || af <= 0.0 {
        return S::NAN;
    }
    if xf == 0.0 {
        return S::ZERO;
    }

    if xf < af + 1.0 {
        // Series expansion: P(a,x) = e^{-x} x^a / Gamma(a) * sum_{n=0}^inf x^n / (a+1)(a+2)...(a+n)
        S::from_f64(gammainc_series(af, xf))
    } else {
        // Use Q = 1 - P via continued fraction
        S::from_f64(1.0 - gammaincc_cf(af, xf))
    }
}

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
pub fn gammaincc<S: Scalar>(a: S, x: S) -> S {
    let af = a.to_f64();
    let xf = x.to_f64();

    if xf < 0.0 || af <= 0.0 {
        return S::NAN;
    }
    if xf == 0.0 {
        return S::ONE;
    }

    if xf < af + 1.0 {
        S::from_f64(1.0 - gammainc_series(af, xf))
    } else {
        S::from_f64(gammaincc_cf(af, xf))
    }
}

/// Series expansion for P(a,x).
fn gammainc_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;

    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }

    // P(a,x) = e^{-x} * x^a * sum / Gamma(a)
    let log_prefix = a * x.ln() - x - libm::lgamma(a);
    sum * log_prefix.exp()
}

/// Continued fraction for Q(a,x) using Lentz's method.
fn gammaincc_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    // CF: Q(a,x) = e^{-x} x^a / Gamma(a) * CF
    // CF = 1/(x+1-a- 1*(1-a)/(x+3-a- 2*(2-a)/(x+5-a- ...)))
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }

    let log_prefix = a * x.ln() - x - libm::lgamma(a);
    log_prefix.exp() * h
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the continued fraction representation (Lentz's method).
pub fn betainc<S: Scalar>(a: S, b: S, x: S) -> S {
    let af = a.to_f64();
    let bf = b.to_f64();
    let xf = x.to_f64();

    if !(0.0..=1.0).contains(&xf) {
        return S::NAN;
    }
    if xf == 0.0 {
        return S::ZERO;
    }
    if xf == 1.0 {
        return S::ONE;
    }

    // Symmetry relation for better convergence
    if xf > (af + 1.0) / (af + bf + 2.0) {
        return S::ONE - S::from_f64(betainc_cf(bf, af, 1.0 - xf));
    }

    S::from_f64(betainc_cf(af, bf, xf))
}

/// Continued fraction for I_x(a,b) using Lentz's method.
fn betainc_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let log_prefix = a * x.ln() + b * (1.0 - x).ln() - libm::lgamma(a) - libm::lgamma(b)
        + libm::lgamma(a + b)
        - a.ln();

    let mut c = 1.0 + tiny;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(tiny);
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + num_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let num_odd = -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + num_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    log_prefix.exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma_integers() {
        // Gamma(n) = (n-1)! for positive integers
        assert_relative_eq!(gamma(1.0_f64), 1.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(2.0_f64), 1.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(3.0_f64), 2.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(4.0_f64), 6.0, epsilon = 1e-12);
        assert_relative_eq!(gamma(5.0_f64), 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_half() {
        // Gamma(1/2) = sqrt(pi)
        let sqrtpi = core::f64::consts::PI.sqrt();
        assert_relative_eq!(gamma(0.5_f64), sqrtpi, epsilon = 1e-12);
    }

    #[test]
    fn test_lgamma() {
        assert_relative_eq!(lgamma(1.0_f64), 0.0, epsilon = 1e-12);
        assert_relative_eq!(lgamma(10.0_f64), (362880.0_f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_digamma_integers() {
        // psi(1) = -gamma_euler ≈ -0.5772156649
        assert_relative_eq!(digamma(1.0_f64), -0.5772156649015329, epsilon = 1e-10);
        // psi(2) = 1 - gamma
        assert_relative_eq!(digamma(2.0_f64), 0.4227843350984671, epsilon = 1e-10);
        // psi(3) = 1.5 - gamma
        assert_relative_eq!(digamma(3.0_f64), 0.9227843350984671, epsilon = 1e-10);
    }

    #[test]
    fn test_digamma_half() {
        // psi(1/2) = -gamma - 2*ln(2)
        let expected = -0.5772156649015329 - 2.0 * 2.0_f64.ln();
        assert_relative_eq!(digamma(0.5_f64), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_beta() {
        // B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
        // B(1,1) = 1
        assert_relative_eq!(beta(1.0_f64, 1.0), 1.0, epsilon = 1e-12);
        // B(2,3) = 1*2 / (4*3*2) = 2/24 = 1/12
        assert_relative_eq!(beta(2.0_f64, 3.0), 1.0 / 12.0, epsilon = 1e-12);
        // B(1/2, 1/2) = pi
        assert_relative_eq!(beta(0.5_f64, 0.5), core::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_gammainc_boundary() {
        assert_relative_eq!(gammainc(1.0_f64, 0.0_f64), 0.0, epsilon = 1e-14);
        // P(1, x) = 1 - e^{-x}
        assert_relative_eq!(
            gammainc(1.0_f64, 1.0_f64),
            1.0 - (-1.0_f64).exp(),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            gammainc(1.0_f64, 5.0_f64),
            1.0 - (-5.0_f64).exp(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gammaincc_complement() {
        let a = 2.5_f64;
        let x = 3.0_f64;
        let p = gammainc(a, x);
        let q = gammaincc(a, x);
        assert_relative_eq!(p + q, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_betainc_boundary() {
        assert_relative_eq!(betainc(2.0_f64, 3.0_f64, 0.0_f64), 0.0, epsilon = 1e-14);
        assert_relative_eq!(betainc(2.0_f64, 3.0_f64, 1.0_f64), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_betainc_known() {
        // I_x(1, b) = 1 - (1-x)^b
        let b = 3.0_f64;
        let x = 0.4_f64;
        let expected = 1.0 - (1.0 - x).powf(b);
        assert_relative_eq!(betainc(1.0_f64, b, x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_f32() {
        assert!((gamma(3.0_f32) - 2.0).abs() < 1e-5);
        assert!((digamma(1.0_f32) - (-0.5772157_f32)).abs() < 1e-4);
        assert!((beta(1.0_f32, 1.0_f32) - 1.0).abs() < 1e-5);
    }
}
