//! Bessel functions: J, Y, I, K for real order and argument.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Bessel function of the first kind J_nu(x).
///
/// For integer order: Miller's backward recurrence.
/// For non-integer order: series expansion for small x, asymptotic for large x.
pub fn besselj<S: Scalar>(nu: S, x: S) -> S {
    let nuf = nu.to_f64();
    let xf = x.to_f64();

    if xf == 0.0 {
        return if nuf == 0.0 { S::ONE } else { S::ZERO };
    }

    S::from_f64(besselj_f64(nuf, xf))
}

/// Bessel function of the second kind Y_nu(x).
///
/// For integer order: uses the Neumann formula.
pub fn bessely<S: Scalar>(nu: S, x: S) -> S {
    let nuf = nu.to_f64();
    let xf = x.to_f64();

    if xf <= 0.0 {
        return S::NAN;
    }

    S::from_f64(bessely_f64(nuf, xf))
}

/// Modified Bessel function of the first kind I_nu(x).
pub fn besseli<S: Scalar>(nu: S, x: S) -> S {
    let nuf = nu.to_f64();
    let xf = x.to_f64();

    if xf == 0.0 {
        return if nuf == 0.0 { S::ONE } else { S::ZERO };
    }

    S::from_f64(besseli_f64(nuf, xf))
}

/// Modified Bessel function of the second kind K_nu(x).
pub fn besselk<S: Scalar>(nu: S, x: S) -> S {
    let xf = x.to_f64();
    let nuf = nu.to_f64();

    if xf <= 0.0 {
        return S::NAN;
    }

    S::from_f64(besselk_f64(nuf, xf))
}

// ============ Internal implementations ============

/// J_nu(x) via power series: J_nu(x) = sum_{k=0}^inf (-1)^k / (k! * Gamma(k+nu+1)) * (x/2)^{2k+nu}
fn besselj_f64(nu: f64, x: f64) -> f64 {
    let xf = x.abs();

    // For large x, use asymptotic
    if xf > 25.0 + nu.abs() {
        return besselj_asymptotic(nu, x);
    }

    // Power series
    let max_terms = 80;
    let eps = 1e-15;
    let half_x = x / 2.0;

    let mut sum = 0.0;
    let mut term = half_x.powf(nu) / libm::tgamma(nu + 1.0);
    sum += term;

    let neg_quarter_x2 = -(x * x) / 4.0;
    for k in 1..max_terms {
        let kf = k as f64;
        term *= neg_quarter_x2 / (kf * (kf + nu));
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum
}

/// Asymptotic expansion for J_nu(x) for large x.
fn besselj_asymptotic(nu: f64, x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let chi = x - (nu / 2.0 + 0.25) * pi;

    // Leading asymptotic: J_nu(x) ~ sqrt(2/(pi*x)) * cos(chi)
    // With first correction terms from Hankel expansion
    let mu = 4.0 * nu * nu;
    let eight_x = 8.0 * x;

    let p0 = 1.0;
    let p1 = -(mu - 1.0) * (mu - 9.0) / (2.0 * eight_x * eight_x);
    let q1 = -(mu - 1.0) / eight_x;

    let amplitude = (2.0 / (pi * x)).sqrt();
    amplitude * ((p0 + p1) * chi.cos() - q1 * chi.sin())
}

/// Y_nu(x): for integer nu use forward recurrence from Y_0 and Y_1.
/// For non-integer nu: Y_nu = (J_nu cos(nu*pi) - J_{-nu}) / sin(nu*pi).
fn bessely_f64(nu: f64, x: f64) -> f64 {
    let nuf_frac = nu - nu.floor();

    if nuf_frac.abs() < 1e-14 {
        // Integer order
        let n = nu.round() as i64;
        if n < 0 {
            // Y_{-n}(x) = (-1)^n Y_n(x)
            let yn = bessely_int((-n) as usize, x);
            if n % 2 == 0 {
                yn
            } else {
                -yn
            }
        } else {
            bessely_int(n as usize, x)
        }
    } else {
        // Non-integer: Y_nu = (J_nu cos(nu*pi) - J_{-nu}) / sin(nu*pi)
        let pi = core::f64::consts::PI;
        let jnu = besselj_f64(nu, x);
        let jneg = besselj_f64(-nu, x);
        (jnu * (nu * pi).cos() - jneg) / (nu * pi).sin()
    }
}

/// Y_n(x) for non-negative integer n via Y_0, Y_1 and forward recurrence.
fn bessely_int(n: usize, x: f64) -> f64 {
    if x > 25.0 + n as f64 {
        return bessely_asymptotic(n as f64, x);
    }

    let y0 = bessely0(x);
    if n == 0 {
        return y0;
    }
    let y1 = bessely1(x);
    if n == 1 {
        return y1;
    }

    // Forward recurrence: Y_{n+1}(x) = 2n/x * Y_n(x) - Y_{n-1}(x)
    let mut ym1 = y0;
    let mut y = y1;
    for k in 1..n {
        let yn = 2.0 * k as f64 / x * y - ym1;
        ym1 = y;
        y = yn;
    }
    y
}

/// Asymptotic expansion for Y_nu(x) for large x.
fn bessely_asymptotic(nu: f64, x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let chi = x - (nu / 2.0 + 0.25) * pi;
    let mu = 4.0 * nu * nu;
    let eight_x = 8.0 * x;

    let p0 = 1.0;
    let p1 = -(mu - 1.0) * (mu - 9.0) / (2.0 * eight_x * eight_x);
    let q1 = -(mu - 1.0) / eight_x;

    let amplitude = (2.0 / (pi * x)).sqrt();
    amplitude * ((p0 + p1) * chi.sin() + q1 * chi.cos())
}

/// Y_0(x) from polynomial/rational approximation (A&S 9.4).
fn bessely0(x: f64) -> f64 {
    if x <= 8.0 {
        bessely0_series(x)
    } else {
        bessely_asymptotic(0.0, x)
    }
}

/// Y_0 via Neumann integral-like series.
fn bessely0_series(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let euler = 0.5772156649015329;
    let half_x = x / 2.0;

    // Y_0(x) = (2/pi)(ln(x/2) + gamma) J_0(x) + (2/pi) * sum
    let j0 = besselj_f64(0.0, x);
    let mut sum = 0.0;
    let neg_quarter_x2 = -(x * x) / 4.0;
    let mut term = 1.0; // (-1)^k (x/2)^{2k} / (k!)^2
    let mut hk = 0.0; // harmonic number H_k

    for k in 1..60 {
        let kf = k as f64;
        term *= neg_quarter_x2 / (kf * kf);
        hk += 1.0 / kf;
        sum += term * hk;
    }

    (2.0 / pi) * ((half_x.ln() + euler) * j0 - sum)
}

/// Y_1(x) from series.
fn bessely1(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let euler = 0.5772156649015329;
    let half_x = x / 2.0;

    if x > 8.0 {
        return bessely_asymptotic(1.0, x);
    }

    // Y_1(x) = (2/pi)(ln(x/2) + gamma) J_1(x) - (2/(pi*x)) + (2/pi) * series
    let j1 = besselj_f64(1.0, x);

    let mut sum = 0.0;
    let neg_quarter_x2 = -(x * x) / 4.0;
    let mut term = half_x; // (x/2)^{2k+1} / (k! * (k+1)!)
    let mut hk = 0.0;

    for k in 1..60 {
        let kf = k as f64;
        term *= neg_quarter_x2 / (kf * (kf + 1.0));
        hk += 1.0 / kf;
        let hk1 = hk + 1.0 / (kf + 1.0);
        sum += term * (hk + hk1);
    }

    (2.0 / pi) * ((half_x.ln() + euler) * j1 - 1.0 / x) - (1.0 / pi) * sum
}

/// I_nu(x) via power series: I_nu(x) = sum_{k=0} (x/2)^{2k+nu} / (k! * Gamma(k+nu+1))
fn besseli_f64(nu: f64, x: f64) -> f64 {
    let xabs = x.abs();

    // For large x, use asymptotic: I_nu(x) ~ e^x / sqrt(2*pi*x)
    if xabs > 30.0 + nu.abs() {
        return besseli_asymptotic(nu, xabs);
    }

    let max_terms = 80;
    let eps = 1e-15;
    let half_x = xabs / 2.0;

    let mut sum = 0.0;
    let mut term = half_x.powf(nu) / libm::tgamma(nu + 1.0);
    sum += term;

    let quarter_x2 = xabs * xabs / 4.0;
    for k in 1..max_terms {
        let kf = k as f64;
        term *= quarter_x2 / (kf * (kf + nu));
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum
}

/// Asymptotic expansion for I_nu(x) for large x.
fn besseli_asymptotic(nu: f64, x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let mu = 4.0 * nu * nu;
    let eight_x = 8.0 * x;

    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=5 {
        let kf = k as f64;
        let factor = (mu - (2.0 * kf - 1.0).powi(2)) / (kf * eight_x);
        term *= factor;
        sum += term;
    }

    x.exp() / (2.0 * pi * x).sqrt() * sum
}

/// K_nu(x) via the relation K_nu(x) = (pi/2) * (I_{-nu}(x) - I_nu(x)) / sin(nu*pi)
/// For integer nu, use limiting form.
fn besselk_f64(nu: f64, x: f64) -> f64 {
    let nuf_frac = nu - nu.floor();

    if nuf_frac.abs() < 1e-14 {
        // Integer order: K_n from K_0 and K_1
        let n = nu.round().abs() as usize; // K_{-n} = K_n
        besselk_int(n, x)
    } else {
        let pi = core::f64::consts::PI;
        let i_pos = besseli_f64(nu, x);
        let i_neg = besseli_f64(-nu, x);
        (pi / 2.0) * (i_neg - i_pos) / (nu * pi).sin()
    }
}

/// K_n(x) for non-negative integer n.
fn besselk_int(n: usize, x: f64) -> f64 {
    // For large x, use asymptotic
    if x > 25.0 {
        return besselk_asymptotic(n as f64, x);
    }

    let k0 = besselk0(x);
    if n == 0 {
        return k0;
    }
    let k1 = besselk1(x);
    if n == 1 {
        return k1;
    }

    // Forward recurrence: K_{n+1}(x) = 2n/x * K_n(x) + K_{n-1}(x)
    let mut km1 = k0;
    let mut k = k1;
    for i in 1..n {
        let kn = 2.0 * i as f64 / x * k + km1;
        km1 = k;
        k = kn;
    }
    k
}

/// Asymptotic K_nu(x) ~ sqrt(pi/(2x)) e^{-x} (1 + ...)
fn besselk_asymptotic(nu: f64, x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let mu = 4.0 * nu * nu;
    let eight_x = 8.0 * x;

    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=5 {
        let kf = k as f64;
        let factor = (mu - (2.0 * kf - 1.0).powi(2)) / (kf * eight_x);
        term *= factor;
        sum += term;
    }

    (pi / (2.0 * x)).sqrt() * (-x).exp() * sum
}

/// K_0(x) via polynomial approximation (A&S 9.6).
fn besselk0(x: f64) -> f64 {
    if x <= 2.0 {
        let euler = 0.5772156649015329;
        let half_x = x / 2.0;
        let i0 = besseli_f64(0.0, x);

        // K_0(x) = -(ln(x/2) + gamma) * I_0(x) + series
        let quarter_x2 = x * x / 4.0;
        let mut sum = 0.0;
        let mut term = 1.0;
        let mut hk = 0.0;
        for k in 1..40 {
            let kf = k as f64;
            term *= quarter_x2 / (kf * kf);
            hk += 1.0 / kf;
            sum += term * hk;
        }

        -(half_x.ln() + euler) * i0 + sum
    } else {
        besselk_asymptotic(0.0, x)
    }
}

/// K_1(x) via forward recurrence from K_0.
fn besselk1(x: f64) -> f64 {
    if x > 2.0 {
        return besselk_asymptotic(1.0, x);
    }

    // K_1(x) = K_0(x) * I_1(x) / I_0(x)  + 1/(x * I_0(x)^2) ... no, use the Wronskian
    // Wronskian: I_0(x)*K_1(x) + I_1(x)*K_0(x) = 1/x
    // So: K_1(x) = (1/x - I_1(x)*K_0(x)) / I_0(x)
    let i0 = besseli_f64(0.0, x);
    let i1 = besseli_f64(1.0, x);
    let k0 = besselk0(x);

    (1.0 / x - i1 * k0) / i0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j0_at_zero() {
        assert!((besselj(0.0_f64, 0.0_f64) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_j0_known() {
        // J_0(1) ≈ 0.7651976865579665
        assert!((besselj(0.0_f64, 1.0_f64) - 0.7651976865579665).abs() < 1e-10);
    }

    #[test]
    fn test_j1_known() {
        // J_1(1) ≈ 0.4400505857449335
        assert!((besselj(1.0_f64, 1.0_f64) - 0.4400505857449335).abs() < 1e-10);
    }

    #[test]
    fn test_j0_large_x() {
        // J_0(10) ≈ -0.2459357644513483
        assert!((besselj(0.0_f64, 10.0_f64) - (-0.2459357644513483)).abs() < 1e-4);
    }

    #[test]
    fn test_i0_at_zero() {
        assert!((besseli(0.0_f64, 0.0_f64) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_i0_known() {
        // I_0(1) ≈ 1.2660658777520082
        assert!((besseli(0.0_f64, 1.0_f64) - 1.2660658777520082).abs() < 1e-10);
    }

    #[test]
    fn test_y0_known() {
        // Y_0(1) ≈ 0.0882569642156769
        assert!((bessely(0.0_f64, 1.0_f64) - 0.0882569642156769).abs() < 1e-4);
    }

    #[test]
    fn test_k0_known() {
        // K_0(1) ≈ 0.4210244382407083
        assert!((besselk(0.0_f64, 1.0_f64) - 0.4210244382407083).abs() < 1e-4);
    }

    #[test]
    fn test_k1_known() {
        // K_1(1) ≈ 0.6019072301972346
        assert!((besselk(1.0_f64, 1.0_f64) - 0.6019072301972346).abs() < 1e-4);
    }

    #[test]
    fn test_y_negative_x() {
        assert!(bessely(0.0_f64, -1.0_f64).to_f64().is_nan());
    }

    #[test]
    fn test_k_negative_x() {
        assert!(besselk(0.0_f64, -1.0_f64).to_f64().is_nan());
    }

    #[test]
    fn test_bessel_f32() {
        assert!((besselj(0.0_f32, 1.0_f32) - 0.7652_f32).abs() < 0.01);
    }
}
