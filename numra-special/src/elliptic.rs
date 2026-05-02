//! Elliptic integrals: complete K(m), E(m), incomplete F(phi, m).
//!
//! Uses the arithmetic-geometric mean (AGM) iteration for fast convergence.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Complete elliptic integral of the first kind K(m).
///
/// K(m) = integral from 0 to pi/2 of dt / sqrt(1 - m*sin^2(t))
///
/// Parameter m = k^2 where k is the modulus. Requires 0 <= m < 1.
pub fn ellipk<S: Scalar>(m: S) -> S {
    let mf = m.to_f64();

    if !(0.0..1.0).contains(&mf) {
        if mf == 1.0 {
            return S::INFINITY;
        }
        return S::NAN;
    }
    if mf == 0.0 {
        return S::from_f64(core::f64::consts::FRAC_PI_2);
    }

    S::from_f64(ellipk_agm(mf))
}

/// Complete elliptic integral of the second kind E(m).
///
/// E(m) = integral from 0 to pi/2 of sqrt(1 - m*sin^2(t)) dt
///
/// Parameter m = k^2. Requires 0 <= m <= 1.
pub fn ellipe<S: Scalar>(m: S) -> S {
    let mf = m.to_f64();

    if !(0.0..=1.0).contains(&mf) {
        return S::NAN;
    }
    if mf == 0.0 {
        return S::from_f64(core::f64::consts::FRAC_PI_2);
    }
    if mf == 1.0 {
        return S::ONE;
    }

    S::from_f64(ellipe_agm(mf))
}

/// Incomplete elliptic integral of the first kind F(phi, m).
///
/// F(phi, m) = integral from 0 to phi of dt / sqrt(1 - m*sin^2(t))
pub fn ellipf<S: Scalar>(phi: S, m: S) -> S {
    let phif = phi.to_f64();
    let mf = m.to_f64();

    if !(0.0..=1.0).contains(&mf) {
        return S::NAN;
    }
    if phif == 0.0 {
        return S::ZERO;
    }

    S::from_f64(ellipf_f64(phif, mf))
}

/// K(m) via AGM: K = pi / (2 * agm(1, sqrt(1-m)))
fn ellipk_agm(m: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let eps = 1e-15;

    for _ in 0..30 {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < eps * a_new {
            return pi / (2.0 * a_new);
        }
        a = a_new;
        b = b_new;
    }

    pi / (2.0 * a)
}

/// E(m) via the AGM with auxiliary sequence.
fn ellipe_agm(m: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let eps = 1e-15;

    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut c2_sum = m; // sum of 2^n * c_n^2
    let mut power = 1.0; // 2^n

    for _ in 0..30 {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        let c = (a - b) / 2.0;
        power *= 2.0;
        c2_sum += power * c * c;

        if (a_new - b_new).abs() < eps * a_new {
            let k = pi / (2.0 * a_new);
            return k * (1.0 - c2_sum / 2.0);
        }
        a = a_new;
        b = b_new;
    }

    let k = pi / (2.0 * a);
    k * (1.0 - c2_sum / 2.0)
}

/// Incomplete F(phi, m) via Carlson's RF.
fn ellipf_f64(phi: f64, m: f64) -> f64 {
    // Reduce to [0, pi/2]
    if phi.abs() > core::f64::consts::FRAC_PI_2 {
        // Use periodicity: F(phi, m) = 2*n*K(m) + F(phi_r, m)
        let k = if m < 1.0 {
            ellipk_agm(m)
        } else {
            f64::INFINITY
        };
        let n = (phi / core::f64::consts::PI + 0.5).floor();
        let phi_r = phi - n * core::f64::consts::PI;
        return 2.0 * n * k + ellipf_f64(phi_r, m);
    }

    let sin_phi = phi.sin();
    let cos_phi = phi.cos();

    // F(phi, m) = sin(phi) * RF(cos^2(phi), 1 - m*sin^2(phi), 1)
    let x = cos_phi * cos_phi;
    let y = 1.0 - m * sin_phi * sin_phi;
    let z = 1.0;

    sin_phi * carlson_rf(x, y, z)
}

/// Carlson's symmetric elliptic integral RF(x, y, z).
fn carlson_rf(x: f64, y: f64, z: f64) -> f64 {
    let eps = 1e-14;
    let mut x = x;
    let mut y = y;
    let mut z = z;

    for _ in 0..30 {
        let a = (x + y + z) / 3.0;
        let dx = 1.0 - x / a;
        let dy = 1.0 - y / a;
        let dz = 1.0 - z / a;

        if dx.abs().max(dy.abs()).max(dz.abs()) < eps {
            // Polynomial approximation
            let e2 = dx * dy + dy * dz + dz * dx;
            let e3 = dx * dy * dz;
            return (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0)
                / a.sqrt();
        }

        let lam = (x * y).sqrt() + (x * z).sqrt() + (y * z).sqrt();
        x = (x + lam) / 4.0;
        y = (y + lam) / 4.0;
        z = (z + lam) / 4.0;
    }

    let a = (x + y + z) / 3.0;
    1.0 / a.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ellipk_zero() {
        // K(0) = pi/2
        assert_relative_eq!(
            ellipk(0.0_f64),
            core::f64::consts::FRAC_PI_2,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_ellipk_half() {
        // K(0.5) ≈ 1.8540746773...
        assert_relative_eq!(ellipk(0.5_f64), 1.8540746773013719, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipk_near_one() {
        // K(m) -> infinity as m -> 1
        let k = ellipk(0.999_f64);
        assert!(k.to_f64() > 4.0);
    }

    #[test]
    fn test_ellipe_zero() {
        // E(0) = pi/2
        assert_relative_eq!(
            ellipe(0.0_f64),
            core::f64::consts::FRAC_PI_2,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_ellipe_one() {
        // E(1) = 1
        assert_relative_eq!(ellipe(1.0_f64), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_ellipe_half() {
        // E(0.5) ≈ 1.3506438810...
        assert_relative_eq!(ellipe(0.5_f64), 1.3506438810476755, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipf_zero_phi() {
        assert_relative_eq!(ellipf(0.0_f64, 0.5_f64), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_ellipf_at_pi_half() {
        // F(pi/2, m) = K(m)
        let m = 0.5_f64;
        let pi_half = core::f64::consts::FRAC_PI_2;
        assert_relative_eq!(ellipf(pi_half, m), ellipk(m), epsilon = 1e-10);
    }

    #[test]
    fn test_elliptic_f32() {
        let k = ellipk(0.5_f32);
        assert!((k.to_f64() - 1.854).abs() < 0.01);
    }
}
