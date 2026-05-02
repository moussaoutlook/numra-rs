//! Orthogonal polynomials: Legendre, Hermite, Laguerre, Chebyshev.
//!
//! All use three-term recurrence relations for numerical stability.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Legendre polynomial P_n(x) via three-term recurrence.
///
/// P_0(x) = 1, P_1(x) = x
/// (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
pub fn legendre_p<S: Scalar>(n: usize, x: S) -> S {
    if n == 0 {
        return S::ONE;
    }
    if n == 1 {
        return x;
    }

    let mut pm1 = S::ONE;
    let mut p = x;
    for k in 1..n {
        let kf = S::from_usize(k);
        let pn = (S::from_f64(2.0 * k as f64 + 1.0) * x * p - kf * pm1) / S::from_usize(k + 1);
        pm1 = p;
        p = pn;
    }
    p
}

/// Physicist's Hermite polynomial H_n(x) via three-term recurrence.
///
/// H_0(x) = 1, H_1(x) = 2x
/// H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
pub fn hermite_h<S: Scalar>(n: usize, x: S) -> S {
    if n == 0 {
        return S::ONE;
    }
    if n == 1 {
        return S::TWO * x;
    }

    let mut hm1 = S::ONE;
    let mut h = S::TWO * x;
    for k in 1..n {
        let hn = S::TWO * x * h - S::TWO * S::from_usize(k) * hm1;
        hm1 = h;
        h = hn;
    }
    h
}

/// Laguerre polynomial L_n(x) via three-term recurrence.
///
/// L_0(x) = 1, L_1(x) = 1 - x
/// (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
pub fn laguerre_l<S: Scalar>(n: usize, x: S) -> S {
    if n == 0 {
        return S::ONE;
    }
    if n == 1 {
        return S::ONE - x;
    }

    let mut lm1 = S::ONE;
    let mut l = S::ONE - x;
    for k in 1..n {
        let kf = S::from_usize(k);
        let ln = (S::from_f64(2.0 * k as f64 + 1.0) - x) * l / S::from_usize(k + 1)
            - kf * lm1 / S::from_usize(k + 1);
        lm1 = l;
        l = ln;
    }
    l
}

/// Chebyshev polynomial of the first kind T_n(x) via three-term recurrence.
///
/// T_0(x) = 1, T_1(x) = x
/// T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
pub fn chebyshev_t<S: Scalar>(n: usize, x: S) -> S {
    if n == 0 {
        return S::ONE;
    }
    if n == 1 {
        return x;
    }

    let mut tm1 = S::ONE;
    let mut t = x;
    for _ in 1..n {
        let tn = S::TWO * x * t - tm1;
        tm1 = t;
        t = tn;
    }
    t
}

/// Associated Legendre function P_l^m(x) for m >= 0.
///
/// Uses the recurrence starting from P_m^m.
pub fn legendre_plm<S: Scalar>(l: usize, m: usize, x: S) -> S {
    if m > l {
        return S::ZERO;
    }

    // P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    let xf = x.to_f64();
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - xf * xf).sqrt();
        let mut fact = 1.0;
        for _i in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if l == m {
        return S::from_f64(pmm);
    }

    // P_{m+1}^m(x) = x (2m+1) P_m^m(x)
    let mut pmmp1 = xf * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return S::from_f64(pmmp1);
    }

    // Recurrence: (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = ((2 * ll - 1) as f64 * xf * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }

    S::from_f64(pll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_legendre_p() {
        // P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2
        assert_relative_eq!(legendre_p(0, 0.5_f64), 1.0, epsilon = 1e-14);
        assert_relative_eq!(legendre_p(1, 0.5_f64), 0.5, epsilon = 1e-14);
        assert_relative_eq!(legendre_p(2, 0.5_f64), -0.125, epsilon = 1e-14);
        assert_relative_eq!(legendre_p(3, 0.5_f64), -0.4375, epsilon = 1e-14);
    }

    #[test]
    fn test_legendre_at_one() {
        // P_n(1) = 1 for all n
        for n in 0..10 {
            assert_relative_eq!(legendre_p(n, 1.0_f64), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_hermite_h() {
        // H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
        let x = 1.5_f64;
        assert_relative_eq!(hermite_h(0, x), 1.0, epsilon = 1e-14);
        assert_relative_eq!(hermite_h(1, x), 3.0, epsilon = 1e-14);
        assert_relative_eq!(hermite_h(2, x), 7.0, epsilon = 1e-14);
        assert_relative_eq!(hermite_h(3, x), 9.0, epsilon = 1e-14);
    }

    #[test]
    fn test_laguerre_l() {
        // L_0 = 1, L_1 = 1-x, L_2 = (x^2 - 4x + 2)/2
        let x = 2.0_f64;
        assert_relative_eq!(laguerre_l(0, x), 1.0, epsilon = 1e-14);
        assert_relative_eq!(laguerre_l(1, x), -1.0, epsilon = 1e-14);
        assert_relative_eq!(laguerre_l(2, x), -1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_chebyshev_t() {
        // T_0 = 1, T_1 = x, T_2 = 2x^2 - 1, T_3 = 4x^3 - 3x
        let x = 0.5_f64;
        assert_relative_eq!(chebyshev_t(0, x), 1.0, epsilon = 1e-14);
        assert_relative_eq!(chebyshev_t(1, x), 0.5, epsilon = 1e-14);
        assert_relative_eq!(chebyshev_t(2, x), -0.5, epsilon = 1e-14);
        assert_relative_eq!(chebyshev_t(3, x), -1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_chebyshev_at_one() {
        // T_n(1) = 1 for all n
        for n in 0..10 {
            assert_relative_eq!(chebyshev_t(n, 1.0_f64), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_chebyshev_trig() {
        // T_n(cos(theta)) = cos(n*theta)
        let theta = 0.7_f64;
        let x = theta.cos();
        for n in 0..8 {
            let expected = (n as f64 * theta).cos();
            assert_relative_eq!(chebyshev_t(n, x), expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_legendre_plm() {
        // P_1^1(x) = -sqrt(1-x^2)
        let x = 0.5_f64;
        let expected = -(1.0 - x * x).sqrt();
        assert_relative_eq!(legendre_plm(1, 1, x), expected, epsilon = 1e-12);

        // P_2^0 = P_2
        assert_relative_eq!(legendre_plm(2, 0, x), legendre_p(2, x), epsilon = 1e-14);
    }

    #[test]
    fn test_orthogonal_f32() {
        assert!((legendre_p(2, 0.5_f32) - (-0.125_f32)).abs() < 1e-6);
        assert!((chebyshev_t(2, 0.5_f32) - (-0.5_f32)).abs() < 1e-6);
    }
}
