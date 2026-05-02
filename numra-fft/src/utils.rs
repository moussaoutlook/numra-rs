//! FFT utility functions: frequencies, shifting, windowing.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Window function types for spectral analysis.
#[derive(Clone, Debug)]
pub enum Window {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann window: 0.5 * (1 - cos(2*pi*n/(N-1))).
    Hann,
    /// Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1)).
    Hamming,
    /// Blackman window.
    Blackman,
    /// Kaiser window with shape parameter beta.
    Kaiser(f64),
    /// Flat-top window for accurate amplitude measurement.
    FlatTop,
}

/// Generate window function coefficients.
///
/// Returns a vector of length `n` with the window weights.
pub fn window_func<S: Scalar>(window: &Window, n: usize) -> Vec<S> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![S::ONE];
    }

    let nm1 = S::from_usize(n - 1);
    let two_pi = S::TWO * S::PI;
    let four_pi = S::from_f64(4.0) * S::PI;

    match window {
        Window::Rectangular => vec![S::ONE; n],
        Window::Hann => (0..n)
            .map(|i| S::HALF * (S::ONE - (two_pi * S::from_usize(i) / nm1).cos()))
            .collect(),
        Window::Hamming => {
            let a0 = S::from_f64(0.54);
            let a1 = S::from_f64(0.46);
            (0..n)
                .map(|i| a0 - a1 * (two_pi * S::from_usize(i) / nm1).cos())
                .collect()
        }
        Window::Blackman => {
            let a0 = S::from_f64(0.42);
            let a1 = S::HALF;
            let a2 = S::from_f64(0.08);
            (0..n)
                .map(|i| {
                    let x = S::from_usize(i) / nm1;
                    a0 - a1 * (two_pi * x).cos() + a2 * (four_pi * x).cos()
                })
                .collect()
        }
        Window::Kaiser(beta) => kaiser_window(S::from_f64(*beta), n),
        Window::FlatTop => {
            // Flat-top window coefficients (five-term cosine sum)
            let a0 = S::from_f64(0.21557895);
            let a1 = S::from_f64(0.41663158);
            let a2 = S::from_f64(0.277263158);
            let a3 = S::from_f64(0.083578947);
            let a4 = S::from_f64(0.006947368);
            let six_pi = S::from_f64(6.0) * S::PI;
            let eight_pi = S::from_f64(8.0) * S::PI;
            (0..n)
                .map(|i| {
                    let x = S::from_usize(i) / nm1;
                    a0 - a1 * (two_pi * x).cos() + a2 * (four_pi * x).cos()
                        - a3 * (six_pi * x).cos()
                        + a4 * (eight_pi * x).cos()
                })
                .collect()
        }
    }
}

/// Kaiser window using the I0 Bessel function approximation.
fn kaiser_window<S: Scalar>(beta: S, n: usize) -> Vec<S> {
    let nm1 = S::from_usize(n - 1);
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let x = S::TWO * S::from_usize(i) / nm1 - S::ONE;
            let arg = beta * (S::ONE - x * x).max(S::ZERO).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

/// Modified Bessel function I0(x) via power series.
fn bessel_i0<S: Scalar>(x: S) -> S {
    let mut sum = S::ONE;
    let mut term = S::ONE;
    let x2_4 = x * x / S::from_f64(4.0);
    for k in 1..50 {
        let kf = S::from_usize(k);
        term = term * x2_4 / (kf * kf);
        sum += term;
        if term.to_f64() < 1e-16 * sum.to_f64() {
            break;
        }
    }
    sum
}

/// Compute the DFT sample frequencies.
///
/// Returns the frequency bin centers in cycles per unit of sample spacing `dt`.
/// For N samples, returns N frequency values.
///
/// f = [0, 1, ..., N/2-1, -N/2, ..., -1] / (dt * N)
pub fn fftfreq<S: Scalar>(n: usize, dt: S) -> Vec<S> {
    let val = S::ONE / (S::from_usize(n) * dt);
    let half = n.div_ceil(2);

    (0..n)
        .map(|i| {
            if i < half {
                S::from_usize(i) * val
            } else {
                (S::from_usize(i) - S::from_usize(n)) * val
            }
        })
        .collect()
}

/// Shift zero-frequency component to the center.
///
/// Rearranges output of `fft` so that the zero-frequency component is centered.
pub fn fftshift<S: Scalar>(x: &mut [S]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    let mid = n / 2;
    x.rotate_left(mid);
}

/// Inverse of `fftshift`: move center back to beginning.
pub fn ifftshift<S: Scalar>(x: &mut [S]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    let mid = n.div_ceil(2);
    x.rotate_left(mid);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fftfreq_even() {
        let f: Vec<f64> = fftfreq(8, 1.0);
        assert_eq!(f.len(), 8);
        assert!((f[0] - 0.0).abs() < 1e-14);
        assert!((f[1] - 0.125).abs() < 1e-14);
        assert!((f[4] - (-0.5)).abs() < 1e-14);
        assert!((f[7] - (-0.125)).abs() < 1e-14);
    }

    #[test]
    fn test_fftfreq_odd() {
        let f: Vec<f64> = fftfreq(5, 0.1);
        // Frequencies: [0, 2, 4, -4, -2]
        assert!((f[0] - 0.0).abs() < 1e-14);
        assert!((f[1] - 2.0).abs() < 1e-14);
        assert!((f[2] - 4.0).abs() < 1e-14);
        assert!((f[3] - (-4.0)).abs() < 1e-14);
        assert!((f[4] - (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn test_fftshift() {
        let mut x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0];
        fftshift(&mut x);
        assert_eq!(x, vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip() {
        let original: Vec<f64> = vec![0.0, 1.0, 2.0, -3.0, -2.0, -1.0];
        let mut x = original.clone();
        fftshift(&mut x);
        ifftshift(&mut x);
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn test_window_rectangular() {
        let w: Vec<f64> = window_func(&Window::Rectangular, 5);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-14));
    }

    #[test]
    fn test_window_hann_endpoints() {
        let w: Vec<f64> = window_func(&Window::Hann, 8);
        assert!(w[0].abs() < 1e-14); // Starts at 0
        assert!(w[7].abs() < 1e-14); // Ends at 0
        assert!((w[4] - 1.0).abs() < 0.1); // Peak near center
    }

    #[test]
    fn test_window_hamming() {
        let w: Vec<f64> = window_func(&Window::Hamming, 8);
        // Hamming doesn't go to zero at edges
        assert!(w[0] > 0.05);
        assert!(w[7] > 0.05);
    }

    #[test]
    fn test_window_blackman() {
        let w: Vec<f64> = window_func(&Window::Blackman, 16);
        assert!(w[0].abs() < 1e-10); // Very small at edges
        assert!(w[8] > 0.9); // Peak near center
    }

    #[test]
    fn test_window_kaiser() {
        let w: Vec<f64> = window_func(&Window::Kaiser(5.0), 16);
        assert!(w.iter().all(|&v| v >= 0.0 && v <= 1.01));
        // Kaiser is symmetric
        for i in 0..8 {
            assert!((w[i] - w[15 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_window_empty() {
        assert!(window_func::<f64>(&Window::Hann, 0).is_empty());
        assert_eq!(window_func::<f64>(&Window::Hann, 1), vec![1.0]);
    }
}
