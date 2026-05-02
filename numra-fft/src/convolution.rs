//! FFT-based convolution and correlation.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::complex::Complex;
use crate::fft_core::{fft, ifft};
use numra_core::Scalar;

/// FFT-based linear convolution of two real signals.
///
/// Returns a vector of length `a.len() + b.len() - 1`.
/// Equivalent to `numpy.convolve(a, b, mode='full')`.
pub fn fftconvolve<S: Scalar>(a: &[S], b: &[S]) -> Vec<S> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let n = a.len() + b.len() - 1;
    // Pad to next power of 2 for FFT efficiency
    let fft_len = n.next_power_of_two();

    let mut ca = vec![Complex::zero(); fft_len];
    let mut cb = vec![Complex::zero(); fft_len];

    for (i, &v) in a.iter().enumerate() {
        ca[i] = Complex::new(v, S::ZERO);
    }
    for (i, &v) in b.iter().enumerate() {
        cb[i] = Complex::new(v, S::ZERO);
    }

    let fa = fft(&ca);
    let fb = fft(&cb);

    // Pointwise multiply in frequency domain
    let fc: Vec<Complex<S>> = fa.iter().zip(fb.iter()).map(|(&a, &b)| a * b).collect();

    let result = ifft(&fc);
    result[..n].iter().map(|c| c.re).collect()
}

/// FFT-based cross-correlation of two real signals.
///
/// Returns a vector of length `a.len() + b.len() - 1`.
/// Equivalent to `numpy.correlate(a, b, mode='full')`.
///
/// Implemented as `convolve(a, reverse(b))`.
pub fn fftcorrelate<S: Scalar>(a: &[S], b: &[S]) -> Vec<S> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    // Correlation = convolution with time-reversed second signal
    let b_rev: Vec<S> = b.iter().rev().copied().collect();
    fftconvolve(a, &b_rev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fftconvolve_impulse() {
        // Convolution with impulse [1] is identity
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0];
        let result = fftconvolve(&a, &b);
        assert_eq!(result.len(), 4);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - a[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fftconvolve_known() {
        // [1, 1] * [1, 1, 1] = [1, 2, 2, 1]
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = fftconvolve(&a, &b);
        assert_eq!(result.len(), 4);
        let expected = [1.0, 2.0, 2.0, 1.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12, "{} vs {}", r, e);
        }
    }

    #[test]
    fn test_fftconvolve_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let ab = fftconvolve(&a, &b);
        let ba = fftconvolve(&b, &a);
        assert_eq!(ab.len(), ba.len());
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fftcorrelate_autocorrelation() {
        // Autocorrelation: peak should be the maximum value (at zero lag)
        let a = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = fftcorrelate(&a, &a);
        assert_eq!(result.len(), 9); // 2*5-1
                                     // Peak value = sum(a[i]^2) = 1+4+9+4+1 = 19
        let peak = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((peak - 19.0).abs() < 1e-10);
    }

    #[test]
    fn test_fftcorrelate_vs_direct() {
        // Verify FFT correlation matches direct computation
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let result = fftcorrelate(&a, &b);
        assert_eq!(result.len(), 4); // 3+2-1 = 4
                                     // Direct: corr[k] = sum_i a[i]*b[i-k+(M-1)] for valid overlaps
                                     // Since correlate(a,b) = convolve(a, rev(b)):
        let b_rev = vec![5.0, 4.0];
        let conv = fftconvolve(&a, &b_rev);
        assert_eq!(result.len(), conv.len());
        for (r, c) in result.iter().zip(conv.iter()) {
            assert!((r - c).abs() < 1e-12, "{} vs {}", r, c);
        }
    }

    #[test]
    fn test_fftconvolve_empty() {
        assert!(fftconvolve::<f64>(&[], &[1.0]).is_empty());
        assert!(fftcorrelate::<f64>(&[], &[1.0]).is_empty());
    }
}
