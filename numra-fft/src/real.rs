//! Real-valued FFT: efficient transforms for real input signals.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::complex::Complex;
use crate::fft_core::{fft, ifft};
use numra_core::Scalar;

/// FFT of a real-valued signal.
///
/// Returns N/2+1 complex values (the non-redundant half of the spectrum,
/// since the DFT of real data has conjugate symmetry).
pub fn rfft<S: Scalar>(x: &[S]) -> Vec<Complex<S>> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let cx: Vec<Complex<S>> = x.iter().map(|&v| Complex::new(v, S::ZERO)).collect();
    let full = fft(&cx);
    full[..n / 2 + 1].to_vec()
}

/// Inverse real FFT.
///
/// Takes N/2+1 complex values and reconstructs an N-point real signal.
/// The parameter `n` specifies the output length (needed to distinguish
/// even vs odd original length).
pub fn irfft<S: Scalar>(x: &[Complex<S>], n: usize) -> Vec<S> {
    if x.is_empty() || n == 0 {
        return vec![];
    }

    // Reconstruct full spectrum from half using conjugate symmetry
    let mut full = vec![Complex::zero(); n];
    let half = x.len().min(n / 2 + 1);

    full[..half].copy_from_slice(&x[..half]);
    // Conjugate symmetry: X[N-k] = conj(X[k])
    for k in 1..n / 2 + 1 {
        if n - k < full.len() && k < x.len() {
            full[n - k] = x[k].conj();
        }
    }

    let result = ifft(&full);
    result.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfft_length() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = rfft(&x);
        assert_eq!(spectrum.len(), 5); // 8/2 + 1 = 5
    }

    #[test]
    fn test_rfft_dc() {
        let x = vec![1.0_f64; 8];
        let spectrum = rfft(&x);
        assert!((spectrum[0].re - 8.0).abs() < 1e-12);
        for k in 1..spectrum.len() {
            assert!(spectrum[k].abs() < 1e-12);
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = x.len();
        let spectrum = rfft(&x);
        let recovered = irfft(&spectrum, n);

        for (a, b) in x.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-12, "expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_rfft_cosine() {
        let n = 32;
        let freq = 4;
        let pi2 = 2.0 * core::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|k| (pi2 * freq as f64 * k as f64 / n as f64).cos())
            .collect();
        let spectrum = rfft(&x);
        // Peak at bin 4
        let amp = n as f64 / 2.0;
        assert!((spectrum[freq].abs() - amp).abs() < 1e-10);
    }

    #[test]
    fn test_rfft_empty() {
        assert!(rfft::<f64>(&[]).is_empty());
        assert!(irfft::<f64>(&[], 0).is_empty());
    }
}
