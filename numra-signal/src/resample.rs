//! Signal resampling via FFT-based interpolation.
//!
//! Resample a signal to a different number of samples using the
//! Fourier method (zero-pad or truncate in the frequency domain).
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_fft::{fft, ifft, Complex};

/// Resample a signal to `num_out` samples using FFT-based interpolation.
///
/// This is the Fourier method: take FFT, zero-pad or truncate the spectrum,
/// then take IFFT. This assumes the signal is periodic (or appropriately windowed).
///
/// For anti-aliased downsampling, apply a lowpass filter before resampling.
///
/// # Example
/// ```
/// use numra_signal::resample;
///
/// // Upsample by 2x
/// let x = vec![1.0, 2.0, 3.0, 4.0];
/// let y = resample(&x, 8);
/// assert_eq!(y.len(), 8);
/// ```
pub fn resample<S: Scalar>(x: &[S], num_out: usize) -> Vec<S> {
    let n = x.len();
    if n == 0 || num_out == 0 {
        return vec![S::ZERO; num_out];
    }
    if n == num_out {
        return x.to_vec();
    }

    // FFT of input
    let cx: Vec<Complex<S>> = x.iter().map(|&v| Complex::new(v, S::ZERO)).collect();
    let spectrum = fft(&cx);

    // Construct new spectrum of length num_out
    let mut new_spectrum = vec![Complex::zero(); num_out];

    let half_s = S::HALF;

    if num_out > n {
        // Upsampling: zero-pad in the middle of the spectrum
        let half = n / 2;
        // Copy positive frequencies
        new_spectrum[..half + 1].copy_from_slice(&spectrum[..half + 1]);
        // Copy negative frequencies (from the end)
        for i in 1..n - half {
            new_spectrum[num_out - i] = spectrum[n - i];
        }
        // If n is even, split the Nyquist bin
        if n % 2 == 0 {
            let nyq = spectrum[half] * half_s;
            new_spectrum[half] = nyq;
            new_spectrum[num_out - half] = nyq;
        }
    } else {
        // Downsampling: truncate the spectrum
        let half = num_out / 2;
        // Copy positive frequencies
        new_spectrum[..half + 1].copy_from_slice(&spectrum[..half + 1]);
        // Copy negative frequencies
        for i in 1..num_out - half {
            new_spectrum[num_out - i] = spectrum[n - i];
        }
        // If num_out is even, combine aliased Nyquist components
        if num_out % 2 == 0 {
            new_spectrum[half] = spectrum[half] + spectrum[n - half];
        }
    }

    // IFFT
    let result = ifft(&new_spectrum);
    let scale = S::from_usize(num_out) / S::from_usize(n);
    result.iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn test_resample_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = resample(&x, 4);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-10, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_resample_upsample_dc() {
        // DC signal upsampled should remain DC
        let x = vec![5.0; 8];
        let y = resample(&x, 16);
        for &yi in &y {
            assert!((yi - 5.0).abs() < 1e-10, "expected 5.0, got {yi}");
        }
    }

    #[test]
    fn test_resample_downsample_dc() {
        let x = vec![3.0; 16];
        let y = resample(&x, 8);
        for &yi in &y {
            assert!((yi - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_resample_sine_upsample() {
        // Upsample a sine wave and verify it matches the analytic sine
        let n = 32;
        let n_out = 64;
        let freq = 3.0; // 3 cycles in the period
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n)
            .map(|i| (pi2 * freq * i as f64 / n as f64).sin())
            .collect();
        let y = resample(&x, n_out);

        let expected: Vec<f64> = (0..n_out)
            .map(|i| (pi2 * freq * i as f64 / n_out as f64).sin())
            .collect();

        for (i, (&yi, &ei)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (yi - ei).abs() < 0.05,
                "sample {i}: expected {ei:.4}, got {yi:.4}"
            );
        }
    }

    #[test]
    fn test_resample_empty() {
        let y = resample::<f64>(&[], 10);
        assert_eq!(y.len(), 10);
        let y = resample(&[1.0, 2.0], 0);
        assert!(y.is_empty());
    }

    #[test]
    fn test_resample_length() {
        let x = vec![1.0; 100];
        assert_eq!(resample(&x, 50).len(), 50);
        assert_eq!(resample(&x, 200).len(), 200);
        assert_eq!(resample(&x, 1).len(), 1);
    }
}
