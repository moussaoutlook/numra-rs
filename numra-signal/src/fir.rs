//! FIR filter design via windowed sinc method.
//!
//! Provides lowpass FIR filter design and convolution-based application.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_fft::{window_func, Window};

use crate::error::SignalError;

/// Design a lowpass FIR filter using the windowed sinc method.
///
/// Returns filter coefficients of length `numtaps`.
///
/// # Arguments
/// - `numtaps` — Number of filter taps (odd recommended for linear phase)
/// - `cutoff` — Cutoff frequency in Hz
/// - `fs` — Sampling frequency in Hz
/// - `window` — Window function to apply
///
/// # Example
/// ```
/// use numra_signal::firwin;
/// use numra_fft::Window;
///
/// let taps = firwin(31, 10.0, 100.0, &Window::Hamming).unwrap();
/// assert_eq!(taps.len(), 31);
/// // Verify taps sum to approximately 1.0 (DC gain = 1)
/// let sum: f64 = taps.iter().sum();
/// assert!((sum - 1.0).abs() < 0.01);
/// ```
pub fn firwin<S: Scalar>(
    numtaps: usize,
    cutoff: f64,
    fs: f64,
    window: &Window,
) -> Result<Vec<S>, SignalError> {
    if numtaps == 0 {
        return Err(SignalError::InvalidParameter("numtaps must be > 0".into()));
    }
    let nyquist = fs / 2.0;
    if cutoff <= 0.0 || cutoff >= nyquist {
        return Err(SignalError::InvalidCutoff { cutoff, nyquist });
    }

    // Normalized cutoff (0 to 1, where 1 = Nyquist)
    let fc = cutoff / nyquist;
    let m = numtaps - 1;
    let half = m as f64 / 2.0;
    let pi = core::f64::consts::PI;

    // Ideal sinc filter (compute in f64)
    let mut h_f64: Vec<f64> = (0..numtaps)
        .map(|i| {
            let n = i as f64 - half;
            if n.abs() < 1e-12 {
                fc // sinc(0) = 1, scaled by fc
            } else {
                (pi * fc * n).sin() / (pi * n)
            }
        })
        .collect();

    // Apply window (compute window in the target scalar type, then use f64 for arithmetic)
    let w: Vec<S> = window_func(window, numtaps);
    for (hi, wi) in h_f64.iter_mut().zip(w.iter()) {
        *hi *= wi.to_f64();
    }

    // Normalize for unity DC gain
    let sum: f64 = h_f64.iter().sum();
    if sum.abs() > 1e-15 {
        for hi in h_f64.iter_mut() {
            *hi /= sum;
        }
    }

    Ok(h_f64.iter().map(|&v| S::from_f64(v)).collect())
}

/// Apply an FIR filter to a signal via direct convolution.
///
/// Returns a vector of the same length as the input (causal, no delay correction).
/// The output is truncated to match the input length.
pub fn fir_filter<S: Scalar>(taps: &[S], x: &[S]) -> Vec<S> {
    let n = x.len();
    let m = taps.len();
    if n == 0 || m == 0 {
        return vec![S::ZERO; n];
    }

    let mut y = vec![S::ZERO; n];
    for i in 0..n {
        let mut sum = S::ZERO;
        for j in 0..m {
            if i >= j {
                sum += taps[j] * x[i - j];
            }
        }
        y[i] = sum;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn test_firwin_basic() {
        let taps: Vec<f64> = firwin(31, 10.0, 100.0, &Window::Hamming).unwrap();
        assert_eq!(taps.len(), 31);
        let sum: f64 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "DC gain should be ~1, got {sum}");
    }

    #[test]
    fn test_firwin_symmetry() {
        // Linear-phase FIR should be symmetric
        let taps: Vec<f64> = firwin(31, 10.0, 100.0, &Window::Hamming).unwrap();
        let n = taps.len();
        for i in 0..n / 2 {
            assert!(
                (taps[i] - taps[n - 1 - i]).abs() < 1e-12,
                "tap {} != tap {}: {} vs {}",
                i,
                n - 1 - i,
                taps[i],
                taps[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_firwin_rectangular() {
        // Rectangular window = ideal sinc (truncated)
        let taps: Vec<f64> = firwin(21, 20.0, 100.0, &Window::Rectangular).unwrap();
        assert_eq!(taps.len(), 21);
        let sum: f64 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_firwin_invalid() {
        assert!(firwin::<f64>(0, 10.0, 100.0, &Window::Hamming).is_err());
        assert!(firwin::<f64>(31, 0.0, 100.0, &Window::Hamming).is_err());
        assert!(firwin::<f64>(31, 50.0, 100.0, &Window::Hamming).is_err());
    }

    #[test]
    fn test_fir_filter_impulse() {
        let taps = vec![0.25, 0.5, 0.25];
        let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let y = fir_filter(&taps, &x);
        assert!((y[2] - 0.25).abs() < 1e-12);
        assert!((y[3] - 0.5).abs() < 1e-12);
        assert!((y[4] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_fir_filter_dc() {
        // FIR taps that sum to 1.0 should pass DC
        let taps: Vec<f64> = firwin(11, 20.0, 100.0, &Window::Hamming).unwrap();
        let x = vec![1.0; 50];
        let y = fir_filter(&taps, &x);
        // After transient, output should be ~1.0
        assert!((y[30] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fir_filter_empty() {
        let y = fir_filter(&[1.0], &Vec::<f64>::new());
        assert!(y.is_empty());
    }

    #[test]
    fn test_fir_lowpass_attenuation() {
        // Design FIR lowpass at 10Hz, fs=100Hz
        let taps: Vec<f64> = firwin(63, 10.0, 100.0, &Window::Hamming).unwrap();

        // Apply to 40Hz sine
        let fs = 100.0;
        let n = 500;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n).map(|i| (pi2 * 40.0 * i as f64 / fs).sin()).collect();
        let y = fir_filter(&taps, &x);

        // After transient, amplitude should be heavily attenuated
        let max_amp: f64 = y[200..].iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_amp < 0.1,
            "40Hz should be attenuated, max_amp = {max_amp}"
        );
    }
}
