//! Power spectral density, Welch's method, and STFT.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use crate::complex::Complex;
use crate::fft_core::fft;
use crate::utils::{window_func, Window};
use numra_core::Scalar;

/// Power spectral density via periodogram.
///
/// Returns `(frequencies, psd)` where frequencies are in Hz and PSD is in
/// power per Hz. The `fs` parameter is the sampling frequency.
pub fn psd<S: Scalar>(x: &[S], fs: S, window: &Window) -> (Vec<S>, Vec<S>) {
    let n = x.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let w: Vec<S> = window_func(window, n);
    let win_sum_sq: S = w.iter().map(|&v| v * v).sum();

    // Apply window and compute FFT
    let windowed: Vec<Complex<S>> = x
        .iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| Complex::new(xi * wi, S::ZERO))
        .collect();
    let spectrum = fft(&windowed);

    // One-sided PSD: only positive frequencies (N/2+1 bins)
    let n_freq = n / 2 + 1;
    let scale = S::ONE / (fs * win_sum_sq);

    let psd_vals: Vec<S> = spectrum[..n_freq]
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let power = c.norm_sqr() * scale;
            // Double non-DC and non-Nyquist bins for one-sided spectrum
            if k > 0 && k < n / 2 {
                S::TWO * power
            } else {
                power
            }
        })
        .collect();

    let freq_step = fs / S::from_usize(n);
    let frequencies: Vec<S> = (0..n_freq).map(|k| S::from_usize(k) * freq_step).collect();

    (frequencies, psd_vals)
}

/// Welch's method for PSD estimation.
///
/// Averages periodograms of overlapping segments for reduced variance.
///
/// - `fs`: sampling frequency
/// - `nperseg`: segment length
/// - `noverlap`: overlap between segments
/// - `window`: window function to apply
///
/// Returns `(frequencies, psd)`.
pub fn welch<S: Scalar>(
    x: &[S],
    fs: S,
    nperseg: usize,
    noverlap: usize,
    window: &Window,
) -> (Vec<S>, Vec<S>) {
    let n = x.len();
    if n == 0 || nperseg == 0 || nperseg > n {
        return (vec![], vec![]);
    }

    let step = nperseg - noverlap;
    if step == 0 {
        return (vec![], vec![]);
    }

    let w: Vec<S> = window_func(window, nperseg);
    let win_sum_sq: S = w.iter().map(|&v| v * v).sum();
    let n_freq = nperseg / 2 + 1;
    let scale = S::ONE / (fs * win_sum_sq);

    let mut psd_avg = vec![S::ZERO; n_freq];
    let mut n_segments: usize = 0;

    let mut start = 0;
    while start + nperseg <= n {
        let segment = &x[start..start + nperseg];

        // Apply window and compute FFT
        let windowed: Vec<Complex<S>> = segment
            .iter()
            .zip(w.iter())
            .map(|(&xi, &wi)| Complex::new(xi * wi, S::ZERO))
            .collect();
        let spectrum = fft(&windowed);

        for k in 0..n_freq {
            let power = spectrum[k].norm_sqr() * scale;
            if k > 0 && k < nperseg / 2 {
                psd_avg[k] += S::TWO * power;
            } else {
                psd_avg[k] += power;
            }
        }

        n_segments += 1;
        start += step;
    }

    if n_segments > 0 {
        let inv_seg = S::ONE / S::from_usize(n_segments);
        for v in &mut psd_avg {
            *v *= inv_seg;
        }
    }

    let freq_step = fs / S::from_usize(nperseg);
    let frequencies: Vec<S> = (0..n_freq).map(|k| S::from_usize(k) * freq_step).collect();

    (frequencies, psd_avg)
}

/// Result of the Short-Time Fourier Transform.
pub struct StftResult<S: Scalar> {
    /// Time values for each window center.
    pub times: Vec<S>,
    /// Frequency bin centers.
    pub frequencies: Vec<S>,
    /// Magnitude of STFT: `magnitude[time_idx][freq_idx]`.
    pub magnitude: Vec<Vec<S>>,
}

/// Short-Time Fourier Transform.
///
/// Computes the STFT by applying a sliding window and FFT to each segment.
///
/// - `fs`: sampling frequency
/// - `nperseg`: segment length
/// - `noverlap`: overlap between segments
/// - `window`: window function to apply
pub fn stft<S: Scalar>(
    x: &[S],
    fs: S,
    nperseg: usize,
    noverlap: usize,
    window: &Window,
) -> StftResult<S> {
    let n = x.len();
    if n == 0 || nperseg == 0 || nperseg > n {
        return StftResult {
            times: vec![],
            frequencies: vec![],
            magnitude: vec![],
        };
    }

    let step = nperseg - noverlap;
    if step == 0 {
        return StftResult {
            times: vec![],
            frequencies: vec![],
            magnitude: vec![],
        };
    }

    let w: Vec<S> = window_func(window, nperseg);
    let n_freq = nperseg / 2 + 1;

    let freq_step = fs / S::from_usize(nperseg);
    let frequencies: Vec<S> = (0..n_freq).map(|k| S::from_usize(k) * freq_step).collect();

    let mut times = Vec::new();
    let mut magnitude = Vec::new();

    let mut start = 0;
    while start + nperseg <= n {
        let segment = &x[start..start + nperseg];

        let windowed: Vec<Complex<S>> = segment
            .iter()
            .zip(w.iter())
            .map(|(&xi, &wi)| Complex::new(xi * wi, S::ZERO))
            .collect();
        let spectrum = fft(&windowed);

        let mag: Vec<S> = spectrum[..n_freq].iter().map(|c| c.abs()).collect();
        magnitude.push(mag);

        let center = S::from_usize(start) + (S::from_usize(nperseg) - S::ONE) / S::TWO;
        times.push(center / fs);

        start += step;
    }

    StftResult {
        times,
        frequencies,
        magnitude,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psd_dc_signal() {
        // Constant signal should have all PSD in DC bin
        let x = vec![1.0_f64; 64];
        let (freqs, psd_vals) = psd(&x, 1.0, &Window::Rectangular);
        assert_eq!(freqs.len(), 33); // 64/2+1
        assert!(psd_vals[0] > 0.0);
        // Non-DC bins should be near zero
        for k in 1..psd_vals.len() {
            assert!(psd_vals[k] < 1e-20, "bin {} = {}", k, psd_vals[k]);
        }
    }

    #[test]
    fn test_psd_single_tone() {
        let n = 256;
        let fs = 256.0;
        let freq = 10.0; // 10 Hz
        let pi2 = 2.0 * core::f64::consts::PI;
        let x: Vec<f64> = (0..n).map(|k| (pi2 * freq * k as f64 / fs).sin()).collect();
        let (freqs, psd_vals) = psd(&x, fs, &Window::Rectangular);

        // Find peak frequency
        let peak_idx = psd_vals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!((freqs[peak_idx] - freq).abs() < fs / n as f64 + 0.01);
    }

    #[test]
    fn test_welch_reduces_variance() {
        // Basic smoke test: Welch should produce valid PSD
        let n = 512;
        let fs = 100.0;
        let pi2 = 2.0 * core::f64::consts::PI;
        let x: Vec<f64> = (0..n).map(|k| (pi2 * 10.0 * k as f64 / fs).sin()).collect();
        let (freqs, psd_vals) = welch(&x, fs, 128, 64, &Window::Hann);
        assert!(!freqs.is_empty());
        assert_eq!(freqs.len(), psd_vals.len());
        // All PSD values should be non-negative
        assert!(psd_vals.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_stft_basic() {
        let n = 256;
        let fs = 100.0;
        let pi2 = 2.0 * core::f64::consts::PI;
        let x: Vec<f64> = (0..n).map(|k| (pi2 * 10.0 * k as f64 / fs).sin()).collect();
        let result = stft(&x, fs, 64, 32, &Window::Hann);
        assert!(!result.times.is_empty());
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.magnitude.len(), result.times.len());
        // Each magnitude vector should have n_freq entries
        for mag in &result.magnitude {
            assert_eq!(mag.len(), result.frequencies.len());
        }
    }

    #[test]
    fn test_psd_empty() {
        let (f, p) = psd::<f64>(&[], 1.0, &Window::Rectangular);
        assert!(f.is_empty());
        assert!(p.is_empty());
    }
}
