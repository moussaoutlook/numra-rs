//! Hilbert transform, analytic signal, and envelope extraction.
//!
//! The Hilbert transform produces the analytic signal, whose magnitude
//! is the instantaneous amplitude (envelope) of the input signal.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_fft::{fft, ifft, Complex};

/// Compute the analytic signal via the Hilbert transform.
///
/// Returns a complex-valued signal where the real part is the original
/// signal and the imaginary part is the Hilbert transform.
///
/// The analytic signal is computed by:
/// 1. FFT the input
/// 2. Zero out negative frequencies, double positive frequencies
/// 3. IFFT back
///
/// # Example
/// ```
/// use numra_signal::hilbert;
///
/// let x: Vec<f64> = (0..64).map(|i| {
///     (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 64.0).sin()
/// }).collect();
/// let z = hilbert(&x);
/// assert_eq!(z.len(), x.len());
/// // The envelope of a pure sine should be approximately 1.0
/// for zi in &z[5..59] {
///     assert!((zi.abs() - 1.0).abs() < 0.15);
/// }
/// ```
pub fn hilbert<S: Scalar>(x: &[S]) -> Vec<Complex<S>> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // FFT of real input
    let cx: Vec<Complex<S>> = x.iter().map(|&v| Complex::new(v, S::ZERO)).collect();
    let mut spectrum = fft(&cx);

    // Create the analytic signal:
    // - DC and Nyquist (if N even) stay the same
    // - Positive frequencies are doubled
    // - Negative frequencies are zeroed
    if n > 1 {
        let half = n.div_ceil(2);

        let two = S::from_f64(2.0);

        // Double positive frequencies (indices 1 to half-1)
        for i in 1..half {
            spectrum[i] = Complex::new(spectrum[i].re * two, spectrum[i].im * two);
        }
        // Nyquist bin stays as-is for even N (index n/2)

        // Zero negative frequencies
        let start_neg = if n % 2 == 0 { n / 2 + 1 } else { n.div_ceil(2) };
        for i in start_neg..n {
            spectrum[i] = Complex::zero();
        }
    }

    // IFFT to get analytic signal
    ifft(&spectrum)
}

/// Extract the instantaneous envelope (amplitude) of a signal.
///
/// Computes `|analytic_signal(x)|` where the analytic signal is obtained
/// via the Hilbert transform.
///
/// # Example
/// ```
/// use numra_signal::envelope;
///
/// // AM signal: carrier modulated by a slow envelope
/// let n = 256;
/// let pi2 = 2.0 * std::f64::consts::PI;
/// let x: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / 256.0;
///     (1.0 + 0.5 * (pi2 * 2.0 * t).cos()) * (pi2 * 30.0 * t).sin()
/// }).collect();
/// let env = envelope(&x);
/// assert_eq!(env.len(), n);
/// ```
pub fn envelope<S: Scalar>(x: &[S]) -> Vec<S> {
    hilbert(x).iter().map(|c| c.abs()).collect()
}

/// Compute the instantaneous frequency of a signal.
///
/// Returns the instantaneous frequency in Hz at each sample, computed
/// as the derivative of the instantaneous phase of the analytic signal.
///
/// # Arguments
/// - `x` — Input signal
/// - `fs` — Sampling frequency in Hz
pub fn instantaneous_frequency<S: Scalar>(x: &[S], fs: f64) -> Vec<S> {
    let analytic = hilbert(x);
    let n = analytic.len();
    if n < 2 {
        return vec![S::ZERO; n];
    }

    // Compute phase (in f64 for stability), then convert back
    let pi = core::f64::consts::PI;
    let pi2 = 2.0 * pi;

    let mut phase: Vec<f64> = analytic.iter().map(|c| c.arg().to_f64()).collect();

    // Unwrap phase
    for i in 1..n {
        let mut d = phase[i] - phase[i - 1];
        while d > pi {
            d -= pi2;
        }
        while d < -pi {
            d += pi2;
        }
        phase[i] = phase[i - 1] + d;
    }

    // Instantaneous frequency = d(phase)/dt / (2*pi)
    let mut freq = vec![0.0_f64; n];
    for i in 1..n - 1 {
        freq[i] = (phase[i + 1] - phase[i - 1]) * fs / (4.0 * pi);
    }
    // Boundary: forward/backward difference
    if n >= 2 {
        freq[0] = (phase[1] - phase[0]) * fs / pi2;
        freq[n - 1] = (phase[n - 1] - phase[n - 2]) * fs / pi2;
    }

    freq.iter().map(|&v| S::from_f64(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn test_hilbert_pure_sine() {
        // Hilbert transform of sin should have envelope ~1.0
        let n = 128;
        let freq = 5.0;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n)
            .map(|i| (pi2 * freq * i as f64 / n as f64).sin())
            .collect();
        let z = hilbert(&x);
        assert_eq!(z.len(), n);

        // Envelope should be approximately 1.0 (away from edges)
        for zi in &z[10..n - 10] {
            let amp = zi.abs();
            assert!(
                (amp.to_f64() - 1.0).abs() < 0.1,
                "envelope should be ~1.0, got {amp:?}"
            );
        }
    }

    #[test]
    fn test_hilbert_real_part() {
        // Real part of analytic signal should equal the original signal
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let z = hilbert(&x);

        for (i, (&xi, zi)) in x.iter().zip(z.iter()).enumerate() {
            assert!(
                (xi - zi.re).abs() < 1e-10,
                "sample {i}: expected {xi}, got {}",
                zi.re
            );
        }
    }

    #[test]
    fn test_hilbert_empty() {
        assert!(hilbert::<f64>(&[]).is_empty());
    }

    #[test]
    fn test_envelope_am_signal() {
        // AM signal: envelope should track the modulation
        let n = 256;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (1.0 + 0.5 * (pi2 * 3.0 * t).cos()) * (pi2 * 20.0 * t).sin()
            })
            .collect();
        let env = envelope(&x);
        assert_eq!(env.len(), n);

        // Envelope should be positive
        assert!(env.iter().all(|&e| e >= 0.0));

        // Envelope should roughly follow 1 + 0.5*cos(...)
        // Check that max envelope is near 1.5 and min is near 0.5
        let max_env: f64 = env[20..n - 20].iter().copied().fold(0.0, f64::max);
        let min_env: f64 = env[20..n - 20].iter().copied().fold(f64::MAX, f64::min);
        assert!(max_env > 1.2, "max envelope = {max_env}");
        assert!(min_env < 0.8, "min envelope = {min_env}");
    }

    #[test]
    fn test_instantaneous_frequency() {
        // Pure 10 Hz tone sampled at 100 Hz
        let n = 128;
        let fs = 128.0;
        let freq = 10.0;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n).map(|i| (pi2 * freq * i as f64 / fs).sin()).collect();
        let inst_freq = instantaneous_frequency(&x, fs);

        // Away from edges, instantaneous frequency should be ~10 Hz
        for &f in &inst_freq[10..n - 10] {
            assert!((f - freq).abs() < 1.0, "expected ~{freq} Hz, got {f} Hz");
        }
    }
}
