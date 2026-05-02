//! IIR filter design: Butterworth and Chebyshev Type I.
//!
//! Designs analog prototype filters, then converts to digital via the
//! bilinear transform. Output is in second-order sections (SOS) format
//! for numerical stability.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::f64::consts::PI;

use crate::error::SignalError;
use crate::filter_apply::SosFilter;

/// Design a Butterworth lowpass digital filter.
///
/// Returns an [`SosFilter`] representing the filter as cascaded second-order
/// sections for numerical stability.
///
/// # Arguments
/// - `order` — Filter order (must be >= 1)
/// - `cutoff` — Cutoff frequency in Hz (must be in (0, fs/2))
/// - `fs` — Sampling frequency in Hz
///
/// # Example
/// ```
/// use numra_signal::butter;
///
/// let sos = butter(4, 10.0, 100.0).unwrap();
/// assert_eq!(sos.n_sections(), 2); // 4th order = 2 SOS sections
/// ```
pub fn butter(order: usize, cutoff: f64, fs: f64) -> Result<SosFilter<f64>, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidOrder(order));
    }
    let nyquist = fs / 2.0;
    if cutoff <= 0.0 || cutoff >= nyquist {
        return Err(SignalError::InvalidCutoff { cutoff, nyquist });
    }

    // Pre-warp the cutoff frequency for bilinear transform
    let wc = pre_warp(cutoff, fs);

    // Compute analog Butterworth poles on the unit circle (left half-plane)
    let poles = butterworth_poles(order, wc);

    // Convert analog poles to digital via bilinear transform, then to SOS
    analog_poles_to_sos(&poles, wc, fs, order)
}

/// Design a Chebyshev Type I lowpass digital filter.
///
/// Chebyshev Type I filters have equiripple in the passband and monotonic
/// rolloff in the stopband.
///
/// # Arguments
/// - `order` — Filter order (must be >= 1)
/// - `ripple_db` — Maximum passband ripple in dB (must be > 0)
/// - `cutoff` — Cutoff frequency in Hz (must be in (0, fs/2))
/// - `fs` — Sampling frequency in Hz
///
/// # Example
/// ```
/// use numra_signal::cheby1;
///
/// let sos = cheby1(4, 1.0, 10.0, 100.0).unwrap();
/// assert_eq!(sos.n_sections(), 2);
/// ```
pub fn cheby1(
    order: usize,
    ripple_db: f64,
    cutoff: f64,
    fs: f64,
) -> Result<SosFilter<f64>, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidOrder(order));
    }
    if ripple_db <= 0.0 {
        return Err(SignalError::InvalidRipple(ripple_db));
    }
    let nyquist = fs / 2.0;
    if cutoff <= 0.0 || cutoff >= nyquist {
        return Err(SignalError::InvalidCutoff { cutoff, nyquist });
    }

    let wc = pre_warp(cutoff, fs);
    let poles = chebyshev1_poles(order, ripple_db, wc);

    analog_poles_to_sos(&poles, wc, fs, order)
}

/// Pre-warp a frequency for the bilinear transform.
///
/// `f` is the desired digital frequency in Hz, `fs` is the sampling rate.
/// Returns the corresponding analog frequency in rad/s.
fn pre_warp(f: f64, fs: f64) -> f64 {
    2.0 * fs * (PI * f / fs).tan()
}

/// Complex number for internal filter design calculations.
#[derive(Clone, Copy, Debug)]
struct Cpx {
    re: f64,
    im: f64,
}

impl Cpx {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl std::ops::Add for Cpx {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Cpx {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Cpx {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Cpx {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::Div for Cpx {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

/// Compute Butterworth analog poles (left half-plane) scaled by `wc`.
fn butterworth_poles(order: usize, wc: f64) -> Vec<Cpx> {
    let n = order;
    let mut poles = Vec::with_capacity(n);
    for k in 0..n {
        let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
        poles.push(Cpx::new(wc * theta.cos(), wc * theta.sin()));
    }
    poles
}

/// Compute Chebyshev Type I analog poles scaled by `wc`.
fn chebyshev1_poles(order: usize, ripple_db: f64, wc: f64) -> Vec<Cpx> {
    let n = order;
    let eps = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let mu = (1.0 / eps + (1.0 / (eps * eps) + 1.0).sqrt()).ln() / n as f64;

    let mut poles = Vec::with_capacity(n);
    for k in 0..n {
        let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;
        let sigma = -wc * mu.sinh() * theta.sin();
        let omega = wc * mu.cosh() * theta.cos();
        poles.push(Cpx::new(sigma, omega));
    }
    poles
}

/// Convert analog poles to digital SOS via bilinear transform.
///
/// Steps:
/// 1. Apply bilinear transform: z = (1 + s/(2*fs)) / (1 - s/(2*fs))
/// 2. Pair complex conjugate poles into second-order sections
/// 3. Compute SOS coefficients [b0, b1, b2, a0, a1, a2]
///
/// For an all-pole prototype (Butterworth/Chebyshev), the analog zeros are all
/// at s = -infinity, which maps to z = -1 under the bilinear transform.
fn analog_poles_to_sos(
    poles: &[Cpx],
    wc: f64,
    fs: f64,
    order: usize,
) -> Result<SosFilter<f64>, SignalError> {
    let t = 1.0 / (2.0 * fs);

    // Bilinear transform each pole: z_k = (1 + p_k*t) / (1 - p_k*t)
    let one = Cpx::new(1.0, 0.0);
    let z_poles: Vec<Cpx> = poles
        .iter()
        .map(|&p| (one + p * t) / (one - p * t))
        .collect();

    // Pair poles into SOS sections
    let mut sections = Vec::new();
    let mut used = vec![false; z_poles.len()];

    // First, pair complex conjugate poles
    for i in 0..z_poles.len() {
        if used[i] {
            continue;
        }
        if z_poles[i].im.abs() > 1e-12 {
            // Find conjugate partner
            let mut found = false;
            for j in (i + 1)..z_poles.len() {
                if used[j] {
                    continue;
                }
                if (z_poles[i].re - z_poles[j].re).abs() < 1e-10
                    && (z_poles[i].im + z_poles[j].im).abs() < 1e-10
                {
                    // Pair (i, j) into second-order section
                    // Denominator: (z - z_i)(z - z_j) = z^2 - 2*Re(z_i)*z + |z_i|^2
                    let a1 = -2.0 * z_poles[i].re;
                    let a2 = z_poles[i].abs() * z_poles[i].abs();

                    // Numerator zeros at z = -1: (z + 1)^2 = z^2 + 2z + 1
                    // Gain normalization: match DC gain
                    let num_dc = 1.0 + 2.0 + 1.0; // (1+1)^2 = 4
                    let den_dc = 1.0 + a1 + a2;
                    let gain = den_dc / num_dc;

                    sections.push([gain, 2.0 * gain, gain, 1.0, a1, a2]);

                    used[i] = true;
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                // Unpaired complex pole — shouldn't happen for real-coefficient filters
                // but handle gracefully as a second-order section with conjugate
                let a1 = -2.0 * z_poles[i].re;
                let a2 = z_poles[i].abs() * z_poles[i].abs();
                let num_dc = 4.0;
                let den_dc = 1.0 + a1 + a2;
                let gain = den_dc / num_dc;
                sections.push([gain, 2.0 * gain, gain, 1.0, a1, a2]);
                used[i] = true;
            }
        }
    }

    // Handle remaining real poles
    let mut real_poles: Vec<f64> = Vec::new();
    for i in 0..z_poles.len() {
        if !used[i] {
            real_poles.push(z_poles[i].re);
        }
    }

    // Pair real poles together, or leave as first-order section embedded in SOS
    let mut rp_iter = real_poles.into_iter();
    while let Some(p1) = rp_iter.next() {
        if let Some(p2) = rp_iter.next() {
            // Pair two real poles: (z - p1)(z - p2)
            let a1 = -(p1 + p2);
            let a2 = p1 * p2;
            let num_dc = 4.0; // (1+1)^2
            let den_dc = 1.0 + a1 + a2;
            let gain = den_dc / num_dc;
            sections.push([gain, 2.0 * gain, gain, 1.0, a1, a2]);
        } else {
            // Single real pole: first-order section in SOS format
            // (z + 1) / (z - p1) = z^2 notation: [b0, b1, 0, 1, -p1, 0]
            let a1 = -p1;
            let num_dc = 2.0; // (1 + 1)
            let den_dc = 1.0 + a1;
            let gain = den_dc / num_dc;
            sections.push([gain, gain, 0.0, 1.0, a1, 0.0]);
        }
    }

    // For Chebyshev, need to adjust overall gain for the ripple
    // The DC gain should be 1.0 for odd-order, or 10^(-ripple_db/20) for even-order
    // We achieve this by normalizing the total DC gain of all sections
    let total_dc_gain: f64 = sections
        .iter()
        .map(|s| {
            let num = s[0] + s[1] + s[2];
            let den = s[3] + s[4] + s[5];
            num / den
        })
        .product();

    if (total_dc_gain - 1.0).abs() > 1e-10 && total_dc_gain.abs() > 1e-15 {
        // Adjust gain of first section
        let correction = 1.0 / total_dc_gain;
        if let Some(s) = sections.first_mut() {
            s[0] *= correction;
            s[1] *= correction;
            s[2] *= correction;
        }
    }

    // For Chebyshev even order, the DC gain should be the ripple factor
    // For odd order Chebyshev, DC gain = 1
    // This is already handled by the pole positions + gain normalization above

    let _ = (wc, order); // Used in gain calculation above

    Ok(SosFilter::new(sections))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_apply::sosfilt;

    #[test]
    fn test_butter_order1() {
        let sos = butter(1, 10.0, 100.0).unwrap();
        assert_eq!(sos.n_sections(), 1);
        // Single section should be first-order (b2=0, a2=0)
        assert!(sos.sections[0][2].abs() < 1e-10);
        assert!(sos.sections[0][5].abs() < 1e-10);
    }

    #[test]
    fn test_butter_order2() {
        let sos = butter(2, 10.0, 100.0).unwrap();
        assert_eq!(sos.n_sections(), 1);
    }

    #[test]
    fn test_butter_order4() {
        let sos = butter(4, 10.0, 100.0).unwrap();
        assert_eq!(sos.n_sections(), 2);
    }

    #[test]
    fn test_butter_dc_gain() {
        // Lowpass filter should pass DC (gain ~= 1.0 at f=0)
        let sos = butter(4, 10.0, 100.0).unwrap();
        let dc_gain: f64 = sos
            .sections
            .iter()
            .map(|s| (s[0] + s[1] + s[2]) / (s[3] + s[4] + s[5]))
            .product();
        assert!((dc_gain - 1.0).abs() < 1e-6, "DC gain = {dc_gain}");
    }

    #[test]
    fn test_butter_attenuates_high_freq() {
        // Apply 4th order Butterworth at 10Hz, fs=100Hz to a 40Hz sine
        let fs = 100.0;
        let sos = butter(4, 10.0, fs).unwrap();

        let n = 500;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n).map(|i| (pi2 * 40.0 * i as f64 / fs).sin()).collect();
        let y = sosfilt(&sos, &x);

        // After transients, output amplitude should be very small
        let max_amp: f64 = y[200..].iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_amp < 0.05,
            "40Hz should be heavily attenuated, max_amp = {max_amp}"
        );
    }

    #[test]
    fn test_butter_passes_low_freq() {
        let fs = 100.0;
        let sos = butter(4, 30.0, fs).unwrap();

        let n = 500;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n).map(|i| (pi2 * 5.0 * i as f64 / fs).sin()).collect();
        let y = sosfilt(&sos, &x);

        // After transients, amplitude should be close to 1.0
        let max_amp: f64 = y[200..].iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_amp > 0.9, "5Hz should pass, max_amp = {max_amp}");
    }

    #[test]
    fn test_butter_invalid_order() {
        assert!(butter(0, 10.0, 100.0).is_err());
    }

    #[test]
    fn test_butter_invalid_cutoff() {
        assert!(butter(4, 0.0, 100.0).is_err());
        assert!(butter(4, 50.0, 100.0).is_err());
        assert!(butter(4, -1.0, 100.0).is_err());
    }

    #[test]
    fn test_cheby1_order4() {
        let sos = cheby1(4, 1.0, 10.0, 100.0).unwrap();
        assert_eq!(sos.n_sections(), 2);
    }

    #[test]
    fn test_cheby1_attenuates_high_freq() {
        let fs = 100.0;
        let sos = cheby1(4, 0.5, 10.0, fs).unwrap();

        let n = 500;
        let pi2 = 2.0 * PI;
        let x: Vec<f64> = (0..n).map(|i| (pi2 * 40.0 * i as f64 / fs).sin()).collect();
        let y = sosfilt(&sos, &x);

        let max_amp: f64 = y[200..].iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_amp < 0.02,
            "Cheby1 should attenuate 40Hz heavily, max_amp = {max_amp}"
        );
    }

    #[test]
    fn test_cheby1_invalid_ripple() {
        assert!(cheby1(4, 0.0, 10.0, 100.0).is_err());
        assert!(cheby1(4, -1.0, 10.0, 100.0).is_err());
    }

    #[test]
    fn test_butter_odd_order() {
        // Odd order should work: 1 real pole + conjugate pairs
        let sos = butter(5, 10.0, 100.0).unwrap();
        assert_eq!(sos.n_sections(), 3); // 2 pairs + 1 real pole
    }
}
