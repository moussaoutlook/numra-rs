//! Apply digital filters via second-order sections.
//!
//! Provides [`SosFilter`] representation and forward/backward filtering.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// A digital filter represented as cascaded second-order sections.
///
/// Each section is `[b0, b1, b2, a0, a1, a2]` where the transfer function is:
///
/// ```text
/// H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
/// ```
///
/// SOS representation is numerically more stable than transfer function
/// (b, a) representation, especially for high-order filters.
#[derive(Clone, Debug)]
pub struct SosFilter<S: Scalar> {
    /// Second-order sections: each is `[b0, b1, b2, a0, a1, a2]`.
    pub sections: Vec<[S; 6]>,
}

impl<S: Scalar> SosFilter<S> {
    /// Create a new SOS filter from sections.
    pub fn new(sections: Vec<[S; 6]>) -> Self {
        Self { sections }
    }

    /// Number of second-order sections.
    pub fn n_sections(&self) -> usize {
        self.sections.len()
    }

    /// Overall filter order (2 * number of sections).
    pub fn order(&self) -> usize {
        self.sections.len() * 2
    }
}

/// Apply an SOS filter to a signal (forward pass only).
///
/// Uses Direct Form II transposed implementation for each section.
///
/// # Example
/// ```
/// use numra_signal::{SosFilter, sosfilt};
///
/// // Identity filter: H(z) = 1
/// let sos = SosFilter::new(vec![[1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]]);
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y = sosfilt(&sos, &x);
/// assert!((y[0] - 1.0).abs() < 1e-12);
/// ```
pub fn sosfilt<S: Scalar>(sos: &SosFilter<S>, x: &[S]) -> Vec<S> {
    if x.is_empty() || sos.sections.is_empty() {
        return x.to_vec();
    }

    let mut y = x.to_vec();

    for section in &sos.sections {
        let b0 = section[0];
        let b1 = section[1];
        let b2 = section[2];
        let a0 = section[3];
        let a1 = section[4];
        let a2 = section[5];

        // Normalize by a0
        let b0 = b0 / a0;
        let b1 = b1 / a0;
        let b2 = b2 / a0;
        let a1 = a1 / a0;
        let a2 = a2 / a0;

        // Direct Form II transposed
        let mut d1 = S::ZERO;
        let mut d2 = S::ZERO;

        for yi in y.iter_mut() {
            let xi = *yi;
            let out = b0 * xi + d1;
            d1 = b1 * xi - a1 * out + d2;
            d2 = b2 * xi - a2 * out;
            *yi = out;
        }
    }

    y
}

/// Zero-phase digital filtering via forward-backward SOS filtering.
///
/// Applies the filter forward, then backward, resulting in zero phase
/// distortion and doubled filter order. The signal is extended at both
/// ends using reflection to reduce edge effects.
///
/// # Example
/// ```
/// use numra_signal::{butter, filtfilt};
///
/// let fs = 100.0;
/// let n = 200;
/// let pi2 = 2.0 * std::f64::consts::PI;
/// // 5 Hz signal + 40 Hz noise
/// let x: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (pi2 * 5.0 * t).sin() + 0.5 * (pi2 * 40.0 * t).sin()
/// }).collect();
///
/// let sos = butter(4, 10.0, fs).unwrap();
/// let y = filtfilt(&sos, &x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn filtfilt<S: Scalar>(sos: &SosFilter<S>, x: &[S]) -> Vec<S> {
    let n = x.len();
    if n < 4 {
        // Too short for meaningful filtfilt; just return forward filter
        return sosfilt(sos, x);
    }

    // Pad length: 3 * max(len(a), len(b)) per scipy convention
    // For SOS, each section is order 2, so padlen = 3 * (2 * n_sections)
    let padlen = (6 * sos.n_sections()).min(n - 1);

    // Reflect-pad the signal: [x[padlen]..x[1] reversed, x, x[n-2]..x[n-1-padlen] reversed]
    let mut padded = Vec::with_capacity(n + 2 * padlen);

    let two = S::from_f64(2.0);

    // Left reflection: 2*x[0] - x[padlen..1]
    let x0 = x[0];
    for i in (1..=padlen).rev() {
        padded.push(two * x0 - x[i]);
    }

    // Original signal
    padded.extend_from_slice(x);

    // Right reflection: 2*x[n-1] - x[n-2..n-1-padlen]
    let xn = x[n - 1];
    for i in (n - 1 - padlen..n - 1).rev() {
        padded.push(two * xn - x[i]);
    }

    // Forward pass
    let mut y = sosfilt(sos, &padded);

    // Reverse
    y.reverse();

    // Backward pass
    y = sosfilt(sos, &y);

    // Reverse again
    y.reverse();

    // Strip padding
    y[padlen..padlen + n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sosfilt_identity() {
        let sos = SosFilter::new(vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = sosfilt(&sos, &x);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sosfilt_first_order() {
        // Simple first-order lowpass: H(z) = 0.5 / (1 - 0.5*z^-1)
        let sos = SosFilter::new(vec![[0.5, 0.0, 0.0, 1.0, -0.5, 0.0]]);
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = sosfilt(&sos, &x);
        // Impulse response: 0.5, 0.25, 0.125, 0.0625, ...
        assert!((y[0] - 0.5).abs() < 1e-12);
        assert!((y[1] - 0.25).abs() < 1e-12);
        assert!((y[2] - 0.125).abs() < 1e-12);
        assert!((y[3] - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn test_sosfilt_empty() {
        let sos = SosFilter::<f64>::new(vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]);
        let y = sosfilt(&sos, &[]);
        assert!(y.is_empty());
    }

    #[test]
    fn test_filtfilt_preserves_length() {
        let sos = SosFilter::new(vec![[0.5, 0.5, 0.0, 1.0, -0.5, 0.0]]);
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y = filtfilt(&sos, &x);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn test_filtfilt_dc_passthrough() {
        // DC signal should pass through a properly designed lowpass filter unchanged
        use crate::filter_design::butter;
        let sos = butter(4, 10.0, 100.0).unwrap();
        let x = vec![3.0; 100];
        let y = filtfilt(&sos, &x);
        // Middle samples should be very close to 3.0
        for &yi in &y[20..80] {
            assert!((yi - 3.0).abs() < 0.01, "expected ~3.0, got {yi}");
        }
    }

    #[test]
    fn test_sosfilt_cascade() {
        // Two identity sections = identity
        let sos = SosFilter::new(vec![
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = sosfilt(&sos, &x);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
