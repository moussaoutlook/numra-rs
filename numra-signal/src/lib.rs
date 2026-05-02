//! # Numra Signal Processing
//!
//! Digital signal processing tools for Numra: IIR/FIR filter design and
//! application, resampling, Hilbert transform, and peak detection.
//!
//! ## Modules
//!
//! - [`filter_design`] — Butterworth and Chebyshev Type I IIR filter design
//! - [`filter_apply`] — SOS filter application (`sosfilt`, `filtfilt`)
//! - [`fir`] — FIR filter design via windowed sinc
//! - [`mod@resample`] — FFT-based signal resampling
//! - [`mod@hilbert`] — Hilbert transform, envelope, instantaneous frequency
//! - [`peak`] — Peak detection with height/distance/prominence constraints
//!
//! ## Example
//!
//! ```rust
//! use numra_signal::{butter, filtfilt, find_peaks, PeakOptions};
//!
//! // Design a 4th-order Butterworth lowpass at 10 Hz, sampled at 100 Hz
//! let sos = butter(4, 10.0, 100.0).unwrap();
//!
//! // Generate a noisy signal
//! let pi2 = 2.0 * std::f64::consts::PI;
//! let x: Vec<f64> = (0..200).map(|i| {
//!     let t = i as f64 / 100.0;
//!     (pi2 * 3.0 * t).sin() + 0.3 * (pi2 * 40.0 * t).sin()
//! }).collect();
//!
//! // Filter the signal (zero-phase)
//! let y = filtfilt(&sos, &x);
//!
//! // Find peaks in the filtered signal
//! let peaks = find_peaks(&y, &PeakOptions::default().height(0.5));
//! assert!(!peaks.is_empty());
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]

pub mod error;
pub mod filter_apply;
pub mod filter_design;
pub mod fir;
pub mod hilbert;
pub mod peak;
pub mod resample;

pub use error::SignalError;
pub use filter_apply::{filtfilt, sosfilt, SosFilter};
pub use filter_design::{butter, cheby1};
pub use fir::{fir_filter, firwin};
pub use hilbert::{envelope, hilbert, instantaneous_frequency};
pub use peak::{find_peaks, PeakOptions};
pub use resample::resample;
