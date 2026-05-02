#![allow(clippy::approx_constant)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_range_loop)]

//! FFT and spectral analysis for Numra.
//!
//! Provides Fast Fourier Transform, power spectral density, windowing,
//! convolution, and short-time Fourier transform. Built on `rustfft`.
//!
//! All public functions are generic over `S: Scalar` (f32/f64), with
//! the FFT computation internally performed in f64 via rustfft.
//!
//! # Example
//!
//! ```rust
//! use numra_fft::{fft, ifft, Complex};
//!
//! let signal: Vec<Complex<f64>> = vec![
//!     Complex::new(1.0_f64, 0.0),
//!     Complex::new(0.0, 0.0),
//!     Complex::new(0.0, 0.0),
//!     Complex::new(0.0, 0.0),
//! ];
//! let spectrum = fft(&signal);
//! let recovered = ifft(&spectrum);
//! for (a, b) in signal.iter().zip(recovered.iter()) {
//!     assert!((a.re - b.re).abs() < 1e-12);
//! }
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod complex;
pub mod convolution;
pub mod fft_core;
pub mod real;
pub mod spectral;
pub mod utils;

pub use complex::{Complex, ComplexF64};
pub use convolution::{fftconvolve, fftcorrelate};
pub use fft_core::{fft, fft2, ifft};
pub use real::{irfft, rfft};
pub use spectral::{psd, stft, welch, StftResult};
pub use utils::{fftfreq, fftshift, ifftshift, window_func, Window};
