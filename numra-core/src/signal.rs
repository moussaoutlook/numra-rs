//! Signal generators for forcing functions.
//!
//! This module provides time-dependent signals commonly used as forcing terms
//! in differential equations (earthquake records, wind loads, control inputs, etc.)
//!
//! # Signal Types
//!
//! ## Deterministic Signals
//! - [`Harmonic`] - Sinusoidal signal: A·sin(2πft + φ)
//! - [`Step`] - Step function with optional smoothing
//! - [`Ramp`] - Linear ramp between start and end times
//! - [`Pulse`] - Rectangular pulse with given duration
//! - [`Chirp`] - Frequency-swept sine wave
//!
//! ## Data-Driven Signals
//! - [`Tabulated`] - Interpolated from time-value pairs
//! - [`FromFile`] - Loaded from CSV file (requires `std` feature)
//!
//! ## Composite Signals
//! - [`Piecewise`] - Different signals in different time ranges
//! - [`Sum`] - Sum of two signals
//! - [`Product`] - Product of two signals
//!
//! # Example
//!
//! ```rust
//! use numra_core::signal::{Signal, Harmonic, Step, Sum};
//!
//! // Create a forcing function: harmonic + step
//! let harmonic = Harmonic::new(1.0, 2.0, 0.0);  // A=1, f=2Hz, φ=0
//! let step = Step::new(0.5, 1.0);                // magnitude=0.5 at t=1
//! let forcing = Sum::new(harmonic, step);
//!
//! // Evaluate at t = 1.5s
//! let value = forcing.eval(1.5);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use crate::Scalar;

/// A time-dependent signal for forcing functions.
///
/// Signals are evaluated at time `t` to produce a scalar value.
/// They are used as forcing terms in ODEs/SDEs.
pub trait Signal<S: Scalar>: Send + Sync {
    /// Evaluate the signal at time t.
    fn eval(&self, t: S) -> S;

    /// Compute the derivative of the signal at time `t`.
    ///
    /// Default: central finite differences with the textbook
    /// precision-aware step `h = cbrt(S::EPSILON) * (1 + |t|)` —
    /// optimal for central FD by the balance of truncation (`O(h²)`)
    /// and round-off (`O(eps_mach / h)`) error. The
    /// `cbrt(S::EPSILON)` form (not a hardcoded constant) keeps FD
    /// useful at every `Scalar` precision: `f64` lands at `≈6.06e-6`,
    /// `f32` at `≈4.92e-3`. A hardcoded `1e-8` would fall below
    /// `f32::EPSILON ≈ 1.19e-7` and quantise the perturbation to zero.
    /// Override when an analytical derivative is available — most
    /// closed-form `Signal` impls in this module already do.
    fn eval_derivative(&self, t: S) -> S {
        let h = S::EPSILON.cbrt() * (S::ONE + t.abs());
        (self.eval(t + h) - self.eval(t - h)) / (S::TWO * h)
    }
}

// ============================================================================
// Harmonic Signal: A·sin(2πft + φ)
// ============================================================================

/// Harmonic (sinusoidal) signal.
///
/// Evaluates to: `amplitude * sin(2π * frequency * t + phase)`
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Harmonic};
///
/// let signal: Harmonic<f64> = Harmonic::new(2.0, 1.0, 0.0);  // 2*sin(2πt)
/// let value = signal.eval(0.25);  // sin(π/2) = 1, so result is 2.0
/// assert!((value - 2.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct Harmonic<S: Scalar> {
    pub amplitude: S,
    pub frequency: S,
    pub phase: S,
}

impl<S: Scalar> Harmonic<S> {
    /// Create a new harmonic signal.
    ///
    /// # Arguments
    /// - `amplitude` - Peak amplitude
    /// - `frequency` - Frequency in Hz
    /// - `phase` - Phase offset in radians
    pub fn new(amplitude: S, frequency: S, phase: S) -> Self {
        Self {
            amplitude,
            frequency,
            phase,
        }
    }
}

impl<S: Scalar> Signal<S> for Harmonic<S> {
    #[inline]
    fn eval(&self, t: S) -> S {
        let omega_t = S::TWO * S::PI * self.frequency * t + self.phase;
        self.amplitude * omega_t.sin()
    }

    #[inline]
    fn eval_derivative(&self, t: S) -> S {
        let omega = S::TWO * S::PI * self.frequency;
        let omega_t = omega * t + self.phase;
        self.amplitude * omega * omega_t.cos()
    }
}

// ============================================================================
// Step Signal
// ============================================================================

/// Step function signal.
///
/// Returns 0 for t < time, and magnitude for t >= time.
/// Optionally applies smooth transition using tanh.
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Step};
///
/// let step: Step<f64> = Step::new(1.0, 2.0);  // Unit step at t=2
/// assert!(step.eval(1.0).abs() < 1e-10);   // Before step
/// assert!((step.eval(3.0) - 1.0).abs() < 1e-10);  // After step
/// ```
#[derive(Clone, Debug)]
pub struct Step<S: Scalar> {
    pub magnitude: S,
    pub time: S,
    /// Smoothing parameter (None = sharp step)
    pub smoothing: Option<S>,
}

impl<S: Scalar> Step<S> {
    /// Create a sharp step function.
    pub fn new(magnitude: S, time: S) -> Self {
        Self {
            magnitude,
            time,
            smoothing: None,
        }
    }

    /// Create a smooth step function with tanh transition.
    ///
    /// The `smoothing` parameter controls the transition width.
    /// Smaller values = sharper transition.
    pub fn smooth(magnitude: S, time: S, smoothing: S) -> Self {
        Self {
            magnitude,
            time,
            smoothing: Some(smoothing),
        }
    }
}

impl<S: Scalar> Signal<S> for Step<S> {
    fn eval(&self, t: S) -> S {
        match self.smoothing {
            None => {
                if t >= self.time {
                    self.magnitude
                } else {
                    S::ZERO
                }
            }
            Some(k) => {
                // Smooth transition using tanh
                let x = (t - self.time) / k;
                self.magnitude * S::HALF * (S::ONE + x.tanh())
            }
        }
    }
}

// ============================================================================
// Ramp Signal
// ============================================================================

/// Linear ramp signal.
///
/// Returns 0 before start, ramps linearly from start to end,
/// then returns final value after end.
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Ramp};
///
/// let ramp: Ramp<f64> = Ramp::new(0.0, 2.0, 1.0, 3.0);  // from 0 to 2 between t=1 and t=3
/// assert!(ramp.eval(0.5).abs() < 1e-10);      // Before ramp
/// assert!((ramp.eval(2.0) - 1.0).abs() < 1e-10);  // Midpoint
/// assert!((ramp.eval(4.0) - 2.0).abs() < 1e-10);  // After ramp
/// ```
#[derive(Clone, Debug)]
pub struct Ramp<S: Scalar> {
    pub start_value: S,
    pub end_value: S,
    pub start_time: S,
    pub end_time: S,
}

impl<S: Scalar> Ramp<S> {
    /// Create a new ramp signal.
    pub fn new(start_value: S, end_value: S, start_time: S, end_time: S) -> Self {
        Self {
            start_value,
            end_value,
            start_time,
            end_time,
        }
    }

    /// Create a ramp from zero with given rate.
    pub fn from_rate(rate: S, start_time: S, end_time: S) -> Self {
        let duration = end_time - start_time;
        Self {
            start_value: S::ZERO,
            end_value: rate * duration,
            start_time,
            end_time,
        }
    }
}

impl<S: Scalar> Signal<S> for Ramp<S> {
    fn eval(&self, t: S) -> S {
        if t <= self.start_time {
            self.start_value
        } else if t >= self.end_time {
            self.end_value
        } else {
            let alpha = (t - self.start_time) / (self.end_time - self.start_time);
            self.start_value + alpha * (self.end_value - self.start_value)
        }
    }

    fn eval_derivative(&self, t: S) -> S {
        if t > self.start_time && t < self.end_time {
            (self.end_value - self.start_value) / (self.end_time - self.start_time)
        } else {
            S::ZERO
        }
    }
}

// ============================================================================
// Pulse Signal
// ============================================================================

/// Rectangular pulse signal.
///
/// Returns magnitude during [start, start + duration], zero otherwise.
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Pulse};
///
/// let pulse: Pulse<f64> = Pulse::new(5.0, 1.0, 2.0);  // magnitude=5, starts at t=1, duration=2
/// assert!(pulse.eval(0.5).abs() < 1e-10);     // Before pulse
/// assert!((pulse.eval(2.0) - 5.0).abs() < 1e-10);  // During pulse
/// assert!(pulse.eval(4.0).abs() < 1e-10);     // After pulse
/// ```
#[derive(Clone, Debug)]
pub struct Pulse<S: Scalar> {
    pub magnitude: S,
    pub start: S,
    pub duration: S,
}

impl<S: Scalar> Pulse<S> {
    /// Create a new pulse signal.
    pub fn new(magnitude: S, start: S, duration: S) -> Self {
        Self {
            magnitude,
            start,
            duration,
        }
    }
}

impl<S: Scalar> Signal<S> for Pulse<S> {
    fn eval(&self, t: S) -> S {
        if t >= self.start && t <= self.start + self.duration {
            self.magnitude
        } else {
            S::ZERO
        }
    }
}

// ============================================================================
// Chirp Signal (Frequency Sweep)
// ============================================================================

/// Chirp (frequency-swept sinusoid) signal.
///
/// Linear frequency sweep from f0 to f1 over duration t1.
/// Useful for structural dynamics testing.
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Chirp};
///
/// let chirp: Chirp<f64> = Chirp::new(1.0, 1.0, 10.0, 5.0);  // A=1, f: 1→10 Hz over 5s
/// let value: f64 = chirp.eval(2.5);  // Somewhere in the middle
/// assert!(value.abs() <= 1.0);  // Bounded by amplitude
/// ```
#[derive(Clone, Debug)]
pub struct Chirp<S: Scalar> {
    pub amplitude: S,
    pub f0: S, // Start frequency (Hz)
    pub f1: S, // End frequency (Hz)
    pub t1: S, // Duration of sweep
}

impl<S: Scalar> Chirp<S> {
    /// Create a new linear chirp.
    pub fn new(amplitude: S, f0: S, f1: S, t1: S) -> Self {
        Self {
            amplitude,
            f0,
            f1,
            t1,
        }
    }
}

impl<S: Scalar> Signal<S> for Chirp<S> {
    fn eval(&self, t: S) -> S {
        if t < S::ZERO || t > self.t1 {
            return S::ZERO;
        }
        // Linear chirp: f(t) = f0 + (f1 - f0) * t / t1
        // Phase: φ(t) = 2π ∫ f(τ) dτ = 2π (f0*t + (f1-f0)*t²/(2*t1))
        let k = (self.f1 - self.f0) / self.t1;
        let phase = S::TWO * S::PI * (self.f0 * t + S::HALF * k * t * t);
        self.amplitude * phase.sin()
    }
}

// ============================================================================
// Tabulated Signal (Interpolated Data)
// ============================================================================

/// Interpolation method for tabulated data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Linear interpolation between points.
    #[default]
    Linear,
    /// Nearest neighbor (zero-order hold).
    Nearest,
    /// Cubic spline (not yet implemented).
    Cubic,
}

/// Signal from tabulated time-value data.
///
/// Interpolates between given data points using linear, nearest-neighbor,
/// or cubic spline interpolation.
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Tabulated, Interpolation};
///
/// let times: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
/// let values: Vec<f64> = vec![0.0, 1.0, 0.0, 2.0];
/// let signal = Tabulated::new(times, values, Interpolation::Linear);
///
/// assert!((signal.value_at(0.5) - 0.5).abs() < 1e-10);  // Midpoint interpolation
/// assert!((signal.value_at(1.0) - 1.0).abs() < 1e-10);  // Exact point
/// ```
///
/// # Cubic Spline Interpolation
///
/// When using `Interpolation::Cubic`, natural cubic splines provide
/// smooth C² continuous interpolation between data points.
#[derive(Clone, Debug)]
pub struct Tabulated<S: Scalar> {
    pub times: Vec<S>,
    pub values: Vec<S>,
    pub interp: Interpolation,
    /// Second derivatives at each point (for cubic spline interpolation).
    /// Computed once during construction when interp == Cubic.
    spline_d2: Option<Vec<S>>,
}

impl<S: Scalar> Tabulated<S> {
    /// Create a new tabulated signal.
    ///
    /// # Panics
    /// Panics if times and values have different lengths, or if times is empty.
    pub fn new(times: Vec<S>, values: Vec<S>, interp: Interpolation) -> Self {
        assert_eq!(
            times.len(),
            values.len(),
            "times and values must have same length"
        );
        assert!(!times.is_empty(), "times must not be empty");

        // Compute spline coefficients if cubic interpolation is requested
        let spline_d2 = if interp == Interpolation::Cubic && times.len() >= 2 {
            Some(Self::compute_spline_coefficients(&times, &values))
        } else {
            None
        };

        Self {
            times,
            values,
            interp,
            spline_d2,
        }
    }

    /// Create from slice pairs.
    pub fn from_pairs(pairs: &[(S, S)], interp: Interpolation) -> Self {
        let (times, values): (Vec<_>, Vec<_>) = pairs.iter().cloned().unzip();
        Self::new(times, values, interp)
    }

    /// Compute natural cubic spline second derivatives.
    ///
    /// Uses the Thomas algorithm to solve the tridiagonal system for
    /// natural spline boundary conditions (second derivative = 0 at endpoints).
    fn compute_spline_coefficients(times: &[S], values: &[S]) -> Vec<S> {
        let n = times.len();
        if n < 2 {
            return vec![S::ZERO; n];
        }
        if n == 2 {
            // Linear case - second derivatives are zero
            return vec![S::ZERO, S::ZERO];
        }

        // Set up the tridiagonal system for natural spline
        // For interior points i = 1, ..., n-2:
        // h_{i-1} * d2[i-1] + 2*(h_{i-1} + h_i) * d2[i] + h_i * d2[i+1] = 6 * delta
        // where delta = (y[i+1] - y[i])/h_i - (y[i] - y[i-1])/h_{i-1}

        let mut d2 = vec![S::ZERO; n];

        // Allocate arrays for Thomas algorithm
        let mut c_prime = vec![S::ZERO; n - 2]; // Modified upper diagonal
        let mut d_prime = vec![S::ZERO; n - 2]; // Modified RHS

        // Compute h_i = t[i+1] - t[i]
        let mut h = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            h.push(times[i + 1] - times[i]);
        }

        // Forward sweep of Thomas algorithm
        for i in 1..n - 1 {
            let h_prev = h[i - 1];
            let h_curr = h[i];

            // Diagonal element: 2 * (h_{i-1} + h_i)
            let diag = S::TWO * (h_prev + h_curr);

            // RHS: 6 * ((y[i+1] - y[i])/h_i - (y[i] - y[i-1])/h_{i-1})
            let slope_curr = (values[i + 1] - values[i]) / h_curr;
            let slope_prev = (values[i] - values[i - 1]) / h_prev;
            let rhs = S::from_f64(6.0) * (slope_curr - slope_prev);

            let idx = i - 1;
            if idx == 0 {
                // First equation (natural BC: d2[0] = 0)
                c_prime[idx] = h_curr / diag;
                d_prime[idx] = rhs / diag;
            } else {
                let m = diag - h_prev * c_prime[idx - 1];
                c_prime[idx] = h_curr / m;
                d_prime[idx] = (rhs - h_prev * d_prime[idx - 1]) / m;
            }
        }

        // Back substitution
        // Natural BC: d2[n-1] = 0, so we only solve for d2[1..n-1]
        let last_idx = n - 3;
        d2[n - 2] = d_prime[last_idx];

        for i in (1..n - 2).rev() {
            let idx = i - 1;
            d2[i] = d_prime[idx] - c_prime[idx] * d2[i + 1];
        }

        // d2[0] = 0 and d2[n-1] = 0 (natural boundary conditions)
        d2[0] = S::ZERO;
        d2[n - 1] = S::ZERO;

        d2
    }

    /// Get the value at time t (convenience method).
    pub fn value_at(&self, t: S) -> S {
        self.interpolate_at(t)
    }

    /// Internal interpolation method.
    fn interpolate_at(&self, t: S) -> S {
        let n = self.times.len();

        // Extrapolation: hold constant
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= self.times[n - 1] {
            return self.values[n - 1];
        }

        let i = self.find_interval(t);
        let t0 = self.times[i];
        let t1 = self.times[i + 1];
        let v0 = self.values[i];
        let v1 = self.values[i + 1];

        match self.interp {
            Interpolation::Nearest => {
                if t - t0 < t1 - t {
                    v0
                } else {
                    v1
                }
            }
            Interpolation::Linear => {
                let alpha = (t - t0) / (t1 - t0);
                v0 + alpha * (v1 - v0)
            }
            Interpolation::Cubic => self.cubic_spline_interp(t, i),
        }
    }

    /// Cubic spline interpolation using precomputed second derivatives.
    fn cubic_spline_interp(&self, t: S, i: usize) -> S {
        let d2 = match &self.spline_d2 {
            Some(d2) => d2,
            None => {
                // Fallback to linear if spline coefficients not available
                let t0 = self.times[i];
                let t1 = self.times[i + 1];
                let alpha = (t - t0) / (t1 - t0);
                return self.values[i] + alpha * (self.values[i + 1] - self.values[i]);
            }
        };

        let t0 = self.times[i];
        let t1 = self.times[i + 1];
        let h = t1 - t0;
        let y0 = self.values[i];
        let y1 = self.values[i + 1];
        let d2_0 = d2[i];
        let d2_1 = d2[i + 1];

        // Cubic spline formula:
        // S(t) = (1-u)*y0 + u*y1 + h²/6 * [(u³-u)*d2_1 + ((1-u)³-(1-u))*d2_0]
        // where u = (t - t0) / h

        let u = (t - t0) / h;
        let one_minus_u = S::ONE - u;

        // Cubic terms
        let u3 = u * u * u;
        let omu3 = one_minus_u * one_minus_u * one_minus_u;

        // h² / 6
        let h2_6 = h * h / S::from_f64(6.0);

        // S(t) = (1-u)*y0 + u*y1 + h²/6 * [(u³-u)*d2_1 + ((1-u)³-(1-u))*d2_0]
        one_minus_u * y0 + u * y1 + h2_6 * ((u3 - u) * d2_1 + (omu3 - one_minus_u) * d2_0)
    }

    /// Binary search for interval containing t.
    fn find_interval(&self, t: S) -> usize {
        let n = self.times.len();
        if t <= self.times[0] {
            return 0;
        }
        if t >= self.times[n - 1] {
            return n - 2;
        }
        // Binary search
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t < self.times[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    }
}

impl<S: Scalar> Signal<S> for Tabulated<S> {
    fn eval(&self, t: S) -> S {
        self.interpolate_at(t)
    }
}

// ============================================================================
// FromFile Signal (std only)
// ============================================================================

/// Signal loaded from a CSV file.
///
/// Reads time-value pairs from a file and interpolates.
/// This is useful for loading earthquake records, experimental data, etc.
///
/// # File Format
///
/// The file should be in CSV format with two columns: time and value.
/// Lines starting with '#' are treated as comments.
///
/// ```text
/// # Time, Value
/// 0.0, 0.0
/// 0.1, 0.5
/// 0.2, 1.0
/// ```
///
/// # Example
///
/// ```
/// use numra_core::signal::{Signal, FromFile};
///
/// // Load a CSV file with time,value columns
/// let signal: FromFile<f64> = FromFile::load("test_data/earthquake.csv").unwrap();
/// let accel = signal.eval(0.15);  // Interpolated value at t=0.15
/// assert!(accel > 0.0 && accel < 1.0);  // Interpolated between 0.5 and 1.0
/// ```
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct FromFile<S: Scalar> {
    /// Underlying tabulated data
    tabulated: Tabulated<S>,
    /// Original file path (for debugging)
    path: std::string::String,
}

#[cfg(feature = "std")]
impl<S: Scalar + std::str::FromStr> FromFile<S> {
    /// Load signal data from a CSV file.
    ///
    /// # Arguments
    /// - `path` - Path to the CSV file
    ///
    /// # Returns
    /// - `Ok(FromFile)` if the file was loaded successfully
    /// - `Err(String)` if there was an error reading or parsing the file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, std::string::String> {
        Self::load_with_interpolation(path, Interpolation::Linear)
    }

    /// Load signal data with specified interpolation method.
    pub fn load_with_interpolation<P: AsRef<std::path::Path>>(
        path: P,
        interp: Interpolation,
    ) -> Result<Self, std::string::String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .map_err(|e| format!("Failed to open file '{}': {}", path_ref.display(), e))?;

        let reader = BufReader::new(file);
        let mut times = Vec::new();
        let mut values = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line =
                line_result.map_err(|e| format!("Failed to read line {}: {}", line_num + 1, e))?;

            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse CSV line
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                return Err(format!(
                    "Line {} has fewer than 2 columns: '{}'",
                    line_num + 1,
                    line
                ));
            }

            let t: S = parts[0].parse().map_err(|_| {
                format!(
                    "Failed to parse time on line {}: '{}'",
                    line_num + 1,
                    parts[0]
                )
            })?;
            let v: S = parts[1].parse().map_err(|_| {
                format!(
                    "Failed to parse value on line {}: '{}'",
                    line_num + 1,
                    parts[1]
                )
            })?;

            times.push(t);
            values.push(v);
        }

        if times.is_empty() {
            return Err("No data found in file".to_string());
        }

        let tabulated = Tabulated::new(times, values, interp);
        let path_string = path_ref.to_string_lossy().into_owned();

        Ok(Self {
            tabulated,
            path: path_string,
        })
    }

    /// Load from raw CSV string content.
    pub fn from_csv_string(
        content: &str,
        interp: Interpolation,
    ) -> Result<Self, std::string::String> {
        let mut times = Vec::new();
        let mut values = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse CSV line
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                return Err(format!(
                    "Line {} has fewer than 2 columns: '{}'",
                    line_num + 1,
                    line
                ));
            }

            let t: S = parts[0].parse().map_err(|_| {
                format!(
                    "Failed to parse time on line {}: '{}'",
                    line_num + 1,
                    parts[0]
                )
            })?;
            let v: S = parts[1].parse().map_err(|_| {
                format!(
                    "Failed to parse value on line {}: '{}'",
                    line_num + 1,
                    parts[1]
                )
            })?;

            times.push(t);
            values.push(v);
        }

        if times.is_empty() {
            return Err("No data found in content".to_string());
        }

        let tabulated = Tabulated::new(times, values, interp);

        Ok(Self {
            tabulated,
            path: "<from_string>".to_string(),
        })
    }

    /// Get the number of data points.
    pub fn len(&self) -> usize {
        self.tabulated.times.len()
    }

    /// Check if the signal is empty.
    pub fn is_empty(&self) -> bool {
        self.tabulated.times.is_empty()
    }

    /// Get the time range.
    pub fn time_range(&self) -> (S, S) {
        let times = &self.tabulated.times;
        (times[0], times[times.len() - 1])
    }

    /// Get the file path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the underlying tabulated data.
    pub fn as_tabulated(&self) -> &Tabulated<S> {
        &self.tabulated
    }
}

#[cfg(feature = "std")]
impl<S: Scalar + std::str::FromStr> Signal<S> for FromFile<S> {
    fn eval(&self, t: S) -> S {
        self.tabulated.eval(t)
    }

    fn eval_derivative(&self, t: S) -> S {
        self.tabulated.eval_derivative(t)
    }
}

// ============================================================================
// Piecewise Signal
// ============================================================================

/// Piecewise signal with different values in different time ranges.
///
/// # Example
///
/// ```rust
/// use numra_core::signal::{Signal, Piecewise};
///
/// // Value 1.0 for t < 1, value 2.0 for 1 ≤ t < 3, value 0.0 for t ≥ 3
/// let signal: Piecewise<f64> = Piecewise::new(vec![
///     (1.0, 1.0),   // until t=1, value=1.0
///     (3.0, 2.0),   // until t=3, value=2.0
/// ], 0.0);          // default value after all segments
///
/// assert!((signal.eval(0.5) - 1.0).abs() < 1e-10);
/// assert!((signal.eval(2.0) - 2.0).abs() < 1e-10);
/// assert!(signal.eval(4.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct Piecewise<S: Scalar> {
    /// Segments as (end_time, value) pairs, sorted by end_time.
    segments: Vec<(S, S)>,
    /// Default value after all segments.
    default: S,
}

impl<S: Scalar> Piecewise<S> {
    /// Create a new piecewise constant signal.
    ///
    /// # Arguments
    /// - `segments` - List of (end_time, value) pairs. The value applies until end_time.
    /// - `default` - Value after all segments end.
    pub fn new(segments: Vec<(S, S)>, default: S) -> Self {
        Self { segments, default }
    }
}

impl<S: Scalar> Signal<S> for Piecewise<S> {
    fn eval(&self, t: S) -> S {
        for &(end_time, value) in &self.segments {
            if t < end_time {
                return value;
            }
        }
        self.default
    }
}

// ============================================================================
// Composite Signals: Sum and Product
// ============================================================================

/// Sum of two signals: f(t) = a(t) + b(t)
#[derive(Clone, Debug)]
pub struct Sum<S: Scalar, A: Signal<S>, B: Signal<S>> {
    pub a: A,
    pub b: B,
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar, A: Signal<S>, B: Signal<S>> Sum<S, A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: Scalar, A: Signal<S>, B: Signal<S>> Signal<S> for Sum<S, A, B> {
    #[inline]
    fn eval(&self, t: S) -> S {
        self.a.eval(t) + self.b.eval(t)
    }

    #[inline]
    fn eval_derivative(&self, t: S) -> S {
        self.a.eval_derivative(t) + self.b.eval_derivative(t)
    }
}

/// Product of two signals: f(t) = a(t) * b(t)
#[derive(Clone, Debug)]
pub struct Product<S: Scalar, A: Signal<S>, B: Signal<S>> {
    pub a: A,
    pub b: B,
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar, A: Signal<S>, B: Signal<S>> Product<S, A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: Scalar, A: Signal<S>, B: Signal<S>> Signal<S> for Product<S, A, B> {
    #[inline]
    fn eval(&self, t: S) -> S {
        self.a.eval(t) * self.b.eval(t)
    }

    #[inline]
    fn eval_derivative(&self, t: S) -> S {
        // Product rule: (a*b)' = a'*b + a*b'
        self.a.eval_derivative(t) * self.b.eval(t) + self.a.eval(t) * self.b.eval_derivative(t)
    }
}

/// Scaled signal: f(t) = scale * inner(t)
#[derive(Clone, Debug)]
pub struct Scaled<S: Scalar, Inner: Signal<S>> {
    pub scale: S,
    pub inner: Inner,
}

impl<S: Scalar, Inner: Signal<S>> Scaled<S, Inner> {
    pub fn new(scale: S, inner: Inner) -> Self {
        Self { scale, inner }
    }
}

impl<S: Scalar, Inner: Signal<S>> Signal<S> for Scaled<S, Inner> {
    #[inline]
    fn eval(&self, t: S) -> S {
        self.scale * self.inner.eval(t)
    }

    #[inline]
    fn eval_derivative(&self, t: S) -> S {
        self.scale * self.inner.eval_derivative(t)
    }
}

// ============================================================================
// Constant Signal
// ============================================================================

/// Constant signal: f(t) = value
#[derive(Clone, Debug)]
pub struct Constant<S: Scalar> {
    pub value: S,
}

impl<S: Scalar> Constant<S> {
    pub fn new(value: S) -> Self {
        Self { value }
    }
}

impl<S: Scalar> Signal<S> for Constant<S> {
    #[inline]
    fn eval(&self, _t: S) -> S {
        self.value
    }

    #[inline]
    fn eval_derivative(&self, _t: S) -> S {
        S::ZERO
    }
}

// ============================================================================
// Zero Signal
// ============================================================================

/// Zero signal: f(t) = 0
#[derive(Clone, Debug, Default)]
pub struct Zero<S: Scalar> {
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar> Zero<S> {
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: Scalar> Signal<S> for Zero<S> {
    #[inline]
    fn eval(&self, _t: S) -> S {
        S::ZERO
    }

    #[inline]
    fn eval_derivative(&self, _t: S) -> S {
        S::ZERO
    }
}

// ============================================================================
// Boxed Signal for Dynamic Dispatch
// ============================================================================

impl<S: Scalar> Signal<S> for Box<dyn Signal<S>> {
    fn eval(&self, t: S) -> S {
        (**self).eval(t)
    }

    fn eval_derivative(&self, t: S) -> S {
        (**self).eval_derivative(t)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_harmonic() {
        let h = Harmonic::new(2.0, 1.0, 0.0); // 2*sin(2πt)

        // sin(0) = 0
        assert!(h.eval(0.0).abs() < TOL);

        // sin(π/2) = 1 at t=0.25
        assert!((h.eval(0.25) - 2.0).abs() < TOL);

        // sin(π) = 0 at t=0.5
        assert!(h.eval(0.5).abs() < TOL);

        // sin(3π/2) = -1 at t=0.75
        assert!((h.eval(0.75) + 2.0).abs() < TOL);
    }

    #[test]
    fn test_harmonic_derivative() {
        let h = Harmonic::new(1.0, 1.0, 0.0); // sin(2πt)

        // d/dt sin(2πt) = 2π cos(2πt)
        // At t=0: 2π*cos(0) = 2π
        let deriv = h.eval_derivative(0.0);
        assert!((deriv - 2.0 * core::f64::consts::PI).abs() < TOL);
    }

    #[test]
    fn test_step_sharp() {
        let s = Step::new(1.0, 2.0);

        assert!(s.eval(1.0).abs() < TOL);
        assert!(s.eval(1.999).abs() < TOL);
        assert!((s.eval(2.0) - 1.0).abs() < TOL);
        assert!((s.eval(3.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_step_smooth() {
        let s = Step::smooth(1.0, 2.0, 0.1);

        // Far before step
        assert!(s.eval(0.0).abs() < 0.01);

        // At step time: should be 0.5
        assert!((s.eval(2.0) - 0.5).abs() < TOL);

        // Far after step
        assert!((s.eval(4.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ramp() {
        let r = Ramp::new(0.0, 2.0, 1.0, 3.0);

        // Before ramp
        assert!(r.eval(0.5).abs() < TOL);

        // Start of ramp
        assert!(r.eval(1.0).abs() < TOL);

        // Midpoint
        assert!((r.eval(2.0) - 1.0).abs() < TOL);

        // End of ramp
        assert!((r.eval(3.0) - 2.0).abs() < TOL);

        // After ramp
        assert!((r.eval(4.0) - 2.0).abs() < TOL);
    }

    #[test]
    fn test_ramp_derivative() {
        let r = Ramp::new(0.0, 4.0, 1.0, 3.0); // 4/(3-1) = 2 slope

        // Derivative should be 2 during ramp
        assert!((r.eval_derivative(2.0) - 2.0).abs() < TOL);

        // Derivative should be 0 outside ramp
        assert!(r.eval_derivative(0.5).abs() < TOL);
        assert!(r.eval_derivative(4.0).abs() < TOL);
    }

    #[test]
    fn test_pulse() {
        let p = Pulse::new(5.0, 1.0, 2.0);

        // Before pulse
        assert!(p.eval(0.5).abs() < TOL);

        // During pulse
        assert!((p.eval(1.0) - 5.0).abs() < TOL);
        assert!((p.eval(2.0) - 5.0).abs() < TOL);
        assert!((p.eval(3.0) - 5.0).abs() < TOL);

        // After pulse
        assert!(p.eval(3.5).abs() < TOL);
    }

    #[test]
    fn test_chirp() {
        let c = Chirp::new(1.0, 1.0, 10.0, 5.0);

        // Should be zero outside [0, t1]
        assert!(c.eval(-1.0).abs() < TOL);
        assert!(c.eval(6.0).abs() < TOL);

        // Should be bounded by amplitude inside
        assert!(c.eval(2.5).abs() <= 1.0 + TOL);

        // At t=0, phase=0, so sin(0)=0
        assert!(c.eval(0.0).abs() < TOL);
    }

    #[test]
    fn test_tabulated_linear() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![0.0, 2.0, 1.0, 3.0];
        let t = Tabulated::new(times, values, Interpolation::Linear);

        // Exact points
        assert!(t.eval(0.0).abs() < TOL);
        assert!((t.eval(1.0) - 2.0).abs() < TOL);
        assert!((t.eval(2.0) - 1.0).abs() < TOL);
        assert!((t.eval(3.0) - 3.0).abs() < TOL);

        // Interpolated
        assert!((t.eval(0.5) - 1.0).abs() < TOL); // Midpoint of 0→2
        assert!((t.eval(1.5) - 1.5).abs() < TOL); // Midpoint of 2→1

        // Extrapolated (hold constant)
        assert!(t.eval(-1.0).abs() < TOL);
        assert!((t.eval(5.0) - 3.0).abs() < TOL);
    }

    #[test]
    fn test_tabulated_nearest() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![0.0, 1.0, 2.0];
        let t = Tabulated::new(times, values, Interpolation::Nearest);

        // Near 0
        assert!(t.eval(0.3).abs() < TOL);

        // Near 1
        assert!((t.eval(0.6) - 1.0).abs() < TOL);
        assert!((t.eval(1.4) - 1.0).abs() < TOL);

        // Near 2
        assert!((t.eval(1.6) - 2.0).abs() < TOL);
    }

    #[test]
    fn test_tabulated_cubic_spline() {
        // Test cubic spline interpolation with a simple dataset
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let t = Tabulated::new(times.clone(), values.clone(), Interpolation::Cubic);

        // Exact points should match
        for (i, (&ti, &vi)) in times.iter().zip(values.iter()).enumerate() {
            let y = t.value_at(ti);
            assert!(
                (y - vi).abs() < 1e-10,
                "Failed at point {}: expected {}, got {}",
                i,
                vi,
                y
            );
        }

        // Cubic spline should be smoother than linear at midpoints
        // For the oscillating data, cubic spline should give values between the neighbors
        let mid_12 = t.value_at(1.5);
        assert!(
            mid_12 < 1.0 && mid_12 > -0.5,
            "Cubic spline midpoint 1.5 out of range: {}",
            mid_12
        );

        // The derivative should be continuous (C² property of natural cubic splines)
        // Check that the second derivative exists by verifying smoothness
        let eps = 0.001;
        let deriv_left = (t.value_at(1.5) - t.value_at(1.5 - eps)) / eps;
        let deriv_right = (t.value_at(1.5 + eps) - t.value_at(1.5)) / eps;
        assert!(
            (deriv_left - deriv_right).abs() < 0.1,
            "Derivative discontinuity at 1.5: left={}, right={}",
            deriv_left,
            deriv_right
        );
    }

    #[test]
    fn test_tabulated_cubic_vs_linear() {
        // Cubic spline should give smoother interpolation than linear
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![0.0, 1.0, 0.5, 1.5];

        let linear = Tabulated::new(times.clone(), values.clone(), Interpolation::Linear);
        let cubic = Tabulated::new(times.clone(), values.clone(), Interpolation::Cubic);

        // At exact points, both should give same values
        for (&ti, &vi) in times.iter().zip(values.iter()) {
            assert!((linear.value_at(ti) - vi).abs() < 1e-10);
            assert!((cubic.value_at(ti) - vi).abs() < 1e-10);
        }

        // At midpoint, cubic will generally differ from linear
        // Both should be reasonable interpolations
        let lin_mid = linear.value_at(1.5);
        let cub_mid = cubic.value_at(1.5);

        // Linear at 1.5: (1.0 + 0.5) / 2 = 0.75
        assert!((lin_mid - 0.75).abs() < 1e-10);

        // Cubic will be different (but still reasonable)
        assert!(
            cub_mid > 0.0 && cub_mid < 2.0,
            "Cubic value {} out of reasonable range",
            cub_mid
        );
    }

    #[test]
    fn test_piecewise() {
        let p = Piecewise::new(vec![(1.0, 10.0), (3.0, 20.0), (5.0, 30.0)], 0.0);

        assert!((p.eval(0.5) - 10.0).abs() < TOL);
        assert!((p.eval(2.0) - 20.0).abs() < TOL);
        assert!((p.eval(4.0) - 30.0).abs() < TOL);
        assert!(p.eval(6.0).abs() < TOL);
    }

    #[test]
    fn test_sum() {
        let a = Constant::new(3.0);
        let b = Constant::new(5.0);
        let sum = Sum::new(a, b);

        assert!((sum.eval(0.0) - 8.0).abs() < TOL);
        assert!((sum.eval(100.0) - 8.0).abs() < TOL);
    }

    #[test]
    fn test_product() {
        let a = Constant::new(3.0);
        let b = Constant::new(5.0);
        let prod = Product::new(a, b);

        assert!((prod.eval(0.0) - 15.0).abs() < TOL);
    }

    #[test]
    fn test_scaled() {
        let h = Harmonic::new(1.0, 1.0, 0.0);
        let scaled = Scaled::new(2.0, h);

        // At t=0.25, sin(π/2) = 1, so scaled = 2
        assert!((scaled.eval(0.25) - 2.0).abs() < TOL);
    }

    /// Pin the f32 viability of the FD-default derivative on `Signal`.
    /// The previous hardcoded `1e-8` step was below
    /// `f32::EPSILON ≈ 1.19e-7`, so the central-FD perturbation
    /// quantised to zero and the derivative came back as `0.0` instead
    /// of the true value. F-FD-STEP switched to the precision-aware
    /// `cbrt(S::EPSILON) * (1 + |t|)`, which scales correctly across
    /// every `Scalar` impl. This test guards against a regression to
    /// the old constant. Uses `Tabulated` because it falls through to
    /// the trait-default FD path (most other built-in signals override
    /// with closed-form derivatives).
    #[test]
    fn test_signal_derivative_f32() {
        // Linear interpolant y(t) = 2t over [0, 4]; analytical
        // derivative is exactly 2.0 everywhere on the interior.
        let times: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let values: Vec<f32> = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let s = Tabulated::new(times, values, Interpolation::Linear);
        let d = s.eval_derivative(2.0_f32);
        assert!(d != 0.0, "FD step quantised to zero on f32");
        assert!(
            (d - 2.0_f32).abs() < 1e-2,
            "central-FD derivative on f32 too inaccurate: d={}",
            d
        );
    }

    #[test]
    fn test_constant_and_zero() {
        let c = Constant::new(42.0);
        let z: Zero<f64> = Zero::new();

        assert!((c.eval(0.0) - 42.0).abs() < TOL);
        assert!((c.eval(1000.0) - 42.0).abs() < TOL);
        assert!(c.eval_derivative(0.0).abs() < TOL);

        assert!(z.eval(0.0).abs() < TOL);
        assert!(z.eval(1000.0).abs() < TOL);
    }

    #[test]
    fn test_composite_signals() {
        // Build: 2*sin(2πt) + step(1, at t=0.5)
        let harmonic = Harmonic::new(2.0, 1.0, 0.0);
        let step = Step::new(1.0, 0.5);
        let composite = Sum::new(harmonic, step);

        // At t=0: sin(0) + 0 = 0
        assert!(composite.eval(0.0).abs() < TOL);

        // At t=0.25: sin(π/2)*2 + 0 = 2
        assert!((composite.eval(0.25) - 2.0).abs() < TOL);

        // At t=0.75: sin(3π/2)*2 + 1 = -2 + 1 = -1
        assert!((composite.eval(0.75) + 1.0).abs() < TOL);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_file_csv_string() {
        let csv = r#"
# Test data
0.0, 0.0
1.0, 2.0
2.0, 1.0
3.0, 3.0
"#;
        let signal: super::FromFile<f64> =
            super::FromFile::from_csv_string(csv, Interpolation::Linear).unwrap();

        assert_eq!(signal.len(), 4);
        assert!(!signal.is_empty());

        let (t_min, t_max) = signal.time_range();
        assert!(t_min.abs() < TOL);
        assert!((t_max - 3.0).abs() < TOL);

        // Test interpolation
        assert!(signal.eval(0.0).abs() < TOL);
        assert!((signal.eval(1.0) - 2.0).abs() < TOL);
        assert!((signal.eval(0.5) - 1.0).abs() < TOL); // Midpoint interpolation
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_file_empty_error() {
        let csv = "# Only comments\n# No data\n";
        let result: Result<super::FromFile<f64>, _> =
            super::FromFile::from_csv_string(csv, Interpolation::Linear);
        assert!(result.is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_file_parse_error() {
        let csv = "0.0, abc\n"; // Invalid value
        let result: Result<super::FromFile<f64>, _> =
            super::FromFile::from_csv_string(csv, Interpolation::Linear);
        assert!(result.is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_file_too_few_columns() {
        let csv = "0.0\n"; // Only one column
        let result: Result<super::FromFile<f64>, _> =
            super::FromFile::from_csv_string(csv, Interpolation::Linear);
        assert!(result.is_err());
    }
}
