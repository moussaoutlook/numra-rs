//! Uncertainty propagation and sensitivity analysis.
//!
//! This module provides tools for:
//! - Representing uncertain values (mean + variance)
//! - First-order uncertainty propagation
//! - Parameter sensitivity analysis
//!
//! # Theory
//!
//! For a function y = f(x) where x has uncertainty σ_x, the first-order
//! approximation of the output uncertainty is:
//!
//! ```text
//! σ_y ≈ |∂f/∂x| * σ_x
//! ```
//!
//! For multiple parameters: σ_y² = Σ (∂f/∂xᵢ)² * σ_xᵢ²
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::Scalar;

/// An uncertain value with mean and variance.
///
/// # Example
///
/// ```rust
/// use numra_core::uncertainty::Uncertain;
///
/// // A measurement of 10.0 ± 0.5 (std dev)
/// let x: Uncertain<f64> = Uncertain::new(10.0, 0.25); // variance = 0.5²
///
/// // Scale by 2: mean scales, variance scales by 4
/// let y = x.scale(2.0);
/// assert!((y.mean - 20.0).abs() < 1e-10);
/// assert!((y.variance - 1.0).abs() < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uncertain<S: Scalar> {
    /// Mean value
    pub mean: S,
    /// Variance (σ²)
    pub variance: S,
}

impl<S: Scalar> Default for Uncertain<S> {
    fn default() -> Self {
        Self {
            mean: S::ZERO,
            variance: S::ZERO,
        }
    }
}

impl<S: Scalar> Uncertain<S> {
    /// Create a new uncertain value.
    pub fn new(mean: S, variance: S) -> Self {
        Self { mean, variance }
    }

    /// Create from mean and standard deviation.
    pub fn from_std(mean: S, std: S) -> Self {
        Self {
            mean,
            variance: std * std,
        }
    }

    /// Create a certain value (zero variance).
    pub fn certain(value: S) -> Self {
        Self {
            mean: value,
            variance: S::ZERO,
        }
    }

    /// Standard deviation.
    pub fn std(&self) -> S {
        self.variance.sqrt()
    }

    /// Coefficient of variation (std/mean).
    pub fn cv(&self) -> S {
        /// Threshold below which the mean is considered zero for CV computation.
        const CV_ZERO_THRESHOLD: f64 = 1e-15;
        if self.mean.abs() < S::from_f64(CV_ZERO_THRESHOLD) {
            S::INFINITY
        } else {
            self.std() / self.mean.abs()
        }
    }

    /// Scale by a constant: y = a*x.
    /// Variance scales by a².
    pub fn scale(&self, a: S) -> Self {
        Self {
            mean: a * self.mean,
            variance: a * a * self.variance,
        }
    }

    /// Add a constant: y = x + c.
    /// Variance unchanged.
    pub fn add_const(&self, c: S) -> Self {
        Self {
            mean: self.mean + c,
            variance: self.variance,
        }
    }

    /// Add another uncertain value (assuming independence).
    /// y = x1 + x2, Var(y) = Var(x1) + Var(x2).
    pub fn add(&self, other: &Self) -> Self {
        Self {
            mean: self.mean + other.mean,
            variance: self.variance + other.variance,
        }
    }

    /// Subtract another uncertain value (assuming independence).
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            mean: self.mean - other.mean,
            variance: self.variance + other.variance,
        }
    }

    /// Multiply by another uncertain value (first-order approximation).
    /// For `y = x1 * x2`:
    /// `E[y] ≈ E[x1] * E[x2]`
    /// `Var(y) ≈ E[x2]² * Var(x1) + E[x1]² * Var(x2)`
    pub fn mul(&self, other: &Self) -> Self {
        let mean = self.mean * other.mean;
        let variance =
            other.mean * other.mean * self.variance + self.mean * self.mean * other.variance;
        Self { mean, variance }
    }

    /// Divide by another uncertain value (first-order approximation).
    /// For `y = x1 / x2`:
    /// `E[y] ≈ E[x1] / E[x2]`
    /// `Var(y) ≈ (1/E[x2]²) * Var(x1) + (E[x1]/E[x2]²)² * Var(x2)`
    pub fn div(&self, other: &Self) -> Self {
        let mean = self.mean / other.mean;
        let inv_mean2 = S::ONE / (other.mean * other.mean);
        let variance = inv_mean2 * self.variance
            + (self.mean * inv_mean2) * (self.mean * inv_mean2) * other.variance;
        Self { mean, variance }
    }

    /// Apply a function with known derivative.
    /// For y = f(x): Var(y) ≈ (f'(x))² * Var(x).
    pub fn apply<F, D>(&self, f: F, df: D) -> Self
    where
        F: Fn(S) -> S,
        D: Fn(S) -> S,
    {
        let mean = f(self.mean);
        let deriv = df(self.mean);
        let variance = deriv * deriv * self.variance;
        Self { mean, variance }
    }

    /// Square: y = x².
    pub fn square(&self) -> Self {
        self.apply(|x| x * x, |x| S::from_f64(2.0) * x)
    }

    /// Square root: y = √x.
    pub fn sqrt_unc(&self) -> Self {
        self.apply(|x| x.sqrt(), |x| S::ONE / (S::from_f64(2.0) * x.sqrt()))
    }

    /// Exponential: y = e^x.
    pub fn exp(&self) -> Self {
        self.apply(|x| x.exp(), |x| x.exp())
    }

    /// Natural log: y = ln(x).
    pub fn ln(&self) -> Self {
        self.apply(|x| x.ln(), |x| S::ONE / x)
    }

    /// Sine: y = sin(x).
    pub fn sin(&self) -> Self {
        self.apply(|x| x.sin(), |x| x.cos())
    }

    /// Cosine: y = cos(x).
    pub fn cos(&self) -> Self {
        self.apply(|x| x.cos(), |x| -x.sin())
    }
}

/// Importance of one parameter for a scalar pure-function output.
///
/// Computed by central finite differences inside [`compute_sensitivities`].
/// Distinct from the ODE forward-sensitivity matrix
/// `numra_ode::SensitivityResult` (which carries `dy(t)/dp` along a trajectory):
/// this type is for ranking parameters of a fixed scalar function `f(p) -> S`.
#[derive(Clone, Debug)]
pub struct ParameterSensitivity<S: Scalar> {
    /// Parameter name
    pub name: String,
    /// Sensitivity coefficient: ∂y/∂p
    pub coefficient: S,
    /// Normalized sensitivity: (p/y) * ∂y/∂p
    pub normalized: S,
}

/// Result of parameter-importance analysis on a scalar pure function.
///
/// Distinct from the ODE forward-sensitivity result type
/// `numra_ode::SensitivityResult`. See [`ParameterSensitivity`] for the
/// distinction.
#[derive(Clone, Debug)]
pub struct ParameterSensitivityResult<S: Scalar> {
    /// Output value at nominal parameters
    pub output: S,
    /// Sensitivities for each parameter
    pub sensitivities: Vec<ParameterSensitivity<S>>,
}

impl<S: Scalar> ParameterSensitivityResult<S> {
    /// Create a new sensitivity result.
    pub fn new(output: S, sensitivities: Vec<ParameterSensitivity<S>>) -> Self {
        Self {
            output,
            sensitivities,
        }
    }

    /// Find the most sensitive parameter.
    pub fn most_sensitive(&self) -> Option<&ParameterSensitivity<S>> {
        self.sensitivities
            .iter()
            .max_by(|a, b| a.normalized.abs().partial_cmp(&b.normalized.abs()).unwrap())
    }

    /// Propagate uncertainty from parameter uncertainties.
    /// Returns the output uncertainty (variance).
    pub fn propagate_uncertainty(&self, param_variances: &[S]) -> S {
        assert_eq!(param_variances.len(), self.sensitivities.len());
        let mut total_var = S::ZERO;
        for (sens, &var) in self.sensitivities.iter().zip(param_variances.iter()) {
            total_var += sens.coefficient * sens.coefficient * var;
        }
        total_var
    }
}

/// Compute finite difference sensitivities.
///
/// # Arguments
/// * `f` - Function to evaluate
/// * `params` - Nominal parameter values
/// * `names` - Parameter names
/// * `h` - Step size for finite differences (default: `cbrt(S::EPSILON)`,
///   the canonical central-FD step factor)
pub fn compute_sensitivities<S: Scalar, F>(
    f: F,
    params: &[S],
    names: &[&str],
    h: Option<S>,
) -> ParameterSensitivityResult<S>
where
    F: Fn(&[S]) -> S,
{
    let h = h.unwrap_or(S::EPSILON.cbrt());
    let output = f(params);

    let mut sensitivities = Vec::with_capacity(params.len());
    let mut params_pert = params.to_vec();

    for (i, name) in names.iter().enumerate() {
        let p0 = params[i];
        let step = h * (S::ONE + p0.abs());

        // Central difference
        params_pert[i] = p0 + step;
        let f_plus = f(&params_pert);
        params_pert[i] = p0 - step;
        let f_minus = f(&params_pert);
        params_pert[i] = p0;

        let coeff = (f_plus - f_minus) / (S::from_f64(2.0) * step);
        const NORMALIZED_ZERO_THRESHOLD: f64 = 1e-15;
        let normalized = if output.abs() > S::from_f64(NORMALIZED_ZERO_THRESHOLD) {
            (p0 / output) * coeff
        } else {
            S::ZERO
        };

        sensitivities.push(ParameterSensitivity {
            name: name.to_string(),
            coefficient: coeff,
            normalized,
        });
    }

    ParameterSensitivityResult::new(output, sensitivities)
}

/// Interval arithmetic for bounds propagation.
///
/// Represents an interval [lower, upper].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval<S: Scalar> {
    /// Lower bound
    pub lower: S,
    /// Upper bound
    pub upper: S,
}

impl<S: Scalar> Interval<S> {
    /// Create a new interval.
    pub fn new(lower: S, upper: S) -> Self {
        debug_assert!(lower <= upper, "Invalid interval: lower > upper");
        Self { lower, upper }
    }

    /// Create a point interval [x, x].
    pub fn point(x: S) -> Self {
        Self { lower: x, upper: x }
    }

    /// Create from center and half-width.
    pub fn from_center(center: S, half_width: S) -> Self {
        Self {
            lower: center - half_width,
            upper: center + half_width,
        }
    }

    /// Width of the interval.
    pub fn width(&self) -> S {
        self.upper - self.lower
    }

    /// Center of the interval.
    pub fn center(&self) -> S {
        (self.lower + self.upper) / S::from_f64(2.0)
    }

    /// Check if interval contains a value.
    pub fn contains(&self, x: S) -> bool {
        x >= self.lower && x <= self.upper
    }

    /// Add two intervals.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            lower: self.lower + other.lower,
            upper: self.upper + other.upper,
        }
    }

    /// Subtract two intervals.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            lower: self.lower - other.upper,
            upper: self.upper - other.lower,
        }
    }

    /// Multiply two intervals.
    pub fn mul(&self, other: &Self) -> Self {
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        let lower = products.iter().fold(S::INFINITY, |a, &b| a.min(b));
        let upper = products.iter().fold(S::NEG_INFINITY, |a, &b| a.max(b));
        Self { lower, upper }
    }

    /// Scale by a constant.
    pub fn scale(&self, a: S) -> Self {
        if a >= S::ZERO {
            Self {
                lower: a * self.lower,
                upper: a * self.upper,
            }
        } else {
            Self {
                lower: a * self.upper,
                upper: a * self.lower,
            }
        }
    }

    /// Union of two intervals.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
        }
    }

    /// Intersection of two intervals (None if empty).
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        if lower <= upper {
            Some(Self { lower, upper })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertain_basic() {
        let x: Uncertain<f64> = Uncertain::from_std(10.0, 0.5);
        assert!((x.mean - 10.0).abs() < 1e-10);
        assert!((x.variance - 0.25).abs() < 1e-10);
        assert!((x.std() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_scale() {
        let x: Uncertain<f64> = Uncertain::from_std(10.0, 1.0);
        let y = x.scale(2.0);
        assert!((y.mean - 20.0).abs() < 1e-10);
        assert!((y.std() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_add() {
        let x: Uncertain<f64> = Uncertain::new(10.0, 1.0);
        let y: Uncertain<f64> = Uncertain::new(5.0, 0.5);
        let z = x.add(&y);
        assert!((z.mean - 15.0).abs() < 1e-10);
        assert!((z.variance - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_mul() {
        // x = 10 ± 1, y = 5 ± 0.5
        // E[xy] ≈ 50
        // Var(xy) ≈ 25*1 + 100*0.25 = 25 + 25 = 50
        let x: Uncertain<f64> = Uncertain::new(10.0, 1.0);
        let y: Uncertain<f64> = Uncertain::new(5.0, 0.25);
        let z = x.mul(&y);
        assert!((z.mean - 50.0).abs() < 1e-10);
        assert!((z.variance - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_exp() {
        // For y = e^x at x=0: dy/dx = e^0 = 1
        // So Var(y) ≈ 1² * Var(x) = Var(x)
        let x: Uncertain<f64> = Uncertain::new(0.0, 0.01);
        let y = x.exp();
        assert!((y.mean - 1.0).abs() < 1e-10);
        assert!((y.variance - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_sensitivity_analysis() {
        // f(a, b) = a * b
        // ∂f/∂a = b = 2, ∂f/∂b = a = 3
        let f = |p: &[f64]| p[0] * p[1];
        let params = [3.0, 2.0];
        let names = ["a", "b"];

        let result = compute_sensitivities(f, &params, &names, None);

        assert!((result.output - 6.0).abs() < 1e-10);
        assert!((result.sensitivities[0].coefficient - 2.0).abs() < 1e-5);
        assert!((result.sensitivities[1].coefficient - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_interval_basic() {
        let i: Interval<f64> = Interval::new(1.0, 3.0);
        assert!((i.width() - 2.0).abs() < 1e-10);
        assert!((i.center() - 2.0).abs() < 1e-10);
        assert!(i.contains(2.0));
        assert!(!i.contains(4.0));
    }

    #[test]
    fn test_interval_add() {
        let a: Interval<f64> = Interval::new(1.0, 2.0);
        let b: Interval<f64> = Interval::new(3.0, 4.0);
        let c = a.add(&b);
        assert!((c.lower - 4.0).abs() < 1e-10);
        assert!((c.upper - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_mul() {
        let a: Interval<f64> = Interval::new(-1.0, 2.0);
        let b: Interval<f64> = Interval::new(1.0, 3.0);
        let c = a.mul(&b);
        // Products: -1*1=-1, -1*3=-3, 2*1=2, 2*3=6
        // So c = [-3, 6]
        assert!((c.lower + 3.0).abs() < 1e-10);
        assert!((c.upper - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_intersection() {
        let a: Interval<f64> = Interval::new(1.0, 4.0);
        let b: Interval<f64> = Interval::new(2.0, 5.0);
        let c = a.intersection(&b).unwrap();
        assert!((c.lower - 2.0).abs() < 1e-10);
        assert!((c.upper - 4.0).abs() < 1e-10);

        let d: Interval<f64> = Interval::new(5.0, 6.0);
        assert!(a.intersection(&d).is_none());
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    #[test]
    fn test_uncertain_zero_variance() {
        let x: Uncertain<f64> = Uncertain::certain(5.0);
        assert!(x.variance.abs() < 1e-15);
        assert!((x.std()).abs() < 1e-15);

        // Operations preserve certainty
        let y = x.scale(2.0);
        assert!(y.variance.abs() < 1e-15);

        let z = x.add_const(10.0);
        assert!(z.variance.abs() < 1e-15);
    }

    #[test]
    fn test_uncertain_zero_mean() {
        let x: Uncertain<f64> = Uncertain::new(0.0, 1.0);

        // CV should be infinity for zero mean
        assert!(x.cv().is_infinite());

        // Scaling zero still gives zero
        let y = x.scale(100.0);
        assert!(y.mean.abs() < 1e-15);
        assert!((y.variance - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_negative_values() {
        let x: Uncertain<f64> = Uncertain::from_std(-10.0, 2.0);
        assert!((x.mean + 10.0).abs() < 1e-10);

        // Scaling by negative flips the mean
        let y = x.scale(-1.0);
        assert!((y.mean - 10.0).abs() < 1e-10);
        assert!((y.variance - x.variance).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_small_values() {
        let x: Uncertain<f64> = Uncertain::from_std(1e-15, 1e-16);
        let y: Uncertain<f64> = Uncertain::from_std(1e-15, 1e-16);

        let z = x.add(&y);
        assert!(z.mean > 0.0);
        assert!(z.variance > 0.0);
    }

    #[test]
    fn test_uncertain_large_values() {
        let x: Uncertain<f64> = Uncertain::from_std(1e15, 1e14);
        let y = x.scale(0.5);
        assert!((y.mean - 5e14).abs() < 1e10);
    }

    #[test]
    fn test_interval_point() {
        let i: Interval<f64> = Interval::point(5.0);
        assert!((i.width()).abs() < 1e-15);
        assert!(i.contains(5.0));
        assert!(!i.contains(5.0 + 1e-10));
    }

    #[test]
    fn test_interval_negative() {
        let i: Interval<f64> = Interval::new(-5.0, -2.0);
        assert!(i.contains(-3.0));
        assert!(!i.contains(0.0));
        assert!((i.center() + 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_interval_scale_negative() {
        let i: Interval<f64> = Interval::new(2.0, 4.0);
        let j = i.scale(-2.0);
        // [-2*4, -2*2] = [-8, -4]
        assert!((j.lower + 8.0).abs() < 1e-10);
        assert!((j.upper + 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_subtract() {
        let a: Interval<f64> = Interval::new(2.0, 4.0);
        let b: Interval<f64> = Interval::new(1.0, 3.0);
        let c = a.sub(&b);
        // [2-3, 4-1] = [-1, 3]
        assert!((c.lower + 1.0).abs() < 1e-10);
        assert!((c.upper - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_union() {
        let a: Interval<f64> = Interval::new(1.0, 3.0);
        let b: Interval<f64> = Interval::new(5.0, 7.0);
        let c = a.union(&b);
        assert!((c.lower - 1.0).abs() < 1e-10);
        assert!((c.upper - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_self_intersection() {
        let a: Interval<f64> = Interval::new(1.0, 5.0);
        let b = a.intersection(&a).unwrap();
        assert!((b.lower - a.lower).abs() < 1e-10);
        assert!((b.upper - a.upper).abs() < 1e-10);
    }

    #[test]
    fn test_uncertain_div_small_denominator() {
        let x: Uncertain<f64> = Uncertain::new(10.0, 1.0);
        let y: Uncertain<f64> = Uncertain::new(0.01, 0.0001);
        let z = x.div(&y);
        // Mean should be large
        assert!(z.mean > 100.0);
    }

    #[test]
    fn test_sensitivity_single_param() {
        // f(x) = x^2
        let f = |p: &[f64]| p[0] * p[0];
        let params = [3.0];
        let names = ["x"];

        let result = compute_sensitivities(f, &params, &names, None);
        // ∂f/∂x = 2x = 6
        assert!((result.sensitivities[0].coefficient - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_sensitivity_zero_output() {
        // f(x, y) = x - x = 0
        let f = |p: &[f64]| p[0] - p[0];
        let params = [5.0];
        let names = ["x"];

        let result = compute_sensitivities(f, &params, &names, None);
        assert!(result.output.abs() < 1e-15);
        // Normalized should be zero because output is zero
        assert!(result.sensitivities[0].normalized.abs() < 1e-10);
    }
}
