//! Memory kernels for integro-differential equations.
//!
//! Common kernel types used in viscoelasticity, heat conduction with memory,
//! and other memory-dependent phenomena.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for memory kernels K(τ) where τ = t - s.
///
/// Convolution kernels depend only on the time difference, not absolute time.
pub trait Kernel<S: Scalar>: Clone {
    /// Evaluate the kernel at time lag τ.
    fn evaluate(&self, tau: S) -> S;

    /// Check if the kernel is a sum of exponentials (Prony series).
    ///
    /// Returns Some((amplitudes, rates)) if the kernel is SOE, None otherwise.
    fn as_prony(&self) -> Option<(Vec<S>, Vec<S>)> {
        None
    }
}

/// Exponential kernel: K(τ) = a * exp(-b * τ).
///
/// Common in viscoelastic materials (Maxwell element).
#[derive(Clone, Debug)]
pub struct ExponentialKernel<S: Scalar> {
    /// Amplitude
    pub a: S,
    /// Decay rate
    pub b: S,
}

impl<S: Scalar> ExponentialKernel<S> {
    pub fn new(a: S, b: S) -> Self {
        Self { a, b }
    }
}

impl<S: Scalar> Kernel<S> for ExponentialKernel<S> {
    fn evaluate(&self, tau: S) -> S {
        self.a * (-self.b * tau).exp()
    }

    fn as_prony(&self) -> Option<(Vec<S>, Vec<S>)> {
        Some((vec![self.a], vec![self.b]))
    }
}

/// Power-law kernel: K(τ) = a * τ^(-α).
///
/// Models fractional memory effects. Note: singular at τ = 0.
#[derive(Clone, Debug)]
pub struct PowerLawKernel<S: Scalar> {
    /// Amplitude
    pub a: S,
    /// Exponent (0 < α < 1 for weak singularity)
    pub alpha: S,
}

impl<S: Scalar> PowerLawKernel<S> {
    pub fn new(a: S, alpha: S) -> Self {
        Self { a, alpha }
    }
}

impl<S: Scalar> Kernel<S> for PowerLawKernel<S> {
    fn evaluate(&self, tau: S) -> S {
        if tau <= S::ZERO {
            S::ZERO // Regularization at τ = 0
        } else {
            self.a * tau.powf(-self.alpha)
        }
    }
}

/// Prony series (Sum of Exponentials) kernel.
///
/// K(τ) = Σᵢ aᵢ * exp(-bᵢ * τ)
///
/// This is the most efficient kernel type for numerical integration
/// because it can be computed recursively without storing full history.
#[derive(Clone, Debug)]
pub struct PronyKernel<S: Scalar> {
    /// Amplitudes aᵢ
    pub amplitudes: Vec<S>,
    /// Decay rates bᵢ
    pub rates: Vec<S>,
}

impl<S: Scalar> PronyKernel<S> {
    /// Create a new Prony series kernel.
    ///
    /// # Arguments
    /// * `amplitudes` - Amplitudes aᵢ
    /// * `rates` - Decay rates bᵢ (must be positive)
    pub fn new(amplitudes: Vec<S>, rates: Vec<S>) -> Self {
        assert_eq!(
            amplitudes.len(),
            rates.len(),
            "Amplitudes and rates must have same length"
        );
        Self { amplitudes, rates }
    }

    /// Create a single exponential kernel.
    pub fn single(a: S, b: S) -> Self {
        Self {
            amplitudes: vec![a],
            rates: vec![b],
        }
    }

    /// Create a two-term Prony series (generalized Maxwell model).
    pub fn two_term(a1: S, b1: S, a2: S, b2: S) -> Self {
        Self {
            amplitudes: vec![a1, a2],
            rates: vec![b1, b2],
        }
    }

    /// Number of terms in the series.
    pub fn num_terms(&self) -> usize {
        self.amplitudes.len()
    }
}

impl<S: Scalar> Kernel<S> for PronyKernel<S> {
    fn evaluate(&self, tau: S) -> S {
        let mut sum = S::ZERO;
        for (a, b) in self.amplitudes.iter().zip(self.rates.iter()) {
            sum += *a * (-*b * tau).exp();
        }
        sum
    }

    fn as_prony(&self) -> Option<(Vec<S>, Vec<S>)> {
        Some((self.amplitudes.clone(), self.rates.clone()))
    }
}

/// Mittag-Leffler kernel (for fractional derivatives).
///
/// K(τ) = τ^(β-1) * E_{α,β}(-λ * τ^α)
///
/// Reduces to exponential when α = β = 1.
#[derive(Clone, Debug)]
pub struct MittagLefflerKernel<S: Scalar> {
    /// Rate parameter λ
    pub lambda: S,
    /// First ML parameter α
    pub alpha: S,
    /// Second ML parameter β
    pub beta: S,
}

impl<S: Scalar> MittagLefflerKernel<S> {
    pub fn new(lambda: S, alpha: S, beta: S) -> Self {
        Self {
            lambda,
            alpha,
            beta,
        }
    }

    /// Standard relaxation kernel with α = β.
    pub fn relaxation(lambda: S, alpha: S) -> Self {
        Self {
            lambda,
            alpha,
            beta: alpha,
        }
    }
}

impl<S: Scalar> Kernel<S> for MittagLefflerKernel<S> {
    fn evaluate(&self, tau: S) -> S {
        if tau <= S::ZERO {
            return S::ZERO;
        }

        // Simple series approximation for E_{α,β}(z)
        let z = -self.lambda * tau.powf(self.alpha);
        let ml = mittag_leffler_approx(z, self.alpha, self.beta, 50);
        tau.powf(self.beta - S::ONE) * ml
    }
}

/// Approximate Mittag-Leffler function via series.
fn mittag_leffler_approx<S: Scalar>(z: S, alpha: S, beta: S, max_terms: usize) -> S {
    let tol = S::from_f64(1e-12);
    let mut sum = S::ZERO;
    let mut z_pow = S::ONE;

    for k in 0..max_terms {
        let gamma_arg = alpha * S::from_usize(k) + beta;
        let gamma_val = gamma_approx(gamma_arg);
        let term = z_pow / gamma_val;
        sum += term;

        if term.abs() < tol * sum.abs() && k > 0 {
            break;
        }
        z_pow *= z;
    }
    sum
}

/// Lanczos approximation for gamma function.
fn gamma_approx<S: Scalar>(z: S) -> S {
    #[allow(clippy::excessive_precision)]
    let p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let z_f = z.to_f64();
    if z_f < 0.5 {
        let pi = std::f64::consts::PI;
        let sin_pz = (pi * z_f).sin();
        let g1mz = gamma_approx(S::ONE - z);
        return S::from_f64(pi) / (S::from_f64(sin_pz) * g1mz);
    }

    let z_f = z_f - 1.0;
    let mut x = p[0];
    for (i, coeff) in p.iter().enumerate().skip(1) {
        x += coeff / (z_f + i as f64);
    }

    let t = z_f + 7.5;
    let result = (2.0 * std::f64::consts::PI).sqrt() * t.powf(z_f + 0.5) * (-t).exp() * x;
    S::from_f64(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_kernel() {
        let k = ExponentialKernel::new(2.0_f64, 0.5);

        // K(0) = 2 * exp(0) = 2
        assert!((k.evaluate(0.0) - 2.0).abs() < 1e-10);

        // K(2) = 2 * exp(-1) ≈ 0.7358
        let expected = 2.0 * (-1.0_f64).exp();
        assert!((k.evaluate(2.0) - expected).abs() < 1e-10);

        // Should be representable as Prony
        let (a, b) = k.as_prony().unwrap();
        assert_eq!(a.len(), 1);
        assert!((a[0] - 2.0).abs() < 1e-10);
        assert!((b[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_power_law_kernel() {
        let k = PowerLawKernel::new(1.0_f64, 0.5);

        // K(1) = 1 * 1^(-0.5) = 1
        assert!((k.evaluate(1.0) - 1.0).abs() < 1e-10);

        // K(4) = 1 * 4^(-0.5) = 0.5
        assert!((k.evaluate(4.0) - 0.5).abs() < 1e-10);

        // K(0) should be regularized to 0
        assert!((k.evaluate(0.0) - 0.0).abs() < 1e-10);

        // Not a Prony series
        assert!(k.as_prony().is_none());
    }

    #[test]
    fn test_prony_kernel() {
        // Two-term Prony: K(τ) = 1*exp(-0.5τ) + 2*exp(-1.0τ)
        let k = PronyKernel::two_term(1.0_f64, 0.5, 2.0, 1.0);

        // K(0) = 1 + 2 = 3
        assert!((k.evaluate(0.0) - 3.0).abs() < 1e-10);

        // K(2) = exp(-1) + 2*exp(-2) ≈ 0.3679 + 0.2707 = 0.6386
        let expected = (-1.0_f64).exp() + 2.0 * (-2.0_f64).exp();
        assert!((k.evaluate(2.0) - expected).abs() < 1e-10);

        assert_eq!(k.num_terms(), 2);
    }

    #[test]
    fn test_prony_single() {
        // PronyKernel::single(a, b) should match ExponentialKernel::new(a, b)
        let prony = PronyKernel::single(2.0_f64, 0.5);
        let exp_k = ExponentialKernel::new(2.0_f64, 0.5);

        for tau in [0.0, 0.5, 1.0, 2.0, 5.0] {
            let p_val = prony.evaluate(tau);
            let e_val = exp_k.evaluate(tau);
            assert!(
                (p_val - e_val).abs() < 1e-10,
                "Prony single should match ExponentialKernel at tau={}: {} vs {}",
                tau,
                p_val,
                e_val
            );
        }
    }

    #[test]
    fn test_power_law_negative_tau() {
        let k = PowerLawKernel::new(1.0_f64, 0.5);

        // tau < 0 should return 0 (regularization)
        assert!((k.evaluate(-1.0)).abs() < 1e-10);
        assert!((k.evaluate(-0.001)).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_clone() {
        let exp_k = ExponentialKernel::new(2.0_f64, 0.5);
        let exp_clone = exp_k.clone();
        assert!((exp_clone.evaluate(1.0) - exp_k.evaluate(1.0)).abs() < 1e-15);

        let pow_k = PowerLawKernel::new(1.0_f64, 0.5);
        let pow_clone = pow_k.clone();
        assert!((pow_clone.evaluate(4.0) - pow_k.evaluate(4.0)).abs() < 1e-15);

        let prony_k = PronyKernel::two_term(1.0_f64, 0.5, 2.0, 1.0);
        let prony_clone = prony_k.clone();
        assert!((prony_clone.evaluate(1.0) - prony_k.evaluate(1.0)).abs() < 1e-15);

        let ml_k = MittagLefflerKernel::new(1.0_f64, 1.0, 1.0);
        let ml_clone = ml_k.clone();
        assert!((ml_clone.evaluate(1.0) - ml_k.evaluate(1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_mittag_leffler_reduces_to_exponential() {
        // E_{1,1}(-z) = exp(-z)
        let ml_kernel = MittagLefflerKernel::new(1.0_f64, 1.0, 1.0);
        let exp_kernel = ExponentialKernel::new(1.0_f64, 1.0);

        for tau in [0.1, 0.5, 1.0, 2.0] {
            let ml = ml_kernel.evaluate(tau);
            let exp = exp_kernel.evaluate(tau);
            assert!(
                (ml - exp).abs() < 0.01,
                "ML({}) = {}, exp = {}",
                tau,
                ml,
                exp
            );
        }
    }
}
