//! SPDE system trait and noise correlation types.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_pde::Grid1D;

/// Noise correlation structure.
#[derive(Clone, Debug, Default)]
pub enum NoiseCorrelation<S: Scalar> {
    /// Spatially white noise (independent at each grid point)
    #[default]
    White,
    /// Spatially colored noise with correlation length
    Colored {
        /// Correlation length (distance over which noise is correlated)
        correlation_length: S,
    },
    /// Trace-class noise (finite number of modes)
    TraceClass {
        /// Number of eigenmodes to include
        n_modes: usize,
        /// Eigenvalue decay rate
        decay_rate: S,
    },
}

/// Trait for Stochastic Partial Differential Equation systems.
///
/// An SPDE has the form:
/// ```text
/// ∂u/∂t = L[u] + σ(u) ξ(x,t)
/// ```
/// where L is a spatial differential operator and ξ is space-time noise.
pub trait SpdeSystem<S: Scalar> {
    /// Spatial dimension (1D, 2D, etc.)
    fn dim(&self) -> usize {
        1
    }

    /// Evaluate the deterministic spatial operator `L[u]`.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `u` - Field values at grid points (interior points only)
    /// * `du` - Output: `L[u]` at each grid point
    /// * `grid` - Spatial grid
    fn drift(&self, t: S, u: &[S], du: &mut [S], grid: &Grid1D<S>);

    /// Evaluate the noise coefficient σ(u).
    ///
    /// For additive noise, this is independent of u.
    /// For multiplicative noise, this depends on u.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `u` - Field values at grid points
    /// * `sigma` - Output: σ(u) at each grid point
    /// * `grid` - Spatial grid
    fn diffusion(&self, t: S, u: &[S], sigma: &mut [S], grid: &Grid1D<S>);

    /// Noise correlation structure.
    fn noise_correlation(&self) -> NoiseCorrelation<S> {
        NoiseCorrelation::White
    }

    /// Whether the noise is additive (σ independent of u).
    fn is_additive(&self) -> bool {
        true
    }
}

/// 1D SPDE with specific boundary conditions.
#[allow(dead_code)]
pub struct Spde1D<S: Scalar, F, G>
where
    F: Fn(S, &[S], &mut [S], &Grid1D<S>),
    G: Fn(S, &[S], &mut [S], &Grid1D<S>),
{
    /// Drift function: L[u]
    drift_fn: F,
    /// Diffusion function: σ(u)
    diffusion_fn: G,
    /// Noise correlation
    correlation: NoiseCorrelation<S>,
    /// Is noise additive?
    additive: bool,
}

#[allow(dead_code)]
impl<S: Scalar, F, G> Spde1D<S, F, G>
where
    F: Fn(S, &[S], &mut [S], &Grid1D<S>),
    G: Fn(S, &[S], &mut [S], &Grid1D<S>),
{
    /// Create a new 1D SPDE.
    pub fn new(drift: F, diffusion: G) -> Self {
        Self {
            drift_fn: drift,
            diffusion_fn: diffusion,
            correlation: NoiseCorrelation::White,
            additive: true,
        }
    }

    /// Set noise correlation.
    pub fn with_correlation(mut self, correlation: NoiseCorrelation<S>) -> Self {
        self.correlation = correlation;
        self
    }

    /// Set whether noise is additive.
    pub fn with_additive(mut self, additive: bool) -> Self {
        self.additive = additive;
        self
    }
}

impl<S: Scalar, F, G> SpdeSystem<S> for Spde1D<S, F, G>
where
    F: Fn(S, &[S], &mut [S], &Grid1D<S>),
    G: Fn(S, &[S], &mut [S], &Grid1D<S>),
{
    fn drift(&self, t: S, u: &[S], du: &mut [S], grid: &Grid1D<S>) {
        (self.drift_fn)(t, u, du, grid)
    }

    fn diffusion(&self, t: S, u: &[S], sigma: &mut [S], grid: &Grid1D<S>) {
        (self.diffusion_fn)(t, u, sigma, grid)
    }

    fn noise_correlation(&self) -> NoiseCorrelation<S> {
        self.correlation.clone()
    }

    fn is_additive(&self) -> bool {
        self.additive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spde1d_construction() {
        let _spde: Spde1D<f64, _, _> = Spde1D::new(
            |_t, u, du, grid| {
                let dx = grid.dx_uniform();
                let n = u.len();
                for i in 0..n {
                    let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                    let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
                    du[i] = (u_left - 2.0 * u[i] + u_right) / (dx * dx);
                }
            },
            |_t, _u, sigma, _grid| {
                for s in sigma.iter_mut() {
                    *s = 0.1;
                }
            },
        );
    }

    #[test]
    fn test_noise_correlation_default() {
        let correlation: NoiseCorrelation<f64> = NoiseCorrelation::default();
        assert!(matches!(correlation, NoiseCorrelation::White));
    }

    #[test]
    fn test_colored_noise() {
        let correlation: NoiseCorrelation<f64> = NoiseCorrelation::Colored {
            correlation_length: 0.1,
        };
        if let NoiseCorrelation::Colored { correlation_length } = correlation {
            assert!((correlation_length - 0.1).abs() < 1e-10);
        }
    }
}
