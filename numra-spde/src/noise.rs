//! Space-time noise generation for SPDEs.
//!
//! This module provides noise generators for the spatial component of SPDE
//! solvers, ranging from simple i.i.d. white noise to spatially correlated
//! colored noise.
//!
//! ## Choosing `correlation_length`
//!
//! For [`ColoredNoiseGenerator`] with spatial correlation:
//! - **`correlation_length` ~ grid spacing:** Noise is nearly white; correlation
//!   has minimal effect. Use white noise instead for efficiency.
//! - **`correlation_length` ~ domain size:** Noise is nearly uniform across
//!   the domain (long-range correlation). May cause Cholesky ill-conditioning
//!   if the grid is fine relative to the correlation length.
//! - **`correlation_length` ~ 5-20 × grid spacing:** Typical choice that
//!   produces smooth spatial noise patterns without numerical issues.
//!
//! If the Cholesky decomposition fails with an ill-conditioning error, try:
//! 1. Increasing `correlation_length` (less extreme correlations)
//! 2. Coarsening the spatial grid (fewer points to correlate)
//! 3. Using trace-class noise instead (eigenvalue decay provides regularization)
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_sde::wiener::{create_wiener, WienerProcess};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Trait for space-time noise processes.
pub trait SpaceTimeNoise<S: Scalar> {
    /// Generate noise increments at all spatial points.
    ///
    /// # Arguments
    /// * `dt` - Time step
    /// * `n_points` - Number of spatial grid points
    /// * `dw` - Output: noise increments at each point
    fn increment(&mut self, dt: S, n_points: usize, dw: &mut [S]);

    /// Reset the noise generator with a new seed.
    fn reseed(&mut self, seed: u64);
}

/// Spatially white (uncorrelated) space-time noise.
///
/// Each grid point receives an independent Wiener increment scaled by dx^(-1/2).
pub struct WhiteNoise<S: Scalar> {
    wiener: WienerProcess<StdRng>,
    dx: S,
    n_wiener: usize,
}

impl<S: Scalar> WhiteNoise<S> {
    /// Create white noise generator.
    ///
    /// # Arguments
    /// * `dx` - Spatial grid spacing
    /// * `n_points` - Number of spatial grid points
    /// * `seed` - Optional random seed
    pub fn new(dx: S, n_points: usize, seed: Option<u64>) -> Self {
        Self {
            wiener: create_wiener(n_points, seed),
            dx,
            n_wiener: n_points,
        }
    }
}

impl<S: Scalar> SpaceTimeNoise<S> for WhiteNoise<S> {
    fn increment(&mut self, dt: S, n_points: usize, dw: &mut [S]) {
        // Resize if needed
        if n_points != self.n_wiener {
            self.wiener = create_wiener(n_points, None);
            self.n_wiener = n_points;
        }

        // For space-time white noise, the covariance is:
        // E[dW(x,t) dW(x',t')] = δ(x-x') δ(t-t') dx dt
        //
        // In discrete form with grid spacing dx:
        // Var[ΔW_i] = dt / dx
        //
        // So we scale by sqrt(dt / dx) to get variance dt/dx
        // But WienerProcess already gives variance dt, so we scale by 1/sqrt(dx)
        let scale = S::ONE / self.dx.sqrt();

        let inc = self.wiener.increment::<S>(dt);
        for i in 0..n_points.min(inc.dw.len()) {
            dw[i] = inc.dw[i] * scale;
        }
    }

    fn reseed(&mut self, seed: u64) {
        self.wiener = create_wiener(self.n_wiener, Some(seed));
    }
}

/// Spatially colored (correlated) noise.
///
/// Noise at nearby points is correlated with correlation function:
/// C(|x - x'|) = exp(-|x - x'| / λ)
///
/// where λ is the correlation length.
pub struct ColoredNoise<S: Scalar> {
    wiener: WienerProcess<StdRng>,
    dx: S,
    correlation_length: S,
    /// Pre-computed correlation matrix (Cholesky factor)
    cholesky: Vec<S>,
    n_cached: usize,
}

impl<S: Scalar> ColoredNoise<S> {
    /// Create colored noise generator.
    ///
    /// # Arguments
    /// * `dx` - Spatial grid spacing
    /// * `correlation_length` - Noise correlation length
    /// * `n_points` - Number of spatial grid points
    /// * `seed` - Optional random seed
    ///
    /// Returns an error if the resulting correlation matrix is not positive
    /// definite or is ill-conditioned for the given parameters.
    pub fn new(
        dx: S,
        correlation_length: S,
        n_points: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        // Build correlation matrix and compute Cholesky factorization
        let cholesky = Self::compute_cholesky(dx, correlation_length, n_points)?;

        Ok(Self {
            wiener: create_wiener(n_points, seed),
            dx,
            correlation_length,
            cholesky,
            n_cached: n_points,
        })
    }

    fn compute_cholesky(dx: S, correlation_length: S, n: usize) -> Result<Vec<S>, String> {
        // Build correlation matrix: C_ij = exp(-|x_i - x_j| / λ)
        let mut c = vec![S::ZERO; n * n];

        for i in 0..n {
            for j in 0..n {
                let dist = dx * S::from_usize(if i > j { i - j } else { j - i });
                c[i * n + j] = (-dist / correlation_length).exp();
            }
        }

        // Cholesky factorization: C = L * L^T
        cholesky_decompose(&c, n)
    }
}

impl<S: Scalar> SpaceTimeNoise<S> for ColoredNoise<S> {
    fn increment(&mut self, dt: S, n_points: usize, dw: &mut [S]) {
        // Recompute Cholesky if grid size changed
        if n_points != self.n_cached {
            self.cholesky = Self::compute_cholesky(self.dx, self.correlation_length, n_points)
                .expect("Cholesky decomposition failed on grid resize; correlation matrix is ill-conditioned");
            self.wiener = create_wiener(n_points, None);
            self.n_cached = n_points;
        }

        // Generate independent increments
        let inc = self.wiener.increment::<S>(dt);

        // Apply Cholesky factor to correlate: dW = L * z
        for i in 0..n_points {
            dw[i] = S::ZERO;
            for j in 0..=i {
                dw[i] = dw[i] + self.cholesky[i * n_points + j] * inc.dw[j];
            }
        }
    }

    fn reseed(&mut self, seed: u64) {
        self.wiener = create_wiener(self.n_cached, Some(seed));
    }
}

/// Trace-class noise using spectral representation.
///
/// Expands the noise in eigenfunctions of the Laplacian:
/// W(x,t) = Σ_k sqrt(λ_k) e_k(x) W_k(t)
///
/// where λ_k are eigenvalues and e_k are eigenfunctions.
#[allow(dead_code)]
pub struct TraceClassNoise<S: Scalar> {
    wiener: WienerProcess<StdRng>,
    n_modes: usize,
    eigenvalues: Vec<S>,
    domain_length: S,
}

#[allow(dead_code)]
impl<S: Scalar> TraceClassNoise<S> {
    /// Create trace-class noise generator.
    ///
    /// # Arguments
    /// * `n_modes` - Number of eigenmodes to include
    /// * `decay_rate` - Eigenvalue decay rate (λ_k ~ k^(-decay_rate))
    /// * `domain_length` - Length of spatial domain
    /// * `seed` - Optional random seed
    pub fn new(n_modes: usize, decay_rate: S, domain_length: S, seed: Option<u64>) -> Self {
        // Eigenvalues decay as k^(-decay_rate)
        let eigenvalues: Vec<S> = (1..=n_modes)
            .map(|k| S::from_usize(k).powf(-decay_rate))
            .collect();

        Self {
            wiener: create_wiener(n_modes, seed),
            n_modes,
            eigenvalues,
            domain_length,
        }
    }
}

impl<S: Scalar> SpaceTimeNoise<S> for TraceClassNoise<S> {
    fn increment(&mut self, dt: S, n_points: usize, dw: &mut [S]) {
        // Initialize output
        for i in 0..n_points {
            dw[i] = S::ZERO;
        }

        let pi = S::PI;

        // Generate mode amplitudes
        let inc = self.wiener.increment::<S>(dt);

        // Sum over modes
        for k in 0..self.n_modes {
            let mode_idx = S::from_usize(k + 1);
            let dw_k = inc.dw[k];
            let sqrt_lambda = self.eigenvalues[k].sqrt();

            // Eigenfunction: e_k(x) = sqrt(2/L) * sin(k*π*x/L)
            let normalization = (S::TWO / self.domain_length).sqrt();

            for i in 0..n_points {
                let x = self.domain_length * S::from_usize(i) / S::from_usize(n_points - 1);
                let basis = normalization * (mode_idx * pi * x / self.domain_length).sin();
                dw[i] = dw[i] + sqrt_lambda * basis * dw_k;
            }
        }
    }

    fn reseed(&mut self, seed: u64) {
        self.wiener = create_wiener(self.n_modes, Some(seed));
    }
}

/// Spatially correlated Gaussian noise generator.
///
/// Generates samples of Gaussian random fields with exponential (squared)
/// correlation function:
///
/// ```text
/// C(x, y) = exp(-|x - y|^2 / (2 * L^2))
/// ```
///
/// where `L` is the correlation length. Uses Cholesky decomposition of the
/// correlation matrix to transform independent standard normal samples into
/// correlated samples.
pub struct ColoredNoiseGenerator<S: Scalar> {
    correlation_length: S,
    rng: StdRng,
}

impl<S: Scalar> ColoredNoiseGenerator<S> {
    /// Create a new colored noise generator.
    ///
    /// # Arguments
    /// * `correlation_length` - The spatial correlation length `L`
    /// * `seed` - Random seed for reproducibility
    pub fn new(correlation_length: S, seed: u64) -> Self {
        Self {
            correlation_length,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate a sample of spatially correlated noise on the given grid.
    ///
    /// Returns a vector of correlated Gaussian values, one per grid point.
    /// Adjacent grid points within the correlation length will have similar values.
    ///
    /// Returns an error if the correlation matrix is not positive definite
    /// or is ill-conditioned for the given grid/correlation_length combination.
    pub fn sample(&mut self, grid: &[S]) -> Result<Vec<S>, String> {
        let n = grid.len();
        let l = self.correlation_length;

        // Build correlation matrix: C_ij = exp(-|x_i - x_j|^2 / (2*L^2))
        let mut c = vec![S::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = grid[i] - grid[j];
                c[i * n + j] = (-dx * dx / (S::from_f64(2.0) * l * l)).exp();
            }
        }

        // Cholesky factorization: C = L_chol * L_chol^T
        let chol = cholesky_decompose(&c, n)?;

        // Generate iid standard normal samples
        let z: Vec<S> = (0..n)
            .map(|_| S::from_f64(self.rng.sample(StandardNormal)))
            .collect();

        // Correlate: y = L_chol * z
        let mut result = vec![S::ZERO; n];
        for i in 0..n {
            for j in 0..=i {
                result[i] = result[i] + chol[i * n + j] * z[j];
            }
        }

        Ok(result)
    }
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Given an n x n SPD matrix `a` stored in row-major order, returns
/// the lower-triangular Cholesky factor `L` such that `A = L * L^T`.
///
/// Returns an error if the matrix is not positive definite or is
/// ill-conditioned (diagonal element below threshold during factorization).
pub(crate) fn cholesky_decompose<S: Scalar>(a: &[S], n: usize) -> Result<Vec<S>, String> {
    let mut l = vec![S::ZERO; n * n];
    let eps = S::from_f64(1e-12);

    for j in 0..n {
        let mut sum = S::ZERO;
        for k in 0..j {
            sum = sum + l[j * n + k] * l[j * n + k];
        }
        let diag = a[j * n + j] - sum;
        if diag < S::ZERO {
            return Err(format!(
                "Correlation matrix not positive definite at index {}: \
                 diagonal element is {}",
                j,
                diag.to_f64()
            ));
        }
        let sqrt_diag = diag.sqrt();
        if sqrt_diag < eps {
            return Err(format!(
                "Correlation matrix ill-conditioned at index {}: \
                 diagonal element {} is below threshold. \
                 Try increasing correlation_length.",
                j,
                diag.to_f64()
            ));
        }
        l[j * n + j] = sqrt_diag;

        for i in (j + 1)..n {
            let mut sum = S::ZERO;
            for k in 0..j {
                sum = sum + l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
        }
    }

    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_noise_variance() {
        let dx = 0.1;
        let dt = 0.01;
        let n = 10;
        let n_samples = 1000;

        let mut noise = WhiteNoise::new(dx, n, Some(42));
        let mut dw = vec![0.0; n];
        let mut sum_sq = vec![0.0; n];

        for _ in 0..n_samples {
            noise.increment(dt, n, &mut dw);
            for i in 0..n {
                sum_sq[i] += dw[i] * dw[i];
            }
        }

        // Expected variance: dt / dx = 0.01 / 0.1 = 0.1
        let expected_var = dt / dx;
        for i in 0..n {
            let var = sum_sq[i] / n_samples as f64;
            assert!(
                (var - expected_var).abs() < 0.05,
                "Variance at point {}: {}, expected: {}",
                i,
                var,
                expected_var
            );
        }
    }

    #[test]
    fn test_colored_noise_correlation() {
        let dx = 0.1;
        let correlation_length = 0.3;
        let n = 20;
        let dt = 0.01;
        let n_samples = 5000;

        let mut noise = ColoredNoise::new(dx, correlation_length, n, Some(42)).unwrap();
        let mut dw = vec![0.0; n];
        let mut cross_sum = 0.0;
        let mut sum_sq = 0.0;

        // Check correlation between adjacent points
        for _ in 0..n_samples {
            noise.increment(dt, n, &mut dw);
            cross_sum += dw[0] * dw[1];
            sum_sq += dw[0] * dw[0];
        }

        let correlation = cross_sum / sum_sq;
        let expected = (-dx / correlation_length).exp();

        // Colored noise should have positive correlation between nearby points
        assert!(
            correlation > 0.0,
            "Adjacent points should be positively correlated"
        );
        assert!(
            (correlation - expected).abs() < 0.2,
            "Correlation: {}, expected: {}",
            correlation,
            expected
        );
    }

    #[test]
    fn test_trace_class_noise() {
        let n_modes = 10;
        let decay_rate = 2.0;
        let domain_length = 1.0;
        let n = 50;
        let dt = 0.01;

        let mut noise = TraceClassNoise::new(n_modes, decay_rate, domain_length, Some(42));
        let mut dw = vec![0.0; n];

        noise.increment(dt, n, &mut dw);

        // Trace-class noise should be smoother than white noise
        // Check that the noise has spatial structure
        let mut total_variation = 0.0;
        for i in 1..n {
            total_variation += (dw[i] - dw[i - 1]).abs();
        }

        // Should be finite and not too wild
        assert!(total_variation.is_finite());
    }

    #[test]
    fn test_colored_noise_generation() {
        use crate::noise::ColoredNoiseGenerator;

        let correlation_length = 0.1_f64;
        let grid: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        let n = grid.len();

        let mut generator = ColoredNoiseGenerator::new(correlation_length, 42);

        // Generate many samples and compute sample correlations
        let n_samples = 2000;
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            samples.push(generator.sample(&grid).unwrap());
        }

        // Compute sample correlation between adjacent points
        // For exponential correlation C(d) = exp(-d^2/(2*L^2)),
        // adjacent points (d = 1/19 ~ 0.053) with L=0.1 should have
        // correlation ~ exp(-0.053^2/0.02) ~ 0.87
        let mean_0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / n_samples as f64;
        let mean_1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / n_samples as f64;
        let var_0: f64 =
            samples.iter().map(|s| (s[0] - mean_0).powi(2)).sum::<f64>() / n_samples as f64;
        let var_1: f64 =
            samples.iter().map(|s| (s[1] - mean_1).powi(2)).sum::<f64>() / n_samples as f64;
        let cov_01: f64 = samples
            .iter()
            .map(|s| (s[0] - mean_0) * (s[1] - mean_1))
            .sum::<f64>()
            / n_samples as f64;
        let corr_01 = cov_01 / (var_0.sqrt() * var_1.sqrt());

        // Correlation should be high for adjacent points with this L
        assert!(corr_01 > 0.5, "Adjacent correlation too low: {}", corr_01);

        // Check correlation decays with distance
        let mean_far: f64 = samples.iter().map(|s| s[n - 1]).sum::<f64>() / n_samples as f64;
        let var_far: f64 = samples
            .iter()
            .map(|s| (s[n - 1] - mean_far).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        let cov_far: f64 = samples
            .iter()
            .map(|s| (s[0] - mean_0) * (s[n - 1] - mean_far))
            .sum::<f64>()
            / n_samples as f64;
        let corr_far = cov_far / (var_0.sqrt() * var_far.sqrt());

        assert!(
            corr_01 > corr_far.abs(),
            "Nearby correlation should be higher than far correlation"
        );
    }

    #[test]
    fn test_cholesky_rejects_non_positive_definite() {
        // Matrix [[1, 2], [2, 1]] has eigenvalues {3, -1}, not positive definite
        let n = 2;
        let matrix = vec![1.0_f64, 2.0, 2.0, 1.0];
        let result = cholesky_decompose::<f64>(&matrix, n);
        assert!(
            result.is_err(),
            "Should reject non-positive-definite matrix"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("not positive definite"),
            "Error message: {}",
            msg
        );
    }

    #[test]
    fn test_cholesky_rejects_ill_conditioned() {
        // Identity with one near-zero diagonal => ill-conditioned
        let n = 3;
        let mut matrix = vec![0.0_f64; n * n];
        matrix[0] = 1.0;
        matrix[4] = 1e-30; // nearly singular
        matrix[8] = 1.0;
        let result = cholesky_decompose::<f64>(&matrix, n);
        assert!(result.is_err(), "Should reject ill-conditioned matrix");
        let msg = result.unwrap_err();
        assert!(msg.contains("ill-conditioned"), "Error message: {}", msg);
    }

    #[test]
    fn test_cholesky_accepts_valid_matrix() {
        // Valid positive-definite matrix: identity
        let n = 3;
        let mut matrix = vec![0.0_f64; n * n];
        matrix[0] = 1.0;
        matrix[4] = 1.0;
        matrix[8] = 1.0;
        let result = cholesky_decompose::<f64>(&matrix, n);
        assert!(result.is_ok(), "Should accept identity matrix");
        let l = result.unwrap();
        // L should be identity for identity input
        assert!((l[0] - 1.0).abs() < 1e-10);
        assert!((l[4] - 1.0).abs() < 1e-10);
        assert!((l[8] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_colored_noise_new_returns_ok_for_valid_params() {
        let result = ColoredNoise::<f64>::new(0.1, 0.3, 10, Some(42));
        assert!(
            result.is_ok(),
            "Should accept reasonable correlation parameters"
        );
    }

    #[test]
    fn test_cholesky_decompose_recovers_factor() {
        // Verify L * L^T = A for a known SPD matrix
        let n = 3;
        let a = vec![4.0_f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let l = cholesky_decompose::<f64>(&a, n).unwrap();

        // Reconstruct A = L * L^T
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[i * n + k] * l[j * n + k];
                }
                assert!(
                    (sum - a[i * n + j]).abs() < 1e-10,
                    "Reconstruction error at ({},{}): {} vs {}",
                    i,
                    j,
                    sum,
                    a[i * n + j]
                );
            }
        }
    }
}
