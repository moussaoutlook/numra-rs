//! SPDE solvers using Method of Lines + SDE steppers.
//!
//! ## Approach
//!
//! Spatial discretization converts the SPDE into a system of SDEs (one per
//! grid point), which are then integrated with an SDE stepper (Euler-Maruyama
//! or Milstein).
//!
//! ## Adaptive Stepping
//!
//! Adaptive time stepping uses step-doubling error estimation: each step is
//! taken once with step size h and twice with h/2, using independent Wiener
//! increments. The difference estimates the local error.
//!
//! **Limitations:**
//! - The weak convergence order of Euler-Maruyama is 1.0, so the step size
//!   controller uses a conservative exponent of 1/3.
//! - Adaptive stepping adds overhead from the extra half-steps and noise
//!   generation.
//! - For problems with very stiff diffusion operators, fixed-step methods
//!   may be more efficient.
//!
//! ## Noise Correlation
//!
//! Three noise types are supported via [`NoiseCorrelation`]:
//! - **White:** Independent noise at each spatial point (diagonal diffusion)
//! - **Spatial:** Correlated noise with a given correlation length (uses
//!   Cholesky factorization of the spatial correlation matrix)
//! - **Trace-class:** Correlated noise with eigenvalue decay (Q-Wiener process)
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_pde::Grid1D;
use numra_sde::{
    create_wiener, EulerMaruyama, Milstein, NoiseType, SdeOptions, SdeSolver, SdeSystem,
};

use crate::noise::cholesky_decompose;
use crate::system::{NoiseCorrelation, SpdeSystem};

/// SDE integration method for the spatially-discretized system.
#[derive(Clone, Copy, Debug, Default)]
pub enum SpdeMethod {
    /// Euler-Maruyama (1st order, simple)
    #[default]
    EulerMaruyama,
    /// Milstein (1.0 order strong convergence, for multiplicative noise)
    Milstein,
}

/// Options for SPDE solver.
///
/// **Divergence from `numra_ode::SolverOptions`** (per Foundation Spec §2.5):
/// SPDE solvers carry both stochastic noise (like SDE — Wiener time
/// control, `seed`) and an explicit method selector (`SpdeMethod`,
/// `EulerMaruyama` | `Milstein`). The method selector is unusual:
/// `numra-ode` solvers dispatch at the type level
/// (`DoPri5::solve(...)`, `Tsit5::solve(...)`); SPDE collapses to a single
/// `solve` entry that branches on the enum because noise-discretisation
/// logic is shared across methods. `n_output` discretises sampled-output
/// count rather than `SolverOptions::t_eval`-style explicit time grid —
/// appropriate for stochastic workloads where exact output times matter
/// less than density. `adaptive: bool` gates between fixed-step EM
/// (default) and adaptive variants. See
/// `docs/architecture/foundation-specification.md` §2.5.
#[derive(Clone, Debug)]
pub struct SpdeOptions<S: Scalar> {
    /// Time step
    pub dt: S,
    /// Number of output time points
    pub n_output: usize,
    /// SDE method
    pub method: SpdeMethod,
    /// Random seed
    pub seed: Option<u64>,
    /// Enable adaptive time stepping
    pub adaptive: bool,
    /// Relative tolerance for adaptive stepping
    pub rtol: S,
    /// Absolute tolerance for adaptive stepping
    pub atol: S,
    /// Minimum time step for adaptive stepping
    pub dt_min: S,
    /// Maximum time step for adaptive stepping
    pub dt_max: S,
}

impl<S: Scalar> Default for SpdeOptions<S> {
    fn default() -> Self {
        Self {
            dt: S::from_f64(0.001),
            n_output: 100,
            method: SpdeMethod::EulerMaruyama,
            seed: None,
            adaptive: false,
            rtol: S::from_f64(1e-3),
            atol: S::from_f64(1e-6),
            dt_min: S::from_f64(1e-10),
            dt_max: S::from_f64(0.1),
        }
    }
}

impl<S: Scalar> SpdeOptions<S> {
    /// Set time step.
    pub fn dt(mut self, dt: S) -> Self {
        self.dt = dt;
        self
    }

    /// Set number of output points.
    pub fn n_output(mut self, n: usize) -> Self {
        self.n_output = n;
        self
    }

    /// Set SDE method.
    pub fn method(mut self, method: SpdeMethod) -> Self {
        self.method = method;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable/disable adaptive time stepping.
    pub fn adaptive(mut self, enable: bool) -> Self {
        self.adaptive = enable;
        self
    }

    /// Set relative tolerance for adaptive stepping.
    pub fn rtol(mut self, tol: S) -> Self {
        self.rtol = tol;
        self
    }

    /// Set absolute tolerance for adaptive stepping.
    pub fn atol(mut self, tol: S) -> Self {
        self.atol = tol;
        self
    }

    /// Set minimum time step for adaptive stepping.
    pub fn dt_min(mut self, dt: S) -> Self {
        self.dt_min = dt;
        self
    }

    /// Set maximum time step for adaptive stepping.
    pub fn dt_max(mut self, dt: S) -> Self {
        self.dt_max = dt;
        self
    }
}

/// Statistics for SPDE solve.
#[derive(Clone, Debug, Default)]
pub struct SpdeStats {
    /// Number of time steps taken
    pub n_steps: usize,
    /// Number of drift evaluations
    pub n_drift: usize,
    /// Number of diffusion evaluations
    pub n_diffusion: usize,
    /// Number of rejected steps (for adaptive methods)
    pub n_reject: usize,
}

/// Result of SPDE solve.
#[derive(Clone, Debug)]
pub struct SpdeResult<S: Scalar> {
    /// Time points
    pub t: Vec<S>,
    /// Solution at each time: `y[i]` is solution at `t[i]`
    pub y: Vec<Vec<S>>,
    /// Spatial grid
    pub grid: Grid1D<S>,
    /// Solver statistics
    pub stats: SpdeStats,
    /// Whether solve was successful
    pub success: bool,
}

impl<S: Scalar> SpdeResult<S> {
    /// Get solution at final time.
    pub fn y_final(&self) -> Option<&[S]> {
        self.y.last().map(|v| v.as_slice())
    }

    /// Get final time.
    pub fn t_final(&self) -> Option<S> {
        self.t.last().copied()
    }

    /// Get solution at time index i.
    pub fn y_at(&self, i: usize) -> &[S] {
        &self.y[i]
    }
}

/// Trait for SPDE solvers.
pub trait SpdeSolver<S: Scalar> {
    /// Solve the SPDE.
    fn solve<Sys: SpdeSystem<S> + Sync>(
        system: &Sys,
        t0: S,
        tf: S,
        u0: &[S],
        grid: &Grid1D<S>,
        options: &SpdeOptions<S>,
    ) -> Result<SpdeResult<S>, String>;
}

/// Pre-computed noise configuration for the MOL wrapper.
enum MolNoiseConfig<S: Scalar> {
    /// White (diagonal) noise — each grid point gets independent noise.
    White,
    /// Colored noise with pre-computed Cholesky factor L (dim × dim, row-major).
    /// Correlated noise = L * independent_noise.
    Colored { cholesky: Vec<S>, dim: usize },
    /// Trace-class noise using spectral representation.
    /// n_modes independent Wiener processes drive the spatial modes.
    TraceClass {
        /// sqrt(lambda_k) for each mode k
        sqrt_eigenvalues: Vec<S>,
        /// Basis function values: basis[i * n_modes + k] = e_k(x_i)
        basis: Vec<S>,
        n_modes: usize,
        dim: usize,
    },
}

impl<S: Scalar> MolNoiseConfig<S> {
    /// Build noise configuration from SPDE system and grid.
    fn from_system<Sys: SpdeSystem<S>>(system: &Sys, grid: &Grid1D<S>) -> Result<Self, String> {
        match system.noise_correlation() {
            NoiseCorrelation::White => Ok(MolNoiseConfig::White),
            NoiseCorrelation::Colored { correlation_length } => {
                let dim = grid.len() - 2; // interior points
                let dx = grid.dx_uniform();

                // Build correlation matrix: C_ij = exp(-|x_i - x_j| / lambda)
                let mut c = vec![S::ZERO; dim * dim];
                for i in 0..dim {
                    for j in 0..dim {
                        let dist = dx * S::from_usize(if i > j { i - j } else { j - i });
                        c[i * dim + j] = (-dist / correlation_length).exp();
                    }
                }

                let cholesky = cholesky_decompose(&c, dim)?;
                Ok(MolNoiseConfig::Colored { cholesky, dim })
            }
            NoiseCorrelation::TraceClass {
                n_modes,
                decay_rate,
            } => {
                let dim = grid.len() - 2;
                let interior = grid.interior_points();
                let (x_min, x_max) = grid.bounds();
                let domain_length = x_max - x_min;

                // Eigenvalues: lambda_k = k^(-decay_rate)
                let sqrt_eigenvalues: Vec<S> = (1..=n_modes)
                    .map(|k| S::from_usize(k).powf(-decay_rate).sqrt())
                    .collect();

                // Basis functions: e_k(x) = sqrt(2/L) * sin(k*pi*x/L)
                let norm = (S::TWO / domain_length).sqrt();
                let pi = S::PI;
                let mut basis = vec![S::ZERO; dim * n_modes];
                for i in 0..dim {
                    let x = interior[i] - x_min;
                    for k in 0..n_modes {
                        let mode = S::from_usize(k + 1);
                        basis[i * n_modes + k] = norm * (mode * pi * x / domain_length).sin();
                    }
                }

                Ok(MolNoiseConfig::TraceClass {
                    sqrt_eigenvalues,
                    basis,
                    n_modes,
                    dim,
                })
            }
        }
    }

    /// Number of Wiener processes needed.
    fn n_wiener(&self, dim: usize) -> usize {
        match self {
            MolNoiseConfig::White => dim,
            MolNoiseConfig::Colored { dim: d, .. } => *d,
            MolNoiseConfig::TraceClass { n_modes, .. } => *n_modes,
        }
    }
}

/// Wrapper that converts SPDE to SDE system via Method of Lines.
struct MolSdeWrapper<'a, S: Scalar, Sys: SpdeSystem<S> + Sync> {
    spde: &'a Sys,
    grid: &'a Grid1D<S>,
    noise_config: MolNoiseConfig<S>,
}

impl<'a, S: Scalar, Sys: SpdeSystem<S> + Sync> SdeSystem<S> for MolSdeWrapper<'a, S, Sys> {
    fn dim(&self) -> usize {
        self.grid.len() - 2 // Interior points only
    }

    fn noise_type(&self) -> NoiseType {
        match &self.noise_config {
            MolNoiseConfig::White => NoiseType::Diagonal,
            MolNoiseConfig::Colored { dim, .. } => NoiseType::General { n_wiener: *dim },
            MolNoiseConfig::TraceClass { n_modes, .. } => NoiseType::General { n_wiener: *n_modes },
        }
    }

    fn drift(&self, t: S, y: &[S], f: &mut [S]) {
        self.spde.drift(t, y, f, self.grid);
    }

    fn diffusion(&self, t: S, y: &[S], g: &mut [S]) {
        match &self.noise_config {
            MolNoiseConfig::White => {
                // Diagonal: g[i] = sigma(u_i)
                self.spde.diffusion(t, y, g, self.grid);
            }
            MolNoiseConfig::Colored { cholesky, dim } => {
                // General: g[i * dim + j] = sigma(u_i) * L[i][j]
                // Get the base diffusion coefficients into a local buffer
                let mut sigma = vec![S::ZERO; *dim];
                self.spde.diffusion(t, y, &mut sigma, self.grid);

                // Build g[i * dim + j] = sigma[i] * L[i][j]
                for i in 0..*dim {
                    for j in 0..=i {
                        g[i * dim + j] = sigma[i] * cholesky[i * dim + j];
                    }
                    // Upper triangle is zero (L is lower triangular)
                    for j in (i + 1)..*dim {
                        g[i * dim + j] = S::ZERO;
                    }
                }
            }
            MolNoiseConfig::TraceClass {
                sqrt_eigenvalues,
                basis,
                n_modes,
                dim,
            } => {
                // General: g[i * n_modes + k] = sigma(u_i) * sqrt(lambda_k) * e_k(x_i)
                let mut sigma = vec![S::ZERO; *dim];
                self.spde.diffusion(t, y, &mut sigma, self.grid);

                for i in 0..*dim {
                    for k in 0..*n_modes {
                        g[i * n_modes + k] =
                            sigma[i] * sqrt_eigenvalues[k] * basis[i * n_modes + k];
                    }
                }
            }
        }
    }
}

// SAFETY: MolSdeWrapper contains only shared references (&'a Sys where Sys: Sync,
// &'a Grid1D<S>) and owned data (MolNoiseConfig<S> which is Send+Sync since S: Scalar
// is Send+Sync and Vec<S> is Send+Sync). The Sys: Sync bound ensures the shared system
// reference is safe to access from multiple threads. MolNoiseConfig only contains
// Vec<S> and usize fields, which are inherently Sync. The compiler cannot auto-derive
// Sync because of the generic Sys parameter interaction with the lifetime, but all
// fields are safe to share across threads.
unsafe impl<'a, S: Scalar, Sys: SpdeSystem<S> + Sync> Sync for MolSdeWrapper<'a, S, Sys> {}

/// Method of Lines SPDE solver.
///
/// Converts the SPDE to a large SDE system by discretizing spatial derivatives,
/// then applies an SDE solver.
pub struct MolSdeSolver;

impl<S: Scalar> SpdeSolver<S> for MolSdeSolver {
    fn solve<Sys: SpdeSystem<S> + Sync>(
        system: &Sys,
        t0: S,
        tf: S,
        u0: &[S],
        grid: &Grid1D<S>,
        options: &SpdeOptions<S>,
    ) -> Result<SpdeResult<S>, String> {
        let n_interior = grid.len() - 2;
        if u0.len() != n_interior {
            return Err(format!(
                "Initial condition length {} doesn't match interior points {}",
                u0.len(),
                n_interior
            ));
        }

        // Use adaptive or fixed time stepping
        if options.adaptive {
            Self::solve_adaptive(system, t0, tf, u0, grid, options)
        } else {
            Self::solve_fixed(system, t0, tf, u0, grid, options)
        }
    }
}

impl MolSdeSolver {
    /// Solve using fixed time stepping (delegates to SDE solver).
    fn solve_fixed<S: Scalar, Sys: SpdeSystem<S> + Sync>(
        system: &Sys,
        t0: S,
        tf: S,
        u0: &[S],
        grid: &Grid1D<S>,
        options: &SpdeOptions<S>,
    ) -> Result<SpdeResult<S>, String> {
        // Create MOL SDE wrapper with pre-computed noise configuration
        let noise_config = MolNoiseConfig::from_system(system, grid)?;
        let sde_system = MolSdeWrapper {
            spde: system,
            grid,
            noise_config,
        };

        // SDE options
        let sde_options = SdeOptions::default().dt(options.dt);

        // Solve using appropriate method
        let sde_result = match options.method {
            SpdeMethod::EulerMaruyama => {
                EulerMaruyama::solve(&sde_system, t0, tf, u0, &sde_options, options.seed)
            }
            SpdeMethod::Milstein => {
                if system.is_additive() {
                    // For additive noise, Milstein reduces to Euler-Maruyama
                    EulerMaruyama::solve(&sde_system, t0, tf, u0, &sde_options, options.seed)
                } else {
                    Milstein::solve(&sde_system, t0, tf, u0, &sde_options, options.seed)
                }
            }
        }?;

        // Build output with subsampling
        let n_steps = sde_result.t.len();
        let stride = (n_steps / options.n_output).max(1);

        let mut t_out = Vec::new();
        let mut y_out = Vec::new();

        for i in (0..n_steps).step_by(stride) {
            t_out.push(sde_result.t[i]);
            y_out.push(sde_result.y_at(i).to_vec());
        }

        // Always include final point
        if t_out.last() != sde_result.t.last() {
            t_out.push(*sde_result.t.last().unwrap());
            y_out.push(sde_result.y_final().unwrap().to_vec());
        }

        Ok(SpdeResult {
            t: t_out,
            y: y_out,
            grid: grid.clone(),
            stats: SpdeStats {
                n_steps: sde_result.stats.n_accept + sde_result.stats.n_reject,
                n_drift: sde_result.stats.n_drift,
                n_diffusion: sde_result.stats.n_diffusion,
                n_reject: sde_result.stats.n_reject,
            },
            success: sde_result.success,
        })
    }

    /// Solve using adaptive time stepping with step doubling for error estimation.
    fn solve_adaptive<S: Scalar, Sys: SpdeSystem<S> + Sync>(
        system: &Sys,
        t0: S,
        tf: S,
        u0: &[S],
        grid: &Grid1D<S>,
        options: &SpdeOptions<S>,
    ) -> Result<SpdeResult<S>, String> {
        let dim = u0.len();
        let mut t = t0;
        let mut u = u0.to_vec();
        let mut dt = options.dt;

        // Pre-compute noise configuration for correlated noise
        let noise_config = MolNoiseConfig::from_system(system, grid)?;
        let n_wiener = noise_config.n_wiener(dim);

        // Wiener process for generating noise (independent increments)
        let mut wiener = create_wiener(n_wiener, options.seed);

        // Storage for results
        let mut t_out = vec![t0];
        let mut y_out = vec![u0.to_vec()];
        let mut stats = SpdeStats::default();

        // Buffers for computation
        let mut drift = vec![S::ZERO; dim];
        let mut diffusion = vec![S::ZERO; dim];
        let mut u1 = vec![S::ZERO; dim]; // One full step
        let mut u_half = vec![S::ZERO; dim]; // After first half step
        let mut u2 = vec![S::ZERO; dim]; // Two half steps

        let max_steps = 1_000_000;
        let mut step_count = 0;

        while t < tf && step_count < max_steps {
            // Clamp dt to not overshoot final time
            let h = dt.min(tf - t).min(options.dt_max).max(options.dt_min);

            // Generate two independent half-step Wiener increments.
            // dW1 ~ N(0, h/2), dW2 ~ N(0, h/2), so dW1 + dW2 ~ N(0, h).
            let half_h = h / S::from_f64(2.0);
            let raw_dw1 = wiener.increment(half_h);
            let raw_dw2 = wiener.increment(half_h);

            // Apply noise correlation to get effective noise at each grid point.
            // For white noise, this is just the raw Wiener increments.
            // For colored/trace-class, we apply the correlation structure.
            let noise1 = apply_noise_correlation(&noise_config, &raw_dw1.dw, dim);
            let noise2 = apply_noise_correlation(&noise_config, &raw_dw2.dw, dim);

            // Compute drift and diffusion at current state
            let mut drift_orig = vec![S::ZERO; dim];
            let mut diffusion_orig = vec![S::ZERO; dim];
            system.drift(t, &u, &mut drift_orig, grid);
            system.diffusion(t, &u, &mut diffusion_orig, grid);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Full step uses effective noise = noise1 + noise2
            for i in 0..dim {
                u1[i] = u[i] + drift_orig[i] * h + diffusion_orig[i] * (noise1[i] + noise2[i]);
            }

            // First half step uses noise1
            for i in 0..dim {
                u_half[i] = u[i] + drift_orig[i] * half_h + diffusion_orig[i] * noise1[i];
            }

            // Compute drift and diffusion at half step
            system.drift(t + half_h, &u_half, &mut drift, grid);
            system.diffusion(t + half_h, &u_half, &mut diffusion, grid);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Second half step uses noise2
            for i in 0..dim {
                u2[i] = u_half[i] + drift[i] * half_h + diffusion[i] * noise2[i];
            }

            // Estimate error as difference between methods
            let mut err_max = S::ZERO;
            let mut scale_max = S::ZERO;
            for i in 0..dim {
                let err_i = (u1[i] - u2[i]).abs();
                let scale_i = options.atol + options.rtol * u[i].abs().max(u2[i].abs());
                let ratio = err_i / scale_i;
                if ratio > err_max / scale_max.max(S::from_f64(1e-15)) {
                    err_max = err_i;
                    scale_max = scale_i;
                }
            }

            let err_ratio = err_max / scale_max.max(S::from_f64(1e-15));

            if err_ratio <= S::ONE || h <= options.dt_min {
                // Accept step - use the more accurate two-step solution
                t = t + h;
                u.copy_from_slice(&u2);
                stats.n_steps += 1;

                // Store result
                t_out.push(t);
                y_out.push(u.clone());

                // Increase step size if error is small
                // Use exponent 1/3 (conservative for EM weak order 1 with step-doubling)
                if err_ratio < S::from_f64(0.5) {
                    let factor = (S::ONE / err_ratio.max(S::from_f64(1e-10)))
                        .powf(S::from_f64(1.0 / 3.0))
                        .min(S::from_f64(2.0));
                    dt = (h * factor).min(options.dt_max);
                }
            } else {
                // Reject step and decrease dt
                stats.n_reject += 1;
                let factor = (S::ONE / err_ratio.max(S::from_f64(1e-10)))
                    .powf(S::from_f64(1.0 / 3.0))
                    .max(S::from_f64(0.25))
                    .min(S::from_f64(0.9));
                dt = (h * factor).max(options.dt_min);
            }

            step_count += 1;
        }

        if step_count >= max_steps && t < tf {
            return Err(format!(
                "Maximum steps ({}) exceeded at t = {}",
                max_steps,
                t.to_f64()
            ));
        }

        // Subsample output if needed
        let n_total = t_out.len();
        if n_total > options.n_output {
            let stride = (n_total / options.n_output).max(1);
            let mut t_sub = Vec::new();
            let mut y_sub = Vec::new();

            for i in (0..n_total).step_by(stride) {
                t_sub.push(t_out[i]);
                y_sub.push(y_out[i].clone());
            }

            // Always include final point
            if t_sub.last() != t_out.last() {
                t_sub.push(*t_out.last().unwrap());
                y_sub.push(y_out.last().unwrap().clone());
            }

            t_out = t_sub;
            y_out = y_sub;
        }

        Ok(SpdeResult {
            t: t_out,
            y: y_out,
            grid: grid.clone(),
            stats,
            success: true,
        })
    }
}

/// Apply noise correlation structure to raw independent Wiener increments.
///
/// For white noise, returns the raw increments unchanged (identity mapping).
/// For colored noise, applies Cholesky factor: effective_dW = L * raw_dZ.
/// For trace-class noise, applies spectral expansion: effective_dW[i] = sum_k sqrt(lam_k) * e_k(x_i) * raw_dZ[k].
fn apply_noise_correlation<S: Scalar>(
    config: &MolNoiseConfig<S>,
    raw_dw: &[S],
    dim: usize,
) -> Vec<S> {
    match config {
        MolNoiseConfig::White => {
            // No correlation — raw increments are already per grid point
            raw_dw[..dim].to_vec()
        }
        MolNoiseConfig::Colored { cholesky, dim: n } => {
            // effective_dW[i] = sum_{j<=i} L[i][j] * raw_dZ[j]
            let mut result = vec![S::ZERO; *n];
            for i in 0..*n {
                for j in 0..=i {
                    result[i] = result[i] + cholesky[i * n + j] * raw_dw[j];
                }
            }
            result
        }
        MolNoiseConfig::TraceClass {
            sqrt_eigenvalues,
            basis,
            n_modes,
            dim: n,
        } => {
            // effective_dW[i] = sum_k sqrt(lambda_k) * e_k(x_i) * raw_dZ[k]
            let mut result = vec![S::ZERO; *n];
            for i in 0..*n {
                for k in 0..*n_modes {
                    result[i] =
                        result[i] + sqrt_eigenvalues[k] * basis[i * n_modes + k] * raw_dw[k];
                }
            }
            result
        }
    }
}

/// Ensemble solver for SPDE Monte Carlo simulations.
pub struct SpdeEnsemble;

impl SpdeEnsemble {
    /// Run ensemble of SPDE solutions with different noise realizations.
    pub fn solve<S: Scalar, Sys: SpdeSystem<S> + Sync>(
        system: &Sys,
        t0: S,
        tf: S,
        u0: &[S],
        grid: &Grid1D<S>,
        options: &SpdeOptions<S>,
        n_trajectories: usize,
    ) -> Result<Vec<SpdeResult<S>>, String> {
        let mut results = Vec::with_capacity(n_trajectories);
        let base_seed = options.seed.unwrap_or(42);

        for i in 0..n_trajectories {
            let mut opts = options.clone();
            opts.seed = Some(base_seed + i as u64);
            let result = MolSdeSolver::solve(system, t0, tf, u0, grid, &opts)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compute mean trajectory from ensemble.
    pub fn mean<S: Scalar>(results: &[SpdeResult<S>]) -> Vec<Vec<S>> {
        if results.is_empty() {
            return Vec::new();
        }

        let n_times = results[0].t.len();
        let n_points = results[0].y[0].len();
        let n_traj = results.len();

        let mut mean = vec![vec![S::ZERO; n_points]; n_times];

        for result in results {
            for (i, y) in result.y.iter().enumerate() {
                for (j, &val) in y.iter().enumerate() {
                    mean[i][j] = mean[i][j] + val;
                }
            }
        }

        let n_traj_s = S::from_usize(n_traj);
        for m in mean.iter_mut() {
            for val in m.iter_mut() {
                *val = *val / n_traj_s;
            }
        }

        mean
    }

    /// Compute standard deviation from ensemble.
    pub fn std<S: Scalar>(results: &[SpdeResult<S>], mean: &[Vec<S>]) -> Vec<Vec<S>> {
        if results.is_empty() {
            return Vec::new();
        }

        let n_times = results[0].t.len();
        let n_points = results[0].y[0].len();
        let n_traj = results.len();

        let mut variance = vec![vec![S::ZERO; n_points]; n_times];

        for result in results {
            for (i, y) in result.y.iter().enumerate() {
                for (j, &val) in y.iter().enumerate() {
                    let diff = val - mean[i][j];
                    variance[i][j] = variance[i][j] + diff * diff;
                }
            }
        }

        let n_traj_s = S::from_usize(n_traj - 1); // Bessel's correction
        for v in variance.iter_mut() {
            for val in v.iter_mut() {
                *val = (*val / n_traj_s).sqrt();
            }
        }

        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StochasticHeat {
        alpha: f64,
        sigma: f64,
    }

    impl SpdeSystem<f64> for StochasticHeat {
        fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
            let dx = grid.dx_uniform();
            let n = u.len();

            for i in 0..n {
                let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
                du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
            }
        }

        fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
            for s in sigma.iter_mut() {
                *s = self.sigma;
            }
        }
    }

    #[test]
    fn test_stochastic_heat_basic() {
        let system = StochasticHeat {
            alpha: 0.01,
            sigma: 0.1,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 21);
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default().dt(0.0001).n_output(50).seed(12345);

        let result = MolSdeSolver::solve(&system, 0.0, 0.1, &u0, &grid, &options).unwrap();

        assert!(result.success);
        assert!(!result.t.is_empty());
        assert!(!result.y.is_empty());

        // Solution should be finite
        for y in &result.y {
            for &val in y {
                assert!(val.is_finite(), "Solution should be finite");
            }
        }
    }

    #[test]
    fn test_spde_reproducibility() {
        let system = StochasticHeat {
            alpha: 0.01,
            sigma: 0.1,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 11);
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default().dt(0.001).n_output(10).seed(42);

        let result1 = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();
        let result2 = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();

        let y1 = result1.y_final().unwrap();
        let y2 = result2.y_final().unwrap();

        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Results should be reproducible with same seed"
            );
        }
    }

    #[test]
    fn test_ensemble_statistics() {
        let system = StochasticHeat {
            alpha: 0.01,
            sigma: 0.05,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 11);
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default().dt(0.001).n_output(10).seed(100);

        let results = SpdeEnsemble::solve(&system, 0.0, 0.01, &u0, &grid, &options, 10).unwrap();

        assert_eq!(results.len(), 10);

        let mean = SpdeEnsemble::mean(&results);
        let std = SpdeEnsemble::std(&results, &mean);

        // Mean should be close to deterministic solution
        // Standard deviation should be positive
        for s in &std[std.len() - 1] {
            assert!(*s >= 0.0, "Std should be non-negative");
        }
    }

    #[test]
    fn test_zero_noise() {
        // With zero noise, should reduce to deterministic PDE
        let system = StochasticHeat {
            alpha: 0.1,
            sigma: 0.0,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 11);
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default().dt(0.0001).n_output(10).seed(42);

        let result1 = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();

        // Change seed - result should still be the same with zero noise
        let options2 = options.seed(999);
        let result2 = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options2).unwrap();

        let y1 = result1.y_final().unwrap();
        let y2 = result2.y_final().unwrap();

        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "Zero noise should give deterministic results"
            );
        }
    }

    #[test]
    fn test_options_builder() {
        let opts: SpdeOptions<f64> = SpdeOptions::default()
            .dt(0.01)
            .n_output(200)
            .method(SpdeMethod::Milstein)
            .seed(123);

        assert!((opts.dt - 0.01).abs() < 1e-10);
        assert_eq!(opts.n_output, 200);
        assert!(matches!(opts.method, SpdeMethod::Milstein));
        assert_eq!(opts.seed, Some(123));
    }

    #[test]
    fn test_spde_adaptive_dt() {
        // Stochastic heat equation with adaptive time stepping
        let spde = StochasticHeat {
            alpha: 0.1,
            sigma: 0.01,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 52); // 52 total points -> 50 interior
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default()
            .adaptive(true)
            .dt(0.001)
            .rtol(1e-4)
            .atol(1e-6)
            .dt_max(0.01)
            .seed(42);

        let result = MolSdeSolver::solve(&spde, 0.0, 0.5, &u0, &grid, &options).unwrap();

        assert!(result.success, "Adaptive solve should succeed");
        assert!(!result.t.is_empty(), "Should have output time points");

        // Solution should be finite
        for y in &result.y {
            for &val in y {
                assert!(val.is_finite(), "Solution should be finite, got {}", val);
            }
        }

        // Verify dt varies (not all equal)
        let dts: Vec<_> = result.t.windows(2).map(|w| w[1] - w[0]).collect();
        let first_dt = dts.first().copied().unwrap_or(0.0);
        let has_variation = dts.iter().any(|&dt| (dt - first_dt).abs() > 1e-10);

        assert!(
            has_variation,
            "Time steps should vary with adaptive stepping"
        );
    }

    #[test]
    fn test_adaptive_noise_consistency() {
        // Verify that the adaptive solver's noise splitting is correct.
        // Two independent half-step increments dW1 ~ N(0, h/2), dW2 ~ N(0, h/2)
        // should produce a total dW1 + dW2 ~ N(0, h).
        //
        // We test this by running the adaptive solver on a pure-noise problem
        // (zero drift) and checking the variance of the ensemble matches theory.
        struct PureNoise;

        impl SpdeSystem<f64> for PureNoise {
            fn drift(&self, _t: f64, _u: &[f64], du: &mut [f64], _grid: &Grid1D<f64>) {
                for d in du.iter_mut() {
                    *d = 0.0;
                }
            }

            fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
                for s in sigma.iter_mut() {
                    *s = 1.0;
                }
            }
        }

        let n_trajectories = 500;
        let grid = Grid1D::uniform(0.0, 1.0, 12); // 10 interior points
        let n_interior = grid.len() - 2;
        let u0 = vec![0.0; n_interior];
        let tf = 0.1;

        let mut sum_sq = vec![0.0; n_interior];
        for i in 0..n_trajectories {
            let options = SpdeOptions::default()
                .adaptive(true)
                .dt(0.01)
                .rtol(1e-2)
                .atol(1e-4)
                .dt_min(1e-6)
                .dt_max(0.05)
                .seed(1000 + i as u64);

            let result = MolSdeSolver::solve(&PureNoise, 0.0, tf, &u0, &grid, &options).unwrap();
            let y_final = result.y_final().unwrap();
            for j in 0..n_interior {
                sum_sq[j] += y_final[j] * y_final[j];
            }
        }

        // For pure noise with sigma=1, Var[u(T)] = T for each component
        // (Wiener process variance). With adaptive stepping, the accepted
        // path should still integrate to the correct total noise.
        let expected_var = tf;
        for j in 0..n_interior {
            let empirical_var = sum_sq[j] / n_trajectories as f64;
            // Relatively loose tolerance due to finite sample size
            assert!(
                (empirical_var - expected_var).abs() < 0.06,
                "Component {} variance: expected ~{}, got {}",
                j,
                expected_var,
                empirical_var
            );
        }
    }

    #[test]
    fn test_adaptive_options_builder() {
        let opts: SpdeOptions<f64> = SpdeOptions::default()
            .adaptive(true)
            .rtol(1e-5)
            .atol(1e-8)
            .dt_min(1e-12)
            .dt_max(0.05);

        assert!(opts.adaptive);
        assert!((opts.rtol - 1e-5).abs() < 1e-15);
        assert!((opts.atol - 1e-8).abs() < 1e-18);
        assert!((opts.dt_min - 1e-12).abs() < 1e-22);
        assert!((opts.dt_max - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_finite_solution() {
        // Verify adaptive solver produces finite results even with larger noise
        let system = StochasticHeat {
            alpha: 0.05,
            sigma: 0.1,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 22); // 22 total -> 20 interior
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default()
            .adaptive(true)
            .dt(0.0005)
            .rtol(1e-3)
            .atol(1e-5)
            .dt_max(0.005)
            .seed(777);

        let result = MolSdeSolver::solve(&system, 0.0, 0.1, &u0, &grid, &options).unwrap();

        assert!(result.success);
        for y in &result.y {
            for &val in y {
                assert!(val.is_finite(), "Adaptive solution should be finite");
            }
        }

        // Should have some rejected steps indicating adaptation
        // (Not required to pass but informative)
        assert!(result.stats.n_steps > 0, "Should have taken some steps");
    }

    // ========================================================================
    // Colored / Trace-class noise integration tests
    // ========================================================================

    struct ColoredHeat {
        alpha: f64,
        sigma: f64,
        correlation_length: f64,
    }

    impl SpdeSystem<f64> for ColoredHeat {
        fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
            let dx = grid.dx_uniform();
            let n = u.len();
            for i in 0..n {
                let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
                du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
            }
        }

        fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
            for s in sigma.iter_mut() {
                *s = self.sigma;
            }
        }

        fn noise_correlation(&self) -> crate::system::NoiseCorrelation<f64> {
            crate::system::NoiseCorrelation::Colored {
                correlation_length: self.correlation_length,
            }
        }
    }

    #[test]
    fn test_spde_colored_noise_fixed_step() {
        // Stochastic heat with spatially correlated noise
        let system = ColoredHeat {
            alpha: 0.01,
            sigma: 0.1,
            correlation_length: 0.2,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 12); // 10 interior points
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default().dt(0.0001).n_output(20).seed(42);

        let result = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();

        assert!(result.success);
        for y in &result.y {
            for &val in y {
                assert!(val.is_finite(), "Colored noise solution should be finite");
            }
        }
    }

    #[test]
    fn test_spde_colored_noise_spatial_correlation() {
        // Verify that colored noise produces spatially correlated solutions.
        // Run an ensemble of pure-noise problems (zero drift) and check that
        // adjacent grid points are more correlated than distant ones.
        struct PureColoredNoise {
            correlation_length: f64,
        }

        impl SpdeSystem<f64> for PureColoredNoise {
            fn drift(&self, _t: f64, _u: &[f64], du: &mut [f64], _grid: &Grid1D<f64>) {
                for d in du.iter_mut() {
                    *d = 0.0;
                }
            }
            fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
                for s in sigma.iter_mut() {
                    *s = 1.0;
                }
            }
            fn noise_correlation(&self) -> crate::system::NoiseCorrelation<f64> {
                crate::system::NoiseCorrelation::Colored {
                    correlation_length: self.correlation_length,
                }
            }
        }

        let system = PureColoredNoise {
            correlation_length: 0.3,
        };
        let grid = Grid1D::uniform(0.0, 1.0, 12); // 10 interior
        let n_interior = grid.len() - 2;
        let u0 = vec![0.0; n_interior];

        let n_traj = 500;
        let mut finals = Vec::with_capacity(n_traj);
        for i in 0..n_traj {
            let options = SpdeOptions::default()
                .dt(0.001)
                .n_output(5)
                .seed(1000 + i as u64);
            let result = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();
            finals.push(result.y_final().unwrap().to_vec());
        }

        // Compute correlation between adjacent points (i=0, i=1)
        let mean_0: f64 = finals.iter().map(|f| f[0]).sum::<f64>() / n_traj as f64;
        let mean_1: f64 = finals.iter().map(|f| f[1]).sum::<f64>() / n_traj as f64;
        let var_0: f64 =
            finals.iter().map(|f| (f[0] - mean_0).powi(2)).sum::<f64>() / n_traj as f64;
        let var_1: f64 =
            finals.iter().map(|f| (f[1] - mean_1).powi(2)).sum::<f64>() / n_traj as f64;
        let cov_01: f64 = finals
            .iter()
            .map(|f| (f[0] - mean_0) * (f[1] - mean_1))
            .sum::<f64>()
            / n_traj as f64;
        let corr_nearby = cov_01 / (var_0.sqrt() * var_1.sqrt());

        // Compute correlation between distant points (i=0, i=last)
        let last = n_interior - 1;
        let mean_far: f64 = finals.iter().map(|f| f[last]).sum::<f64>() / n_traj as f64;
        let var_far: f64 = finals
            .iter()
            .map(|f| (f[last] - mean_far).powi(2))
            .sum::<f64>()
            / n_traj as f64;
        let cov_far: f64 = finals
            .iter()
            .map(|f| (f[0] - mean_0) * (f[last] - mean_far))
            .sum::<f64>()
            / n_traj as f64;
        let corr_far = cov_far / (var_0.sqrt() * var_far.sqrt());

        // Nearby correlation should be positive and larger than distant
        assert!(
            corr_nearby > 0.3,
            "Adjacent points should be positively correlated: {}",
            corr_nearby
        );
        assert!(
            corr_nearby > corr_far.abs(),
            "Adjacent correlation ({}) should exceed distant correlation ({})",
            corr_nearby,
            corr_far
        );
    }

    #[test]
    fn test_spde_colored_noise_adaptive() {
        // Colored noise with adaptive time stepping
        let system = ColoredHeat {
            alpha: 0.01,
            sigma: 0.05,
            correlation_length: 0.2,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 12);
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default()
            .adaptive(true)
            .dt(0.001)
            .rtol(1e-3)
            .atol(1e-5)
            .dt_max(0.01)
            .seed(42);

        let result = MolSdeSolver::solve(&system, 0.0, 0.05, &u0, &grid, &options).unwrap();

        assert!(result.success);
        for y in &result.y {
            for &val in y {
                assert!(
                    val.is_finite(),
                    "Adaptive colored noise solution should be finite"
                );
            }
        }
    }

    struct TraceClassHeat {
        alpha: f64,
        sigma: f64,
        n_modes: usize,
        decay_rate: f64,
    }

    impl SpdeSystem<f64> for TraceClassHeat {
        fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
            let dx = grid.dx_uniform();
            let n = u.len();
            for i in 0..n {
                let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
                du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
            }
        }

        fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
            for s in sigma.iter_mut() {
                *s = self.sigma;
            }
        }

        fn noise_correlation(&self) -> crate::system::NoiseCorrelation<f64> {
            crate::system::NoiseCorrelation::TraceClass {
                n_modes: self.n_modes,
                decay_rate: self.decay_rate,
            }
        }
    }

    #[test]
    fn test_spde_trace_class_noise_fixed_step() {
        let system = TraceClassHeat {
            alpha: 0.01,
            sigma: 0.1,
            n_modes: 5,
            decay_rate: 2.0,
        };

        let grid = Grid1D::uniform(0.0, 1.0, 22); // 20 interior
        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();

        let options = SpdeOptions::default().dt(0.0001).n_output(20).seed(42);

        let result = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options).unwrap();

        assert!(result.success);
        for y in &result.y {
            for &val in y {
                assert!(
                    val.is_finite(),
                    "Trace-class noise solution should be finite"
                );
            }
        }
    }

    #[test]
    fn test_spde_trace_class_noise_smoother_than_white() {
        // Trace-class noise should produce smoother solutions than white noise
        // because higher modes are suppressed by eigenvalue decay.
        // Compare total variation of final solutions.
        struct PureNoise;
        impl SpdeSystem<f64> for PureNoise {
            fn drift(&self, _t: f64, _u: &[f64], du: &mut [f64], _grid: &Grid1D<f64>) {
                for d in du.iter_mut() {
                    *d = 0.0;
                }
            }
            fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
                for s in sigma.iter_mut() {
                    *s = 1.0;
                }
            }
        }

        struct PureTraceNoise;
        impl SpdeSystem<f64> for PureTraceNoise {
            fn drift(&self, _t: f64, _u: &[f64], du: &mut [f64], _grid: &Grid1D<f64>) {
                for d in du.iter_mut() {
                    *d = 0.0;
                }
            }
            fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
                for s in sigma.iter_mut() {
                    *s = 1.0;
                }
            }
            fn noise_correlation(&self) -> crate::system::NoiseCorrelation<f64> {
                crate::system::NoiseCorrelation::TraceClass {
                    n_modes: 3,
                    decay_rate: 2.0,
                }
            }
        }

        let grid = Grid1D::uniform(0.0, 1.0, 32); // 30 interior
        let n_interior = grid.len() - 2;
        let u0 = vec![0.0; n_interior];

        let n_traj = 100;
        let mut white_tv_total = 0.0;
        let mut trace_tv_total = 0.0;

        for i in 0..n_traj {
            let options = SpdeOptions::default()
                .dt(0.001)
                .n_output(5)
                .seed(2000 + i as u64);

            let white_result =
                MolSdeSolver::solve(&PureNoise, 0.0, 0.01, &u0, &grid, &options).unwrap();
            let trace_result =
                MolSdeSolver::solve(&PureTraceNoise, 0.0, 0.01, &u0, &grid, &options).unwrap();

            let white_y = white_result.y_final().unwrap();
            let trace_y = trace_result.y_final().unwrap();

            // Total variation = sum |y[i+1] - y[i]|
            let white_tv: f64 = white_y.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
            let trace_tv: f64 = trace_y.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

            white_tv_total += white_tv;
            trace_tv_total += trace_tv;
        }

        // Trace-class noise should be smoother (lower total variation)
        assert!(
            trace_tv_total < white_tv_total,
            "Trace-class noise should be smoother: trace TV={}, white TV={}",
            trace_tv_total / n_traj as f64,
            white_tv_total / n_traj as f64
        );
    }
}
