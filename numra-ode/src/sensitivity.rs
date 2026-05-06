//! Forward sensitivity analysis for parameterised ODE systems.
//!
//! Given an ODE system parameterised by a vector $p \in \mathbb{R}^{N_s}$,
//!
//! ```text
//! dy/dt = f(t, y, p),    y(t_0) = y_0(p),
//! ```
//!
//! the *forward sensitivity matrix* `S(t) = ∂y(t) / ∂p ∈ ℝ^{N × N_s}` satisfies
//! the variational equations
//!
//! ```text
//! dS/dt = J_y(t, y, p) · S + J_p(t, y, p),    S(t_0) = ∂y_0/∂p
//! ```
//!
//! where `J_y = ∂f/∂y ∈ ℝ^{N×N}` and `J_p = ∂f/∂p ∈ ℝ^{N×N_s}`.
//!
//! Numra integrates the augmented state `z = [y; vec(S)] ∈ ℝ^{N(1+N_s)}` with
//! the user's chosen [`crate::Solver`] (DoPri5, Tsit5, Vern\*, Radau5, BDF,
//! Esdirk\*, Auto). The augmented Jacobian is `block_diag(J_y, …, J_y)` —
//! the standard CVODES *simultaneous-corrector* formulation.
//!
//! # Layout conventions
//!
//! Numra mixes row-major and column-major flattening so that each Jacobian's
//! "natural" axis is contiguous. Implementors of [`ParametricOdeSystem`] must
//! follow these rules exactly:
//!
//! | Quantity                        | Layout         | Index map                         |
//! | ------------------------------- | -------------- | --------------------------------- |
//! | state `y`                       | flat, length `N`           | `y[i]`                |
//! | state Jacobian `J_y`            | row-major, `N × N`         | `jy[i*N + j] = ∂f_i/∂y_j` |
//! | parameter Jacobian `J_p`        | column-major, `N × N_s`    | `jp[k*N + i] = ∂f_i/∂p_k` |
//! | sensitivity `S`                 | column-major, `N × N_s`    | `s[k*N + i] = ∂y_i/∂p_k`  |
//! | augmented state `z`             | `[y; vec(S)]`              | `z[N + k*N + i] = S_{i,k}` |
//! | initial sensitivity `∂y_0/∂p`   | column-major, `N × N_s`    | `s0[k*N + i] = ∂y_{0,i}/∂p_k` |
//!
//! Choosing column-major for `J_p` and `S` matches CVODES indexing, makes the
//! per-parameter sub-vector `S_{:,k}` contiguous (and free to slice via
//! [`SensitivityResult::sensitivity_for_param`]), and improves cache locality
//! in the inner loop `Ṡ_{:,k} = J_y · S_{:,k} + J_p_{:,k}`. `J_y` stays
//! row-major because state-row access is the standard `OdeSystem::jacobian`
//! convention used by every implicit solver in this crate.
//!
//! # Worked example: 2-state, 2-parameter linear ODE
//!
//! Consider the system
//!
//! ```text
//! ẏ_0 = -p_0 · y_0 + p_1 · y_1
//! ẏ_1 =          - p_1 · y_1
//! ```
//!
//! Then
//!
//! ```text
//! J_y = [ -p_0    p_1 ]      J_p = [ -y_0    y_1 ]
//!       [   0    -p_1 ]            [   0    -y_1 ]
//! ```
//!
//! With `N = N_s = 2`, `J_y` is row-major and `J_p` is column-major.
//! Implementing the trait by hand:
//!
//! ```
//! use numra_ode::sensitivity::ParametricOdeSystem;
//!
//! struct Lin2 { p: [f64; 2] }
//!
//! impl ParametricOdeSystem<f64> for Lin2 {
//!     fn n_states(&self) -> usize { 2 }
//!     fn n_params(&self) -> usize { 2 }
//!     fn params(&self) -> &[f64] { &self.p }
//!
//!     fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
//!         dy[0] = -p[0] * y[0] + p[1] * y[1];
//!         dy[1] =                -p[1] * y[1];
//!     }
//!
//!     fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
//!         // Row-major (N × N): jy[i*N + j] = ∂f_i/∂y_j
//!         jy[0*2 + 0] = -self.p[0];   // ∂f_0/∂y_0
//!         jy[0*2 + 1] =  self.p[1];   // ∂f_0/∂y_1
//!         jy[1*2 + 0] =  0.0;         // ∂f_1/∂y_0
//!         jy[1*2 + 1] = -self.p[1];   // ∂f_1/∂y_1
//!     }
//!
//!     fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
//!         // Column-major (N × N_s): jp[k*N + i] = ∂f_i/∂p_k
//!         jp[0*2 + 0] = -y[0];        // ∂f_0/∂p_0
//!         jp[0*2 + 1] =  0.0;         // ∂f_1/∂p_0
//!         jp[1*2 + 0] =  y[1];        // ∂f_0/∂p_1
//!         jp[1*2 + 1] = -y[1];        // ∂f_1/∂p_1
//!     }
//! }
//!
//! let sys = Lin2 { p: [1.0, 0.5] };
//! assert_eq!(sys.n_states(), 2);
//! assert_eq!(sys.n_params(), 2);
//! ```
//!
//! Note the asymmetry: `J_y[0*2 + 1]` is row 0, column 1 of `J_y` (state
//! Jacobian, row-major), while `J_p[1*2 + 0]` is row 0, column 1 of `J_p`
//! (parameter Jacobian, column-major). They look indexed the same way in
//! Rust syntax but address different mathematical entries by design.
//!
//! # Send / Sync
//!
//! [`ParametricOdeSystem`] does **not** require `Send + Sync`, matching the
//! [`crate::OdeSystem`] precedent. Add the bounds at the call site
//! (`where Sys: ParametricOdeSystem<S> + Send + Sync`) when sharing the
//! system across threads — for instance in parallel Monte Carlo. Tightening
//! a trait bound is a breaking change; loosening it is not, so the default
//! is permissive.
//!
//! # Performance note (v1)
//!
//! Numra's v1 ships *correct* forward sensitivity through every solver, with
//! analytical `J_y` and `J_p` overrides replacing FD on the dominant cost
//! path. The block-diagonal structure of the augmented Jacobian — which
//! would let implicit solvers reuse a single LU of `M = (1/γh)·I_N - J_y`
//! across all `N_s + 1` sub-systems — is **not yet exploited**: Radau5 and
//! BDF currently perform dense LU on the full `N(N_s+1) × N(N_s+1)` matrix.
//! Block-aware factorisation is tracked as a named follow-up (see
//! `docs/internal-followups.md`).
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 6 May 2026

use numra_core::Scalar;

use crate::problem::OdeSystem;
use crate::solver::SolverStats;

/// An ODE system parameterised by a parameter vector `p`.
///
/// Implementors expose dimensions, the nominal parameter vector, the
/// parameterised right-hand side, and optionally analytical Jacobians.
/// Forward finite differences are provided as defaults for `jacobian_y` and
/// `jacobian_p`, so a minimal implementation supplies only `n_states`,
/// `n_params`, `params`, and `rhs_with_params`.
///
/// See the [module-level documentation](self) for the layout conventions
/// (`J_y` row-major, `J_p` and `S` column-major) and a worked 2-state,
/// 2-parameter example.
pub trait ParametricOdeSystem<S: Scalar> {
    /// Number of state variables `N` (the dimension of `y`).
    fn n_states(&self) -> usize;

    /// Number of parameters `N_s` (the dimension of `p`).
    fn n_params(&self) -> usize;

    /// Nominal parameter vector. Length must equal [`Self::n_params`].
    fn params(&self) -> &[S];

    /// Evaluate `f(t, y, p) -> dy/dt` for an arbitrary parameter vector `p`.
    ///
    /// This is the primitive form: it lets the FD-default Jacobians perturb
    /// `p` without mutating `self`.
    fn rhs_with_params(&self, t: S, y: &[S], p: &[S], dydt: &mut [S]);

    /// Convenience: evaluate the RHS at the system's nominal parameters.
    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        self.rhs_with_params(t, y, self.params(), dydt);
    }

    /// Fill the state Jacobian `J_y[i,j] = ∂f_i/∂y_j` in **row-major** order
    /// (`jac[i*N + j]`, length `N²`).
    ///
    /// Default: forward finite differences via [`Self::rhs_with_params`].
    /// Override to supply an analytical Jacobian — required for performance
    /// on stiff problems with implicit solvers.
    fn jacobian_y(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.n_states();
        let p = self.params();
        let h_factor = S::EPSILON.sqrt();

        let mut f0 = vec![S::ZERO; n];
        let mut f1 = vec![S::ZERO; n];
        let mut y_pert = y.to_vec();

        self.rhs_with_params(t, y, p, &mut f0);

        for j in 0..n {
            let yj = y_pert[j];
            let h = h_factor * (S::ONE + yj.abs());
            y_pert[j] = yj + h;
            self.rhs_with_params(t, &y_pert, p, &mut f1);
            y_pert[j] = yj;
            for i in 0..n {
                jac[i * n + j] = (f1[i] - f0[i]) / h;
            }
        }
    }

    /// Fill the parameter Jacobian `J_p[i,k] = ∂f_i/∂p_k` in **column-major**
    /// order (`jp[k*N + i]`, length `N · N_s`).
    ///
    /// Default: forward finite differences. Override for performance.
    fn jacobian_p(&self, t: S, y: &[S], jp: &mut [S]) {
        let n = self.n_states();
        let np = self.n_params();
        let p_nominal = self.params();
        let h_factor = S::EPSILON.sqrt();

        let mut f0 = vec![S::ZERO; n];
        let mut f1 = vec![S::ZERO; n];
        let mut p_pert = p_nominal.to_vec();

        self.rhs_with_params(t, y, p_nominal, &mut f0);

        for k in 0..np {
            let pk = p_pert[k];
            let h = h_factor * (S::ONE + pk.abs());
            p_pert[k] = pk + h;
            self.rhs_with_params(t, y, &p_pert, &mut f1);
            p_pert[k] = pk;
            for i in 0..n {
                jp[k * n + i] = (f1[i] - f0[i]) / h;
            }
        }
    }

    /// Initial sensitivity `∂y_0/∂p`, column-major (`s0[k*N + i]`, length
    /// `N · N_s`).
    ///
    /// Defaults to zero, which is correct when `y_0` does not depend on `p`.
    /// Override only if `y_0 = y_0(p)`.
    fn initial_sensitivity(&self, _y0: &[S], s0: &mut [S]) {
        for s in s0.iter_mut() {
            *s = S::ZERO;
        }
    }
}

/// Wraps a [`ParametricOdeSystem`] as an [`OdeSystem`] over the augmented
/// state `z = [y; vec(S)]` (column-major sensitivity flattening).
///
/// The augmented dimension is `N · (1 + N_s)`. The augmented Jacobian is
/// `block_diag(J_y, J_y, …, J_y)` (CVODES simultaneous-corrector form).
pub struct AugmentedSystem<S: Scalar, Sys: ParametricOdeSystem<S>> {
    /// The underlying parameterised system.
    pub system: Sys,
    jy_scratch: std::cell::RefCell<Vec<S>>,
    jp_scratch: std::cell::RefCell<Vec<S>>,
}

impl<S: Scalar, Sys: ParametricOdeSystem<S>> AugmentedSystem<S, Sys> {
    /// Wrap a [`ParametricOdeSystem`] into an `OdeSystem`-shaped augmented
    /// system suitable for any [`crate::Solver`].
    pub fn new(system: Sys) -> Self {
        let n = system.n_states();
        let np = system.n_params();
        Self {
            system,
            jy_scratch: std::cell::RefCell::new(vec![S::ZERO; n * n]),
            jp_scratch: std::cell::RefCell::new(vec![S::ZERO; n * np]),
        }
    }

    /// Total augmented dimension `N · (1 + N_s)`.
    pub fn augmented_dim(&self) -> usize {
        self.system.n_states() * (1 + self.system.n_params())
    }

    /// Build the initial augmented state `[y_0; vec(∂y_0/∂p)]`.
    pub fn initial_augmented(&self, y0: &[S]) -> Vec<S> {
        let n = self.system.n_states();
        let np = self.system.n_params();

        let mut z0 = Vec::with_capacity(n * (1 + np));
        z0.extend_from_slice(y0);

        let mut s0 = vec![S::ZERO; n * np];
        self.system.initial_sensitivity(y0, &mut s0);
        z0.extend_from_slice(&s0);
        z0
    }
}

impl<S: Scalar, Sys: ParametricOdeSystem<S>> OdeSystem<S> for AugmentedSystem<S, Sys> {
    fn dim(&self) -> usize {
        self.augmented_dim()
    }

    /// Augmented RHS: `dy/dt = f(t,y,p)`, `dS_{:,k}/dt = J_y · S_{:,k} + J_p_{:,k}`.
    fn rhs(&self, t: S, z: &[S], dz: &mut [S]) {
        let n = self.system.n_states();
        let np = self.system.n_params();
        let y = &z[..n];

        // (a) original dynamics.
        self.system.rhs(t, y, &mut dz[..n]);

        // (b) Jacobians at the current state.
        let mut jy = self.jy_scratch.borrow_mut();
        let mut jp = self.jp_scratch.borrow_mut();
        self.system.jacobian_y(t, y, &mut jy);
        self.system.jacobian_p(t, y, &mut jp);

        // (c) per-parameter sensitivity column: dS_{:,k}/dt = J_y · S_{:,k} + J_p_{:,k}.
        // Sensitivity column-major: S_{i,k} lives at z[n + k*n + i].
        for k in 0..np {
            for i in 0..n {
                let mut acc = S::ZERO;
                // J_y row-major dot S_{:,k} (contiguous slice of z).
                for j in 0..n {
                    acc = acc + jy[i * n + j] * z[n + k * n + j];
                }
                acc = acc + jp[k * n + i];
                dz[n + k * n + i] = acc;
            }
        }
    }

    /// Augmented Jacobian, row-major over the full `N(1+N_s) × N(1+N_s)` block.
    ///
    /// Fills the `N_s + 1` diagonal blocks with `J_y`. Off-diagonal blocks
    /// (which would carry `∂(J_y · S_{:,k})/∂y` second-derivative coupling)
    /// are zero — the standard CVODES *simultaneous-corrector* approximation.
    /// Implicit solvers in v1 perform dense LU on the full block; the
    /// block-diagonal-aware fast path is a tracked follow-up.
    fn jacobian(&self, t: S, z: &[S], jac: &mut [S]) {
        let n = self.system.n_states();
        let np = self.system.n_params();
        let dim = n * (1 + np);
        let y = &z[..n];

        // Zero the entire augmented matrix first.
        for slot in jac.iter_mut() {
            *slot = S::ZERO;
        }

        let mut jy = self.jy_scratch.borrow_mut();
        self.system.jacobian_y(t, y, &mut jy);

        // Place J_y on each of the n_s + 1 diagonal blocks.
        for block in 0..(1 + np) {
            let row0 = block * n;
            let col0 = block * n;
            for i in 0..n {
                for j in 0..n {
                    jac[(row0 + i) * dim + (col0 + j)] = jy[i * n + j];
                }
            }
        }
    }
}

/// Result of an ODE forward-sensitivity solve.
///
/// Layout matches the conventions in the [module-level docs](self):
/// the state `y` is row-major over time; the sensitivity is row-major over
/// time, column-major within each per-time block.
#[derive(Clone, Debug)]
pub struct SensitivityResult<S: Scalar> {
    /// Output time points (length `n_times`).
    pub t: Vec<S>,
    /// State trajectory: `y[i*N + j] = y_j(t_i)`, length `n_times * N`.
    pub y: Vec<S>,
    /// Sensitivities: per-time block of length `N · N_s`, column-major
    /// within the block, so
    /// `sensitivity[i * (N*N_s) + k*N + j] = ∂y_j(t_i)/∂p_k`.
    pub sensitivity: Vec<S>,
    /// Number of state variables `N`.
    pub n_states: usize,
    /// Number of parameters `N_s`.
    pub n_params: usize,
    /// Solver statistics from the underlying integration.
    pub stats: SolverStats,
    /// Was the integration successful?
    pub success: bool,
    /// Error description if the integration failed.
    pub message: String,
}

impl<S: Scalar> SensitivityResult<S> {
    /// Number of output time points.
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// `true` if there are no output time points.
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// State vector at output index `i` (length `N`).
    pub fn y_at(&self, i: usize) -> &[S] {
        let n = self.n_states;
        let off = i * n;
        &self.y[off..off + n]
    }

    /// Full sensitivity block at output index `i`, column-major
    /// (length `N · N_s`). `sensitivity_at(i)[k*N + j] = ∂y_j(t_i)/∂p_k`.
    pub fn sensitivity_at(&self, i: usize) -> &[S] {
        let block = self.n_states * self.n_params;
        let off = i * block;
        &self.sensitivity[off..off + block]
    }

    /// Contiguous slice `S_{:,k}(t_i) = ∂y/∂p_k` at output index `i`,
    /// length `N`. Free slice thanks to the column-major flattening.
    pub fn sensitivity_for_param(&self, i: usize, k: usize) -> &[S] {
        let n = self.n_states;
        let block = n * self.n_params;
        let off = i * block + k * n;
        &self.sensitivity[off..off + n]
    }

    /// Single component `∂y_state(t_i)/∂p_param`.
    pub fn dyi_dpj(&self, i: usize, state: usize, param: usize) -> S {
        let n = self.n_states;
        let block = n * self.n_params;
        self.sensitivity[i * block + param * n + state]
    }

    /// State at the final output time. Returns an empty slice if the result
    /// has no time points.
    pub fn final_state(&self) -> &[S] {
        if self.is_empty() {
            &[]
        } else {
            self.y_at(self.len() - 1)
        }
    }

    /// Sensitivity block at the final output time, column-major (length
    /// `N · N_s`). Empty slice when the result has no time points.
    pub fn final_sensitivity(&self) -> &[S] {
        if self.is_empty() {
            &[]
        } else {
            self.sensitivity_at(self.len() - 1)
        }
    }

    /// Normalised (logarithmic) sensitivity at output index `i`:
    /// `(p_k / y_j) · ∂y_j/∂p_k`, column-major in the same shape as
    /// [`Self::sensitivity_at`]. Components where `|y_j|` falls below
    /// `1e-15 · max(1, |y|_∞)` are reported as zero to avoid blow-up.
    pub fn normalized_sensitivity_at(&self, i: usize, p_nominal: &[S]) -> Vec<S> {
        assert_eq!(
            p_nominal.len(),
            self.n_params,
            "p_nominal length {} does not match n_params {}",
            p_nominal.len(),
            self.n_params,
        );
        let n = self.n_states;
        let np = self.n_params;
        let y = self.y_at(i);
        let s = self.sensitivity_at(i);

        let mut y_max = S::ZERO;
        for &v in y {
            let av = v.abs();
            if av > y_max {
                y_max = av;
            }
        }
        let zero_threshold = S::from_f64(1e-15) * (S::ONE + y_max);

        let mut out = vec![S::ZERO; n * np];
        for k in 0..np {
            for j in 0..n {
                let yj = y[j];
                if yj.abs() <= zero_threshold {
                    out[k * n + j] = S::ZERO;
                } else {
                    out[k * n + j] = (p_nominal[k] / yj) * s[k * n + j];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal scalar decay system used to drive the trait through
    /// `OdeSystem::rhs` and `OdeSystem::jacobian` without involving a solver.
    struct ExpDecay {
        k: f64,
    }

    impl ParametricOdeSystem<f64> for ExpDecay {
        fn n_states(&self) -> usize {
            1
        }
        fn n_params(&self) -> usize {
            1
        }
        fn params(&self) -> &[f64] {
            std::slice::from_ref(&self.k)
        }
        fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
            dy[0] = -p[0] * y[0];
        }
        fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
            jy[0] = -self.k;
        }
        fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
            jp[0] = -y[0];
        }
    }

    /// 2-state, 2-parameter linear system (mirrors the module-doc example).
    /// Lets us exercise the asymmetric J_y / J_p layouts in code.
    struct Lin2 {
        p: [f64; 2],
    }

    impl ParametricOdeSystem<f64> for Lin2 {
        fn n_states(&self) -> usize {
            2
        }
        fn n_params(&self) -> usize {
            2
        }
        fn params(&self) -> &[f64] {
            &self.p
        }
        fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
            dy[0] = -p[0] * y[0] + p[1] * y[1];
            dy[1] = -p[1] * y[1];
        }
        fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
            jy[0 * 2 + 0] = -self.p[0];
            jy[0 * 2 + 1] = self.p[1];
            jy[1 * 2 + 0] = 0.0;
            jy[1 * 2 + 1] = -self.p[1];
        }
        fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
            jp[0 * 2 + 0] = -y[0];
            jp[0 * 2 + 1] = 0.0;
            jp[1 * 2 + 0] = y[1];
            jp[1 * 2 + 1] = -y[1];
        }
    }

    #[test]
    fn augmented_dim_and_initial_state() {
        let aug = AugmentedSystem::new(Lin2 { p: [1.0, 0.5] });
        assert_eq!(aug.augmented_dim(), 2 * (1 + 2));
        let z0 = aug.initial_augmented(&[1.0, 2.0]);
        assert_eq!(z0.len(), 6);
        assert_eq!(&z0[..2], &[1.0, 2.0]);
        for s in &z0[2..] {
            assert_eq!(*s, 0.0);
        }
    }

    #[test]
    fn augmented_rhs_matches_hand_derivation() {
        // System: dy/dt = -k y, S_{0,0} = ∂y/∂k.
        // dS/dt = J_y * S + J_p = -k * S - y.
        let aug = AugmentedSystem::new(ExpDecay { k: 0.5 });
        let z = vec![2.0, 1.0]; // y = 2.0, S = 1.0
        let mut dz = vec![0.0; 2];
        aug.rhs(0.0, &z, &mut dz);

        // dy/dt = -0.5 * 2.0 = -1.0
        assert!((dz[0] + 1.0).abs() < 1e-12);
        // dS/dt = -0.5 * 1.0 + (-2.0) = -2.5
        assert!((dz[1] + 2.5).abs() < 1e-12);
    }

    #[test]
    fn augmented_rhs_matches_hand_derivation_two_param() {
        // Lin2 system at y = (1, 2), p = (1, 0.5), S = identity (initial sensitivity ID).
        let aug = AugmentedSystem::new(Lin2 { p: [1.0, 0.5] });

        // z = [y_0, y_1, S_{0,0}, S_{1,0}, S_{0,1}, S_{1,1}]
        //     [ 1,    2,    1,       0,       0,       1   ]
        let z = vec![1.0, 2.0, 1.0, 0.0, 0.0, 1.0];
        let mut dz = vec![0.0; 6];
        aug.rhs(0.0, &z, &mut dz);

        // dy/dt
        // dy_0/dt = -1*1 + 0.5*2 = 0.0
        // dy_1/dt =        -0.5*2 = -1.0
        assert!((dz[0] - 0.0).abs() < 1e-12);
        assert!((dz[1] + 1.0).abs() < 1e-12);

        // dS_{:,0}/dt = J_y · [1; 0] + J_p_{:,0}
        // J_y · [1; 0] = [-1; 0];   J_p_{:,0} = [-1; 0]; total = [-2; 0]
        assert!((dz[2] + 2.0).abs() < 1e-12);
        assert!((dz[3] - 0.0).abs() < 1e-12);

        // dS_{:,1}/dt = J_y · [0; 1] + J_p_{:,1}
        // J_y · [0; 1] = [0.5; -0.5]; J_p_{:,1} = [2; -2]; total = [2.5; -2.5]
        assert!((dz[4] - 2.5).abs() < 1e-12);
        assert!((dz[5] + 2.5).abs() < 1e-12);
    }

    #[test]
    fn augmented_jacobian_is_block_diagonal_with_jy() {
        let aug = AugmentedSystem::new(Lin2 { p: [1.0, 0.5] });
        let z = vec![1.0, 2.0, 1.0, 0.0, 0.0, 1.0];
        let dim = aug.augmented_dim();
        let mut jac = vec![0.0; dim * dim];
        aug.jacobian(0.0, &z, &mut jac);

        // Expected J_y at p=(1, 0.5):
        //   [-1.0  0.5]
        //   [ 0.0 -0.5]
        let jy = [-1.0_f64, 0.5, 0.0, -0.5];

        for block in 0..3 {
            let row0 = block * 2;
            let col0 = block * 2;
            for i in 0..2 {
                for j in 0..2 {
                    let got = jac[(row0 + i) * dim + (col0 + j)];
                    let want = jy[i * 2 + j];
                    assert!(
                        (got - want).abs() < 1e-12,
                        "diag block {block}, ({i},{j}): got {got}, want {want}"
                    );
                }
            }
        }

        // Off-diagonal blocks must be zero.
        for br in 0..3 {
            for bc in 0..3 {
                if br == bc {
                    continue;
                }
                for i in 0..2 {
                    for j in 0..2 {
                        let r = br * 2 + i;
                        let c = bc * 2 + j;
                        assert_eq!(jac[r * dim + c], 0.0, "off-diag ({br},{bc}) ({i},{j})");
                    }
                }
            }
        }
    }

    #[test]
    fn fd_default_jacobian_y_matches_analytical() {
        // Re-implement Lin2 without analytical J_y so the FD default is exercised.
        struct Lin2Fd {
            p: [f64; 2],
        }
        impl ParametricOdeSystem<f64> for Lin2Fd {
            fn n_states(&self) -> usize {
                2
            }
            fn n_params(&self) -> usize {
                2
            }
            fn params(&self) -> &[f64] {
                &self.p
            }
            fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
                dy[0] = -p[0] * y[0] + p[1] * y[1];
                dy[1] = -p[1] * y[1];
            }
            // No jacobian_y / jacobian_p override — FD defaults are used.
        }

        let analytical = Lin2 { p: [1.0, 0.5] };
        let fd = Lin2Fd { p: [1.0, 0.5] };

        let y = [1.0, 2.0];
        let mut a_jy = [0.0; 4];
        let mut f_jy = [0.0; 4];
        analytical.jacobian_y(0.0, &y, &mut a_jy);
        fd.jacobian_y(0.0, &y, &mut f_jy);
        for k in 0..4 {
            assert!(
                (a_jy[k] - f_jy[k]).abs() < 1e-7,
                "j_y[{k}] analytical={} fd={}",
                a_jy[k],
                f_jy[k]
            );
        }

        let mut a_jp = [0.0; 4];
        let mut f_jp = [0.0; 4];
        analytical.jacobian_p(0.0, &y, &mut a_jp);
        fd.jacobian_p(0.0, &y, &mut f_jp);
        for k in 0..4 {
            assert!(
                (a_jp[k] - f_jp[k]).abs() < 1e-7,
                "j_p[{k}] analytical={} fd={}",
                a_jp[k],
                f_jp[k]
            );
        }
    }

    #[test]
    fn sensitivity_result_accessors() {
        // Hand-construct a result for a 2-state, 3-param trajectory at 2 time points.
        // sensitivity layout per time block (column-major, length N*N_s = 6):
        //   [ S_{0,0}  S_{1,0}  | S_{0,1}  S_{1,1}  | S_{0,2}  S_{1,2} ]
        let res = SensitivityResult {
            t: vec![0.0, 1.0],
            y: vec![1.0, 2.0, 0.5, 1.5],
            sensitivity: vec![
                // t = 0
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // t = 1
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            ],
            n_states: 2,
            n_params: 3,
            stats: SolverStats::new(),
            success: true,
            message: String::new(),
        };

        assert_eq!(res.len(), 2);
        assert_eq!(res.y_at(1), &[0.5, 1.5]);
        assert_eq!(res.sensitivity_at(0).len(), 6);
        assert_eq!(res.sensitivity_for_param(1, 2), &[1.5, 1.6]);
        assert!((res.dyi_dpj(0, 1, 0) - 0.2).abs() < 1e-12);
        assert!((res.dyi_dpj(1, 0, 2) - 1.5).abs() < 1e-12);
        assert_eq!(res.final_state(), &[0.5, 1.5]);
        assert_eq!(res.final_sensitivity().len(), 6);
    }

    #[test]
    fn normalized_sensitivity_handles_zero_state_safely() {
        let res = SensitivityResult {
            t: vec![0.0],
            y: vec![1.0, 0.0],
            sensitivity: vec![0.5, 0.7],
            n_states: 2,
            n_params: 1,
            stats: SolverStats::new(),
            success: true,
            message: String::new(),
        };
        let norm = res.normalized_sensitivity_at(0, &[2.0]);
        // (p / y_0) · S_{0,0} = (2 / 1) · 0.5 = 1.0
        assert!((norm[0] - 1.0).abs() < 1e-12);
        // y_1 = 0 → reported as zero rather than ∞.
        assert_eq!(norm[1], 0.0);
    }
}
