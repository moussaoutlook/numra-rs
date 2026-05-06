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
//!
//!     // Flag analytical so AugmentedSystem honors the overrides above
//!     // instead of falling through to its inline-FD fast path.
//!     fn has_analytical_jacobian_y(&self) -> bool { true }
//!     fn has_analytical_jacobian_p(&self) -> bool { true }
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

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverStats};

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

    /// Returns `true` iff [`Self::jacobian_y`] has been overridden with an
    /// analytical implementation. Default: `false`.
    ///
    /// `AugmentedSystem` checks this flag inside its hot path. When `false`
    /// (the default), it inlines its own forward-FD using reused scratch
    /// buffers — bypassing both the trait default and any override.
    /// When `true`, it calls `system.jacobian_y(...)` directly so the user's
    /// analytical override is honored.
    ///
    /// This is the pattern used by CVODES (`set_jacobian_user_supplied`):
    /// flagging analytical Jacobians lets the augmented-system path skip
    /// FD scratch allocation in the common FD case while still respecting
    /// analytical overrides in the stiff-problem case where they materially
    /// matter. **If you override `jacobian_y`, also override this method to
    /// return `true`** — otherwise your override is silently bypassed when
    /// running through `solve_forward_sensitivity`.
    fn has_analytical_jacobian_y(&self) -> bool {
        false
    }

    /// Returns `true` iff [`Self::jacobian_p`] has been overridden with an
    /// analytical implementation. Default: `false`. See
    /// [`Self::has_analytical_jacobian_y`] for the rationale and contract.
    fn has_analytical_jacobian_p(&self) -> bool {
        false
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
    // Reusable FD scratch lifted out of the trait's per-call allocations.
    // Touched only on the FD path (when has_analytical_jacobian_{y,p}
    // returns false). Each RefCell is borrowed in isolation; no overlap.
    fd_f0: std::cell::RefCell<Vec<S>>,
    fd_f1: std::cell::RefCell<Vec<S>>,
    fd_y_pert: std::cell::RefCell<Vec<S>>,
    fd_p_pert: std::cell::RefCell<Vec<S>>,
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
            fd_f0: std::cell::RefCell::new(vec![S::ZERO; n]),
            fd_f1: std::cell::RefCell::new(vec![S::ZERO; n]),
            fd_y_pert: std::cell::RefCell::new(vec![S::ZERO; n]),
            fd_p_pert: std::cell::RefCell::new(vec![S::ZERO; np]),
        }
    }

    /// Forward-FD `J_y` using AugmentedSystem-local scratch buffers. Only
    /// invoked when `system.has_analytical_jacobian_y()` is `false`.
    fn fd_jacobian_y_inline(&self, t: S, y: &[S], jy: &mut [S]) {
        let n = self.system.n_states();
        let p = self.system.params();
        let h_factor = S::EPSILON.sqrt();
        let mut f0 = self.fd_f0.borrow_mut();
        let mut f1 = self.fd_f1.borrow_mut();
        let mut y_pert = self.fd_y_pert.borrow_mut();

        self.system.rhs_with_params(t, y, p, &mut f0);
        y_pert.copy_from_slice(y);
        for j in 0..n {
            let yj = y_pert[j];
            let h = h_factor * (S::ONE + yj.abs());
            y_pert[j] = yj + h;
            self.system.rhs_with_params(t, &y_pert, p, &mut f1);
            y_pert[j] = yj;
            for i in 0..n {
                jy[i * n + j] = (f1[i] - f0[i]) / h;
            }
        }
    }

    /// Forward-FD `J_p` using AugmentedSystem-local scratch buffers. Only
    /// invoked when `system.has_analytical_jacobian_p()` is `false`.
    fn fd_jacobian_p_inline(&self, t: S, y: &[S], jp: &mut [S]) {
        let n = self.system.n_states();
        let np = self.system.n_params();
        let p_nominal = self.system.params();
        let h_factor = S::EPSILON.sqrt();
        let mut f0 = self.fd_f0.borrow_mut();
        let mut f1 = self.fd_f1.borrow_mut();
        let mut p_pert = self.fd_p_pert.borrow_mut();

        self.system.rhs_with_params(t, y, p_nominal, &mut f0);
        p_pert.copy_from_slice(p_nominal);
        for k in 0..np {
            let pk = p_pert[k];
            let h = h_factor * (S::ONE + pk.abs());
            p_pert[k] = pk + h;
            self.system.rhs_with_params(t, y, &p_pert, &mut f1);
            p_pert[k] = pk;
            for i in 0..n {
                jp[k * n + i] = (f1[i] - f0[i]) / h;
            }
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

        // (b) Jacobians at the current state. Analytical when the system
        // flags it; otherwise inline FD using local scratch buffers (lifted
        // out of the trait defaults so the integration's hot path doesn't
        // pay per-call allocation).
        let mut jy = self.jy_scratch.borrow_mut();
        let mut jp = self.jp_scratch.borrow_mut();
        if self.system.has_analytical_jacobian_y() {
            self.system.jacobian_y(t, y, &mut jy);
        } else {
            drop(jy);
            self.fd_jacobian_y_inline(t, y, &mut self.jy_scratch.borrow_mut());
            jy = self.jy_scratch.borrow_mut();
        }
        if self.system.has_analytical_jacobian_p() {
            self.system.jacobian_p(t, y, &mut jp);
        } else {
            drop(jp);
            self.fd_jacobian_p_inline(t, y, &mut self.jp_scratch.borrow_mut());
            jp = self.jp_scratch.borrow_mut();
        }

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

/// Forwarding impl: a reference to a parametric system is itself a parametric
/// system. Lets [`solve_forward_sensitivity`] accept `&Sys` without taking
/// ownership of the user's data.
impl<S: Scalar, T: ParametricOdeSystem<S>> ParametricOdeSystem<S> for &T {
    fn n_states(&self) -> usize {
        (*self).n_states()
    }
    fn n_params(&self) -> usize {
        (*self).n_params()
    }
    fn params(&self) -> &[S] {
        (*self).params()
    }
    fn rhs_with_params(&self, t: S, y: &[S], p: &[S], dydt: &mut [S]) {
        (*self).rhs_with_params(t, y, p, dydt)
    }
    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        (*self).rhs(t, y, dydt)
    }
    fn jacobian_y(&self, t: S, y: &[S], jac: &mut [S]) {
        (*self).jacobian_y(t, y, jac)
    }
    fn jacobian_p(&self, t: S, y: &[S], jp: &mut [S]) {
        (*self).jacobian_p(t, y, jp)
    }
    fn initial_sensitivity(&self, y0: &[S], s0: &mut [S]) {
        (*self).initial_sensitivity(y0, s0)
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

/// Closure-shaped adapter for [`solve_forward_sensitivity_with`].
///
/// Wraps a closure of shape `Fn(t, y, p, dydt)` and a fixed parameter vector
/// into a [`ParametricOdeSystem`] using FD-default Jacobians. Public so that
/// callers writing a one-shot harness can construct it explicitly when they
/// need to share it across multiple solver invocations.
pub struct ClosureSystem<S: Scalar, F> {
    rhs: F,
    params: Vec<S>,
    n_states: usize,
}

impl<S: Scalar, F> ClosureSystem<S, F>
where
    F: Fn(S, &[S], &[S], &mut [S]),
{
    /// Build a closure-backed [`ParametricOdeSystem`].
    pub fn new(rhs: F, params: Vec<S>, n_states: usize) -> Self {
        Self {
            rhs,
            params,
            n_states,
        }
    }
}

impl<S: Scalar, F> ParametricOdeSystem<S> for ClosureSystem<S, F>
where
    F: Fn(S, &[S], &[S], &mut [S]),
{
    fn n_states(&self) -> usize {
        self.n_states
    }
    fn n_params(&self) -> usize {
        self.params.len()
    }
    fn params(&self) -> &[S] {
        &self.params
    }
    fn rhs_with_params(&self, t: S, y: &[S], p: &[S], dydt: &mut [S]) {
        (self.rhs)(t, y, p, dydt)
    }
}

/// Compute the forward sensitivity matrix `S(t) = ∂y(t) / ∂p` of an ODE
/// solution using any [`Solver`].
///
/// Builds the augmented state `z = [y; vec(S)]` (column-major sensitivity
/// flattening; see [module-level docs](self)), drives `Sol` over the
/// augmented system, and splits the trajectory back into `y` and
/// sensitivity blocks at every output time the solver produced.
///
/// # Errors
///
/// Returns whatever error `Sol::solve` raises. If the underlying integration
/// returns `success = false` (e.g. step-size collapse or max-steps reached),
/// the returned [`SensitivityResult`] propagates that with empty trajectory
/// vectors and the solver's diagnostic message.
///
/// # Example
///
/// ```
/// use numra_ode::sensitivity::{solve_forward_sensitivity, ParametricOdeSystem};
/// use numra_ode::{DoPri5, SolverOptions};
///
/// // dy/dt = -k * y, y(0) = 1, k = 0.5.
/// // Analytical: y(t) = exp(-k t), ∂y/∂k = -t exp(-k t).
/// struct Decay { k: f64 }
/// impl ParametricOdeSystem<f64> for Decay {
///     fn n_states(&self) -> usize { 1 }
///     fn n_params(&self) -> usize { 1 }
///     fn params(&self) -> &[f64] { std::slice::from_ref(&self.k) }
///     fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
///         dy[0] = -p[0] * y[0];
///     }
///     fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) { jy[0] = -self.k; }
///     fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) { jp[0] = -y[0]; }
///     fn has_analytical_jacobian_y(&self) -> bool { true }
///     fn has_analytical_jacobian_p(&self) -> bool { true }
/// }
///
/// let r = solve_forward_sensitivity::<DoPri5, _, _>(
///     &Decay { k: 0.5 },
///     0.0, 2.0,
///     &[1.0],
///     &SolverOptions::default().rtol(1e-8).atol(1e-10),
/// ).unwrap();
///
/// let last = r.len() - 1;
/// let dy_dk = r.dyi_dpj(last, 0, 0);
/// let analytical = -r.t[last] * (-0.5 * r.t[last]).exp();
/// assert!((dy_dk - analytical).abs() < 1e-5);
/// ```
pub fn solve_forward_sensitivity<Sol, S, Sys>(
    system: &Sys,
    t0: S,
    tf: S,
    y0: &[S],
    options: &SolverOptions<S>,
) -> Result<SensitivityResult<S>, SolverError>
where
    Sol: Solver<S>,
    S: Scalar,
    Sys: ParametricOdeSystem<S>,
{
    let n = system.n_states();
    let np = system.n_params();
    assert_eq!(
        y0.len(),
        n,
        "solve_forward_sensitivity: y0.len() = {} but n_states = {}",
        y0.len(),
        n,
    );
    assert_eq!(
        system.params().len(),
        np,
        "solve_forward_sensitivity: params().len() = {} but n_params = {}",
        system.params().len(),
        np,
    );

    let aug = AugmentedSystem::new(system);
    let z0 = aug.initial_augmented(y0);
    let aug_dim = aug.augmented_dim();

    let aug_result = Sol::solve(&aug, t0, tf, &z0, options)?;

    if !aug_result.success {
        return Ok(SensitivityResult {
            t: Vec::new(),
            y: Vec::new(),
            sensitivity: Vec::new(),
            n_states: n,
            n_params: np,
            stats: aug_result.stats,
            success: false,
            message: aug_result.message,
        });
    }

    let n_times = aug_result.len();
    let mut t = Vec::with_capacity(n_times);
    let mut y = Vec::with_capacity(n_times * n);
    let mut sensitivity = Vec::with_capacity(n_times * n * np);

    for i in 0..n_times {
        t.push(aug_result.t[i]);
        let block_start = i * aug_dim;
        let state_block = &aug_result.y[block_start..block_start + n];
        let sens_block = &aug_result.y[block_start + n..block_start + aug_dim];
        y.extend_from_slice(state_block);
        sensitivity.extend_from_slice(sens_block);
    }

    Ok(SensitivityResult {
        t,
        y,
        sensitivity,
        n_states: n,
        n_params: np,
        stats: aug_result.stats,
        success: true,
        message: String::new(),
    })
}

/// Closure-shaped convenience wrapper around [`solve_forward_sensitivity`].
///
/// Wraps `rhs(t, y, p, dydt)` plus a parameter vector into an internal
/// [`ClosureSystem`] (FD-default Jacobians) and forwards. Suitable for
/// one-shot analyses, tests, and REPL-style scripts. For production usage
/// — particularly stiff problems where analytical Jacobians materially
/// change runtime — implement [`ParametricOdeSystem`] directly and call
/// [`solve_forward_sensitivity`].
///
/// # Example
///
/// ```
/// use numra_ode::{DoPri5, SolverOptions};
/// use numra_ode::sensitivity::solve_forward_sensitivity_with;
///
/// // dy/dt = -k * y, y(0) = 1, k = 0.5.
/// let r = solve_forward_sensitivity_with::<DoPri5, f64, _>(
///     |_t, y, p, dy| { dy[0] = -p[0] * y[0]; },
///     &[1.0],
///     &[0.5],
///     0.0, 2.0,
///     &SolverOptions::default().rtol(1e-8).atol(1e-10),
/// ).unwrap();
///
/// let last = r.len() - 1;
/// let dy_dk = r.dyi_dpj(last, 0, 0);
/// let analytical: f64 = -r.t[last] * (-0.5 * r.t[last]).exp();
/// assert!((dy_dk - analytical).abs() < 1e-4);
/// ```
pub fn solve_forward_sensitivity_with<Sol, S, F>(
    rhs: F,
    y0: &[S],
    params: &[S],
    t0: S,
    tf: S,
    options: &SolverOptions<S>,
) -> Result<SensitivityResult<S>, SolverError>
where
    Sol: Solver<S>,
    S: Scalar,
    F: Fn(S, &[S], &[S], &mut [S]),
{
    let system = ClosureSystem {
        rhs,
        params: params.to_vec(),
        n_states: y0.len(),
    };
    solve_forward_sensitivity::<Sol, S, _>(&system, t0, tf, y0, options)
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
        fn has_analytical_jacobian_y(&self) -> bool {
            true
        }
        fn has_analytical_jacobian_p(&self) -> bool {
            true
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
        fn has_analytical_jacobian_y(&self) -> bool {
            true
        }
        fn has_analytical_jacobian_p(&self) -> bool {
            true
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

    // ---------------------------------------------------------------
    // Solve entry-point sanity tests.
    //
    // The full solver-coverage matrix lives in
    // numra-ode/tests/sensitivity_regression.rs (commit 3).
    // ---------------------------------------------------------------

    use crate::DoPri5;
    use crate::SolverOptions;

    #[test]
    fn solve_forward_sensitivity_with_dopri5_matches_analytical_decay() {
        // dy/dt = -k y, y(0) = 1, k = 0.5.
        // Analytical: ∂y/∂k(t) = -t · exp(-k t).
        let result = solve_forward_sensitivity_with::<DoPri5, f64, _>(
            |_t, y, p, dy| {
                dy[0] = -p[0] * y[0];
            },
            &[1.0],
            &[0.5],
            0.0,
            2.0,
            &SolverOptions::default().rtol(1e-8).atol(1e-10),
        )
        .expect("solve failed");

        assert!(result.success);
        assert_eq!(result.n_states, 1);
        assert_eq!(result.n_params, 1);
        let last = result.len() - 1;
        let dy_dk = result.dyi_dpj(last, 0, 0);
        let t_last = result.t[last];
        let analytical = -t_last * (-0.5 * t_last).exp();
        assert!(
            (dy_dk - analytical).abs() < 1e-5,
            "computed {dy_dk}, analytical {analytical}, |err| = {}",
            (dy_dk - analytical).abs()
        );
    }

    #[test]
    fn solve_trait_form_matches_closure_form() {
        let trait_result = solve_forward_sensitivity::<DoPri5, _, _>(
            &ExpDecay { k: 0.5 },
            0.0,
            2.0,
            &[1.0],
            &SolverOptions::default().rtol(1e-9).atol(1e-12),
        )
        .expect("trait solve failed");

        let closure_result = solve_forward_sensitivity_with::<DoPri5, f64, _>(
            |_t, y, p, dy| {
                dy[0] = -p[0] * y[0];
            },
            &[1.0],
            &[0.5],
            0.0,
            2.0,
            &SolverOptions::default().rtol(1e-9).atol(1e-12),
        )
        .expect("closure solve failed");

        // Trait form has analytical Jacobians; closure form falls back to FD.
        // Both must converge to the same answer at the final time, modulo FD
        // noise.
        let n_t = trait_result.len();
        let n_c = closure_result.len();
        let trait_final = trait_result.dyi_dpj(n_t - 1, 0, 0);
        let closure_final = closure_result.dyi_dpj(n_c - 1, 0, 0);
        assert!(
            (trait_final - closure_final).abs() < 1e-5,
            "trait {trait_final} vs closure {closure_final}, |err| = {}",
            (trait_final - closure_final).abs()
        );
    }

    #[test]
    fn solve_propagates_solver_failure_as_unsuccessful_result() {
        // Forcing max_steps = 1 against a multi-step problem makes the
        // underlying solver return success = false (or an error). We assert
        // that solve_forward_sensitivity surfaces this without panicking.
        let opts = SolverOptions::default().rtol(1e-9).atol(1e-12).max_steps(1);

        let outcome = solve_forward_sensitivity_with::<DoPri5, f64, _>(
            |_t, y, p, dy| {
                dy[0] = -p[0] * y[0];
            },
            &[1.0],
            &[0.5],
            0.0,
            10.0,
            &opts,
        );

        match outcome {
            Ok(r) => {
                assert!(!r.success, "expected unsuccessful result, got success");
                assert!(r.t.is_empty());
                assert!(r.y.is_empty());
                assert!(r.sensitivity.is_empty());
            }
            Err(_) => {
                // Some solvers return Err directly for max_steps; that's also acceptable.
            }
        }
    }
}
