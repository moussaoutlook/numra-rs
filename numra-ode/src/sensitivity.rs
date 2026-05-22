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
//! Esdirk\*). The augmented Jacobian is `block_diag(J_y, …, J_y)` —
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
use crate::problem::{OdeProblem, OdeSystem};
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
    ///
    /// # Debug-build safety net
    ///
    /// To catch the silent-misconfiguration case (override the method,
    /// forget the flag), `AugmentedSystem` runs a one-time consistency
    /// check on the first RHS call: it evaluates [`Self::jacobian_y`]
    /// (which dispatches to the user's override or the FD default) and
    /// compares against the inline FD result. If they disagree by more
    /// than a Frobenius-norm relative threshold of `1e-3`, the system
    /// panics in debug builds with a message naming the missing flag
    /// override. Release builds skip the check entirely (zero overhead).
    /// This is a safety net, not a contract — fixing the panic by
    /// returning `true` here is the canonical resolution.
    fn has_analytical_jacobian_y(&self) -> bool {
        false
    }

    /// Returns `true` iff [`Self::jacobian_p`] has been overridden with an
    /// analytical implementation. Default: `false`. See
    /// [`Self::has_analytical_jacobian_y`] for the rationale, contract,
    /// and debug-build consistency check.
    fn has_analytical_jacobian_p(&self) -> bool {
        false
    }

    // ------------------------------------------------------------------
    // Mass-matrix surface (F-IC-SENS-MASS-SURFACE, 0.1.5, Foundation Spec
    // §6 #14 / §3.8). Mirrors [`OdeSystem`]'s mass-matrix surface so that
    // parameter-sensitivity hosts can declare DAE structure that the
    // augmented `OdeSystem`-impl on [`AugmentedSystem`] then lifts onto
    // the augmented `(N · (1 + N_s))`-dimensional state.
    //
    // Defaults match [`OdeSystem`]'s defaults exactly: `is_autonomous =
    // false`, `has_mass_matrix = false`, `mass_matrix` fills row-major
    // identity, `is_singular_mass = false`, `algebraic_indices = empty`.
    // Every pre-0.1.5 implementor compiles unchanged.
    // ------------------------------------------------------------------

    /// Is the system autonomous? (f does not depend on t explicitly)
    ///
    /// Default: `false`. Override to `true` when `f(t, y, p) = f(y, p)`
    /// independent of `t`. `AugmentedSystem` forwards this from the host
    /// so the augmented system inherits the host's time-dependence
    /// structure — variational equations inherit autonomy from the
    /// state equation.
    fn is_autonomous(&self) -> bool {
        false
    }

    /// Does this system have a mass matrix?
    ///
    /// Default: `false` (the standard parameterised ODE `dy/dt = f(t, y, p)`).
    /// Override to `true` for DAE-shaped parameterised systems
    /// `M · y' = f(t, y, p)` where `M` is non-identity or singular.
    /// See [`OdeSystem::has_mass_matrix`] for the equivalent surface on
    /// the unparameterised side.
    fn has_mass_matrix(&self) -> bool {
        false
    }

    /// Get the mass matrix `M` for the DAE: `M · y' = f(t, y, p)`.
    ///
    /// Default returns identity (standard ODE), regardless of what
    /// [`Self::has_mass_matrix`] reports — callers should consult
    /// [`Self::has_mass_matrix`] before relying on this output.
    ///
    /// Layout: row-major, `mass[i * n + j] = M[i, j]`, length `n²` where
    /// `n = self.n_states()`. Mirrors [`OdeSystem::mass_matrix`].
    fn mass_matrix(&self, mass: &mut [S]) {
        let n = self.n_states();
        for i in 0..n {
            for j in 0..n {
                mass[i * n + j] = if i == j { S::ONE } else { S::ZERO };
            }
        }
    }

    /// Is the mass matrix singular? (i.e., is this a DAE?)
    ///
    /// Default: `false`. Override to `true` for semi-explicit DAEs where
    /// one or more rows of `M` are zero (algebraic constraints).
    fn is_singular_mass(&self) -> bool {
        false
    }

    /// Indices of algebraic variables (rows `i` where `M[i, i] = 0`).
    ///
    /// Default: empty (all rows differential). Override for DAE systems
    /// to enumerate the algebraic rows so downstream solvers can apply
    /// row-aware error scaling and the augmented-system lift can map
    /// each host algebraic index `i` to its `(N_s + 1)` copies on the
    /// augmented diagonal.
    fn algebraic_indices(&self) -> Vec<usize> {
        Vec::new()
    }
}

/// Lift the host's `n × n` mass matrix `M` block-diagonally onto the
/// augmented system's `(n + n·m) × (n + n·m)` state, where `m =
/// n_params` is the number of sensitivity columns.
///
/// # Variational-equation derivation
///
/// The host DAE is `M · y' = f(t, y, p)`. Differentiating each side with
/// respect to a parameter `p_k` gives the variational equation
///
/// ```text
///   M · (∂y/∂p_k)' = J_y · (∂y/∂p_k) + J_p_{:,k}
/// ```
///
/// so each sensitivity column `∂y/∂p_k` lives in the same M-weighted
/// space as the state itself. Stacking the augmented state `z = [y;
/// vec(S)] ∈ ℝ^{n(1+m)}` (column-major sensitivity flattening, see
/// module-level layout conventions) and writing the augmented dynamics in
/// matrix form `M_aug · z' = F(t, z, p)` therefore requires
///
/// ```text
///   M_aug = block_diag(M, M, …, M)  with (m + 1) copies of M
/// ```
///
/// — one M for the state block, plus one M for each of the `m`
/// sensitivity-column blocks. Singular rows of `M` lift through to every
/// block (see `lift_algebraic_indices_for_augmented`).
///
/// # Layout
///
/// `host_m`: row-major, length `n²`, `host_m[i * n + j] = M[i, j]`.
/// `aug_mass`: row-major, length `(n * (1 + m))²`, filled by this
/// function. All cross-block entries are zero (block-diagonal).
fn lift_mass_matrix_for_augmented<S: Scalar>(host_m: &[S], n: usize, m: usize, aug_mass: &mut [S]) {
    let dim = n * (1 + m);
    debug_assert_eq!(host_m.len(), n * n);
    debug_assert_eq!(aug_mass.len(), dim * dim);
    // Zero the augmented matrix first (all off-block entries are zero).
    for slot in aug_mass.iter_mut() {
        *slot = S::ZERO;
    }
    // Place `(m + 1)` copies of M on the diagonal blocks. Block `b`
    // occupies rows `[b*n, (b+1)*n)` and columns `[b*n, (b+1)*n)`.
    for b in 0..=m {
        let row0 = b * n;
        let col0 = b * n;
        for i in 0..n {
            for j in 0..n {
                aug_mass[(row0 + i) * dim + (col0 + j)] = host_m[i * n + j];
            }
        }
    }
}

/// Lift the host's algebraic indices onto the augmented system's `(N_s
/// + 1)` blocks.
///
/// The host's algebraic structure (rows `i` of `M` that are entirely
/// zero) propagates to every sensitivity column through the variational
/// equation: the row-`i` variational equation
/// `M[i,:] · (∂y/∂p_k)' = J_y[i,:] · (∂y/∂p_k) + J_p[i,k]` is algebraic
/// in every block whenever it is algebraic in the host. So for each host
/// algebraic index `i`, the augmented algebraic indices are
/// `{i, n + i, 2n + i, …, m · n + i}` — one copy per block.
///
/// Result is sorted ascending to match the conventional `algebraic_indices`
/// output ordering.
fn lift_algebraic_indices_for_augmented(host_indices: &[usize], n: usize, m: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(host_indices.len() * (1 + m));
    for b in 0..=m {
        let offset = b * n;
        for &i in host_indices {
            out.push(offset + i);
        }
    }
    out.sort_unstable();
    out
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
    // Lifted host algebraic indices for the augmented `(N · (1 + N_s))`
    // state. Computed once at construction by `lift_algebraic_indices_for_
    // augmented` from `system.algebraic_indices()`. Algebraic indices are
    // assumed constant over the host's lifetime (the trait contract does
    // not require this explicitly, but every shipping implementor returns
    // a constant set, and a future foundation change that ever makes them
    // dynamic should revisit this cache).
    augmented_algebraic_indices: Vec<usize>,
    // Debug-only: tracks whether the first-call analytical-vs-FD
    // consistency check has run. See `check_jacobian_flags`.
    #[cfg(debug_assertions)]
    flag_check_done: std::cell::Cell<bool>,
}

impl<S: Scalar, Sys: ParametricOdeSystem<S>> AugmentedSystem<S, Sys> {
    /// Wrap a [`ParametricOdeSystem`] into an `OdeSystem`-shaped augmented
    /// system suitable for any [`crate::Solver`].
    pub fn new(system: Sys) -> Self {
        let n = system.n_states();
        let np = system.n_params();
        // Snapshot the host's algebraic indices once at construction and
        // lift them onto the augmented state's `(N_s + 1)` blocks. See
        // `lift_algebraic_indices_for_augmented` for the lift derivation
        // and the host-stability assumption.
        let augmented_algebraic_indices =
            lift_algebraic_indices_for_augmented(&system.algebraic_indices(), n, np);
        Self {
            system,
            jy_scratch: std::cell::RefCell::new(vec![S::ZERO; n * n]),
            jp_scratch: std::cell::RefCell::new(vec![S::ZERO; n * np]),
            fd_f0: std::cell::RefCell::new(vec![S::ZERO; n]),
            fd_f1: std::cell::RefCell::new(vec![S::ZERO; n]),
            fd_y_pert: std::cell::RefCell::new(vec![S::ZERO; n]),
            fd_p_pert: std::cell::RefCell::new(vec![S::ZERO; np]),
            augmented_algebraic_indices,
            #[cfg(debug_assertions)]
            flag_check_done: std::cell::Cell::new(false),
        }
    }

    /// Debug-build consistency check for the analytical-Jacobian flags.
    ///
    /// Catches the silent-misconfiguration class of bug where a user
    /// implements `jacobian_y` (or `jacobian_p`) analytically but forgets
    /// to override `has_analytical_jacobian_y` (or `_p`) — in which case
    /// the augmented hot path bypasses the analytical override entirely
    /// and runs FD instead. The user sees "Numra is slow on my problem"
    /// with no diagnostic path.
    ///
    /// On the first `rhs` call (and only then), if a flag returns `false`,
    /// we evaluate `system.jacobian_*` (which dispatches to the user's
    /// override or the trait's FD default) and compare against the inline
    /// FD result. If the user did not override the method, both calls
    /// compute FD and agree to floating-point noise. If the user did
    /// override with analytical, the two will typically disagree well
    /// beyond the relative threshold — and we panic with a clear message.
    ///
    /// The threshold is `1e-3` relative (Frobenius). FD is only accurate
    /// to `~sqrt(eps_mach) ≈ 1.5e-8`, so 1e-3 is loose enough to absorb
    /// honest FD-vs-analytical disagreement at the initial state without
    /// false positives, while still catching the "forgot the flag"
    /// scenario where the analytical and FD results differ by O(1).
    ///
    /// Wrapped in `#[cfg(debug_assertions)]` — release builds pay zero.
    #[cfg(debug_assertions)]
    fn check_jacobian_flags(&self, t: S, y: &[S]) {
        let n = self.system.n_states();
        let np = self.system.n_params();
        let threshold = S::from_f64(1e-3);

        if !self.system.has_analytical_jacobian_y() {
            let mut user_jy = vec![S::ZERO; n * n];
            let mut fd_jy = vec![S::ZERO; n * n];
            self.system.jacobian_y(t, y, &mut user_jy);
            self.fd_jacobian_y_inline(t, y, &mut fd_jy);
            if jacobians_differ_significantly(&user_jy, &fd_jy, threshold) {
                panic!(
                    "ParametricOdeSystem implements `jacobian_y` analytically \
                     but `has_analytical_jacobian_y()` returns `false`; the \
                     analytical implementation is being ignored and `AugmentedSystem` \
                     is running finite differences instead. Override \
                     `has_analytical_jacobian_y` to return `true`. \
                     (This check is debug-build only; release builds will \
                     silently use FD.)"
                );
            }
        }

        if !self.system.has_analytical_jacobian_p() && np > 0 {
            let mut user_jp = vec![S::ZERO; n * np];
            let mut fd_jp = vec![S::ZERO; n * np];
            self.system.jacobian_p(t, y, &mut user_jp);
            self.fd_jacobian_p_inline(t, y, &mut fd_jp);
            if jacobians_differ_significantly(&user_jp, &fd_jp, threshold) {
                panic!(
                    "ParametricOdeSystem implements `jacobian_p` analytically \
                     but `has_analytical_jacobian_p()` returns `false`; the \
                     analytical implementation is being ignored and `AugmentedSystem` \
                     is running finite differences instead. Override \
                     `has_analytical_jacobian_p` to return `true`. \
                     (This check is debug-build only; release builds will \
                     silently use FD.)"
                );
            }
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

/// Frobenius-norm relative comparison: returns `true` if
/// `||a - b||_F / max(||b||_F, 1) > threshold`. Used by the debug-build
/// analytical-vs-FD Jacobian flag check.
#[cfg(debug_assertions)]
fn jacobians_differ_significantly<S: Scalar>(a: &[S], b: &[S], threshold: S) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut diff_sq = S::ZERO;
    let mut b_sq = S::ZERO;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        diff_sq = diff_sq + d * d;
        b_sq = b_sq + b[i] * b[i];
    }
    let denom = b_sq.sqrt().max(S::ONE);
    let rel = diff_sq.sqrt() / denom;
    rel > threshold
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

        // Debug-only one-time consistency check: analytical-Jacobian flags
        // vs the user's actual `jacobian_*` override. See
        // `check_jacobian_flags` for rationale.
        #[cfg(debug_assertions)]
        if !self.flag_check_done.get() {
            self.flag_check_done.set(true);
            self.check_jacobian_flags(t, y);
        }

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

    // ------------------------------------------------------------------
    // Mass-matrix + autonomy surface (F-IC-SENS-MASS-SURFACE, 0.1.5,
    // Foundation Spec §6 #14). The pre-0.1.5 `impl OdeSystem for
    // AugmentedSystem` inherited the trait defaults for these five
    // methods — silently dropping the wrapped `ParametricOdeSystem`'s
    // mass-matrix and autonomy declarations onto the augmented state.
    // The downstream solvers (Radau5, BDF) read these methods directly
    // (radau5.rs:234,237; bdf.rs:470,472), so the silent defaults
    // produced M-blind variational integrations for any host with
    // non-identity or singular M. The five overrides below delegate /
    // lift from the host. See `lift_mass_matrix_for_augmented` and
    // `lift_algebraic_indices_for_augmented` for the block-diagonal
    // derivation.
    // ------------------------------------------------------------------

    fn is_autonomous(&self) -> bool {
        // Variational equations inherit the host's time-dependence
        // structure — if `f(t, y, p)` is t-independent, so is the
        // augmented RHS (whose sensitivity columns depend on `J_y`,
        // `J_p`, themselves evaluated at the host's `(t, y, p)`).
        self.system.is_autonomous()
    }

    fn has_mass_matrix(&self) -> bool {
        // Direct delegation. Downstream solvers (Radau5, BDF) gate the
        // M-aware Newton system on this flag; a host that declares
        // `has_mass_matrix = false` must continue to take the identity-M
        // fast path through the augmented integration.
        self.system.has_mass_matrix()
    }

    /// Fill the augmented mass matrix.
    ///
    /// Two distinct cases (correctness, not defense): the
    /// `has_mass_matrix() = false` branch returns the full augmented
    /// identity rather than the host's identity-lifted result, because
    /// downstream solvers dispatch on `has_mass_matrix()` separately
    /// from `mass_matrix()` — a host that explicitly declared
    /// `has_mass_matrix() = true` with `M = I` is semantically distinct
    /// from a host that declared `has_mass_matrix() = false`, and the
    /// branch preserves that distinction. The M-aware branch lifts the
    /// host's `N × N` `M` block-diagonally over `(N_s + 1)` copies via
    /// `lift_mass_matrix_for_augmented`.
    fn mass_matrix(&self, mass: &mut [S]) {
        let n = self.system.n_states();
        let np = self.system.n_params();
        let dim = n * (1 + np);
        if !self.system.has_mass_matrix() {
            // Augmented identity for the full `dim × dim` matrix.
            // Semantically distinct from the M-aware identity branch
            // (see the rustdoc above).
            for i in 0..dim {
                for j in 0..dim {
                    mass[i * dim + j] = if i == j { S::ONE } else { S::ZERO };
                }
            }
            return;
        }
        // Borrow the host's `N × N` M into local scratch and lift onto
        // the augmented diagonal. Local scratch (not a `RefCell` on the
        // struct) because this path is cold relative to `rhs`/`jacobian`
        // — solvers call `mass_matrix` once at the start of integration,
        // not every step.
        let mut host_m = vec![S::ZERO; n * n];
        self.system.mass_matrix(&mut host_m);
        lift_mass_matrix_for_augmented(&host_m, n, np, mass);
    }

    fn is_singular_mass(&self) -> bool {
        // Direct delegation. Singularity lifts through every block (a
        // zero row of host M is zero in every augmented diagonal block).
        self.system.is_singular_mass()
    }

    fn algebraic_indices(&self) -> Vec<usize> {
        // Cached at construction in `new()`; see the field doc on
        // `augmented_algebraic_indices` for the host-stability
        // assumption.
        self.augmented_algebraic_indices.clone()
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
    fn has_analytical_jacobian_y(&self) -> bool {
        (*self).has_analytical_jacobian_y()
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        (*self).has_analytical_jacobian_p()
    }
    // Mass-matrix surface (F-IC-SENS-MASS-SURFACE, 0.1.5). Forwards from
    // the referenced parametric system; otherwise `&Sys` would silently
    // inherit the trait defaults and drop the host's M declaration —
    // exactly the inward-direction boundary gap this PR exists to close
    // (see `feedback_wrapper_boundary_symmetry`).
    fn is_autonomous(&self) -> bool {
        (*self).is_autonomous()
    }
    fn has_mass_matrix(&self) -> bool {
        (*self).has_mass_matrix()
    }
    fn mass_matrix(&self, mass: &mut [S]) {
        (*self).mass_matrix(mass)
    }
    fn is_singular_mass(&self) -> bool {
        (*self).is_singular_mass()
    }
    fn algebraic_indices(&self) -> Vec<usize> {
        (*self).algebraic_indices()
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

// ============================================================================
// Initial-condition sensitivity (state-transition matrix)
//
// Specialises the variational machinery above to the IC case: integrates
// `Ṡ = J_y · S` with `S(t₀) = I_N`, producing `Φ(t) = ∂y(t)/∂y₀` —
// the state-transition matrix of the linearised flow. Reuses
// `AugmentedSystem` and `solve_forward_sensitivity` unchanged; no fork.
//
// The user-facing API takes an `OdeSystem<S>` (or RHS closure) and returns
// a `StateTransitionResult<S>` whose accessors speak Φ(t)-vocabulary only.
// Internally a private `IcAsParametric` wrapper reinterprets the plain
// system as a `ParametricOdeSystem` with `n_params := n_states`,
// `J_p ≡ 0`, and `S(t₀) = I`. The required dummy `params()` slice is the
// only seam, encapsulated behind the wrapper and pinned by the test
// `ic_dummy_params_are_unread` below.
//
// Invariant **IC-NO-PARAM-READ.** `IcAsParametric::rhs_with_params(t,y,p,dy)`
// never reads `p`. The wrapped `OdeSystem::rhs(t,y,dy)` has no `p` argument
// and cannot. Therefore `∂f/∂p ≡ 0` is mathematically exact (not "small in
// some norm"), and `IcAsParametric::jacobian_p` writing zeros with
// `has_analytical_jacobian_p() = true` is a correct analytical declaration
// — `AugmentedSystem`'s debug consistency check at `sensitivity.rs:373` is
// truthfully suppressed by this flag, not silenced.
// ============================================================================

/// Private wrapper: makes an `OdeSystem<S>` look like a `ParametricOdeSystem<S>`
/// whose "parameters" are seeded so the forward-sensitivity flow computes
/// `Φ(t) = ∂y(t)/∂y₀` instead of `∂y(t)/∂p`.
struct IcAsParametric<'a, S: Scalar, Sys: OdeSystem<S> + ?Sized> {
    inner: &'a Sys,
    /// Dummy parameter slice of length `n_states`. Length-only requirement
    /// from `ParametricOdeSystem` (asserted at solve entry, line 852–858).
    /// Provably unread by `rhs_with_params`; pinned by
    /// `ic_dummy_params_are_unread`.
    dummy_p: Vec<S>,
}

impl<'a, S: Scalar, Sys: OdeSystem<S> + ?Sized> ParametricOdeSystem<S>
    for IcAsParametric<'a, S, Sys>
{
    fn n_states(&self) -> usize {
        self.inner.dim()
    }
    fn n_params(&self) -> usize {
        self.inner.dim()
    }
    fn params(&self) -> &[S] {
        &self.dummy_p
    }

    fn rhs_with_params(&self, t: S, y: &[S], _p: &[S], dy: &mut [S]) {
        // IC-NO-PARAM-READ: `_p` is provably unread; `inner.rhs` has no
        // `p` argument and cannot consume it.
        self.inner.rhs(t, y, dy);
    }

    fn jacobian_y(&self, t: S, y: &[S], jy: &mut [S]) {
        // Forwards to the inner system's Jacobian — analytical override if
        // the user supplied one (e.g. `AutodiffJacobianSystem` from the
        // `autodiff` feature), or `OdeSystem`'s FD default otherwise.
        self.inner.jacobian(t, y, jy);
    }

    fn jacobian_p(&self, _t: S, _y: &[S], jp: &mut [S]) {
        // J_p ≡ 0 analytically (IC-NO-PARAM-READ ⇒ ∂f/∂p ≡ 0 exactly).
        for slot in jp.iter_mut() {
            *slot = S::ZERO;
        }
    }

    fn initial_sensitivity(&self, _y0: &[S], s0: &mut [S]) {
        // Column-major identity: s0[k*N + i] = δ_{ik}.
        let n = self.inner.dim();
        for slot in s0.iter_mut() {
            *slot = S::ZERO;
        }
        for k in 0..n {
            s0[k * n + k] = S::ONE;
        }
    }

    fn has_analytical_jacobian_y(&self) -> bool {
        // Claim analytical so AugmentedSystem honours the forwarded
        // `inner.jacobian` (which may itself be analytical or FD-default).
        // The debug consistency check is then irrelevant for this flag.
        true
    }

    fn has_analytical_jacobian_p(&self) -> bool {
        // Truthful: J_p ≡ 0 is analytically exact per IC-NO-PARAM-READ.
        true
    }

    // ------------------------------------------------------------------
    // Mass-matrix + autonomy forwarding (F-IC-SENS-MASS-SURFACE, 0.1.5,
    // Foundation Spec §6 #14). Closes the inward-direction boundary gap
    // F-IC-SENS shipped with: the 0.1.4 wrapper inherited the
    // `ParametricOdeSystem` defaults for these five methods (returning
    // `has_mass_matrix = false`, identity M, etc.), silently dropping
    // the wrapped `OdeSystem`'s declarations on the way into the
    // augmented integration. See `feedback_wrapper_boundary_symmetry`
    // for the both-direction boundary discipline this restores.
    // ------------------------------------------------------------------

    fn is_autonomous(&self) -> bool {
        self.inner.is_autonomous()
    }

    fn has_mass_matrix(&self) -> bool {
        self.inner.has_mass_matrix()
    }

    fn mass_matrix(&self, mass: &mut [S]) {
        self.inner.mass_matrix(mass);
    }

    fn is_singular_mass(&self) -> bool {
        self.inner.is_singular_mass()
    }

    fn algebraic_indices(&self) -> Vec<usize> {
        self.inner.algebraic_indices()
    }
}

/// Result of an initial-condition sensitivity solve.
///
/// Carries the state trajectory `y(t)` and the state-transition matrix
/// `Φ(t) = ∂y(t) / ∂y₀ ∈ ℝ^{N×N}` of the linearised flow, evaluated at every
/// output time the underlying solver produced.
///
/// # Φ(t) layout
///
/// Column-major within each per-time block: `phi(i)[k*N + j] = Φ(t_i)_{j,k}
/// = ∂y_j(t_i) / ∂y_{0,k}`. The column-major flattening makes the j-th
/// column (the trajectory's response at time `t_i` to a unit perturbation
/// of `y_{0,k}`) a free contiguous slice — see [`Self::phi_column`].
///
/// # Vocabulary
///
/// All accessors speak state-transition-matrix vocabulary
/// (`Φ`, `∂y_i/∂y_{0,j}`, monodromy). The type holds the underlying
/// [`SensitivityResult`] in a private field; it does **not** `Deref` to it
/// and does **not** re-export the parameter-shaped accessors.
#[derive(Clone, Debug)]
pub struct StateTransitionResult<S: Scalar> {
    inner: SensitivityResult<S>,
}

impl<S: Scalar> StateTransitionResult<S> {
    /// Output time points (length `n_times`). `t()[i]` is the i-th output
    /// time; `phi(i)` is `Φ(t()[i])`.
    pub fn t(&self) -> &[S] {
        &self.inner.t
    }

    /// Number of output time points.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` iff the result has no output time points.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Underlying state dimension `N`.
    pub fn dim(&self) -> usize {
        self.inner.n_states
    }

    /// State `y(t_i)` at output index `i` (length `N`).
    pub fn y(&self, i: usize) -> &[S] {
        self.inner.y_at(i)
    }

    /// State at the final output time, length `N`. Empty slice when the
    /// result has no time points.
    pub fn final_y(&self) -> &[S] {
        self.inner.final_state()
    }

    /// Full state-transition matrix `Φ(t_i) ∈ ℝ^{N×N}` at output index `i`,
    /// column-major. `phi(i)[k*N + j] = Φ(t_i)_{j,k} = ∂y_j(t_i)/∂y_{0,k}`.
    pub fn phi(&self, i: usize) -> &[S] {
        self.inner.sensitivity_at(i)
    }

    /// Single entry `Φ(t_i)_{row, col} = ∂y_row(t_i) / ∂y_{0, col}`.
    pub fn phi_ij(&self, i: usize, row: usize, col: usize) -> S {
        self.inner.dyi_dpj(i, row, col)
    }

    /// Contiguous slice `Φ(t_i)_{:, col}` of length `N` — the trajectory's
    /// response at `t_i` to a unit perturbation of `y_{0, col}`. Free slice
    /// thanks to the column-major flattening.
    pub fn phi_column(&self, i: usize, col: usize) -> &[S] {
        self.inner.sensitivity_for_param(i, col)
    }

    /// `Φ(t_f)`, the state-transition matrix at the final output time.
    /// When `(t_f − t_0)` equals one period of a periodic orbit this is
    /// the *monodromy matrix*, whose eigenvalues are the orbit's
    /// characteristic (Floquet) multipliers. The type does not enforce
    /// periodicity; it only names the natural use. Empty slice when the
    /// result has no time points.
    pub fn final_phi(&self) -> &[S] {
        self.inner.final_sensitivity()
    }

    /// `true` iff the underlying integration succeeded.
    pub fn success(&self) -> bool {
        self.inner.success
    }

    /// Solver diagnostic message (empty on success).
    pub fn message(&self) -> &str {
        &self.inner.message
    }

    /// Solver statistics from the underlying integration.
    pub fn stats(&self) -> &SolverStats {
        &self.inner.stats
    }
}

/// Compute the state-transition matrix `Φ(t) = ∂y(t) / ∂y₀` of the
/// linearised flow of `dy/dt = f(t, y)`, using any [`Solver`].
///
/// Integrates the variational equations `Ṡ = J_y · S` with seed
/// `S(t₀) = I_N` over the same [`AugmentedSystem`] used for parameter
/// sensitivity, then extracts the state-transition trajectory.
///
/// The user supplies only the ODE system; the IC seeding (identity initial
/// sensitivity, zero parameter forcing) is performed internally. `J_y` is
/// obtained from [`OdeSystem::jacobian`] — the user's analytical override
/// if any, the trait's forward-FD default otherwise. For an
/// exact-to-round-off `J_y` from autodiff, wrap the system in
/// `AutodiffJacobianSystem` (behind the `autodiff` feature).
///
/// # Example: scalar exponential decay
///
/// `dy/dt = -k y` has closed-form `Φ(t) = exp(-k t)`.
///
/// ```
/// use numra_ode::sensitivity::solve_initial_condition_sensitivity_with;
/// use numra_ode::{DoPri5, SolverOptions};
///
/// let k = 0.5_f64;
/// let result = solve_initial_condition_sensitivity_with::<DoPri5, f64, _>(
///     move |_t, y, dy| { dy[0] = -k * y[0]; },
///     &[1.0],
///     0.0, 2.0,
///     &SolverOptions::default().rtol(1e-9).atol(1e-12),
/// ).unwrap();
///
/// let last = result.len() - 1;
/// let phi = result.phi_ij(last, 0, 0);
/// let analytical = (-k * result.t()[last]).exp();
/// assert!((phi - analytical).abs() < 1e-7);
/// ```
pub fn solve_initial_condition_sensitivity<Sol, S, Sys>(
    system: &Sys,
    t0: S,
    tf: S,
    y0: &[S],
    options: &SolverOptions<S>,
) -> Result<StateTransitionResult<S>, SolverError>
where
    Sol: Solver<S>,
    S: Scalar,
    Sys: OdeSystem<S> + ?Sized,
{
    let n = system.dim();
    assert_eq!(
        y0.len(),
        n,
        "solve_initial_condition_sensitivity: y0.len() = {} but dim() = {}",
        y0.len(),
        n,
    );

    let wrapper = IcAsParametric {
        inner: system,
        dummy_p: vec![S::ZERO; n],
    };
    let inner = solve_forward_sensitivity::<Sol, S, _>(&wrapper, t0, tf, y0, options)?;
    Ok(StateTransitionResult { inner })
}

/// Closure-shaped convenience wrapper around
/// [`solve_initial_condition_sensitivity`].
///
/// Wraps `rhs(t, y, dydt)` into an internal [`OdeProblem`] (forward-FD
/// Jacobian via the [`OdeSystem`] default) and forwards. Suitable for
/// one-shot analyses and REPL-style scripts. For an analytical `J_y` or an
/// autodiff-derived `J_y`, implement [`OdeSystem`] directly (with a
/// `jacobian` override or by wrapping in `AutodiffJacobianSystem`)
/// and call [`solve_initial_condition_sensitivity`].
pub fn solve_initial_condition_sensitivity_with<Sol, S, F>(
    rhs: F,
    y0: &[S],
    t0: S,
    tf: S,
    options: &SolverOptions<S>,
) -> Result<StateTransitionResult<S>, SolverError>
where
    Sol: Solver<S>,
    S: Scalar,
    F: Fn(S, &[S], &mut [S]),
{
    let problem = OdeProblem::new(rhs, t0, tf, y0.to_vec());
    solve_initial_condition_sensitivity::<Sol, S, _>(&problem, t0, tf, y0, options)
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

    // ====================================================================
    // F-IC-SENS-MASS-SURFACE Test 6 — structural: IcAsParametric forwards
    // the wrapped OdeSystem's mass-matrix surface and is_autonomous.
    //
    // Closes the inward-direction structural check of the boundary-
    // symmetry pair. Lives here (not in tests/) because IcAsParametric
    // is private. Paired with Test 5 in
    // `numra-ode/tests/ic_sensitivity_mass_matrix.rs::augmented_system_
    // lifts_mass_surface_block_diagonally`, which covers the
    // AugmentedSystem (outward-from-wrapper) lift on the same data.
    // ====================================================================

    struct MockMassOde;

    impl OdeSystem<f64> for MockMassOde {
        fn dim(&self) -> usize {
            3
        }
        fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
            for slot in dy.iter_mut() {
                *slot = 0.0;
            }
        }
        fn is_autonomous(&self) -> bool {
            true
        }
        fn has_mass_matrix(&self) -> bool {
            true
        }
        fn mass_matrix(&self, m: &mut [f64]) {
            // diag(2, 3, 0) row-major.
            for slot in m.iter_mut() {
                *slot = 0.0;
            }
            m[0 * 3 + 0] = 2.0;
            m[1 * 3 + 1] = 3.0;
            m[2 * 3 + 2] = 0.0;
        }
        fn is_singular_mass(&self) -> bool {
            true
        }
        fn algebraic_indices(&self) -> Vec<usize> {
            vec![2]
        }
    }

    #[test]
    fn ic_as_parametric_forwards_mass_surface() {
        let host = MockMassOde;
        let n = host.dim();
        let wrapper = IcAsParametric {
            inner: &host,
            dummy_p: vec![0.0; n],
        };

        // Inward-direction boundary check: every method the host
        // declares on `OdeSystem`'s M surface (and `is_autonomous`)
        // must round-trip through the `ParametricOdeSystem` impl.
        assert!(
            wrapper.is_autonomous(),
            "is_autonomous not forwarded by IcAsParametric"
        );
        assert!(
            wrapper.has_mass_matrix(),
            "has_mass_matrix not forwarded by IcAsParametric"
        );
        assert!(wrapper.is_singular_mass(), "is_singular_mass not forwarded");
        assert_eq!(
            wrapper.algebraic_indices(),
            vec![2],
            "algebraic_indices not forwarded by IcAsParametric",
        );

        let mut m = vec![0.0_f64; n * n];
        wrapper.mass_matrix(&mut m);
        assert_eq!(m[0 * 3 + 0], 2.0, "mass_matrix[0,0] not forwarded");
        assert_eq!(m[1 * 3 + 1], 3.0, "mass_matrix[1,1] not forwarded");
        assert_eq!(
            m[2 * 3 + 2],
            0.0,
            "mass_matrix[2,2] (algebraic) not forwarded"
        );
        // Off-diagonal entries must remain zero (the host M is diagonal).
        assert_eq!(m[0 * 3 + 1], 0.0);
        assert_eq!(m[1 * 3 + 2], 0.0);
    }
}
