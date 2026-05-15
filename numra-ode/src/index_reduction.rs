//! DAE index analysis and automatic index reduction.
//!
//! This module implements the Pantelides algorithm for structural index analysis
//! of Differential-Algebraic Equation (DAE) systems, and automatic index reduction
//! from higher-index (index-2, index-3) to index-1 via constraint differentiation.
//!
//! # Background
//!
//! A DAE in semi-explicit form is:
//! ```text
//! y' = f(t, y, z)     (differential equations)
//! 0  = g(t, y, z)     (algebraic constraints)
//! ```
//!
//! The **differential index** measures how many times the algebraic constraints
//! must be differentiated before the system can be solved as an ODE. Index-1 DAEs
//! can be solved directly by existing BDF/Radau solvers. Higher-index systems
//! need reduction first.
//!
//! # Index Reduction Strategy
//!
//! 1. **Structural analysis** via the Pantelides algorithm determines which
//!    equations need differentiation and how many times.
//! 2. **Constraint differentiation** generates the time derivatives of algebraic
//!    constraints, introducing new variables for the derivatives.
//! 3. The result is an augmented index-1 system solvable by standard methods.
//!
//! # Example: Simple Pendulum (Index-3 → Index-1)
//!
//! ```text
//! Original (index-3):
//!   x'' = -λx           y'' = -λy - g
//!   x² + y² = L²
//!
//! After reduction:
//!   x' = vx             y' = vy
//!   vx' = -λx           vy' = -λy - g
//!   0 = x*vx + y*vy                      (differentiated once: velocity constraint)
//!   0 = vx² + vy² + x*(-λx) + y*(-λy-g) - ... (differentiated twice, or use index-2 form)
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::OdeSystem;
use numra_core::Scalar;

// ============================================================================
// Structural Analysis Types
// ============================================================================

/// Result of structural DAE index analysis.
#[derive(Clone, Debug)]
pub struct DaeIndexInfo {
    /// Structural (Pantelides) index of the DAE system.
    /// - 0: pure ODE (no algebraic equations)
    /// - 1: index-1 DAE (algebraic constraints directly solvable)
    /// - 2: index-2 DAE (constraints need 1 differentiation)
    /// - 3: index-3 DAE (constraints need 2 differentiations)
    pub structural_index: usize,

    /// Number of hidden constraints discovered by the algorithm.
    pub n_hidden_constraints: usize,

    /// Schedule of differentiations: (equation_index, n_times_to_differentiate).
    ///
    /// Each entry says "equation `equation_index` must be differentiated
    /// `n_times_to_differentiate` times to reduce the index."
    pub differentiation_schedule: Vec<(usize, usize)>,

    /// Assignment from the Pantelides algorithm: equation `i` is matched to variable `assign[i]`.
    /// Only meaningful after a successful structural analysis.
    pub assignment: Vec<Option<usize>>,

    /// Number of differential variables in the original system.
    pub n_diff: usize,

    /// Number of algebraic variables in the original system.
    pub n_alg: usize,
}

/// Incidence structure for a DAE system.
///
/// Describes which variables appear in which equations, partitioned into
/// differential and algebraic variables.
#[derive(Clone, Debug)]
pub struct DaeStructure {
    /// Number of differential variables (y)
    pub n_diff: usize,
    /// Number of algebraic variables (z)
    pub n_alg: usize,
    /// Number of differential equations
    pub n_diff_eqs: usize,
    /// Number of algebraic equations (constraints)
    pub n_alg_eqs: usize,
    /// Incidence: for each equation i, the set of variable indices it depends on.
    /// Variable indices 0..n_diff are differential, n_diff..n_diff+n_alg are algebraic.
    pub incidence: Vec<Vec<usize>>,
}

impl DaeStructure {
    /// Total number of variables.
    pub fn n_vars(&self) -> usize {
        self.n_diff + self.n_alg
    }

    /// Total number of equations.
    pub fn n_eqs(&self) -> usize {
        self.n_diff_eqs + self.n_alg_eqs
    }
}

// ============================================================================
// Pantelides Algorithm (Structural Index Analysis)
// ============================================================================

/// Analyze the structural index of a DAE system.
///
/// Uses a constraint-based approach for semi-explicit DAEs:
/// - Differential equations `y_i' = f_i(t, y, z)` are "pre-matched" to their variables y_i
/// - Algebraic constraints `0 = g_j(t, y, z)` must be matched to algebraic variables z_j
/// - If a constraint depends only on differential variables (no algebraic ones),
///   it cannot be matched and must be differentiated → this signals index > 1
///
/// # Arguments
///
/// * `structure` - The incidence structure of the DAE
///
/// # Returns
///
/// A `DaeIndexInfo` with the structural index and differentiation schedule.
pub fn analyze_dae_index(structure: &DaeStructure) -> DaeIndexInfo {
    let n_eqs = structure.n_eqs();
    let _n_vars = structure.n_vars();
    let n_diff = structure.n_diff;
    let n_alg = structure.n_alg;
    let n_diff_eqs = structure.n_diff_eqs;
    let n_alg_eqs = structure.n_alg_eqs;

    // If no algebraic equations, it's a pure ODE (index 0)
    if n_alg_eqs == 0 {
        return DaeIndexInfo {
            structural_index: 0,
            n_hidden_constraints: 0,
            differentiation_schedule: Vec::new(),
            assignment: vec![None; n_eqs],
            n_diff,
            n_alg,
        };
    }

    // For semi-explicit DAEs, the index is determined by whether the algebraic
    // constraints can be "resolved" for the algebraic variables.
    //
    // The key criterion: for each algebraic constraint g_j(t, y, z) = 0,
    // check if it depends on at least one algebraic variable z_k that is not
    // yet "claimed" by another constraint.
    //
    // If a constraint depends only on differential variables (no algebraic ones),
    // it's a "hidden constraint" that must be differentiated (index >= 2).

    // Algebraic variable indices are n_diff..n_diff+n_alg
    let alg_var_start = n_diff;

    // Build bipartite matching: algebraic equations ↔ algebraic variables
    // This is a restricted matching (only algebraic vars are matchable targets)
    let mut alg_incidence: Vec<Vec<usize>> = Vec::new();
    for alg_eq_idx in 0..n_alg_eqs {
        let eq_idx = n_diff_eqs + alg_eq_idx;
        let vars = if eq_idx < structure.incidence.len() {
            &structure.incidence[eq_idx]
        } else {
            continue;
        };

        // Filter to only algebraic variables (remap to 0..n_alg)
        let alg_vars: Vec<usize> = vars
            .iter()
            .filter(|&&v| v >= alg_var_start && v < alg_var_start + n_alg)
            .map(|&v| v - alg_var_start)
            .collect();
        alg_incidence.push(alg_vars);
    }

    // Try to find a maximum matching in the algebraic bipartite graph
    let mut eq_to_var: Vec<Option<usize>> = vec![None; n_alg_eqs];
    let mut var_to_eq: Vec<Option<usize>> = vec![None; n_alg];

    for eq in 0..n_alg_eqs {
        let mut visited = vec![false; n_alg];
        augmenting_path_restricted(
            eq,
            &alg_incidence,
            &mut eq_to_var,
            &mut var_to_eq,
            &mut visited,
        );
    }

    // Count unmatched algebraic equations
    let mut diff_schedule: Vec<(usize, usize)> = Vec::new();
    let mut max_differentiations = 0usize;

    for alg_eq_idx in 0..n_alg_eqs {
        if eq_to_var[alg_eq_idx].is_none() {
            // This constraint could not be matched to any algebraic variable.
            // It must be differentiated (at least once).
            // For the structural index, each differentiation needed adds 1 to the index.
            diff_schedule.push((n_diff_eqs + alg_eq_idx, 1));
            max_differentiations = max_differentiations.max(1);
        }
    }

    // If any constraints needed differentiation, check if a second round is needed.
    // For index-3: differentiated constraints might also fail to match.
    // We do a simple iterative check: after differentiating, do the new constraints match?
    if !diff_schedule.is_empty() {
        // After differentiating, the new constraints depend on y' variables
        // (which are "known" from the differential equations) plus potentially
        // the algebraic variables. If the original constraint depended on NO
        // algebraic variables, its derivative likely depends on y' (matched by
        // diff eqs) but may also introduce new dependencies.
        //
        // For a simple index-2 detection, one round of differentiation suffices.
        // For index-3, we'd need to check the differentiated constraints too.
        // We handle up to index-3 with a second pass.

        // Augment: for each differentiated constraint, create a new "virtual" equation
        // that depends on the original differential variables' derivatives (new vars)
        // plus the original algebraic variables.
        let n_new_eqs = diff_schedule.len();
        let mut aug_alg_incidence: Vec<Vec<usize>> = alg_incidence.clone();

        // New "algebraic variables" for derivatives of diff vars appearing in constraints
        let mut n_aug_alg = n_alg;
        for &(orig_eq_idx, _) in &diff_schedule {
            // The differentiated constraint introduces dependencies on y'_j for each
            // differential variable y_j in the original constraint.
            // These y'_j are "known" from the differential equations, so the
            // differentiated constraint effectively gains new algebraic dependencies
            // (derivative vars become new algebraic vars in the augmented system).
            let orig_vars = if orig_eq_idx < structure.incidence.len() {
                &structure.incidence[orig_eq_idx]
            } else {
                continue;
            };

            let mut new_alg_vars: Vec<usize> = Vec::new();
            for &v in orig_vars {
                if v < n_diff {
                    // y'_v is a new variable
                    new_alg_vars.push(n_aug_alg);
                    n_aug_alg += 1;
                }
            }

            // Original algebraic dependencies still present
            let orig_alg: Vec<usize> = orig_vars
                .iter()
                .filter(|&&v| v >= alg_var_start && v < alg_var_start + n_alg)
                .map(|&v| v - alg_var_start)
                .collect();

            let mut combined = orig_alg;
            combined.extend(new_alg_vars);
            aug_alg_incidence.push(combined);
        }

        // Re-run matching on augmented system
        let total_alg_eqs = n_alg_eqs + n_new_eqs;
        let mut eq_to_var2: Vec<Option<usize>> = vec![None; total_alg_eqs];
        let mut var_to_eq2: Vec<Option<usize>> = vec![None; n_aug_alg];

        for eq in 0..total_alg_eqs {
            let mut visited = vec![false; n_aug_alg];
            augmenting_path_restricted(
                eq,
                &aug_alg_incidence,
                &mut eq_to_var2,
                &mut var_to_eq2,
                &mut visited,
            );
        }

        // Check if any of the NEW equations still couldn't match
        let mut second_round_unmatched = 0;
        for new_eq in n_alg_eqs..total_alg_eqs {
            if eq_to_var2[new_eq].is_none() {
                second_round_unmatched += 1;
            }
        }

        if second_round_unmatched > 0 {
            // Need second differentiation → index 3
            for entry in &mut diff_schedule {
                entry.1 += 1; // Each needs one more differentiation
            }
            max_differentiations += 1;
        }
    }

    let structural_index = if diff_schedule.is_empty() {
        1 // Index-1: all constraints matched to algebraic variables
    } else {
        max_differentiations + 1 // Index = 1 + max differentiations needed
    };

    let n_hidden = diff_schedule.iter().map(|&(_, n)| n).sum::<usize>();

    // Build full assignment
    let mut assignment: Vec<Option<usize>> = vec![None; n_eqs];
    // Differential equations matched to their own variables
    for i in 0..n_diff_eqs.min(n_diff) {
        assignment[i] = Some(i);
    }
    // Algebraic equations matched to algebraic variables
    for (alg_eq_idx, &matched_var) in eq_to_var.iter().enumerate() {
        if let Some(v) = matched_var {
            assignment[n_diff_eqs + alg_eq_idx] = Some(alg_var_start + v);
        }
    }

    DaeIndexInfo {
        structural_index,
        n_hidden_constraints: n_hidden,
        differentiation_schedule: diff_schedule,
        assignment,
        n_diff,
        n_alg,
    }
}

/// Find an augmenting path in the restricted bipartite graph (algebraic eqs ↔ algebraic vars).
fn augmenting_path_restricted(
    eq: usize,
    incidence: &[Vec<usize>],
    eq_to_var: &mut [Option<usize>],
    var_to_eq: &mut [Option<usize>],
    visited: &mut [bool],
) -> bool {
    if eq >= incidence.len() {
        return false;
    }

    for &var in &incidence[eq] {
        if var >= visited.len() || visited[var] {
            continue;
        }
        visited[var] = true;

        let can_reassign = match var_to_eq[var] {
            None => true,
            Some(other_eq) => {
                augmenting_path_restricted(other_eq, incidence, eq_to_var, var_to_eq, visited)
            }
        };

        if can_reassign {
            eq_to_var[eq] = Some(var);
            var_to_eq[var] = Some(eq);
            return true;
        }
    }

    false
}

// ============================================================================
// Automatic Structure Detection
// ============================================================================

/// Detect the incidence structure of a DAE system automatically using finite differences.
///
/// Probes the RHS function to determine which variables appear in which equations.
///
/// # Arguments
///
/// * `system` - The DAE system (must report `is_singular_mass()` and `algebraic_indices()`)
/// * `t0` - Time point for probing
/// * `y0` - State point for probing
///
/// # Returns
///
/// A `DaeStructure` describing the equation-variable dependencies.
pub fn detect_structure<S, Sys>(system: &Sys, t0: S, y0: &[S]) -> DaeStructure
where
    S: Scalar,
    Sys: OdeSystem<S>,
{
    let n = system.dim();
    let alg_indices = system.algebraic_indices();
    let n_alg = alg_indices.len();
    let n_diff = n - n_alg;

    // Map variable index to whether it's algebraic
    let is_algebraic: Vec<bool> = (0..n).map(|i| alg_indices.contains(&i)).collect();

    // Differential equation indices
    let diff_eq_indices: Vec<usize> = (0..n).filter(|i| !is_algebraic[*i]).collect();
    let alg_eq_indices: Vec<usize> = (0..n).filter(|i| is_algebraic[*i]).collect();

    let n_diff_eqs = diff_eq_indices.len();
    let n_alg_eqs = alg_eq_indices.len();

    // Probe RHS at y0
    let h_factor = S::EPSILON.sqrt();
    let mut f0 = vec![S::ZERO; n];
    system.rhs(t0, y0, &mut f0);

    // For each equation, determine which variables it depends on
    let mut incidence = vec![Vec::new(); n_diff_eqs + n_alg_eqs];
    let mut y_pert = y0.to_vec();

    for j in 0..n {
        let yj_save = y_pert[j];
        let h = h_factor * (S::ONE + yj_save.abs());
        y_pert[j] = yj_save + h;

        let mut f1 = vec![S::ZERO; n];
        system.rhs(t0, &y_pert, &mut f1);
        y_pert[j] = yj_save;

        // Check which equations changed
        let threshold = S::from_f64(1e-12);

        // Check differential equations
        for (local_idx, &eq_idx) in diff_eq_indices.iter().enumerate() {
            let diff = (f1[eq_idx] - f0[eq_idx]).abs();
            if diff > threshold * h {
                incidence[local_idx].push(j);
            }
        }

        // Check algebraic equations
        for (local_idx, &eq_idx) in alg_eq_indices.iter().enumerate() {
            let diff = (f1[eq_idx] - f0[eq_idx]).abs();
            if diff > threshold * h {
                incidence[n_diff_eqs + local_idx].push(j);
            }
        }
    }

    DaeStructure {
        n_diff,
        n_alg,
        n_diff_eqs,
        n_alg_eqs,
        incidence,
    }
}

// ============================================================================
// Index Reduction via Constraint Differentiation
// ============================================================================

/// Type alias for boxed RHS function.
type RhsFn<S> = Box<dyn Fn(S, &[S], &mut [S]) + Send + Sync>;
/// Type alias for boxed mass matrix function.
type MassFn<S> = Box<dyn Fn(&mut [S]) + Send + Sync>;

/// A reduced (index-1) DAE system produced by differentiating constraints.
///
/// The reduced system augments the original with:
/// - New variables for the time derivatives of algebraic variables
/// - Differentiated constraint equations
///
/// It implements `OdeSystem<S>` so it can be used directly with BDF/Radau solvers.
impl<S: Scalar> core::fmt::Debug for ReducedDaeSystem<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReducedDaeSystem")
            .field("n_orig", &self.n_orig)
            .field("n_diff", &self.n_diff)
            .field("aug_dim", &self.aug_dim)
            .field("n_new_vars", &self.n_new_vars)
            .field("info", &self.info)
            .finish()
    }
}

pub struct ReducedDaeSystem<S: Scalar> {
    /// Original system dimension
    n_orig: usize,
    /// Number of original differential equations
    n_diff: usize,
    /// Indices of algebraic equations in the original system
    alg_eq_indices: Vec<usize>,
    /// Indices of differential equations in the original system
    diff_eq_indices: Vec<usize>,
    /// The original RHS function (boxed for type erasure)
    rhs_fn: RhsFn<S>,
    /// The original mass matrix function (fills row-major)
    mass_fn: MassFn<S>,
    /// Differentiation info
    info: DaeIndexInfo,
    /// Augmented dimension (original + new derivative variables)
    aug_dim: usize,
    /// Number of new variables added by differentiation
    n_new_vars: usize,
    /// Finite difference epsilon for constraint differentiation
    fd_eps: S,
}

impl<S: Scalar> ReducedDaeSystem<S> {
    /// Get the differentiation info.
    pub fn info(&self) -> &DaeIndexInfo {
        &self.info
    }

    /// Get the number of differential variables in the original system.
    pub fn n_diff(&self) -> usize {
        self.n_diff
    }

    /// Get the augmented system dimension.
    pub fn augmented_dim(&self) -> usize {
        self.aug_dim
    }

    /// Get the original system dimension.
    pub fn original_dim(&self) -> usize {
        self.n_orig
    }

    /// Extract the original state variables from an augmented state vector.
    pub fn extract_original(&self, y_aug: &[S]) -> Vec<S> {
        y_aug[..self.n_orig].to_vec()
    }

    /// Build augmented initial conditions.
    ///
    /// The new variables (time derivatives of algebraic variables) are initialized
    /// by evaluating the constraints' time derivatives at the initial point.
    pub fn augment_initial_conditions(&self, t0: S, y0: &[S]) -> Vec<S> {
        assert_eq!(y0.len(), self.n_orig, "y0 must have original dimension");

        let mut y_aug = vec![S::ZERO; self.aug_dim];
        // Copy original variables
        y_aug[..self.n_orig].copy_from_slice(y0);

        // Estimate initial values for new derivative variables using FD
        // For each algebraic variable z_i, the new variable is dz_i/dt
        // We approximate this from the original RHS
        let mut f0 = vec![S::ZERO; self.n_orig];
        (self.rhs_fn)(t0, y0, &mut f0);

        // The new variables correspond to the time derivatives of algebraic variables
        // For index-2 reduction: new var = dz/dt, estimated from the RHS
        for (k, &alg_eq) in self.alg_eq_indices.iter().enumerate() {
            if self.n_orig + k < self.aug_dim {
                // Use the residual rate of change as initial estimate
                // For a consistent index-1 system, the algebraic residual should be ~0
                // and its time derivative gives the new variable
                y_aug[self.n_orig + k] = f0[alg_eq];
            }
        }

        y_aug
    }

    /// Evaluate the time derivative of an algebraic constraint using FD.
    ///
    /// For constraint g(t, y) = 0, computes dg/dt = ∂g/∂t + (∂g/∂y) * y'
    fn differentiate_constraint(&self, t: S, y: &[S], eq_idx: usize, dydt_diff: &[S]) -> S {
        let n = self.n_orig;
        let eps = self.fd_eps;

        // Compute g(t, y)
        let mut f0 = vec![S::ZERO; n];
        (self.rhs_fn)(t, y, &mut f0);
        let g0 = f0[eq_idx];

        // ∂g/∂t (explicit time dependence)
        let mut f_tp = vec![S::ZERO; n];
        let ht = eps * (S::ONE + t.abs());
        (self.rhs_fn)(t + ht, y, &mut f_tp);
        let dgdt = (f_tp[eq_idx] - g0) / ht;

        // ∂g/∂y_j * dy_j/dt for all variables j
        let mut dgdy_dot = S::ZERO;
        let mut y_pert = y.to_vec();

        for j in 0..n {
            let yj_save = y_pert[j];
            let h = eps * (S::ONE + yj_save.abs());
            y_pert[j] = yj_save + h;

            let mut f1 = vec![S::ZERO; n];
            (self.rhs_fn)(t, &y_pert, &mut f1);
            y_pert[j] = yj_save;

            let dgdyj = (f1[eq_idx] - g0) / h;
            dgdy_dot = dgdy_dot + dgdyj * dydt_diff[j];
        }

        dgdt + dgdy_dot
    }
}

impl<S: Scalar> OdeSystem<S> for ReducedDaeSystem<S> {
    fn dim(&self) -> usize {
        self.aug_dim
    }

    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        let n = self.n_orig;

        // Evaluate original RHS for the original variables
        let y_orig = &y[..n];
        let mut f_orig = vec![S::ZERO; n];
        (self.rhs_fn)(t, y_orig, &mut f_orig);

        // Copy differential equation RHS
        for &i in &self.diff_eq_indices {
            dydt[i] = f_orig[i];
        }

        // For algebraic equations: keep the original constraint as residual
        // (these become 0 = g(t, y) in the mass matrix form)
        for &i in &self.alg_eq_indices {
            dydt[i] = f_orig[i];
        }

        // For the new variables (differentiated constraints):
        // The differentiated constraint dg/dt = 0 becomes a new algebraic equation.
        // We need the current y' to compute dg/dt.
        // Use the original f values as the current y' estimate.
        for (k, &(eq_idx, n_diffs)) in self.info.differentiation_schedule.iter().enumerate() {
            if k >= self.n_new_vars {
                break;
            }

            // For the first differentiation: dg/dt = ∂g/∂t + Σ (∂g/∂y_j) * y_j'
            // The differentiated constraint = 0 is a new algebraic equation
            // whose residual we place in the augmented slot
            let new_var_idx = n + k;
            if new_var_idx < self.aug_dim && n_diffs >= 1 {
                // Compute dg/dt using the current state
                let dg = self.differentiate_constraint(t, y_orig, eq_idx, &f_orig);
                dydt[new_var_idx] = dg;
            }
        }
    }

    fn has_mass_matrix(&self) -> bool {
        true
    }

    fn mass_matrix(&self, mass: &mut [S]) {
        let aug = self.aug_dim;
        // Zero out
        for i in 0..aug * aug {
            mass[i] = S::ZERO;
        }

        // Fill original mass matrix block
        let n = self.n_orig;
        let mut orig_mass = vec![S::ZERO; n * n];
        (self.mass_fn)(&mut orig_mass);

        for i in 0..n {
            for j in 0..n {
                mass[i * aug + j] = orig_mass[i * n + j];
            }
        }

        // New equations are algebraic (mass = 0 on their rows)
        // They are already zero from initialization
        // No need to set M[new_row, new_row] = 0, it's already 0
    }

    fn is_singular_mass(&self) -> bool {
        true
    }

    fn algebraic_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();

        // Original algebraic indices
        for &i in &self.alg_eq_indices {
            indices.push(i);
        }

        // New differentiated constraint equations are also algebraic
        for k in 0..self.n_new_vars {
            indices.push(self.n_orig + k);
        }

        indices
    }
}

// ============================================================================
// Index Reduction Entry Points
// ============================================================================

/// Reduce a higher-index DAE to index-1 by differentiating constraints.
///
/// This function:
/// 1. Detects the DAE structure via FD probing
/// 2. Analyzes the structural index via the Pantelides algorithm
/// 3. If index > 1, builds a reduced system with differentiated constraints
///
/// # Arguments
///
/// * `rhs_fn` - The RHS function f(t, y, dydt)
/// * `mass_fn` - The mass matrix function M(mass_out)
/// * `alg_indices` - Indices of algebraic equations
/// * `n` - System dimension
/// * `t0` - Initial time for structure probing
/// * `y0` - Initial state for structure probing
///
/// # Returns
///
/// A `ReducedDaeSystem` if the index is > 1, or an error if reduction fails.
pub fn reduce_index<S, F, M>(
    rhs_fn: F,
    mass_fn: M,
    alg_indices: &[usize],
    n: usize,
    t0: S,
    y0: &[S],
) -> Result<ReducedDaeSystem<S>, String>
where
    S: Scalar,
    F: Fn(S, &[S], &mut [S]) + Send + Sync + 'static,
    M: Fn(&mut [S]) + Send + Sync + 'static,
{
    // Build structure info
    let n_alg = alg_indices.len();
    let n_diff = n - n_alg;

    let is_algebraic: Vec<bool> = (0..n).map(|i| alg_indices.contains(&i)).collect();
    let diff_eq_indices: Vec<usize> = (0..n).filter(|i| !is_algebraic[*i]).collect();
    let alg_eq_indices: Vec<usize> = alg_indices.to_vec();

    // Detect incidence structure
    let structure = detect_structure_from_fn(&rhs_fn, n, &diff_eq_indices, &alg_eq_indices, t0, y0);

    // Analyze index
    let info = analyze_dae_index(&structure);

    if info.structural_index <= 1 {
        return Err("System is already index-1 or index-0; no reduction needed".to_string());
    }

    // Count new variables needed
    let n_new_vars: usize = info
        .differentiation_schedule
        .iter()
        .map(|&(_, nd)| nd)
        .sum();

    let aug_dim = n + n_new_vars;

    Ok(ReducedDaeSystem {
        n_orig: n,
        n_diff,
        alg_eq_indices,
        diff_eq_indices,
        rhs_fn: Box::new(rhs_fn),
        mass_fn: Box::new(mass_fn),
        info,
        aug_dim,
        n_new_vars,
        fd_eps: S::EPSILON.sqrt(),
    })
}

/// Detect structure from a bare function (not wrapped in OdeSystem).
fn detect_structure_from_fn<S, F>(
    rhs_fn: &F,
    n: usize,
    diff_eq_indices: &[usize],
    alg_eq_indices: &[usize],
    t0: S,
    y0: &[S],
) -> DaeStructure
where
    S: Scalar,
    F: Fn(S, &[S], &mut [S]),
{
    let n_diff_eqs = diff_eq_indices.len();
    let n_alg_eqs = alg_eq_indices.len();
    let n_diff = n - n_alg_eqs;
    let n_alg = n_alg_eqs;

    let h_factor = S::EPSILON.sqrt();
    let mut f0 = vec![S::ZERO; n];
    rhs_fn(t0, y0, &mut f0);

    let mut incidence = vec![Vec::new(); n_diff_eqs + n_alg_eqs];
    let mut y_pert = y0.to_vec();

    for j in 0..n {
        let yj_save = y_pert[j];
        let h = h_factor * (S::ONE + yj_save.abs());
        y_pert[j] = yj_save + h;

        let mut f1 = vec![S::ZERO; n];
        rhs_fn(t0, &y_pert, &mut f1);
        y_pert[j] = yj_save;

        let threshold = S::from_f64(1e-12);

        for (local_idx, &eq_idx) in diff_eq_indices.iter().enumerate() {
            let diff = (f1[eq_idx] - f0[eq_idx]).abs();
            if diff > threshold * h {
                incidence[local_idx].push(j);
            }
        }

        for (local_idx, &eq_idx) in alg_eq_indices.iter().enumerate() {
            let diff = (f1[eq_idx] - f0[eq_idx]).abs();
            if diff > threshold * h {
                incidence[n_diff_eqs + local_idx].push(j);
            }
        }
    }

    DaeStructure {
        n_diff,
        n_alg,
        n_diff_eqs,
        n_alg_eqs,
        incidence,
    }
}

/// Convenience: analyze and reduce a `DaeProblem` directly.
///
/// Returns a `ReducedDaeSystem` if the system is higher-index,
/// or an `Err` if it's already index-1.
pub fn reduce_dae_problem<S, F, M>(
    problem: &DaeProblem<S, F, M>,
    t0: S,
    y0: &[S],
) -> Result<(DaeIndexInfo, ReducedDaeSystem<S>), String>
where
    S: Scalar,
    F: Fn(S, &[S], &mut [S]) + Clone + Send + Sync + 'static,
    M: Fn(&mut [S]) + Clone + Send + Sync + 'static,
{
    let structure = detect_structure(problem, t0, y0);
    let info = analyze_dae_index(&structure);

    if info.structural_index <= 1 {
        return Err(format!(
            "System is index-{}, no reduction needed",
            info.structural_index
        ));
    }

    let rhs_fn = problem.f.clone();
    let mass_fn = problem.mass.clone();
    let alg_indices = problem.alg_indices.clone();
    let n = problem.dim();

    let n_new_vars: usize = info
        .differentiation_schedule
        .iter()
        .map(|&(_, nd)| nd)
        .sum();
    let aug_dim = n + n_new_vars;

    let is_algebraic: Vec<bool> = (0..n).map(|i| alg_indices.contains(&i)).collect();
    let diff_eq_indices: Vec<usize> = (0..n).filter(|i| !is_algebraic[*i]).collect();

    let reduced = ReducedDaeSystem {
        n_orig: n,
        n_diff: n - alg_indices.len(),
        alg_eq_indices: alg_indices,
        diff_eq_indices,
        rhs_fn: Box::new(rhs_fn),
        mass_fn: Box::new(mass_fn),
        info: info.clone(),
        aug_dim,
        n_new_vars,
        fd_eps: S::EPSILON.sqrt(),
    };

    Ok((info, reduced))
}

/// Analyze a DAE system and return its structural index without reducing.
///
/// Useful for diagnostic purposes.
pub fn analyze_system<S, Sys>(system: &Sys, t0: S, y0: &[S]) -> DaeIndexInfo
where
    S: Scalar,
    Sys: OdeSystem<S>,
{
    let structure = detect_structure(system, t0, y0);
    analyze_dae_index(&structure)
}

use crate::DaeProblem;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Structural analysis tests ----

    #[test]
    fn test_pure_ode_index_0() {
        // No algebraic equations => index 0
        let structure = DaeStructure {
            n_diff: 2,
            n_alg: 0,
            n_diff_eqs: 2,
            n_alg_eqs: 0,
            incidence: vec![
                vec![0, 1], // eq0 depends on y0, y1
                vec![0, 1], // eq1 depends on y0, y1
            ],
        };

        let info = analyze_dae_index(&structure);
        assert_eq!(info.structural_index, 0);
        assert_eq!(info.n_hidden_constraints, 0);
        assert!(info.differentiation_schedule.is_empty());
    }

    #[test]
    fn test_index_1_dae() {
        // Index-1 DAE:
        //   y0' = -y0 + y1      (diff eq, depends on y0, y1)
        //   0   = y1 - y0^2     (alg eq, depends on y0, y1)
        //
        // The algebraic equation can be matched to y1 => index 1
        let structure = DaeStructure {
            n_diff: 1,
            n_alg: 1,
            n_diff_eqs: 1,
            n_alg_eqs: 1,
            incidence: vec![
                vec![0, 1], // diff eq depends on y0, y1
                vec![0, 1], // alg eq depends on y0, y1
            ],
        };

        let info = analyze_dae_index(&structure);
        assert_eq!(info.structural_index, 1);
        assert_eq!(info.n_hidden_constraints, 0);
    }

    #[test]
    fn test_index_2_dae() {
        // Index-2 DAE:
        //   x'  = v           (diff, depends on x, v, but really just v)
        //   v'  = -lambda*x   (diff, depends on x, lambda)
        //   0   = x^2 - 1     (alg, depends on x only)
        //
        // Variables: 0=x, 1=v, 2=lambda
        // Diff eqs: eq0 (for x), eq1 (for v)
        // Alg eq: eq2 (constraint)
        //
        // The constraint only depends on x (var 0). Var 0 is already matched
        // to eq0. The constraint can't match to lambda (var 2) since it doesn't
        // depend on it. So it needs differentiation => index 2.
        let structure = DaeStructure {
            n_diff: 2,
            n_alg: 1,
            n_diff_eqs: 2,
            n_alg_eqs: 1,
            incidence: vec![
                vec![0, 1], // eq0: x' = v (depends on x, v)
                vec![0, 2], // eq1: v' = -lambda*x (depends on x, lambda)
                vec![0],    // eq2: 0 = x^2 - 1 (depends on x only)
            ],
        };

        let info = analyze_dae_index(&structure);
        assert!(
            info.structural_index >= 2,
            "Expected index >= 2, got {}",
            info.structural_index
        );
        assert!(info.n_hidden_constraints >= 1);
        assert!(!info.differentiation_schedule.is_empty());
    }

    #[test]
    fn test_detect_structure_index1() {
        // Semi-explicit index-1 DAE:
        //   y0' = -y0        (differential)
        //   0 = y1 - y0^2    (algebraic: y1 = y0^2)
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            0.0,
            1.0,
            vec![2.0, 4.0],
            vec![1],
        );

        let structure = detect_structure(&dae, 0.0, &[2.0, 4.0]);
        assert_eq!(structure.n_diff, 1);
        assert_eq!(structure.n_alg, 1);
        assert_eq!(structure.n_diff_eqs, 1);
        assert_eq!(structure.n_alg_eqs, 1);

        let info = analyze_dae_index(&structure);
        assert_eq!(info.structural_index, 1);
    }

    #[test]
    fn test_detect_structure_index2() {
        // Index-2 DAE (simplified pendulum-like):
        //   x'  = v                    (eq for x)
        //   v'  = -lambda * x          (eq for v)
        //   0   = x^2 - 1.0            (constraint: x on unit circle)
        //
        // State: [x, v, lambda]
        // Algebraic index: 2 (lambda)
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                let x = y[0];
                let v = y[1];
                let lam = y[2];
                dydt[0] = v; // x' = v
                dydt[1] = -lam * x; // v' = -lambda*x
                dydt[2] = x * x - 1.0; // 0 = x^2 - 1
            },
            |mass: &mut [f64]| {
                // 3x3 mass matrix
                for i in 0..9 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0; // x is differential
                mass[4] = 1.0; // v is differential
                               // mass[8] = 0   // lambda is algebraic
            },
            0.0,
            1.0,
            vec![1.0, 0.0, 0.0],
            vec![2],
        );

        let structure = detect_structure(&dae, 0.0, &[1.0, 0.0, 0.0]);
        let info = analyze_dae_index(&structure);
        assert!(
            info.structural_index >= 2,
            "Expected index >= 2, got {}. Schedule: {:?}",
            info.structural_index,
            info.differentiation_schedule
        );
    }

    #[test]
    fn test_reduce_index_creates_augmented_system() {
        // Index-2 system from the pendulum-like example
        let n = 3;
        let alg_indices = vec![2usize];

        let result = reduce_index(
            |_t, y: &[f64], dydt: &mut [f64]| {
                let x = y[0];
                let v = y[1];
                let lam = y[2];
                dydt[0] = v;
                dydt[1] = -lam * x;
                dydt[2] = x * x - 1.0;
            },
            |mass: &mut [f64]| {
                for i in 0..9 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0;
                mass[4] = 1.0;
            },
            &alg_indices,
            n,
            0.0_f64,
            &[1.0, 0.0, 0.0],
        );

        assert!(
            result.is_ok(),
            "Reduction should succeed: {:?}",
            result.err()
        );
        let reduced = result.unwrap();

        // Augmented system should be larger
        assert!(
            reduced.augmented_dim() > n,
            "Augmented dim {} should be > original dim {}",
            reduced.augmented_dim(),
            n
        );

        // Should still report as DAE
        assert!(reduced.is_singular_mass());

        // Test augmented IC
        let y0_aug = reduced.augment_initial_conditions(0.0, &[1.0, 0.0, 0.0]);
        assert_eq!(y0_aug.len(), reduced.augmented_dim());
        // Original part should be preserved
        assert!((y0_aug[0] - 1.0).abs() < 1e-10);
        assert!((y0_aug[1] - 0.0).abs() < 1e-10);
        assert!((y0_aug[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduced_system_rhs_evaluates() {
        // Verify the reduced system can evaluate its RHS without panicking
        let n = 3;
        let alg_indices = vec![2usize];

        let reduced = reduce_index(
            |_t, y: &[f64], dydt: &mut [f64]| {
                let x = y[0];
                let v = y[1];
                let lam = y[2];
                dydt[0] = v;
                dydt[1] = -lam * x;
                dydt[2] = x * x - 1.0;
            },
            |mass: &mut [f64]| {
                for i in 0..9 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0;
                mass[4] = 1.0;
            },
            &alg_indices,
            n,
            0.0_f64,
            &[1.0, 0.0, 0.0],
        )
        .unwrap();

        let aug_dim = reduced.augmented_dim();
        let y0_aug = reduced.augment_initial_conditions(0.0, &[1.0, 0.0, 0.0]);
        let mut dydt = vec![0.0; aug_dim];
        reduced.rhs(0.0, &y0_aug, &mut dydt);

        // Original differential equations should give correct values
        // x' = v = 0
        assert!(
            (dydt[0] - 0.0).abs() < 1e-10,
            "x' should be 0, got {}",
            dydt[0]
        );
        // v' = -lambda*x = 0*1 = 0
        assert!(
            (dydt[1] - 0.0).abs() < 1e-10,
            "v' should be 0, got {}",
            dydt[1]
        );
        // Original constraint: x^2 - 1 = 0 (consistent)
        assert!(
            (dydt[2] - 0.0).abs() < 1e-8,
            "constraint should be ~0, got {}",
            dydt[2]
        );
    }

    #[test]
    fn test_reduced_system_mass_matrix() {
        let n = 3;
        let alg_indices = vec![2usize];

        let reduced = reduce_index(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[2] * y[0];
                dydt[2] = y[0] * y[0] - 1.0;
            },
            |mass: &mut [f64]| {
                for i in 0..9 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0;
                mass[4] = 1.0;
            },
            &alg_indices,
            n,
            0.0_f64,
            &[1.0, 0.0, 0.0],
        )
        .unwrap();

        let aug_dim = reduced.augmented_dim();
        let mut mass = vec![0.0; aug_dim * aug_dim];
        reduced.mass_matrix(&mut mass);

        // Original mass entries preserved
        assert!((mass[0 * aug_dim + 0] - 1.0).abs() < 1e-10); // M[0,0] = 1
        assert!((mass[1 * aug_dim + 1] - 1.0).abs() < 1e-10); // M[1,1] = 1
        assert!((mass[2 * aug_dim + 2] - 0.0).abs() < 1e-10); // M[2,2] = 0 (algebraic)

        // New rows should be zero (algebraic)
        for k in n..aug_dim {
            for j in 0..aug_dim {
                assert!(
                    (mass[k * aug_dim + j] - 0.0).abs() < 1e-10,
                    "New row {} should be all zero, but M[{},{}] = {}",
                    k,
                    k,
                    j,
                    mass[k * aug_dim + j]
                );
            }
        }
    }

    #[test]
    fn test_already_index1_returns_err() {
        // Index-1 system should return Err (no reduction needed)
        let result = reduce_index(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            &[1],
            2,
            0.0_f64,
            &[2.0, 4.0],
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("index-1"));
    }

    #[test]
    fn test_analyze_system_convenience() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
                dydt[1] = y[1] - y[0] * y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
                mass[1] = 0.0;
                mass[2] = 0.0;
                mass[3] = 0.0;
            },
            0.0,
            1.0,
            vec![2.0, 4.0],
            vec![1],
        );

        let info = analyze_system(&dae, 0.0, &[2.0, 4.0]);
        assert_eq!(info.structural_index, 1);
    }

    #[test]
    fn test_extract_original() {
        let n = 3;
        let reduced = reduce_index(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[2] * y[0];
                dydt[2] = y[0] * y[0] - 1.0;
            },
            |mass: &mut [f64]| {
                for i in 0..9 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0;
                mass[4] = 1.0;
            },
            &[2],
            n,
            0.0_f64,
            &[1.0, 0.0, 0.0],
        )
        .unwrap();

        let y_aug = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // some augmented state
        let y_orig = reduced.extract_original(&y_aug);
        assert_eq!(y_orig.len(), 3);
        assert!((y_orig[0] - 1.0).abs() < 1e-10);
        assert!((y_orig[1] - 2.0).abs() < 1e-10);
        assert!((y_orig[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dae_structure_helpers() {
        let structure = DaeStructure {
            n_diff: 2,
            n_alg: 1,
            n_diff_eqs: 2,
            n_alg_eqs: 1,
            incidence: vec![vec![0, 1], vec![0, 2], vec![0]],
        };

        assert_eq!(structure.n_vars(), 3);
        assert_eq!(structure.n_eqs(), 3);
    }

    #[test]
    fn test_pantelides_empty_incidence() {
        // Edge case: algebraic equation with no dependencies
        let structure = DaeStructure {
            n_diff: 1,
            n_alg: 1,
            n_diff_eqs: 1,
            n_alg_eqs: 1,
            incidence: vec![
                vec![0], // diff eq depends on y0
                vec![],  // alg eq depends on nothing — structurally singular
            ],
        };

        let info = analyze_dae_index(&structure);
        // Should handle gracefully, may report index >= 1
        assert!(info.structural_index >= 1);
    }

    #[test]
    fn test_multiple_algebraic_equations() {
        // Two algebraic constraints, both index-1
        // y0' = -y0
        // 0 = y1 - y0
        // 0 = y2 - y0^2
        let structure = DaeStructure {
            n_diff: 1,
            n_alg: 2,
            n_diff_eqs: 1,
            n_alg_eqs: 2,
            incidence: vec![
                vec![0],    // diff eq: y0' = -y0
                vec![0, 1], // alg eq1: y1 - y0 = 0 (depends on y0, y1)
                vec![0, 2], // alg eq2: y2 - y0^2 = 0 (depends on y0, y2)
            ],
        };

        let info = analyze_dae_index(&structure);
        assert_eq!(info.structural_index, 1);
        assert_eq!(info.n_hidden_constraints, 0);
    }
}
