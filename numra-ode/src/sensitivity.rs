//! Forward sensitivity analysis for ODE systems.
//!
//! This module provides infrastructure for computing sensitivities of ODE solutions
//! with respect to parameters. Given an ODE system:
//!
//! ```text
//! dy/dt = f(t, y, p),  y(t0) = y0(p)
//! ```
//!
//! The forward sensitivity equations compute how the solution changes with
//! respect to parameters p:
//!
//! ```text
//! dS/dt = ∂f/∂y * S + ∂f/∂p,  S(t0) = ∂y0/∂p
//! ```
//!
//! where S = ∂y/∂p is the sensitivity matrix.
//!
//! # Example
//!
//! ```
//! use numra_ode::sensitivity::SensitivityEquations;
//!
//! // Define a system with parameters
//! struct Decay {
//!     k: f64,  // decay rate (parameter)
//! }
//!
//! impl SensitivityEquations<f64> for Decay {
//!     fn n_states(&self) -> usize { 1 }
//!     fn n_params(&self) -> usize { 1 }
//!
//!     fn eval(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
//!         dydt[0] = -self.k * y[0];
//!     }
//!
//!     fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
//!         // ∂f/∂y (row-major for 1x1: just -k)
//!         jac[0] = -self.k;
//!     }
//!
//!     fn param_jacobian(&self, _t: f64, y: &[f64], dpdy: &mut [f64]) {
//!         // ∂f/∂p (1 state × 1 param: -y[0])
//!         dpdy[0] = -y[0];
//!     }
//! }
//!
//! let decay = Decay { k: 0.5 };
//! assert_eq!(decay.n_states(), 1);
//! assert_eq!(decay.n_params(), 1);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for ODE systems that support forward sensitivity analysis.
///
/// This trait extends the basic ODE system with methods to compute the
/// Jacobian ∂f/∂y and parameter Jacobian ∂f/∂p, which are needed for
/// the forward sensitivity equations.
pub trait SensitivityEquations<S: Scalar>: Send + Sync {
    /// Number of state variables (dimension of y).
    fn n_states(&self) -> usize;

    /// Number of parameters.
    fn n_params(&self) -> usize;

    /// Evaluate the ODE right-hand side: dydt = f(t, y).
    fn eval(&self, t: S, y: &[S], dydt: &mut [S]);

    /// Compute the Jacobian matrix ∂f/∂y.
    ///
    /// The Jacobian should be stored in row-major order:
    /// jac[i * n + j] = ∂f_i/∂y_j
    ///
    /// where n = n_states().
    fn jacobian(&self, t: S, y: &[S], jac: &mut [S]);

    /// Compute the parameter Jacobian matrix ∂f/∂p.
    ///
    /// The matrix should be stored in row-major order:
    /// dpdy[i * n_params + j] = ∂f_i/∂p_j
    fn param_jacobian(&self, t: S, y: &[S], dpdy: &mut [S]);

    /// Compute the sensitivity of initial conditions: ∂y0/∂p.
    ///
    /// By default, assumes initial conditions don't depend on parameters.
    /// Override if y0 = y0(p).
    fn initial_sensitivity(&self, _y0: &[S], sens0: &mut [S]) {
        for s in sens0.iter_mut() {
            *s = S::ZERO;
        }
    }

    /// Optional: use finite differences for Jacobian if analytical not available.
    fn jacobian_fd(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.n_states();
        let h = S::from_f64(1e-8);

        let mut y_plus = y.to_vec();
        let mut f_base = vec![S::ZERO; n];
        let mut f_plus = vec![S::ZERO; n];

        self.eval(t, y, &mut f_base);

        for j in 0..n {
            let y_j_orig = y_plus[j];
            y_plus[j] = y_j_orig + h;
            self.eval(t, &y_plus, &mut f_plus);

            for i in 0..n {
                jac[i * n + j] = (f_plus[i] - f_base[i]) / h;
            }

            y_plus[j] = y_j_orig;
        }
    }

    /// Optional: use finite differences for parameter Jacobian.
    fn param_jacobian_fd(&self, t: S, y: &[S], params: &[S], dpdy: &mut [S]) {
        let n = self.n_states();
        let np = self.n_params();
        let h = S::from_f64(1e-8);

        let mut f_base = vec![S::ZERO; n];
        let f_plus = vec![S::ZERO; n];

        SensitivityEquations::eval(self, t, y, &mut f_base);

        // Note: This requires the system to accept updated parameters
        // This is a placeholder - actual implementation would need parameter mutation
        let _ = (params, h, dpdy, np, f_plus);
    }
}

/// Augmented state for sensitivity computation.
///
/// Contains both the original state y and the sensitivity matrix S.
#[derive(Clone, Debug)]
pub struct SensitivityState<S: Scalar> {
    /// Original state vector (length n_states)
    pub y: Vec<S>,
    /// Sensitivity matrix in row-major order (n_states × n_params)
    pub sensitivity: Vec<S>,
}

impl<S: Scalar> SensitivityState<S> {
    /// Create a new sensitivity state.
    pub fn new(n_states: usize, n_params: usize) -> Self {
        Self {
            y: vec![S::ZERO; n_states],
            sensitivity: vec![S::ZERO; n_states * n_params],
        }
    }

    /// Create from initial conditions.
    pub fn from_initial(y0: Vec<S>, n_params: usize) -> Self {
        let n_states = y0.len();
        Self {
            y: y0,
            sensitivity: vec![S::ZERO; n_states * n_params],
        }
    }

    /// Get sensitivity of state i with respect to parameter j.
    pub fn get_sensitivity(&self, i: usize, j: usize, n_params: usize) -> S {
        self.sensitivity[i * n_params + j]
    }

    /// Set sensitivity of state i with respect to parameter j.
    pub fn set_sensitivity(&mut self, i: usize, j: usize, n_params: usize, value: S) {
        self.sensitivity[i * n_params + j] = value;
    }

    /// Total dimension of augmented state (n_states + n_states * n_params).
    pub fn total_dim(&self, n_params: usize) -> usize {
        self.y.len() * (1 + n_params)
    }

    /// Convert to flat augmented vector [y; vec(S)].
    pub fn to_augmented(&self) -> Vec<S> {
        let mut aug = self.y.clone();
        aug.extend(&self.sensitivity);
        aug
    }

    /// Create from flat augmented vector.
    pub fn from_augmented(aug: &[S], n_states: usize, _n_params: usize) -> Self {
        let y = aug[..n_states].to_vec();
        let sensitivity = aug[n_states..].to_vec();
        Self { y, sensitivity }
    }
}

/// Wrapper that creates an augmented ODE system for sensitivity computation.
///
/// Given a system with SensitivityEquations trait, this creates a larger
/// system that simultaneously integrates the original ODE and the
/// sensitivity equations.
pub struct AugmentedSystem<S: Scalar, Sys: SensitivityEquations<S>> {
    /// The original system
    pub system: Sys,
    /// Workspace for Jacobian
    jac_workspace: Vec<S>,
    /// Workspace for parameter Jacobian
    param_jac_workspace: Vec<S>,
}

impl<S: Scalar, Sys: SensitivityEquations<S>> AugmentedSystem<S, Sys> {
    /// Create a new augmented system.
    pub fn new(system: Sys) -> Self {
        let n = system.n_states();
        let np = system.n_params();
        Self {
            system,
            jac_workspace: vec![S::ZERO; n * n],
            param_jac_workspace: vec![S::ZERO; n * np],
        }
    }

    /// Dimension of the augmented system.
    pub fn augmented_dim(&self) -> usize {
        let n = self.system.n_states();
        let np = self.system.n_params();
        n * (1 + np)
    }

    /// Evaluate the augmented RHS.
    ///
    /// The augmented state is [y; S] where S is the sensitivity matrix
    /// flattened in row-major order.
    ///
    /// The augmented RHS is:
    /// - dy/dt = f(t, y)
    /// - dS/dt = (∂f/∂y) * S + ∂f/∂p
    pub fn eval_augmented(&mut self, t: S, aug: &[S], daug: &mut [S]) {
        let n = self.system.n_states();
        let np = self.system.n_params();

        // Extract state
        let y = &aug[..n];

        // Evaluate original RHS
        self.system.eval(t, y, &mut daug[..n]);

        // Compute Jacobians
        self.system.jacobian(t, y, &mut self.jac_workspace);
        self.system
            .param_jacobian(t, y, &mut self.param_jac_workspace);

        // Compute dS/dt = J * S + dp/dy
        // S is stored as aug[n..] in row-major: S[i,j] = aug[n + i*np + j]
        for i in 0..n {
            for j in 0..np {
                let idx = n + i * np + j;

                // (J * S)[i,j] = sum_k J[i,k] * S[k,j]
                let mut sum = S::ZERO;
                for k in 0..n {
                    let jac_ik = self.jac_workspace[i * n + k];
                    let s_kj = aug[n + k * np + j];
                    sum = sum + jac_ik * s_kj;
                }

                // Add ∂f_i/∂p_j
                sum = sum + self.param_jac_workspace[i * np + j];

                daug[idx] = sum;
            }
        }
    }

    /// Create initial augmented state.
    pub fn initial_augmented(&self, y0: &[S]) -> Vec<S> {
        let n = self.system.n_states();
        let np = self.system.n_params();

        let mut aug = Vec::with_capacity(n * (1 + np));
        aug.extend_from_slice(y0);

        // Initial sensitivities (typically zero)
        let mut sens0 = vec![S::ZERO; n * np];
        self.system.initial_sensitivity(y0, &mut sens0);
        aug.extend(sens0);

        aug
    }

    /// Extract state from augmented vector.
    pub fn extract_state(&self, aug: &[S]) -> Vec<S> {
        let n = self.system.n_states();
        aug[..n].to_vec()
    }

    /// Extract sensitivity matrix from augmented vector.
    pub fn extract_sensitivity(&self, aug: &[S]) -> Vec<S> {
        let n = self.system.n_states();
        aug[n..].to_vec()
    }
}

/// Result of sensitivity analysis.
#[derive(Clone, Debug)]
pub struct SensitivityResult<S: Scalar> {
    /// Time points
    pub times: Vec<S>,
    /// States at each time point
    pub states: Vec<Vec<S>>,
    /// Sensitivity matrices at each time point (each is n_states × n_params, row-major)
    pub sensitivities: Vec<Vec<S>>,
}

impl<S: Scalar> SensitivityResult<S> {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            sensitivities: Vec::new(),
        }
    }

    /// Add a time point with state and sensitivity.
    pub fn push(&mut self, t: S, y: Vec<S>, sens: Vec<S>) {
        self.times.push(t);
        self.states.push(y);
        self.sensitivities.push(sens);
    }

    /// Get final state.
    pub fn final_state(&self) -> Option<&Vec<S>> {
        self.states.last()
    }

    /// Get final sensitivity matrix.
    pub fn final_sensitivity(&self) -> Option<&Vec<S>> {
        self.sensitivities.last()
    }

    /// Get sensitivity of state i w.r.t. parameter j at final time.
    pub fn final_sensitivity_component(&self, i: usize, j: usize, n_params: usize) -> Option<S> {
        self.sensitivities.last().map(|s| s[i * n_params + j])
    }
}

impl<S: Scalar> Default for SensitivityResult<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple exponential decay: dy/dt = -k*y
    struct ExponentialDecay {
        k: f64,
    }

    impl SensitivityEquations<f64> for ExponentialDecay {
        fn n_states(&self) -> usize {
            1
        }
        fn n_params(&self) -> usize {
            1
        }

        fn eval(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -self.k * y[0];
        }

        fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
            jac[0] = -self.k;
        }

        fn param_jacobian(&self, _t: f64, y: &[f64], dpdy: &mut [f64]) {
            // ∂(-k*y)/∂k = -y
            dpdy[0] = -y[0];
        }
    }

    #[test]
    fn test_sensitivity_state() {
        let mut state = SensitivityState::<f64>::new(2, 3);
        assert_eq!(state.y.len(), 2);
        assert_eq!(state.sensitivity.len(), 6);

        state.set_sensitivity(0, 1, 3, 1.5);
        assert!((state.get_sensitivity(0, 1, 3) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_augmented_system() {
        let system = ExponentialDecay { k: 0.5 };
        let aug_sys = AugmentedSystem::new(system);

        assert_eq!(aug_sys.augmented_dim(), 2); // 1 state + 1 sensitivity

        // Initial state y0 = 1.0
        let aug0 = aug_sys.initial_augmented(&[1.0]);
        assert_eq!(aug0.len(), 2);
        assert!((aug0[0] - 1.0).abs() < 1e-10);
        assert!(aug0[1].abs() < 1e-10); // Initial sensitivity = 0
    }

    #[test]
    fn test_augmented_eval() {
        let system = ExponentialDecay { k: 0.5 };
        let mut aug_sys = AugmentedSystem::new(system);

        // State: y = 2.0, sensitivity S = 1.0
        let aug = vec![2.0, 1.0];
        let mut daug = vec![0.0; 2];

        aug_sys.eval_augmented(0.0, &aug, &mut daug);

        // dy/dt = -k*y = -0.5 * 2 = -1.0
        assert!((daug[0] + 1.0).abs() < 1e-10);

        // dS/dt = (∂f/∂y)*S + ∂f/∂p = (-k)*S + (-y) = -0.5*1.0 + (-2.0) = -2.5
        assert!((daug[1] + 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_sensitivity_result() {
        let mut result = SensitivityResult::<f64>::new();
        result.push(0.0, vec![1.0], vec![0.0]);
        result.push(1.0, vec![0.5], vec![-0.5]);

        assert_eq!(result.times.len(), 2);
        assert!((result.final_state().unwrap()[0] - 0.5).abs() < 1e-10);
        assert!((result.final_sensitivity_component(0, 0, 1).unwrap() + 0.5).abs() < 1e-10);
    }

    // Two-state system: Lotka-Volterra
    struct LotkaVolterra {
        alpha: f64,
        beta: f64,
        delta: f64,
        gamma: f64,
    }

    impl SensitivityEquations<f64> for LotkaVolterra {
        fn n_states(&self) -> usize {
            2
        }
        fn n_params(&self) -> usize {
            4
        }

        fn eval(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            let x = y[0]; // prey
            let y_pred = y[1]; // predator
            dydt[0] = self.alpha * x - self.beta * x * y_pred;
            dydt[1] = self.delta * x * y_pred - self.gamma * y_pred;
        }

        fn jacobian(&self, _t: f64, y: &[f64], jac: &mut [f64]) {
            let x = y[0];
            let y_pred = y[1];
            // ∂f0/∂x, ∂f0/∂y
            jac[0] = self.alpha - self.beta * y_pred;
            jac[1] = -self.beta * x;
            // ∂f1/∂x, ∂f1/∂y
            jac[2] = self.delta * y_pred;
            jac[3] = self.delta * x - self.gamma;
        }

        fn param_jacobian(&self, _t: f64, y: &[f64], dpdy: &mut [f64]) {
            let x = y[0];
            let y_pred = y[1];
            // ∂f0/∂alpha, ∂f0/∂beta, ∂f0/∂delta, ∂f0/∂gamma
            dpdy[0] = x;
            dpdy[1] = -x * y_pred;
            dpdy[2] = 0.0;
            dpdy[3] = 0.0;
            // ∂f1/∂alpha, ∂f1/∂beta, ∂f1/∂delta, ∂f1/∂gamma
            dpdy[4] = 0.0;
            dpdy[5] = 0.0;
            dpdy[6] = x * y_pred;
            dpdy[7] = -y_pred;
        }
    }

    #[test]
    fn test_lotka_volterra_sensitivity() {
        let system = LotkaVolterra {
            alpha: 1.0,
            beta: 0.1,
            delta: 0.075,
            gamma: 1.5,
        };
        let aug_sys = AugmentedSystem::new(system);

        assert_eq!(aug_sys.augmented_dim(), 10); // 2 states + 2*4 sensitivities

        let aug0 = aug_sys.initial_augmented(&[10.0, 5.0]);
        assert_eq!(aug0.len(), 10);
    }
}
