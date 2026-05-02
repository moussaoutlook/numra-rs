//! Step size control for ODE solvers.
//!
//! This module provides step size controllers for adaptive ODE methods.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Result of step size proposal.
#[derive(Clone, Debug)]
pub struct StepProposal<S: Scalar> {
    /// Proposed new step size
    pub h_new: S,
    /// Should the current step be accepted?
    pub accept: bool,
}

/// Trait for step size controllers.
pub trait StepController<S: Scalar> {
    /// Propose a new step size based on the error estimate.
    ///
    /// # Arguments
    /// * `h` - Current step size
    /// * `err` - Error estimate (scaled, should be < 1 for acceptance)
    /// * `order` - Order of the method
    fn propose(&mut self, h: S, err: S, order: usize) -> StepProposal<S>;

    /// Called when a step is accepted.
    fn accept(&mut self, h: S, err: S);

    /// Called when a step is rejected.
    fn reject(&mut self, h: S, err: S);

    /// Reset the controller state.
    fn reset(&mut self);
}

/// PI (Proportional-Integral) step size controller.
///
/// This is the standard controller used in most production ODE codes.
/// It provides smoother step size changes than a simple error-based controller.
///
/// The new step size is computed as:
/// h_new = h * (1/err)^(beta1/p) * (err_old/err)^(beta2/p)
///
/// where p is the method order.
#[derive(Clone, Debug)]
pub struct PIController<S: Scalar> {
    /// Safety factor (typically 0.9)
    pub safety: S,
    /// Minimum step size factor
    pub fac_min: S,
    /// Maximum step size factor
    pub fac_max: S,
    /// I-controller gain (typically 0.7/p)
    pub beta1: S,
    /// P-controller gain (typically 0.4/p)
    pub beta2: S,
    /// Previous error estimate
    err_old: Option<S>,
    /// Previous step size
    h_old: Option<S>,
}

impl<S: Scalar> Default for PIController<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> PIController<S> {
    /// Create a new PI controller with default parameters.
    pub fn new() -> Self {
        Self {
            safety: S::from_f64(0.9),
            fac_min: S::from_f64(0.2),
            fac_max: S::from_f64(10.0),
            beta1: S::from_f64(0.7),
            beta2: S::from_f64(0.4),
            err_old: None,
            h_old: None,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(safety: S, fac_min: S, fac_max: S, beta1: S, beta2: S) -> Self {
        Self {
            safety,
            fac_min,
            fac_max,
            beta1,
            beta2,
            err_old: None,
            h_old: None,
        }
    }

    /// Create a controller tuned for a specific method order.
    pub fn for_order(order: usize) -> Self {
        let p = S::from_usize(order);
        Self {
            safety: S::from_f64(0.9),
            fac_min: S::from_f64(0.2),
            fac_max: S::from_f64(10.0),
            beta1: S::from_f64(0.7) / p,
            beta2: S::from_f64(0.4) / p,
            err_old: None,
            h_old: None,
        }
    }
}

impl<S: Scalar> StepController<S> for PIController<S> {
    fn propose(&mut self, h: S, err: S, order: usize) -> StepProposal<S> {
        let p = S::from_usize(order);

        // Avoid division by zero
        let err_safe = err.max(S::EPSILON * S::from_f64(100.0));

        // Step size factor depends on whether we have previous error (PI mode)
        let fac = match self.err_old {
            Some(err_old) if err_old > S::ZERO => {
                // Full PI controller: h_new = h * safety * (1/err)^beta1 * (err_old/err)^beta2
                let err_old_safe = err_old.max(S::EPSILON * S::from_f64(100.0));
                self.safety
                    * err_safe.powf(-self.beta1)
                    * (err_old_safe / err_safe).powf(self.beta2)
            }
            _ => {
                // First step: use standard I-controller formula (1/err)^(1/p)
                self.safety * err_safe.powf(-S::ONE / p)
            }
        };

        // Apply safety limits
        let fac = fac.clamp(self.fac_min, self.fac_max);

        let h_new = h * fac;
        let accept = err <= S::ONE;

        StepProposal { h_new, accept }
    }

    fn accept(&mut self, h: S, err: S) {
        self.err_old = Some(err);
        self.h_old = Some(h);
    }

    fn reject(&mut self, _h: S, _err: S) {
        // On rejection, reset the PI memory to avoid oscillations
        self.err_old = None;
    }

    fn reset(&mut self) {
        self.err_old = None;
        self.h_old = None;
    }
}

/// Simple I-controller (no proportional term).
///
/// Simpler but can cause step size oscillations.
#[derive(Clone, Debug)]
pub struct IController<S: Scalar> {
    pub safety: S,
    pub fac_min: S,
    pub fac_max: S,
}

impl<S: Scalar> Default for IController<S> {
    fn default() -> Self {
        Self {
            safety: S::from_f64(0.9),
            fac_min: S::from_f64(0.2),
            fac_max: S::from_f64(10.0),
        }
    }
}

impl<S: Scalar> StepController<S> for IController<S> {
    fn propose(&mut self, h: S, err: S, order: usize) -> StepProposal<S> {
        let p = S::from_usize(order);
        let err_safe = err.max(S::EPSILON * S::from_f64(100.0));

        let fac = self.safety * err_safe.powf(-S::ONE / p);
        let fac = fac.clamp(self.fac_min, self.fac_max);

        let h_new = h * fac;
        let accept = err <= S::ONE;

        StepProposal { h_new, accept }
    }

    fn accept(&mut self, _h: S, _err: S) {}
    fn reject(&mut self, _h: S, _err: S) {}
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_controller_accept() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Error = 0.5, should accept and increase step
        let prop = ctrl.propose(0.1, 0.5, 5);
        assert!(prop.accept);
        assert!(prop.h_new > 0.1); // Step should increase
    }

    #[test]
    fn test_pi_controller_reject() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Error = 2.0, should reject and decrease step
        let prop = ctrl.propose(0.1, 2.0, 5);
        assert!(!prop.accept);
        assert!(prop.h_new < 0.1); // Step should decrease
    }

    #[test]
    fn test_pi_controller_limits() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Very small error, step increase should be limited
        let prop = ctrl.propose(0.1, 1e-10, 5);
        assert!(prop.accept);
        assert!(prop.h_new <= 0.1 * ctrl.fac_max);

        // Very large error, step decrease should be limited
        let prop = ctrl.propose(0.1, 1e10, 5);
        assert!(!prop.accept);
        assert!(prop.h_new >= 0.1 * ctrl.fac_min);
    }

    #[test]
    fn test_i_controller() {
        let mut ctrl = IController::<f64>::default();

        // Error = 0.5, should accept
        let prop = ctrl.propose(0.1, 0.5, 5);
        assert!(prop.accept);
        assert!(prop.h_new > 0.1);

        // Error = 2.0, should reject
        let prop = ctrl.propose(0.1, 2.0, 5);
        assert!(!prop.accept);
        assert!(prop.h_new < 0.1);
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    #[test]
    fn test_pi_controller_exact_one_error() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Error exactly 1.0, should accept but not change step much
        let prop = ctrl.propose(0.1, 1.0, 5);
        assert!(prop.accept);
        // h_new should be close to h (within safety factor)
        assert!((prop.h_new - 0.1).abs() < 0.05);
    }

    #[test]
    fn test_pi_controller_zero_error() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Very small error, should still be limited by fac_max
        let prop = ctrl.propose(0.1, 1e-15, 5);
        assert!(prop.accept);
        assert!(prop.h_new <= 0.1 * ctrl.fac_max);
    }

    #[test]
    fn test_pi_controller_reset() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Accept a step to set err_old
        let _ = ctrl.propose(0.1, 0.5, 5);
        ctrl.accept(0.1, 0.5);

        // Reset should clear the memory
        ctrl.reset();

        // Next proposal should use I-controller formula (no memory)
        let prop = ctrl.propose(0.1, 0.5, 5);
        assert!(prop.accept);
    }

    #[test]
    fn test_pi_controller_consecutive_accepts() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Multiple consecutive accepted steps
        for _ in 0..5 {
            let prop = ctrl.propose(0.1, 0.5, 5);
            assert!(prop.accept);
            ctrl.accept(0.1, 0.5);
        }
    }

    #[test]
    fn test_pi_controller_reject_clears_memory() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Accept to set memory
        let _ = ctrl.propose(0.1, 0.5, 5);
        ctrl.accept(0.1, 0.5);

        // Reject should clear memory
        let _ = ctrl.propose(0.1, 5.0, 5);
        ctrl.reject(0.1, 5.0);

        // err_old should be None now
        assert!(ctrl.err_old.is_none());
    }

    #[test]
    fn test_i_controller_stateless() {
        let mut ctrl = IController::<f64>::default();

        // I-controller should give same result regardless of history
        let prop1 = ctrl.propose(0.1, 0.5, 5);
        ctrl.accept(0.1, 0.5);

        let prop2 = ctrl.propose(0.1, 0.5, 5);

        assert!((prop1.h_new - prop2.h_new).abs() < 1e-15);
    }

    #[test]
    fn test_controller_with_order_1() {
        let mut ctrl = PIController::<f64>::for_order(1);

        // First order methods should still work
        let prop = ctrl.propose(0.1, 0.5, 1);
        assert!(prop.accept);
    }

    #[test]
    fn test_controller_large_error_multiple() {
        let mut ctrl = PIController::<f64>::for_order(5);

        // Multiple rejections in a row
        for _ in 0..3 {
            let prop = ctrl.propose(0.1, 100.0, 5);
            assert!(!prop.accept);
            // Step should be reduced but not below fac_min
            assert!(prop.h_new >= 0.1 * ctrl.fac_min);
            ctrl.reject(0.1, 100.0);
        }
    }

    #[test]
    fn test_step_proposal_fields() {
        let prop = StepProposal {
            h_new: 0.05,
            accept: false,
        };
        assert!((prop.h_new - 0.05).abs() < 1e-15);
        assert!(!prop.accept);
    }

    #[test]
    fn test_custom_params() {
        let ctrl = PIController::<f64>::with_params(
            0.8, // safety
            0.1, // fac_min
            5.0, // fac_max
            0.5, // beta1
            0.3, // beta2
        );

        assert!((ctrl.safety - 0.8).abs() < 1e-15);
        assert!((ctrl.fac_min - 0.1).abs() < 1e-15);
        assert!((ctrl.fac_max - 5.0).abs() < 1e-15);
    }
}
