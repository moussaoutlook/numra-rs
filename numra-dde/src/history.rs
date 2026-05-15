//! History management and interpolation for DDEs.
//!
//! Provides efficient storage and interpolation of solution history
//! for evaluating delayed terms y(t - τ).
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for history functions.
pub trait HistoryFunction<S: Scalar>: Fn(S) -> Vec<S> {}
impl<S: Scalar, F: Fn(S) -> Vec<S>> HistoryFunction<S> for F {}

/// A single step's data for history interpolation.
#[derive(Clone, Debug)]
pub struct HistoryStep<S: Scalar> {
    /// Time at start of step
    pub t: S,
    /// State at start of step
    pub y: Vec<S>,
    /// Derivative at start of step
    pub f: Vec<S>,
    /// Time at end of step
    pub t_next: S,
    /// State at end of step
    pub y_next: Vec<S>,
    /// Derivative at end of step
    pub f_next: Vec<S>,
}

impl<S: Scalar> HistoryStep<S> {
    pub fn new(t: S, y: Vec<S>, f: Vec<S>, t_next: S, y_next: Vec<S>, f_next: Vec<S>) -> Self {
        Self {
            t,
            y,
            f,
            t_next,
            y_next,
            f_next,
        }
    }

    /// Step size h = t_next - t.
    pub fn h(&self) -> S {
        self.t_next - self.t
    }

    /// Check if time t is within this step's interval [t, t_next].
    pub fn contains(&self, t: S) -> bool {
        t >= self.t && t <= self.t_next
    }
}

/// Hermite cubic interpolator for a single step.
///
/// Uses the values and derivatives at both endpoints to construct
/// a cubic polynomial that matches y and y' at t and t_next.
pub struct HermiteInterpolator;

impl HermiteInterpolator {
    /// Interpolate at time t within a step.
    ///
    /// Uses cubic Hermite interpolation:
    /// p(θ) = (1-θ)²(1+2θ) y₀ + θ²(3-2θ) y₁ + h θ(1-θ)² f₀ - h θ²(1-θ) f₁
    ///
    /// where θ = (t - t₀) / h.
    pub fn interpolate<S: Scalar>(step: &HistoryStep<S>, t: S) -> Vec<S> {
        let h = step.h();
        if h.abs() < S::from_f64(1e-15) {
            return step.y.clone();
        }

        let theta = (t - step.t) / h;
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;

        // Hermite basis functions
        let h00 = S::ONE - S::from_f64(3.0) * theta2 + S::from_f64(2.0) * theta3; // 1 - 3θ² + 2θ³
        let h10 = theta - S::from_f64(2.0) * theta2 + theta3; // θ - 2θ² + θ³
        let h01 = S::from_f64(3.0) * theta2 - S::from_f64(2.0) * theta3; // 3θ² - 2θ³
        let h11 = -theta2 + theta3; // -θ² + θ³

        let dim = step.y.len();
        let mut result = vec![S::ZERO; dim];

        for i in 0..dim {
            result[i] = h00 * step.y[i]
                + h10 * h * step.f[i]
                + h01 * step.y_next[i]
                + h11 * h * step.f_next[i];
        }

        result
    }

    /// Interpolate derivative at time t within a step.
    pub fn interpolate_derivative<S: Scalar>(step: &HistoryStep<S>, t: S) -> Vec<S> {
        let h = step.h();
        if h.abs() < S::from_f64(1e-15) {
            return step.f.clone();
        }

        let theta = (t - step.t) / h;
        let theta2 = theta * theta;

        // Derivatives of Hermite basis functions (divided by h)
        let dh00 = -S::from_f64(6.0) * theta + S::from_f64(6.0) * theta2;
        let dh10 = S::ONE - S::from_f64(4.0) * theta + S::from_f64(3.0) * theta2;
        let dh01 = S::from_f64(6.0) * theta - S::from_f64(6.0) * theta2;
        let dh11 = -S::from_f64(2.0) * theta + S::from_f64(3.0) * theta2;

        let dim = step.y.len();
        let mut result = vec![S::ZERO; dim];

        for i in 0..dim {
            result[i] = (dh00 * step.y[i]
                + dh10 * h * step.f[i]
                + dh01 * step.y_next[i]
                + dh11 * h * step.f_next[i])
                / h;
        }

        result
    }
}

/// History storage for DDE solver.
///
/// Stores computed solution steps and provides interpolation
/// for accessing y(t - τ) at arbitrary past times.
pub struct History<S: Scalar, H: Fn(S) -> Vec<S>> {
    /// Initial history function for t ≤ t0
    initial_history: H,
    /// Initial time
    t0: S,
    /// Computed solution steps (sorted by time)
    steps: Vec<HistoryStep<S>>,
    /// Dimension of the state
    dim: usize,
}

impl<S: Scalar, H: Fn(S) -> Vec<S>> History<S, H> {
    /// Create a new history with initial function.
    pub fn new(initial_history: H, t0: S, dim: usize) -> Self {
        Self {
            initial_history,
            t0,
            steps: Vec::new(),
            dim,
        }
    }

    /// Add a completed step to the history.
    pub fn add_step(&mut self, step: HistoryStep<S>) {
        self.steps.push(step);
    }

    /// Get the state at time t using interpolation.
    pub fn evaluate(&self, t: S) -> Vec<S> {
        // If t is before t0, use initial history
        if t <= self.t0 {
            return (self.initial_history)(t);
        }

        // Binary search to find the step containing t
        match self.find_step(t) {
            Some(idx) => HermiteInterpolator::interpolate(&self.steps[idx], t),
            None => {
                // t is after all recorded steps - extrapolate from last
                if let Some(last) = self.steps.last() {
                    if t <= last.t_next {
                        HermiteInterpolator::interpolate(last, t)
                    } else {
                        // Beyond computed region - return last known value
                        last.y_next.clone()
                    }
                } else {
                    // No steps yet - use initial history at t0
                    (self.initial_history)(self.t0)
                }
            }
        }
    }

    /// Get the derivative at time t using interpolation.
    pub fn evaluate_derivative(&self, t: S) -> Vec<S> {
        if t <= self.t0 {
            // Can't easily get derivative from arbitrary history function
            // Use finite difference approximation
            let h = S::EPSILON.cbrt() * (S::ONE + t.abs());
            let y1 = (self.initial_history)(t - h);
            let y2 = (self.initial_history)(t + h);
            let mut deriv = vec![S::ZERO; self.dim];
            for i in 0..self.dim {
                deriv[i] = (y2[i] - y1[i]) / (S::from_f64(2.0) * h);
            }
            return deriv;
        }

        match self.find_step(t) {
            Some(idx) => HermiteInterpolator::interpolate_derivative(&self.steps[idx], t),
            None => {
                if let Some(last) = self.steps.last() {
                    last.f_next.clone()
                } else {
                    vec![S::ZERO; self.dim]
                }
            }
        }
    }

    /// Find the step index containing time t.
    fn find_step(&self, t: S) -> Option<usize> {
        if self.steps.is_empty() {
            return None;
        }

        // Binary search
        let mut lo = 0;
        let mut hi = self.steps.len();

        while lo < hi {
            let mid = (lo + hi) / 2;
            if t < self.steps[mid].t {
                hi = mid;
            } else if t > self.steps[mid].t_next {
                lo = mid + 1;
            } else {
                return Some(mid);
            }
        }

        // Check boundaries
        if lo < self.steps.len() && self.steps[lo].contains(t) {
            Some(lo)
        } else if lo > 0 && self.steps[lo - 1].contains(t) {
            Some(lo - 1)
        } else {
            None
        }
    }

    /// Get the current end time of the computed solution.
    pub fn current_time(&self) -> S {
        self.steps.last().map(|s| s.t_next).unwrap_or(self.t0)
    }

    /// Number of stored steps.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Clear history (keep initial function).
    pub fn clear(&mut self) {
        self.steps.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_interpolation() {
        // Test with known polynomial: y = t², y' = 2t
        // On [0, 1]: y(0)=0, y'(0)=0, y(1)=1, y'(1)=2
        let step = HistoryStep {
            t: 0.0,
            y: vec![0.0],
            f: vec![0.0],
            t_next: 1.0,
            y_next: vec![1.0],
            f_next: vec![2.0],
        };

        // Test at midpoint: y(0.5) should be 0.25
        let y_mid = HermiteInterpolator::interpolate(&step, 0.5);
        assert!((y_mid[0] - 0.25).abs() < 1e-10);

        // Test at t=0.25: y(0.25) = 0.0625
        let y_quarter = HermiteInterpolator::interpolate(&step, 0.25);
        assert!((y_quarter[0] - 0.0625).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_derivative() {
        // Same test: y = t², y' = 2t
        let step = HistoryStep {
            t: 0.0,
            y: vec![0.0],
            f: vec![0.0],
            t_next: 1.0,
            y_next: vec![1.0],
            f_next: vec![2.0],
        };

        // Test derivative at midpoint: y'(0.5) = 1.0
        let f_mid = HermiteInterpolator::interpolate_derivative(&step, 0.5);
        assert!((f_mid[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_history_initial() {
        let history_fn = |t: f64| vec![t.sin()];
        let history = History::new(history_fn, 0.0, 1);

        // Before t0, should use initial history
        let y = history.evaluate(-1.0);
        assert!((y[0] - (-1.0_f64).sin()).abs() < 1e-10);
    }

    #[test]
    fn test_history_with_steps() {
        let history_fn = |_t: f64| vec![1.0];
        let mut history = History::new(history_fn, 0.0, 1);

        // Add a step: y goes from 1 to 2 with derivative going from 1 to 1
        // This models y = 1 + t approximately
        history.add_step(HistoryStep {
            t: 0.0,
            y: vec![1.0],
            f: vec![1.0],
            t_next: 1.0,
            y_next: vec![2.0],
            f_next: vec![1.0],
        });

        // Interpolate at t = 0.5: should be approximately 1.5
        let y = history.evaluate(0.5);
        assert!((y[0] - 1.5).abs() < 0.1); // Hermite won't be exact for linear

        // Before t0
        let y_pre = history.evaluate(-0.5);
        assert!((y_pre[0] - 1.0).abs() < 1e-10);

        // At exact step boundary
        let y_end = history.evaluate(1.0);
        assert!((y_end[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_history_multiple_steps() {
        let history_fn = |_t: f64| vec![0.0];
        let mut history = History::new(history_fn, 0.0, 1);

        // Add multiple steps
        for i in 0..5 {
            let t = i as f64;
            let t_next = (i + 1) as f64;
            history.add_step(HistoryStep {
                t,
                y: vec![t],
                f: vec![1.0],
                t_next,
                y_next: vec![t_next],
                f_next: vec![1.0],
            });
        }

        // Test interpolation at various points
        assert!((history.evaluate(2.5)[0] - 2.5).abs() < 0.1);
        assert!((history.evaluate(4.0)[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_derivative_large_t_no_scaling_bug() {
        // Pins F-FD-NOSCALE-BUG for the initial-history FD branch: with
        // unscaled `h = 1e-8`, both `(t - h)` and `(t + h)` quantize to `t`
        // for |t| > ~5e7 in f64, so the FD returned 0 instead of the
        // analytical derivative. With canonical `cbrt(EPSILON) * (1 + |t|)`
        // the derivative is recovered.
        //
        // Initial history is linear in t: y(t) = t, so dy/dt = 1 exactly.
        let history_fn = |t: f64| vec![t];
        let history = History::new(history_fn, 0.0, 1);

        // Query at t = -1e8 (well below t0 = 0, well above the bug threshold).
        let deriv = history.evaluate_derivative(-1e8);
        assert!(
            (deriv[0] - 1.0).abs() < 1e-3,
            "dy/dt = {} should be ≈ 1.0 (within 1e-3); old unscaled formula returns 0",
            deriv[0]
        );
    }
}
