//! Event detection and handling for ODE solvers.
//!
//! This module provides zero-crossing detection (event detection) during ODE integration.
//! Events are defined by a function g(t, y) that crosses zero. When a sign change is
//! detected between steps, bisection is used to locate the precise crossing time.
//!
//! # Example
//!
//! ```rust
//! use numra_ode::events::{EventFunction, EventDirection, EventAction};
//!
//! struct GroundContact;
//!
//! impl EventFunction<f64> for GroundContact {
//!     fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
//!         y[0] // Event when height = 0
//!     }
//!     fn direction(&self) -> EventDirection {
//!         EventDirection::Falling // Only detect when falling through zero
//!     }
//!     fn action(&self) -> EventAction {
//!         EventAction::Stop // Stop integration at event
//!     }
//! }
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Tolerance for event function value considered close enough to zero.
// Note: f64-specific tolerances; for f32 use cases these should be scaled to ~1e-6
const EVENT_ZERO_TOL: f64 = 1e-12;
/// Convergence tolerance for bisection interval width.
// Note: f64-specific tolerances; for f32 use cases these should be scaled to ~1e-6
const EVENT_BISECTION_TOL: f64 = 1e-13;
/// Maximum number of bisection iterations for event location.
const EVENT_MAX_BISECTION: usize = 50;

/// Direction of zero-crossing to detect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventDirection {
    /// Detect when g goes from negative to positive.
    Rising,
    /// Detect when g goes from positive to negative.
    Falling,
    /// Detect zero-crossings in either direction.
    Both,
}

/// Action to take when an event is detected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventAction {
    /// Stop integration at the event time.
    Stop,
    /// Record the event and continue integration.
    Continue,
}

/// Trait for event functions.
///
/// An event function defines a scalar function g(t, y) whose zero-crossings
/// are to be detected during integration.
pub trait EventFunction<S: Scalar>: Send + Sync {
    /// Evaluate the event function g(t, y).
    ///
    /// An event occurs when this function crosses zero.
    fn evaluate(&self, t: S, y: &[S]) -> S;

    /// Direction of zero-crossing to detect.
    ///
    /// Defaults to `Both` (detect crossings in either direction).
    fn direction(&self) -> EventDirection {
        EventDirection::Both
    }

    /// Action to take when event is detected.
    ///
    /// Defaults to `Continue` (record and continue integration).
    fn action(&self) -> EventAction {
        EventAction::Continue
    }
}

/// Recorded event information.
#[derive(Debug, Clone)]
pub struct Event<S: Scalar> {
    /// Time at which the event occurred.
    pub t: S,
    /// State at the event time.
    pub y: Vec<S>,
    /// Index of the event function that triggered this event.
    pub event_index: usize,
}

/// Find the time of a zero-crossing of the event function in [t_lo, t_hi] using bisection.
///
/// Uses the provided interpolation function to evaluate the state at intermediate times.
/// Returns `Some((t_event, y_event))` if a zero-crossing matching the requested direction
/// is found, or `None` if no valid crossing exists.
///
/// # Arguments
///
/// * `event` - The event function to check
/// * `t_lo` - Start time of the interval
/// * `y_lo` - State at t_lo
/// * `t_hi` - End time of the interval
/// * `y_hi` - State at t_hi
/// * `interpolate` - Function that returns the interpolated state at any time in [t_lo, t_hi]
pub fn find_event_time<S: Scalar>(
    event: &dyn EventFunction<S>,
    t_lo: S,
    y_lo: &[S],
    t_hi: S,
    y_hi: &[S],
    interpolate: &dyn Fn(S) -> Vec<S>,
) -> Option<(S, Vec<S>)> {
    let g_lo = event.evaluate(t_lo, y_lo);
    let g_hi = event.evaluate(t_hi, y_hi);

    // Check for sign change
    if g_lo * g_hi > S::ZERO {
        return None;
    }

    // Check direction constraint
    match event.direction() {
        EventDirection::Rising if g_hi <= g_lo => return None,
        EventDirection::Falling if g_hi >= g_lo => return None,
        _ => {}
    }

    // Bisection to locate the zero-crossing
    let mut t_a = t_lo;
    let mut t_b = t_hi;
    let mut y_a = y_lo.to_vec();
    let zero_tol = S::from_f64(EVENT_ZERO_TOL);
    let bisect_tol = S::from_f64(EVENT_BISECTION_TOL);

    for _ in 0..EVENT_MAX_BISECTION {
        let t_mid = (t_a + t_b) / S::TWO;
        let y_mid = interpolate(t_mid);
        let g_mid = event.evaluate(t_mid, &y_mid);

        if g_mid.abs() < zero_tol {
            return Some((t_mid, y_mid));
        }

        let g_a = event.evaluate(t_a, &y_a);
        if g_mid * g_a < S::ZERO {
            t_b = t_mid;
        } else {
            t_a = t_mid;
            y_a = y_mid;
        }

        // Check convergence by interval width
        if (t_b - t_a).abs() < bisect_tol {
            break;
        }
    }

    let t_root = (t_a + t_b) / S::TWO;
    let y_root = interpolate(t_root);
    Some((t_root, y_root))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_direction_default() {
        struct MyEvent;
        impl EventFunction<f64> for MyEvent {
            fn evaluate(&self, _t: f64, _y: &[f64]) -> f64 {
                0.0
            }
        }
        let e = MyEvent;
        assert_eq!(e.direction(), EventDirection::Both);
        assert_eq!(e.action(), EventAction::Continue);
    }

    #[test]
    fn test_find_event_time_simple() {
        // g(t, y) = y[0], linear interpolation from y=1 at t=0 to y=-1 at t=2
        // Zero crossing at t=1
        struct ZeroCross;
        impl EventFunction<f64> for ZeroCross {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
        }

        let event = ZeroCross;
        let t_lo = 0.0;
        let y_lo = [1.0];
        let t_hi = 2.0;
        let y_hi = [-1.0];

        let interpolate = |t: f64| -> Vec<f64> {
            vec![1.0 - t] // linear: y = 1 - t
        };

        let result = find_event_time(&event, t_lo, &y_lo, t_hi, &y_hi, &interpolate);
        assert!(result.is_some());
        let (t_event, y_event) = result.unwrap();
        assert!(
            (t_event - 1.0).abs() < 1e-10,
            "Expected t=1, got t={}",
            t_event
        );
        assert!(y_event[0].abs() < 1e-10);
    }

    #[test]
    fn test_find_event_time_direction_filter() {
        // g goes from +1 to -1 (falling), but we request Rising => should return None
        struct RisingOnly;
        impl EventFunction<f64> for RisingOnly {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
            fn direction(&self) -> EventDirection {
                EventDirection::Rising
            }
        }

        let event = RisingOnly;
        let interpolate = |t: f64| -> Vec<f64> { vec![1.0 - t] };

        let result = find_event_time(&event, 0.0, &[1.0], 2.0, &[-1.0], &interpolate);
        assert!(
            result.is_none(),
            "Rising event should not trigger on falling crossing"
        );
    }

    #[test]
    fn test_no_sign_change() {
        struct MyEvent;
        impl EventFunction<f64> for MyEvent {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
        }

        let event = MyEvent;
        let interpolate = |t: f64| -> Vec<f64> { vec![1.0 + t] };

        let result = find_event_time(&event, 0.0, &[1.0], 2.0, &[3.0], &interpolate);
        assert!(result.is_none(), "No sign change should return None");
    }
}
