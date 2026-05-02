//!
//! Author: Moussa Leblouba
//! Date: 26 April 2026
//! Modified: 2 May 2026

use numra_ode::events::{EventAction, EventDirection, EventFunction};
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};

struct ZeroCrossing {
    direction: EventDirection,
    action: EventAction,
}

impl EventFunction<f64> for ZeroCrossing {
    fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
        y[0]
    }

    fn direction(&self) -> EventDirection {
        self.direction
    }

    fn action(&self) -> EventAction {
        self.action
    }
}

#[test]
fn harmonic_zero_crossings_are_recorded_without_terminating() {
    let problem = OdeProblem::new(
        |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1];
            dydt[1] = -y[0];
        },
        0.0,
        10.0,
        vec![1.0, 0.0],
    );
    let options = SolverOptions::default()
        .rtol(1e-9)
        .atol(1e-11)
        .event(Box::new(ZeroCrossing {
            direction: EventDirection::Both,
            action: EventAction::Continue,
        }));

    let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &options).unwrap();

    assert!(!result.terminated_by_event);
    assert!(result.events.len() >= 3, "events={:?}", result.events);
    for (event, expected) in result.events.iter().zip([
        std::f64::consts::FRAC_PI_2,
        3.0 * std::f64::consts::FRAC_PI_2,
        5.0 * std::f64::consts::FRAC_PI_2,
    ]) {
        assert!(
            (event.t - expected).abs() < 2e-2,
            "expected crossing near {expected}, got {}",
            event.t
        );
        assert!(event.y[0].abs() < 2e-4, "event state={:?}", event.y);
    }
}

#[test]
fn terminal_event_stops_at_first_analytic_crossing() {
    let problem = OdeProblem::new(
        |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1];
            dydt[1] = -y[0];
        },
        0.0,
        10.0,
        vec![1.0, 0.0],
    );
    let options = SolverOptions::default()
        .rtol(1e-9)
        .atol(1e-11)
        .event(Box::new(ZeroCrossing {
            direction: EventDirection::Falling,
            action: EventAction::Stop,
        }));

    let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &options).unwrap();

    assert!(result.terminated_by_event);
    assert_eq!(result.events.len(), 1);
    let event = &result.events[0];
    assert!((event.t - std::f64::consts::FRAC_PI_2).abs() < 2e-2);
    assert!(event.y[0].abs() < 2e-4);
    assert!(
        event.y[1] < 0.0,
        "falling crossing should have negative velocity"
    );
}

#[test]
fn near_initial_crossing_is_located_with_small_step() {
    let problem = OdeProblem::new(
        |_t: f64, _y: &[f64], dydt: &mut [f64]| {
            dydt[0] = 1.0;
        },
        0.0,
        1.0,
        vec![-1e-6],
    );
    let options = SolverOptions::default()
        .h0(1e-4)
        .h_max(1e-4)
        .rtol(1e-9)
        .atol(1e-12)
        .event(Box::new(ZeroCrossing {
            direction: EventDirection::Rising,
            action: EventAction::Stop,
        }));

    let result = DoPri5::solve(&problem, 0.0, 1.0, &[-1e-6], &options).unwrap();

    assert!(result.terminated_by_event);
    let event = &result.events[0];
    assert!(
        (event.t - 1e-6).abs() < 1e-8,
        "expected event at 1e-6, got {}",
        event.t
    );
    assert!(event.y[0].abs() < 1e-8);
}
