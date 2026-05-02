//! Integration smoke tests: adaptive `quad` vs analytic integrals.
//!
//! Author: Moussa Leblouba
//! Date: 26 February 2026
//! Modified: 2 May 2026

use numra_integrate::{quad, trapezoid, QuadOptions};

#[test]
fn quad_polynomial_on_unit_interval() {
    let opts = QuadOptions::<f64>::default();
    // ∫_0^1 x^2 dx = 1/3
    let r = quad(|x: f64| x * x, 0.0, 1.0, &opts).expect("quad");
    assert!(
        (r.value - 1.0 / 3.0).abs() < 1e-9,
        "value {} err_est {}",
        r.value,
        r.error_estimate
    );
}

#[test]
fn trapezoid_matches_quad_on_smooth_function() {
    let n = 100usize;
    let dx = 1.0 / n as f64;
    let ys: Vec<f64> = (0..=n)
        .map(|i| {
            let x = i as f64 / n as f64;
            (std::f64::consts::PI * x).cos()
        })
        .collect();
    let trap = trapezoid(&ys, dx);
    let opts = QuadOptions::<f64>::default();
    let q = quad(|x: f64| (std::f64::consts::PI * x).cos(), 0.0, 1.0, &opts).expect("quad");
    assert!(
        (trap - q.value).abs() < 5e-4,
        "trap {} quad {}",
        trap,
        q.value
    );
}
