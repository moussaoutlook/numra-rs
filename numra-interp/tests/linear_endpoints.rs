//! Interpolation invariants: endpoints and mid-segment values.
//!
//! Author: Moussa Leblouba
//! Date: 26 April 2026
//! Modified: 2 May 2026

use numra_interp::{Interpolant, Linear};

#[test]
fn linear_hits_knots() {
    let x = [0.0_f64, 1.0, 3.0];
    let y = [1.0_f64, 0.0, 9.0];
    let lin = Linear::new(&x, &y).expect("new");
    for i in 0..x.len() {
        let v = lin.interpolate(x[i]);
        assert!((v - y[i]).abs() < 1e-12, "at knot {} got {}", i, v);
    }
}

#[test]
fn linear_midpoint() {
    let lin = Linear::new(&[0.0_f64, 2.0], &[0.0_f64, 4.0]).expect("new");
    assert!((lin.interpolate(1.0) - 2.0).abs() < 1e-12);
}
