// tests/scalar_renamed.rs
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026
use numra_core::Scalar;

#[test]
fn test_gamma_fn_renamed() {
    let x: f64 = 5.0;
    // gamma(5) = 4! = 24
    let result = x.gamma_fn();
    assert!((result - 24.0).abs() < 1e-10);
}

#[test]
fn test_erf_fn_renamed() {
    let x: f64 = 1.0;
    // erf(1) ≈ 0.8427
    let result = x.erf_fn();
    assert!((result - 0.8427007929497149).abs() < 1e-10);
}

#[test]
fn test_erfc_fn_renamed() {
    let x: f64 = 1.0;
    // erfc(1) = 1 - erf(1) ≈ 0.1573
    let result = x.erfc_fn();
    assert!((result - 0.15729920705028513).abs() < 1e-10);
}

#[test]
fn test_gamma_fn_f32() {
    let x: f32 = 5.0;
    // gamma(5) = 4! = 24
    let result = x.gamma_fn();
    assert!((result - 24.0).abs() < 1e-5);
}

#[test]
fn test_erf_fn_f32() {
    let x: f32 = 1.0;
    // erf(1) ≈ 0.8427
    let result = x.erf_fn();
    assert!((result - 0.8427008).abs() < 1e-5);
}
