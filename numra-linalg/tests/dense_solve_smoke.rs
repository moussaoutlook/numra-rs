//! Dense linear solve smoke test (LU path).
//!
//! Author: Moussa Leblouba
//! Date: 26 April 2026
//! Modified: 2 May 2026

use numra_linalg::{DenseMatrix, Matrix};

#[test]
fn solve_diagonal_system() {
    let mut a = DenseMatrix::<f64>::zeros(3, 3);
    a.set(0, 0, 2.0);
    a.set(1, 1, 4.0);
    a.set(2, 2, 6.0);
    let b = vec![1.0_f64, 2.0, 3.0];
    let x = a.solve(&b).expect("solve");
    assert!((x[0] - 0.5).abs() < 1e-12);
    assert!((x[1] - 0.5).abs() < 1e-12);
    assert!((x[2] - 0.5).abs() < 1e-12);
}
