//! Output-grid emission for `SolverOptions::t_eval`.
//!
//! When the caller supplies an explicit output grid, the solver must return
//! `(t, y)` pairs at exactly those times — not at the natural step grid the
//! adaptive controller picked. This module provides a small per-step helper
//! that all solvers wire into their accept branch:
//!
//! 1. Before integration starts, build a [`TEvalEmitter`] from the requested
//!    grid and the integration direction.
//! 2. After each accepted step, call [`TEvalEmitter::emit_step`] with the
//!    step endpoints `(t_old, y_old, dy_old)` and `(t_new, y_new, dy_new)`.
//!    Any requested grid points in `[t_old, t_new]` are flushed to the
//!    solver's `t_out` / `y_out` buffers via Hermite cubic interpolation.
//!
//! The Hermite cubic basis reproduces the endpoints exactly (theta=0 → y_old,
//! theta=1 → y_new), so requested points that coincide with step boundaries
//! come back bit-exact. For points strictly inside a step the interpolation
//! is third-order accurate in `h`, which is well below the working tolerance
//! of every solver in this crate at the step sizes the controller picks.
//!
//! Why Hermite and not each solver's native dense interpolant? Only DoPri5
//! and Verner currently expose dedicated dense-output coefficients, while
//! Tsit5 / BDF / Radau5 / ESDIRK don't. A single Hermite path keeps the
//! contract identical across solvers and makes the wiring auditable in one
//! place. If a higher-order interpolant is needed for a specific solver
//! later, that's a localised optimisation — not a correctness change.

use numra_core::Scalar;

use crate::error::SolverError;

/// Per-step emitter that flushes requested-grid points covered by the step.
///
/// The grid is consumed in order. `direction` is `+S::ONE` for forward
/// integration and `-S::ONE` for backward integration.
pub struct TEvalEmitter<'a, S: Scalar> {
    points: &'a [S],
    idx: usize,
    direction: S,
}

impl<'a, S: Scalar> TEvalEmitter<'a, S> {
    pub fn new(points: &'a [S], direction: S) -> Self {
        Self {
            points,
            idx: 0,
            direction,
        }
    }

    /// `true` if every requested grid point has been emitted.
    pub fn is_done(&self) -> bool {
        self.idx >= self.points.len()
    }

    /// Flush all requested grid points lying in `[t_old, t_new]` (in the
    /// integration direction) using Hermite cubic interpolation between the
    /// step endpoints.
    pub fn emit_step(
        &mut self,
        t_old: S,
        y_old: &[S],
        dy_old: &[S],
        t_new: S,
        y_new: &[S],
        dy_new: &[S],
        t_out: &mut Vec<S>,
        y_out: &mut Vec<S>,
    ) {
        let dim = y_old.len();
        let h = t_new - t_old;
        // Avoid divide-by-zero on a zero-length step (e.g. t0 == tf): just
        // emit any grid points that match the (shared) endpoint exactly.
        if h == S::ZERO {
            while self.idx < self.points.len() && self.points[self.idx] == t_old {
                t_out.push(t_old);
                y_out.extend_from_slice(y_old);
                self.idx += 1;
            }
            return;
        }

        while self.idx < self.points.len() {
            let t_q = self.points[self.idx];
            let in_step = if self.direction > S::ZERO {
                t_q >= t_old && t_q <= t_new
            } else {
                t_q <= t_old && t_q >= t_new
            };
            if !in_step {
                break;
            }

            let theta = (t_q - t_old) / h;
            let theta2 = theta * theta;
            let theta3 = theta2 * theta;
            let three = S::from_f64(3.0);
            let h00 = S::TWO * theta3 - three * theta2 + S::ONE;
            let h10 = theta3 - S::TWO * theta2 + theta;
            let h01 = -S::TWO * theta3 + three * theta2;
            let h11 = theta3 - theta2;

            t_out.push(t_q);
            for i in 0..dim {
                y_out.push(
                    h00 * y_old[i] + h10 * h * dy_old[i] + h01 * y_new[i] + h11 * h * dy_new[i],
                );
            }
            self.idx += 1;
        }
    }
}

/// Validate that a requested output grid is monotonic in the integration
/// direction and stays within `[t0, tf]`.
///
/// Empty grids are allowed — they simply produce an empty result.
pub fn validate_grid<S: Scalar>(grid: &[S], t0: S, tf: S) -> Result<(), SolverError> {
    if grid.is_empty() {
        return Ok(());
    }
    let direction = if tf >= t0 { S::ONE } else { -S::ONE };
    let (lo, hi) = if direction > S::ZERO {
        (t0, tf)
    } else {
        (tf, t0)
    };
    // Allow a small tolerance on bound checks so callers can pass tf
    // exactly without tripping floating-point round-off.
    let span = (tf - t0).abs();
    let tol = S::EPSILON * S::from_f64(16.0) * (span + S::ONE);
    for window in grid.windows(2) {
        let d = window[1] - window[0];
        if d * direction < -tol {
            return Err(SolverError::Other(
                "t_eval must be sorted in the direction of integration".into(),
            ));
        }
    }
    for &t in grid {
        if t < lo - tol || t > hi + tol {
            return Err(SolverError::Other(format!(
                "t_eval contains {} which lies outside [t0, tf] = [{}, {}]",
                t.to_f64(),
                t0.to_f64(),
                tf.to_f64()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_out_of_range() {
        assert!(validate_grid::<f64>(&[0.0, 1.0, 5.0], 0.0, 4.0).is_err());
        assert!(validate_grid::<f64>(&[-1.0, 0.0, 1.0], 0.0, 4.0).is_err());
    }

    #[test]
    fn validate_rejects_unsorted() {
        assert!(validate_grid::<f64>(&[0.0, 2.0, 1.0], 0.0, 4.0).is_err());
        // Backward direction wants strictly descending.
        assert!(validate_grid::<f64>(&[4.0, 1.0, 2.0], 4.0, 0.0).is_err());
    }

    #[test]
    fn validate_accepts_descending_for_backward() {
        assert!(validate_grid::<f64>(&[4.0, 3.0, 2.0, 1.0, 0.0], 4.0, 0.0).is_ok());
    }

    #[test]
    fn emit_reproduces_endpoints_exactly() {
        // Linear y(t) = t with dy/dt = 1. Hermite cubic on (0,0,1)→(1,1,1)
        // is exact for any cubic, so y(0.5) = 0.5 exactly.
        let grid = vec![0.0, 0.5, 1.0];
        let mut emitter = TEvalEmitter::new(&grid, 1.0_f64);

        let mut t_out = Vec::new();
        let mut y_out = Vec::new();

        emitter.emit_step(
            0.0,
            &[0.0],
            &[1.0],
            1.0,
            &[1.0],
            &[1.0],
            &mut t_out,
            &mut y_out,
        );

        assert_eq!(t_out, vec![0.0, 0.5, 1.0]);
        assert!((y_out[0] - 0.0).abs() < 1e-15);
        assert!((y_out[1] - 0.5).abs() < 1e-15);
        assert!((y_out[2] - 1.0).abs() < 1e-15);
        assert!(emitter.is_done());
    }

    #[test]
    fn emit_advances_across_step_boundary_without_double_count() {
        // Grid point exactly on the boundary between two steps must be
        // emitted exactly once (by the first step that reaches it).
        let grid = vec![1.0, 2.0];
        let mut emitter = TEvalEmitter::new(&grid, 1.0_f64);
        let mut t_out = Vec::new();
        let mut y_out = Vec::new();

        // Step 1: [0, 1]. Should emit grid[0]=1.0 once.
        emitter.emit_step(
            0.0,
            &[0.0],
            &[1.0],
            1.0,
            &[1.0],
            &[1.0],
            &mut t_out,
            &mut y_out,
        );
        assert_eq!(t_out.len(), 1);
        // Step 2: [1, 2]. Should emit grid[1]=2.0 once (not grid[0] again).
        emitter.emit_step(
            1.0,
            &[1.0],
            &[1.0],
            2.0,
            &[2.0],
            &[1.0],
            &mut t_out,
            &mut y_out,
        );
        assert_eq!(t_out, vec![1.0, 2.0]);
    }

    #[test]
    fn emit_handles_backward_direction() {
        let grid = vec![1.0, 0.5, 0.0];
        let mut emitter = TEvalEmitter::new(&grid, -1.0_f64);
        let mut t_out = Vec::new();
        let mut y_out = Vec::new();

        emitter.emit_step(
            1.0,
            &[1.0],
            &[1.0],
            0.0,
            &[0.0],
            &[1.0],
            &mut t_out,
            &mut y_out,
        );
        assert_eq!(t_out, vec![1.0, 0.5, 0.0]);
        assert!((y_out[1] - 0.5).abs() < 1e-15);
    }
}
