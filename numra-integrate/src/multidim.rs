//! Multi-dimensional integration.
//!
//! Provides double integral (`dblquad`) via iterated 1D adaptive quadrature.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::adaptive::{quad, QuadOptions, QuadResult};
use crate::error::IntegrationError;

/// Double integral over a region defined by variable y-limits.
///
/// Computes `integral_{xa}^{xb} integral_{ya(x)}^{yb(x)} f(x, y) dy dx`
/// using iterated 1D adaptive quadrature.
///
/// # Arguments
///
/// * `f` - Integrand `f(x, y)`
/// * `xa`, `xb` - Outer integration limits for x
/// * `ya` - Lower y-limit as a function of x
/// * `yb` - Upper y-limit as a function of x
/// * `opts` - Quadrature options (applied to both inner and outer integrals)
///
/// # Example
///
/// ```
/// use numra_integrate::{dblquad, QuadOptions};
///
/// // Integral of x*y over the unit square = 0.25
/// let result = dblquad(
///     |x: f64, y: f64| x * y,
///     0.0, 1.0,
///     |_| 0.0, |_| 1.0,
///     &QuadOptions::default(),
/// ).unwrap();
/// assert!((result.value - 0.25).abs() < 1e-10);
/// ```
pub fn dblquad<S, F, G1, G2>(
    f: F,
    xa: S,
    xb: S,
    ya: G1,
    yb: G2,
    opts: &QuadOptions<S>,
) -> Result<QuadResult<S>, IntegrationError>
where
    S: Scalar,
    F: Fn(S, S) -> S,
    G1: Fn(S) -> S,
    G2: Fn(S) -> S,
{
    let mut total_inner_evals = 0usize;

    let outer_result = quad(
        |x: S| {
            let y_lo = ya(x);
            let y_hi = yb(x);
            match quad(|y: S| f(x, y), y_lo, y_hi, opts) {
                Ok(inner) => {
                    total_inner_evals += inner.n_evaluations;
                    inner.value
                }
                Err(_) => S::NAN,
            }
        },
        xa,
        xb,
        opts,
    )?;

    Ok(QuadResult {
        value: outer_result.value,
        error_estimate: outer_result.error_estimate,
        n_evaluations: outer_result.n_evaluations + total_inner_evals,
        n_subdivisions: outer_result.n_subdivisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dblquad_unit_square() {
        // integral of 1 over [0,1] x [0,1] = 1
        let result = dblquad(
            |_x: f64, _y: f64| 1.0,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            &QuadOptions::default(),
        )
        .unwrap();
        assert_relative_eq!(result.value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dblquad_xy() {
        // integral of x*y over [0,1] x [0,1] = 0.25
        let result = dblquad(
            |x: f64, y: f64| x * y,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            &QuadOptions::default(),
        )
        .unwrap();
        assert_relative_eq!(result.value, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_dblquad_triangular_region() {
        // integral of 1 over triangle {0 <= x <= 1, 0 <= y <= x} = 0.5
        let result = dblquad(
            |_x: f64, _y: f64| 1.0,
            0.0,
            1.0,
            |_| 0.0,
            |x| x,
            &QuadOptions::default(),
        )
        .unwrap();
        assert_relative_eq!(result.value, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_dblquad_x_plus_y() {
        // integral of (x+y) over [0,1] x [0,1] = 1
        let result = dblquad(
            |x: f64, y: f64| x + y,
            0.0,
            1.0,
            |_| 0.0,
            |_| 1.0,
            &QuadOptions::default(),
        )
        .unwrap();
        assert_relative_eq!(result.value, 1.0, epsilon = 1e-10);
    }
}
