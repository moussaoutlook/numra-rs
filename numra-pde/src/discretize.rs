//! Finite difference discretization schemes.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::grid::Grid1D;
use numra_core::Scalar;

/// Difference scheme type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DifferenceScheme {
    /// Central differences (second-order accurate)
    Central,
    /// Forward differences (first-order accurate)
    Forward,
    /// Backward differences (first-order accurate)
    Backward,
    /// Fourth-order central differences
    Central4,
}

/// Finite difference stencil coefficients.
#[derive(Clone, Debug)]
pub struct Stencil<S: Scalar> {
    /// Coefficients for each point in the stencil
    pub coeffs: Vec<S>,
    /// Offset of the leftmost point from center
    pub offset: i32,
}

impl<S: Scalar> Stencil<S> {
    /// Create a first derivative central difference stencil.
    pub fn d1_central(dx: S) -> Self {
        let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);
        Self {
            coeffs: vec![-inv_2dx, S::ZERO, inv_2dx],
            offset: -1,
        }
    }

    /// Create a first derivative forward difference stencil.
    pub fn d1_forward(dx: S) -> Self {
        let inv_dx = S::ONE / dx;
        Self {
            coeffs: vec![-inv_dx, inv_dx],
            offset: 0,
        }
    }

    /// Create a first derivative backward difference stencil.
    pub fn d1_backward(dx: S) -> Self {
        let inv_dx = S::ONE / dx;
        Self {
            coeffs: vec![-inv_dx, inv_dx],
            offset: -1,
        }
    }

    /// Create a second derivative central difference stencil.
    pub fn d2_central(dx: S) -> Self {
        let inv_dx2 = S::ONE / (dx * dx);
        Self {
            coeffs: vec![inv_dx2, -S::from_f64(2.0) * inv_dx2, inv_dx2],
            offset: -1,
        }
    }

    /// Create a fourth-order accurate second derivative stencil.
    pub fn d2_central4(dx: S) -> Self {
        let inv_dx2 = S::ONE / (dx * dx);
        let c = inv_dx2 / S::from_f64(12.0);
        Self {
            coeffs: vec![
                -c,
                S::from_f64(16.0) * c,
                -S::from_f64(30.0) * c,
                S::from_f64(16.0) * c,
                -c,
            ],
            offset: -2,
        }
    }

    /// Apply stencil at point i to array u.
    pub fn apply(&self, u: &[S], i: usize) -> S {
        let mut result = S::ZERO;
        for (k, &coeff) in self.coeffs.iter().enumerate() {
            let idx = (i as i32 + self.offset + k as i32) as usize;
            result = result + coeff * u[idx];
        }
        result
    }
}

/// Finite Difference Method utilities.
pub struct FDM;

impl FDM {
    /// Compute first derivative using central differences.
    ///
    /// `du/dx ≈ (u[i+1] - u[i-1]) / (2*dx)`
    pub fn d1_central<S: Scalar>(u: &[S], dx: S, i: usize) -> S {
        let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);
        (u[i + 1] - u[i - 1]) * inv_2dx
    }

    /// Compute first derivative using forward difference.
    ///
    /// `du/dx ≈ (u[i+1] - u[i]) / dx`
    pub fn d1_forward<S: Scalar>(u: &[S], dx: S, i: usize) -> S {
        (u[i + 1] - u[i]) / dx
    }

    /// Compute first derivative using backward difference.
    ///
    /// `du/dx ≈ (u[i] - u[i-1]) / dx`
    pub fn d1_backward<S: Scalar>(u: &[S], dx: S, i: usize) -> S {
        (u[i] - u[i - 1]) / dx
    }

    /// Compute second derivative using central differences.
    ///
    /// `d²u/dx² ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²`
    pub fn d2_central<S: Scalar>(u: &[S], dx: S, i: usize) -> S {
        let inv_dx2 = S::ONE / (dx * dx);
        (u[i + 1] - S::from_f64(2.0) * u[i] + u[i - 1]) * inv_dx2
    }

    /// Compute second derivative using fourth-order central differences.
    ///
    /// `d²u/dx² ≈ (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx²)`
    pub fn d2_central4<S: Scalar>(u: &[S], dx: S, i: usize) -> S {
        let inv_12dx2 = S::ONE / (S::from_f64(12.0) * dx * dx);
        (-u[i + 2] + S::from_f64(16.0) * u[i + 1] - S::from_f64(30.0) * u[i]
            + S::from_f64(16.0) * u[i - 1]
            - u[i - 2])
            * inv_12dx2
    }

    /// Compute first derivative at left boundary using one-sided differences.
    ///
    /// `du/dx ≈ (-3*u[0] + 4*u[1] - u[2]) / (2*dx)`
    pub fn d1_left_boundary<S: Scalar>(u: &[S], dx: S) -> S {
        let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);
        (-S::from_f64(3.0) * u[0] + S::from_f64(4.0) * u[1] - u[2]) * inv_2dx
    }

    /// Compute first derivative at right boundary using one-sided differences.
    ///
    /// `du/dx ≈ (3*u[n-1] - 4*u[n-2] + u[n-3]) / (2*dx)`
    pub fn d1_right_boundary<S: Scalar>(u: &[S], dx: S) -> S {
        let n = u.len();
        let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);
        (S::from_f64(3.0) * u[n - 1] - S::from_f64(4.0) * u[n - 2] + u[n - 3]) * inv_2dx
    }

    /// Apply Laplacian operator to entire field (interior points only).
    ///
    /// Returns vector of size n-2 (excluding boundary points).
    pub fn laplacian_1d<S: Scalar>(u: &[S], dx: S) -> Vec<S> {
        let n = u.len();
        let inv_dx2 = S::ONE / (dx * dx);
        let two = S::from_f64(2.0);

        (1..n - 1)
            .map(|i| (u[i + 1] - two * u[i] + u[i - 1]) * inv_dx2)
            .collect()
    }

    /// Apply Laplacian operator to entire 2D field (interior points only).
    ///
    /// Uses a 5-point stencil.
    pub fn laplacian_2d<S: Scalar>(u: &[S], nx: usize, ny: usize, dx: S, dy: S) -> Vec<S> {
        let inv_dx2 = S::ONE / (dx * dx);
        let inv_dy2 = S::ONE / (dy * dy);
        let two = S::from_f64(2.0);

        let mut result = vec![S::ZERO; (nx - 2) * (ny - 2)];

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let idx = j * nx + i;
                let d2x = (u[idx + 1] - two * u[idx] + u[idx - 1]) * inv_dx2;
                let d2y = (u[idx + nx] - two * u[idx] + u[idx - nx]) * inv_dy2;

                let result_idx = (j - 1) * (nx - 2) + (i - 1);
                result[result_idx] = d2x + d2y;
            }
        }

        result
    }

    /// Compute gradient at all interior points.
    pub fn gradient_1d<S: Scalar>(u: &[S], dx: S) -> Vec<S> {
        let n = u.len();
        let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);

        (1..n - 1)
            .map(|i| (u[i + 1] - u[i - 1]) * inv_2dx)
            .collect()
    }

    /// Build sparse Laplacian matrix for 1D domain with given BCs.
    ///
    /// Returns a tridiagonal matrix in dense form for simplicity.
    /// For large problems, use sparse matrix formats.
    pub fn laplacian_matrix_1d<S: Scalar>(n_interior: usize, dx: S) -> Vec<Vec<S>> {
        let inv_dx2 = S::ONE / (dx * dx);
        let two = S::from_f64(2.0);

        let mut matrix = vec![vec![S::ZERO; n_interior]; n_interior];

        for i in 0..n_interior {
            matrix[i][i] = -two * inv_dx2;
            if i > 0 {
                matrix[i][i - 1] = inv_dx2;
            }
            if i < n_interior - 1 {
                matrix[i][i + 1] = inv_dx2;
            }
        }

        matrix
    }
}

/// Non-uniform grid finite differences.
///
/// Infrastructure for future non-uniform grid support.
#[allow(dead_code)]
pub struct NonUniformFDM;

#[allow(dead_code)]
impl NonUniformFDM {
    /// Second derivative on non-uniform grid.
    ///
    /// Uses the formula for variable spacing:
    /// d²u/dx² ≈ 2*[(u[i+1] - u[i])/(dx_plus) - (u[i] - u[i-1])/(dx_minus)] / (dx_plus + dx_minus)
    pub fn d2<S: Scalar>(u: &[S], grid: &Grid1D<S>, i: usize) -> S {
        let dx_minus = grid.dx(i - 1);
        let dx_plus = grid.dx(i);
        let sum_dx = dx_minus + dx_plus;

        let two = S::from_f64(2.0);
        two * ((u[i + 1] - u[i]) / dx_plus - (u[i] - u[i - 1]) / dx_minus) / sum_dx
    }

    /// First derivative on non-uniform grid using central difference.
    pub fn d1<S: Scalar>(u: &[S], grid: &Grid1D<S>, i: usize) -> S {
        let dx_minus = grid.dx(i - 1);
        let dx_plus = grid.dx(i);
        let sum_dx = dx_minus + dx_plus;

        (dx_minus * dx_minus * u[i + 1] + (dx_plus * dx_plus - dx_minus * dx_minus) * u[i]
            - dx_plus * dx_plus * u[i - 1])
            / (dx_minus * dx_plus * sum_dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d1_central() {
        // f(x) = x², f'(x) = 2x
        // At x = 1 (middle of [0, 2]), f'(1) = 2
        let u = vec![0.0, 1.0, 4.0]; // x² at x = 0, 1, 2
        let dx = 1.0;
        let deriv = FDM::d1_central(&u, dx, 1);
        assert!((deriv - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_d2_central() {
        // f(x) = x², f''(x) = 2
        let u = vec![0.0, 1.0, 4.0]; // x² at x = 0, 1, 2
        let dx = 1.0;
        let deriv = FDM::d2_central(&u, dx, 1);
        assert!((deriv - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_d2_central_sine() {
        // f(x) = sin(x), f''(x) = -sin(x)
        let n = 101;
        let dx = 0.1;
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * dx).sin()).collect();

        // Check at middle point
        let i = 50;
        let x = i as f64 * dx;
        let d2u = FDM::d2_central(&u, dx, i);
        let exact = -(x).sin();
        assert!(
            (d2u - exact).abs() < 0.001,
            "d2u = {}, exact = {}",
            d2u,
            exact
        );
    }

    #[test]
    fn test_boundary_derivatives() {
        // f(x) = x², f'(x) = 2x
        // More stable test function
        let u = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x² at x = 0, 1, 2, 3, 4
        let dx = 1.0;

        let d1_left = FDM::d1_left_boundary(&u, dx);
        let d1_right = FDM::d1_right_boundary(&u, dx);

        // f'(0) = 0
        assert!((d1_left - 0.0).abs() < 0.1, "d1_left = {}", d1_left);
        // f'(4) = 8
        assert!((d1_right - 8.0).abs() < 0.1, "d1_right = {}", d1_right);
    }

    #[test]
    fn test_laplacian_1d() {
        // f(x) = sin(πx), f''(x) = -π²sin(πx)
        let n = 101;
        let dx = 1.0 / (n as f64 - 1.0);
        let u: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::PI * i as f64 * dx).sin())
            .collect();

        let lap = FDM::laplacian_1d(&u, dx);

        // Check at x = 0.5
        let i = 50;
        let x = i as f64 * dx;
        let exact = -std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x).sin();
        let computed = lap[i - 1];

        assert!(
            (computed - exact).abs() < 0.1,
            "computed = {}, exact = {}",
            computed,
            exact
        );
    }

    #[test]
    fn test_stencil() {
        let dx = 0.1;
        let stencil = Stencil::d2_central(dx);

        let u = vec![0.0, 0.01, 0.04]; // x² at x = 0, 0.1, 0.2
        let d2u = stencil.apply(&u, 1);
        assert!((d2u - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_non_uniform_d2() {
        // f(x) = x², f''(x) = 2
        let points = vec![0.0, 0.3, 0.8, 1.5];
        let grid = Grid1D::from_points(points);
        let u: Vec<f64> = grid.points().iter().map(|x| x * x).collect();

        let d2u = NonUniformFDM::d2(&u, &grid, 1);
        assert!((d2u - 2.0).abs() < 0.1, "d2u = {}", d2u);

        let d2u = NonUniformFDM::d2(&u, &grid, 2);
        assert!((d2u - 2.0).abs() < 0.1, "d2u = {}", d2u);
    }
}
