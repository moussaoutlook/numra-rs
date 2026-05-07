//! Sparse matrix assembly for 2D and 3D finite difference operators.
//!
//! Assembles standard FDM operators as sparse matrices in CSC format,
//! incorporating boundary conditions into the operator.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::boundary::BoundaryCondition;
use crate::boundary2d::{BoundaryConditions2D, BoundaryConditions3D};
use crate::grid::{Grid2D, Grid3D};
use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{LinalgError, SparseMatrix};

/// Trait alias for the faer bounds needed by SparseMatrix.
/// Both f32 and f64 satisfy these bounds.
pub trait SparseScalar: Scalar + SimpleEntity + Conjugate<Canonical = Self> + ComplexField {}
impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> SparseScalar for S {}

/// Coefficients for a general 2D operator:
///   a*u_xx + b*u_yy + c*u_x + d*u_y + e*u
#[derive(Clone, Debug)]
pub struct Operator2DCoefficients<S: Scalar> {
    /// Coefficient for u_xx (second x-derivative)
    pub a: S,
    /// Coefficient for u_yy (second y-derivative)
    pub b: S,
    /// Coefficient for u_x (first x-derivative)
    pub c: S,
    /// Coefficient for u_y (first y-derivative)
    pub d: S,
    /// Coefficient for u (zeroth order)
    pub e: S,
}

impl<S: Scalar> Operator2DCoefficients<S> {
    /// Pure Laplacian: u_xx + u_yy
    pub fn laplacian() -> Self {
        Self {
            a: S::ONE,
            b: S::ONE,
            c: S::ZERO,
            d: S::ZERO,
            e: S::ZERO,
        }
    }

    /// Scaled Laplacian: alpha * (u_xx + u_yy)
    pub fn scaled_laplacian(alpha: S) -> Self {
        Self {
            a: alpha,
            b: alpha,
            c: S::ZERO,
            d: S::ZERO,
            e: S::ZERO,
        }
    }

    /// Advection-diffusion: D*(u_xx + u_yy) - vx*u_x - vy*u_y
    pub fn advection_diffusion(diffusion: S, vx: S, vy: S) -> Self {
        Self {
            a: diffusion,
            b: diffusion,
            c: -vx,
            d: -vy,
            e: S::ZERO,
        }
    }
}

/// Interior-point linear index for 2D grid.
///
/// Maps interior point (i, j) (0-based in interior, so i in 0..nx-2, j in 0..ny-2)
/// to a linear index in the unknown vector.
#[inline]
fn interior_index_2d(i: usize, j: usize, nx_int: usize) -> usize {
    j * nx_int + i
}

/// Assemble 2D Laplacian as a sparse matrix for interior grid points.
///
/// Uses a 5-point stencil on a uniform grid:
///   (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4*u_{i,j}) / h^2
///
/// For Dirichlet boundaries, the known boundary values are moved to the RHS
/// (returned in the `rhs_contribution` vector). For Neumann boundaries, ghost
/// point formulas modify the stencil.
///
/// Returns `(operator, rhs_contribution)` where:
/// - `operator` is the N_int x N_int sparse matrix
/// - `rhs_contribution` is a vector of length N_int with boundary contributions
pub fn assemble_laplacian_2d<S: SparseScalar>(
    grid: &Grid2D<S>,
    bc: &BoundaryConditions2D<S>,
) -> Result<(SparseMatrix<S>, Vec<S>), LinalgError> {
    let coeffs = Operator2DCoefficients::laplacian();
    assemble_operator_2d(grid, &coeffs, bc)
}

/// Assemble a general 2D FDM operator as a sparse matrix.
///
/// Operator: a*u_xx + b*u_yy + c*u_x + d*u_y + e*u
///
/// Uses second-order central differences on a uniform grid.
///
/// Returns `(operator, rhs_contribution)`.
pub fn assemble_operator_2d<S: SparseScalar>(
    grid: &Grid2D<S>,
    coeffs: &Operator2DCoefficients<S>,
    bc: &BoundaryConditions2D<S>,
) -> Result<(SparseMatrix<S>, Vec<S>), LinalgError> {
    let nx = grid.x_grid.len();
    let ny = grid.y_grid.len();
    let nx_int = nx - 2; // interior points in x
    let ny_int = ny - 2; // interior points in y
    let n_int = nx_int * ny_int;

    let dx = grid.x_grid.dx_uniform();
    let dy = grid.y_grid.dx_uniform();
    let inv_dx2 = S::ONE / (dx * dx);
    let inv_dy2 = S::ONE / (dy * dy);
    let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);
    let inv_2dy = S::ONE / (S::from_f64(2.0) * dy);
    let two = S::from_f64(2.0);

    // Stencil coefficients for the general operator
    let center = -two * coeffs.a * inv_dx2 - two * coeffs.b * inv_dy2 + coeffs.e;
    let x_plus = coeffs.a * inv_dx2 + coeffs.c * inv_2dx; // u_{i+1,j}
    let x_minus = coeffs.a * inv_dx2 - coeffs.c * inv_2dx; // u_{i-1,j}
    let y_plus = coeffs.b * inv_dy2 + coeffs.d * inv_2dy; // u_{i,j+1}
    let y_minus = coeffs.b * inv_dy2 - coeffs.d * inv_2dy; // u_{i,j-1}

    let mut triplets = Vec::with_capacity(5 * n_int);
    let mut rhs = vec![S::ZERO; n_int];

    for jj in 0..ny_int {
        for ii in 0..nx_int {
            let row = interior_index_2d(ii, jj, nx_int);
            // Grid indices (1-based in full grid)
            let gi = ii + 1;
            let gj = jj + 1;

            // Center
            triplets.push((row, row, center));

            // x-1 neighbor
            if ii == 0 {
                // Left boundary
                if bc.x_min.is_dirichlet() {
                    let bval = bc.x_min.value(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] + x_minus * bval;
                } else {
                    // Neumann: ghost point u[-1,j] = u[1,j] - 2*dx*flux
                    // So stencil at center gets += x_minus, and rhs gets flux contribution
                    triplets.push((row, row, x_minus));
                    let flux = bc.x_min.flux(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] + x_minus * two * dx * flux;
                }
            } else {
                let col = interior_index_2d(ii - 1, jj, nx_int);
                triplets.push((row, col, x_minus));
            }

            // x+1 neighbor
            if ii == nx_int - 1 {
                // Right boundary
                if bc.x_max.is_dirichlet() {
                    let bval = bc.x_max.value(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] + x_plus * bval;
                } else {
                    triplets.push((row, row, x_plus));
                    let flux = bc.x_max.flux(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] - x_plus * two * dx * flux;
                }
            } else {
                let col = interior_index_2d(ii + 1, jj, nx_int);
                triplets.push((row, col, x_plus));
            }

            // y-1 neighbor
            if jj == 0 {
                // Bottom boundary
                if bc.y_min.is_dirichlet() {
                    let bval = bc.y_min.value(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] + y_minus * bval;
                } else {
                    triplets.push((row, row, y_minus));
                    let flux = bc.y_min.flux(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] + y_minus * two * dy * flux;
                }
            } else {
                let col = interior_index_2d(ii, jj - 1, nx_int);
                triplets.push((row, col, y_minus));
            }

            // y+1 neighbor
            if jj == ny_int - 1 {
                // Top boundary
                if bc.y_max.is_dirichlet() {
                    let bval = bc.y_max.value(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] + y_plus * bval;
                } else {
                    triplets.push((row, row, y_plus));
                    let flux = bc.y_max.flux(S::ZERO).unwrap_or(S::ZERO);
                    rhs[row] = rhs[row] - y_plus * two * dy * flux;
                }
            } else {
                let col = interior_index_2d(ii, jj + 1, nx_int);
                triplets.push((row, col, y_plus));
            }

            let _ = (gi, gj); // suppress unused warnings
        }
    }

    let matrix = SparseMatrix::from_triplets(n_int, n_int, &triplets)?;
    Ok((matrix, rhs))
}

/// Coefficients for a general 3D operator:
///   a*u_xx + b*u_yy + c*u_zz + d*u_x + e*u_y + f*u_z + g*u
#[derive(Clone, Debug)]
pub struct Operator3DCoefficients<S: Scalar> {
    /// Coefficient for u_xx (second x-derivative)
    pub a: S,
    /// Coefficient for u_yy (second y-derivative)
    pub b: S,
    /// Coefficient for u_zz (second z-derivative)
    pub c: S,
    /// Coefficient for u_x (first x-derivative)
    pub d: S,
    /// Coefficient for u_y (first y-derivative)
    pub e: S,
    /// Coefficient for u_z (first z-derivative)
    pub f: S,
    /// Coefficient for u (zeroth order)
    pub g: S,
}

impl<S: Scalar> Operator3DCoefficients<S> {
    /// Pure Laplacian: u_xx + u_yy + u_zz
    pub fn laplacian() -> Self {
        Self {
            a: S::ONE,
            b: S::ONE,
            c: S::ONE,
            d: S::ZERO,
            e: S::ZERO,
            f: S::ZERO,
            g: S::ZERO,
        }
    }

    /// Scaled Laplacian: alpha * (u_xx + u_yy + u_zz)
    pub fn scaled_laplacian(alpha: S) -> Self {
        Self {
            a: alpha,
            b: alpha,
            c: alpha,
            d: S::ZERO,
            e: S::ZERO,
            f: S::ZERO,
            g: S::ZERO,
        }
    }

    /// Advection-diffusion: D*(u_xx + u_yy + u_zz) - vx*u_x - vy*u_y - vz*u_z
    pub fn advection_diffusion(diffusion: S, vx: S, vy: S, vz: S) -> Self {
        Self {
            a: diffusion,
            b: diffusion,
            c: diffusion,
            d: -vx,
            e: -vy,
            f: -vz,
            g: S::ZERO,
        }
    }
}

/// Interior-point linear index for 3D grid.
#[inline]
fn interior_index_3d(i: usize, j: usize, k: usize, nx_int: usize, ny_int: usize) -> usize {
    k * (nx_int * ny_int) + j * nx_int + i
}

/// Assemble 3D Laplacian as a sparse matrix for interior grid points.
///
/// Uses a 7-point stencil on a uniform grid:
///   (u_{i±1,j,k} + u_{i,j±1,k} + u_{i,j,k±1} - 6*u_{i,j,k}) / h^2
///
/// Returns `(operator, rhs_contribution)`.
pub fn assemble_laplacian_3d<S: SparseScalar>(
    grid: &Grid3D<S>,
    bc: &BoundaryConditions3D<S>,
) -> Result<(SparseMatrix<S>, Vec<S>), LinalgError> {
    let coeffs = Operator3DCoefficients::laplacian();
    assemble_operator_3d(grid, &coeffs, bc)
}

/// Assemble a general 3D FDM operator as a sparse matrix.
///
/// Operator: a*u_xx + b*u_yy + c*u_zz + d*u_x + e*u_y + f*u_z + g*u
///
/// Uses second-order central differences on a uniform grid.
///
/// Returns `(operator, rhs_contribution)`.
pub fn assemble_operator_3d<S: SparseScalar>(
    grid: &Grid3D<S>,
    coeffs: &Operator3DCoefficients<S>,
    bc: &BoundaryConditions3D<S>,
) -> Result<(SparseMatrix<S>, Vec<S>), LinalgError> {
    let nx = grid.x_grid.len();
    let ny = grid.y_grid.len();
    let nz = grid.z_grid.len();
    let nx_int = nx - 2;
    let ny_int = ny - 2;
    let nz_int = nz - 2;
    let n_int = nx_int * ny_int * nz_int;

    let dx = grid.x_grid.dx_uniform();
    let dy = grid.y_grid.dx_uniform();
    let dz = grid.z_grid.dx_uniform();
    let inv_dx2 = S::ONE / (dx * dx);
    let inv_dy2 = S::ONE / (dy * dy);
    let inv_dz2 = S::ONE / (dz * dz);
    let inv_2dx = S::ONE / (S::from_f64(2.0) * dx);
    let inv_2dy = S::ONE / (S::from_f64(2.0) * dy);
    let inv_2dz = S::ONE / (S::from_f64(2.0) * dz);
    let two = S::from_f64(2.0);

    // Stencil coefficients for the general operator
    let center =
        -two * coeffs.a * inv_dx2 - two * coeffs.b * inv_dy2 - two * coeffs.c * inv_dz2 + coeffs.g;
    let x_plus = coeffs.a * inv_dx2 + coeffs.d * inv_2dx; // u_{i+1,j,k}
    let x_minus = coeffs.a * inv_dx2 - coeffs.d * inv_2dx; // u_{i-1,j,k}
    let y_plus = coeffs.b * inv_dy2 + coeffs.e * inv_2dy; // u_{i,j+1,k}
    let y_minus = coeffs.b * inv_dy2 - coeffs.e * inv_2dy; // u_{i,j-1,k}
    let z_plus = coeffs.c * inv_dz2 + coeffs.f * inv_2dz; // u_{i,j,k+1}
    let z_minus = coeffs.c * inv_dz2 - coeffs.f * inv_2dz; // u_{i,j,k-1}

    let mut triplets = Vec::with_capacity(7 * n_int);
    let mut rhs = vec![S::ZERO; n_int];

    for kk in 0..nz_int {
        for jj in 0..ny_int {
            for ii in 0..nx_int {
                let row = interior_index_3d(ii, jj, kk, nx_int, ny_int);

                // Center
                triplets.push((row, row, center));

                // x-1
                if ii == 0 {
                    if bc.x_min.is_dirichlet() {
                        let bval = bc.x_min.value(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + x_minus * bval;
                    } else {
                        triplets.push((row, row, x_minus));
                        let flux = bc.x_min.flux(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + x_minus * two * dx * flux;
                    }
                } else {
                    let col = interior_index_3d(ii - 1, jj, kk, nx_int, ny_int);
                    triplets.push((row, col, x_minus));
                }

                // x+1
                if ii == nx_int - 1 {
                    if bc.x_max.is_dirichlet() {
                        let bval = bc.x_max.value(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + x_plus * bval;
                    } else {
                        triplets.push((row, row, x_plus));
                        let flux = bc.x_max.flux(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] - x_plus * two * dx * flux;
                    }
                } else {
                    let col = interior_index_3d(ii + 1, jj, kk, nx_int, ny_int);
                    triplets.push((row, col, x_plus));
                }

                // y-1
                if jj == 0 {
                    if bc.y_min.is_dirichlet() {
                        let bval = bc.y_min.value(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + y_minus * bval;
                    } else {
                        triplets.push((row, row, y_minus));
                        let flux = bc.y_min.flux(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + y_minus * two * dy * flux;
                    }
                } else {
                    let col = interior_index_3d(ii, jj - 1, kk, nx_int, ny_int);
                    triplets.push((row, col, y_minus));
                }

                // y+1
                if jj == ny_int - 1 {
                    if bc.y_max.is_dirichlet() {
                        let bval = bc.y_max.value(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + y_plus * bval;
                    } else {
                        triplets.push((row, row, y_plus));
                        let flux = bc.y_max.flux(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] - y_plus * two * dy * flux;
                    }
                } else {
                    let col = interior_index_3d(ii, jj + 1, kk, nx_int, ny_int);
                    triplets.push((row, col, y_plus));
                }

                // z-1
                if kk == 0 {
                    if bc.z_min.is_dirichlet() {
                        let bval = bc.z_min.value(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + z_minus * bval;
                    } else {
                        triplets.push((row, row, z_minus));
                        let flux = bc.z_min.flux(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + z_minus * two * dz * flux;
                    }
                } else {
                    let col = interior_index_3d(ii, jj, kk - 1, nx_int, ny_int);
                    triplets.push((row, col, z_minus));
                }

                // z+1
                if kk == nz_int - 1 {
                    if bc.z_max.is_dirichlet() {
                        let bval = bc.z_max.value(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] + z_plus * bval;
                    } else {
                        triplets.push((row, row, z_plus));
                        let flux = bc.z_max.flux(S::ZERO).unwrap_or(S::ZERO);
                        rhs[row] = rhs[row] - z_plus * two * dz * flux;
                    }
                } else {
                    let col = interior_index_3d(ii, jj, kk + 1, nx_int, ny_int);
                    triplets.push((row, col, z_plus));
                }
            }
        }
    }

    let matrix = SparseMatrix::from_triplets(n_int, n_int, &triplets)?;
    Ok((matrix, rhs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numra_linalg::Matrix;

    #[test]
    fn test_laplacian_2d_size() {
        // 5x5 grid => 3x3 = 9 interior points
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let (mat, rhs) = assemble_laplacian_2d(&grid, &bc).unwrap();
        assert_eq!(mat.nrows(), 9);
        assert_eq!(mat.ncols(), 9);
        assert_eq!(rhs.len(), 9);
        // All zero Dirichlet => rhs should be all zeros
        for &v in &rhs {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplacian_2d_diagonal() {
        // 4x4 grid => 2x2 = 4 interior points, dx=dy=1/3
        let grid = Grid2D::uniform(0.0, 1.0, 4, 0.0, 1.0, 4);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let (mat, _rhs) = assemble_laplacian_2d(&grid, &bc).unwrap();

        // Diagonal entry: -2/dx^2 - 2/dy^2 = -2*(9) - 2*(9) = -36
        let h = 1.0 / 3.0;
        let expected_diag = -4.0 / (h * h);
        assert!(
            (mat.get(0, 0) - expected_diag).abs() < 1e-8,
            "diag = {}, expected = {}",
            mat.get(0, 0),
            expected_diag
        );
    }

    #[test]
    fn test_laplacian_2d_symmetry() {
        // Laplacian should be symmetric for uniform grid + zero Dirichlet
        let grid = Grid2D::uniform(0.0, 1.0, 6, 0.0, 1.0, 6);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let (mat, _rhs) = assemble_laplacian_2d(&grid, &bc).unwrap();

        let n = mat.nrows();
        let dense = mat.to_dense();
        for i in 0..n {
            for j in i + 1..n {
                let a_ij = dense.get(i, j);
                let a_ji = dense.get(j, i);
                assert!(
                    (a_ij - a_ji).abs() < 1e-10,
                    "Not symmetric at ({}, {}): {} vs {}",
                    i,
                    j,
                    a_ij,
                    a_ji
                );
            }
        }
    }

    #[test]
    fn test_laplacian_2d_matvec() {
        // For u(x,y) = x(1-x)*y(1-y), Laplacian = -2y(1-y) - 2x(1-x)
        // With zero Dirichlet BCs, verify L*u_interior ≈ laplacian values
        let n = 11;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let (mat, rhs) = assemble_laplacian_2d(&grid, &bc).unwrap();

        let nx_int = n - 2;
        let ny_int = n - 2;
        let n_int = nx_int * ny_int;

        // Build u at interior points
        let mut u = vec![0.0_f64; n_int];
        for jj in 0..ny_int {
            for ii in 0..nx_int {
                let x = grid.x_grid.points()[ii + 1];
                let y = grid.y_grid.points()[jj + 1];
                u[jj * nx_int + ii] = x * (1.0 - x) * y * (1.0 - y);
            }
        }

        // L*u + rhs should approximate the analytical Laplacian
        let lu = mat.mul_vec(&u).unwrap();
        for jj in 0..ny_int {
            for ii in 0..nx_int {
                let idx = jj * nx_int + ii;
                let x = grid.x_grid.points()[ii + 1];
                let y = grid.y_grid.points()[jj + 1];
                let exact_lap = -2.0 * y * (1.0 - y) - 2.0 * x * (1.0 - x);
                let computed = lu[idx] + rhs[idx];
                assert!(
                    (computed - exact_lap).abs() < 0.05,
                    "At ({}, {}): computed={}, exact={}",
                    x,
                    y,
                    computed,
                    exact_lap
                );
            }
        }
    }

    #[test]
    fn test_laplacian_2d_dirichlet_rhs() {
        // Non-zero Dirichlet BC: x_min = 1.0, rest = 0
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D {
            x_min: crate::boundary::BoxedBC::dirichlet(1.0),
            x_max: crate::boundary::BoxedBC::dirichlet(0.0),
            y_min: crate::boundary::BoxedBC::dirichlet(0.0),
            y_max: crate::boundary::BoxedBC::dirichlet(0.0),
        };
        let (_mat, rhs) = assemble_laplacian_2d(&grid, &bc).unwrap();

        // First column of interior (ii=0) should have nonzero rhs from x_min BC
        let nx_int = 3;
        for jj in 0..3 {
            let idx = jj * nx_int + 0;
            assert!(
                rhs[idx].abs() > 1e-10,
                "Expected nonzero rhs at ii=0, jj={}",
                jj
            );
        }
        // Last column (ii=2) should have zero rhs (x_max = 0)
        for jj in 0..3 {
            let idx = jj * nx_int + 2;
            assert!(
                rhs[idx].abs() < 1e-10,
                "Expected zero rhs at ii=2, jj={}",
                jj
            );
        }
    }

    #[test]
    fn test_operator_2d_advection_diffusion() {
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let coeffs = Operator2DCoefficients::advection_diffusion(0.1, 1.0, 0.0);
        let (mat, _rhs) = assemble_operator_2d(&grid, &coeffs, &bc).unwrap();
        assert_eq!(mat.nrows(), 9);
        assert_eq!(mat.ncols(), 9);
    }

    #[test]
    fn test_laplacian_3d_size() {
        // 4x4x4 grid => 2x2x2 = 8 interior points
        let grid = Grid3D::uniform(0.0, 1.0, 4, 0.0, 1.0, 4, 0.0, 1.0, 4);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let (mat, rhs) = assemble_laplacian_3d(&grid, &bc).unwrap();
        assert_eq!(mat.nrows(), 8);
        assert_eq!(mat.ncols(), 8);
        assert_eq!(rhs.len(), 8);
    }

    #[test]
    fn test_laplacian_3d_diagonal() {
        let grid = Grid3D::uniform(0.0, 1.0, 4, 0.0, 1.0, 4, 0.0, 1.0, 4);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let (mat, _rhs) = assemble_laplacian_3d(&grid, &bc).unwrap();

        // Diagonal: -2/dx^2 - 2/dy^2 - 2/dz^2 = -6/h^2 where h = 1/3
        let h = 1.0 / 3.0;
        let expected_diag = -6.0 / (h * h);
        assert!(
            (mat.get(0, 0) - expected_diag).abs() < 1e-8,
            "diag = {}, expected = {}",
            mat.get(0, 0),
            expected_diag
        );
    }

    #[test]
    fn test_laplacian_3d_symmetry() {
        let grid = Grid3D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let (mat, _rhs) = assemble_laplacian_3d(&grid, &bc).unwrap();

        let n = mat.nrows();
        let dense = mat.to_dense();
        for i in 0..n {
            for j in i + 1..n {
                let a_ij = dense.get(i, j);
                let a_ji = dense.get(j, i);
                assert!(
                    (a_ij - a_ji).abs() < 1e-10,
                    "Not symmetric at ({}, {}): {} vs {}",
                    i,
                    j,
                    a_ij,
                    a_ji
                );
            }
        }
    }
}
