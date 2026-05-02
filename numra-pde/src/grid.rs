//! Spatial grids for PDE discretization.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// 1D uniform or non-uniform grid.
#[derive(Clone, Debug)]
pub struct Grid1D<S: Scalar> {
    /// Grid points
    points: Vec<S>,
    /// Grid spacings (for non-uniform grids)
    spacings: Vec<S>,
}

impl<S: Scalar> Grid1D<S> {
    /// Create a uniform grid with n points from x_min to x_max.
    pub fn uniform(x_min: S, x_max: S, n: usize) -> Self {
        assert!(n >= 2, "Grid must have at least 2 points");
        let dx = (x_max - x_min) / S::from_usize(n - 1);

        let points: Vec<S> = (0..n).map(|i| x_min + S::from_usize(i) * dx).collect();

        let spacings = vec![dx; n - 1];

        Self { points, spacings }
    }

    /// Create a grid from explicit points.
    pub fn from_points(points: Vec<S>) -> Self {
        assert!(points.len() >= 2, "Grid must have at least 2 points");

        let spacings: Vec<S> = points.windows(2).map(|w| w[1] - w[0]).collect();

        Self { points, spacings }
    }

    /// Create a grid with refinement near boundaries.
    ///
    /// Uses a tanh-based clustering.
    pub fn clustered(x_min: S, x_max: S, n: usize, cluster_factor: S) -> Self {
        assert!(n >= 2, "Grid must have at least 2 points");

        let mut points = Vec::with_capacity(n);
        let length = x_max - x_min;

        for i in 0..n {
            let xi = S::from_f64(-1.0) + S::from_f64(2.0) * S::from_usize(i) / S::from_usize(n - 1);
            // tanh clustering
            let clustered = (cluster_factor * xi).tanh() / cluster_factor.tanh();
            let x = x_min + length * (S::ONE + clustered) / S::from_f64(2.0);
            points.push(x);
        }

        let spacings: Vec<S> = points.windows(2).map(|w| w[1] - w[0]).collect();

        Self { points, spacings }
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if grid is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of interior points (excluding boundaries).
    pub fn n_interior(&self) -> usize {
        self.points.len().saturating_sub(2)
    }

    /// Get all grid points.
    pub fn points(&self) -> &[S] {
        &self.points
    }

    /// Get interior points (excluding boundaries).
    pub fn interior_points(&self) -> &[S] {
        &self.points[1..self.points.len() - 1]
    }

    /// Get grid spacing at index i (between points i and i+1).
    pub fn dx(&self, i: usize) -> S {
        self.spacings[i]
    }

    /// Get uniform spacing (panics if non-uniform).
    pub fn dx_uniform(&self) -> S {
        let first = self.spacings[0];
        debug_assert!(
            self.spacings
                .iter()
                .all(|&dx| (dx - first).abs() < S::from_f64(1e-10) * first),
            "Grid is not uniform"
        );
        first
    }

    /// Check if grid is uniform.
    pub fn is_uniform(&self) -> bool {
        if self.spacings.is_empty() {
            return true;
        }
        let first = self.spacings[0];
        let tol = S::from_f64(1e-10) * first;
        self.spacings.iter().all(|&dx| (dx - first).abs() < tol)
    }

    /// Get domain bounds.
    pub fn bounds(&self) -> (S, S) {
        (*self.points.first().unwrap(), *self.points.last().unwrap())
    }

    /// Domain length.
    pub fn length(&self) -> S {
        let (a, b) = self.bounds();
        b - a
    }

    /// Find index of point closest to x.
    pub fn find_index(&self, x: S) -> usize {
        let (lo, hi) = self.bounds();
        if x <= lo {
            return 0;
        }
        if x >= hi {
            return self.len() - 1;
        }

        // Binary search for non-uniform grids
        let mut left = 0;
        let mut right = self.len() - 1;
        while left < right - 1 {
            let mid = (left + right) / 2;
            if x < self.points[mid] {
                right = mid;
            } else {
                left = mid;
            }
        }

        // Return closest point
        if (x - self.points[left]).abs() < (x - self.points[right]).abs() {
            left
        } else {
            right
        }
    }
}

/// 2D grid (tensor product of 1D grids).
#[derive(Clone, Debug)]
pub struct Grid2D<S: Scalar> {
    /// Grid in x direction
    pub x_grid: Grid1D<S>,
    /// Grid in y direction
    pub y_grid: Grid1D<S>,
}

impl<S: Scalar> Grid2D<S> {
    /// Create a uniform 2D grid.
    pub fn uniform(x_min: S, x_max: S, nx: usize, y_min: S, y_max: S, ny: usize) -> Self {
        Self {
            x_grid: Grid1D::uniform(x_min, x_max, nx),
            y_grid: Grid1D::uniform(y_min, y_max, ny),
        }
    }

    /// Total number of grid points.
    pub fn len(&self) -> usize {
        self.x_grid.len() * self.y_grid.len()
    }

    /// Check if grid is empty.
    pub fn is_empty(&self) -> bool {
        self.x_grid.is_empty() || self.y_grid.is_empty()
    }

    /// Number of interior points.
    pub fn n_interior(&self) -> usize {
        self.x_grid.n_interior() * self.y_grid.n_interior()
    }

    /// Convert (i, j) grid index to linear index.
    pub fn linear_index(&self, i: usize, j: usize) -> usize {
        j * self.x_grid.len() + i
    }

    /// Convert linear index to (i, j) grid index.
    pub fn grid_index(&self, idx: usize) -> (usize, usize) {
        let i = idx % self.x_grid.len();
        let j = idx / self.x_grid.len();
        (i, j)
    }

    /// Get (x, y) coordinates at grid index (i, j).
    pub fn point(&self, i: usize, j: usize) -> (S, S) {
        (self.x_grid.points()[i], self.y_grid.points()[j])
    }

    /// Number of points in x direction.
    pub fn nx(&self) -> usize {
        self.x_grid.len()
    }

    /// Number of points in y direction.
    pub fn ny(&self) -> usize {
        self.y_grid.len()
    }

    /// Number of interior points in x direction.
    pub fn nx_interior(&self) -> usize {
        self.x_grid.n_interior()
    }

    /// Number of interior points in y direction.
    pub fn ny_interior(&self) -> usize {
        self.y_grid.n_interior()
    }
}

/// 3D grid (tensor product of 1D grids).
#[derive(Clone, Debug)]
pub struct Grid3D<S: Scalar> {
    /// Grid in x direction
    pub x_grid: Grid1D<S>,
    /// Grid in y direction
    pub y_grid: Grid1D<S>,
    /// Grid in z direction
    pub z_grid: Grid1D<S>,
}

impl<S: Scalar> Grid3D<S> {
    /// Create a uniform 3D grid.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform(
        x_min: S,
        x_max: S,
        nx: usize,
        y_min: S,
        y_max: S,
        ny: usize,
        z_min: S,
        z_max: S,
        nz: usize,
    ) -> Self {
        Self {
            x_grid: Grid1D::uniform(x_min, x_max, nx),
            y_grid: Grid1D::uniform(y_min, y_max, ny),
            z_grid: Grid1D::uniform(z_min, z_max, nz),
        }
    }

    /// Total number of grid points.
    pub fn len(&self) -> usize {
        self.x_grid.len() * self.y_grid.len() * self.z_grid.len()
    }

    /// Check if grid is empty.
    pub fn is_empty(&self) -> bool {
        self.x_grid.is_empty() || self.y_grid.is_empty() || self.z_grid.is_empty()
    }

    /// Number of interior points.
    pub fn n_interior(&self) -> usize {
        self.x_grid.n_interior() * self.y_grid.n_interior() * self.z_grid.n_interior()
    }

    /// Convert (i, j, k) grid index to linear index (column-major: x varies fastest).
    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.x_grid.len() * self.y_grid.len()) + j * self.x_grid.len() + i
    }

    /// Convert linear index to (i, j, k) grid index.
    pub fn grid_index(&self, idx: usize) -> (usize, usize, usize) {
        let nx = self.x_grid.len();
        let ny = self.y_grid.len();
        let i = idx % nx;
        let j = (idx / nx) % ny;
        let k = idx / (nx * ny);
        (i, j, k)
    }

    /// Get (x, y, z) coordinates at grid index (i, j, k).
    pub fn point(&self, i: usize, j: usize, k: usize) -> (S, S, S) {
        (
            self.x_grid.points()[i],
            self.y_grid.points()[j],
            self.z_grid.points()[k],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_grid() {
        let grid = Grid1D::uniform(0.0, 1.0, 11);
        assert_eq!(grid.len(), 11);
        assert_eq!(grid.n_interior(), 9);
        assert!(grid.is_uniform());
        assert!((grid.dx_uniform() - 0.1).abs() < 1e-10);
        assert!((grid.points()[5] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_grid_bounds() {
        let grid = Grid1D::uniform(-1.0, 2.0, 31);
        let (lo, hi) = grid.bounds();
        assert!((lo - (-1.0)).abs() < 1e-10);
        assert!((hi - 2.0).abs() < 1e-10);
        assert!((grid.length() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_points() {
        let points = vec![0.0, 0.1, 0.3, 0.6, 1.0];
        let grid = Grid1D::from_points(points);
        assert_eq!(grid.len(), 5);
        assert!(!grid.is_uniform());
        assert!((grid.dx(0) - 0.1).abs() < 1e-10);
        assert!((grid.dx(1) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_clustered_grid() {
        let grid = Grid1D::clustered(0.0, 1.0, 21, 2.0);
        assert_eq!(grid.len(), 21);

        // Should have smaller spacing near boundaries
        let dx_boundary = grid.dx(0);
        let dx_middle = grid.dx(9);
        assert!(dx_boundary < dx_middle);
    }

    #[test]
    fn test_find_index() {
        let grid = Grid1D::uniform(0.0, 1.0, 11);
        assert_eq!(grid.find_index(0.0), 0);
        assert_eq!(grid.find_index(0.5), 5);
        assert_eq!(grid.find_index(1.0), 10);
        assert_eq!(grid.find_index(0.49), 5);
    }

    #[test]
    fn test_grid_2d() {
        let grid = Grid2D::uniform(0.0, 1.0, 11, 0.0, 2.0, 21);
        assert_eq!(grid.len(), 11 * 21);
        assert_eq!(grid.n_interior(), 9 * 19);

        let (x, y) = grid.point(5, 10);
        assert!((x - 0.5).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_3d() {
        let grid = Grid3D::uniform(0.0, 1.0, 5, 0.0, 2.0, 11, 0.0, 3.0, 7);
        assert_eq!(grid.len(), 5 * 11 * 7);
        assert_eq!(grid.n_interior(), 3 * 9 * 5);

        let (x, y, z) = grid.point(2, 5, 3);
        assert!((x - 0.5).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
        // z = 3 * 3/6 = 3 * 0.5 = ... no, z_min=0, z_max=3, nz=7, dz=0.5
        // z[3] = 0 + 3*0.5 = 1.5
        assert!((z - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_grid_3d_indexing() {
        let grid = Grid3D::uniform(0.0, 1.0, 4, 0.0, 1.0, 5, 0.0, 1.0, 6);
        // Round-trip test
        for k in 0..6 {
            for j in 0..5 {
                for i in 0..4 {
                    let idx = grid.linear_index(i, j, k);
                    let (ri, rj, rk) = grid.grid_index(idx);
                    assert_eq!((ri, rj, rk), (i, j, k));
                }
            }
        }
    }
}
