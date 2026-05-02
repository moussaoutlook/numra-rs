---
title: "Matrices & Vectors"
---

Dense linear algebra is the foundation that Numra's ODE solvers, eigenvalue
routines, and factorizations are built upon. The `numra-linalg` crate provides
a `Matrix` trait for backend-agnostic operations and a `DenseMatrix` type that
wraps the high-performance [faer](https://crates.io/crates/faer) library.

## The Matrix Trait

All matrix operations in Numra go through the `Matrix<S>` trait, where
`S: Scalar` (either `f32` or `f64`). This abstraction decouples ODE solvers
from any particular linear algebra backend.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{Matrix, DenseMatrix};

pub trait Matrix<S: Scalar>: Clone + Sized {
    fn zeros(rows: usize, cols: usize) -> Self;
    fn identity(n: usize) -> Self;
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn get(&self, i: usize, j: usize) -> S;
    fn set(&mut self, i: usize, j: usize, value: S);
    fn fill_zero(&mut self);
    fn scale(&mut self, alpha: S);
    fn mul_vec(&self, x: &[S], y: &mut [S]);
    fn add_scaled(&mut self, alpha: S, other: &Self);
    fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError>;
    fn is_square(&self) -> bool;
}
```

The trait intentionally provides the minimal surface area needed by numerical
solvers: creation, element access, matrix-vector products, and direct linear
solves. This makes it feasible to implement custom backends (GPU, banded, etc.)
without touching solver code.

## Creating Matrices

### Zero and Identity Matrices

The most common starting points:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

// 3x4 zero matrix
let z: DenseMatrix<f64> = DenseMatrix::zeros(3, 4);
assert_eq!(z.nrows(), 3);
assert_eq!(z.ncols(), 4);

// 3x3 identity matrix
let eye: DenseMatrix<f64> = DenseMatrix::identity(3);
assert!((eye.get(0, 0) - 1.0).abs() < 1e-15);
assert!((eye.get(0, 1) - 0.0).abs() < 1e-15);
```

### From Row-Major Data

When you have matrix entries as a flat array in row-major order (C layout), use
`from_row_major`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::DenseMatrix;

// [1 2 3]
// [4 5 6]
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let a: DenseMatrix<f64> = DenseMatrix::from_row_major(2, 3, &data);

assert!((a.get(0, 2) - 3.0).abs() < 1e-15); // row 0, col 2
assert!((a.get(1, 0) - 4.0).abs() < 1e-15); // row 1, col 0
```

### From Column-Major Data

If your data is in column-major order (Fortran layout), use `from_col_major`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::DenseMatrix;

// col0 = [1, 3], col1 = [2, 4]
let data = [1.0, 3.0, 2.0, 4.0];
let a: DenseMatrix<f64> = DenseMatrix::from_col_major(2, 2, &data);

assert!((a.get(0, 0) - 1.0).abs() < 1e-15);
assert!((a.get(1, 0) - 3.0).abs() < 1e-15);
assert!((a.get(0, 1) - 2.0).abs() < 1e-15);
```

### Element-by-Element Construction

For small matrices or structured patterns, set entries individually:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

// Build a tridiagonal matrix
let n = 4;
let mut a: DenseMatrix<f64> = DenseMatrix::zeros(n, n);
for i in 0..n {
    a.set(i, i, 2.0);
    if i > 0     { a.set(i, i - 1, -1.0); }
    if i < n - 1 { a.set(i, i + 1, -1.0); }
}
```

## Basic Operations

### Element Access

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0);
a.set(0, 1, 2.0);
a.set(1, 0, 3.0);
a.set(1, 1, 4.0);

let val = a.get(1, 0); // 3.0
```

### Scalar Multiplication

Scale all elements by a constant:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::identity(3);
a.scale(5.0);
// a is now diag(5, 5, 5)
```

### Matrix Addition

The `add_scaled` method computes $ A \leftarrow A + \alpha B $:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::identity(3);
let b: DenseMatrix<f64> = DenseMatrix::identity(3);
a.add_scaled(2.0, &b);
// a = I + 2*I = 3*I
assert!((a.get(0, 0) - 3.0).abs() < 1e-15);
```

This operation is used extensively inside implicit ODE solvers to form
matrices of the form $ I - h\gamma J $, where $ J $ is the Jacobian.

### Matrix-Vector Product

Compute $ y = A x $:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0); a.set(0, 1, 2.0);
a.set(1, 0, 3.0); a.set(1, 1, 4.0);

let x = [1.0, 2.0];
let mut y = [0.0, 0.0];
a.mul_vec(&x, &mut y);
// y = [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
```

## Solving Linear Systems

The `solve` method solves $ Ax = b $ using LU factorization with partial
pivoting internally:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
a.set(0, 0, 2.0);
a.set(1, 1, 3.0);
a.set(2, 2, 4.0);

let b = vec![1.0, 2.0, 3.0];
let x = a.solve(&b).unwrap();

assert!((x[0] - 0.5).abs() < 1e-10);
assert!((x[1] - 2.0 / 3.0).abs() < 1e-10);
assert!((x[2] - 0.75).abs() < 1e-10);
```

This is a one-shot convenience method. If you need to solve multiple systems
with the same coefficient matrix, use `LUFactorization` (see
[Factorizations](./factorizations)) to cache the decomposition.

### Error Handling

`solve` returns `Result<Vec<S>, LinalgError>` with typed errors:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

// Non-square matrix
let a: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
let b = vec![1.0, 2.0];
match a.solve(&b) {
    Err(numra::linalg::LinalgError::NotSquare { nrows, ncols }) => {
        println!("Cannot solve: {}x{} is not square", nrows, ncols);
    }
    _ => {}
}

// Dimension mismatch
let a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
let b = vec![1.0, 2.0, 3.0]; // wrong size
match a.solve(&b) {
    Err(numra::linalg::LinalgError::DimensionMismatch { expected, actual }) => {
        println!("Expected {:?}, got {:?}", expected, actual);
    }
    _ => {}
}
```

## Matrix Norms

`DenseMatrix` provides two norms for assessing matrix magnitude and
conditioning:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::DenseMatrix;

let a = DenseMatrix::from_row_major(2, 2, &[1.0, -2.0, 3.0, -4.0]);

// Frobenius norm: sqrt(sum of squares)
// ||A||_F = sqrt(1 + 4 + 9 + 16) = sqrt(30)
let frob = a.norm_frobenius();

// Infinity norm: max absolute row sum
// row 0: |1| + |-2| = 3
// row 1: |3| + |-4| = 7
let inf = a.norm_inf(); // 7.0
```

| Norm | Formula | Use case |
|------|---------|----------|
| Frobenius | $\lVert A \rVert_F = \sqrt{\sum_{i,j} a_{ij}^2}$ | General-purpose matrix size |
| Infinity | $\lVert A \rVert_\infty = \max_i \sum_j \lvert a_{ij} \rvert$ | Row-wise bound, cheap to compute |

## Working with the faer Backend

`DenseMatrix` wraps `faer::Mat<S>` and provides escape hatches when you need
access to the underlying representation:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::DenseMatrix;

let a: DenseMatrix<f64> = DenseMatrix::identity(3);

// Get an immutable reference to the faer matrix
let faer_ref = a.as_faer(); // MatRef<'_, f64>

// Get a mutable reference
let mut a = a;
let faer_mut = a.as_faer_mut(); // MatMut<'_, f64>

// Wrap an existing faer matrix
let faer_mat = faer::Mat::<f64>::zeros(3, 3);
let wrapped = DenseMatrix::from_faer(faer_mat);
```

This interop means you can freely mix Numra operations with faer's broader
ecosystem (sparse solvers, parallel decompositions, etc.) when needed.

## Data Export

Convert back to a flat row-major array for serialization or interop with other
libraries:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);

let flat = a.to_row_major();
assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

## Generic Scalar Support

All matrix operations work with both `f32` and `f64`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 2.0);
a.set(1, 1, 3.0);

let b = vec![4.0f32, 9.0f32];
let x = a.solve(&b).unwrap();

assert!((x[0] - 2.0f32).abs() < 1e-5);
assert!((x[1] - 3.0f32).abs() < 1e-5);
```

Use `f32` when memory is constrained (e.g., large systems on embedded targets)
and `f64` (the default) when you need full double precision.

## Summary

| Operation | Method | Complexity |
|-----------|--------|------------|
| Create zero matrix | `DenseMatrix::zeros(m, n)` | $O(mn)$ |
| Create identity | `DenseMatrix::identity(n)` | $O(n^2)$ |
| Element access | `get(i, j)` / `set(i, j, v)` | $O(1)$ |
| Scale | `scale(alpha)` | $O(mn)$ |
| Add scaled | `add_scaled(alpha, &other)` | $O(mn)$ |
| Matrix-vector product | `mul_vec(&x, &mut y)` | $O(mn)$ |
| Solve $Ax = b$ | `solve(&b)` | $O(n^3)$ |
| Frobenius norm | `norm_frobenius()` | $O(mn)$ |
| Infinity norm | `norm_inf()` | $O(mn)$ |

For systems where you solve $Ax = b$ multiple times with the same $A$,
use [matrix factorizations](./factorizations) to avoid redundant work.
