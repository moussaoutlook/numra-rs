---
title: "Matrix Factorizations"
---

Matrix factorizations decompose a matrix into a product of simpler matrices
that make subsequent operations (solving, least-squares, determinants) much
cheaper. The key insight: factorize once at $O(n^3)$ cost, then solve
repeatedly at $O(n^2)$ cost per right-hand side.

Numra provides four direct factorizations: LU, QR, Cholesky, and SVD. Each
caches the decomposition on construction and reuses it for every subsequent
`solve()` call.

## LU Factorization

LU with partial pivoting decomposes $A$ as:

$$
PA = LU
$$

where $P$ is a permutation matrix, $L$ is unit lower triangular, and
$U$ is upper triangular. This is the general-purpose workhorse for solving
$Ax = b$ with any nonsingular square matrix.

### When to Use

- General (non-symmetric, non-SPD) square systems
- Multiple right-hand sides with the same coefficient matrix
- Inside implicit ODE solvers (Radau5, ESDIRK, BDF) where the Newton system
  is solved many times per step

### API

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, LUFactorization};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0); a.set(0, 1, 2.0);
a.set(1, 0, 3.0); a.set(1, 1, 4.0);

// Factor once
let lu = LUFactorization::new(&a).unwrap();
assert_eq!(lu.dim(), 2);

// Solve Ax = b (reuses the cached factorization)
let b1 = vec![5.0, 11.0];
let x1 = lu.solve(&b1).unwrap();
assert!((x1[0] - 1.0).abs() < 1e-10);
assert!((x1[1] - 2.0).abs() < 1e-10);

// Solve again with a different right-hand side -- no refactorization
let b2 = vec![5.0, 13.0];
let x2 = lu.solve(&b2).unwrap();
assert!((x2[0] - 3.0).abs() < 1e-10);
assert!((x2[1] - 1.0).abs() < 1e-10);
```

### In-Place Solve

When you want to overwrite `b` with the solution:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, LUFactorization};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 2.0); a.set(1, 1, 3.0);

let lu = LUFactorization::new(&a).unwrap();
let mut b = vec![4.0, 9.0];
lu.solve_inplace(&mut b).unwrap();
// b is now [2.0, 3.0]
```

### Multiple Right-Hand Sides

Solve $AX = B$ where $B$ has several columns (stored column-major):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, LUFactorization};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 2.0); a.set(1, 1, 3.0);

let lu = LUFactorization::new(&a).unwrap();

// Two right-hand sides in column-major layout:
// col 0: [2, 6], col 1: [4, 9]
let b = vec![2.0, 6.0, 4.0, 9.0];
let x = lu.solve_multi(&b, 2).unwrap();
// col 0 solution: [1, 2], col 1 solution: [2, 3]
```

### The LUSolver Trait

`DenseMatrix` also implements the `LUSolver` trait for convenience:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, LUSolver};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0); a.set(0, 1, 2.0);
a.set(1, 0, 3.0); a.set(1, 1, 4.0);

// One-shot solve (factors internally)
let x = a.lu_solve(&vec![5.0, 11.0]).unwrap();

// Or get the factorization for reuse
let lu = a.lu_factor().unwrap();
```

### Numerical Properties

| Property | Value |
|----------|-------|
| Cost | $\frac{2}{3}n^3$ flops for factorization, $2n^2$ per solve |
| Stability | Partial pivoting guarantees $\lVert L \rVert \leq n$ |
| Works on | Any nonsingular square matrix |
| Fails when | Matrix is singular (zero pivot) |

## QR Factorization

QR decomposes $A$ as:

$$
A = QR
$$

where $Q$ is orthogonal ($Q^T Q = I$) and $R$ is upper triangular.
QR is numerically more stable than LU for solving and is essential for
least-squares problems.

### When to Use

- Overdetermined systems (more rows than columns): $\min \lVert Ax - b \rVert_2$
- Square systems where extra numerical stability is worth the cost
- Building orthogonal bases (Gram-Schmidt, Krylov methods)

### Square System Solve

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, QRFactorization};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0); a.set(0, 1, 2.0);
a.set(1, 0, 3.0); a.set(1, 1, 4.0);

let qr = QRFactorization::new(&a).unwrap();
assert_eq!(qr.nrows(), 2);
assert_eq!(qr.ncols(), 2);

let b = vec![5.0, 11.0];
let x = qr.solve(&b).unwrap();
assert!((x[0] - 1.0).abs() < 1e-10);
assert!((x[1] - 2.0).abs() < 1e-10);
```

### Least-Squares Problems

For an overdetermined system $A \in \mathbb{R}^{m \times n}$ with
$m > n$, `solve_least_squares` finds $x$ that minimizes
$\lVert Ax - b \rVert_2$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, QRFactorization};

// Fit a line y = a + b*x through points (1,1), (2,2), (3,2)
let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 2);
a.set(0, 0, 1.0); a.set(0, 1, 1.0);
a.set(1, 0, 1.0); a.set(1, 1, 2.0);
a.set(2, 0, 1.0); a.set(2, 1, 3.0);

let qr = QRFactorization::new(&a).unwrap();

let b = vec![1.0, 2.0, 2.0];
let x = qr.solve_least_squares(&b).unwrap();
// x[0] = intercept ~ 2/3, x[1] = slope ~ 0.5
assert!((x[0] - 2.0 / 3.0).abs() < 1e-10);
assert!((x[1] - 0.5).abs() < 1e-10);
```

### Numerical Properties

| Property | Value |
|----------|-------|
| Cost | $\frac{4}{3}mn^2 - \frac{4}{3}n^3$ flops |
| Stability | Always backward stable (orthogonal transformations) |
| Works on | Any $m \times n$ matrix with $m \geq n$ |
| Advantage over LU | Never amplifies rounding errors |

## Cholesky Factorization

For symmetric positive definite (SPD) matrices, Cholesky computes:

$$
A = LL^T
$$

where $L$ is lower triangular. This is roughly twice as fast as LU and
uses half the storage.

### When to Use

- Covariance matrices, mass matrices, stiffness matrices from FEM
- Any system where you know $A$ is SPD
- Detecting positive definiteness (Cholesky fails if $A$ is not SPD)

### API

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, CholeskyFactorization};

// SPD matrix: [4 2; 2 3]
let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 4.0); a.set(0, 1, 2.0);
a.set(1, 0, 2.0); a.set(1, 1, 3.0);

let chol = CholeskyFactorization::new(&a).unwrap();
assert_eq!(chol.dim(), 2);

let b = vec![6.0, 5.0];
let x = chol.solve(&b).unwrap();
assert!((x[0] - 1.0).abs() < 1e-10);
assert!((x[1] - 1.0).abs() < 1e-10);
```

### Detecting Non-SPD Matrices

If the matrix is not positive definite, `new` returns an error:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, CholeskyFactorization};

// Not positive definite: [1 2; 2 1]
let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0); a.set(0, 1, 2.0);
a.set(1, 0, 2.0); a.set(1, 1, 1.0);

match CholeskyFactorization::new(&a) {
    Err(numra::linalg::LinalgError::NotPositiveDefinite) => {
        println!("Matrix is not positive definite");
    }
    _ => {}
}
```

### Numerical Properties

| Property | Value |
|----------|-------|
| Cost | $\frac{1}{3}n^3$ flops (half of LU) |
| Stability | Always stable for SPD matrices |
| Works on | Symmetric positive definite matrices only |
| Advantage | 2x faster and half the storage of LU |

## SVD -- Singular Value Decomposition

The SVD decomposes any $m \times n$ matrix as:

$$
A = U \Sigma V^T
$$

where $U$ is $m \times m$ orthogonal, $\Sigma$ is $m \times n$
diagonal with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$,
and $V$ is $n \times n$ orthogonal.

### Full SVD

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, SvdDecomposition};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
a.set(0, 0, 3.0);
a.set(1, 1, 1.0);
a.set(2, 2, 2.0);

let svd = SvdDecomposition::new(&a).unwrap();

// Singular values in decreasing order
let s = svd.singular_values(); // [3.0, 2.0, 1.0]

// Orthogonal factor matrices
let u = svd.u();  // 3x3
let v = svd.v();  // 3x3
```

### Thin SVD

For tall or wide matrices, the thin SVD saves memory by only computing the
"economy" factors:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, ThinSvdDecomposition};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(4, 2);
a.set(0, 0, 1.0);
a.set(1, 1, 2.0);

let svd = ThinSvdDecomposition::new(&a).unwrap();
let u = svd.u();  // 4x2 (not 4x4)
let v = svd.v();  // 2x2
let s = svd.singular_values(); // [2.0, 1.0]
```

### Pseudoinverse, Rank, and Condition Number

The SVD provides the most numerically reliable way to compute these quantities:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix, SvdDecomposition};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0); a.set(0, 1, 2.0);
a.set(1, 0, 2.0); a.set(1, 1, 4.0);

let svd = SvdDecomposition::new(&a).unwrap();

// Numerical rank (count singular values > tolerance)
let r = svd.rank(1e-10); // 1 (rank-deficient matrix)

// Condition number: sigma_max / sigma_min
let kappa = svd.cond(); // infinity for singular matrix

// Moore-Penrose pseudoinverse: A^+
let pinv = svd.pseudoinverse();
```

### Convenience Methods on DenseMatrix

For quick one-off computations without managing the decomposition object:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{DenseMatrix, Matrix};

let a = DenseMatrix::from_row_major(2, 2, &[3.0, 0.0, 0.0, 1.0]);

let s = a.singular_values();  // [3.0, 1.0]
let k = a.cond();             // 3.0
let r = a.rank(1e-10);        // 2
let pinv = a.pinv().unwrap();  // pseudoinverse

// Least-squares solve via SVD pseudoinverse
let mut tall = DenseMatrix::from_row_major(3, 2,
    &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
let b = vec![1.0, 2.0, 2.0];
let x = tall.lstsq(&b).unwrap();
```

### Numerical Properties

| Property | Value |
|----------|-------|
| Cost | $O(\min(m,n)^2 \cdot \max(m,n))$ flops |
| Stability | The gold standard -- always backward stable |
| Works on | Any $m \times n$ matrix (no restrictions) |
| Extra info | Rank, condition number, null space, range |

## Choosing a Factorization

| Situation | Best choice | Why |
|-----------|-------------|-----|
| General square $Ax = b$ | LU | Fastest general-purpose solver |
| Multiple RHS, same $A$ | LU (cached) | Factor once, solve many |
| SPD system | Cholesky | 2x faster, guaranteed stable |
| Overdetermined least-squares | QR | Numerically stable $\min \lVert Ax - b \rVert$ |
| Rank-deficient system | SVD | Only reliable method |
| Condition number estimate | SVD | Direct access to $\kappa(A)$ |
| ODE implicit step | LU | Newton system $(I - h\gamma J)x = r$ |

### Cost Comparison (n x n matrix)

| Factorization | Factor | Solve | Total (k solves) |
|---------------|--------|-------|-------------------|
| LU | $\frac{2}{3}n^3$ | $2n^2$ | $\frac{2}{3}n^3 + 2kn^2$ |
| QR | $\frac{4}{3}n^3$ | $2n^2$ | $\frac{4}{3}n^3 + 2kn^2$ |
| Cholesky | $\frac{1}{3}n^3$ | $n^2$ | $\frac{1}{3}n^3 + kn^2$ |
| SVD | $O(n^3)$ | $2n^2$ | $O(n^3) + 2kn^2$ |

The factorization cost dominates for single solves. When $k \gg 1$, all
methods amortize to $O(n^2)$ per solve -- but LU and Cholesky have the
smallest constants.
