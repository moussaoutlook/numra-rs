---
title: "Sparse Matrices"
---

Most matrices arising from real-world problems (PDEs, network graphs, FEM
discretizations) are sparse: the vast majority of entries are zero. Storing and
operating on these zeros wastes both memory and computation. Numra provides
`SparseMatrix` for efficient storage and solvers that exploit sparsity.

## When to Use Sparse vs. Dense

| Criterion | Dense | Sparse |
|-----------|-------|--------|
| Non-zero ratio | > 10-20% | < 5-10% |
| Typical size | up to ~5000 x 5000 | 10,000+ x 10,000+ |
| Matrix-vector product | $O(n^2)$ | $O(\text{nnz})$ |
| Storage | $O(n^2)$ | $O(\text{nnz})$ |
| Direct solve | $O(n^3)$ | $O(n)$ to $O(n^{1.5})$ with fill-in |
| Setup complexity | Simple | Requires assembling triplets |

**Rule of thumb:** If your matrix is banded (tridiagonal, pentadiagonal) or
arises from a mesh-based discretization, use sparse. If it is a small dense
block (e.g., a 10x10 Jacobian), use `DenseMatrix`.

## CSC Storage Format

`SparseMatrix` uses Compressed Sparse Column (CSC) format internally via
faer's `SparseColMat`. CSC stores:

- **col_ptrs**: Array of length $n+1$ where `col_ptrs[j]..col_ptrs[j+1]`
  gives the index range of non-zeros in column $j$
- **row_indices**: Row index of each non-zero
- **values**: The non-zero values themselves

This layout is optimal for column-oriented operations and is the standard
format used by MATLAB, SciPy, and most sparse linear algebra libraries.

## Creating Sparse Matrices

### From Triplets (COO Format)

The most common way to build a sparse matrix is from $(i, j, v)$ triplets.
Duplicate entries at the same position are automatically summed:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

// Build a 3x3 tridiagonal matrix: tridiag(-1, 2, -1)
let triplets = vec![
    (0, 0,  2.0), (0, 1, -1.0),
    (1, 0, -1.0), (1, 1,  2.0), (1, 2, -1.0),
    (2, 1, -1.0), (2, 2,  2.0),
];
let a = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();

assert_eq!(a.nrows(), 3);
assert_eq!(a.ncols(), 3);
assert_eq!(a.nnz(), 7);
```

### Duplicate Entry Summation

This is useful for finite element assembly, where multiple elements
contribute to the same matrix entry:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

// Two contributions to position (0, 0)
let triplets = vec![
    (0, 0, 3.0),
    (0, 0, 4.0),
    (1, 1, 1.0),
];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
assert!((a.get(0, 0) - 7.0).abs() < 1e-15); // 3.0 + 4.0
```

### Building a 1D Laplacian

A common pattern for discretizing $-u'' = f$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

fn laplacian_1d(n: usize) -> SparseMatrix<f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0));
        if i > 0     { triplets.push((i, i - 1, -1.0)); }
        if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
    }
    SparseMatrix::from_triplets(n, n, &triplets).unwrap()
}

let a = laplacian_1d(100);
assert_eq!(a.nrows(), 100);
assert_eq!(a.nnz(), 100 + 2 * 99); // diagonal + two off-diagonals
```

## Element Access

Access individual elements. Missing entries return zero:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

let triplets = vec![(0, 0, 5.0), (1, 1, 3.0)];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();

assert!((a.get(0, 0) - 5.0).abs() < 1e-15);
assert!((a.get(0, 1) - 0.0).abs() < 1e-15); // structural zero
```

> **Note:** Element access is $O(\text{nnz per column})$ since it requires
> a linear scan of the column's row indices. For bulk operations, use
> matrix-vector products instead.

## Sparse Matrix-Vector Product

The fundamental operation for iterative solvers. Computes $y = Ax$ in
$O(\text{nnz})$ time:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

// [2 -1 0; -1 2 -1; 0 -1 2] * [1; 1; 1] = [1; 0; 1]
let triplets = vec![
    (0, 0, 2.0), (0, 1, -1.0),
    (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
    (2, 1, -1.0), (2, 2, 2.0),
];
let a = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();

let x = vec![1.0, 1.0, 1.0];
let y = a.mul_vec(&x).unwrap();

assert!((y[0] - 1.0).abs() < 1e-10);
assert!((y[1] - 0.0).abs() < 1e-10);
assert!((y[2] - 1.0).abs() < 1e-10);
```

## Transpose

Compute $A^T$ by swapping row and column indices:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

let triplets = vec![
    (0, 0, 1.0), (0, 1, 2.0),
    (1, 0, 3.0), (1, 1, 4.0),
];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
let at = a.transpose().unwrap();

// at[i][j] = a[j][i]
assert!((at.get(0, 1) - 3.0).abs() < 1e-15);
assert!((at.get(1, 0) - 2.0).abs() < 1e-15);
```

## Converting to Dense

For small sparse matrices, you can convert to `DenseMatrix` for
factorizations or debugging:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{SparseMatrix, Matrix};

let triplets = vec![
    (0, 0, 1.0), (0, 1, 2.0),
    (1, 0, 3.0), (1, 1, 4.0),
];
let sparse = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
let dense = sparse.to_dense();

assert!((dense.get(0, 0) - 1.0).abs() < 1e-15);
assert!((dense.get(1, 1) - 4.0).abs() < 1e-15);
```

## Accessing CSC Internals

When you need low-level access to the compressed storage:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::SparseMatrix;

let triplets = vec![(0, 0, 1.0), (1, 0, 3.0), (0, 1, 2.0), (1, 1, 4.0)];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();

let col_ptrs = a.col_ptrs();     // [0, 2, 4] for a 2-column matrix
let row_indices = a.row_indices(); // row indices of each non-zero
let values = a.values();           // corresponding values
```

## Sparse Direct Solvers

### SparseLU

LU factorization for general sparse systems. Currently converts to dense
internally (suitable for moderate sizes up to a few thousand):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{SparseMatrix, SparseLU};

let triplets = vec![
    (0, 0, 1.0), (0, 1, 2.0),
    (1, 0, 3.0), (1, 1, 4.0),
];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();

let lu = SparseLU::new(&a).unwrap();
assert_eq!(lu.dim(), 2);

let b = vec![5.0, 11.0];
let x = lu.solve(&b).unwrap();
assert!((x[0] - 1.0).abs() < 1e-10);
assert!((x[1] - 2.0).abs() < 1e-10);
```

### SparseCholesky

Cholesky factorization for symmetric positive definite sparse systems:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{SparseMatrix, SparseCholesky};

// SPD: [4 2; 2 3]
let triplets = vec![
    (0, 0, 4.0), (0, 1, 2.0),
    (1, 0, 2.0), (1, 1, 3.0),
];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();

let chol = SparseCholesky::new(&a).unwrap();

let b = vec![6.0, 5.0];
let x = chol.solve(&b).unwrap();
assert!((x[0] - 1.0).abs() < 1e-10);
assert!((x[1] - 1.0).abs() < 1e-10);
```

### Direct vs. Iterative

For large sparse systems, direct methods can be prohibitively expensive due to
fill-in (the factored matrix has more non-zeros than the original). When the
matrix is large and sparse, consider [iterative solvers](./iterative-solvers)
instead.

| Method | Best for | Complexity |
|--------|----------|-----------|
| `SparseLU` | General moderate systems (n < 5000) | $O(n^3)$ worst case |
| `SparseCholesky` | SPD moderate systems (n < 5000) | $O(n^3)$ worst case |
| CG / PCG | Large SPD systems | $O(k \cdot \text{nnz})$ |
| GMRES / BiCGStab | Large non-symmetric systems | $O(k \cdot \text{nnz})$ |

## Verifying Sparse vs. Dense Agreement

A good practice is to verify that sparse and dense solvers produce the same
answer:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{SparseMatrix, SparseLU, Matrix};

let triplets = vec![
    (0, 0, 4.0), (0, 1, -1.0),
    (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
    (2, 1, -1.0), (2, 2, 4.0), (2, 3, -1.0),
    (3, 2, -1.0), (3, 3, 4.0),
];
let sparse = SparseMatrix::from_triplets(4, 4, &triplets).unwrap();
let dense = sparse.to_dense();

let b = vec![1.0, 2.0, 3.0, 4.0];

let x_dense = dense.solve(&b).unwrap();
let lu = SparseLU::new(&sparse).unwrap();
let x_sparse = lu.solve(&b).unwrap();

for i in 0..4 {
    assert!((x_dense[i] - x_sparse[i]).abs() < 1e-10);
}
```

## f32 Support

Sparse matrices work with both `f32` and `f64`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::linalg::{SparseMatrix, SparseLU};

let triplets: Vec<(usize, usize, f32)> = vec![
    (0, 0, 2.0), (1, 1, 3.0),
];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
let lu = SparseLU::new(&a).unwrap();

let b = vec![4.0f32, 9.0f32];
let x = lu.solve(&b).unwrap();
assert!((x[0] - 2.0f32).abs() < 1e-5);
```

## Summary

| Operation | Method | Complexity |
|-----------|--------|------------|
| Build from triplets | `SparseMatrix::from_triplets(m, n, &trips)` | $O(\text{nnz} \log \text{nnz})$ |
| Element access | `get(i, j)` | $O(\text{nnz}/n)$ average |
| Matrix-vector product | `mul_vec(&x)` | $O(\text{nnz})$ |
| Transpose | `transpose()` | $O(\text{nnz})$ |
| Convert to dense | `to_dense()` | $O(n^2)$ |
| LU solve | `SparseLU::new(&a).solve(&b)` | $O(n^3)$ (dense fallback) |
| Cholesky solve | `SparseCholesky::new(&a).solve(&b)` | $O(n^3)$ (dense fallback) |
| Non-zero count | `nnz()` | $O(\text{nnz})$ |

For systems larger than a few thousand unknowns, use the
[iterative solvers](./iterative-solvers) which achieve $O(k \cdot \text{nnz})$
cost where $k$ is the number of iterations (typically $k \ll n$ with a
good preconditioner).
