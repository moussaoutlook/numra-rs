# Eigenvalue Problems

Eigenvalue decompositions reveal the fundamental modes of a linear operator.
Given a square matrix \\(A\\), an eigenvalue \\(\lambda\\) and eigenvector
\\(v\\) satisfy:

\\[
Av = \lambda v
\\]

In ODE solvers, eigenvalues determine stability (stiff vs. non-stiff), while
in applications they capture natural frequencies, principal components, and
decay rates. Numra provides two eigenvalue decompositions: one for symmetric
matrices (real eigenvalues) and one for general matrices (possibly complex
eigenvalues).

## Symmetric Eigendecomposition

For a real symmetric matrix \\(A = A^T\\), the spectral theorem guarantees:

\\[
A = U \Lambda U^T
\\]

where \\(U\\) is orthogonal and \\(\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)\\)
contains real eigenvalues in ascending order.

### API

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, SymEigenDecomposition};

// Symmetric matrix: [2 1; 1 2]
let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 2.0); a.set(0, 1, 1.0);
a.set(1, 0, 1.0); a.set(1, 1, 2.0);

let evd = SymEigenDecomposition::new(&a).unwrap();

// Eigenvalues in ascending order
let eigenvalues = evd.eigenvalues(); // [1.0, 3.0]
assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
assert!((eigenvalues[1] - 3.0).abs() < 1e-10);

// Orthogonal eigenvector matrix
let v = evd.eigenvectors(); // columns are eigenvectors
assert_eq!(evd.dim(), 2);
```

### Verifying Orthogonality

The eigenvector matrix \\(U\\) satisfies \\(U^T U = I\\):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, SymEigenDecomposition};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 2.0); a.set(0, 1, 1.0);
a.set(1, 0, 1.0); a.set(1, 1, 2.0);

let evd = SymEigenDecomposition::new(&a).unwrap();
let v = evd.eigenvectors();

// Check V^T * V = I
for i in 0..2 {
    for j in 0..2 {
        let mut dot = 0.0;
        for k in 0..2 {
            dot += v.get(k, i) * v.get(k, j);
        }
        let expected = if i == j { 1.0 } else { 0.0 };
        assert!((dot - expected).abs() < 1e-10);
    }
}
```

### Reconstruction

Verify the decomposition \\(A = U \Lambda U^T\\):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, SymEigenDecomposition};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
a.set(0, 0, 4.0); a.set(0, 1, 1.0);
a.set(1, 0, 1.0); a.set(1, 1, 3.0); a.set(1, 2, 1.0);
a.set(2, 1, 1.0); a.set(2, 2, 2.0);

let evd = SymEigenDecomposition::new(&a).unwrap();
let eigenvalues = evd.eigenvalues();
let v = evd.eigenvectors();

// Reconstruct: sum_k lambda_k * v[:,k] * v[:,k]^T
for i in 0..3 {
    for j in 0..3 {
        let mut val = 0.0;
        for k in 0..3 {
            val += v.get(i, k) * eigenvalues[k] * v.get(j, k);
        }
        assert!((val - a.get(i, j)).abs() < 1e-10);
    }
}
```

### Convenience Methods

`DenseMatrix` provides shortcuts that skip the intermediate struct:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 3.0);
a.set(1, 1, 7.0);

// Full decomposition
let evd = a.eigh().unwrap();

// Eigenvalues only (faster -- no eigenvectors computed)
let eigenvalues = a.eigvalsh().unwrap();
assert!((eigenvalues[0] - 3.0).abs() < 1e-10);
assert!((eigenvalues[1] - 7.0).abs() < 1e-10);
```

## General Eigendecomposition

For non-symmetric matrices, eigenvalues may be complex even when the matrix is
real. Numra stores eigenvalues as separate real and imaginary parts to avoid
exposing complex number types in the public API.

### Complex Eigenvalues from a Real Matrix

A rotation matrix has purely imaginary eigenvalues:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, EigenDecomposition};

// 90-degree rotation: [[0, -1], [1, 0]]
let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 0.0); a.set(0, 1, -1.0);
a.set(1, 0, 1.0); a.set(1, 1, 0.0);

let evd = EigenDecomposition::new(&a).unwrap();
let re = evd.eigenvalues_real(); // [~0, ~0]
let im = evd.eigenvalues_imag(); // [+1, -1] or [-1, +1]

// Both real parts are zero
assert!(re[0].abs() < 1e-10);
assert!(re[1].abs() < 1e-10);

// Imaginary parts form a conjugate pair: +i and -i
let mut im_sorted = vec![im[0], im[1]];
im_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
assert!((im_sorted[0] - (-1.0)).abs() < 1e-10);
assert!((im_sorted[1] - 1.0).abs() < 1e-10);
```

### Real Eigenvalues of a Diagonal Matrix

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, EigenDecomposition};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
a.set(0, 0, 2.0);
a.set(1, 1, 5.0);
a.set(2, 2, -1.0);

let evd = EigenDecomposition::new(&a).unwrap();
let re = evd.eigenvalues_real();
let im = evd.eigenvalues_imag();

// All imaginary parts are zero
for &v in im.iter() {
    assert!(v.abs() < 1e-10);
}

// Real parts contain {-1, 2, 5} in some order
let mut re_sorted = re.to_vec();
re_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
assert!((re_sorted[0] - (-1.0)).abs() < 1e-10);
assert!((re_sorted[1] - 2.0).abs() < 1e-10);
assert!((re_sorted[2] - 5.0).abs() < 1e-10);
```

### Convenience Method

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 1.0);
a.set(1, 1, 2.0);

let (re, im) = a.eigvals().unwrap();
// re contains real parts, im contains imaginary parts
```

## Connection to ODE Stability Analysis

Eigenvalues are the key to understanding stiffness and stability in ODE systems.
For the linear system:

\\[
\frac{dy}{dt} = Jy
\\]

where \\(J\\) is the Jacobian, the eigenvalues \\(\lambda_i\\) of \\(J\\)
determine the system's behavior:

### Stiffness Detection

A system is stiff when the eigenvalues of the Jacobian span a wide range:

\\[
\text{stiffness ratio} = \frac{\max_i |\lambda_i|}{\min_i |\lambda_i|} \gg 1
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, EigenDecomposition};

// Jacobian of a chemical kinetics system
let mut j: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
j.set(0, 0, -1000.0); j.set(0, 1, 1.0);
j.set(1, 0, 0.0);     j.set(1, 1, -0.1);

let (re, im) = j.eigvals().unwrap();
let magnitudes: Vec<f64> = re.iter().zip(im.iter())
    .map(|(&r, &i)| (r * r + i * i).sqrt())
    .collect();

let max_eig = magnitudes.iter().cloned().fold(0.0_f64, f64::max);
let min_eig = magnitudes.iter().cloned().fold(f64::INFINITY, f64::min);
let stiffness_ratio = max_eig / min_eig;
// stiffness_ratio = 1000/0.1 = 10000 -- very stiff!
// Use an implicit solver (Radau5, BDF, ESDIRK)
```

### Stability Regions

An ODE method is stable when \\(h \lambda_i\\) lies inside the method's
stability region for all eigenvalues \\(\lambda_i\\). This is why implicit
methods are needed for stiff problems: their stability regions include large
portions of the left half-plane.

| Eigenvalue property | System behavior | Solver recommendation |
|---------------------|----------------|----------------------|
| All \\(\text{Re}(\lambda_i) < 0\\), small \\(\lvert\lambda_i\rvert\\) | Stable, non-stiff | DoPri5 / Tsit5 |
| All \\(\text{Re}(\lambda_i) < 0\\), large spread | Stable, stiff | Radau5 / BDF |
| \\(\text{Re}(\lambda_i) > 0\\) for some \\(i\\) | Unstable (growing modes) | Short integration only |
| Purely imaginary \\(\lambda_i\\) | Oscillatory | Symplectic / energy-preserving |

### Oscillatory Systems

For systems with purely imaginary eigenvalues (conservative systems like the
harmonic oscillator), the eigenvalue structure reveals the natural frequencies:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, SymEigenDecomposition};

// Mass-spring system: M^{-1} K for stiffness matrix K
// This is symmetric, so use the symmetric decomposition
let mut k: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
k.set(0, 0, 2.0); k.set(0, 1, -1.0);
k.set(1, 0, -1.0); k.set(1, 1, 2.0); k.set(1, 2, -1.0);
k.set(2, 1, -1.0); k.set(2, 2, 2.0);

let evd = SymEigenDecomposition::new(&k).unwrap();
let eigenvalues = evd.eigenvalues();

// Natural frequencies: omega_i = sqrt(lambda_i)
for (i, &lam) in eigenvalues.iter().enumerate() {
    let omega = lam.sqrt();
    println!("Mode {}: frequency = {:.4} rad/s", i, omega);
}
```

## f32 Support

Both decompositions work with `f32` for memory-constrained applications:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{DenseMatrix, Matrix, SymEigenDecomposition, EigenDecomposition};

// Symmetric f32
let mut a: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
a.set(0, 0, 4.0); a.set(1, 1, 9.0);
let evd = SymEigenDecomposition::new(&a).unwrap();
let eigenvalues = evd.eigenvalues();
assert!((eigenvalues[0] - 4.0f32).abs() < 1e-5);

// General f32
let mut b: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
b.set(0, 1, -1.0); b.set(1, 0, 1.0);
let evd = EigenDecomposition::<f32>::new(&b).unwrap();
```

## Comparison of Eigenvalue Methods

| Feature | `SymEigenDecomposition` | `EigenDecomposition` |
|---------|------------------------|---------------------|
| Matrix type | Symmetric (\\(A = A^T\\)) | Any square |
| Eigenvalues | Real, ascending | Complex (as real + imag parts) |
| Eigenvectors | Yes (orthogonal) | Not exposed (eigenvalues only) |
| Speed | Faster (exploits symmetry) | Slower (Schur decomposition) |
| Typical use | Vibration modes, SPD analysis | Stability analysis, general spectra |

### When to Use Which

- **Use `SymEigenDecomposition`** when the matrix is symmetric. You get real
  eigenvalues and orthogonal eigenvectors, and the algorithm is faster and more
  stable. Common in structural mechanics, covariance analysis, and graph
  Laplacians.

- **Use `EigenDecomposition`** when the matrix is non-symmetric (e.g., Jacobians
  from ODE systems, convection operators). Eigenvalues may come in complex
  conjugate pairs.

## Practical Tips

1. **Check symmetry before choosing.** If your matrix comes from a
   self-adjoint operator (e.g., \\(-\nabla^2\\)), always use `eigh()` /
   `SymEigenDecomposition` for speed and accuracy.

2. **Eigenvalues for stiffness.** Before solving a large ODE system, compute
   the Jacobian's eigenvalues to decide between explicit and implicit solvers.

3. **Condition number from eigenvalues.** For symmetric matrices, the condition
   number is \\(\kappa(A) = |\lambda_{max}| / |\lambda_{min}|\\). For general
   matrices, use SVD instead (see [Factorizations](./factorizations.md)).

4. **Small eigenvalues signal trouble.** An eigenvalue near zero means the
   matrix is nearly singular. The corresponding eigenvector spans the
   near-null space.
