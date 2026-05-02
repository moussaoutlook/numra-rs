# Special Functions

Numra provides implementations of the classical special functions commonly
needed in scientific computing, physics, and engineering. All functions are
generic over `S: Scalar`, working with both `f32` and `f64`.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{
    // Gamma family
    gamma, lgamma, digamma, beta, gammainc, gammaincc, betainc,
    // Error functions
    erf, erfc, erfinv, erfcinv,
    // Bessel functions
    besselj, bessely, besseli, besselk,
    // Elliptic integrals
    ellipk, ellipe, ellipf,
    // Airy functions
    airy_ai, airy_bi,
    // Hypergeometric functions
    hyp1f1, hyp2f1,
    // Orthogonal polynomials
    legendre_p, hermite_h, laguerre_l, chebyshev_t, legendre_plm,
    // Zeta functions
    zeta, hurwitz_zeta,
    // Miscellaneous
    dawson, fresnel_s, fresnel_c, mittag_leffler, mittag_leffler_1,
};
```

## Gamma Family

The gamma function and related functions are foundational to combinatorics,
probability distributions, and many areas of mathematical physics.

### Gamma Function

\\[
\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t}\, dt
\\]

Generalizes the factorial: \\(\Gamma(n) = (n-1)!\\) for positive integers.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::gamma;

assert!((gamma(5.0_f64) - 24.0).abs() < 1e-10);       // Gamma(5) = 4! = 24
assert!((gamma(0.5_f64) - std::f64::consts::PI.sqrt()).abs() < 1e-10); // Gamma(1/2) = sqrt(pi)
```

### Log-Gamma

`lgamma(z)` computes \\(\ln|\Gamma(z)|\\), avoiding overflow for large arguments
where \\(\Gamma(z)\\) itself would exceed floating-point range.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::lgamma;

let val = lgamma(100.0_f64);  // ln(99!) -- too large for Gamma directly
```

### Digamma (Psi) Function

\\[
\psi(x) = \frac{d}{dx} \ln \Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}
\\]

Used in maximum likelihood estimation, Bayesian inference, and as a building
block for the polygamma functions.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::digamma;

let val = digamma(1.0_f64);  // psi(1) = -gamma (Euler-Mascheroni constant)
```

### Beta Function

\\[
B(a, b) = \frac{\Gamma(a)\,\Gamma(b)}{\Gamma(a+b)} = \int_0^1 t^{a-1}(1-t)^{b-1}\, dt
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::beta;

let val = beta(2.0_f64, 3.0);  // B(2,3) = 1*2 / 4! * 3! = 1/12
```

### Incomplete Gamma Functions

The regularized lower and upper incomplete gamma functions:

\\[
P(a, x) = \frac{\gamma(a, x)}{\Gamma(a)}, \qquad Q(a, x) = 1 - P(a, x)
\\]

Essential for chi-squared distributions and Poisson statistics.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{gammainc, gammaincc};

let p = gammainc(1.0_f64, 1.0);   // P(1, 1) = 1 - e^(-1)
let q = gammaincc(1.0_f64, 1.0);  // Q(1, 1) = e^(-1)
assert!((p + q - 1.0).abs() < 1e-12);
```

### Regularized Incomplete Beta

\\[
I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}
\\]

The CDF of the beta distribution and a building block for the F-distribution
and Student's t-distribution.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::betainc;

let val = betainc(2.0_f64, 3.0, 0.5);  // I_{0.5}(2, 3)
```

## Error Functions

The error function and its relatives arise in probability, statistics, and
solutions of the diffusion equation.

### erf and erfc

\\[
\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\, dt
\\]

\\[
\text{erfc}(x) = 1 - \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_x^{\infty} e^{-t^2}\, dt
\\]

The relationship to the normal distribution CDF is:
\\(\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{erf, erfc};

assert!((erf(0.0_f64)).abs() < 1e-15);           // erf(0) = 0
assert!((erf(1e10_f64) - 1.0).abs() < 1e-15);   // erf(inf) = 1
assert!((erfc(0.0_f64) - 1.0).abs() < 1e-15);   // erfc(0) = 1
```

### Inverse Error Functions

`erfinv(p)` returns the value \\(x\\) such that \\(\text{erf}(x) = p\\).
Uses a rational approximation with Newton refinement.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{erfinv, erfcinv};

let x = erfinv(0.5_f64);
assert!((erf(x) - 0.5).abs() < 1e-12);

let x = erfcinv(0.1_f64);
assert!((erfc(x) - 0.1).abs() < 1e-12);
```

**Use cases:** Generating normally-distributed random variates from uniform
ones, computing confidence intervals, quantile functions.

## Bessel Functions

Bessel functions are solutions to Bessel's differential equation:

\\[
x^2 y'' + x y' + (x^2 - \nu^2) y = 0
\\]

They arise in problems with cylindrical symmetry: heat conduction in cylinders,
vibrations of circular membranes, electromagnetic waveguides, and diffraction.

### First and Second Kind

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{besselj, bessely};

// J_0(0) = 1, J_n(0) = 0 for n > 0
assert!((besselj(0.0_f64, 0.0) - 1.0).abs() < 1e-12);
assert!(besselj(1.0_f64, 0.0).abs() < 1e-12);

// Y_0(x) diverges as x -> 0
let y0 = bessely(0.0_f64, 1.0);
```

### Modified Bessel Functions

\\(I_\nu(x)\\) and \\(K_\nu(x)\\) are solutions to the modified Bessel equation
and arise in problems with exponential rather than oscillatory behavior.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{besseli, besselk};

let i0 = besseli(0.0_f64, 1.0);  // I_0(1)
let k0 = besselk(0.0_f64, 1.0);  // K_0(1)
```

**Use cases:** Signal processing (Kaiser window), statistical distributions
(von Mises), heat conduction, elastic vibrations.

## Elliptic Integrals

Elliptic integrals arise in computing arc lengths of ellipses, periods of
pendulums, and in many problems in celestial mechanics and electrostatics.

### Complete Elliptic Integrals

\\[
K(m) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - m\sin^2\theta}}, \qquad
E(m) = \int_0^{\pi/2} \sqrt{1 - m\sin^2\theta}\, d\theta
\\]

where the parameter \\(m = k^2\\) is the squared elliptic modulus (\\(0 \le m < 1\\)).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{ellipk, ellipe};

// K(0) = E(0) = pi/2
assert!((ellipk(0.0_f64) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
assert!((ellipe(0.0_f64) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

// K(m) -> infinity as m -> 1
let k_near_1 = ellipk(0.999_f64);
```

### Incomplete Elliptic Integral of the First Kind

\\[
F(\phi, m) = \int_0^{\phi} \frac{d\theta}{\sqrt{1 - m\sin^2\theta}}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::ellipf;

let val = ellipf(1.0_f64, 0.5);  // F(1.0, 0.5)
```

**Use cases:** Pendulum period beyond small-angle approximation, conformal
mapping, geodesics on an ellipsoid.

## Airy Functions

Solutions to Airy's equation \\(y'' - xy = 0\\):

\\[
\text{Ai}(x), \quad \text{Bi}(x)
\\]

Ai(x) decays exponentially for \\(x > 0\\) and oscillates for \\(x < 0\\).
Bi(x) grows exponentially for \\(x > 0\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{airy_ai, airy_bi};

let ai0 = airy_ai(0.0_f64);  // Ai(0) = 1 / (3^{2/3} * Gamma(2/3))
let bi0 = airy_bi(0.0_f64);  // Bi(0) = 1 / (3^{1/6} * Gamma(2/3))
```

**Use cases:** Quantum mechanics (WKB turning points), optics (diffraction near
caustics), fluid dynamics (Stokes flow near boundaries).

## Hypergeometric Functions

The generalized hypergeometric functions unify many special functions.

### Confluent Hypergeometric -- \\({}_1F_1\\)

\\[
{}_1F_1(a; b; z) = \sum_{k=0}^{\infty} \frac{(a)_k}{(b)_k} \frac{z^k}{k!}
\\]

Also known as Kummer's function \\(M(a, b, z)\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::hyp1f1;

// 1F1(1; 1; z) = e^z
let val = hyp1f1(1.0_f64, 1.0, 1.0);
assert!((val - std::f64::consts::E).abs() < 1e-8);
```

### Gauss Hypergeometric -- \\({}_2F_1\\)

\\[
{}_2F_1(a, b; c; z) = \sum_{k=0}^{\infty} \frac{(a)_k (b)_k}{(c)_k} \frac{z^k}{k!}
\\]

Generalizes many classical functions: Legendre polynomials, Chebyshev polynomials,
the incomplete beta function, and more.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::hyp2f1;

let val = hyp2f1(1.0_f64, 1.0, 2.0, 0.5);  // -ln(1-z)/z at z=0.5 = 2*ln(2)
```

**Use cases:** Angular momentum coupling in quantum mechanics, solutions to
Fuchsian differential equations, statistical distributions.

## Orthogonal Polynomials

Classical orthogonal polynomial families, essential for spectral methods,
Gaussian quadrature, and expansion in orthogonal bases.

### Legendre Polynomials

\\(P_n(x)\\) are orthogonal on \\([-1, 1]\\) with weight 1:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::legendre_p;

assert!((legendre_p(0, 0.5_f64) - 1.0).abs() < 1e-12);    // P_0(x) = 1
assert!((legendre_p(1, 0.5_f64) - 0.5).abs() < 1e-12);    // P_1(x) = x
assert!((legendre_p(2, 0.5_f64) - (-0.125)).abs() < 1e-12); // P_2(x) = (3x^2-1)/2
```

### Associated Legendre Functions

\\(P_n^m(x)\\) for spherical harmonics:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::legendre_plm;

let val = legendre_plm(2, 1, 0.5_f64);  // P_2^1(0.5)
```

### Hermite Polynomials

\\(H_n(x)\\) are orthogonal on \\((-\infty, \infty)\\) with weight \\(e^{-x^2}\\):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::hermite_h;

assert!((hermite_h(0, 1.0_f64) - 1.0).abs() < 1e-12);   // H_0 = 1
assert!((hermite_h(1, 1.0_f64) - 2.0).abs() < 1e-12);   // H_1 = 2x
assert!((hermite_h(2, 1.0_f64) - 2.0).abs() < 1e-12);   // H_2 = 4x^2 - 2
```

**Use cases:** Quantum harmonic oscillator wavefunctions, Gauss-Hermite
quadrature, probability theory.

### Laguerre Polynomials

\\(L_n(x)\\) are orthogonal on \\([0, \infty)\\) with weight \\(e^{-x}\\):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::laguerre_l;

assert!((laguerre_l(0, 1.0_f64) - 1.0).abs() < 1e-12);  // L_0 = 1
assert!((laguerre_l(1, 1.0_f64) - 0.0).abs() < 1e-12);  // L_1 = 1-x
assert!((laguerre_l(2, 1.0_f64) - (-0.5)).abs() < 1e-12); // L_2 = (x^2-4x+2)/2
```

**Use cases:** Hydrogen atom radial wavefunctions, Gauss-Laguerre quadrature.

### Chebyshev Polynomials

\\(T_n(x)\\) are orthogonal on \\([-1, 1]\\) with weight \\(1/\sqrt{1-x^2}\\):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::chebyshev_t;

assert!((chebyshev_t(0, 0.5_f64) - 1.0).abs() < 1e-12);    // T_0 = 1
assert!((chebyshev_t(1, 0.5_f64) - 0.5).abs() < 1e-12);    // T_1 = x
assert!((chebyshev_t(2, 0.5_f64) - (-0.5)).abs() < 1e-12); // T_2 = 2x^2 - 1
```

**Use cases:** Spectral methods, function approximation (Chebyshev expansion),
filter design, numerical analysis.

## Zeta Functions

### Riemann Zeta Function

\\[
\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}, \quad \text{Re}(s) > 1
\\]

Extended to the complex plane via analytic continuation.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::zeta;

assert!((zeta(2.0_f64) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-10);
```

### Hurwitz Zeta Function

\\[
\zeta(s, a) = \sum_{n=0}^{\infty} \frac{1}{(n+a)^s}
\\]

Generalizes the Riemann zeta (\\(\zeta(s) = \zeta(s, 1)\\)).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::hurwitz_zeta;

let val = hurwitz_zeta(2.0_f64, 1.0);  // should equal zeta(2)
```

**Use cases:** Number theory, quantum field theory (Casimir effect),
thermodynamics (Bose-Einstein and Fermi-Dirac integrals).

## Miscellaneous Functions

### Dawson's Integral

\\[
F(x) = e^{-x^2} \int_0^x e^{t^2}\, dt
\\]

Related to the imaginary error function. Used in plasma physics and spectroscopy
(Voigt profile computation).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::dawson;

let val = dawson(1.0_f64);
```

### Fresnel Integrals

\\[
S(x) = \int_0^x \sin\left(\frac{\pi}{2} t^2\right) dt, \qquad
C(x) = \int_0^x \cos\left(\frac{\pi}{2} t^2\right) dt
\\]

Used in optics (near-field diffraction) and road/railway design (clothoid curves).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{fresnel_s, fresnel_c};

let s = fresnel_s(1.0_f64);
let c = fresnel_c(1.0_f64);
```

### Mittag-Leffler Function

\\[
E_{\alpha, \beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}
\\]

Generalizes the exponential function: \\(E_{1,1}(z) = e^z\\). The Mittag-Leffler
function is the natural generalization of the exponential for fractional
calculus.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::special::{mittag_leffler, mittag_leffler_1};

// E_{1,1}(z) = e^z
let val = mittag_leffler(1.0_f64, 1.0, 1.0);
assert!((val - std::f64::consts::E).abs() < 1e-8);

// Shorthand: E_{alpha}(z) = E_{alpha, 1}(z)
let val = mittag_leffler_1(0.5_f64, -1.0);
```

**Use cases:** Solutions of fractional differential equations, viscoelastic
material models, anomalous diffusion, relaxation processes in disordered media.

## Function Family Summary

| Family | Functions | Typical Use Cases |
|--------|-----------|-------------------|
| Gamma | `gamma`, `lgamma`, `digamma`, `beta`, `gammainc`, `gammaincc`, `betainc` | Statistics, combinatorics, distribution CDFs |
| Error | `erf`, `erfc`, `erfinv`, `erfcinv` | Normal distribution, diffusion, heat equation |
| Bessel | `besselj`, `bessely`, `besseli`, `besselk` | Cylindrical symmetry, wave propagation, signal processing |
| Elliptic | `ellipk`, `ellipe`, `ellipf` | Pendulum, celestial mechanics, electrostatics |
| Airy | `airy_ai`, `airy_bi` | Quantum mechanics, optics, WKB approximation |
| Hypergeometric | `hyp1f1`, `hyp2f1` | General framework unifying many special functions |
| Polynomials | `legendre_p`, `hermite_h`, `laguerre_l`, `chebyshev_t`, `legendre_plm` | Spectral methods, quantum mechanics, quadrature |
| Zeta | `zeta`, `hurwitz_zeta` | Number theory, quantum field theory, thermodynamics |
| Misc | `dawson`, `fresnel_s`, `fresnel_c`, `mittag_leffler`, `mittag_leffler_1` | Spectroscopy, optics, fractional calculus |
