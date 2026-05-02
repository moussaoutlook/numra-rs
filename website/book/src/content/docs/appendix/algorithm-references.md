---
title: "Algorithm References"
---

This appendix lists the primary academic references for the algorithms
implemented in Numra.

## ODE Solvers

### Explicit Runge-Kutta

| Solver | Reference |
|--------|-----------|
| **DoPri5** | J.R. Dormand and P.J. Prince, "A family of embedded Runge-Kutta formulae," *J. Comp. Appl. Math.*, 6(1):19-26, 1980. |
| **Tsit5** | Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column simplifying assumption," *Computers & Mathematics with Applications*, 62(2):770-775, 2011. |
| **Vern6** | J.H. Verner, "Numerically optimal Runge-Kutta pairs with interpolants," *Numerical Algorithms*, 53(2-3):383-396, 2010. |
| **Vern7** | J.H. Verner, same reference as Vern6. |
| **Vern8** | J.H. Verner, same reference as Vern6. Extended to 8th order. |

### Implicit Runge-Kutta

| Solver | Reference |
|--------|-----------|
| **Radau5** | E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, Springer, 2nd ed., 1996. Chapter IV. |
| **ESDIRK32, ESDIRK54** | C.A. Kennedy and M.H. Carpenter, "Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review," NASA/TM-2016-219173, 2016. |
| **ESDIRK43** | A. Kvaerno, "Singly diagonally implicit Runge-Kutta methods with an explicit first stage," *BIT Numerical Mathematics*, 44:489-502, 2004. |

### Multistep

| Solver | Reference |
|--------|-----------|
| **BDF** | C.W. Gear, *Numerical Initial Value Problems in Ordinary Differential Equations*, Prentice-Hall, 1971. See also Hairer & Wanner (1996), Chapter III. |

### Adaptive Step Size Control

The step size controllers follow the PI controller framework:

- K. Gustafsson, "Control-theoretic techniques for stepsize selection in
  explicit Runge-Kutta methods," *ACM TOMS*, 17(4):533-554, 1991.

- E. Hairer, S.P. Norsett, G. Wanner, *Solving Ordinary Differential
  Equations I: Nonstiff Problems*, Springer, 2nd ed., 1993. Section II.4.

## SDE Solvers

| Solver | Reference |
|--------|-----------|
| **Euler-Maruyama** | G. Maruyama, "Continuous Markov processes and stochastic equations," *Rendiconti del Circolo Matematico di Palermo*, 4:48-90, 1955. |
| **Milstein** | G.N. Milstein, "Approximate integration of stochastic differential equations," *Theory of Probability & Its Applications*, 19(3):557-562, 1975. |
| **SRI-W1** | A. Rossler, "Runge-Kutta methods for the strong approximation of solutions of stochastic differential equations," *SIAM J. Numer. Anal.*, 48(3):922-952, 2010. |

## DDE Solvers

| Solver | Reference |
|--------|-----------|
| **DDE-DoPri5** | Based on the DoPri5 scheme with continuous output for delay interpolation. See: C.T.H. Baker, C.A.H. Paul, D.R. Wille, "Issues in the numerical solution of evolutionary delay differential equations," *Advances in Computational Mathematics*, 3:171-196, 1995. |

## Linear Algebra

| Algorithm | Reference |
|-----------|-----------|
| **LU factorization** | G.H. Golub and C.F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013. |
| **QR factorization** | Same as above, Chapter 5. |
| **Cholesky** | Same as above, Chapter 4. |
| **Conjugate Gradient** | M.R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving linear systems," *J. Research of the National Bureau of Standards*, 49(6):409-436, 1952. |
| **GMRES** | Y. Saad and M.H. Schultz, "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems," *SIAM J. Sci. Stat. Comput.*, 7(3):856-869, 1986. |
| **BiCGSTAB** | H.A. van der Vorst, "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems," *SIAM J. Sci. Stat. Comput.*, 13(2):631-644, 1992. |

## Numerical Integration

| Algorithm | Reference |
|-----------|-----------|
| **Gauss-Kronrod (G7K15)** | A.S. Kronrod, *Nodes and Weights for Quadrature Formulae*, Consultants Bureau, 1965. |
| **Romberg** | W. Romberg, "Vereinfachte numerische Integration," *Det Kongelige Norske Videnskabers Selskab Forhandlinger*, 28(7):30-36, 1955. |
| **Gauss-Legendre** | See Golub & Van Loan (2013) for nodes/weights computation. |

## Interpolation

| Algorithm | Reference |
|-----------|-----------|
| **Cubic Spline** | C. de Boor, *A Practical Guide to Splines*, Springer, 1978. |
| **PCHIP** | F.N. Fritsch and R.E. Carlson, "Monotone piecewise cubic interpolation," *SIAM J. Numer. Anal.*, 17(2):238-246, 1980. |
| **Akima** | H. Akima, "A new method of interpolation and smooth curve fitting based on local procedures," *J. ACM*, 17(4):589-602, 1970. |
| **Barycentric Lagrange** | J.-P. Berrut and L.N. Trefethen, "Barycentric Lagrange interpolation," *SIAM Review*, 46(3):501-517, 2004. |

## Optimization

| Algorithm | Reference |
|-----------|-----------|
| **BFGS** | R. Fletcher, *Practical Methods of Optimization*, 2nd ed., Wiley, 1987. |
| **L-BFGS** | D.C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," *Mathematical Programming*, 45(1-3):503-528, 1989. |
| **Levenberg-Marquardt** | K. Levenberg (1944) and D.W. Marquardt (1963). See J.J. More, "The Levenberg-Marquardt algorithm: implementation and theory," in *Numerical Analysis*, Springer, 1978. |
| **CMA-ES** | N. Hansen and A. Ostermeier, "Completely derandomized self-adaptation in evolution strategies," *Evolutionary Computation*, 9(2):159-195, 2001. |
| **NSGA-II** | K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," *IEEE Trans. Evolutionary Computation*, 6(2):182-197, 2002. |
| **SQP** | P.T. Boggs and J.W. Tolle, "Sequential quadratic programming," *Acta Numerica*, 4:1-51, 1995. |

## Special Functions

| Function | Reference |
|----------|-----------|
| **Gamma function** | W.J. Cody, "An overview of software development for special functions," in *Numerical Analysis*, Springer, 1975. |
| **Error function** | W.J. Cody, "Rational Chebyshev approximations for the error function," *Mathematics of Computation*, 23(107):631-637, 1969. |
| **Bessel functions** | M. Abramowitz and I.A. Stegun, *Handbook of Mathematical Functions*, Dover, 1964. |
| **Mittag-Leffler** | R. Garrappa, "Numerical computation of the Mittag-Leffler function," *SIAM J. Numer. Anal.*, 53(3):1350-1369, 2015. |

## Signal Processing

| Algorithm | Reference |
|-----------|-----------|
| **FFT** | J.W. Cooley and J.W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Mathematics of Computation*, 19(90):297-301, 1965. |
| **Kaiser window** | J.F. Kaiser, "Nonrecursive digital filter design using the I0-sinh window function," in *Proc. IEEE Int. Symp. Circuits and Systems*, 1974. |
| **Butterworth filter** | S. Butterworth, "On the theory of filter amplifiers," *Wireless Engineer*, 7:536-541, 1930. |

## Textbooks

These comprehensive references cover much of the theory behind Numra:

1. E. Hairer, S.P. Norsett, G. Wanner, *Solving Ordinary Differential
   Equations I: Nonstiff Problems*, 2nd ed., Springer, 1993.

2. E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II:
   Stiff and Differential-Algebraic Problems*, 2nd ed., Springer, 1996.

3. L.N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM, 1997.

4. J. Nocedal and S.J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.

5. P.E. Kloeden and E. Platen, *Numerical Solution of Stochastic Differential
   Equations*, Springer, 1992.
