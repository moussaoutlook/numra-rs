# Numra Capability Audit & Roadmap                                                           
                                                                                             
  Synthesized from 21 crates × 5 parallel auditors. Numbers in parentheses are file
  references.

  **Verified pass (2026-05-08):** Every claim below was checked against source and tests by
  5 parallel verification agents. Corrections applied: numra-linalg f64-only nuance, DDE
  state-dependent delays present, IDE Volterra type-2 only, Gauss-Laguerre/Hermite n≤10
  (not 20), Linear interpolant has derivative/integrate, FD step sizes vary by crate, OCP
  adjoint is private + DoPri5-only.
                                                                                             
  ---
  
## Part 1 — What Numra currently does                                                       
                                                                                           
### 1.1 Foundations                                                                            
                                                                                             
  Crate: numra-core                                                                       
  Provides: Scalar trait (f32/f64 via libm), Vector/Matrix traits, Signal (Harmonic, Step,   
    Ramp, Pulse, Chirp, Tabulated, Piecewise, Sum, Product), Uncertain<S> (mean+variance, 
    1st-order propagation), error types                                                    
  Notes: no-std + alloc; backbone for everything else                                      
  ────────────────────────────────────────                                                   
  Crate: numra-linalg                                                                     
  Provides: Dense LU/QR/Cholesky/SVD, symmetric & general eigendecomp (faer-backed); Krylov: 
    CG, GMRES(30), BiCGSTAB, MINRES; preconditioners: Jacobi, ILU(0), SSOR; sparse CSC via
    faer::SparseColMat                                                                       
  Notes: most ops generic over `S: Entity + ComplexField` (f32 + f64 supported by faer); only general (non-symmetric) eigendecomposition is hardcoded f64 (eigen.rs:63,90); sparse direct solvers convert to dense internally (sparse.rs:148–149)
  ────────────────────────────────────────                                                   
  Crate: numra-autodiff                                                                      
  Provides: Forward-mode Dual<S> (generic), reverse-mode Var with tape, gradient(),        
  hessian(),                                                                                 
    closure bridges                                      
  Notes: reverse mode hard-coded f64; no checkpointing; no sparse Jacobian                 
  ────────────────────────────────────────                                                 
  Crate: numra-special                                                                       
  Provides: Γ/lnΓ/digamma, β, regularized incomplete γ/β, erf/erfc/erfinv, Bessel J/Y/I/K
    (real), elliptic K/E/F, Airy Ai/Bi, ₁F₁ & ₂F₁, Legendre/Hermite/Laguerre/Chebyshev,      
    Riemann/Hurwitz ζ, Dawson, Fresnel, Mittag-Leffler   
  Notes: real arguments only                                                               
  ────────────────────────────────────────                                                 
  Crate: numra-fft                                                                           
  Provides: Complex FFT/iFFT (rustfft: Cooley-Tukey + Bluestein), 2D FFT, real FFT,
    periodogram, Welch, STFT, FFT convolution/correlation, windows                           
    (Hann/Hamming/Blackman/Kaiser/Flattop), fftfreq/fftshift
  Notes: no DCT/DST, no multitaper                                                         
                                                                                           
### 1.2 Differential equations

  Crate: numra-ode                                                                           
  Problem class: IVP dy/dt=f(t,y), mass-matrix DAE up to index-1 (Pantelides for index-2/3)
  Solvers: Explicit: DoPri5, Tsit5, Vern6/7/8. Implicit/stiff: Radau5 (3-stage Radau IIA),   
    ESDIRK32/43/54, BDF/NDF 1–5 (SciPy port). Auto-selector. DAE init via Newton on
    constraints.                                                                           
  Adaptivity: PI step controller, embedded error, dense output, event detection (bisection)  
  Sensitivity: Forward sensitivity via augmented system (ParametricOdeSystem); MC + GUM
    uncertainty                                                                              
  ────────────────────────────────────────               
  Crate: numra-sde                                                                           
  Problem class: Itô SDEs (diagonal/scalar/general noise)                                  
  Solvers: Euler–Maruyama, Milstein (FD ∂g/∂x), SRA1 (strong-1.5), SRA2 (weak-2)             
  Adaptivity: Step-doubling                              
  Sensitivity: None                                                                        
  ────────────────────────────────────────                                                 
  Crate: numra-dde                                                                           
  Problem class: Constant AND state-dependent delays, multiple delays (system.rs:25–47); no neutral DDEs, no distributed delays
  Solvers: Method-of-Steps over DoPri5 + C¹ Hermite history; discontinuity propagation (cap  
    1000)                                                
  Adaptivity: inherited                                                                    
  Sensitivity: None                                                                        
  ────────────────────────────────────────
  Crate: numra-fde                                                                           
  Problem class: Caputo D^α y = f(t,y), 0 < α ≤ 1
  Solvers: L1 scheme (order 2-α) + Mittag-Leffler refs                                       
  Adaptivity: fixed step                                 
  Sensitivity: None                                                                        
  ────────────────────────────────────────                                                 
  Crate: numra-ide                                                                           
  Problem class: Volterra type-2 IDEs only (`y'(t) = f(t,y) + ∫₀ᵗ K(t,s,y(s)) ds`, system.rs:12–14); no type-1 form
  Solvers: Trapezoidal, RK4, Prony solver (O(n_terms) memory for sum-of-exp kernels)         
  Adaptivity: fixed step                                 
  Sensitivity: None                                                                        
  ────────────────────────────────────────                                                 
  Crate: numra-pde                                                                           
  Problem class: Parabolic PDEs (heat, advection-diffusion, reaction-diffusion) in 1D/2D/3D
    via MOL; Stefan moving boundary; Dirichlet/Neumann/Robin/periodic BCs                    
  Solvers: FD stencils → ODE solver of choice; sparse Laplacian assembly; parametric MOL2D/3D
                                                                                           
    with analytical Jacobians                                                              
  Adaptivity: inherited from ODE
  Sensitivity: via ODE sensitivity
  ────────────────────────────────────────
  Crate: numra-spde                                                                          
  Problem class: Parabolic SPDEs ∂u/∂t = L[u] + σ(u)ξ, white/spatial/trace-class noise
  Solvers: MOL + EM/Milstein on the resulting SDE; ensemble runner                           
  Adaptivity: step-doubling                              
  Sensitivity: None                                                                        
                                                                                           
### 1.3 Optimization

  Crate: numra-nonlinear                                                                     
  Capability: Newton-only for F(x)=0; analytical Jacobian or FD (h=1e-8); Armijo; Wolfe line
    search exposed for downstream use                                                        
  ────────────────────────────────────────                                                 
  Crate: numra-optim                                                                       
  Capability: Unconstrained: BFGS, L-BFGS, GD. Box: L-BFGS-B. Constrained: SQP, Augmented    
    Lagrangian. DFO: Nelder-Mead, Powell, CMA-ES, DE. LSQ: Levenberg–Marquardt. LP/QP/MILP:
    revised simplex, active-set, branch-and-bound. Multi-objective: NSGA-II. Stochastic: SAA.
                                                         
    Robust: ellipsoidal worst-case                                                           
  ────────────────────────────────────────                                                 
  Crate: numra-ocp                                                                           
  Capability: Single shooting, multiple shooting, direct collocation (trapezoidal,         
    Hermite-Simpson), forward & adjoint sensitivity, parameter estimation via LM. Wraps      
    numra-ode (DoPri5 only) + numra-optim                
  ────────────────────────────────────────                                                 
  Crate: numra-fit                                                                           
  Capability: LM curve fitting (FD Jacobian only), weighted variant, polynomial via SVD
    (lstsq)                                                                                  
                                                         
### 1.4 Numerical methods                                                                    
                                                                                           
  ┌─────────────────┬─────────────────────────────────────────────────────────────────────┐
  │      Crate      │                             Capability                              │
  ├─────────────────┼─────────────────────────────────────────────────────────────────────┤
  │                 │ Adaptive Gauss-Kronrod G7K15 (≈ SciPy quad),                        │
  │ numra-integrate │ Gauss-Legendre n≤20; Gauss-Laguerre & Gauss-Hermite n≤10            │
  │                 │ (adaptive G7K15, trapezoid/Simpson/cumtrapz, Romberg, dblquad)      │
  ├─────────────────┼─────────────────────────────────────────────────────────────────────┤
  │                 │ 1D only: linear (with derivative + integral), cubic spline          │
  │ numra-interp    │ (natural/clamped/not-a-knot), PCHIP, Akima, barycentric Lagrange    │
  │                 │ (with Chebyshev constructor); only Lagrange returns None for       │
  │                 │ derivative/integrate                                                 │
  ├─────────────────┼─────────────────────────────────────────────────────────────────────┤  
  │                 │ 11 distributions (Normal, Uniform, Exp, Gamma, Beta, LogNormal, χ², │
  │ numra-stats     │  t, F, Poisson, Binomial); descriptive stats; Pearson + Spearman    │  
  │                 │ correlation; t-tests, paired/Welch, χ², KS, ANOVA; OLS regression   │
  │                 │ (simple/multiple/polynomial) with std errors & p-values             │  
  ├─────────────────┼─────────────────────────────────────────────────────────────────────┤
  │                 │ Butterworth + Chebyshev-I lowpass IIR via SOS, FIR via windowed     │
  │ numra-signal    │ sinc, sosfilt + filtfilt, Hilbert + envelope + instantaneous        │  
  │                 │ frequency, FFT resampling, peak detection                           │
  │                 │ (height/distance/prominence)                                        │  
  └─────────────────┴─────────────────────────────────────────────────────────────────────┘
                                                                                           
### 1.5 Composability — the strong story                                                       
   
  This is Numra's distinguishing claim and it largely holds up:                              
                                                         
  - One trait spine — Scalar, Vector, Matrix, OdeSystem, Signal are reused across crates     
  (numra/src/lib.rs:18-28).                                                                
  - Real cross-crate plumbing that already works:                                            
    - PDE → ODE: MOL builds an OdeSystem consumed by any ODE solver (numra-pde/src/mol*.rs)  
    - SPDE → SDE: MOL produces an SDE system (numra-spde/src/solver.rs)                      
    - DDE → ODE: method-of-steps wraps DoPri5 + Hermite history                              
    - OCP → ODE + Optim: shooting/collocation transcribe to NLP                              
    - Forward sensitivity: ParametricOdeSystem augmented with any ODE solver                 
    - Six "interop workflow" integration tests verify ODE→Interp→Quad, ODE→FFT→Signal→Peaks, 
  ParamEst→Sensitivity→Uncertainty, Autodiff→Optim→Fit, PDE→Stats, Stats→MC→ODE              
  (numra/tests/interop_workflows.rs)                                                       
  - 15 worked examples + 13-chapter mdBook covering every advertised area. The self-published
   audit (docs/audit/book-coverage-matrix.md) shows 100% coverage parity between API         
  inventory and book.                                                                      
  - CHANGELOG trajectory: recent work has been about unification — Jacobian path through     
  OdeSystem::jacobian, BDF rewritten to match SciPy, parametric MOL2D/3D, sensitivity        
  benches. The project is currently consolidating, not sprawling.                          
                                                                                             
  ---                                                    
 
## Part 2 — What's missing to be "very complete"                                            
                                                                                             
  Ordered by impact × effort, with the highest-leverage items first.
                                                                                             
### Tier 1 — Foundational gaps that block whole capability classes                             
                                                                                             
  These unlock multiple downstream features at once. Build these first.                      
                                                                                           
  1. AD integration into the solver stack — numra-autodiff exists but is barely wired in:
    - numra-nonlinear, numra-optim, numra-fit, numra-ocp all use FD Jacobians (newton.rs:245
  uses 1e-8 single-sided; optim/problem.rs:488,510 uses 1e-8 central; fit/curve_fit.rs:80
  uses 1e-7 central — inconsistent across crates). numra-autodiff is declared as a Cargo
  dependency in numra-optim but never `use`d. Adding a Jacobian = AutoDiff | Analytical | FD
  enum and threading Dual<S> through means every solver gets exact gradients for free.
    - Reverse-mode currently locked to f64 → generalize to Scalar. Add Jacobian-vector /     
  vector-Jacobian product modes (JVP/VJP) so big problems don't materialize the full         
  Jacobian.                                                                                
    - Add checkpointing (Treeverse/Revolve) for long ODE adjoints.                           
  2. Sparse linear algebra completion — faer sparse is in the door but:                      
    - Sparse direct solvers (SparseLU, sparse Cholesky, AMD ordering) are stubs/half-wired.  
    - No banded/tridiagonal/Toeplitz fast paths (huge for 1D PDEs and BDF).                  
    - No iterative refinement, no equilibration. Add these and the implicit ODE / PDE / NLP  
  stacks all speed up.                                                                       
  3. DAE & mass-matrix story for everything beyond ODE — Index-1 DAE works in ODE; OCP,      
  sensitivity, and PDE coupling don't accept DAE form. Promote the DAE machinery to a        
  first-class problem class.                                                               
  4. Adjoint sensitivity for ODEs — only forward sensitivity exists. Adjoint is essential for
   parameter inference at scale (n_params ≫ n_states), PDE-constrained optimization, and     
  ML-style training. The OCP crate already has an adjoint costate solver — promote that to 
  the public ODE API.                                                                        
                                                         
### Tier 2 — Major missing problem classes                                                   
                                                                                           
  5. Hyperbolic & elliptic PDEs — numra-pde is parabolic-only. Add:                          
    - Elliptic Poisson/Helmholtz solvers (multigrid, AMG, FFT-based for Cartesian).
    - Hyperbolic solvers with flux limiters / WENO / discontinuous Galerkin for conservation 
  laws.                                                                                      
    - Spectral methods (Fourier on periodic, Chebyshev on intervals) — short, high-impact    
  addition.                                                                                  
    - FEM: there is none. A minimal P1/P2 finite-element kernel on simplicial meshes with  
  assembly + sparse solve unlocks structural / fluid / electromagnetics.                     
    - Unstructured grid + AMR.                                                             
  6. Boundary value problems for ODEs — currently only IVPs. Add a BVP module (collocation   
  with COLNEW/bvp4c-style mesh adaptation). Trivial to expose since the OCP collocation code 
  already does most of it.                                                                   
  7. Spectral & pseudospectral methods — collocation, Galerkin, Chebfun-style operations.    
  This is one mid-sized crate (numra-spectral) that would dramatically expand reachable      
  problems.                                                                                
  8. PINN / SciML bridge — operator-learning, neural ODE/PDE, sparse identification (SINDy). 
  At minimum, expose the existing autodiff + ODE adjoint as primitives a separate crate (or  
  feature) can build on.                                                                   
  9. Higher-order SDE schemes & broader noise models:                                        
    - Multi-dimensional Milstein, stochastic RK (Rößler), commutative-noise variants.        
    - Stratonovich form, Itô↔Stratonovich conversion.                                        
    - Jump diffusion / Lévy drives, tau-leap for chemical Langevin.                          
    - Multilevel Monte Carlo (MLMC) — huge variance-reduction win.                           
    - Adjoint sensitivity for SDEs.                                                          
  10. DDEs beyond constant + state-dependent — neutral DDEs (y' evaluated at delayed times),
  distributed delays, characteristic-equation linear stability analysis. (Note: state-
  dependent delays ARE already implemented — system.rs:25–47 — earlier draft was wrong.)
  11. FDE expansion — Riemann–Liouville and Grünwald–Letnikov definitions, distributed-order,
   tempered fractional, Adams-type predictor-corrector schemes, Caputo for α ∈ (1,2].        
                                                                                           
### Tier 3 — Optimization stack completion                                                     
                                                                                           
  12. Interior-point method (general nonlinear, IPOPT-class). The single biggest gap.        
  Combined with proper sparse linear algebra, unlocks PDE-constrained optimization and     
  large-scale NLP.                                                                           
  13. Conic optimization — SOCP and SDP. Required for modern robust/portfolio/control      
  problems.                                                                                  
  14. MPC — receding-horizon controller built on OCP + warm-starting (which OCP also lacks).
  Target real-time embedded use.                                                             
  15. Bayesian optimization / surrogate models — Gaussian processes + acquisition functions.
  Pairs with numra-stats.                                                                    
  16. Mesh refinement for direct collocation — current OCP uses fixed mesh.                
  17. Proximal & nonsmooth methods — ADMM, FISTA, proximal gradient, bundle. Required for    
  sparse regression, total variation, compressed sensing.                                    
  18. Quasi-Newton root-finders beyond plain Newton — Broyden, Anderson acceleration,        
  Levenberg-damped Newton. numra-nonlinear is one file.                                      
                                                                                           
### Tier 4 — Statistics / fit / analysis maturity                                              
                                                                                           
  19. GLM family + IRLS in numra-fit/numra-stats — logistic, Poisson, Gamma, NegBin. Today   
  only OLS.                                                                                
  20. Robust regression — Huber/Tukey M-estimators, RANSAC.                                  
  21. Regularization — Ridge, Lasso (LARS or coordinate descent), elastic-net.               
  22. Bayesian inference — MCMC (HMC/NUTS), variational inference. Pairs with autodiff.      
  23. Confidence/credible intervals + bootstrap + permutation tests across all of            
  numra-stats.                                                                               
  24. Multivariate distributions — multivariate normal, Wishart, Dirichlet, copulas.         
  25. Time series — ARMA/ARIMA, state-space models, Kalman/EKF/UKF/particle filter. Not      
  currently anywhere.                                                                        
  26. Density estimation — KDE, mixture models with EM.                                      
  27. Effect sizes, multiple-comparison correction (Bonferroni/Benjamini-Hochberg),          
  additional non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis).                  
                                                                                             
### Tier 5 — Quadrature, interpolation, signal completeness                                    
                                                                                           
  28. Quadrature: tanh-sinh, Clenshaw-Curtis, Cauchy principal value, Filon/Levin for        
  oscillatory, Genz-Malik adaptive cubature, Smolyak sparse grids.                         
  29. Multivariate interpolation: tensor-product bilinear/bicubic, scattered RBF, thin-plate 
  spline, kriging, B-splines, NURBS.                                                         
  30. Wavelets: DWT, CWT, wavelet packet, Daubechies/Symlet families. Currently absent.    
  31. Spectral estimation beyond Welch: multitaper (Slepian), MUSIC/ESPRIT, AR-based.        
  32. Filter design completeness: Chebyshev-II, elliptic, bandpass/highpass/bandstop, filter 
  banks, Savitzky-Golay, median, Kalman, adaptive (LMS/RLS).                                 
  33. DCT/DST and complex Hilbert across all dimensions.                                     
                                                                                             
### Tier 6 — Cross-cutting infrastructure                                                      
                                                                                             
  34. Parallelism story: rayon is a dep but barely used. Sweep ensemble runners, Jacobian    
  columns, MOL operator applies, Monte Carlo samplers, multi-start optimization to use it  
  explicitly.                                                                                
  35. GPU/SIMD path — even a cfg-gated faer-CUDA or wgpu backend behind the existing Matrix
  trait would matter.                                                                        
  36. f32 + complex first-class in the linalg/AD layer (currently f64-only). Complex<S>    
  exists in FFT but isn't a Scalar peer.                                                     
  37. Serde + checkpointing for long simulations and reproducibility.
  38. Problem-class facade — lib.rs re-exports per-domain entry points, but a top-level      
  Problem/Solve umbrella (à la SciML's solve(prob, alg; kwargs...)) would make the           
  composability story self-evident in a single import.                                       
  39. Verification suite — Numra benches against itself only. A SciPy/SUNDIALS/NLOPT         
  comparison harness (Python subprocess + JSON dump) would be a credibility multiplier.      
  40. Standalone examples for the workflows currently buried in interop_workflows.rs —     
  Optim+Autodiff, Fit, Stats, Linalg.                                                        
                                                         
  ---                                                                                        
  
## Suggested order of attack                                                                
                                                                                             
  If I had to pick five things to build next that maximize the "very complete" payoff per  
  unit effort:                                                                               
                                                         
  1. Wire numra-autodiff into every Jacobian/gradient call site (Tier 1.1). One-pass         
  refactor; immediate accuracy + speed wins everywhere.                                    
  2. Adjoint sensitivity in numra-ode (Tier 1.4). The OCP crate already has the costate
  machinery (`numra-ocp/src/adjoint.rs:adjoint_gradient`) but it is private to OCP and
  hardcoded to DoPri5 (OdeSolverChoice has only that variant). Promotion requires
  generalizing across all ODE solvers, not just re-exporting.
  3. Finish sparse linalg + add banded/tridiagonal (Tier 1.2). Unblocks BDF, PDE, IPM.     
  4. Add an interior-point NLP solver (Tier 3.12) backed by 1+3. This single addition + OCP  
  collocation = PDE-constrained optimization.                                                
  5. BVP + spectral methods + minimal FEM (Tier 2.5–7). Three small crates that close the    
  "PDE story" gap.                                                                           
                                                                                           
  Everything else is incremental. The first five are the structural unlocks.