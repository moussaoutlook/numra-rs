// Clippy: reference values and numerical loops in tests / implementations.
#![allow(clippy::excessive_precision)]
#![allow(clippy::unnecessary_cast)]

//! Special mathematical functions for Numra.
//!
//! This crate provides implementations of special functions commonly needed in
//! scientific computing, physics, and engineering.
//!
//! # Function families
//!
//! - **Gamma**: gamma, lgamma, digamma, beta, incomplete gamma/beta
//! - **Error functions**: erf, erfc, erfinv, erfcinv
//! - **Bessel**: J, Y, I, K for real order
//! - **Elliptic integrals**: complete K(m), E(m), incomplete F(phi,m)
//! - **Airy**: Ai(x), Bi(x)
//! - **Hypergeometric**: 1F1, 2F1
//! - **Orthogonal polynomials**: Legendre, Hermite, Laguerre, Chebyshev
//! - **Zeta**: Riemann zeta, Hurwitz zeta
//! - **Misc**: Dawson, Fresnel, Mittag-Leffler
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod airy;
pub mod bessel;
pub mod elliptic;
pub mod erf;
pub mod error;
pub mod gamma;
pub mod hypergeometric;
pub mod misc;
pub mod orthogonal;
pub mod zeta;

pub use error::SpecialError;

// Gamma family
pub use gamma::{beta, betainc, digamma, gamma, gammainc, gammaincc, lgamma};

// Error functions
pub use erf::{erf, erfc, erfcinv, erfinv};

// Bessel functions
pub use bessel::{besseli, besselj, besselk, bessely};

// Elliptic integrals
pub use elliptic::{ellipe, ellipf, ellipk};

// Airy functions
pub use airy::{airy_ai, airy_bi};

// Hypergeometric functions
pub use hypergeometric::{hyp1f1, hyp2f1};

// Orthogonal polynomials
pub use orthogonal::{chebyshev_t, hermite_h, laguerre_l, legendre_p, legendre_plm};

// Zeta functions
pub use zeta::{hurwitz_zeta, zeta};

// Miscellaneous
pub use misc::{dawson, fresnel_c, fresnel_s, mittag_leffler, mittag_leffler_1};
