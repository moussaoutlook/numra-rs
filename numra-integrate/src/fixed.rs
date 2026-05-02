//! Fixed-order Gaussian quadrature rules.
//!
//! Provides Gauss-Legendre, Gauss-Laguerre, and Gauss-Hermite quadrature
//! with precomputed nodes and weights.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

// Quadrature tables contain precise mathematical constants that happen to
// resemble well-known constants like FRAC_1_SQRT_2. These are genuine
// quadrature node values, not approximations of std constants.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use numra_core::Scalar;

use crate::error::IntegrationError;

// ============================================================================
// Gauss-Legendre nodes and weights for [-1, 1]
// Stored as (positive_node, weight) pairs; rules are symmetric about 0.
// For odd n, the zero node and its weight are stored separately.
// ============================================================================

/// Gauss-Legendre quadrature nodes and weights for n = 1..20.
/// Each entry is (nodes, weights) where nodes are the positive abscissae
/// (the rule is symmetric, so negative abscissae have the same weights).
/// For odd n, the first entry has node = 0.
///
/// Source: Abramowitz & Stegun, Table 25.4; verified against DLMF.
struct GaussLegendreTable {
    /// For odd n: (weight_at_zero, &[(positive_node, weight)])
    /// For even n: (0.0, &[(positive_node, weight)])
    zero_weight: f64,
    pairs: &'static [(f64, f64)],
}

// Precomputed tables for n = 1..20
const GL: [GaussLegendreTable; 20] = [
    // n=1
    GaussLegendreTable {
        zero_weight: 2.0,
        pairs: &[],
    },
    // n=2
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[(0.5773502691896258, 1.0)],
    },
    // n=3
    GaussLegendreTable {
        zero_weight: 0.8888888888888889,
        pairs: &[(0.7745966692414834, 0.5555555555555556)],
    },
    // n=4
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.3399810435848563, 0.6521451548625461),
            (0.8611363115940526, 0.3478548451374538),
        ],
    },
    // n=5
    GaussLegendreTable {
        zero_weight: 0.5688888888888889,
        pairs: &[
            (0.5384693101056831, 0.4786286704993665),
            (0.9061798459386640, 0.2369268850561891),
        ],
    },
    // n=6
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.2386191860831969, 0.4679139345726910),
            (0.6612093864662645, 0.3607615730481386),
            (0.9324695142031521, 0.1713244923791704),
        ],
    },
    // n=7
    GaussLegendreTable {
        zero_weight: 0.4179591836734694,
        pairs: &[
            (0.4058451513773972, 0.3818300505051189),
            (0.7415311855993945, 0.2797053914892767),
            (0.9491079123427585, 0.1294849661688697),
        ],
    },
    // n=8
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.1834346424956498, 0.3626837833783620),
            (0.5255324099163290, 0.3137066458778873),
            (0.7966664774136267, 0.2223810344533745),
            (0.9602898564975363, 0.1012285362903763),
        ],
    },
    // n=9
    GaussLegendreTable {
        zero_weight: 0.3302393550012598,
        pairs: &[
            (0.3242534234038089, 0.3123470770400029),
            (0.6133714327005904, 0.2606106964029354),
            (0.8360311073266358, 0.1806481606948574),
            (0.9681602395076261, 0.0812743883615744),
        ],
    },
    // n=10
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.1488743389816312, 0.2955242247147529),
            (0.4333953941292472, 0.2692667193099963),
            (0.6794095682990244, 0.2190863625159820),
            (0.8650633666889845, 0.1494513491505806),
            (0.9739065285171717, 0.0666713443086881),
        ],
    },
    // n=11
    GaussLegendreTable {
        zero_weight: 0.2729250867779006,
        pairs: &[
            (0.2695431559523450, 0.2628045445102467),
            (0.5190961292068118, 0.2331937645919905),
            (0.7301520055740494, 0.1862902109277343),
            (0.8870625997680953, 0.1255803694649046),
            (0.9782286581460570, 0.0556685671161737),
        ],
    },
    // n=12
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.1252334085114689, 0.2491470458134028),
            (0.3678314989981802, 0.2334925365383548),
            (0.5873179542866175, 0.2031674267230659),
            (0.7699026741943047, 0.1600783285433462),
            (0.9041172563704749, 0.1069393259953184),
            (0.9815606342467192, 0.0471753363865118),
        ],
    },
    // n=13
    GaussLegendreTable {
        zero_weight: 0.2325515532308739,
        pairs: &[
            (0.2304583159551348, 0.2262831802628972),
            (0.4484927510364469, 0.2078160475368885),
            (0.6423493394403402, 0.1781459807619457),
            (0.8015780907333099, 0.1388735102197872),
            (0.9175983992229779, 0.0921214998377285),
            (0.9841830547185881, 0.0404840047653159),
        ],
    },
    // n=14
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.1080549487073437, 0.2152638534631578),
            (0.3191123689278898, 0.2051984637212956),
            (0.5152486363581541, 0.1855383974779378),
            (0.6872929048116855, 0.1572031671581935),
            (0.8272013150697650, 0.1215185706879032),
            (0.9284348836635735, 0.0801580871597602),
            (0.9862838086968123, 0.0351194603317519),
        ],
    },
    // n=15
    GaussLegendreTable {
        zero_weight: 0.2025782419255613,
        pairs: &[
            (0.2011940939974345, 0.1984314853271116),
            (0.3941513470775634, 0.1861610000155622),
            (0.5709721726085388, 0.1662692058169939),
            (0.7244177313601700, 0.1395706779261543),
            (0.8482065834104272, 0.1071592204671719),
            (0.9372733924007060, 0.0703660474881081),
            (0.9879925180204854, 0.0307532419961173),
        ],
    },
    // n=16
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.0950125098376374, 0.1894506104550685),
            (0.2816035507792589, 0.1826034150449236),
            (0.4580167776572274, 0.1691565193950025),
            (0.6178762444026438, 0.1495959888165767),
            (0.7554044083550030, 0.1246289712555339),
            (0.8656312023878318, 0.0951585116824928),
            (0.9445750230732326, 0.0622535239386479),
            (0.9894009349916499, 0.0271524594117541),
        ],
    },
    // n=17
    GaussLegendreTable {
        zero_weight: 0.1794464703562065,
        pairs: &[
            (0.1784841814958479, 0.1765627053669926),
            (0.3512317634538763, 0.1680041021564500),
            (0.5126905370864769, 0.1540457610768103),
            (0.6576711592166907, 0.1351363684685255),
            (0.7815140038968014, 0.1118838471934040),
            (0.8802391537269859, 0.0850361483171792),
            (0.9506755217687678, 0.0554595293739872),
            (0.9905754753144174, 0.0241483028685479),
        ],
    },
    // n=18
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.0847750130417353, 0.1691423829631436),
            (0.2518862256915055, 0.1642764837458327),
            (0.4117511614628426, 0.1546846751262652),
            (0.5597708310739475, 0.1406429146706507),
            (0.6916870430603532, 0.1225552067114785),
            (0.8037049589725231, 0.1009420441062872),
            (0.8926024664975557, 0.0764043285523263),
            (0.9558239495713977, 0.0497145488949698),
            (0.9915651684209309, 0.0216160135264833),
        ],
    },
    // n=19
    GaussLegendreTable {
        zero_weight: 0.1610544498487837,
        pairs: &[
            (0.1603867852552212, 0.1589688433939543),
            (0.3165640999636298, 0.1527660420658597),
            (0.4645707413759609, 0.1426067021736066),
            (0.6005453046616810, 0.1287539625393362),
            (0.7209661773352294, 0.1115666455473340),
            (0.8227146565371428, 0.0914900216224500),
            (0.9031559036148179, 0.0690445427376412),
            (0.9602081521348300, 0.0448142267656996),
            (0.9924068438435844, 0.0194617882297265),
        ],
    },
    // n=20
    GaussLegendreTable {
        zero_weight: 0.0,
        pairs: &[
            (0.0765265211334973, 0.1527533871307258),
            (0.2277858511416451, 0.1491729864726037),
            (0.3737060887154195, 0.1420961093183821),
            (0.5108670019508271, 0.1316886384491766),
            (0.6360536807265150, 0.1181945319615184),
            (0.7463319064601508, 0.1019301198172404),
            (0.8391169718222188, 0.0832767415767048),
            (0.9122344282513259, 0.0626720483341091),
            (0.9639719272779138, 0.0406014298003869),
            (0.9931285991850949, 0.0176140071391521),
        ],
    },
];

/// Gauss-Legendre quadrature over `[a, b]` with `n` points.
///
/// Exact for polynomials of degree `<= 2n - 1`.
/// Supports `n` from 1 to 20 with precomputed nodes/weights.
///
/// # Errors
///
/// Returns `IntegrationError::InvalidInput` if `n` is 0 or greater than 20.
pub fn gauss_legendre<S, F>(f: F, a: S, b: S, n: usize) -> Result<S, IntegrationError>
where
    S: Scalar,
    F: Fn(S) -> S,
{
    if n == 0 || n > 20 {
        return Err(IntegrationError::InvalidInput(format!(
            "gauss_legendre: n must be 1..20, got {}",
            n
        )));
    }

    let table = &GL[n - 1];
    let mid = (a + b) * S::HALF;
    let half_len = (b - a) * S::HALF;

    let mut sum = S::ZERO;

    // Zero node contribution (odd n)
    if table.zero_weight != 0.0 {
        sum += S::from_f64(table.zero_weight) * f(mid);
    }

    // Symmetric pairs
    for &(node, weight) in table.pairs {
        let x_pos = mid + half_len * S::from_f64(node);
        let x_neg = mid - half_len * S::from_f64(node);
        let w = S::from_f64(weight);
        sum += w * (f(x_pos) + f(x_neg));
    }

    Ok(sum * half_len)
}

// ============================================================================
// Gauss-Laguerre nodes and weights for [0, inf) with weight e^(-x)
// ============================================================================

/// Gauss-Laguerre nodes and weights for n = 1..10.
/// Each entry is a slice of (node, weight) pairs.
const LAGUERRE: [&[(f64, f64)]; 10] = [
    // n=1
    &[(1.0, 1.0)],
    // n=2
    &[
        (0.5857864376269050, 0.8535533905932738),
        (3.4142135623730950, 0.1464466094067263),
    ],
    // n=3
    &[
        (0.4157745567834791, 0.7110930099291730),
        (2.2942803602790417, 0.2785177335692409),
        (6.2899450829374792, 0.0103892565015861),
    ],
    // n=4
    &[
        (0.3225476896193923, 0.6031541043416336),
        (1.7457611011583466, 0.3574186924377997),
        (4.5366202969681338, 0.0388879085150054),
        (9.3950709123011331, 0.0005392947055613),
    ],
    // n=5
    &[
        (0.2635603197181409, 0.5217556105828087),
        (1.4134030591065168, 0.3986668110831759),
        (3.5964257710407221, 0.0759424496817076),
        (7.0858100058588376, 0.0036117586799220),
        (12.6408008442757826, 0.0000233699723858),
    ],
    // n=6
    &[
        (0.2228466041792607, 0.4589646739499636),
        (1.1889321016726231, 0.4170008307721210),
        (2.9927363260593141, 0.1133733820740450),
        (5.7751435691045105, 0.0103991974531490),
        (9.8374674183825899, 0.0002610172028150),
        (15.9828739806017017, 0.0000008985479064),
    ],
    // n=7
    &[
        (0.1930436765603623, 0.4093189517012739),
        (1.0266648953391920, 0.4218312778617198),
        (2.5678715903161220, 0.1471263486575052),
        (4.9003530845264845, 0.0206335144687170),
        (8.1821534445628607, 0.0010740101432807),
        (12.7341802917978137, 0.0000158654643486),
        (19.3957278622625403, 0.0000000317031548),
    ],
    // n=8
    &[
        (0.1702796323051010, 0.3691885893416375),
        (0.9037017767993799, 0.4187867808143430),
        (2.2510866298661307, 0.1757949866371718),
        (4.2667001702876588, 0.0333434922612156),
        (7.0459054023934657, 0.0027945362352257),
        (10.7585160101809952, 0.0000907650877336),
        (15.7406786412780046, 0.0000008485746716),
        (22.8631317368892641, 0.0000000010480012),
    ],
    // n=9
    &[
        (0.1523222277318064, 0.3361264114948080),
        (0.8072049551012728, 0.4112139804309697),
        (2.0051351556193953, 0.1992875253708959),
        (3.7834739733309144, 0.0474605627656516),
        (6.2049567778766197, 0.0055999297143511),
        (9.3729852516875898, 0.0003054237032950),
        (13.4662369110920935, 0.0000065921230234),
        (18.8335977889916966, 0.0000000412268815),
        (26.3740718909273768, 0.0000000000329188),
    ],
    // n=10
    &[
        (0.1377934705404924, 0.3084411157650190),
        (0.7294545495031705, 0.4011199291270945),
        (1.8083429017403160, 0.2180682876118094),
        (3.4014336978548995, 0.0620874560986777),
        (5.5524961400638012, 0.0095015169751811),
        (8.3301527467644965, 0.0007530083885876),
        (11.8437858379000656, 0.0000282592334960),
        (16.2792578313781021, 0.0000000424931398),
        (21.9965858119807620, 0.0000000000183956),
        (29.9206970122738766, 0.0000000000000009),
    ],
];

/// Gauss-Laguerre quadrature for semi-infinite integrals.
///
/// Approximates `integral_0^inf f(x) * exp(-x) dx` using `n` points.
/// The user provides `f(x)`; the weight function `exp(-x)` is built in.
///
/// # Errors
///
/// Returns `IntegrationError::InvalidInput` if `n` is 0 or greater than 10.
pub fn gauss_laguerre<S, F>(f: F, n: usize) -> Result<S, IntegrationError>
where
    S: Scalar,
    F: Fn(S) -> S,
{
    if n == 0 || n > 10 {
        return Err(IntegrationError::InvalidInput(format!(
            "gauss_laguerre: n must be 1..10, got {}",
            n
        )));
    }

    let table = LAGUERRE[n - 1];
    let mut sum = S::ZERO;
    for &(node, weight) in table {
        sum += S::from_f64(weight) * f(S::from_f64(node));
    }
    Ok(sum)
}

// ============================================================================
// Gauss-Hermite nodes and weights for (-inf, inf) with weight e^(-x^2)
// ============================================================================

/// Gauss-Hermite nodes and weights for n = 1..10.
/// Stored as (positive_node, weight) pairs; symmetric about 0.
/// For odd n, the zero node weight is stored separately.
struct GaussHermiteTable {
    zero_weight: f64,
    pairs: &'static [(f64, f64)],
}

const GH: [GaussHermiteTable; 10] = [
    // n=1
    GaussHermiteTable {
        zero_weight: 1.7724538509055159, // sqrt(pi)
        pairs: &[],
    },
    // n=2
    GaussHermiteTable {
        zero_weight: 0.0,
        pairs: &[(0.7071067811865476, 0.8862269254527580)],
    },
    // n=3
    GaussHermiteTable {
        zero_weight: 1.1816359006036774,
        pairs: &[(1.2247448713915890, 0.2954089751509193)],
    },
    // n=4
    GaussHermiteTable {
        zero_weight: 0.0,
        pairs: &[
            (0.5246476232752904, 0.8049140900055128),
            (1.6506801238857846, 0.0813128354472452),
        ],
    },
    // n=5
    GaussHermiteTable {
        zero_weight: 0.9453087204829419,
        pairs: &[
            (0.9585724646138185, 0.3936193231522412),
            (2.0201828704560856, 0.0199532420590459),
        ],
    },
    // n=6
    GaussHermiteTable {
        zero_weight: 0.0,
        pairs: &[
            (0.4360774119276165, 0.7246295952243925),
            (1.3358490740136970, 0.1570673203228566),
            (2.3506049736744922, 0.0045300099055088),
        ],
    },
    // n=7
    GaussHermiteTable {
        zero_weight: 0.8102646175568073,
        pairs: &[
            (0.8162878828589647, 0.4256072526101278),
            (1.6735516287674714, 0.0545155828191270),
            (2.6519613568352334, 0.0009717812450995),
        ],
    },
    // n=8
    GaussHermiteTable {
        zero_weight: 0.0,
        pairs: &[
            (0.3811869902073222, 0.6611470125582413),
            (1.1571937124467802, 0.2078023258148919),
            (1.9816567566958429, 0.0170779830074457),
            (2.9306374202572440, 0.0001996040722114),
        ],
    },
    // n=9
    GaussHermiteTable {
        zero_weight: 0.7202352156060510,
        pairs: &[
            (0.7235510187528376, 0.4326515590025558),
            (1.4685532892166679, 0.0886810550706316),
            (2.2665805845318431, 0.0049436242755369),
            (3.1909932017815276, 0.0000396069772633),
        ],
    },
    // n=10
    GaussHermiteTable {
        zero_weight: 0.0,
        pairs: &[
            (0.3429013272237046, 0.6108626337353258),
            (1.0366108297895137, 0.2401386110823147),
            (1.7566836492998818, 0.0338743944554811),
            (2.5327316742327897, 0.0013436457467812),
            (3.4361591188377376, 0.0000076404328552),
        ],
    },
];

/// Gauss-Hermite quadrature for integrals over `(-inf, inf)`.
///
/// Approximates `integral_{-inf}^{inf} f(x) * exp(-x^2) dx` using `n` points.
/// The user provides `f(x)`; the weight function `exp(-x^2)` is built in.
///
/// # Errors
///
/// Returns `IntegrationError::InvalidInput` if `n` is 0 or greater than 10.
pub fn gauss_hermite<S, F>(f: F, n: usize) -> Result<S, IntegrationError>
where
    S: Scalar,
    F: Fn(S) -> S,
{
    if n == 0 || n > 10 {
        return Err(IntegrationError::InvalidInput(format!(
            "gauss_hermite: n must be 1..10, got {}",
            n
        )));
    }

    let table = &GH[n - 1];
    let mut sum = S::ZERO;

    // Zero node
    if table.zero_weight != 0.0 {
        sum += S::from_f64(table.zero_weight) * f(S::ZERO);
    }

    // Symmetric pairs
    for &(node, weight) in table.pairs {
        let w = S::from_f64(weight);
        sum += w * (f(S::from_f64(node)) + f(S::from_f64(-node)));
    }

    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- Gauss-Legendre ---

    #[test]
    fn test_gl_constant() {
        // integral of 1 from -1 to 1 = 2
        let result: f64 = gauss_legendre(|_: f64| 1.0, -1.0, 1.0, 1).unwrap();
        assert_relative_eq!(result, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_gl_x_squared() {
        // n=2 is exact for poly degree <= 3
        let result: f64 = gauss_legendre(|x: f64| x * x, -1.0, 1.0, 2).unwrap();
        assert_relative_eq!(result, 2.0 / 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_gl_x_fourth() {
        // integral of x^4 from -1 to 1 = 2/5, exact for n >= 3
        let result: f64 = gauss_legendre(|x: f64| x.powi(4), -1.0, 1.0, 3).unwrap();
        assert_relative_eq!(result, 2.0 / 5.0, epsilon = 1e-14);
    }

    #[test]
    fn test_gl_interval_transform() {
        // integral of x^2 from 0 to 1 = 1/3
        let result: f64 = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_gl_high_order() {
        // integral of x^10 from -1 to 1 = 2/11, needs n >= 6 for exactness
        let result: f64 = gauss_legendre(|x: f64| x.powi(10), -1.0, 1.0, 6).unwrap();
        assert_relative_eq!(result, 2.0 / 11.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gl_n20_exp() {
        // integral of exp(x) from -1 to 1 = e - 1/e
        let result: f64 = gauss_legendre(|x: f64| x.exp(), -1.0, 1.0, 20).unwrap();
        let expected = core::f64::consts::E - 1.0 / core::f64::consts::E;
        assert_relative_eq!(result, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_gl_invalid_n() {
        let result: Result<f64, _> = gauss_legendre(|x: f64| x, 0.0, 1.0, 0);
        assert!(result.is_err());
        let result: Result<f64, _> = gauss_legendre(|x: f64| x, 0.0, 1.0, 21);
        assert!(result.is_err());
    }

    // --- Gauss-Laguerre ---

    #[test]
    fn test_laguerre_constant() {
        // integral_0^inf 1 * e^(-x) dx = 1
        let result: f64 = gauss_laguerre(|_: f64| 1.0, 5).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_laguerre_x() {
        // integral_0^inf x * e^(-x) dx = 1 (Gamma(2) = 1!)
        let result: f64 = gauss_laguerre(|x: f64| x, 5).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_laguerre_x_squared() {
        // integral_0^inf x^2 * e^(-x) dx = 2 (Gamma(3) = 2!)
        let result: f64 = gauss_laguerre(|x: f64| x * x, 5).unwrap();
        assert_relative_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_laguerre_invalid_n() {
        let result: Result<f64, _> = gauss_laguerre(|x: f64| x, 0);
        assert!(result.is_err());
        let result: Result<f64, _> = gauss_laguerre(|x: f64| x, 11);
        assert!(result.is_err());
    }

    // --- Gauss-Hermite ---

    #[test]
    fn test_hermite_constant() {
        // integral_{-inf}^{inf} 1 * e^(-x^2) dx = sqrt(pi)
        let result: f64 = gauss_hermite(|_: f64| 1.0, 5).unwrap();
        assert_relative_eq!(result, core::f64::consts::PI.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_hermite_x_squared() {
        // integral_{-inf}^{inf} x^2 * e^(-x^2) dx = sqrt(pi)/2
        let result: f64 = gauss_hermite(|x: f64| x * x, 5).unwrap();
        assert_relative_eq!(result, core::f64::consts::PI.sqrt() / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hermite_invalid_n() {
        let result: Result<f64, _> = gauss_hermite(|x: f64| x, 0);
        assert!(result.is_err());
        let result: Result<f64, _> = gauss_hermite(|x: f64| x, 11);
        assert!(result.is_err());
    }

    // --- f32 tests ---

    #[test]
    fn test_gl_f32() {
        let result: f32 = gauss_legendre(|x: f32| x * x, 0.0f32, 1.0f32, 5).unwrap();
        assert!((result - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_laguerre_f32() {
        let result: f32 = gauss_laguerre(|_: f32| 1.0f32, 5).unwrap();
        assert!((result - 1.0).abs() < 1e-5);
    }
}
