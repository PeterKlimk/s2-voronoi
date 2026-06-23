//! Exact canonical predicates (P5): deterministic, shared answers for
//! near-tie clip decisions, evaluated on the same raw f32 bits by every
//! cell. Exactness (Shewchuk adaptive via
//! the `robust` crate) makes the signs permutation-coherent across the
//! phrasings of a 4-point question — the property that kills the
//! parity-contradiction defect class.

use glam::Vec3;
use robust::Coord3D;

#[inline]
fn c3(p: Vec3) -> Coord3D<f64> {
    Coord3D {
        x: p.x as f64,
        y: p.y as f64,
        z: p.z as f64,
    }
}

/// Exact canonical in-circle on the unit sphere.
///
/// Returns +1 if `h` lies strictly inside the circumcircle of (g, a, b)
/// (the vertex (g,a,b) must be cut by bisector(g,h)), -1 if strictly
/// outside (the vertex is kept), 0 on an exact tie (4-cocircular, or a
/// degenerate triple).
///
/// Derivation: the Voronoi vertex v of (g,a,b) is the circumcircle pole
/// with v.g > 0; h inside the cap centered at v through g iff
/// (h-g).v > 0. With v ~ s*(a-g)x(b-g) and s = sign(((a-g)x(b-g)).g)
/// = sign(det[a,b,g]):
///
///   in_circle(g,a,b;h) = sign(det[a-g, b-g, h-g]) * sign(det[a, b, g])
///
/// Both determinants are evaluated exactly (Shewchuk adaptive orient3d;
/// f32 inputs are exactly representable in f64), so the sign is exact and
/// permutation-consistent by construction.
pub(crate) fn in_circle_sphere_sign(g: Vec3, a: Vec3, b: Vec3, h: Vec3) -> i8 {
    // orient3d(pa,pb,pc,pd) computes det of rows (pa-pd, pb-pd, pc-pd).
    let d1 = robust::orient3d(c3(a), c3(b), c3(h), c3(g));
    let d2 = robust::orient3d(
        c3(a),
        c3(b),
        c3(g),
        Coord3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    );
    let prod_sign = (d1 > 0.0) as i8 - (d1 < 0.0) as i8;
    let pole_sign = (d2 > 0.0) as i8 - (d2 < 0.0) as i8;
    prod_sign * pole_sign
}
