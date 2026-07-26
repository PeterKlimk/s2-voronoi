use glam::{DVec3, Vec3};

use crate::policy::{COPLANAR_PERTURBATION_SCALE, REFERENCE_AXIS_COMPONENT_SWITCH_F64};
use crate::tolerances::{NEAR_GREAT_CIRCLE_MAX_PLANE_SIN_TOL, NEAR_GREAT_CIRCLE_RMS_PLANE_SIN_TOL};

#[derive(Debug, Clone, Copy)]
pub(super) struct CoplanarClass {
    pub(super) normal: DVec3,
}

pub(super) fn maybe_perturb_coplanar(
    points: &[Vec3],
    err: &crate::VoronoiError,
) -> Option<Vec<Vec3>> {
    if !matches!(
        err,
        crate::VoronoiError::UnsupportedGeometry { .. } | crate::VoronoiError::ComputationFailed(_)
    ) {
        return None;
    }
    let mut canonical = points.to_vec();
    super::canonicalize_unit_points(&mut canonical);
    let class = classify_exact_affine_circle(&canonical)
        .or_else(|| classify_near_great_circle(&canonical))?;
    Some(perturb_coplanar_points(&canonical, class.normal))
}

/// Certify affine coplanarity in the actual canonical f32 model. The seed
/// plane is selected with bounded linear sweeps for a stable perturbation
/// direction; `orient3d == 0` then decides coplanarity exactly for those
/// binary input coordinates. No tolerance can turn a merely near-coplanar
/// ordinary input into this class.
pub(super) fn classify_exact_affine_circle(points: &[Vec3]) -> Option<CoplanarClass> {
    if points.len() < 4 {
        return None;
    }
    let ([a, b, c], normal) = stable_affine_plane(points)?;
    let a = robust_coord(dvec(points[a]));
    let b = robust_coord(dvec(points[b]));
    let c = robust_coord(dvec(points[c]));
    if points
        .iter()
        .all(|&p| robust::orient3d(a, b, c, robust_coord(dvec(p))) == 0.0)
    {
        Some(CoplanarClass { normal })
    } else {
        None
    }
}

/// Choose a well-spread affine plane seed in a fixed number of linear sweeps.
/// Returns `None` only when fewer than three distinct, non-collinear points are
/// available or the input is non-finite.
fn stable_affine_plane(points: &[Vec3]) -> Option<([usize; 3], DVec3)> {
    fn farthest_from(points: &[Vec3], pivot: usize) -> Option<usize> {
        let a = dvec(points[pivot]);
        let mut best = None;
        let mut best_distance2 = 0.0f64;
        for (i, &p) in points.iter().enumerate() {
            let distance2 = (dvec(p) - a).length_squared();
            if distance2.is_finite() && distance2 > best_distance2 {
                best_distance2 = distance2;
                best = Some(i);
            }
        }
        best
    }

    let mut a = 0usize;
    let mut b = farthest_from(points, a)?;
    a = farthest_from(points, b)?;
    b = farthest_from(points, a)?;

    let pa = dvec(points[a]);
    let ab = dvec(points[b]) - pa;
    let mut c = None;
    let mut best_cross = DVec3::ZERO;
    let mut best_area2 = 0.0f64;
    for (i, &p) in points.iter().enumerate() {
        let cross = ab.cross(dvec(p) - pa);
        let area2 = cross.length_squared();
        if area2.is_finite() && area2 > best_area2 {
            best_area2 = area2;
            best_cross = cross;
            c = Some(i);
        }
    }
    let c = c?;
    Some(([a, b, c], best_cross / best_area2.sqrt()))
}

#[inline]
fn robust_coord(p: DVec3) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

/// Compatibility classifier for nominal great-circle input whose canonical
/// f32 rounding prevents exact affine certification. Unlike the exact path,
/// this tolerance classifier requires full-circle coverage so an ordinary
/// large cell in a hemisphere cannot be misclassified as a degeneracy.
pub(super) fn classify_near_great_circle(points: &[Vec3]) -> Option<CoplanarClass> {
    if points.len() < 4 {
        return None;
    }

    let normal = stable_rank2_normal(points)?;
    let mut max_abs_dot = 0.0f64;
    let mut sum_dot2 = 0.0f64;
    for &p in points {
        let d = normal.dot(dvec(p)).abs();
        max_abs_dot = max_abs_dot.max(d);
        sum_dot2 += d * d;
    }
    let rms_dot = (sum_dot2 / points.len() as f64).sqrt();
    if max_abs_dot > NEAR_GREAT_CIRCLE_MAX_PLANE_SIN_TOL
        || rms_dot > NEAR_GREAT_CIRCLE_RMS_PLANE_SIN_TOL
    {
        return None;
    }

    if !covers_great_circle(points, normal) {
        return None;
    }

    Some(CoplanarClass { normal })
}

/// Find a numerically stable candidate normal in a fixed number of linear
/// sweeps. Fixed-count linear sweeps keep a failed large build from entering a
/// quadratic all-pairs great-circle probe before returning the original error.
///
/// This selection is deliberately conservative: failure to find a pair with
/// enough angular separation merely declines the perturbation retry. It cannot
/// create a false rank-2 classification because `classify_near_great_circle`
/// subsequently checks every point against the candidate plane and verifies
/// full-circle coverage. Re-pivoting at the farthest point handles ordered
/// two-arc inputs where no pair involving `points[0]` is sufficiently stable.
pub(super) fn stable_rank2_normal(points: &[Vec3]) -> Option<DVec3> {
    const SWEEPS: usize = 3;
    const MIN_CROSS_LEN2: f64 = 0.25;

    let mut pivot = 0usize;
    let mut best_cross = DVec3::ZERO;
    let mut best_len2 = 0.0f64;
    for _ in 0..SWEEPS {
        let a = dvec(points[pivot]);
        let mut next_pivot = pivot;
        let mut sweep_best_len2 = 0.0f64;
        for (i, &b32) in points.iter().enumerate() {
            let cross = a.cross(dvec(b32));
            let len2 = cross.length_squared();
            if len2 > sweep_best_len2 {
                sweep_best_len2 = len2;
                next_pivot = i;
            }
            if len2 > best_len2 {
                best_len2 = len2;
                best_cross = cross;
            }
        }
        if next_pivot == pivot {
            break;
        }
        pivot = next_pivot;
    }
    if best_len2 < MIN_CROSS_LEN2 {
        return None;
    }
    Some(best_cross / best_len2.sqrt())
}

fn covers_great_circle(points: &[Vec3], normal: DVec3) -> bool {
    let seed = if normal.x.abs() < REFERENCE_AXIS_COMPONENT_SWITCH_F64 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let e1 = normal.cross(seed).normalize();
    let e2 = normal.cross(e1).normalize();
    let mut angles: Vec<f64> = points
        .iter()
        .map(|&p| {
            let p = dvec(p);
            p.dot(e2).atan2(p.dot(e1)).rem_euclid(std::f64::consts::TAU)
        })
        .collect();
    angles.sort_by(|a, b| a.total_cmp(b));

    let mut max_gap = 0.0f64;
    for w in angles.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    if let (Some(first), Some(last)) = (angles.first(), angles.last()) {
        max_gap = max_gap.max(first + std::f64::consts::TAU - last);
    }

    // A full great-circle set has no empty semicircle. Smaller arcs are better
    // treated as hemisphere/large-cell fallback cases, not SoS perturbation.
    max_gap < std::f64::consts::PI
}

fn perturb_coplanar_points(points: &[Vec3], normal: DVec3) -> Vec<Vec3> {
    // This is a realized robust-mode joggle, not a symbolic-only SoS epsilon.
    // The stored-f32 topology/validation path still sees near-antipodal pole
    // edges for microscopic offsets on exact great-circle fixtures; the named
    // scale is the already-tested small-jitter regime for these inputs.
    let scale = COPLANAR_PERTURBATION_SCALE;
    points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let amp = scale * stable_signed_unit(i as u64);
            let q = (dvec(p) + normal * amp).normalize();
            Vec3::new(q.x as f32, q.y as f32, q.z as f32)
        })
        .collect()
}

fn stable_signed_unit(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = ((x >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let signed = 2.0 * unit - 1.0;
    if signed.abs() < 0.125 {
        if signed < 0.0 {
            -0.125
        } else {
            0.125
        }
    } else {
        signed
    }
}

#[inline]
fn dvec(p: Vec3) -> DVec3 {
    DVec3::new(p.x as f64, p.y as f64, p.z as f64)
}
