//! Owner-plane spherical arc geometry.
//!
//! Endpoint cross products become ill-conditioned as an edge approaches a
//! semicircle. Voronoi edges have stronger information: their two owning
//! generators define the supporting bisector plane independently of the
//! rounded output endpoints.

use glam::{DVec3, Vec3};

const OWNER_PLANE_SIN_TOL: f64 = 2.0e-6;
const EXACT_PI_SIN_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OwnerArcClass {
    Ordinary,
    NearPi,
    ExactPi,
    Invalid,
}

/// A shorter-than-pi arc stabilized against its owners' bisector plane.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OwnerArc {
    start: DVec3,
    tangent: DVec3,
    pub(crate) oriented_normal: DVec3,
    pub(crate) angle: f64,
    pub(crate) class: OwnerArcClass,
}

impl OwnerArc {
    #[inline]
    pub(crate) fn sample(self, t: f64) -> DVec3 {
        let angle = t * self.angle;
        (self.start * angle.cos() + self.tangent * angle.sin()).normalize_or_zero()
    }
}

/// Resolve a shorter-than-pi arc in f64. An error class means the endpoint
/// pair is exactly ambiguous or inconsistent with the supplied owners.
pub(crate) fn resolve_owner_arc(
    start: DVec3,
    end: DVec3,
    owner: DVec3,
    neighbor: DVec3,
    near_pi_dot_eps: f64,
) -> Result<OwnerArc, OwnerArcClass> {
    let start = start.normalize_or_zero();
    let end = end.normalize_or_zero();
    let mut normal = (owner.normalize_or_zero() - neighbor.normalize_or_zero()).normalize_or_zero();
    if start == DVec3::ZERO || end == DVec3::ZERO || normal == DVec3::ZERO {
        return Err(OwnerArcClass::Invalid);
    }

    let start_residual = normal.dot(start).abs();
    let end_residual = normal.dot(end).abs();
    let max_plane_residual = start_residual.max(end_residual);
    let start = (start - normal * normal.dot(start)).normalize_or_zero();
    let end = (end - normal * normal.dot(end)).normalize_or_zero();
    if start == DVec3::ZERO || end == DVec3::ZERO || max_plane_residual > OWNER_PLANE_SIN_TOL {
        return Err(OwnerArcClass::Invalid);
    }

    let cosine = start.dot(end).clamp(-1.0, 1.0);
    let unsigned_sine = start.cross(end).length();
    if cosine < 0.0 && unsigned_sine <= EXACT_PI_SIN_TOL {
        return Err(OwnerArcClass::ExactPi);
    }

    let mut tangent = normal.cross(start).normalize_or_zero();
    if tangent == DVec3::ZERO {
        return Err(OwnerArcClass::Invalid);
    }
    if tangent.dot(end) < 0.0 {
        normal = -normal;
        tangent = -tangent;
    }
    let sine = tangent.dot(end).max(0.0);
    let angle = sine.atan2(cosine);
    let class = if cosine <= -1.0 + near_pi_dot_eps {
        OwnerArcClass::NearPi
    } else {
        OwnerArcClass::Ordinary
    };
    Ok(OwnerArc {
        start,
        tangent,
        oriented_normal: normal,
        angle,
        class,
    })
}

/// Stabilize one Voronoi edge against the bisector plane of its two owners.
pub(crate) fn classify_owner_arc(
    start: Vec3,
    end: Vec3,
    owner: Vec3,
    neighbor: Vec3,
    near_pi_dot_eps: f32,
) -> OwnerArcClass {
    match resolve_owner_arc(
        start.as_dvec3(),
        end.as_dvec3(),
        owner.as_dvec3(),
        neighbor.as_dvec3(),
        near_pi_dot_eps as f64,
    ) {
        Ok(arc) => arc.class,
        Err(class) => class,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_plane_distinguishes_near_pi_from_exact_pi() {
        let owner = Vec3::Z;
        let neighbor = -Vec3::Z;
        let near = classify_owner_arc(
            Vec3::X,
            Vec3::new(-1.0, 1.0e-3, 0.0).normalize(),
            owner,
            neighbor,
            1.0e-5,
        );
        assert_eq!(near, OwnerArcClass::NearPi);

        let resolved = resolve_owner_arc(
            DVec3::X,
            DVec3::new(-1.0, 1.0e-3, 0.0).normalize(),
            DVec3::Z,
            -DVec3::Z,
            1.0e-5,
        )
        .expect("near-pi arc should resolve");
        let midpoint = resolved.sample(0.5);
        assert!(midpoint.dot(DVec3::Y) > 0.999);
        assert!(resolved.angle < std::f64::consts::PI);

        let exact = classify_owner_arc(Vec3::X, -Vec3::X, owner, neighbor, 1.0e-5);
        assert_eq!(exact, OwnerArcClass::ExactPi);
    }

    #[test]
    fn owner_plane_rejects_inconsistent_or_missing_owners() {
        let off_plane = classify_owner_arc(
            Vec3::new(1.0, 0.0, 1.0e-4).normalize(),
            Vec3::new(-1.0, 1.0e-3, 0.0).normalize(),
            Vec3::Z,
            -Vec3::Z,
            1.0e-5,
        );
        assert_eq!(off_plane, OwnerArcClass::Invalid);

        let same_owner = classify_owner_arc(
            Vec3::X,
            Vec3::new(-1.0, 1.0e-3, 0.0).normalize(),
            Vec3::Z,
            Vec3::Z,
            1.0e-5,
        );
        assert_eq!(same_owner, OwnerArcClass::Invalid);
    }
}
