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

/// Stabilize one Voronoi edge against the bisector plane of its two owners.
pub(crate) fn classify_owner_arc(
    start: Vec3,
    end: Vec3,
    owner: Vec3,
    neighbor: Vec3,
    near_pi_dot_eps: f32,
) -> OwnerArcClass {
    let start = start.as_dvec3().normalize_or_zero();
    let end = end.as_dvec3().normalize_or_zero();
    let normal = (owner.as_dvec3().normalize_or_zero() - neighbor.as_dvec3().normalize_or_zero())
        .normalize_or_zero();
    if start == DVec3::ZERO || end == DVec3::ZERO || normal == DVec3::ZERO {
        return OwnerArcClass::Invalid;
    }

    let start_residual = normal.dot(start).abs();
    let end_residual = normal.dot(end).abs();
    let max_plane_residual = start_residual.max(end_residual);
    let start = (start - normal * normal.dot(start)).normalize_or_zero();
    let end = (end - normal * normal.dot(end)).normalize_or_zero();
    if start == DVec3::ZERO || end == DVec3::ZERO || max_plane_residual > OWNER_PLANE_SIN_TOL {
        return OwnerArcClass::Invalid;
    }

    let cosine = start.dot(end).clamp(-1.0, 1.0);
    let sine = start.cross(end).length();
    if cosine < 0.0 && sine <= EXACT_PI_SIN_TOL {
        OwnerArcClass::ExactPi
    } else if start.dot(end) <= -1.0 + near_pi_dot_eps as f64 {
        OwnerArcClass::NearPi
    } else {
        OwnerArcClass::Ordinary
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
