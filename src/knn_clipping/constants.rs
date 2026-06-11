//! Shared constants for kNN clipping Voronoi construction and validation.

/// Treat generators as coincident when their dot product differs from 1 by this amount.
///
/// For unit vectors, `1 - dot ≈ (distance^2) / 2`. This threshold is derived from
/// f32 rounding on normalized inputs, with a small safety multiplier.
/// This is only for duplicate-generator handling; it does not address vertex degeneracy.
pub const COINCIDENT_DOT_TOL: f32 = 64.0 * f32::EPSILON * f32::EPSILON;

/// Squared Euclidean distance threshold for coincident generators.
#[inline]
pub const fn coincident_distance_sq() -> f32 {
    2.0 * COINCIDENT_DOT_TOL
}

/// Euclidean distance threshold for coincident generators.
#[inline]
pub fn coincident_distance() -> f32 {
    coincident_distance_sq().sqrt()
}

/// Default weld radius (chord distance) for `PreprocessMode::Weld`.
///
/// Generators closer than this are welded into one cell before construction.
/// The value is the coincident-distance floor (~1.4e-6), derived from f32
/// input rounding. Empirically (see `docs/correctness-contract.md`), the
/// worst adversarial construction that breaks the raw pipeline does so at a
/// minimum pairwise separation of ~1.2e-7, so this radius carries an order of
/// magnitude of safety margin while staying far below the point spacing of
/// any realistic well-spaced input (~1e-6 spacing requires ~1e13 uniform
/// points).
#[inline]
pub fn weld_radius() -> f32 {
    coincident_distance()
}

#[cfg(test)]
mod tests {
    use super::{coincident_distance, weld_radius};

    #[test]
    fn weld_radius_band() {
        let r = weld_radius();
        assert!(
            (1.0e-6..2.0e-6).contains(&r),
            "weld radius {r} should sit in the ~1e-6 band backed by the margin data"
        );
        assert!(r >= coincident_distance());
    }
}
