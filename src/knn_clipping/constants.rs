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

/// Compute density-adaptive merge threshold.
///
/// At high densities, generators that are very close together can cause
/// numerical issues where different triplets produce the same f32 vertex
/// position. We merge generators that are within a fraction of the mean spacing.
///
/// Returns (fixed_threshold, density_fraction)
/// The actual threshold used is: max(fixed_threshold, mean_spacing * density_fraction)
#[inline]
pub fn merge_threshold_for_density(num_points: usize) -> f32 {
    // Mean angular spacing on unit sphere: sqrt(4π / n)
    // For chord distance: 2 * sin(θ/2) ≈ θ for small θ
    let mean_spacing = (4.0 * std::f32::consts::PI / num_points as f32).sqrt();

    // Use a fraction of mean spacing as merge threshold.
    // This fraction is chosen so that merged points don't significantly
    // affect the diagram quality, but prevents numerical degeneracies.
    //
    // At 4M points: mean_spacing ≈ 0.00177, threshold ≈ 1.8e-5
    // At 1M points: mean_spacing ≈ 0.00354, threshold ≈ 3.5e-5
    let density_threshold = mean_spacing * 0.01;

    // Use the larger of fixed (duplicate) threshold and density threshold
    coincident_distance().max(density_threshold)
}
