#![allow(dead_code)]

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use s2_voronoi::UnitVec3;
use std::f32::consts::PI;

/// Generate random points uniformly distributed on the unit sphere.
pub fn random_sphere_points(n: usize, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    random_sphere_points_with_rng(n, &mut rng)
}

pub fn random_sphere_points_with_rng<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<UnitVec3> {
    (0..n)
        .map(|_| {
            let z: f32 = rng.gen_range(-1.0..1.0);
            let theta: f32 = rng.gen_range(0.0..2.0 * PI);
            let r = (1.0 - z * z).sqrt();
            UnitVec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

/// Generate Fibonacci sphere points (more uniform than random).
pub fn fibonacci_sphere_points(n: usize, jitter: f32, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let golden_angle = PI * (3.0 - 5.0f32.sqrt());

    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f32;

            let mut x = radius * theta.cos();
            let mut z = radius * theta.sin();

            if jitter > 0.0 {
                x += rng.gen_range(-jitter..jitter);
                z += rng.gen_range(-jitter..jitter);
            }

            let len = (x * x + y * y + z * z).sqrt();
            UnitVec3::new(x / len, y / len, z / len)
        })
        .collect()
}

// =============================================================================
// Adversarial Point Generators for Stress Testing
// =============================================================================

/// Generate points along a great circle (all coplanar through origin).
///
/// This is an extreme degenerate case: all Voronoi edges are along the
/// perpendicular great circle, and vertices are at the poles of the circle.
/// The algorithm should handle this gracefully (may degrade but shouldn't panic).
pub fn great_circle_points(n: usize, jitter: f32, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..n)
        .map(|i| {
            let theta = 2.0 * PI * i as f32 / n as f32;
            let mut x = theta.cos();
            let mut y = theta.sin();
            let mut z = 0.0f32;

            if jitter > 0.0 {
                x += rng.gen_range(-jitter..jitter);
                y += rng.gen_range(-jitter..jitter);
                z += rng.gen_range(-jitter..jitter);
            }

            let len = (x * x + y * y + z * z).sqrt();
            UnitVec3::new(x / len, y / len, z / len)
        })
        .collect()
}

/// Generate points clustered in a small spherical cap around the north pole.
///
/// When `cap_radius_rad` is very small (e.g., 0.01), points are extremely close
/// together, stressing numerical precision and vertex deduplication.
///
/// Includes 6 "anchor" points on the axes to prevent any cell from spanning >90°.
/// Without anchors, a single clustered region creates cells that extend beyond
/// the gnomonic projection limit.
pub fn clustered_cap_points(n: usize, cap_radius_rad: f32, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_anchors = 6;
    let n_clustered = n.saturating_sub(n_anchors);

    let mut points = Vec::with_capacity(n);

    // Anchor points on axes (octahedron vertices) to bound cell sizes
    points.push(UnitVec3::new(1.0, 0.0, 0.0));
    points.push(UnitVec3::new(-1.0, 0.0, 0.0));
    points.push(UnitVec3::new(0.0, 1.0, 0.0));
    points.push(UnitVec3::new(0.0, -1.0, 0.0));
    points.push(UnitVec3::new(0.0, 0.0, 1.0));
    points.push(UnitVec3::new(0.0, 0.0, -1.0));

    // Clustered points around north pole
    for _ in 0..n_clustered {
        let u: f32 = rng.gen();
        let cos_theta_max = cap_radius_rad.cos();
        let cos_theta = 1.0 - u * (1.0 - cos_theta_max);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi: f32 = rng.gen_range(0.0..2.0 * PI);
        points.push(UnitVec3::new(
            sin_theta * phi.cos(),
            sin_theta * phi.sin(),
            cos_theta,
        ));
    }

    points
}

/// Generate points near cube vertices (stress cube-face stitching logic).
///
/// Places points near the 8 corners of the inscribed cube, where 3 cube faces
/// meet. This stresses the `CubeMapGrid` neighbor/ring2 computations.
pub fn cube_vertex_stress_points(n: usize, spread_rad: f32, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // 8 cube vertices (normalized to unit sphere)
    let inv_sqrt3 = 1.0 / 3.0f32.sqrt();
    let cube_vertices: [(f32, f32, f32); 8] = [
        (inv_sqrt3, inv_sqrt3, inv_sqrt3),
        (inv_sqrt3, inv_sqrt3, -inv_sqrt3),
        (inv_sqrt3, -inv_sqrt3, inv_sqrt3),
        (inv_sqrt3, -inv_sqrt3, -inv_sqrt3),
        (-inv_sqrt3, inv_sqrt3, inv_sqrt3),
        (-inv_sqrt3, inv_sqrt3, -inv_sqrt3),
        (-inv_sqrt3, -inv_sqrt3, inv_sqrt3),
        (-inv_sqrt3, -inv_sqrt3, -inv_sqrt3),
    ];

    (0..n)
        .map(|i| {
            let (cx, cy, cz) = cube_vertices[i % 8];

            // Generate small perturbation in tangent plane
            let u: f32 = rng.gen_range(-spread_rad..spread_rad);
            let v: f32 = rng.gen_range(-spread_rad..spread_rad);

            // Tangent vectors (arbitrary choice that works for cube vertices)
            let arbitrary = if cx.abs() < 0.9 {
                (1.0, 0.0, 0.0)
            } else {
                (0.0, 1.0, 0.0)
            };
            let t1 = (
                cy * arbitrary.2 - cz * arbitrary.1,
                cz * arbitrary.0 - cx * arbitrary.2,
                cx * arbitrary.1 - cy * arbitrary.0,
            );
            let t1_len = (t1.0 * t1.0 + t1.1 * t1.1 + t1.2 * t1.2).sqrt();
            let t1 = (t1.0 / t1_len, t1.1 / t1_len, t1.2 / t1_len);
            let t2 = (
                cy * t1.2 - cz * t1.1,
                cz * t1.0 - cx * t1.2,
                cx * t1.1 - cy * t1.0,
            );

            let px = cx + u * t1.0 + v * t2.0;
            let py = cy + u * t1.1 + v * t2.1;
            let pz = cz + u * t1.2 + v * t2.2;

            let len = (px * px + py * py + pz * pz).sqrt();
            UnitVec3::new(px / len, py / len, pz / len)
        })
        .collect()
}

/// Generate groups of 4 near-cocircular points (stress vertex stability).
///
/// Each group of 4 points is placed nearly on a circle, so the circumcenter
/// is numerically unstable. This can trigger bad edges due to:
/// - Endpoint mismatch (vertices differ slightly between cells)
/// - One-sided edges (tiny marginal edges only visible from one side)
pub fn near_cocircular_stress_points(
    n_groups: usize,
    perturbation: f32,
    seed: u64,
) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n_groups * 4);

    for _ in 0..n_groups {
        // Random center on sphere
        let z: f32 = rng.gen_range(-1.0..1.0);
        let theta: f32 = rng.gen_range(0.0..2.0 * PI);
        let r = (1.0 - z * z).sqrt();
        let center = (r * theta.cos(), r * theta.sin(), z);

        // Tangent basis at center
        let arbitrary = if center.0.abs() < 0.9 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };
        let t1 = (
            center.1 * arbitrary.2 - center.2 * arbitrary.1,
            center.2 * arbitrary.0 - center.0 * arbitrary.2,
            center.0 * arbitrary.1 - center.1 * arbitrary.0,
        );
        let t1_len = (t1.0 * t1.0 + t1.1 * t1.1 + t1.2 * t1.2).sqrt();
        let t1 = (t1.0 / t1_len, t1.1 / t1_len, t1.2 / t1_len);
        let t2 = (
            center.1 * t1.2 - center.2 * t1.1,
            center.2 * t1.0 - center.0 * t1.2,
            center.0 * t1.1 - center.1 * t1.0,
        );

        // Circle radius (angular distance from center)
        let circle_radius: f32 = rng.gen_range(0.05..0.2);

        // Place 4 points nearly on the circle
        for k in 0..4 {
            let angle = (k as f32 / 4.0) * 2.0 * PI + rng.gen_range(-0.1..0.1);
            let r_perturb = circle_radius + rng.gen_range(-perturbation..perturbation);

            let px = center.0 + r_perturb * (angle.cos() * t1.0 + angle.sin() * t2.0);
            let py = center.1 + r_perturb * (angle.cos() * t1.1 + angle.sin() * t2.1);
            let pz = center.2 + r_perturb * (angle.cos() * t1.2 + angle.sin() * t2.2);

            let len = (px * px + py * py + pz * pz).sqrt();
            points.push(UnitVec3::new(px / len, py / len, pz / len));
        }
    }

    points
}

/// Generate points on a hemisphere (asymmetric distribution).
///
/// Tests behavior when points don't cover the full sphere.
/// The opposite hemisphere will have very large cells or undefined behavior.
pub fn hemisphere_points(n: usize, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..n)
        .map(|_| {
            let z: f32 = rng.gen_range(0.0..1.0); // Only upper hemisphere
            let theta: f32 = rng.gen_range(0.0..2.0 * PI);
            let r = (1.0 - z * z).sqrt();
            UnitVec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

/// Generate bimodal distribution: dense cluster + sparse background.
///
/// Half the points are clustered tightly, half are spread uniformly.
/// Tests algorithms that rely on uniform density assumptions.
///
/// The sparse points naturally serve as anchors, preventing cells from spanning >90°.
pub fn bimodal_density_points(n: usize, cluster_radius_rad: f32, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Ensure at least 6 sparse anchor points
    let min_sparse = 6;
    let n_clustered = (n / 2).min(n.saturating_sub(min_sparse));
    let n_sparse = n - n_clustered;

    let mut points = Vec::with_capacity(n);

    // Sparse half first (serve as anchors) - uniform on sphere
    for _ in 0..n_sparse {
        let z: f32 = rng.gen_range(-1.0..1.0);
        let theta: f32 = rng.gen_range(0.0..2.0 * PI);
        let r = (1.0 - z * z).sqrt();
        points.push(UnitVec3::new(r * theta.cos(), r * theta.sin(), z));
    }

    // Clustered half (around north pole)
    for _ in 0..n_clustered {
        let u: f32 = rng.gen();
        let cos_theta_max = cluster_radius_rad.cos();
        let cos_theta = 1.0 - u * (1.0 - cos_theta_max);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi: f32 = rng.gen_range(0.0..2.0 * PI);
        points.push(UnitVec3::new(
            sin_theta * phi.cos(),
            sin_theta * phi.sin(),
            cos_theta,
        ));
    }

    points
}
