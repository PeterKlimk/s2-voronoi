//! Adversarial geometry tests.
//!
//! Tests using degenerate or stress-inducing point distributions to verify
//! robustness. Contract tests assert stable expected outcomes. Known-stress
//! diagnostics keep reporting validation quality without pinning current
//! invalidity as a desired outcome.

mod support;

use s2_voronoi::{compute, validation::validate, VoronoiError};
use support::points::*;

fn expect_strict_success(name: &str, result: Result<s2_voronoi::SphericalVoronoi, VoronoiError>) {
    let diagram = result.unwrap_or_else(|err| panic!("{name} should succeed, got {err:?}"));
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "{name} should be strictly valid, got {}",
        report.headline()
    );
}

fn expect_defined_failure(
    name: &str,
    result: Result<s2_voronoi::SphericalVoronoi, VoronoiError>,
) {
    assert!(
        matches!(
            result,
            Err(VoronoiError::UnsupportedGeometry { .. }) | Err(VoronoiError::ComputationFailed(_))
        ),
        "{name} should fail cleanly with an explicit error, got {:?}",
        result
    );
}

// =============================================================================
// Great Circle Tests (All Coplanar)
//
// NOTE: Pure great circle configurations are beyond the algorithm's design.
// Gnomonic projection fails for cells spanning >90° from the generator, and
// coplanar points create cells that wrap around the sphere.
// =============================================================================

#[test]
fn test_great_circle_with_jitter() {
    // Great circle with significant jitter to break coplanarity.
    let points = great_circle_points(50, 0.2, 42);
    expect_strict_success("great_circle_with_jitter", compute(&points));
}

#[test]
fn test_great_circle_small_jitter() {
    // Small jitter is still enough to break the coplanar failure mode for this
    // fixed test distribution.
    let points = great_circle_points(20, 0.01, 42);
    expect_strict_success("great_circle_small_jitter", compute(&points));
}

#[test]
fn test_great_circle_pure_degenerate() {
    // Pure great circle with no jitter - maximally degenerate
    // This CANNOT work with current algorithm - cells span 180°
    let points = great_circle_points(50, 0.0, 42);
    expect_defined_failure("great_circle_pure_degenerate", compute(&points));
}

// =============================================================================
// Clustered Cap Tests (Extreme Density)
//
// These use anchor points on the 6 axis directions to prevent cells from
// spanning >90°.
// =============================================================================

#[test]
fn test_clustered_cap_small() {
    // 100 points total: 6 anchors + 94 in 5-degree cap.
    let points = clustered_cap_points(100, 0.087, 42); // ~5 degrees
    expect_strict_success("clustered_cap_small", compute(&points));
}

#[test]
fn test_clustered_cap_tight() {
    // 100 points total: 6 anchors + 94 in 1-degree cap - very close together.
    let points = clustered_cap_points(100, 0.0175, 42); // ~1 degree
    let result = compute(&points);

    expect_strict_success("clustered_cap_tight", result);
}

#[test]
fn test_clustered_cap_extreme() {
    // 50 points total: 6 anchors + 44 in 0.1-degree cap - numerically challenging
    let points = clustered_cap_points(50, 0.00175, 42); // ~0.1 degrees
    let result = compute(&points);

    // Should work now with anchor points
    assert!(result.is_ok(), "clustered cap extreme should not panic");
    let diagram = result.unwrap();
    let report = validate(&diagram);
    eprintln!("clustered_cap_extreme: {}", report.headline());
}

// =============================================================================
// No-Preprocessing Variants
//
// These tests run without preprocessing to isolate and compare behavior.
// Preprocessing merges near-coincident points, which can hide or amplify issues.
// =============================================================================

#[test]
fn test_clustered_cap_small_no_preprocess() {
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    let points = clustered_cap_points(100, 0.087, 42);
    let config = VoronoiConfig {
        preprocess_mode: PreprocessMode::Disabled,
        ..Default::default()
    };
    expect_strict_success("clustered_cap_small_no_preprocess", compute_with(&points, config));
}

#[test]
fn test_clustered_cap_tight_no_preprocess() {
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    let points = clustered_cap_points(100, 0.0175, 42);
    let config = VoronoiConfig {
        preprocess_mode: PreprocessMode::Disabled,
        ..Default::default()
    };
    expect_strict_success("clustered_cap_tight_no_preprocess", compute_with(&points, config));
}

#[test]
fn test_cocircular_tight_no_preprocess() {
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    let points = near_cocircular_stress_points(25, 0.001, 42);
    let config = VoronoiConfig {
        preprocess_mode: PreprocessMode::Disabled,
        ..Default::default()
    };
    expect_strict_success("cocircular_tight_no_preprocess", compute_with(&points, config));
}

// =============================================================================
// Cube Vertex Stress Tests (Face Stitching)
// =============================================================================

#[test]
fn test_cube_vertices_basic() {
    // 80 points near cube corners (10 per corner)
    let points = cube_vertex_stress_points(80, 0.1, 42);
    expect_strict_success("cube_vertices_basic", compute(&points));
}

#[test]
fn test_cube_vertices_tight() {
    // 160 points very close to cube corners
    let points = cube_vertex_stress_points(160, 0.02, 42);
    expect_strict_success("cube_vertices_tight", compute(&points));
}

// =============================================================================
// Near-Cocircular Tests (Vertex Instability → Bad Edges)
// =============================================================================

#[test]
fn test_cocircular_basic() {
    // 25 groups of 4 = 100 points with moderate perturbation
    let points = near_cocircular_stress_points(25, 0.01, 42);
    expect_strict_success("cocircular_basic", compute(&points));
}

#[test]
fn test_cocircular_tight() {
    // 25 groups with very small perturbation - near-degenerate but currently stable.
    let points = near_cocircular_stress_points(25, 0.001, 42);
    expect_strict_success("cocircular_tight", compute(&points));
}

#[test]
fn test_cocircular_extreme() {
    // Near-perfect cocircular groups - still a supported success on this fixed seed.
    let points = near_cocircular_stress_points(25, 0.0001, 42);
    expect_strict_success("cocircular_extreme", compute(&points));
}

// =============================================================================
// Hemisphere Tests (Asymmetric)
// =============================================================================

#[test]
fn test_hemisphere_basic() {
    // 100 points on upper hemisphere
    let points = hemisphere_points(100, 42);
    let result = compute(&points);

    assert!(
        matches!(
            result,
            Err(VoronoiError::UnsupportedGeometry { .. }) | Err(VoronoiError::ComputationFailed(_))
        ),
        "hemisphere_basic should fail cleanly with an explicit error, got {:?}",
        result
    );
}

#[test]
fn test_hemisphere_dense() {
    // 500 points on upper hemisphere - likely to create cells >90°
    let points = hemisphere_points(500, 42);
    let result = compute(&points);

    assert!(
        matches!(
            result,
            Err(VoronoiError::UnsupportedGeometry { .. }) | Err(VoronoiError::ComputationFailed(_))
        ),
        "hemisphere_dense should fail cleanly with an explicit error, got {:?}",
        result
    );
}

// =============================================================================
// Bimodal Density Tests
// =============================================================================

#[test]
fn test_bimodal_basic() {
    // 100 points: 50 clustered + 50 sparse
    let points = bimodal_density_points(100, 0.1, 42); // 10-degree cluster
    expect_strict_success("bimodal_basic", compute(&points));
}

#[test]
fn test_bimodal_extreme() {
    // 200 points: 100 in 2-degree cluster + 100 sparse.
    // The sparse half anchors the geometry enough that this fixed seed is now
    // a supported strict-success case.
    let points = bimodal_density_points(200, 0.035, 42);
    expect_strict_success("bimodal_extreme", compute(&points));
}

// =============================================================================
// Combined Stress Tests
// =============================================================================

#[test]
#[ignore = "manual stress scan across mixed valid, invalid, and unsupported distributions"]
fn test_multi_distribution_robustness() {
    // Run through all adversarial distributions and verify no panics
    let test_cases: Vec<(&str, Vec<_>)> = vec![
        ("great_circle", great_circle_points(50, 0.2, 1)), // Needs enough jitter
        ("clustered_cap", clustered_cap_points(50, 0.1, 2)),
        ("cube_vertices", cube_vertex_stress_points(48, 0.05, 3)),
        ("cocircular", near_cocircular_stress_points(12, 0.005, 4)),
        ("hemisphere", hemisphere_points(50, 5)),
        ("bimodal", bimodal_density_points(50, 0.1, 6)),
    ];

    for (name, points) in test_cases {
        let result = compute(&points);
        match result {
            Ok(diagram) => {
                let report = validate(&diagram);
                eprintln!(
                    "{}: {} cells, {}",
                    name,
                    diagram.num_cells(),
                    report.headline()
                );
            }
            Err(e) => {
                eprintln!("{}: failed with {:?}", name, e);
            }
        }
    }
}

// =============================================================================
// High-Volume Fuzz Tests
//
// Diagnostic validation sweeps at large point counts (2-4M). Historical
// "bad edges" in this range were the pre-weld merge remap duplicating cells
// (docs/engineering-findings.md #7); with weld semantics these runs validate
// strictly, modulo intentional orphan vertices on some seeds (finding #9).
// Marked #[ignore] for runtime; promote to scheduled CI asserting
// STRICT_VALID once the orphan policy (todo P0.2) lands.
// Run with: cargo test --test adversarial fuzz -- --ignored --nocapture
// =============================================================================

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_2m_random_seeds() {
    // 2M points with 5 different seeds
    const N: usize = 2_000_000;
    let seeds: [u64; 5] = [42, 123, 456, 789, 1001];

    for seed in seeds {
        let points = random_sphere_points(N, seed);
        let result = compute(&points);

        match result {
            Ok(diagram) => {
                let report = validate(&diagram);
                let status = if report.is_strictly_valid() {
                    "STRICT_VALID"
                } else {
                    "INVALID"
                };
                eprintln!(
                    "fuzz_2m seed={}: {} cells, {} - {}",
                    seed,
                    diagram.num_cells(),
                    report.headline(),
                    status
                );
            }
            Err(e) => {
                eprintln!("fuzz_2m seed={}: FAILED {:?}", seed, e);
            }
        }
    }
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_3m_random_seeds() {
    // 3M points with 5 different seeds
    const N: usize = 3_000_000;
    let seeds: [u64; 5] = [42, 123, 456, 789, 1001];

    for seed in seeds {
        let points = random_sphere_points(N, seed);
        let result = compute(&points);

        match result {
            Ok(diagram) => {
                let report = validate(&diagram);
                let status = if report.is_strictly_valid() {
                    "STRICT_VALID"
                } else {
                    "INVALID"
                };
                eprintln!(
                    "fuzz_3m seed={}: {} cells, {} - {}",
                    seed,
                    diagram.num_cells(),
                    report.headline(),
                    status
                );
            }
            Err(e) => {
                eprintln!("fuzz_3m seed={}: FAILED {:?}", seed, e);
            }
        }
    }
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_4m_random_seeds() {
    // 4M points with 5 different seeds - in the range where bad edges occur
    const N: usize = 4_000_000;
    let seeds: [u64; 5] = [42, 123, 456, 789, 1001];

    for seed in seeds {
        let points = random_sphere_points(N, seed);
        let result = compute(&points);

        match result {
            Ok(diagram) => {
                let report = validate(&diagram);
                let status = if report.is_strictly_valid() {
                    "STRICT_VALID"
                } else {
                    "INVALID"
                };
                eprintln!(
                    "fuzz_4m seed={}: {} cells, {} - {}",
                    seed,
                    diagram.num_cells(),
                    report.headline(),
                    status
                );
            }
            Err(e) => {
                eprintln!("fuzz_4m seed={}: FAILED {:?}", seed, e);
            }
        }
    }
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_sweep_sizes() {
    // Sweep through sizes from 2M to 4.5M in 500k increments
    let sizes: [usize; 6] = [
        2_000_000, 2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000,
    ];
    let seed: u64 = 42;

    for n in sizes {
        let points = random_sphere_points(n, seed);
        let result = compute(&points);

        match result {
            Ok(diagram) => {
                let report = validate(&diagram);
                let status = if report.is_strictly_valid() {
                    "STRICT_VALID"
                } else {
                    "INVALID"
                };
                eprintln!(
                    "fuzz_sweep n={}: {} cells, {} - {}",
                    n,
                    diagram.num_cells(),
                    report.headline(),
                    status
                );
            }
            Err(e) => {
                eprintln!("fuzz_sweep n={}: FAILED {:?}", n, e);
            }
        }
    }
}

// =============================================================================
// Coincident-Input Weld Contract
//
// Sub-weld-radius configurations that break the raw pipeline (see
// docs/correctness-contract.md margin data) must produce strictly valid
// diagrams under the default config, with welded twins sharing one cell.
// =============================================================================

#[test]
fn test_weld_exact_duplicate_pair_strictly_valid_by_default() {
    let mut points = random_sphere_points(2_000, 7);
    points.push(points[123]);

    let diagram = compute(&points).expect("exact duplicate should weld, not fail");
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "welded exact duplicate should validate strictly: {}",
        report.headline()
    );
    assert_eq!(report.welded_twin_cells, 1);
    assert_eq!(diagram.canonical_cell_index(2_000), 123);
    assert_eq!(
        diagram.cell(2_000).vertex_indices,
        diagram.cell(123).vertex_indices
    );
}

#[test]
fn test_weld_ulp_cluster_strictly_valid_by_default() {
    // A 5-point cluster within ~1 ulp hard-fails construction (ClippedAway on
    // the enclosed micro-cell) without welding; the default weld absorbs it.
    let mut points = random_sphere_points(2_000, 11);
    let p = points[500];
    points.push(s2_voronoi::UnitVec3::new(p.x.next_up(), p.y, p.z));
    points.push(s2_voronoi::UnitVec3::new(p.x, p.y.next_up(), p.z));
    points.push(s2_voronoi::UnitVec3::new(p.x, p.y, p.z.next_up()));
    points.push(s2_voronoi::UnitVec3::new(p.x.next_down(), p.y.next_up(), p.z));

    let diagram = compute(&points).expect("ulp cluster should weld, not fail");
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "welded ulp cluster should validate strictly: {}",
        report.headline()
    );
    assert_eq!(report.welded_twin_cells, 4);
    for twin in 2_000..2_004 {
        assert_eq!(diagram.canonical_cell_index(twin), 500);
    }
}

#[test]
fn test_weld_seam_ulp_pairs_strictly_valid_by_default() {
    // Ulp pairs at exactly symmetric positions (cube corners, axis poles, 45
    // degree points) are the worst known aligned regime without welding.
    let mut points = random_sphere_points(2_000, 13);
    let inv3 = 1.0f32 / 3.0f32.sqrt();
    let inv2 = 1.0f32 / 2.0f32.sqrt();
    let mut seam_points: Vec<s2_voronoi::UnitVec3> = Vec::new();
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                seam_points.push(s2_voronoi::UnitVec3::new(sx * inv3, sy * inv3, sz * inv3));
            }
        }
    }
    for s in [-1.0f32, 1.0] {
        seam_points.push(s2_voronoi::UnitVec3::new(s * inv2, s * inv2, 0.0));
        seam_points.push(s2_voronoi::UnitVec3::new(s, 0.0, 0.0));
        seam_points.push(s2_voronoi::UnitVec3::new(0.0, 0.0, s));
    }
    let num_pairs = seam_points.len();
    for p in seam_points {
        let twin = s2_voronoi::UnitVec3::new(
            if p.x.abs() > 0.5 { p.x.next_up() } else { p.x },
            if p.y.abs() > 0.5 { p.y.next_up() } else { p.y },
            if p.z.abs() > 0.5 { p.z.next_up() } else { p.z },
        );
        points.push(p);
        points.push(twin);
    }

    let diagram = compute(&points).expect("seam ulp pairs should weld, not fail");
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "welded seam pairs should validate strictly: {}",
        report.headline()
    );
    assert_eq!(report.welded_twin_cells, num_pairs);
}
