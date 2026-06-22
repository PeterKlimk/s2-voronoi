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

fn expect_defined_failure(name: &str, result: Result<s2_voronoi::SphericalVoronoi, VoronoiError>) {
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
        ..VoronoiConfig::default()
    };
    expect_strict_success(
        "clustered_cap_small_no_preprocess",
        compute_with(&points, config),
    );
}

#[test]
fn test_clustered_cap_tight_no_preprocess() {
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    let points = clustered_cap_points(100, 0.0175, 42);
    let config = VoronoiConfig {
        preprocess_mode: PreprocessMode::Disabled,
        ..VoronoiConfig::default()
    };
    expect_strict_success(
        "clustered_cap_tight_no_preprocess",
        compute_with(&points, config),
    );
}

#[test]
fn test_cocircular_tight_no_preprocess() {
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    let points = near_cocircular_stress_points(25, 0.001, 42);
    let config = VoronoiConfig {
        preprocess_mode: PreprocessMode::Disabled,
        ..VoronoiConfig::default()
    };
    expect_strict_success(
        "cocircular_tight_no_preprocess",
        compute_with(&points, config),
    );
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

#[test]
fn test_cube_vertices_100k_seed3_defect_carrier() {
    // Discovered by the robustness campaign: a cheap (~0.5s), deterministic
    // defect carrier — the only known sphere input that exercises the
    // post-assembly repair net below the multi-million-point scale (it
    // produces InBinThirdsMismatch + InBinUnconsumedCheck records that
    // repair to strict validity). The contract is strict validity; `compute`
    // additionally errors on any post-repair residual, so this also guards
    // the always-caught path.
    let points = cube_vertex_stress_points(100_000, 0.01, 3);
    expect_strict_success("cube_vertices_100k_seed3", compute(&points));
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
// Contract sweeps at large point counts (2-4.5M): every run must validate
// strictly. Historical "bad edges" in this range were the pre-weld merge
// remap duplicating cells (docs/engineering-findings.md #7); orphan vertices
// are a documented representation note (finding #9), not an invalidity.
// Marked #[ignore] for runtime only - suitable for scheduled CI.
// Run with: cargo test --test adversarial fuzz -- --ignored --nocapture
// =============================================================================

/// Compute, validate, and assert strict validity for one fuzz case.
fn assert_fuzz_strictly_valid(name: &str, points: &[s2_voronoi::UnitVec3]) {
    let diagram = compute(points).unwrap_or_else(|e| panic!("{name}: compute failed with {e:?}"));
    let report = validate(&diagram);
    eprintln!(
        "{name}: {} cells, {}",
        diagram.num_cells(),
        report.headline()
    );
    assert!(
        report.is_strictly_valid(),
        "{name}: expected strict validity, got {}",
        report.headline()
    );
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_2m_random_seeds() {
    const N: usize = 2_000_000;
    for seed in [42u64, 123, 456, 789, 1001] {
        let points = random_sphere_points(N, seed);
        assert_fuzz_strictly_valid(&format!("fuzz_2m seed={seed}"), &points);
    }
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_3m_random_seeds() {
    const N: usize = 3_000_000;
    for seed in [42u64, 123, 456, 789, 1001] {
        let points = random_sphere_points(N, seed);
        assert_fuzz_strictly_valid(&format!("fuzz_3m seed={seed}"), &points);
    }
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_4m_random_seeds() {
    const N: usize = 4_000_000;
    for seed in [42u64, 123, 456, 789, 1001] {
        let points = random_sphere_points(N, seed);
        assert_fuzz_strictly_valid(&format!("fuzz_4m seed={seed}"), &points);
    }
}

#[test]
#[ignore = "high-volume fuzz test - run manually"]
fn test_fuzz_sweep_sizes() {
    // Sweep through sizes from 2M to 4.5M in 500k increments
    let sizes: [usize; 6] = [
        2_000_000, 2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000,
    ];
    for n in sizes {
        let points = random_sphere_points(n, 42);
        assert_fuzz_strictly_valid(&format!("fuzz_sweep n={n}"), &points);
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
    points.push(s2_voronoi::UnitVec3::new(
        p.x.next_down(),
        p.y.next_up(),
        p.z,
    ));

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

#[test]
fn test_nan_input_rejected_with_index() {
    let mut points = random_sphere_points(1_000, 17);
    points[437] = s2_voronoi::UnitVec3::new(f32::NAN, 0.5, 0.5);

    match compute(&points) {
        Err(VoronoiError::InvalidInput { point_index, .. }) => {
            assert_eq!(point_index, 437);
        }
        other => panic!("expected InvalidInput at point 437, got {other:?}"),
    }
}

#[test]
fn test_infinite_input_rejected_with_index() {
    let mut points = random_sphere_points(1_000, 17);
    points[12] = s2_voronoi::UnitVec3::new(0.0, f32::INFINITY, 0.0);

    match compute(&points) {
        Err(VoronoiError::InvalidInput { point_index, .. }) => {
            assert_eq!(point_index, 12);
        }
        other => panic!("expected InvalidInput at point 12, got {other:?}"),
    }
}

#[test]
fn test_sub_weld_cluster_without_welding_is_degenerate_input() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    // Sub-weld clusters (5 points scattered within ~1e-8) hard-fail
    // construction when welding is off (ClippedAway on the enclosed
    // micro-cell, see docs/correctness-contract.md); that failure must
    // surface as an actionable DegenerateInput naming the coincident
    // generators, not a generic ComputationFailed. Same construction as the
    // margin-sweep probe that established the failure.
    let mut points = random_sphere_points(20_000, 11);
    let mut rng = ChaCha8Rng::seed_from_u64(404);
    for i in 0..50 {
        let c = points[i * 31];
        let c64 = glam::DVec3::new(c.x as f64, c.y as f64, c.z as f64).normalize();
        let mut placed = 0;
        let mut tries = 0;
        while placed < 4 && tries < 200 {
            tries += 1;
            let r = glam::DVec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            let t = (r - c64 * r.dot(c64)).normalize();
            let q64 = (c64 + t * (1e-8 * rng.gen_range(0.1..1.0))).normalize();
            let q = s2_voronoi::UnitVec3::new(q64.x as f32, q64.y as f32, q64.z as f32);
            let dup = points.iter().any(|p| (p.x, p.y, p.z) == (q.x, q.y, q.z));
            if !dup {
                points.push(q);
                placed += 1;
            }
        }
    }

    let result = compute_with(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::Disabled,
            ..VoronoiConfig::default()
        },
    );
    match result {
        Err(VoronoiError::DegenerateInput {
            coincident_pairs,
            message,
        }) => {
            eprintln!("sub_weld_cluster: DegenerateInput ({coincident_pairs} pairs)");
            assert!(coincident_pairs >= 1);
            assert!(
                message.contains("Weld"),
                "message should point at the welding fix: {message}"
            );
        }
        Ok(_) => {
            // Construction order can let the cluster survive; if it computes,
            // that is also acceptable - the contract is no *generic* failure.
            eprintln!("sub_weld_cluster: computed without failure");
        }
        Err(other) => panic!("expected DegenerateInput, got {other:?}"),
    }
}

// =============================================================================
// Resolvable-Regime Contract (welding disabled)
//
// Above the weld radius the raw pipeline must resolve close pairs without
// preprocessing - the other half of the coincidence contract. Derived from
// the margin-mapping probes in tests/coincidence_probes.rs.
// =============================================================================

#[test]
fn test_above_weld_pairs_resolve_without_welding() {
    use glam::DVec3;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    // Pairs just above the weld radius (~1.4e-6): separations 2e-6 and 1e-5.
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for &sep in &[2e-6f64, 1e-5] {
        let mut points = random_sphere_points(20_000, 11);
        for i in 0..100 {
            let p = points[i * 7];
            let p64 = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
            let r = DVec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            let t = (r - p64 * r.dot(p64)).normalize();
            let q64 = (p64 + t * sep).normalize();
            points.push(s2_voronoi::UnitVec3::new(
                q64.x as f32,
                q64.y as f32,
                q64.z as f32,
            ));
        }
        let result = compute_with(
            &points,
            VoronoiConfig {
                preprocess_mode: PreprocessMode::Disabled,
                ..VoronoiConfig::default()
            },
        );
        expect_strict_success(&format!("above_weld_pairs_sep_{sep:.0e}"), result);
    }
}

#[test]
fn test_rotated_symmetric_pairs_resolve_without_welding() {
    use glam::DVec3;
    use s2_voronoi::{compute_with, PreprocessMode, VoronoiConfig};

    // The rotated control from the seam probes: sub-weld pairs with the
    // same relative geometry as the catastrophic seam-aligned regime, but
    // at generic (rotated) positions, must resolve without welding.
    //
    // Twins are placed *tangentially* (~3e-7, sub-weld): entry
    // canonicalization (P5 stage 0) erases radial-only ulp distinctions by
    // design — a radial ulp is the same direction on the sphere — and a
    // collapsed pair is an exact bit-duplicate, which `Disabled` has never
    // supported (its contract requires no sub-weld pairs).
    let rot = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.71, 1.13, 2.41);
    let inv3 = 1.0f64 / 3.0f64.sqrt();
    let inv2 = 1.0f64 / 2.0f64.sqrt();
    let mut centers: Vec<DVec3> = Vec::new();
    for sx in [-1.0f64, 1.0] {
        for sy in [-1.0f64, 1.0] {
            for sz in [-1.0f64, 1.0] {
                centers.push(DVec3::new(sx * inv3, sy * inv3, sz * inv3));
            }
        }
    }
    for s in [-1.0f64, 1.0] {
        centers.push(DVec3::new(s * inv2, s * inv2, 0.0));
        centers.push(DVec3::new(s, 0.0, 0.0));
        centers.push(DVec3::new(0.0, 0.0, s));
    }

    let mut points = random_sphere_points(20_000, 11);
    for &c in &centers {
        let r = (rot * c).normalize();
        let arbitrary = if r.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let t = r.cross(arbitrary).normalize();
        let q = (r + t * 3e-7).normalize();
        points.push(s2_voronoi::UnitVec3::new(
            r.x as f32, r.y as f32, r.z as f32,
        ));
        points.push(s2_voronoi::UnitVec3::new(
            q.x as f32, q.y as f32, q.z as f32,
        ));
    }

    let result = compute_with(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::Disabled,
            ..VoronoiConfig::default()
        },
    );
    expect_strict_success("rotated_symmetric_ulp_pairs", result);
}
