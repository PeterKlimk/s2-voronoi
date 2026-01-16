//! Adversarial geometry tests.
//!
//! Tests using degenerate or stress-inducing point distributions to verify
//! robustness. These tests document expected behavior (success, graceful
//! degradation, or defined error) rather than asserting perfect results.

mod support;

use s2_voronoi::{compute, validation::validate};
use support::points::*;

// =============================================================================
// Great Circle Tests (All Coplanar)
//
// NOTE: Pure great circle configurations are beyond the algorithm's design.
// Gnomonic projection fails for cells spanning >90° from the generator, and
// coplanar points create cells that wrap around the sphere. Tests with
// insufficient jitter are marked #[ignore].
// =============================================================================

#[test]
fn test_great_circle_with_jitter() {
    // Great circle with significant jitter to break coplanarity
    // The jitter creates enough 3D spread that cells stay bounded
    let points = great_circle_points(50, 0.2, 42);
    let result = compute(&points);

    assert!(result.is_ok(), "great circle with jitter should work");
    let diagram = result.unwrap();
    assert_eq!(diagram.num_cells(), 50);

    let report = validate(&diagram);
    eprintln!("great_circle_with_jitter: {}", report.summary());
}

#[test]
#[ignore = "gnomonic projection cannot handle cells spanning >90° from generator"]
fn test_great_circle_small_jitter() {
    // Small jitter - may or may not work depending on luck
    let points = great_circle_points(20, 0.01, 42);
    let result = compute(&points);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("great_circle_small_jitter: {}", report.summary());
        }
        Err(e) => {
            eprintln!("great_circle_small_jitter failed: {:?}", e);
        }
    }
}

#[test]
#[ignore = "gnomonic projection cannot handle cells spanning >90° from generator"]
fn test_great_circle_pure_degenerate() {
    // Pure great circle with no jitter - maximally degenerate
    // This CANNOT work with current algorithm - cells span 180°
    let points = great_circle_points(50, 0.0, 42);
    let result = compute(&points);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("pure great circle: {}", report.summary());
        }
        Err(e) => {
            eprintln!("pure great circle failed (expected): {:?}", e);
        }
    }
}

// =============================================================================
// Clustered Cap Tests (Extreme Density)
//
// These use anchor points on the 6 axis directions to prevent cells from
// spanning >90°.
// =============================================================================

#[test]
fn test_clustered_cap_small() {
    // 100 points total: 6 anchors + 94 in 5-degree cap
    let points = clustered_cap_points(100, 0.087, 42); // ~5 degrees
    let result = compute(&points);

    assert!(result.is_ok(), "clustered cap small should not panic");
    let diagram = result.unwrap();
    assert_eq!(diagram.num_cells(), 100);
}

#[test]
fn test_clustered_cap_tight() {
    // 100 points total: 6 anchors + 94 in 1-degree cap - very close together
    let points = clustered_cap_points(100, 0.0175, 42); // ~1 degree
    let result = compute(&points);

    assert!(result.is_ok(), "clustered cap tight should not panic");
    let diagram = result.unwrap();
    let report = validate(&diagram);
    eprintln!("clustered_cap_tight: {}", report.summary());
    eprintln!(
        "  cells={}, vertices={}, total_cell_vertices={}, unique_cells={}, duplicates={}",
        report.num_cells,
        report.num_vertices,
        report.total_cell_vertices,
        report.unique_cells,
        report.duplicate_cells_count
    );
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
    eprintln!("clustered_cap_extreme: {}", report.summary());
}

// =============================================================================
// No-Preprocessing Variants
//
// These tests run without preprocessing to isolate and compare behavior.
// Preprocessing merges near-coincident points, which can hide or amplify issues.
// =============================================================================

#[test]
#[ignore = "compare behavior with preprocessing disabled"]
fn test_clustered_cap_small_no_preprocess() {
    use s2_voronoi::{compute_with, VoronoiConfig};

    let points = clustered_cap_points(100, 0.087, 42);
    let config = VoronoiConfig {
        preprocess: false,
        ..Default::default()
    };
    let result = compute_with(&points, config);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("clustered_cap_small (no preprocess): {}", report.summary());
            assert_eq!(diagram.num_cells(), 100);
        }
        Err(e) => {
            eprintln!("clustered_cap_small (no preprocess) failed: {:?}", e);
        }
    }
}

#[test]
#[ignore = "compare behavior with preprocessing disabled"]
fn test_clustered_cap_tight_no_preprocess() {
    use s2_voronoi::{compute_with, VoronoiConfig};

    let points = clustered_cap_points(100, 0.0175, 42);
    let config = VoronoiConfig {
        preprocess: false,
        ..Default::default()
    };
    let result = compute_with(&points, config);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("clustered_cap_tight (no preprocess): {}", report.summary());
        }
        Err(e) => {
            eprintln!("clustered_cap_tight (no preprocess) failed: {:?}", e);
        }
    }
}

#[test]
#[ignore = "compare behavior with preprocessing disabled"]
fn test_cocircular_tight_no_preprocess() {
    use s2_voronoi::{compute_with, VoronoiConfig};

    let points = near_cocircular_stress_points(25, 0.001, 42);
    let config = VoronoiConfig {
        preprocess: false,
        ..Default::default()
    };
    let result = compute_with(&points, config);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("cocircular_tight (no preprocess): {}", report.summary());
        }
        Err(e) => {
            eprintln!("cocircular_tight (no preprocess) failed: {:?}", e);
        }
    }
}

// =============================================================================
// Cube Vertex Stress Tests (Face Stitching)
// =============================================================================

#[test]
fn test_cube_vertices_basic() {
    // 80 points near cube corners (10 per corner)
    let points = cube_vertex_stress_points(80, 0.1, 42);
    let result = compute(&points);

    assert!(result.is_ok(), "cube vertices basic should not panic");
    let diagram = result.unwrap();
    assert_eq!(diagram.num_cells(), 80);
}

#[test]
fn test_cube_vertices_tight() {
    // 160 points very close to cube corners
    let points = cube_vertex_stress_points(160, 0.02, 42);
    let result = compute(&points);

    assert!(result.is_ok(), "cube vertices tight should not panic");
    let diagram = result.unwrap();
    let report = validate(&diagram);
    eprintln!("cube_vertices_tight: {}", report.summary());
}

// =============================================================================
// Near-Cocircular Tests (Vertex Instability → Bad Edges)
// =============================================================================

#[test]
fn test_cocircular_basic() {
    // 25 groups of 4 = 100 points with moderate perturbation
    let points = near_cocircular_stress_points(25, 0.01, 42);
    let result = compute(&points);

    assert!(result.is_ok(), "cocircular basic should not panic");
    let diagram = result.unwrap();
    assert_eq!(diagram.num_cells(), 100);
}

#[test]
fn test_cocircular_tight() {
    // 25 groups with very small perturbation - likely to trigger bad edges
    let points = near_cocircular_stress_points(25, 0.001, 42);
    let result = compute(&points);

    assert!(result.is_ok(), "cocircular tight should not panic");
    let diagram = result.unwrap();
    let report = validate(&diagram);
    eprintln!("cocircular_tight: {}", report.summary());
}

#[test]
fn test_cocircular_extreme() {
    // Near-perfect cocircular groups - maximally unstable vertices
    let points = near_cocircular_stress_points(25, 0.0001, 42);
    let result = compute(&points);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("cocircular_extreme: {}", report.summary());
            // Likely has non-degree-3 vertices or other issues
        }
        Err(e) => {
            eprintln!("cocircular_extreme failed (acceptable): {:?}", e);
        }
    }
}

// =============================================================================
// Hemisphere Tests (Asymmetric)
// =============================================================================

#[test]
#[ignore = "gnomonic projection cannot handle cells spanning >90° - awaiting fallback"]
fn test_hemisphere_basic() {
    // 100 points on upper hemisphere
    let points = hemisphere_points(100, 42);
    let result = compute(&points);

    // May fail due to cells spanning >90° from generator
    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("hemisphere_basic: {}", report.summary());
            assert_eq!(diagram.num_cells(), 100);
        }
        Err(e) => {
            eprintln!("hemisphere_basic failed (gnomonic limit): {:?}", e);
        }
    }
}

#[test]
#[ignore = "gnomonic projection cannot handle cells spanning >90° - awaiting fallback"]
fn test_hemisphere_dense() {
    // 500 points on upper hemisphere - likely to create cells >90°
    let points = hemisphere_points(500, 42);
    let result = compute(&points);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("hemisphere_dense: {}", report.summary());
        }
        Err(e) => {
            eprintln!("hemisphere_dense failed (expected): {:?}", e);
        }
    }
}

// =============================================================================
// Bimodal Density Tests
// =============================================================================

#[test]
fn test_bimodal_basic() {
    // 100 points: 50 clustered + 50 sparse
    let points = bimodal_density_points(100, 0.1, 42); // 10-degree cluster
    let result = compute(&points);

    assert!(result.is_ok(), "bimodal basic should not panic");
    let diagram = result.unwrap();
    assert_eq!(diagram.num_cells(), 100);
}

#[test]
#[ignore = "gnomonic projection cannot handle cells spanning >90° - awaiting fallback"]
fn test_bimodal_extreme() {
    // 200 points: 100 in 2-degree cluster + 100 sparse
    // Sparse points may create cells that span >90°
    let points = bimodal_density_points(200, 0.035, 42);
    let result = compute(&points);

    match result {
        Ok(diagram) => {
            let report = validate(&diagram);
            eprintln!("bimodal_extreme: {}", report.summary());
        }
        Err(e) => {
            eprintln!("bimodal_extreme failed (expected): {:?}", e);
        }
    }
}

// =============================================================================
// Combined Stress Tests
// =============================================================================

#[test]
#[ignore = "some distributions create cells spanning >90° - awaiting fallback"]
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
                    report.summary()
                );
            }
            Err(e) => {
                eprintln!("{}: failed with {:?}", name, e);
            }
        }
    }
}
