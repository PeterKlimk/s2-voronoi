//! Public API integration tests for s2-voronoi.

mod support;

use s2_voronoi::{
    compute, compute_with, compute_with_report, validation::validate, PreprocessMode, UnitVec3,
    VoronoiConfig, VoronoiError,
};
use support::points::{
    clustered_cap_points, fibonacci_sphere_points, hemisphere_points, random_sphere_points,
};

#[test]
fn test_compute_basic() {
    let points = random_sphere_points(100, 12345);
    let diagram = compute(&points).expect("compute should succeed");

    assert_eq!(diagram.num_cells(), 100);
    assert!(diagram.num_vertices() > 0);
}

#[test]
fn test_compute_small_set() {
    // Small point set (algorithm needs enough neighbors to work)
    let points = random_sphere_points(20, 12345);
    let diagram = compute(&points).expect("20 points should work");
    assert_eq!(diagram.num_cells(), 20);
}

#[test]
fn test_compute_insufficient_points() {
    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
    ];
    let result = compute(&points);
    assert!(matches!(result, Err(VoronoiError::InsufficientPoints(3))));
}

#[test]
fn test_compute_octahedron() {
    // 6 axis-aligned points form an octahedron
    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
    ];
    let diagram = compute(&points).expect("octahedron should work");

    assert_eq!(diagram.num_cells(), 6);
    // Each cell should have 4 vertices (square face)
    for cell in diagram.iter_cells() {
        assert_eq!(cell.len(), 4, "octahedron cells should have 4 vertices");
    }
}

#[test]
fn test_compute_various_sizes() {
    for n in [10, 50, 100, 500] {
        let points = fibonacci_sphere_points(n, 0.1, 42);
        let diagram = compute(&points).unwrap_or_else(|_| panic!("n={} should work", n));
        assert_eq!(diagram.num_cells(), n);
    }
}

#[test]
fn test_compute_with_explicit_preprocess_modes() {
    let points = random_sphere_points(50, 13579);

    let density = compute_with(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::MergeDensity,
            ..VoronoiConfig::default()
        },
    )
    .expect("density-based preprocessing should succeed");
    assert_eq!(density.num_cells(), 50);

    let disabled = compute_with(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::Disabled,
            ..VoronoiConfig::default()
        },
    )
    .expect("disabled preprocessing should succeed");
    assert_eq!(disabled.num_cells(), 50);
}

#[test]
fn test_compute_with_report_surfaces_preprocess_outcome() {
    let points = random_sphere_points(50, 24680);

    let output = compute_with_report(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::MergeDensity,
            ..VoronoiConfig::default()
        },
    )
    .expect("compute_with_report should succeed");

    assert_eq!(output.diagram.num_cells(), 50);
    assert_eq!(
        output.report.preprocess.requested_mode,
        PreprocessMode::MergeDensity
    );
    assert_eq!(output.report.preprocess.original_points, 50);
    assert_eq!(
        output.report.preprocess.effective_points + output.report.preprocess.num_merged,
        50
    );
    assert!(output.report.preprocess.threshold_used.is_some());
}

#[test]
fn test_clustered_cap_tight_report_shows_default_preprocessing_merges_points() {
    let points = clustered_cap_points(100, 0.0175, 42);

    let output = compute_with_report(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::MergeDensity,
            ..VoronoiConfig::default()
        },
    )
    .expect("clustered cap should still compute with default preprocessing");

    assert_eq!(
        output.report.preprocess.requested_mode,
        PreprocessMode::MergeDensity
    );
    assert!(
        output.report.preprocess.did_merge(),
        "expected default preprocessing to merge close clustered-cap generators"
    );
    assert!(
        output.report.preprocess.num_merged > 0,
        "expected a nonzero clustered-cap merge count"
    );
    assert!(
        output.report.preprocess.threshold_used.is_some(),
        "expected density-based preprocessing to record a threshold"
    );

    let report = validate(&output.diagram);
    assert!(
        !report.is_strictly_valid(),
        "current clustered-cap collapse regression should still be surfaced as invalid: {}",
        report.headline()
    );
}

#[test]
fn test_cell_iteration() {
    let points = random_sphere_points(50, 99999);
    let diagram = compute(&points).unwrap();

    let mut count = 0;
    for cell in diagram.iter_cells() {
        assert!(cell.generator_index < 50);
        count += 1;
    }
    assert_eq!(count, 50);
}

#[test]
fn test_vertex_indices_valid() {
    let points = random_sphere_points(100, 54321);
    let diagram = compute(&points).unwrap();

    let num_vertices = diagram.num_vertices();
    for cell in diagram.iter_cells() {
        for &idx in cell.vertex_indices {
            assert!(
                (idx as usize) < num_vertices,
                "vertex index {} out of bounds ({})",
                idx,
                num_vertices
            );
        }
    }
}

#[test]
fn test_generators_preserved() {
    let points = random_sphere_points(20, 77777);
    let diagram = compute(&points).unwrap();

    // Generators should match input points
    assert_eq!(diagram.generators().len(), points.len());
    for (i, (gen, orig)) in diagram.generators().iter().zip(points.iter()).enumerate() {
        let diff =
            ((gen.x - orig.x).powi(2) + (gen.y - orig.y).powi(2) + (gen.z - orig.z).powi(2)).sqrt();
        assert!(diff < 1e-6, "generator {} differs from input: {}", i, diff);
    }
}

#[test]
fn test_input_types() {
    // Test that different input types work via UnitVec3Like trait
    // Use enough points for the algorithm to work
    let base_points = random_sphere_points(50, 88888);

    // Convert to array format
    let arr_points: Vec<[f32; 3]> = base_points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let diagram = compute(&arr_points).expect("array input should work");
    assert_eq!(diagram.num_cells(), 50);

    // Convert to tuple format
    let tuple_points: Vec<(f32, f32, f32)> = base_points.iter().map(|p| (p.x, p.y, p.z)).collect();
    let diagram = compute(&tuple_points).expect("tuple input should work");
    assert_eq!(diagram.num_cells(), 50);
}

#[test]
fn test_compute_reports_hemisphere_limit_as_error() {
    let points = hemisphere_points(100, 42);
    let result = compute(&points);
    assert!(
        matches!(
            result,
            Err(VoronoiError::UnsupportedGeometry { .. }) | Err(VoronoiError::ComputationFailed(_))
        ),
        "hemisphere-limited inputs should fail cleanly, got {:?}",
        result
    );
}

#[test]
#[cfg(feature = "qhull")]
fn test_qhull_available() {
    use glam::Vec3;
    use s2_voronoi::compute_voronoi_qhull;

    let points: Vec<Vec3> = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    let voronoi = compute_voronoi_qhull(&points);
    assert_eq!(voronoi.num_cells(), 6);
}
