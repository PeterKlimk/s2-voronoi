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
            preprocess_mode: PreprocessMode::Weld,
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
            preprocess_mode: PreprocessMode::Weld,
            ..VoronoiConfig::default()
        },
    )
    .expect("compute_with_report should succeed");

    assert_eq!(output.diagram.num_cells(), 50);
    assert_eq!(
        output.report.preprocess.requested_mode,
        PreprocessMode::Weld
    );
    assert_eq!(output.report.preprocess.original_points, 50);
    assert_eq!(
        output.report.preprocess.effective_points + output.report.preprocess.num_merged,
        50
    );
    assert!(output.report.preprocess.threshold_used.is_some());
    assert!(
        output.report.returned_validation.is_strictly_valid(),
        "expected random-sphere output to validate strictly"
    );
    assert!(
        output.report.effective_validation.is_none(),
        "effective validation should only be present when preprocessing changes the generator set"
    );
    assert!(
        output.effective_diagram.is_none(),
        "effective diagram should only be present when preprocessing changes the generator set"
    );
    assert!(
        output.report.preferred_validation().is_strictly_valid(),
        "preferred validation should agree with returned validation when no merges occur"
    );
    assert_eq!(
        output.preferred_diagram().num_cells(),
        output.diagram.num_cells()
    );
}

#[test]
fn test_clustered_cap_tight_report_keeps_default_preprocessing_nonintrusive() {
    let points = clustered_cap_points(100, 0.0175, 42);

    let output = compute_with_report(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::Weld,
            ..VoronoiConfig::default()
        },
    )
    .expect("clustered cap should still compute with default preprocessing");

    assert_eq!(
        output.report.preprocess.requested_mode,
        PreprocessMode::Weld
    );
    assert!(
        !output.report.preprocess.did_merge(),
        "default density preprocessing should stay non-intrusive on this clustered-cap fixture"
    );
    assert!(
        output.report.preprocess.threshold_used.is_some(),
        "expected density-based preprocessing to record a threshold"
    );

    let report = validate(&output.diagram);
    assert!(
        report.is_strictly_valid(),
        "clustered-cap output should remain strictly valid under the less aggressive default merge policy: {}",
        report.headline()
    );
    assert!(
        output.report.returned_validation.is_strictly_valid(),
        "returned validation should stay strict-valid when no remap collapse occurs"
    );
    assert!(output.report.effective_validation.is_none());
    assert!(output.effective_diagram.is_none());
    assert!(output.report.preferred_validation().is_strictly_valid());
    assert_eq!(
        output.preferred_diagram().num_cells(),
        output.diagram.num_cells()
    );
}

#[test]
fn test_clustered_cap_extreme_weld_keeps_returned_diagram_strictly_valid() {
    let points = clustered_cap_points(50, 0.00175, 42);

    // Coarse explicit threshold that forces welds on this tight fixture
    // (the default weld radius is far below its point spacing).
    let output = compute_with_report(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::MergeWithin(3.5e-4),
            ..VoronoiConfig::default()
        },
    )
    .expect("clustered_cap_extreme should compute under coarse welding");

    assert!(
        output.report.preprocess.did_merge(),
        "coarse threshold should exercise the weld-altered contract on this fixture"
    );
    assert!(
        output.effective_diagram.is_some(),
        "welding should expose the effective solved diagram"
    );
    assert!(
        output
            .report
            .effective_validation
            .as_ref()
            .expect("welding should surface effective validation")
            .is_strictly_valid(),
        "effective solved diagram should validate strictly"
    );
    // The strengthened contract: the returned diagram is also strictly valid,
    // with welded twins sharing their canonical cell instead of duplicating it.
    assert!(
        output.report.returned_validation.is_strictly_valid(),
        "returned diagram with welds should validate strictly: {}",
        output.report.returned_validation.headline()
    );
    assert_eq!(
        output.report.returned_validation.welded_twin_cells,
        output.report.preprocess.num_merged
    );

    let weld_map = output
        .diagram
        .weld_map()
        .expect("welds occurred, weld map should be present");
    assert_eq!(weld_map.len(), points.len());
    assert_eq!(
        output.diagram.welded_twin_count(),
        output.report.preprocess.num_merged
    );
    for i in 0..points.len() {
        let canonical = output.diagram.canonical_cell_index(i);
        assert_eq!(
            output.diagram.canonical_cell_index(canonical),
            canonical,
            "canonical cells must map to themselves"
        );
        assert!(
            canonical <= i,
            "canonical index is the smallest input index in the weld class"
        );
        assert_eq!(
            output.diagram.cell(i).vertex_indices,
            output.diagram.cell(canonical).vertex_indices,
            "welded twins must alias their canonical cell's boundary"
        );
    }
}

#[test]
fn test_compute_with_report_exposes_effective_diagram_when_merges_occur() {
    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(0.999_999_94, 0.0003, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
    ];

    let output = compute_with_report(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::MergeWithin(0.001),
            ..VoronoiConfig::default()
        },
    )
    .expect("explicit merge preprocessing should still compute");

    assert!(output.report.preprocess.did_merge());
    assert_eq!(output.diagram.num_cells(), points.len());

    let effective_diagram = output
        .effective_diagram
        .as_ref()
        .expect("merged preprocessing should expose the effective solved diagram");
    assert_eq!(
        effective_diagram.num_cells(),
        output.report.preprocess.effective_points
    );
    assert_eq!(effective_diagram.num_cells(), points.len() - 1);
    assert!(
        output
            .report
            .effective_validation
            .as_ref()
            .expect("merged preprocessing should surface effective validation")
            .is_strictly_valid(),
        "effective merged diagram should validate strictly"
    );
    assert_eq!(
        output.preferred_diagram().num_cells(),
        effective_diagram.num_cells()
    );
    assert!(
        output.report.returned_validation.is_strictly_valid(),
        "returned diagram with welds should validate strictly: {}",
        output.report.returned_validation.headline()
    );
    // Points 0 and 1 are the welded pair; 0 is the canonical index.
    assert_eq!(output.diagram.canonical_cell_index(1), 0);
    assert_eq!(
        output.diagram.cell(1).vertex_indices,
        output.diagram.cell(0).vertex_indices
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
