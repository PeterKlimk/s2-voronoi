mod support;

use s2_voronoi::{compute, compute_with, PreprocessMode, VoronoiConfig};
use s2_voronoi::validation::validate;
use support::points::{clustered_cap_points, fibonacci_sphere_points};

#[test]
fn test_validation_basic() {
    let points = fibonacci_sphere_points(100, 0.0, 0);
    let diagram = compute(&points).unwrap();
    let report = validate(&diagram);

    assert!(
        report.is_strictly_valid(),
        "Expected strictly valid diagram: {}",
        report
    );
    assert_eq!(report.num_cells, 100);
    assert_eq!(report.vertices_off_sphere, 0);
}

#[test]
fn test_validation_larger() {
    let points = fibonacci_sphere_points(1000, 0.0, 0);
    let diagram = compute(&points).unwrap();
    let report = validate(&diagram);

    assert!(
        report.is_strictly_valid(),
        "Expected strictly valid diagram: {}",
        report
    );
    assert_eq!(report.connected_components, 1);
}

#[test]
fn test_validation_rejects_duplicate_cell_collapse() {
    let points = clustered_cap_points(100, 0.0175, 42);
    let diagram = compute(&points).expect("clustered cap should still compute");
    let report = validate(&diagram);

    assert!(
        !report.is_strictly_valid(),
        "Expected clustered duplicate-cell collapse to be invalid: {}",
        report
    );
    assert!(
        report.duplicate_cells_count > 0,
        "Expected duplicate-cell collapse to be surfaced: {}",
        report
    );
}

#[test]
fn test_validation_clustered_cap_tight_without_preprocess_is_strictly_valid() {
    let points = clustered_cap_points(100, 0.0175, 42);
    let diagram = compute_with(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::Disabled,
            ..VoronoiConfig::default()
        },
    )
    .expect("clustered cap should compute without preprocessing");
    let report = validate(&diagram);

    assert!(
        report.is_strictly_valid(),
        "Expected clustered cap without preprocessing to remain strictly valid: {}",
        report
    );
}
