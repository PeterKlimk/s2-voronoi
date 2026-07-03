mod support;

use s2_voronoi::validation::validate;
use s2_voronoi::{compute, compute_with, PreprocessMode, VoronoiConfig};
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
fn test_validation_clustered_cap_tight_with_default_preprocessing_is_strictly_valid() {
    let points = clustered_cap_points(100, 0.0175, 42);
    let diagram = compute(&points).expect("clustered cap should compute");
    let report = validate(&diagram);

    assert!(
        report.is_strictly_valid(),
        "Expected clustered cap to remain strictly valid under the default preprocessing policy: {}",
        report
    );
}

#[test]
fn test_validation_clustered_cap_tight_without_preprocess_is_strictly_valid() {
    let points = clustered_cap_points(100, 0.0175, 42);
    let diagram = compute_with(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Disabled),
    )
    .expect("clustered cap should compute without preprocessing");
    let report = validate(&diagram);

    assert!(
        report.is_strictly_valid(),
        "Expected clustered cap without preprocessing to remain strictly valid: {}",
        report
    );
}
