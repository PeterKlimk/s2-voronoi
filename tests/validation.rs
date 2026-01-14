mod support;

use s2_voronoi::compute;
use s2_voronoi::validation::validate;
use support::points::fibonacci_sphere_points;

#[test]
fn test_validation_basic() {
    let points = fibonacci_sphere_points(100, 0.0, 0);
    let diagram = compute(&points).unwrap();
    let report = validate(&diagram);

    assert!(report.is_valid(), "Expected valid diagram: {}", report);
    assert_eq!(report.num_cells, 100);
    assert_eq!(report.vertices_off_sphere, 0);
}

#[test]
fn test_validation_larger() {
    let points = fibonacci_sphere_points(1000, 0.0, 0);
    let diagram = compute(&points).unwrap();
    let report = validate(&diagram);

    assert!(report.is_valid(), "Expected valid diagram: {}", report);
    // For 1000 cells, expect ~1996 vertices
    assert!(
        (report.num_vertices as i32 - report.expected_vertices as i32).abs() <= 10,
        "Vertex count far from expected: {}",
        report
    );
}
