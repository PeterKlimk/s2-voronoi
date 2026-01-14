//! Public API integration tests for s2-voronoi.

mod support;

use s2_voronoi::{compute, UnitVec3, VoronoiError};
use support::points::{fibonacci_sphere_points, random_sphere_points};

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
    assert_eq!(diagram.generators.len(), points.len());
    for (i, (gen, orig)) in diagram.generators.iter().zip(points.iter()).enumerate() {
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
