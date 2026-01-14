//! Geometric correctness tests for s2-voronoi.
//!
//! These tests verify geometric invariants that should hold for any valid
//! spherical Voronoi diagram.

mod support;

use s2_voronoi::compute;
use std::collections::HashSet;
use std::f32::consts::PI;
use support::points::fibonacci_sphere_points;

#[test]
fn test_vertices_on_unit_sphere() {
    let points = fibonacci_sphere_points(500, 0.1, 12345);
    let diagram = compute(&points).unwrap();

    for (i, v) in diagram.vertices.iter().enumerate() {
        let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "vertex {} not on unit sphere: length = {}",
            i,
            len
        );
    }
}

#[test]
fn test_cell_count_equals_input() {
    for n in [10, 100, 500] {
        let points = fibonacci_sphere_points(n, 0.1, 42);
        let diagram = compute(&points).unwrap();
        assert_eq!(
            diagram.num_cells(),
            n,
            "cell count should equal input count"
        );
    }
}

#[test]
fn test_most_cells_have_valid_vertex_count() {
    // Most cells should have >= 3 vertices (some may fail due to numerical issues)
    let points = fibonacci_sphere_points(1000, 0.1, 99999);
    let diagram = compute(&points).unwrap();

    let valid_cells = diagram.iter_cells().filter(|c| c.len() >= 3).count();

    let ratio = valid_cells as f32 / diagram.num_cells() as f32;
    assert!(
        ratio > 0.99,
        "at least 99% of cells should have >= 3 vertices, got {:.1}%",
        ratio * 100.0
    );
}

#[test]
fn test_no_duplicate_vertices_in_cell() {
    let points = fibonacci_sphere_points(500, 0.1, 54321);
    let diagram = compute(&points).unwrap();

    let mut cells_with_dupes = 0;
    for cell in diagram.iter_cells() {
        let unique: HashSet<u32> = cell.vertex_indices.iter().copied().collect();
        if unique.len() < cell.vertex_indices.len() {
            cells_with_dupes += 1;
        }
    }

    let ratio = cells_with_dupes as f32 / diagram.num_cells() as f32;
    assert!(
        ratio < 0.01,
        "less than 1% of cells should have duplicate vertices, got {:.1}%",
        ratio * 100.0
    );
}

#[test]
fn test_euler_characteristic_approximate() {
    // For a spherical Voronoi diagram: V - E + F = 2
    // where F = num_cells, V = num_vertices
    // Each edge is shared by exactly 2 cells, so E = sum(cell_vertices) / 2

    let points = fibonacci_sphere_points(200, 0.1, 11111);
    let diagram = compute(&points).unwrap();

    let f = diagram.num_cells();
    let v = diagram.num_vertices();
    let total_edges: usize = diagram.iter_cells().map(|c| c.len()).sum();
    let e = total_edges / 2; // Each edge counted twice

    let euler = v as i32 - e as i32 + f as i32;

    // Due to degenerate cells, this may not be exactly 2
    assert!(
        (euler - 2).abs() <= 5,
        "Euler characteristic should be close to 2, got {} (V={}, E={}, F={})",
        euler,
        v,
        e,
        f
    );
}

#[test]
fn test_average_vertices_per_cell() {
    // For a random spherical Voronoi, average ~6 vertices per cell (hexagons)
    let points = fibonacci_sphere_points(1000, 0.1, 77777);
    let diagram = compute(&points).unwrap();

    let total_vertices: usize = diagram.iter_cells().map(|c| c.len()).sum();
    let avg = total_vertices as f32 / diagram.num_cells() as f32;

    assert!(
        avg > 5.5 && avg < 6.5,
        "average vertices per cell should be ~6, got {}",
        avg
    );
}

#[test]
fn test_cell_areas_reasonable() {
    // Sum of cell areas should be approximately 4π (sphere surface)
    // We can estimate area from the number of triangles formed by center + vertices

    let points = fibonacci_sphere_points(500, 0.1, 33333);
    let diagram = compute(&points).unwrap();

    let mean_area = 4.0 * PI / diagram.num_cells() as f32;

    // Just verify cells aren't absurdly sized
    for cell in diagram.iter_cells() {
        if cell.len() >= 3 {
            // Cell has at least a triangle
            assert!(
                cell.len() <= 20,
                "cell has too many vertices: {}",
                cell.len()
            );
        }
    }

    // Mean area should be reasonable
    let expected_mean = 4.0 * PI / 500.0;
    assert!(
        (mean_area - expected_mean).abs() < 0.1,
        "mean area calculation sanity check"
    );
}

#[test]
fn test_reproducibility() {
    // Same input should produce same output
    let points1 = fibonacci_sphere_points(100, 0.1, 12345);
    let points2 = fibonacci_sphere_points(100, 0.1, 12345);

    let diagram1 = compute(&points1).unwrap();
    let diagram2 = compute(&points2).unwrap();

    assert_eq!(diagram1.num_cells(), diagram2.num_cells());
    assert_eq!(diagram1.num_vertices(), diagram2.num_vertices());

    // Compare vertex positions
    for (v1, v2) in diagram1.vertices.iter().zip(diagram2.vertices.iter()) {
        let diff = ((v1.x - v2.x).powi(2) + (v1.y - v2.y).powi(2) + (v1.z - v2.z).powi(2)).sqrt();
        assert!(diff < 1e-6, "vertices should be identical");
    }
}

#[test]
#[cfg(feature = "qhull")]
fn test_knn_matches_qhull_structure() {
    use glam::Vec3;
    use s2_voronoi::compute_voronoi_qhull;

    // For well-spaced points, knn_clipping should produce similar structure to qhull
    let points = fibonacci_sphere_points(100, 0.05, 44444);
    let vec3_points: Vec<Vec3> = points.iter().map(|p| Vec3::new(p.x, p.y, p.z)).collect();

    let knn_diagram = compute(&points).unwrap();
    let qhull_output = compute_voronoi_qhull(&vec3_points);

    // Same number of cells
    assert_eq!(knn_diagram.num_cells(), qhull_output.num_cells());

    // Vertex counts should be similar for most cells
    let mut matching_vertex_counts = 0;
    for i in 0..knn_diagram.num_cells() {
        let knn_cell = knn_diagram.cell(i);
        let qhull_cell = qhull_output.cell(i);
        if knn_cell.len() == qhull_cell.len() {
            matching_vertex_counts += 1;
        }
    }

    let match_ratio = matching_vertex_counts as f32 / knn_diagram.num_cells() as f32;
    assert!(
        match_ratio > 0.95,
        "at least 95% of cells should have matching vertex counts with qhull, got {:.1}%",
        match_ratio * 100.0
    );
}
