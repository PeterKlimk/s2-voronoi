//! Geometric correctness tests for voronoi-mesh.
//!
//! These tests verify geometric invariants that should hold for any valid
//! spherical Voronoi diagram.

mod support;

use std::collections::HashSet;
use std::f32::consts::PI;
use support::points::fibonacci_sphere_points;
use voronoi_mesh::compute;

#[test]
fn test_vertices_on_unit_sphere() {
    let points = fibonacci_sphere_points(500, 0.1, 12345);
    let diagram = compute(&points).unwrap();

    for (i, v) in diagram.vertices().iter().enumerate() {
        let len = v.length_squared().sqrt();
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
fn test_all_cells_have_valid_vertex_count() {
    // Every cell of a supported input must be a real spherical polygon.
    let points = fibonacci_sphere_points(1000, 0.1, 99999);
    let diagram = compute(&points).unwrap();

    for cell in diagram.iter_cells() {
        assert!(
            cell.len() >= 3,
            "cell {} has only {} vertices",
            cell.generator_index,
            cell.len()
        );
    }
}

#[test]
fn test_no_duplicate_vertices_in_cell() {
    let points = fibonacci_sphere_points(500, 0.1, 54321);
    let diagram = compute(&points).unwrap();

    for cell in diagram.iter_cells() {
        let unique: HashSet<u32> = cell.vertex_indices.iter().copied().collect();
        assert_eq!(
            unique.len(),
            cell.vertex_indices.len(),
            "cell {} repeats a boundary vertex",
            cell.generator_index
        );
    }
}

#[test]
fn test_euler_characteristic_exact() {
    // For a spherical Voronoi diagram: V - E + F = 2, with F = num_cells,
    // V counted over referenced vertices (the representation may carry
    // unreferenced leftovers; see the orphan-vertices note in
    // docs/correctness.md), and E = sum(cell boundary lengths) / 2
    // since each edge is shared by exactly 2 cells.

    let points = fibonacci_sphere_points(200, 0.1, 11111);
    let diagram = compute(&points).unwrap();

    let f = diagram.num_cells();
    let used: HashSet<u32> = diagram
        .iter_cells()
        .flat_map(|c| c.vertex_indices.iter().copied())
        .collect();
    let v = used.len();
    let total_edges: usize = diagram.iter_cells().map(|c| c.len()).sum();
    assert_eq!(total_edges % 2, 0, "directed edges must pair up");
    let e = total_edges / 2;

    let euler = v as i32 - e as i32 + f as i32;
    assert_eq!(
        euler, 2,
        "Euler characteristic must be exactly 2 (V={}, E={}, F={})",
        v, e, f
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
    for (v1, v2) in diagram1.vertices().iter().zip(diagram2.vertices().iter()) {
        let [x1, y1, z1] = v1.to_array();
        let [x2, y2, z2] = v2.to_array();
        let diff = ((x1 - x2).powi(2) + (y1 - y2).powi(2) + (z1 - z2).powi(2)).sqrt();
        assert!(diff < 1e-6, "vertices should be identical");
    }
}

#[test]
fn test_cell_areas_sum_to_sphere_area() {
    let points = fibonacci_sphere_points(5_000, 0.1, 2024);
    let diagram = compute(&points).unwrap();

    let mut total = 0.0f64;
    for i in 0..diagram.num_cells() {
        if diagram.canonical_cell_index(i) != i {
            continue;
        }
        let area = diagram.cell_area(i);
        assert!(area > 0.0, "cell {i} must have positive area");
        total += area;
    }
    let sphere = 4.0 * std::f64::consts::PI;
    assert!(
        (total - sphere).abs() / sphere < 1e-6,
        "canonical cell areas must sum to 4*pi, got {total} (expected {sphere})"
    );
}

#[test]
fn test_octahedron_cells_have_equal_areas() {
    use voronoi_mesh::UnitVec3;
    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
    ];
    let diagram = compute(&points).unwrap();
    let expected = 4.0 * std::f64::consts::PI / 6.0;
    for i in 0..6 {
        let area = diagram.cell_area(i);
        assert!(
            (area - expected).abs() / expected < 1e-5,
            "octahedron cell {i} area {area} should be {expected}"
        );
    }
}

#[test]
fn test_welded_twins_share_conditioned_measure_results() {
    use voronoi_mesh::UnitVec3;

    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
        UnitVec3::new(1.0, 0.0, 0.0),
    ];
    let diagram = compute(&points).unwrap();
    assert_eq!(diagram.canonical_cell_index(6), 0);
    assert_eq!(
        diagram.cell_area(6).to_bits(),
        diagram.cell_area(0).to_bits()
    );
    assert_eq!(diagram.cell_centroid(6), diagram.cell_centroid(0));
    let lloyd = diagram.lloyd_step();
    assert_eq!(lloyd[6], lloyd[0]);
}

#[test]
fn test_cell_centroids_unit_length_and_near_generator() {
    let points = fibonacci_sphere_points(2_000, 0.1, 4096);
    let diagram = compute(&points).unwrap();

    // Mean spacing ~ sqrt(4*pi/n); the centroid must land well within the
    // cell's scale of its generator.
    let mean_spacing = (4.0 * std::f64::consts::PI / 2_000.0f64).sqrt();
    for i in 0..diagram.num_cells() {
        let c = diagram.cell_centroid(i);
        let [cx, cy, cz] = c.to_array();
        let len = (cx as f64).hypot(cy as f64).hypot(cz as f64);
        assert!((len - 1.0).abs() < 1e-6, "centroid must be unit length");
        let g = diagram.generator(i);
        let dot = c.dot(g) as f64;
        let angle = dot.clamp(-1.0, 1.0).acos();
        assert!(
            angle < mean_spacing,
            "cell {i} centroid is {angle} rad from its generator (spacing {mean_spacing})"
        );
    }
}

#[test]
fn test_lloyd_relaxation_converges_toward_centroidal() {
    let points = fibonacci_sphere_points(500, 0.5, 99);

    let mean_displacement = |diagram: &voronoi_mesh::SphericalVoronoi| -> f64 {
        (0..diagram.num_cells())
            .map(|i| {
                let g = diagram.generator(i);
                let c = diagram.cell_centroid(i);
                let dot = (g.dot(c) as f64).clamp(-1.0, 1.0);
                dot.acos()
            })
            .sum::<f64>()
            / diagram.num_cells() as f64
    };

    let mut last = f64::MAX;
    let mut diagram = compute(&points).unwrap();
    for iteration in 0..3 {
        let displacement = mean_displacement(&diagram);
        assert!(
            displacement < last,
            "Lloyd iteration {iteration} should reduce mean generator-centroid \
             displacement ({displacement} >= {last})"
        );
        last = displacement;
        diagram = compute(&diagram.lloyd_step()).unwrap();
    }
}
