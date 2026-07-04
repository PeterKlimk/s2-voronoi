//! Public API integration tests for voronoi-mesh.

mod support;

use support::points::{
    clustered_cap_points, fibonacci_sphere_points, great_circle_points, hemisphere_points,
    random_sphere_points,
};
use voronoi_mesh::{
    compute, compute_with, compute_with_report, validation::validate, DegenerateMode,
    PreprocessMode, UnitVec3, VoronoiConfig, VoronoiError,
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
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Weld),
    )
    .expect("density-based preprocessing should succeed");
    assert_eq!(density.num_cells(), 50);

    let disabled = compute_with(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Disabled),
    )
    .expect("disabled preprocessing should succeed");
    assert_eq!(disabled.num_cells(), 50);
}

#[test]
fn test_compute_with_report_surfaces_preprocess_outcome() {
    let points = random_sphere_points(50, 24680);

    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Weld),
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
fn test_rank2_great_circle_policy_is_explicit() {
    let points = great_circle_points(50, 0.0, 42);

    let strict = compute_with(
        &points,
        VoronoiConfig::default().with_degenerate_mode(DegenerateMode::Strict),
    );
    assert!(
        matches!(
            strict,
            Err(VoronoiError::UnsupportedGeometry { .. })
                | Err(VoronoiError::ComputationFailed(_))
                | Err(VoronoiError::DegenerateInput { .. })
                | Err(VoronoiError::RepresentationLimit(_))
        ),
        "rank-2 great-circle input should fail cleanly in strict mode, got {strict:?}"
    );

    let perturbed = compute_with_report(&points, VoronoiConfig::default())
        .expect("default rank-2 great-circle perturbation should return a valid nearby diagram");

    assert_eq!(
        perturbed.report.degenerate.requested_mode,
        DegenerateMode::PerturbGreatCircle
    );
    assert!(
        perturbed.report.degenerate.perturbation_applied,
        "rank-2 great-circle fixture should take the perturbation retry"
    );
    assert!(
        perturbed.report.preferred_validation().is_strictly_valid(),
        "perturbed rank-2 diagram should validate strictly: {}",
        perturbed.report.preferred_validation().headline()
    );
}

#[test]
fn test_clustered_cap_tight_report_keeps_default_preprocessing_nonintrusive() {
    let points = clustered_cap_points(100, 0.0175, 42);

    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Weld),
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
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::MergeWithin(3.5e-4)),
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

    // Compaction keeps the diagram strictly valid and weld-consistent.
    let mut compacted = output.diagram.clone();
    compacted.compact_vertices();
    let report = validate(&compacted);
    assert!(
        report.is_strictly_valid(),
        "compacted welded diagram should validate strictly: {}",
        report.headline()
    );
    assert_eq!(report.orphan_vertices, 0);
    assert_eq!(
        compacted.welded_twin_count(),
        output.diagram.welded_twin_count()
    );
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
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::MergeWithin(0.001)),
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
fn test_compute_solves_upper_hemisphere_large_cells() {
    let points = hemisphere_points(100, 42);
    let diagram = compute(&points).expect("hemisphere-limited inputs should compute");
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "hemisphere-limited output should validate strictly: {}",
        report.headline()
    );
}

#[test]
#[cfg(feature = "qhull")]
fn test_qhull_available() {
    use glam::Vec3;
    use voronoi_mesh::compute_voronoi_qhull;

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

#[test]
#[cfg(feature = "qhull")]
fn test_qhull_normalizes_inputs_before_exact_3d_hull() {
    use glam::Vec3;
    use voronoi_mesh::compute_voronoi_qhull;

    let points: Vec<Vec3> = vec![
        Vec3::new(1.0001, 0.0, 0.0),
        Vec3::new(-0.9999, 0.0, 0.0),
        Vec3::new(0.0, 1.0002, 0.0),
        Vec3::new(0.0, -0.9998, 0.0),
        Vec3::new(0.0, 0.0, 1.0003),
        Vec3::new(0.0, 0.0, -0.9997),
    ];
    let voronoi = compute_voronoi_qhull(&points);
    assert_eq!(voronoi.num_cells(), 6);
    for (i, g) in voronoi.generators().iter().enumerate() {
        let len = (g.x * g.x + g.y * g.y + g.z * g.z).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-6,
            "qhull generator {i} was not normalized: {len}"
        );
    }
    for (i, v) in voronoi.vertices().iter().enumerate() {
        let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-6,
            "qhull vertex {i} was not normalized: {len}"
        );
    }
}

#[test]
fn test_adjacency_symmetric_complete_and_edge_aligned() {
    use std::collections::HashSet;

    let points = random_sphere_points(2_000, 31337);
    let diagram = compute(&points).unwrap();
    assert!(validate(&diagram).is_strictly_valid());

    let adjacency = diagram.build_adjacency();
    assert_eq!(adjacency.num_cells(), diagram.num_cells());
    assert!(
        adjacency.is_complete(),
        "strictly valid diagram must have a neighbor across every edge"
    );

    let mut total_neighbor_entries = 0usize;
    for i in 0..diagram.num_cells() {
        let cell = diagram.cell(i);
        let neighbors = adjacency.neighbors_of(i);
        assert_eq!(
            neighbors.len(),
            cell.len(),
            "one neighbor per boundary edge"
        );
        total_neighbor_entries += neighbors.len();

        for (k, &j) in neighbors.iter().enumerate() {
            let j = j as usize;
            assert_ne!(j, i, "a cell cannot neighbor itself");

            // Symmetry: i appears in j's neighbor list.
            assert!(
                adjacency.neighbors_of(j).contains(&(i as u32)),
                "adjacency must be symmetric ({i} -> {j})"
            );

            // Edge alignment: the shared edge's vertices appear in j's boundary.
            let a = cell.vertex_indices[k];
            let b = cell.vertex_indices[(k + 1) % cell.len()];
            let j_vertices: HashSet<u32> = diagram.cell(j).vertex_indices.iter().copied().collect();
            assert!(
                j_vertices.contains(&a) && j_vertices.contains(&b),
                "neighbor {j} must share the boundary edge ({a}, {b}) of cell {i}"
            );
        }
    }

    // Every edge contributes one entry per side; average degree ~6.
    let avg = total_neighbor_entries as f64 / diagram.num_cells() as f64;
    assert!(
        (5.5..6.5).contains(&avg),
        "average neighbor count should be ~6, got {avg}"
    );
}

#[test]
fn test_adjacency_with_welded_twins() {
    let mut points = random_sphere_points(2_000, 31338);
    points.push(points[123]); // exact duplicate -> welded twin of 123

    let diagram = compute(&points).unwrap();
    assert_eq!(diagram.canonical_cell_index(2_000), 123);

    let adjacency = diagram.build_adjacency();
    assert!(adjacency.is_complete());

    // The twin reports its canonical cell's neighbors.
    assert_eq!(adjacency.neighbors_of(2_000), adjacency.neighbors_of(123));

    // Twins never appear as neighbors; canonical indices do.
    for i in 0..diagram.num_cells() {
        assert!(
            !adjacency.neighbors_of(i).contains(&2_000),
            "welded twin must not appear as a neighbor"
        );
    }
    let canonical_appears =
        (0..diagram.num_cells()).any(|i| adjacency.neighbors_of(i).contains(&123));
    assert!(
        canonical_appears,
        "canonical cell should appear as a neighbor"
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_serde_roundtrip_preserves_diagram_and_welds() {
    use voronoi_mesh::SphericalVoronoi;

    let mut points = random_sphere_points(500, 777);
    points.push(points[42]); // welded twin -> weld map must survive

    let diagram = compute(&points).unwrap();
    let json = serde_json::to_string(&diagram).expect("serialize");
    let restored: SphericalVoronoi = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored.num_cells(), diagram.num_cells());
    assert_eq!(restored.num_vertices(), diagram.num_vertices());
    assert_eq!(restored.generators(), diagram.generators());
    assert_eq!(restored.vertices(), diagram.vertices());
    for i in 0..diagram.num_cells() {
        assert_eq!(
            restored.cell(i).vertex_indices,
            diagram.cell(i).vertex_indices
        );
    }
    assert_eq!(restored.weld_map(), diagram.weld_map());
    assert_eq!(restored.canonical_cell_index(500), 42);
    assert!(validate(&restored).is_strictly_valid());

    // Adjacency round-trips too.
    let adjacency = diagram.build_adjacency();
    let adj_json = serde_json::to_string(&adjacency).expect("serialize adjacency");
    let adj_restored: voronoi_mesh::CellAdjacency =
        serde_json::from_str(&adj_json).expect("deserialize adjacency");
    for i in 0..diagram.num_cells() {
        assert_eq!(adj_restored.neighbors_of(i), adjacency.neighbors_of(i));
    }
}

#[test]
fn test_merge_within_large_radius_uses_standalone_detector() {
    // A radius far above the grid-adjacency bound (1/(16*res)) must route
    // through the standalone weld detector + grid rebuild fallback and
    // still produce a strictly valid welded diagram.
    let mut points = random_sphere_points(2_000, 77);
    let base = points[10];
    points.push(base);

    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::MergeWithin(0.05)),
    )
    .expect("large MergeWithin radius should compute via the fallback path");

    assert!(
        output.report.preprocess.did_merge(),
        "the duplicated point (and any natural sub-0.05 pairs) must weld"
    );
    assert!(
        output.report.returned_validation.is_strictly_valid(),
        "returned diagram should validate strictly: {}",
        output.report.returned_validation.headline()
    );
    assert_eq!(
        output.report.returned_validation.welded_twin_cells,
        output.report.preprocess.num_merged
    );
}
