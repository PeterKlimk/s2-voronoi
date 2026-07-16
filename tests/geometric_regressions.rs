//! Quarantined regressions for confirmed geometric-fidelity defects.

use std::collections::BTreeMap;

use voronoi_mesh::{
    compute_with, compute_with_report, validation::validate, RepairMode, UnitVec3, UnitVec3Like,
    VoronoiConfig,
};

const HEALTHY_F32_DOT_RESIDUAL: f64 = 2.0e-6;

fn dot<A: UnitVec3Like, B: UnitVec3Like>(a: &A, b: &B) -> f64 {
    a.x() as f64 * b.x() as f64 + a.y() as f64 * b.y() as f64 + a.z() as f64 * b.z() as f64
}

fn normalized_chord_sample<A: UnitVec3Like, B: UnitVec3Like>(a: &A, b: &B, t: f64) -> UnitVec3 {
    let x = a.x() as f64 * (1.0 - t) + b.x() as f64 * t;
    let y = a.y() as f64 * (1.0 - t) + b.y() as f64 * t;
    let z = a.z() as f64 * (1.0 - t) + b.z() as f64 * t;
    let inv_len = 1.0 / x.hypot(y).hypot(z);
    UnitVec3::new(
        (x * inv_len) as f32,
        (y * inv_len) as f32,
        (z * inv_len) as f32,
    )
}

fn aud_002_points() -> [UnitVec3; 5] {
    // Explicit public-input counterexample from docs/audit-triage.md. Keep the
    // coordinates here so changes to random fixture generation cannot hide it.
    [
        UnitVec3::new(0.580_496_5, -0.535_992_44, -0.612_973),
        UnitVec3::new(-0.953_108_8, -0.299_328_3, 0.044_567_585),
        UnitVec3::new(0.086_291_43, -0.342_774_1, -0.935_446_26),
        UnitVec3::new(0.526_091_1, -0.658_014_7, -0.538_743_73),
        UnitVec3::new(0.134_592_53, 0.759_519, -0.636_408_57),
    ]
}

fn assert_aud_002_voronoi_geometry() {
    let points = aud_002_points();
    let output = compute_with_report(&points, VoronoiConfig::default())
        .expect("AUD-002 fixture should compute successfully");
    let diagram = &output.diagram;
    let validation = validate(diagram);
    assert!(
        validation.is_strictly_valid(),
        "AUD-002 fixture must remain a strict subdivision: {}",
        validation.headline()
    );

    let mut incident_cells = vec![Vec::<usize>::new(); diagram.num_vertices()];
    let mut edge_cells = BTreeMap::<(u32, u32), Vec<usize>>::new();
    for cell in diagram.iter_cells() {
        for &vertex in cell.vertex_indices {
            incident_cells[vertex as usize].push(cell.generator_index);
        }
        for edge_idx in 0..cell.len() {
            let a = cell.vertex_indices[edge_idx];
            let b = cell.vertex_indices[(edge_idx + 1) % cell.len()];
            edge_cells
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(cell.generator_index);
        }
    }

    let mut max_incident_dot_spread = 0.0_f64;
    for (vertex_idx, cells) in incident_cells.iter().enumerate() {
        if cells.len() < 2 {
            continue;
        }
        let vertex = &diagram.vertices()[vertex_idx];
        let mut min_dot = f64::INFINITY;
        let mut max_dot = f64::NEG_INFINITY;
        for &cell_idx in cells {
            let site_dot = dot(vertex, &diagram.generator(cell_idx));
            min_dot = min_dot.min(site_dot);
            max_dot = max_dot.max(site_dot);
        }
        max_incident_dot_spread = max_incident_dot_spread.max(max_dot - min_dot);
    }

    let mut max_shared_edge_bisector_residual = 0.0_f64;
    for ((a, b), mut cells) in edge_cells {
        cells.sort_unstable();
        cells.dedup();
        assert_eq!(
            cells.len(),
            2,
            "strict shared edge must have two cell owners"
        );

        let va = &diagram.vertices()[a as usize];
        let vb = &diagram.vertices()[b as usize];
        let ga = diagram.generator(cells[0]);
        let gb = diagram.generator(cells[1]);
        for sample_idx in 0..=4 {
            let sample = normalized_chord_sample(va, vb, sample_idx as f64 / 4.0);
            let residual = (dot(&sample, &ga) - dot(&sample, &gb)).abs();
            max_shared_edge_bisector_residual = max_shared_edge_bisector_residual.max(residual);
        }
    }

    assert!(
        max_incident_dot_spread <= HEALTHY_F32_DOT_RESIDUAL
            && max_shared_edge_bisector_residual <= HEALTHY_F32_DOT_RESIDUAL,
        "AUD-002 geometry exceeds healthy f32 residual {HEALTHY_F32_DOT_RESIDUAL:.1e}: \
         max incident-site dot spread={max_incident_dot_spread:.10e}, \
         max shared-edge bisector residual={max_shared_edge_bisector_residual:.10e}"
    );
}

#[test]
fn aud_002_five_sites_preserve_voronoi_geometry() {
    let disabled = VoronoiConfig::default().with_repair_mode(RepairMode::Disabled);
    if let Ok(diagram) = compute_with(&aud_002_points(), disabled) {
        let validation = validate(&diagram);
        assert!(
            validation.is_strictly_valid(),
            "repair-disabled construction may succeed directly, but never invalidly: {}",
            validation.headline()
        );
    }
    assert_aud_002_voronoi_geometry();
}

#[test]
fn aud_016_near_pi_edge_is_a_strict_valid_edge() {
    let points = [
        UnitVec3::new(-0.346_064_27, -0.758_758, -0.551_838_64),
        UnitVec3::new(0.672_760_4, -0.217_307_08, -0.707_227_7),
        UnitVec3::new(-0.753_194_45, 0.368_890_3, 0.544_626_5),
        UnitVec3::new(-0.661_814_2, -0.681_742_25, -0.311_816_45),
    ];
    let output = compute_with_report(&points, VoronoiConfig::default())
        .expect("AUD-016 near-pi fixture should compute");
    let validation = validate(&output.diagram);
    assert!(
        validation.is_strictly_valid(),
        "near-pi edge is shorter than pi and must remain supported: {}",
        validation.headline()
    );
    assert_eq!(validation.euler_characteristic, 2);
    assert_eq!(validation.antipodal_edges, 0);

    let area_sum: f64 = (0..output.diagram.num_cells())
        .map(|i| {
            let area = output.diagram.cell_area(i);
            assert!(area > 0.0, "AUD-016 cell {i} must have positive area");
            area
        })
        .sum();
    assert!(
        (area_sum - 4.0 * std::f64::consts::PI).abs() < 1.0e-4,
        "AUD-016 areas must sum to 4*pi within stored-f32 error, got {area_sum}"
    );
    let lloyd = output.diagram.lloyd_step();
    for i in 0..output.diagram.num_cells() {
        let centroid = output.diagram.cell_centroid(i);
        assert_eq!(centroid, lloyd[i]);
        let owner = output.diagram.generator(i);
        let owner_dot = centroid.dot(owner);
        for j in 0..output.diagram.num_cells() {
            assert!(
                owner_dot >= centroid.dot(output.diagram.generator(j)) - 2.0e-6,
                "AUD-016 centroid {i} must lie in its owner cell"
            );
        }
    }
}
