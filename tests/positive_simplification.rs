//! Explicit positive-threshold cell-mesh simplification contracts.

mod support;

use std::collections::BTreeSet;

use support::points::{fibonacci_sphere_points, TestPoint};
use voronoi_mesh::{
    compute_with_report, CellSimplificationErrorKind, CellSimplificationOptions,
    CellSimplificationThresholdError, PreprocessMode, SimplificationCellPolicy, VoronoiConfig,
};

fn disabled_weld_cell_killing_points() -> Vec<TestPoint> {
    fn displaced(mut base: [f64; 3], theta: f64, phi: f64) -> TestPoint {
        let length = (base[0] * base[0] + base[1] * base[1] + base[2] * base[2]).sqrt();
        for component in &mut base {
            *component /= length;
        }
        let equatorial = (base[0] * base[0] + base[1] * base[1]).sqrt();
        let e = [-base[1] / equatorial, base[0] / equatorial, 0.0];
        let f = [
            base[1] * e[2] - base[2] * e[1],
            base[2] * e[0] - base[0] * e[2],
            base[0] * e[1] - base[1] * e[0],
        ];
        let cosine = theta.cos();
        let sine = theta.sin();
        TestPoint::new(
            (cosine * base[0] + sine * (phi.cos() * e[0] + phi.sin() * f[0])) as f32,
            (cosine * base[1] + sine * (phi.cos() * e[1] + phi.sin() * f[1])) as f32,
            (cosine * base[2] + sine * (phi.cos() * e[2] + phi.sin() * f[2])) as f32,
        )
        .normalize()
    }

    let base = [-0.61, -0.27, 0.74];
    let theta = 9.0e-8;
    let phase = 3.0 * 0.071;
    let ring = 8;
    let mut points = vec![displaced(base, 0.0, 0.0)];
    for index in 0..ring {
        points.push(displaced(
            base,
            theta,
            phase + std::f64::consts::TAU * index as f64 / ring as f64,
        ));
    }
    let local = points.clone();
    points.extend(
        local
            .into_iter()
            .map(|point| TestPoint::new(-point.x, -point.y, -point.z)),
    );
    points
}

fn stored_chord(a: voronoi_mesh::SpherePoint, b: voronoi_mesh::SpherePoint) -> f64 {
    let a = a.to_array();
    let b = b.to_array();
    let x = a[0] as f64 - b[0] as f64;
    let y = a[1] as f64 - b[1] as f64;
    let z = a[2] as f64 - b[2] as f64;
    (x * x + y * y + z * z).sqrt()
}

#[test]
fn chord_options_reject_invalid_thresholds() {
    assert!(matches!(
        CellSimplificationOptions::from_chord_length(0.0),
        Err(CellSimplificationThresholdError::NonPositive)
    ));
    assert!(matches!(
        CellSimplificationOptions::from_chord_length(-1.0),
        Err(CellSimplificationThresholdError::NonPositive)
    ));
    assert!(matches!(
        CellSimplificationOptions::from_chord_length(f32::NAN),
        Err(CellSimplificationThresholdError::NonFinite)
    ));
    assert!(matches!(
        CellSimplificationOptions::from_chord_length(f32::INFINITY),
        Err(CellSimplificationThresholdError::NonFinite)
    ));
    assert!(matches!(
        CellSimplificationOptions::from_chord_length(2.000_001),
        Err(CellSimplificationThresholdError::ExceedsSphereDiameter)
    ));
    assert!(CellSimplificationOptions::from_chord_length(f32::from_bits(1)).is_ok());
    assert!(CellSimplificationOptions::from_chord_length(2.0).is_ok());
}

#[test]
fn candidate_free_preserve_conversion_is_a_valid_identity() {
    let points = fibonacci_sphere_points(32, 0.05, 7);
    let output = compute_with_report(&points, VoronoiConfig::default()).unwrap();
    let source_cells = output.preferred_diagram().num_cells();
    let source_vertices = output.preferred_diagram().num_vertices();
    let simplified = output
        .into_simplified_cell_mesh(
            CellSimplificationOptions::from_chord_length(f32::from_bits(1)).unwrap(),
        )
        .unwrap();

    assert_eq!(simplified.mesh.num_cells(), source_cells);
    assert_eq!(simplified.mesh.num_vertices(), source_vertices);
    assert_eq!(
        simplified
            .simplification_report
            .positive_components_committed,
        0
    );
    assert_eq!(simplified.simplification_report.remaining_positive_edges, 0);
    assert!(simplified.mesh.validate().is_strictly_valid());
}

#[test]
fn preserve_contracts_a_real_isolated_short_edge() {
    let points = fibonacci_sphere_points(48, 0.35, 91);
    let output = compute_with_report(&points, VoronoiConfig::default()).unwrap();
    let diagram = output.preferred_diagram();
    let mut edges = BTreeSet::new();
    for cell in diagram.iter_cells() {
        for offset in 0..cell.vertex_indices.len() {
            let a = cell.vertex_indices[offset];
            let b = cell.vertex_indices[(offset + 1) % cell.vertex_indices.len()];
            edges.insert((a.min(b), a.max(b)));
        }
    }
    let mut thresholds: Vec<f64> = edges
        .into_iter()
        .map(|(a, b)| stored_chord(diagram.vertex(a as usize), diagram.vertex(b as usize)))
        .collect();
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|a, b| a.to_bits() == b.to_bits());

    for threshold in thresholds.into_iter().take(32) {
        let threshold = (threshold * (1.0 + 8.0 * f32::EPSILON as f64)) as f32;
        let simplified = output.clone().into_simplified_cell_mesh(
            CellSimplificationOptions::from_chord_length(threshold).unwrap(),
        );
        let Ok(simplified) = simplified else {
            continue;
        };
        if simplified
            .simplification_report
            .positive_components_committed
            > 0
        {
            assert!(simplified.mesh.validate().is_strictly_valid());
            assert!(simplified.simplification_report.max_component_diameter <= threshold as f64);
            assert!(
                simplified
                    .simplification_report
                    .max_representative_displacement
                    <= threshold as f64
            );
            return;
        }
    }
    panic!("fixture did not expose an admissible isolated short edge");
}

#[test]
fn elide_matches_the_exact_cell_killing_fixture() {
    let points = disabled_weld_cell_killing_points();
    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Disabled),
    )
    .unwrap();
    let exact = output.clone().into_elided_cell_mesh().unwrap();
    let simplified = output
        .into_simplified_cell_mesh(
            CellSimplificationOptions::from_chord_length(f32::from_bits(1))
                .unwrap()
                .with_cell_policy(SimplificationCellPolicy::Elide),
        )
        .unwrap();

    assert_eq!(simplified.mesh.num_cells(), 16);
    assert_eq!(simplified.mesh.vertices(), exact.mesh.vertices());
    for cell in 0..simplified.mesh.num_cells() {
        assert_eq!(
            simplified.mesh.cell(cell).vertex_indices,
            exact.mesh.cell(cell).vertex_indices
        );
    }
    assert_eq!(simplified.simplification_report.effective_cells_elided, 2);
    assert_eq!(simplified.simplification_report.source_inputs_elided, 2);
    assert_eq!(simplified.simplification_report.remaining_exact_edges, 0);
    assert_eq!(simplified.simplification_report.remaining_positive_edges, 0);
    assert_eq!(
        simplified.simplification_report.exact_suppression_members,
        2
    );
    assert!(simplified.mesh.validate().is_strictly_valid());
}

#[test]
fn error_policy_reports_the_first_unresolved_exact_group_and_returns_source() {
    let points = disabled_weld_cell_killing_points();
    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Disabled),
    )
    .unwrap();
    let source_cells = output.diagram.num_cells();
    let error = output
        .into_simplified_cell_mesh(
            CellSimplificationOptions::from_chord_length(f32::from_bits(1))
                .unwrap()
                .with_cell_policy(SimplificationCellPolicy::Error),
        )
        .unwrap_err();

    assert_eq!(
        error.kind(),
        CellSimplificationErrorKind::UnresolvedExactGroup
    );
    assert!(!error.report().affected_original_inputs.is_empty());
    assert_eq!(error.into_source_output().diagram.num_cells(), source_cells);
}
