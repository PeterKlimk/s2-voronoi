//! Output-resolution regressions, including the minimized Hex3 zero-edge core.

use voronoi_mesh::{compute_with_report, PreprocessMode, UnitVec3, VoronoiConfig};

fn point(x: f32, y: f32, z: f32) -> UnitVec3 {
    UnitVec3::new(x, y, z)
}

#[test]
fn hex3_four_generator_core_contracts_stored_zero_edge() {
    // Four generators incident to the two bit-identical f32 Voronoi vertices
    // in Hex3's seed-12345 250k handoff. Antipodal copies are only distant
    // supports that close the spherical diagram; they do not define the local
    // empty-cap corner.
    let core = [
        [0.5433253049850464, 0.6803092956542969, 0.4919114410877228],
        [0.5451860427856445, 0.6789295077323914, 0.4917590022087097],
        [0.5443676710128784, 0.6787402629852295, 0.4929254353046417],
        [0.5445437431335449, 0.6799840927124023, 0.49101296067237854],
    ];
    let mut points = Vec::with_capacity(8);
    for [x, y, z] in core {
        points.push(point(x, y, z));
    }
    for [x, y, z] in core {
        points.push(point(-x, -y, -z));
    }

    let output = compute_with_report(&points, VoronoiConfig::default())
        .expect("minimized Hex3 core should compute");
    assert!(output.report.returned_validation.is_strictly_valid());
    assert!(
        output.report.output_resolution.exact_zero_edges_detected > 0,
        "fixture must retain the incident's stored-zero local corner"
    );
    assert_eq!(
        output.report.output_resolution.exact_zero_edges_remaining, 0,
        "non-cell-killing exact-zero edge must be canonicalized"
    );
    assert_eq!(
        output
            .report
            .output_resolution
            .cell_killing_components_preserved,
        0
    );
    assert_eq!(
        output.report.output_resolution.exact_zero_edges_contracted,
        output.report.output_resolution.exact_zero_edges_detected
    );
    assert_eq!(output.report.returned_validation.zero_length_edges, 0);
    assert_eq!(output.diagram.num_cells(), points.len());
}

#[test]
fn separated_tiny_cell_survives_increasing_weld_radius() {
    // A center site surrounded by six sites at ~2.5x the configured weld
    // radius has an intentionally tiny but positive cell. Coarse equatorial
    // and southern supports close the rest of the spherical diagram. This is
    // not a proof of the cap-area bound, but it pins the expected behavior as
    // the separation floor moves thousands of f32 ulps above storage noise.
    for radius in [1.4e-6f32, 1.0e-5, 1.0e-4, 1.0e-3] {
        let theta = 2.5 * radius;
        let mut points = vec![point(0.0, 0.0, 1.0)];
        for k in 0..6 {
            let phi = std::f32::consts::TAU * (k as f32 + 0.07 * (k % 2) as f32) / 6.0;
            let local_theta = theta * (1.0 + 0.025 * k as f32);
            points.push(point(
                local_theta.sin() * phi.cos(),
                local_theta.sin() * phi.sin(),
                local_theta.cos(),
            ));
        }
        points.extend([
            point(1.0, 0.0, 0.0),
            point(0.0, 1.0, 0.0),
            point(-1.0, 0.0, 0.0),
            point(0.0, -1.0, 0.0),
            point(0.0, 0.0, -1.0),
        ]);

        let config =
            VoronoiConfig::default().with_preprocess_mode(PreprocessMode::MergeWithin(radius));
        let output = compute_with_report(&points, config)
            .unwrap_or_else(|error| panic!("weld radius {radius:.3e}: {error}"));
        assert_eq!(
            output.report.preprocess.num_merged, 0,
            "fixture sites must remain outside weld radius {radius:.3e}"
        );
        assert!(
            output.report.returned_validation.is_strictly_valid(),
            "weld radius {radius:.3e}: {}",
            output.report.returned_validation.headline()
        );
        assert_eq!(
            output
                .report
                .output_resolution
                .cell_killing_components_preserved,
            0,
            "weld radius {radius:.3e} unexpectedly produced a cell-killing zero component"
        );
        assert_eq!(
            output.report.returned_validation.zero_length_edges, 0,
            "weld radius {radius:.3e} returned zero geometry"
        );
        assert!(
            output.diagram.cell(0).len() >= 3,
            "center cell must remain representable at radius {radius:.3e}"
        );
    }
}

/// A representable set of distinct f32 generators whose stored Voronoi output
/// contains cell-killing exact-zero edges when preprocessing welding is off.
/// Keep this fixture reusable as the public output policy gains Error/Elide.
fn disabled_weld_cell_killing_points() -> Vec<UnitVec3> {
    fn displaced(mut b: [f64; 3], theta: f64, phi: f64) -> UnitVec3 {
        let bl = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        for x in &mut b {
            *x /= bl;
        }
        let el = (b[0] * b[0] + b[1] * b[1]).sqrt();
        let e = [-b[1] / el, b[0] / el, 0.0];
        let f = [
            b[1] * e[2] - b[2] * e[1],
            b[2] * e[0] - b[0] * e[2],
            b[0] * e[1] - b[1] * e[0],
        ];
        let c = theta.cos();
        let s = theta.sin();
        UnitVec3::new(
            (c * b[0] + s * (phi.cos() * e[0] + phi.sin() * f[0])) as f32,
            (c * b[1] + s * (phi.cos() * e[1] + phi.sin() * f[1])) as f32,
            (c * b[2] + s * (phi.cos() * e[2] + phi.sin() * f[2])) as f32,
        )
        .normalize()
    }

    let base = [-0.61, -0.27, 0.74];
    let theta = 9.0e-8;
    let phase = 3.0 * 0.071;
    let ring = 8;
    let mut points = vec![displaced(base, 0.0, 0.0)];
    for k in 0..ring {
        points.push(displaced(
            base,
            theta,
            phase + std::f64::consts::TAU * k as f64 / ring as f64,
        ));
    }
    let local = points.clone();
    points.extend(local.into_iter().map(|p| point(-p.x, -p.y, -p.z)));
    points
}

#[test]
fn disabled_welding_preserves_cell_killing_zero_components() {
    let points = disabled_weld_cell_killing_points();
    for (i, point) in points.iter().enumerate() {
        assert!(
            !points[..i].contains(point),
            "fixture generators {i} and an earlier generator must be distinct at f32"
        );
    }

    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_preprocess_mode(PreprocessMode::Disabled),
    )
    .expect("distinct-generator cell-killing fixture should compute");
    let resolution = output.report.output_resolution;

    assert_eq!(output.report.preprocess.num_merged, 0);
    assert!(output.diagram.weld_map().is_none());
    assert!(output.report.returned_validation.is_strictly_valid());
    assert_eq!(resolution.exact_zero_edges_detected, 3);
    assert_eq!(resolution.exact_zero_components_detected, 3);
    assert_eq!(resolution.exact_zero_edges_contracted, 0);
    assert_eq!(resolution.exact_zero_components_contracted, 0);
    assert_eq!(resolution.cell_killing_components_preserved, 3);
    assert_eq!(resolution.topology_rejected_components, 0);
    assert_eq!(resolution.exact_zero_edges_remaining, 3);
    assert_eq!(output.report.returned_validation.zero_length_edges, 3);
    assert_eq!(output.diagram.num_cells(), points.len());

    // This is not merely a declined conservative transaction: at least one
    // remaining zero edge bounds a triangle, so contracting it would turn the
    // owning generator's cell into a two-vertex cycle.
    let has_triangle_witness = output.diagram.iter_cells().any(|cell| {
        cell.len() == 3
            && (0..cell.len()).any(|edge| {
                let a = cell.vertex_indices[edge] as usize;
                let b = cell.vertex_indices[(edge + 1) % cell.len()] as usize;
                a != b && output.diagram.vertex(a) == output.diagram.vertex(b)
            })
    });
    assert!(
        has_triangle_witness,
        "fixture must contain a cell-killing zero edge"
    );
}
