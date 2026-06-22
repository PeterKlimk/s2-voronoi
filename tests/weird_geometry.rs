//! Weird-geometry contract fixtures.
//!
//! These are not random stress tests. They are named geometric regimes that
//! either already succeed through welding / fallback / stitching, or currently
//! fail cleanly because the input asks for a lower-dimensional or very large-cell
//! spherical diagram. If we later add a "no fail with welding" mode, this file is
//! the checklist of rows to promote from clean failure to strict success.

mod support;

use s2_voronoi::{
    compute, compute_with, validation::validate, PreprocessMode, UnitVec3, VoronoiConfig,
    VoronoiError,
};
use std::f32::consts::PI;
use support::points::{
    clustered_cap_points, cubed_sphere_points, great_circle_points, hemisphere_points,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    StrictSuccess,
    CleanFailure,
}

fn u(x: f32, y: f32, z: f32) -> UnitVec3 {
    let l = (x * x + y * y + z * z).sqrt();
    UnitVec3::new(x / l, y / l, z / l)
}

fn expect_case(name: &str, points: Vec<UnitVec3>, expected: Expected) {
    let result = compute(&points);
    match expected {
        Expected::StrictSuccess => {
            let diagram = result.unwrap_or_else(|err| panic!("{name}: expected success: {err:?}"));
            let report = validate(&diagram);
            assert!(
                report.is_strictly_valid(),
                "{name}: expected strict validity, got {}",
                report.headline()
            );
        }
        Expected::CleanFailure => {
            assert!(
                matches!(
                    result,
                    Err(VoronoiError::UnsupportedGeometry { .. })
                        | Err(VoronoiError::ComputationFailed(_))
                        | Err(VoronoiError::DegenerateInput { .. })
                        | Err(VoronoiError::RepresentationLimit(_))
                ),
                "{name}: expected clean failure, got {result:?}"
            );
        }
    }
}

fn equator_with_poles(n: usize) -> Vec<UnitVec3> {
    let mut points = great_circle_points(n, 0.0, 0);
    points.push(u(0.0, 0.0, 1.0));
    points.push(u(0.0, 0.0, -1.0));
    points
}

fn latitude_ring_with_apex(n: usize) -> Vec<UnitVec3> {
    let r = (0.5f32).sqrt();
    let mut points = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = 2.0 * PI * i as f32 / n as f32;
        points.push(u(r * t.cos(), r * t.sin(), r));
    }
    points.push(u(0.0, 0.0, -1.0));
    points
}

#[test]
fn weird_geometry_contract_cases() {
    let cases: [(&str, Vec<UnitVec3>, Expected); 8] = [
        (
            "pure_great_circle_rank2",
            great_circle_points(50, 0.0, 42),
            Expected::CleanFailure,
        ),
        (
            "small_jitter_great_circle",
            great_circle_points(20, 0.01, 42),
            Expected::StrictSuccess,
        ),
        (
            "upper_hemisphere_large_cells",
            hemisphere_points(100, 42),
            Expected::CleanFailure,
        ),
        (
            "upper_hemisphere_dense_large_cells",
            hemisphere_points(500, 42),
            Expected::CleanFailure,
        ),
        (
            "anchored_clustered_cap",
            clustered_cap_points(100, 0.0175, 42),
            Expected::StrictSuccess,
        ),
        (
            "cubed_sphere_degree4_grid",
            cubed_sphere_points(6 * 12 * 12, 0),
            Expected::StrictSuccess,
        ),
        (
            "exact_cocircular_pyramid",
            latitude_ring_with_apex(6),
            Expected::StrictSuccess,
        ),
        (
            "great_circle_with_two_poles",
            equator_with_poles(24),
            Expected::StrictSuccess,
        ),
    ];

    for (name, points, expected) in cases {
        expect_case(name, points, expected);
    }
}

#[test]
fn welding_subradius_cluster_is_strictly_valid() {
    let mut points = vec![
        u(1.0, 0.0, 0.0),
        u(-1.0, 0.0, 0.0),
        u(0.0, 1.0, 0.0),
        u(0.0, -1.0, 0.0),
        u(0.0, 0.0, 1.0),
        u(0.0, 0.0, -1.0),
    ];
    // A cluster well below the explicit weld threshold. The point of this case
    // is not exact geometry; it pins that welding turns a sub-resolution local
    // feature into a valid solved effective problem.
    for i in 0..8 {
        let t = 2.0 * PI * i as f32 / 8.0;
        points.push(u(1.0, 1e-5 * t.cos(), 1e-5 * t.sin()));
    }

    let output = s2_voronoi::compute_with_report(
        &points,
        VoronoiConfig {
            preprocess_mode: PreprocessMode::MergeWithin(1e-4),
            ..VoronoiConfig::default()
        },
    )
    .expect("explicit welding should solve the subradius cluster");

    assert!(output.report.preprocess.did_merge());
    assert!(
        output.report.preferred_validation().is_strictly_valid(),
        "welded effective diagram should validate strictly: {}",
        output.report.preferred_validation().headline()
    );
}

#[test]
#[ignore = "future goal: no-fail weird geometries under welding/fallback mode"]
fn future_no_fail_with_welding_targets() {
    for (name, points) in [
        ("pure_great_circle_rank2", great_circle_points(50, 0.0, 42)),
        ("upper_hemisphere_large_cells", hemisphere_points(100, 42)),
        (
            "upper_hemisphere_dense_large_cells",
            hemisphere_points(500, 42),
        ),
    ] {
        let diagram = compute_with(
            &points,
            VoronoiConfig {
                preprocess_mode: PreprocessMode::Weld,
                ..VoronoiConfig::default()
            },
        )
        .unwrap_or_else(|err| panic!("{name}: future no-fail target still errors: {err:?}"));
        let report = validate(&diagram);
        assert!(
            report.is_strictly_valid(),
            "{name}: future no-fail target returned invalid diagram: {}",
            report.headline()
        );
    }
}
