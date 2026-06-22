//! Weird-geometry contract fixtures.
//!
//! These are not random stress tests. They are named geometric regimes that
//! either already succeed through welding / fallback / stitching, or use default
//! robust perturbation for rank-deficient spherical input. Pure rank-2
//! great-circle input is a clean failure only under explicit
//! `DegenerateMode::Strict`.

mod support;

use s2_voronoi::{
    compute, validation::validate, DegenerateMode, PreprocessMode, RepairMode, UnitVec3,
    VoronoiConfig,
};
use std::f32::consts::PI;
use support::points::{
    benchmark_cap_points, clustered_cap_points, cubed_sphere_points, great_circle_points,
    hemisphere_points,
};

fn u(x: f32, y: f32, z: f32) -> UnitVec3 {
    let l = (x * x + y * y + z * z).sqrt();
    UnitVec3::new(x / l, y / l, z / l)
}

fn expect_strict_success(name: &str, points: Vec<UnitVec3>) {
    let diagram =
        compute(&points).unwrap_or_else(|err| panic!("{name}: expected success: {err:?}"));
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "{name}: expected strict validity, got {}",
        report.headline()
    );
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

fn pole_with_latitude_ring(n: usize, z: f32) -> Vec<UnitVec3> {
    let r = (1.0 - z * z).sqrt();
    let mut points = Vec::with_capacity(n + 2);
    points.push(u(0.0, 0.0, 1.0));
    points.push(u(0.0, 0.0, -1.0));
    for i in 0..n {
        let t = 2.0 * PI * i as f32 / n as f32;
        points.push(u(r * t.cos(), r * t.sin(), z));
    }
    points
}

#[test]
fn weird_geometry_contract_cases() {
    let cases: [(&str, Vec<UnitVec3>); 8] = [
        ("pure_great_circle_rank2", great_circle_points(50, 0.0, 42)),
        (
            "small_jitter_great_circle",
            great_circle_points(20, 0.01, 42),
        ),
        ("upper_hemisphere_large_cells", hemisphere_points(100, 42)),
        (
            "upper_hemisphere_dense_large_cells",
            hemisphere_points(500, 42),
        ),
        (
            "anchored_clustered_cap",
            clustered_cap_points(100, 0.0175, 42),
        ),
        (
            "cubed_sphere_degree4_grid",
            cubed_sphere_points(6 * 12 * 12, 0),
        ),
        ("exact_cocircular_pyramid", latitude_ring_with_apex(6)),
        ("great_circle_with_two_poles", equator_with_poles(24)),
    ];

    for (name, points) in cases {
        expect_strict_success(name, points);
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
fn robust_great_circle_perturbation_solves_rank2_fixture() {
    let points = great_circle_points(50, 0.0, 42);
    let output = s2_voronoi::compute_with_report(&points, VoronoiConfig::default())
        .expect("default great-circle perturbation should solve rank-2 fixture");

    assert_eq!(
        output.report.degenerate.requested_mode,
        DegenerateMode::PerturbGreatCircle
    );
    assert!(
        output.report.degenerate.perturbation_applied,
        "rank-2 fixture should take the perturbation retry"
    );
    assert!(
        output.report.preferred_validation().is_strictly_valid(),
        "perturbed great-circle diagram should validate strictly: {}",
        output.report.preferred_validation().headline()
    );
}

#[test]
fn dense_cap_frontier_requires_repair_but_default_is_valid() {
    let points = benchmark_cap_points(50_000, 0.05, 7);

    let raw = s2_voronoi::compute_with_report(
        &points,
        VoronoiConfig {
            repair_mode: RepairMode::Disabled,
            ..VoronoiConfig::default()
        },
    )
    .expect("dense-cap raw path should build far enough to report the residual defect");

    assert!(
        !raw.report.returned_validation.is_strictly_valid(),
        "dense-cap fixture should remain a raw fast-path defect; replace it if this becomes valid"
    );
    assert!(
        !raw.report.pre_repair_edge_mismatches.is_empty(),
        "raw dense-cap defect should be visible as pre-repair edge mismatches"
    );
    assert!(
        raw.report.has_post_repair_residuals(),
        "raw dense-cap defect should leave post-repair residuals when local repair is disabled"
    );

    let repaired = s2_voronoi::compute_with_report(&points, VoronoiConfig::default())
        .expect("default repair should solve the dense-cap frontier fixture");
    assert!(
        repaired.report.returned_validation.is_strictly_valid(),
        "default repair should return a strict-valid dense-cap diagram: {}",
        repaired.report.returned_validation.headline()
    );
    assert!(
        !repaired.report.pre_repair_edge_mismatches.is_empty(),
        "default dense-cap fixture should still exercise the repair path"
    );
    assert!(
        !repaired.report.has_post_repair_residuals(),
        "accepted default repair should clear output-invariant residuals"
    );
}

#[test]
#[ignore = "diagnostic: prints the current weird-geometry failure taxonomy"]
fn classify_weird_geometry_failures() {
    for (name, points) in [
        ("pure_great_circle_rank2", great_circle_points(50, 0.0, 42)),
        ("upper_hemisphere_large_cells", hemisphere_points(100, 42)),
        (
            "upper_hemisphere_dense_large_cells",
            hemisphere_points(500, 42),
        ),
        (
            "latitude_ring_32_near_north_pole",
            pole_with_latitude_ring(32, 0.5),
        ),
        (
            "latitude_ring_64_near_north_pole",
            pole_with_latitude_ring(64, 0.5),
        ),
        (
            "dense_cap_50k_repair_frontier",
            benchmark_cap_points(50_000, 0.05, 7),
        ),
    ] {
        match s2_voronoi::compute_with_report(&points, VoronoiConfig::default()) {
            Ok(output) => eprintln!(
                "WEIRDCASE {name}: ok cells={} validation={} pre_repair={}",
                output.preferred_diagram().num_cells(),
                output.report.preferred_validation().headline(),
                output.report.pre_repair_edge_mismatches.len()
            ),
            Err(err) => eprintln!("WEIRDCASE {name}: err {err:?}"),
        }
    }
}
