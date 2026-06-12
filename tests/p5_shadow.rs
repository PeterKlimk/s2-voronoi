//! P5 stage-1 shadow-audit probes (feature `p5_shadow`): run representative
//! inputs and print the margin/disagreement report. Diagnostic, not a
//! contract test — run with
//! `cargo test --release --features p5_shadow --test p5_shadow -- --ignored --nocapture`
#![cfg(feature = "p5_shadow")]

mod support;

use s2_voronoi::{compute_with_report, UnitVec3Like, VoronoiConfig};
use support::points::*;

/// The defect site from uniform 2M seed 1 (see tests/edge_repair_net.rs).
const SITE_CENTER: (f32, f32, f32) = (-0.419_580_3, -0.220_115_17, -0.880_625_7);

fn defect_fixture() -> Vec<s2_voronoi::UnitVec3> {
    let mean_spacing = (4.0 * std::f32::consts::PI / 2_000_000.0).sqrt();
    let cos_window = (10.0 * mean_spacing).cos();
    let cos_excl = (15.0 * mean_spacing).cos();
    let c = SITE_CENTER;
    let mut fixture: Vec<s2_voronoi::UnitVec3> = random_sphere_points(2_000_000, 1)
        .into_iter()
        .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 >= cos_window)
        .collect();
    fixture.extend(
        random_sphere_points(2_000, 4242)
            .into_iter()
            .filter(|p| p.x() * c.0 + p.y() * c.1 + p.z() * c.2 < cos_excl),
    );
    fixture
}

fn run_case(name: &str, points: &[s2_voronoi::UnitVec3]) {
    s2_voronoi::p5_shadow::reset();
    let out = compute_with_report(points, VoronoiConfig::default()).expect(name);
    println!(
        "=== {name}: n={} defects={} ===",
        points.len(),
        out.report.unresolved_edge_pairs.len()
    );
    print!("{}", s2_voronoi::p5_shadow::report());
}

#[test]
#[ignore]
fn probe_shadow_audit() {
    run_case("defect_fixture", &defect_fixture());
    run_case("uniform_100k_s2", &random_sphere_points(100_000, 2));
    run_case("uniform_2m_s1", &random_sphere_points(2_000_000, 1));
}
