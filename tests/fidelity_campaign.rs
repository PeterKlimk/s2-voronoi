#![cfg(feature = "tools")]
//! Diagnostic geometric-fidelity campaign (ignored by default).
//!
//! Each invocation computes one environment-selected case, requires the
//! established strict topology contract, and emits one `FIDELITYRESULT` line.
//! `scripts/fidelity_campaign.sh` runs a matrix in isolated processes so large
//! cases do not accumulate allocator high-water memory.

mod support;

use support::points::*;
use voronoi_mesh::{compute_with_report, quality, RepairMode, UnitVec3, VoronoiConfig};

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn welded_points(n: usize, seed: u64) -> Vec<UnitVec3> {
    let duplicates = 16.min(n / 4);
    let mut points = random_sphere_points(n.saturating_sub(duplicates), seed);
    let twins: Vec<UnitVec3> = points.iter().take(duplicates).copied().collect();
    points.extend(twins);
    points
}

fn make_points(dist: &str, n: usize, seed: u64, param: f32) -> Vec<UnitVec3> {
    match dist {
        "uniform" => random_sphere_points(n, seed),
        "fibonacci" => fibonacci_sphere_points(n, param, seed),
        "clustered" => clustered_cap_points(n, param, seed),
        "bimodal" => bimodal_density_points(n, param, seed),
        "cube" => cube_vertex_stress_points(n, param, seed),
        "cocircular" => near_cocircular_stress_points(n, param, seed),
        "mega" => mega_points(n, param, seed),
        "grid" => cubed_sphere_points(n, seed),
        "hemisphere" => hemisphere_points(n, seed),
        "cap" => benchmark_cap_points(n, param, seed),
        "great_circle" => great_circle_points(n, param, seed),
        "welded" => welded_points(n, seed),
        other => panic!("unknown VORONOI_MESH_CASE_DIST '{other}'"),
    }
}

#[test]
#[ignore]
fn fidelity_case() {
    let dist = env_str("VORONOI_MESH_CASE_DIST", "uniform");
    let n: usize = env_parse("VORONOI_MESH_CASE_N", 100_000usize);
    let seed: u64 = env_parse("VORONOI_MESH_CASE_SEED", 1u64);
    let param: f32 = env_parse("VORONOI_MESH_CASE_PARAM", 0.0f32);
    let sampled_cells: usize = env_parse("VORONOI_MESH_FIDELITY_CELLS", 1_024usize);
    let edge_samples: usize = env_parse("VORONOI_MESH_FIDELITY_EDGE_SAMPLES", 3usize);
    let repair_mode = match env_str("VORONOI_MESH_REPAIR_MODE", "local3d").as_str() {
        "local3d" => RepairMode::Local3d,
        "projected" => RepairMode::LocalProjected,
        "disabled" => RepairMode::Disabled,
        mode => panic!("unknown VORONOI_MESH_REPAIR_MODE '{mode}'"),
    };

    let points = make_points(&dist, n, seed, param);
    let original_n = points.len();
    let output = compute_with_report(
        &points,
        VoronoiConfig::default().with_repair_mode(repair_mode),
    )
    .unwrap_or_else(|e| panic!("{dist} n={original_n} seed={seed}: compute failed: {e}"));
    let validation = output.report.preferred_validation();
    assert!(
        validation.is_strictly_valid(),
        "{dist} n={original_n} seed={seed}: strict validation failed: {validation}"
    );
    assert!(
        !output.report.has_post_repair_residuals(),
        "{dist} n={original_n} seed={seed}: default output retained a repair residual"
    );

    let mut report = quality::assess_with_config(
        output.preferred_diagram(),
        quality::QualityConfig {
            max_sampled_cells: sampled_cells,
            edge_samples_per_edge: edge_samples,
        },
    );
    report.canonicalization_angular_error =
        quality::assess_canonicalization(&points, &output.diagram);

    println!(
        "FIDELITYRESULT dist={dist} original_n={original_n} effective_n={} seed={seed} param={param} merged={} perturbed={} pre_defects={} repair_attempted={} repair_accepted={} {}",
        output.preferred_diagram().num_cells(),
        output.report.preprocess.num_merged,
        output.report.degenerate.perturbation_applied,
        output.report.pre_repair_edge_mismatch_count,
        output.report.repair.attempted,
        output.report.repair.accepted,
        report.fidelity_kv_fields(),
    );
}
