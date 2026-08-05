#![cfg(feature = "manual_probes")]
//! Robustness evidence campaign (ignored by default): drive many seeds,
//! sizes, and distributions through `compute_with_report` and record, per
//! input, the defect count, the detection-origin histogram (incl. the
//! post-reconciliation output-invariant backstop), and strict validity.
//!
//! Motivation: the residual-defect story must not rest on the handful of
//! historically known sites. Every new natural input either reaches
//! strict validity (evidence the hardened net holds) or surfaces a defect
//! class we have not characterized — which is exactly what we want to find
//! here rather than in the field.
//!
//! ## Execution model: ONE PROCESS PER CASE
//!
//! Running the whole sweep in a single process OOMs small-RAM boxes — not
//! because any one build is large (3M peaks at ~2.3 GB; memory is ~linear,
//! ~0.65 KB/point) but because glibc does not return freed arenas to the OS
//! between builds, so dozens of sequential builds pin a high-water mark
//! that climbs into swap-thrash and the OOM killer. The fix is process
//! isolation: each case runs in its own process via `campaign_case`
//! (parameterized by env vars), so the OS reclaims fully between cases and
//! peak RSS is bounded to a single build. `scripts/robustness_campaign.sh`
//! is the driver; it builds this binary once and invokes it per case.
//!
//! Run the full campaign:
//!   ./scripts/robustness_campaign.sh
//!
//! Run one case by hand:
//!   VORONOI_MESH_CASE_DIST=uniform VORONOI_MESH_CASE_N=2000000 VORONOI_MESH_CASE_SEED=1 \
//!     cargo test --release --features manual_probes --test robustness_campaign -- --ignored \
//!     campaign_case --nocapture

mod support;

use std::collections::BTreeMap;

use support::points::*;
use voronoi_mesh::{
    compute_with, compute_with_report, validation::validate, EdgeMismatchOrigin, LocalRebuildMode,
    VoronoiConfig,
};

/// Process peak RSS (high-water mark) in MB, from /proc/self/status.
fn peak_rss_mb() -> u64 {
    read_status_kb("VmHWM").map(|kb| kb / 1024).unwrap_or(0)
}

/// Current process RSS in MB, from /proc/self/status.
fn rss_mb() -> u64 {
    read_status_kb("VmRSS").map(|kb| kb / 1024).unwrap_or(0)
}

fn read_status_kb(field: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<u64>().ok())
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Build the point set for a named distribution. `n` is the point count
/// (group count for `cocircular`, which emits 4*n points); `param` is the
/// distribution's shape knob (cap radius, perturbation, jitter), ignored by
/// `uniform`.
fn make_points(dist: &str, n: usize, seed: u64, param: f32) -> Vec<TestPoint> {
    match dist {
        "uniform" => random_sphere_points(n, seed),
        "clustered" => clustered_cap_points(n, param, seed),
        "bimodal" => bimodal_density_points(n, param, seed),
        "cube" => cube_vertex_stress_points(n, param, seed),
        "cocircular" => near_cocircular_stress_points(n, param, seed),
        "fibonacci" => fibonacci_sphere_points(n, param, seed),
        // `param` is the cap fraction; the only distribution dense enough to
        // drive the contested near-cocircular regime (which now errors loudly
        // per the valid-or-error contract).
        "mega" => mega_points(n, param, seed),
        // Clean structured quad grid: O(n) high-degree vertices at normal
        // density — the high-degeneracy/reconcile-load regime.
        "grid" => cubed_sphere_points(n, seed),
        other => panic!("unknown VORONOI_MESH_CASE_DIST '{other}'"),
    }
}

/// One campaign case, parameterized entirely by environment so the driver
/// script can fork a fresh process per case:
///   VORONOI_MESH_CASE_DIST  (uniform|clustered|bimodal|cube|cocircular|fibonacci|mega)
///   VORONOI_MESH_CASE_N     point count (group count for cocircular)
///   VORONOI_MESH_CASE_SEED  rng seed
///   VORONOI_MESH_CASE_PARAM shape knob (f32; default 0.3 — for mega it is the cap fraction)
///
/// Emits a single machine-parseable `CASERESULT` line. Every matrix case is a
/// supported finite/default-preprocessed input and therefore must succeed,
/// clear every reconciliation/local-rebuild residual, and validate strictly. Errors
/// or surviving diagnostics are campaign failures, not accepted outcomes.
#[test]
#[ignore]
fn campaign_case() {
    let dist = env_str("VORONOI_MESH_CASE_DIST", "uniform");
    let n: usize = env_parse("VORONOI_MESH_CASE_N", 1_000_000usize);
    let seed: u64 = env_parse("VORONOI_MESH_CASE_SEED", 1u64);
    let param: f32 = env_parse("VORONOI_MESH_CASE_PARAM", 0.3f32);

    let points = make_points(&dist, n, seed, param);
    let actual_n = points.len();

    match compute_with_report(&points, VoronoiConfig::default()) {
        Ok(out) => {
            let mut origins: BTreeMap<EdgeMismatchOrigin, usize> = BTreeMap::new();
            for &(_, _, origin) in &out.report.reconciliation_edge_records {
                *origins.entry(origin).or_default() += 1;
            }
            let valid = out.report.validation.is_strictly_valid();
            let defects = out.report.reconciliation_edge_records.len();
            let reconciliation_residual = out.report.residual_reconciliation_pairs.len();
            let residual_edges = out.report.residual_unpaired_edges.len() + reconciliation_residual;
            let local_rebuild_attempted = out.report.local_rebuild.attempted();
            let local_rebuild_accepted = out.report.local_rebuild.accepted();
            let origins_str = if origins.is_empty() {
                "none".to_string()
            } else {
                origins
                    .iter()
                    .map(|(o, c)| format!("{o:?}:{c}"))
                    .collect::<Vec<_>>()
                    .join("|")
            };
            println!(
                "CASERESULT dist={dist} n={actual_n} seed={seed} param={param} \
                 result=ok defects={defects} residual_edges={residual_edges} reconciliation_residual={reconciliation_residual} valid={valid} \
                 local_rebuild_attempted={local_rebuild_attempted} local_rebuild_accepted={local_rebuild_accepted} \
                 peak_mb={} origins={origins_str}",
                peak_rss_mb()
            );
            assert!(
                valid && residual_edges == 0,
                "{dist} n={actual_n} seed={seed}: default pipeline did not produce a clean, \
                 strictly valid diagram (valid={valid}, residual_edges={residual_edges}, \
                 origins={origins_str})"
            );
        }
        Err(e) => {
            println!(
                "CASERESULT dist={dist} n={actual_n} seed={seed} param={param} \
                 result=err defects=- valid=- peak_mb={} origins=err:{e:?}",
                peak_rss_mb()
            );
            panic!("{dist} n={actual_n} seed={seed}: supported campaign case failed: {e}");
        }
    }
}

/// Smaller differential for the actual plain-API contract with local rebuilding
/// disabled. A clean error is allowed because this mode deliberately refuses
/// a Hull3d rebuild; any returned diagram must still satisfy the strict sphere
/// validator. This is separate from `campaign_case`, whose default
/// matrix requires success for every supported input.
#[test]
#[ignore]
fn campaign_disabled_case() {
    let dist = env_str("VORONOI_MESH_CASE_DIST", "uniform");
    let n: usize = env_parse("VORONOI_MESH_CASE_N", 1_000_000usize);
    let seed: u64 = env_parse("VORONOI_MESH_CASE_SEED", 1u64);
    let param: f32 = env_parse("VORONOI_MESH_CASE_PARAM", 0.3f32);

    let points = make_points(&dist, n, seed, param);
    let actual_n = points.len();
    let config = VoronoiConfig::default().with_local_rebuild_mode(LocalRebuildMode::Disabled);

    match compute_with(&points, config) {
        Ok(diagram) => {
            let report = validate(&diagram);
            let valid = report.is_strictly_valid();
            println!(
                "DISABLEDRESULT dist={dist} n={actual_n} seed={seed} param={param} \
                 result=ok valid={valid} peak_mb={}",
                peak_rss_mb()
            );
            assert!(
                valid,
                "{dist} n={actual_n} seed={seed}: disabled-mode plain compute \
                 returned a diagram that failed strict validation: {report}"
            );
        }
        Err(e) => println!(
            "DISABLEDRESULT dist={dist} n={actual_n} seed={seed} param={param} \
             result=err valid=- peak_mb={} error={e:?}",
            peak_rss_mb()
        ),
    }
}

/// Memory-accumulation diagnostic: build several moderate diagrams in ONE
/// process, printing RSS after each. Flat RSS => per-case memory; rising
/// RSS => the process is not returning freed memory between cases (glibc
/// arena retention / fragmentation), the reason the campaign uses one
/// process per case. Bounded to 1M x 6 so peak stays well under any ceiling.
#[test]
#[ignore]
fn diag_memory_accumulation() {
    println!("baseline RSS: {} MB", rss_mb());
    for i in 0..6 {
        let pts = random_sphere_points(1_000_000, (i + 1) as u64);
        let out = compute_with_report(&pts, VoronoiConfig::default()).expect("compute");
        let defects = out.report.reconciliation_edge_records.len();
        drop(out);
        drop(pts);
        println!(
            "after build {} (1m s{}): RSS {} MB, defects {}",
            i + 1,
            i + 1,
            rss_mb(),
            defects
        );
    }
    println!("final RSS: {} MB, peak {} MB", rss_mb(), peak_rss_mb());
}

#[test]
#[ignore]
fn diag_single_build_peak() {
    let n: usize = env_parse("VORONOI_MESH_CASE_N", 1_000_000usize);
    let pts = random_sphere_points(n, 1);
    let out = compute_with_report(&pts, VoronoiConfig::default()).expect("compute");
    let defects = out.report.reconciliation_edge_records.len();
    let valid = out.report.validation.is_strictly_valid();
    println!(
        "PEAKPROBE n={n} peak_rss={} MB defects={defects} valid={valid}",
        peak_rss_mb()
    );
}
