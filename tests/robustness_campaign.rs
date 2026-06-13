//! Robustness evidence campaign (ignored by default): drive many seeds,
//! sizes, and distributions through `compute_with_report` and record, per
//! input, the defect count, the detection-origin histogram (incl. the
//! post-repair output-invariant backstop), and strict validity.
//!
//! Motivation: the residual-defect story must not rest on the handful of
//! historically known sites. Every new natural input either repairs to
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
//!   S2_CASE_DIST=uniform S2_CASE_N=2000000 S2_CASE_SEED=1 \
//!     cargo test --release --test robustness_campaign -- --ignored \
//!     campaign_case --nocapture

mod support;

use std::collections::BTreeMap;

use s2_voronoi::{compute_with_report, UnitVec3, UnresolvedEdgeOrigin, VoronoiConfig};
use support::points::*;

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
fn make_points(dist: &str, n: usize, seed: u64, param: f32) -> Vec<UnitVec3> {
    match dist {
        "uniform" => random_sphere_points(n, seed),
        "clustered" => clustered_cap_points(n, param, seed),
        "bimodal" => bimodal_density_points(n, param, seed),
        "cube" => cube_vertex_stress_points(n, param, seed),
        "cocircular" => near_cocircular_stress_points(n, param, seed),
        "fibonacci" => fibonacci_sphere_points(n, param, seed),
        other => panic!("unknown S2_CASE_DIST '{other}'"),
    }
}

/// One campaign case, parameterized entirely by environment so the driver
/// script can fork a fresh process per case:
///   S2_CASE_DIST  (uniform|clustered|bimodal|cube|cocircular|fibonacci)
///   S2_CASE_N     point count (group count for cocircular)
///   S2_CASE_SEED  rng seed
///   S2_CASE_PARAM shape knob (f32; default 0.3)
///
/// Emits a single machine-parseable `CASERESULT` line. A build that returns
/// an error is recorded as `result=err` (documented out-of-envelope
/// behavior) WITHOUT failing the test; a build that succeeds but is not
/// strictly valid fails the test (a real, newly found invalid-output bug).
#[test]
#[ignore]
fn campaign_case() {
    let dist = env_str("S2_CASE_DIST", "uniform");
    let n: usize = env_parse("S2_CASE_N", 1_000_000usize);
    let seed: u64 = env_parse("S2_CASE_SEED", 1u64);
    let param: f32 = env_parse("S2_CASE_PARAM", 0.3f32);

    let points = make_points(&dist, n, seed, param);
    let actual_n = points.len();

    match compute_with_report(&points, VoronoiConfig::default()) {
        Ok(out) => {
            let mut origins: BTreeMap<UnresolvedEdgeOrigin, usize> = BTreeMap::new();
            for &(_, _, origin) in &out.report.unresolved_edge_pairs {
                *origins.entry(origin).or_default() += 1;
            }
            let valid = out.report.preferred_validation().is_strictly_valid();
            let defects = out.report.unresolved_edge_pairs.len();
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
                 result=ok defects={defects} valid={valid} peak_mb={} origins={origins_str}",
                peak_rss_mb()
            );
            assert!(
                valid,
                "{dist} n={actual_n} seed={seed}: built diagram is NOT strictly valid \
                 (origins {origins_str}) — a real invalid-output defect"
            );
        }
        Err(e) => {
            // Out-of-envelope (e.g. vertex-budget overflow on pathological
            // density). Recorded, not a correctness failure.
            println!(
                "CASERESULT dist={dist} n={actual_n} seed={seed} param={param} \
                 result=err defects=- valid=- peak_mb={} origins=err:{e:?}",
                peak_rss_mb()
            );
        }
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
        let defects = out.report.unresolved_edge_pairs.len();
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

/// Single-build transient-peak probe: ONE build of `S2_CASE_N` uniform
/// points, reporting the construction high-water mark. Used to measure the
/// peak-vs-size curve in isolated processes.
#[test]
#[ignore]
fn diag_single_build_peak() {
    let n: usize = env_parse("S2_CASE_N", 1_000_000usize);
    let pts = random_sphere_points(n, 1);
    let out = compute_with_report(&pts, VoronoiConfig::default()).expect("compute");
    let defects = out.report.unresolved_edge_pairs.len();
    let valid = out.report.preferred_validation().is_strictly_valid();
    println!(
        "PEAKPROBE n={n} peak_rss={} MB defects={defects} valid={valid}",
        peak_rss_mb()
    );
}
