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

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{
    compute_plane_periodic_with_report, compute_plane_with_report, compute_with_report, PlanePoint,
    PlaneRect, UnitVec3, UnresolvedEdgeOrigin, VoronoiConfig,
};
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
        // `param` is the cap fraction; the only distribution dense enough to
        // drive the contested near-cocircular regime (which now errors loudly
        // per the valid-or-error contract).
        "mega" => mega_points(n, param, seed),
        // Clean structured quad grid: O(n) high-degree vertices at normal
        // density — the high-degeneracy/reconcile-load regime.
        "grid" => cubed_sphere_points(n, seed),
        other => panic!("unknown S2_CASE_DIST '{other}'"),
    }
}

/// One campaign case, parameterized entirely by environment so the driver
/// script can fork a fresh process per case:
///   S2_CASE_DIST  (uniform|clustered|bimodal|cube|cocircular|fibonacci|mega)
///   S2_CASE_N     point count (group count for cocircular)
///   S2_CASE_SEED  rng seed
///   S2_CASE_PARAM shape knob (f32; default 0.3 — for mega it is the cap fraction)
///
/// Emits a single machine-parseable `CASERESULT` line. A build that returns an
/// error is recorded as `result=err` (documented out-of-envelope behavior)
/// WITHOUT failing the test. The contract checked is the silent-invalid guard:
/// a returned diagram with NO surviving `PostRepairUnpaired` residual must be
/// strictly valid — otherwise the test fails (a real, newly found
/// silent-invalid bug). A diagram that left a residual is out-of-envelope on
/// this report API (the hot `compute` path loud-fails on it) and is recorded,
/// not failed.
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
            // Residuals that survived to the RETURNED diagram. `compute` (the
            // hot path) loud-fails on these; `compute_with_report` returns them
            // as diagnostics instead. Their absence is the silent-invalid
            // contract: no surviving residual MUST mean a strictly-valid graph.
            let post_repair = out.report.post_repair_unpaired_edges.len();
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
                 result=ok defects={defects} post_repair={post_repair} valid={valid} \
                 peak_mb={} origins={origins_str}",
                peak_rss_mb()
            );
            // THE contract: a returned diagram with NO surviving residual must be
            // strictly valid. A diagram that left a `PostRepairUnpaired` residual
            // is out-of-envelope on the report API (the hot `compute` path
            // loud-fails on it) — recorded, not a fuzzer failure.
            assert!(
                valid || post_repair > 0,
                "{dist} n={actual_n} seed={seed}: returned diagram is NOT strictly valid \
                 yet left NO post-repair residual (origins {origins_str}) — a SILENT \
                 invalid-output defect"
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

// ---------------------------------------------------------------------------
// Plane campaign
//
// Rule-agnostic: drives `compute_plane_with_report` (or the periodic variant)
// and records residuals + strict validity, exactly like the sphere case. The
// clip rule in force is whatever the binary was COMPILED with
// (`PLANE_CLIP_EPS_INSIDE`): build with the production bias for a regression
// sweep, or flip the constant to 0.0 and rebuild for the strict-plane
// campaign (the gate before flipping that default in production — it moves
// every plane fingerprint). The case does not touch the constant itself, so
// the same committed harness serves both rules; the driver script prints
// which rule is active.
// ---------------------------------------------------------------------------

fn plane_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Uniform points in `rect`.
fn plane_uniform(rect: PlaneRect, n: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut r = plane_rng(seed);
    (0..n)
        .map(|_| {
            [
                r.gen_range(rect.min.x..rect.max.x),
                r.gen_range(rect.min.y..rect.max.y),
            ]
        })
        .collect()
}

/// A dense tight cluster (jitter `param` about the centre) plus two far
/// corner anchors — the sliver-prone geometry that historically broke the
/// strict plane rule (the 402-cell torture class). Stresses near-coincident
/// generators and zero-length-edge attribution.
fn plane_clustered(n: usize, seed: u64, param: f32) -> Vec<[f32; 2]> {
    let mut r = plane_rng(seed);
    let j = if param > 0.0 { param } else { 0.01 };
    let mut pts: Vec<[f32; 2]> = (0..n.saturating_sub(2))
        .map(|_| {
            [
                (0.5 + r.gen_range(-j..j)).clamp(0.0, 1.0),
                (0.5 + r.gen_range(-j..j)).clamp(0.0, 1.0),
            ]
        })
        .collect();
    pts.push([0.01, 0.01]);
    pts.push([0.99, 0.99]);
    pts
}

/// A square lattice: every interior vertex is an exact 4-cocircular tie.
/// `n` is rounded down to the nearest square. The hardest exact-tie case
/// for any keep rule.
fn plane_lattice(n: usize) -> Vec<[f32; 2]> {
    let side = (n as f64).sqrt() as usize;
    let side = side.max(1);
    let mut pts = Vec::with_capacity(side * side);
    for i in 0..side {
        for j in 0..side {
            pts.push([
                (i as f32 + 0.5) / side as f32,
                (j as f32 + 0.5) / side as f32,
            ]);
        }
    }
    pts
}

/// Near-cocircular rings: `n` groups of 4 points each placed slightly off a
/// common circle (perturbation `param`). The planar parity-defect class.
fn plane_cocircular(n: usize, seed: u64, param: f32) -> Vec<[f32; 2]> {
    let mut r = plane_rng(seed);
    let eps = if param > 0.0 { param } else { 1e-4 };
    let mut pts = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let cx = r.gen_range(0.1f32..0.9);
        let cy = r.gen_range(0.1f32..0.9);
        let rad = r.gen_range(0.005f32..0.03);
        for k in 0..4 {
            let a = std::f32::consts::FRAC_PI_2 * k as f32 + r.gen_range(-eps..eps);
            let rr = rad + r.gen_range(-eps..eps) * rad;
            pts.push([
                (cx + rr * a.cos()).clamp(0.0, 1.0),
                (cy + rr * a.sin()).clamp(0.0, 1.0),
            ]);
        }
    }
    pts
}

/// One plane campaign case, parameterized by environment:
///   S2_PLANE_DIST  (uniform|clustered|lattice|cocircular|periodic)
///   S2_CASE_N      point count (group count for cocircular)
///   S2_CASE_SEED   rng seed
///   S2_CASE_PARAM  shape knob (jitter/perturbation; f32, default 0.01)
///   S2_PLANE_RECT  optional "minx,miny,maxx,maxy" (default unit square)
///
/// Emits a single machine-parseable `PLANECASE` line. As with the sphere
/// case: an error is recorded `result=err` without failing; a successful but
/// not-strictly-valid build fails the test (a real invalid-output defect).
#[test]
#[ignore]
fn plane_campaign_case() {
    let dist = env_str("S2_PLANE_DIST", "uniform");
    let n: usize = env_parse("S2_CASE_N", 50_000usize);
    let seed: u64 = env_parse("S2_CASE_SEED", 1u64);
    let param: f32 = env_parse("S2_CASE_PARAM", 0.01f32);

    let rect = match std::env::var("S2_PLANE_RECT") {
        Ok(s) => {
            let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            assert_eq!(v.len(), 4, "S2_PLANE_RECT must be minx,miny,maxx,maxy");
            PlaneRect::new(PlanePoint::new(v[0], v[1]), PlanePoint::new(v[2], v[3]))
        }
        Err(_) => PlaneRect::unit(),
    };

    let periodic = dist == "periodic";
    let points = match dist.as_str() {
        "uniform" | "periodic" => plane_uniform(rect, n, seed),
        "clustered" => plane_clustered(n, seed, param),
        "lattice" => plane_lattice(n),
        "cocircular" => plane_cocircular(n, seed, param),
        other => panic!("unknown S2_PLANE_DIST '{other}'"),
    };
    let actual_n = points.len();

    let built = if periodic {
        compute_plane_periodic_with_report(&points, rect)
    } else {
        compute_plane_with_report(&points, rect)
    };

    match built {
        Ok(out) => {
            let valid = out.report.validation.is_strictly_valid();
            let defects = out.report.unresolved_edge_pairs.len();
            println!(
                "PLANECASE dist={dist} n={actual_n} seed={seed} param={param} \
                 result=ok defects={defects} valid={valid} peak_mb={}",
                peak_rss_mb()
            );
            assert!(
                valid,
                "plane {dist} n={actual_n} seed={seed}: built diagram is NOT strictly \
                 valid ({} residual pair(s)) — a real invalid-output defect",
                defects
            );
        }
        Err(e) => {
            println!(
                "PLANECASE dist={dist} n={actual_n} seed={seed} param={param} \
                 result=err defects=- valid=- peak_mb={} note={e:?}",
                peak_rss_mb()
            );
        }
    }
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
