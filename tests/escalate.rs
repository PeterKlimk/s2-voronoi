//! Step-C vertical slice (feature `escalate_probe`): drive the exact local
//! rebuild over REAL fast-path defects and confirm it produces a consistent
//! local diagram where the fast path left an unpaired edge.
//!
//!   cargo test --release --features escalate_probe --test escalate -- --nocapture
#![cfg(feature = "escalate_probe")]

mod support;

use std::collections::HashMap;

use glam::Vec3;
use s2_voronoi::escalate_probe::{
    check_cell_internally_paired, gather_local, rebuild_cells, set_escalation_enabled, RebuiltCell,
};
use s2_voronoi::{compute_with_report, UnitVec3, VoronoiConfig};
use support::points::*;

fn to_vec3(points: &[UnitVec3]) -> Vec<Vec3> {
    points.iter().map(|p| Vec3::new(p.x, p.y, p.z)).collect()
}

/// For each fast-path defect edge (a,b), rebuild the local neighborhood as ONE
/// exact hull and check that a's and b's rebuilt cells AGREE on the a-b edge
/// (same shared-vertex triples) — i.e. the rebuild pairs what the fast path
/// could not. This proves the gather -> one-hull -> dual-read plumbing on real
/// defect data; the dual-consistency invariant guarantees agreement whenever
/// both seeds are interior to the gathered set.
#[test]
fn rebuild_resolves_mega_defects() {
    let points = mega_points(100_000, 0.8, 3);
    let pts3 = to_vec3(&points);

    let out = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    let defects: Vec<(u32, u32)> = out
        .report
        .unresolved_edge_pairs
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect();
    assert!(
        !defects.is_empty(),
        "expected mega defects to exercise the rebuild"
    );
    println!("mega 100k s3: {} defect edges", defects.len());

    // Sweep the gather size. Consistency (both cells agree) must hold at every
    // k — that is the invariant. Whether the edge EXISTS can move with k: a
    // defect whose edge appears only once the gather is large was a
    // local-set-adequacy (considered-neighbor / step-B) issue, not a spurious
    // fast-path edge.
    for k in [48usize, 96, 192] {
        let mut with_edge = 0usize;
        let mut bailed = 0usize;
        for &(a, b) in &defects {
            let local = gather_local(&pts3, &[a, b], k);
            let Some(cells) = rebuild_cells(&pts3, &local, &[a, b]) else {
                bailed += 1;
                continue;
            };
            let by_gen: HashMap<u32, &RebuiltCell> =
                cells.iter().map(|c| (c.generator, c)).collect();
            let (Some(ca), Some(cb)) = (by_gen.get(&a), by_gen.get(&b)) else {
                bailed += 1;
                continue;
            };
            let edge_from_a = ca.shared_edge_with(b);
            let edge_from_b = cb.shared_edge_with(a);
            assert_eq!(
                edge_from_a, edge_from_b,
                "k={k}: rebuilt cells disagree on edge ({a},{b}): \
                 {edge_from_a:?} vs {edge_from_b:?}"
            );
            if !edge_from_a.is_empty() {
                with_edge += 1;
            }
        }
        println!(
            "k={k:>3}: {with_edge}/{} defects rebuild to a consistent PAIRED edge \
             ({bailed} bailed)",
            defects.len()
        );
    }
}

/// Stronger than the per-edge check: rebuild each defect's whole neighborhood as
/// ONE hull and verify EVERY edge of BOTH defect cells pairs internally — i.e.
/// the defect cells are fully-valid local subdivisions, not just consistent on
/// the one a-b edge. `check_cell_internally_paired` panics on any interior
/// pairing disagreement, so a clean run is the proof.
#[test]
fn defect_cells_are_internally_valid_after_rebuild() {
    let points = mega_points(100_000, 0.8, 3);
    let pts3 = to_vec3(&points);
    let out = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    let defects: Vec<(u32, u32)> = out
        .report
        .unresolved_edge_pairs
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect();
    assert!(!defects.is_empty());

    let mut total_rim = 0usize;
    let mut total_edges = 0usize;
    for &(a, b) in &defects {
        // Gather the defect's neighborhood, then rebuild EVERY gathered cell so
        // a's and b's neighbors are present to pair against.
        let local = gather_local(&pts3, &[a, b], 96);
        let cells = rebuild_cells(&pts3, &local, &local).expect("hull");
        let by_gen: HashMap<u32, RebuiltCell> =
            cells.into_iter().map(|c| (c.generator, c)).collect();

        for &g in &[a, b] {
            let cell = &by_gen[&g];
            total_edges += cell.vertices.len();
            // Panics on any interior pairing disagreement; rim edges (neighbor
            // outside the gathered set) are counted, not failed.
            total_rim += check_cell_internally_paired(cell, &by_gen);
        }
    }
    println!(
        "{} defects: all defect-cell interior edges paired; {total_rim} rim edges \
         (neighbor outside gather) of {total_edges} total uncheckable",
        defects.len()
    );
}

/// Probe runner: build mega 100k at seed `S2_ESCALATE_SEED` (default 3) with
/// escalation enabled, so the env-gated probes inside `escalate_diagram`
/// (S2_ESCALATE_PROBE_*) fire on that seed. Asserts nothing.
#[test]
#[ignore = "probe driver; run with S2_ESCALATE_PROBE_*=1 and --ignored --nocapture"]
fn escalation_probe_runner() {
    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let dist = std::env::var("S2_ESCALATE_DIST").unwrap_or_else(|_| "mega".to_string());
    let points = match dist.as_str() {
        "uniform" => random_sphere_points(n, seed),
        "fib" => fibonacci_sphere_points(n, 0.0, seed),
        "clustered" => clustered_cap_points(n, 0.3, seed),
        "bimodal" => bimodal_density_points(n, 0.2, seed),
        _ => mega_points(n, 0.8, seed),
    };
    println!("dist={dist} seed={seed} n={n}");
    set_escalation_enabled(true);
    let out = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    println!(
        "seed={seed} n={n}: {}",
        if out.report.returned_validation.is_strictly_valid() {
            "VALID".to_string()
        } else {
            format!("{:?}", out.report.returned_validation.subdivision_issues())
        }
    );
}

/// A0 with an EXACT reference (delaunator, exact predicates) via stereographic
/// projection. Stashes the fast per-cell triples from a build, computes the true
/// spherical Delaunay, and classifies cells: which CHANGED (fast≠exact) and which
/// are DEFECTS (fast unpaired). Decides the clip-time-exact (A) cost/soundness:
/// changed = cells A must canonicalize; defects ⊆ changed is required; the
/// changed count vs defect-closure shows A (canonicalize-cap) vs B (surgical).
///   S2_ESCALATE_DIST=mega S2_ESCALATE_N=100000 S2_ESCALATE_SEED=3 \
///     cargo test --release --features escalate_probe --test escalate \
///     a0_exact_reference_delaunator -- --ignored --nocapture
#[test]
#[ignore = "A0 exact-reference probe; run individually with env + --ignored --nocapture"]
fn a0_exact_reference_delaunator() {
    use s2_voronoi::escalate_probe::take_a0_fast;
    use std::collections::BTreeSet;

    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let dist = std::env::var("S2_ESCALATE_DIST").unwrap_or_else(|_| "mega".to_string());
    let points = match dist.as_str() {
        "uniform" => random_sphere_points(n, seed),
        "fib" => fibonacci_sphere_points(n, 0.0, seed),
        "clustered" => clustered_cap_points(n, 0.3, seed),
        _ => mega_points(n, 0.8, seed),
    };

    // Trigger the stash inside escalate_diagram.
    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    let (pts, fast_triples) = take_a0_fast().expect("A0 stash");
    let m = pts.len();

    // Stereographic projection from the antipode of the centroid (so the dense
    // region maps compact and the projection pole sits in empty space).
    let mut c = Vec3::ZERO;
    for p in &pts {
        c += *p;
    }
    let pole = (-c).normalize();
    // basis perpendicular to pole
    let a = if pole.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (a - pole * a.dot(pole)).normalize();
    let e2 = pole.cross(e1);
    let proj: Vec<delaunator::Point> = pts
        .iter()
        .map(|&p| {
            let denom = (1.0 - p.dot(pole)).max(1e-12);
            delaunator::Point {
                x: (p.dot(e1) / denom) as f64,
                y: (p.dot(e2) / denom) as f64,
            }
        })
        .collect();

    let tri = delaunator::triangulate(&proj);
    // exact per-generator triple-set + mark hull-region (incomplete-fan) gens
    let mut exact: Vec<BTreeSet<[u32; 3]>> = vec![BTreeSet::new(); m];
    for t in tri.triangles.chunks_exact(3) {
        let mut k = [t[0] as u32, t[1] as u32, t[2] as u32];
        k.sort_unstable();
        for &g in t {
            exact[g].insert(k);
        }
    }
    let hull_set: BTreeSet<usize> = tri.hull.iter().copied().collect();
    // hull-region = on the hull or sharing a triangle with a hull vertex
    let mut hull_region = vec![false; m];
    for &h in &hull_set {
        hull_region[h] = true;
    }
    for t in tri.triangles.chunks_exact(3) {
        if t.iter().any(|&g| hull_set.contains(&g)) {
            for &g in t {
                hull_region[g] = true;
            }
        }
    }

    // fast defects: triple-level unpaired scan over fast_triples
    let mut dir: std::collections::HashMap<([u32; 3], [u32; 3]), u32> =
        std::collections::HashMap::new();
    for fan in &fast_triples {
        let f = fan.len();
        if f < 3 {
            continue;
        }
        for i in 0..f {
            *dir.entry((fan[i], fan[(i + 1) % f])).or_default() += 1;
        }
    }
    let mut defect = vec![false; m];
    // mark a generator defective if any of its directed edges is unpaired
    for (g, fan) in fast_triples.iter().enumerate() {
        let f = fan.len();
        if f < 3 {
            continue;
        }
        for i in 0..f {
            let (a, b) = (fan[i], fan[(i + 1) % f]);
            let fwd = dir.get(&(a, b)).copied().unwrap_or(0);
            let rev = dir.get(&(b, a)).copied().unwrap_or(0);
            if fwd != 1 || rev != 1 {
                defect[g] = true;
            }
        }
    }

    let mut changed = 0usize;
    let mut changed_interior = 0usize;
    let mut defect_cells = 0usize;
    let mut defect_not_changed = 0usize;
    for g in 0..m {
        let fast_set: BTreeSet<[u32; 3]> = fast_triples[g].iter().copied().collect();
        let ch = fast_set != exact[g];
        if ch {
            changed += 1;
            if !hull_region[g] {
                changed_interior += 1;
            }
        }
        if defect[g] {
            defect_cells += 1;
            if !ch {
                defect_not_changed += 1;
            }
        }
    }
    let hull_region_count = hull_region.iter().filter(|&&h| h).count();
    println!(
        "A0-exact dist={dist} n={m}: hull_region={hull_region_count} | \
         changed={changed} (interior={changed_interior}, {:.2}%) | \
         defect_cells={defect_cells} defect_not_changed={defect_not_changed}",
        100.0 * changed_interior as f64 / (m - hull_region_count).max(1) as f64,
    );
}

/// Build the exact spherical Delaunay (delaunator, exact predicates) for `pts`
/// via stereographic projection; return per-generator exact triple-sets and a
/// hull-region mask (cells with incomplete fans at the projection boundary).
#[cfg(test)]
fn exact_triples_delaunator(
    pts: &[Vec3],
) -> (Vec<std::collections::BTreeSet<[u32; 3]>>, Vec<bool>) {
    use std::collections::BTreeSet;
    let m = pts.len();
    let mut c = Vec3::ZERO;
    for p in pts {
        c += *p;
    }
    let pole = (-c).normalize();
    let a = if pole.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (a - pole * a.dot(pole)).normalize();
    let e2 = pole.cross(e1);
    let proj: Vec<delaunator::Point> = pts
        .iter()
        .map(|&p| {
            let denom = (1.0 - p.dot(pole)).max(1e-12);
            delaunator::Point {
                x: (p.dot(e1) / denom) as f64,
                y: (p.dot(e2) / denom) as f64,
            }
        })
        .collect();
    let tri = delaunator::triangulate(&proj);
    let mut exact: Vec<BTreeSet<[u32; 3]>> = vec![BTreeSet::new(); m];
    for t in tri.triangles.chunks_exact(3) {
        let mut k = [t[0] as u32, t[1] as u32, t[2] as u32];
        k.sort_unstable();
        for &g in t {
            exact[g].insert(k);
        }
    }
    let hull_set: BTreeSet<usize> = tri.hull.iter().copied().collect();
    let mut hull_region = vec![false; m];
    for &h in &hull_set {
        hull_region[h] = true;
    }
    for t in tri.triangles.chunks_exact(3) {
        if t.iter().any(|&g| hull_set.contains(&g)) {
            for &g in t {
                hull_region[g] = true;
            }
        }
    }
    (exact, hull_region)
}

/// Local exact Delaunay over a generator subset (gather), same metric as the
/// global delaunator reference: stereographic-project the subset, triangulate,
/// return each seed generator's triple-set in GLOBAL indices. A seed's fan is
/// complete iff the subset contains all its true neighbors (generous gather).
#[cfg(test)]
fn local_delaunator_cell(
    pts: &[Vec3],
    subset: &[u32],
    seeds: &[u32],
) -> std::collections::HashMap<u32, std::collections::BTreeSet<[u32; 3]>> {
    use std::collections::{BTreeSet, HashMap};
    // project the subset (pole = antipode of subset centroid)
    let mut c = Vec3::ZERO;
    for &g in subset {
        c += pts[g as usize];
    }
    let pole = (-c).normalize();
    let a = if pole.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (a - pole * a.dot(pole)).normalize();
    let e2 = pole.cross(e1);
    let proj: Vec<delaunator::Point> = subset
        .iter()
        .map(|&g| {
            let p = pts[g as usize];
            let denom = (1.0 - p.dot(pole)).max(1e-12);
            delaunator::Point {
                x: (p.dot(e1) / denom) as f64,
                y: (p.dot(e2) / denom) as f64,
            }
        })
        .collect();
    let tri = delaunator::triangulate(&proj);
    let seedset: BTreeSet<u32> = seeds.iter().copied().collect();
    let mut out: HashMap<u32, BTreeSet<[u32; 3]>> =
        seeds.iter().map(|&g| (g, BTreeSet::new())).collect();
    for t in tri.triangles.chunks_exact(3) {
        let mut k = [subset[t[0]], subset[t[1]], subset[t[2]]];
        k.sort_unstable();
        for &li in t {
            let g = subset[li];
            if seedset.contains(&g) {
                out.get_mut(&g).unwrap().insert(k);
            }
        }
    }
    out
}

/// Does a LOCAL delaunator (over a generous gather) reproduce the GLOBAL
/// delaunator on the repair cells? If yes, "local stereographic + delaunator"
/// is the consistent local oracle the repair needs (matching the trusted global
/// reference), unlike `local_hull` (wrong metric → never matches).
#[test]
#[ignore = "local-delaunator-vs-global; run with env + --ignored --nocapture"]
fn local_delaunator_vs_global() {
    use s2_voronoi::escalate_probe::{gather_local, take_a0_fast};
    use std::collections::BTreeSet;

    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let points = mega_points(n, 0.8, seed);

    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    let (pts, fast_triples) = take_a0_fast().expect("A0 stash");

    let (exact, hull_region) = exact_triples_delaunator(&pts);
    let targets: Vec<u32> = (0..pts.len() as u32)
        .filter(|&g| {
            !hull_region[g as usize] && {
                let fs: BTreeSet<[u32; 3]> = fast_triples[g as usize].iter().copied().collect();
                fs != exact[g as usize]
            }
        })
        .collect();
    println!(
        "local-delaunator: {} repair-target cells, seed={seed}",
        targets.len()
    );
    for k in [64usize, 128, 256, 512] {
        // gather a generous neighborhood around ALL targets (one local patch)
        let subset = gather_local(&pts, &targets, k);
        let cells = local_delaunator_cell(&pts, &subset, &targets);
        let matched = targets
            .iter()
            .filter(|&&g| {
                cells
                    .get(&g)
                    .map(|s| s == &exact[g as usize])
                    .unwrap_or(false)
            })
            .count();
        println!(
            "  k={k:>4} (subset={}): local==global delaunator on {matched}/{} targets",
            subset.len(),
            targets.len()
        );
    }
}

/// Tests the projection theory: the fast clipper (gnomonic chart) and delaunator
/// (stereographic chart) both decide near-cocircular ties on PROJECTED coords,
/// while `local_hull` uses exact `orient3d` on RAW 3D coords. Prediction: the two
/// projected methods agree closely (small fast Δ delaunator) and both differ from
/// the raw method (large fast Δ local_hull ≈ delaunator Δ local_hull). Computes
/// all three on a small mega input and reports pairwise changed-cell counts over
/// interior (non-projection-rim) cells.
#[test]
#[ignore = "projection-theory 3-way diagnostic; small n; env + --ignored --nocapture"]
fn projection_theory_3way() {
    use s2_voronoi::escalate_probe::{rebuild_cells, take_a0_fast};
    use std::collections::BTreeSet;

    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12_000);
    let dist = std::env::var("S2_ESCALATE_DIST").unwrap_or_else(|_| "mega".to_string());
    let points = match dist.as_str() {
        "uniform" => random_sphere_points(n, seed),
        "clustered" => clustered_cap_points(n, 0.3, seed),
        _ => mega_points(n, 0.8, seed),
    };

    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    let (pts, fast_triples) = take_a0_fast().expect("A0 stash");
    let m = pts.len();

    // delaunator (stereographic, projected) and its hull-region (incomplete fans)
    let (delan, hull_region) = exact_triples_delaunator(&pts);

    // local_hull GLOBAL (raw 3D orient3d): ONE hull over all points, all cells.
    let all: Vec<u32> = (0..m as u32).collect();
    let lh_cells = rebuild_cells(&pts, &all, &all).expect("global hull");
    let mut lh: Vec<BTreeSet<[u32; 3]>> = vec![BTreeSet::new(); m];
    for c in lh_cells {
        lh[c.generator as usize] = c.vertices.into_iter().collect();
    }

    let fast: Vec<BTreeSet<[u32; 3]>> = fast_triples
        .iter()
        .map(|f| f.iter().copied().collect())
        .collect();

    // Compare over interior cells (exclude projection rim AND cells local_hull
    // couldn't build / are empty, i.e. back-face artifacts on the cluster edge).
    let (mut fast_delan, mut fast_lh, mut delan_lh, mut interior) = (0, 0, 0, 0);
    for g in 0..m {
        if hull_region[g] || lh[g].is_empty() || fast[g].is_empty() || delan[g].is_empty() {
            continue;
        }
        interior += 1;
        if fast[g] != delan[g] {
            fast_delan += 1;
        }
        if fast[g] != lh[g] {
            fast_lh += 1;
        }
        if delan[g] != lh[g] {
            delan_lh += 1;
        }
    }
    println!(
        "PROJ-THEORY dist={dist} n={m} seed={seed}: interior_cells={interior} | \
         fastΔdelaunator={fast_delan} fastΔlocal_hull={fast_lh} delaunatorΔlocal_hull={delan_lh}"
    );
}

/// The actually-right oracle test: detect defects → replace the cluster cells
/// from ONE `local_hull` (3D exact `orient3d`, projection-independent) over the
/// cluster's gather → expand by any still-inconsistent generator → repeat. The
/// repair is valid iff every triple is referenced by exactly its 3 generators.
/// Measures repair-set size + convergence with the dependency-free 3D oracle
/// (the real target, not "match delaunator").
#[test]
#[ignore = "detect+fix+expand with one-local_hull oracle; env + --ignored --nocapture"]
fn detect_fix_expand_localhull() {
    use s2_voronoi::escalate_probe::{gather_local, rebuild_cells, take_a0_fast};
    use std::collections::{BTreeSet, HashMap};

    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let k: usize = std::env::var("S2_ESCALATE_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let dist = std::env::var("S2_ESCALATE_DIST").unwrap_or_else(|_| "mega".to_string());
    let points = match dist.as_str() {
        "uniform" => random_sphere_points(n, seed),
        _ => mega_points(n, 0.8, seed),
    };

    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    let (pts, fast_triples) = take_a0_fast().expect("A0 stash");
    let m = pts.len();
    let fast_set: Vec<BTreeSet<[u32; 3]>> = fast_triples
        .iter()
        .map(|f| f.iter().copied().collect())
        .collect();

    // Inconsistent triples (referenced by != its 3 generators) → implicated gens,
    // given a repair set R rebuilt from ONE local_hull over gather(R, k).
    let analyze = |r: &BTreeSet<u32>| -> (BTreeSet<u32>, usize) {
        let rv: Vec<u32> = r.iter().copied().collect();
        let oracle: HashMap<u32, BTreeSet<[u32; 3]>> = if rv.is_empty() {
            HashMap::new()
        } else {
            let gather = gather_local(&pts, &rv, k);
            rebuild_cells(&pts, &gather, &rv)
                .unwrap_or_default()
                .into_iter()
                .map(|c| (c.generator, c.vertices.into_iter().collect()))
                .collect()
        };
        let mut refs: HashMap<[u32; 3], Vec<u32>> = HashMap::new();
        for g in 0..m as u32 {
            let s = oracle.get(&g).unwrap_or(&fast_set[g as usize]);
            for &t in s {
                refs.entry(t).or_default().push(g);
            }
        }
        let mut bad = BTreeSet::new();
        let mut nbad = 0usize;
        for (t, gens) in &refs {
            let expected: BTreeSet<u32> = t.iter().copied().collect();
            let got: BTreeSet<u32> = gens.iter().copied().collect();
            if got != expected {
                nbad += 1;
                for &g in t.iter().chain(gens.iter()) {
                    bad.insert(g);
                }
            }
        }
        (bad, nbad)
    };

    let (seed_bad, base_nbad) = analyze(&BTreeSet::new());
    let mut r = seed_bad.clone();
    let mut rounds = 0;
    let mut history = vec![r.len()];
    let mut final_bad = base_nbad;
    for _ in 0..60 {
        rounds += 1;
        let (bad, nbad) = analyze(&r);
        final_bad = nbad;
        let newbad: Vec<u32> = bad.difference(&r).copied().collect();
        if newbad.is_empty() {
            break;
        }
        for g in newbad {
            r.insert(g);
        }
        history.push(r.len());
    }
    println!(
        "EXPAND-LH dist={dist} n={m} seed={seed} k={k}: base_bad_triples={base_nbad} \
         repair_set={} rounds={rounds} final_bad_triples={final_bad} {} | traj={:?}",
        r.len(),
        if final_bad == 0 { "CLEAN" } else { "RESIDUAL" },
        history,
    );
}

/// First goal for the repair: a LOCAL exact oracle whose cells match the global
/// exact reference (delaunator). This measures how well the current `local_hull`
/// (via `rebuild_cells` over a k-NN gather) matches delaunator on the cells that
/// matter — the CHANGED cells (where fast ≠ exact, i.e. the repair targets) —
/// across gather sizes. Mismatches diagnose what the oracle needs (completeness
/// vs ties). A perfect local oracle would match delaunator on every changed cell.
#[test]
#[ignore = "local-oracle-vs-delaunator diagnostic; run with env + --ignored --nocapture"]
fn local_oracle_vs_delaunator() {
    use s2_voronoi::escalate_probe::{rebuild_cells, take_a0_fast};
    use std::collections::BTreeSet;

    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let points = mega_points(n, 0.8, seed);

    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    let (pts, fast_triples) = take_a0_fast().expect("A0 stash");

    let (exact, hull_region) = exact_triples_delaunator(&pts);
    // repair targets = changed cells (fast ≠ exact), excluding the projection rim.
    let targets: Vec<u32> = (0..pts.len() as u32)
        .filter(|&g| {
            !hull_region[g as usize] && {
                let fs: BTreeSet<[u32; 3]> = fast_triples[g as usize].iter().copied().collect();
                fs != exact[g as usize]
            }
        })
        .collect();
    let pts3: Vec<glam::Vec3> = pts.clone();

    println!(
        "local-oracle: {} changed (repair-target) cells, seed={seed} n={}",
        targets.len(),
        pts.len()
    );
    for k in [48usize, 96, 192, 384, 768] {
        let mut matched = 0usize;
        for &g in &targets {
            let local = s2_voronoi::escalate_probe::gather_local(&pts3, &[g], k);
            let Some(cells) = rebuild_cells(&pts3, &local, &[g]) else {
                continue;
            };
            let Some(rc) = cells.into_iter().next() else {
                continue;
            };
            let lh: BTreeSet<[u32; 3]> = rc.vertices.into_iter().collect();
            if lh == exact[g as usize] {
                matched += 1;
            }
        }
        println!(
            "  k={k:>4}: local_hull matches delaunator on {matched}/{} changed cells",
            targets.len()
        );
    }
}

/// THE demonstration: with a CONSISTENT exact oracle, "detect bad edges → fix
/// both incident cells → expand until agreement" terminates at the small
/// changed-closure (does NOT explode to the cap). Seeds the repair set from the
/// fast diagram's defects, replaces those cells with their exact (delaunator)
/// cells, and grows by any generator still inconsistent — measuring the final
/// repair-set size (should be ~tens) and residual (should be 0).
#[test]
#[ignore = "detect+fix+expand demonstration; run individually with env + --ignored --nocapture"]
fn detect_fix_expand_delaunator() {
    use s2_voronoi::escalate_probe::take_a0_fast;
    use std::collections::{BTreeSet, HashMap};

    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let dist = std::env::var("S2_ESCALATE_DIST").unwrap_or_else(|_| "mega".to_string());
    let points = match dist.as_str() {
        "uniform" => random_sphere_points(n, seed),
        "clustered" => clustered_cap_points(n, 0.3, seed),
        _ => mega_points(n, 0.8, seed),
    };

    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    let (pts, fast_triples) = take_a0_fast().expect("A0 stash");
    let m = pts.len();

    let (exact, hull_region) = exact_triples_delaunator(&pts);
    let fast_set: Vec<BTreeSet<[u32; 3]>> = fast_triples
        .iter()
        .map(|f| f.iter().copied().collect())
        .collect();

    // A triple {a,b,c} is consistent iff exactly cells {a,b,c} reference it.
    // Given a repair set R (use exact for those, fast otherwise), find the
    // generators implicated by any inconsistent (non-hull) triple.
    let implicated = |r: &BTreeSet<u32>| -> BTreeSet<u32> {
        let mut refs: HashMap<[u32; 3], Vec<u32>> = HashMap::new();
        for g in 0..m {
            let s = if r.contains(&(g as u32)) {
                &exact[g]
            } else {
                &fast_set[g]
            };
            for &t in s {
                refs.entry(t).or_default().push(g as u32);
            }
        }
        let mut bad = BTreeSet::new();
        for (t, gens) in &refs {
            if t.iter().any(|&g| hull_region[g as usize]) {
                continue; // incomplete fan at the projection boundary
            }
            let expected: BTreeSet<u32> = t.iter().copied().collect();
            let got: BTreeSet<u32> = gens.iter().copied().collect();
            if got != expected {
                for &g in t.iter().chain(gens.iter()) {
                    bad.insert(g);
                }
            }
        }
        bad
    };

    // Seed R from the fast diagram's inconsistencies (the "detected bad edges").
    let mut r: BTreeSet<u32> = implicated(&BTreeSet::new());
    let seed_size = r.len();
    let mut rounds = 0;
    let mut converged = false;
    let mut history: Vec<usize> = vec![r.len()];
    for _ in 0..50 {
        rounds += 1;
        let bad = implicated(&r);
        let newbad: Vec<u32> = bad.difference(&r).copied().collect();
        if newbad.is_empty() {
            converged = bad.is_empty();
            break;
        }
        for g in newbad {
            r.insert(g);
        }
        history.push(r.len());
    }
    println!(
        "EXPAND dist={dist} n={m} seed={seed}: detected_seed={seed_size} -> repair_set={} \
         in {rounds} rounds {} | trajectory={:?}",
        r.len(),
        if converged {
            "CONVERGED (valid, bounded — no explosion)"
        } else {
            "did not converge"
        },
        history,
    );
}

/// Generalization sweep (report-only): does consensus repair turn every mega
/// config strictly valid, and is it a no-op (validity preserved) on clean input?
#[test]
#[ignore = "report-only sweep; run with --ignored --nocapture"]
fn escalation_generalization_sweep() {
    let run = |label: &str, pts: &[UnitVec3]| {
        set_escalation_enabled(false);
        let before = compute_with_report(pts, VoronoiConfig::default()).expect("build");
        set_escalation_enabled(true);
        let after = compute_with_report(pts, VoronoiConfig::default()).expect("build");
        set_escalation_enabled(false);
        println!(
            "{label}: before={} after={}",
            if before.report.returned_validation.is_strictly_valid() {
                "VALID".to_string()
            } else {
                format!(
                    "{:?}",
                    before.report.returned_validation.subdivision_issues()
                )
            },
            if after.report.returned_validation.is_strictly_valid() {
                "VALID".to_string()
            } else {
                format!(
                    "{:?}",
                    after.report.returned_validation.subdivision_issues()
                )
            },
        );
    };
    for seed in 1..=5u64 {
        run(
            &format!("mega 100k s{seed}"),
            &mega_points(100_000, 0.8, seed),
        );
    }
    run("mega 300k s3", &mega_points(300_000, 0.8, 3));
    run("uniform 100k s7 (clean)", &random_sphere_points(100_000, 7));
}

/// Narrow WIN, but NOT general — mega 100k s3's defect is a single CONTAINED
/// malformed cell (cell 11548: neighbor 17853 in 4 boundary vertices ⇒ an
/// invalid non-convex polygon; the only malformed cell in 99996), and all 5 of
/// its unpaired edges contain it. Consensus repair (rebuild the malformed cell
/// from its valid neighbors' attributed triples, adopting their diagonals) makes
/// THIS input strictly valid with zero cascade.
///
/// It does NOT generalize (see `escalation_generalization_sweep`): the general
/// mega defect is cross-cell decision divergence between individually-VALID cells
/// (no malformed cell), which consensus does not address and can even regress
/// (mega 300k s3: 51→52). So this is kept `#[ignore]` — a real but narrow result,
/// not a green gate. The full story (hull splice diverges; reclip is source-
/// compatible but f32 can't fix; surgical/component rebuild cascade; consensus is
/// a contained-malformation-only fix) is in docs/escalation-build-state-2026-06.md
/// and the memory note `route-a-splice-diverges`.
///
/// `set_escalation_enabled` is a process-global toggle only THIS test flips.
#[test]
#[ignore = "narrow: consensus repair fixes the contained-malformation case (s3) only; does not generalize"]
fn escalation_splice_makes_mega_strictly_valid() {
    let points = mega_points(100_000, 0.8, 3);

    // Baseline: escalation off — the fast path leaves a real defect residual.
    set_escalation_enabled(false);
    let before = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    let before_valid = before.report.returned_validation.is_strictly_valid();
    println!(
        "escalation OFF: {} | {}",
        if before_valid { "VALID" } else { "INVALID" },
        before.report.returned_validation
    );
    assert!(
        !before_valid,
        "expected mega 100k s3 to be invalid without escalation (no defect to fix?)"
    );

    // Escalation on — the residual is resolved by exact local rebuild + splice.
    set_escalation_enabled(true);
    let after = compute_with_report(&points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    let after_report = &after.report.returned_validation;
    println!("escalation ON:  {after_report}");
    assert!(
        after_report.is_strictly_valid(),
        "escalation did not produce a strictly valid diagram: {:?}",
        after_report.subdivision_issues()
    );
}
