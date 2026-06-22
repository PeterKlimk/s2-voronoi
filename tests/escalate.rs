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
use s2_voronoi::{compute_with_report, RepairMode, UnitVec3, VoronoiConfig};
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

fn probe_points_from_env(default_n: usize) -> (String, u64, Vec<UnitVec3>) {
    let seed: u64 = std::env::var("S2_ESCALATE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n: usize = std::env::var("S2_ESCALATE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_n);
    let dist = std::env::var("S2_ESCALATE_DIST").unwrap_or_else(|_| "uniform".to_string());
    let points = match dist.as_str() {
        "fib" => fibonacci_sphere_points(n, 0.0, seed),
        "clustered" => clustered_cap_points(n, 0.3, seed),
        "bimodal" => bimodal_density_points(n, 0.2, seed),
        "mega" => mega_points(n, 0.8, seed),
        _ => random_sphere_points(n, seed),
    };
    (dist, seed, points)
}

fn stash_fast_triples(points: &[UnitVec3]) -> (Vec<Vec3>, Vec<Vec<[u32; 3]>>) {
    use s2_voronoi::escalate_probe::take_a0_fast;

    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    take_a0_fast().expect("A0 stash")
}

fn stash_fast_triples_with_audit(
    points: &[UnitVec3],
) -> (
    Vec<Vec3>,
    Vec<Vec<[u32; 3]>>,
    Vec<s2_voronoi::escalate_probe::CellAudit>,
) {
    use s2_voronoi::escalate_probe::{reset_proactive_audit, take_a0_fast, take_proactive_audit};

    reset_proactive_audit();
    std::env::set_var("S2_PROACTIVE_AUDIT", "1");
    std::env::set_var("S2_ESCALATE_PROBE_A0", "1");
    set_escalation_enabled(true);
    let _ = compute_with_report(points, VoronoiConfig::default()).expect("build");
    set_escalation_enabled(false);
    std::env::remove_var("S2_ESCALATE_PROBE_A0");
    std::env::remove_var("S2_PROACTIVE_AUDIT");
    let fast = take_a0_fast().expect("A0 stash");
    let audit = take_proactive_audit();
    (fast.0, fast.1, audit)
}

fn exact_triples_norm3d(pts: &[Vec3]) -> (Vec<std::collections::BTreeSet<[u32; 3]>>, Vec<bool>) {
    use s2_voronoi::escalate_probe::rebuild_cells;
    use std::collections::BTreeSet;

    let m = pts.len();
    let all: Vec<u32> = (0..m as u32).collect();
    let mut exact: Vec<BTreeSet<[u32; 3]>> = vec![BTreeSet::new(); m];
    let Some(cells) = rebuild_cells(pts, &all, &all) else {
        return (exact, vec![false; m]);
    };
    let mut complete = vec![false; m];
    for c in cells {
        complete[c.generator as usize] = !c.vertices.is_empty();
        exact[c.generator as usize] = c.vertices.into_iter().collect();
    }
    (exact, complete)
}

fn exact_triples_cgal_hull3(
    pts: &[Vec3],
) -> (Vec<std::collections::BTreeSet<[u32; 3]>>, Vec<bool>, String) {
    use std::collections::BTreeSet;
    use std::io::Write;
    use std::process::{Command, Stdio};

    let bin = std::env::var("S2_CGAL_HULL3_BIN")
        .expect("set S2_CGAL_HULL3_BIN to scripts/cgal_hull3.cpp compiled binary");
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn CGAL hull probe");
    {
        let stdin = child.stdin.as_mut().expect("CGAL hull probe stdin");
        for (i, p) in pts.iter().enumerate() {
            let q = glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64).normalize();
            writeln!(stdin, "{} {:.17e} {:.17e} {:.17e}", i, q.x, q.y, q.z)
                .expect("write CGAL hull probe input");
        }
    }
    let output = child.wait_with_output().expect("wait for CGAL hull probe");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "CGAL hull probe failed with status {:?}: {stderr}",
        output.status.code()
    );

    let mut exact: Vec<BTreeSet<[u32; 3]>> = vec![BTreeSet::new(); pts.len()];
    let mut complete = vec![false; pts.len()];
    let stdout = String::from_utf8_lossy(&output.stdout);
    for (line_no, line) in stdout.lines().enumerate() {
        let ids: Vec<u32> = line
            .split_whitespace()
            .map(|s| s.parse::<u32>())
            .collect::<Result<_, _>>()
            .unwrap_or_else(|err| panic!("parse CGAL hull output line {}: {err}", line_no + 1));
        assert_eq!(
            ids.len(),
            3,
            "CGAL hull output line {} was not a triangle: {line}",
            line_no + 1
        );
        let tri = [ids[0], ids[1], ids[2]];
        for &g in &tri {
            exact[g as usize].insert(tri);
            complete[g as usize] = true;
        }
    }
    (exact, complete, stderr)
}

fn defect_cells_from_triples(fast_triples: &[Vec<[u32; 3]>]) -> Vec<bool> {
    let mut dir: std::collections::HashMap<([u32; 3], [u32; 3]), u32> =
        std::collections::HashMap::new();
    for fan in fast_triples {
        let f = fan.len();
        if f < 3 {
            continue;
        }
        for i in 0..f {
            *dir.entry((fan[i], fan[(i + 1) % f])).or_default() += 1;
        }
    }

    let mut defect = vec![false; fast_triples.len()];
    for (g, fan) in fast_triples.iter().enumerate() {
        let f = fan.len();
        if f < 3 {
            defect[g] = true;
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
    defect
}

fn sorted_set(v: &[[u32; 3]]) -> std::collections::BTreeSet<[u32; 3]> {
    v.iter().copied().collect()
}

fn norm3d_quad_margin(pts: &[Vec3], g: u32, m: u32, x: u32, y: u32) -> f64 {
    let p = |i: u32| {
        let v = pts[i as usize];
        glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64).normalize()
    };
    let (g, m, x, y) = (p(g), p(m), p(x), p(y));
    let a = m - g;
    let b = x - g;
    let c = y - g;
    let denom = (a.length() * b.length() * c.length()).max(1e-300);
    a.cross(b).dot(c).abs() / denom
}

fn min_fast_norm3d_quad_margin(pts: &[Vec3], g: u32, fan: &[[u32; 3]]) -> Option<f64> {
    if fan.len() < 3 {
        return None;
    }
    let mut best = f64::INFINITY;
    for i in 0..fan.len() {
        let t0 = fan[i];
        let t1 = fan[(i + 1) % fan.len()];
        let common: Vec<u32> = t0
            .iter()
            .copied()
            .filter(|&v| v != g && t1.contains(&v))
            .collect();
        if common.len() != 1 {
            continue;
        }
        let m = common[0];
        let Some(x) = t0.iter().copied().find(|&v| v != g && v != m) else {
            continue;
        };
        let Some(y) = t1.iter().copied().find(|&v| v != g && v != m) else {
            continue;
        };
        best = best.min(norm3d_quad_margin(pts, g, m, x, y));
    }
    best.is_finite().then_some(best)
}

fn flag_bands_from_env() -> Vec<f64> {
    let mut bands: Vec<f64> = std::env::var("S2_NORM3D_FLAG_BANDS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse::<f64>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]);
    bands.sort_by(f64::total_cmp);
    bands
}

fn print_norm3d_flag_recall(
    label: &str,
    dist: &str,
    seed: u64,
    pts: &[Vec3],
    fast_triples: &[Vec<[u32; 3]>],
    changed: &std::collections::BTreeSet<usize>,
    defect: &[bool],
    audit: Option<&[s2_voronoi::escalate_probe::CellAudit]>,
) {
    use std::collections::BTreeSet;

    let m = pts.len();
    let margins: Vec<Option<f64>> = (0..m)
        .map(|g| min_fast_norm3d_quad_margin(pts, g as u32, &fast_triples[g]))
        .collect();
    let defect_set: BTreeSet<usize> = defect
        .iter()
        .enumerate()
        .filter_map(|(g, &d)| d.then_some(g))
        .collect();

    println!(
        "{label} dist={dist} n={m} seed={seed}: changed={} ({:.4}%) defect_cells={}",
        changed.len(),
        100.0 * changed.len() as f64 / m.max(1) as f64,
        defect_set.len(),
    );
    let audit_by_gen: Vec<Option<s2_voronoi::escalate_probe::CellAudit>> = audit
        .map(|records| {
            let mut by_gen = vec![None; m];
            for &r in records {
                if let Some(slot) = by_gen.get_mut(r.generator as usize) {
                    *slot = Some(r);
                }
            }
            by_gen
        })
        .unwrap_or_default();
    println!("  band        flagged  hit  missed  recall%  false_pos  fp%     +defect_recall%");
    for band in flag_bands_from_env() {
        let flagged: BTreeSet<usize> = margins
            .iter()
            .enumerate()
            .filter_map(|(g, m)| m.is_some_and(|v| v <= band).then_some(g))
            .collect();
        let hit = flagged.intersection(changed).count();
        let missed = changed.len().saturating_sub(hit);
        let false_pos = flagged.len().saturating_sub(hit);
        let recall = 100.0 * hit as f64 / changed.len().max(1) as f64;
        let fp = 100.0 * false_pos as f64 / m.saturating_sub(changed.len()).max(1) as f64;

        let flagged_or_defect: BTreeSet<usize> = flagged.union(&defect_set).copied().collect();
        let hit_plus = flagged_or_defect.intersection(changed).count();
        let recall_plus = 100.0 * hit_plus as f64 / changed.len().max(1) as f64;
        println!(
            "  {band:9.1e} {:8} {:4} {:7} {:7.2} {:10} {:7.3} {:14.2}",
            flagged.len(),
            hit,
            missed,
            recall,
            false_pos,
            fp,
            recall_plus,
        );
    }
    if audit.is_some() {
        let term_band = std::env::var("S2_TERM_FLAG_BAND")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-6);
        let transition_band = std::env::var("S2_TRANSITION_FLAG_BAND")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-8);
        let early_band = std::env::var("S2_EARLY_UNCHANGED_FLAG_BAND")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-12);
        let fallback: BTreeSet<usize> = audit_by_gen
            .iter()
            .enumerate()
            .filter_map(|(g, r)| {
                r.is_some_and(|r| r.fallback_projection || r.fallback_polygon_cap)
                    .then_some(g)
            })
            .collect();
        let near_term: BTreeSet<usize> = audit_by_gen
            .iter()
            .enumerate()
            .filter_map(|(g, r)| {
                r.and_then(|r| r.termination_clearance)
                    .is_some_and(|c| c <= term_band)
                    .then_some(g)
            })
            .collect();
        let near_transition: BTreeSet<usize> = audit_by_gen
            .iter()
            .enumerate()
            .filter_map(|(g, r)| {
                r.and_then(|r| r.transition_delta)
                    .is_some_and(|c| c <= transition_band)
                    .then_some(g)
            })
            .collect();
        let near_early: BTreeSet<usize> = audit_by_gen
            .iter()
            .enumerate()
            .filter_map(|(g, r)| {
                r.and_then(|r| r.early_unchanged_clearance)
                    .is_some_and(|c| c <= early_band)
                    .then_some(g)
            })
            .collect();
        let margin_1e6: BTreeSet<usize> = margins
            .iter()
            .enumerate()
            .filter_map(|(g, m)| m.is_some_and(|v| v <= 1e-6).then_some(g))
            .collect();
        let joined: BTreeSet<usize> = defect_set
            .union(&fallback)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&near_term)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&near_transition)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&near_early)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&margin_1e6)
            .copied()
            .collect();
        println!(
            "  audit: fallback_cells={} near_term<={term_band:.1e}={} \
             near_transition<={transition_band:.1e}={} near_early<={early_band:.1e}={}",
            fallback.len(),
            near_term.len(),
            near_transition.len(),
            near_early.len(),
        );
        println!(
            "  audit_hits: defect/fallback/term/transition/early/margin1e-6/all={}/{}/{}/{}/{}/{}/{}",
            defect_set.intersection(changed).count(),
            fallback.intersection(changed).count(),
            near_term.intersection(changed).count(),
            near_transition.intersection(changed).count(),
            near_early.intersection(changed).count(),
            margin_1e6.intersection(changed).count(),
            joined.intersection(changed).count(),
        );
        for &g in changed.iter().take(12) {
            let margin = margins[g]
                .map(|m| format!("{m:.3e}"))
                .unwrap_or_else(|| "none".into());
            let audit = audit_by_gen.get(g).and_then(|r| *r);
            let term = audit
                .and_then(|r| r.termination_clearance)
                .map(|c| format!("{c:.3e}"))
                .unwrap_or_else(|| "none".into());
            let transition = audit
                .and_then(|r| r.transition_delta)
                .map(|c| format!("{c:.3e}"))
                .unwrap_or_else(|| "none".into());
            let early = audit
                .and_then(|r| r.early_unchanged_clearance)
                .map(|c| format!("{c:.3e}"))
                .unwrap_or_else(|| "none".into());
            let fallback = audit.is_some_and(|r| r.fallback_projection || r.fallback_polygon_cap);
            let bound = audit
                .and_then(|r| r.termination_bound)
                .map(|c| format!("{c:.3e}"))
                .unwrap_or_else(|| "none".into());
            let stream = audit
                .map(|r| {
                    format!(
                        "processed={} edges={} exhausted={} used_knn={} packed_tail={} packed_safe={}",
                        r.neighbors_processed,
                        r.final_edges,
                        r.knn_exhausted,
                        r.used_knn,
                        r.packed_tail_used,
                        r.packed_safe_exhausted,
                    )
                })
                .unwrap_or_else(|| "none".into());
            println!(
                "    changed g={g}: defect={} margin={} fallback={} term={} bound={} \
                 transition={} early={} stream={}",
                defect[g], margin, fallback, term, bound, transition, early, stream
            );
        }
    }
}

fn fast_cell_neighbors(g: u32, fan: &[[u32; 3]]) -> Vec<u32> {
    let mut out = Vec::new();
    if fan.len() < 3 {
        return out;
    }
    for i in 0..fan.len() {
        let t0 = fan[i];
        let t1 = fan[(i + 1) % fan.len()];
        let common: Vec<u32> = t0
            .iter()
            .copied()
            .filter(|&v| v != g && t1.contains(&v))
            .collect();
        if common.len() == 1 {
            out.push(common[0]);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn exact_cell_neighbors(g: u32, fan: &std::collections::BTreeSet<[u32; 3]>) -> Vec<u32> {
    let mut out = Vec::new();
    for tri in fan {
        for &v in tri {
            if v != g {
                out.push(v);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn print_changed_neighbor_diffs(
    pts: &[Vec3],
    changed: &std::collections::BTreeSet<usize>,
    fast_triples: &[Vec<[u32; 3]>],
    exact: &[std::collections::BTreeSet<[u32; 3]>],
    audit: Option<&[s2_voronoi::escalate_probe::CellAudit]>,
) {
    use std::collections::BTreeSet;

    let audit_by_gen: Vec<Option<s2_voronoi::escalate_probe::CellAudit>> = audit
        .map(|records| {
            let mut by_gen = vec![None; pts.len()];
            for &r in records {
                if let Some(slot) = by_gen.get_mut(r.generator as usize) {
                    *slot = Some(r);
                }
            }
            by_gen
        })
        .unwrap_or_default();

    let mut exact_match = 0usize;
    let mut neighbor_diff = 0usize;
    for &g in changed {
        let g_u32 = g as u32;
        let fast: BTreeSet<u32> = fast_cell_neighbors(g_u32, &fast_triples[g])
            .into_iter()
            .collect();
        let exact: BTreeSet<u32> = exact_cell_neighbors(g_u32, &exact[g]).into_iter().collect();
        if fast == exact {
            exact_match += 1;
        } else {
            neighbor_diff += 1;
        }
    }

    println!(
        "  changed_neighbor_sets: same={} different={}",
        exact_match, neighbor_diff
    );

    for &g in changed.iter().take(12) {
        let g_u32 = g as u32;
        let fast: BTreeSet<u32> = fast_cell_neighbors(g_u32, &fast_triples[g])
            .into_iter()
            .collect();
        let exact: BTreeSet<u32> = exact_cell_neighbors(g_u32, &exact[g]).into_iter().collect();
        if fast == exact {
            println!("    changed g={g}: neighbor_set=same");
        } else {
            let missing: Vec<u32> = exact.difference(&fast).copied().collect();
            let extra: Vec<u32> = fast.difference(&exact).copied().collect();
            let generator = pts[g].normalize();
            let max_missing_dot = missing
                .iter()
                .map(|&m| generator.dot(pts[m as usize].normalize()) as f64)
                .max_by(f64::total_cmp);
            let term_bound = audit_by_gen
                .get(g)
                .and_then(|r| *r)
                .and_then(|r| r.termination_bound);
            let bound_gap = max_missing_dot
                .zip(term_bound)
                .map(|(dot, bound)| dot - bound);
            let max_missing_dot = max_missing_dot
                .map(|d| format!("{d:.9e}"))
                .unwrap_or_else(|| "none".into());
            let term_bound = term_bound
                .map(|d| format!("{d:.9e}"))
                .unwrap_or_else(|| "none".into());
            let bound_gap = bound_gap
                .map(|d| format!("{d:.3e}"))
                .unwrap_or_else(|| "none".into());
            println!(
                "    changed g={g}: missing_neighbors={:?} extra_neighbors={:?} \
                 max_missing_dot={} term_bound={} dot_minus_bound={}",
                missing, extra, max_missing_dot, term_bound, bound_gap
            );
        }
    }
}

fn missing_neighbor_pairs(
    changed: &std::collections::BTreeSet<usize>,
    fast_triples: &[Vec<[u32; 3]>],
    exact: &[std::collections::BTreeSet<[u32; 3]>],
) -> Vec<(u32, u32)> {
    use std::collections::BTreeSet;

    let mut pairs = BTreeSet::new();
    for &g in changed {
        let g_u32 = g as u32;
        let fast: BTreeSet<u32> = fast_cell_neighbors(g_u32, &fast_triples[g])
            .into_iter()
            .collect();
        let exact: BTreeSet<u32> = exact_cell_neighbors(g_u32, &exact[g]).into_iter().collect();
        pairs.extend(exact.difference(&fast).map(|&m| (g_u32, m)));
    }
    pairs.into_iter().collect()
}

fn print_watched_missing_neighbor_attempts(points: &[UnitVec3], pairs: &[(u32, u32)]) {
    use s2_voronoi::escalate_probe::{
        clear_watch_pairs, reset_proactive_audit, set_watch_pairs, take_watched_clips,
        WatchedClipResult,
    };
    use std::collections::BTreeMap;

    if pairs.is_empty() {
        return;
    }

    reset_proactive_audit();
    set_watch_pairs(pairs);
    set_escalation_enabled(true);
    let _ = compute_with_report(points, VoronoiConfig::default()).expect("watch-pair build");
    set_escalation_enabled(false);
    let watched = take_watched_clips();
    clear_watch_pairs();

    let mut by_pair: BTreeMap<(u32, u32), Vec<WatchedClipResult>> = BTreeMap::new();
    for record in watched {
        by_pair
            .entry((record.generator, record.neighbor))
            .or_default()
            .push(record.result);
    }
    let attempted = pairs
        .iter()
        .filter(|pair| by_pair.contains_key(pair))
        .count();
    println!(
        "  watched_missing_neighbors: pairs={} attempted={} never_attempted={}",
        pairs.len(),
        attempted,
        pairs.len().saturating_sub(attempted),
    );
    for &pair in pairs.iter().take(16) {
        match by_pair.get(&pair) {
            Some(results) => {
                println!("    watched {pair:?}: attempted {:?}", results);
            }
            None => {
                println!("    watched {pair:?}: never_attempted");
            }
        }
    }
}

fn changed_component_summary(changed: &[bool], fast_triples: &[Vec<[u32; 3]>]) -> (usize, usize) {
    let mut seen = vec![false; changed.len()];
    let mut components = 0usize;
    let mut max_size = 0usize;
    for seed in 0..changed.len() {
        if !changed[seed] || seen[seed] {
            continue;
        }
        components += 1;
        let mut stack = vec![seed as u32];
        seen[seed] = true;
        let mut size = 0usize;
        while let Some(g) = stack.pop() {
            size += 1;
            for h in fast_cell_neighbors(g, &fast_triples[g as usize]) {
                let hi = h as usize;
                if hi < changed.len() && changed[hi] && !seen[hi] {
                    seen[hi] = true;
                    stack.push(h);
                }
            }
        }
        max_size = max_size.max(size);
    }
    (components, max_size)
}

/// Compare the fast gnomonic graph against a normalized 3D exact Delaunay
/// reference (global `LocalHull`). This is the cell-error measurement for a
/// future exact-by-construction mode. The normalization inside `LocalHull` is
/// essential: raw f32 radius drift is not the S2 graph.
///
/// Example:
///   S2_ESCALATE_DIST=uniform S2_ESCALATE_N=12000 S2_ESCALATE_SEED=3 \
///     cargo test --release --features escalate_probe --test escalate \
///     probe_fast_vs_norm3d_reference -- --ignored --nocapture
#[test]
#[ignore = "normalized 3D exact-reference probe; run with env + --ignored --nocapture"]
fn probe_fast_vs_norm3d_reference() {
    let (dist, seed, points) = probe_points_from_env(12_000);
    let (pts, fast_triples) = stash_fast_triples(&points);
    let m = pts.len();
    let (exact, exact_complete) = exact_triples_norm3d(&pts);
    let defect = defect_cells_from_triples(&fast_triples);

    let mut eligible = 0usize;
    let mut changed = 0usize;
    let mut changed_defective = 0usize;
    let mut changed_valid = 0usize;
    let mut defect_cells = 0usize;
    let mut defect_not_changed = 0usize;
    let mut incomplete_exact = 0usize;
    let mut changed_mask = vec![false; m];
    for g in 0..m {
        if defect[g] {
            defect_cells += 1;
        }
        if !exact_complete[g] || fast_triples[g].is_empty() {
            incomplete_exact += 1;
            continue;
        }
        eligible += 1;
        let ch = sorted_set(&fast_triples[g]) != exact[g];
        if ch {
            changed_mask[g] = true;
            changed += 1;
            if defect[g] {
                changed_defective += 1;
            } else {
                changed_valid += 1;
            }
        } else if defect[g] {
            defect_not_changed += 1;
        }
    }
    let (changed_components, changed_max_component) =
        changed_component_summary(&changed_mask, &fast_triples);
    println!(
        "NORM3D-REF dist={dist} n={m} seed={seed}: eligible={eligible} \
         incomplete_exact={incomplete_exact} changed={changed} ({:.4}%) \
         changed_defective={changed_defective} changed_valid={changed_valid} \
         defect_cells={defect_cells} defect_not_changed={defect_not_changed} \
         changed_components={changed_components} changed_max_component={changed_max_component}",
        100.0 * changed as f64 / eligible.max(1) as f64,
    );
}

#[test]
#[ignore = "external CGAL exact hull probe; set S2_CGAL_HULL3_BIN and run with --ignored --nocapture"]
fn probe_cgal_hull3_vs_local_hull_reference() {
    let (dist, seed, points) = probe_points_from_env(2_000);
    let pts = to_vec3(&points);
    let (local, local_complete) = exact_triples_norm3d(&pts);
    let (cgal, cgal_complete, cgal_log) = exact_triples_cgal_hull3(&pts);

    let mut eligible = 0usize;
    let mut changed = 0usize;
    let mut local_incomplete = 0usize;
    let mut cgal_incomplete = 0usize;
    for g in 0..pts.len() {
        if !local_complete[g] {
            local_incomplete += 1;
        }
        if !cgal_complete[g] {
            cgal_incomplete += 1;
        }
        if !local_complete[g] || !cgal_complete[g] {
            continue;
        }
        eligible += 1;
        changed += usize::from(local[g] != cgal[g]);
    }

    println!(
        "CGAL-HULL3-vs-LOCAL dist={dist} n={} seed={seed}: eligible={eligible} \
         changed={changed} ({:.4}%) local_incomplete={local_incomplete} \
         cgal_incomplete={cgal_incomplete} | {}",
        pts.len(),
        100.0 * changed as f64 / eligible.max(1) as f64,
        cgal_log.trim(),
    );
}

#[test]
#[ignore = "fast graph vs external CGAL exact hull probe; set S2_CGAL_HULL3_BIN and run with --ignored --nocapture"]
fn probe_fast_vs_cgal_hull3_reference() {
    let (dist, seed, points) = probe_points_from_env(12_000);
    let (pts, fast_triples) = stash_fast_triples(&points);
    let m = pts.len();
    let (exact, exact_complete, cgal_log) = exact_triples_cgal_hull3(&pts);
    let defect = defect_cells_from_triples(&fast_triples);

    let mut eligible = 0usize;
    let mut changed = 0usize;
    let mut changed_defective = 0usize;
    let mut changed_valid = 0usize;
    let mut defect_cells = 0usize;
    let mut defect_not_changed = 0usize;
    let mut incomplete_exact = 0usize;
    let mut changed_mask = vec![false; m];
    for g in 0..m {
        if defect[g] {
            defect_cells += 1;
        }
        if !exact_complete[g] || fast_triples[g].is_empty() {
            incomplete_exact += 1;
            continue;
        }
        eligible += 1;
        let ch = sorted_set(&fast_triples[g]) != exact[g];
        if ch {
            changed_mask[g] = true;
            changed += 1;
            if defect[g] {
                changed_defective += 1;
            } else {
                changed_valid += 1;
            }
        } else if defect[g] {
            defect_not_changed += 1;
        }
    }
    let (changed_components, changed_max_component) =
        changed_component_summary(&changed_mask, &fast_triples);
    println!(
        "CGAL-HULL3-REF dist={dist} n={m} seed={seed}: eligible={eligible} \
         incomplete_exact={incomplete_exact} changed={changed} ({:.4}%) \
         changed_defective={changed_defective} changed_valid={changed_valid} \
         defect_cells={defect_cells} defect_not_changed={defect_not_changed} \
         changed_components={changed_components} changed_max_component={changed_max_component} | {}",
        100.0 * changed as f64 / eligible.max(1) as f64,
        cgal_log.trim(),
    );
}

/// Candidate flag recall against the normalized-3D changed-cell set. The flag
/// here is deliberately simple: mark a cell if any consecutive pair of fast-cell
/// vertices forms a near-cocircular normalized 3D quad with volume <= BAND.
/// This tests whether a question-intrinsic S2 margin could be a viable cheap
/// detector before trying to make it hot-path/provable.
///
/// Optional env:
/// - `S2_NORM3D_FLAG_BANDS=1e-12,1e-10,1e-8,1e-6`
#[test]
#[ignore = "normalized 3D flag-recall probe; run with env + --ignored --nocapture"]
fn probe_norm3d_flag_recall() {
    use std::collections::BTreeSet;

    let (dist, seed, points) = probe_points_from_env(12_000);
    let (pts, fast_triples) = stash_fast_triples(&points);
    let m = pts.len();
    let (exact, exact_complete) = exact_triples_norm3d(&pts);
    let defect = defect_cells_from_triples(&fast_triples);

    let changed: BTreeSet<usize> = (0..m)
        .filter(|&g| {
            exact_complete[g]
                && !fast_triples[g].is_empty()
                && sorted_set(&fast_triples[g]) != exact[g]
        })
        .collect();
    print_norm3d_flag_recall(
        "NORM3D-FLAG",
        &dist,
        seed,
        &pts,
        &fast_triples,
        &changed,
        &defect,
        None,
    );
}

#[test]
#[ignore = "external CGAL truth flag-recall probe; set S2_CGAL_HULL3_BIN and run with --ignored --nocapture"]
fn probe_cgal_hull3_flag_recall() {
    use std::collections::BTreeSet;

    let (dist, seed, points) = probe_points_from_env(12_000);
    let (pts, fast_triples, audit) = stash_fast_triples_with_audit(&points);
    let m = pts.len();
    let (exact, complete, cgal_log) = exact_triples_cgal_hull3(&pts);
    let defect = defect_cells_from_triples(&fast_triples);

    let changed: BTreeSet<usize> = (0..m)
        .filter(|&g| {
            complete[g] && !fast_triples[g].is_empty() && sorted_set(&fast_triples[g]) != exact[g]
        })
        .collect();
    print_norm3d_flag_recall(
        "CGAL-HULL3-FLAG",
        &dist,
        seed,
        &pts,
        &fast_triples,
        &changed,
        &defect,
        Some(&audit),
    );
    print_changed_neighbor_diffs(&pts, &changed, &fast_triples, &exact, Some(&audit));
    let missing_pairs = missing_neighbor_pairs(&changed, &fast_triples, &exact);
    print_watched_missing_neighbor_attempts(&points, &missing_pairs);
    println!("  {}", cgal_log.trim());
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
/// delaunator on the repair cells? This was the first projected-oracle sanity
/// check. After the normalized-CGAL finding, normalized local 3D hull is also a
/// viable oracle candidate.
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

/// Tests the old projection-theory hypothesis. The normalized-CGAL probe showed
/// that exact 3D hulls must renormalize f32 inputs back onto S2; once local_hull
/// does that, fast, projected delaunator, and local_hull should agree on these
/// interior cells.
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

    // local_hull GLOBAL (normalized S2 orient3d): ONE hull over all points, all cells.
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
/// exact reference (delaunator). This measures how well normalized `local_hull`
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
    let off = || VoronoiConfig {
        repair_mode: RepairMode::Disabled,
        ..VoronoiConfig::default()
    };
    let on = VoronoiConfig::default;
    let run = |label: &str, pts: &[UnitVec3]| {
        let before = compute_with_report(pts, off()).expect("build");
        let after = compute_with_report(pts, on()).expect("build");
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

/// Regression: defect-driven repair (projected delaunator oracle + grow on
/// unpaired-edges & low-incidence vertices + valid-or-revert gate) turns the
/// defective mega builds into STRICTLY VALID diagrams. Covers the formerly-hard
/// seeds: s2 (cross-cell decision divergence between individually-valid cells)
/// and s15 (a low-incidence-only defect with NO unpaired edge). Off by default;
/// `set_escalation_enabled` is a process-global toggle only THIS test flips, so
/// the on/off builds below are reliable under parallel execution.
#[test]
fn escalation_repair_makes_mega_strictly_valid() {
    let off = || VoronoiConfig {
        repair_mode: RepairMode::Disabled,
        ..VoronoiConfig::default()
    };
    let on = VoronoiConfig::default;
    let mut fixed_at_least_one = false;
    for seed in [1u64, 2, 15] {
        let points = mega_points(100_000, 0.8, seed);

        let before = compute_with_report(&points, off()).expect("build");
        let before_valid = before.report.returned_validation.is_strictly_valid();

        let after = compute_with_report(&points, on()).expect("build");
        let after_report = &after.report.returned_validation;

        println!(
            "mega 100k s{seed}: off={} on={}",
            if before_valid { "VALID" } else { "INVALID" },
            if after_report.is_strictly_valid() {
                "VALID".to_string()
            } else {
                format!("{:?}", after_report.subdivision_issues())
            }
        );
        // The repair must always reach strict validity (the valid-or-revert gate
        // guarantees the output is never worse than the fast path, so this is the
        // meaningful bar).
        assert!(
            after_report.is_strictly_valid(),
            "seed {seed}: repair did not produce a strictly valid diagram: {:?}",
            after_report.subdivision_issues()
        );
        fixed_at_least_one |= !before_valid;
    }
    // Sanity: at least one of these seeds is a genuine defect the repair fixed
    // (otherwise the test would pass trivially on already-valid inputs).
    assert!(
        fixed_at_least_one,
        "expected at least one mega seed to be invalid without the repair"
    );
}
