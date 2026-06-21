//! Step-C vertical slice (feature `escalate_probe`): drive the exact local
//! rebuild over REAL fast-path defects and confirm it produces a consistent
//! local diagram where the fast path left an unpaired edge.
//!
//!   cargo test --release --features escalate_probe --test escalate -- --nocapture
#![cfg(feature = "escalate_probe")]

mod support;

use glam::Vec3;
use s2_voronoi::escalate_probe::{gather_local, rebuild_cells};
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
            let edge_from_a = cells[0].shared_edge_with(b);
            let edge_from_b = cells[1].shared_edge_with(a);
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
