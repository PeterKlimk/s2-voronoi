//! Defect-driven escalation (step C of the adaptive-canonical-clip plan, see
//! `docs/adaptive-canonical-clip-design-2026-06.md`).
//!
//! Rebuilds a near-degenerate neighborhood as a SINGLE exact local Delaunay
//! (`local_hull`) and reads each generator's Voronoi cell off the shared dual.
//! Because every cell comes from one triangulation, they pair on shared edges
//! by construction (see `local_hull::tests::dual_cells_agree_on_shared_edges`),
//! which is the property the reverted per-cell/pin-by-key repairs lacked.
//!
//! This is the vertical-slice core: brute-force neighbor gather + rebuild + a
//! consistency read. The production loop (connected-component growth, the
//! considered-neighbor set, re-assembly, grow-until-clean-rim) builds on top.

// The production consumer (the escalation loop) lands in a follow-up; until
// then this is exercised by the escalate_probe integration test only.
#![allow(dead_code)]

use glam::Vec3;

use super::local_hull::LocalHull;

/// A generator's rebuilt Voronoi cell: its vertices as the ordered cyclic fan
/// of sorted global-id triples (each triple = the three generators meeting at
/// that Voronoi vertex). Same identity space as the production `VertexKey`.
#[derive(Debug, Clone)]
pub struct RebuiltCell {
    /// The generator (global id) whose Voronoi cell this is.
    pub generator: u32,
    /// The cell's vertices, in cyclic fan order, each a sorted global-id triple.
    pub vertices: Vec<[u32; 3]>,
}

impl RebuiltCell {
    /// The vertices of this cell incident to the edge it shares with `other`
    /// (the two Voronoi vertices whose triple contains both generators), as a
    /// sorted set. An interior edge has exactly two; non-adjacent cells, zero.
    pub fn shared_edge_with(&self, other: u32) -> Vec<[u32; 3]> {
        let mut v: Vec<[u32; 3]> = self
            .vertices
            .iter()
            .copied()
            .filter(|t| t.contains(&self.generator) && t.contains(&other))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// Gather a local generator id set: the `seeds` plus their `k` nearest
/// neighbors (largest dot product on the unit sphere). Brute force — the
/// production path will pull this from the spatial grid / the cell's retained
/// considered-neighbor set.
pub fn gather_local(points: &[Vec3], seeds: &[u32], k: usize) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<u32> = seeds.iter().copied().collect();
    let mut scored: Vec<(f32, u32)> = Vec::with_capacity(points.len());
    for &s in seeds {
        let sp = points[s as usize];
        scored.clear();
        scored.extend(
            points
                .iter()
                .enumerate()
                .map(|(i, &p)| (sp.dot(p), i as u32)),
        );
        // Partial sort: largest dot = nearest. (Test-scale brute force.)
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(_, id) in scored.iter().take(k + 1) {
            set.insert(id);
        }
    }
    set.into_iter().collect()
}

/// Rebuild the `seeds`' Voronoi cells from one exact local hull over
/// `local_ids`. Returns one [`RebuiltCell`] per seed, or `None` if the local
/// set has no 3D hull (all generators coplanar — a measure-zero pathology).
pub fn rebuild_cells(
    points: &[Vec3],
    local_ids: &[u32],
    seeds: &[u32],
) -> Option<Vec<RebuiltCell>> {
    let pos: Vec<Vec3> = local_ids.iter().map(|&g| points[g as usize]).collect();
    let hull = LocalHull::build(&pos)?;

    let local_of = |g: u32| -> Option<usize> { local_ids.iter().position(|&x| x == g) };

    let mut out = Vec::with_capacity(seeds.len());
    for &g in seeds {
        let lg = local_of(g)?;
        let fan = hull.cell_faces(lg);
        if fan.is_empty() {
            return None; // broken fan — caller bails (shouldn't happen interior)
        }
        let vertices = fan
            .iter()
            .map(|&fi| {
                let [a, b, c] = hull.faces()[fi];
                let mut t = [local_ids[a], local_ids[b], local_ids[c]];
                t.sort_unstable();
                t
            })
            .collect();
        out.push(RebuiltCell {
            generator: g,
            vertices,
        });
    }
    Some(out)
}
