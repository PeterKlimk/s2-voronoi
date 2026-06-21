//! Boundary extraction for the Tier-2 totality patch synthesizer (spherical).
//!
//! A contested component `C` (a set of generators whose cells disagree about a
//! near-degenerate vertex) is repaired by *reproducing its boundary verbatim*
//! and re-filling the interior. The boundary is the set of edges separating a
//! `C`-cell from a non-`C` (ring) cell. Those edges are already paired by the
//! valid ring cells, so reproducing them exactly keeps the seam paired no matter
//! how the interior is rebuilt (the firewall is the ring, not anything inside
//! `C` — see `docs/reclip-hull-snap-experiment-2026-06.md`).
//!
//! This module turns the ring cells' references to `C` into a set of **oriented
//! simple cycles** of existing vids, wound so the contested region is on the
//! LEFT. The fill (next stage) reproduces each cycle edge from the inside and
//! tiles the enclosed region with the component's generators.
//!
//! The hard part Codex flagged is that the boundary is a *graph*, not always one
//! clean cycle: it can be several disjoint loops, or pinch through a shared
//! vertex (figure-eight). Cycle decomposition therefore uses the planar
//! embedding (angular order of edges around each vertex) to pair the chains
//! correctly at pinch points, rather than assuming degree ≤ 1.
//!
//! Errors are surfaced, not silently bailed: each `BoundaryError` is a contract
//! the synthesizer relies on, and which variant fires on the mega seeds tells us
//! which topology actually occurs in practice.

use super::reclip_repair::{cell_span, key_common_pair};
use crate::diagram::VoronoiCell;
use crate::live_dedup::ShardedVertexKeys;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

type VertexKey3 = [u32; 3];
/// Inside-oriented directed boundary edges, a `vid → key` pin map, and a
/// `directed-edge → C-owner generator` map (the C-side cell of each boundary
/// edge, needed by the fill to assign boundary arcs to generators).
type CollectedBoundary = (
    Vec<(u32, u32)>,
    HashMap<u32, VertexKey3>,
    HashMap<(u32, u32), u32>,
);

/// A vertex on the contested component's boundary, pinned to an existing global
/// vertex id (recovered from a valid ring cell) and carrying its generator-triple
/// key plus the C-generator that owns the boundary edge LEAVING this vertex (the
/// next edge in the loop) — the fill uses it to group the loop into per-generator
/// arcs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoundaryVert {
    pub(crate) vid: u32,
    pub(crate) key: VertexKey3,
    /// C-generator owning the boundary edge `vid → next`.
    pub(crate) edge_owner: u32,
}

/// The extracted boundary of a contested component: oriented simple cycles of
/// existing vids, wound so the contested region is on the LEFT (the inside-on-left
/// convention produced by reversing each ring cell's outward-wound edge).
#[derive(Debug)]
pub(crate) struct BoundaryExtract {
    pub(crate) loops: Vec<Vec<BoundaryVert>>,
}

/// Why boundary extraction could not produce clean oriented loops. Each variant
/// is a structural assumption of the patch synthesizer; surfacing it (rather than
/// a silent bail) is the point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BoundaryError {
    /// The directed boundary is not a union of cycles: some vid's in-degree and
    /// out-degree differ, so the ring references do not enclose a region.
    UnbalancedVertex { vid: u32, in_deg: u32, out_deg: u32 },
    /// The same directed boundary edge was contributed twice (ring over-emit) —
    /// the inside region cannot traverse one edge twice in the same orientation.
    DuplicateDirectedEdge { from: u32, to: u32 },
    /// A boundary vid has no usable position, so a pinch cannot be resolved by
    /// angular order. Only raised when the decomposition actually needs it.
    MissingPosition { vid: u32 },
    /// A boundary vid had no recoverable key (should not happen — every vid was
    /// pinned from a valid ring cell that has a key).
    MissingKey { vid: u32 },
    /// The walk consumed an edge but could not continue / close the cycle. Means
    /// the directed multigraph is malformed in a way the balance check missed.
    OpenWalk { at: u32 },
}

/// How boundary edges are collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CollectMode {
    /// Key domain: scan ring cells, infer the edge's two cells via the common
    /// generator pair of its endpoint keys. Fragile at near-degenerate vertices
    /// where consecutive ring vertices do not share a clean 2-generator edge.
    Keys,
    /// Edge-pairing domain: a boundary edge is a `C`-cell directed edge whose
    /// reverse is owned by a non-`C` cell (so it is paired across the rim by
    /// construction). Manifold-by-construction when the rim is paired-clean;
    /// genuinely-unpaired `C`-cell edges (the residual, interior to `C`) are
    /// skipped. This is the robust collector.
    Paired,
}

/// Collect the inside-oriented directed boundary edges of component `C`.
///
/// See [`CollectMode`]. Returns the directed edges and a `vid → key` pin map.
fn collect_boundary_edges(
    gset: &HashSet<u32>,
    ring_candidates: &HashSet<u32>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    mode: CollectMode,
) -> Result<CollectedBoundary, BoundaryError> {
    match mode {
        CollectMode::Keys => {
            collect_boundary_edges_keys(gset, ring_candidates, cells, cell_indices, vertex_keys)
        }
        CollectMode::Paired => {
            collect_boundary_edges_paired(gset, ring_candidates, cells, cell_indices, vertex_keys)
        }
    }
}

/// Edge-pairing-domain collection (the robust path). Build a directed-edge →
/// owning-cell map over `C ∪ ring`, then a `C`-cell edge `va→vb` is a boundary
/// edge iff its reverse `vb→va` is owned by a non-`C` cell. The `C`-cell traverses
/// it with the `C`-region on its left, so emit `va→vb` directly (inside-on-left,
/// no reversal). Edges whose reverse has no owner are genuinely unpaired (residual,
/// interior to `C`) and are skipped — by construction the boundary then closes
/// without growing, and any unpaired `C↔`ring edge would manifest as an imbalance
/// (which is then a real, detectable mismatch).
fn collect_boundary_edges_paired(
    gset: &HashSet<u32>,
    ring_candidates: &HashSet<u32>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
) -> Result<CollectedBoundary, BoundaryError> {
    // Undirected edge {lo,hi} -> the cells using it, with orientation. Built over
    // C ∪ ring so both sides of every rim edge are seen.
    #[derive(Default)]
    struct Uses {
        // (cell, forward) where forward = traversed lo->hi.
        uses: Vec<(u32, bool)>,
    }
    let mut edge_uses: HashMap<(u32, u32), Uses> = HashMap::new();
    let record = |g: u32, map: &mut HashMap<(u32, u32), Uses>| {
        if (g as usize) >= cells.len() {
            return;
        }
        if let Some(span) = cell_span(cells, cell_indices, g) {
            let n = span.len();
            for i in 0..n {
                let a = span[i];
                let b = span[(i + 1) % n];
                if a == b {
                    continue;
                }
                let (lo, hi, fwd) = if a < b { (a, b, true) } else { (b, a, false) };
                map.entry((lo, hi)).or_default().uses.push((g, fwd));
            }
        }
    };
    for &g in gset {
        record(g, &mut edge_uses);
    }
    for &h in ring_candidates {
        if !gset.contains(&h) {
            record(h, &mut edge_uses);
        }
    }

    // Emit the topological cut: an undirected edge is a clean boundary edge iff it
    // has EXACTLY two uses, of OPPOSITE orientation, with EXACTLY ONE owner in C
    // (Codex's full paired-edge contract). Orientation is taken from the RING
    // (trusted) use and reversed, so the contested region is on the left. Anything
    // else involving a C cell (singleton, >2 uses, same orientation, both-in-C with
    // a third use) is NOT a clean cut: it is interior residual / non-manifold and
    // is left for the fill / surfaced as an imbalance — never papered over here.
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut key_of: HashMap<u32, VertexKey3> = HashMap::new();
    let mut owner_of: HashMap<(u32, u32), u32> = HashMap::new();
    for (&(lo, hi), u) in &edge_uses {
        if u.uses.len() != 2 {
            continue;
        }
        let (c0, f0) = u.uses[0];
        let (c1, f1) = u.uses[1];
        if f0 == f1 {
            continue; // same orientation: not a manifold pairing
        }
        let in0 = gset.contains(&c0);
        let in1 = gset.contains(&c1);
        if in0 == in1 {
            continue; // both in C (interior) or both ring (not C's boundary)
        }
        // Ring use = the one not in C; C owner = the one in C. Inside-on-left =
        // reverse of the ring use.
        let (ring_fwd, c_owner) = if in0 { (f1, c0) } else { (f0, c1) };
        // ring_fwd true means ring traverses lo->hi, so inside edge is hi->lo.
        let (from, to) = if ring_fwd { (hi, lo) } else { (lo, hi) };
        if let Some(k) = vertex_keys.get(from) {
            key_of.insert(from, k);
        }
        if let Some(k) = vertex_keys.get(to) {
            key_of.insert(to, k);
        }
        owner_of.insert((from, to), c_owner);
        edges.push((from, to));
    }
    Ok((edges, key_of, owner_of))
}

/// Key-domain collection (legacy/fragile). Scans every ring candidate cell `h`
/// (a non-`C` cell). For each polygon edge `va→vb` of `h` whose two incident
/// generators are `{h, across}` with `across ∈ C`, the edge separates the ring
/// from the contested region; `h` is wound with `h` on its left, so the contested
/// region lies on the left of the **reverse** edge `vb→va`.
fn collect_boundary_edges_keys(
    gset: &HashSet<u32>,
    ring_candidates: &HashSet<u32>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
) -> Result<CollectedBoundary, BoundaryError> {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut key_of: HashMap<u32, VertexKey3> = HashMap::new();
    let mut owner_of: HashMap<(u32, u32), u32> = HashMap::new();

    for &h in ring_candidates {
        if gset.contains(&h) || (h as usize) >= cells.len() {
            continue;
        }
        let Some(span) = cell_span(cells, cell_indices, h) else {
            continue;
        };
        let n = span.len();
        for i in 0..n {
            let va = span[i];
            let vb = span[(i + 1) % n];
            let (Some(ka), Some(kb)) = (vertex_keys.get(va), vertex_keys.get(vb)) else {
                continue;
            };
            let Some((a, b)) = key_common_pair(ka, kb) else {
                continue;
            };
            // The edge's two incident generators are {a, b}. In cell h one of
            // them is h itself; the other is the cell across the edge (the C owner).
            let across = if a == h {
                b
            } else if b == h {
                a
            } else {
                continue;
            };
            if !gset.contains(&across) {
                continue; // ring↔ring edge, not part of C's boundary
            }
            key_of.insert(va, ka);
            key_of.insert(vb, kb);
            // Inside-on-left: reverse of h's outward edge.
            if !seen.insert((vb, va)) {
                return Err(BoundaryError::DuplicateDirectedEdge { from: vb, to: va });
            }
            owner_of.insert((vb, va), across);
            edges.push((vb, va));
        }
    }

    Ok((edges, key_of, owner_of))
}

/// Angle of `neighbor` as seen from `center`, measured CCW in the tangent plane
/// at `center` viewed from outside the sphere (the `+center` side). The basis is
/// right-handed `(t1, t2, n)` with `n = center`, so increasing angle is CCW from
/// the outside — matching the cell winding convention (signed area positive when
/// CCW as seen from the generator side).
fn angle_at(center: Vec3, neighbor: Vec3) -> f64 {
    let n = center.normalize();
    // A tangent axis robustly perpendicular to n.
    let seed = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t1 = (seed - n * seed.dot(n)).normalize();
    let t2 = n.cross(t1); // right-handed: viewed from +n, t1 -> t2 is CCW
    let d = neighbor - n * neighbor.dot(n);
    (d.dot(t2) as f64).atan2(d.dot(t1) as f64)
}

/// Decompose a balanced directed edge set into oriented simple cycles, using the
/// planar embedding (angular order at each vertex) to continue the correct chain
/// at pinch vertices.
///
/// Contract: the directed edges must form a union of cycles (every vid has equal
/// in/out degree) with no repeated directed edge. The inside-on-left convention
/// (each edge has the contested region on its left) makes the angular rule a
/// standard CCW face traversal: arriving `prev→cur`, the next edge is the
/// unused out-edge whose direction is the **first CCW** from the reverse of the
/// incoming direction (smallest positive angular gap). With out-degree 1 the
/// rule is trivial and no position is needed.
fn decompose_cycles(
    edges: &[(u32, u32)],
    pos_of: impl Fn(u32) -> Option<Vec3>,
) -> Result<Vec<Vec<u32>>, BoundaryError> {
    // Adjacency + degree balance.
    let mut out_adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut in_deg: HashMap<u32, u32> = HashMap::new();
    for &(a, b) in edges {
        out_adj.entry(a).or_default().push(b);
        *in_deg.entry(b).or_default() += 1;
        in_deg.entry(a).or_default();
        out_adj.entry(b).or_default();
    }
    for (&vid, outs) in &out_adj {
        let od = outs.len() as u32;
        let id = in_deg.get(&vid).copied().unwrap_or(0);
        if od != id {
            return Err(BoundaryError::UnbalancedVertex {
                vid,
                in_deg: id,
                out_deg: od,
            });
        }
    }

    let mut used: HashSet<(u32, u32)> = HashSet::new();
    let mut cycles: Vec<Vec<u32>> = Vec::new();
    let guard = edges.len() + 1;

    for &(s0, t0) in edges {
        if used.contains(&(s0, t0)) {
            continue;
        }
        used.insert((s0, t0));
        let mut nodes = vec![s0];
        let mut prev = s0;
        let mut cur = t0;
        let mut steps = 0usize;
        while cur != s0 {
            nodes.push(cur);
            let nxt = pick_next(cur, prev, &out_adj, &used, &pos_of)?;
            used.insert((cur, nxt));
            prev = cur;
            cur = nxt;
            steps += 1;
            if steps > guard {
                return Err(BoundaryError::OpenWalk { at: cur });
            }
        }
        cycles.push(nodes);
    }

    Ok(cycles)
}

/// Pick the next out-edge from `cur`, having arrived from `prev`, among unused
/// edges. With one choice the angular order is irrelevant; with several (a pinch)
/// take the first out-direction CCW from the reverse of the incoming direction.
fn pick_next(
    cur: u32,
    prev: u32,
    out_adj: &HashMap<u32, Vec<u32>>,
    used: &HashSet<(u32, u32)>,
    pos_of: &impl Fn(u32) -> Option<Vec3>,
) -> Result<u32, BoundaryError> {
    let outs: Vec<u32> = out_adj
        .get(&cur)
        .into_iter()
        .flatten()
        .copied()
        .filter(|&w| !used.contains(&(cur, w)))
        .collect();
    match outs.len() {
        0 => Err(BoundaryError::OpenWalk { at: cur }),
        1 => Ok(outs[0]),
        _ => {
            let pc = pos_of(cur).ok_or(BoundaryError::MissingPosition { vid: cur })?;
            let pp = pos_of(prev).ok_or(BoundaryError::MissingPosition { vid: prev })?;
            let r = angle_at(pc, pp); // reverse of incoming direction
            const TAU: f64 = std::f64::consts::TAU;
            let mut best: Option<u32> = None;
            let mut best_gap = f64::INFINITY;
            for w in outs {
                let pw = pos_of(w).ok_or(BoundaryError::MissingPosition { vid: w })?;
                let mut gap = angle_at(pc, pw) - r;
                // Strictly positive CCW gap; a near-zero gap (going straight back
                // the way we came) is pushed to a full turn so it is chosen last.
                while gap <= 1e-9 {
                    gap += TAU;
                }
                if gap < best_gap {
                    best_gap = gap;
                    best = Some(w);
                }
            }
            best.ok_or(BoundaryError::OpenWalk { at: cur })
        }
    }
}

/// Extract the oriented boundary loops of contested component `C`.
///
/// `ring_candidates` must be a (super)set of every non-`C` cell adjacent to `C`;
/// over-inclusion is safe (non-adjacent cells contribute no boundary edges),
/// under-inclusion makes the boundary unbalanced and is reported.
pub(crate) fn extract_boundary(
    gset: &HashSet<u32>,
    ring_candidates: &HashSet<u32>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertices: &[Vec3],
    vertex_keys: &ShardedVertexKeys,
    mode: CollectMode,
) -> Result<BoundaryExtract, BoundaryError> {
    let (edges, key_of, owner_of) = collect_boundary_edges(
        gset,
        ring_candidates,
        cells,
        cell_indices,
        vertex_keys,
        mode,
    )?;
    let pos_of = |vid: u32| vertices.get(vid as usize).copied();
    let cycles = decompose_cycles(&edges, pos_of)?;
    let mut loops = Vec::with_capacity(cycles.len());
    for cycle in cycles {
        let n = cycle.len();
        let mut loop_verts = Vec::with_capacity(n);
        for (i, &vid) in cycle.iter().enumerate() {
            let key = *key_of.get(&vid).ok_or(BoundaryError::MissingKey { vid })?;
            // Owner of the edge leaving this vid (vid -> next in the loop).
            let next = cycle[(i + 1) % n];
            let edge_owner = owner_of
                .get(&(vid, next))
                .copied()
                .ok_or(BoundaryError::MissingKey { vid })?;
            loop_verts.push(BoundaryVert {
                vid,
                key,
                edge_owner,
            });
        }
        loops.push(loop_verts);
    }
    Ok(BoundaryExtract { loops })
}

/// Per-vertex degree imbalance in the directed boundary, for the probe.
#[derive(Debug, Clone)]
pub(crate) struct Imbalance {
    pub(crate) vid: u32,
    pub(crate) key: VertexKey3,
    pub(crate) in_deg: u32,
    pub(crate) out_deg: u32,
}

/// Diagnostic snapshot of a component's directed boundary (no decomposition):
/// the collected inside-oriented edges and every vid whose in/out degree differ.
/// Used by the `S2_BOUNDARY_PROBE` harness to explain `UnbalancedVertex` failures.
#[derive(Debug, Default)]
pub(crate) struct BoundaryDiag {
    pub(crate) num_edges: usize,
    pub(crate) num_verts: usize,
    pub(crate) imbalances: Vec<Imbalance>,
}

/// Collect the boundary and report all degree imbalances (rather than bailing on
/// the first, as `extract_boundary` does). Pure diagnostic.
pub(crate) fn diagnose_boundary(
    gset: &HashSet<u32>,
    ring_candidates: &HashSet<u32>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    mode: CollectMode,
) -> Result<BoundaryDiag, BoundaryError> {
    let (edges, key_of, _owner_of) = collect_boundary_edges(
        gset,
        ring_candidates,
        cells,
        cell_indices,
        vertex_keys,
        mode,
    )?;
    let mut out_deg: HashMap<u32, u32> = HashMap::new();
    let mut in_deg: HashMap<u32, u32> = HashMap::new();
    for &(a, b) in &edges {
        *out_deg.entry(a).or_default() += 1;
        in_deg.entry(a).or_default();
        *in_deg.entry(b).or_default() += 1;
        out_deg.entry(b).or_default();
    }
    let mut imbalances = Vec::new();
    for (&vid, &od) in &out_deg {
        let id = in_deg.get(&vid).copied().unwrap_or(0);
        if od != id {
            imbalances.push(Imbalance {
                vid,
                key: key_of.get(&vid).copied().unwrap_or([u32::MAX; 3]),
                in_deg: id,
                out_deg: od,
            });
        }
    }
    imbalances.sort_unstable_by_key(|i| i.vid);
    Ok(BoundaryDiag {
        num_edges: edges.len(),
        num_verts: out_deg.len(),
        imbalances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Normalize a cycle to start at its minimum vid (rotation-invariant) for
    /// comparison, preserving direction.
    fn canon(cycle: &[u32]) -> Vec<u32> {
        let n = cycle.len();
        let start = (0..n).min_by_key(|&i| cycle[i]).unwrap();
        (0..n).map(|k| cycle[(start + k) % n]).collect()
    }

    fn sorted_canon(cycles: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut v: Vec<Vec<u32>> = cycles.iter().map(|c| canon(c)).collect();
        v.sort();
        v
    }

    // Positions on a small cap near the north pole, addressed by vid. A vid's
    // position is placed at planar angle `theta` (deg) and a small tilt off the
    // pole, then normalized — enough to give each vid a distinct tangent-plane
    // bearing as seen from a shared pinch vertex at the pole.
    fn cap_pos(theta_deg: f64, tilt: f64) -> Vec3 {
        let t = theta_deg.to_radians();
        Vec3::new((tilt * t.cos()) as f32, (tilt * t.sin()) as f32, 1.0).normalize()
    }

    #[test]
    fn single_square_one_loop() {
        // 0->1->2->3->0
        let edges = [(0u32, 1u32), (1, 2), (2, 3), (3, 0)];
        let cycles = decompose_cycles(&edges, |_| Some(Vec3::Z)).unwrap();
        assert_eq!(sorted_canon(&cycles), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn two_disjoint_loops() {
        let edges = [
            (0u32, 1u32),
            (1, 2),
            (2, 0), // triangle A
            (3, 4),
            (4, 5),
            (5, 3), // triangle B
        ];
        let cycles = decompose_cycles(&edges, |_| Some(Vec3::Z)).unwrap();
        assert_eq!(sorted_canon(&cycles), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn figure_eight_pinch_splits_into_two_loops() {
        // Vertex 0 is shared by two triangles in disjoint angular sectors at 0.
        // Triangle 1: 0->1->2->0 around bearing ~0deg; triangle 2: 0->3->4->0
        // around bearing ~180deg. The angular rule must keep the sectors apart:
        // arriving 2->0 continues to 1 (not 3), arriving 4->0 continues to 3.
        //
        // Edge order matters for the test: starting on (1,2) forces the walk to
        // PASS THROUGH the pinch vertex 0 mid-loop (rather than closing trivially
        // at it), which is the only path that exercises the angular branch.
        let edges = [(1u32, 2u32), (2, 0), (0, 1), (4, 0), (0, 3), (3, 4)];
        let pos = |vid: u32| {
            Some(match vid {
                0 => cap_pos(0.0, 0.0),    // the pinch, at the pole
                1 => cap_pos(20.0, 0.02),  // triangle 1 sector
                2 => cap_pos(-20.0, 0.02), // triangle 1 sector
                3 => cap_pos(160.0, 0.02), // triangle 2 sector
                4 => cap_pos(200.0, 0.02), // triangle 2 sector
                _ => unreachable!(),
            })
        };
        let cycles = decompose_cycles(&edges, pos).unwrap();
        let got = sorted_canon(&cycles);
        assert_eq!(
            got,
            vec![vec![0, 1, 2], vec![0, 3, 4]],
            "pinch must split into the two embedded triangles, got {got:?}"
        );
    }

    #[test]
    fn unbalanced_vertex_is_reported() {
        let edges = [(0u32, 1u32)]; // 0 out=1/in=0, 1 out=0/in=1
        let err = decompose_cycles(&edges, |_| Some(Vec3::Z)).unwrap_err();
        assert!(matches!(err, BoundaryError::UnbalancedVertex { .. }));
    }

    #[test]
    fn missing_position_only_when_pinch_needs_it() {
        // Two triangles sharing vertex 0 (a pinch) -> needs positions. Returning
        // None for the pinch vertex must surface MissingPosition, not a wrong
        // pairing. Edge order forces a pass-through of the pinch (see the
        // figure-eight test).
        let edges = [(1u32, 2u32), (2, 0), (0, 1), (4, 0), (0, 3), (3, 4)];
        let err = decompose_cycles(&edges, |_| None).unwrap_err();
        assert!(matches!(err, BoundaryError::MissingPosition { .. }));
    }

    // ---- collect_boundary_edges with a hand-built mini diagram ----

    /// Build a `ShardedVertexKeys` from a flat key list (single shard).
    fn keys_from(flat: &[VertexKey3]) -> ShardedVertexKeys {
        ShardedVertexKeys::new(vec![0, flat.len() as u32], vec![flat.to_vec()])
    }

    /// One contested triangle cell g=0 with vertices V0,V1,V2 surrounded by ring
    /// cells 1,2,3. Each ring cell shares exactly one edge with cell 0:
    ///   - ring 2 owns edge (V0,V1)  [common pair {0,2}]
    ///   - ring 3 owns edge (V1,V2)  [common pair {0,3}]
    ///   - ring 1 owns edge (V2,V0)  [common pair {0,1}]
    /// so the boundary of C is the single triangle loop, inside-on-left.
    #[test]
    fn collect_and_extract_single_triangle() {
        // vids:  V0=0, V1=1, V2=2, plus outer ring-only vertices 3,4,5.
        // keys keyed by vid index:
        let keys = [
            [0u32, 1, 2], // 0: V0  (cells 0,1,2)
            [0, 2, 3],    // 1: V1  (cells 0,2,3)
            [0, 3, 1],    // 2: V2  (cells 0,3,1)
            [1, 2, 6],    // 3: outer vertex for ring cell 2 / ring cell 1
            [2, 3, 7],    // 4: outer vertex for ring cell 3 / ring cell 2
            [3, 1, 8],    // 5: outer vertex for ring cell 1 / ring cell 3
        ];
        let vk = keys_from(&keys);

        // Cells. Cell 0 (contested) = [V0,V1,V2]. Ring cells wound so the
        // contested region is OUTSIDE them, i.e. each traverses its shared edge
        // opposite to cell 0's CCW order. Cell 0 CCW: V0->V1->V2.
        //   ring 2 must traverse V1->V0 (reverse of V0->V1)
        //   ring 3 must traverse V2->V1
        //   ring 1 must traverse V0->V2
        // Each ring cell is a triangle: shared edge + one outer vertex.
        //   ring 1 (g=1): owns edge (V2,V0)->traverse V0->V2; verts V0,V2,outer5
        //   ring 2 (g=2): owns edge (V0,V1)->traverse V1->V0; verts V1,V0,outer3
        //   ring 3 (g=3): owns edge (V1,V2)->traverse V2->V1; verts V2,V1,outer4
        // cell_indices flat, cells reference spans.
        let mut cell_indices: Vec<u32> = Vec::new();
        let mut cells: Vec<VoronoiCell> = Vec::new();
        let push_cell = |idx: &mut Vec<u32>, cells: &mut Vec<VoronoiCell>, span: &[u32]| {
            let start = idx.len() as u32;
            idx.extend_from_slice(span);
            cells.push(VoronoiCell::new(start, span.len() as u16));
        };
        push_cell(&mut cell_indices, &mut cells, &[0, 1, 2]); // g0
        push_cell(&mut cell_indices, &mut cells, &[0, 2, 5]); // g1: V0->V2->outer5
        push_cell(&mut cell_indices, &mut cells, &[1, 0, 3]); // g2: V1->V0->outer3
        push_cell(&mut cell_indices, &mut cells, &[2, 1, 4]); // g3: V2->V1->outer4

        // Positions (only used if a pinch occurs; here all degree 1). Put the
        // three boundary vids around the pole.
        let vertices = vec![
            cap_pos(90.0, 0.05),  // V0
            cap_pos(210.0, 0.05), // V1
            cap_pos(330.0, 0.05), // V2
            cap_pos(150.0, 0.1),  // outer3
            cap_pos(270.0, 0.1),  // outer4
            cap_pos(30.0, 0.1),   // outer5
        ];

        let gset: HashSet<u32> = [0u32].into_iter().collect();
        let ring: HashSet<u32> = [1u32, 2, 3].into_iter().collect();

        // Both collection modes agree on this clean fixture: the inside-on-left
        // loop around g0 is V0->V1->V2.
        for mode in [CollectMode::Keys, CollectMode::Paired] {
            let extract =
                extract_boundary(&gset, &ring, &cells, &cell_indices, &vertices, &vk, mode)
                    .unwrap();
            assert_eq!(extract.loops.len(), 1, "one boundary loop ({mode:?})");
            let vids: Vec<u32> = extract.loops[0].iter().map(|bv| bv.vid).collect();
            assert_eq!(canon(&vids), vec![0, 1, 2], "{mode:?} got {vids:?}");
            for bv in &extract.loops[0] {
                assert_eq!(bv.key, keys[bv.vid as usize]);
            }
        }
    }

    /// Paired mode is robust to a ring vertex whose KEY is degenerate (shares only
    /// one generator with its neighbor, so `key_common_pair` would drop the edge):
    /// the edge-pairing domain recovers the boundary from actual vid-pairs. Here
    /// ring cell 2's vertex V1 carries a "wrong" 3rd generator (99) so its key no
    /// longer shares a clean 2-gen pair across the V0–V1 edge, yet the vid-pair
    /// V0–V1 is still cleanly used by cell 0 (V0->V1) and cell 2 (V1->V0).
    #[test]
    fn paired_mode_survives_degenerate_key() {
        let keys = [
            [0u32, 1, 2], // 0: V0
            [0, 2, 99], // 1: V1  -- degenerate 3rd gen (not 3): breaks key pair on V0-V1 and V1-V2
            [0, 3, 1],  // 2: V2
            [1, 2, 6],  // 3
            [2, 3, 7],  // 4
            [3, 1, 8],  // 5
        ];
        let vk = keys_from(&keys);
        let mut cell_indices: Vec<u32> = Vec::new();
        let mut cells: Vec<VoronoiCell> = Vec::new();
        let push_cell = |idx: &mut Vec<u32>, cells: &mut Vec<VoronoiCell>, span: &[u32]| {
            let start = idx.len() as u32;
            idx.extend_from_slice(span);
            cells.push(VoronoiCell::new(start, span.len() as u16));
        };
        push_cell(&mut cell_indices, &mut cells, &[0, 1, 2]); // g0 (C): V0->V1->V2
        push_cell(&mut cell_indices, &mut cells, &[0, 2, 5]); // g1: V0->V2->5
        push_cell(&mut cell_indices, &mut cells, &[1, 0, 3]); // g2: V1->V0->3
        push_cell(&mut cell_indices, &mut cells, &[2, 1, 4]); // g3: V2->V1->4
        let vertices = vec![
            cap_pos(90.0, 0.05),
            cap_pos(210.0, 0.05),
            cap_pos(330.0, 0.05),
            cap_pos(150.0, 0.1),
            cap_pos(270.0, 0.1),
            cap_pos(30.0, 0.1),
        ];
        let gset: HashSet<u32> = [0u32].into_iter().collect();
        let ring: HashSet<u32> = [1u32, 2, 3].into_iter().collect();

        // Paired mode reconstructs the full triangle loop despite the bad key.
        let paired = extract_boundary(
            &gset,
            &ring,
            &cells,
            &cell_indices,
            &vertices,
            &vk,
            CollectMode::Paired,
        )
        .unwrap();
        assert_eq!(paired.loops.len(), 1);
        let vids: Vec<u32> = paired.loops[0].iter().map(|bv| bv.vid).collect();
        assert_eq!(canon(&vids), vec![0, 1, 2], "paired got {vids:?}");

        // Key mode drops the V0-V1 and V1-V2 edges (no clean common pair) and so
        // dangles / fails to close — demonstrating the fragility paired mode fixes.
        let key_err = extract_boundary(
            &gset,
            &ring,
            &cells,
            &cell_indices,
            &vertices,
            &vk,
            CollectMode::Keys,
        );
        assert!(
            key_err.is_err(),
            "key mode should fail on the degenerate key, got {key_err:?}"
        );
    }
}
