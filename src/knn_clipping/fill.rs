//! Fill stage for the Tier-2 totality patch synthesizer (spherical).
//!
//! Consumes the clean oriented boundary loops produced by [`super::boundary`] and
//! re-tiles the enclosed region of a contested component `C` with one cell per
//! generator, reproducing every boundary edge verbatim (so the seam stays paired
//! with the untouched ring) and filling the interior with fresh synthetic Voronoi
//! vertices (empty-circumcircle circumcenters of the `C` generators).
//!
//! This is "Option B": near-Voronoi local assembly. It is the same interior +
//! per-cell-cycle assembly the older `resolve_component_attempt` uses, but the
//! boundary is fed in CLEAN from the edge-domain extractor instead of being
//! re-derived (fragilely) from ring-cell key pairs. A crude validity-only fill
//! ("Option A") is the fallback when this bails; both sit behind the whole-diagram
//! validate-or-revert gate, so totality never ships invalid topology.
//!
//! Output is a [`FilledPatch`]: per-cell polygons over first-class vertex refs
//! ([`FillRef`]) plus the synthetic interior vertex positions. `repair`
//! materializes it by appending the synthetics and emitting each cell's span.

use super::boundary::BoundaryVert;
use super::reclip_repair::{build_cycle_from_edges, delaunay, jitter_unit, key3};
use crate::cube_grid::CubeMapGrid;
use glam::{DVec3, Vec3};
use std::collections::{HashMap, HashSet};

type VertexKey3 = [u32; 3];
type KeyEdge = (VertexKey3, VertexKey3);

/// A first-class vertex reference in a filled patch: either an existing global
/// vertex id (a pinned boundary vertex, shared with the untouched ring) or a
/// fresh synthetic interior vertex (indexed into [`FilledPatch::synthetic`]).
/// Replaces the older "fake generator-triple key" representation so the crude
/// fill can introduce a hub vertex that is not any generator triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum FillRef {
    Existing(u32),
    Synthetic(u32),
}

/// A re-tiled contested component: one simple-cycle polygon per generator over
/// first-class vertex refs, plus the positions of the synthetic interior
/// vertices. `repair` materializes this by appending `synthetic` to the vertex
/// array and emitting each cell's span.
#[derive(Debug, Default)]
pub(crate) struct FilledPatch {
    pub(crate) polys: Vec<(u32, Vec<FillRef>)>,
    pub(crate) synthetic: Vec<Vec3>,
}

/// Why the near-Voronoi fill could not assemble a clean component (caller falls
/// back to the crude fill / leaves the component as residual).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FillError {
    /// Two distinct boundary vids collapsed onto the same generator-triple key —
    /// the key-space assembly cannot represent them separately.
    KeyCollision { key: VertexKey3 },
    /// A generator in C received no usable edges (e.g. a fully-interior generator
    /// the empty-circumcircle interior did not surround, or one whose edges were
    /// all pruned).
    NoEdges { gen: u32 },
    /// A generator's edge set does not assemble into one simple cycle.
    OpenCell { gen: u32 },
    /// A generator id is out of range for the point set.
    BadGenerator { gen: u32 },
    /// The crude fill cannot tile a multi-loop (annulus/holes) region yet.
    MultiLoopUnsupported { loops: usize },
    /// The crude fill needs `m >= 3` generators and `k >= m` boundary edges.
    CrudeUnsupported { m: usize, k: usize },
}

/// Jittered unit position of a generator: a deterministic ~1e-9 perturbation
/// (shared with `reclip_repair`) so no triple is exactly cocircular/coincident,
/// making the degenerate-split choice jitter-determined but consistent.
#[inline]
fn gjit(points: &[Vec3], g: u32) -> DVec3 {
    let p = points[g as usize];
    let base = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
    const J: f64 = 1e-9;
    let jit = DVec3::new(jitter_unit(g, 0), jitter_unit(g, 1), jitter_unit(g, 2)) * J;
    (base + jit).normalize()
}

/// Re-tile component `C` (the generators in `gset`) given its clean oriented
/// boundary `loops`. Returns per-cell key polygons + interior positions + boundary
/// pins, or a [`FillError`] to bail.
pub(crate) fn fill_component(
    gset: &HashSet<u32>,
    loops: &[Vec<BoundaryVert>],
    points: &[Vec3],
    grid: &CubeMapGrid,
) -> Result<FilledPatch, FillError> {
    let mut gvec: Vec<u32> = gset.iter().copied().collect();
    gvec.sort_unstable();
    if let Some(&g) = gvec.iter().find(|&&g| (g as usize) >= points.len()) {
        return Err(FillError::BadGenerator { gen: g });
    }

    // 1. Boundary from the extracted loops. Each loop edge (cur -> next) belongs
    //    to cur.edge_owner (a C generator). Keys pin to existing vids.
    let mut boundary_pin: HashMap<VertexKey3, u32> = HashMap::new();
    let mut boundary_edges: HashMap<u32, Vec<KeyEdge>> = HashMap::new();
    let mut boundary_of: HashMap<u32, Vec<VertexKey3>> = HashMap::new();
    for lp in loops {
        let n = lp.len();
        for i in 0..n {
            let cur = lp[i];
            let next = lp[(i + 1) % n];
            if let Some(&v) = boundary_pin.get(&cur.key) {
                if v != cur.vid {
                    return Err(FillError::KeyCollision { key: cur.key });
                }
            }
            boundary_pin.insert(cur.key, cur.vid);
            let g = cur.edge_owner;
            boundary_edges
                .entry(g)
                .or_default()
                .push((cur.key, next.key));
            boundary_of
                .entry(g)
                .or_default()
                .extend([cur.key, next.key]);
        }
    }
    for keys in boundary_of.values_mut() {
        keys.sort_unstable();
        keys.dedup();
    }

    // 2. Interior Voronoi vertices: every all-C triple whose circumcircle is empty
    //    of the complete local neighborhood (grid 3x3 + ring-2 of every C gen).
    //    Brute force over triples (no triangulation, so no generator is orphaned);
    //    jitter removes cocircular over-counting.
    let filter = neighborhood_filter(&gvec, points, grid);
    let gp: Vec<DVec3> = gvec.iter().map(|&g| gjit(points, g)).collect();
    const EMPTY_TOL: f64 = 1e-12;
    let mut interior_pos: HashMap<VertexKey3, Vec3> = HashMap::new();
    for i in 0..gvec.len() {
        for j in (i + 1)..gvec.len() {
            for k in (j + 1)..gvec.len() {
                let Some(cc) = delaunay::circumcenter(gp[i], gp[j], gp[k]) else {
                    continue;
                };
                let g3 = [gvec[i], gvec[j], gvec[k]];
                let radius = cc.dot(gp[i]);
                let empty = filter
                    .iter()
                    .all(|&(d, dp)| g3.contains(&d) || cc.dot(dp) <= radius + EMPTY_TOL);
                if !empty {
                    continue;
                }
                interior_pos.insert(
                    key3(g3[0], g3[1], g3[2]),
                    Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32),
                );
            }
        }
    }

    // 3. Prune interior vertices whose implied C-pairs lack a second endpoint
    //    (else assembly would invent a one-sided edge). Repeat to a fixed point.
    prune_unsupported(&mut interior_pos, &boundary_of, gset);

    // 4. Per-cell edges. Boundary (rim) edges plus interior/closing edges: for
    //    every C-generator pair, the vertices (boundary or interior) that contain
    //    BOTH generators form that shared Voronoi edge.
    let mut cell_edges = boundary_edges;
    let mut pair_keys: HashMap<(u32, u32), Vec<VertexKey3>> = HashMap::new();
    let mut all_keys: HashSet<VertexKey3> = HashSet::new();
    for keys in boundary_of.values() {
        all_keys.extend(keys.iter().copied());
    }
    all_keys.extend(interior_pos.keys().copied());
    for key in all_keys {
        let members: Vec<u32> = key.into_iter().filter(|g| gset.contains(g)).collect();
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let a = members[i].min(members[j]);
                let b = members[i].max(members[j]);
                pair_keys.entry((a, b)).or_default().push(key);
            }
        }
    }
    for (_, keys) in pair_keys.iter_mut() {
        keys.sort_unstable();
        keys.dedup();
    }
    let mut multi_pairs = 0usize;
    for (&(a, b), keys) in &pair_keys {
        if keys.len() == 2 {
            cell_edges.entry(a).or_default().push((keys[0], keys[1]));
            cell_edges.entry(b).or_default().push((keys[0], keys[1]));
        } else if keys.len() > 2 {
            // A multi-vertex shared edge: order the vertices along the a|b bisector
            // and connect consecutively (a polyline), rather than dropping it.
            multi_pairs += 1;
            let chain = order_along_bisector(keys, a, b, points);
            for w in chain.windows(2) {
                cell_edges.entry(a).or_default().push((w[0], w[1]));
                cell_edges.entry(b).or_default().push((w[0], w[1]));
            }
        }
    }
    let _ = multi_pairs; // measured via the probe; no behavioral use here.
    for edges in cell_edges.values_mut() {
        for (a, b) in edges.iter_mut() {
            if *b < *a {
                std::mem::swap(a, b);
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }

    // 5. Assemble each cell into one simple cycle.
    let pos_of = |key: VertexKey3| {
        delaunay::circumcenter(
            gjit(points, key[0]),
            gjit(points, key[1]),
            gjit(points, key[2]),
        )
    };
    let mut key_polys: Vec<(u32, Vec<VertexKey3>)> = Vec::with_capacity(gvec.len());
    for &g in &gvec {
        let Some(edges) = cell_edges.get(&g) else {
            return Err(FillError::NoEdges { gen: g });
        };
        let mut verts: Vec<VertexKey3> = Vec::new();
        for &(a, b) in edges {
            verts.extend([a, b]);
        }
        verts.sort_unstable();
        verts.dedup();
        let gc = gjit(points, g);
        let Some(poly) = build_cycle_from_edges(g, gc, &verts, edges, pos_of) else {
            return Err(FillError::OpenCell { gen: g });
        };
        key_polys.push((g, poly));
    }

    // Map key polygons to first-class FillRefs: boundary keys pin to existing vids,
    // interior keys become synthetic vertices in deterministic key order.
    let mut interior_keys: Vec<VertexKey3> = interior_pos.keys().copied().collect();
    interior_keys.sort_unstable();
    let mut synth_index: HashMap<VertexKey3, u32> = HashMap::new();
    let mut synthetic: Vec<Vec3> = Vec::with_capacity(interior_keys.len());
    for k in interior_keys {
        synth_index.insert(k, synthetic.len() as u32);
        synthetic.push(interior_pos[&k]);
    }
    let mut polys: Vec<(u32, Vec<FillRef>)> = Vec::with_capacity(key_polys.len());
    for (g, poly) in key_polys {
        let mut refs = Vec::with_capacity(poly.len());
        for key in poly {
            let r = if let Some(&vid) = boundary_pin.get(&key) {
                FillRef::Existing(vid)
            } else if let Some(&si) = synth_index.get(&key) {
                FillRef::Synthetic(si)
            } else {
                // Should not happen: every poly key is a boundary pin or interior.
                return Err(FillError::OpenCell { gen: g });
            };
            refs.push(r);
        }
        polys.push((g, refs));
    }

    Ok(FilledPatch { polys, synthetic })
}

/// Crude validity-only fill (Option A, the totality fallback). Topology-only:
/// the validator checks no geometry, so a wedge fan from a single synthetic hub
/// is valid for ANY single loop regardless of region shape. Partitions the loop
/// into `m` contiguous arcs (one per generator) and emits `[hub, arc...]` wedges;
/// consecutive wedges share a radial edge (paired), every boundary edge is
/// reproduced once, the hub has incidence `m`.
///
/// Limitations (return `FillError`, caller leaves the component as residual):
/// multi-loop regions (an annulus the single hub cannot tile — handled later by
/// bridging), too few boundary edges for one-per-generator (`k < m`), or `m < 3`
/// (hub would be under-incident).
pub(crate) fn fill_component_crude(
    gset: &HashSet<u32>,
    loops: &[Vec<BoundaryVert>],
    points: &[Vec3],
) -> Result<FilledPatch, FillError> {
    if loops.len() != 1 {
        return Err(FillError::MultiLoopUnsupported { loops: loops.len() });
    }
    let lp = &loops[0];
    let k = lp.len();
    let mut gvec: Vec<u32> = gset.iter().copied().collect();
    gvec.sort_unstable();
    let m = gvec.len();
    if let Some(&g) = gvec.iter().find(|&&g| (g as usize) >= points.len()) {
        return Err(FillError::BadGenerator { gen: g });
    }
    if m < 3 || k < m {
        return Err(FillError::CrudeUnsupported { m, k });
    }

    // Hub at the normalized centroid of the component's generators (inside the
    // region; geometric quality is not required, only on-sphere + non-antipodal
    // edges, which hold for a small cap component).
    let mut c = Vec3::ZERO;
    for &g in &gvec {
        c += points[g as usize];
    }
    let hub = if c.length_squared() > 1e-20 {
        c.normalize()
    } else {
        points[gvec[0] as usize]
    };
    let synthetic = vec![hub];
    let hub_ref = FillRef::Synthetic(0);

    // Contiguous arc cut points: distribute k edges among m arcs as evenly as
    // possible (each >= 1 edge since k >= m).
    let mut cuts: Vec<usize> = (0..m).map(|i| i * k / m).collect();
    cuts.push(k);

    let mut polys: Vec<(u32, Vec<FillRef>)> = Vec::with_capacity(m);
    for i in 0..m {
        let (start, end) = (cuts[i], cuts[i + 1]); // edges [start, end); vertices start..=end
        let mut refs = Vec::with_capacity(end - start + 2);
        refs.push(hub_ref);
        for j in start..=end {
            refs.push(FillRef::Existing(lp[j % k].vid));
        }
        polys.push((gvec[i], refs));
    }
    Ok(FilledPatch { polys, synthetic })
}

/// Try the near-Voronoi fill (B); on any failure fall back to the crude fill (A).
/// Returns the patch and whether the fallback was used, or `None` if both bail.
pub(crate) fn fill_or_fallback(
    gset: &HashSet<u32>,
    loops: &[Vec<BoundaryVert>],
    points: &[Vec3],
    grid: &CubeMapGrid,
) -> Option<(FilledPatch, bool)> {
    match fill_component(gset, loops, points, grid) {
        Ok(p) => Some((p, false)),
        Err(_) => fill_component_crude(gset, loops, points)
            .ok()
            .map(|p| (p, true)),
    }
}

/// Verify a filled patch against (almost) every validator invariant, locally and
/// without mutation — so the probe's success count reflects what the whole-diagram
/// gate will actually accept. Mirrors `verify_sphere_effective_strict` except for
/// global Euler/connectivity (left to the gate). Hardened per Codex review:
///
/// - exactly one cell per generator (no duplicate / missing generator);
/// - each cell a simple cycle (>=3 distinct vertices, no self-loop, no dup vid);
/// - **directed** boundary reproduction: each loop edge `cur -> next` appears once,
///   in the inside-on-left direction (catches B's signed-area rewind flipping a
///   seam edge — the local check Codex flagged was missing);
/// - interior edges used exactly twice, opposite orientation;
/// - no antipodal edge (fatal to the validator);
/// - **global incidence >= 3** for every referenced vertex: boundary vids count
///   `ring_ref_count` (non-C cells referencing them, precomputed) + patch cells;
///   synthetic vids count patch cells. This is the check that catches the crude
///   fill's owner-transition `{g1,g2,R}` vertex dropping to incidence 2.
/// - no duplicate patch-cell signature.
///
/// `vertices` provides positions for `Existing` refs (antipodal test);
/// `ring_ref_count` maps a boundary vid to the number of non-C cells referencing
/// it (so post-patch incidence is exact).
pub(crate) fn fill_check(
    patch: &FilledPatch,
    loops: &[Vec<BoundaryVert>],
    gset: &HashSet<u32>,
    vertices: &[Vec3],
    ring_ref_count: &HashMap<u32, u32>,
) -> Result<(), String> {
    // Synthetic positions on-sphere + refs in range.
    for (i, v) in patch.synthetic.iter().enumerate() {
        if (v.length_squared() - 1.0).abs() > 1e-4 {
            return Err(format!("synthetic vertex {i} not on sphere"));
        }
    }

    // Exactly one cell per generator.
    let mut gens_seen: HashSet<u32> = HashSet::new();
    for (g, _) in &patch.polys {
        if !gset.contains(g) {
            return Err(format!("cell for non-component generator {g}"));
        }
        if !gens_seen.insert(*g) {
            return Err(format!("duplicate cell for generator {g}"));
        }
    }
    if gens_seen.len() != gset.len() {
        return Err(format!(
            "filled {}/{} generators",
            gens_seen.len(),
            gset.len()
        ));
    }

    let pos = |r: FillRef| -> Option<Vec3> {
        match r {
            FillRef::Existing(v) => vertices.get(v as usize).copied(),
            FillRef::Synthetic(i) => patch.synthetic.get(i as usize).copied(),
        }
    };
    let undir = |a: FillRef, b: FillRef| if a <= b { (a, b) } else { (b, a) };

    // Loop directed edges (inside-on-left): the patch must traverse each exactly
    // this way so it pairs with the ring's reverse.
    let mut loop_dir: HashSet<(u32, u32)> = HashSet::new();
    let mut loop_undir: HashSet<(FillRef, FillRef)> = HashSet::new();
    for lp in loops {
        let n = lp.len();
        for i in 0..n {
            let (a, b) = (lp[i].vid, lp[(i + 1) % n].vid);
            loop_dir.insert((a, b));
            loop_undir.insert(undir(FillRef::Existing(a), FillRef::Existing(b)));
        }
    }

    // Per-vertex patch incidence (distinct cells) + edge uses + signatures.
    let mut incidence: HashMap<FillRef, u32> = HashMap::new();
    let mut uses: HashMap<(FillRef, FillRef), (u32, u32)> = HashMap::new(); // (count, forward)
    let mut signatures: HashSet<Vec<FillRef>> = HashSet::new();
    for (g, poly) in &patch.polys {
        let n = poly.len();
        if n < 3 {
            return Err(format!("cell {g} has {n} < 3 vertices"));
        }
        let mut distinct: Vec<FillRef> = poly.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != n {
            return Err(format!("duplicate vertex in cell {g}"));
        }
        if !signatures.insert(distinct.clone()) {
            return Err(format!("duplicate cell signature (cell {g})"));
        }
        for &r in &distinct {
            *incidence.entry(r).or_default() += 1;
        }
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            if a == b {
                return Err(format!("self-loop edge in cell {g}"));
            }
            // Antipodal check.
            if let (Some(pa), Some(pb)) = (pos(a), pos(b)) {
                if pa.dot(pb) <= -1.0 + 1e-6 {
                    return Err(format!("antipodal edge in cell {g}"));
                }
            } else {
                return Err(format!("edge with no position in cell {g}"));
            }
            let (lo, hi) = undir(a, b);
            let fwd = u32::from((lo, hi) == (a, b));
            let e = uses.entry((lo, hi)).or_insert((0, 0));
            e.0 += 1;
            e.1 += fwd;
        }
    }

    // Edge pairing + directed boundary reproduction.
    let mut seen_loop: HashSet<(FillRef, FillRef)> = HashSet::new();
    for (&(lo, hi), &(count, fwd)) in &uses {
        if loop_undir.contains(&(lo, hi)) {
            if count != 1 {
                return Err(format!("boundary edge used {count}x inside (want 1)"));
            }
            // Direction must be the inside-on-left one.
            let (FillRef::Existing(a), FillRef::Existing(b)) = (lo, hi) else {
                return Err("boundary edge with synthetic endpoint".into());
            };
            let directed = if fwd == 1 { (a, b) } else { (b, a) };
            if !loop_dir.contains(&directed) {
                return Err(format!("boundary edge {directed:?} reversed inside"));
            }
            seen_loop.insert((lo, hi));
        } else if count != 2 || fwd != 1 {
            return Err(format!(
                "interior edge used {count}x (fwd={fwd}); want 2 opposite"
            ));
        }
    }
    if seen_loop.len() != loop_undir.len() {
        return Err(format!(
            "reproduced {}/{} boundary edges",
            seen_loop.len(),
            loop_undir.len()
        ));
    }

    // Global incidence >= 3 for every referenced vertex.
    for (&r, &patch_refs) in &incidence {
        let total = match r {
            FillRef::Existing(v) => patch_refs + ring_ref_count.get(&v).copied().unwrap_or(0),
            FillRef::Synthetic(_) => patch_refs,
        };
        if total < 3 {
            return Err(format!("vertex {r:?} incidence {total} < 3"));
        }
    }
    Ok(())
}

/// The complete local neighborhood (grid cell + 3x3 + ring-2) of every C
/// generator, with jittered positions — the emptiness filter for interior
/// circumcenters. Using the grid (not the buggy cell keys) is essential: a
/// key-derived filter would be incomplete and let spurious triples pass.
fn neighborhood_filter(gvec: &[u32], points: &[Vec3], grid: &CubeMapGrid) -> Vec<(u32, DVec3)> {
    let mut filter: Vec<(u32, DVec3)> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut add = |x: u32, filter: &mut Vec<(u32, DVec3)>| {
        if (x as usize) < points.len() && seen.insert(x) {
            filter.push((x, gjit(points, x)));
        }
    };
    for &g in gvec {
        let cell = grid.point_index_to_cell(g as usize);
        for &nc in grid.cell_neighbors(cell).iter() {
            if nc != u32::MAX {
                for &pt in grid.cell_points(nc as usize) {
                    add(pt, &mut filter);
                }
            }
        }
        for &nc in grid.cell_ring2(cell) {
            if nc != u32::MAX {
                for &pt in grid.cell_points(nc as usize) {
                    add(pt, &mut filter);
                }
            }
        }
    }
    filter
}

/// Drop interior vertices whose implied component-component edges lack a second
/// endpoint, to a fixed point. An interior triple `{a,b,c}` is kept only if each
/// of its three C-pairs is shared by at least two vertices (so it can form an
/// edge rather than a dangling stub).
fn prune_unsupported(
    interior_pos: &mut HashMap<VertexKey3, Vec3>,
    boundary_of: &HashMap<u32, Vec<VertexKey3>>,
    gset: &HashSet<u32>,
) {
    loop {
        let mut pair_count: HashMap<(u32, u32), u32> = HashMap::new();
        let mut seen: HashSet<VertexKey3> = HashSet::new();
        let bump = |key: VertexKey3, pc: &mut HashMap<(u32, u32), u32>| {
            let members: Vec<u32> = key.into_iter().filter(|g| gset.contains(g)).collect();
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    let a = members[i].min(members[j]);
                    let b = members[i].max(members[j]);
                    *pc.entry((a, b)).or_default() += 1;
                }
            }
        };
        for keys in boundary_of.values() {
            for &key in keys {
                if seen.insert(key) {
                    bump(key, &mut pair_count);
                }
            }
        }
        for &key in interior_pos.keys() {
            bump(key, &mut pair_count);
        }
        let before = interior_pos.len();
        interior_pos.retain(|key, _| {
            (0..3).all(|i| {
                ((i + 1)..3).all(|j| {
                    let a = key[i].min(key[j]);
                    let b = key[i].max(key[j]);
                    pair_count.get(&(a, b)).copied().unwrap_or(0) >= 2
                })
            })
        });
        if interior_pos.len() == before {
            break;
        }
    }
}

/// Order the vertices sharing a generator pair `(a,b)` along their common Voronoi
/// edge (the `a|b` great-circle bisector), so a multi-vertex shared edge can be
/// chained consecutively. Each vertex's position is its key's circumcenter; sort
/// by projection onto an axis lying in the bisector plane.
fn order_along_bisector(keys: &[VertexKey3], a: u32, b: u32, points: &[Vec3]) -> Vec<VertexKey3> {
    let axis = gjit(points, a).cross(gjit(points, b));
    let axis = if axis.length_squared() > 0.0 {
        axis.normalize()
    } else {
        return keys.to_vec();
    };
    let mut keyed: Vec<(f64, VertexKey3)> = keys
        .iter()
        .map(|&k| {
            let pos =
                delaunay::circumcenter(gjit(points, k[0]), gjit(points, k[1]), gjit(points, k[2]));
            let t = pos.map(|p| p.dot(axis)).unwrap_or(0.0);
            (t, k)
        })
        .collect();
    keyed.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    keyed.into_iter().map(|(_, k)| k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bv(vid: u32, owner: u32) -> BoundaryVert {
        BoundaryVert {
            vid,
            key: [vid, 0, 0],
            edge_owner: owner,
        }
    }

    fn cap(theta_deg: f64) -> Vec3 {
        let t = theta_deg.to_radians();
        Vec3::new((0.1 * t.cos()) as f32, (0.1 * t.sin()) as f32, 1.0).normalize()
    }

    /// Crude fill of a single square loop with 4 generators: 4 wedge cells from
    /// one hub, locally valid (every boundary edge once, radial edges paired).
    #[test]
    fn crude_single_loop_is_valid() {
        // Generators 10..=13 near the pole; loop vids 0..=3.
        let mut points = vec![Vec3::Z; 14];
        for (i, g) in (10u32..14).enumerate() {
            points[g as usize] = cap(i as f64 * 90.0 + 45.0);
        }
        let gset: HashSet<u32> = (10u32..14).collect();
        let loops = vec![vec![bv(0, 10), bv(1, 11), bv(2, 12), bv(3, 13)]];

        let patch = fill_component_crude(&gset, &loops, &points).unwrap();
        assert_eq!(patch.polys.len(), 4, "one cell per generator");
        assert_eq!(patch.synthetic.len(), 1, "one hub vertex");
        for (_, poly) in &patch.polys {
            assert!(poly.len() >= 3);
            assert!(poly.contains(&FillRef::Synthetic(0)), "hub in every wedge");
        }
        // Positions for the 4 loop vids (0..3); each is an arc endpoint shared by
        // 2 wedges, so 1 ring ref each gives incidence 3.
        let vertices: Vec<Vec3> = (0..4).map(|i| cap(i as f64 * 90.0)).collect();
        let ring_ref_count: HashMap<u32, u32> = (0u32..4).map(|v| (v, 1)).collect();
        fill_check(&patch, &loops, &gset, &vertices, &ring_ref_count)
            .expect("crude patch must be locally valid");
    }

    /// `fill_check` rejects a patch whose interior edge is used only once.
    #[test]
    fn fill_check_rejects_unpaired_interior() {
        // Two triangles sharing the hub but NOT a common interior edge: hub-0 is
        // used once (only by cell A), so it is an unpaired interior edge.
        let loops = vec![vec![bv(1, 0), bv(2, 0), bv(3, 0)]];
        let patch = FilledPatch {
            polys: vec![
                (
                    0,
                    vec![
                        FillRef::Synthetic(0),
                        FillRef::Existing(1),
                        FillRef::Existing(2),
                    ],
                ),
                (
                    1,
                    vec![
                        FillRef::Existing(2),
                        FillRef::Existing(3),
                        FillRef::Existing(1),
                    ],
                ),
            ],
            synthetic: vec![Vec3::Z],
        };
        let gset: HashSet<u32> = [0u32, 1].into_iter().collect();
        let vertices: Vec<Vec3> = (0..4).map(|i| cap(i as f64 * 90.0)).collect();
        let ring_ref_count: HashMap<u32, u32> = HashMap::new();
        assert!(fill_check(&patch, &loops, &gset, &vertices, &ring_ref_count).is_err());
    }

    /// Crude fill bails (not panics) on multi-loop and on too-few-edges.
    #[test]
    fn crude_bails_on_unsupported() {
        let points = vec![Vec3::Z; 16];
        let gset: HashSet<u32> = (10u32..14).collect();
        let two_loops = vec![
            vec![bv(0, 10), bv(1, 11), bv(2, 12)],
            vec![bv(3, 10), bv(4, 11), bv(5, 12)],
        ];
        assert!(matches!(
            fill_component_crude(&gset, &two_loops, &points),
            Err(FillError::MultiLoopUnsupported { loops: 2 })
        ));
        // k < m: 5 generators, 3-edge loop.
        let gset5: HashSet<u32> = (10u32..15).collect();
        let small = vec![vec![bv(0, 10), bv(1, 11), bv(2, 12)]];
        assert!(matches!(
            fill_component_crude(&gset5, &small, &points),
            Err(FillError::CrudeUnsupported { m: 5, k: 3 })
        ));
    }
}
