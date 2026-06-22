//! Defect-driven escalation (step C of the adaptive-canonical-clip plan, see
//! `docs/adaptive-canonical-clip-design-2026-06.md`).
//!
//! Rebuilds a near-degenerate neighborhood as a SINGLE normalized local 3D hull
//! and reads each generator's Voronoi cell off the shared dual.
//! Because every repaired cell comes from one triangulation, they pair on shared
//! edges by construction, which is the property the reverted per-cell/pin-by-key
//! repairs lacked.
//!
//! This is the vertical-slice core: brute-force neighbor gather + rebuild + a
//! consistency read. The production loop (connected-component growth, the
//! considered-neighbor set, re-assembly, grow-until-clean-rim) builds on top.

// Much of this module is diagnostic scaffolding for the older projected and
// fill experiments. The active production repair path is `repair_local_hull`.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use glam::{DVec3, Vec3};
use rustc_hash::FxHashMap;

use super::local_hull::LocalHull;
use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch, DirectedEligibility};
use crate::diagram::VoronoiCell;
use crate::live_dedup::ShardedVertexKeys;
use crate::packed_layout::PackedSlotLayout;

// Packed-slot layout params for the repair's grid kNN gather, mirroring the
// point locator (see `locate.rs`). The grid is built over the effective points
// with an identity slot→generator map, so a query emits every nearby point.
const REPAIR_LOCAL_SHIFT: u32 = 24;
const REPAIR_LOCAL_MASK: u32 = (1u32 << REPAIR_LOCAL_SHIFT) - 1;

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

/// O(local) replacement for [`gather_local`]'s O(n) brute force, used by the
/// production repair oracle. For each seed, walk the cube-map shell frontier
/// nearest-first (the same machinery the point locator uses) and collect the
/// `k + 1` nearest generators, unioned with the seeds — the same neighbor set as
/// `gather_local` (modulo equal-dot tie-breaking), but proportional to the local
/// neighborhood rather than to `n`.
///
/// `slot_gen_map` is an all-zero (all-emit) eligibility layout so every nearby
/// point is emitted regardless of `n`; `grid.point_indices()[slot]` maps an
/// emitted slot back to its generator id. The ring certificate (`unseen_bound`)
/// bounds every unseen point's dot, so collection stops once the `k + 1` nearest
/// seen are provably nearer than anything unseen.
fn gather_knn_grid(
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    points: &[Vec3],
    seeds: &[u32],
    k: usize,
) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<u32> = seeds.iter().copied().collect();
    let n = slot_gen_map.len();
    let mut batch: Vec<u32> = Vec::new();
    let mut collected: Vec<(f32, u32)> = Vec::new();
    for &s in seeds {
        let query = points[s as usize];
        let layout = PackedSlotLayout::new(slot_gen_map, REPAIR_LOCAL_SHIFT, REPAIR_LOCAL_MASK);
        let ctx = DirectedEligibility::from_layout(1, 0, layout);
        let mut frontier = grid.shell_frontier(query, n, scratch, ctx);
        collected.clear();
        while let Some(layer) = frontier.frontier(&mut batch) {
            for &slot in &batch {
                let id = grid.point_indices()[slot as usize];
                let dot = query.dot(points[id as usize]);
                collected.push((dot, id));
            }
            // Once we hold at least k+1 candidates, the (k+1)-th nearest's dot
            // certifies completeness: if it is already >= the unseen bound, no
            // unseen point can displace the top k+1, so stop.
            if collected.len() >= k + 1 {
                let idx = collected.len() - (k + 1);
                collected.select_nth_unstable_by(idx, |a, b| a.0.partial_cmp(&b.0).unwrap());
                if collected[idx].0 >= layer.unseen_bound {
                    break;
                }
            }
            frontier.advance();
        }
        // `collected` is a superset of the k+1 nearest; take exactly those.
        collected.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(_, id) in collected.iter().take(k + 1) {
            set.insert(id);
        }
    }
    set.into_iter().collect()
}

/// The common third generator of two vertices of `g`'s cell that share an edge:
/// each vertex triple contains `g`; consecutive cell vertices also share the
/// neighbor `m` whose bisector carries the edge `g–m`. Returns `m`, or `None`
/// if the two triples don't share exactly one other generator (not an edge).
pub fn shared_neighbor(g: u32, t0: [u32; 3], t1: [u32; 3]) -> Option<u32> {
    let common: Vec<u32> = t0
        .iter()
        .copied()
        .filter(|&x| x != g && t1.contains(&x))
        .collect();
    if common.len() == 1 {
        Some(common[0])
    } else {
        None
    }
}

/// Whether two vertices are cyclically adjacent in a cell's fan.
fn cyclically_adjacent(verts: &[[u32; 3]], a: [u32; 3], b: [u32; 3]) -> bool {
    let n = verts.len();
    let pa = verts.iter().position(|&t| t == a);
    let pb = verts.iter().position(|&t| t == b);
    match (pa, pb) {
        (Some(pa), Some(pb)) => {
            let d = (pa as isize - pb as isize).rem_euclid(n as isize);
            d == 1 || d == n as isize - 1
        }
        _ => false,
    }
}

/// Verify every interior edge of `g`'s rebuilt cell pairs: for each edge `g–m`
/// whose neighbor `m` is also rebuilt (`cells_by_gen`), `m`'s cell must carry
/// the same two endpoint triples, cyclically adjacent. Returns the number of
/// edges that could NOT be checked because `m` is outside the rebuilt set (the
/// rim) — 0 means the cell is fully internally paired. Panics on a genuine
/// pairing disagreement (an interior defect the rebuild failed to resolve).
pub fn check_cell_internally_paired(
    cell: &RebuiltCell,
    cells_by_gen: &std::collections::HashMap<u32, RebuiltCell>,
) -> usize {
    let g = cell.generator;
    let n = cell.vertices.len();
    let mut rim = 0usize;
    for i in 0..n {
        let t0 = cell.vertices[i];
        let t1 = cell.vertices[(i + 1) % n];
        let Some(m) = shared_neighbor(g, t0, t1) else {
            continue; // coincident / degenerate vertex pair — skip
        };
        match cells_by_gen.get(&m) {
            Some(mc) => assert!(
                cyclically_adjacent(&mc.vertices, t0, t1),
                "edge {g}-{m} present in {g}'s cell ({t0:?},{t1:?}) but not paired in {m}'s cell"
            ),
            None => rim += 1,
        }
    }
    rim
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
        let Some(lg) = local_of(g) else { continue };
        let fan = hull.cell_faces(lg);
        if fan.is_empty() {
            // Broken fan: a boundary generator of the local patch whose true
            // neighbors aren't all gathered (the rim). Skip it — interior cells
            // (the defect site) are clean. Callers treat a missing cell as rim.
            continue;
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

/// Source-compatible local RE-CLIP (step a): rebuild each seed's cell by driving
/// the SAME production clipper (`Topo2DBuilder`) over the local candidate set,
/// nearest-first (mirroring the directed neighbor stream), and reading the
/// triple-keyed fan off `to_vertex_data_full`. Unlike `rebuild_cells` (from-
/// scratch exact Delaunay), this reproduces the fast path's own clip decisions,
/// so it should AGREE with the fast diagram on every well-conditioned cell and
/// only differ where the fast path was unstable. This is the A1 variant: still
/// f32 (no exact escalation yet), used to measure source-compatibility — does a
/// re-clip match fast outside the defect?
pub fn reclip_cells(points: &[Vec3], local_ids: &[u32], seeds: &[u32]) -> Option<Vec<RebuiltCell>> {
    use crate::knn_clipping::topo2d::{BuilderClipOutcome, Topo2DBuilder};
    use crate::live_dedup::CellOutputBuffer;

    let mut buf = CellOutputBuffer::default();
    let mut out = Vec::with_capacity(seeds.len());
    for &g in seeds {
        let gp = points[g as usize];
        // Nearest-first candidate order (largest dot), matching the fast stream.
        let mut cands: Vec<(f32, u32)> = local_ids
            .iter()
            .copied()
            .filter(|&x| x != g)
            .map(|x| (gp.dot(points[x as usize]), x))
            .collect();
        cands.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut builder = Topo2DBuilder::new(g as usize, gp);
        let mut failed = false;
        for (_, n) in cands {
            match builder.clip_with_slot_result_policy(n as usize, u32::MAX, points[n as usize]) {
                Ok(BuilderClipOutcome::Applied(_)) => {}
                Ok(BuilderClipOutcome::NeedsFallback(req)) => builder.enter_fallback(points, req),
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        if failed || !builder.is_bounded() {
            continue; // couldn't form a bounded cell from the local set — rim
        }
        buf.clear();
        if builder.to_vertex_data_full(&mut buf).is_err() {
            continue;
        }
        let vertices: Vec<[u32; 3]> = buf.vertices.iter().map(|&(k, _)| k).collect();
        out.push(RebuiltCell {
            generator: g,
            vertices,
        });
    }
    Some(out)
}

/// Projected exact oracle (feature `escalate_probe`): exact 2D Delaunay via
/// stereographic projection + `delaunator`, read as each seed's ordered Voronoi
/// fan. This remains useful as an A/B reference for the local projected repair.
/// Normalized local 3D hull is now also viable; older raw-3D cascade results
/// were caused by exact predicates seeing f32 radius drift.
#[cfg(feature = "escalate_probe")]
pub fn delaunator_cells(
    points: &[Vec3],
    local_ids: &[u32],
    seeds: &[u32],
) -> Option<Vec<RebuiltCell>> {
    use std::collections::{BTreeSet, HashMap};
    if local_ids.len() < 3 {
        return None;
    }
    let mut c = Vec3::ZERO;
    for &g in local_ids {
        c += points[g as usize];
    }
    let pole = (-c).normalize();
    let a = if pole.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (a - pole * a.dot(pole)).normalize();
    let e2 = pole.cross(e1);
    let proj: Vec<delaunator::Point> = local_ids
        .iter()
        .map(|&g| {
            let p = points[g as usize];
            let d = (1.0 - p.dot(pole)).max(1e-12);
            delaunator::Point {
                x: (p.dot(e1) / d) as f64,
                y: (p.dot(e2) / d) as f64,
            }
        })
        .collect();
    let tri = delaunator::triangulate(&proj);
    if tri.triangles.is_empty() {
        return None;
    }
    let seedset: BTreeSet<u32> = seeds.iter().copied().collect();
    // Incident sorted-global triples per seed generator.
    let mut incident: HashMap<u32, Vec<[u32; 3]>> = HashMap::new();
    for t in tri.triangles.chunks_exact(3) {
        let gs = [local_ids[t[0]], local_ids[t[1]], local_ids[t[2]]];
        let mut tri3 = gs;
        tri3.sort_unstable();
        for &g in &gs {
            if seedset.contains(&g) {
                incident.entry(g).or_default().push(tri3);
            }
        }
    }
    let adj = |a: [u32; 3], b: [u32; 3]| a != b && a.iter().filter(|x| b.contains(x)).count() == 2;
    let mut out = Vec::with_capacity(seeds.len());
    for &g in seeds {
        let Some(tris) = incident.get(&g) else {
            continue;
        };
        let n = tris.len();
        if n < 3 {
            continue;
        }
        // Chain incident triangles into the cyclic fan (each shares edge g-x with
        // the next). delaunator's fans are clean, so this closes — a generator on
        // the projection hull won't close and is skipped (treated as rim).
        let mut used = vec![false; n];
        let mut order = vec![tris[0]];
        used[0] = true;
        let mut ok = true;
        for _ in 1..n {
            let cur = *order.last().unwrap();
            match (0..n).find(|&j| !used[j] && adj(cur, tris[j])) {
                Some(j) => {
                    used[j] = true;
                    order.push(tris[j]);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && adj(*order.last().unwrap(), order[0]) {
            out.push(RebuiltCell {
                generator: g,
                vertices: order,
            });
        }
    }
    Some(out)
}

/// Probe defect repair with the projected external oracle (`delaunator_cells`).
/// The production-style projected repair is `repair_local_exact`; this global
/// oracle remains as an A/B scaffold in `escalate_probe`.
#[cfg(feature = "escalate_probe")]
pub fn repair_delaunator(
    points: &[Vec3],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    _gather_k: usize,
    max_rounds: usize,
) -> EscalationStats {
    use std::collections::HashMap;
    let mut stats = EscalationStats::default();
    let defect_gens: std::collections::BTreeSet<u32> =
        defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let target_sign = work.winding_convention(points, &defect_gens);

    // ONE GLOBAL stereographic Delaunay (fixed pole) over all generators. A
    // global pole is what makes the rebuilt rim agree with the fast diagram
    // (fast ≈ the global-pole Delaunay); a per-gather local pole picks different
    // near-cocircular diagonals → rim mismatch → overgrowth/skips. Defective
    // inputs are clustered, so a single pole = antipode of the centroid is sound.
    let mut centroid = Vec3::ZERO;
    for &p in points {
        centroid += p;
    }
    let pole = (-centroid).normalize();
    let a = if pole.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (a - pole * a.dot(pole)).normalize();
    let e2 = pole.cross(e1);
    let proj: Vec<delaunator::Point> = points
        .iter()
        .map(|&p| {
            let d = (1.0 - p.dot(pole)).max(1e-12);
            delaunator::Point {
                x: (p.dot(e1) / d) as f64,
                y: (p.dot(e2) / d) as f64,
            }
        })
        .collect();
    let tri = delaunator::triangulate(&proj);
    if tri.triangles.is_empty() {
        return stats;
    }
    // generator -> incident sorted-global triples (the global Delaunay fan)
    let mut incident: HashMap<u32, Vec<[u32; 3]>> = HashMap::new();
    for t in tri.triangles.chunks_exact(3) {
        let mut k = [t[0] as u32, t[1] as u32, t[2] as u32];
        k.sort_unstable();
        for &li in t {
            incident.entry(li as u32).or_default().push(k);
        }
    }
    let adj = |a: [u32; 3], b: [u32; 3]| a != b && a.iter().filter(|x| b.contains(x)).count() == 2;
    // Ordered fan for g from its incident triples (combinatorial chain); None if
    // it doesn't close (g on the projection hull — a far-hemisphere rim cell).
    let extract = |g: u32| -> Option<Vec<[u32; 3]>> {
        let tris = incident.get(&g)?;
        let n = tris.len();
        if n < 3 {
            return None;
        }
        let mut used = vec![false; n];
        let mut order = vec![tris[0]];
        used[0] = true;
        for _ in 1..n {
            let cur = *order.last().unwrap();
            let j = (0..n).find(|&j| !used[j] && adj(cur, tris[j]))?;
            used[j] = true;
            order.push(tris[j]);
        }
        if adj(*order.last().unwrap(), order[0]) {
            Some(order)
        } else {
            None
        }
    };

    let mut spliced: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    // Seed the closure from the unpaired-edge generators AND any low-incidence
    // vertices (degree 1/2 — a sliver/near-coincident vertex can be a defect with
    // NO unpaired edge, which the unpaired-only trigger misses).
    let mut closure: std::collections::BTreeSet<u32> = defect_gens;
    {
        let mut refcount: HashMap<u32, u32> = HashMap::new();
        for list in &work.cells {
            for &v in list {
                *refcount.entry(v).or_default() += 1;
            }
        }
        for (&v, &c) in &refcount {
            if (1..3).contains(&c) {
                for x in work.vkey[v as usize] {
                    closure.insert(x);
                }
            }
        }
    }
    for _ in 0..max_rounds {
        if closure.is_empty() {
            break;
        }
        stats.rounds += 1;
        let cells: Vec<RebuiltCell> = closure
            .iter()
            .filter_map(|&g| {
                extract(g).map(|vertices| RebuiltCell {
                    generator: g,
                    vertices,
                })
            })
            .collect();
        for c in &cells {
            work.splice_generator(points, c.generator, &c.vertices, target_sign);
            spliced.insert(c.generator);
        }
        // Grow on residual defects: unpaired edges AND low-incidence vertices.
        // A vertex referenced by 1 or 2 cells is a degree-1/2 artifact — e.g. a
        // closure cell re-fanned away a fast vertex its NON-closure neighbors
        // still reference, leaving it degree-2 OUTSIDE the closure. Scan globally
        // and add that triple's generators so the next round completes them.
        let mut implicated: std::collections::BTreeSet<u32> =
            work.unpaired_generators().into_iter().collect();
        {
            let mut refcount: HashMap<u32, u32> = HashMap::new();
            for list in &work.cells {
                for &v in list {
                    *refcount.entry(v).or_default() += 1;
                }
            }
            for (&v, &c) in &refcount {
                if (1..3).contains(&c) {
                    for x in work.vkey[v as usize] {
                        implicated.insert(x);
                    }
                }
            }
        }
        let new: Vec<u32> = implicated
            .iter()
            .copied()
            .filter(|g| !closure.contains(g))
            .collect();
        if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
            eprintln!(
                "  repair round {}: closure={} spliced={} implicated={} new={}",
                stats.rounds,
                closure.len(),
                spliced.len(),
                implicated.len(),
                new.len(),
            );
        }
        if new.is_empty() {
            stats.stuck_components = usize::from(!implicated.is_empty());
            break;
        }
        for g in new {
            closure.insert(g);
        }
    }
    stats.spliced_generators = spliced.len();
    if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
        eprintln!(
            "repair_delaunator: {:?}; final unpaired-implicated={}",
            stats,
            work.unpaired_generators().len()
        );
    }
    stats
}

/// Cyclically chain a generator's incident sorted-global triples into its
/// Voronoi fan: each consecutive pair shares the edge `g–x` (two common
/// generators). Returns `None` if the fan doesn't close (the generator is on the
/// gather frontier — an incomplete neighborhood, deferred to the grow loop), so
/// only genuinely-interior cells are spliced. Shared by the local and (probe)
/// delaunator engines.
fn chain_fan(tris: &[[u32; 3]]) -> Option<Vec<[u32; 3]>> {
    let adj = |a: [u32; 3], b: [u32; 3]| a != b && a.iter().filter(|x| b.contains(x)).count() == 2;
    let n = tris.len();
    if n < 3 {
        return None;
    }
    let mut used = vec![false; n];
    let mut order = vec![tris[0]];
    used[0] = true;
    for _ in 1..n {
        let cur = *order.last().unwrap();
        let j = (0..n).find(|&j| !used[j] && adj(cur, tris[j]))?;
        used[j] = true;
        order.push(tris[j]);
    }
    if adj(*order.last().unwrap(), order[0]) {
        Some(order)
    } else {
        None
    }
}

/// Exact 2D Delaunay triangulation of `proj` via Bowyer–Watson incremental
/// insertion (dependency-free: `robust` exact `incircle`/`orient2d`, already a
/// crate dep). Returns CCW triangles as input-index triples; triangles touching
/// the bounding super-triangle are dropped. The point set is small (a local
/// gather), so the simple O(n·triangles) cavity walk is ample.
///
/// One triangulation makes every interior generator's incident triangles a
/// closed, manifold fan by construction — the property a per-generator
/// empty-circle enumeration can't guarantee for cluster-boundary cells (whose
/// fans thread through far neighbors a per-cell candidate ring won't surface).
fn local_delaunay_2d(proj: &[robust::Coord<f64>]) -> Vec<[usize; 3]> {
    use robust::{incircle, orient2d, Coord};
    let n = proj.len();
    if n < 3 {
        return Vec::new();
    }
    let (mut minx, mut miny, mut maxx, mut maxy) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for c in proj {
        minx = minx.min(c.x);
        miny = miny.min(c.y);
        maxx = maxx.max(c.x);
        maxy = maxy.max(c.y);
    }
    let span = (maxx - minx).max(maxy - miny).max(1e-9);
    let (midx, midy) = ((minx + maxx) * 0.5, (miny + maxy) * 0.5);
    // Super-triangle vertices (indices n, n+1, n+2), generously enclosing all.
    let big = span * 1000.0;
    let mut pts: Vec<Coord<f64>> = proj.to_vec();
    pts.push(Coord {
        x: midx - 2.0 * big,
        y: midy - big,
    });
    pts.push(Coord {
        x: midx + 2.0 * big,
        y: midy - big,
    });
    pts.push(Coord {
        x: midx,
        y: midy + 2.0 * big,
    });
    let ccw = |a: usize, b: usize, c: usize| -> [usize; 3] {
        if orient2d(pts[a], pts[b], pts[c]) > 0.0 {
            [a, b, c]
        } else {
            [a, c, b]
        }
    };
    let mut tris: Vec<[usize; 3]> = vec![ccw(n, n + 1, n + 2)];
    let mut bad_edges: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    for i in 0..n {
        let p = pts[i];
        // Triangles whose circumcircle contains p (each stored CCW).
        bad_edges.clear();
        let mut keep: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
        for &t in &tris {
            if incircle(pts[t[0]], pts[t[1]], pts[t[2]], p) > 0.0 {
                for &(u, v) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *bad_edges.entry((u, v)).or_default() += 1;
                }
            } else {
                keep.push(t);
            }
        }
        tris = keep;
        // Re-triangulate the cavity: each boundary directed edge (u,v) — one whose
        // reverse (v,u) is not also a bad edge — forms a new CCW triangle (u,v,i).
        for (&(u, v), _) in bad_edges.iter() {
            if !bad_edges.contains_key(&(v, u)) {
                tris.push([u, v, i]);
            }
        }
    }
    tris.retain(|t| t.iter().all(|&v| v < n));
    tris
}

/// Per-generator incident Delaunay triangles for `closure`, read off ONE exact
/// 2D Delaunay (`robust::incircle`) built in a single shared stereographic chart
/// over a local gather — the dependency-free, local analog of the global
/// stereographic delaunator oracle.
///
/// A triple `(g,a,b)` is a Delaunay triangle (hence a Voronoi vertex of `g`) iff
/// no other nearby generator `h` falls strictly inside its circumcircle —
/// `in_circle_sphere_sign(g,a,b,h) <= 0` for all `h`. That predicate is a pure
/// function of the four RAW points (Shewchuk-exact orient3d), so it returns the
/// IDENTICAL verdict no matter which cell evaluates it. That cross-cell
/// consistency is exactly what the per-cell gnomonic CHART f64 rounding lacks —
/// resolving it here with no external crate and no global triangulation.
///
/// Correctness is local because a point inside a closure triangle's circumcircle
/// lies within ~2× the cell radius, so it is present in the SHARED union gather
/// `L` that every emptiness test scans — this is a proper restricted local
/// Delaunay, not a set of per-generator triangle tests (which over-admit: a
/// candidate whose intruder sits near `a`/`b` but outside `g`'s own list would be
/// falsely accepted, giving a non-manifold incident set that never chains).
///
/// CLUSTER-BOUNDARY cells are the subtlety: a generator on the rim of a dense
/// cluster has a Voronoi cell that reaches far into the sparse region, so its
/// true fan includes FAR generators a kNN gather can't see — rebuilding from kNN
/// alone leaves an open arc. The fix: seed each generator's candidate neighbors
/// (and the union gather) from its CURRENT assembled cell's vertex triples. The
/// fast path already found those far neighbors correctly; we keep them and only
/// RE-DECIDE the contested near-cocircular vertices with the exact predicate. At
/// the well-conditioned rim the exact sign equals the fast clipper's decision, so
/// the rebuilt rim triples reuse the surrounding cells' vids and pair by
/// construction. Any genuine shortfall surfaces as a residual the grow loop
/// expands, behind the whole-diagram never-worse gate.
fn local_exact_incident(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &WorkingDiagram,
    closure: &[u32],
    gather_k: usize,
) -> FxHashMap<u32, Vec<[u32; 3]>> {
    use robust::Coord;
    use std::collections::BTreeSet;

    // g's current Voronoi neighbors, read off its assembled cell's vertex triples.
    let cell_neighbors = |g: u32| -> Vec<u32> {
        let mut ns: Vec<u32> = work.cells[g as usize]
            .iter()
            .flat_map(|&v| work.vkey[v as usize])
            .filter(|&x| x != g && x != u32::MAX)
            .collect();
        ns.sort_unstable();
        ns.dedup();
        ns
    };

    // Union gather (2-RING): the closure, each closure cell's neighbors (so a
    // cluster-boundary cell's far Voronoi neighbors are present), then each of
    // those seeds' kNN (so every gathered cell's own neighborhood is complete and
    // its triangulated fan is the true Delaunay, not a gather-boundary artifact).
    let mut seeds2: BTreeSet<u32> = closure.iter().copied().collect();
    for &g in closure {
        seeds2.extend(cell_neighbors(g));
    }
    let seeds2: Vec<u32> = seeds2.into_iter().collect();
    let ring_k = gather_k.min(32);
    let local_ids: Vec<u32> = gather_knn_grid(grid, scratch, slot_gen_map, points, &seeds2, ring_k);

    // SINGLE shared stereographic chart over the gather. This is the current
    // production repair oracle: one exact 2D triangulation produces all closure
    // fans, so repaired cells agree with each other. Normalized local 3D hull is
    // now a viable equivalent oracle on tested S2 inputs; the previous raw-3D
    // cascade diagnosis was an off-sphere f32-radius artifact, not projection
    // drift. Keep this projected path as the default until the normalized local
    // 3D broad sweep is complete.
    let mut centroid = Vec3::ZERO;
    for &id in &local_ids {
        centroid += points[id as usize];
    }
    let pole = (-centroid).normalize();
    let ax = if pole.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (ax - pole * ax.dot(pole)).normalize();
    let e2 = pole.cross(e1);
    let proj: Vec<Coord<f64>> = local_ids
        .iter()
        .map(|&id| {
            let p = points[id as usize];
            let d = (1.0 - p.dot(pole)).max(1e-12);
            Coord {
                x: (p.dot(e1) / d) as f64,
                y: (p.dot(e2) / d) as f64,
            }
        })
        .collect();

    // One Delaunay triangulation; read each closure generator's incident fan off
    // it (the same triple seen from a/b's cells, so spliced cells pair).
    let tri = local_delaunay_2d(&proj);
    let closureset: BTreeSet<u32> = closure.iter().copied().collect();
    let mut incident: FxHashMap<u32, Vec<[u32; 3]>> = FxHashMap::default();
    for &g in closure {
        incident.entry(g).or_default();
    }
    for t in &tri {
        let gs = [local_ids[t[0]], local_ids[t[1]], local_ids[t[2]]];
        let mut sorted = gs;
        sorted.sort_unstable();
        for &g in &gs {
            if closureset.contains(&g) {
                incident.get_mut(&g).unwrap().push(sorted);
            }
        }
    }
    incident
}

fn local_hull_incident(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &WorkingDiagram,
    closure: &[u32],
    gather_k: usize,
) -> FxHashMap<u32, Vec<[u32; 3]>> {
    use std::collections::BTreeSet;

    if closure.is_empty() {
        return FxHashMap::default();
    }

    let cell_neighbors = |g: u32| -> Vec<u32> {
        let mut ns: Vec<u32> = work.cells[g as usize]
            .iter()
            .flat_map(|&v| work.vkey[v as usize])
            .filter(|&x| x != g && x != u32::MAX)
            .collect();
        ns.sort_unstable();
        ns.dedup();
        ns
    };

    let mut seeds2: BTreeSet<u32> = closure.iter().copied().collect();
    for &g in closure {
        seeds2.extend(cell_neighbors(g));
    }
    let seeds2: Vec<u32> = seeds2.into_iter().collect();
    let ring_k = gather_k.min(32);
    let local_ids: Vec<u32> = gather_knn_grid(grid, scratch, slot_gen_map, points, &seeds2, ring_k);

    let closureset: BTreeSet<u32> = closure.iter().copied().collect();
    let mut incident: FxHashMap<u32, Vec<[u32; 3]>> = FxHashMap::default();
    for &g in closure {
        incident.entry(g).or_default();
    }
    if let Some(cells) = rebuild_cells(points, &local_ids, closure) {
        for cell in cells {
            if !closureset.contains(&cell.generator) {
                continue;
            }
            incident.insert(cell.generator, cell.vertices);
        }
    }
    incident
}

/// Dependency-free, local defect repair: the production escalation engine.
///
/// Same shape as the (probe-only) [`repair_delaunator`] — seed the closure from
/// the unpaired-edge and low-incidence generators, splice each closure cell from
/// the consistent exact oracle, grow on the residual until it closes, behind the
/// caller's whole-diagram never-worse gate — but the oracle is
/// [`local_exact_incident`]: one exact 2D Delaunay (`robust::incircle`) in a
/// single shared stereographic chart over a LOCAL gather, instead of a global
/// stereographic delaunator. No external crate, no global O(n log n)
/// triangulation: a true local repair.
pub(crate) fn repair_local_exact(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    gather_k: usize,
    max_rounds: usize,
) -> EscalationStats {
    use std::collections::BTreeSet;
    let mut stats = EscalationStats::default();

    // Seed: the defect-pair generators UNION any low-incidence (degree 1/2)
    // vertex's generators — a sliver vertex can be a defect with no unpaired edge.
    let mut closure: BTreeSet<u32> = defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let low_incidence_gens = |work: &WorkingDiagram| -> Vec<u32> {
        let mut refcount: FxHashMap<u32, u32> = FxHashMap::default();
        for list in &work.cells {
            for &v in list {
                *refcount.entry(v).or_default() += 1;
            }
        }
        let mut out = Vec::new();
        for (&v, &c) in &refcount {
            if (1..3).contains(&c) {
                out.extend(work.vkey[v as usize]);
            }
        }
        out
    };
    for g in low_incidence_gens(work) {
        closure.insert(g);
    }
    let defect_gens: BTreeSet<u32> = defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let target_sign = work.winding_convention(points, &defect_gens);

    let mut spliced: BTreeSet<u32> = BTreeSet::new();
    for _ in 0..max_rounds {
        if closure.is_empty() {
            break;
        }
        stats.rounds += 1;
        let closure_vec: Vec<u32> = closure.iter().copied().collect();
        let incident =
            local_exact_incident(points, grid, scratch, slot_gen_map, work, &closure_vec, gather_k);
        for &g in &closure_vec {
            let Some(tris) = incident.get(&g) else {
                continue;
            };
            let Some(fan) = chain_fan(tris) else {
                continue; // frontier generator — defer to a later, wider round
            };
            work.splice_generator(points, g, &fan, target_sign);
            spliced.insert(g);
        }
        // Grow on the residual: generators named by any still-unpaired edge, plus
        // low-incidence vertices left by re-fanning (a vertex an unspliced
        // neighbor still references, now orphaned). A full scan sees both.
        let mut implicated: BTreeSet<u32> = work.unpaired_generators().into_iter().collect();
        for g in low_incidence_gens(work) {
            implicated.insert(g);
        }
        let new: Vec<u32> = implicated
            .iter()
            .copied()
            .filter(|g| !closure.contains(g))
            .collect();
        if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
            eprintln!(
                "  local repair round {}: closure={} spliced={} implicated={} new={}",
                stats.rounds,
                closure.len(),
                spliced.len(),
                implicated.len(),
                new.len(),
            );
        }
        if new.is_empty() {
            stats.stuck_components = usize::from(!implicated.is_empty());
            break;
        }
        for g in new {
            closure.insert(g);
        }
    }
    stats.spliced_generators = spliced.len();
    if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
        eprintln!(
            "repair_local_exact: {:?}; final unpaired-implicated={}",
            stats,
            work.unpaired_generators().len()
        );
    }
    stats
}

/// Dependency-free local 3D repair: use normalized local 3D hulls as the oracle
/// instead of projected exact 2D Delaunay.
///
/// This is the preferred final-backstop shape because exact 3D construction has
/// no single-chart/pole failure mode. `local_hull` normalizes S2 directions
/// before exact predicates, so it solves the crate's spherical input problem
/// rather than an off-sphere f32-radius hull problem.
pub(crate) fn repair_local_hull(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    gather_k: usize,
    max_rounds: usize,
) -> EscalationStats {
    use std::collections::BTreeSet;
    let mut stats = EscalationStats::default();

    let mut closure: BTreeSet<u32> = defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let low_incidence_gens = |work: &WorkingDiagram| -> Vec<u32> {
        let mut refcount: FxHashMap<u32, u32> = FxHashMap::default();
        for list in &work.cells {
            for &v in list {
                *refcount.entry(v).or_default() += 1;
            }
        }
        let mut out = Vec::new();
        for (&v, &c) in &refcount {
            if (1..3).contains(&c) {
                out.extend(work.vkey[v as usize]);
            }
        }
        out
    };
    for g in low_incidence_gens(work) {
        closure.insert(g);
    }
    let defect_gens: BTreeSet<u32> = defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let target_sign = work.winding_convention(points, &defect_gens);

    let mut spliced: BTreeSet<u32> = BTreeSet::new();
    for _ in 0..max_rounds {
        if closure.is_empty() {
            break;
        }
        stats.rounds += 1;
        let closure_vec: Vec<u32> = closure.iter().copied().collect();
        let incident =
            local_hull_incident(points, grid, scratch, slot_gen_map, work, &closure_vec, gather_k);
        for &g in &closure_vec {
            let Some(fan) = incident.get(&g) else {
                continue;
            };
            if fan.len() < 3 {
                continue;
            }
            work.splice_generator(points, g, fan, target_sign);
            spliced.insert(g);
        }

        let mut implicated: BTreeSet<u32> = work.unpaired_generators().into_iter().collect();
        for g in low_incidence_gens(work) {
            implicated.insert(g);
        }
        let new: Vec<u32> = implicated
            .iter()
            .copied()
            .filter(|g| !closure.contains(g))
            .collect();
        if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
            eprintln!(
                "  local hull repair round {}: closure={} spliced={} implicated={} new={}",
                stats.rounds,
                closure.len(),
                spliced.len(),
                implicated.len(),
                new.len(),
            );
        }
        if new.is_empty() {
            stats.stuck_components = usize::from(!implicated.is_empty());
            break;
        }
        for g in new {
            closure.insert(g);
        }
    }
    stats.spliced_generators = spliced.len();
    if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
        eprintln!(
            "repair_local_hull: {:?}; final unpaired-implicated={}",
            stats,
            work.unpaired_generators().len()
        );
    }
    stats
}

/// Select the active cell source from `S2_ESCALATE_SOURCE` (`reclip` | `hull`),
/// defaulting to the passed-in `source`. Lets the probes A/B the local re-clip
/// against the from-scratch hull without rewiring callers.
fn pick_source(default: CellSource) -> CellSource {
    match std::env::var("S2_ESCALATE_SOURCE").as_deref() {
        Ok("reclip") => reclip_cells,
        Ok("hull") => rebuild_cells,
        _ => default,
    }
}

// ===========================================================================
// Step C → route (a): splice rebuilt cells into the assembled diagram.
//
// The splice is SOURCE-AGNOSTIC: it consumes triple-keyed cells (`RebuiltCell`)
// and does not care whether they came from the local-hull dual (`rebuild_cells`
// today) or a future exact local re-clip. The only contract is "a generator's
// cell as an ordered cyclic fan of sorted global-id triples", same identity
// space as the production `VertexKey`.
//
// The load-bearing trick: a rebuilt vertex looks up the EXISTING fast-path
// vertex id for its triple and reuses it. On the well-conditioned rim
// (fast == exact) the triple is already present, so the rebuilt cell shares the
// surrounding cells' vertices and pairs with them automatically. Only the
// corrected near-cocircular defect vertices are minted fresh.
// ===========================================================================

/// The f64 spherical circumcenter of a generator triple, as a (near-)unit
/// `Vec3` — the Voronoi vertex of the three generators. Used to position a
/// freshly minted vertex; deterministic and source-independent (any producer of
/// the same triple agrees on its position).
pub fn triple_circumcenter(points: &[Vec3], t: [u32; 3]) -> Vec3 {
    let p = |i: u32| {
        let v = points[i as usize];
        DVec3::new(v.x as f64, v.y as f64, v.z as f64)
    };
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    let mut n = (b - a).cross(c - a).normalize();
    // Outward = same side as the generators (matches local_hull::face_circumcenter).
    if n.dot(a) < 0.0 {
        n = -n;
    }
    Vec3::new(n.x as f32, n.y as f32, n.z as f32)
}

/// Order a cell's vertex triples into a cyclic boundary by the angular position
/// of each circumcenter in generator `g`'s tangent plane. Robust for a mixed
/// rim(fast)+interior(hull) vertex set — unlike a greedy shared-generator cycle
/// walk, it never bails on a non-manifold/degenerate set (the grow loop + the
/// never-worse gate catch any residual inconsistency).
fn order_cell_angular(points: &[Vec3], g: u32, triples: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let gp = points[g as usize];
    let a = if gp.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = (a - gp * a.dot(gp)).normalize();
    let e2 = gp.cross(e1);
    let mut keyed: Vec<(f32, [u32; 3])> = triples
        .iter()
        .map(|&t| {
            let c = triple_circumcenter(points, t);
            (c.dot(e2).atan2(c.dot(e1)), t)
        })
        .collect();
    keyed.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().map(|(_, t)| t).collect()
}

/// A mutable triple-keyed view of the assembled diagram, in effective-generator
/// index space. Splicing replaces a generator's whole boundary with a rebuilt
/// fan, minting vertices for any triple the fast path never produced.
pub struct WorkingDiagram {
    /// Vertex positions (grows as new triples are minted).
    pub vpos: Vec<Vec3>,
    /// Triple per vertex, parallel to `vpos`.
    pub vkey: Vec<[u32; 3]>,
    /// Per effective generator: its boundary as an ordered list of vertex ids.
    pub cells: Vec<Vec<u32>>,
    /// First vertex id that carries a given triple (mint-once cache).
    triple_to_vid: FxHashMap<[u32; 3], u32>,
}

impl WorkingDiagram {
    /// Build from the assembled global arrays (post-reconcile). `keys` is the
    /// per-vertex triple store; `cells`/`cell_indices` the flat CSR boundaries.
    pub fn from_assembled(
        vertices: &[Vec3],
        keys: &ShardedVertexKeys,
        cells: &[VoronoiCell],
        cell_indices: &[u32],
    ) -> Self {
        let mut vkey = vec![[u32::MAX; 3]; vertices.len()];
        keys.for_each(|vid, k| vkey[vid as usize] = k);

        // First-vid-per-triple. A triple may recur at a few vids the
        // proximity-merge missed; reusing the first is fine (same circumcenter).
        let mut triple_to_vid: FxHashMap<[u32; 3], u32> =
            FxHashMap::with_capacity_and_hasher(vertices.len(), Default::default());
        for (vid, &k) in vkey.iter().enumerate() {
            triple_to_vid.entry(k).or_insert(vid as u32);
        }

        let cell_lists = cells
            .iter()
            .map(|c| cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()].to_vec())
            .collect();

        Self {
            vpos: vertices.to_vec(),
            vkey,
            cells: cell_lists,
            triple_to_vid,
        }
    }

    /// Vertex id carrying triple `t`, minting (with the triple's circumcenter) if
    /// the fast path never produced it.
    fn vid_for(&mut self, points: &[Vec3], t: [u32; 3]) -> u32 {
        if let Some(&vid) = self.triple_to_vid.get(&t) {
            return vid;
        }
        let vid = self.vpos.len() as u32;
        self.vpos.push(triple_circumcenter(points, t));
        self.vkey.push(t);
        self.triple_to_vid.insert(t, vid);
        vid
    }

    /// Replace generator `g`'s boundary with the rebuilt fan `fan`, oriented to
    /// match the diagram's global winding convention (`target_sign`). The
    /// local-hull dual fan can come out either way; a rim edge only pairs with
    /// its unspliced neighbor when both wind the same direction, so the fan is
    /// reversed if its signed orientation disagrees.
    fn splice_generator(&mut self, points: &[Vec3], g: u32, fan: &[[u32; 3]], target_sign: f32) {
        if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
            let before = self.vpos.len();
            let reused = fan
                .iter()
                .filter(|t| self.triple_to_vid.contains_key(*t))
                .count();
            REUSE_HIT.fetch_add(reused as u64, Ordering::Relaxed);
            REUSE_MISS.fetch_add((fan.len() - reused) as u64, Ordering::Relaxed);
            let _ = before;
        }
        let mut list: Vec<u32> = fan.iter().map(|&t| self.vid_for(points, t)).collect();
        if target_sign != 0.0 && self.polygon_sign(points, g, &list) * target_sign < 0.0 {
            list.reverse();
        }
        self.cells[g as usize] = list;
    }

    /// Signed orientation of generator `g`'s boundary `list`: the sphere-surface
    /// area normal dotted with the generator direction. Positive and negative
    /// distinguish CCW vs CW as seen from outside the sphere.
    fn polygon_sign(&self, points: &[Vec3], g: u32, list: &[u32]) -> f32 {
        let n = list.len();
        if n < 3 {
            return 0.0;
        }
        let mut acc = Vec3::ZERO;
        for i in 0..n {
            acc += self.vpos[list[i] as usize].cross(self.vpos[list[(i + 1) % n] as usize]);
        }
        acc.dot(points[g as usize])
    }

    /// Majority signed-orientation of the existing (unspliced) cells — the
    /// global winding convention the spliced cells must match. Sampled over the
    /// first cells with a real boundary, skipping the defect generators.
    fn winding_convention(&self, points: &[Vec3], skip: &std::collections::BTreeSet<u32>) -> f32 {
        let mut pos = 0i32;
        let mut neg = 0i32;
        for (g, list) in self.cells.iter().enumerate() {
            if skip.contains(&(g as u32)) || list.len() < 3 {
                continue;
            }
            let s = self.polygon_sign(points, g as u32, list);
            if s > 0.0 {
                pos += 1;
            } else if s < 0.0 {
                neg += 1;
            }
            if pos + neg >= 256 {
                break;
            }
        }
        if neg > pos {
            -1.0
        } else {
            1.0
        }
    }

    /// Generators implicated by every unpaired boundary edge in the WHOLE
    /// diagram. An undirected edge `{va,vb}` is paired iff it is used by exactly
    /// two directed half-edges in opposite directions; anything else (one use,
    /// three+ uses, or two same-direction uses) is a defect, and the generators
    /// named by the two endpoints' triples are returned. This is the source of
    /// truth the grow loop converges against: rebuilding a cell `g` orphans the
    /// stale edges of any UNSPLICED neighbor still referencing `g`'s old defect
    /// vertices, and only a full scan sees those.
    fn unpaired_generators(&self) -> Vec<u32> {
        // Directed half-edge → count.
        let mut dir: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        for list in &self.cells {
            let n = list.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                *dir.entry((list[i], list[(i + 1) % n])).or_default() += 1;
            }
        }
        let paired = |a: u32, b: u32| {
            dir.get(&(a, b)).copied().unwrap_or(0) == 1
                && dir.get(&(b, a)).copied().unwrap_or(0) == 1
        };
        let mut grow: Vec<u32> = Vec::new();
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for &(a, b) in dir.keys() {
            let key = (a.min(b), a.max(b));
            if !seen.insert(key) {
                continue;
            }
            if !paired(a, b) {
                let (ka, kb) = (self.vkey[a as usize], self.vkey[b as usize]);
                for &x in ka.iter().chain(kb.iter()) {
                    grow.push(x);
                }
            }
        }
        grow.sort_unstable();
        grow.dedup();
        grow
    }

    /// Unpaired undirected boundary edges `{lo,hi}` and the set of generators
    /// whose cell uses either direction of one (the "broken" cells). Diagnostic.
    fn unpaired_edges_and_cells(&self) -> (Vec<(u32, u32)>, std::collections::BTreeSet<u32>) {
        let mut dir: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        for list in &self.cells {
            let n = list.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                *dir.entry((list[i], list[(i + 1) % n])).or_default() += 1;
            }
        }
        let paired = |a: u32, b: u32| {
            dir.get(&(a, b)).copied().unwrap_or(0) == 1
                && dir.get(&(b, a)).copied().unwrap_or(0) == 1
        };
        let mut edges: Vec<(u32, u32)> = Vec::new();
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for &(a, b) in dir.keys() {
            let key = (a.min(b), a.max(b));
            if !seen.insert(key) {
                continue;
            }
            if !paired(a, b) {
                edges.push(key);
            }
        }
        // Both cells bordering each unpaired Voronoi edge are "broken": the
        // generators shared by the two endpoints' triples (the edge g–m). This
        // captures the side that HAS the stray edge and the side MISSING it.
        let mut broken: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for &(va, vb) in &edges {
            let (ka, kb) = (self.vkey[va as usize], self.vkey[vb as usize]);
            for &x in ka.iter() {
                if kb.contains(&x) {
                    broken.insert(x);
                }
            }
        }
        (edges, broken)
    }

    /// Cells whose f32 clip produced an INVALID polygon: a Voronoi cell is
    /// convex, so each neighbor's bisector gives at most one boundary edge = 2
    /// vertices; a neighbor appearing in >=3 of the cell's vertex triples means
    /// the cell is non-convex / self-overlapping. These are the repair targets.
    fn malformed_cells(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        for (g, list) in self.cells.iter().enumerate() {
            if list.len() < 3 {
                continue;
            }
            counts.clear();
            let mut bad = false;
            for &v in list {
                for &x in self.vkey[v as usize].iter() {
                    if x != g as u32 {
                        let c = counts.entry(x).or_default();
                        *c += 1;
                        if *c >= 3 {
                            bad = true;
                        }
                    }
                }
            }
            if bad {
                out.push(g as u32);
            }
        }
        out
    }

    /// Rebuild generator `g`'s boundary from its NEIGHBORS' consensus: the set of
    /// vertex triples that other (trusted, non-`skip`) cells attribute to `g`,
    /// ordered into a cycle. Adopts the neighbors' existing (valid) diagonals, so
    /// splicing it pairs by construction and does not cascade. Returns `None` if
    /// the consensus vertices do not form a single clean cycle (a neighbor is
    /// also untrusted — defer to a later round once it is repaired).
    fn consensus_cycle(
        &self,
        g: u32,
        skip: &std::collections::BTreeSet<u32>,
    ) -> Option<Vec<[u32; 3]>> {
        let mut set: std::collections::BTreeSet<[u32; 3]> = std::collections::BTreeSet::new();
        for (h, list) in self.cells.iter().enumerate() {
            if h as u32 == g || skip.contains(&(h as u32)) {
                continue;
            }
            for &v in list {
                let t = self.vkey[v as usize];
                if t.contains(&g) {
                    set.insert(t);
                }
            }
        }
        let verts: Vec<[u32; 3]> = set.into_iter().collect();
        let n = verts.len();
        if n < 3 {
            return None;
        }
        // (g,a,b) ~ (g,a,c): share g + exactly one other generator.
        let adj =
            |a: [u32; 3], b: [u32; 3]| a != b && a.iter().filter(|x| b.contains(x)).count() == 2;
        let mut used = vec![false; n];
        let mut order = Vec::with_capacity(n);
        order.push(verts[0]);
        used[0] = true;
        for _ in 1..n {
            let cur = *order.last().unwrap();
            let nxt = (0..n).find(|&j| !used[j] && adj(cur, verts[j]))?;
            used[nxt] = true;
            order.push(verts[nxt]);
        }
        if adj(*order.last().unwrap(), order[0]) {
            Some(order)
        } else {
            None
        }
    }

    /// The defect cluster to repair: every cell incident to an unpaired edge,
    /// plus every malformed (invalid-polygon) cell.
    fn defect_cluster(&self) -> std::collections::BTreeSet<u32> {
        let (_edges, broken) = self.unpaired_edges_and_cells();
        let mut cluster = broken;
        for g in self.malformed_cells() {
            cluster.insert(g);
        }
        cluster
    }

    /// One fill pass over `cluster`: rebuild each cluster cell with its RIM
    /// vertices pinned from trusted (non-cluster) neighbors and its INTERIOR
    /// (cluster–cluster) vertices taken from one exact local hull over the
    /// cluster's neighborhood, ordered into a cycle. Returns the number of cells
    /// successfully rebuilt (a cell whose vertices do not chain into a clean
    /// cycle is left unchanged for a later, larger-cluster round).
    fn fill_cluster_pass(
        &mut self,
        points: &[Vec3],
        cluster: &std::collections::BTreeSet<u32>,
        gather_k: usize,
        target_sign: f32,
    ) -> usize {
        let cvec: Vec<u32> = cluster.iter().copied().collect();
        let local = gather_local(points, &cvec, gather_k);
        let hull_cells = rebuild_cells(points, &local, &cvec).unwrap_or_default();
        let hull_by_gen: FxHashMap<u32, &RebuiltCell> =
            hull_cells.iter().map(|c| (c.generator, c)).collect();
        let adj =
            |a: [u32; 3], b: [u32; 3]| a != b && a.iter().filter(|x| b.contains(x)).count() == 2;

        let mut filled = 0usize;
        for &g in &cvec {
            let mut verts: std::collections::BTreeSet<[u32; 3]> = std::collections::BTreeSet::new();
            // Rim: triples containing g that trusted (non-cluster) cells attribute to it.
            for (h, list) in self.cells.iter().enumerate() {
                if cluster.contains(&(h as u32)) {
                    continue;
                }
                for &v in list {
                    let t = self.vkey[v as usize];
                    if t.contains(&g) {
                        verts.insert(t);
                    }
                }
            }
            // Interior: g's hull-fan triples whose other two gens are in the cluster.
            if let Some(rc) = hull_by_gen.get(&g) {
                for &t in &rc.vertices {
                    if t.iter().filter(|x| **x != g).all(|x| cluster.contains(x)) {
                        verts.insert(t);
                    }
                }
            }
            let vs: Vec<[u32; 3]> = verts.into_iter().collect();
            if vs.len() < 3 {
                continue;
            }
            // Robust cyclic order: sort the (rim+interior) vertex set by the
            // angular position of each circumcenter in g's tangent plane. Unlike
            // a greedy shared-generator cycle walk, this never bails on a
            // non-manifold / degenerate vertex set — any residual inconsistency
            // is caught by the grow loop and the whole-diagram never-worse gate.
            let order = order_cell_angular(points, g, &vs);
            self.splice_generator(points, g, &order, target_sign);
            filled += 1;
        }
        let _ = adj;
        filled
    }

    /// Flatten back into assembled global arrays `(vertices, cells, cell_indices)`.
    pub fn into_flat(self) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
        let total: usize = self.cells.iter().map(|c| c.len()).sum();
        let mut cells = Vec::with_capacity(self.cells.len());
        let mut cell_indices = Vec::with_capacity(total);
        for list in &self.cells {
            let start = cell_indices.len() as u32;
            cell_indices.extend_from_slice(list);
            cells.push(VoronoiCell::new(start, list.len() as u16));
        }
        (self.vpos, cells, cell_indices)
    }
}

/// The cell producer used by the escalation loop. Source-agnostic seam:
/// `rebuild_cells` (local-hull dual) is the current implementation; an exact
/// local re-clip can be dropped in with the same signature.
pub type CellSource = fn(&[Vec3], &[u32], &[u32]) -> Option<Vec<RebuiltCell>>;

/// Outcome of an escalation pass (diagnostics for tests / probes).
#[derive(Debug, Default, Clone, Copy)]
pub struct EscalationStats {
    /// Connected components of the defect-edge graph that were processed.
    pub components: usize,
    /// Total grow rounds across all components.
    pub rounds: usize,
    /// Distinct generators whose cells were rebuilt and spliced.
    pub spliced_generators: usize,
    /// Components that hit the round cap without a clean rim (a residual).
    pub stuck_components: usize,
}

/// Connected components of the undirected defect-edge graph (union-find over the
/// generators named by the defect pairs).
fn defect_components(pairs: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut ids: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    ids.sort_unstable();
    ids.dedup();
    let index: FxHashMap<u32, usize> = ids.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &(a, b) in pairs {
        let (ra, rb) = (find(&mut parent, index[&a]), find(&mut parent, index[&b]));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut groups: FxHashMap<usize, Vec<u32>> = FxHashMap::default();
    for (i, &g) in ids.iter().enumerate() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(g);
    }
    groups.into_values().collect()
}

/// Resolve the near-cocircular defect residual by exact local rebuild.
///
/// For each connected component of the defect-edge graph, rebuild its
/// generators (and enough ring to triangulate them) as ONE local subdivision
/// via `source`, splice the rebuilt cells into `work`, and GROW the spliced set
/// until every edge incident to it pairs — i.e. until the component's rim lands
/// on well-conditioned vertices the surrounding (unspliced) cells already share.
///
/// `gather_k` is the per-seed neighbor count for the local set; `max_rounds`
/// caps the grow loop per component.
pub fn escalate_diagram(
    points: &[Vec3],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    gather_k: usize,
    max_rounds: usize,
    source: CellSource,
) -> EscalationStats {
    let source = pick_source(source);
    let gather_k = std::env::var("S2_ESCALATE_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(gather_k);
    let max_rounds = std::env::var("S2_ESCALATE_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(max_rounds);

    let mut stats = EscalationStats::default();
    let mut spliced: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

    // Global winding convention, sampled before any splice (skip defect cells).
    let defect_gens: std::collections::BTreeSet<u32> =
        defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let target_sign = work.winding_convention(points, &defect_gens);

    if std::env::var("S2_ESCALATE_PROBE_A0_LOCALHULL").is_ok() {
        a0_full_exact_vs_flag(points, work, gather_k);
        return stats;
    }
    if std::env::var("S2_ESCALATE_PROBE_A0").is_ok() {
        // Stash the fast per-cell triples + effective points so a test can build
        // the TRUE exact reference (delaunator, exact predicates) and classify.
        let triples: Vec<Vec<[u32; 3]>> = work
            .cells
            .iter()
            .map(|c| c.iter().map(|&v| work.vkey[v as usize]).collect())
            .collect();
        A0_STASH.with(|s| *s.borrow_mut() = Some((points.to_vec(), triples)));
        return stats;
    }

    if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
        // Aliasing: how many vids share a triple already used by an earlier vid?
        let mut seen: std::collections::HashSet<[u32; 3]> = std::collections::HashSet::new();
        let mut aliased = 0usize;
        for &k in &work.vkey {
            if !seen.insert(k) {
                aliased += 1;
            }
        }
        eprintln!(
            "escalate pre-splice: target_sign={target_sign}, defect_pairs={}, \
             vkey_aliased={aliased}/{}, my-scan unpaired-implicated gens={}",
            defect_pairs.len(),
            work.vkey.len(),
            work.unpaired_generators().len()
        );
    }

    // Isolation probe: splice a few KNOWN-GOOD non-defect cells one at a time
    // and report how many new unpaired edges each creates. A correctly rebuilt
    // good cell is identical to its fast cell (pure reuse) and must create ZERO
    // new unpaired edges; a nonzero count is a pure splice bug (winding/order).
    if std::env::var("S2_ESCALATE_PROBE_GOOD").is_ok() {
        let base = work.unpaired_generators().len();
        let mut checked = 0;
        for g in 0..work.cells.len() as u32 {
            if defect_gens.contains(&g) || work.cells[g as usize].len() < 3 {
                continue;
            }
            let before_len = work.vpos.len();
            let local = gather_local(points, &[g], gather_k);
            let Some(rebuilt) = source(points, &local, &[g]) else {
                continue;
            };
            let before = work.unpaired_generators().len();
            for cell in &rebuilt {
                work.splice_generator(points, cell.generator, &cell.vertices, target_sign);
            }
            let after = work.unpaired_generators().len();
            let minted = work.vpos.len() - before_len;
            eprintln!(
                "  PROBE good g={g}: unpaired {before}->{after} (base {base}), minted={minted}, \
                 cell_len={}",
                work.cells[g as usize].len()
            );
            checked += 1;
            if checked >= 6 {
                break;
            }
        }
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_FILL").is_ok() {
        // Proper cluster fill: pin each cluster cell's RIM from trusted (non-
        // cluster) neighbors (their attributed triples), and fill the INTERIOR
        // (cluster–cluster vertices the neighbors can't see) from the exact hull
        // over the cluster. Order the union into a cycle. This breaks the
        // consensus deadlock (adjacent malformed cells) without re-Delaunaying
        // the rim. Does it make the tangled-cluster seeds valid?
        // Cluster = all BROKEN (unpaired-incident) cells ∪ malformed cells. The
        // cross-cell-divergence cells are broken-but-individually-valid, so the
        // malformed-only set under-identifies the cluster and wrongly trusts them
        // as rim.
        let (_edges, broken) = work.unpaired_edges_and_cells();
        let mut cluster: std::collections::BTreeSet<u32> = broken.iter().copied().collect();
        for g in work.malformed_cells() {
            cluster.insert(g);
        }
        if cluster.is_empty() {
            eprintln!("FILL: no defect cells");
            return stats;
        }
        let base = work.unpaired_edges_and_cells().0.len();
        let k = std::env::var("S2_ESCALATE_CLUSTER_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64usize);
        let max_grow = std::env::var("S2_ESCALATE_ROUNDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30usize);

        let adj =
            |a: [u32; 3], b: [u32; 3]| a != b && a.iter().filter(|x| b.contains(x)).count() == 2;
        let mut converged = false;
        for round in 0..max_grow {
            let cvec: Vec<u32> = cluster.iter().copied().collect();
            let local = gather_local(points, &cvec, k);
            let hull_cells = rebuild_cells(points, &local, &cvec).unwrap_or_default();
            let hull_by_gen: FxHashMap<u32, &RebuiltCell> =
                hull_cells.iter().map(|c| (c.generator, c)).collect();
            for &g in &cvec {
                // Rim: triples containing g that trusted (non-cluster) cells attribute to it.
                let mut verts: std::collections::BTreeSet<[u32; 3]> =
                    std::collections::BTreeSet::new();
                for (h, list) in work.cells.iter().enumerate() {
                    if cluster.contains(&(h as u32)) {
                        continue;
                    }
                    for &v in list {
                        let t = work.vkey[v as usize];
                        if t.contains(&g) {
                            verts.insert(t);
                        }
                    }
                }
                // Interior: g's hull-fan triples whose other two gens are in the cluster.
                if let Some(rc) = hull_by_gen.get(&g) {
                    for &t in &rc.vertices {
                        if t.iter().filter(|x| **x != g).all(|x| cluster.contains(x)) {
                            verts.insert(t);
                        }
                    }
                }
                let vs: Vec<[u32; 3]> = verts.into_iter().collect();
                let n = vs.len();
                if n < 3 {
                    continue;
                }
                let mut used = vec![false; n];
                let mut order = vec![vs[0]];
                used[0] = true;
                let mut ok = true;
                for _ in 1..n {
                    let cur = *order.last().unwrap();
                    match (0..n).find(|&j| !used[j] && adj(cur, vs[j])) {
                        Some(j) => {
                            used[j] = true;
                            order.push(vs[j]);
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && adj(*order.last().unwrap(), order[0]) {
                    work.splice_generator(points, g, &order, target_sign);
                }
            }
            let implicated = work.unpaired_generators();
            let new: Vec<u32> = implicated
                .iter()
                .copied()
                .filter(|g| !cluster.contains(g))
                .collect();
            if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
                eprintln!(
                    "  FILL round {round}: cluster={} implicated={} new={}",
                    cluster.len(),
                    implicated.len(),
                    new.len()
                );
            }
            if new.is_empty() {
                converged = implicated.is_empty();
                break;
            }
            for g in new {
                cluster.insert(g);
            }
        }
        let after = work.unpaired_edges_and_cells().0.len();
        eprintln!(
            "FILL: final cluster={} | unpaired {base}->{after} | {}",
            cluster.len(),
            if converged && after == 0 {
                "CONVERGED CLEAN (fill-with-grow is local → post-hoc tractable)"
            } else {
                "did NOT converge (grew unbounded / residual)"
            },
        );
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_CLUSTER").is_ok() {
        // Test whether rebuilding the connected MALFORMED CLUSTER stays local.
        // Cluster = malformed cells connected via a shared vertex. Rebuild all
        // cluster cells from one exact hull over gather(cluster, k); splice only
        // cluster cells; measure cascade (implicated outside the cluster).
        let malformed = work.malformed_cells();
        // Connected components of malformed cells by shared vid.
        let idx: FxHashMap<u32, usize> =
            malformed.iter().enumerate().map(|(i, &g)| (g, i)).collect();
        let mut parent: Vec<usize> = (0..malformed.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut vid_owner: FxHashMap<u32, usize> = FxHashMap::default();
        for (i, &g) in malformed.iter().enumerate() {
            for &v in &work.cells[g as usize] {
                if let Some(&j) = vid_owner.get(&v) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    if a != b {
                        parent[a] = b;
                    }
                } else {
                    vid_owner.insert(v, i);
                }
            }
        }
        let mut clusters: FxHashMap<usize, Vec<u32>> = FxHashMap::default();
        for (i, &g) in malformed.iter().enumerate() {
            clusters.entry(find(&mut parent, i)).or_default().push(g);
        }
        let _ = idx;
        let k = std::env::var("S2_ESCALATE_CLUSTER_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(48usize);
        let base = work.unpaired_edges_and_cells().0.len();
        eprintln!(
            "CLUSTER: {} malformed cells in {} clusters (base unpaired={base})",
            malformed.len(),
            clusters.len()
        );
        for cluster in clusters.values() {
            let cset: std::collections::BTreeSet<u32> = cluster.iter().copied().collect();
            let local = gather_local(points, cluster, k);
            let Some(rebuilt) = source(points, &local, cluster) else {
                continue;
            };
            let mut minted = 0usize;
            for cell in &rebuilt {
                let before = work.vpos.len();
                work.splice_generator(points, cell.generator, &cell.vertices, target_sign);
                minted += work.vpos.len() - before;
            }
            let implicated = work.unpaired_generators();
            let outside = implicated.iter().filter(|g| !cset.contains(g)).count();
            eprintln!(
                "  cluster size={} rebuilt={} minted={minted} → implicated={} outside_cluster={outside}",
                cluster.len(),
                rebuilt.len(),
                implicated.len(),
            );
        }
        let after = work.unpaired_edges_and_cells().0.len();
        eprintln!(
            "CLUSTER: unpaired {base}->{after} | {}",
            if after == 0 {
                "CLEAN (cluster rebuild is local)"
            } else {
                "residual / cascade"
            }
        );
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_CONSENSUS").is_ok() {
        // Probe C2: rebuild each MALFORMED cell from its NEIGHBORS' consensus —
        // the set of vertex triples that other cells attribute to it, ordered
        // into a cycle. This adopts the neighbors' (valid, possibly non-Delaunay)
        // diagonals instead of imposing Delaunay, so it should NOT cascade. Test:
        // does fixing only the malformed cell(s) resolve all unpaired edges?
        let (edges, _broken) = work.unpaired_edges_and_cells();
        let base = edges.len();
        // Malformed cells (neighbor in >=3 of own verts).
        let mut malformed: Vec<u32> = Vec::new();
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        for (g, list) in work.cells.iter().enumerate() {
            if list.len() < 3 {
                continue;
            }
            counts.clear();
            for &v in list {
                for &x in work.vkey[v as usize].iter() {
                    if x != g as u32 {
                        *counts.entry(x).or_default() += 1;
                    }
                }
            }
            if counts.values().any(|&c| c >= 3) {
                malformed.push(g as u32);
            }
        }
        eprintln!("CONSENSUS: {} malformed cells to rebuild", malformed.len());

        // One pass: gen -> triples other cells attribute to it (the consensus).
        let mut consensus: FxHashMap<u32, std::collections::BTreeSet<[u32; 3]>> =
            FxHashMap::default();
        let mal_set: std::collections::BTreeSet<u32> = malformed.iter().copied().collect();
        for (g, list) in work.cells.iter().enumerate() {
            if mal_set.contains(&(g as u32)) {
                continue; // skip the malformed cell's own (bad) boundary
            }
            for &v in list {
                let t = work.vkey[v as usize];
                for &x in t.iter() {
                    if mal_set.contains(&x) {
                        consensus.entry(x).or_default().insert(t);
                    }
                }
            }
        }

        // Order a generator's consensus vertices into a cycle: (g,a,b)~(g,a,c)
        // share edge g-a (exactly 2 common gens). Each vertex must have exactly
        // two such neighbors for a clean cycle.
        let order_cycle = |g: u32, verts: &[[u32; 3]]| -> Option<Vec<[u32; 3]>> {
            let n = verts.len();
            if n < 3 {
                return None;
            }
            let adj = |a: [u32; 3], b: [u32; 3]| -> bool {
                let common = a.iter().filter(|x| b.contains(x)).count();
                common == 2 && a != b // share g + exactly one other
            };
            let mut used = vec![false; n];
            let mut order = vec![verts[0]];
            used[0] = true;
            for _ in 1..n {
                let cur = *order.last().unwrap();
                let nxt = (0..n).find(|&j| !used[j] && adj(cur, verts[j]));
                match nxt {
                    Some(j) => {
                        used[j] = true;
                        order.push(verts[j]);
                    }
                    None => return None, // broken chain — consensus inconsistent
                }
            }
            // close the loop
            if adj(*order.last().unwrap(), order[0]) {
                let _ = g;
                Some(order)
            } else {
                None
            }
        };

        let mut rebuilt_ok = 0usize;
        for &g in &malformed {
            let Some(set) = consensus.get(&g) else {
                eprintln!("  g={g}: no consensus vertices");
                continue;
            };
            let verts: Vec<[u32; 3]> = set.iter().copied().collect();
            match order_cycle(g, &verts) {
                Some(cycle) => {
                    work.splice_generator(points, g, &cycle, target_sign);
                    rebuilt_ok += 1;
                    eprintln!(
                        "  g={g}: consensus cycle len {} (was {})",
                        cycle.len(),
                        verts.len()
                    );
                }
                None => eprintln!(
                    "  g={g}: {} consensus verts do NOT form a clean cycle",
                    verts.len()
                ),
            }
        }
        let after = work.unpaired_edges_and_cells().0.len();
        let implicated = work.unpaired_generators().len();
        eprintln!(
            "CONSENSUS: rebuilt {rebuilt_ok}/{} | unpaired {base}->{after} | implicated={implicated} | {}",
            malformed.len(),
            if after == 0 { "CLEAN (contained consensus fix works!)" } else { "residual remains" },
        );
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_MALFORMED").is_ok() {
        // Probe C: a Voronoi cell is convex, so each neighbor's bisector gives at
        // most ONE boundary edge = exactly 2 vertices sharing that neighbor. A
        // neighbor appearing in >=3 of a cell's vertex triples ⇒ the f32 clip
        // produced an INVALID (non-convex/self-overlapping) polygon. Count such
        // malformed cells across the whole diagram and their overlap with the
        // broken (unpaired-incident) set: contained bug vs pervasive.
        let (_edges, broken) = work.unpaired_edges_and_cells();
        let mut malformed: Vec<(u32, u32, u32)> = Vec::new(); // (gen, worst_neighbor, count)
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        for (g, list) in work.cells.iter().enumerate() {
            if list.len() < 3 {
                continue;
            }
            counts.clear();
            for &v in list {
                for &x in work.vkey[v as usize].iter() {
                    if x != g as u32 {
                        *counts.entry(x).or_default() += 1;
                    }
                }
            }
            if let Some((&m, &c)) = counts.iter().max_by_key(|(_, &c)| c) {
                if c >= 3 {
                    malformed.push((g as u32, m, c));
                }
            }
        }
        let malformed_set: std::collections::BTreeSet<u32> =
            malformed.iter().map(|&(g, _, _)| g).collect();
        let broken_malformed = broken.iter().filter(|g| malformed_set.contains(g)).count();
        let malformed_not_broken = malformed_set.iter().filter(|g| !broken.contains(g)).count();
        eprintln!(
            "MALFORMED: {} cells with a neighbor in >=3 verts (of {} total) | \
             broken={} of which malformed={} | malformed-but-not-broken={}",
            malformed_set.len(),
            work.cells.len(),
            broken.len(),
            broken_malformed,
            malformed_not_broken,
        );
        let mut sample = malformed.clone();
        sample.sort_unstable_by_key(|&(_, _, c)| std::cmp::Reverse(c));
        for &(g, m, c) in sample.iter().take(8) {
            eprintln!(
                "  malformed g={g}: neighbor {m} appears {c}x ({} verts){}",
                work.cells[g as usize].len(),
                if broken.contains(&g) { " [broken]" } else { "" },
            );
        }
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_STABLE").is_ok() {
        // Path-A test: ONE fixed hull over a big neighborhood of the defect;
        // grow the spliced set WITHIN that fixed hull (cells' fans never shift
        // between rounds) until the rim pairs with the surrounding fast cells.
        // Converges at a bounded size ⇔ the near-cocircular component is bounded
        // and rebuild-the-component is viable. Diverges/exhausts ⇔ totality.
        let seeds: Vec<u32> = defect_gens.iter().copied().collect();
        let big = std::env::var("S2_ESCALATE_BIG_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(400usize);
        let local = gather_local(points, &seeds, big);
        let local_set: std::collections::BTreeSet<u32> = local.iter().copied().collect();
        let Some(rebuilt_all) = source(points, &local, &local) else {
            return stats;
        };
        let reb: FxHashMap<u32, &RebuiltCell> =
            rebuilt_all.iter().map(|c| (c.generator, c)).collect();
        let mut splice_set: std::collections::BTreeSet<u32> = seeds.iter().copied().collect();
        let mut converged = false;
        for round in 0..200 {
            for &g in &splice_set {
                if let Some(rc) = reb.get(&g) {
                    work.splice_generator(points, g, &rc.vertices, target_sign);
                }
            }
            let implicated = work.unpaired_generators();
            let new: Vec<u32> = implicated
                .iter()
                .copied()
                .filter(|g| local_set.contains(g) && reb.contains_key(g) && !splice_set.contains(g))
                .collect();
            let outside = implicated.iter().filter(|g| !local_set.contains(g)).count();
            if new.is_empty() {
                converged = implicated.is_empty();
                eprintln!(
                    "STABLE round {round}: splice_set={} implicated={} (outside_hull={outside}) — STOP",
                    splice_set.len(),
                    implicated.len(),
                );
                break;
            }
            if round % 5 == 0 {
                eprintln!(
                    "STABLE round {round}: splice_set={} implicated={} new={}",
                    splice_set.len(),
                    implicated.len(),
                    new.len(),
                );
            }
            for g in new {
                splice_set.insert(g);
            }
        }
        let final_unpaired = work.unpaired_edges_and_cells().0.len();
        eprintln!(
            "STABLE: big_k={big} hull={} | final splice_set={} | final_unpaired={final_unpaired} | {}",
            local.len(),
            splice_set.len(),
            if converged && final_unpaired == 0 {
                "CONVERGED CLEAN (bounded component → path A viable)"
            } else {
                "did NOT converge clean (grew to hull edge / residual remains)"
            }
        );
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_DUMP").is_ok() {
        use crate::knn_clipping::canonical::in_circle_sphere_sign;
        let (edges, broken) = work.unpaired_edges_and_cells();
        eprintln!(
            "DUMP: {} unpaired edges, {} broken cells",
            edges.len(),
            broken.len()
        );
        for &(va, vb) in &edges {
            let (ka, kb) = (work.vkey[va as usize], work.vkey[vb as usize]);
            // shared generators g,m of this Voronoi edge; thirds are the rest.
            let shared: Vec<u32> = ka.iter().copied().filter(|x| kb.contains(x)).collect();
            eprintln!("  edge vids({va},{vb}) triples {ka:?},{kb:?} shared(g,m)={shared:?}",);
            // For the contested vertex (g,a,b), the cutter is the OTHER cell's
            // non-shared third. Report in_circle for both orderings.
            if shared.len() == 2 {
                let (g, m) = (shared[0], shared[1]);
                let third_a: Vec<u32> =
                    ka.iter().copied().filter(|x| !shared.contains(x)).collect();
                let third_b: Vec<u32> =
                    kb.iter().copied().filter(|x| !shared.contains(x)).collect();
                let p = |i: u32| points[i as usize];
                if let (Some(&xa), Some(&xb)) = (third_a.first(), third_b.first()) {
                    // Does xb cut vertex (g,m,xa)? and does xa cut (g,m,xb)?
                    let s1 = in_circle_sphere_sign(p(g), p(m), p(xa), p(xb));
                    let s2 = in_circle_sphere_sign(p(g), p(m), p(xb), p(xa));
                    eprintln!(
                        "    in_circle(g,m,{xa}; cut by {xb})={s1}  in_circle(g,m,{xb}; cut by {xa})={s2}",
                    );
                }
            }
        }
        for &g in &broken {
            let tris: Vec<[u32; 3]> = work.cells[g as usize]
                .iter()
                .map(|&v| work.vkey[v as usize])
                .collect();
            eprintln!("  broken cell g={g} ({} verts): {tris:?}", tris.len());
        }
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_SURGICAL").is_ok() {
        // Codex's surgical-consensus test: for the residual unpaired edges,
        // compute the rewrite CLOSURE (all generators named by both endpoint
        // triples of every unpaired edge), rebuild ONLY those cells consistently
        // (one shot, no grow), and measure whether the newly-implicated set stays
        // INSIDE the closure (surgical repair is local → real) or jumps OUTSIDE
        // into the near-cocircular sea (cascades → surgical dead).
        let (edges, _broken) = work.unpaired_edges_and_cells();
        let base_unpaired = edges.len();
        let mut closure: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for &(va, vb) in &edges {
            for &x in work.vkey[va as usize]
                .iter()
                .chain(work.vkey[vb as usize].iter())
            {
                closure.insert(x);
            }
        }
        let closure_vec: Vec<u32> = closure.iter().copied().collect();
        let k = std::env::var("S2_ESCALATE_CLOSURE_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(48usize);
        let local = gather_local(points, &closure_vec, k);
        if let Some(rebuilt) = source(points, &local, &closure_vec) {
            let mut minted = 0usize;
            for cell in &rebuilt {
                let before = work.vpos.len();
                work.splice_generator(points, cell.generator, &cell.vertices, target_sign);
                minted += work.vpos.len() - before;
            }
            let implicated = work.unpaired_generators();
            let outside = implicated.iter().filter(|g| !closure.contains(g)).count();
            let after = work.unpaired_edges_and_cells().0.len();
            eprintln!(
                "SURGICAL: closure_gens={} rebuilt={} (minted={minted}) | \
                 unpaired {base_unpaired}->{after} | implicated={} outside_closure={outside}",
                closure_vec.len(),
                rebuilt.len(),
                implicated.len(),
            );
            eprintln!(
                "  => {}",
                if outside == 0 && after == 0 {
                    "CLEAN local repair (surgical real)"
                } else if outside == 0 {
                    "contained but unresolved (needs closure refinement)"
                } else {
                    "CASCADES into the sea (surgical dead for this defect)"
                }
            );
        }
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_CLASSIFY").is_ok() {
        // Step (b): on the UNSPLICED fast diagram, separate the defect patch into
        //   - ACTUALLY-UNPAIRED  : cells incident to a real boundary edge (defect)
        //   - PAIRED-NON-DELAUNAY: locally valid (all edges pair) but fan != exact
        //   - PAIRED-AGREE       : locally valid and fan == exact
        // and measure the true broken-core blast radius + connectedness. If the
        // broken core is a small bounded cluster surrounded by paired-non-Delaunay
        // cells, a source-compatible re-clip (step a) has a clean rim to hit.
        let (edges, broken) = work.unpaired_edges_and_cells();

        // Connected components of broken cells (adjacency = shared vertex id).
        let broken_vec: Vec<u32> = broken.iter().copied().collect();
        let idx: FxHashMap<u32, usize> = broken_vec
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i))
            .collect();
        let mut parent: Vec<usize> = (0..broken_vec.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut vid_owner: FxHashMap<u32, usize> = FxHashMap::default();
        for (i, &g) in broken_vec.iter().enumerate() {
            for &v in &work.cells[g as usize] {
                if let Some(&j) = vid_owner.get(&v) {
                    let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                } else {
                    vid_owner.insert(v, i);
                }
            }
        }
        let mut comps: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for i in 0..broken_vec.len() {
            comps.insert(find(&mut parent, i));
        }
        let _ = idx;
        eprintln!(
            "CLASSIFY: unpaired_edges={} broken_cells={} broken_components={}",
            edges.len(),
            broken.len(),
            comps.len(),
        );

        // Classify the defect patch's deep-interior cells against exact Delaunay.
        let seeds: Vec<u32> = defect_gens.iter().copied().collect();
        let big = std::env::var("S2_ESCALATE_BIG_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600usize);
        let core_k = std::env::var("S2_ESCALATE_CORE_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120usize);
        let local = gather_local(points, &seeds, big);
        let core: std::collections::BTreeSet<u32> =
            gather_local(points, &seeds, core_k).into_iter().collect();
        if let Some(rebuilt) = source(points, &local, &local) {
            let reb_by_gen: FxHashMap<u32, &RebuiltCell> =
                rebuilt.iter().map(|c| (c.generator, c)).collect();
            let (mut c_broken, mut c_nondel, mut c_agree, mut checked) = (0, 0, 0, 0);
            for &g in &local {
                if !core.contains(&g) {
                    continue;
                }
                let Some(rc) = reb_by_gen.get(&g) else {
                    continue;
                };
                checked += 1;
                if broken.contains(&g) {
                    c_broken += 1;
                    continue;
                }
                let fast: std::collections::BTreeSet<[u32; 3]> = work.cells[g as usize]
                    .iter()
                    .map(|&v| work.vkey[v as usize])
                    .collect();
                let reb: std::collections::BTreeSet<[u32; 3]> =
                    rc.vertices.iter().copied().collect();
                if fast == reb {
                    c_agree += 1;
                } else {
                    c_nondel += 1;
                }
            }
            eprintln!(
                "CLASSIFY core (big_k={big} core_k={core_k}): checked={checked} \
                 broken={c_broken} paired_non_delaunay={c_nondel} paired_agree={c_agree}",
            );
        }
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_REGION").is_ok() {
        // Gather a BIG neighborhood around all defect seeds, rebuild every
        // gathered cell, and count how many disagree with their fast fan. This
        // is the size of the fast≠exact region — i.e. how far out a clean rim
        // (where fast==exact) sits. Bloated fast cells are also tallied.
        let seeds: Vec<u32> = match std::env::var("S2_ESCALATE_PROBE_SEED")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        {
            Some(s) => vec![s],
            None => defect_gens.iter().copied().collect(),
        };
        let big = std::env::var("S2_ESCALATE_BIG_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(400usize);
        let local = gather_local(points, &seeds, big);
        // Inner core: cells deep inside the big context hull (no cap artifact).
        let core_k = std::env::var("S2_ESCALATE_CORE_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(big / 4);
        let core: std::collections::BTreeSet<u32> =
            gather_local(points, &seeds, core_k).into_iter().collect();
        let Some(rebuilt) = source(points, &local, &local) else {
            return stats;
        };
        let reb_by_gen: FxHashMap<u32, &RebuiltCell> =
            rebuilt.iter().map(|c| (c.generator, c)).collect();
        let mut disagree = 0usize;
        let mut bloated = 0usize;
        let mut checked = 0usize;
        for &g in &local {
            if !core.contains(&g) {
                continue; // only judge deep-interior cells
            }
            let Some(rc) = reb_by_gen.get(&g) else {
                continue;
            };
            checked += 1;
            let fast: std::collections::BTreeSet<[u32; 3]> = work.cells[g as usize]
                .iter()
                .map(|&v| work.vkey[v as usize])
                .collect();
            let reb: std::collections::BTreeSet<[u32; 3]> = rc.vertices.iter().copied().collect();
            if fast != reb {
                disagree += 1;
            }
            if work.cells[g as usize].len() > 9 {
                bloated += 1;
            }
        }
        eprintln!(
            "escalate region probe (big_k={big}): local={} rebuilt={} checked={checked} \
             disagree_with_fast={disagree} bloated_fast_cells={bloated}",
            local.len(),
            rebuilt.len(),
        );
        return stats;
    }

    if std::env::var("S2_ESCALATE_PROBE_DEFECT").is_ok() {
        // Compare each defect cell's FAST fan (existing vids → triples) against
        // its REBUILT fan (exact-hull triples). How many triples match tells us
        // whether the disagreement is the localized broken edge(s) or a broad
        // near-cocircular re-triangulation.
        for &g in defect_gens.iter() {
            let fast: std::collections::BTreeSet<[u32; 3]> = work.cells[g as usize]
                .iter()
                .map(|&v| work.vkey[v as usize])
                .collect();
            let local = gather_local(points, &[g], gather_k);
            let Some(rebuilt) = source(points, &local, &[g]) else {
                continue;
            };
            let reb: std::collections::BTreeSet<[u32; 3]> =
                rebuilt[0].vertices.iter().copied().collect();
            let common = fast.intersection(&reb).count();
            eprintln!(
                "  PROBE defect g={g}: fast_len={} reb_len={} common={common} \
                 fast_only={} reb_only={}",
                fast.len(),
                reb.len(),
                fast.len() - common,
                reb.len() - common,
            );
        }
        return stats;
    }

    // Cluster fill-with-grow (the mechanism the step-C probes established):
    // the defect is a connected cluster of cells whose f32 clip is mutually
    // inconsistent (malformed and/or cross-cell-divergent). Rebuild the whole
    // cluster as one consistent patch — each cell's RIM pinned from its trusted
    // (non-cluster) neighbors' attributed triples, its cluster-INTERIOR vertices
    // taken from one exact local hull — then grow the cluster by any newly
    // unpaired generator until the rim is well-conditioned (fast == exact there).
    // Pinning the rim is what stops the Delaunay-rebuild cascade; growing closes
    // the seam. Converges cleanly on most mega seeds; a residual that does not
    // close is left for the caller's whole-diagram never-worse gate.
    let _ = (source, defect_pairs, &defect_gens);
    let mut cluster = work.defect_cluster();
    for _ in 0..max_rounds {
        if cluster.is_empty() {
            break;
        }
        stats.rounds += 1;
        stats.components = stats.components.max(1);
        let n_spliced = work.fill_cluster_pass(points, &cluster, gather_k, target_sign);
        for &g in &cluster {
            spliced.insert(g);
        }
        let _ = n_spliced;
        let implicated = work.unpaired_generators();
        let new: Vec<u32> = implicated
            .iter()
            .copied()
            .filter(|g| !cluster.contains(g))
            .collect();
        if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
            eprintln!(
                "  fill round {}: cluster={} implicated={} new={}",
                stats.rounds,
                cluster.len(),
                implicated.len(),
                new.len()
            );
        }
        if new.is_empty() {
            if !implicated.is_empty() {
                stats.stuck_components = 1;
            }
            break;
        }
        for g in new {
            cluster.insert(g);
        }
    }
    stats.spliced_generators = spliced.len();
    if std::env::var("S2_ESCALATE_DEBUG").is_ok() {
        eprintln!(
            "escalate: {:?}; final unpaired edges={}",
            stats,
            work.unpaired_edges_and_cells().0.len(),
        );
    }
    stats
}

/// A0 (clip-time-exact de-risk): build the FULL exact diagram (one exact
/// `local_hull` cell per generator), and answer three questions that together
/// decide whether the adaptive primary clip (option A) is sound:
///   1. Is the full exact diagram self-consistent (every triple-keyed edge pairs)?
///      — i.e. do per-cell exact rebuilds agree across cells (complete-considered
///      / tie concerns), so "valid by construction" actually holds.
///   2. Which cells CHANGED vs the fast diagram (exact triple-set ≠ fast set)?
///   3. Is `changed ⊆ flagged` achievable by a near-cocircularity band, and at
///      what trip rate (cost)? The flag here is the IDEAL geometric `in_circle`
///      margin (the production chart-margin flag is a cheap proxy for it; if the
///      ideal flag can't separate, no proxy can).
fn a0_full_exact_vs_flag(points: &[Vec3], work: &WorkingDiagram, gather_k: usize) {
    let n = work.cells.len();
    // Normalized in-circle margin of the quad (g,m,x,y): how close the diagonal
    // is to a 4-cocircular tie. 0 = exactly cocircular (max suspicion).
    let quad_margin = |g: u32, m: u32, x: u32, y: u32| -> f64 {
        let p = |i: u32| {
            let v = points[i as usize];
            DVec3::new(v.x as f64, v.y as f64, v.z as f64)
        };
        let (a, b, c, d) = (p(g), p(m), p(x), p(y));
        // det[a-d, b-d, c-d] (orient3d) normalized by the edge lengths.
        let det = (a - d).cross(b - d).dot(c - d).abs();
        let denom = (a - d).length() * (b - d).length() * (c - d).length();
        if denom > 0.0 {
            det / denom
        } else {
            0.0
        }
    };

    // Build per-cell exact fans (triple-keyed) and the fast triple-sets.
    let mut exact: Vec<Vec<[u32; 3]>> = vec![Vec::new(); n];
    let mut changed = vec![false; n];
    let mut score = vec![f64::INFINITY; n]; // min quad margin over the cell
    let mut built = 0usize;
    for g in 0..n as u32 {
        let local = gather_local(points, &[g], gather_k);
        let Some(cells) = rebuild_cells(points, &local, &[g]) else {
            continue;
        };
        let Some(rc) = cells.into_iter().next() else {
            continue;
        };
        built += 1;
        let fan = rc.vertices;
        // near-cocircularity score: min margin over consecutive fan faces that
        // share edge (g,m) — the cell's own diagonals.
        let f = fan.len();
        for i in 0..f {
            let (t0, t1) = (fan[i], fan[(i + 1) % f]);
            if let Some(mm) = shared_neighbor(g, t0, t1) {
                let x = t0.iter().copied().find(|&v| v != g && v != mm);
                let y = t1.iter().copied().find(|&v| v != g && v != mm);
                if let (Some(x), Some(y)) = (x, y) {
                    score[g as usize] = score[g as usize].min(quad_margin(g, mm, x, y));
                }
            }
        }
        let fast_set: std::collections::BTreeSet<[u32; 3]> = work.cells[g as usize]
            .iter()
            .map(|&v| work.vkey[v as usize])
            .collect();
        let exact_set: std::collections::BTreeSet<[u32; 3]> = fan.iter().copied().collect();
        changed[g as usize] = fast_set != exact_set;
        exact[g as usize] = fan;
    }

    // (1) Is the full exact diagram self-consistent? Triple-level edge pairing:
    // each cell's consecutive triples form a directed edge; every undirected
    // edge must be used exactly twice, opposite directions.
    let mut dir: FxHashMap<([u32; 3], [u32; 3]), u32> = FxHashMap::default();
    for fan in &exact {
        let f = fan.len();
        if f < 3 {
            continue;
        }
        for i in 0..f {
            *dir.entry((fan[i], fan[(i + 1) % f])).or_default() += 1;
        }
    }
    let mut exact_unpaired = 0usize;
    let mut seen: std::collections::HashSet<([u32; 3], [u32; 3])> =
        std::collections::HashSet::new();
    for &(a, b) in dir.keys() {
        let key = if a <= b { (a, b) } else { (b, a) };
        if !seen.insert(key) {
            continue;
        }
        let fwd = dir.get(&(a, b)).copied().unwrap_or(0);
        let rev = dir.get(&(b, a)).copied().unwrap_or(0);
        if fwd != 1 || rev != 1 {
            exact_unpaired += 1;
        }
    }

    let n_changed = changed.iter().filter(|&&c| c).count();
    eprintln!(
        "A0: n={n} built_exact={built} | exact-diagram unpaired(triple)={exact_unpaired} \
         | changed_cells={n_changed}"
    );

    // (3) changed ⊆ flagged at several bands: a cell is flagged if score < band.
    // Report trip rate and any "changed but NOT flagged" (false negatives = seam).
    for band in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let flagged = |g: usize| score[g] < band;
        let trip = (0..n).filter(|&g| flagged(g)).count();
        let missed = (0..n).filter(|&g| changed[g] && !flagged(g)).count();
        eprintln!(
            "  band={band:>8.0e}: trip={trip} ({:.3}%) | changed_not_flagged={missed} {}",
            100.0 * trip as f64 / n as f64,
            if missed == 0 { "← superset OK" } else { "" },
        );
    }
}

/// A0 probe stash payload: (effective points, fast per-cell triples).
pub type A0Stash = (Vec<Vec3>, Vec<Vec<[u32; 3]>>);

thread_local! {
    /// A0 probe stash from the last build, for an exact-reference comparison
    /// test. See `take_a0_fast`.
    static A0_STASH: std::cell::RefCell<Option<A0Stash>> = const { std::cell::RefCell::new(None) };
}

/// Stash the assembled fast per-cell triple fans for A0 exact-reference probes.
pub(crate) fn stash_a0_fast(
    points: &[Vec3],
    keys: &ShardedVertexKeys,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) {
    let mut vkey = vec![[u32::MAX; 3]; keys.len()];
    keys.for_each(|vid, k| {
        if let Some(slot) = vkey.get_mut(vid as usize) {
            *slot = k;
        }
    });
    let triples: Vec<Vec<[u32; 3]>> = cells
        .iter()
        .map(|c| {
            cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()]
                .iter()
                .map(|&v| vkey[v as usize])
                .collect()
        })
        .collect();
    A0_STASH.with(|s| *s.borrow_mut() = Some((points.to_vec(), triples)));
}

/// Take the A0 stash (effective points + fast per-cell triples) from the last
/// build that ran with `S2_ESCALATE_PROBE_A0` set. Probe API.
pub fn take_a0_fast() -> Option<A0Stash> {
    A0_STASH.with(|s| s.borrow_mut().take())
}

/// Process-global enable for the escalation pass (probe / opt-in). Off by
/// default; the production fast path is unaffected until this is set.
static ESCALATE_ENABLED: AtomicBool = AtomicBool::new(false);
static REUSE_HIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_MISS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Enable or disable defect-driven escalation (probe API).
pub fn set_escalation_enabled(on: bool) {
    ESCALATE_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether escalation is currently enabled.
pub(crate) fn escalation_enabled() -> bool {
    ESCALATE_ENABLED.load(Ordering::Relaxed)
}
