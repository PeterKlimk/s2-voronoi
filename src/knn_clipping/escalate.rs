//! Defect-driven local repair (current state in `docs/correctness.md`).
//!
//! When post-assembly detection finds residual topology defects (unpaired
//! edges / degree-1-or-2 vertices), the affected neighborhood is rebuilt from
//! ONE consistent exact oracle and spliced back into the assembled diagram.
//! Because every repaired cell comes from the same oracle, repaired cells pair
//! with each other on shared edges by construction; on the well-conditioned rim
//! the oracle agrees with the fast clipper, so rebuilt rim vertices reuse the
//! surrounding cells' vertex ids and pair with the unspliced neighbors too.
//!
//! Two production oracles share one grow loop ([`repair_grow_loop`]):
//! - [`repair_local_hull`] (default, [`crate::RepairMode::Local3d`]): a
//!   normalized local 3D hull ([`LocalHull`], exact `orient3d`).
//! - [`repair_local_exact`] ([`crate::RepairMode::LocalProjected`]): exact 2D
//!   Delaunay (`robust::incircle`) in a single shared stereographic chart.
//!
//! The caller (`maybe_repair_effective`) commits the result only if the whole
//! repaired diagram passes strict validation — the never-worse gate.

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

#[cfg(feature = "escalate_probe")]
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
/// neighbors (largest dot product on the unit sphere). Brute force — probe/test
/// use only; the production path uses [`gather_knn_grid`].
#[cfg(feature = "escalate_probe")]
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

/// O(local) kNN gather used by the production repair oracles. For each seed,
/// walk the cube-map shell frontier nearest-first (the same machinery the point
/// locator uses) and collect the `k + 1` nearest generators, unioned with the
/// seeds — proportional to the local neighborhood rather than to `n`.
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
            if collected.len() > k {
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
#[cfg(feature = "escalate_probe")]
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
#[cfg(feature = "escalate_probe")]
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
#[cfg(feature = "escalate_probe")]
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

    // Seed → local index. On the deep-cascade cold path both `local_ids` and
    // `seeds` reach thousands, so an O(L) scan per seed is a real cost there.
    let local_of: FxHashMap<u32, usize> =
        local_ids.iter().enumerate().map(|(i, &g)| (g, i)).collect();

    let mut out = Vec::with_capacity(seeds.len());
    for &g in seeds {
        let Some(&lg) = local_of.get(&g) else {
            continue;
        };
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

/// Cyclically chain a generator's incident sorted-global triples into its
/// Voronoi fan: each consecutive pair shares the edge `g–x` (two common
/// generators). Returns `None` if the fan doesn't close (the generator is on the
/// gather frontier — an incomplete neighborhood, deferred to the grow loop), so
/// only genuinely-interior cells are spliced.
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

/// Union gather (2-RING) for a repair oracle: the closure, each closure cell's
/// current Voronoi neighbors (so a cluster-boundary cell's far Voronoi neighbors
/// are present), then each of those seeds' kNN (so every gathered cell's own
/// neighborhood is complete and its triangulated fan is the true Delaunay, not a
/// gather-boundary artifact).
///
/// CLUSTER-BOUNDARY cells are why the current cell's neighbors seed the gather:
/// a generator on the rim of a dense cluster has a Voronoi cell that reaches far
/// into the sparse region, so its true fan includes FAR generators a kNN gather
/// can't see. The fast path already found those far neighbors correctly; seeding
/// from the assembled cell's vertex triples keeps them, and the oracle only
/// RE-DECIDES the contested near-cocircular vertices.
fn gather_two_ring(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &WorkingDiagram,
    closure: &[u32],
    ring_k: usize,
) -> Vec<u32> {
    use std::collections::BTreeSet;

    // g's current Voronoi neighbors, read off its assembled cell's vertex triples.
    let cell_neighbors = |g: u32| -> Vec<u32> {
        let mut ns: Vec<u32> = work
            .boundary(g)
            .iter()
            .flat_map(|&v| work.vkey(v))
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
    gather_knn_grid(grid, scratch, slot_gen_map, points, &seeds2, ring_k)
}

/// Per-generator ready-to-splice fans for `closure`, read off ONE exact 2D
/// Delaunay (`robust::incircle`) built in a single shared stereographic chart
/// over a local gather.
///
/// A triple `(g,a,b)` is a Delaunay triangle (hence a Voronoi vertex of `g`) iff
/// no other nearby generator falls strictly inside its circumcircle. That
/// predicate is a pure function of the four points, so it returns the IDENTICAL
/// verdict no matter which cell evaluates it — the cross-cell consistency that
/// per-cell gnomonic-chart f64 rounding lacks. Correctness is local because a
/// point inside a closure triangle's circumcircle lies within ~2× the cell
/// radius, so it is present in the SHARED union gather every emptiness test
/// scans — a proper restricted local Delaunay, not per-generator triangle tests.
fn local_exact_fans(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &WorkingDiagram,
    closure: &[u32],
    ring_k: usize,
) -> FxHashMap<u32, Vec<[u32; 3]>> {
    use robust::Coord;
    use std::collections::BTreeSet;

    let local_ids = gather_two_ring(points, grid, scratch, slot_gen_map, work, closure, ring_k);

    // SINGLE shared stereographic chart over the gather, pole at the antipode of
    // the local centroid. One exact 2D triangulation produces all closure fans,
    // so repaired cells agree with each other.
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
    for t in &tri {
        let gs = [local_ids[t[0]], local_ids[t[1]], local_ids[t[2]]];
        let mut sorted = gs;
        sorted.sort_unstable();
        for &g in &gs {
            if closureset.contains(&g) {
                incident.entry(g).or_default().push(sorted);
            }
        }
    }
    incident
        .into_iter()
        .filter_map(|(g, tris)| chain_fan(&tris).map(|fan| (g, fan)))
        .collect()
}

/// Per-generator ready-to-splice fans for `closure`, read off ONE normalized
/// local 3D hull over a local gather. This is the default production oracle:
/// exact 3D construction has no single-chart/pole failure mode. `local_hull`
/// normalizes S2 directions before exact predicates, so it solves the crate's
/// spherical input problem rather than an off-sphere f32-radius hull problem.
fn local_hull_fans(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &WorkingDiagram,
    closure: &[u32],
    ring_k: usize,
) -> FxHashMap<u32, Vec<[u32; 3]>> {
    if closure.is_empty() {
        return FxHashMap::default();
    }
    let local_ids = gather_two_ring(points, grid, scratch, slot_gen_map, work, closure, ring_k);

    let mut fans: FxHashMap<u32, Vec<[u32; 3]>> = FxHashMap::default();
    if let Some(cells) = rebuild_cells(points, &local_ids, closure) {
        for cell in cells {
            fans.insert(cell.generator, cell.vertices);
        }
    }
    fans
}

/// The shared repair engine: seed the closure from the defect-pair generators
/// and any low-incidence (degree 1/2) vertex's generators (a sliver vertex can
/// be a defect with no unpaired edge), splice each closure cell's fan from the
/// consistent oracle `fans_for`, and grow on the residual until it closes (or
/// `max_rounds`). The caller's whole-diagram never-worse gate makes any
/// non-converged residual safe: an unrepaired diagram is simply not committed.
fn repair_grow_loop(
    points: &[Vec3],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    max_rounds: usize,
    debug_name: &str,
    mut fans_for: impl FnMut(&WorkingDiagram, &[u32]) -> FxHashMap<u32, Vec<[u32; 3]>>,
) -> EscalationStats {
    use std::collections::BTreeSet;
    let mut stats = EscalationStats::default();
    let debug = std::env::var("VORONOI_MESH_ESCALATE_DEBUG").is_ok();

    let defect_gens: BTreeSet<u32> = defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let mut closure: BTreeSet<u32> = defect_gens.clone();
    let t_seed = std::time::Instant::now();
    closure.extend(low_incidence_gens(work));
    if debug {
        eprintln!("  {debug_name} seed scan {:?}", t_seed.elapsed());
    }
    let target_sign = work.winding_convention(points, &defect_gens);

    let mut spliced: BTreeSet<u32> = BTreeSet::new();
    for _ in 0..max_rounds {
        if closure.is_empty() {
            break;
        }
        stats.rounds += 1;
        let closure_vec: Vec<u32> = closure.iter().copied().collect();
        let t_oracle = std::time::Instant::now();
        let fans = fans_for(work, &closure_vec);
        let oracle_elapsed = t_oracle.elapsed();
        for &g in &closure_vec {
            let Some(fan) = fans.get(&g) else {
                continue; // frontier generator — defer to a later, wider round
            };
            if fan.len() < 3 {
                continue;
            }
            work.splice_generator(points, g, fan, target_sign);
            spliced.insert(g);
        }
        // Grow on the residual: generators named by any still-unpaired edge, plus
        // low-incidence vertices left by re-fanning (a vertex an unspliced
        // neighbor still references, now orphaned). A full scan sees both.
        let t_scan = std::time::Instant::now();
        let implicated: BTreeSet<u32> = work.residual_generators().into_iter().collect();
        let new: Vec<u32> = implicated
            .iter()
            .copied()
            .filter(|g| !closure.contains(g))
            .collect();
        if debug {
            eprintln!(
                "  {debug_name} round {}: closure={} spliced={} implicated={} new={} \
                 (oracle {:?}, residual scan {:?})",
                stats.rounds,
                closure.len(),
                spliced.len(),
                implicated.len(),
                new.len(),
                oracle_elapsed,
                t_scan.elapsed(),
            );
        }
        if new.is_empty() {
            stats.stuck_components = usize::from(!implicated.is_empty());
            break;
        }
        closure.extend(new);
    }
    stats.spliced_generators = spliced.len();
    if debug {
        eprintln!(
            "{debug_name}: {:?}; final unpaired-implicated={}",
            stats,
            work.unpaired_generators().len()
        );
    }
    stats
}

/// Dependency-free local 3D repair (default, [`crate::RepairMode::Local3d`]):
/// the grow loop over the normalized-local-3D-hull oracle ([`local_hull_fans`]).
#[allow(clippy::too_many_arguments)] // grid/scratch/slot_gen_map travel together as the gather index
pub(crate) fn repair_local_hull(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    ring_k: usize,
    max_rounds: usize,
) -> EscalationStats {
    repair_grow_loop(
        points,
        work,
        defect_pairs,
        max_rounds,
        "repair_local_hull",
        |work, closure| local_hull_fans(points, grid, scratch, slot_gen_map, work, closure, ring_k),
    )
}

/// Projected local repair ([`crate::RepairMode::LocalProjected`]): the grow loop
/// over the shared-stereographic-chart exact 2D Delaunay oracle
/// ([`local_exact_fans`]). Kept as a projected-oracle diagnostic path.
#[allow(clippy::too_many_arguments)] // grid/scratch/slot_gen_map travel together as the gather index
pub(crate) fn repair_local_exact(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    slot_gen_map: &[u32],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    ring_k: usize,
    max_rounds: usize,
) -> EscalationStats {
    repair_grow_loop(
        points,
        work,
        defect_pairs,
        max_rounds,
        "repair_local_exact",
        |work, closure| {
            local_exact_fans(points, grid, scratch, slot_gen_map, work, closure, ring_k)
        },
    )
}

/// Probe-only global oracle (feature `escalate_probe`): ONE GLOBAL stereographic
/// Delaunay (fixed pole, `delaunator`) over all generators, read per closure
/// generator through the shared grow loop. A global pole is what makes the
/// rebuilt rim agree with the fast diagram (fast ≈ the global-pole Delaunay).
/// A/B reference for the local repairs; not a production path.
#[cfg(feature = "escalate_probe")]
pub fn repair_delaunator(
    points: &[Vec3],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    _gather_k: usize,
    max_rounds: usize,
) -> EscalationStats {
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
        return EscalationStats::default();
    }
    // generator -> incident sorted-global triples (the global Delaunay fan)
    let mut incident: FxHashMap<u32, Vec<[u32; 3]>> = FxHashMap::default();
    for t in tri.triangles.chunks_exact(3) {
        let mut k = [t[0] as u32, t[1] as u32, t[2] as u32];
        k.sort_unstable();
        for &li in t {
            incident.entry(li as u32).or_default().push(k);
        }
    }
    repair_grow_loop(
        points,
        work,
        defect_pairs,
        max_rounds,
        "repair_delaunator",
        |_work, closure| {
            closure
                .iter()
                .filter_map(|&g| {
                    incident
                        .get(&g)
                        .and_then(|tris| chain_fan(tris))
                        .map(|fan| (g, fan))
                })
                .collect()
        },
    )
}

// ===========================================================================
// Splicing rebuilt cells into the assembled diagram.
//
// The splice is SOURCE-AGNOSTIC: it consumes triple-keyed fans and does not
// care which oracle produced them. The only contract is "a generator's cell as
// an ordered cyclic fan of sorted global-id triples", same identity space as
// the production `VertexKey`.
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
fn triple_circumcenter(points: &[Vec3], t: [u32; 3]) -> Vec3 {
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

/// Generators of every vertex referenced by exactly 1 or 2 live cells — a real
/// sub-3-incidence defect (e.g. a sliver/near-coincident vertex) that can exist
/// with NO unpaired edge, which the unpaired-only trigger would miss.
fn low_incidence_gens(work: &WorkingDiagram) -> Vec<u32> {
    let mut cnt = vec![0u32; work.num_vertices()];
    for g in 0..work.num_cells() as u32 {
        for &v in work.boundary(g) {
            cnt[v as usize] += 1;
        }
    }
    let mut out = Vec::new();
    for (v, &c) in cnt.iter().enumerate() {
        if c == 1 || c == 2 {
            out.extend(work.vkey(v as u32));
        }
    }
    out
}

/// A triple-keyed OVERLAY view of the assembled diagram, in effective-generator
/// index space. The base arrays are borrowed read-only; splicing records a
/// per-generator boundary override, and freshly minted vertices live in side
/// arrays (their vids continue past the base vertex count). Building the view
/// is O(1) and splicing is O(defect region) — the repair's entry cost no longer
/// scales with the diagram (the old form copied every vertex, built a
/// triple→vid map over all of them, and materialized every cell as its own
/// `Vec`, ~1s at 2.5M generators before a single defect was examined).
pub struct WorkingDiagram<'a> {
    base_vertices: &'a [Vec3],
    base_keys: &'a ShardedVertexKeys,
    base_cells: &'a [VoronoiCell],
    base_cell_indices: &'a [u32],
    /// Spliced boundaries: generator → replacement vertex-id list.
    overrides: FxHashMap<u32, Vec<u32>>,
    /// Positions of minted vertices (vid = base vertex count + index).
    minted_pos: Vec<Vec3>,
    /// Triples of minted vertices, parallel to `minted_pos`.
    minted_key: Vec<[u32; 3]>,
    /// Memoized triple → vid (resolved lazily from the owner cells; see
    /// `vid_for`). Memoization also pins one deterministic answer per triple
    /// across grow rounds, so all spliced cells agree on shared vertices.
    triple_to_vid: FxHashMap<[u32; 3], u32>,
}

impl<'a> WorkingDiagram<'a> {
    /// Overlay over the assembled global arrays (post-reconcile). `keys` is the
    /// per-vertex triple store; `cells`/`cell_indices` the flat CSR boundaries.
    pub fn from_assembled(
        vertices: &'a [Vec3],
        keys: &'a ShardedVertexKeys,
        cells: &'a [VoronoiCell],
        cell_indices: &'a [u32],
    ) -> Self {
        Self {
            base_vertices: vertices,
            base_keys: keys,
            base_cells: cells,
            base_cell_indices: cell_indices,
            overrides: FxHashMap::default(),
            minted_pos: Vec::new(),
            minted_key: Vec::new(),
            triple_to_vid: FxHashMap::default(),
        }
    }

    /// Number of effective generators (splices never add generators).
    fn num_cells(&self) -> usize {
        self.base_cells.len()
    }

    /// Total vertex-id space: base vertices plus minted ones.
    fn num_vertices(&self) -> usize {
        self.base_vertices.len() + self.minted_pos.len()
    }

    /// Generator `g`'s current boundary: its override if spliced, else its live
    /// CSR window in the base arrays.
    fn boundary(&self, g: u32) -> &[u32] {
        if let Some(list) = self.overrides.get(&g) {
            return list;
        }
        let c = &self.base_cells[g as usize];
        &self.base_cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()]
    }

    /// Position of vertex `vid` (base or minted).
    fn vpos(&self, vid: u32) -> Vec3 {
        let base = self.base_vertices.len();
        if (vid as usize) < base {
            self.base_vertices[vid as usize]
        } else {
            self.minted_pos[vid as usize - base]
        }
    }

    /// Triple of vertex `vid` (base or minted). A base vid past the key store
    /// (a vertex appended by reconciliation without a key) reads as all-MAX,
    /// matching the old flattened-array initialization.
    fn vkey(&self, vid: u32) -> [u32; 3] {
        let base = self.base_vertices.len();
        if (vid as usize) < base {
            self.base_keys.get(vid).unwrap_or([u32::MAX; 3])
        } else {
            self.minted_key[vid as usize - base]
        }
    }

    /// Vertex id carrying triple `t`, minting (with the triple's circumcenter)
    /// if no referenced vertex carries it.
    ///
    /// Lookup is LOCAL: a vertex keyed `(a,b,c)` can only be referenced by the
    /// boundaries of cells `a`, `b`, `c` (per-cell emission puts the owning
    /// generator in every key, and spliced fans preserve this), so scanning
    /// those three boundaries finds any LIVE vertex with the triple — the
    /// rim-reuse property that makes spliced cells pair with unspliced
    /// neighbors. Ties (a triple at several vids, proximity-merge leftovers)
    /// resolve to the smallest vid, deterministically.
    ///
    /// Divergence from the old global-map form, accepted under the valid-or-
    /// error contract (the whole-diagram gate is unchanged): the global map
    /// also indexed UNREFERENCED vertices — e.g. a vertex orphaned by a
    /// reconciliation merge — and would resurrect such a vid instead of
    /// minting a twin. Both choices leave the surrounding cells referencing a
    /// different vid than the spliced cell, so both feed the same grow-or-
    /// reject machinery; only the vertex id (and its f32-vs-recomputed
    /// position) differs.
    fn vid_for(&mut self, points: &[Vec3], t: [u32; 3]) -> u32 {
        if let Some(&vid) = self.triple_to_vid.get(&t) {
            return vid;
        }
        let mut found: Option<u32> = None;
        for &g in &t {
            if g == u32::MAX || g as usize >= self.num_cells() {
                continue;
            }
            for &v in self.boundary(g) {
                if self.vkey(v) == t {
                    found = Some(match found {
                        Some(best) => best.min(v),
                        None => v,
                    });
                }
            }
        }
        let vid = found.unwrap_or_else(|| {
            let vid = self.num_vertices() as u32;
            self.minted_pos.push(triple_circumcenter(points, t));
            self.minted_key.push(t);
            vid
        });
        self.triple_to_vid.insert(t, vid);
        vid
    }

    /// Replace generator `g`'s boundary with the rebuilt fan `fan`, oriented to
    /// match the diagram's global winding convention (`target_sign`). An oracle
    /// fan can come out either way; a rim edge only pairs with its unspliced
    /// neighbor when both wind the same direction, so the fan is reversed if its
    /// signed orientation disagrees.
    fn splice_generator(&mut self, points: &[Vec3], g: u32, fan: &[[u32; 3]], target_sign: f32) {
        let mut list: Vec<u32> = fan.iter().map(|&t| self.vid_for(points, t)).collect();
        if target_sign != 0.0 && self.polygon_sign(points, g, &list) * target_sign < 0.0 {
            list.reverse();
        }
        self.overrides.insert(g, list);
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
            acc += self.vpos(list[i]).cross(self.vpos(list[(i + 1) % n]));
        }
        acc.dot(points[g as usize])
    }

    /// Majority signed-orientation of the existing (unspliced) cells — the
    /// global winding convention the spliced cells must match. Sampled over the
    /// first cells with a real boundary, skipping the defect generators.
    fn winding_convention(&self, points: &[Vec3], skip: &std::collections::BTreeSet<u32>) -> f32 {
        let mut pos = 0i32;
        let mut neg = 0i32;
        for g in 0..self.num_cells() as u32 {
            let list = self.boundary(g);
            if skip.contains(&g) || list.len() < 3 {
                continue;
            }
            let s = self.polygon_sign(points, g, list);
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
        self.residual_scan(false)
    }

    /// Union of both whole-diagram residual signals the grow loop converges
    /// against — `unpaired_generators` plus the generators of every degree-1/2
    /// vertex (`low_incidence_gens`' criterion) — from ONE boundary walk.
    /// The grow loop runs this every round; as two independent scans it paid
    /// the boundary walk (and its per-cell override lookup) twice.
    fn residual_generators(&self) -> Vec<u32> {
        self.residual_scan(true)
    }

    fn residual_scan(&self, include_low_incidence: bool) -> Vec<u32> {
        // One record per directed half-edge: (canonical undirected key, is
        // lower-id direction). Sort + run-scan instead of a hashmap build —
        // the map (unreserved, ~2E entries, rebuilt every grow round) was the
        // dominant repair cost at scale (~1.3s/round at 1M cells; the sorted
        // scan is ~10x cheaper and parallelizes).
        let mut uses: Vec<(u64, bool)> = Vec::with_capacity(self.base_cell_indices.len() + 64);
        // Per-vertex live-cell reference counts, matching `low_incidence_gens`
        // (counts every boundary, including sub-3 ones the edge scan skips).
        let mut cnt: Vec<u32> = if include_low_incidence {
            vec![0u32; self.num_vertices()]
        } else {
            Vec::new()
        };
        for g in 0..self.num_cells() as u32 {
            let list = self.boundary(g);
            if include_low_incidence {
                for &v in list {
                    cnt[v as usize] += 1;
                }
            }
            let n = list.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (list[i], list[(i + 1) % n]);
                let (lo, hi, fwd) = if a <= b { (a, b, true) } else { (b, a, false) };
                uses.push((((lo as u64) << 32) | hi as u64, fwd));
            }
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::slice::ParallelSliceMut;
            uses.par_sort_unstable();
        }
        #[cfg(not(feature = "parallel"))]
        uses.sort_unstable();

        let mut grow: Vec<u32> = Vec::new();
        let mut i = 0usize;
        while i < uses.len() {
            let key = uses[i].0;
            let mut fwd_count = 0usize;
            let mut j = i;
            while j < uses.len() && uses[j].0 == key {
                fwd_count += usize::from(uses[j].1);
                j += 1;
            }
            let group_len = j - i;
            let (a, b) = ((key >> 32) as u32, key as u32);
            // Paired = exactly two uses in opposite directions. A self-loop
            // key (a == b) has both "directions" in one record, so a single
            // use reads as paired — matching the directed-count map this
            // replaces (the gate rejects self-loops regardless).
            let paired = if a == b {
                group_len == 1
            } else {
                group_len == 2 && fwd_count == 1
            };
            if !paired {
                let (ka, kb) = (self.vkey(a), self.vkey(b));
                grow.extend(ka.iter().chain(kb.iter()));
            }
            i = j;
        }
        for (v, &c) in cnt.iter().enumerate() {
            if c == 1 || c == 2 {
                grow.extend(self.vkey(v as u32));
            }
        }
        grow.sort_unstable();
        grow.dedup();
        grow
    }

    /// Materialize the overlay into flat cell arrays. Returns
    /// `(minted_vertex_positions, cells, cell_indices)`: minted vids were
    /// assigned past the base vertex count, so the caller appends the minted
    /// positions to its base vertex array and swaps in the cell arrays on
    /// acceptance (truncating the appended positions again on rejection).
    pub fn into_flat(self) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
        let n = self.base_cells.len();
        let mut cells = Vec::with_capacity(n);
        let mut cell_indices = Vec::with_capacity(self.base_cell_indices.len());
        for g in 0..n as u32 {
            let list = if let Some(list) = self.overrides.get(&g) {
                list.as_slice()
            } else {
                let c = &self.base_cells[g as usize];
                &self.base_cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()]
            };
            let start = cell_indices.len() as u32;
            cell_indices.extend_from_slice(list);
            cells.push(VoronoiCell::new(start, list.len() as u16));
        }
        (self.minted_pos, cells, cell_indices)
    }
}

/// Outcome of a repair pass (diagnostics for tests / debug output).
#[derive(Debug, Default, Clone, Copy)]
pub struct EscalationStats {
    /// Total grow rounds.
    pub rounds: usize,
    /// Distinct generators whose cells were rebuilt and spliced.
    pub spliced_generators: usize,
    /// 1 if the grow loop stopped with a residual (no new implicated generators
    /// but the implicated set is non-empty), else 0.
    pub stuck_components: usize,
}

/// A0 probe stash payload: (effective points, fast per-cell triples).
#[cfg(feature = "escalate_probe")]
pub type A0Stash = (Vec<Vec3>, Vec<Vec<[u32; 3]>>);

#[cfg(feature = "escalate_probe")]
thread_local! {
    /// A0 probe stash from the last build, for an exact-reference comparison
    /// test. See `take_a0_fast`.
    static A0_STASH: std::cell::RefCell<Option<A0Stash>> = const { std::cell::RefCell::new(None) };
}

/// Stash the assembled fast per-cell triple fans for A0 exact-reference probes.
#[cfg(feature = "escalate_probe")]
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
/// build that ran with `VORONOI_MESH_ESCALATE_PROBE_A0` set. Probe API.
#[cfg(feature = "escalate_probe")]
pub fn take_a0_fast() -> Option<A0Stash> {
    A0_STASH.with(|s| s.borrow_mut().take())
}

/// Process-global enable for the repair pass (probe / opt-in): forces the
/// repair trigger on even when the configured [`crate::RepairMode`] is
/// `Disabled`. Off by default; only the probe API sets it.
static ESCALATE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable defect-driven escalation (probe API).
#[cfg(feature = "escalate_probe")]
pub fn set_escalation_enabled(on: bool) {
    ESCALATE_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether escalation is currently force-enabled.
pub(crate) fn escalation_enabled() -> bool {
    ESCALATE_ENABLED.load(Ordering::Relaxed)
}
