//! Defect-driven local rebuild; its output contract is documented in
//! `docs/correctness.md`.
//!
//! When post-assembly detection finds residual topology defects (unpaired
//! edges / degree-1-or-2 vertices), the affected neighborhood is rebuilt from
//! ONE consistent exact oracle and spliced back into the assembled diagram.
//! Because every rebuilt cell comes from the same oracle, rebuilt cells pair
//! with each other on shared edges by construction; on the well-conditioned rim
//! the oracle agrees with the fast clipper, so rebuilt rim vertices reuse the
//! surrounding cells' vertex ids and pair with the unspliced neighbors too.
//!
//! The rebuild oracle is a normalized local 3D hull ([`LocalHull`], exact
//! `orient3d`) shared by every cell in the affected neighborhood.
//!
//! The caller (`maybe_rebuild_effective`) commits the result only if the whole
//! rebuilt diagram passes strict validation — the never-worse gate.

use glam::{DVec3, Vec3};
use rustc_hash::FxHashMap;

use super::local_hull::LocalHull;
use crate::cell_layout::LiveCellLayout;
use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch};
use crate::diagram::VoronoiCell;
use crate::live_dedup::ShardedVertexKeys;

#[derive(Clone, Copy)]
struct RebuildVertex {
    key: [u32; 3],
    /// Oracle-selected position for a newly minted vertex.
    mint_pos: Vec3,
}

type RebuildFan = Vec<RebuildVertex>;

/// O(local) kNN gather used by the production rebuild oracles. For each seed,
/// walk the cube-map shell frontier nearest-first (the same machinery the point
/// locator uses) and collect the `k + 1` nearest generators, unioned with the
/// seeds — proportional to the local neighborhood rather than to `n`.
///
/// `grid.point_indices()[slot]` maps an emitted slot back to its generator id.
/// The ring certificate (`unseen_bound`) bounds every unseen point's dot, so
/// collection stops once the `k + 1` nearest seen are provably nearer than
/// anything unseen.
fn gather_knn_grid(
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    points: &[Vec3],
    seeds: &[u32],
    k: usize,
) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<u32> = seeds.iter().copied().collect();
    let mut batch: Vec<u32> = Vec::new();
    let mut collected: Vec<(f32, u32)> = Vec::new();
    for &s in seeds {
        let query = points[s as usize];
        let mut frontier = grid.unrestricted_shell_frontier(query, s as usize, scratch);
        collected.clear();
        while let Some(layer) = frontier.frontier(&mut batch) {
            for &slot in &batch {
                let id = grid.point_indices()[slot as usize];
                let candidate = points[id as usize];
                // The shell frontier certifies the crate's canonical raw-f32
                // dot operation. Scoring with glam's independently associated
                // `Vec3::dot` could round across `layer.unseen_bound` and stop
                // the rebuild gather before its actual k-th candidate was seen.
                let dot = crate::fp::dot3_f32(
                    query.x,
                    query.y,
                    query.z,
                    candidate.x,
                    candidate.y,
                    candidate.z,
                );
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

/// Build Hull3d fans while retaining each face's oriented support normal
/// alongside the sorted triple identity.
fn rebuild_hull_cells(
    points: &[Vec3],
    local_ids: &[u32],
    seeds: &[u32],
) -> Option<Vec<(u32, RebuildFan)>> {
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
                let n = hull.face_circumcenter(fi);
                RebuildVertex {
                    key: t,
                    mint_pos: Vec3::new(n.x as f32, n.y as f32, n.z as f32),
                }
            })
            .collect();
        out.push((g, vertices));
    }
    Some(out)
}

/// Union gather (2-RING) for a rebuild oracle: the closure, each closure cell's
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
            .flat_map(|&v| work.vkey(VertexId::new(v)))
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
    gather_knn_grid(grid, scratch, points, &seeds2, ring_k)
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
    work: &WorkingDiagram,
    closure: &[u32],
    ring_k: usize,
) -> FxHashMap<u32, RebuildFan> {
    if closure.is_empty() {
        return FxHashMap::default();
    }
    let local_ids = gather_two_ring(points, grid, scratch, work, closure, ring_k);

    let mut fans: FxHashMap<u32, RebuildFan> = FxHashMap::default();
    if let Some(cells) = rebuild_hull_cells(points, &local_ids, closure) {
        for (generator, fan) in cells {
            fans.insert(generator, fan);
        }
    }
    fans
}

/// The shared rebuild engine: seed the closure from the defect-pair generators
/// and any low-incidence (degree 1/2) vertex's generators (a sliver vertex can
/// be a defect with no unpaired edge), splice each closure cell's fan from the
/// consistent oracle `fans_for`, and grow on the residual until it closes (or
/// `max_rounds`). The caller's whole-diagram never-worse gate makes any
/// non-converged residual safe: an unrebuilt diagram is simply not committed.
#[derive(Clone, Copy)]
struct GrowthDiagnostics {
    enabled: bool,
    name: &'static str,
}

fn run_rebuild_growth(
    points: &[Vec3],
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    merge_affected: &[u32],
    max_rounds: usize,
    diagnostics: GrowthDiagnostics,
    mut fans_for: impl FnMut(&WorkingDiagram, &[u32]) -> FxHashMap<u32, RebuildFan>,
) -> LocalRebuildStats {
    use std::collections::BTreeSet;
    let mut stats = LocalRebuildStats::default();
    let GrowthDiagnostics {
        enabled: debug,
        name: debug_name,
    } = diagnostics;

    let defect_gens: BTreeSet<u32> = defect_pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let mut closure: BTreeSet<u32> = defect_gens.clone();
    closure.extend(low_incidence_gens(work));
    let target_sign = work.winding_convention(points, &defect_gens);

    // Every vertex id whose set of referencing boundaries may have changed:
    // splicing is the loop's only mutation, so that is exactly the vids on a
    // spliced cell's boundary, captured immediately before and after each
    // splice. Feeds the localized residual scan's dirty region.
    let mut touched_vids: BTreeSet<u32> = BTreeSet::new();
    let mut spliced: BTreeSet<u32> = BTreeSet::new();
    for _ in 0..max_rounds {
        if closure.is_empty() {
            break;
        }
        stats.rounds += 1;
        let closure_vec: Vec<u32> = closure.iter().copied().collect();
        let fans = fans_for(work, &closure_vec);
        for &g in &closure_vec {
            let Some(fan) = fans.get(&g) else {
                continue; // frontier generator — defer to a later, wider round
            };
            if fan.len() < 3 {
                continue;
            }
            touched_vids.extend(work.boundary(g).iter().copied());
            work.splice_generator(points, CellId::new(g), fan, target_sign);
            touched_vids.extend(work.boundary(g).iter().copied());
            spliced.insert(g);
        }
        // Geometry is not a reliable winding oracle in the dense cases that
        // reach rebuild: rounded/reconciled f32 circumcenters can make a tiny
        // cell look self-crossing. Enforce the combinatorial invariant instead.
        // Shared edges constrain two rebuilt cells to opposite directions;
        // rim edges anchor each connected rebuilt component to an unspliced
        // neighbor. This is a parity solve over the already-spliced overlay.
        work.reconcile_override_winding(points, &spliced, target_sign);
        // Grow on the residual: generators named by any still-unpaired edge, plus
        // low-incidence vertices left by re-fanning (a vertex an unspliced
        // neighbor still references, now orphaned). The localized scan sees both
        // without walking the whole diagram (debug builds oracle-check it).
        let implicated = work.residual_generators_local(&closure, &touched_vids, merge_affected);
        let new: Vec<u32> = implicated
            .iter()
            .copied()
            .filter(|g| !closure.contains(g))
            .collect();
        #[cfg(debug_assertions)]
        {
            // The locality argument (key-ownership invariant + merge-affected
            // carve-out) says the localized scan finds every residual the
            // whole-diagram scan would, up to generators already in the
            // closure. Verify it exhaustively in debug builds — a divergence
            // here means a defect class escaped the dirty region.
            let global_new: Vec<u32> = work
                .residual_generators()
                .into_iter()
                .filter(|g| !closure.contains(g))
                .collect();
            assert_eq!(
                new, global_new,
                "{debug_name}: localized residual scan diverges from the global scan \
                 (key-ownership locality argument violated)"
            );
        }
        if debug {
            eprintln!(
                "  {debug_name} round {}: closure={} spliced={} implicated={} new={}",
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

/// Dependency-free local 3D rebuild (default, [`crate::LocalRebuildMode::Hull3d`]):
/// the grow loop over the normalized-local-3D-hull oracle ([`local_hull_fans`]).
#[allow(clippy::too_many_arguments)] // grid and scratch form the reusable gather index
pub(crate) fn rebuild_with_local_hull(
    points: &[Vec3],
    grid: &CubeMapGrid,
    scratch: &mut CubeMapGridScratch,
    work: &mut WorkingDiagram,
    defect_pairs: &[(u32, u32)],
    merge_affected: &[u32],
    ring_k: usize,
    max_rounds: usize,
    debug: bool,
) -> LocalRebuildStats {
    run_rebuild_growth(
        points,
        work,
        defect_pairs,
        merge_affected,
        max_rounds,
        GrowthDiagnostics {
            enabled: debug,
            name: "rebuild_with_local_hull",
        },
        |work, closure| local_hull_fans(points, grid, scratch, work, closure, ring_k),
    )
}

// ===========================================================================
// Splicing rebuilt cells into the assembled diagram.
//
// The splice consumes Hull3d fans keyed by sorted generator triples, the same
// identity space as the production `VertexKey`.
//
// The load-bearing trick: a rebuilt vertex looks up the EXISTING fast-path
// vertex id for its triple and reuses it. On the well-conditioned rim
// (fast == exact) the triple is already present, so the rebuilt cell shares the
// surrounding cells' vertices and pairs with them automatically. Only the
// corrected near-cocircular defect vertices are minted fresh.
// ===========================================================================

/// Signed spherical polygon orientation, accumulated in f64. Rebuild is entered
/// precisely for numerically difficult neighborhoods, and the O(radius²)
/// polygon-area signal can be smaller than the rounding left by summing its
/// O(radius) edge cross-products in f32.
fn polygon_sign_f64(generator: Vec3, n: usize, mut vertex: impl FnMut(usize) -> Vec3) -> f64 {
    if n < 3 {
        return 0.0;
    }
    let mut acc = DVec3::ZERO;
    for i in 0..n {
        let a = vertex(i).as_dvec3();
        let b = vertex((i + 1) % n).as_dvec3();
        acc += a.cross(b);
    }
    acc.dot(generator.as_dvec3())
}

/// Solve cell-reversal parity constraints. An adjacency bit is the required
/// XOR between its endpoint cells; an anchor bit is the required reversal of a
/// cell next to an unmodified neighbor. `None` means the constraints conflict,
/// in which case the caller leaves the winding correction unapplied for the
/// strict gate to reject rather than guessing.
fn solve_winding_parity(
    nodes: &std::collections::BTreeSet<u32>,
    adj: &FxHashMap<u32, Vec<(u32, bool)>>,
    anchors: &FxHashMap<u32, Vec<bool>>,
    mut unanchored_root_flip: impl FnMut(u32) -> bool,
) -> Option<Vec<u32>> {
    use std::collections::VecDeque;

    let mut parity: FxHashMap<u32, bool> = FxHashMap::default();
    let mut flips = Vec::new();
    for &root in nodes {
        if parity.contains_key(&root) {
            continue;
        }
        parity.insert(root, false);
        let mut component = Vec::new();
        let mut queue = VecDeque::from([root]);
        while let Some(g) = queue.pop_front() {
            component.push(g);
            let gp = parity[&g];
            for &(m, xor) in adj.get(&g).map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(&mp) = parity.get(&m) {
                    if mp != gp ^ xor {
                        return None;
                    }
                } else {
                    parity.insert(m, gp ^ xor);
                    queue.push_back(m);
                }
            }
        }

        let mut root_flip = None;
        for &g in &component {
            for &required in anchors.get(&g).map(Vec::as_slice).unwrap_or(&[]) {
                let candidate = required ^ parity[&g];
                if root_flip.is_some_and(|existing| existing != candidate) {
                    return None;
                }
                root_flip = Some(candidate);
            }
        }
        let root_flip = root_flip.unwrap_or_else(|| unanchored_root_flip(root));
        for g in component {
            if root_flip ^ parity[&g] {
                flips.push(g);
            }
        }
    }
    Some(flips)
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
            out.extend(work.vkey(VertexId::new(v as u32)));
        }
    }
    out
}

/// Effective-cell identity at overlay mutation seams.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellId(u32);

impl CellId {
    #[inline]
    const fn new(raw: u32) -> Self {
        Self(raw)
    }

    #[inline]
    const fn get(self) -> u32 {
        self.0
    }
}

/// Overlay vertex identity at position/key lookup seams.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VertexId(u32);

impl VertexId {
    #[inline]
    const fn new(raw: u32) -> Self {
        Self(raw)
    }

    #[inline]
    const fn get(self) -> u32 {
        self.0
    }
}

/// A triple-keyed OVERLAY view of the assembled diagram, in effective-generator
/// index space. The base arrays are borrowed read-only; splicing records a
/// per-generator boundary override, and freshly minted vertices live in side
/// arrays (their vids continue past the base vertex count). Building the view
/// is O(1) and splicing is O(defect region); entry does not copy every vertex,
/// build a diagram-wide triple map, or materialize every cell. See
/// `docs/performance.md#source-pinned-performance-decisions`.
pub(crate) struct WorkingDiagram<'a> {
    base_vertices: &'a [Vec3],
    base_keys: &'a ShardedVertexKeys,
    base_layout: LiveCellLayout<'a, 'a>,
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
    /// Overlay over the reconciled global arrays. `keys` is the per-vertex
    /// triple store; `base_layout` owns the flat CSR boundary pairing.
    pub(crate) fn from_reconciled(
        vertices: &'a [Vec3],
        keys: &'a ShardedVertexKeys,
        base_layout: LiveCellLayout<'a, 'a>,
    ) -> Self {
        #[cfg(debug_assertions)]
        base_layout.debug_assert_valid();
        Self {
            base_vertices: vertices,
            base_keys: keys,
            base_layout,
            overrides: FxHashMap::default(),
            minted_pos: Vec::new(),
            minted_key: Vec::new(),
            triple_to_vid: FxHashMap::default(),
        }
    }

    /// Number of effective generators (splices never add generators).
    fn num_cells(&self) -> usize {
        self.base_layout.cell_count()
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
        self.base_layout.span(g as usize)
    }

    /// Position of vertex `vid` (base or minted).
    fn vpos(&self, vertex: VertexId) -> Vec3 {
        let vid = vertex.get();
        let base = self.base_vertices.len();
        if (vid as usize) < base {
            self.base_vertices[vid as usize]
        } else {
            self.minted_pos[vid as usize - base]
        }
    }

    /// Triple of vertex `vid` (base or minted). A base vid past the key store
    /// (a vertex appended by reconciliation without a key) reads as all-MAX,
    /// matching the flattened-array sentinel contract.
    fn vkey(&self, vertex: VertexId) -> [u32; 3] {
        let vid = vertex.get();
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
    /// Difference from the eager global-map oracle, accepted under the
    /// valid-or-error contract (the whole-diagram gate is unchanged): that map
    /// also indexed UNREFERENCED vertices — e.g. a vertex orphaned by a
    /// reconciliation merge — and would resurrect such a vid instead of
    /// minting a twin. Both choices leave the surrounding cells referencing a
    /// different vid than the spliced cell, so both feed the same grow-or-
    /// reject machinery; only the vertex id (and its f32-vs-recomputed
    /// position) differs.
    fn vid_for(&mut self, vertex: RebuildVertex) -> VertexId {
        let t = vertex.key;
        if let Some(&vid) = self.triple_to_vid.get(&t) {
            return VertexId::new(vid);
        }
        let mut found: Option<u32> = None;
        for &g in &t {
            if g == u32::MAX || g as usize >= self.num_cells() {
                continue;
            }
            for &v in self.boundary(g) {
                if self.vkey(VertexId::new(v)) == t {
                    found = Some(match found {
                        Some(best) => best.min(v),
                        None => v,
                    });
                }
            }
        }
        let vid = found.unwrap_or_else(|| {
            let vid = self.num_vertices() as u32;
            let position = vertex.mint_pos;
            self.minted_pos.push(position);
            self.minted_key.push(t);
            vid
        });
        self.triple_to_vid.insert(t, vid);
        VertexId::new(vid)
    }

    /// Replace generator `g`'s boundary with the rebuilt fan `fan`, oriented to
    /// match the diagram's global winding convention (`target_sign`). An oracle
    /// fan can come out either way; a rim edge only pairs with its unspliced
    /// neighbor when both wind the same direction, so the fan is reversed if its
    /// signed orientation disagrees.
    fn splice_generator(
        &mut self,
        points: &[Vec3],
        cell: CellId,
        fan: &[RebuildVertex],
        target_sign: f64,
    ) {
        let g = cell.get();
        let mut list: Vec<u32> = fan
            .iter()
            .map(|&vertex| self.vid_for(vertex).get())
            .collect();
        if target_sign != 0.0 && self.polygon_sign(points, g, &list) * target_sign < 0.0 {
            list.reverse();
        }
        self.overrides.insert(g, list);
    }

    /// Make every shared edge in the rebuilt overlay run in opposite
    /// directions, without consulting its numerically fragile vertex geometry.
    /// Each cell reversal is a boolean; a same-direction edge requires exactly
    /// one endpoint cell to reverse, while an already-opposite edge requires
    /// equal reversal parity. Edges to unspliced cells anchor the component.
    fn reconcile_override_winding(
        &mut self,
        points: &[Vec3],
        active: &std::collections::BTreeSet<u32>,
        target_sign: f64,
    ) {
        use std::collections::BTreeSet;

        let nodes: BTreeSet<u32> = active
            .iter()
            .copied()
            .filter(|g| self.overrides.contains_key(g))
            .collect();
        let mut adj: FxHashMap<u32, Vec<(u32, bool)>> = FxHashMap::default();
        let mut anchors: FxHashMap<u32, Vec<bool>> = FxHashMap::default();

        for &g in &nodes {
            let list = self.boundary(g);
            for i in 0..list.len() {
                let (from, to) = (list[i], list[(i + 1) % list.len()]);
                let (lo, hi, g_fwd) = if from <= to {
                    (from, to, true)
                } else {
                    (to, from, false)
                };
                let (ka, kb) = (self.vkey(VertexId::new(lo)), self.vkey(VertexId::new(hi)));
                let mut common = ka
                    .iter()
                    .copied()
                    .filter(|&x| x != g && x != u32::MAX && kb.contains(&x));
                let Some(m) = common.next() else {
                    continue;
                };
                if common.next().is_some() || m as usize >= self.num_cells() {
                    continue;
                }
                let other = self.boundary(m);
                let mut m_fwd = None;
                for j in 0..other.len() {
                    let (a, b) = (other[j], other[(j + 1) % other.len()]);
                    if a == lo && b == hi {
                        m_fwd = Some(true);
                        break;
                    }
                    if a == hi && b == lo {
                        m_fwd = Some(false);
                        break;
                    }
                }
                let Some(m_fwd) = m_fwd else {
                    continue;
                };
                let one_must_flip = g_fwd == m_fwd;
                if nodes.contains(&m) {
                    adj.entry(g).or_default().push((m, one_must_flip));
                    adj.entry(m).or_default().push((g, one_must_flip));
                } else {
                    anchors.entry(g).or_default().push(one_must_flip);
                }
            }
        }

        if let Some(flips) = solve_winding_parity(&nodes, &adj, &anchors, |root| {
            target_sign != 0.0
                && self.polygon_sign(points, root, self.boundary(root)) * target_sign < 0.0
        }) {
            for g in flips {
                self.overrides.get_mut(&g).unwrap().reverse();
            }
        }
    }

    /// Signed orientation of generator `g`'s boundary `list`: the sphere-surface
    /// area normal dotted with the generator direction. Positive and negative
    /// distinguish CCW vs CW as seen from outside the sphere.
    fn polygon_sign(&self, points: &[Vec3], g: u32, list: &[u32]) -> f64 {
        polygon_sign_f64(points[g as usize], list.len(), |i| {
            self.vpos(VertexId::new(list[i]))
        })
    }

    /// Majority signed-orientation of the existing (unspliced) cells — the
    /// global winding convention the spliced cells must match. Sampled over the
    /// first cells with a real boundary, skipping the defect generators.
    fn winding_convention(&self, points: &[Vec3], skip: &std::collections::BTreeSet<u32>) -> f64 {
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
    /// named by the two endpoints' triples are returned. Debug-print diagnostic.
    fn unpaired_generators(&self) -> Vec<u32> {
        self.residual_scan(false)
    }

    /// Whole-diagram union of both residual signals the grow loop converges
    /// against — `unpaired_generators` plus the generators of every degree-1/2
    /// vertex (`low_incidence_gens`' criterion) — from ONE boundary walk.
    /// Reference form: the grow loop runs `residual_generators_local` and
    /// debug builds assert it matches this scan.
    #[cfg(debug_assertions)]
    fn residual_generators(&self) -> Vec<u32> {
        self.residual_scan(true)
    }

    /// The distinct generators that can legally reference vertex `vid`: the
    /// non-sentinel entries of its key triple. Per-cell emission puts the
    /// owning generator in every key and spliced fans preserve this, so a
    /// vertex keyed `(a, b, c)` appears only in the boundaries of cells `a`,
    /// `b`, `c` (the same invariant `vid_for`'s local lookup rests on). The
    /// one production exception — reconcile merges remapping a reference into
    /// a foreign cell — is carved out by the caller via `merge_affected`.
    fn owners(&self, vertex: VertexId) -> impl Iterator<Item = u32> + '_ {
        let k = self.vkey(vertex);
        let n = self.num_cells();
        (0..3).filter_map(move |i| {
            let g = k[i];
            ((g as usize) < n && !k[..i].contains(&g)).then_some(g)
        })
    }

    /// Localized form of `residual_generators`: the same defect verdicts,
    /// computed over the defect neighborhood instead of the whole diagram.
    ///
    /// Region construction (all sets O(defect region)):
    /// - DIRTY — cells whose edge pairing or vertex incidence may have changed
    ///   since the last scan: the closure (splices happen only there), the
    ///   key-owners of every `touched_vids` entry (an unspliced neighbor's
    ///   pairing changes only via a shared vertex, and it can share only
    ///   vertices whose triples name it), and the reconcile merge-affected
    ///   cells (whose references evade the key-ownership rule).
    /// - EMIT — DIRTY plus the key-owners of every vertex on a DIRTY
    ///   boundary. Any cell using an edge `{va, vb}` of a DIRTY cell
    ///   references `va`, so it is a key-owner of `va` (or merge-affected):
    ///   EMIT contains the complete use set of every evaluated edge, and the
    ///   complete reference set of every DIRTY-boundary vertex.
    ///
    /// Edge groups with no DIRTY use are skipped: their users are unspliced,
    /// so their status is unchanged, and any entry defect's owning cells are
    /// already in the closure (seeded from the detection residuals). Verdicts
    /// on evaluated groups are exact, so the grow set equals the global
    /// scan's, up to generators already in the closure — asserted per round
    /// against `residual_generators` in debug builds.
    fn residual_generators_local(
        &self,
        closure: &std::collections::BTreeSet<u32>,
        touched_vids: &std::collections::BTreeSet<u32>,
        merge_affected: &[u32],
    ) -> Vec<u32> {
        use std::collections::BTreeSet;
        let n_cells = self.num_cells();
        let mut dirty: BTreeSet<u32> = closure
            .iter()
            .chain(merge_affected)
            .copied()
            .filter(|&g| (g as usize) < n_cells)
            .collect();
        for &v in touched_vids {
            dirty.extend(self.owners(VertexId::new(v)));
        }
        let mut dirty_vids: BTreeSet<u32> = BTreeSet::new();
        for &g in &dirty {
            dirty_vids.extend(self.boundary(g).iter().copied());
        }
        let mut emit: BTreeSet<u32> = dirty.clone();
        for &v in &dirty_vids {
            emit.extend(self.owners(VertexId::new(v)));
        }

        // Same record scheme as `residual_scan`, plus a from-DIRTY bit; the
        // incidence counts cover exactly the DIRTY-boundary vertices (their
        // reference sets are complete within EMIT).
        let mut uses: Vec<(u64, u8)> = Vec::new();
        let mut cnt: FxHashMap<u32, u32> = dirty_vids.iter().map(|&v| (v, 0u32)).collect();
        for &g in &emit {
            let from_dirty = dirty.contains(&g);
            let list = self.boundary(g);
            for &v in list {
                if let Some(c) = cnt.get_mut(&v) {
                    *c += 1;
                }
            }
            let n = list.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (list[i], list[(i + 1) % n]);
                let (lo, hi, fwd) = if a <= b { (a, b, 1u8) } else { (b, a, 0u8) };
                uses.push((
                    ((lo as u64) << 32) | hi as u64,
                    fwd | (u8::from(from_dirty) << 1),
                ));
            }
        }
        uses.sort_unstable();

        let mut grow: Vec<u32> = Vec::new();
        let mut i = 0usize;
        while i < uses.len() {
            let key = uses[i].0;
            let mut fwd_count = 0usize;
            let mut any_dirty = false;
            let mut j = i;
            while j < uses.len() && uses[j].0 == key {
                fwd_count += usize::from(uses[j].1 & 1);
                any_dirty |= uses[j].1 & 2 != 0;
                j += 1;
            }
            let group_len = j - i;
            if any_dirty {
                let (a, b) = ((key >> 32) as u32, key as u32);
                // Pairing verdict identical to `residual_scan` (self-loop
                // single-use reads as paired; the gate rejects it regardless).
                let paired = if a == b {
                    group_len == 1
                } else {
                    group_len == 2 && fwd_count == 1
                };
                if !paired {
                    let (ka, kb) = (self.vkey(VertexId::new(a)), self.vkey(VertexId::new(b)));
                    grow.extend(ka.iter().chain(kb.iter()));
                }
            }
            i = j;
        }
        for (&v, &c) in &cnt {
            if c == 1 || c == 2 {
                grow.extend(self.vkey(VertexId::new(v)));
            }
        }
        grow.sort_unstable();
        grow.dedup();
        grow
    }

    fn residual_scan(&self, include_low_incidence: bool) -> Vec<u32> {
        // One record per directed half-edge: (canonical undirected key, is
        // lower-id direction). Sort + run-scan avoids rebuilding a ~2E-entry
        // hash map every grow round and permits parallel sorting. See
        // docs/performance.md#source-pinned-performance-decisions.
        let mut uses: Vec<(u64, bool)> = Vec::with_capacity(self.base_layout.index_count() + 64);
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
                let (ka, kb) = (self.vkey(VertexId::new(a)), self.vkey(VertexId::new(b)));
                grow.extend(ka.iter().chain(kb.iter()));
            }
            i = j;
        }
        for (v, &c) in cnt.iter().enumerate() {
            if c == 1 || c == 2 {
                grow.extend(self.vkey(VertexId::new(v as u32)));
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
    pub(crate) fn overridden_cells(&self) -> Vec<u32> {
        let mut cells: Vec<u32> = self.overrides.keys().copied().collect();
        cells.sort_unstable();
        cells
    }

    pub(crate) fn into_flat(self) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
        let n = self.base_layout.cell_count();
        let mut cells = Vec::with_capacity(n);
        let mut cell_indices = Vec::with_capacity(self.base_layout.index_count());
        for g in 0..n as u32 {
            let list = if let Some(list) = self.overrides.get(&g) {
                list.as_slice()
            } else {
                self.base_layout.span(g as usize)
            };
            let start = cell_indices.len() as u32;
            cell_indices.extend_from_slice(list);
            cells.push(VoronoiCell::new(start, list.len() as u16));
        }
        (self.minted_pos, cells, cell_indices)
    }
}

/// Outcome of a rebuild pass (diagnostics for tests / debug output).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LocalRebuildStats {
    /// Total grow rounds.
    pub rounds: usize,
    /// Distinct generators whose cells were rebuilt and spliced.
    pub spliced_generators: usize,
    /// 1 if the grow loop stopped with a residual (no new implicated generators
    /// but the implicated set is non-empty), else 0.
    pub stuck_components: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize()
    }

    #[test]
    fn local3d_mint_preserves_hull_selected_circumcenter_sign() {
        let points = [
            u(0.70, 0.00, 0.714),
            u(-0.35, 0.61, 0.714),
            u(-0.35, -0.61, 0.714),
            u(0.18, 0.12, 0.976),
            u(-0.12, 0.16, 0.980),
        ];
        let ids = [0, 1, 2, 3, 4];
        let cells = rebuild_hull_cells(&points, &ids, &ids).unwrap();
        let vertex = cells
            .iter()
            .flat_map(|(_, fan)| fan)
            .copied()
            .find(|v| v.mint_pos.dot(points[v.key[0] as usize]) < 0.0)
            .expect("origin-outside hull must have an opposing support normal");
        let expected = vertex.mint_pos;

        let keys = ShardedVertexKeys::new(vec![0], vec![]);
        let cells = vec![VoronoiCell::new(0, 0); points.len()];
        let mut work =
            WorkingDiagram::from_reconciled(&[], &keys, LiveCellLayout::new(&cells, &[]));
        let vid = work.vid_for(vertex);
        assert_eq!(work.vkey(vid), vertex.key);
        assert_eq!(work.vpos(vid), expected);
    }

    #[test]
    fn overlay_ids_are_layout_transparent() {
        assert_eq!(CellId::new(7).get(), 7);
        assert_eq!(std::mem::size_of::<CellId>(), std::mem::size_of::<u32>());
        assert_eq!(std::mem::align_of::<CellId>(), std::mem::align_of::<u32>());
        assert_eq!(VertexId::new(11).get(), 11);
        assert_eq!(std::mem::size_of::<VertexId>(), std::mem::size_of::<u32>());
        assert_eq!(
            std::mem::align_of::<VertexId>(),
            std::mem::align_of::<u32>()
        );
    }

    #[test]
    fn local3d_overlay_reports_every_spliced_cell_for_resolution_scan() {
        let points = [Vec3::X, Vec3::Y, Vec3::Z];
        let keys = ShardedVertexKeys::new(vec![0], vec![]);
        let cells = vec![VoronoiCell::new(0, 0); points.len()];
        let mut work =
            WorkingDiagram::from_reconciled(&[], &keys, LiveCellLayout::new(&cells, &[]));
        let fan = [
            RebuildVertex {
                key: [0, 1, 2],
                mint_pos: Vec3::X,
            },
            RebuildVertex {
                key: [0, 1, 2],
                mint_pos: Vec3::Y,
            },
            RebuildVertex {
                key: [0, 1, 2],
                mint_pos: Vec3::Z,
            },
        ];
        work.splice_generator(&points, CellId::new(2), &fan, 0.0);
        work.splice_generator(&points, CellId::new(0), &fan, 0.0);
        assert_eq!(work.overridden_cells(), [0, 2]);
    }

    #[test]
    fn overlay_uses_live_base_spans_and_materializes_overrides() {
        let keys = ShardedVertexKeys::new(vec![0], vec![]);
        let cells = [VoronoiCell::new(1, 2), VoronoiCell::new(4, 1)];
        let indices = [99, 10, 11, 98, 20, 97];
        let mut work =
            WorkingDiagram::from_reconciled(&[], &keys, LiveCellLayout::new(&cells, &indices));

        assert_eq!(work.num_cells(), 2);
        assert_eq!(work.boundary(0), [10, 11]);
        assert_eq!(work.boundary(1), [20]);

        work.overrides.insert(1, vec![7, 8, 9]);
        assert_eq!(work.boundary(1), [7, 8, 9]);

        let (minted, flat_cells, flat_indices) = work.into_flat();
        assert!(minted.is_empty());
        assert_eq!(flat_indices, [10, 11, 7, 8, 9]);
        assert_eq!(flat_cells[0].vertex_start(), 0);
        assert_eq!(flat_cells[0].vertex_count(), 2);
        assert_eq!(flat_cells[1].vertex_start(), 2);
        assert_eq!(flat_cells[1].vertex_count(), 3);
    }

    #[test]
    fn tiny_polygon_winding_uses_f64_accumulation() {
        let generator = Vec3::new(0.31, -0.27, 0.911_043_35);
        let vertices = [
            Vec3::new(0.310_181_26, -0.270_194_2, 0.911_472_8),
            Vec3::new(0.310_227_72, -0.270_119_3, 0.911_479_2),
            Vec3::new(0.310_170_1, -0.270_078_45, 0.911_510_9),
            Vec3::new(0.310_107_02, -0.270_104_62, 0.911_524_6),
            Vec3::new(0.310_113_82, -0.270_169_26, 0.911_503_14),
        ];

        // The former f32 accumulator reports the opposite winding for this
        // small but otherwise ordinary convex spherical polygon.
        let mut f32_acc = Vec3::ZERO;
        for i in 0..vertices.len() {
            f32_acc += vertices[i].cross(vertices[(i + 1) % vertices.len()]);
        }
        assert!(f32_acc.dot(generator) < 0.0);
        assert!(polygon_sign_f64(generator, vertices.len(), |i| vertices[i]) > 0.0);
    }

    #[test]
    fn winding_parity_propagates_rim_anchor() {
        let nodes = [10, 20, 30].into_iter().collect();
        let mut adj: FxHashMap<u32, Vec<(u32, bool)>> = FxHashMap::default();
        // 10/20 already run oppositely (same flip parity); 20/30 currently
        // run the same way (exactly one must flip).
        adj.insert(10, vec![(20, false)]);
        adj.insert(20, vec![(10, false), (30, true)]);
        adj.insert(30, vec![(20, true)]);
        let anchors = FxHashMap::from_iter([(10, vec![false])]);

        let flips = solve_winding_parity(&nodes, &adj, &anchors, |_| {
            panic!("anchored component must not use the geometric fallback")
        })
        .unwrap();
        assert_eq!(flips, vec![30]);
    }

    #[test]
    fn winding_parity_rejects_conflicting_rim() {
        let nodes = [10, 20].into_iter().collect();
        let adj = FxHashMap::from_iter([(10, vec![(20, false)]), (20, vec![(10, false)])]);
        let anchors = FxHashMap::from_iter([(10, vec![false]), (20, vec![true])]);
        assert!(solve_winding_parity(&nodes, &adj, &anchors, |_| false).is_none());
    }
}
