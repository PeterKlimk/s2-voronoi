//! Tier-2 re-clip repair (spherical).
//!
//! Runs only on the residual that the cheap Tier-1 stitch (`edge_reconcile`)
//! could not pair. Each surviving unpaired interior edge marks a contested
//! cluster of cells that disagree on a high-degree degenerate vertex; this
//! pass re-resolves each cluster jointly against a pinned boundary of
//! already-valid cells. See `docs/reclip-repair-design.md`.
//!
//! Opt-in via `S2_RECLIP_REPAIR` while under development; the default path is
//! unchanged (the loud `residual_error` still fires on any survivor).

use crate::diagram::VoronoiCell;
use crate::knn_clipping::edge_reconcile::{self, VertexKeys};
use crate::knn_clipping::union_find::SparseUnionFind;
use crate::live_dedup::ShardedVertexKeys;
use glam::{DVec3, Vec3};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Cap on a single component's cell count; larger contested regions fail loud
/// (returned as residual) rather than risk an unbounded re-clip. The interior
/// solve is O(|G|^3 * filter), so this also bounds cost; the de-risking
/// measurement put the largest real component at ~50 generators.
const MAX_COMPONENT_CELLS: usize = 128;

/// Whether the Tier-2 re-clip pass is enabled (off by default).
pub(crate) fn enabled() -> bool {
    std::env::var("S2_RECLIP_REPAIR").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn trace() -> bool {
    std::env::var("S2_RECLIP_TRACE").is_ok()
}

/// One contested cluster: the cells (generators) that disagree about a
/// high-degree degenerate vertex, plus the raw unpaired interior edges within
/// it. The boundary (already-paired edges to outside cells) is derived later
/// by the resolver.
pub(crate) struct Component {
    /// Generator indices of the cells in this component (sorted, unique).
    pub(crate) cells: Vec<u32>,
    /// Raw unpaired interior edges `(va, vb, owner)` belonging to this cluster.
    pub(crate) edges: Vec<(u32, u32, u32)>,
}

/// Identify the contested connected components from Tier-1's residual.
///
/// The component is the transitive closure of *contested adjacency*: every
/// generator named in either endpoint key of an unpaired edge is unioned, so a
/// degenerate vertex's whole petal lands in one component (matches the
/// de-risking measurement). By construction the component's outer boundary
/// consists only of already-paired edges (a contested vertex on the boundary
/// would pull its outside cell in), so it is safe to pin.
fn identify_components(
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
    residual: &[(u32, u32)],
) -> Result<Vec<Component>, crate::VoronoiError> {
    // Candidate cells = those named by Tier-1's residual pairs. Tier-1 derived
    // those pairs from the residual edges, so every residual edge's owner is
    // covered; the localized scan re-surfaces the raw `(va, vb, owner)` edges.
    let mut candidates: Vec<u32> = residual.iter().flat_map(|&(a, b)| [a, b]).collect();
    candidates.sort_unstable();
    candidates.dedup();

    let raw = edge_reconcile::scan_unpaired_interior(
        cells,
        cell_indices,
        VertexKeys::Sharded(vertex_keys),
        &candidates,
        &|_, _| false,
    )?;

    // Union every generator incident to a contested edge's two endpoints.
    let mut uf = SparseUnionFind::new();
    for &(va, vb, owner) in &raw {
        let mut gens: Vec<u32> = Vec::with_capacity(6);
        for v in [va, vb] {
            if let Some(k) = vertex_keys.get(v) {
                gens.extend_from_slice(&k);
            }
        }
        if gens.is_empty() {
            gens.push(owner);
        }
        let first = gens[0];
        for &g in &gens[1..] {
            uf.union(first, g);
        }
    }

    // Group edges and cells by component root.
    let mut by_root: BTreeMap<u32, Component> = BTreeMap::new();
    for &(va, vb, owner) in &raw {
        // Attribute the edge to the component of its first incident generator.
        let anchor = vertex_keys
            .get(va)
            .map(|k| k[0])
            .or_else(|| vertex_keys.get(vb).map(|k| k[0]))
            .unwrap_or(owner);
        let r = uf.find(anchor);
        let comp = by_root.entry(r).or_insert_with(|| Component {
            cells: Vec::new(),
            edges: Vec::new(),
        });
        comp.edges.push((va, vb, owner));
        for v in [va, vb] {
            if let Some(k) = vertex_keys.get(v) {
                comp.cells.extend_from_slice(&k);
            }
        }
    }

    let mut comps: Vec<Component> = by_root.into_values().collect();
    for c in &mut comps {
        c.cells.sort_unstable();
        c.cells.dedup();
    }
    Ok(comps)
}

#[inline]
fn key3(a: u32, b: u32, c: u32) -> [u32; 3] {
    let mut k = [a, b, c];
    k.sort_unstable();
    k
}

/// Deterministic pseudo-random value in [-1, 1] keyed by `(gen, axis)`. Used to
/// jitter projected generators so no two are coincident/cocircular; stable per
/// generator id so every component sees the same perturbation.
#[inline]
fn jitter_unit(gen: u32, axis: u32) -> f64 {
    // SplitMix64-style avalanche on a mixed key.
    let mut z = (gen as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((axis as u64) << 1 | 1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Map to [-1, 1).
    (z as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// Order cell `g`'s vertices into a boundary cycle by *combinatorial* adjacency
/// (two vertices are adjacent iff they share a neighbor of `g`), not by angle —
/// angle-ordering is unstable for the near-coincident vertices of a degenerate
/// corner and produces spurious adjacencies. `None` if the vertices do not form
/// a single clean cycle (every neighbor must appear in exactly two of `g`'s
/// vertices), which flags a malformed/incomplete cell.
fn build_cycle(g: u32, verts: &[[u32; 3]]) -> Option<Vec<[u32; 3]>> {
    let n = verts.len();
    if n < 3 {
        return None;
    }
    // neighbor -> indices of g's vertices containing it (must be exactly 2).
    let mut by_nbr: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, v) in verts.iter().enumerate() {
        for &x in v {
            if x != g {
                by_nbr.entry(x).or_default().push(i);
            }
        }
    }
    // Build the 2-regular adjacency graph (each shared neighbor links 2 verts).
    let mut adj: Vec<[usize; 2]> = vec![[usize::MAX; 2]; n];
    let mut deg = vec![0usize; n];
    for vs in by_nbr.values() {
        if vs.len() != 2 {
            return None;
        }
        let (a, b) = (vs[0], vs[1]);
        if deg[a] == 2 || deg[b] == 2 {
            return None;
        }
        adj[a][deg[a]] = b;
        deg[a] += 1;
        adj[b][deg[b]] = a;
        deg[b] += 1;
    }
    if deg.iter().any(|&d| d != 2) {
        return None;
    }
    // Walk the single cycle.
    let mut order = Vec::with_capacity(n);
    let (mut prev, mut cur) = (usize::MAX, 0usize);
    for _ in 0..n {
        order.push(cur);
        let nxt = if adj[cur][0] != prev {
            adj[cur][0]
        } else {
            adj[cur][1]
        };
        prev = cur;
        cur = nxt;
    }
    if cur != 0 || order.len() != n {
        return None; // multiple disjoint cycles -> not a simple polygon
    }
    Some(order.into_iter().map(|i| verts[i]).collect())
}

#[inline]
fn cell_span<'a>(cells: &[VoronoiCell], cell_indices: &'a [u32], g: u32) -> Option<&'a [u32]> {
    let c = cells.get(g as usize)?;
    let start = c.vertex_start();
    let end = start + c.vertex_count();
    cell_indices.get(start..end)
}

/// Per-cell re-resolved polygons (as ordered vertex keys) plus the positions of
/// the new interior vertices.
struct Resolved {
    polys: Vec<(u32, Vec<[u32; 3]>)>,
    interior_pos: HashMap<[u32; 3], Vec3>,
}

/// Re-resolve one component into consistent per-cell key polygons by
/// triangulating its local generator set in a single shared chart. `None` to
/// bail (component is left as residual): horizon/degeneracy, or a non-triangle
/// fan that would not be a clean cell.
fn resolve_component(
    comp: &Component,
    points: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &ShardedVertexKeys,
) -> Option<Resolved> {
    let gset: HashSet<u32> = comp.cells.iter().copied().collect();
    // Jittered unit positions: a deterministic per-generator perturbation (~1e-9,
    // keyed by global id) so no triple is exactly cocircular/coincident. This
    // makes the degenerate-split choice jitter-determined — but consistently, as
    // the jitter is shared across cells — and keeps positions within ~1e-9 of
    // true (well inside the near-Voronoi tolerance). All geometry uses jittered
    // positions so the emptiness decision and the output vertex agree.
    let gjit = |g: u32| -> DVec3 {
        let p = points[g as usize];
        let base = DVec3::new(p.x as f64, p.y as f64, p.z as f64);
        const J: f64 = 1e-9;
        let jit = DVec3::new(jitter_unit(g, 0), jitter_unit(g, 1), jitter_unit(g, 2)) * J;
        (base + jit).normalize()
    };

    // Filter set = every generator in a component cell's existing keys (G plus
    // the 1-ring). An interior vertex must be closer to its three generators
    // than to any of these.
    let mut filter: Vec<(u32, DVec3)> = Vec::new();
    let mut filter_seen: HashSet<u32> = HashSet::new();
    for &g in &comp.cells {
        for &vid in cell_span(cells, cell_indices, g)? {
            if let Some(k) = vertex_keys.get(vid) {
                for x in k {
                    if filter_seen.insert(x) {
                        filter.push((x, gjit(x)));
                    }
                }
            }
        }
    }

    // Interior Voronoi vertices: every all-G triple whose circumcircle is empty
    // of all filter generators. Brute force over triples — no triangulation, so
    // no generator can be orphaned — and jitter removes the cocircular
    // over-counting that made the per-cell emptiness inconsistent. The triple set
    // is a single deterministic computation, so every component cell inherits the
    // same vertices (consistent by construction).
    const EMPTY_TOL: f64 = 1e-12; // f64 noise floor; jitter (1e-9) dominates ties.
    let gvec = &comp.cells;
    let gp: Vec<DVec3> = gvec.iter().map(|&g| gjit(g)).collect();
    let mut interior_pos: HashMap<[u32; 3], Vec3> = HashMap::new();
    let mut incident_interior: HashMap<u32, Vec<[u32; 3]>> = HashMap::new();
    for i in 0..gvec.len() {
        for j in (i + 1)..gvec.len() {
            for k in (j + 1)..gvec.len() {
                let Some(cc) = delaunay::circumcenter(gp[i], gp[j], gp[k]) else {
                    continue; // collinear triple: not a vertex
                };
                let g3 = [gvec[i], gvec[j], gvec[k]];
                let radius = cc.dot(gp[i]);
                let empty = filter
                    .iter()
                    .all(|&(d, dp)| g3.contains(&d) || cc.dot(dp) <= radius + EMPTY_TOL);
                if !empty {
                    continue;
                }
                let key = key3(g3[0], g3[1], g3[2]);
                interior_pos.insert(key, Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32));
                for &g in &g3 {
                    incident_interior.entry(g).or_default().push(key);
                }
            }
        }
    }
    if trace() {
        eprintln!(
            "[reclip]   diag: {} cells, {} filter gens, {} interior vertices",
            gvec.len(),
            filter.len(),
            interior_pos.len()
        );
    }

    // Build each cell's polygon: existing boundary vertices (keys touching
    // outside G — pinned) plus the incident interior vertices, ordered into a
    // cycle by shared-neighbor adjacency (robust to the near-coincident vertices
    // of a degenerate corner, where angle-ordering is unstable).
    let mut polys: Vec<(u32, Vec<[u32; 3]>)> = Vec::with_capacity(comp.cells.len());
    for &g in &comp.cells {
        let mut verts: Vec<[u32; 3]> = Vec::new();
        for &vid in cell_span(cells, cell_indices, g)? {
            if let Some(k) = vertex_keys.get(vid) {
                if k.iter().any(|x| !gset.contains(x)) {
                    verts.push(k);
                }
            }
        }
        if let Some(keys) = incident_interior.get(&g) {
            verts.extend_from_slice(keys);
        }
        verts.sort_unstable();
        verts.dedup();
        let Some(poly) = build_cycle(g, &verts) else {
            if trace() {
                eprintln!(
                    "[reclip]     bail cell {g}: {} vertices do not form a clean cycle",
                    verts.len()
                );
            }
            return None;
        };
        polys.push((g, poly));
    }

    Some(Resolved {
        polys,
        interior_pos,
    })
}

/// Re-resolve contested components that survived Tier-1, returning the residual
/// still unpaired afterward (empty on full success). See
/// `docs/reclip-repair-design.md`.
// `&mut Vec` (not `&mut [_]`): re-stitch appends re-resolved cell spans to
// `cell_indices` and new interior vertices to `vertices`.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
pub(crate) fn repair(
    points: &[Vec3],
    vertices: &mut Vec<Vec3>,
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: &ShardedVertexKeys,
    residual: Vec<(u32, u32)>,
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    let comps = identify_components(cells, cell_indices, vertex_keys, &residual)?;
    if trace() {
        eprintln!(
            "[reclip] {} residual pair(s) -> {} component(s)",
            residual.len(),
            comps.len()
        );
    }

    // Existing key -> vid, for pinning boundary vertices (built from the current
    // polygons of all component cells; boundary keys are already consistent).
    let mut existing: HashMap<[u32; 3], u32> = HashMap::new();
    for comp in &comps {
        for &g in &comp.cells {
            if let Some(span) = cell_span(cells, cell_indices, g) {
                for &vid in span {
                    if let Some(k) = vertex_keys.get(vid) {
                        existing.entry(k).or_insert(vid);
                    }
                }
            }
        }
    }

    let orig_len = vertices.len() as u32;
    let mut interior_vid: HashMap<[u32; 3], u32> = HashMap::new();
    let mut overlay: Vec<[u32; 3]> = Vec::new();
    let mut residual_out: Vec<(u32, u32)> = Vec::new();
    let mut touched: Vec<u32> = Vec::new();

    'comp: for comp in &comps {
        let bail = |out: &mut Vec<(u32, u32)>| {
            for &(_, _, owner) in &comp.edges {
                out.push((owner, owner));
            }
        };
        if comp.cells.len() > MAX_COMPONENT_CELLS {
            if trace() {
                eprintln!(
                    "[reclip]   bail: component too large ({} cells)",
                    comp.cells.len()
                );
            }
            bail(&mut residual_out);
            continue;
        }
        let Some(res) = resolve_component(comp, points, cells, cell_indices, vertex_keys) else {
            if trace() {
                eprintln!(
                    "[reclip]   bail: resolve_component failed ({} cells)",
                    comp.cells.len()
                );
            }
            bail(&mut residual_out);
            continue;
        };

        // Validate: every boundary key (not interior) must already exist to pin.
        for (_, poly) in &res.polys {
            for key in poly {
                if !res.interior_pos.contains_key(key) && !existing.contains_key(key) {
                    if trace() {
                        eprintln!("[reclip]   bail: unpinnable boundary key {key:?}");
                    }
                    bail(&mut residual_out);
                    continue 'comp;
                }
            }
        }

        // Assign interior vids (append positions + overlay keys, deduped by key).
        for (key, pos) in &res.interior_pos {
            interior_vid.entry(*key).or_insert_with(|| {
                let v = orig_len + overlay.len() as u32;
                overlay.push(*key);
                vertices.push(*pos);
                v
            });
        }

        // Re-stitch: append each new polygon and repoint its cell.
        for (g, poly) in &res.polys {
            let start = cell_indices.len() as u32;
            for key in poly {
                let vid = interior_vid
                    .get(key)
                    .copied()
                    .or_else(|| existing.get(key).copied())
                    .expect("validated above");
                cell_indices.push(vid);
            }
            cells[*g as usize] = VoronoiCell::new(start, poly.len() as u16);
            touched.push(*g);
        }
    }

    // Re-detect, matching the validator exactly: count directed edge uses by
    // undirected VID pair (not by key — a malformed re-stitched polygon can have
    // consecutive vertices that share no clean neighbor, which a key-based check
    // would skip but the validator counts). Scope to the affected region: the
    // touched cells plus every cell sharing a vertex with them. An edge with a
    // touched owner must be used exactly twice; anything else is residual. This
    // is the SAME invariant `validation::validate` enforces, so a clean result
    // here means the returned diagram is a valid subdivision (no silent ship).
    let touched_set: HashSet<u32> = touched.iter().copied().collect();
    let key_of = |vid: u32| -> Option<[u32; 3]> {
        if vid < orig_len {
            vertex_keys.get(vid)
        } else {
            overlay.get((vid - orig_len) as usize).copied()
        }
    };
    let mut region: HashSet<u32> = touched_set.clone();
    for &g in &touched {
        if let Some(span) = cell_span(cells, cell_indices, g) {
            for &vid in span {
                if let Some(k) = key_of(vid) {
                    region.extend(k);
                }
            }
        }
    }
    let mut edge_use: HashMap<(u32, u32), (u32, bool, [u32; 2])> = HashMap::new();
    for &c in &region {
        let is_touched = touched_set.contains(&c);
        let Some(span) = cell_span(cells, cell_indices, c) else {
            continue;
        };
        let n = span.len();
        for i in 0..n {
            let a = span[i];
            let b = span[(i + 1) % n];
            let e = (a.min(b), a.max(b));
            let slot = edge_use
                .entry(e)
                .or_insert((0, false, [u32::MAX, u32::MAX]));
            if (slot.0 as usize) < 2 {
                slot.2[slot.0 as usize] = c;
            }
            slot.0 += 1;
            slot.1 |= is_touched;
        }
    }
    for (edge, (count, touched_owner, owners)) in edge_use {
        // Only edges incident to a re-stitched cell are our concern; edges
        // entirely among unchanged outside cells (touched_owner=false) pair as
        // before. An in-scope edge must be used by exactly two cells.
        if touched_owner && count != 2 {
            if trace() {
                eprintln!(
                    "[reclip]   BADEDGE vids=({},{}) keys=({:?},{:?}) count={count} owners={owners:?}",
                    edge.0,
                    edge.1,
                    key_of(edge.0),
                    key_of(edge.1)
                );
            }
            let a = owners[0];
            let b = if owners[1] == u32::MAX {
                owners[0]
            } else {
                owners[1]
            };
            residual_out.push((a.min(b), a.max(b)));
        }
    }
    residual_out.sort_unstable();
    residual_out.dedup();

    if trace() {
        eprintln!(
            "[reclip] re-stitched {} cell(s), {} new interior vertices; {} residual edge(s) remain",
            touched.len(),
            overlay.len(),
            residual_out.len()
        );
    }
    Ok(residual_out)
}

/// Spherical circumcenter helper.
mod delaunay {
    use glam::DVec3;

    /// Unit direction equidistant (spherical) from three unit generators — the
    /// Voronoi vertex of triangle `(a,b,c)`. `None` if degenerate (collinear).
    pub(super) fn circumcenter(a: DVec3, b: DVec3, c: DVec3) -> Option<DVec3> {
        let nrm = (a - b).cross(b - c);
        let len2 = nrm.length_squared();
        if !len2.is_finite() || len2 <= 1e-30 {
            return None;
        }
        let dir = nrm / len2.sqrt();
        Some(if dir.dot(a) >= 0.0 { dir } else { -dir })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn circumcenter_is_equidistant() {
            let a = DVec3::new(0.0, 0.0, 1.0);
            let b = DVec3::new(0.1, 0.0, 1.0).normalize();
            let c = DVec3::new(0.0, 0.1, 1.0).normalize();
            let p = circumcenter(a, b, c).unwrap();
            let (da, db, dc) = (p.dot(a), p.dot(b), p.dot(c));
            assert!(
                (da - db).abs() < 1e-9 && (db - dc).abs() < 1e-9,
                "equidistant"
            );
        }
    }
}
