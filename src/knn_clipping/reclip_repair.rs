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
/// (returned as residual) rather than risk an unbounded re-clip. The de-risking
/// measurement put the largest real component at ~50 generators.
const MAX_COMPONENT_CELLS: usize = 512;

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

/// The single generator shared by two vertex keys other than `g` — the
/// neighbor across the edge between them. `None` if they share != 1 such.
fn shared_other(ki: [u32; 3], kj: [u32; 3], g: u32) -> Option<u32> {
    let mut found = None;
    for &x in &ki {
        if x != g && kj.contains(&x) {
            if found.is_some() {
                return None;
            }
            found = Some(x);
        }
    }
    found
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
    let gpos = |g: u32| {
        let p = points[g as usize];
        DVec3::new(p.x as f64, p.y as f64, p.z as f64)
    };
    let gset: HashSet<u32> = comp.cells.iter().copied().collect();

    // Chart anchored at the component centroid (component cells are a tight
    // cluster). Built from G only, so a far sparse 1-ring neighbor can't tilt it.
    let g_positions: Vec<DVec3> = comp.cells.iter().map(|&g| gpos(g)).collect();
    let chart = delaunay::Chart::new(&g_positions)?;

    // Triangulate ONLY the component cells G (a tight, chart-centered cluster).
    // The 1-ring is kept separately for the emptiness filter, not triangulated:
    // it spreads outward and, projected from the component centroid, distorts
    // enough to wreck the triangulation of the dense cluster.
    let tri_gens: Vec<u32> = comp.cells.clone();
    let tri_proj: Vec<[f64; 2]> = tri_gens
        .iter()
        .map(|&g| chart.project(gpos(g)))
        .collect::<Option<_>>()?;

    // Filter set = every generator in a component cell's existing keys (G plus
    // the 1-ring), as 3-D unit vectors. An interior vertex must be closer to its
    // three generators than to any of these.
    let mut filter: Vec<(u32, DVec3)> = Vec::new();
    let mut filter_seen: HashSet<u32> = HashSet::new();
    for &g in &comp.cells {
        for &vid in cell_span(cells, cell_indices, g)? {
            if let Some(k) = vertex_keys.get(vid) {
                for x in k {
                    if filter_seen.insert(x) {
                        filter.push((x, gpos(x)));
                    }
                }
            }
        }
    }

    let tris = delaunay::triangulate(&tri_proj);
    if tris.is_empty() {
        return None;
    }

    // Interior Voronoi vertices: circumcenters of G-triangles whose circumcircle
    // is empty of every filter generator (so the vertex is a real triple point,
    // not one bounded by a nearer 1-ring neighbor).
    // Map each component generator to the interior vertices incident to it.
    let mut interior_pos: HashMap<[u32; 3], Vec3> = HashMap::new();
    let mut incident_interior: HashMap<u32, Vec<[u32; 3]>> = HashMap::new();
    if trace() {
        let mut incident_all: HashMap<u32, usize> = HashMap::new();
        for t in &tris {
            for &li in t {
                *incident_all.entry(tri_gens[li]).or_insert(0) += 1;
            }
        }
        let missing: Vec<u32> = comp
            .cells
            .iter()
            .copied()
            .filter(|g| !incident_all.contains_key(g))
            .collect();
        eprintln!(
            "[reclip]   diag: {} tri_gens, {} tris; component cells with ZERO incident triangles: {:?}",
            tri_gens.len(),
            tris.len(),
            missing
        );
    }
    // On-circle slack for the emptiness test (chord units): a generator within
    // this of the circumradius is treated as on the circle, not inside.
    const EMPTY_TOL: f64 = 1e-7;
    for t in &tris {
        let g3 = [tri_gens[t[0]], tri_gens[t[1]], tri_gens[t[2]]];
        // (all in G by construction now)
        let key = key3(g3[0], g3[1], g3[2]);
        let Some(cc) = delaunay::circumcenter(gpos(g3[0]), gpos(g3[1]), gpos(g3[2])) else {
            if trace() {
                eprintln!("[reclip]     bail: interior circumcenter degenerate {key:?}");
            }
            return None;
        };
        // Empty-circumcircle test against the full filter set: closer = larger
        // dot, so the vertex is real iff no other generator beats its radius.
        let radius_dot = cc.dot(gpos(g3[0]));
        let empty = filter
            .iter()
            .all(|&(d, dp)| g3.contains(&d) || cc.dot(dp) <= radius_dot + EMPTY_TOL);
        if !empty {
            continue;
        }
        interior_pos.insert(key, Vec3::new(cc.x as f32, cc.y as f32, cc.z as f32));
        for &g in &g3 {
            incident_interior.entry(g).or_default().push(key);
        }
    }

    // Build each component cell's polygon: existing boundary vertices (keys with
    // a non-G generator — pinned) merged with the recomputed interior vertices,
    // ordered by angle around the generator in the shared chart.
    let mut polys: Vec<(u32, Vec<[u32; 3]>)> = Vec::with_capacity(comp.cells.len());
    for &g in &comp.cells {
        let g_xy = chart.ortho(gpos(g));
        // angle of a vertex key around g, from its triple's circumcenter.
        let angle_of = |key: [u32; 3]| -> Option<f64> {
            let cc = delaunay::circumcenter(gpos(key[0]), gpos(key[1]), gpos(key[2]))?;
            let xy = chart.ortho(cc);
            Some((xy[1] - g_xy[1]).atan2(xy[0] - g_xy[0]))
        };
        let mut verts: Vec<(f64, [u32; 3])> = Vec::new();
        // Boundary vertices from g's existing polygon (keys touching outside G).
        for &vid in cell_span(cells, cell_indices, g)? {
            if let Some(k) = vertex_keys.get(vid) {
                if k.iter().any(|x| !gset.contains(x)) {
                    let Some(a) = angle_of(k) else {
                        if trace() {
                            eprintln!(
                                "[reclip]     bail cell {g}: boundary angle degenerate {k:?}"
                            );
                        }
                        return None;
                    };
                    verts.push((a, k));
                }
            }
        }
        // Interior vertices incident to g.
        if let Some(keys) = incident_interior.get(&g) {
            for &k in keys {
                let Some(a) = angle_of(k) else {
                    if trace() {
                        eprintln!("[reclip]     bail cell {g}: interior angle degenerate {k:?}");
                    }
                    return None;
                };
                verts.push((a, k));
            }
        }
        verts.sort_by(|a, b| a.0.total_cmp(&b.0));
        verts.dedup_by_key(|(_, k)| *k);
        if verts.len() < 3 {
            if trace() {
                eprintln!("[reclip]     bail cell {g}: only {} vertices", verts.len());
            }
            return None;
        }
        polys.push((g, verts.into_iter().map(|(_, k)| k).collect()));
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

    // Re-detect: every interior edge (between two touched cells) must now be
    // used by exactly two cells. Boundary edges are pinned (unchanged) and pair
    // by construction, so only interior edges need checking.
    let touched_set: HashSet<u32> = touched.iter().copied().collect();
    let key_of = |vid: u32| -> Option<[u32; 3]> {
        if vid < orig_len {
            vertex_keys.get(vid)
        } else {
            overlay.get((vid - orig_len) as usize).copied()
        }
    };
    let mut edge_use: HashMap<(u32, u32), (u32, [u32; 2])> = HashMap::new();
    for &g in &touched {
        let Some(span) = cell_span(cells, cell_indices, g) else {
            continue;
        };
        let n = span.len();
        for i in 0..n {
            let vi = span[i];
            let vj = span[(i + 1) % n];
            let (Some(ki), Some(kj)) = (key_of(vi), key_of(vj)) else {
                continue;
            };
            let Some(m) = shared_other(ki, kj, g) else {
                continue;
            };
            if !touched_set.contains(&m) {
                continue; // boundary edge: pinned, trusted
            }
            let e = (vi.min(vj), vi.max(vj));
            let slot = edge_use.entry(e).or_insert((0, [g, u32::MAX]));
            if slot.0 < 2 {
                slot.1[slot.0 as usize] = g;
            }
            slot.0 += 1;
        }
    }
    for (_, (count, owners)) in edge_use {
        if count != 2 {
            residual_out.push((owners[0].min(owners[1]), owners[0].max(owners[1])));
        }
    }

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

/// Deterministic local triangulation + spherical circumcenters.
///
/// The contested interior of a component is re-resolved from a single shared
/// computation so every component cell agrees (consistency, not exactness — see
/// the design doc). Bowyer–Watson over a shared gnomonic chart gives one
/// deterministic triangulation; near-cocircular ties resolve to *a* consistent
/// choice (acceptable because such vertices are geometrically ~ambiguous).
mod delaunay {
    use glam::DVec3;
    use std::collections::HashMap;

    /// Shared gnomonic chart for a local generator set: project each unit
    /// generator `p` to the tangent plane at the set centroid.
    pub(super) struct Chart {
        center: DVec3,
        e1: DVec3,
        e2: DVec3,
    }

    impl Chart {
        pub(super) fn new(gens: &[DVec3]) -> Option<Self> {
            let mut c = DVec3::ZERO;
            for &g in gens {
                c += g;
            }
            let len2 = c.length_squared();
            if !len2.is_finite() || len2 <= 1e-18 {
                return None;
            }
            let center = c / len2.sqrt();
            // Any stable tangent basis at `center`.
            let helper = if center.x.abs() < 0.9 {
                DVec3::X
            } else {
                DVec3::Y
            };
            let e1 = (helper - center * helper.dot(center)).normalize();
            let e2 = center.cross(e1);
            Some(Self { center, e1, e2 })
        }

        /// Gnomonic projection; `None` if `p` is on/over the chart horizon.
        pub(super) fn project(&self, p: DVec3) -> Option<[f64; 2]> {
            let d = p.dot(self.center);
            if d <= 1e-6 {
                return None;
            }
            let s = p / d;
            Some([s.dot(self.e1), s.dot(self.e2)])
        }

        /// Orthographic tangent coordinates (no horizon): robust for angular
        /// ordering of vertices around a generator.
        pub(super) fn ortho(&self, p: DVec3) -> [f64; 2] {
            [p.dot(self.e1), p.dot(self.e2)]
        }
    }

    #[inline]
    fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    /// `> 0` when `d` lies strictly inside the circumcircle of CCW triangle
    /// `(a,b,c)`. Orientation-independent (multiplies by the triangle's
    /// orientation sign), so callers need not pre-order the triangle.
    fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
        let adx = a[0] - d[0];
        let ady = a[1] - d[1];
        let bdx = b[0] - d[0];
        let bdy = b[1] - d[1];
        let cdx = c[0] - d[0];
        let cdy = c[1] - d[1];
        let ad = adx * adx + ady * ady;
        let bd = bdx * bdx + bdy * bdy;
        let cd = cdx * cdx + cdy * cdy;
        let det = adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx)
            + ad * (bdx * cdy - bdy * cdx);
        det * orient2d(a, b, c)
    }

    /// Bowyer–Watson triangulation of `pts` (indices into the returned tris are
    /// indices into `pts`). Returns CCW-or-CW triangles covering the convex
    /// hull. Deterministic; ties (cocircular) resolve via the strict `> tol`
    /// test to a single consistent triangulation.
    pub(super) fn triangulate(pts: &[[f64; 2]]) -> Vec<[usize; 3]> {
        let n = pts.len();
        if n < 3 {
            return Vec::new();
        }
        // Super-triangle covering the bbox, vertices indexed n, n+1, n+2.
        let (mut minx, mut miny, mut maxx, mut maxy) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for p in pts {
            minx = minx.min(p[0]);
            miny = miny.min(p[1]);
            maxx = maxx.max(p[0]);
            maxy = maxy.max(p[1]);
        }
        let dmax = (maxx - minx).max(maxy - miny).max(1e-12);
        let midx = 0.5 * (minx + maxx);
        let midy = 0.5 * (miny + maxy);
        let st = [
            [midx - 1000.0 * dmax, midy - 1000.0 * dmax],
            [midx, midy + 1000.0 * dmax],
            [midx + 1000.0 * dmax, midy - 1000.0 * dmax],
        ];
        let coord = |i: usize| -> [f64; 2] {
            if i < n {
                pts[i]
            } else {
                st[i - n]
            }
        };
        let tol = 1e-12 * dmax * dmax;

        let mut tris: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];
        for (i, &p) in pts.iter().enumerate() {
            // Triangles whose circumcircle contains p are "bad".
            let mut bad = Vec::new();
            for (ti, t) in tris.iter().enumerate() {
                if in_circle(coord(t[0]), coord(t[1]), coord(t[2]), p) > tol {
                    bad.push(ti);
                }
            }
            // Cavity boundary = undirected edges appearing in exactly one bad
            // triangle. Orientation-independent (count by sorted pair): the
            // triangles carry no consistent winding, so directed-edge matching
            // would mis-cancel genuine boundary edges and lose points.
            let mut edge_count: HashMap<(usize, usize), u32> = HashMap::new();
            for &ti in &bad {
                let t = tris[ti];
                for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *edge_count.entry((e.0.min(e.1), e.0.max(e.1))).or_insert(0) += 1;
                }
            }
            // Remove bad triangles (descending index so swap_remove is safe).
            bad.sort_unstable_by(|a, b| b.cmp(a));
            for ti in bad {
                tris.swap_remove(ti);
            }
            // Re-triangulate the cavity: connect p to each boundary edge,
            // oriented CCW so created triangles have consistent winding.
            for ((u, v), c) in edge_count {
                if c != 1 {
                    continue;
                }
                let tri = if orient2d(coord(u), coord(v), p) >= 0.0 {
                    [u, v, i]
                } else {
                    [v, u, i]
                };
                tris.push(tri);
            }
        }
        // Drop triangles touching a super-triangle vertex.
        tris.retain(|t| t.iter().all(|&v| v < n));
        tris
    }

    /// Unit direction equidistant (spherical) from three unit generators — the
    /// Voronoi vertex of triangle `(a,b,c)`. `None` if degenerate.
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

        fn valid_triangulation(pts: &[[f64; 2]], tris: &[[usize; 3]]) -> bool {
            // No triangle's circumcircle strictly contains another point
            // (Delaunay), and every triangle is non-degenerate.
            for t in tris {
                if orient2d(pts[t[0]], pts[t[1]], pts[t[2]]).abs() < 1e-15 {
                    return false;
                }
                for (j, &p) in pts.iter().enumerate() {
                    if j == t[0] || j == t[1] || j == t[2] {
                        continue;
                    }
                    if in_circle(pts[t[0]], pts[t[1]], pts[t[2]], p) > 1e-9 {
                        return false;
                    }
                }
            }
            true
        }

        #[test]
        fn triangulates_square_consistently() {
            // 4 cocircular corners → 2 triangles, deterministic.
            let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            let t1 = triangulate(&pts);
            let t2 = triangulate(&pts);
            assert_eq!(t1.len(), 2, "square → 2 triangles");
            assert_eq!(t1, t2, "deterministic");
            assert!(valid_triangulation(&pts, &t1));
        }

        #[test]
        fn triangulates_grid_covers_all_points() {
            // 6x6 jittered grid: every interior point must be incident to a
            // triangle, and the count must match Euler (2n - 2 - hull).
            let mut pts = Vec::new();
            for i in 0..6u32 {
                for j in 0..6u32 {
                    let jit = ((i * 7 + j * 13) % 5) as f64 * 1e-3;
                    pts.push([i as f64 + jit, j as f64 - jit]);
                }
            }
            let tris = triangulate(&pts);
            let mut incident = vec![false; pts.len()];
            for t in &tris {
                for &v in t {
                    incident[v] = true;
                }
            }
            assert!(incident.iter().all(|&x| x), "every point incident");
            // A correct triangulation has 2n-2-h triangles (h = hull size); for
            // this mostly-interior grid that is ~50. The lost-points bug gave
            // ~n; require comfortably more than n to catch it.
            assert!(
                tris.len() >= pts.len() + pts.len() / 3,
                "triangle count {} too low for {} points (Bowyer–Watson lost points)",
                tris.len(),
                pts.len()
            );
            assert!(valid_triangulation(&pts, &tris));
        }

        #[test]
        fn triangulates_pentagon_fan() {
            let pts = [[0.0, 0.0], [1.0, 0.1], [0.6, 1.0], [-0.6, 0.9], [-1.0, 0.0]];
            let tris = triangulate(&pts);
            // Convex position, 5 points → 2n-2-h = 10-2-5 = 3 triangles.
            assert_eq!(tris.len(), 3);
            assert!(valid_triangulation(&pts, &tris));
        }

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

        #[test]
        fn chart_projects_round_trip_order() {
            let gens = [
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(0.05, 0.0, 1.0).normalize(),
                DVec3::new(0.0, 0.05, 1.0).normalize(),
            ];
            let chart = Chart::new(&gens).unwrap();
            for g in gens {
                assert!(chart.project(g).is_some());
            }
        }
    }
}
