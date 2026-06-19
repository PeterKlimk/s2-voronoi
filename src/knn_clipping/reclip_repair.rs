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
use glam::Vec3;
use std::collections::BTreeMap;

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

/// Re-resolve contested components that survived Tier-1, returning the residual
/// still unpaired afterward (empty on full success).
///
/// Currently identifies the components (and traces them under `S2_RECLIP_TRACE`)
/// but does not yet re-clip — returns the input residual unchanged. The
/// resolver and re-stitch phases land incrementally.
// `&mut Vec` (not `&mut [_]`): the re-stitch phase appends re-resolved cell
// spans to `cell_indices`, so it needs to grow the buffer, not just mutate it.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
pub(crate) fn repair(
    _points: &[Vec3],
    _vertices: &[Vec3],
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
        for (i, c) in comps.iter().enumerate() {
            eprintln!(
                "  comp {i}: {} cells, {} contested edges; cells={:?}",
                c.cells.len(),
                c.edges.len(),
                c.cells
            );
        }
    }
    Ok(residual)
}

/// Deterministic local triangulation + spherical circumcenters.
///
/// The contested interior of a component is re-resolved from a single shared
/// computation so every component cell agrees (consistency, not exactness — see
/// the design doc). Bowyer–Watson over a shared gnomonic chart gives one
/// deterministic triangulation; near-cocircular ties resolve to *a* consistent
/// choice (acceptable because such vertices are geometrically ~ambiguous).
mod delaunay {
    // Tested primitives; wired into the resolver in the next step (interior
    // re-resolution + re-stitch). Allow dead_code until then.
    #![allow(dead_code)]
    use glam::DVec3;

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
            [midx - 20.0 * dmax, midy - dmax],
            [midx, midy + 20.0 * dmax],
            [midx + 20.0 * dmax, midy - dmax],
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
            // Cavity boundary = edges appearing in exactly one bad triangle.
            let mut edges: Vec<[usize; 2]> = Vec::new();
            for &ti in &bad {
                let t = tris[ti];
                for e in [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] {
                    // Shared iff the reverse edge is also present among bad tris.
                    if let Some(pos) = edges.iter().position(|x| {
                        (x[0] == e[1] && x[1] == e[0]) || (x[0] == e[0] && x[1] == e[1])
                    }) {
                        edges.swap_remove(pos);
                    } else {
                        edges.push(e);
                    }
                }
            }
            // Remove bad triangles (descending index so swap_remove is safe).
            bad.sort_unstable_by(|a, b| b.cmp(a));
            for ti in bad {
                tris.swap_remove(ti);
            }
            // Re-triangulate the cavity by connecting p to each boundary edge.
            for e in edges {
                tris.push([e[0], e[1], i]);
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
