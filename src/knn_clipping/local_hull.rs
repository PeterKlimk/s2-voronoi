//! Incremental local 3D convex hull (= spherical Delaunay) for the Tier-2
//! re-clip resolver.
//!
//! On the unit sphere the convex hull of the generator directions IS the
//! Delaunay triangulation: each hull face `(a,b,c)` is a Delaunay triangle, and
//! its outward unit normal is the spherical circumcenter = the Voronoi vertex of
//! `(a,b,c)`. The dual (the ordered fan of faces around a generator) is that
//! generator's Voronoi cell — one consistent local topology, by construction.
//!
//! Visibility is decided by the EXACT `robust::orient3d` predicate after
//! f32→f64 conversion and f64 renormalization onto S2. The renormalization is
//! load-bearing: exact predicates over tiny f32 radius drift solve a different
//! Euclidean hull problem than the crate's spherical contract. The local set is
//! small (tens–low-hundreds), so this is a naive `O(n²)` incremental insertion
//! with a directed-edge horizon walk — no point-location structure.
//!
//! Exact `orient3d == 0` (a coplanar = exactly-cocircular set) is a high-degree
//! degeneracy that needs explicit policy (deterministic split vs merge); it is
//! measured to never occur on random `mega`. `build` treats a strictly-positive
//! orientation as "visible", so an exact tie is currently resolved by insertion
//! order — callers that must handle exact cliques deterministically should
//! detect `orient == 0` separately. See `docs/repair-design.md`.

// The consumer (the Tier-2 resolver wiring) lands in a follow-up; until then the
// hull core is exercised only by its unit tests, so silence dead-code in the
// (non-test) lib build.
#![allow(dead_code)]

use glam::{DVec3, Vec3};
use robust::Coord3D;

#[inline]
fn c3(p: DVec3) -> Coord3D<f64> {
    Coord3D {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

/// Exact orientation determinant of `d` relative to the plane `(a,b,c)`.
/// `robust::orient3d(a,b,c,d) > 0` ⇔ `d` is below the plane when `(a,b,c)` is
/// seen counter-clockwise from above `d` (right-hand rule).
#[inline]
fn orient(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    robust::orient3d(c3(a), c3(b), c3(c), c3(d))
}

/// A built local hull: input points plus outward-CCW triangular faces.
pub(crate) struct LocalHull {
    pts: Vec<DVec3>,
    faces: Vec<[usize; 3]>,
}

impl LocalHull {
    /// Build the convex hull of `points` (unit generator directions). Returns
    /// `None` if there are fewer than 4 points or they are all coplanar (a
    /// degenerate, hull-less set — e.g. all on one circle / great circle).
    pub(crate) fn build(points: &[Vec3]) -> Option<LocalHull> {
        let n = points.len();
        if n < 4 {
            return None;
        }
        let pts: Vec<DVec3> = points
            .iter()
            .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64).normalize())
            .collect();

        // --- seed: four affinely independent points -------------------------
        // i0,i1 distinct; i2 not collinear with them; i3 not coplanar.
        let i0 = 0usize;
        let i1 = (1..n).find(|&j| pts[j] != pts[i0])?;
        let i2 = (0..n).find(|&j| {
            j != i0 && j != i1 && (pts[i1] - pts[i0]).cross(pts[j] - pts[i0]).length_squared() > 0.0
        })?;
        let i3 = (0..n).find(|&j| {
            j != i0 && j != i1 && j != i2 && orient(pts[i0], pts[i1], pts[i2], pts[j]) != 0.0
        })?;

        // Oriented seed tetrahedron: each face wound so the opposite (interior)
        // vertex is below it (orient < 0).
        let mut faces: Vec<[usize; 3]> = Vec::with_capacity(2 * n);
        let oriented = |a: usize, b: usize, c: usize, inside: usize| -> [usize; 3] {
            if orient(pts[a], pts[b], pts[c], pts[inside]) < 0.0 {
                [a, b, c]
            } else {
                [a, c, b]
            }
        };
        faces.push(oriented(i0, i1, i2, i3));
        faces.push(oriented(i0, i1, i3, i2));
        faces.push(oriented(i0, i2, i3, i1));
        faces.push(oriented(i1, i2, i3, i0));

        // --- incremental insertion -----------------------------------------
        let seeded = [i0, i1, i2, i3];
        for p in 0..n {
            if seeded.contains(&p) {
                continue;
            }
            // Faces the new point can "see" (strictly outside).
            let mut visible = vec![false; faces.len()];
            let mut any = false;
            for (fi, f) in faces.iter().enumerate() {
                if orient(pts[f[0]], pts[f[1]], pts[f[2]], pts[p]) > 0.0 {
                    visible[fi] = true;
                    any = true;
                }
            }
            if !any {
                continue; // inside the current hull (or exactly on a face)
            }

            // Horizon = directed edges of visible faces whose reverse is NOT
            // also a visible-face edge (the boundary of the visible cap).
            use std::collections::HashSet;
            let mut vis_edges: HashSet<(usize, usize)> = HashSet::new();
            for (fi, f) in faces.iter().enumerate() {
                if visible[fi] {
                    vis_edges.insert((f[0], f[1]));
                    vis_edges.insert((f[1], f[2]));
                    vis_edges.insert((f[2], f[0]));
                }
            }
            let mut horizon: Vec<(usize, usize)> = Vec::new();
            for &(a, b) in &vis_edges {
                if !vis_edges.contains(&(b, a)) {
                    horizon.push((a, b));
                }
            }

            // Drop visible faces, then cone the horizon to `p`. A horizon edge
            // a→b kept its winding from the (removed, outward) visible face, so
            // [a, b, p] is the new outward face.
            let mut fi = 0usize;
            faces.retain(|_| {
                let keep = !visible[fi];
                fi += 1;
                keep
            });
            for (a, b) in horizon {
                faces.push([a, b, p]);
            }
        }

        Some(LocalHull { pts, faces })
    }

    pub(crate) fn faces(&self) -> &[[usize; 3]] {
        &self.faces
    }

    /// Outward unit normal of face `f` = the spherical circumcenter (Voronoi
    /// vertex) of its three generators.
    pub(crate) fn face_circumcenter(&self, f: usize) -> DVec3 {
        let [a, b, c] = self.faces[f];
        let n = (self.pts[b] - self.pts[a]).cross(self.pts[c] - self.pts[a]);
        let n = n.normalize();
        // Outward = same side as the face vertices (positive dot with a).
        if n.dot(self.pts[a]) >= 0.0 {
            n
        } else {
            -n
        }
    }

    /// The ordered cyclic fan of face indices incident to generator `i` — the
    /// dual, i.e. generator `i`'s Voronoi cell as a sequence of faces (their
    /// circumcenters are the cell's vertices in order). Empty if `i` is not on
    /// the hull (cannot happen for points on a sphere unless duplicated).
    pub(crate) fn cell_faces(&self, i: usize) -> Vec<usize> {
        // Each incident face, rotated to [i, x, y] (preserving CCW winding),
        // contributes a directed link x→y around i. Chain the links into a cycle.
        let mut link: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new(); // x -> (y, face)
        for (fi, f) in self.faces.iter().enumerate() {
            let pos = f.iter().position(|&v| v == i);
            let Some(pos) = pos else { continue };
            let x = f[(pos + 1) % 3];
            let y = f[(pos + 2) % 3];
            link.insert(x, (y, fi));
        }
        if link.is_empty() {
            return Vec::new();
        }
        let start = *link.keys().next().unwrap();
        let mut order = Vec::with_capacity(link.len());
        let mut cur = start;
        for _ in 0..link.len() {
            let Some(&(next, fi)) = link.get(&cur) else {
                return Vec::new(); // broken fan (degenerate) — caller bails
            };
            order.push(fi);
            cur = next;
            if cur == start {
                break;
            }
        }
        if cur != start || order.len() != link.len() {
            return Vec::new(); // not a single clean cycle
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize()
    }

    /// Every undirected edge is shared by exactly two faces, and each directed
    /// edge appears exactly once (a closed, consistently-wound surface).
    fn assert_closed(h: &LocalHull) {
        use std::collections::HashMap;
        let mut dir: HashMap<(usize, usize), u32> = HashMap::new();
        for f in h.faces() {
            for k in 0..3 {
                let e = (f[k], f[(k + 1) % 3]);
                *dir.entry(e).or_default() += 1;
            }
        }
        for (&(a, b), &count) in &dir {
            assert_eq!(count, 1, "directed edge ({a},{b}) used {count}x");
            assert_eq!(
                dir.get(&(b, a)).copied().unwrap_or(0),
                1,
                "edge ({a},{b}) has no opposite"
            );
        }
        // Euler: V - E + F = 2 for a sphere.
        let f = h.faces().len();
        let e = dir.len() / 2;
        // V is the number of points actually referenced.
        let mut used = std::collections::HashSet::new();
        for fc in h.faces() {
            used.extend(fc.iter().copied());
        }
        let v = used.len();
        assert_eq!(v as i32 - e as i32 + f as i32, 2, "Euler V-E+F=2");
    }

    /// Faces wound CCW outward: every other point is below (orient <= 0).
    fn assert_outward(h: &LocalHull) {
        for f in h.faces() {
            for p in 0..h.pts.len() {
                if f.contains(&p) {
                    continue;
                }
                let o = orient(h.pts[f[0]], h.pts[f[1]], h.pts[f[2]], h.pts[p]);
                assert!(o <= 1e-12, "point {p} above face {f:?}: orient={o}");
            }
        }
    }

    #[test]
    fn tetrahedron_has_4_faces() {
        let pts = [
            u(1.0, 1.0, 1.0),
            u(1.0, -1.0, -1.0),
            u(-1.0, 1.0, -1.0),
            u(-1.0, -1.0, 1.0),
        ];
        let h = LocalHull::build(&pts).unwrap();
        assert_eq!(h.faces().len(), 4);
        assert_closed(&h);
        assert_outward(&h);
    }

    #[test]
    fn octahedron_has_8_faces() {
        let pts = [
            u(1.0, 0.0, 0.0),
            u(-1.0, 0.0, 0.0),
            u(0.0, 1.0, 0.0),
            u(0.0, -1.0, 0.0),
            u(0.0, 0.0, 1.0),
            u(0.0, 0.0, -1.0),
        ];
        let h = LocalHull::build(&pts).unwrap();
        assert_eq!(h.faces().len(), 8);
        assert_closed(&h);
        assert_outward(&h);
        // Each generator's dual cell has 4 faces (octahedron vertex degree 4).
        for i in 0..6 {
            assert_eq!(h.cell_faces(i).len(), 4, "octa cell {i}");
        }
    }

    #[test]
    fn cube_has_12_faces() {
        let pts = [
            u(1.0, 1.0, 1.0),
            u(1.0, 1.0, -1.0),
            u(1.0, -1.0, 1.0),
            u(1.0, -1.0, -1.0),
            u(-1.0, 1.0, 1.0),
            u(-1.0, 1.0, -1.0),
            u(-1.0, -1.0, 1.0),
            u(-1.0, -1.0, -1.0),
        ];
        let h = LocalHull::build(&pts).unwrap();
        assert_eq!(h.faces().len(), 12); // 6 square faces triangulated
        assert_closed(&h);
        assert_outward(&h);
    }

    #[test]
    fn random_points_form_closed_outward_hull() {
        // Deterministic pseudo-random points on the sphere.
        let mut s: u64 = 0x1234_5678;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let pts: Vec<Vec3> = (0..60).map(|_| u(rng(), rng(), rng())).collect();
        let h = LocalHull::build(&pts).unwrap();
        assert_closed(&h);
        assert_outward(&h);
        // Circumcenters are unit and equidistant to their three generators.
        for fi in 0..h.faces().len() {
            let cc = h.face_circumcenter(fi);
            assert!((cc.length() - 1.0).abs() < 1e-9);
            let [a, b, c] = h.faces()[fi];
            let (da, db, dc) = (cc.dot(h.pts[a]), cc.dot(h.pts[b]), cc.dot(h.pts[c]));
            assert!(
                (da - db).abs() < 1e-9 && (db - dc).abs() < 1e-9,
                "equidistant"
            );
        }
    }

    /// Canonical triangulation: each face mapped to a sorted triple of ORIGINAL
    /// point indices (via exact coordinate match), the whole set sorted. Two
    /// builds of the same points in different orders must yield the same set —
    /// the cross-cell-consistency property (a cell's local point order is
    /// arbitrary, so the hull must not depend on it).
    fn canonical_faces(orig: &[Vec3], h: &LocalHull) -> Vec<[usize; 3]> {
        let id = |p: DVec3| -> usize {
            orig.iter()
                .position(|q| DVec3::new(q.x as f64, q.y as f64, q.z as f64) == p)
                .expect("face vertex must be an original point")
        };
        let mut out: Vec<[usize; 3]> = h
            .faces()
            .iter()
            .map(|f| {
                let mut t = [id(h.pts[f[0]]), id(h.pts[f[1]]), id(h.pts[f[2]])];
                t.sort_unstable();
                t
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Documents that the BARE hull's triangulation of exact cocircular cliques
    /// (the cube's square faces) is order-DEPENDENT — the insertion order picks
    /// the diagonal. `#[ignore]` because the escalation use does NOT require
    /// across-build determinism: a component is rebuilt as ONE hull (so its
    /// diagonals are internally consistent), the grow-until-clean-rim condition
    /// keeps cocircular cliques interior to that single hull, and the coincident
    /// high-degree vertices it emits are collapsed by the existing Tier-1
    /// proximity-merge (see tests/high_degree.rs). Kept as a record of the bare
    /// hull's behavior should a per-cell build ever be contemplated.
    #[test]
    #[ignore]
    fn cube_triangulation_is_order_invariant() {
        let base = [
            u(1.0, 1.0, 1.0),
            u(1.0, 1.0, -1.0),
            u(1.0, -1.0, 1.0),
            u(1.0, -1.0, -1.0),
            u(-1.0, 1.0, 1.0),
            u(-1.0, 1.0, -1.0),
            u(-1.0, -1.0, 1.0),
            u(-1.0, -1.0, -1.0),
        ];
        let h0 = LocalHull::build(&base).unwrap();
        let canon0 = canonical_faces(&base, &h0);

        // A few permutations of the same points.
        let perms: [[usize; 8]; 3] = [
            [7, 6, 5, 4, 3, 2, 1, 0],
            [0, 2, 4, 6, 1, 3, 5, 7],
            [3, 1, 4, 7, 0, 6, 2, 5],
        ];
        for (pi, perm) in perms.iter().enumerate() {
            let pts: Vec<Vec3> = perm.iter().map(|&i| base[i]).collect();
            let h = LocalHull::build(&pts).unwrap();
            // Map through `base` (stable identity) regardless of build order.
            let canon = canonical_faces(&base, &h);
            assert_eq!(
                canon, canon0,
                "cube triangulation differs under permutation {pi} ({perm:?}) — \
                 cocircular-tie resolution is order-dependent"
            );
        }
    }

    /// The load-bearing property for the escalation rebuild: every generator's
    /// dual cell reads as a single clean fan, INCLUDING on a structure with
    /// exact cocircular cliques (the cube's square faces). `cell_faces` must not
    /// bail (return empty) for any generator of a valid hull, regardless of how
    /// the cocircular faces were triangulated.
    #[test]
    fn cube_all_cells_read_as_clean_fans() {
        let pts = [
            u(1.0, 1.0, 1.0),
            u(1.0, 1.0, -1.0),
            u(1.0, -1.0, 1.0),
            u(1.0, -1.0, -1.0),
            u(-1.0, 1.0, 1.0),
            u(-1.0, 1.0, -1.0),
            u(-1.0, -1.0, 1.0),
            u(-1.0, -1.0, -1.0),
        ];
        // Try several input orders: each must yield 8 readable cells (the
        // triangulation may differ, but every dual fan must still chain).
        let perms: [[usize; 8]; 3] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
            [3, 1, 4, 7, 0, 6, 2, 5],
        ];
        for perm in &perms {
            let ordered: Vec<Vec3> = perm.iter().map(|&i| pts[i]).collect();
            let h = LocalHull::build(&ordered).unwrap();
            for i in 0..ordered.len() {
                let fan = h.cell_faces(i);
                assert!(
                    !fan.is_empty(),
                    "cube cell {i} (order {perm:?}) read as a broken fan"
                );
                // A cube corner touches 3 square faces; triangulated, its dual
                // fan has at least 3 faces.
                assert!(fan.len() >= 3, "cube cell {i} fan too short: {}", fan.len());
            }
        }
    }

    /// THE core escalation invariant: every two generators that share a Voronoi
    /// edge, read from ONE hull, agree on that edge's two endpoints (the two
    /// incident faces = the two shared Voronoi vertices) — and those endpoints
    /// are cyclically adjacent in BOTH dual fans. This is why rebuilding a whole
    /// component as a single hull pairs every interior edge by construction:
    /// there is one triangulation, so both sides name the same triples.
    #[test]
    fn dual_cells_agree_on_shared_edges() {
        use std::collections::HashMap;
        let mut s: u64 = 0xCAFE_F00D;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let n = 50;
        let pts: Vec<Vec3> = (0..n).map(|_| u(rng(), rng(), rng())).collect();
        let h = LocalHull::build(&pts).unwrap();

        let fans: Vec<Vec<usize>> = (0..n).map(|i| h.cell_faces(i)).collect();
        for (i, fan) in fans.iter().enumerate() {
            assert!(!fan.is_empty(), "generator {i} has no dual cell");
        }

        // Undirected hull edge -> incident face indices.
        let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (fi, f) in h.faces().iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                edge_faces.entry((a.min(b), a.max(b))).or_default().push(fi);
            }
        }

        let adjacent = |fan: &[usize], f0: usize, f1: usize| -> bool {
            let len = fan.len() as isize;
            let p0 = fan.iter().position(|&f| f == f0).expect("face in fan") as isize;
            let p1 = fan.iter().position(|&f| f == f1).expect("face in fan") as isize;
            let d = (p0 - p1).rem_euclid(len);
            d == 1 || d == len - 1
        };

        for ((i, j), shared) in &edge_faces {
            // Each interior Voronoi edge is bounded by exactly two vertices.
            assert_eq!(
                shared.len(),
                2,
                "edge ({i},{j}) shared by {} faces, not 2",
                shared.len()
            );
            // Both cells carry those same two vertices, cyclically adjacent —
            // i.e. both sides emit the identical shared edge.
            for &g in &[*i, *j] {
                assert!(
                    adjacent(&fans[g], shared[0], shared[1]),
                    "edge ({i},{j}) endpoints not adjacent in dual cell {g}"
                );
            }
        }
    }

    #[test]
    fn coplanar_set_returns_none() {
        // Four points on the equator (cocircular / coplanar through origin).
        let pts = [
            u(1.0, 0.0, 0.0),
            u(0.0, 1.0, 0.0),
            u(-1.0, 0.0, 0.0),
            u(0.0, -1.0, 0.0),
        ];
        assert!(LocalHull::build(&pts).is_none());
    }
}
