//! Delaunay export tests: duality counts, CCW winding, the empty
//! circumcircle property, adjacency consistency, weld canonicalization,
//! and degenerate (cocircular) inputs.

mod support;

use std::collections::BTreeSet;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{
    adjacency::NO_NEIGHBOR, compute, compute_plane, compute_plane_periodic, PlanePoint, PlaneRect,
};
use support::points::random_sphere_points;

fn edge_set(triangles: &[[u32; 3]]) -> BTreeSet<(u32, u32)> {
    let mut set = BTreeSet::new();
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            set.insert((a.min(b), a.max(b)));
        }
    }
    set
}

fn adjacency_edge_set(
    num_cells: usize,
    weld_map: Option<&[u32]>,
    neighbors_of: impl Fn(usize) -> Vec<u32>,
) -> BTreeSet<(u32, u32)> {
    let mut set = BTreeSet::new();
    for i in 0..num_cells {
        if weld_map.is_some_and(|m| m[i] as usize != i) {
            continue;
        }
        for &j in neighbors_of(i).iter() {
            if j == NO_NEIGHBOR {
                continue;
            }
            set.insert(((i as u32).min(j), (i as u32).max(j)));
        }
    }
    set
}

#[test]
fn sphere_delaunay_is_complete_dual() {
    let mut points = random_sphere_points(500, 5005);
    points.push(points[42]); // welded twin: must not appear in any triangle

    let diagram = compute(&points).unwrap();
    let triangles = diagram.delaunay_triangles();

    // Euler: a triangulation of the sphere over c vertices has 2c - 4 faces.
    let c = 500;
    assert_eq!(triangles.len(), 2 * c - 4);

    let weld = diagram.weld_map().unwrap();
    for t in &triangles {
        // Canonical indices only.
        for &i in t {
            assert_eq!(weld[i as usize], i, "non-canonical index in triangle");
        }
        // CCW viewed from outside: positive scalar triple product.
        let g = |i: u32| {
            let p = diagram.generator(i as usize);
            glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64)
        };
        let det = g(t[0]).cross(g(t[1])).dot(g(t[2]));
        assert!(det > 0.0, "triangle {t:?} not CCW (det {det})");
    }

    // Triangle edges are exactly the Voronoi adjacency pairs.
    let adjacency = diagram.build_adjacency();
    let adj_edges = adjacency_edge_set(diagram.num_cells(), diagram.weld_map(), |i| {
        adjacency.neighbors_of(i).to_vec()
    });
    assert_eq!(edge_set(&triangles), adj_edges);
}

#[test]
fn sphere_delaunay_empty_circumcircle() {
    let points = random_sphere_points(120, 6006);
    let diagram = compute(&points).unwrap();
    let g = |i: u32| {
        let p = diagram.generator(i as usize);
        glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64)
    };
    for t in diagram.delaunay_triangles() {
        let (a, b, c) = (g(t[0]), g(t[1]), g(t[2]));
        let mut cc = (b - a).cross(c - a).normalize();
        if cc.dot(a) < 0.0 {
            cc = -cc;
        }
        let r_dot = cc.dot(a);
        // Candidates come from the diagram's own generators: the pipeline
        // canonicalizes inputs at entry (P5 stage 0), so the raw input
        // points differ from the solved generator set by up to ~1 ulp.
        for i in 0..diagram.num_cells() {
            assert!(
                cc.dot(g(i as u32)) <= r_dot + 1e-9,
                "generator {i} strictly inside circumcircle of {t:?}"
            );
        }
    }
}

fn random_rect_points(n: usize, rect: &PlaneRect, seed: u64) -> Vec<PlanePoint> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            PlanePoint::new(
                rng.gen_range(rect.min.x..rect.max.x),
                rng.gen_range(rect.min.y..rect.max.y),
            )
        })
        .collect()
}

#[test]
fn plane_bounded_delaunay_subset_dual() {
    let rect = PlaneRect::new(PlanePoint::new(-3.0, 2.0), PlanePoint::new(5.0, 7.0));
    let mut points = random_rect_points(400, &rect, 7007);
    points.push(points[7]); // welded twin

    let diagram = compute_plane(&points, rect).unwrap();
    let triangles = diagram.delaunay_triangles();
    assert!(!triangles.is_empty());

    let weld = diagram.weld_map().unwrap();
    let gp = |i: u32| {
        let p = diagram.generator(i as usize);
        glam::DVec2::new(p.x as f64, p.y as f64)
    };
    for t in &triangles {
        for &i in t {
            assert_eq!(weld[i as usize], i, "non-canonical index in triangle");
        }
        // CCW in standard axes.
        let (a, b, c) = (gp(t[0]), gp(t[1]), gp(t[2]));
        let cross = (b - a).perp_dot(c - a);
        assert!(cross > 0.0, "triangle {t:?} not CCW (cross {cross})");

        // Empty circumcircle among ALL generators (the bounded subset is
        // still a set of true Delaunay triangles).
        let d2 = |p: glam::DVec2, q: glam::DVec2| (p - q).length_squared();
        let cc = circumcenter(a, b, c);
        let r2 = d2(cc, a);
        for (i, p) in points.iter().enumerate() {
            let other = glam::DVec2::new(p.x as f64, p.y as f64);
            assert!(
                d2(cc, other) >= r2 * (1.0 - 1e-9) - 1e-12,
                "generator {i} strictly inside circumcircle of {t:?}"
            );
        }
    }

    // Triangle edges are a subset of adjacency pairs (an edge whose dual
    // Voronoi edge has both endpoints on walls has no triangle).
    let adjacency = diagram.build_adjacency();
    let adj_edges = adjacency_edge_set(diagram.num_cells(), diagram.weld_map(), |i| {
        adjacency.neighbors_of(i).to_vec()
    });
    assert!(edge_set(&triangles).is_subset(&adj_edges));
}

fn circumcenter(a: glam::DVec2, b: glam::DVec2, c: glam::DVec2) -> glam::DVec2 {
    let (bx, by) = (b.x - a.x, b.y - a.y);
    let (cx, cy) = (c.x - a.x, c.y - a.y);
    let d = 2.0 * (bx * cy - by * cx);
    let ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d;
    let uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d;
    glam::DVec2::new(a.x + ux, a.y + uy)
}

#[test]
fn periodic_delaunay_is_complete_torus_triangulation() {
    let rect = PlaneRect::new(PlanePoint::new(0.0, 0.0), PlanePoint::new(2.0, 1.0));
    let points = random_rect_points(300, &rect, 8008);
    let diagram = compute_plane_periodic(&points, rect).unwrap();
    let triangles = diagram.delaunay_triangles();

    // Torus Euler: F = 2V triangles, E = 3V edges.
    assert_eq!(triangles.len(), 2 * 300);
    let edges = edge_set(&triangles);
    assert_eq!(edges.len(), 3 * 300);

    // CCW with minimum-image displacements (triangle corners are nearest
    // images of each other under the half-period guard).
    let (px, py) = (rect.width() as f64, rect.height() as f64);
    let wrap = |d: f64, p: f64| {
        let w = d.rem_euclid(p);
        if w > p / 2.0 {
            w - p
        } else {
            w
        }
    };
    let gp = |i: u32| {
        let p = diagram.generator(i as usize);
        (p.x as f64, p.y as f64)
    };
    for t in &triangles {
        let (ax, ay) = gp(t[0]);
        let (bx, by) = gp(t[1]);
        let (cx, cy) = gp(t[2]);
        let (ux, uy) = (wrap(bx - ax, px), wrap(by - ay, py));
        let (vx, vy) = (wrap(cx - ax, px), wrap(cy - ay, py));
        let cross = ux * vy - uy * vx;
        assert!(cross > 0.0, "torus triangle {t:?} not CCW (cross {cross})");
    }

    // Edge set matches the toroidal adjacency exactly.
    let adjacency = diagram.build_adjacency();
    assert!(adjacency.is_complete());
    let adj_edges = adjacency_edge_set(diagram.num_cells(), diagram.weld_map(), |i| {
        adjacency.neighbors_of(i).to_vec()
    });
    assert_eq!(edges, adj_edges);
}

#[test]
fn degenerate_lattice_does_not_panic_and_stays_consistent() {
    // Every interior vertex is a 4-cocircular tie: exercise the >3-incidence
    // fan path (when reconciliation merges) and tie-broken triplets alike.
    let mut points = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            points.push([(i as f32 + 0.5) / 8.0, (j as f32 + 0.5) / 8.0]);
        }
    }
    let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
    let triangles = diagram.delaunay_triangles();
    assert!(!triangles.is_empty());

    let mut seen = BTreeSet::new();
    for t in &triangles {
        for &i in t {
            assert!((i as usize) < diagram.num_cells());
        }
        let mut key = *t;
        key.sort_unstable();
        assert!(seen.insert(key), "duplicate triangle {t:?}");
        // Ties may produce zero-area slivers; winding must never be CW.
        let gp = |i: u32| {
            let p = diagram.generator(i as usize);
            glam::DVec2::new(p.x as f64, p.y as f64)
        };
        let cross = (gp(t[1]) - gp(t[0])).perp_dot(gp(t[2]) - gp(t[0]));
        assert!(cross >= -1e-12, "lattice triangle {t:?} wound CW");
    }
}
