//! Delaunay export tests: duality counts, CCW winding, the empty
//! circumcircle property, adjacency consistency, and weld canonicalization.

mod support;

use std::collections::BTreeSet;

use support::points::random_sphere_points;
use voronoi_mesh::{adjacency::NO_NEIGHBOR, compute};

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
