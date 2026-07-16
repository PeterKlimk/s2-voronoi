//! Point-location API tests: the locator must agree with brute-force
//! nearest generator (ties tolerated by distance) and resolve welded twins
//! to canonical cells.

mod support;

use support::points::random_sphere_points;
use voronoi_mesh::{compute, SpherePoint, UnitVec3, UnitVec3Like};

/// Distance-comparison slack: locator and brute force may round differently
/// (normalized vs raw coordinates), so a tie can resolve to a different
/// index whose distance differs by a few ULPs.
const REL_EPS: f64 = 1e-5;

fn dot<A: UnitVec3Like, B: UnitVec3Like>(a: &A, b: &B) -> f64 {
    a.x() as f64 * b.x() as f64 + a.y() as f64 * b.y() as f64 + a.z() as f64 * b.z() as f64
}

fn brute_nearest_sphere(generators: &[SpherePoint], q: &UnitVec3) -> (usize, f64) {
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, g) in generators.iter().enumerate() {
        let d = dot(g, q);
        if d > best.1 {
            best = (i, d);
        }
    }
    best
}

#[test]
fn sphere_locate_matches_brute_force() {
    let points = random_sphere_points(700, 11);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    let queries = random_sphere_points(3000, 12);
    for q in &queries {
        let found = locator.locate(q);
        let (brute, brute_dot) = brute_nearest_sphere(diagram.generators(), q);
        if found != brute {
            let found_dot = dot(&diagram.generator(found), q);
            assert!(
                found_dot >= brute_dot - REL_EPS,
                "locate returned {found} (dot {found_dot}), brute force {brute} (dot {brute_dot})"
            );
        }
    }
}

#[test]
fn sphere_locate_at_generators_is_identity() {
    let points = random_sphere_points(500, 21);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    for i in 0..diagram.num_cells() {
        let g = diagram.generator(i);
        assert_eq!(locator.locate(&g), diagram.canonical_cell_index(i));
    }
}

#[test]
fn sphere_locate_welded_returns_canonical() {
    let mut points = random_sphere_points(300, 31);
    // Exact duplicates of a few generators; the pipeline welds them.
    for i in [5usize, 77, 123] {
        let p = points[i];
        points.push(p);
    }
    let diagram = compute(&points).unwrap();
    assert!(diagram.weld_map().is_some(), "duplicates should weld");
    let mut locator = diagram.build_locator();
    for (i, p) in points.iter().enumerate().skip(300) {
        let got = locator.locate(p);
        assert_eq!(got, diagram.canonical_cell_index(i));
        assert_eq!(got, diagram.canonical_cell_index(got), "must be canonical");
    }
}

#[test]
fn locate_small_diagrams() {
    // Tiny inputs stress the grid's minimum-resolution paths. (Sphere
    // compute needs enough points that every cell bounds; 16 random
    // points is comfortably above that while far below one per grid cell.)
    let points = random_sphere_points(16, 91);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    let queries = random_sphere_points(200, 92);
    for q in &queries {
        let found = locator.locate(q);
        let (brute, brute_dot) = brute_nearest_sphere(diagram.generators(), q);
        if found != brute {
            let found_dot = dot(&diagram.generator(found), q);
            assert!(found_dot >= brute_dot - REL_EPS);
        }
    }
}

#[test]
fn locate_many_matches_locate() {
    let points = random_sphere_points(400, 101);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    let queries = random_sphere_points(500, 102);
    let batch = locator.locate_many(&queries);
    for (q, &got) in queries.iter().zip(&batch) {
        assert_eq!(got, locator.locate(q));
    }
}
