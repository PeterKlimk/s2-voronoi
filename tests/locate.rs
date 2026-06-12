//! Point-location API tests: locators must agree with brute-force nearest
//! generator on all three geometries (ties tolerated by distance), handle
//! out-of-rect and wrapping queries, and resolve welded twins to canonical
//! cells.

mod support;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{compute, compute_plane, compute_plane_periodic, PlanePoint, PlaneRect, UnitVec3};
use support::points::random_sphere_points;

/// Distance-comparison slack: locator and brute force may round differently
/// (normalized vs raw coordinates), so a tie can resolve to a different
/// index whose distance differs by a few ULPs.
const REL_EPS: f64 = 1e-5;

fn dot(a: UnitVec3, b: UnitVec3) -> f64 {
    a.x as f64 * b.x as f64 + a.y as f64 * b.y as f64 + a.z as f64 * b.z as f64
}

fn brute_nearest_sphere(generators: &[UnitVec3], q: UnitVec3) -> (usize, f64) {
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, g) in generators.iter().enumerate() {
        let d = dot(*g, q);
        if d > best.1 {
            best = (i, d);
        }
    }
    best
}

fn dist_sq(a: PlanePoint, b: PlanePoint) -> f64 {
    let dx = a.x as f64 - b.x as f64;
    let dy = a.y as f64 - b.y as f64;
    dx * dx + dy * dy
}

fn min_image_dist_sq(a: PlanePoint, b: PlanePoint, rect: &PlaneRect) -> f64 {
    let (px, py) = (rect.width() as f64, rect.height() as f64);
    let mut dx = (a.x as f64 - b.x as f64).rem_euclid(px);
    if dx > px / 2.0 {
        dx = px - dx;
    }
    let mut dy = (a.y as f64 - b.y as f64).rem_euclid(py);
    if dy > py / 2.0 {
        dy = py - dy;
    }
    dx * dx + dy * dy
}

#[test]
fn sphere_locate_matches_brute_force() {
    let points = random_sphere_points(700, 11);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    let queries = random_sphere_points(3000, 12);
    for q in &queries {
        let found = locator.locate(q);
        let (brute, brute_dot) = brute_nearest_sphere(diagram.generators(), *q);
        if found != brute {
            let found_dot = dot(diagram.generator(found), *q);
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
fn plane_locate_matches_brute_force() {
    // Non-unit, offset rect to exercise the normalization transform.
    let rect = PlaneRect::new(PlanePoint::new(-2.0, 1.0), PlanePoint::new(3.0, 4.0));
    let points = random_rect_points(600, &rect, 41);
    let diagram = compute_plane(&points, rect).unwrap();
    let mut locator = diagram.build_locator();

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for k in 0..3000 {
        // Every third query lands outside the rect: still locates to the
        // globally nearest generator.
        let (qx, qy) = if k % 3 == 0 {
            (rng.gen_range(-6.0..8.0), rng.gen_range(-3.0..9.0))
        } else {
            (
                rng.gen_range(rect.min.x..rect.max.x),
                rng.gen_range(rect.min.y..rect.max.y),
            )
        };
        let q = PlanePoint::new(qx, qy);
        let found = locator.locate(qx, qy);
        let mut brute = (0usize, f64::INFINITY);
        for (i, g) in diagram.generators().iter().enumerate() {
            let d = dist_sq(*g, q);
            if d < brute.1 {
                brute = (i, d);
            }
        }
        if found != brute.0 {
            let found_d = dist_sq(diagram.generator(found), q);
            assert!(
                found_d <= brute.1 * (1.0 + REL_EPS) + 1e-12,
                "locate returned {found} (d2 {found_d}), brute force {} (d2 {})",
                brute.0,
                brute.1
            );
        }
    }
}

#[test]
fn plane_locate_at_generators_is_identity() {
    let rect = PlaneRect::unit();
    let points = random_rect_points(400, &rect, 51);
    let diagram = compute_plane(&points, rect).unwrap();
    let mut locator = diagram.build_locator();
    for i in 0..diagram.num_cells() {
        let g = diagram.generator(i);
        assert_eq!(locator.locate(g.x, g.y), diagram.canonical_cell_index(i));
    }
}

#[test]
fn plane_locate_welded_returns_canonical() {
    let rect = PlaneRect::unit();
    let mut points = random_rect_points(300, &rect, 61);
    for i in [3usize, 150, 299] {
        let p = points[i];
        points.push(p);
    }
    let diagram = compute_plane(&points, rect).unwrap();
    assert!(diagram.weld_map().is_some(), "duplicates should weld");
    let mut locator = diagram.build_locator();
    for (i, p) in points.iter().enumerate().skip(300) {
        let got = locator.locate(p.x, p.y);
        assert_eq!(got, diagram.canonical_cell_index(i));
    }
}

#[test]
fn periodic_locate_matches_brute_force_with_wrapping() {
    // Anisotropic torus.
    let rect = PlaneRect::new(PlanePoint::new(0.0, 0.0), PlanePoint::new(2.0, 1.0));
    let points = random_rect_points(500, &rect, 71);
    let diagram = compute_plane_periodic(&points, rect).unwrap();
    let mut locator = diagram.build_locator();

    let mut rng = ChaCha8Rng::seed_from_u64(72);
    for k in 0..3000 {
        // Mix of in-domain queries, seam-hugging queries, and far
        // out-of-domain queries (must wrap onto the torus).
        let (qx, qy) = match k % 3 {
            0 => (rng.gen_range(0.0..2.0f32), rng.gen_range(0.0..1.0f32)),
            1 => (
                rng.gen_range(-1e-3..1e-3f32),
                rng.gen_range(0.999..1.001f32),
            ),
            _ => (rng.gen_range(-9.0..9.0f32), rng.gen_range(-7.0..7.0f32)),
        };
        let q = PlanePoint::new(qx, qy);
        let found = locator.locate(qx, qy);
        let mut brute = (0usize, f64::INFINITY);
        for (i, g) in diagram.generators().iter().enumerate() {
            let d = min_image_dist_sq(*g, q, &rect);
            if d < brute.1 {
                brute = (i, d);
            }
        }
        if found != brute.0 {
            let found_d = min_image_dist_sq(diagram.generator(found), q, &rect);
            assert!(
                found_d <= brute.1 * (1.0 + REL_EPS) + 1e-12,
                "locate returned {found} (d2 {found_d}), brute force {} (d2 {}) at ({qx}, {qy})",
                brute.0,
                brute.1
            );
        }
    }
}

#[test]
fn periodic_locate_at_generators_is_identity() {
    let rect = PlaneRect::unit();
    let points = random_rect_points(400, &rect, 81);
    let diagram = compute_plane_periodic(&points, rect).unwrap();
    let mut locator = diagram.build_locator();
    for i in 0..diagram.num_cells() {
        let g = diagram.generator(i);
        assert_eq!(locator.locate(g.x, g.y), diagram.canonical_cell_index(i));
        // The wrapped image one period away locates to the same cell.
        assert_eq!(
            locator.locate(g.x + rect.width(), g.y - rect.height()),
            diagram.canonical_cell_index(i)
        );
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
        let (brute, brute_dot) = brute_nearest_sphere(diagram.generators(), *q);
        if found != brute {
            let found_dot = dot(diagram.generator(found), *q);
            assert!(found_dot >= brute_dot - REL_EPS);
        }
    }

    let rect = PlaneRect::unit();
    let pts = vec![
        PlanePoint::new(0.25, 0.25),
        PlanePoint::new(0.75, 0.25),
        PlanePoint::new(0.5, 0.75),
    ];
    let diagram = compute_plane(&pts, rect).unwrap();
    let mut locator = diagram.build_locator();
    assert_eq!(locator.locate(0.2, 0.2), 0);
    assert_eq!(locator.locate(0.8, 0.2), 1);
    assert_eq!(locator.locate(0.5, 0.9), 2);
}

#[test]
fn locate_many_matches_locate() {
    // Sphere.
    let points = random_sphere_points(400, 101);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    let queries = random_sphere_points(500, 102);
    let batch = locator.locate_many(&queries);
    for (q, &got) in queries.iter().zip(&batch) {
        assert_eq!(got, locator.locate(q));
    }

    // Bounded plane (queries inside and outside the rect).
    let rect = PlaneRect::new(PlanePoint::new(-1.0, 0.0), PlanePoint::new(2.0, 2.0));
    let points = random_rect_points(300, &rect, 103);
    let diagram = compute_plane(&points, rect).unwrap();
    let mut locator = diagram.build_locator();
    let mut rng = ChaCha8Rng::seed_from_u64(104);
    let queries: Vec<PlanePoint> = (0..500)
        .map(|_| PlanePoint::new(rng.gen_range(-3.0..4.0), rng.gen_range(-2.0..4.0)))
        .collect();
    let batch = locator.locate_many(&queries);
    for (q, &got) in queries.iter().zip(&batch) {
        assert_eq!(got, locator.locate(q.x, q.y));
    }

    // Periodic (wrapping queries).
    let rect = PlaneRect::unit();
    let points = random_rect_points(300, &rect, 105);
    let diagram = compute_plane_periodic(&points, rect).unwrap();
    let mut locator = diagram.build_locator();
    let queries: Vec<PlanePoint> = (0..500)
        .map(|_| PlanePoint::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0)))
        .collect();
    let batch = locator.locate_many(&queries);
    for (q, &got) in queries.iter().zip(&batch) {
        assert_eq!(got, locator.locate(q.x, q.y));
    }
}
