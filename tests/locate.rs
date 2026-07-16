//! Point-location API tests: the locator must agree with brute-force
//! nearest generator (ties tolerated by distance) and resolve welded twins
//! to canonical cells.

mod support;

use support::points::random_sphere_points;
use voronoi_mesh::{compute, SpherePoint, SphereQueryError, UnitVec3, UnitVec3Like};

/// Distance-comparison slack: locator canonicalization and the raw-query
/// brute-force oracle may round differently, so a tie can resolve to a
/// different index whose distance differs by a few ULPs.
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
        let found = locator.locate(q).unwrap();
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
        assert_eq!(locator.locate(&g).unwrap(), diagram.canonical_cell_index(i));
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
        let got = locator.locate(p).unwrap();
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
        let found = locator.locate(q).unwrap();
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
    let batch = locator.locate_many(&queries).unwrap();
    for (q, &got) in queries.iter().zip(&batch) {
        assert_eq!(got, locator.locate(q).unwrap());
    }
}

#[test]
fn scaled_query_uses_the_same_normalized_ranking_and_bound() {
    let points = random_sphere_points(700, 11);
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    // Before locator normalization, the raw candidate dot was multiplied by
    // 16 while the unseen-cell bound was normalized. That made the search
    // stop in the first occupied shell and returned generator 211 rather
    // than the actual nearest generator 576 for this query.
    let query = UnitVec3::new(-0.022_981_111, 0.561_943_7, 0.826_856_14);
    let scaled = UnitVec3::new(query.x * 16.0, query.y * 16.0, query.z * 16.0);
    let unit_result = locator.locate(&query).unwrap();
    let scaled_result = locator.locate(&scaled).unwrap();
    let (brute, _) = brute_nearest_sphere(diagram.generators(), &query);
    assert_eq!(unit_result, brute);
    assert_eq!(scaled_result, unit_result);
}

#[test]
fn locator_rejects_directionless_and_non_finite_queries() {
    let diagram = compute(&random_sphere_points(100, 111)).unwrap();
    let mut locator = diagram.build_locator();

    assert_eq!(
        locator.locate(&[0.0f32; 3]),
        Err(SphereQueryError::Directionless)
    );
    assert_eq!(
        locator.locate(&[1.0, f32::NAN, 0.0]),
        Err(SphereQueryError::NonFinite { component: 1 })
    );
    assert_eq!(
        locator.locate(&[1.0, 0.0, f32::INFINITY]),
        Err(SphereQueryError::NonFinite { component: 2 })
    );
}

#[test]
fn locate_many_reports_the_lowest_invalid_query_index() {
    let diagram = compute(&random_sphere_points(100, 112)).unwrap();
    let locator = diagram.build_locator();
    let mut queries = random_sphere_points(20, 113);
    queries[11].z = f32::NAN;
    queries[4] = UnitVec3::new(0.0, 0.0, 0.0);

    let error = locator.locate_many(&queries).unwrap_err();
    assert_eq!(error.query_index(), 4);
    assert_eq!(error.query_error(), SphereQueryError::Directionless);
}
