//! Clean, exact high-degree constructions — the regime opposite to `mega`'s
//! clustered noise: generators spread out at normal density whose Voronoi
//! diagrams have genuine degree-4+ vertices (4+ cells meeting at one point).
//!
//! These pass today via Tier-1 reconcile's coincident-vertex merge (the four
//! cells at a degree-4 vertex emit different triple-keys at the same point;
//! proximity-merge collapses them into one degree-4 vertex, leaving the
//! duplicates as benign orphans). The strict validator accepts degree-4+
//! vertices, so the contract is just: strictly valid. These are a regression
//! guard for the high-degeneracy regime and a heavy exerciser of the reconcile
//! at O(n) defects (a structured grid produces thousands; see
//! docs/optimization-ideas.md "structured high-degeneracy inputs").

mod support;
use support::points::cubed_sphere_points;

use s2_voronoi::{compute, validation::validate, UnitVec3};

fn u(x: f32, y: f32, z: f32) -> UnitVec3 {
    let l = (x * x + y * y + z * z).sqrt();
    UnitVec3::new(x / l, y / l, z / l)
}

fn assert_valid(name: &str, pts: &[UnitVec3]) {
    let diagram = compute(pts).unwrap_or_else(|e| panic!("{name}: compute failed: {e:?}"));
    let report = validate(&diagram);
    assert!(
        report.is_strictly_valid(),
        "{name} ({} gens): not strictly valid — {}",
        pts.len(),
        report.headline()
    );
}

#[test]
fn octahedron_degree3_control() {
    // 6 generators -> Voronoi is the cube: 8 degree-3 vertices. Control case.
    let pts = [
        u(1.0, 0.0, 0.0),
        u(-1.0, 0.0, 0.0),
        u(0.0, 1.0, 0.0),
        u(0.0, -1.0, 0.0),
        u(0.0, 0.0, 1.0),
        u(0.0, 0.0, -1.0),
    ];
    assert_valid("octahedron", &pts);
}

#[test]
fn cube_degree4() {
    // 8 generators -> Voronoi has 6 face-center vertices, each exactly DEGREE 4.
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
    assert_valid("cube", &pts);
}

#[test]
fn cuboctahedron_mixed_degree() {
    // 12 generators (edge midpoints of the cube) -> mixed degree-3 / degree-4.
    let pts = [
        u(1.0, 1.0, 0.0),
        u(1.0, -1.0, 0.0),
        u(-1.0, 1.0, 0.0),
        u(-1.0, -1.0, 0.0),
        u(1.0, 0.0, 1.0),
        u(1.0, 0.0, -1.0),
        u(-1.0, 0.0, 1.0),
        u(-1.0, 0.0, -1.0),
        u(0.0, 1.0, 1.0),
        u(0.0, 1.0, -1.0),
        u(0.0, -1.0, 1.0),
        u(0.0, -1.0, -1.0),
    ];
    assert_valid("cuboctahedron", &pts);
}

#[test]
fn cubed_sphere_grid_degree4_at_scale() {
    // Structured quad grid: O(n) degree-4 vertices at normal density. Produces
    // thousands of reconcile defects yet must stay strictly valid (the heavy
    // exerciser of the coincident-vertex merge path). Kept modest for CI; the
    // full size sweep is manual (docs/optimization-ideas.md).
    let pts = cubed_sphere_points(6 * 24 * 24, 0); // ~3.5k generators
    assert_valid("cubed_sphere_k24", &pts);
}
