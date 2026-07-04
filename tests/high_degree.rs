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
//! at O(n) defects (a structured grid produces thousands).

mod support;
use support::points::cubed_sphere_points;

use std::f32::consts::PI;
use voronoi_mesh::{compute, compute_with_report, validation::validate, UnitVec3, VoronoiConfig};

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
    // full size sweep is manual.
    let pts = cubed_sphere_points(6 * 24 * 24, 0); // ~3.5k generators
    assert_valid("cubed_sphere_k24", &pts);
}

/// `n` generators equally spaced on the latitude-45°N small circle (exactly
/// cocircular) plus a south-pole apex: the true Voronoi has a single degree-`n`
/// vertex at the north pole. Exact cocircular clique fixture.
fn cocircular_pyramid(n: usize) -> Vec<UnitVec3> {
    let r = (0.5f32).sqrt(); // latitude 45N
    let mut pts: Vec<UnitVec3> = (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f32 / n as f32;
            u(r * t.cos(), r * t.sin(), r)
        })
        .collect();
    pts.push(u(0.0, 0.0, -1.0));
    pts
}

#[test]
fn pentagonal_pyramid_cocircular5() {
    // Exact 5-cocircular clique. Resolves cleanly (the odd clique SPLITS into
    // degree-3 vertices rather than merging like the cube's degree-4) — either
    // way the contract is strict validity.
    assert_valid("pentagonal_pyramid", &cocircular_pyramid(5));
}

#[test]
fn hexagonal_pyramid_cocircular6() {
    // Exact 6-cocircular clique.
    assert_valid("hexagonal_pyramid", &cocircular_pyramid(6));
}

/// Manual scaling probe (`#[ignore]`): how the structured-grid reconcile load and
/// validity behave as the grid refines (spacing shrinks → conditioning degrades).
/// Documents where, if anywhere, a clean grid stops being handled by Tier-1 and
/// starts leaving a residual. Run: `cargo test --release --test high_degree --
/// --ignored cubed_sphere_scaling_probe --nocapture`.
#[test]
#[ignore = "manual scaling characterization"]
fn cubed_sphere_scaling_probe() {
    for k in [20usize, 40, 80, 120] {
        let pts = cubed_sphere_points(6 * k * k, 0);
        let out = compute_with_report(&pts, VoronoiConfig::default())
            .unwrap_or_else(|e| panic!("k={k}: {e:?}"));
        let post = out.report.post_repair_unpaired_edges.len();
        let v = out.report.preferred_validation();
        let [_, _, _, d3, d4] = v.degree_counts;
        eprintln!(
            "k={k} n={}: defects={} post_repair={post} valid={} deg[d3={d3} d4+={d4}]",
            pts.len(),
            out.report.unresolved_edge_pairs.len(),
            v.is_strictly_valid(),
        );
    }
}
