//! Planar coincidence probes: the plane's analog of the sphere's
//! `coincidence_probes.rs`, asserting strict validity under adversarial
//! near-coincidence with the production weld in place.
//!
//! History (2026-06-12 margin probes, raw pipeline without a radius weld):
//! PAIRS resolve at any distinct-f32 separation — including straddling grid
//! walls and at rect corners; the sphere's chart-divergence failure class
//! has no planar analog. CLUSTERS (k >= 3) within ~1 ulp of unit scale
//! produce invalid topology (degenerate cells, overused edges): invalid at
//! min-separation 3e-8, valid from 6e-8 in every configuration, plus the
//! subnormal-separation regime near the origin. Hence PLANE_WELD_DIST =
//! 1e-6 (~30x margin); these scenarios now exercise the weld + pipeline
//! end to end.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{compute_plane, validation, PlaneRect};

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn uniform(n: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut r = rng(seed);
    (0..n)
        .map(|_| [r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)])
        .collect()
}

fn assert_strict(points: &[[f32; 2]], name: &str) {
    let diagram = compute_plane(points, PlaneRect::unit())
        .unwrap_or_else(|e| panic!("{name}: compute_plane failed: {e}"));
    let report = validation::validate_plane(&diagram);
    assert!(
        report.is_strictly_valid(),
        "{name}: strict validation failed: {report:#?}"
    );
}

/// Nudge a coordinate by `ulps` representable steps (sign of `ulps` picks
/// the direction); exact f32 lattice steps, never a no-op for ulps != 0.
fn nudge(x: f32, ulps: i32) -> f32 {
    let mut v = x;
    for _ in 0..ulps.abs() {
        v = if ulps > 0 {
            f32::from_bits(v.to_bits() + 1)
        } else if v == 0.0 {
            -f32::from_bits(1)
        } else {
            f32::from_bits(v.to_bits() - 1)
        };
    }
    v
}

#[test]
fn plane_probe_ulp_pairs_generic() {
    // 1-ulp twins at generic positions (within the weld radius: welded).
    let mut points = uniform(10_000, 11);
    for i in 0..200 {
        let base = points[i * 37];
        points.push([nudge(base[0], 1), base[1]]);
    }
    assert_strict(&points, "ulp_pairs_generic");
}

#[test]
fn plane_probe_ulp_clusters() {
    // k = 3..9 clusters within a few ulps: the sphere's hard-fail regime.
    let mut points = uniform(5_000, 13);
    let mut r = rng(17);
    for k in 3..=9 {
        for _ in 0..8 {
            let cx = r.gen_range(0.1f32..0.9);
            let cy = r.gen_range(0.1f32..0.9);
            for j in 0..k {
                points.push([nudge(cx, j % 3 - 1), nudge(cy, j / 3 - 1)]);
            }
        }
    }
    // The whole cluster welds to one generator.
    assert_strict(&points, "ulp_clusters");
}

#[test]
fn plane_probe_ulp_pairs_on_grid_walls() {
    // The plane's only classification branches are grid-cell walls: twins
    // straddling a wall land in different cells. Walls depend on the policy
    // resolution, so probe simple fractions that are walls across many
    // resolutions, plus their ulp neighbors.
    let mut points = uniform(20_000, 19);
    for &wall in &[0.25f32, 0.5, 0.75, 0.125, 0.375] {
        for i in 0..20 {
            let y = 0.05 + 0.045 * i as f32;
            points.push([nudge(wall, -1), y]);
            points.push([wall, nudge(y, 1)]);
            points.push([nudge(wall, 1), y]);
            // And the transposed configuration on a horizontal wall.
            points.push([y, nudge(wall, -1)]);
            points.push([nudge(y, 1), wall]);
            points.push([y, nudge(wall, 1)]);
        }
    }
    assert_strict(&points, "ulp_pairs_on_walls");
}

#[test]
fn plane_probe_ulp_clusters_at_rect_boundary() {
    // Clusters hugging the rect corners and edges: wall half-planes meet
    // epsilon-scale bisectors.
    let mut points = uniform(5_000, 23);
    for &(cx, cy) in &[
        (0.0f32, 0.0f32),
        (1.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (1.0, 0.5),
        (0.5, 1.0),
    ] {
        for j in 0..5i32 {
            // One-sided nudges keep coordinates inside [0, 1].
            let dir_x = if cx == 0.0 { 1 } else { -1 };
            let dir_y = if cy == 0.0 { 1 } else { -1 };
            points.push([nudge(cx, dir_x * j), nudge(cy, dir_y * (j % 3))]);
        }
    }
    assert_strict(&points, "ulp_clusters_rect_boundary");
}

#[test]
fn plane_probe_subnormal_separations_near_origin() {
    // Near the origin the f32 lattice is astronomically fine: distinct
    // points can be ~1e-40 apart in normalized units. The bisector stays
    // well-formed (no f64 underflow for any distinct f32 pair); the cells
    // between them are sub-resolution slivers whose topology must still be
    // strictly valid.
    let mut points = uniform(2_000, 29);
    for j in 1..30 {
        points.push([f32::from_bits(j), f32::from_bits(j * 7 % 31)]);
    }
    points.push([0.0, 0.0]);
    assert_strict(&points, "subnormal_near_origin");
}

#[test]
fn plane_probe_mixed_scale_weld_interaction() {
    // Exact duplicates and 1-ulp neighbors of the same base points: all
    // within the weld radius; the weld map and cells must stay consistent.
    let mut points = uniform(5_000, 31);
    for i in 0..50 {
        let base = points[i * 41];
        points.push(base); // welds
        points.push([nudge(base[0], 1), base[1]]); // stays distinct
        points.push([base[0], nudge(base[1], -1)]); // stays distinct
    }
    assert_strict(&points, "mixed_weld_interaction");
}

/// Heavier randomized sweep over cluster sizes and positions; run with
/// `--ignored` (scheduled CI).
#[test]
#[ignore]
fn plane_probe_randomized_cluster_sweep() {
    let mut r = rng(101);
    for round in 0..20 {
        let mut points = uniform(50_000, 1000 + round);
        for _ in 0..200 {
            let k = r.gen_range(2..=9);
            let cx = r.gen_range(0.0f32..1.0);
            let cy = r.gen_range(0.0f32..1.0);
            for j in 0..k {
                points.push([
                    nudge(cx, r.gen_range(-2..=2)),
                    nudge(cy, r.gen_range(-2..=2)),
                ]);
                let _ = j;
            }
        }
        assert_strict(&points, &format!("random_sweep_round_{round}"));
    }
}
