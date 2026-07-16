//! Exhaustive-within-a-declared-range intrinsic geometry checks.
//!
//! These tests do not compare against another Voronoi implementation. They
//! ask whether the returned complex actually satisfies the defining nearest-
//! site and bisector conditions for its own normalized canonical generators.

mod support;

use std::collections::BTreeMap;

use glam::DVec3;
use support::points::{
    bimodal_density_points, clustered_cap_points, cube_vertex_stress_points,
    fibonacci_sphere_points, hemisphere_points, near_cocircular_stress_points,
    random_sphere_points,
};
use voronoi_mesh::{
    compute_with_report, validation::validate, SphericalVoronoi, UnitVec3, UnitVec3Like,
    VoronoiConfig,
};

// Small diagrams can contain nearly antipodal edge endpoints, where rounding
// stored f32 vertices is amplified by normalized chord sampling. This remains
// two orders below a visible 1e-3-radian geometric defect and far below the
// former AUD-002 corruption.
const MAX_INTRINSIC_ANGULAR_ERROR: f64 = 1.0e-5;

#[derive(Debug, Default)]
struct IntrinsicGeometry {
    vertex_owner_samples: usize,
    edge_samples: usize,
    interior_samples: usize,
    max_ownership_violation_rad: f64,
    max_vertex_cross_track_rad: f64,
    max_edge_cross_track_rad: f64,
}

impl IntrinsicGeometry {
    fn max_error(&self) -> f64 {
        self.max_ownership_violation_rad
            .max(self.max_vertex_cross_track_rad)
            .max(self.max_edge_cross_track_rad)
    }
}

fn normalized<P: UnitVec3Like>(p: &P) -> DVec3 {
    DVec3::new(p.x() as f64, p.y() as f64, p.z() as f64).normalize()
}

fn angular_distance(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b).clamp(-1.0, 1.0))
}

fn ownership_violation(sample: DVec3, owner: usize, generators: &[DVec3]) -> f64 {
    let owner_distance = angular_distance(sample, generators[owner]);
    let best_distance = generators
        .iter()
        .copied()
        .map(|g| angular_distance(sample, g))
        .fold(f64::INFINITY, f64::min);
    (owner_distance - best_distance).max(0.0)
}

fn cross_track_radians(sample: DVec3, a: DVec3, b: DVec3) -> f64 {
    let bisector_normal = a - b;
    let length = bisector_normal.length();
    if length == 0.0 || !length.is_finite() {
        return f64::INFINITY;
    }
    (sample.dot(bisector_normal).abs() / length).min(1.0).asin()
}

fn owner_plane_arc_sample(a: DVec3, b: DVec3, ga: DVec3, gb: DVec3, t: f64) -> DVec3 {
    let normal = (ga - gb).normalize();
    let a = (a - normal * normal.dot(a)).normalize();
    let b = (b - normal * normal.dot(b)).normalize();
    let mut tangent = normal.cross(a).normalize();
    if tangent.dot(b) < 0.0 {
        tangent = -tangent;
    }
    let angle = tangent.dot(b).max(0.0).atan2(a.dot(b).clamp(-1.0, 1.0));
    (a * (t * angle).cos() + tangent * (t * angle).sin()).normalize()
}

fn measure_intrinsic_geometry<G: UnitVec3Like, V: UnitVec3Like>(
    generators: &[G],
    vertices: &[V],
    cells: &[Vec<u32>],
) -> IntrinsicGeometry {
    let generators: Vec<DVec3> = generators.iter().map(normalized).collect();
    let vertices: Vec<DVec3> = vertices.iter().map(normalized).collect();
    let mut incident_cells = vec![Vec::<usize>::new(); vertices.len()];
    let mut edge_cells = BTreeMap::<(u32, u32), Vec<usize>>::new();

    for (cell_idx, cell) in cells.iter().enumerate() {
        for &vertex in cell {
            incident_cells[vertex as usize].push(cell_idx);
        }
        for edge_idx in 0..cell.len() {
            let a = cell[edge_idx];
            let b = cell[(edge_idx + 1) % cell.len()];
            edge_cells
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(cell_idx);
        }
    }

    let mut result = IntrinsicGeometry::default();
    for (vertex_idx, cells) in incident_cells.iter_mut().enumerate() {
        cells.sort_unstable();
        cells.dedup();
        let sample = vertices[vertex_idx];
        for &owner in cells.iter() {
            result.vertex_owner_samples += 1;
            result.max_ownership_violation_rad = result
                .max_ownership_violation_rad
                .max(ownership_violation(sample, owner, &generators));
        }
        for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                result.max_vertex_cross_track_rad = result.max_vertex_cross_track_rad.max(
                    cross_track_radians(sample, generators[cells[i]], generators[cells[j]]),
                );
            }
        }
    }

    for ((a, b), owners) in edge_cells.iter_mut() {
        owners.sort_unstable();
        owners.dedup();
        if owners.len() != 2 {
            result.max_edge_cross_track_rad = f64::INFINITY;
            continue;
        }
        let va = vertices[*a as usize];
        let vb = vertices[*b as usize];
        let ga = generators[owners[0]];
        let gb = generators[owners[1]];
        for k in 0..=4 {
            let sample = owner_plane_arc_sample(va, vb, ga, gb, k as f64 / 4.0);
            result.edge_samples += 1;
            result.max_edge_cross_track_rad = result.max_edge_cross_track_rad.max(
                cross_track_radians(sample, generators[owners[0]], generators[owners[1]]),
            );
            for &owner in owners.iter() {
                result.max_ownership_violation_rad = result
                    .max_ownership_violation_rad
                    .max(ownership_violation(sample, owner, &generators));
            }
        }
    }

    // A generator-biased point inside every cell-edge wedge catches a cell
    // that owns the right boundary geometry but the wrong side of an edge.
    for (cell_idx, cell) in cells.iter().enumerate() {
        for edge_idx in 0..cell.len() {
            let a = vertices[cell[edge_idx] as usize];
            let b = vertices[cell[(edge_idx + 1) % cell.len()] as usize];
            let sample = (generators[cell_idx] * 2.0 + a + b).normalize();
            result.interior_samples += 1;
            result.max_ownership_violation_rad = result
                .max_ownership_violation_rad
                .max(ownership_violation(sample, cell_idx, &generators));
        }
    }

    result
}

fn diagram_cells(diagram: &SphericalVoronoi) -> Vec<Vec<u32>> {
    diagram
        .iter_cells()
        .map(|cell| cell.vertex_indices.to_vec())
        .collect()
}

fn assert_case(points: &[UnitVec3], label: &str) -> IntrinsicGeometry {
    let output = compute_with_report(points, VoronoiConfig::default())
        .unwrap_or_else(|err| panic!("{label}: computation failed: {err}; points={points:?}"));
    assert!(
        !output.report.preprocess.did_merge(),
        "{label}: ordinary small-n fixture unexpectedly welded {} generators",
        output.report.preprocess.num_merged
    );
    let cells = diagram_cells(&output.diagram);
    let geometry = measure_intrinsic_geometry(
        output.diagram.generators(),
        output.diagram.vertices(),
        &cells,
    );
    let validation = validate(&output.diagram);
    assert!(
        validation.is_strictly_valid(),
        "{label}: strict validation failed: {}; geometry={geometry:?}; points={points:?}; vertices={:?}; cells={cells:?}",
        validation.headline(),
        output.diagram.vertices(),
    );
    assert!(
        geometry.max_error() <= MAX_INTRINSIC_ANGULAR_ERROR,
        "{label}: intrinsic geometry exceeded {MAX_INTRINSIC_ANGULAR_ERROR:.1e} rad: {geometry:?}"
    );
    geometry
}

#[test]
fn uniform_small_n_exhaustive_range_has_intrinsic_voronoi_geometry() {
    // Includes random_sphere_points(5, 2), the former topologically valid but
    // geometrically wrong AUD-002 counterexample.
    for n in 4..=32 {
        for seed in 0..64u64 {
            let points = random_sphere_points(n, seed);
            assert_case(&points, &format!("uniform n={n} seed={seed}"));
        }
    }
}

#[test]
fn structured_small_n_cases_have_intrinsic_voronoi_geometry() {
    for n in [8usize, 12, 16, 24, 32] {
        for seed in 0..8u64 {
            let cases = [
                ("fibonacci", fibonacci_sphere_points(n, 1e-4, seed)),
                ("clustered", clustered_cap_points(n, 0.05, seed)),
                ("cube", cube_vertex_stress_points(n, 0.02, seed)),
                (
                    "cocircular",
                    near_cocircular_stress_points((n / 4).max(1), 1e-4, seed),
                ),
                ("hemisphere", hemisphere_points(n, seed)),
                ("bimodal", bimodal_density_points(n, 0.05, seed)),
            ];
            for (distribution, points) in cases {
                assert_case(&points, &format!("{distribution} n={n} seed={seed}"));
            }
        }
    }
}

#[test]
fn intrinsic_oracle_rejects_topologically_unchanged_vertex_rotation() {
    let points = [
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
    ];
    let diagram = voronoi_mesh::compute(&points).unwrap();
    assert!(validate(&diagram).is_strictly_valid());

    // A global rotation preserves the closed complex, winding, areas, and
    // Euler characteristic. Rotating only vertices relative to generators is
    // nevertheless not their Voronoi diagram and must fail this oracle.
    let axis = DVec3::new(0.3, -0.5, 0.8).normalize();
    let angle = 0.17f64;
    let rotate = |p: DVec3| {
        p * angle.cos() + axis.cross(p) * angle.sin() + axis * axis.dot(p) * (1.0 - angle.cos())
    };
    let rotated: Vec<UnitVec3> = diagram
        .vertices()
        .iter()
        .map(|p| rotate(normalized(p)))
        .map(|p| UnitVec3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let geometry =
        measure_intrinsic_geometry(diagram.generators(), &rotated, &diagram_cells(&diagram));
    assert!(
        geometry.max_error() > 1e-2,
        "rotated negative control escaped intrinsic oracle: {geometry:?}"
    );
}

#[test]
#[ignore = "extended deterministic campaign; run manually in release mode"]
fn extended_uniform_small_n_intrinsic_geometry_campaign() {
    let max_n = std::env::var("VORONOI_SMALL_N_MAX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64usize)
        .max(4);
    let seeds = std::env::var("VORONOI_SMALL_N_SEEDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024u64);

    let mut cases = 0usize;
    let mut worst = IntrinsicGeometry::default();
    for n in 4..=max_n {
        for seed in 0..seeds {
            let points = random_sphere_points(n, seed);
            let geometry = assert_case(&points, &format!("uniform n={n} seed={seed}"));
            if geometry.max_error() > worst.max_error() {
                worst = geometry;
            }
            cases += 1;
        }
    }
    eprintln!("small-n intrinsic campaign: cases={cases} worst={worst:?}");
}
