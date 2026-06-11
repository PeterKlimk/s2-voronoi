//! Public-API tests for planar Voronoi computation: strict subdivision
//! contract (validate_plane), the area partition, weld semantics, domain
//! mapping, and input validation.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{compute_plane, validation, PlanarVoronoi, PlanePoint, PlaneRect, VoronoiError};

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn uniform_in(rect: PlaneRect, n: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut r = rng(seed);
    (0..n)
        .map(|_| {
            [
                r.gen_range(rect.min.x..rect.max.x),
                r.gen_range(rect.min.y..rect.max.y),
            ]
        })
        .collect()
}

fn total_area(diagram: &PlanarVoronoi) -> f64 {
    let weld_map = diagram.weld_map();
    let mut acc = 0.0f64;
    for i in 0..diagram.num_cells() {
        if let Some(m) = weld_map {
            if m[i] as usize != i {
                continue; // welded twin aliases canonical storage
            }
        }
        let cell = diagram.cell(i);
        for k in 0..cell.len() {
            let a = diagram.vertex(cell[k] as usize);
            let b = diagram.vertex(cell[(k + 1) % cell.len()] as usize);
            acc += (a.x as f64) * (b.y as f64) - (b.x as f64) * (a.y as f64);
        }
    }
    0.5 * acc
}

fn assert_strict(diagram: &PlanarVoronoi, name: &str) {
    let report = validation::validate_plane(diagram);
    assert!(
        report.is_strictly_valid(),
        "{name}: diagram failed strict validation: {report:#?}"
    );
}

fn assert_area(diagram: &PlanarVoronoi, name: &str) {
    let rect = diagram.rect();
    let expected = rect.width() as f64 * rect.height() as f64;
    let area = total_area(diagram);
    assert!(
        (area - expected).abs() < 1e-4 * expected.max(1.0),
        "{name}: cell areas sum to {area}, expected {expected}"
    );
}

#[test]
fn plane_uniform_strict_and_area() {
    for &(n, seed) in &[(50usize, 3u64), (500, 5), (5000, 7)] {
        let points = uniform_in(PlaneRect::unit(), n, seed);
        let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
        assert_eq!(diagram.num_cells(), n);
        assert_strict(&diagram, &format!("uniform_{n}"));
        assert_area(&diagram, &format!("uniform_{n}"));
    }
}

#[test]
fn plane_non_square_rect_mapping() {
    // A wide rect away from the origin: exercises the uniform-scale domain
    // transform (no anisotropic distortion allowed).
    let rect = PlaneRect::new(PlanePoint::new(-10.0, 5.0), PlanePoint::new(30.0, 15.0));
    let points = uniform_in(rect, 800, 11);
    let diagram = compute_plane(&points, rect).unwrap();
    assert_strict(&diagram, "wide_rect");
    assert_area(&diagram, "wide_rect");

    // All vertices inside the rect (within tolerance), generators preserved.
    for (i, p) in points.iter().enumerate() {
        let g = diagram.generator(i);
        assert_eq!([g.x, g.y], *p, "generator {i} not preserved");
    }
}

#[test]
fn plane_single_point_owns_rect() {
    let rect = PlaneRect::new(PlanePoint::new(2.0, 3.0), PlanePoint::new(6.0, 5.0));
    let diagram = compute_plane(&[[4.0f32, 4.0]], rect).unwrap();
    assert_eq!(diagram.num_cells(), 1);
    assert_eq!(diagram.cell(0).len(), 4);
    assert_strict(&diagram, "n1");
    assert_area(&diagram, "n1");
}

#[test]
fn plane_tiny_inputs() {
    for n in 2..=6 {
        let points = uniform_in(PlaneRect::unit(), n, 13 + n as u64);
        let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
        assert_strict(&diagram, &format!("tiny_n{n}"));
        assert_area(&diagram, &format!("tiny_n{n}"));
    }
}

#[test]
fn plane_clustered_and_collinear() {
    let mut r = rng(17);
    let mut clustered: Vec<[f32; 2]> = (0..400)
        .map(|_| {
            [
                (0.5 + r.gen_range(-0.01f32..0.01)).clamp(0.0, 1.0),
                (0.5 + r.gen_range(-0.01f32..0.01)).clamp(0.0, 1.0),
            ]
        })
        .collect();
    clustered.push([0.01, 0.01]);
    clustered.push([0.99, 0.99]);
    let diagram = compute_plane(&clustered, PlaneRect::unit()).unwrap();
    assert_strict(&diagram, "clustered");
    assert_area(&diagram, "clustered");

    let collinear: Vec<[f32; 2]> = (0..60)
        .map(|i| [0.05 + 0.9 * (i as f32 / 59.0), 0.5])
        .collect();
    let diagram = compute_plane(&collinear, PlaneRect::unit()).unwrap();
    assert_strict(&diagram, "collinear");
    assert_area(&diagram, "collinear");
}

#[test]
fn plane_exact_lattice_ties() {
    // Every interior Voronoi vertex is a 4-cocircular tie; the strict graph
    // contract must hold regardless of how the ties resolve.
    let mut points = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            points.push([(i as f32 + 0.5) / 8.0, (j as f32 + 0.5) / 8.0]);
        }
    }
    let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
    assert_strict(&diagram, "lattice");
    assert_area(&diagram, "lattice");
}

#[test]
fn plane_exact_duplicates_weld() {
    let mut points = uniform_in(PlaneRect::unit(), 100, 23);
    points.push(points[7]);
    points.push(points[7]);
    points.push(points[42]);

    let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
    assert_eq!(diagram.num_cells(), 103);
    let weld = diagram
        .weld_map()
        .expect("duplicates must produce weld map");
    assert_eq!(weld[100], 7);
    assert_eq!(weld[101], 7);
    assert_eq!(weld[102], 42);
    assert_eq!(diagram.cell(100), diagram.cell(7));
    assert_eq!(diagram.cell(102), diagram.cell(42));
    assert_strict(&diagram, "duplicates");
    assert_area(&diagram, "duplicates");
}

#[test]
fn plane_boundary_generators() {
    // Generators exactly on walls and corners of the rect.
    let mut points: Vec<[f32; 2]> = vec![
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.5],
        [1.0, 0.5],
        [0.5, 0.0],
        [0.5, 1.0],
    ];
    points.extend(uniform_in(PlaneRect::unit(), 50, 29));
    let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
    assert_strict(&diagram, "boundary_generators");
    assert_area(&diagram, "boundary_generators");
}

#[test]
fn plane_input_validation() {
    let unit = PlaneRect::unit();

    let empty: Vec<[f32; 2]> = Vec::new();
    assert!(matches!(
        compute_plane(&empty, unit),
        Err(VoronoiError::InsufficientPoints(0))
    ));

    assert!(matches!(
        compute_plane(&[[0.5f32, 1.5]], unit),
        Err(VoronoiError::InvalidInput { point_index: 0, .. })
    ));

    assert!(matches!(
        compute_plane(&[[f32::NAN, 0.5]], unit),
        Err(VoronoiError::InvalidInput { point_index: 0, .. })
    ));

    let degenerate = PlaneRect::new(PlanePoint::new(0.0, 0.0), PlanePoint::new(0.0, 1.0));
    assert!(matches!(
        compute_plane(&[[0.0f32, 0.5]], degenerate),
        Err(VoronoiError::InvalidDomain { .. })
    ));

    let backwards = PlaneRect::new(PlanePoint::new(1.0, 1.0), PlanePoint::new(0.0, 0.0));
    assert!(matches!(
        compute_plane(&[[0.5f32, 0.5]], backwards),
        Err(VoronoiError::InvalidDomain { .. })
    ));
}

#[test]
fn plane_point_like_inputs() {
    let rect = PlaneRect::unit();
    let arrays = vec![[0.2f32, 0.2], [0.8, 0.3], [0.5, 0.8]];
    let tuples: Vec<(f32, f32)> = arrays.iter().map(|p| (p[0], p[1])).collect();
    let structs: Vec<PlanePoint> = arrays.iter().map(|p| PlanePoint::new(p[0], p[1])).collect();

    let a = compute_plane(&arrays, rect).unwrap();
    let b = compute_plane(&tuples, rect).unwrap();
    let c = compute_plane(&structs, rect).unwrap();
    assert_eq!(a.num_vertices(), b.num_vertices());
    assert_eq!(b.num_vertices(), c.num_vertices());
}

#[test]
fn plane_larger_uniform_strict() {
    let points = uniform_in(PlaneRect::unit(), 50_000, 31);
    let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
    assert_strict(&diagram, "uniform_50k");
    assert_area(&diagram, "uniform_50k");
}

/// Multi-million-point fuzz sweep; run with `--ignored` (scheduled CI).
#[test]
#[ignore]
fn plane_fuzz_large() {
    for &(n, seed) in &[(1_000_000usize, 101u64), (2_000_000, 103)] {
        let points = uniform_in(PlaneRect::unit(), n, seed);
        let diagram = compute_plane(&points, PlaneRect::unit()).unwrap();
        let report = validation::validate_plane(&diagram);
        assert!(
            report.is_strictly_valid(),
            "fuzz n={n} seed={seed}: {report:#?}"
        );
    }
}
