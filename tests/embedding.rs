use std::cell::Cell;

use voronoi_mesh::{
    compute, compute_on_sphere, compute_on_sphere_with_report, SphereEmbedding,
    SphereEmbeddingError, SphereProjectionError, UnitVec3, VoronoiConfig, VoronoiError,
    WorldVec3Like,
};

struct NonSyncPoint {
    xyz: [f64; 3],
    marker: Cell<()>,
}

impl WorldVec3Like for NonSyncPoint {
    fn x(&self) -> f64 {
        self.marker.set(());
        self.xyz[0]
    }

    fn y(&self) -> f64 {
        self.xyz[1]
    }

    fn z(&self) -> f64 {
        self.xyz[2]
    }
}

fn octahedron_unit() -> Vec<UnitVec3> {
    vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
    ]
}

fn fibonacci_unit(count: usize) -> Vec<UnitVec3> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..count)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let radial = (1.0 - z * z).sqrt();
            let theta = i as f64 * golden_angle;
            UnitVec3::new(
                (radial * theta.cos()) as f32,
                (radial * theta.sin()) as f32,
                z as f32,
            )
        })
        .collect()
}

fn embed(center: [f64; 3], direction: UnitVec3, distance: f64) -> [f64; 3] {
    [
        center[0] + distance * direction.x as f64,
        center[1] + distance * direction.y as f64,
        center[2] + distance * direction.z as f64,
    ]
}

fn assert_same_diagram(a: &voronoi_mesh::SphericalVoronoi, b: &voronoi_mesh::SphericalVoronoi) {
    assert_eq!(a.generators(), b.generators());
    assert_eq!(a.vertices(), b.vertices());
    assert_eq!(a.num_cells(), b.num_cells());
    assert_eq!(a.num_vertices(), b.num_vertices());
    assert_eq!(a.weld_map(), b.weld_map());
    for i in 0..a.num_cells() {
        assert_eq!(a.cell(i).vertex_indices, b.cell(i).vertex_indices);
    }
}

#[test]
fn translated_scaled_octahedron_matches_unit_computation() {
    let unit = octahedron_unit();
    let center = [10.0, -20.0, 30.0];
    let embedding = SphereEmbedding::new(center, 4.0).unwrap();
    let world: Vec<[f64; 3]> = unit
        .iter()
        .map(|&direction| embed(center, direction, 4.0))
        .collect();

    let canonical = compute(&unit).unwrap();
    let embedded = compute_on_sphere(&world, embedding).unwrap();
    assert_same_diagram(&canonical, embedded.diagram());

    for (i, &point) in world.iter().enumerate() {
        assert_eq!(embedded.generator_world(i), point);
    }
    let total_area: f64 = (0..embedded.diagram().num_cells())
        .map(|i| embedded.cell_area_world(i))
        .sum();
    let expected_area = 4.0 * std::f64::consts::PI * 16.0;
    assert!((total_area - expected_area).abs() / expected_area < 1e-6);
}

#[test]
fn translated_scaled_non_axis_points_match_recovered_unit_computation() {
    let unit = fibonacci_unit(257);
    let center = [12.125, -98.75, 0.03125];
    let embedding = SphereEmbedding::new(center, 11.5).unwrap();
    let world: Vec<[f64; 3]> = unit
        .iter()
        .enumerate()
        .map(|(i, &direction)| embed(center, direction, 0.25 + (i % 19) as f64))
        .collect();
    let recovered: Vec<UnitVec3> = world
        .iter()
        .map(|point| {
            let u = embedding.project_world_to_unit(point).unwrap();
            UnitVec3::new(u[0] as f32, u[1] as f32, u[2] as f32)
        })
        .collect();

    let canonical = compute(&recovered).unwrap();
    let embedded = compute_on_sphere(&world, embedding).unwrap();
    assert_same_diagram(&canonical, embedded.diagram());
}

#[test]
fn embedded_compute_and_locator_accept_non_sync_point_types() {
    let embedding = SphereEmbedding::new([2.0, 3.0, 5.0], 7.0).unwrap();
    let world: Vec<NonSyncPoint> = octahedron_unit()
        .into_iter()
        .map(|direction| NonSyncPoint {
            xyz: embed(embedding.center(), direction, embedding.radius()),
            marker: Cell::new(()),
        })
        .collect();

    let embedded = compute_on_sphere(&world, embedding).unwrap();
    let locator = embedded.build_locator_world();
    assert_eq!(
        locator.locate_many_world(&world).unwrap(),
        (0..6).collect::<Vec<_>>()
    );
}

#[test]
fn radial_distance_is_deliberately_discarded() {
    let unit = octahedron_unit();
    let center = [3.0, 5.0, -7.0];
    let distances = [0.125, 250.0, 2.0, 17.0, 0.5, 1_000.0];
    let world: Vec<[f64; 3]> = unit
        .iter()
        .zip(distances)
        .map(|(&direction, distance)| embed(center, direction, distance))
        .collect();
    let embedding = SphereEmbedding::new(center, 9.0).unwrap();

    let canonical = compute(&unit).unwrap();
    let embedded = compute_on_sphere(&world, embedding).unwrap();
    assert_same_diagram(&canonical, embedded.diagram());
    for i in 0..unit.len() {
        let projected = embedding
            .project_world_to_unit(&world[i])
            .expect("off-shell point still defines a direction");
        assert_eq!(
            projected,
            [unit[i].x as f64, unit[i].y as f64, unit[i].z as f64]
        );
        assert_eq!(
            embedded.generator_world(i),
            embedding.unit_to_world(projected)
        );
    }
}

#[test]
fn projection_is_scale_safe_for_extreme_finite_displacements() {
    let origin = SphereEmbedding::new([0.0; 3], 1.0).unwrap();

    let tiny = origin
        .project_world_to_unit(&[1e-300, -2e-300, 2e-300])
        .unwrap();
    assert!((tiny[0] - 1.0 / 3.0).abs() < 1e-15);
    assert!((tiny[1] + 2.0 / 3.0).abs() < 1e-15);
    assert!((tiny[2] - 2.0 / 3.0).abs() < 1e-15);

    let subnormal = f64::from_bits(1);
    assert_eq!(
        origin
            .project_world_to_unit(&[subnormal, 0.0, 0.0])
            .unwrap(),
        [1.0, 0.0, 0.0]
    );

    let shifted = SphereEmbedding::new([-f64::MAX / 2.0, 0.0, 0.0], 1.0).unwrap();
    assert_eq!(
        shifted
            .project_world_to_unit(&[f64::MAX, 0.0, 0.0])
            .unwrap(),
        [1.0, 0.0, 0.0]
    );
}

#[test]
fn embedding_and_projection_errors_are_explicit() {
    assert!(matches!(
        SphereEmbedding::new([f64::NAN, 0.0, 0.0], 1.0),
        Err(SphereEmbeddingError::NonFiniteCenter { component: 0 })
    ));
    assert!(matches!(
        SphereEmbedding::new([0.0; 3], 0.0),
        Err(SphereEmbeddingError::InvalidRadius)
    ));
    assert!(matches!(
        SphereEmbedding::new([f64::MAX, 0.0, 0.0], 1.0),
        Err(SphereEmbeddingError::UnrepresentableExtent { component: 0 })
    ));

    let embedding = SphereEmbedding::new([1.0, 2.0, 3.0], 1.0).unwrap();
    assert_eq!(
        embedding.project_world_to_unit(&[1.0, 2.0, 3.0]),
        Err(SphereProjectionError::PointAtCenter)
    );
    assert_eq!(
        embedding.project_world_to_unit(&[1.0, f64::INFINITY, 3.0]),
        Err(SphereProjectionError::NonFinitePoint { component: 1 })
    );

    let too_short = [[1.0, 2.0, 3.0]];
    assert!(matches!(
        compute_on_sphere(&too_short, embedding),
        Err(VoronoiError::InsufficientPoints(1))
    ));

    let mut world: Vec<[f64; 3]> = octahedron_unit()
        .iter()
        .map(|&direction| embed(embedding.center(), direction, 1.0))
        .collect();
    world[2] = embedding.center();
    world[4][0] = f64::NAN;
    assert!(matches!(
        compute_on_sphere(&world, embedding),
        Err(VoronoiError::InvalidInput { point_index: 2, .. })
    ));
}

#[test]
fn embedded_report_wraps_returned_and_effective_diagrams() {
    let center = [7.0, -11.0, 13.0];
    let embedding = SphereEmbedding::new(center, 5.0).unwrap();
    let mut unit = octahedron_unit();
    unit.push(unit[0]);
    let mut world: Vec<[f64; 3]> = unit
        .iter()
        .map(|&direction| embed(center, direction, 5.0))
        .collect();
    world[6] = embed(center, unit[6], 2.0); // same ray, deliberately off shell

    let output =
        compute_on_sphere_with_report(&world, embedding, VoronoiConfig::default()).unwrap();
    assert_eq!(output.diagram.diagram().num_cells(), 7);
    assert_eq!(output.diagram.diagram().canonical_cell_index(6), 0);
    assert_eq!(output.diagram.embedding(), embedding);
    assert_eq!(output.report.preprocess.original_points, 7);
    assert_eq!(output.report.preprocess.effective_points, 6);
    assert!(output.report.preprocess.did_merge());
    let effective = output.effective_diagram.as_ref().unwrap();
    assert_eq!(effective.diagram().num_cells(), 6);
    assert_eq!(effective.embedding(), embedding);
    assert_eq!(output.preferred_diagram().diagram().num_cells(), 6);
}

#[test]
fn world_locator_and_lloyd_targets_match_unit_space() {
    let center = [-4.0, 8.0, 12.0];
    let embedding = SphereEmbedding::new(center, 3.0).unwrap();
    let unit = octahedron_unit();
    let world: Vec<[f64; 3]> = unit
        .iter()
        .enumerate()
        .map(|(i, &direction)| embed(center, direction, i as f64 + 0.5))
        .collect();
    let embedded = compute_on_sphere(&world, embedding).unwrap();

    let mut locator = embedded.build_locator_world();
    for (i, point) in world.iter().enumerate() {
        assert_eq!(locator.locate_world(point).unwrap(), i);
    }
    assert_eq!(
        locator.locate_many_world(&world).unwrap(),
        (0..6).collect::<Vec<_>>()
    );
    let mut invalid = world.clone();
    invalid[4] = center;
    invalid[5][1] = f64::INFINITY;
    let err = locator.locate_many_world(&invalid).unwrap_err();
    assert_eq!(err.point_index(), 4);
    assert_eq!(err.projection_error(), SphereProjectionError::PointAtCenter);

    let world_lloyd = embedded.lloyd_step_world();
    for (i, point) in world_lloyd.iter().enumerate() {
        let projected = embedding.project_world_to_unit(point).unwrap();
        let unit_centroid = embedded.diagram().cell_centroid(i);
        assert!((projected[0] - unit_centroid.x as f64).abs() < 1e-7);
        assert!((projected[1] - unit_centroid.y as f64).abs() < 1e-7);
        assert!((projected[2] - unit_centroid.z as f64).abs() < 1e-7);
        assert_eq!(*point, embedded.cell_centroid_world(i));
    }
}

#[test]
fn physical_area_overflow_follows_ieee_semantics() {
    let embedding = SphereEmbedding::new([0.0; 3], 1e200).unwrap();
    assert!(embedding.solid_angle_to_area(1.0).is_infinite());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_projection_reports_lowest_invalid_index() {
    let embedding = SphereEmbedding::new([0.0; 3], 1.0).unwrap();
    let mut world = vec![[1.0, 0.0, 0.0]; 20_000];
    world[17_000][2] = f64::NAN;
    world[1_234] = embedding.center();
    assert!(matches!(
        compute_on_sphere(&world, embedding),
        Err(VoronoiError::InvalidInput {
            point_index: 1_234,
            ..
        })
    ));
}

#[cfg(feature = "glam")]
#[test]
fn glam_dvec3_is_a_world_input() {
    let unit = octahedron_unit();
    let embedding = SphereEmbedding::new([0.0; 3], 2.0).unwrap();
    let world: Vec<glam::DVec3> = unit
        .iter()
        .map(|p| glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64) * 2.0)
        .collect();
    assert_eq!(
        compute_on_sphere(&world, embedding)
            .unwrap()
            .diagram()
            .num_cells(),
        6
    );
}

#[cfg(feature = "serde")]
#[test]
fn embedded_serde_is_checked_and_round_trips() {
    let center = [10.0, 20.0, 30.0];
    let embedding = SphereEmbedding::new(center, 4.0).unwrap();
    let world: Vec<[f64; 3]> = octahedron_unit()
        .iter()
        .map(|&direction| embed(center, direction, 4.0))
        .collect();
    let embedded = compute_on_sphere(&world, embedding).unwrap();

    let json = serde_json::to_string(&embedded).unwrap();
    let restored: voronoi_mesh::EmbeddedSphericalVoronoi = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.embedding(), embedding);
    assert_same_diagram(restored.diagram(), embedded.diagram());

    let mut malformed: serde_json::Value = serde_json::from_str(&json).unwrap();
    malformed["embedding"]["radius"] = serde_json::json!(0.0);
    let err = serde_json::from_value::<voronoi_mesh::EmbeddedSphericalVoronoi>(malformed)
        .expect_err("invalid embedded radius must be rejected");
    assert!(
        err.to_string().contains("radius"),
        "unexpected error: {err}"
    );
}
