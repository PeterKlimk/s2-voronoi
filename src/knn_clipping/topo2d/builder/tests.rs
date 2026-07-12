use super::projection::MIN_PROJECTION_COS;
use super::*;
use crate::knn_clipping::cell_build::CellOutputBuffer;
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane, INVALID_PLANE_ID};
use glam::Vec3;
use std::cmp::Ordering;

fn generic_sphere_points(n: usize) -> Vec<Vec3> {
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f32;
            let jitter_x = (((i * 37 + 11) as f32) * 0.12345).sin() * 0.002;
            let jitter_z = (((i * 53 + 7) as f32) * 0.23456).cos() * 0.002;
            let x = radius * theta.cos() + jitter_x;
            let z = radius * theta.sin() + jitter_z;
            Vec3::new(x, y, z).normalize()
        })
        .collect()
}

fn clip_all_neighbors(builder: &mut Topo2DBuilder, points: &[Vec3], i: usize) {
    let mut neighbors: Vec<usize> = (0..points.len()).filter(|&j| j != i).collect();
    neighbors.sort_by(|&a, &b| {
        points[i]
            .dot(points[b])
            .partial_cmp(&points[i].dot(points[a]))
            .unwrap_or(Ordering::Equal)
    });

    for &j in &neighbors {
        let result = builder.clip_with_slot_result(j, j as u32, points[j]);
        assert!(
            result.is_ok(),
            "cell {} clipping against neighbor {} failed with {:?}",
            i,
            j,
            result
        );
    }
}

fn fallback_points(g: Vec3, h1: Vec3, h2: Vec3, h3: Vec3) -> Vec<Vec3> {
    let mut points = vec![Vec3::ZERO; 14];
    points[0] = g;
    points[11] = h1;
    points[12] = h2;
    points[13] = h3;
    points
}

fn fallback_points4(g: Vec3, h1: Vec3, h2: Vec3, h3: Vec3, h4: Vec3) -> Vec<Vec3> {
    let mut points = fallback_points(g, h1, h2, h3);
    points.resize(15, Vec3::ZERO);
    points[14] = h4;
    points
}

#[test]
fn changed_clip_fails_when_bounded_polygon_reaches_projection_limit() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    gnomonic.poly_b.clear();
    gnomonic.poly_b.len = 3;
    gnomonic.poly_b.has_bounding_ref = false;
    let invalid_min_cos = MIN_PROJECTION_COS * 0.5;
    gnomonic.poly_b.max_r2 = (1.0 / (invalid_min_cos * invalid_min_cos)) - 1.0;

    let err = builder
        .as_gnomonic_mut()
        .commit_clip(
            ClipResult::Changed,
            HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0),
            1,
            u32::MAX,
        )
        .expect_err("expected projection-invalid bounded cell to fail");
    assert_eq!(
        err,
        crate::knn_clipping::cell_build::CellFailure::ProjectionInvalid
    );
    assert_eq!(
        builder.failure(),
        Some(crate::knn_clipping::cell_build::CellFailure::ProjectionInvalid)
    );
}

#[test]
fn changed_clip_allows_bounded_polygon_inside_projection_limit() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    gnomonic.poly_b.clear();
    gnomonic.poly_b.len = 3;
    gnomonic.poly_b.has_bounding_ref = false;
    let valid_min_cos = MIN_PROJECTION_COS * 2.0;
    gnomonic.poly_b.max_r2 = (1.0 / (valid_min_cos * valid_min_cos)) - 1.0;

    let result = builder.as_gnomonic_mut().commit_clip(
        ClipResult::Changed,
        HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0),
        1,
        u32::MAX,
    );
    assert_eq!(result, Ok(ClipResult::Changed));
    assert_eq!(builder.failure(), None);
}

#[test]
fn unchanged_clip_skips_projection_validation() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    gnomonic.poly_a.clear();
    gnomonic.poly_a.len = 3;
    gnomonic.poly_a.has_bounding_ref = false;
    gnomonic.poly_a.max_r2 = f64::INFINITY;

    let result = gnomonic.commit_clip(
        ClipResult::Unchanged,
        HalfPlane::new_unnormalized(1.0, 0.0, 1.0, 0),
        1,
        u32::MAX,
    );

    assert_eq!(result, Ok(ClipResult::Unchanged));
    assert_eq!(gnomonic.constraints.len(), 0);
    assert_eq!(builder.failure(), None);
}

#[test]
fn generic_full_sphere_cells_do_not_clip_away() {
    let points = generic_sphere_points(24);

    for i in 0..points.len() {
        let mut builder = Topo2DBuilder::new(i, points[i]);
        clip_all_neighbors(&mut builder, &points, i);
        assert!(
            builder.is_bounded(),
            "cell {} should be bounded after clipping against all neighbors",
            i
        );
        assert_eq!(
            builder.failure(),
            None,
            "cell {} should not fail after clipping against all neighbors",
            i
        );
    }
}

#[test]
fn valid_bounded_cells_reconstruct_vertices_with_healthy_norm() {
    let points = generic_sphere_points(24);

    for i in 0..points.len() {
        let mut builder = Topo2DBuilder::new(i, points[i]);
        clip_all_neighbors(&mut builder, &points, i);
        assert!(builder.is_bounded(), "cell {} should be bounded", i);

        let poly = builder.as_gnomonic().current_poly();
        let basis = &builder.as_gnomonic().basis;
        for v in 0..poly.len {
            let u = poly.us[v];
            let w = poly.vs[v];
            let dir = basis.g + basis.t1 * u + basis.t2 * w;
            let len2 = dir.length_squared();
            assert!(
                len2.is_finite() && len2 > 0.5,
                "cell {} vertex {} reconstructed direction has suspicious len2={}",
                i,
                v,
                len2
            );
        }

        let mut buffer = CellOutputBuffer::default();
        builder
            .to_vertex_data_full(&mut buffer)
            .expect("valid bounded cell should extract vertex data");
        assert!(
            !buffer.vertices.is_empty(),
            "cell {} should produce at least one vertex",
            i
        );
    }
}

#[test]
fn extraction_failure_reports_invalid_vertex_plane_metadata() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    gnomonic.poly_b.clear();
    gnomonic.poly_b.len = 3;
    gnomonic.poly_b.has_bounding_ref = false;
    gnomonic.poly_b.max_r2 = 1.0;
    gnomonic.poly_b.us[0] = 0.0;
    gnomonic.poly_b.vs[0] = 0.0;
    gnomonic.poly_b.us[1] = 0.5;
    gnomonic.poly_b.vs[1] = 0.0;
    gnomonic.poly_b.us[2] = 0.0;
    gnomonic.poly_b.vs[2] = 0.5;
    gnomonic.poly_b.vertex_planes[0] = (7, 8);
    gnomonic.poly_b.vertex_planes[1] = (7, 8);
    gnomonic.poly_b.vertex_planes[2] = (7, 8);
    gnomonic.poly_b.edge_planes[0] = INVALID_PLANE_ID;
    gnomonic.poly_b.edge_planes[1] = INVALID_PLANE_ID;
    gnomonic.poly_b.edge_planes[2] = INVALID_PLANE_ID;
    gnomonic.use_a = false;

    let mut buffer = CellOutputBuffer::default();
    buffer.vertices.push(([u32::MAX; 3], Vec3::X));
    buffer.edge_neighbor_globals.push(u32::MAX);
    buffer.edge_neighbor_slots.push(u32::MAX);
    buffer.edge_neighbor_eps.push(f32::INFINITY);
    assert_eq!(
        gnomonic.to_vertex_data_full(&mut buffer),
        Err(crate::knn_clipping::cell_build::CellFailure::NoValidSeed)
    );
    assert!(buffer.vertices.is_empty());
    assert!(buffer.edge_neighbor_globals.is_empty());
    assert!(buffer.edge_neighbor_slots.is_empty());
    assert!(buffer.edge_neighbor_eps.is_empty());

    assert_eq!(
        builder.debug_extraction_failure(),
        Some(ExtractionInvariantFailure::InvalidVertexPlane {
            vertex: 0,
            plane_a: 7,
            plane_b: 8,
            neighbor_index_count: 0,
        })
    );
}

#[test]
fn projection_invalid_is_classified_as_fallback_handoff() {
    assert_eq!(
        Topo2DBuilder::classify_clip_result(Err(
            crate::knn_clipping::cell_build::CellFailure::ProjectionInvalid,
        )),
        Ok(BuilderClipOutcome::NeedsFallback(BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        }))
    );
}

#[test]
fn too_many_vertices_is_classified_as_polygon_fallback_handoff() {
    assert_eq!(
        Topo2DBuilder::classify_clip_result(Err(
            crate::knn_clipping::cell_build::CellFailure::TooManyVertices,
        )),
        Ok(BuilderClipOutcome::NeedsFallback(BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::PolygonVertexLimit,
        }))
    );
}

#[test]
fn clipped_away_is_classified_as_exact_fallback_handoff() {
    assert_eq!(
        Topo2DBuilder::classify_clip_result(Err(
            crate::knn_clipping::cell_build::CellFailure::ClippedAway,
        )),
        Ok(BuilderClipOutcome::NeedsFallback(BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ClippedAway,
        }))
    );
}

#[test]
fn clipped_away_handoff_rebuilds_from_constraints() {
    let g = Vec3::Z;
    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();
    let mut builder = Topo2DBuilder::new(0, g);
    builder.clip_with_slot_policy(11, 21, h1).unwrap();
    builder.clip_with_slot_policy(12, 22, h2).unwrap();
    builder.clip_with_slot_policy(13, 23, h3).unwrap();

    // Model the gnomonic failure state: the accepted constraints are intact,
    // but the rounded chart polygon has collapsed below three vertices.
    let gnomonic = builder.as_gnomonic_mut();
    gnomonic.poly_a.len = 0;
    gnomonic.poly_b.len = 0;
    gnomonic.failed = Some(crate::knn_clipping::cell_build::CellFailure::ClippedAway);

    let switched = builder.try_enter_fallback(
        &fallback_points(g, h1, h2, h3),
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ClippedAway,
        },
    );
    assert!(switched);
    assert!(builder.is_bounded());
    let mut buffer = CellOutputBuffer::default();
    builder.to_vertex_data_full(&mut buffer).unwrap();
    assert_eq!(buffer.vertices.len(), 3);
}

#[test]
fn clipped_away_handoff_rejection_preserves_failed_builder() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    for (neighbor_idx, neighbor_slot) in [(11, 21), (12, 22), (13, 23), (14, 24)] {
        gnomonic.constraints.push(super::GnomonicConstraint {
            half_plane: HalfPlane::new_unnormalized(1.0, 0.0, 0.0, neighbor_idx),
            neighbor_idx,
            neighbor_slot,
        });
    }
    gnomonic.poly_a.len = 0;
    gnomonic.poly_b.len = 0;
    gnomonic.failed = Some(crate::knn_clipping::cell_build::CellFailure::ClippedAway);

    let eps = f32::EPSILON;
    let points = fallback_points4(
        Vec3::Z,
        Vec3::new(eps, 0.0, 1.0).normalize(),
        Vec3::new(-eps, 0.0, 1.0).normalize(),
        Vec3::new(0.0, eps, 1.0).normalize(),
        Vec3::new(0.0, -eps, 1.0).normalize(),
    );
    assert!(!builder.try_enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ClippedAway,
        },
    ));
    assert!(!builder.is_fallback());
    assert!(builder.is_failed());
    assert_eq!(
        builder.failure(),
        Some(crate::knn_clipping::cell_build::CellFailure::ClippedAway)
    );
}

#[test]
fn too_many_vertices_records_current_constraint_before_fallback() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    let hp = HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0);

    assert_eq!(
        gnomonic.commit_clip(ClipResult::TooManyVertices, hp, 11, 21,),
        Err(crate::knn_clipping::cell_build::CellFailure::TooManyVertices)
    );
    assert_eq!(gnomonic.constraints.len(), 1);
    assert_eq!(gnomonic.constraints[0].neighbor_idx, 11);
    assert_eq!(gnomonic.constraints[0].neighbor_slot, 21);
}

#[test]
fn polygon_vertex_limit_handoff_replays_overflowing_constraint() {
    let g = Vec3::new(0.0, 0.0, 1.0);
    let mut builder = Topo2DBuilder::new(0, g);

    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();
    let h4 = Vec3::new(0.0, 1.0, 0.75).normalize();

    assert_eq!(
        builder.clip_with_slot_edgecheck_policy(11, 21, h1, 0.125),
        Ok(BuilderStepOutcome::Applied)
    );
    assert_eq!(
        builder.clip_with_slot_policy(12, 22, h2),
        Ok(BuilderStepOutcome::Applied)
    );
    assert_eq!(
        builder.clip_with_slot_policy(13, 23, h3),
        Ok(BuilderStepOutcome::Applied)
    );

    let hp = HalfPlane::new_unnormalized(1.0, 0.0, 0.0, builder.accepted_constraint_count());
    assert_eq!(
        builder
            .as_gnomonic_mut()
            .commit_clip(ClipResult::TooManyVertices, hp, 14, 24,),
        Err(crate::knn_clipping::cell_build::CellFailure::TooManyVertices)
    );

    let points = fallback_points4(g, h1, h2, h3, h4);
    assert!(builder.try_enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::PolygonVertexLimit,
        },
    ));

    assert_eq!(
        builder.as_fallback().trigger,
        BuilderFallbackTrigger::PolygonVertexLimit
    );
    assert_eq!(builder.as_fallback().constraints.len(), 4);
    assert_eq!(builder.as_fallback().constraints[3].neighbor_idx, 14);
    assert_eq!(builder.as_fallback().constraints[3].neighbor_slot, 24);

    let mut buffer = CellOutputBuffer::default();
    builder
        .to_vertex_data_full(&mut buffer)
        .expect("polygon-cap fallback replay should produce extractable vertices");
    assert!(buffer.vertices.len() >= 3);
}

#[test]
fn fallback_handoff_switches_builder_variant_and_replays_constraints() {
    let g = Vec3::new(0.0, 0.0, 1.0);
    let mut builder = Topo2DBuilder::new(17, g);

    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

    builder
        .clip_with_slot_edgecheck_policy(11, 21, h1, 0.125)
        .expect("edgecheck clip should apply");
    builder
        .clip_with_slot_policy(12, 22, h2)
        .expect("normal clip should apply");
    builder
        .clip_with_slot_policy(13, 23, h3)
        .expect("normal clip should apply");

    let outcome = Topo2DBuilder::handle_clip_result(Err(
        crate::knn_clipping::cell_build::CellFailure::ProjectionInvalid,
    ))
    .expect("projection invalid should be converted to a fallback handoff");
    let points = fallback_points(g, h1, h2, h3);
    assert!(builder.try_enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        },
    ));

    assert_eq!(
        outcome,
        BuilderClipOutcome::NeedsFallback(BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        })
    );
    assert!(!builder.is_failed());
    assert!(builder.is_bounded());
    assert_eq!(builder.failure(), None);
    assert_eq!(
        builder.as_fallback().trigger,
        BuilderFallbackTrigger::ProjectionLimit
    );
    assert_eq!(builder.as_fallback().constraints.len(), 3);
    assert_eq!(builder.as_fallback().constraints[0].neighbor_idx, 11);
    assert_eq!(builder.as_fallback().constraints[0].neighbor_slot, 21);
    assert_eq!(builder.as_fallback().constraints[0].hp_eps, Some(0.125));
    assert_eq!(builder.as_fallback().constraints[1].neighbor_idx, 12);
    assert_eq!(builder.as_fallback().constraints[2].neighbor_idx, 13);

    let mut buffer = CellOutputBuffer::default();
    builder
        .to_vertex_data_full(&mut buffer)
        .expect("fallback replay should produce extractable vertices");
    assert_eq!(buffer.vertices.len(), 3);
}

#[test]
fn fallback_reconstruction_preserves_constraint_order_and_eps() {
    let g = Vec3::new(0.0, 0.0, 1.0);
    let mut builder = Topo2DBuilder::new(0, g);

    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

    assert_eq!(
        builder.clip_with_slot_edgecheck_policy(11, 21, h1, 0.125),
        Ok(BuilderStepOutcome::Applied)
    );
    assert_eq!(
        builder.clip_with_slot_policy(12, 22, h2),
        Ok(BuilderStepOutcome::Applied)
    );
    assert_eq!(
        builder.clip_with_slot_policy(13, 23, h3),
        Ok(BuilderStepOutcome::Applied)
    );

    let points = fallback_points(g, h1, h2, h3);
    assert!(builder.try_enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        },
    ));

    assert_eq!(builder.as_fallback().generator_idx, 0);
    assert_eq!(builder.as_fallback().constraints.len(), 3);
    assert_eq!(builder.as_fallback().constraints[0].neighbor_idx, 11);
    assert_eq!(builder.as_fallback().constraints[0].neighbor_slot, 21);
    assert_eq!(builder.as_fallback().constraints[0].hp_eps, Some(0.125));
    assert_eq!(builder.as_fallback().constraints[1].neighbor_idx, 12);
    assert_eq!(builder.as_fallback().constraints[1].neighbor_slot, 22);
    // Ordinary clips carry the strict-rule eps (CLIP_EPS_INSIDE = 0.0),
    // preserved exactly through reconstruction.
    assert_eq!(builder.as_fallback().constraints[1].hp_eps, Some(0.0));
    assert_eq!(builder.as_fallback().constraints[2].neighbor_idx, 13);
    assert_eq!(builder.as_fallback().constraints[2].neighbor_slot, 23);
    assert_eq!(builder.as_fallback().constraints[2].hp_eps, Some(0.0));
}

#[test]
fn fallback_reconstruction_normalizes_s2_constraints() {
    let g = Vec3::new(0.0, 0.0, 1.0002);
    let mut builder = Topo2DBuilder::new(0, g);

    // Off-unit S2 positions: the property under test is that the fallback
    // reconstruction normalizes what it reads from the `points` array
    // (`FallbackConstraint::from_neighbor(.., points[neighbor_idx])`). The
    // clip path itself has a canonicalized-unit precondition (debug-asserted
    // in `bisector_coefficients`; production inputs pass through
    // `canonicalize_unit_points`), so the clips receive the normalized forms
    // and only `points` carries the off-unit values.
    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize() * 0.9997;
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize() * 1.0003;
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize() * 0.9998;

    builder
        .clip_with_slot_edgecheck_policy(11, 21, h1.normalize(), 0.125)
        .expect("edgecheck clip should apply");
    builder
        .clip_with_slot_policy(12, 22, h2.normalize())
        .expect("normal clip should apply");
    builder
        .clip_with_slot_policy(13, 23, h3.normalize())
        .expect("normal clip should apply");

    let points = fallback_points(g, h1, h2, h3);
    assert!(builder.try_enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        },
    ));

    let expected = |h: Vec3| {
        (glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64).normalize()
            - glam::DVec3::new(h.x as f64, h.y as f64, h.z as f64).normalize())
        .normalize_or_zero()
    };
    for (constraint, h) in builder.as_fallback().constraints.iter().zip([h1, h2, h3]) {
        let diff = constraint.normal - expected(h);
        assert!(
            diff.length() < 1e-12,
            "fallback constraint did not normalize S2 inputs: diff={diff:?}"
        );
    }
}

#[test]
fn fallback_stale_corner_is_rebuilt_from_all_constraints() {
    let t = 0.2;
    let delta = 1.0e-6;
    let normal = |x: f64, y: f64, z: f64| glam::DVec3::new(x, y, z).normalize();
    let constraint = |normal: glam::DVec3, neighbor_idx: usize| FallbackConstraint {
        normal,
        neighbor_idx,
        neighbor_slot: neighbor_idx as u32,
        hp_eps: None,
    };

    // A/B/D/E bound a square around +Z. C cuts off the A/B corner, but the
    // incremental polygon deliberately retains the stale pre-C corner and
    // edge labels. This models the rare extraction fallthrough: exact A/B is
    // outside C, while C is far enough from the stale corner that the local
    // split-plane lookup does not attribute it.
    let constraints = vec![
        constraint(normal(1.0, 0.0, t), 11),               // A: x >= -t z
        constraint(normal(0.0, 1.0, t), 12),               // B: y >= -t z
        constraint(normal(-1.0, 0.0, t), 13),              // D: x <=  t z
        constraint(normal(0.0, -1.0, t), 14),              // E: y <=  t z
        constraint(normal(1.0, 1.0, 2.0 * t - delta), 15), // C
    ];
    let corner = |x: f64, y: f64| SphericalPolyVertex {
        position: glam::DVec3::new(x, y, 1.0).normalize(),
    };
    let builder = FallbackBuilder {
        generator_idx: 0,
        generator: glam::DVec3::Z,
        constraints,
        poly: SphericalPoly {
            vertices: vec![corner(-t, -t), corner(t, -t), corner(t, t), corner(-t, t)],
            // Each entry labels the outgoing edge; A is therefore the
            // incoming label at the first (stale) corner.
            edge_planes: vec![1, 2, 3, 0],
        },
        trigger: BuilderFallbackTrigger::ProjectionLimit,
    };

    let mut buffer = CellOutputBuffer::default();
    builder
        .to_vertex_data_full(&mut buffer)
        .expect("all-constraints retry should recover the cut corner");

    assert_eq!(buffer.vertices.len(), 5);
    assert!(!buffer.vertices.iter().any(|(key, _)| *key == [0, 11, 12]));
    assert!(buffer.vertices.iter().any(|(key, _)| *key == [0, 11, 15]));
    assert!(buffer.vertices.iter().any(|(key, _)| *key == [0, 12, 15]));
    for (_, position) in &buffer.vertices {
        let position = glam::DVec3::new(position.x as f64, position.y as f64, position.z as f64);
        assert!(
            builder
                .constraints
                .iter()
                .all(|constraint| constraint.normal.dot(position) >= -1.0e-7),
            "extracted vertex lies outside an accepted constraint: {position:?}"
        );
    }
}

/// Find an adversarial-but-legitimate f32 generator on the ring just above
/// the TangentBasis pole cutoff: |g|² − 1 within ~1 ulp of the z component
/// (any f32 normalization can land there), maximizing chart metric skew.
fn worst_ring_generator() -> Vec3 {
    // The exact f32 ring just above the -0.9999999 cutoff (`-1 + 2^-23`).
    #[allow(clippy::excessive_precision)]
    let z: f32 = -0.999_999_88;
    let mut worst = (0.0f64, Vec3::ZERO);
    // A legitimately-normalized f32 point with this z can carry any
    // tangential radius whose exact direction rounds its z here: the
    // half-ulp z freedom (±3e-8) is a ±12% radius band at this latitude.
    let r0 = (1.0f64 - (z as f64) * (z as f64)).max(0.0).sqrt();
    for k in 0..256 {
        let phi = (k as f64) * (std::f64::consts::TAU / 256.0);
        for fr in 0..400 {
            let f = 0.88 + (fr as f64) * (0.24 / 400.0);
            let x = (r0 * f * phi.cos()) as f32;
            let y = (r0 * f * phi.sin()) as f32;
            let g = glam::DVec3::new(x as f64, y as f64, z as f64);
            let delta = g.length_squared() - 1.0;
            // Prove this exact f32 vector can be emitted by the production
            // f64-normalize -> f32-round canonicalization, rather than merely
            // accepting a similar-sized norm error.
            let unit = g.normalize();
            let roundtrip = Vec3::new(unit.x as f32, unit.y as f32, unit.z as f32);
            if roundtrip != Vec3::new(x, y, z) {
                continue;
            }
            debug_assert!(delta.abs() <= 6.0e-8);
            let basis = TangentBasis::new(g);
            let inv_g2 = 1.0 / g.length_squared();
            let g11 = basis.t1.dot(basis.t1) * inv_g2;
            let g22 = basis.t2.dot(basis.t2) * inv_g2;
            let g12 = basis.t1.dot(basis.t2) * inv_g2;
            let dev = (g11 - 1.0).abs().max((g22 - 1.0).abs()).max(g12.abs());
            if dev > worst.0 {
                worst = (dev, Vec3::new(x, y, z));
            }
        }
    }
    assert!(
        worst.0 > 0.05,
        "expected large metric distortion on the cutoff ring, got {}",
        worst.0
    );
    worst.1
}

#[test]
fn polar_projection_limit_uses_chart_metric_bound() {
    let g = worst_ring_generator();
    let mut builder = Topo2DBuilder::new(0, g);
    let gnomonic = builder.as_gnomonic_mut();
    assert!(
        gnomonic.chart_metric_r2_scale > 1.5,
        "expected a strongly stretched polar chart, got scale {}",
        gnomonic.chart_metric_r2_scale
    );

    gnomonic.poly_b.clear();
    gnomonic.poly_b.len = 3;
    gnomonic.poly_b.has_bounding_ref = false;
    // The legacy unscaled radius appears 10% inside the permitted chart,
    // while the true metric-bound radius is already beyond the limit.
    let legacy_min_cos = MIN_PROJECTION_COS * 1.1;
    gnomonic.poly_b.max_r2 = (1.0 / (legacy_min_cos * legacy_min_cos)) - 1.0;

    let err = gnomonic
        .commit_clip(
            ClipResult::Changed,
            HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0),
            1,
            u32::MAX,
        )
        .expect_err("metric-stretched chart must hand off at the projection limit");
    assert_eq!(
        err,
        crate::knn_clipping::cell_build::CellFailure::ProjectionInvalid
    );
}

/// Build a small cell around `g` from neighbors at angular distance `2*theta`
/// spread over azimuths, using real f32 unit points.
fn clip_ring_neighbors(
    builder: &mut Topo2DBuilder,
    g: Vec3,
    theta: f64,
    count: usize,
) -> Vec<Vec3> {
    let gd = glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64).normalize();
    // Any orthonormal frame for placing neighbors (exactness not needed here).
    let helper = if gd.z.abs() < 0.9 {
        glam::DVec3::Z
    } else {
        glam::DVec3::X
    };
    let e1 = (helper - gd * helper.dot(gd)).normalize();
    let e2 = gd.cross(e1);
    let mut pts = Vec::new();
    for i in 0..count {
        let az = (i as f64) * std::f64::consts::TAU / (count as f64);
        let dir = e1 * az.cos() + e2 * az.sin();
        let p = (gd * (2.0 * theta).cos() + dir * (2.0 * theta).sin()).normalize();
        let p32 = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        builder
            .clip_with_slot_result(100 + i, 100 + i as u32, p32)
            .expect("neighbor clip");
        pts.push(p32);
    }
    pts
}

/// SMOKING GUN (pre-fix): on a worst-ring generator, the scalar termination
/// certificate accepts an unseen-dot bound while a candidate AT that bound
/// still produces a `Changed` clip — i.e. `can_terminate` is unsound under
/// near-pole chart metric distortion. After the metric-bound fix this must
/// no longer be constructible.
#[test]
fn polar_termination_certificate_soundness() {
    let g = worst_ring_generator();
    let gd = glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64);

    let theta = 0.01f64;
    let mut builder = Topo2DBuilder::new(0, g);
    clip_ring_neighbors(&mut builder, g, theta, 6);
    assert!(builder.is_bounded());

    // True max vertex angle from extracted chart vertices vs the claimed one.
    let (claimed, true_max) = {
        let gn = builder.as_gnomonic();
        let poly = gn.current_poly();
        let claimed = (1.0 / (1.0 + poly.max_r2).sqrt()).acos();
        let mut true_max = 0.0f64;
        for i in 0..poly.len {
            let p = gn.basis.g + gn.basis.t1 * poly.us[i] + gn.basis.t2 * poly.vs[i];
            let ang = (p.dot(gd) / (p.length() * gd.length()))
                .clamp(-1.0, 1.0)
                .acos();
            true_max = true_max.max(ang);
        }
        (claimed, true_max)
    };
    let pad = 8.0 * f32::EPSILON as f64;
    eprintln!(
        "claimed theta_max = {claimed:.9}, true = {true_max:.9}, err = {:.3e} (pad {pad:.3e})",
        true_max - claimed
    );

    // Search for a violating candidate: direction of the true-farthest vertex,
    // at angles gamma where the certificate claims safety.
    let (vdir, _) = {
        let gn = builder.as_gnomonic();
        let poly = gn.current_poly();
        let mut best = (glam::DVec3::ZERO, 0.0f64);
        for i in 0..poly.len {
            let p = gn.basis.g + gn.basis.t1 * poly.us[i] + gn.basis.t2 * poly.vs[i];
            let ang = (p.dot(gd) / (p.length() * gd.length()))
                .clamp(-1.0, 1.0)
                .acos();
            if ang > best.1 {
                best = (p.normalize(), ang);
            }
        }
        best
    };
    let gdn = gd.normalize();
    let tangent = (vdir - gdn * vdir.dot(gdn)).normalize();

    let mut violations = 0usize;
    for step in 0..2000 {
        let gamma = 2.0 * (claimed + pad) + 1.0e-6 + (step as f64) * 1.0e-6;
        let c = (gdn * gamma.cos() + tangent * gamma.sin()).normalize();
        let c32 = Vec3::new(c.x as f32, c.y as f32, c.z as f32);
        let dot = crate::fp::dot3_f32(g.x, g.y, g.z, c32.x, c32.y, c32.z);
        if !builder.can_terminate(dot) {
            continue; // certificate (correctly) refuses; not a candidate
        }
        // Certificate claims: nothing at or beyond this dot can cut the cell.
        // Replay the exact clip decision for this candidate.
        let gn = builder.as_gnomonic();
        let (a, b, cc) = gn.bisector_coefficients(c32);
        let hp = HalfPlane::new_unnormalized(a, b, cc, 0);
        let poly = gn.current_poly();
        let neg_eps = -hp.eps;
        let cuts = (0..poly.len).any(|i| hp.signed_dist(poly.us[i], poly.vs[i]) < neg_eps);
        if cuts {
            violations += 1;
        }
    }
    eprintln!("violations found: {violations}");
    assert_eq!(
        violations, 0,
        "can_terminate accepted a bound while a candidate at that bound still cuts the cell"
    );
}

/// SMOKING GUN, take 2: elongate the cell along the chart's stretched axis so
/// `max_r2` is set by a distorted vertex. The scalar termination certificate
/// must never accept a bound while a candidate at that bound still cuts.
#[test]
fn polar_termination_certificate_soundness_elongated() {
    let g = worst_ring_generator();
    let gd = glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64);
    let basis = TangentBasis::new(gd);
    // Chart axes as 3D unit directions (the stretch lives along these).
    let e1 = basis.t1.normalize();
    let e2 = basis.t2.normalize();
    let gdn = gd.normalize();

    let theta = 0.01f64;
    let mut builder = Topo2DBuilder::new(0, g);
    // Tight walls along ±e2, far walls along ±e1 → far vertices lie in the
    // e1 direction where the chart metric is distorted.
    let mut place = |dir: glam::DVec3, ang: f64, idx: usize| {
        let p = (gdn * ang.cos() + dir * ang.sin()).normalize();
        let p32 = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        builder
            .clip_with_slot_result(100 + idx, 100 + idx as u32, p32)
            .expect("neighbor clip");
    };
    place(e2, 2.0 * theta, 0);
    place(-e2, 2.0 * theta, 1);
    place(e1, 4.0 * theta, 2);
    place(-e1, 4.0 * theta, 3);
    assert!(builder.is_bounded());

    let (claimed, true_max) = {
        let gn = builder.as_gnomonic();
        let poly = gn.current_poly();
        let claimed = (1.0 / (1.0 + poly.max_r2).sqrt()).acos();
        let mut true_max = 0.0f64;
        for i in 0..poly.len {
            let p = gn.basis.g + gn.basis.t1 * poly.us[i] + gn.basis.t2 * poly.vs[i];
            let ang = (p.dot(gd) / (p.length() * gd.length()))
                .clamp(-1.0, 1.0)
                .acos();
            true_max = true_max.max(ang);
        }
        (claimed, true_max)
    };
    let pad = 8.0 * f32::EPSILON as f64;
    eprintln!(
        "claimed theta_max = {claimed:.9}, true = {true_max:.9}, err = {:.3e} (pad {pad:.3e})",
        true_max - claimed
    );

    // Candidates along the elongated (distorted) axis, at angles the
    // certificate believes are beyond the security radius.
    let mut violations = 0usize;
    let mut first: Option<f64> = None;
    for step in 0..4000 {
        let gamma = 2.0 * (claimed + pad) + 1.0e-6 + (step as f64) * 2.0e-6;
        for dir in [e1, -e1] {
            let c = (gdn * gamma.cos() + dir * gamma.sin()).normalize();
            let c32 = Vec3::new(c.x as f32, c.y as f32, c.z as f32);
            let dot = crate::fp::dot3_f32(g.x, g.y, g.z, c32.x, c32.y, c32.z);
            if !builder.can_terminate(dot) {
                continue;
            }
            let gn = builder.as_gnomonic();
            let (a, b, cc) = gn.bisector_coefficients(c32);
            let hp = HalfPlane::new_unnormalized(a, b, cc, 0);
            let poly = gn.current_poly();
            let neg_eps = -hp.eps;
            if (0..poly.len).any(|i| hp.signed_dist(poly.us[i], poly.vs[i]) < neg_eps) {
                violations += 1;
                first.get_or_insert(gamma);
            }
        }
    }
    eprintln!("violations found: {violations} (first at gamma = {first:?})");
    assert_eq!(
        violations, 0,
        "can_terminate accepted a bound while a candidate at that bound still cuts the cell"
    );
}
