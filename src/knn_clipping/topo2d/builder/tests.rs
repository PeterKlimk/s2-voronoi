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
    assert_eq!(gnomonic.half_planes.len(), 0);
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
fn too_many_vertices_records_current_constraint_before_fallback() {
    let mut builder = Topo2DBuilder::new(0, Vec3::Z);
    let gnomonic = builder.as_gnomonic_mut();
    let hp = HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0);

    assert_eq!(
        gnomonic.commit_clip(ClipResult::TooManyVertices, hp, 11, 21,),
        Err(crate::knn_clipping::cell_build::CellFailure::TooManyVertices)
    );
    assert_eq!(gnomonic.half_planes.len(), 1);
    assert_eq!(gnomonic.neighbor_indices, vec![11]);
    assert_eq!(gnomonic.neighbor_slots, vec![21]);
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
    builder.enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::PolygonVertexLimit,
        },
    );

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
    builder.enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        },
    );

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
    builder.enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        },
    );

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

    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize() * 0.9997;
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize() * 1.0003;
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize() * 0.9998;

    builder
        .clip_with_slot_edgecheck_policy(11, 21, h1, 0.125)
        .expect("edgecheck clip should apply");
    builder
        .clip_with_slot_policy(12, 22, h2)
        .expect("normal clip should apply");
    builder
        .clip_with_slot_policy(13, 23, h3)
        .expect("normal clip should apply");

    let points = fallback_points(g, h1, h2, h3);
    builder.enter_fallback(
        &points,
        BuilderFallbackRequest {
            trigger: BuilderFallbackTrigger::ProjectionLimit,
        },
    );

    let expected = |h: Vec3| {
        glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64).normalize()
            - glam::DVec3::new(h.x as f64, h.y as f64, h.z as f64).normalize()
    };
    for (constraint, h) in builder.as_fallback().constraints.iter().zip([h1, h2, h3]) {
        let diff = constraint.normal - expected(h);
        assert!(
            diff.length() < 1e-12,
            "fallback constraint did not normalize S2 inputs: diff={diff:?}"
        );
    }
}
