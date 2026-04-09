use super::projection::MIN_PROJECTION_COS;
use super::*;
use crate::knn_clipping::cell_build::CellOutputBuffer;
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane};
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
    gnomonic.poly_b.edge_planes[0] = usize::MAX;
    gnomonic.poly_b.edge_planes[1] = usize::MAX;
    gnomonic.poly_b.edge_planes[2] = usize::MAX;
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
