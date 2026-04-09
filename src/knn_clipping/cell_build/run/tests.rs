use super::failure::classify_terminal_failure;
use super::{build_cell_into, CellBuildContext, CellBuildRequest};
use crate::cube_grid::{CubeMapGrid, DirectedEligibility};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::TerminationConfig;
use glam::Vec3;

fn octahedron_points() -> Vec<Vec3> {
    vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
}

#[test]
fn projection_invalid_stays_distinct_from_exhausted_unbounded() {
    assert_eq!(
        classify_terminal_failure(false, Some(CellFailure::ProjectionInvalid), true),
        Some(CellFailure::ProjectionInvalid)
    );
    assert_eq!(
        classify_terminal_failure(false, None, true),
        Some(CellFailure::UnboundedAfterExhaustion)
    );
}

#[test]
fn too_many_vertices_is_a_structured_failure() {
    assert_eq!(
        classify_terminal_failure(true, Some(CellFailure::TooManyVertices), false),
        Some(CellFailure::TooManyVertices)
    );
}

#[test]
fn bounded_nonfailed_cell_has_no_terminal_failure() {
    assert_eq!(classify_terminal_failure(true, None, true), None);
    assert_eq!(classify_terminal_failure(true, None, false), None);
}

#[test]
fn direct_cursor_builds_normal_cell() {
    let points = octahedron_points();
    let grid = CubeMapGrid::new(&points, 4);
    let policy = TerminationConfig::default().knn_policy(points.len());
    let mut ctx = CellBuildContext::new(&grid, policy);
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);

    let stats = build_cell_into(
        &mut ctx,
        CellBuildRequest {
            points: &points,
            grid: &grid,
            generator_idx: 0,
            directed_ctx,
            termination: policy.termination(),
            packed: None,
            seed_neighbors: &[],
        },
    )
    .expect("cell build should succeed");

    assert!(ctx.output_buffer().vertices.len() >= 3);
    assert!(!stats.knn_exhausted || !stats.did_packed);
}
