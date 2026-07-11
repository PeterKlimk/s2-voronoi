use glam::Vec3;

use super::super::{CellBuildError, CellFailure};
use super::{BuildCounters, BuildTrace, CellBuildContext};

pub(super) fn classify_terminal_failure(
    bounded: bool,
    failure: Option<CellFailure>,
    knn_exhausted: bool,
) -> Option<CellFailure> {
    if let Some(
        failure @ (CellFailure::ProjectionInvalid
        | CellFailure::TooManyVertices
        | CellFailure::ClippedAway),
    ) = failure
    {
        return Some(failure);
    }
    if !bounded && knn_exhausted {
        return Some(CellFailure::UnboundedAfterExhaustion);
    }
    None
}

pub(super) fn unexpected_failure_error(
    ctx: &CellBuildContext,
    points: &[Vec3],
    generator_idx: usize,
    trace: &BuildTrace,
    counters: &BuildCounters,
    context: &str,
    explicit_failure: Option<CellFailure>,
) -> CellBuildError {
    let (active, total) = ctx.builder.count_active_planes();
    let builder = ctx.builder.debug_state();
    let extraction_failure = ctx.builder.debug_extraction_failure();
    let generator = points[generator_idx];
    let neighbor_indices: Vec<usize> = ctx.builder.neighbor_indices_iter().collect();
    let failure = explicit_failure.or(ctx.builder.failure());
    let failure = failure.unwrap_or(CellFailure::NoValidSeed);

    CellBuildError {
        generator_idx,
        failure,
        detail: Some(format!(
            "unexpected {} failure: bounded={}, failure={:?}, \
             planes={}, active={}, vertices={}, poly_len={}, has_bounding_ref={}, min_cos={:?}, \
             half_plane_count={}, neighbor_index_count={}, neighbor_slot_count={}, extraction_failure={:?}, neighbors_processed={}, \
             did_packed={}, did_takeover={}, knn_exhausted={}, \
             last_clip_phase={}, last_batch_source={:?}, last_neighbor_idx={:?}, last_neighbor_slot={:?}; \
             generator_pos={:?}; first_10_neighbor_indices={:?}",
            context,
            builder.bounded,
            failure,
            total,
            active,
            ctx.builder.vertex_count(),
            builder.poly_len,
            builder.has_bounding_ref,
            builder.min_cos,
            builder.half_plane_count,
            builder.neighbor_index_count,
            builder.neighbor_slot_count,
            extraction_failure,
            counters.neighbors_processed,
            counters.did_packed,
            counters.used_knn,
            counters.knn_exhausted,
            trace.last_clip_phase,
            trace.last_batch_source,
            trace.last_neighbor_idx,
            trace.last_neighbor_slot,
            generator,
            &neighbor_indices[..neighbor_indices.len().min(10)],
        )),
    }
}
