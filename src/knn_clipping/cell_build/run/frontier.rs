use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream,
};
use crate::policy::TerminationPolicy;

#[inline]
pub(super) fn probe_frontier<'a, 'm, 'p, 'g>(
    stream: &mut DirectedNeighborStream<'a, 'm, 'p, 'g>,
    packed_chunk: &mut Vec<u32>,
    used_knn: &mut bool,
    knn_stage: &mut crate::knn_clipping::timing::KnnCellStage,
    knn_query_time: &mut Duration,
) -> DirectedNeighborFrontier {
    let t_knn = crate::knn_clipping::timing::Timer::start();
    let cursor_stage_before = stream.is_cursor_stage();
    let frontier = stream.frontier(packed_chunk);
    let frontier_is_cursor = match frontier {
        DirectedNeighborFrontier::ExactBatch(batch) => matches!(
            batch.source,
            DirectedNeighborBatchSource::DirectedCursor | DirectedNeighborBatchSource::ShellExpand
        ),
        DirectedNeighborFrontier::UnknownButBounded { .. }
        | DirectedNeighborFrontier::Exhausted => cursor_stage_before,
    };
    if frontier_is_cursor {
        *used_knn = true;
        *knn_stage = crate::knn_clipping::timing::KnnCellStage::DirectedCursor;
        *knn_query_time += t_knn.elapsed();
    }
    frontier
}

#[inline]
pub(super) fn maybe_terminate_or_advance_frontier<'a, 'm, 'p, 'g>(
    stream: &mut DirectedNeighborStream<'a, 'm, 'p, 'g>,
    packed_chunk: &mut Vec<u32>,
    builder: &mut crate::knn_clipping::topo2d::Topo2DBuilder,
    termination: TerminationPolicy,
    neighbors_processed: usize,
    used_knn: &mut bool,
    knn_stage: &mut crate::knn_clipping::timing::KnnCellStage,
    knn_query_time: &mut Duration,
) -> bool {
    let frontier = probe_frontier(stream, packed_chunk, used_knn, knn_stage, knn_query_time);
    let can_check_cursor = termination.should_check(neighbors_processed);

    match frontier {
        DirectedNeighborFrontier::ExactBatch(batch) => {
            let can_check =
                batch.source != DirectedNeighborBatchSource::DirectedCursor || can_check_cursor;
            can_check && builder.can_terminate(batch.first_dot)
        }
        DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
            let can_check = !stream.is_cursor_stage() || can_check_cursor;
            if can_check && builder.can_terminate(dot_upper_bound) {
                true
            } else {
                stream.advance_frontier();
                false
            }
        }
        DirectedNeighborFrontier::Exhausted => true,
    }
}
