use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream,
};

#[inline]
pub(super) fn probe_frontier<'a, 'm, 'p, 'g>(
    stream: &mut DirectedNeighborStream<'a, 'm, 'p, 'g>,
    packed_chunk: &mut Vec<u32>,
    used_knn: &mut bool,
    knn_stage: &mut crate::knn_clipping::timing::KnnCellStage,
    knn_query_time: &mut Duration,
) -> DirectedNeighborFrontier {
    let t_knn = crate::knn_clipping::timing::Timer::start();
    let takeover_before = stream.is_takeover_stage();
    let frontier = stream.frontier(packed_chunk);
    let frontier_is_takeover = match frontier {
        DirectedNeighborFrontier::ExactBatch(batch) => {
            batch.source == DirectedNeighborBatchSource::ShellExpand
        }
        DirectedNeighborFrontier::UnknownButBounded { .. }
        | DirectedNeighborFrontier::Exhausted => takeover_before,
    };
    if frontier_is_takeover {
        *used_knn = true;
        *knn_stage = crate::knn_clipping::timing::KnnCellStage::ShellExpand;
        *knn_query_time += t_knn.elapsed();
    }
    frontier
}

#[inline]
pub(super) fn maybe_terminate_or_advance_frontier<'a, 'm, 'p, 'g>(
    stream: &mut DirectedNeighborStream<'a, 'm, 'p, 'g>,
    packed_chunk: &mut Vec<u32>,
    builder: &mut crate::knn_clipping::topo2d::Topo2DBuilder,
    _pos_slots: &[crate::cube_grid::SlotPoint],
    counters: &mut super::BuildCounters,
) -> bool {
    let frontier = probe_frontier(
        stream,
        packed_chunk,
        &mut counters.used_knn,
        &mut counters.knn_stage,
        &mut counters.knn_query_time,
    );

    match frontier {
        DirectedNeighborFrontier::ExactBatch(batch) => {
            // Termination before consuming a batch must bound everything
            // unseen: the batch itself plus what lies beyond. Packed batches
            // dominate their unseen set, so first_dot suffices; shell layers
            // do not (the next layer can beat this layer's best), so combine
            // with the layer certificate.
            let bound = if batch.source == DirectedNeighborBatchSource::ShellExpand {
                batch.first_dot.max(batch.unseen_bound)
            } else {
                batch.first_dot
            };
            if builder.can_terminate(bound) {
                return true;
            }
            #[cfg(feature = "timing")]
            super::audit_directional_batch_skip(
                builder,
                &packed_chunk[..batch.n],
                batch.unseen_bound,
                _pos_slots,
                counters,
            );
            false
        }
        DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
            if builder.can_terminate(dot_upper_bound) {
                true
            } else {
                stream.advance_frontier();
                false
            }
        }
        DirectedNeighborFrontier::Exhausted => true,
    }
}
