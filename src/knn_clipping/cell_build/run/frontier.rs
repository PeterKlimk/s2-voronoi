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
    phase: &mut super::StreamPhase<'_>,
    pos_slots: &[crate::cube_grid::SlotPoint],
    generator_idx: usize,
    counters: &mut super::BuildCounters,
) -> bool {
    let frontier = probe_frontier(
        stream,
        phase.packed_chunk,
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
            if phase.builder.can_terminate(bound) {
                return true;
            }
            // Directional attempt over the whole probed batch: the scalar
            // certificate covers everything beyond it, so if every eligible
            // batch candidate certifies as non-cutting the cell is complete
            // without consuming the batch. A blocked attempt records nothing —
            // the batch is then clipped normally (v1 drops the certified
            // prefix here; the mid-batch attempt re-derives it cheaply).
            if phase.directional_term && phase.builder.can_terminate(batch.unseen_bound) {
                #[cfg(feature = "timing")]
                {
                    counters.dir_term_attempts += 1;
                }
                if super::directional_certify_slots(
                    phase.builder,
                    &phase.packed_chunk[..batch.n],
                    batch.source,
                    pos_slots,
                    generator_idx,
                    phase.attempted_neighbors,
                )
                .is_ok()
                {
                    #[cfg(feature = "timing")]
                    {
                        counters.dir_term_terminations += 1;
                        counters.dir_term_saved += batch.n;
                    }
                    return true;
                }
            }
            #[cfg(feature = "timing")]
            super::audit_directional_batch_skip(
                phase.builder,
                &phase.packed_chunk[..batch.n],
                batch.unseen_bound,
                pos_slots,
                counters,
            );
            false
        }
        DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
            if phase.builder.can_terminate(dot_upper_bound) {
                true
            } else {
                stream.advance_frontier();
                false
            }
        }
        DirectedNeighborFrontier::Exhausted => true,
    }
}
