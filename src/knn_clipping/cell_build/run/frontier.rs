use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatch, DirectedNeighborBatchSource, DirectedNeighborFrontier,
    DirectedNeighborStream,
};

/// Combine the best known dot in an exact batch remainder with the bound for
/// everything after that batch. Neither set is allowed to disappear from the
/// termination certificate, regardless of how the batch was produced.
#[inline(always)]
pub(super) fn complete_exact_bound(batch_remainder_bound: f32, unseen_bound: f32) -> f32 {
    batch_remainder_bound.max(unseen_bound)
}

#[inline(always)]
fn exact_frontier_bound(batch: DirectedNeighborBatch) -> f32 {
    complete_exact_bound(batch.first_dot, batch.unseen_bound)
}

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
            // unseen: the batch itself plus what lies beyond it.
            let bound = exact_frontier_bound(batch);
            if builder.can_terminate(bound) {
                #[cfg(test)]
                counters.record_termination_checkpoint(match batch.source {
                    DirectedNeighborBatchSource::ShellExpand => super::TerminationCheckpoint::Shell,
                    DirectedNeighborBatchSource::PackedChunk0
                    | DirectedNeighborBatchSource::PackedTail => {
                        super::TerminationCheckpoint::PackedPreBatch
                    }
                });
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
                #[cfg(test)]
                counters
                    .record_termination_checkpoint(super::TerminationCheckpoint::PackedPostBatch);
                true
            } else {
                stream.advance_frontier();
                false
            }
        }
        DirectedNeighborFrontier::Exhausted => true,
    }
}

#[cfg(test)]
mod tests {
    use super::{complete_exact_bound, exact_frontier_bound};
    use crate::cube_grid::{DirectedNeighborBatch, DirectedNeighborBatchSource};

    #[test]
    fn packed_exact_bounds_keep_the_post_batch_certificate() {
        for source in [
            DirectedNeighborBatchSource::PackedChunk0,
            DirectedNeighborBatchSource::PackedTail,
        ] {
            for (first_dot, unseen_bound) in [(0.25, 0.75), (0.75, 0.75), (0.75, 0.25)] {
                let batch = DirectedNeighborBatch {
                    n: 2,
                    first_dot,
                    unseen_bound,
                    source,
                };

                assert_eq!(exact_frontier_bound(batch), first_dot.max(unseen_bound));
            }
        }

        // The same composition is used after consuming a packed prefix, where
        // the known remainder is represented by next_dot instead of first_dot.
        for (next_dot, unseen_bound) in [(0.5, 0.75), (0.75, 0.75), (0.75, 0.5)] {
            assert_eq!(
                complete_exact_bound(next_dot, unseen_bound),
                next_dot.max(unseen_bound)
            );
        }
    }
}
