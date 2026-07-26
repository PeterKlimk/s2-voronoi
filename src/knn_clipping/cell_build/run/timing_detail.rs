#[cfg(feature = "timing")]
mod real {
    use crate::cube_grid::DirectedNeighborBatchSource;

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct CellTimingDetail {
        shell_layer_batches: usize,
        shell_layer_slots: usize,
        shell_layer_prefix_consumed: usize,
        shell_midlayer_terminations: usize,
        packed_exact_slots_visited: [usize; 4],
        packed_exact_slots_abandoned: [usize; 4],
        /// Candidates examined after the final polygon-changing constraint.
        neighbors_after_last_progress: usize,
        /// Exhaustion recovery lacks per-candidate clip outcomes, so its tail
        /// is excluded rather than reported as precise.
        progress_tail_valid: bool,
    }

    impl CellTimingDetail {
        #[inline]
        pub(crate) fn record_into(
            &self,
            cell_sub: &mut crate::timing::CellSubAccum,
            neighbors_processed: usize,
        ) {
            cell_sub.add_shell_layer_usage(
                self.shell_layer_batches,
                self.shell_layer_slots,
                self.shell_layer_prefix_consumed,
                self.shell_midlayer_terminations,
            );
            cell_sub.add_packed_batch_usage(
                self.packed_exact_slots_visited,
                self.packed_exact_slots_abandoned,
            );
            cell_sub.add_work_profile(
                neighbors_processed,
                self.neighbors_after_last_progress,
                self.progress_tail_valid,
            );
        }
    }

    pub(crate) struct BuildTimingDetail {
        shell_layer_batches: usize,
        shell_layer_slots: usize,
        shell_layer_prefix_consumed: usize,
        shell_midlayer_terminations: usize,
        packed_exact_batch_usage_counts: [usize; 2],
        packed_exact_slots_visited: [usize; 4],
        packed_exact_slots_abandoned: [usize; 4],
        last_progress_neighbor: usize,
        progress_tail_valid: bool,
        directional_shadow_terminated: bool,
    }

    impl BuildTimingDetail {
        #[inline]
        pub(crate) fn new() -> Self {
            Self {
                shell_layer_batches: 0,
                shell_layer_slots: 0,
                shell_layer_prefix_consumed: 0,
                shell_midlayer_terminations: 0,
                packed_exact_batch_usage_counts: [0; 2],
                packed_exact_slots_visited: [0; 4],
                packed_exact_slots_abandoned: [0; 4],
                last_progress_neighbor: 0,
                progress_tail_valid: true,
                directional_shadow_terminated: false,
            }
        }

        #[inline]
        pub(crate) fn invalidate_progress_tail(&mut self) {
            self.progress_tail_valid = false;
        }

        #[inline]
        pub(crate) fn record_progress(&mut self, neighbors_processed: usize) {
            self.last_progress_neighbor = neighbors_processed;
        }

        #[inline]
        pub(crate) fn record_packed_batch_usage(
            &mut self,
            source: DirectedNeighborBatchSource,
            emitted: usize,
            visited: usize,
        ) {
            debug_assert!(visited <= emitted);
            let stage = match source {
                DirectedNeighborBatchSource::PackedChunk0 => 0,
                DirectedNeighborBatchSource::PackedTail => 1,
                DirectedNeighborBatchSource::ShellExpand => return,
            };
            let first = self.packed_exact_batch_usage_counts[stage] == 0;
            self.packed_exact_batch_usage_counts[stage] += 1;
            let class = stage * 2 + usize::from(!first);
            self.packed_exact_slots_visited[class] += visited;
            self.packed_exact_slots_abandoned[class] += emitted - visited;
        }

        #[inline]
        pub(crate) fn record_shell_batch<const SHELL: bool>(
            &mut self,
            emitted: usize,
            visited: usize,
            terminated: bool,
        ) {
            if SHELL {
                self.shell_layer_batches += 1;
                self.shell_layer_slots += emitted;
                self.shell_layer_prefix_consumed += visited;
                self.shell_midlayer_terminations += (terminated && visited < emitted) as usize;
            }
        }

        #[inline]
        pub(crate) fn directional_shadow_terminated(&self) -> bool {
            self.directional_shadow_terminated
        }

        #[inline]
        pub(crate) fn mark_directional_shadow_terminated(&mut self) {
            self.directional_shadow_terminated = true;
        }

        #[inline]
        pub(crate) fn finish(&self, neighbors_processed: usize) -> CellTimingDetail {
            CellTimingDetail {
                shell_layer_batches: self.shell_layer_batches,
                shell_layer_slots: self.shell_layer_slots,
                shell_layer_prefix_consumed: self.shell_layer_prefix_consumed,
                shell_midlayer_terminations: self.shell_midlayer_terminations,
                packed_exact_slots_visited: self.packed_exact_slots_visited,
                packed_exact_slots_abandoned: self.packed_exact_slots_abandoned,
                neighbors_after_last_progress: neighbors_processed
                    .saturating_sub(self.last_progress_neighbor),
                progress_tail_valid: self.progress_tail_valid,
            }
        }
    }
}

#[cfg(not(feature = "timing"))]
mod stub {
    use crate::cube_grid::DirectedNeighborBatchSource;

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct CellTimingDetail;

    impl CellTimingDetail {
        #[inline(always)]
        pub(crate) fn record_into(
            &self,
            _cell_sub: &mut crate::timing::CellSubAccum,
            _neighbors_processed: usize,
        ) {
        }
    }

    pub(crate) struct BuildTimingDetail;

    impl BuildTimingDetail {
        #[inline(always)]
        pub(crate) fn new() -> Self {
            Self
        }

        #[inline(always)]
        pub(crate) fn invalidate_progress_tail(&mut self) {}

        #[inline(always)]
        pub(crate) fn record_progress(&mut self, _neighbors_processed: usize) {}

        #[inline(always)]
        pub(crate) fn record_packed_batch_usage(
            &mut self,
            _source: DirectedNeighborBatchSource,
            _emitted: usize,
            _visited: usize,
        ) {
        }

        #[inline(always)]
        pub(crate) fn record_shell_batch<const SHELL: bool>(
            &mut self,
            _emitted: usize,
            _visited: usize,
            _terminated: bool,
        ) {
        }

        #[inline(always)]
        pub(crate) fn finish(&self, _neighbors_processed: usize) -> CellTimingDetail {
            CellTimingDetail
        }
    }
}

#[cfg(feature = "timing")]
pub(super) use real::{BuildTimingDetail, CellTimingDetail};
#[cfg(not(feature = "timing"))]
pub(super) use stub::{BuildTimingDetail, CellTimingDetail};
