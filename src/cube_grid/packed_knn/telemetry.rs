#[cfg(feature = "telemetry")]
#[derive(Debug, Clone, Default)]
pub(crate) struct PackedKnnTelemetry {
    pub tail_builds: u64,
    pub keys_materialized: u64,
    pub key_capacity_peak: u64,
    pub tail_possible_queries: u64,
    pub tail_requested_queries: u64,
    pub ring_tail_rescans: u64,
    pub ring_tail_empty_rescans: u64,
    pub ring_tail_dot_evaluations: u64,
    pub center_tail_keys: u64,
    pub unused_center_tail_keys: u64,
    pub center_tail_dot_evaluations: u64,
    pub chunk0_keys: u64,
    pub unused_chunk0_keys: u64,
    pub exact_batch_counts: [u64; 4],
    pub exact_slots_emitted: [u64; 4],
}
#[cfg(not(feature = "telemetry"))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PackedKnnTelemetry;

#[cfg(feature = "telemetry")]
impl PackedKnnTelemetry {
    #[inline]
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
    #[inline]
    pub(crate) fn inc_tail_builds(&mut self) {
        self.tail_builds += 1;
    }
    #[inline]
    pub(crate) fn observe_key_storage(&mut self, added: usize, capacity: usize) {
        self.keys_materialized += added as u64;
        self.key_capacity_peak = self.key_capacity_peak.max(capacity as u64);
    }
    #[inline]
    pub(crate) fn add_tail_possible_queries(&mut self, count: usize) {
        self.tail_possible_queries += count as u64;
    }
    #[inline]
    pub(crate) fn inc_tail_requested_queries(&mut self) {
        self.tail_requested_queries += 1;
    }
    #[inline]
    pub(crate) fn add_ring_tail_rescan(&mut self, empty: bool, dots: usize) {
        self.ring_tail_rescans += 1;
        self.ring_tail_empty_rescans += empty as u64;
        self.ring_tail_dot_evaluations += dots as u64;
    }
    #[inline]
    pub(crate) fn add_center_tail_keys(&mut self, count: usize) {
        self.center_tail_keys += count as u64;
    }
    #[inline]
    pub(crate) fn add_unused_center_tail_keys(&mut self, count: usize) {
        self.unused_center_tail_keys += count as u64;
    }
    #[inline]
    pub(crate) fn add_center_tail_dot_evaluations(&mut self, count: usize) {
        self.center_tail_dot_evaluations += count as u64;
    }
    #[inline]
    pub(crate) fn add_chunk0_keys(&mut self, count: usize) {
        self.chunk0_keys += count as u64;
    }
    #[inline]
    pub(crate) fn add_unused_chunk0_keys(&mut self, count: usize) {
        self.unused_chunk0_keys += count as u64;
    }
    #[inline]
    pub(crate) fn record_exact_batch_emitted(
        &mut self,
        source: super::PackedNeighborBatchSource,
        first: bool,
        slots: usize,
    ) {
        let stage = match source {
            super::PackedNeighborBatchSource::Chunk0 => 0,
            super::PackedNeighborBatchSource::Tail => 1,
        };
        let class = stage * 2 + usize::from(!first);
        self.exact_batch_counts[class] += 1;
        self.exact_slots_emitted[class] += slots as u64;
    }
}
#[cfg(not(feature = "telemetry"))]
impl PackedKnnTelemetry {
    pub(crate) fn clear(&mut self) {}
    pub(crate) fn inc_tail_builds(&mut self) {}
    pub(crate) fn observe_key_storage(&mut self, _added: usize, _capacity: usize) {}
    pub(crate) fn add_tail_possible_queries(&mut self, _count: usize) {}
    pub(crate) fn inc_tail_requested_queries(&mut self) {}
    pub(crate) fn add_ring_tail_rescan(&mut self, _empty: bool, _dots: usize) {}
    pub(crate) fn add_center_tail_keys(&mut self, _count: usize) {}
    pub(crate) fn add_center_tail_dot_evaluations(&mut self, _count: usize) {}
    pub(crate) fn add_chunk0_keys(&mut self, _count: usize) {}
}
