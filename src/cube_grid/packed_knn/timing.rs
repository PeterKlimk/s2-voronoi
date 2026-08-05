use std::time::Duration;

/// Fine-grained timing breakdown for the packed-kNN per-cell-group flow.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub(crate) struct PackedKnnTimings {
    pub setup: Duration,
    pub query_cache: Duration,
    pub security_thresholds: Duration,
    pub center_pass: Duration,
    pub ring_thresholds: Duration,
    pub ring_pass: Duration,
    pub ring_fallback: Duration,
    pub select_prep: Duration,
    pub select_query_prep: Duration,
    pub select_partition: Duration,
    pub select_sort: Duration,
    pub select_scatter: Duration,
    /// Number of times tail candidates were built (per query, but counted at most once per group).
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
    /// Exact packed batches and slots emitted, split into chunk0-first,
    /// chunk0-later, tail-first, and tail-later classes.
    pub exact_batch_counts: [u64; 4],
    pub exact_slots_emitted: [u64; 4],
}

#[cfg(feature = "timing")]
impl Default for PackedKnnTimings {
    fn default() -> Self {
        Self {
            setup: Duration::ZERO,
            query_cache: Duration::ZERO,
            security_thresholds: Duration::ZERO,
            center_pass: Duration::ZERO,
            ring_thresholds: Duration::ZERO,
            ring_pass: Duration::ZERO,
            ring_fallback: Duration::ZERO,
            select_prep: Duration::ZERO,
            select_query_prep: Duration::ZERO,
            select_partition: Duration::ZERO,
            select_sort: Duration::ZERO,
            select_scatter: Duration::ZERO,
            tail_builds: 0,
            keys_materialized: 0,
            key_capacity_peak: 0,
            tail_possible_queries: 0,
            tail_requested_queries: 0,
            ring_tail_rescans: 0,
            ring_tail_empty_rescans: 0,
            ring_tail_dot_evaluations: 0,
            center_tail_keys: 0,
            unused_center_tail_keys: 0,
            center_tail_dot_evaluations: 0,
            chunk0_keys: 0,
            unused_chunk0_keys: 0,
            exact_batch_counts: [0; 4],
            exact_slots_emitted: [0; 4],
        }
    }
}

/// Dummy timings when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PackedKnnTimings;

#[cfg(feature = "timing")]
impl PackedKnnTimings {
    #[inline]
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub(crate) fn add_setup(&mut self, d: Duration) {
        self.setup += d;
    }

    #[inline]
    pub(crate) fn add_query_cache(&mut self, d: Duration) {
        self.query_cache += d;
    }

    #[inline]
    pub(crate) fn add_security_thresholds(&mut self, d: Duration) {
        self.security_thresholds += d;
    }

    #[inline]
    pub(crate) fn add_center_pass(&mut self, d: Duration) {
        self.center_pass += d;
    }

    #[inline]
    pub(crate) fn add_ring_thresholds(&mut self, d: Duration) {
        self.ring_thresholds += d;
    }

    #[inline]
    pub(crate) fn add_ring_pass(&mut self, d: Duration) {
        self.ring_pass += d;
    }

    #[inline]
    pub(crate) fn add_ring_fallback(&mut self, d: Duration) {
        self.ring_fallback += d;
    }

    #[inline]
    pub(crate) fn add_select_prep(&mut self, d: Duration) {
        self.select_prep += d;
    }

    #[inline]
    pub(crate) fn add_select_query_prep(&mut self, d: Duration) {
        self.select_query_prep += d;
    }

    #[inline]
    pub(crate) fn add_select_partition(&mut self, d: Duration) {
        self.select_partition += d;
    }

    #[inline]
    pub(crate) fn add_select_sort_sized(&mut self, d: Duration, _n: usize) {
        self.select_sort += d;
    }

    #[inline]
    pub(crate) fn add_select_scatter(&mut self, d: Duration) {
        self.select_scatter += d;
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
    pub(crate) fn add_ring_tail_rescan(&mut self, empty: bool, dot_evaluations: usize) {
        self.ring_tail_rescans += 1;
        self.ring_tail_empty_rescans += empty as u64;
        self.ring_tail_dot_evaluations += dot_evaluations as u64;
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

    #[inline]
    pub(crate) fn total(&self) -> Duration {
        self.setup
            + self.query_cache
            + self.security_thresholds
            + self.center_pass
            + self.ring_thresholds
            + self.ring_pass
            + self.ring_fallback
            + self.select_prep
            + self.select_query_prep
            + self.select_partition
            + self.select_sort
            + self.select_scatter
    }
}

#[cfg(not(feature = "timing"))]
impl PackedKnnTimings {
    pub(crate) fn clear(&mut self) {}

    pub(crate) fn add_setup(&mut self, _d: Duration) {}
    pub(crate) fn add_query_cache(&mut self, _d: Duration) {}
    pub(crate) fn add_security_thresholds(&mut self, _d: Duration) {}
    pub(crate) fn add_center_pass(&mut self, _d: Duration) {}
    pub(crate) fn add_ring_thresholds(&mut self, _d: Duration) {}
    pub(crate) fn add_ring_pass(&mut self, _d: Duration) {}
    pub(crate) fn add_ring_fallback(&mut self, _d: Duration) {}
    pub(crate) fn add_select_prep(&mut self, _d: Duration) {}
    pub(crate) fn add_select_query_prep(&mut self, _d: Duration) {}
    pub(crate) fn add_select_partition(&mut self, _d: Duration) {}
    pub(crate) fn add_select_sort_sized(&mut self, _d: Duration, _n: usize) {}
    pub(crate) fn add_select_scatter(&mut self, _d: Duration) {}
    pub(crate) fn inc_tail_builds(&mut self) {}
    pub(crate) fn observe_key_storage(&mut self, _added: usize, _capacity: usize) {}
    pub(crate) fn add_tail_possible_queries(&mut self, _count: usize) {}
    pub(crate) fn inc_tail_requested_queries(&mut self) {}
    pub(crate) fn add_ring_tail_rescan(&mut self, _empty: bool, _dot_evaluations: usize) {}
    pub(crate) fn add_center_tail_keys(&mut self, _count: usize) {}
    pub(crate) fn add_center_tail_dot_evaluations(&mut self, _count: usize) {}
    pub(crate) fn add_chunk0_keys(&mut self, _count: usize) {}
}
