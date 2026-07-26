use super::KnnCellStage;
use std::time::Duration;

use crate::cube_grid::{packed_knn::PackedKnnTimings, CubeMapGridBuildTimings};

mod report;

/// Timer that tracks elapsed time when timing is enabled.
pub(crate) struct Timer(std::time::Instant);

impl Timer {
    #[inline]
    pub(crate) fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub(crate) fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Timer optimized for sequential sub-phase timing: each `lap()` uses a single `Instant::now()`.
pub(crate) struct LapTimer(std::time::Instant);

impl LapTimer {
    #[inline]
    pub(crate) fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub(crate) fn lap(&mut self) -> Duration {
        let now = std::time::Instant::now();
        let d = now.duration_since(self.0);
        self.0 = now;
        d
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StageCounts {
    packed_chunk0: u64,
    packed_tail: u64,
    shell_expand: u64,
}

impl StageCounts {
    #[inline]
    fn add(&mut self, stage: KnnCellStage) {
        match stage {
            KnnCellStage::PackedChunk0 => self.packed_chunk0 += 1,
            KnnCellStage::PackedTail => self.packed_tail += 1,
            KnnCellStage::ShellExpand => self.shell_expand += 1,
        }
    }

    #[inline]
    fn merge(&mut self, other: &StageCounts) {
        self.packed_chunk0 += other.packed_chunk0;
        self.packed_tail += other.packed_tail;
        self.shell_expand += other.shell_expand;
    }
}

// Keep the common range exact and compress only genuinely long tails. Timing
// builds can then characterize ordinary workloads precisely without allocating
// one sample per cell.
const WORK_EXACT_BUCKETS: usize = 256;
const WORK_LOG_BUCKETS: usize = u64::BITS as usize - 8;
const WORK_BUCKETS: usize = WORK_EXACT_BUCKETS + WORK_LOG_BUCKETS;

#[derive(Debug, Clone)]
struct WorkHistogram {
    buckets: [u64; WORK_BUCKETS],
    samples: u64,
    max: u64,
}

impl Default for WorkHistogram {
    fn default() -> Self {
        Self {
            buckets: [0; WORK_BUCKETS],
            samples: 0,
            max: 0,
        }
    }
}

impl WorkHistogram {
    #[inline]
    fn bucket(value: u64) -> usize {
        if value < WORK_EXACT_BUCKETS as u64 {
            value as usize
        } else {
            let log2 = u64::BITS as usize - 1 - value.leading_zeros() as usize;
            WORK_EXACT_BUCKETS + log2 - 8
        }
    }

    #[inline]
    fn bucket_lower_bound(bucket: usize) -> u64 {
        if bucket < WORK_EXACT_BUCKETS {
            bucket as u64
        } else {
            1_u64 << (8 + bucket - WORK_EXACT_BUCKETS)
        }
    }

    #[inline]
    fn record(&mut self, value: usize) {
        let value = value as u64;
        self.buckets[Self::bucket(value)] += 1;
        self.samples += 1;
        self.max = self.max.max(value);
    }

    #[inline]
    fn merge(&mut self, other: &Self) {
        for (dst, src) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            *dst += *src;
        }
        self.samples += other.samples;
        self.max = self.max.max(other.max);
    }

    fn quantile_lower_bound(&self, numerator: u64, denominator: u64) -> u64 {
        if self.samples == 0 {
            return 0;
        }
        let rank = self
            .samples
            .saturating_mul(numerator)
            .saturating_add(denominator - 1)
            / denominator;
        let mut cumulative = 0_u64;
        for (bucket, count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= rank.max(1) {
                return Self::bucket_lower_bound(bucket);
            }
        }
        self.max
    }

    /// Conservative count when `threshold` cuts through a logarithmic bucket:
    /// only buckets whose lower bound meets the threshold are included.
    fn count_at_least_lower_bound(&self, threshold: u64) -> u64 {
        self.buckets
            .iter()
            .enumerate()
            .filter(|(bucket, _)| Self::bucket_lower_bound(*bucket) >= threshold)
            .map(|(_, count)| count)
            .sum()
    }

    fn summary(&self) -> WorkDistribution {
        let p50_bucket_lower = self.quantile_lower_bound(1, 2);
        let relative_median_base = p50_bucket_lower.max(1);
        let relative_count = |factor: u64| {
            self.count_at_least_lower_bound(relative_median_base.saturating_mul(factor))
        };
        WorkDistribution {
            samples: self.samples,
            p50_bucket_lower,
            p90_bucket_lower: self.quantile_lower_bound(9, 10),
            p99_bucket_lower: self.quantile_lower_bound(99, 100),
            p999_bucket_lower: self.quantile_lower_bound(999, 1000),
            max: self.max,
            relative_median_base,
            count_ge_4x_median_lower: relative_count(4),
            count_ge_16x_median_lower: relative_count(16),
            count_ge_64x_median_lower: relative_count(64),
        }
    }
}

/// Scale-relative per-cell work shape. Quantiles are exact below 256 and the
/// lower bound of a power-of-two bucket above that. Relative-tail counts are
/// conservative when their threshold cuts through such a bucket.
#[derive(Debug, Clone, Default)]
pub(crate) struct WorkDistribution {
    pub samples: u64,
    pub p50_bucket_lower: u64,
    pub p90_bucket_lower: u64,
    pub p99_bucket_lower: u64,
    pub p999_bucket_lower: u64,
    pub max: u64,
    /// p50 bucket lower bound, floored at one so a zero-progress median still
    /// has useful positive relative-tail thresholds.
    pub relative_median_base: u64,
    pub count_ge_4x_median_lower: u64,
    pub count_ge_16x_median_lower: u64,
    pub count_ge_64x_median_lower: u64,
}

/// Per-cell-group timing totals aggregated across all shards.
#[derive(Debug, Clone, Default)]
pub(crate) struct CellSubPhases {
    pub knn_query: Duration,
    pub packed_knn: Duration,
    pub packed_setup: Duration,
    pub packed_security: Duration,
    pub packed_center_pass: Duration,
    pub packed_ring_thresholds: Duration,
    pub packed_ring_pass: Duration,
    pub packed_ring_fallback: Duration,
    pub packed_select_prep: Duration,
    pub packed_select_partition: Duration,
    pub packed_select_sort: Duration,
    pub packed_select_scatter: Duration,
    pub clipping: Duration,
    pub certification: Duration,
    pub key_dedup: Duration,
    pub edge_collect: Duration,
    pub edge_resolve: Duration,
    pub edge_emit: Duration,
    pub cells_knn_exhausted: u64,
    pub cells_packed_tail_used: u64,
    pub cells_packed_safe_exhausted: u64,
    pub cells_used_knn: u64,
    pub fallback_projection: u64,
    pub fallback_polygon_cap: u64,
    pub fallback_all_constraints: u64,
    pub packed_tail_builds: u64,
    pub packed_keys_materialized: u64,
    pub packed_key_capacity_peak: u64,
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
    /// Counts are split into chunk0-first, chunk0-later, tail-first, and
    /// tail-later exact packed batches.
    pub packed_exact_batch_counts: [u64; 4],
    pub packed_exact_slots_emitted: [u64; 4],
    pub packed_exact_slots_visited: [u64; 4],
    pub packed_exact_slots_abandoned: [u64; 4],
    pub shell_layer_batches: u64,
    pub shell_layer_slots: u64,
    pub shell_layer_prefix_consumed: u64,
    pub shell_midlayer_terminations: u64,
    /// Sum of neighbors processed before termination across all cells
    /// (mean = total / n; input for the grid-density tuning model).
    pub neighbors_processed_total: u64,
    pub neighbors_processed_max: u64,
    /// Sum of final cell degrees across all cells. Used with
    /// `neighbors_processed_total` to size examine-and-reject headroom.
    pub final_edges_total: u64,
    pub final_edges_max: u64,
    /// Total examined candidates per cell, reported relative to this run's
    /// median so expected growth with input size is not itself pathological.
    pub candidate_work: WorkDistribution,
    /// Examined candidates after the final polygon-changing constraint.
    pub no_progress_tail: WorkDistribution,
    /// Cells omitted from `no_progress_tail` because exhaustion recovery
    /// batches constraints without retaining individual clip outcomes.
    pub no_progress_tail_excluded: u64,
    /// Shadow direction-aware batch-skip probe counters. These do not affect
    /// construction; they estimate candidates a conservative known-batch
    /// directional certificate could skip.
    pub directional_shadow_checks: u64,
    pub directional_shadow_candidate_tests: u64,
    pub directional_shadow_hits: u64,
    pub directional_shadow_saved: u64,
    pub directional_support_candidate_tests: u64,
    pub directional_support_hits: u64,
    pub directional_support_saved: u64,
    pub directional_support_false_positive_hits: u64,
}

/// Fine-grained dedup timing and a few size counters.
#[derive(Debug, Clone, Default)]
pub(crate) struct DedupSubPhases {
    pub bookkeeping: Duration,
    pub edge_check_overflow: Duration,
    pub deferred_patching: Duration,
    pub finalize_shards: Duration,
    pub concat_vertices: Duration,
    pub emit_cell_prefixes: Duration,
    pub incidence_summary: Duration,
    pub scatter_cell_indices: Duration,
    pub patch_reference_overrides: Duration,
    pub exact_zero_hints: Duration,
    pub shard_order_descents: u64,
    pub shard_order_pairs: u64,
    pub shard_order_abs_delta: u64,
    pub scatter_by_shard: bool,
    pub triplet_keys: u64,
    pub edge_mismatches_count: u64,
    pub primary_cell_references: u64,
    pub reference_overrides: u64,
}

/// Accumulator for cell sub-phase timings (used per-bin, then merged).
#[derive(Clone, Default)]
pub(crate) struct CellSubAccum {
    knn_query: Duration,
    packed_knn: Duration,
    packed_setup: Duration,
    packed_security: Duration,
    packed_center_pass: Duration,
    packed_ring_thresholds: Duration,
    packed_ring_pass: Duration,
    packed_ring_fallback: Duration,
    packed_select_prep: Duration,
    packed_select_partition: Duration,
    packed_select_sort: Duration,
    packed_select_scatter: Duration,
    clipping: Duration,
    certification: Duration,
    key_dedup: Duration,
    edge_collect: Duration,
    edge_resolve: Duration,
    edge_emit: Duration,
    stage_counts: StageCounts,
    cells_knn_exhausted: u64,
    cells_packed_tail_used: u64,
    cells_packed_safe_exhausted: u64,
    cells_used_knn: u64,
    fallback_projection: u64,
    fallback_polygon_cap: u64,
    fallback_all_constraints: u64,
    packed_tail_builds: u64,
    packed_keys_materialized: u64,
    packed_key_capacity_peak: u64,
    tail_possible_queries: u64,
    tail_requested_queries: u64,
    ring_tail_rescans: u64,
    ring_tail_empty_rescans: u64,
    ring_tail_dot_evaluations: u64,
    center_tail_keys: u64,
    unused_center_tail_keys: u64,
    center_tail_dot_evaluations: u64,
    chunk0_keys: u64,
    unused_chunk0_keys: u64,
    packed_exact_batch_counts: [u64; 4],
    packed_exact_slots_emitted: [u64; 4],
    packed_exact_slots_visited: [u64; 4],
    packed_exact_slots_abandoned: [u64; 4],
    shell_layer_batches: u64,
    shell_layer_slots: u64,
    shell_layer_prefix_consumed: u64,
    shell_midlayer_terminations: u64,
    neighbors_processed_total: u64,
    neighbors_processed_max: u64,
    final_edges_total: u64,
    final_edges_max: u64,
    candidate_work: WorkHistogram,
    no_progress_tail: WorkHistogram,
    no_progress_tail_excluded: u64,
    directional_shadow_checks: u64,
    directional_shadow_candidate_tests: u64,
    directional_shadow_hits: u64,
    directional_shadow_saved: u64,
    directional_support_candidate_tests: u64,
    directional_support_hits: u64,
    directional_support_saved: u64,
    directional_support_false_positive_hits: u64,
}

impl CellSubAccum {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn add_knn(&mut self, d: Duration) {
        self.knn_query += d;
    }

    #[inline]
    pub(crate) fn add_packed_knn(&mut self, d: Duration) {
        self.packed_knn += d;
    }

    #[inline]
    pub(crate) fn add_packed_knn_breakdown(&mut self, timings: &PackedKnnTimings) {
        self.packed_setup += timings.setup + timings.query_cache;
        self.packed_security += timings.security_thresholds;
        self.packed_center_pass += timings.center_pass;
        self.packed_ring_thresholds += timings.ring_thresholds;
        self.packed_ring_pass += timings.ring_pass;
        self.packed_ring_fallback += timings.ring_fallback;
        self.packed_select_prep += timings.select_prep + timings.select_query_prep;
        self.packed_select_partition += timings.select_partition;
        self.packed_select_sort += timings.select_sort;
        self.packed_select_scatter += timings.select_scatter;
        self.packed_tail_builds += timings.tail_builds;
        self.packed_keys_materialized += timings.keys_materialized;
        self.packed_key_capacity_peak =
            self.packed_key_capacity_peak.max(timings.key_capacity_peak);
        self.tail_possible_queries += timings.tail_possible_queries;
        self.tail_requested_queries += timings.tail_requested_queries;
        self.ring_tail_rescans += timings.ring_tail_rescans;
        self.ring_tail_empty_rescans += timings.ring_tail_empty_rescans;
        self.ring_tail_dot_evaluations += timings.ring_tail_dot_evaluations;
        self.center_tail_keys += timings.center_tail_keys;
        self.unused_center_tail_keys += timings.unused_center_tail_keys;
        self.center_tail_dot_evaluations += timings.center_tail_dot_evaluations;
        self.chunk0_keys += timings.chunk0_keys;
        self.unused_chunk0_keys += timings.unused_chunk0_keys;
        for class in 0..4 {
            self.packed_exact_batch_counts[class] += timings.exact_batch_counts[class];
            self.packed_exact_slots_emitted[class] += timings.exact_slots_emitted[class];
        }
    }

    #[inline]
    pub(crate) fn add_clip(&mut self, d: Duration) {
        self.clipping += d;
    }

    #[inline]
    pub(crate) fn add_cert(&mut self, d: Duration) {
        self.certification += d;
    }

    #[inline]
    pub(crate) fn add_key_dedup(&mut self, d: Duration) {
        self.key_dedup += d;
    }

    #[inline]
    pub(crate) fn add_edge_collect(&mut self, d: Duration) {
        self.edge_collect += d;
    }

    #[inline]
    pub(crate) fn add_edge_resolve(&mut self, d: Duration) {
        self.edge_resolve += d;
    }

    #[inline]
    pub(crate) fn add_edge_emit(&mut self, d: Duration) {
        self.edge_emit += d;
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_cell_stage(
        &mut self,
        stage: KnnCellStage,
        knn_exhausted: bool,
        neighbors_processed: usize,
        final_edges: usize,
        packed_tail_used: bool,
        packed_safe_exhausted: bool,
        used_knn: bool,
        _incoming_edgechecks: usize,
        _edgecheck_seed_clips: usize,
    ) {
        self.stage_counts.add(stage);
        self.cells_knn_exhausted += knn_exhausted as u64;
        self.cells_packed_tail_used += packed_tail_used as u64;
        self.cells_packed_safe_exhausted += packed_safe_exhausted as u64;
        self.cells_used_knn += used_knn as u64;
        self.neighbors_processed_total += neighbors_processed as u64;
        self.neighbors_processed_max = self.neighbors_processed_max.max(neighbors_processed as u64);
        self.final_edges_total += final_edges as u64;
        self.final_edges_max = self.final_edges_max.max(final_edges as u64);
    }

    #[inline]
    pub(crate) fn add_work_profile(
        &mut self,
        candidates: usize,
        candidates_after_last_progress: usize,
        progress_tail_valid: bool,
    ) {
        self.candidate_work.record(candidates);
        if progress_tail_valid {
            self.no_progress_tail.record(candidates_after_last_progress);
        } else {
            self.no_progress_tail_excluded += 1;
        }
    }

    #[inline]
    pub(crate) fn add_fallbacks(
        &mut self,
        projection: usize,
        polygon_cap: usize,
        all_constraints: usize,
    ) {
        self.fallback_projection += projection as u64;
        self.fallback_polygon_cap += polygon_cap as u64;
        self.fallback_all_constraints += all_constraints as u64;
    }

    #[inline]
    pub(crate) fn add_shell_layer_usage(
        &mut self,
        batches: usize,
        slots: usize,
        prefix_consumed: usize,
        midlayer_terminations: usize,
    ) {
        self.shell_layer_batches += batches as u64;
        self.shell_layer_slots += slots as u64;
        self.shell_layer_prefix_consumed += prefix_consumed as u64;
        self.shell_midlayer_terminations += midlayer_terminations as u64;
    }

    #[inline]
    pub(crate) fn add_packed_batch_usage(&mut self, visited: [usize; 4], abandoned: [usize; 4]) {
        for class in 0..4 {
            self.packed_exact_slots_visited[class] += visited[class] as u64;
            self.packed_exact_slots_abandoned[class] += abandoned[class] as u64;
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_directional_shadow(
        &mut self,
        checks: usize,
        candidate_tests: usize,
        hits: usize,
        saved: usize,
        support_candidate_tests: usize,
        support_hits: usize,
        support_saved: usize,
        support_false_positive_hits: usize,
    ) {
        self.directional_shadow_checks += checks as u64;
        self.directional_shadow_candidate_tests += candidate_tests as u64;
        self.directional_shadow_hits += hits as u64;
        self.directional_shadow_saved += saved as u64;
        self.directional_support_candidate_tests += support_candidate_tests as u64;
        self.directional_support_hits += support_hits as u64;
        self.directional_support_saved += support_saved as u64;
        self.directional_support_false_positive_hits += support_false_positive_hits as u64;
    }

    #[inline]
    pub(crate) fn merge(&mut self, other: &CellSubAccum) {
        self.knn_query += other.knn_query;
        self.packed_knn += other.packed_knn;
        self.packed_setup += other.packed_setup;
        self.packed_security += other.packed_security;
        self.packed_center_pass += other.packed_center_pass;
        self.packed_ring_thresholds += other.packed_ring_thresholds;
        self.packed_ring_pass += other.packed_ring_pass;
        self.packed_ring_fallback += other.packed_ring_fallback;
        self.packed_select_prep += other.packed_select_prep;
        self.packed_select_partition += other.packed_select_partition;
        self.packed_select_sort += other.packed_select_sort;
        self.packed_select_scatter += other.packed_select_scatter;
        self.clipping += other.clipping;
        self.certification += other.certification;
        self.key_dedup += other.key_dedup;
        self.edge_collect += other.edge_collect;
        self.edge_resolve += other.edge_resolve;
        self.edge_emit += other.edge_emit;
        self.stage_counts.merge(&other.stage_counts);
        self.cells_knn_exhausted += other.cells_knn_exhausted;
        self.cells_packed_tail_used += other.cells_packed_tail_used;
        self.cells_packed_safe_exhausted += other.cells_packed_safe_exhausted;
        self.cells_used_knn += other.cells_used_knn;
        self.fallback_projection += other.fallback_projection;
        self.fallback_polygon_cap += other.fallback_polygon_cap;
        self.fallback_all_constraints += other.fallback_all_constraints;
        self.packed_tail_builds += other.packed_tail_builds;
        self.packed_keys_materialized += other.packed_keys_materialized;
        self.packed_key_capacity_peak = self
            .packed_key_capacity_peak
            .max(other.packed_key_capacity_peak);
        self.tail_possible_queries += other.tail_possible_queries;
        self.tail_requested_queries += other.tail_requested_queries;
        self.ring_tail_rescans += other.ring_tail_rescans;
        self.ring_tail_empty_rescans += other.ring_tail_empty_rescans;
        self.ring_tail_dot_evaluations += other.ring_tail_dot_evaluations;
        self.center_tail_keys += other.center_tail_keys;
        self.unused_center_tail_keys += other.unused_center_tail_keys;
        self.center_tail_dot_evaluations += other.center_tail_dot_evaluations;
        self.chunk0_keys += other.chunk0_keys;
        self.unused_chunk0_keys += other.unused_chunk0_keys;
        for class in 0..4 {
            self.packed_exact_batch_counts[class] += other.packed_exact_batch_counts[class];
            self.packed_exact_slots_emitted[class] += other.packed_exact_slots_emitted[class];
            self.packed_exact_slots_visited[class] += other.packed_exact_slots_visited[class];
            self.packed_exact_slots_abandoned[class] += other.packed_exact_slots_abandoned[class];
        }
        self.shell_layer_batches += other.shell_layer_batches;
        self.shell_layer_slots += other.shell_layer_slots;
        self.shell_layer_prefix_consumed += other.shell_layer_prefix_consumed;
        self.shell_midlayer_terminations += other.shell_midlayer_terminations;
        self.neighbors_processed_total += other.neighbors_processed_total;
        self.neighbors_processed_max = self
            .neighbors_processed_max
            .max(other.neighbors_processed_max);
        self.final_edges_total += other.final_edges_total;
        self.final_edges_max = self.final_edges_max.max(other.final_edges_max);
        self.candidate_work.merge(&other.candidate_work);
        self.no_progress_tail.merge(&other.no_progress_tail);
        self.no_progress_tail_excluded += other.no_progress_tail_excluded;
        self.directional_shadow_checks += other.directional_shadow_checks;
        self.directional_shadow_candidate_tests += other.directional_shadow_candidate_tests;
        self.directional_shadow_hits += other.directional_shadow_hits;
        self.directional_shadow_saved += other.directional_shadow_saved;
        self.directional_support_candidate_tests += other.directional_support_candidate_tests;
        self.directional_support_hits += other.directional_support_hits;
        self.directional_support_saved += other.directional_support_saved;
        self.directional_support_false_positive_hits +=
            other.directional_support_false_positive_hits;
    }

    #[inline]
    pub(crate) fn into_sub_phases(self) -> CellSubPhases {
        for class in 0..4 {
            debug_assert_eq!(
                self.packed_exact_slots_emitted[class],
                self.packed_exact_slots_visited[class] + self.packed_exact_slots_abandoned[class],
                "every emitted exact packed slot must be visited or abandoned"
            );
        }
        CellSubPhases {
            knn_query: self.knn_query,
            packed_knn: self.packed_knn,
            packed_setup: self.packed_setup,
            packed_security: self.packed_security,
            packed_center_pass: self.packed_center_pass,
            packed_ring_thresholds: self.packed_ring_thresholds,
            packed_ring_pass: self.packed_ring_pass,
            packed_ring_fallback: self.packed_ring_fallback,
            packed_select_prep: self.packed_select_prep,
            packed_select_partition: self.packed_select_partition,
            packed_select_sort: self.packed_select_sort,
            packed_select_scatter: self.packed_select_scatter,
            clipping: self.clipping,
            certification: self.certification,
            key_dedup: self.key_dedup,
            edge_collect: self.edge_collect,
            edge_resolve: self.edge_resolve,
            edge_emit: self.edge_emit,
            cells_knn_exhausted: self.cells_knn_exhausted,
            cells_packed_tail_used: self.cells_packed_tail_used,
            cells_packed_safe_exhausted: self.cells_packed_safe_exhausted,
            cells_used_knn: self.cells_used_knn,
            fallback_projection: self.fallback_projection,
            fallback_polygon_cap: self.fallback_polygon_cap,
            fallback_all_constraints: self.fallback_all_constraints,
            packed_tail_builds: self.packed_tail_builds,
            packed_keys_materialized: self.packed_keys_materialized,
            packed_key_capacity_peak: self.packed_key_capacity_peak,
            tail_possible_queries: self.tail_possible_queries,
            tail_requested_queries: self.tail_requested_queries,
            ring_tail_rescans: self.ring_tail_rescans,
            ring_tail_empty_rescans: self.ring_tail_empty_rescans,
            ring_tail_dot_evaluations: self.ring_tail_dot_evaluations,
            center_tail_keys: self.center_tail_keys,
            unused_center_tail_keys: self.unused_center_tail_keys,
            center_tail_dot_evaluations: self.center_tail_dot_evaluations,
            chunk0_keys: self.chunk0_keys,
            unused_chunk0_keys: self.unused_chunk0_keys,
            packed_exact_batch_counts: self.packed_exact_batch_counts,
            packed_exact_slots_emitted: self.packed_exact_slots_emitted,
            packed_exact_slots_visited: self.packed_exact_slots_visited,
            packed_exact_slots_abandoned: self.packed_exact_slots_abandoned,
            shell_layer_batches: self.shell_layer_batches,
            shell_layer_slots: self.shell_layer_slots,
            shell_layer_prefix_consumed: self.shell_layer_prefix_consumed,
            shell_midlayer_terminations: self.shell_midlayer_terminations,
            neighbors_processed_total: self.neighbors_processed_total,
            neighbors_processed_max: self.neighbors_processed_max,
            final_edges_total: self.final_edges_total,
            final_edges_max: self.final_edges_max,
            candidate_work: self.candidate_work.summary(),
            no_progress_tail: self.no_progress_tail.summary(),
            no_progress_tail_excluded: self.no_progress_tail_excluded,
            directional_shadow_checks: self.directional_shadow_checks,
            directional_shadow_candidate_tests: self.directional_shadow_candidate_tests,
            directional_shadow_hits: self.directional_shadow_hits,
            directional_shadow_saved: self.directional_shadow_saved,
            directional_support_candidate_tests: self.directional_support_candidate_tests,
            directional_support_hits: self.directional_support_hits,
            directional_support_saved: self.directional_support_saved,
            directional_support_false_positive_hits: self.directional_support_false_positive_hits,
        }
    }
}

/// Phase-level timings for a full Voronoi run.
#[derive(Debug, Clone)]
pub(crate) struct PhaseTimings {
    pub total: Duration,
    pub preprocess: Duration,
    pub weld_pairs: u64,
    pub weld_pair_capacity: u64,
    pub knn_build: Duration,
    pub knn_build_sub: Option<CubeMapGridBuildTimings>,
    pub cell_construction: Duration,
    pub cell_sub: CellSubPhases,
    pub dedup: Duration,
    pub dedup_sub: DedupSubPhases,
    pub edge_reconcile: Duration,
    pub merge_safety_scan_cells: u64,
    pub merge_safety_global_fallbacks: u64,
    pub assemble: Duration,
    /// Query-grid shape: resolution, max cell occupancy, and whether the
    /// occupancy-feedback rebuild fired.
    pub grid_res: usize,
    pub grid_max_occupancy: u64,
    pub grid_rebuilt: bool,
    pub resolution_drift_fallback: bool,
    pub resolution_reconcile_scan_cells: u64,
    pub resolution_rebuild_scan_cells: u64,
    pub resolution_hint_cells: u64,
    pub resolution_hinted_candidates: u64,
    pub resolution_detected_edges: u64,
}

/// Builder for collecting phase timings.
pub(crate) struct TimingBuilder {
    t_start: std::time::Instant,
    preprocess: Duration,
    weld_pairs: u64,
    weld_pair_capacity: u64,
    knn_build: Duration,
    knn_build_sub: Option<CubeMapGridBuildTimings>,
    cell_construction: Duration,
    cell_sub: CellSubPhases,
    dedup: Duration,
    dedup_sub: DedupSubPhases,
    edge_reconcile: Duration,
    merge_safety_scan_cells: u64,
    merge_safety_global_fallbacks: u64,
    assemble: Duration,
    grid_res: usize,
    grid_max_occupancy: u64,
    grid_rebuilt: bool,
    resolution_drift_fallback: bool,
    resolution_reconcile_scan_cells: u64,
    resolution_rebuild_scan_cells: u64,
    resolution_hint_cells: u64,
    resolution_hinted_candidates: u64,
    resolution_detected_edges: u64,
}

impl TimingBuilder {
    pub(crate) fn new() -> Self {
        Self {
            t_start: std::time::Instant::now(),
            preprocess: Duration::ZERO,
            weld_pairs: 0,
            weld_pair_capacity: 0,
            knn_build: Duration::ZERO,
            knn_build_sub: None,
            cell_construction: Duration::ZERO,
            cell_sub: CellSubPhases::default(),
            dedup: Duration::ZERO,
            dedup_sub: DedupSubPhases::default(),
            edge_reconcile: Duration::ZERO,
            merge_safety_scan_cells: 0,
            merge_safety_global_fallbacks: 0,
            assemble: Duration::ZERO,
            grid_res: 0,
            grid_max_occupancy: 0,
            grid_rebuilt: false,
            resolution_drift_fallback: false,
            resolution_reconcile_scan_cells: 0,
            resolution_rebuild_scan_cells: 0,
            resolution_hint_cells: 0,
            resolution_hinted_candidates: 0,
            resolution_detected_edges: 0,
        }
    }

    pub(crate) fn set_grid_stats(&mut self, res: usize, max_occupancy: u64, rebuilt: bool) {
        self.grid_res = res;
        self.grid_max_occupancy = max_occupancy;
        self.grid_rebuilt = rebuilt;
    }

    pub(crate) fn set_preprocess(&mut self, d: Duration) {
        self.preprocess = d;
    }

    pub(crate) fn set_weld_pair_stats(&mut self, len: usize, capacity: usize) {
        self.weld_pairs = len as u64;
        self.weld_pair_capacity = capacity as u64;
    }

    pub(crate) fn set_knn_build(&mut self, d: Duration) {
        self.knn_build = d;
    }

    pub(crate) fn add_knn_build(&mut self, d: Duration) {
        self.knn_build += d;
    }

    pub(crate) fn set_knn_build_sub(&mut self, sub: CubeMapGridBuildTimings) {
        self.knn_build_sub = Some(sub);
    }

    pub(crate) fn set_cell_construction(&mut self, d: Duration, sub: CellSubPhases) {
        self.cell_construction = d;
        self.cell_sub = sub;
    }

    pub(crate) fn set_dedup(&mut self, d: Duration, sub: DedupSubPhases) {
        self.dedup = d;
        self.dedup_sub = sub;
    }

    pub(crate) fn set_edge_reconcile(
        &mut self,
        d: Duration,
        merge_safety_scan_cells: usize,
        merge_safety_global_fallbacks: usize,
    ) {
        self.edge_reconcile = d;
        self.merge_safety_scan_cells = merge_safety_scan_cells as u64;
        self.merge_safety_global_fallbacks = merge_safety_global_fallbacks as u64;
    }

    pub(crate) fn set_assemble(&mut self, d: Duration) {
        self.assemble = d;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_output_resolution_discovery(
        &mut self,
        drift_fallback: bool,
        reconcile_scan_cells: usize,
        rebuild_scan_cells: usize,
        hint_cells: usize,
        hinted_candidates: usize,
        detected_edges: usize,
    ) {
        self.resolution_drift_fallback = drift_fallback;
        self.resolution_reconcile_scan_cells = reconcile_scan_cells as u64;
        self.resolution_rebuild_scan_cells = rebuild_scan_cells as u64;
        self.resolution_hint_cells = hint_cells as u64;
        self.resolution_hinted_candidates = hinted_candidates as u64;
        self.resolution_detected_edges = detected_edges as u64;
    }

    pub(crate) fn finish(self) -> PhaseTimings {
        PhaseTimings {
            total: self.t_start.elapsed(),
            preprocess: self.preprocess,
            weld_pairs: self.weld_pairs,
            weld_pair_capacity: self.weld_pair_capacity,
            knn_build: self.knn_build,
            knn_build_sub: self.knn_build_sub,
            cell_construction: self.cell_construction,
            cell_sub: self.cell_sub,
            dedup: self.dedup,
            dedup_sub: self.dedup_sub,
            edge_reconcile: self.edge_reconcile,
            merge_safety_scan_cells: self.merge_safety_scan_cells,
            merge_safety_global_fallbacks: self.merge_safety_global_fallbacks,
            assemble: self.assemble,
            grid_res: self.grid_res,
            grid_max_occupancy: self.grid_max_occupancy,
            grid_rebuilt: self.grid_rebuilt,
            resolution_drift_fallback: self.resolution_drift_fallback,
            resolution_reconcile_scan_cells: self.resolution_reconcile_scan_cells,
            resolution_rebuild_scan_cells: self.resolution_rebuild_scan_cells,
            resolution_hint_cells: self.resolution_hint_cells,
            resolution_hinted_candidates: self.resolution_hinted_candidates,
            resolution_detected_edges: self.resolution_detected_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TimingBuilder, WorkHistogram};

    #[test]
    fn work_histogram_reports_exact_body_and_bounded_tail() {
        let mut histogram = WorkHistogram::default();
        for value in [2, 4, 4, 8, 16, 64, 300] {
            histogram.record(value);
        }

        let summary = histogram.summary();
        assert_eq!(summary.samples, 7);
        assert_eq!(summary.p50_bucket_lower, 8);
        assert_eq!(summary.p90_bucket_lower, 256);
        assert_eq!(summary.p99_bucket_lower, 256);
        assert_eq!(summary.p999_bucket_lower, 256);
        assert_eq!(summary.max, 300);
        assert_eq!(summary.relative_median_base, 8);
        assert_eq!(summary.count_ge_4x_median_lower, 2);
        assert_eq!(summary.count_ge_16x_median_lower, 1);
        // The 300 sample cannot be conservatively claimed to exceed 512 from
        // its [256, 512) logarithmic bucket.
        assert_eq!(summary.count_ge_64x_median_lower, 0);
    }

    #[test]
    fn work_histogram_merge_preserves_samples_and_quantiles() {
        let mut left = WorkHistogram::default();
        let mut right = WorkHistogram::default();
        for value in [3, 5, 7] {
            left.record(value);
        }
        for value in [9, 11, 13] {
            right.record(value);
        }
        left.merge(&right);

        let summary = left.summary();
        assert_eq!(summary.samples, 6);
        assert_eq!(summary.p50_bucket_lower, 7);
        assert_eq!(summary.p90_bucket_lower, 13);
        assert_eq!(summary.max, 13);
        assert_eq!(summary.relative_median_base, 7);
    }

    #[test]
    fn zero_median_uses_one_candidate_relative_base() {
        let mut histogram = WorkHistogram::default();
        for value in [0, 0, 0, 4] {
            histogram.record(value);
        }

        let summary = histogram.summary();
        assert_eq!(summary.p50_bucket_lower, 0);
        assert_eq!(summary.relative_median_base, 1);
        assert_eq!(summary.count_ge_4x_median_lower, 1);
    }

    #[test]
    fn output_resolution_discovery_fields_survive_finish() {
        for drift_fallback in [false, true] {
            let mut builder = TimingBuilder::new();
            builder.set_edge_reconcile(std::time::Duration::from_millis(2), 17, 1);
            builder.set_output_resolution_discovery(drift_fallback, 13, 5, 11, 7, 3);
            let timings = builder.finish();
            assert_eq!(timings.merge_safety_scan_cells, 17);
            assert_eq!(timings.merge_safety_global_fallbacks, 1);
            assert_eq!(timings.resolution_drift_fallback, drift_fallback);
            assert_eq!(timings.resolution_reconcile_scan_cells, 13);
            assert_eq!(timings.resolution_rebuild_scan_cells, 5);
            assert_eq!(timings.resolution_hint_cells, 11);
            assert_eq!(timings.resolution_hinted_candidates, 7);
            assert_eq!(timings.resolution_detected_edges, 3);
        }
    }
}
