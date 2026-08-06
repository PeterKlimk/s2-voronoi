use super::KnnCellStage;
use crate::cube_grid::packed_knn::PackedKnnTelemetry;

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
    fn merge(&mut self, other: &Self) {
        self.packed_chunk0 += other.packed_chunk0;
        self.packed_tail += other.packed_tail;
        self.shell_expand += other.shell_expand;
    }
}

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
        let mut cumulative = 0;
        for (bucket, count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= rank.max(1) {
                return Self::bucket_lower_bound(bucket);
            }
        }
        self.max
    }
    fn count_at_least_lower_bound(&self, threshold: u64) -> u64 {
        self.buckets
            .iter()
            .enumerate()
            .filter(|(bucket, _)| Self::bucket_lower_bound(*bucket) >= threshold)
            .map(|(_, count)| count)
            .sum()
    }
    fn summary(&self) -> WorkDistribution {
        let p50 = self.quantile_lower_bound(1, 2);
        let base = p50.max(1);
        WorkDistribution {
            samples: self.samples,
            p50_bucket_lower: p50,
            p90_bucket_lower: self.quantile_lower_bound(9, 10),
            p99_bucket_lower: self.quantile_lower_bound(99, 100),
            p999_bucket_lower: self.quantile_lower_bound(999, 1000),
            max: self.max,
            relative_median_base: base,
            count_ge_4x_median_lower: self.count_at_least_lower_bound(base.saturating_mul(4)),
            count_ge_16x_median_lower: self.count_at_least_lower_bound(base.saturating_mul(16)),
            count_ge_64x_median_lower: self.count_at_least_lower_bound(base.saturating_mul(64)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WorkDistribution {
    pub samples: u64,
    pub p50_bucket_lower: u64,
    pub p90_bucket_lower: u64,
    pub p99_bucket_lower: u64,
    pub p999_bucket_lower: u64,
    pub max: u64,
    pub relative_median_base: u64,
    pub count_ge_4x_median_lower: u64,
    pub count_ge_16x_median_lower: u64,
    pub count_ge_64x_median_lower: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CellTelemetry {
    pub bin_count: u64,
    pub bin_population_max: u64,
    pub cells_knn_exhausted: u64,
    pub cells_packed_tail_used: u64,
    pub cells_packed_safe_exhausted: u64,
    pub cells_used_knn: u64,
    pub packed_chunk0_cells: u64,
    pub packed_tail_cells: u64,
    pub shell_expand_cells: u64,
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
    pub packed_exact_batch_counts: [u64; 4],
    pub packed_exact_slots_emitted: [u64; 4],
    pub packed_exact_slots_visited: [u64; 4],
    pub packed_exact_slots_abandoned: [u64; 4],
    pub shell_layer_batches: u64,
    pub shell_layer_slots: u64,
    pub shell_layer_prefix_consumed: u64,
    pub shell_midlayer_terminations: u64,
    pub neighbors_processed_total: u64,
    pub neighbors_processed_max: u64,
    pub final_edges_total: u64,
    pub final_edges_max: u64,
    pub candidate_work: WorkDistribution,
    pub no_progress_tail: WorkDistribution,
    pub no_progress_tail_excluded: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DedupTelemetry {
    pub edge_check_overflow_records: u64,
    pub shard_order_descents: u64,
    pub shard_order_pairs: u64,
    pub shard_order_abs_delta: u64,
    pub scatter_by_shard: bool,
    pub triplet_keys: u64,
    pub edge_mismatches_count: u64,
    pub primary_cell_references: u64,
    pub reference_overrides: u64,
}

#[derive(Clone, Default)]
pub(crate) struct CellTelemetryAccum {
    bin_count: u64,
    bin_population_max: u64,
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
}

impl CellTelemetryAccum {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn record_bin_schedule(&mut self, bins: &[Vec<usize>]) {
        self.bin_count = bins.len() as u64;
        self.bin_population_max = bins.iter().map(|v| v.len() as u64).max().unwrap_or(0);
    }
    pub(crate) fn add_packed_telemetry(&mut self, t: &PackedKnnTelemetry) {
        self.packed_tail_builds += t.tail_builds;
        self.packed_keys_materialized += t.keys_materialized;
        self.packed_key_capacity_peak = self.packed_key_capacity_peak.max(t.key_capacity_peak);
        self.tail_possible_queries += t.tail_possible_queries;
        self.tail_requested_queries += t.tail_requested_queries;
        self.ring_tail_rescans += t.ring_tail_rescans;
        self.ring_tail_empty_rescans += t.ring_tail_empty_rescans;
        self.ring_tail_dot_evaluations += t.ring_tail_dot_evaluations;
        self.center_tail_keys += t.center_tail_keys;
        self.unused_center_tail_keys += t.unused_center_tail_keys;
        self.center_tail_dot_evaluations += t.center_tail_dot_evaluations;
        self.chunk0_keys += t.chunk0_keys;
        self.unused_chunk0_keys += t.unused_chunk0_keys;
        for i in 0..4 {
            self.packed_exact_batch_counts[i] += t.exact_batch_counts[i];
            self.packed_exact_slots_emitted[i] += t.exact_slots_emitted[i];
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_cell_stage(
        &mut self,
        stage: KnnCellStage,
        exhausted: bool,
        neighbors: usize,
        edges: usize,
        tail: bool,
        safe_exhausted: bool,
        used_knn: bool,
        _incoming: usize,
        _seed_clips: usize,
    ) {
        self.stage_counts.add(stage);
        self.cells_knn_exhausted += exhausted as u64;
        self.cells_packed_tail_used += tail as u64;
        self.cells_packed_safe_exhausted += safe_exhausted as u64;
        self.cells_used_knn += used_knn as u64;
        self.neighbors_processed_total += neighbors as u64;
        self.neighbors_processed_max = self.neighbors_processed_max.max(neighbors as u64);
        self.final_edges_total += edges as u64;
        self.final_edges_max = self.final_edges_max.max(edges as u64);
    }
    pub(crate) fn add_work_profile(&mut self, candidates: usize, tail: usize, valid: bool) {
        self.candidate_work.record(candidates);
        if valid {
            self.no_progress_tail.record(tail);
        } else {
            self.no_progress_tail_excluded += 1;
        }
    }
    pub(crate) fn add_fallbacks(&mut self, projection: usize, polygon: usize, all: usize) {
        self.fallback_projection += projection as u64;
        self.fallback_polygon_cap += polygon as u64;
        self.fallback_all_constraints += all as u64;
    }
    pub(crate) fn add_shell_layer_usage(
        &mut self,
        batches: usize,
        slots: usize,
        prefix: usize,
        mid: usize,
    ) {
        self.shell_layer_batches += batches as u64;
        self.shell_layer_slots += slots as u64;
        self.shell_layer_prefix_consumed += prefix as u64;
        self.shell_midlayer_terminations += mid as u64;
    }
    pub(crate) fn add_packed_batch_usage(&mut self, visited: [usize; 4], abandoned: [usize; 4]) {
        for i in 0..4 {
            self.packed_exact_slots_visited[i] += visited[i] as u64;
            self.packed_exact_slots_abandoned[i] += abandoned[i] as u64;
        }
    }
    pub(crate) fn merge(&mut self, o: &Self) {
        self.stage_counts.merge(&o.stage_counts);
        self.bin_count = self.bin_count.max(o.bin_count);
        self.bin_population_max = self.bin_population_max.max(o.bin_population_max);
        macro_rules! add { ($($f:ident),+ $(,)?) => { $(self.$f += o.$f;)+ }; }
        add!(
            cells_knn_exhausted,
            cells_packed_tail_used,
            cells_packed_safe_exhausted,
            cells_used_knn,
            fallback_projection,
            fallback_polygon_cap,
            fallback_all_constraints,
            packed_tail_builds,
            packed_keys_materialized,
            tail_possible_queries,
            tail_requested_queries,
            ring_tail_rescans,
            ring_tail_empty_rescans,
            ring_tail_dot_evaluations,
            center_tail_keys,
            unused_center_tail_keys,
            center_tail_dot_evaluations,
            chunk0_keys,
            unused_chunk0_keys,
            shell_layer_batches,
            shell_layer_slots,
            shell_layer_prefix_consumed,
            shell_midlayer_terminations,
            neighbors_processed_total,
            final_edges_total,
            no_progress_tail_excluded
        );
        self.packed_key_capacity_peak = self
            .packed_key_capacity_peak
            .max(o.packed_key_capacity_peak);
        self.neighbors_processed_max = self.neighbors_processed_max.max(o.neighbors_processed_max);
        self.final_edges_max = self.final_edges_max.max(o.final_edges_max);
        for i in 0..4 {
            self.packed_exact_batch_counts[i] += o.packed_exact_batch_counts[i];
            self.packed_exact_slots_emitted[i] += o.packed_exact_slots_emitted[i];
            self.packed_exact_slots_visited[i] += o.packed_exact_slots_visited[i];
            self.packed_exact_slots_abandoned[i] += o.packed_exact_slots_abandoned[i];
        }
        self.candidate_work.merge(&o.candidate_work);
        self.no_progress_tail.merge(&o.no_progress_tail);
    }
    pub(crate) fn into_telemetry(self) -> CellTelemetry {
        CellTelemetry {
            bin_count: self.bin_count,
            bin_population_max: self.bin_population_max,
            cells_knn_exhausted: self.cells_knn_exhausted,
            cells_packed_tail_used: self.cells_packed_tail_used,
            cells_packed_safe_exhausted: self.cells_packed_safe_exhausted,
            cells_used_knn: self.cells_used_knn,
            packed_chunk0_cells: self.stage_counts.packed_chunk0,
            packed_tail_cells: self.stage_counts.packed_tail,
            shell_expand_cells: self.stage_counts.shell_expand,
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
        }
    }
}

#[derive(Default)]
pub(crate) struct TelemetryBuilder {
    weld_pairs: u64,
    weld_pair_capacity: u64,
    grid_res: usize,
    grid_max_occupancy: u64,
    grid_rebuilt: bool,
    grid_input_order_abs_delta: u64,
    grid_input_order_pairs: u64,
    grid_materialized_by_slot: bool,
    grid_topology_overlapped: bool,
    grid_topology_reused: bool,
    cell: CellTelemetry,
    dedup: DedupTelemetry,
    merge_safety_scan_cells: u64,
    merge_safety_global_fallbacks: u64,
    resolution_drift_fallback: bool,
    resolution_reconcile_scan_cells: u64,
    resolution_rebuild_scan_cells: u64,
    resolution_hint_cells: u64,
    resolution_hinted_candidates: u64,
    resolution_detected_edges: u64,
}
impl TelemetryBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn set_weld_pair_stats(&mut self, len: usize, cap: usize) {
        self.weld_pairs = len as u64;
        self.weld_pair_capacity = cap as u64;
    }
    pub(crate) fn set_grid_stats(&mut self, res: usize, max: u64, rebuilt: bool) {
        self.grid_res = res;
        self.grid_max_occupancy = max;
        self.grid_rebuilt = rebuilt;
    }
    pub(crate) fn set_grid_build_stats(
        &mut self,
        stats: &crate::cube_grid::CubeMapGridBuildTelemetry,
    ) {
        self.grid_input_order_abs_delta = stats.input_order_abs_delta;
        self.grid_input_order_pairs = stats.input_order_pairs;
        self.grid_materialized_by_slot = stats.materialize_coordinates_by_slot;
        self.grid_topology_overlapped = stats.topology_overlapped;
        self.grid_topology_reused = stats.topology_reused;
    }
    pub(crate) fn set_cell(&mut self, cell: CellTelemetry) {
        self.cell = cell;
    }
    pub(crate) fn set_dedup(&mut self, dedup: DedupTelemetry) {
        self.dedup = dedup;
    }
    pub(crate) fn set_edge_reconcile(&mut self, cells: usize, fallbacks: usize) {
        self.merge_safety_scan_cells = cells as u64;
        self.merge_safety_global_fallbacks = fallbacks as u64;
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_output_resolution_discovery(
        &mut self,
        drift: bool,
        reconcile: usize,
        rebuild: usize,
        hints: usize,
        candidates: usize,
        edges: usize,
    ) {
        self.resolution_drift_fallback = drift;
        self.resolution_reconcile_scan_cells = reconcile as u64;
        self.resolution_rebuild_scan_cells = rebuild as u64;
        self.resolution_hint_cells = hints as u64;
        self.resolution_hinted_candidates = candidates as u64;
        self.resolution_detected_edges = edges as u64;
    }
    pub(crate) fn report(&self, n: usize) {
        report(self, n);
    }
}

fn report(t: &TelemetryBuilder, n: usize) {
    let c = &t.cell;
    let d = &t.dedup;
    eprintln!("telemetry n={n}");
    eprintln!(
        "  grid: res={} max_occ={} rebuilt={} input_pairs={} input_abs_delta={} materialized_by_slot={} topology_overlapped={} topology_reused={} weld_pairs={} weld_capacity={}",
        t.grid_res,
        t.grid_max_occupancy,
        t.grid_rebuilt,
        t.grid_input_order_pairs,
        t.grid_input_order_abs_delta,
        t.grid_materialized_by_slot,
        t.grid_topology_overlapped,
        t.grid_topology_reused,
        t.weld_pairs,
        t.weld_pair_capacity
    );
    eprintln!(
        "  bins: count={} max_population={}",
        c.bin_count, c.bin_population_max
    );
    eprintln!(
        "  cells: chunk0={} packed_tail={} shell={} used_knn={} exhausted={} safe_exhausted={}",
        c.packed_chunk0_cells,
        c.packed_tail_cells,
        c.shell_expand_cells,
        c.cells_used_knn,
        c.cells_knn_exhausted,
        c.cells_packed_safe_exhausted
    );
    eprintln!(
        "  neighbors: mean={:.2} max={} final_degree_mean={:.2} max={}",
        c.neighbors_processed_total as f64 / n.max(1) as f64,
        c.neighbors_processed_max,
        c.final_edges_total as f64 / n.max(1) as f64,
        c.final_edges_max
    );
    eprintln!(
        "  candidate_work: p50={} p90={} p99={} p999={} max={} ge4x={} ge16x={} ge64x={}",
        c.candidate_work.p50_bucket_lower,
        c.candidate_work.p90_bucket_lower,
        c.candidate_work.p99_bucket_lower,
        c.candidate_work.p999_bucket_lower,
        c.candidate_work.max,
        c.candidate_work.count_ge_4x_median_lower,
        c.candidate_work.count_ge_16x_median_lower,
        c.candidate_work.count_ge_64x_median_lower
    );
    eprintln!(
        "  no_progress_tail: samples={} excluded={} p50={} p90={} p99={} p999={} max={}",
        c.no_progress_tail.samples,
        c.no_progress_tail_excluded,
        c.no_progress_tail.p50_bucket_lower,
        c.no_progress_tail.p90_bucket_lower,
        c.no_progress_tail.p99_bucket_lower,
        c.no_progress_tail.p999_bucket_lower,
        c.no_progress_tail.max
    );
    eprintln!(
        "  packed_batches: counts={:?} emitted={:?} visited={:?} abandoned={:?}",
        c.packed_exact_batch_counts,
        c.packed_exact_slots_emitted,
        c.packed_exact_slots_visited,
        c.packed_exact_slots_abandoned
    );
    eprintln!(
        "  shell_layers: batches={} slots={} prefix_consumed={} midlayer_terminations={}",
        c.shell_layer_batches,
        c.shell_layer_slots,
        c.shell_layer_prefix_consumed,
        c.shell_midlayer_terminations
    );
    eprintln!("  dedup: triplet_keys={} overflow_records={} mismatches={} primary_refs={} overrides={} scatter_by_shard={}", d.triplet_keys, d.edge_check_overflow_records, d.edge_mismatches_count, d.primary_cell_references, d.reference_overrides, d.scatter_by_shard);
    if std::env::var_os("VORONOI_MESH_TELEMETRY_KV").is_some() {
        eprintln!("TELEMETRY_KV n={n} weld_pairs={} weld_pair_capacity={} grid_res={} grid_max_occ={} grid_rebuilt={} grid_input_order_pairs={} grid_input_order_abs_delta={} grid_materialized_by_slot={} grid_topology_overlapped={} grid_topology_reused={} bin_count={} bin_population_max={} cells_used_knn={} cells_packed_tail_used={} cells_packed_safe_exhausted={} packed_chunk0_cells={} packed_tail_cells={} shell_expand_cells={} fallback_projection={} fallback_polygon_cap={} fallback_all_constraints={} packed_tail_builds={} packed_keys_materialized={} packed_key_capacity_peak={} tail_possible_queries={} tail_requested_queries={} ring_tail_rescans={} ring_tail_empty_rescans={} ring_tail_dot_evaluations={} center_tail_keys={} unused_center_tail_keys={} center_tail_dot_evaluations={} chunk0_keys={} unused_chunk0_keys={} neighbors_total={} neighbors_max={} candidate_work_samples={} candidate_work_p50_lb={} candidate_work_p90_lb={} candidate_work_p99_lb={} candidate_work_p999_lb={} candidate_work_max={} candidate_work_relative_base={} candidate_work_ge4x_median_lb={} candidate_work_ge16x_median_lb={} candidate_work_ge64x_median_lb={} no_progress_tail_samples={} no_progress_tail_excluded={} no_progress_tail_p50_lb={} no_progress_tail_p90_lb={} no_progress_tail_p99_lb={} no_progress_tail_p999_lb={} no_progress_tail_max={} no_progress_tail_relative_base={} no_progress_tail_ge4x_median_lb={} no_progress_tail_ge16x_median_lb={} no_progress_tail_ge64x_median_lb={} final_edges_total={} final_edges_max={} edge_check_overflow_records={} triplet_keys={} edge_mismatches={} primary_refs={} reference_overrides={} shard_order_descents={} shard_order_pairs={} shard_order_abs_delta={} scatter_by_shard={} merge_safety_scan_cells={} merge_safety_global_fallbacks={} resolution_fallback_drift={} resolution_reconcile_scan_cells={} resolution_rebuild_scan_cells={} resolution_hint_cells={} resolution_hinted_candidates={} resolution_detected_edges={}",
            t.weld_pairs,t.weld_pair_capacity,t.grid_res,t.grid_max_occupancy,t.grid_rebuilt as u8,t.grid_input_order_pairs,t.grid_input_order_abs_delta,t.grid_materialized_by_slot as u8,t.grid_topology_overlapped as u8,t.grid_topology_reused as u8,c.bin_count,c.bin_population_max,c.cells_used_knn,c.cells_packed_tail_used,c.cells_packed_safe_exhausted,c.packed_chunk0_cells,c.packed_tail_cells,c.shell_expand_cells,c.fallback_projection,c.fallback_polygon_cap,c.fallback_all_constraints,c.packed_tail_builds,c.packed_keys_materialized,c.packed_key_capacity_peak,c.tail_possible_queries,c.tail_requested_queries,c.ring_tail_rescans,c.ring_tail_empty_rescans,c.ring_tail_dot_evaluations,c.center_tail_keys,c.unused_center_tail_keys,c.center_tail_dot_evaluations,c.chunk0_keys,c.unused_chunk0_keys,c.neighbors_processed_total,c.neighbors_processed_max,c.candidate_work.samples,c.candidate_work.p50_bucket_lower,c.candidate_work.p90_bucket_lower,c.candidate_work.p99_bucket_lower,c.candidate_work.p999_bucket_lower,c.candidate_work.max,c.candidate_work.relative_median_base,c.candidate_work.count_ge_4x_median_lower,c.candidate_work.count_ge_16x_median_lower,c.candidate_work.count_ge_64x_median_lower,c.no_progress_tail.samples,c.no_progress_tail_excluded,c.no_progress_tail.p50_bucket_lower,c.no_progress_tail.p90_bucket_lower,c.no_progress_tail.p99_bucket_lower,c.no_progress_tail.p999_bucket_lower,c.no_progress_tail.max,c.no_progress_tail.relative_median_base,c.no_progress_tail.count_ge_4x_median_lower,c.no_progress_tail.count_ge_16x_median_lower,c.no_progress_tail.count_ge_64x_median_lower,c.final_edges_total,c.final_edges_max,d.edge_check_overflow_records,d.triplet_keys,d.edge_mismatches_count,d.primary_cell_references,d.reference_overrides,d.shard_order_descents,d.shard_order_pairs,d.shard_order_abs_delta,d.scatter_by_shard as u8,t.merge_safety_scan_cells,t.merge_safety_global_fallbacks,t.resolution_drift_fallback as u8,t.resolution_reconcile_scan_cells,t.resolution_rebuild_scan_cells,t.resolution_hint_cells,t.resolution_hinted_candidates,t.resolution_detected_edges);
    }
}
