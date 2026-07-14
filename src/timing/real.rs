use super::KnnCellStage;
use std::time::Duration;

use crate::cube_grid::{packed_knn::PackedKnnTimings, CubeMapGridBuildTimings};

/// Timer that tracks elapsed time when timing is enabled.
pub struct Timer(std::time::Instant);

impl Timer {
    #[inline]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Timer optimized for sequential sub-phase timing: each `lap()` uses a single `Instant::now()`.
pub struct LapTimer(std::time::Instant);

impl LapTimer {
    #[inline]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub fn lap(&mut self) -> Duration {
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

/// Per-cell-group timing totals aggregated across all shards.
#[derive(Debug, Clone, Default)]
pub struct CellSubPhases {
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
pub struct DedupSubPhases {
    pub triplet_keys: u64,
    pub unresolved_edges_count: u64,
}

/// Accumulator for cell sub-phase timings (used per-bin, then merged).
#[derive(Clone, Default)]
pub struct CellSubAccum {
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
    shell_layer_batches: u64,
    shell_layer_slots: u64,
    shell_layer_prefix_consumed: u64,
    shell_midlayer_terminations: u64,
    neighbors_processed_total: u64,
    neighbors_processed_max: u64,
    final_edges_total: u64,
    final_edges_max: u64,
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
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add_knn(&mut self, d: Duration) {
        self.knn_query += d;
    }

    #[inline]
    pub fn add_packed_knn(&mut self, d: Duration) {
        self.packed_knn += d;
    }

    #[inline]
    pub fn add_packed_knn_breakdown(&mut self, timings: &PackedKnnTimings) {
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
    }

    #[inline]
    pub fn add_clip(&mut self, d: Duration) {
        self.clipping += d;
    }

    #[inline]
    pub fn add_cert(&mut self, d: Duration) {
        self.certification += d;
    }

    #[inline]
    pub fn add_key_dedup(&mut self, d: Duration) {
        self.key_dedup += d;
    }

    #[inline]
    pub fn add_edge_collect(&mut self, d: Duration) {
        self.edge_collect += d;
    }

    #[inline]
    pub fn add_edge_resolve(&mut self, d: Duration) {
        self.edge_resolve += d;
    }

    #[inline]
    pub fn add_edge_emit(&mut self, d: Duration) {
        self.edge_emit += d;
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn add_cell_stage(
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
    pub fn add_fallbacks(&mut self, projection: usize, polygon_cap: usize, all_constraints: usize) {
        self.fallback_projection += projection as u64;
        self.fallback_polygon_cap += polygon_cap as u64;
        self.fallback_all_constraints += all_constraints as u64;
    }

    #[inline]
    pub fn add_shell_layer_usage(
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
    #[allow(clippy::too_many_arguments)]
    pub fn add_directional_shadow(
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
    pub fn merge(&mut self, other: &CellSubAccum) {
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
    pub fn into_sub_phases(self) -> CellSubPhases {
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
            shell_layer_batches: self.shell_layer_batches,
            shell_layer_slots: self.shell_layer_slots,
            shell_layer_prefix_consumed: self.shell_layer_prefix_consumed,
            shell_midlayer_terminations: self.shell_midlayer_terminations,
            neighbors_processed_total: self.neighbors_processed_total,
            neighbors_processed_max: self.neighbors_processed_max,
            final_edges_total: self.final_edges_total,
            final_edges_max: self.final_edges_max,
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
pub struct PhaseTimings {
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
    pub assemble: Duration,
    /// Query-grid shape: resolution, max cell occupancy, and whether the
    /// occupancy-feedback rebuild fired.
    pub grid_res: usize,
    pub grid_max_occupancy: u64,
    pub grid_rebuilt: bool,
    pub resolution_certified_hint: bool,
    pub resolution_drift_fallback: bool,
    pub resolution_reconcile_scan_cells: u64,
    pub resolution_repair_scan_cells: u64,
    pub resolution_hint_cells: u64,
    pub resolution_hinted_candidates: u64,
    pub resolution_detected_edges: u64,
}

impl PhaseTimings {
    pub fn report(&self, n: usize) {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let total_ms = ms(self.total);

        let pct = |d: Duration| {
            if self.total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / self.total.as_secs_f64() * 100.0
            }
        };

        eprintln!("timing n={}", n);
        if self.preprocess.as_nanos() > 0 {
            eprintln!(
                "  preprocess:        {:7.1}ms ({:4.1}%)",
                ms(self.preprocess),
                pct(self.preprocess)
            );
        }
        eprintln!(
            "  knn_build:         {:7.1}ms ({:4.1}%)",
            ms(self.knn_build),
            pct(self.knn_build)
        );
        if let Some(sub) = &self.knn_build_sub {
            if sub.total().as_nanos() > 0 && self.knn_build.as_nanos() > 0 {
                let sub_pct = |d: Duration| d.as_secs_f64() / self.knn_build.as_secs_f64() * 100.0;
                eprintln!(
                    "    grid_count:      {:7.1}ms ({:4.1}%)",
                    ms(sub.count_cells),
                    sub_pct(sub.count_cells)
                );
                eprintln!(
                    "    grid_prefix:     {:7.1}ms ({:4.1}%)",
                    ms(sub.prefix_sum),
                    sub_pct(sub.prefix_sum)
                );
                eprintln!(
                    "    grid_scatter:    {:7.1}ms ({:4.1}%)",
                    ms(sub.scatter_soa),
                    sub_pct(sub.scatter_soa)
                );
                eprintln!(
                    "    grid_neighbors:  {:7.1}ms ({:4.1}%)",
                    ms(sub.neighbors),
                    sub_pct(sub.neighbors)
                );
                eprintln!(
                    "    grid_ring2:      {:7.1}ms ({:4.1}%)",
                    ms(sub.ring2),
                    sub_pct(sub.ring2)
                );
                eprintln!(
                    "    grid_bounds:     {:7.1}ms ({:4.1}%)",
                    ms(sub.cell_bounds),
                    sub_pct(sub.cell_bounds)
                );
                eprintln!(
                    "    grid_security:   {:7.1}ms ({:4.1}%)",
                    ms(sub.security_3x3),
                    sub_pct(sub.security_3x3)
                );
            }
        }

        eprintln!(
            "  cell_construction: {:7.1}ms ({:4.1}%)",
            ms(self.cell_construction),
            pct(self.cell_construction)
        );

        // Estimate wall time contributions from per-cell CPU totals (parallel runs).
        let cpu_total = self.cell_sub.knn_query
            + self.cell_sub.packed_knn
            + self.cell_sub.clipping
            + self.cell_sub.certification
            + self.cell_sub.key_dedup
            + self.cell_sub.edge_collect
            + self.cell_sub.edge_resolve
            + self.cell_sub.edge_emit;
        let cpu_total_secs = cpu_total.as_secs_f64();
        let wall_secs = self.cell_construction.as_secs_f64();
        let cpu_to_wall = if cpu_total_secs > 0.0 {
            wall_secs / cpu_total_secs
        } else {
            1.0
        };
        let sub_pct = |d: Duration| {
            if cpu_total_secs > 0.0 {
                d.as_secs_f64() / cpu_total_secs * 100.0
            } else {
                0.0
            }
        };
        let est_wall_ms = |d: Duration| d.as_secs_f64() * cpu_to_wall * 1000.0;

        if cpu_total.as_nanos() > 0 {
            eprintln!(
                "    knn_query:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.knn_query),
                sub_pct(self.cell_sub.knn_query)
            );
            if self.cell_sub.packed_knn.as_nanos() > 0 {
                eprintln!(
                    "    packed_knn:      {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn),
                    sub_pct(self.cell_sub.packed_knn)
                );
                let kernel = [
                    ("setup", self.cell_sub.packed_setup),
                    ("security", self.cell_sub.packed_security),
                    ("center_pass", self.cell_sub.packed_center_pass),
                    ("ring_thresholds", self.cell_sub.packed_ring_thresholds),
                    ("ring_pass", self.cell_sub.packed_ring_pass),
                    ("ring_fallback", self.cell_sub.packed_ring_fallback),
                    ("select_prep", self.cell_sub.packed_select_prep),
                    ("select_partition", self.cell_sub.packed_select_partition),
                    ("select_sort", self.cell_sub.packed_select_sort),
                    ("select_scatter", self.cell_sub.packed_select_scatter),
                ];
                for (label, d) in kernel {
                    if d.as_nanos() > 0 {
                        eprintln!("      {:16} {:7.1}ms", label, est_wall_ms(d));
                    }
                }
                if self.cell_sub.packed_tail_builds > 0 {
                    eprintln!(
                        "      packed_builds: tail={}",
                        self.cell_sub.packed_tail_builds,
                    );
                    eprintln!(
                        "      tail_queries: possible={} requested={} ring_rescans={} empty={} ring_dot_evals={}",
                        self.cell_sub.tail_possible_queries,
                        self.cell_sub.tail_requested_queries,
                        self.cell_sub.ring_tail_rescans,
                        self.cell_sub.ring_tail_empty_rescans,
                        self.cell_sub.ring_tail_dot_evaluations,
                    );
                    eprintln!(
                        "      center_tail_candidates: total={} unrequested={} recomputed_dots={}",
                        self.cell_sub.center_tail_keys,
                        self.cell_sub.unused_center_tail_keys,
                        self.cell_sub.center_tail_dot_evaluations,
                    );
                    eprintln!(
                        "      chunk0_keys: total={} unused={}",
                        self.cell_sub.chunk0_keys, self.cell_sub.unused_chunk0_keys,
                    );
                }
            }
            eprintln!(
                "    clipping:        {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.clipping),
                sub_pct(self.cell_sub.clipping)
            );
            eprintln!(
                "    certification:   {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.certification),
                sub_pct(self.cell_sub.certification)
            );
            eprintln!(
                "    key_dedup:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.key_dedup),
                sub_pct(self.cell_sub.key_dedup)
            );
            eprintln!(
                "    edge_collect:    {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_collect),
                sub_pct(self.cell_sub.edge_collect)
            );
            eprintln!(
                "    edge_resolve:    {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_resolve),
                sub_pct(self.cell_sub.edge_resolve)
            );
            eprintln!(
                "    edge_emit:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_emit),
                sub_pct(self.cell_sub.edge_emit)
            );
            eprintln!(
                "    cells: used_knn={} knn_exhausted={} packed_tail_used={} packed_safe_exhausted={}",
                self.cell_sub.cells_used_knn,
                self.cell_sub.cells_knn_exhausted,
                self.cell_sub.cells_packed_tail_used,
                self.cell_sub.cells_packed_safe_exhausted
            );
            if self.cell_sub.shell_layer_batches > 0 {
                eprintln!(
                    "    shell_layers: batches={} slots={} prefix_consumed={} midlayer_terminations={}",
                    self.cell_sub.shell_layer_batches,
                    self.cell_sub.shell_layer_slots,
                    self.cell_sub.shell_layer_prefix_consumed,
                    self.cell_sub.shell_midlayer_terminations,
                );
            }
            if self.cell_sub.fallback_projection > 0
                || self.cell_sub.fallback_polygon_cap > 0
                || self.cell_sub.fallback_all_constraints > 0
            {
                eprintln!(
                    "    fallbacks: projection={} polygon_cap={} all_constraints={}",
                    self.cell_sub.fallback_projection,
                    self.cell_sub.fallback_polygon_cap,
                    self.cell_sub.fallback_all_constraints
                );
            }
            eprintln!(
                "    neighbors: mean={:.1} max={} (grid res={} max_occ={} rebuilt={})",
                self.cell_sub.neighbors_processed_total as f64 / n.max(1) as f64,
                self.cell_sub.neighbors_processed_max,
                self.grid_res,
                self.grid_max_occupancy,
                self.grid_rebuilt
            );
            let examine_per_edge = if self.cell_sub.final_edges_total > 0 {
                self.cell_sub.neighbors_processed_total as f64
                    / self.cell_sub.final_edges_total as f64
            } else {
                0.0
            };
            eprintln!(
                "    final_edges: mean={:.2} max={} examine_per_edge={:.3}",
                self.cell_sub.final_edges_total as f64 / n.max(1) as f64,
                self.cell_sub.final_edges_max,
                examine_per_edge
            );
            if self.cell_sub.directional_shadow_checks > 0 {
                eprintln!(
                    "    dir_shadow: checks={} tests={} hits={} saved={} support_tests={} support_hits={} support_saved={} support_false_pos={}",
                    self.cell_sub.directional_shadow_checks,
                    self.cell_sub.directional_shadow_candidate_tests,
                    self.cell_sub.directional_shadow_hits,
                    self.cell_sub.directional_shadow_saved,
                    self.cell_sub.directional_support_candidate_tests,
                    self.cell_sub.directional_support_hits,
                    self.cell_sub.directional_support_saved,
                    self.cell_sub.directional_support_false_positive_hits
                );
            }
        }

        eprintln!(
            "  dedup:             {:7.1}ms ({:4.1}%)",
            ms(self.dedup),
            pct(self.dedup)
        );
        eprintln!(
            "    keys: triplet={} unresolved_edges={}",
            self.dedup_sub.triplet_keys, self.dedup_sub.unresolved_edges_count
        );
        eprintln!(
            "  edge_reconcile:    {:7.1}ms ({:4.1}%)",
            ms(self.edge_reconcile),
            pct(self.edge_reconcile)
        );
        eprintln!(
            "  assemble:          {:7.1}ms ({:4.1}%)",
            ms(self.assemble),
            pct(self.assemble)
        );
        eprintln!(
            "  output_resolution: mode={} drift_fallback={} local_scan(reconcile_cells={},repair_cells={}) hint_cells={} hinted_candidates={} detected_edges={}",
            if self.resolution_certified_hint {
                "certified_hint"
            } else {
                "exhaustive"
            },
            self.resolution_drift_fallback as u8,
            self.resolution_reconcile_scan_cells,
            self.resolution_repair_scan_cells,
            self.resolution_hint_cells,
            self.resolution_hinted_candidates,
            self.resolution_detected_edges,
        );

        if std::env::var_os("VORONOI_MESH_TIMING_KV").is_some() {
            eprintln!(
                "TIMING_KV n={n} total_ms={total:.3} preprocess_ms={pre:.3} weld_pairs={wp} weld_pair_capacity={wpc} knn_build_ms={kb:.3} cell_construction_ms={cc:.3} dedup_ms={dd:.3} edge_reconcile_ms={er:.3} assemble_ms={asmb:.3} resolution_certified_hint={rch} resolution_fallback_drift={rfd} resolution_reconcile_scan_cells={rrsc} resolution_repair_scan_cells={rpsc} resolution_hint_cells={rhc} resolution_hinted_candidates={rhcand} resolution_detected_edges={rde} cells_used_knn={cuk} cells_packed_tail_used={cpt} fallback_projection={fpj} fallback_polygon_cap={fpc} fallback_all_constraints={fac} packed_tail_builds={ptb} packed_keys_materialized={pkm} packed_key_capacity_peak={pkp} tail_possible_queries={tpq} tail_requested_queries={trq} ring_tail_rescans={rtr} ring_tail_empty_rescans={rte} ring_tail_dot_evaluations={rtd} center_tail_keys={ctk} unused_center_tail_keys={uctk} center_tail_dot_evaluations={ctd} chunk0_keys={c0k} unused_chunk0_keys={uc0k} shell_layer_batches={slb} shell_layer_slots={sls} shell_layer_prefix_consumed={slp} shell_midlayer_terminations={slm} neighbors_total={nt} neighbors_max={nm} final_edges_total={fet} final_edges_max={fem} examine_per_edge={epe:.6} dir_shadow_checks={dsc} dir_shadow_candidate_tests={dst} dir_shadow_hits={dsh} dir_shadow_saved={dss} dir_support_candidate_tests={dpt} dir_support_hits={dph} dir_support_saved={dps} dir_support_false_positive_hits={dpf} grid_res={gr} grid_max_occ={gmo} grid_rebuilt={grb}",
                n = n,
                total = total_ms,
                pre = ms(self.preprocess),
                wp = self.weld_pairs,
                wpc = self.weld_pair_capacity,
                kb = ms(self.knn_build),
                cc = ms(self.cell_construction),
                dd = ms(self.dedup),
                er = ms(self.edge_reconcile),
                asmb = ms(self.assemble),
                rch = self.resolution_certified_hint as u8,
                rfd = self.resolution_drift_fallback as u8,
                rrsc = self.resolution_reconcile_scan_cells,
                rpsc = self.resolution_repair_scan_cells,
                rhc = self.resolution_hint_cells,
                rhcand = self.resolution_hinted_candidates,
                rde = self.resolution_detected_edges,
                cuk = self.cell_sub.cells_used_knn,
                cpt = self.cell_sub.cells_packed_tail_used,
                fpj = self.cell_sub.fallback_projection,
                fpc = self.cell_sub.fallback_polygon_cap,
                fac = self.cell_sub.fallback_all_constraints,
                ptb = self.cell_sub.packed_tail_builds,
                pkm = self.cell_sub.packed_keys_materialized,
                pkp = self.cell_sub.packed_key_capacity_peak,
                tpq = self.cell_sub.tail_possible_queries,
                trq = self.cell_sub.tail_requested_queries,
                rtr = self.cell_sub.ring_tail_rescans,
                rte = self.cell_sub.ring_tail_empty_rescans,
                rtd = self.cell_sub.ring_tail_dot_evaluations,
                ctk = self.cell_sub.center_tail_keys,
                uctk = self.cell_sub.unused_center_tail_keys,
                ctd = self.cell_sub.center_tail_dot_evaluations,
                c0k = self.cell_sub.chunk0_keys,
                uc0k = self.cell_sub.unused_chunk0_keys,
                slb = self.cell_sub.shell_layer_batches,
                sls = self.cell_sub.shell_layer_slots,
                slp = self.cell_sub.shell_layer_prefix_consumed,
                slm = self.cell_sub.shell_midlayer_terminations,
                nt = self.cell_sub.neighbors_processed_total,
                nm = self.cell_sub.neighbors_processed_max,
                fet = self.cell_sub.final_edges_total,
                fem = self.cell_sub.final_edges_max,
                epe = if self.cell_sub.final_edges_total > 0 {
                    self.cell_sub.neighbors_processed_total as f64
                        / self.cell_sub.final_edges_total as f64
                } else {
                    0.0
                },
                dsc = self.cell_sub.directional_shadow_checks,
                dst = self.cell_sub.directional_shadow_candidate_tests,
                dsh = self.cell_sub.directional_shadow_hits,
                dss = self.cell_sub.directional_shadow_saved,
                dpt = self.cell_sub.directional_support_candidate_tests,
                dph = self.cell_sub.directional_support_hits,
                dps = self.cell_sub.directional_support_saved,
                dpf = self.cell_sub.directional_support_false_positive_hits,
                gr = self.grid_res,
                gmo = self.grid_max_occupancy,
                grb = self.grid_rebuilt as u8,
            );
        }
    }
}

/// Builder for collecting phase timings.
pub struct TimingBuilder {
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
    assemble: Duration,
    grid_res: usize,
    grid_max_occupancy: u64,
    grid_rebuilt: bool,
    resolution_certified_hint: bool,
    resolution_drift_fallback: bool,
    resolution_reconcile_scan_cells: u64,
    resolution_repair_scan_cells: u64,
    resolution_hint_cells: u64,
    resolution_hinted_candidates: u64,
    resolution_detected_edges: u64,
}

impl TimingBuilder {
    pub fn new() -> Self {
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
            assemble: Duration::ZERO,
            grid_res: 0,
            grid_max_occupancy: 0,
            grid_rebuilt: false,
            resolution_certified_hint: false,
            resolution_drift_fallback: false,
            resolution_reconcile_scan_cells: 0,
            resolution_repair_scan_cells: 0,
            resolution_hint_cells: 0,
            resolution_hinted_candidates: 0,
            resolution_detected_edges: 0,
        }
    }

    pub fn set_grid_stats(&mut self, res: usize, max_occupancy: u64, rebuilt: bool) {
        self.grid_res = res;
        self.grid_max_occupancy = max_occupancy;
        self.grid_rebuilt = rebuilt;
    }

    pub fn set_preprocess(&mut self, d: Duration) {
        self.preprocess = d;
    }

    pub fn set_weld_pair_stats(&mut self, len: usize, capacity: usize) {
        self.weld_pairs = len as u64;
        self.weld_pair_capacity = capacity as u64;
    }

    pub fn set_knn_build(&mut self, d: Duration) {
        self.knn_build = d;
    }

    pub fn add_knn_build(&mut self, d: Duration) {
        self.knn_build += d;
    }

    pub fn set_knn_build_sub(&mut self, sub: CubeMapGridBuildTimings) {
        self.knn_build_sub = Some(sub);
    }

    pub fn set_cell_construction(&mut self, d: Duration, sub: CellSubPhases) {
        self.cell_construction = d;
        self.cell_sub = sub;
    }

    pub fn set_dedup(&mut self, d: Duration, sub: DedupSubPhases) {
        self.dedup = d;
        self.dedup_sub = sub;
    }

    pub fn set_edge_reconcile(&mut self, d: Duration) {
        self.edge_reconcile = d;
    }

    pub fn set_assemble(&mut self, d: Duration) {
        self.assemble = d;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_output_resolution_discovery(
        &mut self,
        certified_hint: bool,
        drift_fallback: bool,
        reconcile_scan_cells: usize,
        repair_scan_cells: usize,
        hint_cells: usize,
        hinted_candidates: usize,
        detected_edges: usize,
    ) {
        self.resolution_certified_hint = certified_hint;
        self.resolution_drift_fallback = drift_fallback;
        self.resolution_reconcile_scan_cells = reconcile_scan_cells as u64;
        self.resolution_repair_scan_cells = repair_scan_cells as u64;
        self.resolution_hint_cells = hint_cells as u64;
        self.resolution_hinted_candidates = hinted_candidates as u64;
        self.resolution_detected_edges = detected_edges as u64;
    }

    pub fn finish(self) -> PhaseTimings {
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
            assemble: self.assemble,
            grid_res: self.grid_res,
            grid_max_occupancy: self.grid_max_occupancy,
            grid_rebuilt: self.grid_rebuilt,
            resolution_certified_hint: self.resolution_certified_hint,
            resolution_drift_fallback: self.resolution_drift_fallback,
            resolution_reconcile_scan_cells: self.resolution_reconcile_scan_cells,
            resolution_repair_scan_cells: self.resolution_repair_scan_cells,
            resolution_hint_cells: self.resolution_hint_cells,
            resolution_hinted_candidates: self.resolution_hinted_candidates,
            resolution_detected_edges: self.resolution_detected_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TimingBuilder;

    #[test]
    fn output_resolution_discovery_fields_survive_finish() {
        let mut builder = TimingBuilder::new();
        builder.set_output_resolution_discovery(false, true, 13, 5, 11, 7, 3);
        let timings = builder.finish();
        assert!(!timings.resolution_certified_hint);
        assert!(timings.resolution_drift_fallback);
        assert_eq!(timings.resolution_reconcile_scan_cells, 13);
        assert_eq!(timings.resolution_repair_scan_cells, 5);
        assert_eq!(timings.resolution_hint_cells, 11);
        assert_eq!(timings.resolution_hinted_candidates, 7);
        assert_eq!(timings.resolution_detected_edges, 3);
    }
}
