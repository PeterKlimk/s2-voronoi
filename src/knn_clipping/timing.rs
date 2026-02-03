//! Zero-cost timing instrumentation for knn_clipping.
//!
//! When the `timing` feature is enabled, this module provides timing
//! infrastructure that measures and reports phase durations.
//! When disabled, all timing code compiles away to nothing.
//!
//! Usage:
//!   cargo run --release --features timing

use std::time::Duration;

#[cfg(feature = "timing")]
use crate::cube_grid::CubeMapGridBuildTimings;
#[cfg(feature = "timing")]
use rustc_hash::FxHashMap;

/// Histogram of neighbors processed at termination.
/// Buckets 0-47: exact counts for 1-48 neighbors (bucket i = i+1 neighbors)
/// Bucket 48: 49-64 neighbors
/// Bucket 49: 65-96 neighbors
/// Bucket 50: 97+ neighbors
#[cfg(feature = "timing")]
pub const NEIGHBOR_HIST_BUCKETS: usize = 51;

/// Buckets for "incoming edgechecks provided" when collecting termination stats.
///
/// Buckets 0-15: exact edgecheck counts (bucket i = i edgechecks).
/// Bucket 16: 16+ edgechecks.
#[cfg(feature = "timing")]
pub const EDGECHECK_BUCKETS: usize = 17;

/// Histogram buckets for "extra neighbors beyond edgecheck seeds".
///
/// Bucket 0: 0 neighbors
/// Buckets 1-48: exact counts for 1-48 neighbors (bucket i = i neighbors)
/// Bucket 49: 49-64 neighbors
/// Bucket 50: 65-96 neighbors
/// Bucket 51: 97+ neighbors
#[cfg(feature = "timing")]
pub const EXTRA_NEIGHBOR_HIST_BUCKETS: usize = 52;

/// K-NN stage that a cell terminated at.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnnCellStage {
    /// Terminated during packed chunk0 (r=1).
    PackedChunk0,
    /// Terminated during packed tail (r=1, dot >= security).
    PackedTail,
    /// Terminated during resume stage with given K value
    Resume(usize),
    /// Terminated during restart stage with given K value
    Restart(usize),
    /// Ran full scan as fallback
    FullScanFallback,
}

/// Counts of which k-NN stage cells terminated at.
///
/// This is heavily performance-sensitive when timing is enabled: it is updated once per cell.
/// We keep the common stages in fixed counters and only fall back to maps for rare K values.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub struct StageCounts {
    pub packed_chunk0: u64,
    pub packed_tail: u64,
    pub full_scan: u64,
    pub resume_default: u64,
    pub restart_k0: u64,
    pub restart_kmax: u64,
    pub resume_other: FxHashMap<usize, u64>,
    pub restart_other: FxHashMap<usize, u64>,
}

#[cfg(feature = "timing")]
impl StageCounts {
    #[inline]
    pub fn add(&mut self, stage: KnnCellStage) {
        match stage {
            KnnCellStage::PackedChunk0 => self.packed_chunk0 += 1,
            KnnCellStage::PackedTail => self.packed_tail += 1,
            KnnCellStage::FullScanFallback => self.full_scan += 1,
            KnnCellStage::Resume(k) => {
                if k == super::KNN_RESUME_K {
                    self.resume_default += 1;
                } else {
                    *self.resume_other.entry(k).or_insert(0) += 1;
                }
            }
            KnnCellStage::Restart(k) => {
                if k == super::KNN_RESTART_K0 {
                    self.restart_k0 += 1;
                } else if k == super::KNN_RESTART_MAX {
                    self.restart_kmax += 1;
                } else {
                    *self.restart_other.entry(k).or_insert(0) += 1;
                }
            }
        }
    }

    #[inline]
    pub fn total(&self) -> u64 {
        let mut total = self.packed_chunk0
            + self.packed_tail
            + self.full_scan
            + self.resume_default
            + self.restart_k0
            + self.restart_kmax;
        total += self.resume_other.values().copied().sum::<u64>();
        total += self.restart_other.values().copied().sum::<u64>();
        total
    }

    pub fn iter_resume(&self) -> Vec<(usize, u64)> {
        let mut v = Vec::with_capacity(1 + self.resume_other.len());
        if self.resume_default > 0 {
            v.push((super::KNN_RESUME_K, self.resume_default));
        }
        v.extend(self.resume_other.iter().map(|(&k, &c)| (k, c)));
        v.sort_by_key(|(k, _)| *k);
        v
    }

    pub fn iter_restart(&self) -> Vec<(usize, u64)> {
        let mut v = Vec::with_capacity(2 + self.restart_other.len());
        if self.restart_k0 > 0 {
            v.push((super::KNN_RESTART_K0, self.restart_k0));
        }
        if self.restart_kmax > 0 {
            v.push((super::KNN_RESTART_MAX, self.restart_kmax));
        }
        v.extend(self.restart_other.iter().map(|(&k, &c)| (k, c)));
        v.sort_by_key(|(k, _)| *k);
        v
    }

    pub fn merge(&mut self, other: &StageCounts) {
        self.packed_chunk0 += other.packed_chunk0;
        self.packed_tail += other.packed_tail;
        self.full_scan += other.full_scan;
        self.resume_default += other.resume_default;
        self.restart_k0 += other.restart_k0;
        self.restart_kmax += other.restart_kmax;
        for (&k, &c) in &other.resume_other {
            *self.resume_other.entry(k).or_insert(0) += c;
        }
        for (&k, &c) in &other.restart_other {
            *self.restart_other.entry(k).or_insert(0) += c;
        }
    }
}

/// Sub-phase timings within cell construction.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub struct CellSubPhases {
    pub knn_query: Duration,
    pub packed_knn: Duration,
    pub packed_knn_setup: Duration,
    pub packed_knn_query_cache: Duration,
    pub packed_knn_security_thresholds: Duration,
    pub packed_knn_center_pass: Duration,
    pub packed_knn_ring_thresholds: Duration,
    pub packed_knn_ring_pass: Duration,
    pub packed_knn_ring_fallback: Duration,
    pub packed_knn_select_prep: Duration,
    pub packed_knn_select_query_prep: Duration,
    pub packed_knn_select_partition: Duration,
    pub packed_knn_select_sort: Duration,
    pub packed_knn_select_scatter: Duration,
    pub packed_knn_unaccounted: Duration,
    /// Packed-kNN sort-size histogram (call count, bucketed by n).
    pub packed_knn_sort_len_counts: [u64; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
    /// Packed-kNN sort-size histogram (time, in nanoseconds, bucketed by n).
    pub packed_knn_sort_len_nanos: [u64; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
    /// Packed-kNN sort-size histogram (call count, exact by n, capped + overflow).
    pub packed_knn_sort_len_exact_counts:
        [u64; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
    /// Packed-kNN sort-size histogram (time, in nanoseconds, exact by n, capped + overflow).
    pub packed_knn_sort_len_exact_nanos:
        [u64; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
    pub clipping: Duration,
    pub certification: Duration,
    /// Live-dedup: per-vertex ownership checks and shard-local dedup work during cell build.
    pub key_dedup: Duration,
    /// Per-cell edge check bookkeeping and matching.
    pub edge_checks: Duration,
    /// Per-cell edge collection (building edge lists).
    pub edge_collect: Duration,
    /// Per-cell edge check matching + index application.
    pub edge_resolve: Duration,
    /// Per-cell edge emission (pending + overflow).
    pub edge_emit: Duration,
    /// Per-cell k-NN stage distribution (final stage used per cell).
    pub stage_counts: StageCounts,
    /// Cells where the k-NN search loop exhausted (typically means it hit brute force).
    pub cells_knn_exhausted: u64,
    /// Cells that entered packed tail emission (even if they later fell back to kNN).
    pub cells_packed_tail_used: u64,
    /// Cells that exhausted all safe packed candidates in r=1 (even if they later fell back).
    pub cells_packed_safe_exhausted: u64,
    /// Cells that executed any kNN query stage (resume/restart growth).
    pub cells_used_knn: u64,
    /// Packed security mode usage (query count): interior u/v line planes.
    pub packed_security_planes_queries: u64,
    /// Packed security mode usage (query count): ring2 cap bound.
    pub packed_security_cap_queries: u64,
    /// Packed queries (cells) processed via packed_knn.
    pub packed_queries: u64,
    /// Packed queries (cells) with `tail_possible` set after `prepare_group_*`.
    pub packed_tail_possible_queries: u64,
    /// Packed query groups prepared successfully (PackedKnnCellStatus::Ok).
    pub packed_groups: u64,
    /// Packed groups that built tail candidates at least once.
    pub packed_tail_build_groups: u64,
    /// Histogram of neighbors processed at termination.
    pub neighbors_histogram: [u64; NEIGHBOR_HIST_BUCKETS],
    /// Histogram of "extra neighbors beyond edgecheck seeds", bucketed by incoming edgecheck count.
    pub edgecheck_extra_histogram: [[u64; EXTRA_NEIGHBOR_HIST_BUCKETS]; EDGECHECK_BUCKETS],
    /// Per-edgecheck bucket cell counts.
    pub edgecheck_bucket_counts: [u64; EDGECHECK_BUCKETS],
    /// Per-edgecheck bucket sum of edgecheck seed clips actually used.
    pub edgecheck_seed_used_sum: [u64; EDGECHECK_BUCKETS],
    /// Per-edgecheck bucket sum of extra neighbors (for mean reporting).
    pub edgecheck_extra_sum: [u64; EDGECHECK_BUCKETS],
    /// Packed-kNN: chunk0 candidate count sampled per packed query, bucketed by incoming edgechecks.
    pub packed_knn_chunk0_by_edgechecks_counts: [u64; EDGECHECK_BUCKETS],
    /// Packed-kNN: sum(chunk0_len) by incoming edgecheck bucket.
    pub packed_knn_chunk0_by_edgechecks_sum_chunk0: [u64; EDGECHECK_BUCKETS],
    /// Packed-kNN: sum(k_needed at chunk0) by incoming edgecheck bucket.
    pub packed_knn_chunk0_by_edgechecks_sum_needed: [u64; EDGECHECK_BUCKETS],
    /// Packed-kNN: sum(max(chunk0_len - k_needed, 0)) by incoming edgecheck bucket.
    pub packed_knn_chunk0_by_edgechecks_sum_excess: [u64; EDGECHECK_BUCKETS],
    /// Packed-kNN correlation sample count.
    pub packed_knn_chunk0_corr_n: u64,
    /// Pearson sums for corr(chunk0_len, incoming_edgechecks).
    pub packed_knn_chunk0_corr_sum_edge: f64,
    pub packed_knn_chunk0_corr_sum_edge2: f64,
    pub packed_knn_chunk0_corr_sum_chunk0: f64,
    pub packed_knn_chunk0_corr_sum_chunk02: f64,
    pub packed_knn_chunk0_corr_sum_chunk0_edge: f64,
    /// Pearson sums for corr(excess, incoming_edgechecks).
    pub packed_knn_chunk0_corr_sum_excess: f64,
    pub packed_knn_chunk0_corr_sum_excess2: f64,
    pub packed_knn_chunk0_corr_sum_excess_edge: f64,
}

#[cfg(feature = "timing")]
impl Default for CellSubPhases {
    fn default() -> Self {
        Self {
            knn_query: Duration::ZERO,
            packed_knn: Duration::ZERO,
            packed_knn_setup: Duration::ZERO,
            packed_knn_query_cache: Duration::ZERO,
            packed_knn_security_thresholds: Duration::ZERO,
            packed_knn_center_pass: Duration::ZERO,
            packed_knn_ring_thresholds: Duration::ZERO,
            packed_knn_ring_pass: Duration::ZERO,
            packed_knn_ring_fallback: Duration::ZERO,
            packed_knn_select_prep: Duration::ZERO,
            packed_knn_select_query_prep: Duration::ZERO,
            packed_knn_select_partition: Duration::ZERO,
            packed_knn_select_sort: Duration::ZERO,
            packed_knn_select_scatter: Duration::ZERO,
            packed_knn_unaccounted: Duration::ZERO,
            packed_knn_sort_len_counts: [0; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
            packed_knn_sort_len_nanos: [0; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
            packed_knn_sort_len_exact_counts: [0;
                crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
            packed_knn_sort_len_exact_nanos: [0; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX
                + 2],
            clipping: Duration::ZERO,
            certification: Duration::ZERO,
            key_dedup: Duration::ZERO,
            edge_checks: Duration::ZERO,
            edge_collect: Duration::ZERO,
            edge_resolve: Duration::ZERO,
            edge_emit: Duration::ZERO,
            stage_counts: StageCounts::default(),
            cells_knn_exhausted: 0,
            cells_packed_tail_used: 0,
            cells_packed_safe_exhausted: 0,
            cells_used_knn: 0,
            packed_security_planes_queries: 0,
            packed_security_cap_queries: 0,
            packed_queries: 0,
            packed_tail_possible_queries: 0,
            packed_groups: 0,
            packed_tail_build_groups: 0,
            neighbors_histogram: [0; NEIGHBOR_HIST_BUCKETS],
            edgecheck_extra_histogram: [[0; EXTRA_NEIGHBOR_HIST_BUCKETS]; EDGECHECK_BUCKETS],
            edgecheck_bucket_counts: [0; EDGECHECK_BUCKETS],
            edgecheck_seed_used_sum: [0; EDGECHECK_BUCKETS],
            edgecheck_extra_sum: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_counts: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_sum_chunk0: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_sum_needed: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_sum_excess: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_corr_n: 0,
            packed_knn_chunk0_corr_sum_edge: 0.0,
            packed_knn_chunk0_corr_sum_edge2: 0.0,
            packed_knn_chunk0_corr_sum_chunk0: 0.0,
            packed_knn_chunk0_corr_sum_chunk02: 0.0,
            packed_knn_chunk0_corr_sum_chunk0_edge: 0.0,
            packed_knn_chunk0_corr_sum_excess: 0.0,
            packed_knn_chunk0_corr_sum_excess2: 0.0,
            packed_knn_chunk0_corr_sum_excess_edge: 0.0,
        }
    }
}

/// Sub-phase timings within dedup.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub struct DedupSubPhases {
    /// Live-dedup: overflow bucketing before flush.
    pub overflow_collect: Duration,
    /// Live-dedup: overflow flush into owner shards.
    pub overflow_flush: Duration,
    /// Edge-check overflow sort/scan pass.
    pub edge_checks_overflow: Duration,
    /// Edge-check overflow sort time.
    pub edge_checks_overflow_sort: Duration,
    /// Edge-check overflow match/patch time.
    pub edge_checks_overflow_match: Duration,
    /// Deferred fallback map scan/patch.
    pub deferred_fallback: Duration,
    /// Live-dedup: concatenating shard vertex buffers.
    pub concat_vertices: Duration,
    /// Live-dedup: emitting cells in generator order.
    pub emit_cells: Duration,
    /// Number of triplet keys processed.
    pub triplet_keys: u64,
    /// Number of support keys processed.
    pub support_keys: u64,
    /// Number of bad edges detected (endpoint mismatch or one-sided edges).
    pub bad_edges_count: u64,
}

/// Dummy dedup sub-phases when feature is disabled.
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DedupSubPhases;

/// Phase timings for the Voronoi algorithm.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub struct PhaseTimings {
    pub total: Duration,
    pub preprocess: Duration,
    pub knn_build: Duration,
    pub knn_build_sub: Option<CubeMapGridBuildTimings>,
    pub cell_construction: Duration,
    pub cell_sub: CellSubPhases,
    pub dedup: Duration,
    pub dedup_sub: DedupSubPhases,
    pub edge_repair: Duration,
    pub assemble: Duration,
}

#[cfg(feature = "timing")]
impl PhaseTimings {
    pub fn report(&self, n: usize) {
        let total_ms = self.total.as_secs_f64() * 1000.0;
        let pct = |d: Duration| {
            if self.total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / self.total.as_secs_f64() * 100.0
            }
        };

        eprintln!("[timing] knn_clipping n={}", n);
        if self.preprocess.as_nanos() > 0 {
            eprintln!(
                "  preprocess:        {:7.1}ms ({:4.1}%)",
                self.preprocess.as_secs_f64() * 1000.0,
                pct(self.preprocess)
            );
        }
        eprintln!(
            "  knn_build:         {:7.1}ms ({:4.1}%)",
            self.knn_build.as_secs_f64() * 1000.0,
            pct(self.knn_build)
        );
        if let Some(sub) = &self.knn_build_sub {
            if sub.total().as_nanos() > 0 {
                // Sub-phase breakdown inside the grid build.
                let sub_pct_knn = |d: Duration| {
                    if self.knn_build.as_nanos() == 0 {
                        0.0
                    } else {
                        d.as_secs_f64() / self.knn_build.as_secs_f64() * 100.0
                    }
                };
                let ms = |d: Duration| d.as_secs_f64() * 1000.0;
                eprintln!(
                    "    grid_count:      {:7.1}ms ({:4.1}%)",
                    ms(sub.count_cells),
                    sub_pct_knn(sub.count_cells)
                );
                eprintln!(
                    "    grid_prefix:     {:7.1}ms ({:4.1}%)",
                    ms(sub.prefix_sum),
                    sub_pct_knn(sub.prefix_sum)
                );
                eprintln!(
                    "    grid_scatter:    {:7.1}ms ({:4.1}%)",
                    ms(sub.scatter_soa),
                    sub_pct_knn(sub.scatter_soa)
                );
                eprintln!(
                    "    grid_neighbors:  {:7.1}ms ({:4.1}%)",
                    ms(sub.neighbors),
                    sub_pct_knn(sub.neighbors)
                );
                eprintln!(
                    "    grid_ring2:      {:7.1}ms ({:4.1}%)",
                    ms(sub.ring2),
                    sub_pct_knn(sub.ring2)
                );
                eprintln!(
                    "    grid_bounds:     {:7.1}ms ({:4.1}%)",
                    ms(sub.cell_bounds),
                    sub_pct_knn(sub.cell_bounds)
                );
                eprintln!(
                    "    grid_security:   {:7.1}ms ({:4.1}%)",
                    ms(sub.security_3x3),
                    sub_pct_knn(sub.security_3x3)
                );
            }
        }
        eprintln!(
            "  cell_construction: {:7.1}ms ({:4.1}%)",
            self.cell_construction.as_secs_f64() * 1000.0,
            pct(self.cell_construction)
        );

        // Sub-phase breakdown: estimate wall time from CPU time using parent ratio
        let cpu_total = self.cell_sub.knn_query
            + self.cell_sub.packed_knn
            + self.cell_sub.clipping
            + self.cell_sub.certification
            + self.cell_sub.key_dedup
            + self.cell_sub.edge_checks;
        let cpu_total_secs = cpu_total.as_secs_f64();
        let wall_secs = self.cell_construction.as_secs_f64();

        // Ratio to convert CPU time to estimated wall time
        let cpu_to_wall = if cpu_total_secs > 0.0 {
            wall_secs / cpu_total_secs
        } else {
            1.0
        };
        let parallelism = if wall_secs > 0.0 {
            cpu_total_secs / wall_secs
        } else {
            1.0
        };

        let sub_pct = |d: Duration| {
            if cpu_total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / cpu_total_secs * 100.0
            }
        };
        let est_wall_ms = |d: Duration| d.as_secs_f64() * cpu_to_wall * 1000.0;

        // Optional machine-readable one-liner for scripts.
        //
        // Intentionally uses the same "estimated wall ms" convention as the human report for
        // cell sub-phases, so comparisons match what you see printed (especially in parallel).
        if std::env::var_os("S2_VORONOI_TIMING_KV").is_some() {
            eprintln!(
                concat!(
                    "TIMING_KV n={n} ",
                    "total_ms={total:.3} preprocess_ms={pre:.3} knn_build_ms={kb:.3} ",
                    "cell_construction_ms={cc:.3} dedup_ms={dd:.3} edge_repair_ms={er:.3} assemble_ms={asmb:.3} ",
                    "packed_knn_ms={pk:.3} pk_setup_ms={pks:.3} pk_queries_ms={pkq:.3} pk_security_ms={pksec:.3} ",
                    "pk_center_ms={pkc:.3} pk_thresh_ms={pkt:.3} pk_ring_ms={pkr:.3} pk_fallback_ms={pkf:.3} ",
                    "pk_sel_prep_ms={pksp:.3} pk_sel_qprep_ms={pksq:.3} pk_partition_ms={pkp:.3} pk_sort_ms={pksort:.3} ",
                    "pk_scatter_ms={pksc:.3} pk_unaccounted_ms={pku:.3}"
                ),
                n = n,
                total = total_ms,
                pre = self.preprocess.as_secs_f64() * 1000.0,
                kb = self.knn_build.as_secs_f64() * 1000.0,
                cc = self.cell_construction.as_secs_f64() * 1000.0,
                dd = self.dedup.as_secs_f64() * 1000.0,
                er = self.edge_repair.as_secs_f64() * 1000.0,
                asmb = self.assemble.as_secs_f64() * 1000.0,
                pk = est_wall_ms(self.cell_sub.packed_knn),
                pks = est_wall_ms(self.cell_sub.packed_knn_setup),
                pkq = est_wall_ms(self.cell_sub.packed_knn_query_cache),
                pksec = est_wall_ms(self.cell_sub.packed_knn_security_thresholds),
                pkc = est_wall_ms(self.cell_sub.packed_knn_center_pass),
                pkt = est_wall_ms(self.cell_sub.packed_knn_ring_thresholds),
                pkr = est_wall_ms(self.cell_sub.packed_knn_ring_pass),
                pkf = est_wall_ms(self.cell_sub.packed_knn_ring_fallback),
                pksp = est_wall_ms(self.cell_sub.packed_knn_select_prep),
                pksq = est_wall_ms(self.cell_sub.packed_knn_select_query_prep),
                pkp = est_wall_ms(self.cell_sub.packed_knn_select_partition),
                pksort = est_wall_ms(self.cell_sub.packed_knn_select_sort),
                pksc = est_wall_ms(self.cell_sub.packed_knn_select_scatter),
                pku = est_wall_ms(self.cell_sub.packed_knn_unaccounted),
            );
        }

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
            let pk_total = self.cell_sub.packed_knn;
            let pk_pct = |d: Duration| {
                if pk_total.as_nanos() == 0 {
                    0.0
                } else {
                    d.as_secs_f64() / pk_total.as_secs_f64() * 100.0
                }
            };

            if self.cell_sub.packed_knn_setup.as_nanos() > 0 {
                eprintln!(
                    "      pk_setup:     {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_setup),
                    pk_pct(self.cell_sub.packed_knn_setup)
                );
            }
            if self.cell_sub.packed_knn_query_cache.as_nanos() > 0 {
                eprintln!(
                    "      pk_queries:   {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_query_cache),
                    pk_pct(self.cell_sub.packed_knn_query_cache)
                );
            }
            if self.cell_sub.packed_knn_security_thresholds.as_nanos() > 0 {
                eprintln!(
                    "      pk_security:  {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_security_thresholds),
                    pk_pct(self.cell_sub.packed_knn_security_thresholds)
                );
            }
            if self.cell_sub.packed_knn_center_pass.as_nanos() > 0 {
                eprintln!(
                    "      pk_center:    {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_center_pass),
                    pk_pct(self.cell_sub.packed_knn_center_pass)
                );
            }
            if self.cell_sub.packed_knn_ring_thresholds.as_nanos() > 0 {
                eprintln!(
                    "      pk_thresh:    {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_ring_thresholds),
                    pk_pct(self.cell_sub.packed_knn_ring_thresholds)
                );
            }
            if self.cell_sub.packed_knn_ring_pass.as_nanos() > 0 {
                eprintln!(
                    "      pk_ring:      {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_ring_pass),
                    pk_pct(self.cell_sub.packed_knn_ring_pass)
                );
            }
            if self.cell_sub.packed_knn_ring_fallback.as_nanos() > 0 {
                eprintln!(
                    "      pk_fallback:  {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_ring_fallback),
                    pk_pct(self.cell_sub.packed_knn_ring_fallback)
                );
            }
            if self.cell_sub.packed_knn_select_prep.as_nanos() > 0 {
                eprintln!(
                    "      pk_sel_prep:  {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_select_prep),
                    pk_pct(self.cell_sub.packed_knn_select_prep)
                );
            }
            if self.cell_sub.packed_knn_select_query_prep.as_nanos() > 0 {
                eprintln!(
                    "      pk_sel_qprep: {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_select_query_prep),
                    pk_pct(self.cell_sub.packed_knn_select_query_prep)
                );
            }
            if self.cell_sub.packed_knn_select_partition.as_nanos() > 0 {
                eprintln!(
                    "      pk_partition:{:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_select_partition),
                    pk_pct(self.cell_sub.packed_knn_select_partition)
                );
            }
            if self.cell_sub.packed_knn_select_sort.as_nanos() > 0 {
                eprintln!(
                    "      pk_sort:      {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_select_sort),
                    pk_pct(self.cell_sub.packed_knn_select_sort)
                );
                if std::env::var_os("S2_VORONOI_TIMING_PK_SORT_HIST").is_some() {
                    let nanos = &self.cell_sub.packed_knn_sort_len_nanos;
                    let counts = &self.cell_sub.packed_knn_sort_len_counts;
                    let total_nanos: u64 = nanos.iter().sum();
                    if total_nanos > 0 {
                        use std::cmp::Reverse;

                        let bins = crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS;
                        let label = |b: usize| -> String {
                            if b == 0 {
                                "n=0".to_string()
                            } else if b == 1 {
                                "n=1".to_string()
                            } else if b == 2 {
                                "n=2".to_string()
                            } else {
                                let lower = (1usize << (b - 2)) + 1;
                                if b == bins - 1 {
                                    format!("n>={}", lower)
                                } else {
                                    let upper = 1usize << (b - 1);
                                    format!("{}-{}", lower, upper)
                                }
                            }
                        };

                        let mut v: Vec<(usize, u64, u64)> = nanos
                            .iter()
                            .enumerate()
                            .filter_map(|(b, &t)| {
                                if t == 0 {
                                    None
                                } else {
                                    Some((b, t, counts[b]))
                                }
                            })
                            .collect();
                        v.sort_by_key(|&(_, t, _)| Reverse(t));

                        let mut detail = String::from("      pk_sort_hist:");
                        for (b, t, c) in v.into_iter().take(6) {
                            let ms = (t as f64) / 1e6;
                            let pct = (t as f64) / (total_nanos as f64) * 100.0;
                            detail.push_str(&format!(
                                " {}={:.1}ms({:.1}%,c={})",
                                label(b),
                                ms,
                                pct,
                                c
                            ));
                        }
                        eprintln!("{}", detail);
                    }
                }
                if std::env::var_os("S2_VORONOI_TIMING_PK_SORT_PER_N").is_some() {
                    let nanos = &self.cell_sub.packed_knn_sort_len_exact_nanos;
                    let counts = &self.cell_sub.packed_knn_sort_len_exact_counts;
                    let total_nanos: u64 = nanos.iter().sum();
                    if total_nanos > 0 {
                        use std::cmp::Reverse;

                        let max_n = crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX;
                        let mut v: Vec<(usize, u64, u64)> = nanos
                            .iter()
                            .enumerate()
                            .filter_map(|(n, &t)| {
                                if t == 0 {
                                    None
                                } else {
                                    Some((n, t, counts[n]))
                                }
                            })
                            .collect();
                        v.sort_by_key(|&(_, t, _)| Reverse(t));

                        let mut detail = String::from("      pk_sort_per_n:");
                        for (n, t, c) in v.into_iter().take(10) {
                            let label = if n == max_n + 1 {
                                format!("n>{}", max_n)
                            } else {
                                format!("n={}", n)
                            };
                            let ms = (t as f64) / 1e6;
                            let pct = (t as f64) / (total_nanos as f64) * 100.0;
                            detail
                                .push_str(&format!(" {}={:.1}ms({:.1}%,c={})", label, ms, pct, c));
                        }
                        eprintln!("{}", detail);
                    }
                }
            }
            if self.cell_sub.packed_knn_select_scatter.as_nanos() > 0 {
                eprintln!(
                    "      pk_scatter:   {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_select_scatter),
                    pk_pct(self.cell_sub.packed_knn_select_scatter)
                );
            }
            if self.cell_sub.packed_knn_unaccounted.as_nanos() > 0 {
                eprintln!(
                    "      pk_unaccount: {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.packed_knn_unaccounted),
                    pk_pct(self.cell_sub.packed_knn_unaccounted)
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
        if self.cell_sub.key_dedup.as_nanos() > 0 {
            eprintln!(
                "    key_dedup:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.key_dedup),
                sub_pct(self.cell_sub.key_dedup)
            );
        }
        if self.cell_sub.edge_checks.as_nanos() > 0 {
            eprintln!(
                "    edge_checks:    {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.edge_checks),
                sub_pct(self.cell_sub.edge_checks)
            );
            if self.cell_sub.edge_collect.as_nanos() > 0 {
                eprintln!(
                    "      edge_collect:{:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.edge_collect),
                    sub_pct(self.cell_sub.edge_collect)
                );
            }
            if self.cell_sub.edge_resolve.as_nanos() > 0 {
                eprintln!(
                    "      edge_resolve:{:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.edge_resolve),
                    sub_pct(self.cell_sub.edge_resolve)
                );
            }
            if self.cell_sub.edge_emit.as_nanos() > 0 {
                eprintln!(
                    "      edge_emit:   {:7.1}ms ({:4.1}%)",
                    est_wall_ms(self.cell_sub.edge_emit),
                    sub_pct(self.cell_sub.edge_emit)
                );
            }
        }
        eprintln!("    ({:.1}x parallelism)", parallelism);

        // Collect and sort stage counts for display
        let total_cells: u64 = self.cell_sub.stage_counts.total().max(1);
        let pct_cells = |c: u64| c as f64 / total_cells as f64 * 100.0;

        let resume_stages = self.cell_sub.stage_counts.iter_resume();
        let restart_stages = self.cell_sub.stage_counts.iter_restart();

        let full_scan = self.cell_sub.stage_counts.full_scan;

        // Build output string
        let packed_chunk0 = self.cell_sub.stage_counts.packed_chunk0;
        let packed_tail = self.cell_sub.stage_counts.packed_tail;

        let mut stages_str = String::from("    cell_stages:");
        if packed_chunk0 > 0 {
            stages_str.push_str(&format!(
                " pk0={} ({:.1}%)",
                packed_chunk0,
                pct_cells(packed_chunk0)
            ));
        }
        if packed_tail > 0 {
            stages_str.push_str(&format!(
                " pk_tail={} ({:.1}%)",
                packed_tail,
                pct_cells(packed_tail)
            ));
        }
        for (k, count) in &resume_stages {
            stages_str.push_str(&format!(" k{}={} ({:.1}%)", k, count, pct_cells(*count)));
        }
        for (k, count) in &restart_stages {
            stages_str.push_str(&format!(" K{}={} ({:.1}%)", k, count, pct_cells(*count)));
        }
        if full_scan > 0 {
            stages_str.push_str(&format!(
                " full_scan={} ({:.1}%)",
                full_scan,
                pct_cells(full_scan)
            ));
        }

        stages_str.push_str(&format!(
            " exhausted={} ({:.1}%)",
            self.cell_sub.cells_knn_exhausted,
            pct_cells(self.cell_sub.cells_knn_exhausted)
        ));
        eprintln!("{}", stages_str);
        eprintln!(
            "    packed: tail_used={} ({:.1}%) safe_exhausted={} ({:.1}%)",
            self.cell_sub.cells_packed_tail_used,
            pct_cells(self.cell_sub.cells_packed_tail_used),
            self.cell_sub.cells_packed_safe_exhausted,
            pct_cells(self.cell_sub.cells_packed_safe_exhausted)
        );
        if self.cell_sub.packed_queries > 0 {
            let pct = (self.cell_sub.packed_tail_possible_queries as f64)
                / (self.cell_sub.packed_queries as f64)
                * 100.0;
            eprintln!(
                "    pk_tail_possible: q={} ({:.1}%)",
                self.cell_sub.packed_tail_possible_queries, pct
            );
        }
        if self.cell_sub.packed_groups > 0 {
            let pct = (self.cell_sub.packed_tail_build_groups as f64)
                / (self.cell_sub.packed_groups as f64)
                * 100.0;
            eprintln!(
                "    pk_tail_build: groups={} ({:.1}%)",
                self.cell_sub.packed_tail_build_groups, pct
            );
        }
        let sec_total = self.cell_sub.packed_security_planes_queries
            + self.cell_sub.packed_security_cap_queries;
        if sec_total > 0 {
            let pct_q = |n: u64| (n as f64) / (sec_total as f64) * 100.0;
            eprintln!(
                "    pk_security: planes_q={} ({:.1}%) caps_q={} ({:.1}%)",
                self.cell_sub.packed_security_planes_queries,
                pct_q(self.cell_sub.packed_security_planes_queries),
                self.cell_sub.packed_security_cap_queries,
                pct_q(self.cell_sub.packed_security_cap_queries)
            );
        }
        eprintln!(
            "    knn: used={} ({:.1}%)",
            self.cell_sub.cells_used_knn,
            pct_cells(self.cell_sub.cells_used_knn)
        );

        // Neighbor histogram: compute percentiles
        let hist = &self.cell_sub.neighbors_histogram;
        let hist_total: u64 = hist.iter().sum();
        if hist_total > 0 {
            // Convert bucket index to neighbor count
            let bucket_to_neighbors = |bucket: usize| -> usize {
                if bucket < 48 {
                    bucket + 1 // buckets 0-47 = 1-48 neighbors
                } else if bucket == 48 {
                    64 // 49-64 range, report upper bound
                } else if bucket == 49 {
                    96 // 65-96 range
                } else {
                    97 // 97+ range
                }
            };

            // Find percentiles by scanning cumulative distribution
            let find_percentile = |p: f64| -> usize {
                // Use ceil to avoid a 0 target on small sample sizes.
                let target = ((hist_total as f64 * p).ceil() as u64).max(1);
                let mut cumulative = 0u64;
                for (bucket, &count) in hist.iter().enumerate() {
                    cumulative += count;
                    if cumulative >= target {
                        return bucket_to_neighbors(bucket);
                    }
                }
                bucket_to_neighbors(NEIGHBOR_HIST_BUCKETS - 1)
            };

            // Find max (last non-zero bucket)
            let max_bucket = hist
                .iter()
                .enumerate()
                .rev()
                .find(|(_, &c)| c > 0)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let max_neighbors = bucket_to_neighbors(max_bucket);

            eprintln!(
                "    neighbors: p50={} p90={} p99={} max={}",
                find_percentile(0.50),
                find_percentile(0.90),
                find_percentile(0.99),
                max_neighbors,
            );

            // Dump non-zero buckets as: n=count (pct%)
            let mut detail = String::from("    neighbors_detail:");
            for (bucket, &count) in hist.iter().enumerate() {
                if count > 0 {
                    let n = bucket_to_neighbors(bucket);
                    let pct = count as f64 / hist_total as f64 * 100.0;
                    detail.push_str(&format!(" {}={:.1}%", n, pct));
                }
            }
            eprintln!("{}", detail);
        }

        // Extra neighbors needed (beyond edgecheck seed clips), bucketed by incoming edgechecks.
        let by_edge = &self.cell_sub.edgecheck_extra_histogram;
        let edge_counts = &self.cell_sub.edgecheck_bucket_counts;
        if edge_counts.iter().any(|&c| c > 0) {
            // Convert histogram bucket index to extra-neighbor count (reported as an upper bound
            // for the wide buckets).
            let bucket_to_extra_neighbors = |bucket: usize| -> usize {
                if bucket == 0 {
                    0
                } else if bucket <= 48 {
                    bucket // 1..=48 are exact
                } else if bucket == 49 {
                    64
                } else if bucket == 50 {
                    96
                } else {
                    97
                }
            };

            let find_percentile =
                |hist: &[u64; EXTRA_NEIGHBOR_HIST_BUCKETS], total: u64, p: f64| -> usize {
                    // Use ceil to avoid a 0 target on small sample sizes.
                    let target = ((total as f64 * p).ceil() as u64).max(1);
                    let mut cumulative = 0u64;
                    for (bucket, &count) in hist.iter().enumerate() {
                        cumulative += count;
                        if cumulative >= target {
                            return bucket_to_extra_neighbors(bucket);
                        }
                    }
                    bucket_to_extra_neighbors(EXTRA_NEIGHBOR_HIST_BUCKETS - 1)
                };

            eprintln!("    extra_neighbors_by_edgechecks:");
            for edge_bucket in 0..EDGECHECK_BUCKETS {
                let cells = edge_counts[edge_bucket];
                if cells == 0 {
                    continue;
                }
                let hist = &by_edge[edge_bucket];
                let hist_total: u64 = hist.iter().sum();
                if hist_total == 0 {
                    continue;
                }
                let max_bucket = hist
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, &c)| c > 0)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let max_extra = bucket_to_extra_neighbors(max_bucket);
                let label = if edge_bucket == EDGECHECK_BUCKETS - 1 {
                    "e>=16".to_string()
                } else {
                    format!("e={}", edge_bucket)
                };
                let seed_mean =
                    (self.cell_sub.edgecheck_seed_used_sum[edge_bucket] as f64) / (cells as f64);
                let extra_mean =
                    (self.cell_sub.edgecheck_extra_sum[edge_bucket] as f64) / (cells as f64);
                eprintln!(
                    "      {:>5}: cells={} seed_mean={:.2} extra_mean={:.2} extra_p50={} extra_p90={} extra_p99={} extra_max={}",
                    label,
                    cells,
                    seed_mean,
                    extra_mean,
                    find_percentile(hist, hist_total, 0.50),
                    find_percentile(hist, hist_total, 0.90),
                    find_percentile(hist, hist_total, 0.99),
                    max_extra,
                );
            }
        }

        // Packed-kNN: chunk0 size / overfetch vs incoming edgechecks.
        let pk_counts = &self.cell_sub.packed_knn_chunk0_by_edgechecks_counts;
        if pk_counts.iter().any(|&c| c > 0) {
            let pk_sum_chunk0 = &self.cell_sub.packed_knn_chunk0_by_edgechecks_sum_chunk0;
            let pk_sum_needed = &self.cell_sub.packed_knn_chunk0_by_edgechecks_sum_needed;
            let pk_sum_excess = &self.cell_sub.packed_knn_chunk0_by_edgechecks_sum_excess;

            let pearson = |n: f64, sum_x: f64, sum_y: f64, sum_x2: f64, sum_y2: f64, sum_xy: f64| {
                if n < 2.0 {
                    return f64::NAN;
                }
                let num = n * sum_xy - sum_x * sum_y;
                let den_x = n * sum_x2 - sum_x * sum_x;
                let den_y = n * sum_y2 - sum_y * sum_y;
                if den_x <= 0.0 || den_y <= 0.0 {
                    return f64::NAN;
                }
                num / (den_x * den_y).sqrt()
            };

            let n = self.cell_sub.packed_knn_chunk0_corr_n as f64;
            let corr_chunk0 = pearson(
                n,
                self.cell_sub.packed_knn_chunk0_corr_sum_edge,
                self.cell_sub.packed_knn_chunk0_corr_sum_chunk0,
                self.cell_sub.packed_knn_chunk0_corr_sum_edge2,
                self.cell_sub.packed_knn_chunk0_corr_sum_chunk02,
                self.cell_sub.packed_knn_chunk0_corr_sum_chunk0_edge,
            );
            let corr_excess = pearson(
                n,
                self.cell_sub.packed_knn_chunk0_corr_sum_edge,
                self.cell_sub.packed_knn_chunk0_corr_sum_excess,
                self.cell_sub.packed_knn_chunk0_corr_sum_edge2,
                self.cell_sub.packed_knn_chunk0_corr_sum_excess2,
                self.cell_sub.packed_knn_chunk0_corr_sum_excess_edge,
            );

            eprintln!("    packed_chunk0_by_edgechecks:");
            for edge_bucket in 0..EDGECHECK_BUCKETS {
                let cells = pk_counts[edge_bucket];
                if cells == 0 {
                    continue;
                }
                let label = if edge_bucket == EDGECHECK_BUCKETS - 1 {
                    "e>=16".to_string()
                } else {
                    format!("e={}", edge_bucket)
                };
                let denom = cells as f64;
                let mean_chunk0 = (pk_sum_chunk0[edge_bucket] as f64) / denom;
                let mean_needed = (pk_sum_needed[edge_bucket] as f64) / denom;
                let mean_excess = (pk_sum_excess[edge_bucket] as f64) / denom;
                eprintln!(
                    "      {:>5}: cells={} chunk0_mean={:.2} need_mean={:.2} excess_mean={:.2}",
                    label, cells, mean_chunk0, mean_needed, mean_excess
                );
            }
            eprintln!(
                "      corr(chunk0_len, edgechecks)={:.3} corr(excess, edgechecks)={:.3} samples={}",
                corr_chunk0, corr_excess, self.cell_sub.packed_knn_chunk0_corr_n
            );
        }

        eprintln!(
            "  dedup:             {:7.1}ms ({:4.1}%)",
            self.dedup.as_secs_f64() * 1000.0,
            pct(self.dedup)
        );

        // Dedup sub-phase breakdown
        let dedup_total = self.dedup_sub.overflow_collect
            + self.dedup_sub.overflow_flush
            + self.dedup_sub.edge_checks_overflow
            + self.dedup_sub.deferred_fallback
            + self.dedup_sub.concat_vertices
            + self.dedup_sub.emit_cells;
        if dedup_total.as_nanos() > 0 {
            let dedup_pct = |d: Duration| d.as_secs_f64() / dedup_total.as_secs_f64() * 100.0;
            if self.dedup_sub.overflow_collect.as_nanos() > 0 {
                eprintln!(
                    "    overflow_collect:{:7.1}ms ({:4.1}%)",
                    self.dedup_sub.overflow_collect.as_secs_f64() * 1000.0,
                    dedup_pct(self.dedup_sub.overflow_collect)
                );
            }
            if self.dedup_sub.overflow_flush.as_nanos() > 0 {
                eprintln!(
                    "    overflow_flush: {:7.1}ms ({:4.1}%)",
                    self.dedup_sub.overflow_flush.as_secs_f64() * 1000.0,
                    dedup_pct(self.dedup_sub.overflow_flush)
                );
            }
            if self.dedup_sub.edge_checks_overflow.as_nanos() > 0 {
                eprintln!(
                    "    edge_checks:    {:7.1}ms ({:4.1}%)",
                    self.dedup_sub.edge_checks_overflow.as_secs_f64() * 1000.0,
                    dedup_pct(self.dedup_sub.edge_checks_overflow)
                );
                if self.dedup_sub.edge_checks_overflow_sort.as_nanos() > 0 {
                    eprintln!(
                        "      edge_sort:   {:7.1}ms ({:4.1}%)",
                        self.dedup_sub.edge_checks_overflow_sort.as_secs_f64() * 1000.0,
                        dedup_pct(self.dedup_sub.edge_checks_overflow_sort)
                    );
                }
                if self.dedup_sub.edge_checks_overflow_match.as_nanos() > 0 {
                    eprintln!(
                        "      edge_match:  {:7.1}ms ({:4.1}%)",
                        self.dedup_sub.edge_checks_overflow_match.as_secs_f64() * 1000.0,
                        dedup_pct(self.dedup_sub.edge_checks_overflow_match)
                    );
                }
            }
            if self.dedup_sub.deferred_fallback.as_nanos() > 0 {
                eprintln!(
                    "    deferred_fallback:{:5.1}ms ({:4.1}%)",
                    self.dedup_sub.deferred_fallback.as_secs_f64() * 1000.0,
                    dedup_pct(self.dedup_sub.deferred_fallback)
                );
            }
            if self.dedup_sub.concat_vertices.as_nanos() > 0 {
                eprintln!(
                    "    concat_vertices:{:7.1}ms ({:4.1}%)",
                    self.dedup_sub.concat_vertices.as_secs_f64() * 1000.0,
                    dedup_pct(self.dedup_sub.concat_vertices)
                );
            }
            if self.dedup_sub.emit_cells.as_nanos() > 0 {
                eprintln!(
                    "    emit_cells:     {:7.1}ms ({:4.1}%)",
                    self.dedup_sub.emit_cells.as_secs_f64() * 1000.0,
                    dedup_pct(self.dedup_sub.emit_cells)
                );
            }
            let total_keys = self.dedup_sub.triplet_keys + self.dedup_sub.support_keys;
            if total_keys > 0 {
                let key_pct = |k: u64| k as f64 / total_keys as f64 * 100.0;
                eprintln!(
                    "    keys: triplet={} ({:.1}%) support={} ({:.1}%)",
                    self.dedup_sub.triplet_keys,
                    key_pct(self.dedup_sub.triplet_keys),
                    self.dedup_sub.support_keys,
                    key_pct(self.dedup_sub.support_keys),
                );
            }
        }

        // Bad edges detected (before repair)
        if self.dedup_sub.bad_edges_count > 0 {
            eprintln!(
                "  bad_edges:         {} detected",
                self.dedup_sub.bad_edges_count
            );
        }

        if self.edge_repair.as_nanos() > 0 {
            eprintln!(
                "  edge_repair:       {:7.1}ms ({:4.1}%)",
                self.edge_repair.as_secs_f64() * 1000.0,
                pct(self.edge_repair)
            );
        }

        eprintln!(
            "  assemble:          {:7.1}ms ({:4.1}%)",
            self.assemble.as_secs_f64() * 1000.0,
            pct(self.assemble)
        );
        eprintln!("  total:             {:7.1}ms", total_ms);
    }
}

/// Dummy sub-phases when feature is disabled.
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CellSubPhases;

/// Dummy timings when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy)]
pub struct PhaseTimings;

#[cfg(not(feature = "timing"))]
impl PhaseTimings {
    #[inline(always)]
    pub fn report(&self, _n: usize) {}
}

/// Timer that tracks elapsed time when timing is enabled.
#[cfg(feature = "timing")]
pub struct Timer(std::time::Instant);

#[cfg(feature = "timing")]
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

/// Dummy timer when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
pub struct Timer;

#[cfg(not(feature = "timing"))]
impl Timer {
    #[inline(always)]
    pub fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

/// Timer optimized for sequential sub-phase timing: each `lap()` uses a single `Instant::now()`.
#[cfg(feature = "timing")]
pub struct LapTimer(std::time::Instant);

#[cfg(feature = "timing")]
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

/// Dummy lap timer when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
pub struct LapTimer;

#[cfg(not(feature = "timing"))]
impl LapTimer {
    #[inline(always)]
    pub fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub fn lap(&mut self) -> Duration {
        Duration::ZERO
    }
}

/// Accumulator for cell sub-phase timings (used per-chunk, then merged).
#[cfg(feature = "timing")]
#[derive(Clone)]
pub struct CellSubAccum {
    pub knn_query: Duration,
    pub packed_knn: Duration,
    pub packed_knn_setup: Duration,
    pub packed_knn_query_cache: Duration,
    pub packed_knn_security_thresholds: Duration,
    pub packed_knn_center_pass: Duration,
    pub packed_knn_ring_thresholds: Duration,
    pub packed_knn_ring_pass: Duration,
    pub packed_knn_ring_fallback: Duration,
    pub packed_knn_select_prep: Duration,
    pub packed_knn_select_query_prep: Duration,
    pub packed_knn_select_partition: Duration,
    pub packed_knn_select_sort: Duration,
    pub packed_knn_select_scatter: Duration,
    pub packed_knn_unaccounted: Duration,
    pub packed_knn_sort_len_counts: [u64; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
    pub packed_knn_sort_len_nanos: [u64; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
    pub packed_knn_sort_len_exact_counts:
        [u64; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
    pub packed_knn_sort_len_exact_nanos:
        [u64; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
    pub clipping: Duration,
    pub certification: Duration,
    pub key_dedup: Duration,
    pub edge_checks: Duration,
    pub edge_collect: Duration,
    pub edge_resolve: Duration,
    pub edge_emit: Duration,
    pub stage_counts: StageCounts,
    pub cells_knn_exhausted: u64,
    pub cells_packed_tail_used: u64,
    pub cells_packed_safe_exhausted: u64,
    pub cells_used_knn: u64,
    pub packed_security_planes_queries: u64,
    pub packed_security_cap_queries: u64,
    pub packed_queries: u64,
    pub packed_tail_possible_queries: u64,
    pub packed_groups: u64,
    pub packed_tail_build_groups: u64,
    pub neighbors_histogram: [u64; NEIGHBOR_HIST_BUCKETS],
    pub edgecheck_extra_histogram: [[u64; EXTRA_NEIGHBOR_HIST_BUCKETS]; EDGECHECK_BUCKETS],
    pub edgecheck_bucket_counts: [u64; EDGECHECK_BUCKETS],
    pub edgecheck_seed_used_sum: [u64; EDGECHECK_BUCKETS],
    pub edgecheck_extra_sum: [u64; EDGECHECK_BUCKETS],
    pub packed_knn_chunk0_by_edgechecks_counts: [u64; EDGECHECK_BUCKETS],
    pub packed_knn_chunk0_by_edgechecks_sum_chunk0: [u64; EDGECHECK_BUCKETS],
    pub packed_knn_chunk0_by_edgechecks_sum_needed: [u64; EDGECHECK_BUCKETS],
    pub packed_knn_chunk0_by_edgechecks_sum_excess: [u64; EDGECHECK_BUCKETS],
    pub packed_knn_chunk0_corr_n: u64,
    pub packed_knn_chunk0_corr_sum_edge: f64,
    pub packed_knn_chunk0_corr_sum_edge2: f64,
    pub packed_knn_chunk0_corr_sum_chunk0: f64,
    pub packed_knn_chunk0_corr_sum_chunk02: f64,
    pub packed_knn_chunk0_corr_sum_chunk0_edge: f64,
    pub packed_knn_chunk0_corr_sum_excess: f64,
    pub packed_knn_chunk0_corr_sum_excess2: f64,
    pub packed_knn_chunk0_corr_sum_excess_edge: f64,
}

#[cfg(feature = "timing")]
impl Default for CellSubAccum {
    fn default() -> Self {
        Self {
            knn_query: Duration::ZERO,
            packed_knn: Duration::ZERO,
            packed_knn_setup: Duration::ZERO,
            packed_knn_query_cache: Duration::ZERO,
            packed_knn_security_thresholds: Duration::ZERO,
            packed_knn_center_pass: Duration::ZERO,
            packed_knn_ring_thresholds: Duration::ZERO,
            packed_knn_ring_pass: Duration::ZERO,
            packed_knn_ring_fallback: Duration::ZERO,
            packed_knn_select_prep: Duration::ZERO,
            packed_knn_select_query_prep: Duration::ZERO,
            packed_knn_select_partition: Duration::ZERO,
            packed_knn_select_sort: Duration::ZERO,
            packed_knn_select_scatter: Duration::ZERO,
            packed_knn_unaccounted: Duration::ZERO,
            packed_knn_sort_len_counts: [0; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
            packed_knn_sort_len_nanos: [0; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
            packed_knn_sort_len_exact_counts: [0;
                crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
            packed_knn_sort_len_exact_nanos: [0; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX
                + 2],
            clipping: Duration::ZERO,
            certification: Duration::ZERO,
            key_dedup: Duration::ZERO,
            edge_checks: Duration::ZERO,
            edge_collect: Duration::ZERO,
            edge_resolve: Duration::ZERO,
            edge_emit: Duration::ZERO,
            stage_counts: StageCounts::default(),
            cells_knn_exhausted: 0,
            cells_packed_tail_used: 0,
            cells_packed_safe_exhausted: 0,
            cells_used_knn: 0,
            packed_security_planes_queries: 0,
            packed_security_cap_queries: 0,
            packed_queries: 0,
            packed_tail_possible_queries: 0,
            packed_groups: 0,
            packed_tail_build_groups: 0,
            neighbors_histogram: [0; NEIGHBOR_HIST_BUCKETS],
            edgecheck_extra_histogram: [[0; EXTRA_NEIGHBOR_HIST_BUCKETS]; EDGECHECK_BUCKETS],
            edgecheck_bucket_counts: [0; EDGECHECK_BUCKETS],
            edgecheck_seed_used_sum: [0; EDGECHECK_BUCKETS],
            edgecheck_extra_sum: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_counts: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_sum_chunk0: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_sum_needed: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_by_edgechecks_sum_excess: [0; EDGECHECK_BUCKETS],
            packed_knn_chunk0_corr_n: 0,
            packed_knn_chunk0_corr_sum_edge: 0.0,
            packed_knn_chunk0_corr_sum_edge2: 0.0,
            packed_knn_chunk0_corr_sum_chunk0: 0.0,
            packed_knn_chunk0_corr_sum_chunk02: 0.0,
            packed_knn_chunk0_corr_sum_chunk0_edge: 0.0,
            packed_knn_chunk0_corr_sum_excess: 0.0,
            packed_knn_chunk0_corr_sum_excess2: 0.0,
            packed_knn_chunk0_corr_sum_excess_edge: 0.0,
        }
    }
}

#[cfg(feature = "timing")]
impl CellSubAccum {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_knn(&mut self, d: Duration) {
        self.knn_query += d;
    }

    pub fn add_packed_knn(&mut self, d: Duration) {
        self.packed_knn += d;
    }

    pub fn add_packed_knn_setup(&mut self, d: Duration) {
        self.packed_knn_setup += d;
    }

    pub fn add_packed_knn_query_cache(&mut self, d: Duration) {
        self.packed_knn_query_cache += d;
    }

    pub fn add_packed_knn_security_thresholds(&mut self, d: Duration) {
        self.packed_knn_security_thresholds += d;
    }

    pub fn add_packed_security_queries(&mut self, planes: u64, caps: u64) {
        self.packed_security_planes_queries += planes;
        self.packed_security_cap_queries += caps;
    }

    pub fn add_packed_tail_possible_queries(
        &mut self,
        total_packed_queries: u64,
        tail_possible: u64,
    ) {
        self.packed_queries += total_packed_queries;
        self.packed_tail_possible_queries += tail_possible;
    }

    pub fn add_packed_groups(&mut self, groups: u64) {
        self.packed_groups += groups;
    }

    pub fn add_packed_tail_build_groups(&mut self, groups: u64) {
        self.packed_tail_build_groups += groups;
    }

    pub fn add_packed_knn_center_pass(&mut self, d: Duration) {
        self.packed_knn_center_pass += d;
    }

    pub fn add_packed_knn_ring_thresholds(&mut self, d: Duration) {
        self.packed_knn_ring_thresholds += d;
    }

    pub fn add_packed_knn_ring_pass(&mut self, d: Duration) {
        self.packed_knn_ring_pass += d;
    }

    pub fn add_packed_knn_ring_fallback(&mut self, d: Duration) {
        self.packed_knn_ring_fallback += d;
    }

    pub fn add_packed_knn_select_prep(&mut self, d: Duration) {
        self.packed_knn_select_prep += d;
    }

    pub fn add_packed_knn_select_query_prep(&mut self, d: Duration) {
        self.packed_knn_select_query_prep += d;
    }

    pub fn add_packed_knn_select_partition(&mut self, d: Duration) {
        self.packed_knn_select_partition += d;
    }

    pub fn add_packed_knn_select_sort(&mut self, d: Duration) {
        self.packed_knn_select_sort += d;
    }

    pub fn add_packed_knn_sort_len_hist(
        &mut self,
        counts: &[u64; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
        nanos: &[u64; crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS],
    ) {
        for i in 0..crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS {
            self.packed_knn_sort_len_counts[i] += counts[i];
            self.packed_knn_sort_len_nanos[i] += nanos[i];
        }
    }

    pub fn add_packed_knn_sort_len_exact_hist(
        &mut self,
        counts: &[u64; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
        nanos: &[u64; crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2],
    ) {
        for i in 0..crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2 {
            self.packed_knn_sort_len_exact_counts[i] += counts[i];
            self.packed_knn_sort_len_exact_nanos[i] += nanos[i];
        }
    }

    pub fn add_packed_knn_select_scatter(&mut self, d: Duration) {
        self.packed_knn_select_scatter += d;
    }

    #[inline]
    pub fn add_packed_knn_chunk0_overfetch_sample(
        &mut self,
        incoming_edgechecks: usize,
        k_needed: usize,
        chunk0_len: usize,
    ) {
        let edge_bucket = incoming_edgechecks.min(EDGECHECK_BUCKETS - 1);
        self.packed_knn_chunk0_by_edgechecks_counts[edge_bucket] += 1;
        self.packed_knn_chunk0_by_edgechecks_sum_chunk0[edge_bucket] += chunk0_len as u64;
        self.packed_knn_chunk0_by_edgechecks_sum_needed[edge_bucket] += k_needed as u64;
        let excess = chunk0_len.saturating_sub(k_needed);
        self.packed_knn_chunk0_by_edgechecks_sum_excess[edge_bucket] += excess as u64;

        self.packed_knn_chunk0_corr_n += 1;
        let e = incoming_edgechecks as f64;
        let x = chunk0_len as f64;
        let ex = excess as f64;
        self.packed_knn_chunk0_corr_sum_edge += e;
        self.packed_knn_chunk0_corr_sum_edge2 += e * e;
        self.packed_knn_chunk0_corr_sum_chunk0 += x;
        self.packed_knn_chunk0_corr_sum_chunk02 += x * x;
        self.packed_knn_chunk0_corr_sum_chunk0_edge += x * e;
        self.packed_knn_chunk0_corr_sum_excess += ex;
        self.packed_knn_chunk0_corr_sum_excess2 += ex * ex;
        self.packed_knn_chunk0_corr_sum_excess_edge += ex * e;
    }

    #[allow(dead_code)]
    pub fn add_packed_knn_unaccounted(&mut self, d: Duration) {
        self.packed_knn_unaccounted += d;
    }

    pub fn add_clip(&mut self, d: Duration) {
        self.clipping += d;
    }

    pub fn add_cert(&mut self, d: Duration) {
        self.certification += d;
    }

    pub fn add_key_dedup(&mut self, d: Duration) {
        self.key_dedup += d;
    }

    pub fn add_edge_collect(&mut self, d: Duration) {
        self.edge_collect += d;
        self.edge_checks += d;
    }

    pub fn add_edge_resolve(&mut self, d: Duration) {
        self.edge_resolve += d;
        self.edge_checks += d;
    }

    pub fn add_edge_emit(&mut self, d: Duration) {
        self.edge_emit += d;
        self.edge_checks += d;
    }

    pub fn add_cell_stage(
        &mut self,
        stage: KnnCellStage,
        knn_exhausted: bool,
        neighbors_processed: usize,
        packed_tail_used: bool,
        packed_safe_exhausted: bool,
        used_knn: bool,
        incoming_edgechecks: usize,
        edgecheck_seed_clips: usize,
    ) {
        self.stage_counts.add(stage);
        if knn_exhausted {
            self.cells_knn_exhausted += 1;
        }
        if packed_tail_used {
            self.cells_packed_tail_used += 1;
        }
        if packed_safe_exhausted {
            self.cells_packed_safe_exhausted += 1;
        }
        if used_knn {
            self.cells_used_knn += 1;
        }
        // Record histogram bucket
        let bucket = if neighbors_processed <= 48 {
            neighbors_processed.saturating_sub(1) // 1->0, 2->1, ..., 48->47
        } else if neighbors_processed <= 64 {
            48
        } else if neighbors_processed <= 96 {
            49
        } else {
            50
        };
        self.neighbors_histogram[bucket] += 1;

        // Extra-neighbor histogram by incoming edgechecks.
        let edge_bucket = incoming_edgechecks.min(EDGECHECK_BUCKETS - 1);
        self.edgecheck_bucket_counts[edge_bucket] += 1;
        self.edgecheck_seed_used_sum[edge_bucket] += edgecheck_seed_clips as u64;
        let extra = neighbors_processed.saturating_sub(edgecheck_seed_clips);
        self.edgecheck_extra_sum[edge_bucket] += extra as u64;
        let extra_bucket = if extra == 0 {
            0
        } else if extra <= 48 {
            extra
        } else if extra <= 64 {
            49
        } else if extra <= 96 {
            50
        } else {
            51
        };
        self.edgecheck_extra_histogram[edge_bucket][extra_bucket] += 1;
    }

    pub fn merge(&mut self, other: &CellSubAccum) {
        self.knn_query += other.knn_query;
        self.packed_knn += other.packed_knn;
        self.packed_knn_setup += other.packed_knn_setup;
        self.packed_knn_query_cache += other.packed_knn_query_cache;
        self.packed_knn_security_thresholds += other.packed_knn_security_thresholds;
        self.packed_knn_center_pass += other.packed_knn_center_pass;
        self.packed_knn_ring_thresholds += other.packed_knn_ring_thresholds;
        self.packed_knn_ring_pass += other.packed_knn_ring_pass;
        self.packed_knn_ring_fallback += other.packed_knn_ring_fallback;
        self.packed_knn_select_prep += other.packed_knn_select_prep;
        self.packed_knn_select_query_prep += other.packed_knn_select_query_prep;
        self.packed_knn_select_partition += other.packed_knn_select_partition;
        self.packed_knn_select_sort += other.packed_knn_select_sort;
        self.packed_knn_select_scatter += other.packed_knn_select_scatter;
        self.packed_knn_unaccounted += other.packed_knn_unaccounted;
        for i in 0..crate::cube_grid::packed_knn::PACKED_SORT_HIST_BINS {
            self.packed_knn_sort_len_counts[i] += other.packed_knn_sort_len_counts[i];
            self.packed_knn_sort_len_nanos[i] += other.packed_knn_sort_len_nanos[i];
        }
        for i in 0..crate::cube_grid::packed_knn::PACKED_SORT_LEN_MAX + 2 {
            self.packed_knn_sort_len_exact_counts[i] += other.packed_knn_sort_len_exact_counts[i];
            self.packed_knn_sort_len_exact_nanos[i] += other.packed_knn_sort_len_exact_nanos[i];
        }
        self.clipping += other.clipping;
        self.certification += other.certification;
        self.key_dedup += other.key_dedup;
        self.edge_checks += other.edge_checks;
        self.edge_collect += other.edge_collect;
        self.edge_resolve += other.edge_resolve;
        self.edge_emit += other.edge_emit;
        self.stage_counts.merge(&other.stage_counts);
        self.cells_knn_exhausted += other.cells_knn_exhausted;
        self.cells_packed_tail_used += other.cells_packed_tail_used;
        self.cells_packed_safe_exhausted += other.cells_packed_safe_exhausted;
        self.cells_used_knn += other.cells_used_knn;
        self.packed_security_planes_queries += other.packed_security_planes_queries;
        self.packed_security_cap_queries += other.packed_security_cap_queries;
        self.packed_queries += other.packed_queries;
        self.packed_tail_possible_queries += other.packed_tail_possible_queries;
        self.packed_groups += other.packed_groups;
        self.packed_tail_build_groups += other.packed_tail_build_groups;
        for (i, &count) in other.neighbors_histogram.iter().enumerate() {
            self.neighbors_histogram[i] += count;
        }
        for e in 0..EDGECHECK_BUCKETS {
            self.edgecheck_bucket_counts[e] += other.edgecheck_bucket_counts[e];
            self.edgecheck_seed_used_sum[e] += other.edgecheck_seed_used_sum[e];
            self.edgecheck_extra_sum[e] += other.edgecheck_extra_sum[e];
            self.packed_knn_chunk0_by_edgechecks_counts[e] +=
                other.packed_knn_chunk0_by_edgechecks_counts[e];
            self.packed_knn_chunk0_by_edgechecks_sum_chunk0[e] +=
                other.packed_knn_chunk0_by_edgechecks_sum_chunk0[e];
            self.packed_knn_chunk0_by_edgechecks_sum_needed[e] +=
                other.packed_knn_chunk0_by_edgechecks_sum_needed[e];
            self.packed_knn_chunk0_by_edgechecks_sum_excess[e] +=
                other.packed_knn_chunk0_by_edgechecks_sum_excess[e];
            for b in 0..EXTRA_NEIGHBOR_HIST_BUCKETS {
                self.edgecheck_extra_histogram[e][b] += other.edgecheck_extra_histogram[e][b];
            }
        }
        self.packed_knn_chunk0_corr_n += other.packed_knn_chunk0_corr_n;
        self.packed_knn_chunk0_corr_sum_edge += other.packed_knn_chunk0_corr_sum_edge;
        self.packed_knn_chunk0_corr_sum_edge2 += other.packed_knn_chunk0_corr_sum_edge2;
        self.packed_knn_chunk0_corr_sum_chunk0 += other.packed_knn_chunk0_corr_sum_chunk0;
        self.packed_knn_chunk0_corr_sum_chunk02 += other.packed_knn_chunk0_corr_sum_chunk02;
        self.packed_knn_chunk0_corr_sum_chunk0_edge += other.packed_knn_chunk0_corr_sum_chunk0_edge;
        self.packed_knn_chunk0_corr_sum_excess += other.packed_knn_chunk0_corr_sum_excess;
        self.packed_knn_chunk0_corr_sum_excess2 += other.packed_knn_chunk0_corr_sum_excess2;
        self.packed_knn_chunk0_corr_sum_excess_edge += other.packed_knn_chunk0_corr_sum_excess_edge;
    }

    pub fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases {
            knn_query: self.knn_query,
            packed_knn: self.packed_knn,
            packed_knn_setup: self.packed_knn_setup,
            packed_knn_query_cache: self.packed_knn_query_cache,
            packed_knn_security_thresholds: self.packed_knn_security_thresholds,
            packed_knn_center_pass: self.packed_knn_center_pass,
            packed_knn_ring_thresholds: self.packed_knn_ring_thresholds,
            packed_knn_ring_pass: self.packed_knn_ring_pass,
            packed_knn_ring_fallback: self.packed_knn_ring_fallback,
            packed_knn_select_prep: self.packed_knn_select_prep,
            packed_knn_select_query_prep: self.packed_knn_select_query_prep,
            packed_knn_select_partition: self.packed_knn_select_partition,
            packed_knn_select_sort: self.packed_knn_select_sort,
            packed_knn_select_scatter: self.packed_knn_select_scatter,
            packed_knn_unaccounted: self.packed_knn_unaccounted,
            packed_knn_sort_len_counts: self.packed_knn_sort_len_counts,
            packed_knn_sort_len_nanos: self.packed_knn_sort_len_nanos,
            packed_knn_sort_len_exact_counts: self.packed_knn_sort_len_exact_counts,
            packed_knn_sort_len_exact_nanos: self.packed_knn_sort_len_exact_nanos,
            clipping: self.clipping,
            certification: self.certification,
            key_dedup: self.key_dedup,
            edge_checks: self.edge_checks,
            edge_collect: self.edge_collect,
            edge_resolve: self.edge_resolve,
            edge_emit: self.edge_emit,
            stage_counts: self.stage_counts,
            cells_knn_exhausted: self.cells_knn_exhausted,
            cells_packed_tail_used: self.cells_packed_tail_used,
            cells_packed_safe_exhausted: self.cells_packed_safe_exhausted,
            cells_used_knn: self.cells_used_knn,
            packed_security_planes_queries: self.packed_security_planes_queries,
            packed_security_cap_queries: self.packed_security_cap_queries,
            packed_queries: self.packed_queries,
            packed_tail_possible_queries: self.packed_tail_possible_queries,
            packed_groups: self.packed_groups,
            packed_tail_build_groups: self.packed_tail_build_groups,
            neighbors_histogram: self.neighbors_histogram,
            edgecheck_extra_histogram: self.edgecheck_extra_histogram,
            edgecheck_bucket_counts: self.edgecheck_bucket_counts,
            edgecheck_seed_used_sum: self.edgecheck_seed_used_sum,
            edgecheck_extra_sum: self.edgecheck_extra_sum,
            packed_knn_chunk0_by_edgechecks_counts: self.packed_knn_chunk0_by_edgechecks_counts,
            packed_knn_chunk0_by_edgechecks_sum_chunk0: self.packed_knn_chunk0_by_edgechecks_sum_chunk0,
            packed_knn_chunk0_by_edgechecks_sum_needed: self.packed_knn_chunk0_by_edgechecks_sum_needed,
            packed_knn_chunk0_by_edgechecks_sum_excess: self.packed_knn_chunk0_by_edgechecks_sum_excess,
            packed_knn_chunk0_corr_n: self.packed_knn_chunk0_corr_n,
            packed_knn_chunk0_corr_sum_edge: self.packed_knn_chunk0_corr_sum_edge,
            packed_knn_chunk0_corr_sum_edge2: self.packed_knn_chunk0_corr_sum_edge2,
            packed_knn_chunk0_corr_sum_chunk0: self.packed_knn_chunk0_corr_sum_chunk0,
            packed_knn_chunk0_corr_sum_chunk02: self.packed_knn_chunk0_corr_sum_chunk02,
            packed_knn_chunk0_corr_sum_chunk0_edge: self.packed_knn_chunk0_corr_sum_chunk0_edge,
            packed_knn_chunk0_corr_sum_excess: self.packed_knn_chunk0_corr_sum_excess,
            packed_knn_chunk0_corr_sum_excess2: self.packed_knn_chunk0_corr_sum_excess2,
            packed_knn_chunk0_corr_sum_excess_edge: self.packed_knn_chunk0_corr_sum_excess_edge,
        }
    }
}

/// Dummy accumulator when feature is disabled.
#[cfg(not(feature = "timing"))]
#[derive(Default, Clone, Copy)]
pub struct CellSubAccum;

#[cfg(not(feature = "timing"))]
impl CellSubAccum {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }
    #[inline(always)]
    pub fn add_knn(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_packed_knn(&mut self, _d: Duration) {}
    #[allow(dead_code)]
    #[inline(always)]
    pub fn add_packed_security_queries(&mut self, _planes: u64, _caps: u64) {}
    #[inline(always)]
    pub fn add_packed_tail_possible_queries(
        &mut self,
        _total_packed_queries: u64,
        _tail_possible: u64,
    ) {
    }
    #[inline(always)]
    pub fn add_packed_groups(&mut self, _groups: u64) {}
    #[inline(always)]
    pub fn add_packed_tail_build_groups(&mut self, _groups: u64) {}
    #[inline(always)]
    pub fn add_clip(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_cert(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_key_dedup(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_edge_collect(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_edge_resolve(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_edge_emit(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_cell_stage(
        &mut self,
        _stage: KnnCellStage,
        _knn_exhausted: bool,
        _neighbors_processed: usize,
        _packed_tail_used: bool,
        _packed_safe_exhausted: bool,
        _used_knn: bool,
        _incoming_edgechecks: usize,
        _edgecheck_seed_clips: usize,
    ) {
    }
    #[inline(always)]
    pub fn merge(&mut self, _other: &CellSubAccum) {}
    #[inline(always)]
    pub fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases
    }
}

#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnCellStage {
    PackedChunk0,
    PackedTail,
    /// Terminated during resume stage with given K value
    Resume(usize),
    /// Terminated during restart stage with given K value
    Restart(usize),
    /// Ran full scan as fallback
    FullScanFallback,
}

/// Builder for collecting phase timings.
#[cfg(feature = "timing")]
pub struct TimingBuilder {
    t_start: std::time::Instant,
    preprocess: Duration,
    knn_build: Duration,
    knn_build_sub: Option<CubeMapGridBuildTimings>,
    cell_construction: Duration,
    cell_sub: CellSubPhases,
    dedup: Duration,
    dedup_sub: DedupSubPhases,
    edge_repair: Duration,
    assemble: Duration,
}

#[cfg(feature = "timing")]
impl TimingBuilder {
    pub fn new() -> Self {
        Self {
            t_start: std::time::Instant::now(),
            preprocess: Duration::ZERO,
            knn_build: Duration::ZERO,
            knn_build_sub: None,
            cell_construction: Duration::ZERO,
            cell_sub: CellSubPhases::default(),
            dedup: Duration::ZERO,
            dedup_sub: DedupSubPhases::default(),
            edge_repair: Duration::ZERO,
            assemble: Duration::ZERO,
        }
    }

    pub fn set_preprocess(&mut self, d: Duration) {
        self.preprocess = d;
    }

    pub fn set_knn_build(&mut self, d: Duration) {
        self.knn_build = d;
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

    pub fn set_edge_repair(&mut self, d: Duration) {
        self.edge_repair = d;
    }

    pub fn set_assemble(&mut self, d: Duration) {
        self.assemble = d;
    }

    pub fn finish(self) -> PhaseTimings {
        PhaseTimings {
            total: self.t_start.elapsed(),
            preprocess: self.preprocess,
            knn_build: self.knn_build,
            knn_build_sub: self.knn_build_sub,
            cell_construction: self.cell_construction,
            cell_sub: self.cell_sub,
            dedup: self.dedup,
            dedup_sub: self.dedup_sub,
            edge_repair: self.edge_repair,
            assemble: self.assemble,
        }
    }
}

/// Dummy builder when feature is disabled.
#[cfg(not(feature = "timing"))]
pub struct TimingBuilder;

#[cfg(not(feature = "timing"))]
impl TimingBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub fn set_knn_build(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn set_preprocess(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn set_cell_construction(&mut self, _d: Duration, _sub: CellSubPhases) {}

    #[inline(always)]
    pub fn set_dedup(&mut self, _d: Duration, _sub: DedupSubPhases) {}
    #[inline(always)]
    pub fn set_edge_repair(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn set_assemble(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn finish(self) -> PhaseTimings {
        PhaseTimings
    }
}
