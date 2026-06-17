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
    pub packed_tail_builds: u64,
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
    packed_tail_builds: u64,
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
        self.packed_tail_builds += other.packed_tail_builds;
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
            packed_tail_builds: self.packed_tail_builds,
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

        if std::env::var_os("S2_VORONOI_TIMING_KV").is_some() {
            eprintln!(
                "TIMING_KV n={n} total_ms={total:.3} preprocess_ms={pre:.3} knn_build_ms={kb:.3} cell_construction_ms={cc:.3} dedup_ms={dd:.3} edge_reconcile_ms={er:.3} edge_repair_ms={er:.3} assemble_ms={asmb:.3} cells_used_knn={cuk} cells_packed_tail_used={cpt} packed_tail_builds={ptb} neighbors_total={nt} neighbors_max={nm} final_edges_total={fet} final_edges_max={fem} examine_per_edge={epe:.6} dir_shadow_checks={dsc} dir_shadow_candidate_tests={dst} dir_shadow_hits={dsh} dir_shadow_saved={dss} dir_support_candidate_tests={dpt} dir_support_hits={dph} dir_support_saved={dps} dir_support_false_positive_hits={dpf} grid_res={gr} grid_max_occ={gmo} grid_rebuilt={grb}",
                n = n,
                total = total_ms,
                pre = ms(self.preprocess),
                kb = ms(self.knn_build),
                cc = ms(self.cell_construction),
                dd = ms(self.dedup),
                er = ms(self.edge_reconcile),
                asmb = ms(self.assemble),
                cuk = self.cell_sub.cells_used_knn,
                cpt = self.cell_sub.cells_packed_tail_used,
                ptb = self.cell_sub.packed_tail_builds,
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
}

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
            edge_reconcile: Duration::ZERO,
            assemble: Duration::ZERO,
            grid_res: 0,
            grid_max_occupancy: 0,
            grid_rebuilt: false,
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

    pub fn set_edge_reconcile(&mut self, d: Duration) {
        self.edge_reconcile = d;
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
            edge_reconcile: self.edge_reconcile,
            assemble: self.assemble,
            grid_res: self.grid_res,
            grid_max_occupancy: self.grid_max_occupancy,
            grid_rebuilt: self.grid_rebuilt,
        }
    }
}
