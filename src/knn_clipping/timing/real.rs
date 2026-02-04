use super::KnnCellStage;
use std::time::Duration;

use crate::cube_grid::CubeMapGridBuildTimings;

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
    full_scan: u64,
    resume_default: u64,
    resume_other: u64,
    restart_k0: u64,
    restart_kmax: u64,
    restart_other: u64,
}

impl StageCounts {
    #[inline]
    fn add(&mut self, stage: KnnCellStage) {
        match stage {
            KnnCellStage::PackedChunk0 => self.packed_chunk0 += 1,
            KnnCellStage::PackedTail => self.packed_tail += 1,
            KnnCellStage::FullScanFallback => self.full_scan += 1,
            KnnCellStage::Resume(k) => {
                if k == super::super::KNN_RESUME_K {
                    self.resume_default += 1;
                } else {
                    self.resume_other += 1;
                }
            }
            KnnCellStage::Restart(k) => {
                if k == super::super::KNN_RESTART_K0 {
                    self.restart_k0 += 1;
                } else if k == super::super::KNN_RESTART_MAX {
                    self.restart_kmax += 1;
                } else {
                    self.restart_other += 1;
                }
            }
        }
    }

    #[inline]
    fn merge(&mut self, other: &StageCounts) {
        self.packed_chunk0 += other.packed_chunk0;
        self.packed_tail += other.packed_tail;
        self.full_scan += other.full_scan;
        self.resume_default += other.resume_default;
        self.resume_other += other.resume_other;
        self.restart_k0 += other.restart_k0;
        self.restart_kmax += other.restart_kmax;
        self.restart_other += other.restart_other;
    }
}

/// Per-cell-group timing totals aggregated across all shards.
#[derive(Debug, Clone, Default)]
pub struct CellSubPhases {
    pub knn_query: Duration,
    pub packed_knn: Duration,
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
}

/// Fine-grained dedup timing and a few size counters.
#[derive(Debug, Clone, Default)]
pub struct DedupSubPhases {
    pub triplet_keys: u64,
    pub support_keys: u64,
    pub bad_edges_count: u64,
}

/// Accumulator for cell sub-phase timings (used per-bin, then merged).
#[derive(Clone, Default)]
pub struct CellSubAccum {
    knn_query: Duration,
    packed_knn: Duration,
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
        _neighbors_processed: usize,
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
    }

    #[inline]
    pub fn merge(&mut self, other: &CellSubAccum) {
        self.knn_query += other.knn_query;
        self.packed_knn += other.packed_knn;
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
    }

    #[inline]
    pub fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases {
            knn_query: self.knn_query,
            packed_knn: self.packed_knn,
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
    pub edge_repair: Duration,
    pub assemble: Duration,
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
            eprintln!("  preprocess:        {:7.1}ms ({:4.1}%)", ms(self.preprocess), pct(self.preprocess));
        }
        eprintln!("  knn_build:         {:7.1}ms ({:4.1}%)", ms(self.knn_build), pct(self.knn_build));
        if let Some(sub) = &self.knn_build_sub {
            if sub.total().as_nanos() > 0 && self.knn_build.as_nanos() > 0 {
                let sub_pct = |d: Duration| d.as_secs_f64() / self.knn_build.as_secs_f64() * 100.0;
                eprintln!("    grid_count:      {:7.1}ms ({:4.1}%)", ms(sub.count_cells), sub_pct(sub.count_cells));
                eprintln!("    grid_prefix:     {:7.1}ms ({:4.1}%)", ms(sub.prefix_sum), sub_pct(sub.prefix_sum));
                eprintln!("    grid_scatter:    {:7.1}ms ({:4.1}%)", ms(sub.scatter_soa), sub_pct(sub.scatter_soa));
                eprintln!("    grid_neighbors:  {:7.1}ms ({:4.1}%)", ms(sub.neighbors), sub_pct(sub.neighbors));
                eprintln!("    grid_ring2:      {:7.1}ms ({:4.1}%)", ms(sub.ring2), sub_pct(sub.ring2));
                eprintln!("    grid_bounds:     {:7.1}ms ({:4.1}%)", ms(sub.cell_bounds), sub_pct(sub.cell_bounds));
                eprintln!("    grid_security:   {:7.1}ms ({:4.1}%)", ms(sub.security_3x3), sub_pct(sub.security_3x3));
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
        let cpu_to_wall = if cpu_total_secs > 0.0 { wall_secs / cpu_total_secs } else { 1.0 };
        let sub_pct = |d: Duration| if cpu_total_secs > 0.0 { d.as_secs_f64() / cpu_total_secs * 100.0 } else { 0.0 };
        let est_wall_ms = |d: Duration| d.as_secs_f64() * cpu_to_wall * 1000.0;

        if cpu_total.as_nanos() > 0 {
            eprintln!("    knn_query:       {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.knn_query), sub_pct(self.cell_sub.knn_query));
            if self.cell_sub.packed_knn.as_nanos() > 0 {
                eprintln!("    packed_knn:      {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.packed_knn), sub_pct(self.cell_sub.packed_knn));
            }
            eprintln!("    clipping:        {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.clipping), sub_pct(self.cell_sub.clipping));
            eprintln!("    certification:   {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.certification), sub_pct(self.cell_sub.certification));
            eprintln!("    key_dedup:       {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.key_dedup), sub_pct(self.cell_sub.key_dedup));
            eprintln!("    edge_collect:    {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.edge_collect), sub_pct(self.cell_sub.edge_collect));
            eprintln!("    edge_resolve:    {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.edge_resolve), sub_pct(self.cell_sub.edge_resolve));
            eprintln!("    edge_emit:       {:7.1}ms ({:4.1}%)", est_wall_ms(self.cell_sub.edge_emit), sub_pct(self.cell_sub.edge_emit));
            eprintln!(
                "    cells: used_knn={} knn_exhausted={} packed_tail_used={} packed_safe_exhausted={}",
                self.cell_sub.cells_used_knn,
                self.cell_sub.cells_knn_exhausted,
                self.cell_sub.cells_packed_tail_used,
                self.cell_sub.cells_packed_safe_exhausted
            );
        }

        eprintln!(
            "  dedup:             {:7.1}ms ({:4.1}%)",
            ms(self.dedup),
            pct(self.dedup)
        );
        eprintln!(
            "    keys: triplet={} support={} bad_edges={}",
            self.dedup_sub.triplet_keys, self.dedup_sub.support_keys, self.dedup_sub.bad_edges_count
        );
        eprintln!(
            "  edge_repair:       {:7.1}ms ({:4.1}%)",
            ms(self.edge_repair),
            pct(self.edge_repair)
        );
        eprintln!(
            "  assemble:          {:7.1}ms ({:4.1}%)",
            ms(self.assemble),
            pct(self.assemble)
        );

        if std::env::var_os("S2_VORONOI_TIMING_KV").is_some() {
            eprintln!(
                "TIMING_KV n={n} total_ms={total:.3} preprocess_ms={pre:.3} knn_build_ms={kb:.3} cell_construction_ms={cc:.3} dedup_ms={dd:.3} edge_repair_ms={er:.3} assemble_ms={asmb:.3}",
                n = n,
                total = total_ms,
                pre = ms(self.preprocess),
                kb = ms(self.knn_build),
                cc = ms(self.cell_construction),
                dd = ms(self.dedup),
                er = ms(self.edge_repair),
                asmb = ms(self.assemble),
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
    edge_repair: Duration,
    assemble: Duration,
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
