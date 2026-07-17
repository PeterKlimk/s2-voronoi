use super::KnnCellStage;
use std::time::Duration;

use crate::cube_grid::packed_knn::PackedKnnTimings;

/// Dummy timer when `timing` is disabled (zero-sized).
pub(crate) struct Timer;

impl Timer {
    #[inline(always)]
    pub(crate) fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

/// Dummy lap timer when `timing` is disabled (zero-sized).
pub(crate) struct LapTimer;

impl LapTimer {
    #[inline(always)]
    pub(crate) fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn lap(&mut self) -> Duration {
        Duration::ZERO
    }
}

/// Dummy cell sub-phases when `timing` is disabled (zero-sized).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CellSubPhases;

/// Dummy dedup sub-phases when `timing` is disabled (zero-sized).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DedupSubPhases;

/// Dummy accumulator when `timing` is disabled (zero-sized).
#[derive(Clone, Copy, Default)]
pub(crate) struct CellSubAccum;

impl CellSubAccum {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self
    }
    #[inline(always)]
    pub(crate) fn add_knn(&mut self, _d: Duration) {}
    #[inline(always)]
    pub(crate) fn add_packed_knn(&mut self, _d: Duration) {}
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn add_packed_knn_breakdown(&mut self, _timings: &PackedKnnTimings) {}
    #[inline(always)]
    pub(crate) fn add_clip(&mut self, _d: Duration) {}
    #[inline(always)]
    pub(crate) fn add_cert(&mut self, _d: Duration) {}
    #[inline(always)]
    pub(crate) fn add_key_dedup(&mut self, _d: Duration) {}
    #[inline(always)]
    pub(crate) fn add_edge_collect(&mut self, _d: Duration) {}
    #[inline(always)]
    pub(crate) fn add_edge_resolve(&mut self, _d: Duration) {}
    #[inline(always)]
    pub(crate) fn add_edge_emit(&mut self, _d: Duration) {}
    #[inline(always)]
    #[allow(clippy::too_many_arguments)] // mirrors the real timing API
    pub(crate) fn add_cell_stage(
        &mut self,
        _stage: KnnCellStage,
        _knn_exhausted: bool,
        _neighbors_processed: usize,
        _final_edges: usize,
        _packed_tail_used: bool,
        _packed_safe_exhausted: bool,
        _used_knn: bool,
        _incoming_edgechecks: usize,
        _edgecheck_seed_clips: usize,
    ) {
    }
    #[inline(always)]
    #[allow(clippy::too_many_arguments)] // mirrors the real timing API
    pub(crate) fn add_directional_shadow(
        &mut self,
        _checks: usize,
        _candidate_tests: usize,
        _hits: usize,
        _saved: usize,
        _support_candidate_tests: usize,
        _support_hits: usize,
        _support_saved: usize,
        _support_false_positive_hits: usize,
    ) {
    }
    #[inline(always)]
    pub(crate) fn add_fallbacks(
        &mut self,
        _projection: usize,
        _polygon_cap: usize,
        _all_constraints: usize,
    ) {
    }
    #[inline(always)]
    pub(crate) fn merge(&mut self, _other: &CellSubAccum) {}
    #[inline(always)]
    pub(crate) fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases
    }
}

/// Dummy timings when `timing` is disabled (zero-sized).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PhaseTimings;

impl PhaseTimings {
    #[inline(always)]
    pub(crate) fn report(&self, _n: usize) {}
}

/// Dummy builder when `timing` is disabled.
pub(crate) struct TimingBuilder;

impl TimingBuilder {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn set_preprocess(&mut self, _d: Duration) {}

    #[inline(always)]
    pub(crate) fn set_weld_pair_stats(&mut self, _len: usize, _capacity: usize) {}

    #[inline(always)]
    pub(crate) fn set_knn_build(&mut self, _d: Duration) {}

    #[inline(always)]
    pub(crate) fn add_knn_build(&mut self, _d: Duration) {}

    #[inline(always)]
    pub(crate) fn set_grid_stats(&mut self, _res: usize, _max_occupancy: u64, _rebuilt: bool) {}

    #[inline(always)]
    pub(crate) fn set_cell_construction(&mut self, _d: Duration, _sub: CellSubPhases) {}

    #[inline(always)]
    pub(crate) fn set_dedup(&mut self, _d: Duration, _sub: DedupSubPhases) {}

    #[inline(always)]
    pub(crate) fn set_edge_reconcile(
        &mut self,
        _d: Duration,
        _merge_safety_scan_cells: usize,
        _merge_safety_global_fallbacks: usize,
    ) {
    }

    #[inline(always)]
    pub(crate) fn set_assemble(&mut self, _d: Duration) {}

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_output_resolution_discovery(
        &mut self,
        _certified_hint: bool,
        _drift_fallback: bool,
        _reconcile_scan_cells: usize,
        _rebuild_scan_cells: usize,
        _hint_cells: usize,
        _hinted_candidates: usize,
        _detected_edges: usize,
    ) {
    }

    #[inline(always)]
    pub(crate) fn finish(self) -> PhaseTimings {
        PhaseTimings
    }
}
