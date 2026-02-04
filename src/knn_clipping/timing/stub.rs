use super::KnnCellStage;
use std::time::Duration;

/// Dummy timer when `timing` is disabled (zero-sized).
pub struct Timer;

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

/// Dummy lap timer when `timing` is disabled (zero-sized).
pub struct LapTimer;

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

/// Dummy cell sub-phases when `timing` is disabled (zero-sized).
#[derive(Debug, Clone, Copy, Default)]
pub struct CellSubPhases;

/// Dummy dedup sub-phases when `timing` is disabled (zero-sized).
#[derive(Debug, Clone, Copy, Default)]
pub struct DedupSubPhases;

/// Dummy accumulator when `timing` is disabled (zero-sized).
#[derive(Clone, Copy, Default)]
pub struct CellSubAccum;

impl CellSubAccum {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }
    #[inline(always)]
    pub fn add_knn(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_packed_knn(&mut self, _d: Duration) {}
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

/// Dummy timings when `timing` is disabled (zero-sized).
#[derive(Debug, Clone, Copy)]
pub struct PhaseTimings;

impl PhaseTimings {
    #[inline(always)]
    pub fn report(&self, _n: usize) {}
}

/// Dummy builder when `timing` is disabled.
pub struct TimingBuilder;

impl TimingBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub fn set_preprocess(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn set_knn_build(&mut self, _d: Duration) {}

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

