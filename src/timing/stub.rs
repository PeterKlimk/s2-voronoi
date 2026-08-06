use std::time::Duration;

pub(crate) struct Timer;
impl Timer {
    #[inline]
    pub(crate) fn start() -> Self {
        Self
    }
    #[inline]
    pub(crate) fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PhaseTimings;
impl PhaseTimings {
    pub(crate) fn report(&self, _n: usize) {}
}

pub(crate) struct TimingBuilder;
impl TimingBuilder {
    pub(crate) fn new() -> Self {
        Self
    }
    pub(crate) fn set_input_validation(&mut self, _d: Duration) {}
    pub(crate) fn set_preprocess(&mut self, _d: Duration) {}
    pub(crate) fn add_grid_build(&mut self, _d: Duration) {}
    pub(crate) fn set_cell_construction(&mut self, _d: Duration) {}
    pub(crate) fn set_shard_assembly(&mut self, _d: Duration) {}
    pub(crate) fn set_edge_reconcile(&mut self, _d: Duration) {}
    pub(crate) fn set_postprocess(&mut self, _d: Duration) {}
    pub(crate) fn set_output_remap(&mut self, _d: Duration) {}
    pub(crate) fn set_output_validation(&mut self, _d: Duration) {}
    pub(crate) fn finish(self) -> PhaseTimings {
        PhaseTimings
    }
}
