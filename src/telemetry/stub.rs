#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CellTelemetry;
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DedupTelemetry;
#[derive(Clone, Default)]
pub(crate) struct CellTelemetryAccum;
impl CellTelemetryAccum {
    pub(crate) fn new() -> Self {
        Self
    }
    pub(crate) fn record_bin_schedule(&mut self, _bins: &[Vec<usize>]) {}
    pub(crate) fn add_packed_telemetry(
        &mut self,
        _t: &crate::cube_grid::packed_knn::PackedKnnTelemetry,
    ) {
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_cell_stage(
        &mut self,
        _stage: super::KnnCellStage,
        _exhausted: bool,
        _neighbors: usize,
        _edges: usize,
        _tail: bool,
        _safe: bool,
        _used: bool,
        _incoming: usize,
        _seed: usize,
    ) {
    }
    pub(crate) fn add_fallbacks(&mut self, _projection: usize, _polygon: usize, _all: usize) {}
    pub(crate) fn merge(&mut self, _other: &Self) {}
    pub(crate) fn into_telemetry(self) -> CellTelemetry {
        CellTelemetry
    }
}
#[derive(Default)]
pub(crate) struct TelemetryBuilder;
impl TelemetryBuilder {
    pub(crate) fn new() -> Self {
        Self
    }
    pub(crate) fn set_weld_pair_stats(&mut self, _len: usize, _cap: usize) {}
    pub(crate) fn set_grid_stats(&mut self, _res: usize, _max: u64, _rebuilt: bool) {}
    pub(crate) fn set_grid_build_stats(
        &mut self,
        _stats: &crate::cube_grid::CubeMapGridBuildTelemetry,
    ) {
    }
    pub(crate) fn set_cell(&mut self, _cell: CellTelemetry) {}
    pub(crate) fn set_dedup(&mut self, _dedup: DedupTelemetry) {}
    pub(crate) fn set_edge_reconcile(&mut self, _cells: usize, _fallbacks: usize) {}
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_output_resolution_discovery(
        &mut self,
        _drift: bool,
        _reconcile: usize,
        _rebuild: usize,
        _hints: usize,
        _candidates: usize,
        _edges: usize,
    ) {
    }
    pub(crate) fn report(&self, _n: usize) {}
}
