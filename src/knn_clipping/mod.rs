//! Spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from nearby neighbors. This structure is friendly to data-parallel CPU
//! implementations.

pub(crate) mod cell_builder;
pub(crate) mod compute;
pub(crate) mod constants;
pub(crate) mod edge_repair;
pub(crate) mod live_dedup;
pub(crate) mod preprocess;
pub(crate) mod timing;
pub(crate) mod topo2d;
pub(crate) mod union_find;

// Re-exports (internal use)
#[allow(unused_imports)]
pub use compute::compute_voronoi_knn_clipping_with_config_owned;
pub use preprocess::merge_close_points;

pub type MergeResult = preprocess::MergeResult;

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    /// Enables adaptive early termination checks.
    pub check_start: usize,
    pub check_step: usize,
    /// Legacy compatibility field retained in the public config.
    /// The directed cursor fallback is no-K and ignores this cap.
    pub max_k_cap: Option<usize>,
}

/// Packed-kNN initial `Chunk0` size (r=1).
pub(super) const PACKED_K0: usize = 16;

/// Packed-kNN chunk size after `Chunk0` (and for tail emission).
/// Smaller reduces upfront packed work but may increase fallback iterations.
pub(super) const PACKED_K1: usize = 8;

/// Target points per cell for the cube-map KNN grid.
/// Lower = more cells, faster scans, more heap overhead.
/// Higher = fewer cells, longer scans, less overhead.
pub(super) const KNN_GRID_TARGET_DENSITY: f64 = 16.0;

// Default termination cadence.
const DEFAULT_TERMINATION_CHECK_START: usize = 8;
const DEFAULT_TERMINATION_CHECK_STEP: usize = 1;

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            check_start: DEFAULT_TERMINATION_CHECK_START,
            check_step: DEFAULT_TERMINATION_CHECK_STEP,
            max_k_cap: None,
        }
    }
}

impl TerminationConfig {
    #[inline]
    pub fn should_check(&self, neighbors_processed: usize) -> bool {
        self.check_step > 0
            && neighbors_processed >= self.check_start
            && (neighbors_processed - self.check_start).is_multiple_of(self.check_step)
    }
}

// NOTE: benchmark_voronoi function was removed during crate extraction.
// It compared knn_clipping vs qhull backends and belongs in hex3's benchmarks.
