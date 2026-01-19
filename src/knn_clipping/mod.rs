//! GPU-friendly spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This enables massive parallelism on GPU.

macro_rules! maybe_par_into_iter {
    ($v:expr) => {{
        #[cfg(feature = "parallel")]
        {
            $v.into_par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            $v.into_iter()
        }
    }};
}

mod cell_builder;
mod compute;
mod constants;
mod edge_repair;
mod live_dedup;
mod preprocess;
mod timing;
mod topo2d;

// Re-exports (internal use)
#[allow(unused_imports)]
pub use compute::compute_voronoi_gpu_style_with_config;
pub use preprocess::merge_close_points;

pub type MergeResult = preprocess::MergeResult;

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    /// Enables adaptive k-NN + early termination checks.
    ///
    /// When disabled, the builder will still run the initial k-NN pass but will not
    /// attempt to terminate early; the k-NN schedule still runs to ensure
    /// correctness (so this should generally remain enabled for performance).
    pub check_start: usize,
    pub check_step: usize,
    /// Optional cap on k if termination keeps requesting more neighbors.
    /// None means unbounded.
    pub max_k_cap: Option<usize>,
}

// Keep the k-NN schedule and the default termination cadence in one place.
pub(super) const KNN_RESUME_K: usize = 18;
pub(super) const KNN_RESTART_MAX: usize = 48;
pub(super) const KNN_RESTART_KS: [usize; 2] = [24, KNN_RESTART_MAX];

/// Target points per cell for the cube-map KNN grid.
/// Lower = more cells, faster scans, more heap overhead.
/// Higher = fewer cells, longer scans, less overhead.
pub(super) const KNN_GRID_TARGET_DENSITY: f64 = 16.0;

// Default termination cadence:
// - start near the end of the initial k pass
// - then check roughly twice per initial-k window
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
