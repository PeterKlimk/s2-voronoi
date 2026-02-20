//! Spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This structure is friendly to data-parallel CPU
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

// Keep the kNN schedule and termination knobs in one place.
//
// Tuning notes:
// - The "resumable kNN" schedule (`KNN_RESUME_K`, `KNN_RESTART_*`) is used for exact kNN queries.
// - The packed schedule (`PACKED_*`) is used only for the fast r=1 packed-kNN path.
// - If termination still isn't proven after `KNN_RESTART_MAX`, we grow k using
//   `TERMINATION_GROW_*` until the (optional) cap.
pub(super) const KNN_RESUME_K: usize = 18;
pub(super) const KNN_RESTART_K0: usize = 24;
pub(super) const KNN_RESTART_MAX: usize = 48;
pub(super) const KNN_RESTART_KS: [usize; 2] = [KNN_RESTART_K0, KNN_RESTART_MAX];

/// Packed-kNN initial `Chunk0` size (r=1).
///
/// This is intentionally separate from the resumable kNN schedule; it only affects the packed
/// path. Defaults to the same value as `KNN_RESTART_K0`.
pub(super) const PACKED_K0: usize = 16;

/// Packed-kNN chunk size after `Chunk0` (and for tail emission).
///
/// Smaller reduces upfront packed work but may increase loop iterations and/or fallback to kNN.
pub(super) const PACKED_K1: usize = 8;

/// How aggressively to grow k when we have a bounded-but-unproven cell after the scheduled kNN.
pub(super) const TERMINATION_GROW_MULTIPLIER: usize = 2;
pub(super) const TERMINATION_GROW_MIN_STEP: usize = 32;

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
