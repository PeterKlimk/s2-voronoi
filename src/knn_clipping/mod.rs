//! Spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from nearby neighbors. This structure is friendly to data-parallel CPU
//! implementations.

use crate::policy::{
    KnnPolicy, TerminationPolicy, DEFAULT_TERMINATION_CHECK_START, DEFAULT_TERMINATION_CHECK_STEP,
};

pub(crate) mod cell_build;
pub(crate) mod compute;
pub(crate) mod edge_reconcile;
pub(crate) mod live_dedup;
pub(crate) mod preprocess;
pub(crate) mod timing;
pub(crate) mod topo2d;
pub(crate) mod union_find;

// Re-exports (internal use)
#[allow(unused_imports)]
pub use compute::compute_voronoi_knn_clipping_with_config_owned;
#[allow(unused_imports)]
pub use compute::compute_voronoi_knn_clipping_with_report_owned;
pub use preprocess::merge_close_points;

pub type MergeResult = preprocess::MergeResult;

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    /// Enables adaptive early termination checks.
    pub check_start: usize,
    pub check_step: usize,
    /// Enable a cold packed r=2 expansion stage before directed cursor fallback.
    pub packed_expand_r2: bool,
    /// Legacy compatibility field retained in the public config.
    /// The directed cursor fallback is no-K and ignores this cap.
    pub max_k_cap: Option<usize>,
}

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            check_start: DEFAULT_TERMINATION_CHECK_START,
            check_step: DEFAULT_TERMINATION_CHECK_STEP,
            packed_expand_r2: true,
            max_k_cap: None,
        }
    }
}

impl TerminationConfig {
    #[inline]
    pub(crate) fn termination_policy(&self) -> TerminationPolicy {
        TerminationPolicy::new(self.check_start, self.check_step, self.max_k_cap)
    }

    #[inline]
    pub(crate) fn knn_policy(&self, num_points: usize) -> KnnPolicy {
        KnnPolicy::for_point_count(num_points, self.termination_policy(), self.packed_expand_r2)
    }
}

// NOTE: benchmark_voronoi function was removed during crate extraction.
// It compared knn_clipping vs qhull backends and belongs in hex3's benchmarks.
