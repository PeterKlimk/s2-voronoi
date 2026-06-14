//! Spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from nearby neighbors. This structure is friendly to data-parallel CPU
//! implementations.

use crate::policy::PackedNeighborPolicy;

pub(crate) mod cell_build;
pub(crate) mod compute;
pub(crate) mod driver;
pub(crate) mod edge_reconcile;
// The live-dedup engine moved to the crate root (it is geometry-agnostic
// and serves both the spherical and planar drivers); re-exported so
// existing paths keep working.
pub(crate) use crate::live_dedup;
pub(crate) mod canonical;
#[cfg(feature = "p5_shadow")]
pub(crate) mod p5_shadow;
pub(crate) mod preprocess;
// Timing instrumentation moved to the crate root (the live-dedup engine
// uses it too); re-exported so existing paths keep working.
pub(crate) use crate::timing;
pub(crate) mod topo2d;
pub(crate) mod union_find;

// Re-exports (internal use)
#[allow(unused_imports)]
pub use compute::compute_voronoi_knn_clipping_with_config_owned;
#[allow(unused_imports)]
pub use compute::compute_voronoi_knn_clipping_with_report_owned;
pub use preprocess::merge_close_points;

pub type MergeResult = preprocess::MergeResult;

#[derive(Debug, Clone, Copy, Default)]
pub struct TerminationConfig {
    /// Enable a cold packed r=2 expansion stage before the shell takeover.
    ///
    /// Defaults to `false` (the bool default): measured net-negative as
    /// implemented — see `VoronoiConfig::packed_knn_expand_r2` docs and
    /// `docs/multi-regime-perf.md` items 1 & 9.
    pub packed_expand_r2: bool,
}

impl TerminationConfig {
    #[inline]
    pub(crate) fn packed_policy(&self, num_points: usize) -> PackedNeighborPolicy {
        PackedNeighborPolicy::for_point_count(num_points, self.packed_expand_r2)
    }
}

// NOTE: benchmark_voronoi function was removed during crate extraction.
// It compared knn_clipping vs qhull backends and belongs in hex3's benchmarks.
