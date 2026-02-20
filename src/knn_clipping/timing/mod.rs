//! Zero-cost timing instrumentation for knn_clipping.
//!
//! When the `timing` feature is enabled, this module collects and reports
//! coarse phase timings plus a small set of cell sub-phase totals.
//!
//! When disabled, all types become zero-sized and all methods compile away.

/// K-NN stage that a cell terminated at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnCellStage {
    /// Terminated during packed chunk0 (r=1).
    PackedChunk0,
    /// Terminated during packed tail (r=1, dot >= security).
    PackedTail,
    /// Ran directed no-K cursor fallback.
    DirectedCursor,
}

#[cfg(feature = "timing")]
mod real;
#[cfg(not(feature = "timing"))]
mod stub;

#[cfg(feature = "timing")]
pub use real::*;
#[cfg(not(feature = "timing"))]
pub use stub::*;
