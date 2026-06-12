//! Timing hooks for the planar packed stage.
//!
//! The sphere's [`PackedKnnTimings`] is reused verbatim — the planar stage
//! has the same sub-phases and feeds the same `CellSubAccum` breakdown seam
//! — so the only local piece is the lap timer alias. Under the `timing`
//! feature laps are real `Duration`s; otherwise everything is zero-sized
//! and compiles away (the sphere's real/stub split).

pub use crate::cube_grid::packed_knn::PackedKnnTimings as PlanePackedTimings;
pub(crate) use crate::timing::LapTimer as PlanePackedLapTimer;
