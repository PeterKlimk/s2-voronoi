//! Single-cell construction for the kNN + clipping backend.
//!
//! This phase owns neighbor seeding, directed neighbor-stream consumption,
//! clipping, terminal failure classification, and final vertex extraction.
//! Downstream live dedup consumes the extracted cell output and handles shard
//! ownership, deferred slots, and edge-check propagation.

mod run;

// The cell-output vocabulary moved into the live-dedup engine (it is the
// engine's input format, shared with the planar driver); re-exported here
// so existing paths keep working.
pub use crate::live_dedup::{CellBuildError, CellFailure, CellOutputBuffer, VertexKey};

pub(crate) use run::{build_cell_into, CellBuildContext, CellBuildRequest};
