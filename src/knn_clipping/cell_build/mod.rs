//! Single-cell construction for the kNN + clipping backend.
//!
//! This phase owns neighbor seeding, directed neighbor-stream consumption,
//! clipping, terminal failure classification, and final vertex extraction.
//! Downstream live dedup consumes the extracted cell output and handles shard
//! ownership, deferred slots, and edge-check propagation.

mod run;

pub(crate) use run::{build_cell_into, CellBuildContext, CellBuildRequest};
