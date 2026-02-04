//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! This module is an internal performance component. The only consumer in this crate is the
//! directed live-dedup backend, so we keep the implementation focused on that use-case.

mod scratch;
mod timing;

pub use scratch::{PackedKnnCellScratch, PackedKnnCellStatus};
pub use timing::PackedKnnTimings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedStage {
    Chunk0,
    Tail,
}

#[derive(Debug, Clone, Copy)]
pub struct PackedChunk {
    pub n: usize,
    pub unseen_bound: f32,
}

