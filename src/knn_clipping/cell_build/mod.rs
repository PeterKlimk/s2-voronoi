//! Single-cell construction for the kNN + clipping backend.
//!
//! This phase owns neighbor seeding, directed neighbor-stream consumption,
//! clipping, terminal failure classification, and final vertex extraction.
//! Downstream live dedup consumes the extracted cell output and handles shard
//! ownership, deferred slots, and edge-check propagation.

use glam::Vec3;

mod run;

/// Vertex key for deduplication: sorted triplet of generator indices.
/// The triplet `(A, B, C)` represents the circumcenter of generators `A, B, C`.
pub type VertexKey = [u32; 3];

/// Vertex data: `(key, position)`. Uses `u32` indices to save space.
pub type VertexData<P = Vec3> = (VertexKey, P);

/// Reasons a cell build can terminate unsuccessfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellFailure {
    /// Exceeded vertex budget during clipping.
    TooManyVertices,
    /// Cell was completely clipped away (all vertices outside a plane).
    ClippedAway,
    /// The clipped cell reaches the generator hemisphere boundary, so gnomonic projection
    /// is no longer a valid model for the current feasible region.
    ProjectionInvalid,
    /// The neighbor stream was exhausted before the cell ever became bounded.
    ///
    /// This is not the same thing as a proven projection failure. It indicates we ended
    /// cell construction without a valid bounded polygon and should be classified separately
    /// from mathematically established unsupported geometry.
    UnboundedAfterExhaustion,
    /// Extraction invariants failed despite a supposedly valid bounded polygon.
    NoValidSeed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellBuildError {
    pub generator_idx: usize,
    pub failure: CellFailure,
    pub detail: Option<String>,
}

/// A reusable buffer to hold the extracted output of clipping a cell.
#[derive(Default)]
pub struct CellOutputBuffer<P = Vec3> {
    pub vertices: Vec<VertexData<P>>,
    pub edge_neighbor_globals: Vec<u32>,
    pub edge_neighbor_slots: Vec<u32>,
    pub edge_neighbor_eps: Vec<f32>,
}

impl<P> CellOutputBuffer<P> {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edge_neighbor_globals.clear();
        self.edge_neighbor_slots.clear();
        self.edge_neighbor_eps.clear();
    }
}

pub(crate) use run::{build_cell_into, CellBuildContext, CellBuildRequest, SeedNeighbor};
