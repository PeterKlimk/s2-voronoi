//! Shared cell-building types for spherical Voronoi computation.

use glam::Vec3;

/// Vertex key for deduplication: sorted triplet of generator indices.
/// The triplet (A, B, C) represents the circumcenter of generators A, B, C.
pub type VertexKey = [u32; 3];

/// Vertex data: (key, position). Uses u32 indices to save space.
pub type VertexData = (VertexKey, Vec3);

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
    /// Failed to construct a valid seed polygon.
    NoValidSeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellBuildError {
    pub generator_idx: usize,
    pub failure: CellFailure,
}

/// A buffer to hold the output of clipping a cell.
#[derive(Default)]
pub struct CellOutputBuffer {
    pub vertices: Vec<VertexData>,
    pub edge_neighbor_globals: Vec<u32>,
    pub edge_neighbor_slots: Vec<u32>,
    pub edge_neighbor_eps: Vec<f32>,
}

impl CellOutputBuffer {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edge_neighbor_globals.clear();
        self.edge_neighbor_slots.clear();
        self.edge_neighbor_eps.clear();
    }
}
