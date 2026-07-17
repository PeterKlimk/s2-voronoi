//! Per-cell extraction output and failure types consumed by live dedup.

use glam::Vec3;

/// Vertex key for deduplication: sorted triplet of generator indices.
/// The triplet `(A, B, C)` represents the circumcenter of generators `A, B, C`.
pub(crate) type VertexKey = [u32; 3];

/// Vertex data: `(key, position)`. Uses `u32` indices to save space.
pub(crate) type VertexData = (VertexKey, Vec3);

/// Reasons a cell build can terminate unsuccessfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CellFailure {
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
pub(crate) struct CellBuildError {
    pub(crate) generator_idx: usize,
    pub(crate) failure: CellFailure,
    pub(crate) detail: Option<String>,
}

/// A reusable buffer to hold the extracted output of clipping a cell.
#[derive(Default)]
pub(crate) struct CellOutputBuffer {
    pub(crate) vertices: Vec<VertexData>,
    pub(crate) edge_neighbor_globals: Vec<u32>,
    pub(crate) edge_neighbor_slots: Vec<u32>,
    /// True when the extractor guarantees every real edge's neighbor appears
    /// in BOTH endpoint vertex keys (the emit engine's key/edge-consistency
    /// precondition). The incremental gnomonic clip maintains this by
    /// construction and sets it unconditionally (debug-asserted); the
    /// fallback extractors — whose split-plane corner resolution can strand
    /// a foreign plane in a surviving key — verify per edge (cold path) and
    /// set it accordingly. Emit uses the unchecked XOR "third" when set, and
    /// the checked malformed-endpoint-recording path when clear, so the
    /// common case pays nothing for the fallback's hazard.
    pub(crate) edge_keys_verified: bool,
}

impl CellOutputBuffer {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(capacity),
            edge_neighbor_globals: Vec::with_capacity(capacity),
            edge_neighbor_slots: Vec::with_capacity(capacity),
            edge_keys_verified: false,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.vertices.clear();
        self.edge_neighbor_globals.clear();
        self.edge_neighbor_slots.clear();
        self.edge_keys_verified = false;
    }
}
