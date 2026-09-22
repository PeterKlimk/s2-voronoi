//! Per-cell extraction output and failure types consumed by live dedup.

use glam::Vec3;
use std::hint::select_unpredictable;

/// Vertex key for deduplication: sorted triplet of generator indices.
/// The triplet `(A, B, C)` represents the circumcenter of generators `A, B, C`.
pub(crate) type VertexKey = [u32; 3];

/// Generator triple attributed to an extracted corner, without an ordering
/// requirement. Endpoint membership and XOR are independent of its order.
pub(crate) type VertexAttribution = [u32; 3];

/// Extracted vertex attribution and position. The generator triple may be
/// unordered: edge checks consume it as a set/XOR, and emission canonicalizes
/// unresolved triples before owner selection or persistent key storage. The
/// non-AVX2 extractor must emit sorted triples for its owner-first emission path.
pub(crate) type VertexData = (VertexAttribution, Vec3);

#[inline(always)]
fn cswap_u32(a: &mut u32, b: &mut u32) {
    let va = *a;
    let vb = *b;
    let cond = va <= vb;
    *a = select_unpredictable(cond, va, vb);
    *b = select_unpredictable(cond, vb, va);
}

/// Canonicalize a corner attribution for owner selection and stored key identity.
#[inline(always)]
pub(crate) fn sort3_u32(a: u32, b: u32, c: u32) -> VertexKey {
    // Sorting network (3 elements): (0,1) (1,2) (0,1)
    let mut x0 = a;
    let mut x1 = b;
    let mut x2 = c;
    cswap_u32(&mut x0, &mut x1);
    cswap_u32(&mut x1, &mut x2);
    cswap_u32(&mut x0, &mut x1);
    [x0, x1, x2]
}

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
    /// Native AVX2 emission accepts unordered attributions and canonicalizes
    /// only unresolved ones. Other targets require canonical triples because
    /// their owner-first emission shape precedes the resolved-index test.
    pub(crate) vertices: Vec<VertexData>,
    pub(crate) edge_neighbor_globals: Vec<u32>,
    pub(crate) edge_neighbor_slots: Vec<u32>,
    /// Dedup resolution slot initialized alongside extraction output and
    /// patched while incoming edge checks are collected.
    #[cfg(target_feature = "avx2")]
    pub(crate) vertex_indices: Vec<u32>,
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
            #[cfg(target_feature = "avx2")]
            vertex_indices: Vec::with_capacity(capacity),
            edge_keys_verified: false,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.vertices.clear();
        self.edge_neighbor_globals.clear();
        self.edge_neighbor_slots.clear();
        #[cfg(target_feature = "avx2")]
        self.vertex_indices.clear();
        self.edge_keys_verified = false;
    }
}
