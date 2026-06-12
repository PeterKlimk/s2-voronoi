//! Shared POD-like types for live dedup bookkeeping.

use glam::{Vec2, Vec3};

/// Vertex position type threaded through the dedup/assembly layer.
///
/// The engine is geometry-agnostic: it stores positions, dedups by
/// [`VertexKey`], and only ever needs a squared distance (edge-reconcile's
/// degenerate-length check). The sphere instantiates `Vec3`, the plane
/// `Vec2`; type defaults keep the spherical call sites spelled as before.
pub(crate) trait VertexPosition: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Squared Euclidean distance (chord distance on the unit sphere).
    fn dist_sq(self, other: Self) -> f32;
}

impl VertexPosition for Vec3 {
    #[inline]
    fn dist_sq(self, other: Self) -> f32 {
        (self - other).length_squared()
    }
}

impl VertexPosition for Vec2 {
    #[inline]
    fn dist_sq(self, other: Self) -> f32 {
        (self - other).length_squared()
    }
}

use crate::knn_clipping::cell_build::VertexKey;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct BinId(u8);

impl BinId {
    pub(crate) fn from_usize(value: usize) -> Self {
        Self(u8::try_from(value).expect("bin id must fit in u8"))
    }

    pub(crate) fn as_u8(self) -> u8 {
        self.0
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u8> for BinId {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl From<BinId> for u8 {
    fn from(value: BinId) -> Self {
        value.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct LocalId(u32);

impl LocalId {
    pub(crate) fn from_usize(value: usize) -> Self {
        Self(u32::try_from(value).expect("local id must fit in u32"))
    }

    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LocalId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<LocalId> for u32 {
    fn from(value: LocalId) -> Self {
        value.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct EdgeKey(u64);

impl EdgeKey {
    pub(crate) fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for EdgeKey {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<EdgeKey> for u64 {
    fn from(value: EdgeKey) -> Self {
        value.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeRecord {
    pub(crate) key: EdgeKey,
}

/// Which detection path recorded an unresolved shared-edge mismatch.
///
/// Diagnostic only (cold path, rare records): lets tests prove that each
/// detection path — the three within-bin cases and their cross-bin overflow
/// analogues — is actually exercised, instead of inferring coverage from
/// aggregate counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnresolvedEdgeOrigin {
    /// Within-bin: the later cell has an edge to an earlier same-bin
    /// neighbor, but no incoming edge check matched (the earlier cell
    /// concluded the edge does not exist).
    InBinMissingCheck,
    /// Within-bin: an incoming edge check matched the edge, but the endpoint
    /// "third" generators do not fully reconcile (the cells disagree on
    /// endpoint identity).
    InBinThirdsMismatch,
    /// Within-bin: an incoming edge check was never matched by any of the
    /// later cell's edges (the later cell concluded the edge does not exist).
    InBinUnconsumedCheck,
    /// Cross-bin: both overflow sides matched by key, but the endpoint
    /// "third" generators do not fully reconcile.
    CrossBinThirdsMismatch,
    /// Cross-bin: only one side emitted an overflow record for this edge key.
    CrossBinSingleSided,
    /// Cross-bin: both overflow records carried the same side tag (bug trap;
    /// debug-asserted).
    CrossBinDuplicateSide,
}

/// Historical name: this records an unresolved shared-edge reconciliation mismatch.
///
/// These are produced by edge-check matching when the two sides of an undirected edge cannot be
/// reconciled during live dedup. They are the only inputs to the narrow post-pass
/// reconciliation in `edge_reconcile.rs`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnresolvedEdgeMismatch {
    pub(crate) key: EdgeKey,
    pub(crate) origin: UnresolvedEdgeOrigin,
}

#[derive(Clone, Copy)]
pub(crate) struct EdgeCheck {
    pub(crate) key: EdgeKey,
    /// Half-plane epsilon to use when clipping this neighbor as an edgecheck-derived seed.
    ///
    /// This is stored to avoid recomputing a normalization-dependent epsilon (sqrt) in the hot
    /// edgecheck seeding path. Tiny cross-side differences are not important; we only need a
    /// stable tolerance scale.
    pub(crate) hp_eps: f32,
    /// For edge (A, B), each endpoint vertex key is (A, B, T).
    /// Store just the "third" generator T for each endpoint, in canonical order.
    pub(super) thirds: [u32; 2],
    pub(super) indices: [u32; 2],
}

#[derive(Clone, Copy)]
pub(super) struct EdgeCheckOverflow {
    pub(super) key: EdgeKey,
    pub(super) side: u8,
    pub(super) source_bin: BinId,
    /// See `EdgeCheck::thirds`.
    pub(super) thirds: [u32; 2],
    pub(super) indices: [u32; 2],
    pub(super) slots: [u32; 2],
}

/// Edge record to later-local neighbors (emitted into their incoming edgecheck queues).
///
/// This is ephemeral (per-cell scratch) and optimized for cache-friendly iteration in the emit
/// phase.
#[derive(Clone, Copy)]
pub(super) struct EdgeToLater {
    pub(super) key: EdgeKey,
    pub(super) local_b: LocalId,
    pub(super) locals: [u8; 2],
    pub(crate) hp_eps: f32,
}

/// Flattened for size: 16 bytes instead of 24.
/// Layout: key (8) + locals (2) + side (1) + 5 padding = 16
#[derive(Clone, Copy)]
pub(super) struct EdgeOverflowLocal {
    pub(super) key: EdgeKey,
    pub(super) locals: [u8; 2],
    pub(super) side: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct DeferredSlot<P = Vec3> {
    /// Canonical vertex key that identifies the eventual owner bin.
    pub(super) key: VertexKey,
    pub(super) pos: P,
    /// Bin/cell slot that still needs to be patched once ownership is resolved.
    pub(super) source_bin: BinId,
    pub(super) source_slot: u32,
}

// Packed-kNN data is handled via chunked emission from `cube_grid::packed_knn`.
