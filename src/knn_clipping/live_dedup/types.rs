//! Shared POD-like types for live dedup bookkeeping.

use glam::Vec3;

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

/// Historical name: this records an unresolved shared-edge reconciliation mismatch.
///
/// These are produced by edge-check matching when the two sides of an undirected edge cannot be
/// reconciled during live dedup. They are the only inputs to the narrow post-pass
/// reconciliation in `edge_reconcile.rs`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnresolvedEdgeMismatch {
    pub(crate) key: EdgeKey,
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
pub(crate) struct DeferredSlot {
    /// Canonical vertex key that identifies the eventual owner bin.
    pub(super) key: VertexKey,
    pub(super) pos: Vec3,
    /// Bin/cell slot that still needs to be patched once ownership is resolved.
    pub(super) source_bin: BinId,
    pub(super) source_slot: u32,
}

// Packed-kNN data is handled via chunked emission from `cube_grid::packed_knn`.
