//! Shared POD-like types for live dedup bookkeeping.

use glam::Vec3;

use crate::knn_clipping::cell_builder::VertexKey;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(in crate::knn_clipping) struct BinId(u8);

impl BinId {
    pub(in crate::knn_clipping) fn from_usize(value: usize) -> Self {
        Self(u8::try_from(value).expect("bin id must fit in u8"))
    }

    pub(in crate::knn_clipping) fn as_u8(self) -> u8 {
        self.0
    }

    pub(in crate::knn_clipping) fn as_usize(self) -> usize {
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
pub(in crate::knn_clipping) struct LocalId(u32);

impl LocalId {
    pub(in crate::knn_clipping) fn from_usize(value: usize) -> Self {
        Self(u32::try_from(value).expect("local id must fit in u32"))
    }

    pub(in crate::knn_clipping) fn as_u32(self) -> u32 {
        self.0
    }

    pub(in crate::knn_clipping) fn as_usize(self) -> usize {
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
pub(in crate::knn_clipping) struct EdgeKey(u64);

impl EdgeKey {
    pub(in crate::knn_clipping) fn as_u64(self) -> u64 {
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
pub(in crate::knn_clipping) struct EdgeRecord {
    pub(in crate::knn_clipping) key: EdgeKey,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::knn_clipping) struct BadEdgeRecord {
    pub(in crate::knn_clipping) key: EdgeKey,
}

#[derive(Clone, Copy)]
pub(super) struct EdgeCheck {
    pub(super) key: EdgeKey,
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

/// Flattened for size: 16 bytes instead of 24.
/// Layout: key (8) + local_b (4) + locals (2) + 2 padding = 16
#[derive(Clone, Copy)]
pub(super) struct EdgeToLater {
    pub(super) key: EdgeKey,
    pub(super) local_b: LocalId,
    pub(super) locals: [u8; 2],
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
pub(super) struct DeferredSlot {
    pub(super) key: VertexKey,
    pub(super) pos: Vec3,
    pub(super) source_bin: BinId,
    pub(super) source_slot: u32,
}

pub(super) struct SupportOverflow {
    pub(super) source_bin: BinId,
    pub(super) target_bin: BinId,
    pub(super) source_slot: u32,
    pub(super) support: Vec<u32>,
    pub(super) pos: Vec3,
}

// Packed-kNN data is handled via chunked emission from `cube_grid::packed_knn`.
