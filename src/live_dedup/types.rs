//! Shared POD-like types for live dedup bookkeeping.

use glam::Vec3;

use crate::live_dedup::VertexKey;

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

/// An unresolved undirected generator edge handed from live assembly to the
/// narrow post-pass reconciliation stage.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EdgeRecord {
    pub(crate) key: EdgeKey,
}

#[derive(Clone, Copy)]
pub(crate) struct EdgeCheck {
    /// Grid slot of the earlier same-bin generator that forwarded this check.
    /// Geometry seeding consumes the slot directly and recovers the global
    /// generator id from the same `SlotPoint` load. The destination generator
    /// is known by every consumer, so cold diagnostics can still reconstruct
    /// the canonical edge key losslessly.
    pub(crate) neighbor_slot: u32,
    /// For edge (A, B), each endpoint vertex key is (A, B, T).
    /// Store just the "third" generator T for each endpoint, in canonical
    /// order. `u32::MAX` (`edge_checks::MALFORMED_THIRD`) marks an endpoint
    /// whose key did not name both edge endpoints (recorded as an
    /// `EndpointKeyMismatch` defect at the emitter); it never matches during
    /// endpoint reconciliation.
    pub(super) thirds: [u32; 2],
    pub(super) indices: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<EdgeCheck>() == 20);

#[derive(Clone, Copy)]
pub(super) struct EdgeCheckOverflow {
    pub(super) key: EdgeKey,
    pub(super) side: u8,
    pub(super) source_bin: BinId,
    /// Opposite endpoint's bin. This occupies existing record padding and
    /// lets assembly group cross-bin checks without another global-id lookup.
    pub(super) target_bin: BinId,
    /// See `EdgeCheck::thirds`.
    pub(super) thirds: [u32; 2],
    pub(super) indices: [u32; 2],
    pub(super) slots: [u32; 2],
    pub(super) source_cell: u32,
    pub(super) source_offsets: [u8; 2],
}

const _: () = assert!(std::mem::size_of::<EdgeCheckOverflow>() == 48);

/// Edge record to later-local neighbors (emitted into their incoming edgecheck queues).
///
/// This is ephemeral (per-cell scratch) and optimized for cache-friendly iteration in the emit
/// phase.
#[derive(Clone, Copy)]
pub(super) struct EdgeToLater {
    pub(super) key: EdgeKey,
    pub(super) local_b: LocalId,
    pub(super) locals: [u8; 2],
}

const _: () = assert!(std::mem::size_of::<EdgeToLater>() == 16);

/// Flattened for size: 16 bytes instead of 24.
/// Layout: key (8) + locals (2) + side (1) + 5 padding = 16
#[derive(Clone, Copy)]
pub(super) struct EdgeOverflowLocal {
    pub(super) key: EdgeKey,
    pub(super) locals: [u8; 2],
    pub(super) side: u8,
    pub(super) target_bin: BinId,
}

const _: () = assert!(std::mem::size_of::<EdgeOverflowLocal>() == 16);

#[derive(Clone, Copy)]
pub(crate) struct DeferredSlot {
    /// Canonical vertex key that identifies the eventual owner bin.
    pub(super) key: VertexKey,
    pub(super) pos: Vec3,
    /// Bin/cell slot that still needs to be patched once ownership is resolved.
    pub(super) source_bin: BinId,
    pub(super) source_slot: u32,
    pub(super) source_cell: u32,
    pub(super) source_offset: u8,
}

/// Sparse override for a cell reference whose vertex is owned by another
/// shard. The temporary per-shard lookup maps source slots to these records
/// during reconciliation; `source_cell`/`source_offset` identify the final CSR
/// destination directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CellReferenceOverride {
    pub(super) source_cell: u32,
    pub(super) owner_local: u32,
    pub(super) source_offset: u8,
    pub(super) owner_bin: BinId,
}

const _: () = assert!(std::mem::size_of::<CellReferenceOverride>() == 12);

// Packed-kNN data is handled via chunked emission from `cube_grid::packed_knn`.
