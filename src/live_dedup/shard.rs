//! Shard-local state for live dedup.

use super::types::{DeferredSlot, EdgeCheck, EdgeCheckOverflow, LocalId, UnresolvedEdgeMismatch};
use crate::knn_clipping::cell_build::VertexKey;
use glam::Vec3;

use super::types::VertexPosition;

/// One-pointer per-local queue handle. The `Vec` header lives only for
/// populated queues and moves intact through take/recycle.
#[allow(clippy::box_collection)] // the outer box is the one-pointer thin handle
pub(crate) struct EdgeCheckQueue(Option<Box<Vec<EdgeCheck>>>);

#[allow(clippy::box_collection)]
impl EdgeCheckQueue {
    #[inline]
    pub(super) fn from_box(queue: Option<Box<Vec<EdgeCheck>>>) -> Self {
        Self(queue)
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[EdgeCheck] {
        self.0.as_deref().map_or(&[], Vec::as_slice)
    }

    #[inline]
    pub(super) fn into_box(self) -> Option<Box<Vec<EdgeCheck>>> {
        self.0
    }
}

impl From<Vec<EdgeCheck>> for EdgeCheckQueue {
    fn from(queue: Vec<EdgeCheck>) -> Self {
        if queue.is_empty() {
            Self(None)
        } else {
            Self(Some(Box::new(queue)))
        }
    }
}

impl std::ops::Deref for EdgeCheckQueue {
    type Target = [EdgeCheck];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Data only needed during vertex deduplication (dropped after overflow flush).
#[allow(clippy::box_collection, clippy::vec_box)] // intentional thin-queue experiment
pub(crate) struct ShardDedup {
    /// Per-local one-pointer handles; only populated queues allocate a header.
    pub(super) edge_checks: Vec<Option<Box<Vec<EdgeCheck>>>>,
    /// Pool of reusable queue headers and their existing payload capacity.
    pub(super) edge_check_pool: Vec<Box<Vec<EdgeCheck>>>,
}

impl ShardDedup {
    pub(super) fn new(num_local_generators: usize) -> Self {
        Self {
            edge_checks: (0..num_local_generators).map(|_| None).collect(),
            edge_check_pool: Vec::new(),
        }
    }
}

/// Output data needed for final assembly.
pub(crate) struct ShardOutput<P = Vec3> {
    pub(crate) vertices: Vec<P>,
    pub(crate) vertex_keys: Vec<VertexKey>,
    pub(super) unresolved_edges: Vec<UnresolvedEdgeMismatch>,
    pub(super) edge_check_overflow: Vec<EdgeCheckOverflow>,
    /// Cell slots whose owner bin is off-shard and must be patched during assembly.
    pub(crate) deferred_slots: Vec<DeferredSlot<P>>,
    pub(crate) cell_indices: Vec<u64>,
    pub(super) cell_starts: Vec<u32>,
    pub(super) cell_counts: Vec<u8>,
    pub(crate) exact_zero_edge_hint_cells: Vec<u32>,
    pub(crate) resolution_drift_exceeded: bool,
}

impl<P: VertexPosition> ShardOutput<P> {
    pub(super) fn new(num_local_generators: usize) -> Self {
        Self {
            vertices: Vec::new(),
            vertex_keys: Vec::new(),
            unresolved_edges: Vec::new(),
            edge_check_overflow: Vec::new(),
            deferred_slots: Vec::new(),
            cell_indices: Vec::new(),
            cell_starts: vec![0; num_local_generators],
            cell_counts: vec![0; num_local_generators],
            exact_zero_edge_hint_cells: Vec::new(),
            resolution_drift_exceeded: false,
        }
    }

    #[inline(always)]
    pub(crate) fn set_cell_start(&mut self, local: LocalId, start: u32) {
        self.cell_starts[local.as_usize()] = start;
    }

    #[inline(always)]
    pub(super) fn cell_start(&self, local: LocalId) -> u32 {
        self.cell_starts[local.as_usize()]
    }

    #[inline(always)]
    pub(super) fn set_cell_count(&mut self, local: LocalId, count: u8) {
        self.cell_counts[local.as_usize()] = count;
    }

    #[inline(always)]
    pub(super) fn cell_count(&self, local: LocalId) -> u8 {
        self.cell_counts[local.as_usize()]
    }
}

/// Per-shard state during cell construction.
pub(crate) struct ShardState<P = Vec3> {
    pub(crate) dedup: ShardDedup,
    pub(crate) output: ShardOutput<P>,
    #[cfg(feature = "timing")]
    pub(super) triplet_keys: u64,
}

impl<P: VertexPosition> ShardState<P> {
    pub(crate) fn new(num_local_generators: usize) -> Self {
        Self {
            dedup: ShardDedup::new(num_local_generators),
            output: ShardOutput::new(num_local_generators),
            #[cfg(feature = "timing")]
            triplet_keys: 0,
        }
    }

    pub(super) fn into_final(self) -> ShardFinal<P> {
        ShardFinal {
            output: self.output,
            #[cfg(feature = "timing")]
            triplet_keys: self.triplet_keys,
        }
        // self.dedup dropped here automatically
    }
}

/// Shard state after construction, with dedup dropped.
pub(super) struct ShardFinal<P = Vec3> {
    pub(crate) output: ShardOutput<P>,
    #[cfg(feature = "timing")]
    pub(super) triplet_keys: u64,
}
