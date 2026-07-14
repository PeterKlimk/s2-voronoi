//! Shard-local state for live dedup.

use super::types::{DeferredSlot, EdgeCheck, EdgeCheckOverflow, LocalId, UnresolvedEdgeMismatch};
use crate::knn_clipping::cell_build::VertexKey;
use glam::Vec3;

use super::types::VertexPosition;

/// Data only needed during vertex deduplication (dropped after overflow flush).
pub(crate) struct ShardDedup {
    /// Per-local edge checks (Vec-based for cache locality)
    pub(super) edge_checks: Vec<Vec<EdgeCheck>>,
    /// Pool of reusable Vecs with existing capacity
    pub(super) edge_check_pool: Vec<Vec<EdgeCheck>>,
}

impl ShardDedup {
    pub(super) fn new(num_local_generators: usize) -> Self {
        Self {
            edge_checks: (0..num_local_generators).map(|_| Vec::new()).collect(),
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
