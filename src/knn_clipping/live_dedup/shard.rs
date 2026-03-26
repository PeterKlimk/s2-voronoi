//! Shard-local state for live dedup.

use glam::Vec3;
use rustc_hash::FxHashMap;

use super::types::{
    DeferredSlot, EdgeCheck, EdgeCheckOverflow, LocalId, RemoteSupportVertexWrite,
    UnresolvedEdgeMismatch,
};
use crate::knn_clipping::cell_builder::VertexKey;

/// Data only needed during vertex deduplication (dropped after overflow flush).
pub(super) struct ShardDedup {
    /// Support-set-owned vertices keyed by their canonical support set.
    pub(super) support_vertex_map: FxHashMap<Vec<u32>, u32>,
    /// Cross-bin patch records for support-set-owned vertices.
    pub(super) remote_support_writes: Vec<RemoteSupportVertexWrite>,
    /// Per-local edge checks (Vec-based for cache locality)
    pub(super) edge_checks: Vec<Vec<EdgeCheck>>,
    /// Pool of reusable Vecs with existing capacity
    pub(super) edge_check_pool: Vec<Vec<EdgeCheck>>,
}

impl ShardDedup {
    pub(super) fn new(num_local_generators: usize) -> Self {
        Self {
            support_vertex_map: FxHashMap::default(),
            remote_support_writes: Vec::new(),
            edge_checks: (0..num_local_generators).map(|_| Vec::new()).collect(),
            edge_check_pool: Vec::new(),
        }
    }
}

/// Output data needed for final assembly.
pub(super) struct ShardOutput {
    pub(super) vertices: Vec<Vec3>,
    pub(super) vertex_keys: Vec<VertexKey>,
    pub(super) unresolved_edges: Vec<UnresolvedEdgeMismatch>,
    pub(super) edge_check_overflow: Vec<EdgeCheckOverflow>,
    /// Cell slots whose owner bin is off-shard and must be patched during assembly.
    pub(super) deferred_slots: Vec<DeferredSlot>,
    pub(super) cell_indices: Vec<u64>,
    pub(super) cell_starts: Vec<u32>,
    pub(super) cell_counts: Vec<u8>,
}

impl ShardOutput {
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
        }
    }

    #[inline(always)]
    pub(super) fn set_cell_start(&mut self, local: LocalId, start: u32) {
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
pub(super) struct ShardState {
    pub(super) dedup: ShardDedup,
    pub(super) output: ShardOutput,
    #[cfg(feature = "timing")]
    pub(super) triplet_keys: u64,
    #[cfg(feature = "timing")]
    pub(super) support_keys: u64,
}

impl ShardState {
    pub(super) fn new(num_local_generators: usize) -> Self {
        Self {
            dedup: ShardDedup::new(num_local_generators),
            output: ShardOutput::new(num_local_generators),
            #[cfg(feature = "timing")]
            triplet_keys: 0,
            #[cfg(feature = "timing")]
            support_keys: 0,
        }
    }

    #[inline(always)]
    pub(super) fn dedup_owned_support_vertex(&mut self, support: Vec<u32>, pos: Vec3) -> u32 {
        if let Some(&idx) = self.dedup.support_vertex_map.get(support.as_slice()) {
            return idx;
        }
        let idx = self.output.vertices.len() as u32;
        self.output.vertices.push(pos);
        self.dedup.support_vertex_map.insert(support, idx);
        idx
    }

    pub(super) fn into_final(self) -> ShardFinal {
        ShardFinal {
            output: self.output,
            #[cfg(feature = "timing")]
            triplet_keys: self.triplet_keys,
            #[cfg(feature = "timing")]
            support_keys: self.support_keys,
        }
        // self.dedup dropped here automatically
    }
}

/// Shard state after construction, with dedup dropped.
pub(super) struct ShardFinal {
    pub(super) output: ShardOutput,
    #[cfg(feature = "timing")]
    pub(super) triplet_keys: u64,
    #[cfg(feature = "timing")]
    pub(super) support_keys: u64,
}
