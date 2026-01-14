//! Shared POD-like types for live dedup bookkeeping.

use glam::Vec3;

use crate::knn_clipping::cell_builder::VertexKey;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(in crate::knn_clipping) struct EdgeRecord {
    pub(in crate::knn_clipping) key: u64,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::knn_clipping) enum BadEdgeReason {
    MissingSide,
    EndpointMismatch,
    DuplicateSide,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::knn_clipping) struct BadEdgeRecord {
    pub(in crate::knn_clipping) key: u64,
    pub(in crate::knn_clipping) reason: BadEdgeReason,
}

#[derive(Clone, Copy)]
pub(super) struct EdgeCheck {
    pub(super) key: u64,
    pub(super) endpoints: [VertexKey; 2],
    pub(super) indices: [u32; 2],
}

#[derive(Clone, Copy)]
pub(super) struct EdgeCheckNode {
    pub(super) check: EdgeCheck,
    pub(super) next: u32,
}

#[derive(Clone, Copy)]
pub(super) struct EdgeCheckOverflow {
    pub(super) key: u64,
    pub(super) side: u8,
    pub(super) source_bin: u32,
    pub(super) endpoints: [VertexKey; 2],
    pub(super) indices: [u32; 2],
    pub(super) slots: [u32; 2],
}

#[derive(Clone, Copy)]
pub(super) struct EdgeLocal {
    pub(super) key: u64,
    pub(super) locals: [u8; 2],
}

#[derive(Clone, Copy)]
pub(super) struct EdgeToLater {
    pub(super) edge: EdgeLocal,
    pub(super) local_b: u32,
}

#[derive(Clone, Copy)]
pub(super) struct EdgeOverflowLocal {
    pub(super) edge: EdgeLocal,
    pub(super) side: u8,
}

#[derive(Clone, Copy)]
pub(super) struct DeferredSlot {
    pub(super) key: VertexKey,
    pub(super) pos: Vec3,
    pub(super) source_bin: u32,
    pub(super) source_slot: u32,
}

pub(super) struct SupportOverflow {
    pub(super) source_bin: u32,
    pub(super) target_bin: u32,
    pub(super) source_slot: u32,
    pub(super) support: Vec<u32>,
    pub(super) pos: Vec3,
}

pub(super) struct PackedSeed<'a> {
    pub(super) neighbors: &'a [u32],
    pub(super) count: usize,
    pub(super) security: f32,
    pub(super) k: usize,
}
