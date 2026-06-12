//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! This module is an internal performance component. The only consumer in this crate is the
//! directed live-dedup backend, so we keep the implementation focused on that use-case.

mod scratch;
mod timing;

use super::CubeMapGrid;
use crate::packed_layout::PackedSlotLayout;
use crate::policy::PackedNeighborPolicy;

pub use scratch::PackedKnnCellScratch;
pub(crate) use scratch::{PreparedPackedGroup, PreparedPackedGroupStatus};
pub use timing::PackedKnnTimings;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedGroupInput<'a> {
    cell: usize,
    query_bin: u8,
    queries: &'a [u32],
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    query_local_start: u32,
    layout: PackedSlotLayout<'a>,
}

impl<'a> PackedGroupInput<'a> {
    #[inline]
    pub(crate) fn new(
        cell: usize,
        query_bin: u8,
        queries: &'a [u32],
        query_local_start: u32,
        layout: PackedSlotLayout<'a>,
    ) -> Self {
        Self {
            cell,
            query_bin,
            queries,
            query_local_start,
            layout,
        }
    }

    #[inline]
    pub(crate) fn cell(self) -> usize {
        self.cell
    }

    #[inline]
    pub(crate) fn query_bin(self) -> u8 {
        self.query_bin
    }

    #[inline]
    pub(crate) fn queries(self) -> &'a [u32] {
        self.queries
    }

    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn query_local(self, query_index: usize) -> u32 {
        self.query_local_start + query_index as u32
    }

    #[inline]
    pub(crate) fn slot_gen_map(self) -> &'a [u32] {
        self.layout.slot_gen_map()
    }

    #[inline]
    pub(crate) fn local_shift(self) -> u32 {
        self.layout.local_shift()
    }

    #[inline]
    pub(crate) fn local_mask(self) -> u32 {
        self.layout.local_mask()
    }

    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_matches_grid(self, grid: &CubeMapGrid) {
        let start = grid.cell_offsets()[self.cell] as usize;
        let end = grid.cell_offsets()[self.cell + 1] as usize;
        debug_assert_eq!(
            self.queries.len(),
            end - start,
            "directed packed group must cover the full center cell"
        );
        debug_assert!(
            self.queries
                .iter()
                .enumerate()
                .all(|(offset, &slot)| slot as usize == start + offset),
            "directed packed group queries must be the center cell in slot order"
        );
        debug_assert!(
            self.queries
                .iter()
                .enumerate()
                .all(|(offset, _)| self.query_local(offset)
                    == self.query_local_start + offset as u32),
            "directed packed group locals must be contiguous in slot order"
        );
        debug_assert!(
            self.queries.iter().enumerate().all(|(offset, &slot)| {
                let (bin, local) = self.layout.bin_local(slot);
                bin == self.query_bin && local == self.query_local(offset)
            }),
            "directed packed group (slot -> bin,local) mapping must match query inputs"
        );
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    pub(crate) fn debug_assert_matches_grid(self, _grid: &CubeMapGrid) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedStage {
    Chunk0,
    Tail,
    ExpandR2,
}

#[derive(Debug, Clone, Copy)]
struct PackedChunk {
    n: usize,
    unseen_bound: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackedNeighborBatchSource {
    Chunk0,
    Tail,
    ExpandR2,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedNeighborBatch {
    pub(crate) n: usize,
    pub(crate) first_dot: f32,
    pub(crate) unseen_bound: f32,
    pub(crate) source: PackedNeighborBatchSource,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PackedNeighborFrontier {
    ExactBatch(PackedNeighborBatch),
    UnknownButBounded { dot_upper_bound: f32 },
    Exhausted,
}

#[derive(Debug, Clone)]
enum CachedFrontier {
    ExactBatch(PackedNeighborBatch),
    UnknownButBounded { dot_upper_bound: f32 },
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedQueryStage {
    Chunk0,
    Tail,
    ExpandR2,
    Exhausted,
}

pub(crate) struct PackedQuery<'a, 'p, 'g> {
    prepared: &'a mut PreparedPackedGroup<'p, 'g>,
    timings: &'a mut PackedKnnTimings,
    query_index: usize,
    query_x: f32,
    query_y: f32,
    query_z: f32,
    policy: PackedNeighborPolicy,
    stage: PackedQueryStage,
    cached_frontier: Option<CachedFrontier>,
    tail_used: bool,
    expand_r2_used: bool,
    safe_exhausted: bool,
    cached_slots: Vec<u32>,
}

impl<'a, 'p, 'g> PackedQuery<'a, 'p, 'g> {
    pub(crate) fn new(
        prepared: &'a mut PreparedPackedGroup<'p, 'g>,
        timings: &'a mut PackedKnnTimings,
        grid: &CubeMapGrid,
        query_index: usize,
        policy: PackedNeighborPolicy,
    ) -> Self {
        let query_slot = prepared.group().queries()[query_index];
        let query_slot_usize = query_slot as usize;
        Self {
            prepared,
            timings,
            query_index,
            query_x: grid.cell_points_x[query_slot_usize],
            query_y: grid.cell_points_y[query_slot_usize],
            query_z: grid.cell_points_z[query_slot_usize],
            policy,
            stage: PackedQueryStage::Chunk0,
            cached_frontier: None,
            tail_used: false,
            expand_r2_used: false,
            safe_exhausted: false,
            cached_slots: Vec::new(),
        }
    }

    #[inline]
    fn slot_dot(&self, grid: &CubeMapGrid, slot: u32) -> f32 {
        let slot = slot as usize;
        crate::fp::dot3_f32(
            self.query_x,
            self.query_y,
            self.query_z,
            grid.cell_points_x[slot],
            grid.cell_points_y[slot],
            grid.cell_points_z[slot],
        )
    }

    pub(crate) fn frontier(
        &mut self,
        grid: &CubeMapGrid,
        out: &mut Vec<u32>,
    ) -> PackedNeighborFrontier {
        if let Some(cached) = &self.cached_frontier {
            match cached {
<<<<<<< HEAD
                CachedFrontier::ExactBatch { batch, slots } => {
                    out.clear();
                    out.extend_from_slice(slots);
=======
                CachedFrontier::ExactBatch(batch) => {
                    out.extend_from_slice(&self.cached_slots);
>>>>>>> agent/micro-opt-frontier-cache-ordf32
                    return PackedNeighborFrontier::ExactBatch(*batch);
                }
                CachedFrontier::UnknownButBounded { dot_upper_bound } => {
                    out.clear();
                    return PackedNeighborFrontier::UnknownButBounded {
                        dot_upper_bound: *dot_upper_bound,
                    };
                }
                CachedFrontier::Exhausted => {
                    out.clear();
                    return PackedNeighborFrontier::Exhausted;
                }
            }
        }

        let (stage, source, k) = match self.stage {
            PackedQueryStage::Chunk0 => (
                PackedStage::Chunk0,
                PackedNeighborBatchSource::Chunk0,
                self.policy.chunk0_size(),
            ),
            PackedQueryStage::Tail => (
                PackedStage::Tail,
                PackedNeighborBatchSource::Tail,
                self.policy.chunk_size(),
            ),
            PackedQueryStage::ExpandR2 => (
                PackedStage::ExpandR2,
                PackedNeighborBatchSource::ExpandR2,
                self.policy.chunk_size(),
            ),
            PackedQueryStage::Exhausted => {
                self.cached_frontier = Some(CachedFrontier::Exhausted);
                out.clear();
                return PackedNeighborFrontier::Exhausted;
            }
        };

        if out.len() < k {
            out.resize(k, 0);
        } else {
            out.truncate(k);
        }
        if let Some(chunk) = self
            .prepared
            .next_chunk(self.query_index, stage, k, out, self.timings)
        {
            out.truncate(chunk.n);
            let first_dot = if chunk.n > 0 {
                self.slot_dot(grid, out[0])
            } else {
                -1.0
            };
            let batch = PackedNeighborBatch {
                n: chunk.n,
                first_dot,
                unseen_bound: chunk.unseen_bound,
                source,
            };
            self.cached_slots.clear();
            self.cached_slots.extend_from_slice(out);
            self.cached_frontier = Some(CachedFrontier::ExactBatch(batch));
            return PackedNeighborFrontier::ExactBatch(batch);
        }

        out.clear();

        let dot_upper_bound = if self.stage == PackedQueryStage::Chunk0
            && self.prepared.tail_possible(self.query_index)
        {
            self.prepared.tail_upper_bound(self.query_index)
        } else {
            self.prepared.resume_security(self.query_index)
        };
        self.cached_frontier = Some(CachedFrontier::UnknownButBounded { dot_upper_bound });
        PackedNeighborFrontier::UnknownButBounded { dot_upper_bound }
    }

    pub(crate) fn advance_frontier(&mut self, grid: &CubeMapGrid) {
        let cached = self.cached_frontier.take();
        match cached {
            Some(CachedFrontier::ExactBatch(_)) => {}
            Some(CachedFrontier::UnknownButBounded { .. }) => self.advance_stage(grid),
            Some(CachedFrontier::Exhausted) | None => {}
        }
    }

    fn advance_stage(&mut self, grid: &CubeMapGrid) {
        if self.stage == PackedQueryStage::Chunk0 && self.prepared.tail_possible(self.query_index) {
            self.prepared
                .ensure_tail_directed_for(self.query_index, grid, self.timings);
            self.stage = PackedQueryStage::Tail;
            self.tail_used = true;
            return;
        }

        if self.stage != PackedQueryStage::ExpandR2
            && self.policy.expand_r2_enabled()
            && self.prepared.ensure_expand_r2_band_directed_for(
                self.query_index,
                grid,
                self.timings,
            )
        {
            self.stage = PackedQueryStage::ExpandR2;
            self.expand_r2_used = true;
            return;
        }

        self.safe_exhausted = true;
        self.stage = PackedQueryStage::Exhausted;
    }

    #[inline]
    pub(crate) fn tail_used(&self) -> bool {
        self.tail_used
    }

    #[inline]
    pub(crate) fn expand_r2_used(&self) -> bool {
        self.expand_r2_used
    }

    #[inline]
    pub(crate) fn safe_exhausted(&self) -> bool {
        self.safe_exhausted
    }

    #[inline]
    pub(crate) fn is_exhausted(&self) -> bool {
        self.stage == PackedQueryStage::Exhausted
    }
}
