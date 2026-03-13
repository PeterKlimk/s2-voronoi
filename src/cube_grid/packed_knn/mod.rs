//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! This module is an internal performance component. The only consumer in this crate is the
//! directed live-dedup backend, so we keep the implementation focused on that use-case.

mod scratch;
mod timing;

use super::CubeMapGrid;

pub use scratch::{PackedKnnCellScratch, PackedKnnCellStatus};
pub use timing::PackedKnnTimings;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectedCellGroup<'a> {
    cell: usize,
    query_bin: u8,
    queries: &'a [u32],
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    query_locals: &'a [u32],
    slot_gen_map: &'a [u32],
    local_shift: u32,
    local_mask: u32,
}

impl<'a> DirectedCellGroup<'a> {
    #[inline]
    pub(crate) fn new(
        cell: usize,
        query_bin: u8,
        queries: &'a [u32],
        query_locals: &'a [u32],
        slot_gen_map: &'a [u32],
        local_shift: u32,
        local_mask: u32,
    ) -> Self {
        Self {
            cell,
            query_bin,
            queries,
            query_locals,
            slot_gen_map,
            local_shift,
            local_mask,
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
    pub(crate) fn query_locals(self) -> &'a [u32] {
        self.query_locals
    }

    #[inline]
    pub(crate) fn slot_gen_map(self) -> &'a [u32] {
        self.slot_gen_map
    }

    #[inline]
    pub(crate) fn local_shift(self) -> u32 {
        self.local_shift
    }

    #[inline]
    pub(crate) fn local_mask(self) -> u32 {
        self.local_mask
    }

    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_matches_grid(self, grid: &CubeMapGrid) {
        debug_assert_eq!(
            self.queries.len(),
            self.query_locals.len(),
            "directed packed group queries/locals length mismatch"
        );

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
            self.query_locals.windows(2).all(|w| w[1] == w[0] + 1),
            "directed packed group locals must be contiguous in slot order"
        );
        debug_assert!(
            self.queries
                .iter()
                .zip(self.query_locals.iter())
                .all(|(&slot, &ql)| {
                    let packed = self.slot_gen_map[slot as usize];
                    let bin = (packed >> self.local_shift) as u8;
                    let local = packed & self.local_mask;
                    bin == self.query_bin && local == ql
                }),
            "directed packed group (slot -> bin,local) mapping must match query inputs"
        );
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    pub(crate) fn debug_assert_matches_grid(self, _grid: &CubeMapGrid) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedStage {
    Chunk0,
    Tail,
}

#[derive(Debug, Clone, Copy)]
pub struct PackedChunk {
    pub n: usize,
    pub unseen_bound: f32,
}
