use crate::fp;
use glam::Vec3;
use std::cmp::Reverse;

use super::super::{CubeMapGrid, CubeMapGridScratch, OrdF32};

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectedCtx<'a> {
    query_bin: u8,
    query_local: u32,
    slot_gen_map: &'a [u32],
    local_shift: u32,
    local_mask: u32,
}

impl<'a> DirectedCtx<'a> {
    #[inline]
    pub(crate) fn new(
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &'a [u32],
        local_shift: u32,
        local_mask: u32,
    ) -> Self {
        Self {
            query_bin,
            query_local,
            slot_gen_map,
            local_shift,
            local_mask,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectedCellMode {
    TransitOnly,
    EmitAll,
    EmitCenterDirected,
}

pub(crate) struct DirectedNoKCursor<'a, 'b> {
    grid: &'a CubeMapGrid,
    scratch: &'a mut CubeMapGridScratch,
    query: Vec3,
    query_idx: usize,
    start_cell: u32,
    query_bin: u8,
    query_local: u32,
    slot_gen_map: &'b [u32],
    local_shift: u32,
    local_mask: u32,
    exhausted: bool,
}

impl<'a, 'b> DirectedNoKCursor<'a, 'b> {
    fn new(
        grid: &'a CubeMapGrid,
        query: Vec3,
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        ctx: DirectedCtx<'b>,
    ) -> Self {
        let start_cell = if query_idx < grid.point_cells.len() {
            grid.point_cells[query_idx]
        } else {
            grid.point_to_cell(query) as u32
        };

        scratch.cell_heap.clear();
        scratch.point_heap.clear();
        scratch.exhausted = false;
        scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
        if scratch.stamp == u32::MAX {
            scratch.visited_stamp.fill(0);
            scratch.stamp = 1;
        }
        scratch.mark_visited(start_cell);
        scratch.push_cell(start_cell, 0.0);

        Self {
            grid,
            scratch,
            query,
            query_idx,
            start_cell,
            query_bin: ctx.query_bin,
            query_local: ctx.query_local,
            slot_gen_map: ctx.slot_gen_map,
            local_shift: ctx.local_shift,
            local_mask: ctx.local_mask,
            exhausted: false,
        }
    }

    #[inline]
    fn cell_mode(&self, cell: usize) -> DirectedCellMode {
        let start = self.grid.cell_offsets[cell] as usize;
        let end = self.grid.cell_offsets[cell + 1] as usize;
        if start >= end {
            return DirectedCellMode::TransitOnly;
        }

        let packed = self.slot_gen_map[start];
        let (bin_b, _) = CubeMapGrid::unpack_bin_local(packed, self.local_shift, self.local_mask);
        if bin_b != self.query_bin {
            return DirectedCellMode::EmitAll;
        }

        let cell_u32 = cell as u32;
        if cell_u32 < self.start_cell {
            DirectedCellMode::TransitOnly
        } else if cell_u32 == self.start_cell {
            DirectedCellMode::EmitCenterDirected
        } else {
            DirectedCellMode::EmitAll
        }
    }

    #[inline]
    fn push_neighbors(&mut self, cell: usize) {
        for &ncell in self.grid.cell_neighbors(cell) {
            if ncell == u32::MAX {
                continue;
            }
            if !self.scratch.mark_visited(ncell) {
                continue;
            }
            let bound = self.grid.cell_min_dist_sq(self.query, ncell as usize);
            self.scratch.push_cell(ncell, bound);
        }
    }

    #[inline]
    fn push_point_slot_dist(&mut self, slot: u32, dist_sq: f32) {
        self.scratch
            .point_heap
            .push(Reverse((OrdF32::new(dist_sq), slot)));
    }

    fn scan_emit_points(&mut self, cell: usize, mode: DirectedCellMode) {
        if mode == DirectedCellMode::TransitOnly {
            return;
        }

        let start = self.grid.cell_offsets[cell] as usize;
        let end = self.grid.cell_offsets[cell + 1] as usize;
        if start >= end {
            return;
        }

        let xs = &self.grid.cell_points_x[start..end];
        let ys = &self.grid.cell_points_y[start..end];
        let zs = &self.grid.cell_points_z[start..end];
        let indices = &self.grid.point_indices[start..end];
        let (qx, qy, qz) = (self.query.x, self.query.y, self.query.z);

        for i in 0..xs.len() {
            let global = indices[i] as usize;
            if global == self.query_idx {
                continue;
            }
            let slot = (start + i) as u32;

            if mode == DirectedCellMode::EmitCenterDirected {
                let packed = self.slot_gen_map[slot as usize];
                let (bin_b, local_b) =
                    CubeMapGrid::unpack_bin_local(packed, self.local_shift, self.local_mask);
                if bin_b == self.query_bin && local_b < self.query_local {
                    continue;
                }
            }

            let dot = fp::dot3_f32(xs[i], ys[i], zs[i], qx, qy, qz);
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            self.push_point_slot_dist(slot, dist_sq);
        }
    }

    fn expand_cell(&mut self, cell: usize) {
        let mode = self.cell_mode(cell);
        self.scan_emit_points(cell, mode);
        self.push_neighbors(cell);
    }

    fn peel_to_emit_bound_sq(&mut self) -> Option<f32> {
        loop {
            let (bound, cell) = self.scratch.peek_cell()?;
            if self.cell_mode(cell as usize) != DirectedCellMode::TransitOnly {
                return Some(bound);
            }
            let (_, transit_cell) = self.scratch.pop_cell().expect("cell heap out of sync");
            self.expand_cell(transit_cell as usize);
        }
    }

    #[inline]
    fn pop_best_point_slot(&mut self) -> Option<u32> {
        self.scratch.point_heap.pop().map(|Reverse((_, slot))| slot)
    }

    #[inline]
    fn best_point_dist_sq(&self) -> Option<f32> {
        self.scratch
            .point_heap
            .peek()
            .map(|Reverse((dist, _))| dist.get())
    }

    pub(crate) fn pop_next_proven_slot(&mut self) -> Option<u32> {
        if self.exhausted {
            return self.pop_best_point_slot();
        }

        loop {
            let emit_bound = self.peel_to_emit_bound_sq();
            if let Some(best_dist_sq) = self.best_point_dist_sq() {
                let proven = match emit_bound {
                    Some(bound_sq) => best_dist_sq <= bound_sq,
                    None => true,
                };
                if proven {
                    return self.pop_best_point_slot();
                }
            }

            let Some((_, cell)) = self.scratch.pop_cell() else {
                self.exhausted = true;
                self.scratch.exhausted = true;
                return self.pop_best_point_slot();
            };
            self.expand_cell(cell as usize);
        }
    }

    pub(crate) fn unseen_dot_upper_bound(&mut self) -> f32 {
        match self.peel_to_emit_bound_sq() {
            Some(min_dist_sq) => (1.0 - 0.5 * min_dist_sq).clamp(-1.0, 1.0),
            None => -1.0,
        }
    }

    #[inline]
    fn queued_dot_upper_bound(&self) -> f32 {
        match self.best_point_dist_sq() {
            Some(min_dist_sq) => (1.0 - 0.5 * min_dist_sq).clamp(-1.0, 1.0),
            None => -1.0,
        }
    }

    #[inline]
    pub(crate) fn remaining_dot_upper_bound(&mut self) -> f32 {
        self.unseen_dot_upper_bound()
            .max(self.queued_dot_upper_bound())
    }

    #[inline]
    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

impl CubeMapGrid {
    #[inline]
    pub(crate) fn directed_no_k_cursor<'a, 'b>(
        &'a self,
        query: Vec3,
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        ctx: DirectedCtx<'b>,
    ) -> DirectedNoKCursor<'a, 'b> {
        DirectedNoKCursor::new(self, query, query_idx, scratch, ctx)
    }

    #[inline(always)]
    fn unpack_bin_local(packed: u32, local_shift: u32, local_mask: u32) -> (u8, u32) {
        let bin = (packed >> local_shift) as u8;
        let local = packed & local_mask;
        (bin, local)
    }
}

#[cfg(test)]
#[path = "directed_tests.rs"]
mod directed_tests;
