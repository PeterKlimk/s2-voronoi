//! Packed-kNN scratch + implementation.

mod cold;
mod emit;
mod helpers;
mod prepare;
#[cfg(test)]
mod tests;

use super::super::CubeMapGrid;
use super::{PackedChunk, PackedGroupInput, PackedKnnTimings, PackedStage};

// Hard cap on total candidates in a 3x3 neighborhood to avoid pathological allocations.
const MAX_CANDIDATES_HARD: usize = 65_536;

/// Reusable scratch buffers for packed per-cell group queries.
pub struct PackedKnnCellScratch {
    cell_ranges: Vec<PackedCellRange>,
    next_group_gen: u32,
    chunk0_keys: Vec<Vec<u64>>,
    tail_keys: Vec<Vec<u64>>,
    chunk0_pos: Vec<usize>,
    tail_pos: Vec<usize>,
    tail_possible: Vec<bool>,
    tail_ready_gen: Vec<u32>,
    security_thresholds: Vec<f32>,
    thresholds: Vec<f32>,
    expand_r2_cells: Vec<PackedCellRange>,
    expand_r2_cells_gen: u32,
    ring3_cells: Vec<u32>,
    ring3_cells_gen: u32,
    expand2_keys: Vec<Vec<u64>>,
    expand2_pos: Vec<usize>,
    expand2_ready_gen: Vec<u32>,
    security2: Vec<f32>,
    security2_ready_gen: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct PackedCellRange {
    soa_start: usize,
    soa_end: usize,
    kind: PackedCellRangeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedCellRangeKind {
    /// Neighbor cell is in a different bin than the queries (no directed filter needed).
    CrossBin,
    /// Neighbor cell is in the same bin and strictly earlier than the group cell
    /// (all locals are < any query local; can skip the whole cell).
    SameBinEarlier,
    /// Neighbor cell is in the same bin and strictly later than the group cell
    /// (no locals are < query local; no directed filter needed).
    SameBinLater,
}

pub(crate) struct PreparedPackedGroup<'a, 'g> {
    scratch: &'a mut PackedKnnCellScratch,
    group: PackedGroupInput<'g>,
    group_gen: u32,
    tail_built_any: bool,
}

pub(crate) enum PreparedPackedGroupStatus<'a, 'g> {
    Ready(PreparedPackedGroup<'a, 'g>),
    SlowPath,
}

impl<'a, 'g> PreparedPackedGroup<'a, 'g> {
    #[inline]
    pub(super) fn group(&self) -> PackedGroupInput<'g> {
        self.group
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn security(&self, qi: usize) -> f32 {
        self.scratch.security(qi)
    }

    #[inline]
    pub(super) fn next_chunk(
        &mut self,
        qi: usize,
        stage: PackedStage,
        k: usize,
        out: &mut [u32],
        timings: &mut PackedKnnTimings,
    ) -> Option<PackedChunk> {
        self.scratch
            .next_chunk(qi, self.group_gen, stage, k, out, timings)
    }

    #[inline]
    pub(super) fn ensure_tail_directed_for(
        &mut self,
        qi: usize,
        grid: &CubeMapGrid,
        timings: &mut PackedKnnTimings,
    ) {
        self.scratch.ensure_tail_directed_for(
            qi,
            self.group.queries(),
            self.group_gen,
            &mut self.tail_built_any,
            grid,
            self.group.slot_gen_map(),
            self.group.local_shift(),
            self.group.local_mask(),
            timings,
        );
    }

    #[inline]
    pub(super) fn ensure_expand_r2_band_directed_for(
        &mut self,
        qi: usize,
        grid: &CubeMapGrid,
        timings: &mut PackedKnnTimings,
    ) -> bool {
        self.scratch.ensure_expand_r2_band_directed_for(
            qi,
            self.group,
            self.group_gen,
            grid,
            self.group.slot_gen_map(),
            self.group.local_shift(),
            self.group.local_mask(),
            timings,
        )
    }

    #[inline]
    pub(super) fn resume_security(&self, qi: usize) -> f32 {
        self.scratch.resume_security(qi, self.group_gen)
    }

    #[inline]
    pub(super) fn tail_possible(&self, qi: usize) -> bool {
        self.scratch.tail_possible(qi)
    }

    #[inline]
    pub(super) fn tail_upper_bound(&self, qi: usize) -> f32 {
        self.scratch.tail_upper_bound(qi)
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn ensure_security2_for(
        &mut self,
        qi: usize,
        grid: &CubeMapGrid,
        timings: &mut PackedKnnTimings,
    ) -> f32 {
        self.scratch
            .ensure_security2_for(qi, self.group, self.group_gen, grid, timings)
    }
}

impl PackedKnnCellScratch {
    pub fn new() -> Self {
        Self {
            cell_ranges: Vec::with_capacity(9),
            next_group_gen: 1,
            chunk0_keys: Vec::new(),
            tail_keys: Vec::new(),
            chunk0_pos: Vec::new(),
            tail_pos: Vec::new(),
            tail_possible: Vec::new(),
            tail_ready_gen: Vec::new(),
            security_thresholds: Vec::new(),
            thresholds: Vec::new(),
            expand_r2_cells: Vec::new(),
            expand_r2_cells_gen: 0,
            ring3_cells: Vec::new(),
            ring3_cells_gen: 0,
            expand2_keys: Vec::new(),
            expand2_pos: Vec::new(),
            expand2_ready_gen: Vec::new(),
            security2: Vec::new(),
            security2_ready_gen: Vec::new(),
        }
    }

    #[inline]
    fn ensure_cold_query_storage(&mut self, num_queries: usize) {
        if self.expand2_keys.len() < num_queries {
            self.expand2_keys.resize_with(num_queries, Vec::new);
        }
        if self.expand2_pos.len() < num_queries {
            self.expand2_pos.resize(num_queries, 0);
        }
        if self.expand2_ready_gen.len() < num_queries {
            self.expand2_ready_gen.resize(num_queries, 0);
        }
        if self.security2.len() < num_queries {
            self.security2.resize(num_queries, -1.0);
        }
        if self.security2_ready_gen.len() < num_queries {
            self.security2_ready_gen.resize(num_queries, 0);
        }
    }

    #[inline]
    #[cfg(test)]
    pub fn security(&self, qi: usize) -> f32 {
        self.security_thresholds[qi]
    }

    #[inline]
    pub(super) fn resume_security(&self, qi: usize, group_gen: u32) -> f32 {
        if self.expand2_ready_gen.get(qi).copied().unwrap_or(0) == group_gen {
            self.security2[qi]
        } else {
            self.security_thresholds[qi]
        }
    }

    #[inline]
    pub(super) fn tail_possible(&self, qi: usize) -> bool {
        self.tail_possible.get(qi).copied().unwrap_or(false)
    }

    #[inline]
    pub(super) fn tail_upper_bound(&self, qi: usize) -> f32 {
        self.thresholds[qi]
    }
}
