//! Packed-kNN scratch + implementation.

mod emit;
mod helpers;
mod prepare;
#[cfg(test)]
mod tests;

use super::super::CubeMapGrid;
use super::{PackedChunk, PackedGroupInput, PackedKnnTimings, PackedStage};

// Hard cap on total candidates in a 3x3 neighborhood to avoid pathological allocations.
const MAX_CANDIDATES_HARD: usize = 65_536;
// Bound aggregate per-group work and retained key storage, not just the 3x3
// candidate population. Over-budget groups use the resumable shell fallback.
const MAX_AGGREGATE_CANDIDATE_WORK: usize = 1 << 20;

/// Reusable scratch buffers for packed per-cell group queries.
pub(crate) struct PackedKnnCellScratch {
    cell_ranges: Vec<PackedCellRange>,
    next_group_gen: u32,
    chunk0_keys: Vec<Vec<u64>>,
    tail_keys: Vec<Vec<u64>>,
    chunk0_pos: Vec<usize>,
    tail_pos: Vec<usize>,
    tail_possible: Vec<bool>,
    tail_ready_gen: Vec<u32>,
    center_tail_counts: Vec<usize>,
    security_thresholds: Vec<f32>,
    thresholds: Vec<f32>,
    /// Per-query completeness floor of the packed center coverage: the dot
    /// below which the packed stages have NOT certified completeness. Equals
    /// `security_thresholds[qi]` on the normal full-cell scan; raised to the
    /// dense band bound (`1 - r²/2 > security`) for band-pruned dense queries,
    /// where the shell takeover covers everything below it.
    center_bound: Vec<f32>,
    /// Whether query `qi` took the dense band path (center covered only to
    /// `center_bound[qi]`, with the takeover handling the rest).
    band_mode: Vec<bool>,
    /// Reusable scratch for the dense band's gathered candidate slots.
    band_scratch: Vec<u32>,
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
            self.group,
            self.group_gen,
            &mut self.tail_built_any,
            grid,
            timings,
        );
    }

    #[inline]
    pub(super) fn resume_security(&self, qi: usize) -> f32 {
        self.scratch.resume_security(qi)
    }

    #[inline]
    pub(super) fn tail_possible(&self, qi: usize) -> bool {
        self.scratch.tail_possible(qi)
    }

    #[inline]
    pub(super) fn tail_upper_bound(&self, qi: usize) -> f32 {
        self.scratch.tail_upper_bound(qi)
    }

    #[cfg(feature = "timing")]
    pub(crate) fn record_tail_usage(&mut self, timings: &mut PackedKnnTimings) {
        const COMPACT_CAP: usize = 64;
        const COMPACT_BLOCK_SIZE: usize = 16;

        let mut unused_center_keys = 0usize;
        let mut unused_chunk0_keys = 0usize;
        let mut any_high_blocks = Vec::new();
        let mut deferred_blocks = Vec::new();
        for qi in 0..self.group.query_count() {
            if self.scratch.tail_ready_gen.get(qi).copied().unwrap_or(0) != self.group_gen {
                unused_center_keys += self
                    .scratch
                    .center_tail_counts
                    .get(qi)
                    .copied()
                    .unwrap_or(0);
            }
            unused_chunk0_keys += self.scratch.chunk0_keys[qi]
                .len()
                .saturating_sub(self.scratch.chunk0_pos[qi]);

            let high_keys = &mut self.scratch.chunk0_keys[qi];
            let high_count = high_keys.len();
            let emitted = self.scratch.chunk0_pos[qi];
            any_high_blocks.clear();
            deferred_blocks.clear();
            if high_count > COMPACT_CAP {
                // The group is finished, so mutating its dead key list cannot
                // affect frontier behavior. Full-key order is the production
                // descending-dot/ascending-slot order.
                high_keys.sort_unstable();
                any_high_blocks.extend(
                    high_keys
                        .iter()
                        .map(|&key| helpers::key_to_idx(key) as usize / COMPACT_BLOCK_SIZE),
                );
                deferred_blocks.extend(
                    high_keys[COMPACT_CAP..]
                        .iter()
                        .map(|&key| helpers::key_to_idx(key) as usize / COMPACT_BLOCK_SIZE),
                );
                any_high_blocks.sort_unstable();
                any_high_blocks.dedup();
                deferred_blocks.sort_unstable();
                deferred_blocks.dedup();
            }
            let any_high_rescan_slots = eligible_slots_in_blocks(
                &any_high_blocks,
                COMPACT_BLOCK_SIZE,
                self.group,
                qi,
                &self.scratch.cell_ranges,
            );
            let deferred_rescan_slots = eligible_slots_in_blocks(
                &deferred_blocks,
                COMPACT_BLOCK_SIZE,
                self.group,
                qi,
                &self.scratch.cell_ranges,
            );
            let eligible_slots =
                eligible_slots_in_ranges(self.group, qi, &self.scratch.cell_ranges);
            timings.add_compact_overflow_shadow(
                self.scratch.band_mode.get(qi).copied().unwrap_or(false),
                high_count,
                emitted,
                COMPACT_CAP,
                any_high_blocks.len(),
                deferred_blocks.len(),
                any_high_rescan_slots,
                deferred_rescan_slots,
                eligible_slots,
            );
        }
        timings.add_unused_center_tail_keys(unused_center_keys);
        timings.add_unused_chunk0_keys(unused_chunk0_keys);
    }
}

#[cfg(feature = "timing")]
fn eligible_slots_in_ranges(
    group: PackedGroupInput<'_>,
    qi: usize,
    ranges: &[PackedCellRange],
) -> usize {
    let query_slot = group.query_slot_start() as usize + qi;
    ranges
        .iter()
        .enumerate()
        .filter(|(_, range)| range.kind != PackedCellRangeKind::SameBinEarlier)
        .map(|(range_index, range)| {
            let start = if range_index == 0 {
                range.soa_start.max(query_slot + 1)
            } else {
                range.soa_start
            };
            range.soa_end.saturating_sub(start)
        })
        .sum()
}

#[cfg(feature = "timing")]
fn eligible_slots_in_blocks(
    blocks: &[usize],
    block_size: usize,
    group: PackedGroupInput<'_>,
    qi: usize,
    ranges: &[PackedCellRange],
) -> usize {
    let query_slot = group.query_slot_start() as usize + qi;
    let mut covered = 0usize;
    for &block in blocks {
        let block_start = block * block_size;
        let block_end = block_start + block_size;
        for (range_index, range) in ranges.iter().enumerate() {
            if range.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }
            let eligible_start = if range_index == 0 {
                range.soa_start.max(query_slot + 1)
            } else {
                range.soa_start
            };
            let start = eligible_start.max(block_start);
            let end = range.soa_end.min(block_end);
            covered += end.saturating_sub(start);
        }
    }
    covered
}

impl PackedKnnCellScratch {
    pub(crate) fn new() -> Self {
        Self {
            cell_ranges: Vec::with_capacity(9),
            next_group_gen: 1,
            chunk0_keys: Vec::new(),
            tail_keys: Vec::new(),
            chunk0_pos: Vec::new(),
            tail_pos: Vec::new(),
            tail_possible: Vec::new(),
            tail_ready_gen: Vec::new(),
            center_tail_counts: Vec::new(),
            security_thresholds: Vec::new(),
            thresholds: Vec::new(),
            center_bound: Vec::new(),
            band_mode: Vec::new(),
            band_scratch: Vec::new(),
        }
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn security(&self, qi: usize) -> f32 {
        self.security_thresholds[qi]
    }

    #[inline]
    pub(super) fn resume_security(&self, qi: usize) -> f32 {
        if self.band_mode.get(qi).copied().unwrap_or(false) {
            return self.center_bound[qi];
        }
        self.security_thresholds[qi]
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
