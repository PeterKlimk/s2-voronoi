//! Packed-kNN scratch + implementation.

use super::timing::PackedLapTimer;
use super::{DirectedCellGroup, PackedChunk, PackedKnnTimings, PackedStage};

use super::super::{cell_to_face_ij, CubeMapGrid};

use crate::fp;
use crate::policy::{
    PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER, PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER,
    PACKED_HI_BUDGET, PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY,
};
#[cfg(feature = "packed_knn_sort_small")]
use crate::sort::sort_small as sort_small_u64;
use glam::Vec3;
use std::collections::VecDeque;
use std::simd::f32x8;
use std::simd::{cmp::SimdPartialOrd, Mask};

// Hard cap on total candidates in a 3×3 neighborhood to avoid pathological allocations.
const MAX_CANDIDATES_HARD: usize = 65_536;

// Target maximum size for the initial "hi" candidate list (`chunk0_keys`) per query.
//
// `next_chunk` avoids `select_nth_unstable` when `remaining.len() <= 2 * n_target`. With the
// current packed schedule (`chunk0` = 16), keeping the hi list around <= 32 tends to avoid the
// expensive partition path.

/// Reusable scratch buffers for packed per-cell group queries.
pub struct PackedKnnCellScratch {
    cell_ranges: Vec<PackedCellRange>,
    center_lens: Vec<usize>,
    min_center_dot: Vec<f32>,
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
    group: DirectedCellGroup<'g>,
    group_gen: u32,
    tail_built_any: bool,
}

pub(crate) enum PreparedPackedGroupStatus<'a, 'g> {
    Ready(PreparedPackedGroup<'a, 'g>),
    SlowPath,
}

impl<'a, 'g> PreparedPackedGroup<'a, 'g> {
    #[inline]
    pub(super) fn group(&self) -> DirectedCellGroup<'g> {
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
            center_lens: Vec::new(),
            min_center_dot: Vec::new(),
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
    fn classify_cell_range(
        &self,
        grid: &CubeMapGrid,
        cell: usize,
        group: DirectedCellGroup<'_>,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
    ) -> Option<PackedCellRange> {
        let start = grid.cell_offsets[cell] as usize;
        let end = grid.cell_offsets[cell + 1] as usize;
        if start >= end {
            return None;
        }

        let kind = if cell == group.cell() {
            PackedCellRangeKind::CrossBin
        } else {
            let (bin_b, _) = unpack_bin_local(slot_gen_map[start], local_shift, local_mask);
            if bin_b != group.query_bin() {
                PackedCellRangeKind::CrossBin
            } else if cell < group.cell() {
                PackedCellRangeKind::SameBinEarlier
            } else {
                PackedCellRangeKind::SameBinLater
            }
        };

        Some(PackedCellRange {
            soa_start: start,
            soa_end: end,
            kind,
        })
    }

    fn ensure_expand_r2_cells(
        &mut self,
        grid: &CubeMapGrid,
        group: DirectedCellGroup<'_>,
        group_gen: u32,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
    ) {
        if self.expand_r2_cells_gen == group_gen {
            return;
        }

        self.expand_r2_cells.clear();
        self.expand_r2_cells_gen = group_gen;

        let num_cells = grid.cell_offsets().len() - 1;
        let mut seen = vec![false; num_cells];
        let mut push_cell = |this: &mut Self, cell: usize| {
            if seen[cell] {
                return;
            }
            seen[cell] = true;
            if let Some(range) =
                this.classify_cell_range(grid, cell, group, slot_gen_map, local_shift, local_mask)
            {
                this.expand_r2_cells.push(range);
            }
        };

        push_cell(self, group.cell());
        for &ncell in grid.cell_neighbors(group.cell()) {
            if ncell != u32::MAX {
                push_cell(self, ncell as usize);
            }
        }
        for &cell in grid.cell_ring2(group.cell()) {
            push_cell(self, cell as usize);
        }
    }

    fn ensure_ring3_cells(&mut self, grid: &CubeMapGrid, group_cell: usize, group_gen: u32) {
        if self.ring3_cells_gen == group_gen {
            return;
        }

        self.ring3_cells.clear();
        self.ring3_cells_gen = group_gen;

        let num_cells = grid.cell_offsets().len() - 1;
        let mut depth = vec![u8::MAX; num_cells];
        let mut queue = VecDeque::new();
        depth[group_cell] = 0;
        queue.push_back(group_cell);

        while let Some(cell) = queue.pop_front() {
            let d = depth[cell];
            if d == 3 {
                continue;
            }
            for &ncell in grid.cell_neighbors(cell) {
                if ncell == u32::MAX {
                    continue;
                }
                let next = ncell as usize;
                if depth[next] != u8::MAX {
                    continue;
                }
                depth[next] = d + 1;
                queue.push_back(next);
            }
        }

        for (cell, &d) in depth.iter().enumerate() {
            if d == 3 {
                self.ring3_cells.push(cell as u32);
            }
        }
    }

    fn ensure_security2_for(
        &mut self,
        qi: usize,
        group: DirectedCellGroup<'_>,
        group_gen: u32,
        grid: &CubeMapGrid,
        timings: &mut PackedKnnTimings,
    ) -> f32 {
        self.ensure_cold_query_storage(group.queries().len());
        if self.security2_ready_gen[qi] == group_gen {
            return self.security2[qi];
        }

        let mut t = PackedLapTimer::start();
        self.ensure_ring3_cells(grid, group.cell(), group_gen);
        let query_slot = group.queries()[qi] as usize;
        let security = if self.ring3_cells.is_empty() {
            -1.0
        } else {
            outside_max_dot_xyz(
                grid.cell_points_x[query_slot],
                grid.cell_points_y[query_slot],
                grid.cell_points_z[query_slot],
                &self.ring3_cells,
                grid,
            )
        };
        timings.add_ring_thresholds(t.lap());

        self.security2[qi] = security;
        self.security2_ready_gen[qi] = group_gen;
        security
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn ensure_expand_r2_band_directed_for(
        &mut self,
        qi: usize,
        group: DirectedCellGroup<'_>,
        group_gen: u32,
        grid: &CubeMapGrid,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
        timings: &mut PackedKnnTimings,
    ) -> bool {
        self.ensure_cold_query_storage(group.queries().len());
        if self.expand2_ready_gen[qi] == group_gen {
            return true;
        }

        let security2 = self.ensure_security2_for(qi, group, group_gen, grid, timings);
        self.ensure_expand_r2_cells(
            grid,
            group,
            group_gen,
            slot_gen_map,
            local_shift,
            local_mask,
        );

        let query_slot = group.queries()[qi];
        let query_slot_usize = query_slot as usize;
        let qx = grid.cell_points_x[query_slot_usize];
        let qy = grid.cell_points_y[query_slot_usize];
        let qz = grid.cell_points_z[query_slot_usize];
        let security1 = self.security_thresholds[qi];

        let keys = &mut self.expand2_keys[qi];
        keys.clear();
        self.expand2_pos[qi] = 0;
        timings.inc_expand_r2_builds();

        let mut t = PackedLapTimer::start();
        for range in &self.expand_r2_cells {
            if range.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }

            for slot in range.soa_start..range.soa_end {
                let slot_u32 = slot as u32;
                if slot_u32 == query_slot {
                    continue;
                }
                if range.soa_start == self.cell_ranges[0].soa_start
                    && range.soa_end == self.cell_ranges[0].soa_end
                    && slot_u32 < query_slot
                {
                    continue;
                }

                let dot = fp::dot3_f32(
                    grid.cell_points_x[slot],
                    grid.cell_points_y[slot],
                    grid.cell_points_z[slot],
                    qx,
                    qy,
                    qz,
                );
                if dot > security2 && dot <= security1 {
                    keys.push(make_desc_key(dot, slot_u32));
                    if keys.len() > PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY {
                        keys.clear();
                        timings.inc_expand_r2_cap_skips();
                        return false;
                    }
                }
            }
        }
        timings.add_expand_r2_scan(t.lap());

        self.expand2_ready_gen[qi] = group_gen;
        true
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(crate) fn prepare_group_directed<'a, 'g>(
        &'a mut self,
        grid: &CubeMapGrid,
        group: DirectedCellGroup<'g>,
        timings: &mut PackedKnnTimings,
    ) -> PreparedPackedGroupStatus<'a, 'g> {
        timings.clear();

        group.debug_assert_matches_grid(grid);

        let cell = group.cell();
        let queries = group.queries();
        let query_bin = group.query_bin();
        let slot_gen_map = group.slot_gen_map();
        let local_shift = group.local_shift();
        let local_mask = group.local_mask();
        let num_queries = queries.len();
        let mut group_gen = self.next_group_gen.wrapping_add(1).max(1);
        if group_gen == u32::MAX {
            // Reserve `0` as "never set"; on wrap, clear all generation stamps.
            group_gen = 1;
            self.tail_ready_gen.fill(0);
            self.expand_r2_cells_gen = 0;
            self.ring3_cells_gen = 0;
            self.expand2_ready_gen.fill(0);
            self.security2_ready_gen.fill(0);
        }
        self.next_group_gen = group_gen;

        let num_cells = 6 * grid.res * grid.res;
        if cell >= num_cells {
            return PreparedPackedGroupStatus::Ready(PreparedPackedGroup {
                scratch: self,
                group,
                group_gen,
                tail_built_any: false,
            });
        }

        let mut t = PackedLapTimer::start();
        self.cell_ranges.clear();

        let q_start = grid.cell_offsets[cell] as usize;
        let q_end = grid.cell_offsets[cell + 1] as usize;
        self.cell_ranges.push(PackedCellRange {
            soa_start: q_start,
            soa_end: q_end,
            kind: PackedCellRangeKind::CrossBin,
        });

        for &ncell in grid.cell_neighbors(cell) {
            if ncell == u32::MAX || ncell == cell as u32 {
                continue;
            }
            let nc = ncell as usize;
            let n_start = grid.cell_offsets[nc] as usize;
            let n_end = grid.cell_offsets[nc + 1] as usize;
            if n_start < n_end {
                // Bin/local ids are properties of the whole cell: all points in a grid cell share
                // the same bin, and within a bin, locals are assigned in increasing cell order.
                //
                // This allows:
                // - skipping same-bin neighbor cells strictly earlier than the center cell
                // - avoiding per-point (bin,local) decoding for all other neighbor cells
                let (bin_b, _) = unpack_bin_local(slot_gen_map[n_start], local_shift, local_mask);
                let kind = if bin_b != query_bin {
                    PackedCellRangeKind::CrossBin
                } else if ncell < cell as u32 {
                    PackedCellRangeKind::SameBinEarlier
                } else {
                    PackedCellRangeKind::SameBinLater
                };

                self.cell_ranges.push(PackedCellRange {
                    soa_start: n_start,
                    soa_end: n_end,
                    kind,
                });
            }
        }

        let mut num_candidates = 0usize;
        for r in &self.cell_ranges {
            // If this neighbor cell is earlier-local in the same bin, we never consider it for
            // directed kNN (the earlier side already sent adjacency via edge checks).
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }
            num_candidates += r.soa_end - r.soa_start;
        }
        if num_candidates > MAX_CANDIDATES_HARD {
            timings.add_setup(t.lap());
            return PreparedPackedGroupStatus::SlowPath;
        }
        let mut ring_candidates_eligible = 0usize;
        let mut ring_candidates_all = 0usize;
        for r in &self.cell_ranges[1..] {
            ring_candidates_all += r.soa_end - r.soa_start;
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }
            ring_candidates_eligible += r.soa_end - r.soa_start;
        }
        timings.add_setup(t.lap());

        let ring2 = grid.cell_ring2(cell);
        let interior_planes = security_planes_3x3_interior(cell, grid);

        // live_dedup invariant: groups are complete center-cell runs in slot order.
        // Query coordinates are already stored in SoA order by slot; borrow the center-cell
        // slices directly instead of copying into scratch.
        let qx_src = &grid.cell_points_x[q_start..q_end];
        let qy_src = &grid.cell_points_y[q_start..q_end];
        let qz_src = &grid.cell_points_z[q_start..q_end];
        timings.add_query_cache(t.lap());

        self.security_thresholds.clear();
        self.security_thresholds.reserve(num_queries);
        match interior_planes {
            Some(planes) => {
                for qi in 0..num_queries {
                    let qx = qx_src[qi];
                    let qy = qy_src[qi];
                    let qz = qz_src[qi];

                    let mut s_min = 1.0f32;
                    for n in &planes {
                        s_min = s_min.min(fp::dot3_f32(n.x, n.y, n.z, qx, qy, qz));
                    }

                    // For the interior single-face case, the nearest outside point is reached
                    // by crossing the closest boundary great circle. If we ever see a non-positive
                    // signed distance (numerical issues), fall back to the existing cap bound.
                    let security = if s_min > 0.0 && s_min.is_finite() {
                        const PAD: f32 = 1e-6;
                        let s = (s_min - PAD).clamp(0.0, 1.0);
                        (1.0 - s * s).max(0.0).sqrt()
                    } else {
                        outside_max_dot_xyz(qx, qy, qz, ring2, grid)
                    };
                    self.security_thresholds.push(security);
                }
            }
            None => {
                self.security_thresholds.extend(
                    qx_src
                        .iter()
                        .zip(qy_src.iter())
                        .zip(qz_src.iter())
                        .map(|((&x, &y), &z)| outside_max_dot_xyz(x, y, z, ring2, grid)),
                );
            }
        }
        timings.add_security_thresholds(t.lap());

        self.min_center_dot.resize(num_queries, f32::INFINITY);
        self.min_center_dot.fill(f32::INFINITY);
        self.center_lens.resize(num_queries, 0);
        self.thresholds.resize(num_queries, 0.0);

        // Don't shrink `Vec<Vec<_>>` to avoid dropping inner buffers when group sizes vary.
        if self.chunk0_keys.len() < num_queries {
            self.chunk0_keys.resize_with(num_queries, Vec::new);
        }
        for v in &mut self.chunk0_keys[..num_queries] {
            v.clear();
        }
        self.chunk0_pos.resize(num_queries, 0);
        self.chunk0_pos.fill(0);

        // Same rationale as `chunk0_keys` above.
        if self.tail_keys.len() < num_queries {
            self.tail_keys.resize_with(num_queries, Vec::new);
        }
        for v in &mut self.tail_keys[..num_queries] {
            v.clear();
        }
        self.tail_pos.resize(num_queries, 0);
        self.tail_pos.fill(0);
        self.tail_possible.resize(num_queries, false);
        if self.tail_ready_gen.len() < num_queries {
            self.tail_ready_gen.resize(num_queries, 0);
        }
        timings.add_select_prep(t.lap());

        // === Center cell pass (directed triangular).
        let PackedCellRange {
            soa_start: center_soa_start,
            soa_end: center_soa_end,
            ..
        } = self.cell_ranges[0];
        let center_len = center_soa_end - center_soa_start;
        let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
        let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
        let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

        // Center pass is immediately after selection prep; continue lapping.
        // live_dedup invariant: groups are complete center-cell runs in slot order.
        debug_assert_eq!(num_queries, center_len, "center-cell query length mismatch");

        // Directed center cell: since all points in a grid cell are in the same bin, the within-bin
        // filter reduces to "skip earlier slots in this same cell".
        let query_x = qx_src;
        let query_y = qy_src;
        let query_z = qz_src;
        let security_thresholds = &self.security_thresholds[..num_queries];
        let chunk0_keys = &mut self.chunk0_keys[..num_queries];

        let full_chunks = center_len / 8;
        for chunk in 0..full_chunks {
            let i = chunk * 8;
            let cx = f32x8::from_slice(&xs[i..]);
            let cy = f32x8::from_slice(&ys[i..]);
            let cz = f32x8::from_slice(&zs[i..]);

            // Candidate positions in this chunk are [i, i+7]. A query at position qi only
            // needs to consider this chunk if qi <= i+7.
            let qi_end = (i + 8).min(num_queries);
            for qi in 0..qi_end {
                let qx = f32x8::splat(query_x[qi]);
                let qy = f32x8::splat(query_y[qi]);
                let qz = f32x8::splat(query_z[qi]);
                let dots = fp::dot3_f32x8(cx, cy, cz, qx, qy, qz);

                let thresh_vec = f32x8::splat(security_thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;
                if mask_bits == 0 {
                    continue;
                }

                // Directed intra-bin filter for center cell:
                // allowed candidates are those with position >= qi, excluding self.
                if qi >= i {
                    let rel = qi - i;
                    debug_assert!(rel < 8);
                    if rel > 0 {
                        mask_bits &= !((1u32 << rel) - 1);
                    }
                    mask_bits &= !(1u32 << rel);
                    if mask_bits == 0 {
                        continue;
                    }
                }

                let dots_arr: [f32; 8] = dots.into();
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let slot = (center_soa_start + i + lane) as u32;
                    let dot = dots_arr[lane];
                    chunk0_keys[qi].push(make_desc_key(dot, slot));
                    self.min_center_dot[qi] = self.min_center_dot[qi].min(dot);
                    mask_bits &= mask_bits - 1;
                }
            }
        }

        let tail_start = full_chunks * 8;
        for pos in tail_start..center_len {
            let cx = xs[pos];
            let cy = ys[pos];
            let cz = zs[pos];
            let slot = (center_soa_start + pos) as u32;

            // Candidate position is `pos`. A query at position qi can only see this candidate
            // if qi <= pos, excluding qi == pos (self).
            let qi_end = (pos + 1).min(num_queries);
            for qi in 0..qi_end {
                if qi == pos {
                    continue;
                }
                let dot = fp::dot3_f32(cx, cy, cz, query_x[qi], query_y[qi], query_z[qi]);
                if dot > security_thresholds[qi] {
                    chunk0_keys[qi].push(make_desc_key(dot, slot));
                    self.min_center_dot[qi] = self.min_center_dot[qi].min(dot);
                }
            }
        }
        timings.add_center_pass(t.lap());

        for (qi, v) in chunk0_keys.iter().enumerate() {
            self.center_lens[qi] = v.len();
        }

        // === Threshold selection.
        //
        // The old threshold is derived from the worst (minimum) center dot, ensuring that all
        // safe center candidates remain "hi" and that we can't miss any ring candidate that would
        // outrank a kept center candidate.
        //
        // To reduce the cost of large ring candidate sets (and especially `select_nth_unstable`),
        // we allow tightening the hi threshold above the worst center dot. Any safe center
        // candidates below the tightened threshold get demoted into the tail set, preserving the
        // ordering/correctness invariant: if we keep a candidate with dot d in "hi", we must not
        // miss any other safe candidate with dot > d.
        //
        // We choose a heuristic tightened threshold based on counts: treat the eligible
        // neighborhood size as `ring_candidates + (num_queries - qi - 1)` (directed center cell),
        // and pick a dot threshold t in [security, 1] that targets ~PACKED_HI_BUDGET candidates above t
        // under a simple "uniform on [security, 1]" model. We never loosen below the old
        // worst-center threshold.
        let ring_candidates_est = if PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER {
            ring_candidates_all
        } else {
            ring_candidates_eligible
        };

        for qi in 0..num_queries {
            let security = security_thresholds[qi];
            let center_len = self.center_lens[qi];
            let min_dot = self.min_center_dot[qi];
            let _old_t = if center_len > 0 {
                security.max(min_dot - 1e-6)
            } else {
                security
            };

            let center_eligible = if PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER {
                num_queries.saturating_sub(1)
            } else {
                num_queries.saturating_sub(qi + 1)
            };
            let n_total = ring_candidates_est + center_eligible;
            let t_count = if n_total == 0 {
                security
            } else {
                let ratio = ((PACKED_HI_BUDGET as f32) / (n_total as f32)).min(1.0);
                let t = 1.0 - (1.0 - security) * ratio;
                t.clamp(security, 1.0)
            };

            self.thresholds[qi] = t_count;
            self.tail_possible[qi] = t_count > security;
        }

        // Demote center candidates at/below the tightened threshold into tail.
        //
        // This ensures that any candidate remaining in chunk0 ("hi") has dot > thresholds[qi],
        // so the ring pass (which uses dot > thresholds[qi]) cannot miss a ring candidate that
        // outranks a kept center candidate.
        for qi in 0..num_queries {
            let t = self.thresholds[qi];
            let v = &mut chunk0_keys[qi];
            if v.is_empty() {
                continue;
            }

            let tail_v = &mut self.tail_keys[qi];
            let mut write = 0usize;
            let len = v.len();
            for idx in 0..len {
                let key = v[idx];
                let dot = key_to_dot(key);
                if dot > t {
                    v[write] = key;
                    write += 1;
                } else {
                    tail_v.push(key);
                }
            }
            v.truncate(write);

            // Tail may be needed either due to a tightened threshold or due to demoted center
            // candidates. Keep the flag conservative.
            if !tail_v.is_empty() {
                self.tail_possible[qi] = true;
            }
        }
        timings.add_ring_thresholds(t.lap());

        // === Ring pass: collect "hi" candidates into chunk0.
        let thresholds = &self.thresholds[..num_queries];
        for r in &self.cell_ranges[1..] {
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }

            let soa_start = r.soa_start;
            let soa_end = r.soa_end;
            let range_len = soa_end - soa_start;
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];

            let full_chunks = range_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let cx = f32x8::from_slice(&xs[i..]);
                let cy = f32x8::from_slice(&ys[i..]);
                let cz = f32x8::from_slice(&zs[i..]);

                for (qi, &query_slot) in queries.iter().enumerate() {
                    let qx = f32x8::splat(query_x[qi]);
                    let qy = f32x8::splat(query_y[qi]);
                    let qz = f32x8::splat(query_z[qi]);
                    let dots = fp::dot3_f32x8(cx, cy, cz, qx, qy, qz);

                    let thresh_vec = f32x8::splat(thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr: [f32; 8] = dots.into();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        debug_assert_ne!(
                            slot, query_slot,
                            "ring pass should never revisit the query slot"
                        );
                        let dot = dots_arr[lane];
                        chunk0_keys[qi].push(make_desc_key(dot, slot));
                        mask_bits &= mask_bits - 1;
                    }
                }
            }

            let rem = range_len % 8;
            if rem != 0 {
                let i = full_chunks * 8;
                debug_assert!(i < range_len);
                let valid_bits = (1u32 << (rem as u32)) - 1;

                let mut xbuf = [0.0f32; 8];
                let mut ybuf = [0.0f32; 8];
                let mut zbuf = [0.0f32; 8];
                xbuf[..rem].copy_from_slice(&xs[i..]);
                ybuf[..rem].copy_from_slice(&ys[i..]);
                zbuf[..rem].copy_from_slice(&zs[i..]);

                let cx = f32x8::from_array(xbuf);
                let cy = f32x8::from_array(ybuf);
                let cz = f32x8::from_array(zbuf);

                for (qi, &query_slot) in queries.iter().enumerate() {
                    let qx = f32x8::splat(query_x[qi]);
                    let qy = f32x8::splat(query_y[qi]);
                    let qz = f32x8::splat(query_z[qi]);
                    let dots = fp::dot3_f32x8(cx, cy, cz, qx, qy, qz);

                    let thresh_vec = f32x8::splat(thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = (mask.to_bitmask() as u32) & valid_bits;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr: [f32; 8] = dots.into();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        debug_assert_ne!(
                            slot, query_slot,
                            "ring pass should never revisit the query slot"
                        );
                        let dot = dots_arr[lane];
                        chunk0_keys[qi].push(make_desc_key(dot, slot));
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }
        timings.add_ring_pass(t.lap());

        PreparedPackedGroupStatus::Ready(PreparedPackedGroup {
            scratch: self,
            group,
            group_gen,
            tail_built_any: false,
        })
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn ensure_tail_directed_for(
        &mut self,
        qi: usize,
        group_queries: &[u32],
        group_gen: u32,
        tail_built_any: &mut bool,
        grid: &CubeMapGrid,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
        timings: &mut PackedKnnTimings,
    ) {
        // Tail candidates have already been partitioned into (hi, tail, unsafe) buckets
        // during `prepare_group_directed`. Bin/local decoding is only needed there.
        let _ = (slot_gen_map, local_shift, local_mask);

        let Some(gen) = self.tail_ready_gen.get(qi).copied() else {
            return;
        };
        if gen == group_gen {
            return;
        }
        if !*tail_built_any {
            *tail_built_any = true;
            timings.inc_tail_builds();
        }
        self.tail_ready_gen[qi] = group_gen;

        // Keep any precomputed center-tail candidates already stored in `tail_keys[qi]` and
        // append ring-tail candidates here.
        self.tail_pos[qi] = 0;
        debug_assert!(self.tail_possible.get(qi).copied().unwrap_or(false));

        let mut t_tail = PackedLapTimer::start();
        for r in &self.cell_ranges[1..] {
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }

            let soa_start = r.soa_start;
            let soa_end = r.soa_end;
            let range_len = soa_end - soa_start;
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];

            let query_slot = group_queries[qi];
            let query_slot_usize = query_slot as usize;
            let qx_s = grid.cell_points_x[query_slot_usize];
            let qy_s = grid.cell_points_y[query_slot_usize];
            let qz_s = grid.cell_points_z[query_slot_usize];

            let full_chunks = range_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let cx = f32x8::from_slice(&xs[i..]);
                let cy = f32x8::from_slice(&ys[i..]);
                let cz = f32x8::from_slice(&zs[i..]);

                let qx = f32x8::splat(qx_s);
                let qy = f32x8::splat(qy_s);
                let qz = f32x8::splat(qz_s);
                let dots = fp::dot3_f32x8(cx, cy, cz, qx, qy, qz);

                let hi_vec = f32x8::splat(self.thresholds[qi]);
                let safe_vec = f32x8::splat(self.security_thresholds[qi]);

                let safe_mask: Mask<i32, 8> = dots.simd_gt(safe_vec);
                let hi_mask: Mask<i32, 8> = dots.simd_gt(hi_vec);

                let mut tail_bits =
                    (safe_mask.to_bitmask() as u32) & !(hi_mask.to_bitmask() as u32);
                if tail_bits == 0 {
                    continue;
                }

                let dots_arr: [f32; 8] = dots.into();
                while tail_bits != 0 {
                    let lane = tail_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    if slot != query_slot {
                        let dot = dots_arr[lane];
                        self.tail_keys[qi].push(make_desc_key(dot, slot));
                    }
                    tail_bits &= tail_bits - 1;
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                let slot = (soa_start + i) as u32;

                if slot == query_slot {
                    continue;
                }

                let dot = fp::dot3_f32(cx, cy, cz, qx_s, qy_s, qz_s);
                if dot > self.security_thresholds[qi] && dot <= self.thresholds[qi] {
                    self.tail_keys[qi].push(make_desc_key(dot, slot));
                }
            }
        }
        timings.add_ring_fallback(t_tail.lap());

        if self.tail_keys[qi].is_empty() {
            self.tail_possible[qi] = false;
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn next_chunk(
        &mut self,
        qi: usize,
        group_gen: u32,
        stage: PackedStage,
        k: usize,
        out: &mut [u32],
        timings: &mut PackedKnnTimings,
    ) -> Option<PackedChunk> {
        if k == 0 || out.is_empty() {
            return None;
        }

        match stage {
            PackedStage::Chunk0 => {
                let mut t = PackedLapTimer::start();
                let keys = &mut self.chunk0_keys.get_mut(qi)?;
                let start = *self.chunk0_pos.get(qi)?;
                if start >= keys.len() {
                    return None;
                }
                let remaining = &mut keys[start..];
                let n_target = k.min(out.len());
                if remaining.len() == 0 {
                    return None;
                }
                timings.add_select_query_prep(t.lap());

                // If remaining candidates are close to the requested emission size,
                // skip partition and just sort the remainder.
                //
                // Important: this must scale with `n_target` (k can shrink after the first
                // packed chunk), otherwise we end up sorting large remainders when asking
                // for small k (e.g. k=8).
                if remaining.len() <= 2 * n_target {
                    let emit = remaining.len().min(n_target);
                    let sort_len = remaining.len();
                    sort_keys_u64(remaining);
                    timings.add_select_sort_sized(t.lap(), sort_len);
                    for (dst, key) in out[..emit].iter_mut().zip(remaining.iter()) {
                        *dst = key_to_idx(*key);
                    }
                    timings.add_select_scatter(t.lap());
                    let last_dot = key_to_dot(remaining[emit - 1]);
                    self.chunk0_pos[qi] = start + emit;
                    let has_more = self.chunk0_pos[qi] < keys.len();
                    let unseen_bound = if has_more {
                        last_dot
                    } else if self.tail_possible[qi] {
                        self.thresholds[qi]
                    } else {
                        self.security_thresholds[qi]
                    };
                    return Some(PackedChunk {
                        n: emit,
                        unseen_bound,
                    });
                }

                // Large chunk: partition to extract top n, then sort those.
                let n = n_target.min(remaining.len());
                if remaining.len() > n {
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t.lap());
                }
                sort_keys_u64(&mut remaining[..n]);
                timings.add_select_sort_sized(t.lap(), n);
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t.lap());
                let last_dot = key_to_dot(remaining[n - 1]);
                self.chunk0_pos[qi] = start + n;
                let has_more = self.chunk0_pos[qi] < keys.len();
                let unseen_bound = if has_more {
                    last_dot
                } else if self.tail_possible[qi] {
                    self.thresholds[qi]
                } else {
                    self.security_thresholds[qi]
                };
                Some(PackedChunk { n, unseen_bound })
            }
            PackedStage::Tail => {
                debug_assert!(
                    self.tail_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "tail stage requested before ensure_tail"
                );
                let mut t = PackedLapTimer::start();
                let keys = &mut self.tail_keys.get_mut(qi)?;
                let start = *self.tail_pos.get(qi)?;
                if start >= keys.len() {
                    return None;
                }
                let remaining = &mut keys[start..];
                let n = k.min(out.len()).min(remaining.len());
                if n == 0 {
                    return None;
                }
                timings.add_select_query_prep(t.lap());
                if remaining.len() > n {
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t.lap());
                }
                sort_keys_u64(&mut remaining[..n]);
                timings.add_select_sort_sized(t.lap(), n);
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t.lap());
                let last_dot = key_to_dot(remaining[n - 1]);
                self.tail_pos[qi] = start + n;
                let has_more = self.tail_pos[qi] < keys.len();
                let unseen_bound = if has_more {
                    last_dot
                } else {
                    self.security_thresholds[qi]
                };
                Some(PackedChunk { n, unseen_bound })
            }
            PackedStage::ExpandR2 => {
                debug_assert!(
                    self.expand2_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "expand_r2 stage requested before ensure_expand_r2"
                );
                let mut t_stage = PackedLapTimer::start();
                let mut t = PackedLapTimer::start();
                let keys = &mut self.expand2_keys.get_mut(qi)?;
                let start = *self.expand2_pos.get(qi)?;
                if start >= keys.len() {
                    return None;
                }
                let remaining = &mut keys[start..];
                let n = k.min(out.len()).min(remaining.len());
                if n == 0 {
                    return None;
                }
                timings.add_select_query_prep(t.lap());
                if remaining.len() > n {
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t.lap());
                }
                sort_keys_u64(&mut remaining[..n]);
                timings.add_select_sort_sized(t.lap(), n);
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t.lap());
                let last_dot = key_to_dot(remaining[n - 1]);
                self.expand2_pos[qi] = start + n;
                let has_more = self.expand2_pos[qi] < keys.len();
                let unseen_bound = if has_more {
                    last_dot
                } else {
                    self.security2[qi]
                };
                timings.add_expand_r2_select(t_stage.lap());
                Some(PackedChunk { n, unseen_bound })
            }
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

#[inline(always)]
fn unpack_bin_local(packed: u32, local_shift: u32, local_mask: u32) -> (u8, u32) {
    let bin = (packed >> local_shift) as u8;
    let local = packed & local_mask;
    (bin, local)
}

#[inline(always)]
fn f32_to_ordered_u32(val: f32) -> u32 {
    let b = val.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    }
}

#[inline(always)]
fn make_desc_key(dot: f32, idx: u32) -> u64 {
    // Sorting networks use `u64::MAX` as a padding sentinel (via `sort_small`). For finite floats,
    // `f32_to_ordered_u32` is never 0, so `desc` is never all-ones and the full key can never be
    // `u64::MAX` (even if `idx == u32::MAX`).
    debug_assert!(dot.is_finite());
    // Bigger dot = smaller key, so ascending sort gives descending dot.
    let ord = f32_to_ordered_u32(dot);
    let desc = !ord;
    ((desc as u64) << 32) | (idx as u64)
}

#[inline(always)]
fn sort_keys_u64(keys: &mut [u64]) {
    #[cfg(feature = "packed_knn_sort_small")]
    {
        if keys.len() <= 35 {
            sort_small_u64(keys);
            return;
        }
    }
    keys.sort_unstable();
}

#[inline(always)]
fn key_to_idx(key: u64) -> u32 {
    (key & 0xFFFF_FFFF) as u32
}

#[inline(always)]
fn ordered_u32_to_f32(val: u32) -> f32 {
    let b = if val & 0x8000_0000 != 0 {
        val ^ 0x8000_0000
    } else {
        !val
    };
    f32::from_bits(b)
}

#[inline(always)]
fn key_to_dot(key: u64) -> f32 {
    let desc = (key >> 32) as u32;
    let ord = !desc;
    ordered_u32_to_f32(ord)
}

#[inline]
fn max_dot_to_cap_xyz(qx: f32, qy: f32, qz: f32, center: Vec3, cos_r: f32, sin_r: f32) -> f32 {
    let cos_d = (qx * center.x + qy * center.y + qz * center.z).clamp(-1.0, 1.0);
    if cos_d > cos_r {
        return 1.0;
    }

    let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
    (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0)
}

#[inline]
fn security_planes_3x3_interior(cell: usize, grid: &CubeMapGrid) -> Option<[Vec3; 4]> {
    let res = grid.res;
    if res < 3 {
        return None;
    }

    // 3×3 neighborhood stays on a single face iff the center cell is not on the face boundary.
    let (face, iu, iv) = cell_to_face_ij(cell, res);
    if iu < 1 || iv < 1 || iu + 1 >= res || iv + 1 >= res {
        return None;
    }

    // Outer boundaries for the 3×3 envelope: lines at (iu-1, iu+2) and (iv-1, iv+2).
    let mut planes = [
        grid.face_u_line_plane(face, iu - 1),
        grid.face_u_line_plane(face, iu + 2),
        grid.face_v_line_plane(face, iv - 1),
        grid.face_v_line_plane(face, iv + 2),
    ];

    // Orient all planes so that the interior (containing the cell center) has `n·p >= 0`.
    let center = grid.cell_centers[cell];
    for n in &mut planes {
        if n.dot(center) < 0.0 {
            *n = -*n;
        }
    }

    Some(planes)
}

#[inline]
fn outside_max_dot_xyz(qx: f32, qy: f32, qz: f32, ring2: &[u32], grid: &CubeMapGrid) -> f32 {
    debug_assert!(!ring2.is_empty(), "ring2 must be non-empty");
    let mut max_dot = f32::NEG_INFINITY;
    for &cell in ring2 {
        let idx = cell as usize;
        let center = grid.cell_centers[idx];
        let cos_r = grid.cell_cos_radius[idx];
        let sin_r = grid.cell_sin_radius[idx];
        let dot = max_dot_to_cap_xyz(qx, qy, qz, center, cos_r, sin_r);
        if dot > max_dot {
            max_dot = dot;
            if max_dot >= 1.0 {
                break;
            }
        }
    }
    max_dot
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const QUERY_BIN: u8 = 0;
    const LOCAL_SHIFT: u32 = 24;
    const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;

    fn random_unit_points(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n);
        while points.len() < n {
            let p = Vec3::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let len_sq = p.length_squared();
            if len_sq <= 1e-12 {
                continue;
            }
            points.push(p / len_sq.sqrt());
        }
        points
    }

    fn fullest_cell(grid: &CubeMapGrid) -> usize {
        let mut best_cell = 0usize;
        let mut best_len = 0usize;
        for cell in 0..(grid.cell_offsets().len() - 1) {
            let len = grid.cell_points(cell).len();
            if len > best_len {
                best_len = len;
                best_cell = cell;
            }
        }
        assert!(best_len > 0, "expected at least one non-empty cell");
        best_cell
    }

    fn expected_safe_slots(
        grid: &CubeMapGrid,
        points: &[Vec3],
        query_idx: usize,
        query_local: u32,
        security: f32,
    ) -> Vec<u32> {
        let query = points[query_idx];
        let mut candidates = Vec::new();
        for (slot, &neighbor_idx_u32) in grid.point_indices().iter().enumerate() {
            let neighbor_idx = neighbor_idx_u32 as usize;
            if neighbor_idx == query_idx {
                continue;
            }
            if (slot as u32) < query_local {
                continue;
            }
            let dot = query.dot(points[neighbor_idx]);
            if dot > security {
                candidates.push((dot, slot as u32));
            }
        }
        candidates.sort_unstable_by(|&(da, sa), &(db, sb)| db.total_cmp(&da).then(sa.cmp(&sb)));
        candidates.into_iter().map(|(_, slot)| slot).collect()
    }

    fn cell_neighbor_depths(grid: &CubeMapGrid, start_cell: usize, max_depth: u8) -> Vec<u8> {
        let num_cells = grid.cell_offsets().len() - 1;
        let mut depth = vec![u8::MAX; num_cells];
        let mut queue = VecDeque::new();
        depth[start_cell] = 0;
        queue.push_back(start_cell);

        while let Some(cell) = queue.pop_front() {
            let d = depth[cell];
            if d == max_depth {
                continue;
            }
            for &ncell in grid.cell_neighbors(cell) {
                if ncell == u32::MAX {
                    continue;
                }
                let next = ncell as usize;
                if depth[next] != u8::MAX {
                    continue;
                }
                depth[next] = d + 1;
                queue.push_back(next);
            }
        }

        depth
    }

    #[test]
    fn packed_chunks_match_safe_bruteforce_order_and_bounds() {
        const N: usize = 384;
        const RES: usize = 10;
        const EPS: f32 = 1e-5;

        for &seed in &[11u64, 37] {
            let points = random_unit_points(N, seed);
            let grid = CubeMapGrid::new(&points, RES);
            let cell = fullest_cell(&grid);
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
            let query_locals = queries.clone();
            let mut slot_gen_map = vec![0u32; points.len()];
            for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
                *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
            }

            let group = DirectedCellGroup::new(
                cell,
                QUERY_BIN,
                &queries,
                &query_locals,
                &slot_gen_map,
                LOCAL_SHIFT,
                LOCAL_MASK,
            );
            let mut scratch = PackedKnnCellScratch::new();
            let mut timings = PackedKnnTimings::default();
            let PreparedPackedGroupStatus::Ready(mut prepared) =
                scratch.prepare_group_directed(&grid, group, &mut timings)
            else {
                panic!("packed prepare unexpectedly fell back to slow path");
            };

            for qi in 0..queries.len() {
                let query_slot = queries[qi];
                let query_idx = grid.point_indices()[query_slot as usize] as usize;
                let security = prepared.security(qi);
                let expected =
                    expected_safe_slots(&grid, &points, query_idx, query_locals[qi], security);
                let mut emitted = Vec::new();
                let mut prev_bound = 1.0f32;
                let mut stage = PackedStage::Chunk0;

                loop {
                    let k = match stage {
                        PackedStage::Chunk0 => 16,
                        PackedStage::Tail => 8,
                        PackedStage::ExpandR2 => 8,
                    };
                    let mut out = vec![u32::MAX; k];
                    let chunk = prepared.next_chunk(qi, stage, k, &mut out, &mut timings);
                    match chunk {
                        Some(chunk) => {
                            assert!(
                                chunk.unseen_bound <= prev_bound + EPS,
                                "unseen bound increased for seed={seed}, qi={qi}"
                            );
                            prev_bound = chunk.unseen_bound;
                            emitted.extend_from_slice(&out[..chunk.n]);

                            if let Some(&next_slot) = expected.get(emitted.len()) {
                                let next_idx = grid.point_indices()[next_slot as usize] as usize;
                                let next_dot = points[query_idx].dot(points[next_idx]);
                                assert!(
                                    next_dot <= chunk.unseen_bound + EPS,
                                    "unseen bound was not conservative for seed={seed}, qi={qi}"
                                );
                            }
                        }
                        None if stage == PackedStage::Chunk0 && prepared.tail_possible(qi) => {
                            prepared.ensure_tail_directed_for(qi, &grid, &mut timings);
                            stage = PackedStage::Tail;
                        }
                        None => break,
                    }
                }

                assert_eq!(
                    emitted, expected,
                    "safe packed order mismatch for seed={seed}, qi={qi}"
                );
            }
        }
    }

    #[test]
    fn expand_r2_security_is_conservative() {
        const N: usize = 384;
        const RES: usize = 10;
        const EPS: f32 = 1e-5;

        for &seed in &[13u64, 41, 97] {
            let points = random_unit_points(N, seed);
            let grid = CubeMapGrid::new(&points, RES);
            let cell = fullest_cell(&grid);
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
            let query_locals = queries.clone();
            let mut slot_gen_map = vec![0u32; points.len()];
            for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
                *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
            }

            let group = DirectedCellGroup::new(
                cell,
                QUERY_BIN,
                &queries,
                &query_locals,
                &slot_gen_map,
                LOCAL_SHIFT,
                LOCAL_MASK,
            );
            let mut scratch = PackedKnnCellScratch::new();
            let mut timings = PackedKnnTimings::default();
            let PreparedPackedGroupStatus::Ready(mut prepared) =
                scratch.prepare_group_directed(&grid, group, &mut timings)
            else {
                panic!("packed prepare unexpectedly fell back to slow path");
            };

            let depth = cell_neighbor_depths(&grid, cell, 3);
            for qi in 0..queries.len() {
                let query_slot = queries[qi] as usize;
                let query_idx = grid.point_indices()[query_slot] as usize;
                let security2 = prepared.ensure_security2_for(qi, &grid, &mut timings);

                let mut brute_max = f32::NEG_INFINITY;
                for &neighbor_idx_u32 in grid.point_indices() {
                    let neighbor_idx = neighbor_idx_u32 as usize;
                    let neighbor_cell = grid.point_index_to_cell(neighbor_idx);
                    if depth[neighbor_cell] <= 2 {
                        continue;
                    }
                    let dot = points[query_idx].dot(points[neighbor_idx]);
                    brute_max = brute_max.max(dot);
                }

                assert!(
                    brute_max <= security2 + EPS,
                    "security2 not conservative for seed={seed}, qi={qi}: brute_max={brute_max}, security2={security2}"
                );
            }
        }
    }

    #[test]
    fn expand_r2_band_matches_bruteforce_order_and_bounds() {
        const N: usize = 384;
        const RES: usize = 10;
        const EPS: f32 = 1e-5;

        for &seed in &[19u64, 73] {
            let points = random_unit_points(N, seed);
            let grid = CubeMapGrid::new(&points, RES);
            let cell = fullest_cell(&grid);
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
            let query_locals = queries.clone();
            let mut slot_gen_map = vec![0u32; points.len()];
            for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
                *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
            }

            let group = DirectedCellGroup::new(
                cell,
                QUERY_BIN,
                &queries,
                &query_locals,
                &slot_gen_map,
                LOCAL_SHIFT,
                LOCAL_MASK,
            );
            let mut scratch = PackedKnnCellScratch::new();
            let mut timings = PackedKnnTimings::default();
            let PreparedPackedGroupStatus::Ready(mut prepared) =
                scratch.prepare_group_directed(&grid, group, &mut timings)
            else {
                panic!("packed prepare unexpectedly fell back to slow path");
            };

            for qi in 0..queries.len() {
                let query_slot = queries[qi];
                let query_idx = grid.point_indices()[query_slot as usize] as usize;
                assert!(
                    prepared.ensure_expand_r2_band_directed_for(qi, &grid, &mut timings),
                    "cold r=2 expansion unexpectedly exceeded cap for seed={seed}, qi={qi}"
                );

                let security2 = prepared.resume_security(qi);
                let expected =
                    expected_safe_slots(&grid, &points, query_idx, query_locals[qi], security2);
                let mut emitted = Vec::new();
                let mut prev_bound = 1.0f32;
                let mut stage = PackedStage::Chunk0;

                loop {
                    let k = match stage {
                        PackedStage::Chunk0 => 16,
                        PackedStage::Tail => 8,
                        PackedStage::ExpandR2 => 8,
                    };
                    let mut out = vec![u32::MAX; k];
                    let chunk = prepared.next_chunk(qi, stage, k, &mut out, &mut timings);
                    match chunk {
                        Some(chunk) => {
                            assert!(
                                chunk.unseen_bound <= prev_bound + EPS,
                                "unseen bound increased for seed={seed}, qi={qi}"
                            );
                            prev_bound = chunk.unseen_bound;
                            emitted.extend_from_slice(&out[..chunk.n]);

                            if let Some(&next_slot) = expected.get(emitted.len()) {
                                let next_idx = grid.point_indices()[next_slot as usize] as usize;
                                let next_dot = points[query_idx].dot(points[next_idx]);
                                assert!(
                                    next_dot <= chunk.unseen_bound + EPS,
                                    "unseen bound was not conservative for seed={seed}, qi={qi}"
                                );
                            }
                        }
                        None if stage == PackedStage::Chunk0 && prepared.tail_possible(qi) => {
                            prepared.ensure_tail_directed_for(qi, &grid, &mut timings);
                            stage = PackedStage::Tail;
                        }
                        None if stage != PackedStage::ExpandR2 => {
                            stage = PackedStage::ExpandR2;
                        }
                        None => break,
                    }
                }

                assert_eq!(
                    emitted, expected,
                    "expanded packed order mismatch for seed={seed}, qi={qi}"
                );
            }
        }
    }
}
