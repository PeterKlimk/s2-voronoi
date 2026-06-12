//! Planar packed-kNN scratch + group preparation, selection, and the lazy
//! tail / ring-2 expansion stages.
//!
//! The planar port of `cube_grid::packed_knn::scratch` with the semantic
//! inversions documented per site: squared-distance keys sort ASCENDING
//! (nearest first), candidate masks are `mask_lt`, and every conservative
//! spherical bound (security planes, ring-2 cap sweeps, ring-3 BFS) is
//! replaced by the exact box certificate [`super::super::outside_box_dist_sq`]
//! — anything outside the radius-`r` cell box is at least that far away, so
//! the 5x5 box bound directly replaces the sphere's ring-3 machinery.

use crate::sort::sort_small as sort_small_u64;

use super::timing::{PlanePackedLapTimer, PlanePackedTimings};
use super::{PackedGeometry, PlanePackedChunk, PlanePackedGroupInput, PlanePackedStage};
use crate::fp;
use crate::policy::{PACKED_HI_BUDGET, PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY};

// Hard cap on total candidates in a 3x3 neighborhood to avoid pathological
// allocations (same value as the sphere).
const MAX_CANDIDATES_HARD: usize = 65_536;

#[inline(always)]
fn unpack_bin_local(packed: u32, local_shift: u32, local_mask: u32) -> (u8, u32) {
    let bin = (packed >> local_shift) as u8;
    let local = packed & local_mask;
    (bin, local)
}

/// Ascending key: smaller dist_sq sorts first (nearest first). `dist_sq` is
/// finite and non-negative, so its bit pattern is order-preserving and the
/// key can never be `u64::MAX` (the sorting-network padding sentinel).
#[inline(always)]
fn make_asc_key(dist_sq: f32, idx: u32) -> u64 {
    debug_assert!(dist_sq.is_finite() && dist_sq >= 0.0);
    ((dist_sq.to_bits() as u64) << 32) | (idx as u64)
}

#[inline(always)]
fn key_to_idx(key: u64) -> u32 {
    (key & 0xFFFF_FFFF) as u32
}

#[inline(always)]
fn key_to_dist_sq(key: u64) -> f32 {
    f32::from_bits((key >> 32) as u32)
}

#[inline(always)]
fn sort_keys_u64(keys: &mut [u64]) {
    // Always-on small-N sorting networks (see the cube twin for the data).
    if keys.len() <= 35 {
        sort_small_u64(keys);
        return;
    }
    keys.sort_unstable();
}

#[derive(Debug, Clone, Copy)]
struct PackedCellRange {
    soa_start: usize,
    soa_end: usize,
    kind: PackedCellRangeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedCellRangeKind {
    /// Neighbor cell is in a different bin (no directed filter needed).
    CrossBin,
    /// Same bin, strictly earlier cell: skip entirely (the earlier side
    /// already forwarded adjacency via edge checks).
    SameBinEarlier,
    /// Same bin, strictly later cell: no directed filter needed.
    SameBinLater,
}

/// Reusable scratch buffers for planar packed per-cell group queries.
pub struct PlanePackedScratch {
    cell_ranges: Vec<PackedCellRange>,
    next_group_gen: u32,
    chunk0_keys: Vec<Vec<u64>>,
    tail_keys: Vec<Vec<u64>>,
    chunk0_pos: Vec<usize>,
    tail_pos: Vec<usize>,
    tail_possible: Vec<bool>,
    tail_ready_gen: Vec<u32>,
    /// Lower bound on dist_sq of anything outside the 3x3 box.
    security_thresholds: Vec<f32>,
    /// Tightened "hi" threshold: chunk0 holds candidates with
    /// `dist_sq < thresholds[qi]`; the tail holds the rest below security.
    thresholds: Vec<f32>,
    expand_r2_cells: Vec<PackedCellRange>,
    expand_r2_cells_gen: u32,
    expand2_keys: Vec<Vec<u64>>,
    expand2_pos: Vec<usize>,
    expand2_ready_gen: Vec<u32>,
    /// Lower bound on dist_sq of anything outside the 5x5 box.
    security2: Vec<f32>,
    security2_ready_gen: Vec<u32>,
}

pub(crate) struct PlanePreparedGroup<'a, 'g> {
    scratch: &'a mut PlanePackedScratch,
    group: PlanePackedGroupInput<'g>,
    group_gen: u32,
}

pub(crate) enum PlanePreparedGroupStatus<'a, 'g> {
    Ready(PlanePreparedGroup<'a, 'g>),
    SlowPath,
}

impl<'a, 'g> PlanePreparedGroup<'a, 'g> {
    #[inline]
    pub(super) fn next_chunk(
        &mut self,
        qi: usize,
        stage: PlanePackedStage,
        k: usize,
        out: &mut Vec<u32>,
        out_dists: &mut Vec<f32>,
        timings: &mut PlanePackedTimings,
    ) -> Option<PlanePackedChunk> {
        self.scratch
            .next_chunk(qi, self.group_gen, stage, k, out, out_dists, timings)
    }

    #[inline]
    pub(super) fn ensure_tail_for<G: PackedGeometry>(
        &mut self,
        qi: usize,
        grid: &G,
        timings: &mut PlanePackedTimings,
    ) {
        self.scratch
            .ensure_tail_for(qi, self.group.queries(), self.group_gen, grid, timings);
    }

    #[inline]
    pub(super) fn ensure_expand_r2_band_for<G: PackedGeometry>(
        &mut self,
        qi: usize,
        grid: &G,
        timings: &mut PlanePackedTimings,
    ) -> bool {
        self.scratch
            .ensure_expand_r2_band_for(qi, self.group, self.group_gen, grid, timings)
    }

    #[inline]
    pub(super) fn resume_security(&self, qi: usize) -> f32 {
        if self.scratch.expand2_ready_gen.get(qi).copied().unwrap_or(0) == self.group_gen {
            self.scratch.security2[qi]
        } else {
            self.scratch.security_thresholds[qi]
        }
    }

    #[inline]
    pub(super) fn tail_possible(&self, qi: usize) -> bool {
        self.scratch.tail_possible.get(qi).copied().unwrap_or(false)
    }

    #[inline]
    pub(super) fn tail_lower_bound(&self, qi: usize) -> f32 {
        self.scratch.thresholds[qi]
    }
}

impl Default for PlanePackedScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanePackedScratch {
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
            self.security2.resize(num_queries, f32::INFINITY);
        }
        if self.security2_ready_gen.len() < num_queries {
            self.security2_ready_gen.resize(num_queries, 0);
        }
    }

    /// Classify a neighbor cell's SoA range for the directed filter (bin and
    /// cell-order properties hold cell-wide, so one decode per cell).
    #[inline]
    fn classify_cell_range<G: PackedGeometry>(
        grid: &G,
        cell: usize,
        center_cell: usize,
        query_bin: u8,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
    ) -> Option<PackedCellRange> {
        let start = grid.cell_offsets()[cell] as usize;
        let end = grid.cell_offsets()[cell + 1] as usize;
        if start >= end {
            return None;
        }
        let kind = if cell == center_cell {
            PackedCellRangeKind::CrossBin
        } else {
            let (bin_b, _) = unpack_bin_local(slot_gen_map[start], local_shift, local_mask);
            if bin_b != query_bin {
                PackedCellRangeKind::CrossBin
            } else if cell < center_cell {
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

    #[cfg_attr(feature = "profiling", inline(never))]
    // Loop indices address several parallel per-query arrays at once;
    // iterator zips would obscure rather than clarify.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn prepare_group<'a, 'g, G: PackedGeometry>(
        &'a mut self,
        grid: &G,
        group: PlanePackedGroupInput<'g>,
        timings: &mut PlanePackedTimings,
    ) -> PlanePreparedGroupStatus<'a, 'g> {
        timings.clear();
        group.debug_assert_matches_grid(grid);
        // Wrapped boxes must enumerate distinct cells (tiny torus grids
        // fall back to the shell stream, which stamps visited cells).
        if !grid.box_radius_distinct(1) {
            return PlanePreparedGroupStatus::SlowPath;
        }

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
            self.expand2_ready_gen.fill(0);
            self.security2_ready_gen.fill(0);
        }
        self.next_group_gen = group_gen;

        let res = grid.res();
        let (cx, cy) = (cell % res, cell / res);

        let mut t = PlanePackedLapTimer::start();
        self.cell_ranges.clear();

        let q_start = grid.cell_offsets()[cell] as usize;
        let q_end = grid.cell_offsets()[cell + 1] as usize;
        self.cell_ranges.push(PackedCellRange {
            soa_start: q_start,
            soa_end: q_end,
            kind: PackedCellRangeKind::CrossBin,
        });
        grid.for_each_box_cell(cx, cy, 1, |ncell| {
            if ncell == cell {
                return;
            }
            if let Some(range) = Self::classify_cell_range(
                grid,
                ncell,
                cell,
                query_bin,
                slot_gen_map,
                local_shift,
                local_mask,
            ) {
                self.cell_ranges.push(range);
            }
        });

        let mut num_candidates = 0usize;
        let mut ring_candidates_eligible = 0usize;
        for (i, r) in self.cell_ranges.iter().enumerate() {
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }
            let len = r.soa_end - r.soa_start;
            num_candidates += len;
            if i > 0 {
                ring_candidates_eligible += len;
            }
        }
        if num_candidates > MAX_CANDIDATES_HARD {
            timings.add_setup(t.lap());
            return PlanePreparedGroupStatus::SlowPath;
        }
        timings.add_setup(t.lap());

        // live_dedup invariant: groups are complete center-cell runs in slot
        // order, so the query coordinates are the center cell's SoA slices.
        let qx_src = &grid.points_x()[q_start..q_end];
        let qy_src = &grid.points_y()[q_start..q_end];

        // Security: exact lower bound on anything outside the 3x3 box.
        self.security_thresholds.clear();
        self.security_thresholds.reserve(num_queries);
        for qi in 0..num_queries {
            self.security_thresholds
                .push(grid.outside_box_dist_sq(cx, cy, 1, qx_src[qi], qy_src[qi]));
        }
        timings.add_security_thresholds(t.lap());

        self.thresholds.resize(num_queries, 0.0);
        if self.chunk0_keys.len() < num_queries {
            self.chunk0_keys.resize_with(num_queries, Vec::new);
        }
        for v in &mut self.chunk0_keys[..num_queries] {
            v.clear();
        }
        self.chunk0_pos.resize(num_queries, 0);
        self.chunk0_pos.fill(0);
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
        let center_len = q_end - q_start;
        debug_assert_eq!(num_queries, center_len, "center-cell query length mismatch");
        let xs = qx_src;
        let ys = qy_src;
        let security_thresholds = &self.security_thresholds[..num_queries];
        let chunk0_keys = &mut self.chunk0_keys[..num_queries];

        let full_chunks = center_len / 8;
        for chunk in 0..full_chunks {
            let i = chunk * 8;
            let candidates = fp::PlaneChunk8::from_slices(&xs[i..], &ys[i..]);

            // Candidate positions in this chunk are [i, i+7]. A query at
            // position qi only sees candidates at positions >= qi.
            let qi_end = (i + 8).min(num_queries);
            for qi in 0..qi_end {
                let dists = grid.chunk_dist_sqs(&candidates, xs[qi], ys[qi]);
                let mut mask_bits = dists.mask_lt(security_thresholds[qi]);
                if mask_bits == 0 {
                    continue;
                }

                // Directed intra-bin filter for the center cell: allowed
                // candidates are those with position >= qi, excluding self.
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

                let dists_arr = dists.to_array();
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let slot = (q_start + i + lane) as u32;
                    chunk0_keys[qi].push(make_asc_key(dists_arr[lane], slot));
                    mask_bits &= mask_bits - 1;
                }
            }
        }
        let tail_start = full_chunks * 8;
        for pos in tail_start..center_len {
            let slot = (q_start + pos) as u32;
            let qi_end = (pos + 1).min(num_queries);
            for qi in 0..qi_end {
                if qi == pos {
                    continue;
                }
                let dist_sq = grid.dist_sq(xs[pos], ys[pos], xs[qi], ys[qi]);
                if dist_sq < security_thresholds[qi] {
                    chunk0_keys[qi].push(make_asc_key(dist_sq, slot));
                }
            }
        }
        timings.add_center_pass(t.lap());

        // === Threshold selection.
        //
        // Tighten the hi threshold so chunk0 targets ~PACKED_HI_BUDGET
        // candidates (caps select/sort costs on dense neighborhoods). Model:
        // under uniform density the number of candidates with dist_sq < t
        // grows linearly in t (area), so t = security * (budget / n_total)
        // targets the budget. (The sphere's interpolation in dot space is
        // the same linear-in-area model.) Demoted center candidates and the
        // ring's [t, security) band go to the lazily built tail.
        for qi in 0..num_queries {
            let security = security_thresholds[qi];
            let center_eligible = num_queries.saturating_sub(qi + 1);
            let n_total = ring_candidates_eligible + center_eligible;
            let t_count = if n_total == 0 {
                security
            } else {
                let ratio = ((PACKED_HI_BUDGET as f32) / (n_total as f32)).min(1.0);
                // security may be INFINITY (box covers the grid); INF * ratio
                // = INF keeps "no tightening" semantics.
                (security * ratio).clamp(0.0, security)
            };
            self.thresholds[qi] = t_count;
            self.tail_possible[qi] = t_count < security;
        }

        // Demote center candidates at/above the tightened threshold into the
        // tail, so every kept chunk0 candidate has dist_sq < thresholds[qi]
        // and the ring pass (mask_lt(threshold)) cannot miss anything that
        // outranks a kept candidate.
        for qi in 0..num_queries {
            let t_hi = self.thresholds[qi];
            let v = &mut chunk0_keys[qi];
            if v.is_empty() {
                continue;
            }
            let tail_v = &mut self.tail_keys[qi];
            let mut write = 0usize;
            for idx in 0..v.len() {
                let key = v[idx];
                if key_to_dist_sq(key) < t_hi {
                    v[write] = key;
                    write += 1;
                } else {
                    tail_v.push(key);
                }
            }
            v.truncate(write);
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
            let range_len = r.soa_end - soa_start;
            let rxs = &grid.points_x()[soa_start..r.soa_end];
            let rys = &grid.points_y()[soa_start..r.soa_end];

            let full_chunks = range_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let candidates = fp::PlaneChunk8::from_slices(&rxs[i..], &rys[i..]);
                for qi in 0..num_queries {
                    let dists = grid.chunk_dist_sqs(&candidates, xs[qi], ys[qi]);
                    let mut mask_bits = dists.mask_lt(thresholds[qi]);
                    if mask_bits == 0 {
                        continue;
                    }
                    let dists_arr = dists.to_array();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        chunk0_keys[qi].push(make_asc_key(dists_arr[lane], slot));
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
            let rem = range_len % 8;
            if rem != 0 {
                let i = full_chunks * 8;
                let valid_bits = (1u32 << (rem as u32)) - 1;
                let mut xbuf = [0.0f32; 8];
                let mut ybuf = [0.0f32; 8];
                xbuf[..rem].copy_from_slice(&rxs[i..]);
                ybuf[..rem].copy_from_slice(&rys[i..]);
                let candidates = fp::PlaneChunk8::from_arrays(xbuf, ybuf);
                for qi in 0..num_queries {
                    let dists = grid.chunk_dist_sqs(&candidates, xs[qi], ys[qi]);
                    let mut mask_bits = dists.mask_lt(thresholds[qi]) & valid_bits;
                    if mask_bits == 0 {
                        continue;
                    }
                    let dists_arr = dists.to_array();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        chunk0_keys[qi].push(make_asc_key(dists_arr[lane], slot));
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }
        timings.add_ring_pass(t.lap());

        PlanePreparedGroupStatus::Ready(PlanePreparedGroup {
            scratch: self,
            group,
            group_gen,
        })
    }

    /// Lazily build the tail for one query: the [threshold, security) band
    /// over the ring cells (demoted center candidates are already stored).
    fn ensure_tail_for<G: PackedGeometry>(
        &mut self,
        qi: usize,
        group_queries: &[u32],
        group_gen: u32,
        grid: &G,
        timings: &mut PlanePackedTimings,
    ) {
        let Some(gen) = self.tail_ready_gen.get(qi).copied() else {
            return;
        };
        if gen == group_gen {
            return;
        }
        self.tail_ready_gen[qi] = group_gen;
        self.tail_pos[qi] = 0;
        debug_assert!(self.tail_possible.get(qi).copied().unwrap_or(false));
        timings.inc_tail_builds();

        let query_slot = group_queries[qi];
        let qx = grid.points_x()[query_slot as usize];
        let qy = grid.points_y()[query_slot as usize];
        let security = self.security_thresholds[qi];
        let t_hi = self.thresholds[qi];

        let mut t_tail = PlanePackedLapTimer::start();
        for r in &self.cell_ranges[1..] {
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }
            let soa_start = r.soa_start;
            let range_len = r.soa_end - soa_start;
            let rxs = &grid.points_x()[soa_start..r.soa_end];
            let rys = &grid.points_y()[soa_start..r.soa_end];

            let full_chunks = range_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let candidates = fp::PlaneChunk8::from_slices(&rxs[i..], &rys[i..]);
                let dists = grid.chunk_dist_sqs(&candidates, qx, qy);
                let safe_bits = dists.mask_lt(security);
                let hi_bits = dists.mask_lt(t_hi);
                let mut tail_bits = safe_bits & !hi_bits;
                if tail_bits == 0 {
                    continue;
                }
                let dists_arr = dists.to_array();
                while tail_bits != 0 {
                    let lane = tail_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    if slot != query_slot {
                        self.tail_keys[qi].push(make_asc_key(dists_arr[lane], slot));
                    }
                    tail_bits &= tail_bits - 1;
                }
            }
            let rem_start = full_chunks * 8;
            for i in rem_start..range_len {
                let slot = (soa_start + i) as u32;
                if slot == query_slot {
                    continue;
                }
                let dist_sq = grid.dist_sq(rxs[i], rys[i], qx, qy);
                if dist_sq < security && dist_sq >= t_hi {
                    self.tail_keys[qi].push(make_asc_key(dist_sq, slot));
                }
            }
        }
        timings.add_ring_fallback(t_tail.lap());

        if self.tail_keys[qi].is_empty() {
            self.tail_possible[qi] = false;
        }
    }

    fn ensure_expand_r2_cells<G: PackedGeometry>(
        &mut self,
        grid: &G,
        group: PlanePackedGroupInput<'_>,
        group_gen: u32,
    ) {
        if self.expand_r2_cells_gen == group_gen {
            return;
        }
        self.expand_r2_cells.clear();
        self.expand_r2_cells_gen = group_gen;

        let res = grid.res();
        let center = group.cell();
        let (cx, cy) = (center % res, center / res);
        let mut ranges: Vec<PackedCellRange> = Vec::new();
        grid.for_each_box_cell(cx, cy, 2, |cell| {
            if let Some(range) = Self::classify_cell_range(
                grid,
                cell,
                center,
                group.query_bin(),
                group.slot_gen_map(),
                group.local_shift(),
                group.local_mask(),
            ) {
                ranges.push(range);
            }
        });
        self.expand_r2_cells = ranges;
    }

    /// Lower bound on dist_sq of anything outside the 5x5 box (lazy).
    fn ensure_security2_for<G: PackedGeometry>(
        &mut self,
        qi: usize,
        group: PlanePackedGroupInput<'_>,
        group_gen: u32,
        grid: &G,
    ) -> f32 {
        self.ensure_cold_query_storage(group.queries().len());
        if self.security2_ready_gen[qi] == group_gen {
            return self.security2[qi];
        }
        let res = grid.res();
        let center = group.cell();
        let (cx, cy) = (center % res, center / res);
        let query_slot = group.queries()[qi] as usize;
        let security = grid.outside_box_dist_sq(
            cx,
            cy,
            2,
            grid.points_x()[query_slot],
            grid.points_y()[query_slot],
        );
        self.security2[qi] = security;
        self.security2_ready_gen[qi] = group_gen;
        security
    }

    /// Lazily build the ring-2 expansion band for one query: candidates in
    /// the 5x5 box with `security1 <= dist_sq < security2`. Returns false
    /// when the band exceeds the per-query cap (caller falls through to the
    /// shell takeover).
    fn ensure_expand_r2_band_for<G: PackedGeometry>(
        &mut self,
        qi: usize,
        group: PlanePackedGroupInput<'_>,
        group_gen: u32,
        grid: &G,
        timings: &mut PlanePackedTimings,
    ) -> bool {
        if !grid.box_radius_distinct(2) {
            // Tiny wrapped grids: the 5x5 box would revisit cells; hand off
            // to the shell takeover instead.
            return false;
        }
        self.ensure_cold_query_storage(group.queries().len());
        if self.expand2_ready_gen[qi] == group_gen {
            return true;
        }

        let security2 = self.ensure_security2_for(qi, group, group_gen, grid);
        self.ensure_expand_r2_cells(grid, group, group_gen);

        let query_slot = group.queries()[qi];
        let qx = grid.points_x()[query_slot as usize];
        let qy = grid.points_y()[query_slot as usize];
        let security1 = self.security_thresholds[qi];
        let center_range = self.cell_ranges[0];

        let keys = &mut self.expand2_keys[qi];
        keys.clear();
        self.expand2_pos[qi] = 0;
        timings.inc_expand_r2_builds();

        let mut t = PlanePackedLapTimer::start();
        for range in &self.expand_r2_cells {
            if range.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }
            for slot in range.soa_start..range.soa_end {
                let slot_u32 = slot as u32;
                if slot_u32 == query_slot {
                    continue;
                }
                // Directed center-cell filter (same-bin same-cell: local >=
                // query local, i.e. slot >= query slot).
                if range.soa_start == center_range.soa_start
                    && range.soa_end == center_range.soa_end
                    && slot_u32 < query_slot
                {
                    continue;
                }
                let dist_sq = grid.dist_sq(grid.points_x()[slot], grid.points_y()[slot], qx, qy);
                if dist_sq >= security1 && dist_sq < security2 {
                    keys.push(make_asc_key(dist_sq, slot_u32));
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

    /// Emit the next sorted chunk for a stage. `out`/`out_dists` receive the
    /// slots and their squared distances (ascending); the chunk's
    /// `unseen_bound` lower-bounds everything not yet emitted.
    #[cfg_attr(feature = "profiling", inline(never))]
    #[allow(clippy::too_many_arguments)] // hot-path SoA plumbing, mirrors the sphere
    fn next_chunk(
        &mut self,
        qi: usize,
        group_gen: u32,
        stage: PlanePackedStage,
        k: usize,
        out: &mut Vec<u32>,
        out_dists: &mut Vec<f32>,
        timings: &mut PlanePackedTimings,
    ) -> Option<PlanePackedChunk> {
        if k == 0 {
            return None;
        }

        let (keys, pos, exhausted_bound) = match stage {
            PlanePackedStage::Chunk0 => {
                let bound = if self.tail_possible.get(qi).copied().unwrap_or(false) {
                    self.thresholds[qi]
                } else {
                    self.security_thresholds[qi]
                };
                (
                    self.chunk0_keys.get_mut(qi)?,
                    &mut self.chunk0_pos[qi],
                    bound,
                )
            }
            PlanePackedStage::Tail => {
                debug_assert!(
                    self.tail_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "tail stage requested before ensure_tail"
                );
                let bound = self.security_thresholds[qi];
                (self.tail_keys.get_mut(qi)?, &mut self.tail_pos[qi], bound)
            }
            PlanePackedStage::ExpandR2 => {
                debug_assert!(
                    self.expand2_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "expand_r2 stage requested before ensure_expand_r2"
                );
                let bound = self.security2[qi];
                (
                    self.expand2_keys.get_mut(qi)?,
                    &mut self.expand2_pos[qi],
                    bound,
                )
            }
        };

        let start = *pos;
        if start >= keys.len() {
            return None;
        }
        let remaining = &mut keys[start..];
        let n_target = k;
        let mut t = PlanePackedLapTimer::start();
        timings.add_select_query_prep(t.lap());

        // Small remainder: skip the partition and sort everything. Must
        // scale with n_target so a shrunken k never sorts large remainders.
        let n = if remaining.len() <= 2 * n_target {
            let sort_len = remaining.len();
            sort_keys_u64(remaining);
            timings.add_select_sort_sized(t.lap(), sort_len);
            remaining.len().min(n_target)
        } else {
            let n = n_target;
            remaining.select_nth_unstable(n - 1);
            timings.add_select_partition(t.lap());
            sort_keys_u64(&mut remaining[..n]);
            timings.add_select_sort_sized(t.lap(), n);
            n
        };

        out.clear();
        out_dists.clear();
        out.extend(remaining[..n].iter().map(|&key| key_to_idx(key)));
        out_dists.extend(remaining[..n].iter().map(|&key| key_to_dist_sq(key)));
        timings.add_select_scatter(t.lap());

        let last_dist = key_to_dist_sq(remaining[n - 1]);
        *pos = start + n;
        let has_more = *pos < keys.len();
        let unseen_bound = if has_more { last_dist } else { exhausted_bound };
        Some(PlanePackedChunk { n, unseen_bound })
    }
}
