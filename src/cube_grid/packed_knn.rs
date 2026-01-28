//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! This module is an internal performance component. The only consumer in this crate is the
//! directed live-dedup backend, so we keep the implementation focused on that use-case.

use super::{cell_to_face_ij, CubeMapGrid};
use crate::fp;
use glam::Vec3;
use std::simd::f32x8;
use std::simd::{cmp::SimdPartialOrd, Mask};
use std::time::Duration;

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

/// Timer that tracks elapsed time when timing is enabled.
#[cfg(feature = "timing")]
struct PackedLapTimer(std::time::Instant);

#[cfg(feature = "timing")]
impl PackedLapTimer {
    #[inline]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub fn lap(&mut self) -> Duration {
        let now = std::time::Instant::now();
        let d = now.duration_since(self.0);
        self.0 = now;
        d
    }
}

/// Dummy timer when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
struct PackedLapTimer;

#[cfg(not(feature = "timing"))]
impl PackedLapTimer {
    #[inline(always)]
    pub fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub fn lap(&mut self) -> Duration {
        Duration::ZERO
    }
}

/// Fine-grained timing breakdown for the packed-kNN per-cell-group flow.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub struct PackedKnnTimings {
    pub setup: Duration,
    pub query_cache: Duration,
    pub security_thresholds: Duration,
    pub center_pass: Duration,
    pub ring_thresholds: Duration,
    pub ring_pass: Duration,
    pub ring_fallback: Duration,
    pub select_prep: Duration,
    pub select_query_prep: Duration,
    pub select_partition: Duration,
    pub select_sort: Duration,
    pub select_scatter: Duration,
    /// Number of times tail candidates were built (per query, but counted at most once per group).
    pub tail_builds: u64,
}

/// Dummy timings when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct PackedKnnTimings;

#[cfg(feature = "timing")]
impl PackedKnnTimings {
    #[inline]
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub fn add_setup(&mut self, d: Duration) {
        self.setup += d;
    }

    #[inline]
    pub fn add_query_cache(&mut self, d: Duration) {
        self.query_cache += d;
    }

    #[inline]
    pub fn add_security_thresholds(&mut self, d: Duration) {
        self.security_thresholds += d;
    }

    #[inline]
    pub fn add_center_pass(&mut self, d: Duration) {
        self.center_pass += d;
    }

    #[inline]
    pub fn add_ring_thresholds(&mut self, d: Duration) {
        self.ring_thresholds += d;
    }

    #[inline]
    pub fn add_ring_pass(&mut self, d: Duration) {
        self.ring_pass += d;
    }

    #[inline]
    pub fn add_ring_fallback(&mut self, d: Duration) {
        self.ring_fallback += d;
    }

    #[inline]
    pub fn add_select_prep(&mut self, d: Duration) {
        self.select_prep += d;
    }

    #[inline]
    pub fn add_select_query_prep(&mut self, d: Duration) {
        self.select_query_prep += d;
    }

    #[inline]
    pub fn add_select_partition(&mut self, d: Duration) {
        self.select_partition += d;
    }

    #[inline]
    pub fn add_select_sort(&mut self, d: Duration) {
        self.select_sort += d;
    }

    #[inline]
    pub fn add_select_scatter(&mut self, d: Duration) {
        self.select_scatter += d;
    }

    #[inline]
    pub fn inc_tail_builds(&mut self) {
        self.tail_builds += 1;
    }

    #[inline]
    pub fn total(&self) -> Duration {
        self.setup
            + self.query_cache
            + self.security_thresholds
            + self.center_pass
            + self.ring_thresholds
            + self.ring_pass
            + self.ring_fallback
            + self.select_prep
            + self.select_query_prep
            + self.select_partition
            + self.select_sort
            + self.select_scatter
    }
}

#[cfg(not(feature = "timing"))]
impl PackedKnnTimings {
    #[inline(always)]
    pub fn clear(&mut self) {}

    #[inline(always)]
    pub fn add_setup(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_query_cache(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_security_thresholds(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_center_pass(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_ring_thresholds(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_ring_pass(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_ring_fallback(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_prep(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_query_prep(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_partition(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_sort(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_select_scatter(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn inc_tail_builds(&mut self) {}
}

// Hard cap on total candidates in a 3×3 neighborhood to avoid pathological allocations.
const MAX_CANDIDATES_HARD: usize = 65_536;

/// Reusable scratch buffers for packed per-cell group queries.
pub struct PackedKnnCellScratch {
    cell_ranges: Vec<PackedCellRange>,
    center_lens: Vec<usize>,
    min_center_dot: Vec<f32>,
    group_queries: Vec<u32>,
    group_cell: usize,
    group_query_bin: u8,
    group_gen: u32,
    chunk0_keys: Vec<Vec<u64>>,
    tail_keys: Vec<Vec<u64>>,
    chunk0_pos: Vec<usize>,
    tail_pos: Vec<usize>,
    tail_possible: Vec<bool>,
    tail_ready_gen: Vec<u32>,
    tail_built_any: bool,
    security_thresholds: Vec<f32>,
    thresholds: Vec<f32>,
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

impl PackedKnnCellScratch {
    pub fn new() -> Self {
        Self {
            cell_ranges: Vec::with_capacity(9),
            center_lens: Vec::new(),
            min_center_dot: Vec::new(),
            group_queries: Vec::new(),
            group_cell: 0,
            group_query_bin: 0,
            group_gen: 1,
            chunk0_keys: Vec::new(),
            tail_keys: Vec::new(),
            chunk0_pos: Vec::new(),
            tail_pos: Vec::new(),
            tail_possible: Vec::new(),
            tail_ready_gen: Vec::new(),
            tail_built_any: false,
            security_thresholds: Vec::new(),
            thresholds: Vec::new(),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn prepare_group_directed(
        &mut self,
        grid: &CubeMapGrid,
        cell: usize,
        queries: &[u32],
        query_locals: &[u32],
        query_bin: u8,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
        timings: &mut PackedKnnTimings,
    ) -> PackedKnnCellStatus {
        timings.clear();

        let num_queries = queries.len();
        if num_queries == 0 {
            return PackedKnnCellStatus::Ok;
        }

        self.group_cell = cell;
        self.group_query_bin = query_bin;
        self.group_queries.clear();
        self.group_queries.extend_from_slice(queries);
        self.group_gen = self.group_gen.wrapping_add(1);
        if self.group_gen == 0 {
            // Reserve `0` as "never set"; on wrap, clear all generation stamps.
            self.group_gen = 1;
            self.tail_ready_gen.fill(0);
        }

        let num_cells = 6 * grid.res * grid.res;
        if cell >= num_cells {
            return PackedKnnCellStatus::Ok;
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
            return PackedKnnCellStatus::SlowPath;
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
        debug_assert_eq!(
            num_queries,
            qx_src.len(),
            "directed packed group must cover the full center cell"
        );

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
        self.tail_built_any = false;
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
        debug_assert_eq!(
            num_queries, center_len,
            "directed packed group must cover the full center cell"
        );
        debug_assert!(
            queries
                .iter()
                .enumerate()
                .all(|(qi, &s)| s as usize == center_soa_start + qi),
            "directed packed group queries must be the center cell in slot order"
        );
        debug_assert!(
            query_locals.windows(2).all(|w| w[1] == w[0] + 1),
            "directed packed group locals must be contiguous in slot order"
        );
        debug_assert!(
            queries.iter().zip(query_locals.iter()).all(|(&slot, &ql)| {
                let packed = slot_gen_map[slot as usize];
                let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                bin_b == query_bin && local_b == ql
            }),
            "directed packed group (slot -> bin,local) mapping must match query inputs"
        );

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

        for qi in 0..num_queries {
            let security = security_thresholds[qi];
            let center_len = self.center_lens[qi];
            let min_dot = self.min_center_dot[qi];
            self.thresholds[qi] = if center_len > 0 {
                security.max(min_dot - 1e-6)
            } else {
                security
            };
            self.tail_possible[qi] = self.thresholds[qi] > security;
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

        PackedKnnCellStatus::Ok
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn ensure_tail_directed_for(
        &mut self,
        qi: usize,
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
        if gen == self.group_gen {
            return;
        }
        if !self.tail_built_any {
            self.tail_built_any = true;
            timings.inc_tail_builds();
        }
        self.tail_ready_gen[qi] = self.group_gen;

        self.tail_keys[qi].clear();
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

            let query_slot = self.group_queries[qi];
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
    pub fn next_chunk(
        &mut self,
        qi: usize,
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
                    remaining.sort_unstable();
                    timings.add_select_sort(t.lap());
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
                remaining[..n].sort_unstable();
                timings.add_select_sort(t.lap());
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
                    self.tail_ready_gen.get(qi).copied().unwrap_or(0) == self.group_gen,
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
                remaining[..n].sort_unstable();
                timings.add_select_sort(t.lap());
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
        }
    }

    #[inline]
    pub fn security(&self, qi: usize) -> f32 {
        self.security_thresholds[qi]
    }

    #[inline]
    pub fn tail_possible(&self, qi: usize) -> bool {
        self.tail_possible.get(qi).copied().unwrap_or(false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedKnnCellStatus {
    Ok,
    SlowPath,
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
    // Bigger dot = smaller key, so ascending sort gives descending dot.
    let ord = f32_to_ordered_u32(dot);
    let desc = !ord;
    ((desc as u64) << 32) | (idx as u64)
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
