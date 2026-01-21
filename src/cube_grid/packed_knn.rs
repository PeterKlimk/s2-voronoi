//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! Dominant mode: PackedV4 (batched, cell-local, SIMD dot products).

use super::{cell_to_face_ij, CubeMapGrid};
use glam::Vec3;
use std::mem::MaybeUninit;
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
struct PackedTimer(std::time::Instant);

#[cfg(feature = "timing")]
impl PackedTimer {
    #[inline]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Dummy timer when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
struct PackedTimer;

#[cfg(not(feature = "timing"))]
impl PackedTimer {
    #[inline(always)]
    pub fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

/// Fine-grained timing breakdown for `packed_knn_cell_stream`.
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
    /// Time spent inside the per-query callback (`on_query`), excluded from `packed_knn` overhead.
    pub callback: Duration,
    /// Number of times tail candidates were built (group-level, since tail build is shared).
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
    pub fn add_callback(&mut self, d: Duration) {
        self.callback += d;
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
    pub fn add_callback(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn inc_tail_builds(&mut self) {}
}

// Fast path stores candidate keys in a dense slab of size (num_queries * num_candidates).
// For typical KNN_GRID_TARGET_DENSITY (32) this is small (~9*density^2 ≈ 9k),
// but it grows quadratically with density. Keep a generous per-cell cap while
// still preventing pathological allocations.
const MAX_CANDIDATES_FAST: usize = 1024;
const MAX_KEYS_SLAB_FAST: usize = 200_000;
const MAX_CANDIDATES_HARD: usize = 65_536;

#[allow(dead_code)]
#[inline(always)]
fn dense_scan_range<const UPDATE_MIN: bool>(
    stride: usize,
    soa_start: usize,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    queries: &[u32],
    query_x: &[f32],
    query_y: &[f32],
    query_z: &[f32],
    thresholds: &[f32],
    keys_slab: &mut [MaybeUninit<u64>],
    lens: &mut [usize],
    min_center_dot: &mut [f32],
) {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(xs.len(), zs.len());
    debug_assert_eq!(queries.len(), lens.len());
    debug_assert_eq!(queries.len(), thresholds.len());
    debug_assert_eq!(queries.len(), query_x.len());
    debug_assert_eq!(queries.len(), query_y.len());
    debug_assert_eq!(queries.len(), query_z.len());
    debug_assert_eq!(queries.len(), min_center_dot.len());
    debug_assert!(keys_slab.len() >= queries.len() * stride);

    let num_queries = queries.len();
    let query_x = &query_x[..num_queries];
    let query_y = &query_y[..num_queries];
    let query_z = &query_z[..num_queries];
    let thresholds = &thresholds[..num_queries];
    let lens = &mut lens[..num_queries];
    let min_center_dot = &mut min_center_dot[..num_queries];

    let range_len = xs.len();
    let full_chunks = range_len / 8;
    for chunk in 0..full_chunks {
        let i = chunk * 8;
        let cx = f32x8::from_slice(&xs[i..]);
        let cy = f32x8::from_slice(&ys[i..]);
        let cz = f32x8::from_slice(&zs[i..]);

        let it = queries
            .iter()
            .zip(query_x.iter())
            .zip(query_y.iter())
            .zip(query_z.iter())
            .zip(thresholds.iter())
            .zip(lens.iter_mut())
            .zip(min_center_dot.iter_mut());

        let mut slab_base = 0usize;
        for ((((((query_slot, qx), qy), qz), thr), len_ref), min_ref) in it {
            let slab_base_this = slab_base;
            let qx = f32x8::splat(*qx);
            let qy = f32x8::splat(*qy);
            let qz = f32x8::splat(*qz);
            let dots = cx * qx + cy * qy + cz * qz;

            let thresh_vec = f32x8::splat(*thr);
            let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
            let mut mask_bits = mask.to_bitmask() as u32;

            if mask_bits != 0 {
                let dots_arr: [f32; 8] = dots.into();
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    if slot != *query_slot {
                        let dot = dots_arr[lane];
                        let slab_idx = slab_base_this + *len_ref;
                        keys_slab[slab_idx].write(make_desc_key(dot, slot));
                        *len_ref += 1;
                        if UPDATE_MIN {
                            *min_ref = (*min_ref).min(dot);
                        }
                    }
                    mask_bits &= mask_bits - 1;
                }
            }

            slab_base += stride;
        }
    }

    let tail_start = full_chunks * 8;
    for i in tail_start..range_len {
        let cx = xs[i];
        let cy = ys[i];
        let cz = zs[i];
        let slot = (soa_start + i) as u32;

        let it = queries
            .iter()
            .zip(query_x.iter())
            .zip(query_y.iter())
            .zip(query_z.iter())
            .zip(thresholds.iter())
            .zip(lens.iter_mut())
            .zip(min_center_dot.iter_mut());

        let mut slab_base = 0usize;
        for ((((((query_slot, qx), qy), qz), thr), len_ref), min_ref) in it {
            let slab_base_this = slab_base;
            if slot == *query_slot {
                slab_base += stride;
                continue;
            }
            let dot = cx * *qx + cy * *qy + cz * *qz;
            if dot > *thr {
                let slab_idx = slab_base_this + *len_ref;
                keys_slab[slab_idx].write(make_desc_key(dot, slot));
                *len_ref += 1;
                if UPDATE_MIN {
                    *min_ref = (*min_ref).min(dot);
                }
            }

            slab_base += stride;
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn dense_scan_range_directed<const UPDATE_MIN: bool>(
    stride: usize,
    soa_start: usize,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    queries: &[u32],
    query_locals: &[u32],
    query_bin: u8,
    slot_gen_map: &[u32],
    local_shift: u32,
    local_mask: u32,
    query_x: &[f32],
    query_y: &[f32],
    query_z: &[f32],
    thresholds: &[f32],
    keys_slab: &mut [MaybeUninit<u64>],
    lens: &mut [usize],
    min_center_dot: &mut [f32],
) {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(xs.len(), zs.len());
    debug_assert_eq!(queries.len(), lens.len());
    debug_assert_eq!(queries.len(), thresholds.len());
    debug_assert_eq!(queries.len(), query_x.len());
    debug_assert_eq!(queries.len(), query_y.len());
    debug_assert_eq!(queries.len(), query_z.len());
    debug_assert_eq!(queries.len(), min_center_dot.len());
    debug_assert_eq!(queries.len(), query_locals.len());
    debug_assert!(keys_slab.len() >= queries.len() * stride);

    let num_queries = queries.len();
    let query_x = &query_x[..num_queries];
    let query_y = &query_y[..num_queries];
    let query_z = &query_z[..num_queries];
    let thresholds = &thresholds[..num_queries];
    let lens = &mut lens[..num_queries];
    let min_center_dot = &mut min_center_dot[..num_queries];
    let query_locals = &query_locals[..num_queries];

    let range_len = xs.len();
    let full_chunks = range_len / 8;
    for chunk in 0..full_chunks {
        let i = chunk * 8;
        let cx = f32x8::from_slice(&xs[i..]);
        let cy = f32x8::from_slice(&ys[i..]);
        let cz = f32x8::from_slice(&zs[i..]);

        let it = queries
            .iter()
            .zip(query_locals.iter())
            .zip(query_x.iter())
            .zip(query_y.iter())
            .zip(query_z.iter())
            .zip(thresholds.iter())
            .zip(lens.iter_mut())
            .zip(min_center_dot.iter_mut());

        let mut slab_base = 0usize;
        for (((((((query_slot, &query_local), qx), qy), qz), thr), len_ref), min_ref) in it {
            let slab_base_this = slab_base;
            let qx = f32x8::splat(*qx);
            let qy = f32x8::splat(*qy);
            let qz = f32x8::splat(*qz);
            let dots = cx * qx + cy * qy + cz * qz;

            let thresh_vec = f32x8::splat(*thr);
            let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
            let mut mask_bits = mask.to_bitmask() as u32;

            if mask_bits != 0 {
                let dots_arr: [f32; 8] = dots.into();
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    if slot != *query_slot {
                        let packed = slot_gen_map[slot as usize];
                        let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                        if bin_b == query_bin && local_b < query_local {
                            mask_bits &= mask_bits - 1;
                            continue;
                        }

                        let dot = dots_arr[lane];
                        let slab_idx = slab_base_this + *len_ref;
                        keys_slab[slab_idx].write(make_desc_key(dot, slot));
                        *len_ref += 1;
                        if UPDATE_MIN {
                            *min_ref = (*min_ref).min(dot);
                        }
                    }
                    mask_bits &= mask_bits - 1;
                }
            }

            slab_base += stride;
        }
    }

    let tail_start = full_chunks * 8;
    for i in tail_start..range_len {
        let cx = xs[i];
        let cy = ys[i];
        let cz = zs[i];
        let slot = (soa_start + i) as u32;

        let it = queries
            .iter()
            .zip(query_locals.iter())
            .zip(query_x.iter())
            .zip(query_y.iter())
            .zip(query_z.iter())
            .zip(thresholds.iter())
            .zip(lens.iter_mut())
            .zip(min_center_dot.iter_mut());

        let mut slab_base = 0usize;
        for (((((((query_slot, &query_local), qx), qy), qz), thr), len_ref), min_ref) in it {
            let slab_base_this = slab_base;
            if slot == *query_slot {
                slab_base += stride;
                continue;
            }
            let packed = slot_gen_map[slot as usize];
            let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
            if bin_b == query_bin && local_b < query_local {
                slab_base += stride;
                continue;
            }

            let dot = cx * *qx + cy * *qy + cz * *qz;
            if dot > *thr {
                let slab_idx = slab_base_this + *len_ref;
                keys_slab[slab_idx].write(make_desc_key(dot, slot));
                *len_ref += 1;
                if UPDATE_MIN {
                    *min_ref = (*min_ref).min(dot);
                }
            }

            slab_base += stride;
        }
    }
}

#[allow(dead_code)]
#[cold]
#[inline(never)]
fn dense_ring_fallback(
    grid: &CubeMapGrid,
    queries: &[u32],
    k: usize,
    stride: usize,
    ring_ranges: &[(usize, usize)],
    query_x: &[f32],
    query_y: &[f32],
    query_z: &[f32],
    security_thresholds: &[f32],
    ring_thresholds: &[f32],
    keys_slab: &mut [MaybeUninit<u64>],
    lens: &mut [usize],
    center_lens: &[usize],
) {
    for (qi, &query_slot) in queries.iter().enumerate() {
        let ring_added = lens[qi].saturating_sub(center_lens[qi]);
        let need = k.saturating_sub(center_lens[qi]);
        if ring_added >= need {
            continue;
        }
        if ring_thresholds[qi] <= security_thresholds[qi] {
            continue;
        }

        lens[qi] = center_lens[qi];
        let qx = query_x[qi];
        let qy = query_y[qi];
        let qz = query_z[qi];
        let thr = security_thresholds[qi];

        for &(soa_start, soa_end) in ring_ranges {
            let range_len = soa_end - soa_start;
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];

            for i in 0..range_len {
                let slot = (soa_start + i) as u32;
                if slot == query_slot {
                    continue;
                }
                let dot = xs[i] * qx + ys[i] * qy + zs[i] * qz;
                if dot > thr {
                    let slab_idx = qi * stride + lens[qi];
                    keys_slab[slab_idx].write(make_desc_key(dot, slot));
                    lens[qi] += 1;
                }
            }
        }
    }
}

#[cold]
#[inline(never)]
#[allow(dead_code)]
fn dense_ring_fallback_directed(
    grid: &CubeMapGrid,
    queries: &[u32],
    query_locals: &[u32],
    query_bin: u8,
    slot_gen_map: &[u32],
    local_shift: u32,
    local_mask: u32,
    k: usize,
    stride: usize,
    ring_ranges: &[(usize, usize)],
    query_x: &[f32],
    query_y: &[f32],
    query_z: &[f32],
    security_thresholds: &[f32],
    ring_thresholds: &[f32],
    keys_slab: &mut [MaybeUninit<u64>],
    lens: &mut [usize],
    center_lens: &[usize],
) {
    debug_assert_eq!(queries.len(), query_locals.len());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let ring_added = lens[qi].saturating_sub(center_lens[qi]);
        let need = k.saturating_sub(center_lens[qi]);
        if ring_added >= need {
            continue;
        }
        if ring_thresholds[qi] <= security_thresholds[qi] {
            continue;
        }

        lens[qi] = center_lens[qi];
        let qx = query_x[qi];
        let qy = query_y[qi];
        let qz = query_z[qi];
        let thr = security_thresholds[qi];
        let query_local = query_locals[qi];

        for &(soa_start, soa_end) in ring_ranges {
            let range_len = soa_end - soa_start;
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];

            for i in 0..range_len {
                let slot = (soa_start + i) as u32;
                if slot == query_slot {
                    continue;
                }
                let packed = slot_gen_map[slot as usize];
                let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                if bin_b == query_bin && local_b < query_local {
                    continue;
                }

                let dot = xs[i] * qx + ys[i] * qy + zs[i] * qz;
                if dot > thr {
                    let slab_idx = qi * stride + lens[qi];
                    keys_slab[slab_idx].write(make_desc_key(dot, slot));
                    lens[qi] += 1;
                }
            }
        }
    }
}

#[allow(dead_code)]
#[cold]
#[inline(never)]
fn packed_knn_cell_stream_large(
    grid: &CubeMapGrid,
    queries: &[u32],
    k: usize,
    scratch: &mut PackedKnnCellScratch,
    timings: &mut PackedKnnTimings,
    center_soa_start: usize,
    center_soa_end: usize,
    on_query: &mut dyn FnMut(usize, u32, &[u32], usize, f32),
) -> PackedKnnCellStatus {
    let num_queries = queries.len();

    let t_setup = PackedTimer::start();
    scratch.keys_slab.clear();
    let topk_slab_size = num_queries * k;
    if scratch.keys_slab.capacity() < topk_slab_size {
        scratch
            .keys_slab
            .reserve(topk_slab_size.saturating_sub(scratch.keys_slab.len()));
    }
    unsafe { scratch.keys_slab.set_len(topk_slab_size) };
    scratch.top_worst_key.resize(num_queries, 0);
    scratch.top_worst_key.fill(0);
    scratch.top_worst_pos.resize(num_queries, 0);
    scratch.top_worst_pos.fill(0);
    timings.add_setup(t_setup.elapsed());

    let mut push_topk = |qi: usize, key: u64| {
        let len = scratch.lens[qi];
        let base = qi * k;
        if len < k {
            scratch.keys_slab[base + len].write(key);
            let new_len = len + 1;
            scratch.lens[qi] = new_len;
            if new_len == k {
                let mut worst_key = 0u64;
                let mut worst_pos = 0usize;
                for j in 0..k {
                    let v = unsafe { scratch.keys_slab[base + j].assume_init() };
                    if v > worst_key {
                        worst_key = v;
                        worst_pos = j;
                    }
                }
                scratch.top_worst_key[qi] = worst_key;
                scratch.top_worst_pos[qi] = worst_pos;
            }
            return;
        }

        let worst_key = scratch.top_worst_key[qi];
        if key >= worst_key {
            return;
        }
        let worst_pos = scratch.top_worst_pos[qi];
        scratch.keys_slab[base + worst_pos].write(key);
        let mut new_worst_key = 0u64;
        let mut new_worst_pos = 0usize;
        for j in 0..k {
            let v = unsafe { scratch.keys_slab[base + j].assume_init() };
            if v > new_worst_key {
                new_worst_key = v;
                new_worst_pos = j;
            }
        }
        scratch.top_worst_key[qi] = new_worst_key;
        scratch.top_worst_pos[qi] = new_worst_pos;
    };

    let center_len = center_soa_end - center_soa_start;
    let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
    let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
    let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

    let t_center = PackedTimer::start();
    let full_chunks = center_len / 8;
    for chunk in 0..full_chunks {
        let i = chunk * 8;
        let cx = f32x8::from_slice(&xs[i..]);
        let cy = f32x8::from_slice(&ys[i..]);
        let cz = f32x8::from_slice(&zs[i..]);

        for (qi, &query_slot) in queries.iter().enumerate() {
            let qx = f32x8::splat(scratch.query_x[qi]);
            let qy = f32x8::splat(scratch.query_y[qi]);
            let qz = f32x8::splat(scratch.query_z[qi]);
            let dots = cx * qx + cy * qy + cz * qz;

            let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
            let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
            let mut mask_bits = mask.to_bitmask() as u32;

            if mask_bits != 0 {
                let dots_arr: [f32; 8] = dots.into();
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let slot = (center_soa_start + i + lane) as u32;
                    if slot != query_slot {
                        let dot = dots_arr[lane];
                        push_topk(qi, make_desc_key(dot, slot));
                    }
                    mask_bits &= mask_bits - 1;
                }
            }
        }
    }

    let tail_start = full_chunks * 8;
    for i in tail_start..center_len {
        let cx = xs[i];
        let cy = ys[i];
        let cz = zs[i];
        let slot = (center_soa_start + i) as u32;
        for (qi, &query_slot) in queries.iter().enumerate() {
            if slot == query_slot {
                continue;
            }
            let dot =
                cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
            if dot > scratch.security_thresholds[qi] {
                push_topk(qi, make_desc_key(dot, slot));
            }
        }
    }
    timings.add_center_pass(t_center.elapsed());

    let t_ring = PackedTimer::start();
    for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
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
                let qx = f32x8::splat(scratch.query_x[qi]);
                let qy = f32x8::splat(scratch.query_y[qi]);
                let qz = f32x8::splat(scratch.query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        if slot != query_slot {
                            let dot = dots_arr[lane];
                            push_topk(qi, make_desc_key(dot, slot));
                        }
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }

        let tail_start = full_chunks * 8;
        for i in tail_start..range_len {
            let cx = xs[i];
            let cy = ys[i];
            let cz = zs[i];
            let slot = (soa_start + i) as u32;

            let it = queries
                .iter()
                .take(num_queries)
                .zip(scratch.query_x.iter())
                .zip(scratch.query_y.iter())
                .zip(scratch.query_z.iter())
                .zip(scratch.security_thresholds.iter())
                .map(|((((q, qx), qy), qz), thr)| (q, qx, qy, qz, thr));

            for (qi, (q, qx, qy, qz, thr)) in it.enumerate() {
                if slot == *q {
                    continue;
                }

                let dot = cx * *qx + cy * *qy + cz * *qz;

                if dot > *thr {
                    push_topk(qi, make_desc_key(dot, slot));
                }
            }
        }
    }
    timings.add_ring_pass(t_ring.elapsed());

    let t_select_prep = PackedTimer::start();
    scratch.neighbors.resize(num_queries * k, u32::MAX);
    timings.add_select_prep(t_select_prep.elapsed());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let t_qprep = PackedTimer::start();
        let query_global = grid.point_indices[query_slot as usize];
        let m = scratch.lens[qi].min(k);
        timings.add_select_query_prep(t_qprep.elapsed());
        if m == 0 {
            let t_cb = PackedTimer::start();
            on_query(qi, query_global, &[], 0, scratch.security_thresholds[qi]);
            timings.add_callback(t_cb.elapsed());
            continue;
        }

        let keys_uninit = &mut scratch.keys_slab[qi * k..qi * k + m];
        let keys_slice =
            unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };
        let t_sort = PackedTimer::start();
        keys_slice.sort_unstable();
        timings.add_select_sort(t_sort.elapsed());

        let out_start = qi * k;
        let t_scatter = PackedTimer::start();
        for (neighbor, key) in scratch.neighbors[out_start..out_start + m]
            .iter_mut()
            .zip(keys_slice.iter())
        {
            *neighbor = key_to_idx(*key);
        }
        timings.add_select_scatter(t_scatter.elapsed());

        let t_cb = PackedTimer::start();
        on_query(
            qi,
            query_global,
            &scratch.neighbors[out_start..out_start + m],
            m,
            scratch.security_thresholds[qi],
        );
        timings.add_callback(t_cb.elapsed());
    }

    PackedKnnCellStatus::Ok
}

#[cold]
#[inline(never)]
#[allow(dead_code)]
fn packed_knn_cell_stream_large_directed(
    grid: &CubeMapGrid,
    queries: &[u32],
    query_locals: &[u32],
    query_bin: u8,
    slot_gen_map: &[u32],
    local_shift: u32,
    local_mask: u32,
    k: usize,
    scratch: &mut PackedKnnCellScratch,
    timings: &mut PackedKnnTimings,
    center_soa_start: usize,
    center_soa_end: usize,
    on_query: &mut dyn FnMut(usize, u32, &[u32], usize, f32),
) -> PackedKnnCellStatus {
    let num_queries = queries.len();
    debug_assert_eq!(num_queries, query_locals.len());

    let t_setup = PackedTimer::start();
    scratch.keys_slab.clear();
    let topk_slab_size = num_queries * k;
    if scratch.keys_slab.capacity() < topk_slab_size {
        scratch
            .keys_slab
            .reserve(topk_slab_size.saturating_sub(scratch.keys_slab.len()));
    }
    unsafe { scratch.keys_slab.set_len(topk_slab_size) };
    scratch.top_worst_key.resize(num_queries, 0);
    scratch.top_worst_key.fill(0);
    scratch.top_worst_pos.resize(num_queries, 0);
    scratch.top_worst_pos.fill(0);
    timings.add_setup(t_setup.elapsed());

    let mut push_topk = |qi: usize, key: u64| {
        let len = scratch.lens[qi];
        let base = qi * k;
        if len < k {
            scratch.keys_slab[base + len].write(key);
            let new_len = len + 1;
            scratch.lens[qi] = new_len;
            if new_len == k {
                let mut worst_key = 0u64;
                let mut worst_pos = 0usize;
                for j in 0..k {
                    let v = unsafe { scratch.keys_slab[base + j].assume_init() };
                    if v > worst_key {
                        worst_key = v;
                        worst_pos = j;
                    }
                }
                scratch.top_worst_key[qi] = worst_key;
                scratch.top_worst_pos[qi] = worst_pos;
            }
            return;
        }

        let worst_key = scratch.top_worst_key[qi];
        if key >= worst_key {
            return;
        }
        let worst_pos = scratch.top_worst_pos[qi];
        scratch.keys_slab[base + worst_pos].write(key);
        let mut new_worst_key = 0u64;
        let mut new_worst_pos = 0usize;
        for j in 0..k {
            let v = unsafe { scratch.keys_slab[base + j].assume_init() };
            if v > new_worst_key {
                new_worst_key = v;
                new_worst_pos = j;
            }
        }
        scratch.top_worst_key[qi] = new_worst_key;
        scratch.top_worst_pos[qi] = new_worst_pos;
    };

    let center_len = center_soa_end - center_soa_start;
    let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
    let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
    let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

    let t_center = PackedTimer::start();
    let full_chunks = center_len / 8;
    for chunk in 0..full_chunks {
        let i = chunk * 8;
        let cx = f32x8::from_slice(&xs[i..]);
        let cy = f32x8::from_slice(&ys[i..]);
        let cz = f32x8::from_slice(&zs[i..]);

        for (qi, &query_slot) in queries.iter().enumerate() {
            let qx = f32x8::splat(scratch.query_x[qi]);
            let qy = f32x8::splat(scratch.query_y[qi]);
            let qz = f32x8::splat(scratch.query_z[qi]);
            let dots = cx * qx + cy * qy + cz * qz;

            let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
            let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
            let mut mask_bits = mask.to_bitmask() as u32;

            if mask_bits != 0 {
                let dots_arr: [f32; 8] = dots.into();
                let query_local = query_locals[qi];
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let slot = (center_soa_start + i + lane) as u32;
                    if slot != query_slot {
                        let packed = slot_gen_map[slot as usize];
                        let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                        if bin_b == query_bin && local_b < query_local {
                            mask_bits &= mask_bits - 1;
                            continue;
                        }
                        let dot = dots_arr[lane];
                        push_topk(qi, make_desc_key(dot, slot));
                    }
                    mask_bits &= mask_bits - 1;
                }
            }
        }
    }

    let tail_start = full_chunks * 8;
    for i in tail_start..center_len {
        let cx = xs[i];
        let cy = ys[i];
        let cz = zs[i];
        let slot = (center_soa_start + i) as u32;
        for (qi, &query_slot) in queries.iter().enumerate() {
            if slot == query_slot {
                continue;
            }
            let query_local = query_locals[qi];
            let packed = slot_gen_map[slot as usize];
            let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
            if bin_b == query_bin && local_b < query_local {
                continue;
            }
            let dot =
                cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
            if dot > scratch.security_thresholds[qi] {
                push_topk(qi, make_desc_key(dot, slot));
            }
        }
    }
    timings.add_center_pass(t_center.elapsed());

    let t_ring = PackedTimer::start();
    for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
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
                let qx = f32x8::splat(scratch.query_x[qi]);
                let qy = f32x8::splat(scratch.query_y[qi]);
                let qz = f32x8::splat(scratch.query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    let query_local = query_locals[qi];
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        if slot != query_slot {
                            let packed = slot_gen_map[slot as usize];
                            let (bin_b, local_b) =
                                unpack_bin_local(packed, local_shift, local_mask);
                            if bin_b == query_bin && local_b < query_local {
                                mask_bits &= mask_bits - 1;
                                continue;
                            }
                            let dot = dots_arr[lane];
                            push_topk(qi, make_desc_key(dot, slot));
                        }
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }

        let tail_start = full_chunks * 8;
        for i in tail_start..range_len {
            let cx = xs[i];
            let cy = ys[i];
            let cz = zs[i];
            let slot = (soa_start + i) as u32;

            let it = queries
                .iter()
                .take(num_queries)
                .zip(query_locals.iter())
                .zip(scratch.query_x.iter())
                .zip(scratch.query_y.iter())
                .zip(scratch.query_z.iter())
                .zip(scratch.security_thresholds.iter())
                .map(|(((((q, &ql), qx), qy), qz), thr)| (q, ql, qx, qy, qz, thr));

            for (qi, (q, ql, qx, qy, qz, thr)) in it.enumerate() {
                if slot == *q {
                    continue;
                }

                let packed = slot_gen_map[slot as usize];
                let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                if bin_b == query_bin && local_b < ql {
                    continue;
                }

                let dot = cx * *qx + cy * *qy + cz * *qz;

                if dot > *thr {
                    push_topk(qi, make_desc_key(dot, slot));
                }
            }
        }
    }
    timings.add_ring_pass(t_ring.elapsed());

    let t_select_prep = PackedTimer::start();
    scratch.neighbors.resize(num_queries * k, u32::MAX);
    timings.add_select_prep(t_select_prep.elapsed());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let t_qprep = PackedTimer::start();
        let query_global = grid.point_indices[query_slot as usize];
        let m = scratch.lens[qi].min(k);
        timings.add_select_query_prep(t_qprep.elapsed());
        if m == 0 {
            let t_cb = PackedTimer::start();
            on_query(qi, query_global, &[], 0, scratch.security_thresholds[qi]);
            timings.add_callback(t_cb.elapsed());
            continue;
        }

        let keys_uninit = &mut scratch.keys_slab[qi * k..qi * k + m];
        let keys_slice =
            unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };
        let t_sort = PackedTimer::start();
        keys_slice.sort_unstable();
        timings.add_select_sort(t_sort.elapsed());

        let out_start = qi * k;
        let t_scatter = PackedTimer::start();
        for (neighbor, key) in scratch.neighbors[out_start..out_start + m]
            .iter_mut()
            .zip(keys_slice.iter())
        {
            *neighbor = key_to_idx(*key);
        }
        timings.add_select_scatter(t_scatter.elapsed());

        let t_cb = PackedTimer::start();
        on_query(
            qi,
            query_global,
            &scratch.neighbors[out_start..out_start + m],
            m,
            scratch.security_thresholds[qi],
        );
        timings.add_callback(t_cb.elapsed());
    }

    PackedKnnCellStatus::Ok
}

/// Statistics from PackedV4 batched k-NN.
#[derive(Clone, Debug, Default)]

/// Reusable scratch buffers for packed per-cell streaming queries.
pub struct PackedKnnCellScratch {
    cell_ranges: Vec<(usize, usize)>,
    keys_slab: Vec<MaybeUninit<u64>>,
    lens: Vec<usize>,
    center_lens: Vec<usize>,
    min_center_dot: Vec<f32>,
    group_queries: Vec<u32>,
    group_query_locals: Vec<u32>,
    group_cell: usize,
    group_query_bin: u8,
    chunk0_keys: Vec<Vec<u64>>,
    tail_keys: Vec<Vec<u64>>,
    chunk0_pos: Vec<usize>,
    tail_pos: Vec<usize>,
    tail_possible: Vec<bool>,
    tail_ready: Vec<bool>,
    tail_built_any: bool,
    top_worst_key: Vec<u64>,
    top_worst_pos: Vec<usize>,
    security_thresholds: Vec<f32>,
    thresholds: Vec<f32>,
    query_x: Vec<f32>,
    query_y: Vec<f32>,
    query_z: Vec<f32>,
    neighbors: Vec<u32>,
}

impl PackedKnnCellScratch {
    pub fn new() -> Self {
        Self {
            cell_ranges: Vec::with_capacity(9),
            keys_slab: Vec::new(),
            lens: Vec::new(),
            center_lens: Vec::new(),
            min_center_dot: Vec::new(),
            group_queries: Vec::new(),
            group_query_locals: Vec::new(),
            group_cell: 0,
            group_query_bin: 0,
            chunk0_keys: Vec::new(),
            tail_keys: Vec::new(),
            chunk0_pos: Vec::new(),
            tail_pos: Vec::new(),
            tail_possible: Vec::new(),
            tail_ready: Vec::new(),
            tail_built_any: false,
            top_worst_key: Vec::new(),
            top_worst_pos: Vec::new(),
            security_thresholds: Vec::new(),
            thresholds: Vec::new(),
            query_x: Vec::new(),
            query_y: Vec::new(),
            query_z: Vec::new(),
            neighbors: Vec::new(),
        }
    }

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
        self.group_query_locals.clear();
        self.group_query_locals.extend_from_slice(query_locals);

        let num_cells = 6 * grid.res * grid.res;
        if cell >= num_cells {
            return PackedKnnCellStatus::Ok;
        }

        let t_setup = PackedTimer::start();
        self.cell_ranges.clear();

        let q_start = grid.cell_offsets[cell] as usize;
        let q_end = grid.cell_offsets[cell + 1] as usize;
        self.cell_ranges.push((q_start, q_end));

        for &ncell in grid.cell_neighbors(cell) {
            if ncell == u32::MAX || ncell == cell as u32 {
                continue;
            }
            let nc = ncell as usize;
            let n_start = grid.cell_offsets[nc] as usize;
            let n_end = grid.cell_offsets[nc + 1] as usize;
            if n_start < n_end {
                self.cell_ranges.push((n_start, n_end));
            }
        }

        let mut num_candidates = 0usize;
        for &(start, end) in &self.cell_ranges {
            num_candidates += end - start;
        }
        if num_candidates > MAX_CANDIDATES_HARD {
            timings.add_setup(t_setup.elapsed());
            return PackedKnnCellStatus::SlowPath;
        }
        timings.add_setup(t_setup.elapsed());

        let ring2 = grid.cell_ring2(cell);
        let interior_planes = security_planes_3x3_interior(cell, grid);

        let t_query_cache = PackedTimer::start();
        self.query_x.resize(num_queries, 0.0);
        self.query_y.resize(num_queries, 0.0);
        self.query_z.resize(num_queries, 0.0);
        for (qi, &query_slot) in queries.iter().enumerate() {
            let slot = query_slot as usize;
            self.query_x[qi] = grid.cell_points_x[slot];
            self.query_y[qi] = grid.cell_points_y[slot];
            self.query_z[qi] = grid.cell_points_z[slot];
        }
        timings.add_query_cache(t_query_cache.elapsed());

        let t_security = PackedTimer::start();
        self.security_thresholds.clear();
        self.security_thresholds.reserve(num_queries);
        match interior_planes {
            Some(planes) => {
                for qi in 0..num_queries {
                    let qx = self.query_x[qi];
                    let qy = self.query_y[qi];
                    let qz = self.query_z[qi];

                    let mut s_min = 1.0f32;
                    for n in &planes {
                        s_min = s_min.min(n.x * qx + n.y * qy + n.z * qz);
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
                    self.query_x
                        .iter()
                        .zip(self.query_y.iter())
                        .zip(self.query_z.iter())
                        .map(|((&x, &y), &z)| outside_max_dot_xyz(x, y, z, ring2, grid)),
                );
            }
        }
        timings.add_security_thresholds(t_security.elapsed());

        self.min_center_dot.resize(num_queries, f32::INFINITY);
        self.min_center_dot.fill(f32::INFINITY);
        self.center_lens.resize(num_queries, 0);
        self.center_lens.fill(0);
        self.thresholds.resize(num_queries, 0.0);
        self.thresholds.fill(0.0);

        self.chunk0_keys.resize_with(num_queries, Vec::new);
        for v in &mut self.chunk0_keys[..num_queries] {
            v.clear();
        }
        self.chunk0_pos.resize(num_queries, 0);
        self.chunk0_pos.fill(0);

        self.tail_keys.resize_with(num_queries, Vec::new);
        for v in &mut self.tail_keys[..num_queries] {
            v.clear();
        }
        self.tail_pos.resize(num_queries, 0);
        self.tail_pos.fill(0);
        self.tail_possible.resize(num_queries, false);
        self.tail_possible.fill(false);
        self.tail_ready.resize(num_queries, false);
        self.tail_ready.fill(false);
        self.tail_built_any = false;

        let (center_soa_start, center_soa_end) = self.cell_ranges[0];
        let center_len = center_soa_end - center_soa_start;
        let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
        let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
        let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

        let t_center = PackedTimer::start();
        let queries_cover_center_cell = num_queries == center_len
            && queries
                .first()
                .copied()
                .is_some_and(|s| s as usize == center_soa_start)
            && queries
                .last()
                .copied()
                .is_some_and(|s| s as usize + 1 == center_soa_end)
            && queries.windows(2).all(|w| w[1].wrapping_sub(w[0]) == 1);

        // Fast path: when queries are the entire center cell in slot order (the live_dedup case),
        // the directed within-bin filter reduces to "skip earlier slots in this same cell".
        //
        // This avoids per-candidate bin/local unpacking and also avoids computing dots for chunks
        // that are entirely earlier than a given query.
        if queries_cover_center_cell {
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
                    let qx = f32x8::splat(self.query_x[qi]);
                    let qy = f32x8::splat(self.query_y[qi]);
                    let qz = f32x8::splat(self.query_z[qi]);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(self.security_thresholds[qi]);
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
                        self.chunk0_keys[qi].push(make_desc_key(dot, slot));
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
                    let dot = cx * self.query_x[qi] + cy * self.query_y[qi] + cz * self.query_z[qi];
                    if dot > self.security_thresholds[qi] {
                        self.chunk0_keys[qi].push(make_desc_key(dot, slot));
                        self.min_center_dot[qi] = self.min_center_dot[qi].min(dot);
                    }
                }
            }
        } else {
            // Generic path: apply directed filter via packed (bin, local) decode.
            let full_chunks = center_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let cx = f32x8::from_slice(&xs[i..]);
                let cy = f32x8::from_slice(&ys[i..]);
                let cz = f32x8::from_slice(&zs[i..]);

                for (qi, &query_slot) in queries.iter().enumerate() {
                    let qx = f32x8::splat(self.query_x[qi]);
                    let qy = f32x8::splat(self.query_y[qi]);
                    let qz = f32x8::splat(self.query_z[qi]);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(self.security_thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr: [f32; 8] = dots.into();
                    let query_local = query_locals[qi];
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (center_soa_start + i + lane) as u32;
                        if slot != query_slot {
                            let packed = slot_gen_map[slot as usize];
                            let (bin_b, local_b) =
                                unpack_bin_local(packed, local_shift, local_mask);
                            if bin_b == query_bin && local_b < query_local {
                                mask_bits &= mask_bits - 1;
                                continue;
                            }

                            let dot = dots_arr[lane];
                            self.chunk0_keys[qi].push(make_desc_key(dot, slot));
                            self.min_center_dot[qi] = self.min_center_dot[qi].min(dot);
                        }
                        mask_bits &= mask_bits - 1;
                    }
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..center_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                let slot = (center_soa_start + i) as u32;

                for (qi, &query_slot) in queries.iter().enumerate() {
                    if slot == query_slot {
                        continue;
                    }
                    let query_local = query_locals[qi];
                    let packed = slot_gen_map[slot as usize];
                    let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                    if bin_b == query_bin && local_b < query_local {
                        continue;
                    }

                    let dot = cx * self.query_x[qi] + cy * self.query_y[qi] + cz * self.query_z[qi];
                    if dot > self.security_thresholds[qi] {
                        self.chunk0_keys[qi].push(make_desc_key(dot, slot));
                        self.min_center_dot[qi] = self.min_center_dot[qi].min(dot);
                    }
                }
            }
        }
        timings.add_center_pass(t_center.elapsed());

        for (qi, v) in self.chunk0_keys.iter().take(num_queries).enumerate() {
            self.center_lens[qi] = v.len();
        }

        let t_thresholds = PackedTimer::start();
        for qi in 0..num_queries {
            let security = self.security_thresholds[qi];
            let center_len = self.center_lens[qi];
            let min_dot = self.min_center_dot[qi];
            self.thresholds[qi] = if center_len > 0 {
                security.max(min_dot - 1e-6)
            } else {
                security
            };
            self.tail_possible[qi] = self.thresholds[qi] > security;
        }
        timings.add_ring_thresholds(t_thresholds.elapsed());

        let t_ring = PackedTimer::start();
        for &(soa_start, soa_end) in &self.cell_ranges[1..] {
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
                    let qx = f32x8::splat(self.query_x[qi]);
                    let qy = f32x8::splat(self.query_y[qi]);
                    let qz = f32x8::splat(self.query_z[qi]);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(self.thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr: [f32; 8] = dots.into();
                    let query_local = query_locals[qi];
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        if slot != query_slot {
                            let packed = slot_gen_map[slot as usize];
                            let (bin_b, local_b) =
                                unpack_bin_local(packed, local_shift, local_mask);
                            if bin_b == query_bin && local_b < query_local {
                                mask_bits &= mask_bits - 1;
                                continue;
                            }

                            let dot = dots_arr[lane];
                            self.chunk0_keys[qi].push(make_desc_key(dot, slot));
                        }
                        mask_bits &= mask_bits - 1;
                    }
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                let slot = (soa_start + i) as u32;

                for (qi, &query_slot) in queries.iter().enumerate() {
                    if slot == query_slot {
                        continue;
                    }

                    let query_local = query_locals[qi];
                    let packed = slot_gen_map[slot as usize];
                    let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                    if bin_b == query_bin && local_b < query_local {
                        continue;
                    }

                    let dot = cx * self.query_x[qi] + cy * self.query_y[qi] + cz * self.query_z[qi];
                    if dot > self.thresholds[qi] {
                        self.chunk0_keys[qi].push(make_desc_key(dot, slot));
                    }
                }
            }
        }
        timings.add_ring_pass(t_ring.elapsed());

        PackedKnnCellStatus::Ok
    }

    pub fn ensure_tail_directed_for(
        &mut self,
        qi: usize,
        grid: &CubeMapGrid,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
        timings: &mut PackedKnnTimings,
    ) {
        let Some(tail_ready) = self.tail_ready.get(qi).copied() else {
            return;
        };
        if tail_ready {
            return;
        }
        if !self.tail_built_any {
            self.tail_built_any = true;
            timings.inc_tail_builds();
        }
        self.tail_ready[qi] = true;

        let num_queries = self.group_queries.len();
        debug_assert_eq!(num_queries, self.group_query_locals.len());

        self.tail_keys[qi].clear();
        self.tail_pos[qi] = 0;
        debug_assert!(self.tail_possible.get(qi).copied().unwrap_or(false));

        let t_tail = PackedTimer::start();
        for &(soa_start, soa_end) in &self.cell_ranges[1..] {
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

                let query_slot = self.group_queries[qi];

                let qx = f32x8::splat(self.query_x[qi]);
                let qy = f32x8::splat(self.query_y[qi]);
                let qz = f32x8::splat(self.query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

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
                let query_local = self.group_query_locals[qi];
                while tail_bits != 0 {
                    let lane = tail_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    if slot != query_slot {
                        let packed = slot_gen_map[slot as usize];
                        let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                        if bin_b == self.group_query_bin && local_b < query_local {
                            tail_bits &= tail_bits - 1;
                            continue;
                        }

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

                let query_slot = self.group_queries[qi];
                if slot == query_slot {
                    continue;
                }

                let query_local = self.group_query_locals[qi];
                let packed = slot_gen_map[slot as usize];
                let (bin_b, local_b) = unpack_bin_local(packed, local_shift, local_mask);
                if bin_b == self.group_query_bin && local_b < query_local {
                    continue;
                }

                let dot = cx * self.query_x[qi] + cy * self.query_y[qi] + cz * self.query_z[qi];
                if dot > self.security_thresholds[qi] && dot <= self.thresholds[qi] {
                    self.tail_keys[qi].push(make_desc_key(dot, slot));
                }
            }
        }
        timings.add_ring_fallback(t_tail.elapsed());

        if self.tail_keys[qi].is_empty() {
            self.tail_possible[qi] = false;
        }
    }

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
                let t_qprep = PackedTimer::start();
                let keys = &mut self.chunk0_keys.get_mut(qi)?;
                let start = *self.chunk0_pos.get(qi)?;
                if start >= keys.len() {
                    return None;
                }
                let remaining = &mut keys[start..];
                let n = k.min(out.len()).min(remaining.len());
                if n == 0 {
                    return None;
                }
                timings.add_select_query_prep(t_qprep.elapsed());
                if remaining.len() > n {
                    let t_part = PackedTimer::start();
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t_part.elapsed());
                }
                let t_sort = PackedTimer::start();
                remaining[..n].sort_unstable();
                timings.add_select_sort(t_sort.elapsed());
                let t_scatter = PackedTimer::start();
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t_scatter.elapsed());
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
                    self.tail_ready.get(qi).copied().unwrap_or(false),
                    "tail stage requested before ensure_tail"
                );
                let t_qprep = PackedTimer::start();
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
                timings.add_select_query_prep(t_qprep.elapsed());
                if remaining.len() > n {
                    let t_part = PackedTimer::start();
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t_part.elapsed());
                }
                let t_sort = PackedTimer::start();
                remaining[..n].sort_unstable();
                timings.add_select_sort(t_sort.elapsed());
                let t_scatter = PackedTimer::start();
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t_scatter.elapsed());
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

/// PackedV4 per-cell k-NN for a subset of queries, streaming results to a callback.
///
/// Queries are assumed to lie in the center cell (same as the full packed path),
/// but may be a strict subset of that cell's points.
#[allow(dead_code)]
pub fn packed_knn_cell_stream(
    grid: &CubeMapGrid,
    cell: usize,
    queries: &[u32],
    k: usize,
    scratch: &mut PackedKnnCellScratch,
    timings: &mut PackedKnnTimings,
    mut on_query: impl FnMut(usize, u32, &[u32], usize, f32),
) -> PackedKnnCellStatus {
    timings.clear();

    let num_queries = queries.len();
    if num_queries == 0 || k == 0 {
        return PackedKnnCellStatus::Ok;
    }

    let num_cells = 6 * grid.res * grid.res;
    if cell >= num_cells {
        return PackedKnnCellStatus::Ok;
    }

    let t_setup = PackedTimer::start();
    scratch.cell_ranges.clear();

    let q_start = grid.cell_offsets[cell] as usize;
    let q_end = grid.cell_offsets[cell + 1] as usize;
    scratch.cell_ranges.push((q_start, q_end));

    for &ncell in grid.cell_neighbors(cell) {
        if ncell == u32::MAX || ncell == cell as u32 {
            continue;
        }
        let nc = ncell as usize;
        let n_start = grid.cell_offsets[nc] as usize;
        let n_end = grid.cell_offsets[nc + 1] as usize;
        if n_start < n_end {
            scratch.cell_ranges.push((n_start, n_end));
        }
    }

    let mut num_candidates = 0usize;
    for &(start, end) in &scratch.cell_ranges {
        num_candidates += end - start;
    }
    if num_candidates == 0 {
        timings.add_setup(t_setup.elapsed());
        return PackedKnnCellStatus::Ok;
    }
    if num_candidates > MAX_CANDIDATES_HARD {
        timings.add_setup(t_setup.elapsed());
        return PackedKnnCellStatus::SlowPath;
    }

    let ring2 = grid.cell_ring2(cell);
    let interior_planes = security_planes_3x3_interior(cell, grid);

    scratch.lens.resize(num_queries, 0);
    scratch.lens.fill(0);
    scratch.min_center_dot.resize(num_queries, f32::INFINITY);
    scratch.min_center_dot.fill(f32::INFINITY);

    timings.add_setup(t_setup.elapsed());

    let t_query_cache = PackedTimer::start();
    scratch.query_x.resize(num_queries, 0.0);
    scratch.query_y.resize(num_queries, 0.0);
    scratch.query_z.resize(num_queries, 0.0);
    for (qi, &query_slot) in queries.iter().enumerate() {
        let slot = query_slot as usize;
        scratch.query_x[qi] = grid.cell_points_x[slot];
        scratch.query_y[qi] = grid.cell_points_y[slot];
        scratch.query_z[qi] = grid.cell_points_z[slot];
    }
    timings.add_query_cache(t_query_cache.elapsed());

    let t_security = PackedTimer::start();
    scratch.security_thresholds.clear();
    scratch.security_thresholds.reserve(num_queries);
    match interior_planes {
        Some(planes) => {
            for qi in 0..num_queries {
                let qx = scratch.query_x[qi];
                let qy = scratch.query_y[qi];
                let qz = scratch.query_z[qi];

                let mut s_min = 1.0f32;
                for n in &planes {
                    s_min = s_min.min(n.x * qx + n.y * qy + n.z * qz);
                }

                let security = if s_min > 0.0 && s_min.is_finite() {
                    const PAD: f32 = 1e-6;
                    let s = (s_min - PAD).clamp(0.0, 1.0);
                    (1.0 - s * s).max(0.0).sqrt()
                } else {
                    outside_max_dot_xyz(qx, qy, qz, ring2, grid)
                };
                scratch.security_thresholds.push(security);
            }
        }
        None => {
            scratch.security_thresholds.extend(
                scratch
                    .query_x
                    .iter()
                    .zip(scratch.query_y.iter())
                    .zip(scratch.query_z.iter())
                    .map(|((&x, &y), &z)| outside_max_dot_xyz(x, y, z, ring2, grid)),
            );
        }
    }
    timings.add_security_thresholds(t_security.elapsed());

    let stride = num_candidates;
    let dense_slab_size = num_queries * stride;
    let use_dense = num_candidates <= MAX_CANDIDATES_FAST && dense_slab_size <= MAX_KEYS_SLAB_FAST;

    let (center_soa_start, center_soa_end) = scratch.cell_ranges[0];
    let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
    let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
    let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

    // === Large-candidate path: streaming top-k per query (O(num_queries*k) memory).
    // Keeps PackedV4 behavior (dot > security threshold), but avoids dense (q*c) slab.
    if !use_dense {
        let on_query = &mut on_query as &mut dyn FnMut(usize, u32, &[u32], usize, f32);
        return packed_knn_cell_stream_large(
            grid,
            queries,
            k,
            scratch,
            timings,
            center_soa_start,
            center_soa_end,
            on_query,
        );
    }

    // === Dense slab path (O(num_queries*num_candidates) memory) for small cells.
    let t_setup = PackedTimer::start();
    scratch.keys_slab.clear();
    if scratch.keys_slab.capacity() < dense_slab_size {
        scratch
            .keys_slab
            .reserve(dense_slab_size.saturating_sub(scratch.keys_slab.len()));
    }
    unsafe { scratch.keys_slab.set_len(dense_slab_size) };
    timings.add_setup(t_setup.elapsed());

    let t_center = PackedTimer::start();
    dense_scan_range::<true>(
        stride,
        center_soa_start,
        xs,
        ys,
        zs,
        queries,
        &scratch.query_x,
        &scratch.query_y,
        &scratch.query_z,
        &scratch.security_thresholds,
        &mut scratch.keys_slab,
        &mut scratch.lens,
        &mut scratch.min_center_dot,
    );
    timings.add_center_pass(t_center.elapsed());

    scratch.center_lens.resize(num_queries, 0);
    scratch.center_lens.copy_from_slice(&scratch.lens);

    let t_thresholds = PackedTimer::start();
    scratch.thresholds.clear();
    scratch.thresholds.reserve(num_queries);
    scratch.thresholds.extend(
        scratch
            .center_lens
            .iter()
            .zip(scratch.security_thresholds.iter())
            .zip(scratch.min_center_dot.iter())
            .map(|((&center_len, &security), &min_dot)| {
                if center_len > 0 {
                    security.max(min_dot - 1e-6)
                } else {
                    security
                }
            }),
    );
    timings.add_ring_thresholds(t_thresholds.elapsed());

    let t_ring = PackedTimer::start();
    for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
        let xs = &grid.cell_points_x[soa_start..soa_end];
        let ys = &grid.cell_points_y[soa_start..soa_end];
        let zs = &grid.cell_points_z[soa_start..soa_end];
        dense_scan_range::<false>(
            stride,
            soa_start,
            xs,
            ys,
            zs,
            queries,
            &scratch.query_x,
            &scratch.query_y,
            &scratch.query_z,
            &scratch.thresholds,
            &mut scratch.keys_slab,
            &mut scratch.lens,
            &mut scratch.min_center_dot,
        );
    }
    timings.add_ring_pass(t_ring.elapsed());

    let t_fallback = PackedTimer::start();
    dense_ring_fallback(
        grid,
        queries,
        k,
        stride,
        &scratch.cell_ranges[1..],
        &scratch.query_x,
        &scratch.query_y,
        &scratch.query_z,
        &scratch.security_thresholds,
        &scratch.thresholds,
        &mut scratch.keys_slab,
        &mut scratch.lens,
        &scratch.center_lens,
    );
    timings.add_ring_fallback(t_fallback.elapsed());

    let t_select_prep = PackedTimer::start();
    scratch.neighbors.resize(num_queries * k, u32::MAX);
    timings.add_select_prep(t_select_prep.elapsed());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let t_qprep = PackedTimer::start();
        let query_global = grid.point_indices[query_slot as usize];
        let m = scratch.lens[qi];
        let keys_uninit = &mut scratch.keys_slab[qi * stride..qi * stride + m];
        let keys_slice =
            unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };

        let k_actual = k.min(m);
        timings.add_select_query_prep(t_qprep.elapsed());
        if k_actual > 0 {
            if m > k_actual {
                let t_part = PackedTimer::start();
                keys_slice.select_nth_unstable(k_actual - 1);
                timings.add_select_partition(t_part.elapsed());
            }
            let t_sort = PackedTimer::start();
            keys_slice[..k_actual].sort_unstable();
            timings.add_select_sort(t_sort.elapsed());

            let out_start = qi * k;
            let t_scatter = PackedTimer::start();
            for (neighbor, key) in scratch.neighbors[out_start..out_start + k_actual]
                .iter_mut()
                .zip(keys_slice.iter())
            {
                *neighbor = key_to_idx(*key);
            }
            timings.add_select_scatter(t_scatter.elapsed());
        }

        let out_start = qi * k;
        let out_end = out_start + k_actual;
        let t_cb = PackedTimer::start();
        on_query(
            qi,
            query_global,
            &scratch.neighbors[out_start..out_end],
            k_actual,
            scratch.security_thresholds[qi],
        );
        timings.add_callback(t_cb.elapsed());
    }

    PackedKnnCellStatus::Ok
}

/// PackedV4 per-cell k-NN with directed within-bin candidate filtering.
///
/// Filters out any candidate slot where `(bin == query_bin && local < query_local[qi])`.
/// Cross-bin candidates are always considered.
///
/// This is intended for the live-dedup directed edgecheck scheme, where earlier-local
/// same-bin neighbors are sourced exclusively from incoming edgechecks.
#[allow(dead_code)]
pub fn packed_knn_cell_stream_directed(
    grid: &CubeMapGrid,
    cell: usize,
    queries: &[u32],
    query_locals: &[u32],
    query_bin: u8,
    slot_gen_map: &[u32],
    local_shift: u32,
    local_mask: u32,
    k: usize,
    scratch: &mut PackedKnnCellScratch,
    timings: &mut PackedKnnTimings,
    mut on_query: impl FnMut(usize, u32, &[u32], usize, f32),
) -> PackedKnnCellStatus {
    timings.clear();

    let num_queries = queries.len();
    if num_queries == 0 || k == 0 {
        return PackedKnnCellStatus::Ok;
    }
    debug_assert_eq!(
        num_queries,
        query_locals.len(),
        "query_locals must match queries length"
    );

    let num_cells = 6 * grid.res * grid.res;
    if cell >= num_cells {
        return PackedKnnCellStatus::Ok;
    }

    let t_setup = PackedTimer::start();
    scratch.cell_ranges.clear();

    let q_start = grid.cell_offsets[cell] as usize;
    let q_end = grid.cell_offsets[cell + 1] as usize;
    scratch.cell_ranges.push((q_start, q_end));

    for &ncell in grid.cell_neighbors(cell) {
        if ncell == u32::MAX || ncell == cell as u32 {
            continue;
        }
        let nc = ncell as usize;
        let n_start = grid.cell_offsets[nc] as usize;
        let n_end = grid.cell_offsets[nc + 1] as usize;
        if n_start < n_end {
            scratch.cell_ranges.push((n_start, n_end));
        }
    }

    let mut num_candidates = 0usize;
    for &(start, end) in &scratch.cell_ranges {
        num_candidates += end - start;
    }
    if num_candidates == 0 {
        timings.add_setup(t_setup.elapsed());
        return PackedKnnCellStatus::Ok;
    }
    if num_candidates > MAX_CANDIDATES_HARD {
        timings.add_setup(t_setup.elapsed());
        return PackedKnnCellStatus::SlowPath;
    }

    let ring2 = grid.cell_ring2(cell);
    let interior_planes = security_planes_3x3_interior(cell, grid);

    scratch.lens.resize(num_queries, 0);
    scratch.lens.fill(0);
    scratch.min_center_dot.resize(num_queries, f32::INFINITY);
    scratch.min_center_dot.fill(f32::INFINITY);

    timings.add_setup(t_setup.elapsed());

    let t_query_cache = PackedTimer::start();
    scratch.query_x.resize(num_queries, 0.0);
    scratch.query_y.resize(num_queries, 0.0);
    scratch.query_z.resize(num_queries, 0.0);
    for (qi, &query_slot) in queries.iter().enumerate() {
        let slot = query_slot as usize;
        scratch.query_x[qi] = grid.cell_points_x[slot];
        scratch.query_y[qi] = grid.cell_points_y[slot];
        scratch.query_z[qi] = grid.cell_points_z[slot];
    }
    timings.add_query_cache(t_query_cache.elapsed());

    let t_security = PackedTimer::start();
    scratch.security_thresholds.clear();
    scratch.security_thresholds.reserve(num_queries);
    match interior_planes {
        Some(planes) => {
            for qi in 0..num_queries {
                let qx = scratch.query_x[qi];
                let qy = scratch.query_y[qi];
                let qz = scratch.query_z[qi];

                let mut s_min = 1.0f32;
                for n in &planes {
                    s_min = s_min.min(n.x * qx + n.y * qy + n.z * qz);
                }

                let security = if s_min > 0.0 && s_min.is_finite() {
                    const PAD: f32 = 1e-6;
                    let s = (s_min - PAD).clamp(0.0, 1.0);
                    (1.0 - s * s).max(0.0).sqrt()
                } else {
                    outside_max_dot_xyz(qx, qy, qz, ring2, grid)
                };
                scratch.security_thresholds.push(security);
            }
        }
        None => {
            scratch.security_thresholds.extend(
                scratch
                    .query_x
                    .iter()
                    .zip(scratch.query_y.iter())
                    .zip(scratch.query_z.iter())
                    .map(|((&x, &y), &z)| outside_max_dot_xyz(x, y, z, ring2, grid)),
            );
        }
    }
    timings.add_security_thresholds(t_security.elapsed());

    let stride = num_candidates;
    let dense_slab_size = num_queries * stride;
    let use_dense = num_candidates <= MAX_CANDIDATES_FAST && dense_slab_size <= MAX_KEYS_SLAB_FAST;

    let (center_soa_start, center_soa_end) = scratch.cell_ranges[0];
    let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
    let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
    let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

    // === Large-candidate path: streaming top-k per query (O(num_queries*k) memory).
    if !use_dense {
        let on_query = &mut on_query as &mut dyn FnMut(usize, u32, &[u32], usize, f32);
        return packed_knn_cell_stream_large_directed(
            grid,
            queries,
            query_locals,
            query_bin,
            slot_gen_map,
            local_shift,
            local_mask,
            k,
            scratch,
            timings,
            center_soa_start,
            center_soa_end,
            on_query,
        );
    }

    // === Dense slab path (O(num_queries*num_candidates) memory) for small cells.
    let t_setup = PackedTimer::start();
    scratch.keys_slab.clear();
    if scratch.keys_slab.capacity() < dense_slab_size {
        scratch
            .keys_slab
            .reserve(dense_slab_size.saturating_sub(scratch.keys_slab.len()));
    }
    unsafe { scratch.keys_slab.set_len(dense_slab_size) };
    timings.add_setup(t_setup.elapsed());

    let t_center = PackedTimer::start();
    dense_scan_range_directed::<true>(
        stride,
        center_soa_start,
        xs,
        ys,
        zs,
        queries,
        query_locals,
        query_bin,
        slot_gen_map,
        local_shift,
        local_mask,
        &scratch.query_x,
        &scratch.query_y,
        &scratch.query_z,
        &scratch.security_thresholds,
        &mut scratch.keys_slab,
        &mut scratch.lens,
        &mut scratch.min_center_dot,
    );
    timings.add_center_pass(t_center.elapsed());

    scratch.center_lens.resize(num_queries, 0);
    scratch.center_lens.copy_from_slice(&scratch.lens);

    let t_thresholds = PackedTimer::start();
    scratch.thresholds.clear();
    scratch.thresholds.reserve(num_queries);
    scratch.thresholds.extend(
        scratch
            .center_lens
            .iter()
            .zip(scratch.security_thresholds.iter())
            .zip(scratch.min_center_dot.iter())
            .map(|((&center_len, &security), &min_dot)| {
                if center_len > 0 {
                    security.max(min_dot - 1e-6)
                } else {
                    security
                }
            }),
    );
    timings.add_ring_thresholds(t_thresholds.elapsed());

    let t_ring = PackedTimer::start();
    for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
        let xs = &grid.cell_points_x[soa_start..soa_end];
        let ys = &grid.cell_points_y[soa_start..soa_end];
        let zs = &grid.cell_points_z[soa_start..soa_end];
        dense_scan_range_directed::<false>(
            stride,
            soa_start,
            xs,
            ys,
            zs,
            queries,
            query_locals,
            query_bin,
            slot_gen_map,
            local_shift,
            local_mask,
            &scratch.query_x,
            &scratch.query_y,
            &scratch.query_z,
            &scratch.thresholds,
            &mut scratch.keys_slab,
            &mut scratch.lens,
            &mut scratch.min_center_dot,
        );
    }
    timings.add_ring_pass(t_ring.elapsed());

    let t_fallback = PackedTimer::start();
    dense_ring_fallback_directed(
        grid,
        queries,
        query_locals,
        query_bin,
        slot_gen_map,
        local_shift,
        local_mask,
        k,
        stride,
        &scratch.cell_ranges[1..],
        &scratch.query_x,
        &scratch.query_y,
        &scratch.query_z,
        &scratch.security_thresholds,
        &scratch.thresholds,
        &mut scratch.keys_slab,
        &mut scratch.lens,
        &scratch.center_lens,
    );
    timings.add_ring_fallback(t_fallback.elapsed());

    let t_select_prep = PackedTimer::start();
    scratch.neighbors.resize(num_queries * k, u32::MAX);
    timings.add_select_prep(t_select_prep.elapsed());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let t_qprep = PackedTimer::start();
        let query_global = grid.point_indices[query_slot as usize];
        let m = scratch.lens[qi];
        let keys_uninit = &mut scratch.keys_slab[qi * stride..qi * stride + m];
        let keys_slice =
            unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };

        let k_actual = k.min(m);
        timings.add_select_query_prep(t_qprep.elapsed());
        if k_actual > 0 {
            if m > k_actual {
                let t_part = PackedTimer::start();
                keys_slice.select_nth_unstable(k_actual - 1);
                timings.add_select_partition(t_part.elapsed());
            }
            let t_sort = PackedTimer::start();
            keys_slice[..k_actual].sort_unstable();
            timings.add_select_sort(t_sort.elapsed());

            let out_start = qi * k;
            let t_scatter = PackedTimer::start();
            for (neighbor, key) in scratch.neighbors[out_start..out_start + k_actual]
                .iter_mut()
                .zip(keys_slice.iter())
            {
                *neighbor = key_to_idx(*key);
            }
            timings.add_select_scatter(t_scatter.elapsed());
        }

        let out_start = qi * k;
        let out_end = out_start + k_actual;
        let t_cb = PackedTimer::start();
        on_query(
            qi,
            query_global,
            &scratch.neighbors[out_start..out_end],
            k_actual,
            scratch.security_thresholds[qi],
        );
        timings.add_callback(t_cb.elapsed());
    }

    PackedKnnCellStatus::Ok
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
