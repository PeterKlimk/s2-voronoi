//! Batched k-NN using PackedV4 filtering for unit vectors on a cube-map grid.
//!
//! Dominant mode: PackedV4 (batched, cell-local, SIMD dot products).

use super::CubeMapGrid;
use glam::Vec3;
use std::mem::MaybeUninit;
use std::simd::f32x8;
use std::simd::{cmp::SimdPartialOrd, Mask};
use std::time::Duration;

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
    pub select_sort: Duration,
    /// Time spent inside the per-query callback (`on_query`), excluded from `packed_knn` overhead.
    pub callback: Duration,
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
    pub fn add_select_sort(&mut self, d: Duration) {
        self.select_sort += d;
    }

    #[inline]
    pub fn add_callback(&mut self, d: Duration) {
        self.callback += d;
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
            + self.select_sort
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
    pub fn add_select_sort(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_callback(&mut self, _d: Duration) {}
}

// Fast path stores candidate keys in a dense slab of size (num_queries * num_candidates).
// For typical KNN_GRID_TARGET_DENSITY (32) this is small (~9*density^2 ≈ 9k),
// but it grows quadratically with density. Keep a generous per-cell cap while
// still preventing pathological allocations.
const MAX_CANDIDATES_FAST: usize = 1024;
const MAX_KEYS_SLAB_FAST: usize = 200_000;
const MAX_CANDIDATES_HARD: usize = 65_536;

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
    timings.add_select_sort(t_select_prep.elapsed());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let query_global = grid.point_indices[query_slot as usize];
        let m = scratch.lens[qi].min(k);
        if m == 0 {
            let t_cb = PackedTimer::start();
            on_query(qi, query_global, &[], 0, scratch.security_thresholds[qi]);
            timings.add_callback(t_cb.elapsed());
            continue;
        }

        let t_select = PackedTimer::start();
        let keys_uninit = &mut scratch.keys_slab[qi * k..qi * k + m];
        let keys_slice =
            unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };
        keys_slice.sort_unstable();

        let out_start = qi * k;
        for (neighbor, key) in scratch.neighbors[out_start..out_start + m]
            .iter_mut()
            .zip(keys_slice.iter())
        {
            *neighbor = key_to_idx(*key);
        }
        timings.add_select_sort(t_select.elapsed());

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedKnnCellStatus {
    Ok,
    SlowPath,
}

/// PackedV4 per-cell k-NN for a subset of queries, streaming results to a callback.
///
/// Queries are assumed to lie in the center cell (same as the full packed path),
/// but may be a strict subset of that cell's points.
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
    scratch.security_thresholds.extend(
        scratch
            .query_x
            .iter()
            .zip(scratch.query_y.iter())
            .zip(scratch.query_z.iter())
            .map(|((&x, &y), &z)| outside_max_dot_xyz(x, y, z, ring2, grid)),
    );
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
    timings.add_select_sort(t_select_prep.elapsed());
    for (qi, &query_slot) in queries.iter().enumerate() {
        let query_global = grid.point_indices[query_slot as usize];
        let m = scratch.lens[qi];
        let keys_uninit = &mut scratch.keys_slab[qi * stride..qi * stride + m];
        let keys_slice =
            unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };

        let k_actual = k.min(m);
        if k_actual > 0 {
            let t_select = PackedTimer::start();
            if m > k_actual {
                keys_slice.select_nth_unstable(k_actual - 1);
            }
            keys_slice[..k_actual].sort_unstable();

            let out_start = qi * k;
            for (neighbor, key) in scratch.neighbors[out_start..out_start + k_actual]
                .iter_mut()
                .zip(keys_slice.iter())
            {
                *neighbor = key_to_idx(*key);
            }
            timings.add_select_sort(t_select.elapsed());
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
