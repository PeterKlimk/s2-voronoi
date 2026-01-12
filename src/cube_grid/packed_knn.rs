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

/// Result of packed k-NN: for each point, the indices of its k nearest neighbors.
/// Layout: [p0_n0, p0_n1, ..., p0_n(k-1), p1_n0, ...]
pub struct PackedKnnResult {
    pub neighbors: Vec<u32>,
    pub k: usize,
}

impl PackedKnnResult {
    #[inline]
    pub fn get(&self, point_idx: usize) -> &[u32] {
        let start = point_idx * self.k;
        &self.neighbors[start..start + self.k]
    }

    /// Returns the valid prefix of `get()` (stops at the first `u32::MAX` sentinel).
    ///
    /// Packed mode may return fewer than `k` neighbors for some points.
    #[inline]
    pub fn get_valid(&self, point_idx: usize) -> &[u32] {
        let slice = self.get(point_idx);
        let len = slice
            .iter()
            .position(|&idx| idx == u32::MAX)
            .unwrap_or(self.k);
        &slice[..len]
    }
}

/// Statistics from PackedV4 batched k-NN.
#[derive(Clone, Debug, Default)]
pub struct PackedKnnStats {
    pub total_candidates: u64,
    pub filtered_out: u64,
    pub fallback_queries: u64,
    pub slow_path_cells: u64,
    pub under_k_count: u64,
    pub num_queries: u64,
}

impl PackedKnnStats {
    pub fn filter_rate(&self) -> f64 {
        if self.total_candidates == 0 {
            0.0
        } else {
            self.filtered_out as f64 / self.total_candidates as f64
        }
    }
}

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
    points: &[Vec3],
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
    for (qi, &query_idx) in queries.iter().enumerate() {
        let q = points[query_idx as usize];
        scratch.query_x[qi] = q.x;
        scratch.query_y[qi] = q.y;
        scratch.query_z[qi] = q.z;
    }
    timings.add_query_cache(t_query_cache.elapsed());

    let t_security = PackedTimer::start();
    scratch.security_thresholds.clear();
    scratch.security_thresholds.reserve(num_queries);
    for qi in 0..num_queries {
        scratch.security_thresholds.push(outside_max_dot_xyz(
            scratch.query_x[qi],
            scratch.query_y[qi],
            scratch.query_z[qi],
            ring2,
            grid,
        ));
    }
    timings.add_security_thresholds(t_security.elapsed());

    let stride = num_candidates;
    let dense_slab_size = num_queries * stride;
    let use_dense = num_candidates <= MAX_CANDIDATES_FAST && dense_slab_size <= MAX_KEYS_SLAB_FAST;

    let (center_soa_start, center_soa_end) = scratch.cell_ranges[0];
    let center_len = center_soa_end - center_soa_start;
    let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
    let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
    let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

    // === Large-candidate path: streaming top-k per query (O(num_queries*k) memory).
    // Keeps PackedV4 behavior (dot > security threshold), but avoids dense (q*c) slab.
    if !use_dense {
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

        let t_center = PackedTimer::start();
        let full_chunks = center_len / 8;
        for chunk in 0..full_chunks {
            let i = chunk * 8;
            let cx = f32x8::from_slice(&xs[i..]);
            let cy = f32x8::from_slice(&ys[i..]);
            let cz = f32x8::from_slice(&zs[i..]);

            for qi in 0..num_queries {
                let qx = f32x8::splat(scratch.query_x[qi]);
                let qy = f32x8::splat(scratch.query_y[qi]);
                let qz = f32x8::splat(scratch.query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    let query_idx = queries[qi];
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let cand_global = grid.point_indices[center_soa_start + i + lane];
                        if cand_global != query_idx {
                            let dot = dots_arr[lane];
                            push_topk(qi, make_desc_key(dot, cand_global));
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
            let cand_global = grid.point_indices[center_soa_start + i];
            for qi in 0..num_queries {
                if cand_global == queries[qi] {
                    continue;
                }
                let dot =
                    cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
                if dot > scratch.security_thresholds[qi] {
                    push_topk(qi, make_desc_key(dot, cand_global));
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

                for qi in 0..num_queries {
                    let qx = f32x8::splat(scratch.query_x[qi]);
                    let qy = f32x8::splat(scratch.query_y[qi]);
                    let qz = f32x8::splat(scratch.query_z[qi]);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;

                    if mask_bits != 0 {
                        let dots_arr: [f32; 8] = dots.into();
                        let query_idx = queries[qi];
                        while mask_bits != 0 {
                            let lane = mask_bits.trailing_zeros() as usize;
                            let cand_global = grid.point_indices[soa_start + i + lane];
                            if cand_global != query_idx {
                                let dot = dots_arr[lane];
                                push_topk(qi, make_desc_key(dot, cand_global));
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
                let cand_global = grid.point_indices[soa_start + i];
                for qi in 0..num_queries {
                    if cand_global == queries[qi] {
                        continue;
                    }
                    let dot = cx * scratch.query_x[qi]
                        + cy * scratch.query_y[qi]
                        + cz * scratch.query_z[qi];
                    if dot > scratch.security_thresholds[qi] {
                        push_topk(qi, make_desc_key(dot, cand_global));
                    }
                }
            }
        }
        timings.add_ring_pass(t_ring.elapsed());

        let t_select_prep = PackedTimer::start();
        scratch.neighbors.resize(num_queries * k, u32::MAX);
        timings.add_select_sort(t_select_prep.elapsed());
        for (qi, &query_idx) in queries.iter().enumerate() {
            let m = scratch.lens[qi].min(k);
            if m == 0 {
                let t_cb = PackedTimer::start();
                on_query(qi, query_idx, &[], 0, scratch.security_thresholds[qi]);
                timings.add_callback(t_cb.elapsed());
                continue;
            }

            let t_select = PackedTimer::start();
            let keys_uninit = &mut scratch.keys_slab[qi * k..qi * k + m];
            let keys_slice =
                unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };
            keys_slice.sort_unstable();

            let out_start = qi * k;
            for j in 0..m {
                scratch.neighbors[out_start + j] = key_to_idx(keys_slice[j]);
            }
            timings.add_select_sort(t_select.elapsed());

            let t_cb = PackedTimer::start();
            on_query(
                qi,
                query_idx,
                &scratch.neighbors[out_start..out_start + m],
                m,
                scratch.security_thresholds[qi],
            );
            timings.add_callback(t_cb.elapsed());
        }

        return PackedKnnCellStatus::Ok;
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
    let full_chunks = center_len / 8;
    for chunk in 0..full_chunks {
        let i = chunk * 8;
        let cx = f32x8::from_slice(&xs[i..]);
        let cy = f32x8::from_slice(&ys[i..]);
        let cz = f32x8::from_slice(&zs[i..]);

        for qi in 0..num_queries {
            let qx = f32x8::splat(scratch.query_x[qi]);
            let qy = f32x8::splat(scratch.query_y[qi]);
            let qz = f32x8::splat(scratch.query_z[qi]);
            let dots = cx * qx + cy * qy + cz * qz;

            let thresh_vec = f32x8::splat(scratch.security_thresholds[qi]);
            let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
            let mut mask_bits = mask.to_bitmask() as u32;

            if mask_bits != 0 {
                let dots_arr: [f32; 8] = dots.into();
                let query_idx = queries[qi];
                while mask_bits != 0 {
                    let lane = mask_bits.trailing_zeros() as usize;
                    let cand_global = grid.point_indices[center_soa_start + i + lane];
                    if cand_global != query_idx {
                        let dot = dots_arr[lane];
                        let slab_idx = qi * stride + scratch.lens[qi];
                        scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                        scratch.lens[qi] += 1;
                        scratch.min_center_dot[qi] = scratch.min_center_dot[qi].min(dot);
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
        let cand_global = grid.point_indices[center_soa_start + i];
        for qi in 0..num_queries {
            if cand_global == queries[qi] {
                continue;
            }
            let dot =
                cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
            if dot > scratch.security_thresholds[qi] {
                let slab_idx = qi * stride + scratch.lens[qi];
                scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                scratch.lens[qi] += 1;
                scratch.min_center_dot[qi] = scratch.min_center_dot[qi].min(dot);
            }
        }
    }
    timings.add_center_pass(t_center.elapsed());

    scratch.center_lens.resize(num_queries, 0);
    scratch.center_lens.copy_from_slice(&scratch.lens);

    let t_thresholds = PackedTimer::start();
    scratch.thresholds.clear();
    scratch.thresholds.reserve(num_queries);
    for qi in 0..num_queries {
        let threshold = if scratch.center_lens[qi] > 0 {
            scratch.security_thresholds[qi].max(scratch.min_center_dot[qi] - 1e-6)
        } else {
            scratch.security_thresholds[qi]
        };
        scratch.thresholds.push(threshold);
    }
    timings.add_ring_thresholds(t_thresholds.elapsed());

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

            for qi in 0..num_queries {
                let qx = f32x8::splat(scratch.query_x[qi]);
                let qy = f32x8::splat(scratch.query_y[qi]);
                let qz = f32x8::splat(scratch.query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(scratch.thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    let query_idx = queries[qi];
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let cand_global = grid.point_indices[soa_start + i + lane];
                        if cand_global != query_idx {
                            let dot = dots_arr[lane];
                            let slab_idx = qi * stride + scratch.lens[qi];
                            scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                            scratch.lens[qi] += 1;
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
            let cand_global = grid.point_indices[soa_start + i];
            for qi in 0..num_queries {
                if cand_global == queries[qi] {
                    continue;
                }
                let dot =
                    cx * scratch.query_x[qi] + cy * scratch.query_y[qi] + cz * scratch.query_z[qi];
                if dot > scratch.thresholds[qi] {
                    let slab_idx = qi * stride + scratch.lens[qi];
                    scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                    scratch.lens[qi] += 1;
                }
            }
        }
    }
    timings.add_ring_pass(t_ring.elapsed());

    let t_fallback = PackedTimer::start();
    for qi in 0..num_queries {
        let ring_added = scratch.lens[qi] - scratch.center_lens[qi];
        let need = k.saturating_sub(scratch.center_lens[qi]);
        if ring_added < need {
            scratch.lens[qi] = scratch.center_lens[qi];
            for &(soa_start, soa_end) in &scratch.cell_ranges[1..] {
                let range_len = soa_end - soa_start;
                let xs = &grid.cell_points_x[soa_start..soa_end];
                let ys = &grid.cell_points_y[soa_start..soa_end];
                let zs = &grid.cell_points_z[soa_start..soa_end];

                for i in 0..range_len {
                    let cand_global = grid.point_indices[soa_start + i];
                    if cand_global == queries[qi] {
                        continue;
                    }
                    let dot = xs[i] * scratch.query_x[qi]
                        + ys[i] * scratch.query_y[qi]
                        + zs[i] * scratch.query_z[qi];
                    if dot > scratch.security_thresholds[qi] {
                        let slab_idx = qi * stride + scratch.lens[qi];
                        scratch.keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                        scratch.lens[qi] += 1;
                    }
                }
            }
        }
    }
    timings.add_ring_fallback(t_fallback.elapsed());

    let t_select_prep = PackedTimer::start();
    scratch.neighbors.resize(num_queries * k, u32::MAX);
    timings.add_select_sort(t_select_prep.elapsed());
    for (qi, &query_idx) in queries.iter().enumerate() {
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
            for i in 0..k_actual {
                scratch.neighbors[out_start + i] = key_to_idx(keys_slice[i]);
            }
            timings.add_select_sort(t_select.elapsed());
        }

        let out_start = qi * k;
        let out_end = out_start + k_actual;
        let t_cb = PackedTimer::start();
        on_query(
            qi,
            query_idx,
            &scratch.neighbors[out_start..out_end],
            k_actual,
            scratch.security_thresholds[qi],
        );
        timings.add_callback(t_cb.elapsed());
    }

    PackedKnnCellStatus::Ok
}

/// PackedV4 batched k-NN (fast path, no stats instrumentation).
pub fn packed_knn(grid: &CubeMapGrid, points: &[Vec3], k: usize) -> PackedKnnResult {
    packed_knn_impl(grid, points, k, None)
}

/// PackedV4 batched k-NN with stats.
pub fn packed_knn_stats(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
) -> (PackedKnnResult, PackedKnnStats) {
    let mut stats = PackedKnnStats::default();
    let result = packed_knn_impl(grid, points, k, Some(&mut stats));
    (result, stats)
}

fn packed_knn_impl(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    mut stats: Option<&mut PackedKnnStats>,
) -> PackedKnnResult {
    let n = points.len();
    let num_cells = 6 * grid.res * grid.res;
    let mut neighbors = vec![u32::MAX; n * k];

    let mut cell_ranges: Vec<(usize, usize)> = Vec::with_capacity(9);

    let mut keys_slab: Vec<MaybeUninit<u64>> = Vec::new();
    let mut lens: Vec<usize> = Vec::new();
    let mut center_lens: Vec<usize> = Vec::new();
    let mut min_center_dot: Vec<f32> = Vec::new();
    let mut security_thresholds: Vec<f32> = Vec::new();
    let mut thresholds: Vec<f32> = Vec::new();
    let mut query_x: Vec<f32> = Vec::new();
    let mut query_y: Vec<f32> = Vec::new();
    let mut query_z: Vec<f32> = Vec::new();

    for cell in 0..num_cells {
        let query_points = grid.cell_points(cell);
        let num_queries = query_points.len();
        if num_queries == 0 {
            continue;
        }

        // Gather indices and track SoA ranges.
        cell_ranges.clear();

        let q_start = grid.cell_offsets[cell] as usize;
        let q_end = grid.cell_offsets[cell + 1] as usize;
        cell_ranges.push((q_start, q_end));

        for &ncell in grid.cell_neighbors(cell) {
            if ncell == u32::MAX || ncell == cell as u32 {
                continue;
            }
            let nc = ncell as usize;
            let n_start = grid.cell_offsets[nc] as usize;
            let n_end = grid.cell_offsets[nc + 1] as usize;
            if n_start < n_end {
                cell_ranges.push((n_start, n_end));
            }
        }

        let mut num_candidates = 0usize;
        for &(start, end) in &cell_ranges {
            num_candidates += end - start;
        }
        if num_candidates == 0 {
            continue;
        }

        let ring2 = grid.cell_ring2(cell);

        // Worst-case: fall back to an unbounded (slow) path rather than silently truncating.
        let stride = num_candidates;
        let slab_size = num_queries * stride;
        if num_candidates > MAX_CANDIDATES_FAST || slab_size > MAX_KEYS_SLAB_FAST {
            if let Some(stats) = stats.as_deref_mut() {
                stats.slow_path_cells += 1;
            }
            packed_knn_fallback_cell(
                grid,
                points,
                k,
                query_points,
                &cell_ranges,
                ring2,
                num_candidates,
                &mut neighbors,
                stats.as_deref_mut(),
            );
            continue;
        }

        // Per-query state.
        keys_slab.clear();
        if keys_slab.capacity() < slab_size {
            keys_slab.reserve(slab_size.saturating_sub(keys_slab.len()));
        }
        unsafe { keys_slab.set_len(slab_size) };
        lens.resize(num_queries, 0);
        lens.fill(0);
        min_center_dot.resize(num_queries, f32::INFINITY);
        min_center_dot.fill(f32::INFINITY);

        let (center_soa_start, center_soa_end) = cell_ranges[0];
        let center_len = center_soa_end - center_soa_start;
        let xs = &grid.cell_points_x[center_soa_start..center_soa_end];
        let ys = &grid.cell_points_y[center_soa_start..center_soa_end];
        let zs = &grid.cell_points_z[center_soa_start..center_soa_end];

        // Cache query vectors (queries are exactly the center cell points).
        query_x.resize(num_queries, 0.0);
        query_y.resize(num_queries, 0.0);
        query_z.resize(num_queries, 0.0);
        for qi in 0..num_queries {
            query_x[qi] = xs[qi];
            query_y[qi] = ys[qi];
            query_z[qi] = zs[qi];
        }

        // Precompute security_3x3 per query.
        security_thresholds.clear();
        security_thresholds.reserve(num_queries);
        for qi in 0..num_queries {
            security_thresholds.push(outside_max_dot_xyz(
                query_x[qi],
                query_y[qi],
                query_z[qi],
                ring2,
                grid,
            ));
        }

        // Pass A: center range only - compute dots, filter, track min.
        let full_chunks = center_len / 8;
        for chunk in 0..full_chunks {
            let i = chunk * 8;
            let cx = f32x8::from_slice(&xs[i..]);
            let cy = f32x8::from_slice(&ys[i..]);
            let cz = f32x8::from_slice(&zs[i..]);

            for qi in 0..num_queries {
                let qx = f32x8::splat(query_x[qi]);
                let qy = f32x8::splat(query_y[qi]);
                let qz = f32x8::splat(query_z[qi]);
                let dots = cx * qx + cy * qy + cz * qz;

                let thresh_vec = f32x8::splat(security_thresholds[qi]);
                let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                let mut mask_bits = mask.to_bitmask() as u32;

                // Clear self bit if in this chunk.
                let base = i;
                if base <= qi && qi < base + 8 {
                    mask_bits &= !(1u32 << (qi - base));
                }

                if mask_bits != 0 {
                    let dots_arr: [f32; 8] = dots.into();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let cand_idx = i + lane;
                        let dot = dots_arr[lane];
                        let slab_idx = qi * stride + lens[qi];
                        let cand_global = grid.point_indices[center_soa_start + cand_idx];
                        keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                        lens[qi] += 1;
                        min_center_dot[qi] = min_center_dot[qi].min(dot);
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }

        // Center tail.
        let tail_start = full_chunks * 8;
        for i in tail_start..center_len {
            let cx = xs[i];
            let cy = ys[i];
            let cz = zs[i];
            for qi in 0..num_queries {
                if i == qi {
                    continue;
                }
                let dot = cx * query_x[qi] + cy * query_y[qi] + cz * query_z[qi];
                if dot > security_thresholds[qi] {
                    let slab_idx = qi * stride + lens[qi];
                    let cand_global = grid.point_indices[center_soa_start + i];
                    keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                    lens[qi] += 1;
                    min_center_dot[qi] = min_center_dot[qi].min(dot);
                }
            }
        }

        center_lens.resize(num_queries, 0);
        center_lens.copy_from_slice(&lens);

        // Compute per-query thresholds for ring pass (PackedV4).
        thresholds.clear();
        thresholds.reserve(num_queries);
        for qi in 0..num_queries {
            let threshold = if center_lens[qi] > 0 {
                security_thresholds[qi].max(min_center_dot[qi] - 1e-6)
            } else {
                security_thresholds[qi]
            };
            thresholds.push(threshold);
        }

        // Pass B: ring ranges - compute dots, filter by threshold.
        for &(soa_start, soa_end) in &cell_ranges[1..] {
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

                for qi in 0..num_queries {
                    let qx = f32x8::splat(query_x[qi]);
                    let qy = f32x8::splat(query_y[qi]);
                    let qz = f32x8::splat(query_z[qi]);
                    let dots = cx * qx + cy * qy + cz * qz;

                    let thresh_vec = f32x8::splat(thresholds[qi]);
                    let mask: Mask<i32, 8> = dots.simd_gt(thresh_vec);
                    let mut mask_bits = mask.to_bitmask() as u32;

                    if mask_bits != 0 {
                        let dots_arr: [f32; 8] = dots.into();
                        while mask_bits != 0 {
                            let lane = mask_bits.trailing_zeros() as usize;
                            let dot = dots_arr[lane];
                            let slab_idx = qi * stride + lens[qi];
                            let cand_global = grid.point_indices[soa_start + i + lane];
                            keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                            lens[qi] += 1;
                            mask_bits &= mask_bits - 1;
                        }
                    }
                }
            }

            // Ring tail.
            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                for qi in 0..num_queries {
                    let dot = cx * query_x[qi] + cy * query_y[qi] + cz * query_z[qi];
                    if dot > thresholds[qi] {
                        let slab_idx = qi * stride + lens[qi];
                        let cand_global = grid.point_indices[soa_start + i];
                        keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                        lens[qi] += 1;
                    }
                }
            }
        }

        // Fallback: re-run ring with security threshold if we didn't get enough.
        for qi in 0..num_queries {
            let ring_added = lens[qi] - center_lens[qi];
            let need = k.saturating_sub(center_lens[qi]);
            if ring_added < need {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.fallback_queries += 1;
                }
                lens[qi] = center_lens[qi];
                for &(soa_start, soa_end) in &cell_ranges[1..] {
                    let range_len = soa_end - soa_start;
                    let xs = &grid.cell_points_x[soa_start..soa_end];
                    let ys = &grid.cell_points_y[soa_start..soa_end];
                    let zs = &grid.cell_points_z[soa_start..soa_end];

                    for i in 0..range_len {
                        let dot = xs[i] * query_x[qi] + ys[i] * query_y[qi] + zs[i] * query_z[qi];
                        if dot > security_thresholds[qi] {
                            let slab_idx = qi * stride + lens[qi];
                            let cand_global = grid.point_indices[soa_start + i];
                            keys_slab[slab_idx].write(make_desc_key(dot, cand_global));
                            lens[qi] += 1;
                        }
                    }
                }
            }
        }

        // Select+sort per query.
        for (qi, &query_idx) in query_points.iter().enumerate() {
            let m = lens[qi];
            if let Some(stats) = stats.as_deref_mut() {
                stats.total_candidates += num_candidates as u64;
                stats.filtered_out += (num_candidates - m) as u64;
                stats.num_queries += 1;
            }

            let keys_uninit = &mut keys_slab[qi * stride..qi * stride + m];
            let keys_slice =
                unsafe { std::slice::from_raw_parts_mut(keys_uninit.as_mut_ptr() as *mut u64, m) };

            let k_actual = k.min(m);
            if k_actual < k {
                if let Some(stats) = stats.as_deref_mut() {
                    stats.under_k_count += 1;
                }
            }
            if k_actual > 0 {
                if m > k_actual {
                    keys_slice.select_nth_unstable(k_actual - 1);
                }
                keys_slice[..k_actual].sort_unstable();

                let out_start = query_idx as usize * k;
                for i in 0..k_actual {
                    neighbors[out_start + i] = key_to_idx(keys_slice[i]);
                }
            }
        }
    }

    PackedKnnResult { neighbors, k }
}

/// Worst-case fallback when the per-cell candidate list is too large for the fixed-size fast path.
fn packed_knn_fallback_cell(
    grid: &CubeMapGrid,
    points: &[Vec3],
    k: usize,
    query_points: &[u32],
    cell_ranges: &[(usize, usize)],
    ring2: &[u32],
    num_candidates: usize,
    neighbors: &mut [u32],
    mut stats: Option<&mut PackedKnnStats>,
) {
    let (center_soa_start, center_soa_end) = cell_ranges[0];
    let center_len = center_soa_end - center_soa_start;

    let mut keys: Vec<u64> = Vec::new();

    for (qi, &query_idx) in query_points.iter().enumerate() {
        let q = points[query_idx as usize];
        let security = outside_max_dot(q, ring2, grid);

        keys.clear();
        keys.reserve(k.min(num_candidates));

        let center_xs = &grid.cell_points_x[center_soa_start..center_soa_end];
        let center_ys = &grid.cell_points_y[center_soa_start..center_soa_end];
        let center_zs = &grid.cell_points_z[center_soa_start..center_soa_end];

        let mut min_center_dot = f32::INFINITY;
        let mut center_added = 0usize;

        for i in 0..center_len {
            if i == qi {
                continue;
            }
            let dot = center_xs[i] * q.x + center_ys[i] * q.y + center_zs[i] * q.z;
            if dot > security {
                let cand_global = grid.point_indices[center_soa_start + i];
                keys.push(make_desc_key(dot, cand_global));
                min_center_dot = min_center_dot.min(dot);
                center_added += 1;
            }
        }

        let threshold = if center_added > 0 {
            security.max(min_center_dot - 1e-6)
        } else {
            security
        };

        // Ring pass.
        let center_keys_len = keys.len();
        for &(soa_start, soa_end) in &cell_ranges[1..] {
            let xs = &grid.cell_points_x[soa_start..soa_end];
            let ys = &grid.cell_points_y[soa_start..soa_end];
            let zs = &grid.cell_points_z[soa_start..soa_end];
            let range_len = soa_end - soa_start;

            for i in 0..range_len {
                let dot = xs[i] * q.x + ys[i] * q.y + zs[i] * q.z;
                if dot > threshold {
                    let cand_global = grid.point_indices[soa_start + i];
                    keys.push(make_desc_key(dot, cand_global));
                }
            }
        }

        // Fallback to security threshold if we didn't get enough ring candidates.
        let ring_added = keys.len() - center_keys_len;
        let need = k.saturating_sub(center_added);
        if ring_added < need {
            if let Some(stats) = stats.as_deref_mut() {
                stats.fallback_queries += 1;
            }
            keys.truncate(center_keys_len);
            for &(soa_start, soa_end) in &cell_ranges[1..] {
                let xs = &grid.cell_points_x[soa_start..soa_end];
                let ys = &grid.cell_points_y[soa_start..soa_end];
                let zs = &grid.cell_points_z[soa_start..soa_end];
                let range_len = soa_end - soa_start;

                for i in 0..range_len {
                    let dot = xs[i] * q.x + ys[i] * q.y + zs[i] * q.z;
                    if dot > security {
                        let cand_global = grid.point_indices[soa_start + i];
                        keys.push(make_desc_key(dot, cand_global));
                    }
                }
            }
        }

        let m = keys.len();
        if let Some(stats) = stats.as_deref_mut() {
            stats.total_candidates += num_candidates as u64;
            stats.filtered_out += (num_candidates.saturating_sub(m)) as u64;
            stats.num_queries += 1;
        }

        let k_actual = k.min(m);
        if k_actual < k {
            if let Some(stats) = stats.as_deref_mut() {
                stats.under_k_count += 1;
            }
        }
        if k_actual > 0 {
            if m > k_actual {
                keys.select_nth_unstable(k_actual - 1);
            }
            keys[..k_actual].sort_unstable();
            let out_start = query_idx as usize * k;
            for i in 0..k_actual {
                neighbors[out_start + i] = key_to_idx(keys[i]);
            }
        }
    }
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
fn outside_max_dot(q: Vec3, ring2: &[u32], grid: &CubeMapGrid) -> f32 {
    outside_max_dot_xyz(q.x, q.y, q.z, ring2, grid)
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

    /// Generate Fibonacci sphere points with optional jitter.
    fn fibonacci_sphere_points(n: usize, jitter: f32, rng: &mut impl Rng) -> Vec<Vec3> {
        use std::f32::consts::PI;
        let golden_angle = PI * (3.0 - 5.0f32.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
                let radius = (1.0 - y * y).sqrt();
                let theta = golden_angle * i as f32;

                let mut x = radius * theta.cos();
                let mut z = radius * theta.sin();

                if jitter > 0.0 {
                    x += rng.gen_range(-jitter..jitter);
                    z += rng.gen_range(-jitter..jitter);
                }

                Vec3::new(x, y, z).normalize()
            })
            .collect()
    }

    fn gen_fibonacci(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        fibonacci_sphere_points(n, spacing * 0.1, &mut rng)
    }

    fn res_for_target(n: usize, target: f64) -> usize {
        ((n as f64 / (6.0 * target)).sqrt() as usize).max(4)
    }

    #[test]
    fn test_packed_knn_basic() {
        let n = 10_000;
        let k = 24;
        let points = gen_fibonacci(n, 12345);
        let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));

        let result = packed_knn(&grid, &points, k);

        assert_eq!(result.neighbors.len(), n * k);

        for qi in [0, 100, 5000, 9999] {
            let neighbors = result.get_valid(qi);
            assert!(neighbors.len() <= k);
            assert!(!neighbors.contains(&(qi as u32)));
            for &idx in neighbors {
                assert!((idx as usize) < n);
            }
        }
    }

    #[test]
    fn test_packed_knn_no_duplicate_neighbors() {
        let n = 6000;
        let k = 24;
        for seed in [1u64, 2, 3, 4, 5] {
            let points = gen_fibonacci(n, seed);
            let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));
            let result = packed_knn(&grid, &points, k);

            for qi in 0..n {
                let neighbors = result.get_valid(qi);
                let mut seen = std::collections::HashSet::<u32>::with_capacity(neighbors.len());
                for &idx in neighbors {
                    assert!(
                        seen.insert(idx),
                        "duplicate neighbor: seed={}, query={}, neighbor={}",
                        seed,
                        qi,
                        idx
                    );
                }
            }
        }
    }
}
