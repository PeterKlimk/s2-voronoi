//! Query helpers for CubeMapGrid.

use glam::Vec3;
use std::cmp::Reverse;

use super::{
    face_uv_to_cell, point_to_face_uv, unit_vec_dist_sq_generic, CubeMapGrid, CubeMapGridScratch,
    KnnStatus, OrdF32, UnitVec,
};

impl CubeMapGridScratch {
    pub(super) const MAX_TRACK: usize = 128;

    pub fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            cell_heap: std::collections::BinaryHeap::new(),
            track_limit: 0,
            candidates_fixed: [(f32::INFINITY, 0); Self::MAX_TRACK],
            candidates_len: 0,
            use_fixed: true,
            candidates_vec: Vec::new(),
            candidates_dot: Vec::new(),
            exhausted: false,
        }
    }

    #[inline]
    fn begin_query(&mut self, k: usize, track_limit: usize) {
        self.cell_heap.clear();
        self.exhausted = false;
        self.track_limit = track_limit;
        self.candidates_len = 0;
        self.use_fixed = track_limit <= Self::MAX_TRACK;
        self.candidates_vec.clear();
        self.candidates_dot.clear();
        if !self.use_fixed {
            let reserve = track_limit.max(k);
            if self.candidates_vec.capacity() < reserve {
                self.candidates_vec
                    .reserve(reserve - self.candidates_vec.capacity());
            }
        }

        // Stamp 0 means "unvisited". Avoid ever using stamp 0 for a query.
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.visited_stamp.fill(0);
            self.stamp = 1;
        }
    }

    #[inline]
    fn mark_visited(&mut self, cell: u32) -> bool {
        let idx = cell as usize;
        if self.visited_stamp[idx] == self.stamp {
            return false;
        }
        self.visited_stamp[idx] = self.stamp;
        true
    }

    #[inline]
    fn push_cell(&mut self, cell: u32, bound_dist_sq: f32) {
        self.cell_heap
            .push(Reverse((OrdF32::new(bound_dist_sq), cell)));
    }

    #[inline]
    fn peek_cell(&self) -> Option<(f32, u32)> {
        self.cell_heap
            .peek()
            .map(|Reverse((bound, cell))| (bound.get(), *cell))
    }

    #[inline]
    fn pop_cell(&mut self) -> Option<(f32, u32)> {
        self.cell_heap
            .pop()
            .map(|Reverse((bound, cell))| (bound.get(), cell))
    }

    #[inline]
    fn kth_dist_sq(&self, k: usize) -> f32 {
        if self.use_fixed {
            self.candidates_fixed[k - 1].0
        } else {
            self.candidates_vec[k - 1].0
        }
    }

    #[inline]
    fn have_k(&self, k: usize) -> bool {
        if self.use_fixed {
            self.candidates_len >= k
        } else {
            self.candidates_vec.len() >= k
        }
    }

    fn try_add_neighbor(&mut self, slot: u32, dist_sq: f32) {
        let len = if self.use_fixed {
            self.candidates_len
        } else {
            self.candidates_vec.len()
        };
        let k = self.track_limit;

        if len < k {
            if self.use_fixed {
                self.candidates_fixed[len] = (dist_sq, slot);
                self.candidates_len += 1;
            } else {
                self.candidates_vec.push((dist_sq, slot));
            }

            // Just reached k: sort in-place.
            if len + 1 == k {
                if self.use_fixed {
                    let slice = &mut self.candidates_fixed[..k];
                    slice.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else {
                    self.candidates_vec
                        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
            }
            return;
        }

        if dist_sq >= self.kth_dist_sq(k) {
            return;
        }

        // Replace worst, then rescan to find new worst (k is small: 12/24/48).
        if self.use_fixed {
            self.candidates_fixed[k - 1] = (dist_sq, slot);
            let slice = &mut self.candidates_fixed[..k];
            slice.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        } else {
            self.candidates_vec[k - 1] = (dist_sq, slot);
            self.candidates_vec
                .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
    }

    #[inline]
    fn copy_k_indices_into(&self, k: usize, out: &mut Vec<usize>) {
        out.clear();

        let k = if self.use_fixed {
            k.min(self.candidates_len)
        } else {
            k.min(self.candidates_vec.len())
        };

        out.reserve(k);
        if self.use_fixed {
            for i in 0..k {
                out.push(self.candidates_fixed[i].1 as usize);
            }
        } else {
            for i in 0..k {
                out.push(self.candidates_vec[i].1 as usize);
            }
        }
    }

    #[inline]
    fn copy_k_slots_into(&self, k: usize, out: &mut Vec<u32>) {
        out.clear();

        let k = if self.use_fixed {
            k.min(self.candidates_len)
        } else {
            k.min(self.candidates_vec.len())
        };

        out.reserve(k);
        if self.use_fixed {
            for i in 0..k {
                out.push(self.candidates_fixed[i].1);
            }
        } else {
            for i in 0..k {
                out.push(self.candidates_vec[i].1);
            }
        }
    }
}

impl CubeMapGrid {
    /// Get cell index for a point.
    #[inline]
    pub fn point_to_cell(&self, p: Vec3) -> usize {
        let (face, u, v) = point_to_face_uv(p);
        face_uv_to_cell(face, u, v, self.res)
    }

    /// Get the precomputed cell index for `points[idx]` used to build this grid.
    #[inline]
    pub fn point_index_to_cell(&self, idx: usize) -> usize {
        self.point_cells[idx] as usize
    }

    /// Get grid resolution (cells per face).
    #[inline]
    pub fn res(&self) -> usize {
        self.res
    }

    /// Get cell offsets array (length = num_cells + 1).
    #[inline]
    pub fn cell_offsets(&self) -> &[u32] {
        &self.cell_offsets
    }

    /// Get point indices array (SOA layout, length = n).
    #[inline]
    pub fn point_indices(&self) -> &[u32] {
        &self.point_indices
    }

    /// Get points in a cell.
    #[inline]
    pub fn cell_points(&self, cell: usize) -> &[u32] {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;
        &self.point_indices[start..end]
    }

    /// Get the 9 neighbor cells (including self) for a cell.
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> &[u32; 9] {
        let base = cell * 9;
        self.neighbors[base..base + 9].try_into().unwrap()
    }

    /// Get the ring-2 cells (Chebyshev distance 2) for a cell.
    #[inline]
    pub fn cell_ring2(&self, cell: usize) -> &[u32] {
        let len = self.ring2_lens[cell] as usize;
        &self.ring2[cell][..len]
    }

    /// Get precomputed security_3x3 threshold for a cell.
    /// This is the max dot from any point in the cell to any ring-2 cell.

    /// Create a reusable scratch buffer for fast repeated queries.
    pub fn make_scratch(&self) -> CubeMapGridScratch {
        CubeMapGridScratch::new(6 * self.res * self.res)
    }

    /// Conservative lower bound on squared Euclidean distance from `query` to any point in `cell`.
    ///
    /// Uses a spherical cap that contains the cell and triangle inequality on the sphere.
    #[inline]
    fn cell_min_dist_sq(&self, query: Vec3, cell: usize) -> f32 {
        let center = self.cell_centers[cell];
        let mut cos_d = query.dot(center);
        cos_d = cos_d.clamp(-1.0, 1.0);

        let cos_r = self.cell_cos_radius[cell];
        let sin_r = self.cell_sin_radius[cell];

        // If the query direction is within the cell's cap, the minimum distance can be 0.
        if cos_d > cos_r {
            return 0.0;
        }

        // cos(d - r) = cos d cos r + sin d sin r
        let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
        let max_dot_upper = (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0);
        2.0 - 2.0 * max_dot_upper
    }

    /// Scratch-based k-NN query that writes results into `out_indices` (sorted closest-first).
    ///
    /// This is the preferred high-throughput API: it avoids per-query allocations.

    /// Non-resumable scratch-based k-NN query optimized for unit vectors:
    /// maintains an unsorted top-k by dot product and sorts once at the end.

    /// Resumable kNN query that returns slots (SOA indices) instead of global indices.
    pub fn find_k_nearest_resumable_slots_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_slots: &mut Vec<u32>,
    ) -> KnnStatus {
        self.find_k_nearest_resumable_into_impl(
            points,
            query,
            query_idx,
            k,
            track_limit,
            scratch,
            None,
            Some(out_slots),
        )
    }

    /// Resume a k-NN query to fetch additional neighbors.
    ///
    /// Call this after `find_k_nearest_resumable_into` when you need more neighbors.
    /// `new_k` should be larger than the previous k but within the original `track_limit`.

    /// Resume a k-NN query and append only the new neighbors to `out_indices`.
    ///
    /// `prev_k` is the number of neighbors previously produced into `out_indices` for this
    /// same scratch/query state. On success, this appends indices for the range
    /// `prev_k..new_k` (or less if exhausted).
    ///
    /// This is an optimization over `resume_k_nearest_into` when the caller wants to process
    /// only the newly discovered neighbors.

    fn bruteforce_fill_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        if scratch.use_fixed {
            scratch.candidates_len = 0;
        } else {
            scratch.candidates_vec.clear();
        }

        for (idx, p) in points.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            let dist_sq = unit_vec_dist_sq_generic(*p, query);
            scratch.try_add_neighbor(idx as u32, dist_sq);
        }
        scratch.exhausted = true;
    }

    #[inline]
    fn scan_cell_points_impl<P: UnitVec>(
        &self,
        _points: &[P],
        query: P,
        query_idx: usize,
        cell: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;

        // Use SoA layout for contiguous memory access
        let xs = &self.cell_points_x[start..end];
        let ys = &self.cell_points_y[start..end];
        let zs = &self.cell_points_z[start..end];
        let indices = &self.point_indices[start..end];

        let qv = query.to_vec3();
        let (qx, qy, qz) = (qv.x, qv.y, qv.z);

        for i in 0..xs.len() {
            let pidx = indices[i] as usize;
            if pidx == query_idx {
                continue;
            }
            // Contiguous SoA access - should auto-vectorize well
            let dot = xs[i] * qx + ys[i] * qy + zs[i] * qz;
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            let slot = (start + i) as u32;
            scratch.try_add_neighbor(slot, dist_sq);
        }
    }

    #[inline]

    fn seed_start_cell_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        start_cell: u32,
        scratch: &mut CubeMapGridScratch,
    ) -> usize {
        scratch.mark_visited(start_cell);
        self.scan_cell_points_impl(points, query, query_idx, start_cell as usize, scratch);

        let query_vec3 = query.to_vec3();
        let neighbors = self.cell_neighbors(start_cell as usize);
        for &ncell in neighbors.iter() {
            if ncell == u32::MAX || ncell == start_cell {
                continue;
            }
            if !scratch.mark_visited(ncell) {
                continue;
            }
            let bound = self.cell_min_dist_sq(query_vec3, ncell as usize);
            scratch.push_cell(ncell, bound);
        }

        1
    }

    #[allow(clippy::too_many_arguments)]
    fn find_k_nearest_resumable_into_impl<P: UnitVec>(
        &self,
        points: &[P],
        query: P,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        mut out_indices: Option<&mut Vec<usize>>,
        mut out_slots: Option<&mut Vec<u32>>,
    ) -> KnnStatus {
        let n = points.len();
        out_indices.as_deref_mut().map(|v| v.clear());
        out_slots.as_deref_mut().map(|v| v.clear());

        if k == 0 || n <= 1 {
            return KnnStatus::CanResume;
        }
        let k = k.min(n - 1);
        let track_limit = track_limit.min(n - 1).max(k);

        let query_vec3 = query.to_vec3();
        let num_cells = 6 * self.res * self.res;
        scratch.begin_query(k, track_limit);

        let start_cell = if query_idx < self.point_cells.len() {
            self.point_cells[query_idx]
        } else {
            self.point_to_cell(query_vec3) as u32
        };

        let visited = self.seed_start_cell_impl(points, query, query_idx, start_cell, scratch);

        let mut visited = visited;
        loop {
            let (bound_dist_sq, _cell) = match scratch.peek_cell() {
                Some(v) => v,
                None => break,
            };

            if scratch.have_k(k) && bound_dist_sq >= scratch.kth_dist_sq(k) {
                break;
            }

            let (_, cell) = scratch.pop_cell().expect("cell heap out of sync");
            self.scan_cell_points_impl(points, query, query_idx, cell as usize, scratch);

            let neighbors = self.cell_neighbors(cell as usize);
            for &ncell in neighbors.iter() {
                if ncell == u32::MAX {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let bound = self.cell_min_dist_sq(query_vec3, ncell as usize);
                scratch.push_cell(ncell, bound);
            }

            visited += 1;
            if visited >= num_cells {
                break;
            }
        }

        if !scratch.have_k(k) {
            // Not enough results found; brute force remaining points
            self.bruteforce_fill_impl(points, query, query_idx, scratch);
        }

        if let Some(out_indices) = out_indices {
            scratch.copy_k_indices_into(k, out_indices);
            if !scratch.exhausted {
                // Convert slots back to global indices if we have slots
                for idx in out_indices.iter_mut() {
                    let slot = *idx;
                    *idx = self.point_indices[slot] as usize;
                }
            }
        }
        if let Some(out_slots) = out_slots {
            if !scratch.exhausted {
                scratch.copy_k_slots_into(k, out_slots);
            } else {
                // Brute force used: we have globals, not slots. Return u32::MAX.
                let count = if scratch.use_fixed {
                    k.min(scratch.candidates_len)
                } else {
                    k.min(scratch.candidates_vec.len())
                };
                out_slots.clear();
                out_slots.resize(count, u32::MAX);
            }
        }

        if scratch.exhausted {
            KnnStatus::Exhausted
        } else {
            KnnStatus::CanResume
        }
    }
}
