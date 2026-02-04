use crate::fp;
use glam::Vec3;

use super::super::{CubeMapGrid, CubeMapGridScratch, KnnStatus};

impl CubeMapGrid {
    /// Resumable kNN query that returns slots (SOA indices) instead of global indices.
    #[allow(dead_code)]
    pub fn find_k_nearest_resumable_slots_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_slots: &mut Vec<u32>,
    ) -> KnnStatus {
        self.find_k_nearest_resumable_into_impl(
            query,
            query_idx,
            k,
            track_limit,
            scratch,
            out_slots,
        )
    }

    #[allow(dead_code)]
    fn bruteforce_fill_impl(&self, query: Vec3, query_idx: usize, scratch: &mut CubeMapGridScratch) {
        if scratch.use_fixed {
            scratch.candidates_len = 0;
        } else {
            scratch.candidates_vec.clear();
        }

        let (qx, qy, qz) = (query.x, query.y, query.z);
        let n = self.point_indices.len();

        let indices = &self.point_indices[..n];
        let xs = &self.cell_points_x[..n];
        let ys = &self.cell_points_y[..n];
        let zs = &self.cell_points_z[..n];

        for (slot, (global, ((x, y), z))) in indices
            .iter()
            .zip(xs.iter().zip(ys.iter()).zip(zs.iter()))
            .enumerate()
        {
            let global = *global as usize;
            if global == query_idx {
                continue;
            }
            let dot = fp::dot3_f32(*x, *y, *z, qx, qy, qz);
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            scratch.try_add_neighbor(slot as u32, dist_sq);
        }
        scratch.exhausted = true;
    }

    #[inline]
    #[allow(dead_code)]
    fn scan_cell_points_impl(
        &self,
        query: Vec3,
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

        let (qx, qy, qz) = (query.x, query.y, query.z);

        for i in 0..xs.len() {
            let pidx = indices[i] as usize;
            if pidx == query_idx {
                continue;
            }
            // Contiguous SoA access - should auto-vectorize well
            let dot = fp::dot3_f32(xs[i], ys[i], zs[i], qx, qy, qz);
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            let slot = (start + i) as u32;
            scratch.try_add_neighbor(slot, dist_sq);
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn seed_start_cell_impl(
        &self,
        query: Vec3,
        query_idx: usize,
        start_cell: u32,
        scratch: &mut CubeMapGridScratch,
    ) -> usize {
        scratch.mark_visited(start_cell);
        self.scan_cell_points_impl(query, query_idx, start_cell as usize, scratch);

        let neighbors = self.cell_neighbors(start_cell as usize);
        for &ncell in neighbors.iter() {
            if ncell == u32::MAX || ncell == start_cell {
                continue;
            }
            if !scratch.mark_visited(ncell) {
                continue;
            }
            let bound = self.cell_min_dist_sq(query, ncell as usize);
            scratch.push_cell(ncell, bound);
        }

        1
    }

    #[allow(dead_code)]
    fn find_k_nearest_resumable_into_impl(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_slots: &mut Vec<u32>,
    ) -> KnnStatus {
        let n = self.point_indices.len();
        out_slots.clear();

        if k == 0 || n <= 1 {
            return KnnStatus::CanResume;
        }
        let k = k.min(n - 1);
        let track_limit = track_limit.min(n - 1).max(k);

        let num_cells = 6 * self.res * self.res;
        scratch.begin_query(k, track_limit);

        let start_cell = if query_idx < self.point_cells.len() {
            self.point_cells[query_idx]
        } else {
            self.point_to_cell(query) as u32
        };

        let visited = self.seed_start_cell_impl(query, query_idx, start_cell, scratch);

        let mut visited = visited;
        while let Some((bound_dist_sq, _cell)) = scratch.peek_cell() {
            if scratch.have_k(k) && bound_dist_sq >= scratch.kth_dist_sq(k) {
                break;
            };

            let (_, cell) = scratch.pop_cell().expect("cell heap out of sync");
            self.scan_cell_points_impl(query, query_idx, cell as usize, scratch);

            let neighbors = self.cell_neighbors(cell as usize);
            for &ncell in neighbors.iter() {
                if ncell == u32::MAX {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let bound = self.cell_min_dist_sq(query, ncell as usize);
                scratch.push_cell(ncell, bound);
            }

            visited += 1;
            if visited >= num_cells {
                break;
            }
        }

        if !scratch.have_k(k) {
            // Not enough results found; brute force remaining points
            self.bruteforce_fill_impl(query, query_idx, scratch);
        }

        scratch.copy_k_slots_into(k, out_slots);

        if scratch.exhausted {
            KnnStatus::Exhausted
        } else {
            KnnStatus::CanResume
        }
    }
}

