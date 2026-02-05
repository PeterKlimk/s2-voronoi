use crate::fp;
use glam::Vec3;

use super::super::{CubeMapGrid, CubeMapGridScratch, KnnStatus};

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectedCtx<'a> {
    query_bin: u8,
    query_local: u32,
    slot_gen_map: &'a [u32],
    local_shift: u32,
    local_mask: u32,
}

impl<'a> DirectedCtx<'a> {
    #[inline]
    pub(crate) fn new(
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &'a [u32],
        local_shift: u32,
        local_mask: u32,
    ) -> Self {
        Self {
            query_bin,
            query_local,
            slot_gen_map,
            local_shift,
            local_mask,
        }
    }
}

impl CubeMapGrid {
    #[inline]
    pub(crate) fn find_k_nearest_resumable_slots_directed_ctx_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_slots: &mut Vec<u32>,
        ctx: DirectedCtx<'_>,
    ) -> KnnStatus {
        self.find_k_nearest_resumable_into_directed_impl(
            query,
            query_idx,
            k,
            track_limit,
            scratch,
            out_slots,
            ctx.query_bin,
            ctx.query_local,
            ctx.slot_gen_map,
            ctx.local_shift,
            ctx.local_mask,
        )
    }

    #[inline(always)]
    fn unpack_bin_local(packed: u32, local_shift: u32, local_mask: u32) -> (u8, u32) {
        let bin = (packed >> local_shift) as u8;
        let local = packed & local_mask;
        (bin, local)
    }

    fn bruteforce_fill_directed_impl(
        &self,
        query: Vec3,
        query_idx: usize,
        scratch: &mut CubeMapGridScratch,
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
    ) {
        if scratch.use_fixed {
            scratch.candidates_len = 0;
        } else {
            scratch.candidates_vec.clear();
        }

        let (qx, qy, qz) = (query.x, query.y, query.z);
        let n = self.point_indices.len();
        debug_assert_eq!(
            slot_gen_map.len(),
            n,
            "slot_gen_map must match point_indices length"
        );

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

            let packed = slot_gen_map[slot];
            let (bin_b, local_b) = Self::unpack_bin_local(packed, local_shift, local_mask);
            if bin_b == query_bin && local_b < query_local {
                continue;
            }

            let dot = fp::dot3_f32(*x, *y, *z, qx, qy, qz);
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            scratch.try_add_neighbor(slot as u32, dist_sq);
        }
        scratch.exhausted = true;
    }

    #[inline]
    fn scan_cell_points_directed_impl(
        &self,
        query: Vec3,
        query_idx: usize,
        cell: usize,
        scratch: &mut CubeMapGridScratch,
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
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
            let slot = (start + i) as u32;
            let packed = slot_gen_map[slot as usize];
            let (bin_b, local_b) = Self::unpack_bin_local(packed, local_shift, local_mask);
            if bin_b == query_bin && local_b < query_local {
                continue;
            }

            let dot = fp::dot3_f32(xs[i], ys[i], zs[i], qx, qy, qz);
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            scratch.try_add_neighbor(slot, dist_sq);
        }
    }

    #[inline]
    fn seed_start_cell_directed_impl(
        &self,
        query: Vec3,
        query_idx: usize,
        start_cell: u32,
        scratch: &mut CubeMapGridScratch,
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
    ) -> usize {
        scratch.mark_visited(start_cell);
        self.scan_cell_points_directed_impl(
            query,
            query_idx,
            start_cell as usize,
            scratch,
            query_bin,
            query_local,
            slot_gen_map,
            local_shift,
            local_mask,
        );

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

    fn find_k_nearest_resumable_into_directed_impl(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_slots: &mut Vec<u32>,
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
    ) -> KnnStatus {
        let n = self.point_indices.len();
        out_slots.clear();

        if k == 0 || n <= 1 {
            return KnnStatus::CanResume;
        }
        debug_assert_eq!(
            slot_gen_map.len(),
            n,
            "slot_gen_map must match point_indices length"
        );

        let k = k.min(n - 1);
        let track_limit = track_limit.min(n - 1).max(k);

        let num_cells = 6 * self.res * self.res;
        scratch.begin_query(k, track_limit);

        let start_cell = if query_idx < self.point_cells.len() {
            self.point_cells[query_idx]
        } else {
            self.point_to_cell(query) as u32
        };

        let visited = self.seed_start_cell_directed_impl(
            query,
            query_idx,
            start_cell,
            scratch,
            query_bin,
            query_local,
            slot_gen_map,
            local_shift,
            local_mask,
        );

        let mut visited = visited;
        while let Some((bound_dist_sq, _cell)) = scratch.peek_cell() {
            if scratch.have_k(k) && bound_dist_sq >= scratch.kth_dist_sq(k) {
                break;
            };

            let (_, cell) = scratch.pop_cell().expect("cell heap out of sync");
            self.scan_cell_points_directed_impl(
                query,
                query_idx,
                cell as usize,
                scratch,
                query_bin,
                query_local,
                slot_gen_map,
                local_shift,
                local_mask,
            );

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
            self.bruteforce_fill_directed_impl(
                query,
                query_idx,
                scratch,
                query_bin,
                query_local,
                slot_gen_map,
                local_shift,
                local_mask,
            );
        }

        scratch.copy_k_slots_into(k, out_slots);

        if scratch.exhausted {
            KnnStatus::Exhausted
        } else {
            KnnStatus::CanResume
        }
    }
}
