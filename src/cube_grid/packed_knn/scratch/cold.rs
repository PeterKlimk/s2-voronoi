use super::super::timing::PackedLapTimer;
use super::helpers::{make_desc_key, outside_max_dot_xyz, unpack_bin_local};
use super::*;
use crate::fp;
use crate::policy::PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY;
use std::collections::VecDeque;

impl PackedKnnCellScratch {
    #[inline]
    fn classify_cell_range(
        &self,
        grid: &CubeMapGrid,
        cell: usize,
        group: PackedGroupInput<'_>,
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
        group: PackedGroupInput<'_>,
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

    pub(crate) fn ensure_security2_for(
        &mut self,
        qi: usize,
        group: PackedGroupInput<'_>,
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
        group: PackedGroupInput<'_>,
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
}
