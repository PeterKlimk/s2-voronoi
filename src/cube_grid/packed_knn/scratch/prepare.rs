use super::super::timing::PackedLapTimer;
use super::helpers::{
    make_desc_key, outside_max_dot_xyz, security_planes_3x3_interior, unpack_bin_local,
};
use super::*;
use crate::fp;
use crate::policy::{
    PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER, PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER,
    PACKED_HI_BUDGET,
};

impl PackedKnnCellScratch {
    #[cfg_attr(feature = "profiling", inline(never))]
    pub(crate) fn prepare_group_directed<'a, 'g>(
        &'a mut self,
        grid: &CubeMapGrid,
        group: PackedGroupInput<'g>,
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
            let candidates = fp::PointChunk8::from_slices(&xs[i..], &ys[i..], &zs[i..]);

            // Candidate positions in this chunk are [i, i+7]. A query at position qi only
            // needs to consider this chunk if qi <= i+7.
            let qi_end = (i + 8).min(num_queries);
            for qi in 0..qi_end {
                let dots = candidates.dots(query_x[qi], query_y[qi], query_z[qi]);
                let mut mask_bits = dots.mask_gt(security_thresholds[qi]);
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

                let dots_arr = dots.to_array();
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
                let dot = super::helpers::key_to_dot(key);
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
                let candidates = fp::PointChunk8::from_slices(&xs[i..], &ys[i..], &zs[i..]);

                for (qi, &query_slot) in queries.iter().enumerate() {
                    let dots = candidates.dots(query_x[qi], query_y[qi], query_z[qi]);
                    let mut mask_bits = dots.mask_gt(thresholds[qi]);
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr = dots.to_array();
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

                let candidates = fp::PointChunk8::from_arrays(xbuf, ybuf, zbuf);

                for (qi, &query_slot) in queries.iter().enumerate() {
                    let dots = candidates.dots(query_x[qi], query_y[qi], query_z[qi]);
                    let mut mask_bits = dots.mask_gt(thresholds[qi]) & valid_bits;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr = dots.to_array();
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
}
