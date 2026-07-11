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
    // Loop indices address several parallel per-query arrays at once;
    // iterator zips would obscure rather than clarify.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn prepare_group_directed<'a, 'g>(
        &'a mut self,
        grid: &CubeMapGrid,
        group: PackedGroupInput<'g>,
        timings: &mut PackedKnnTimings,
    ) -> PreparedPackedGroupStatus<'a, 'g> {
        timings.clear();

        let cell = group.cell();
        let num_cells = 6 * grid.res * grid.res;
        if cell >= num_cells {
            return PreparedPackedGroupStatus::SlowPath;
        }

        group.debug_assert_matches_grid(grid);

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
        }
        self.next_group_gen = group_gen;

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
        if num_queries
            .checked_mul(num_candidates)
            .is_none_or(|work| work > MAX_AGGREGATE_CANDIDATE_WORK)
        {
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
                        let s = (s_min - crate::tolerances::GRID_PLANE_PAD).clamp(0.0, 1.0);
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

        self.thresholds.resize(num_queries, 0.0);

        // Cleared so non-dense groups don't inherit band state from a previous
        // dense group. Dense groups resize these lazily below.
        self.center_bound.clear();
        self.band_mode.clear();

        // Don't shrink `Vec<Vec<_>>` to avoid dropping inner buffers when group sizes vary.
        if self.chunk0_keys.len() < num_queries {
            self.chunk0_keys.resize_with(num_queries, Vec::new);
        }
        for v in &mut self.chunk0_keys[..num_queries] {
            v.clear();
        }
        self.chunk0_pos.clear();
        self.chunk0_pos.resize(num_queries, 0);

        // Same rationale as `chunk0_keys` above.
        if self.tail_keys.len() < num_queries {
            self.tail_keys.resize_with(num_queries, Vec::new);
        }
        for v in &mut self.tail_keys[..num_queries] {
            v.clear();
        }
        self.tail_pos.clear();
        self.tail_pos.resize(num_queries, 0);
        self.tail_possible.resize(num_queries, false);
        self.center_tail_counts.clear();
        self.center_tail_counts.resize(num_queries, 0);
        if self.tail_ready_gen.len() < num_queries {
            self.tail_ready_gen.resize(num_queries, 0);
        }
        timings.add_select_prep(t.lap());

        // === Threshold selection (before the center pass: the count model
        // uses only the security bound and candidate counts, so the center
        // pass can split hi/tail directly and no demotion pass is needed).
        //
        // We tighten the hi threshold above the security floor based on
        // counts: treat the eligible neighborhood size as
        // `ring_candidates + (num_queries - qi - 1)` (directed center cell),
        // and pick a dot threshold t in [security, 1] that targets
        // ~PACKED_HI_BUDGET candidates above t under a simple "uniform on
        // [security, 1]" model. Anything safe at/below t goes to the tail.
        let ring_candidates_est = if PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER {
            ring_candidates_all
        } else {
            ring_candidates_eligible
        };
        for qi in 0..num_queries {
            let security = self.security_thresholds[qi];
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
        timings.add_ring_thresholds(t.lap());

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
        let tail_keys = &mut self.tail_keys[..num_queries];

        // Dense center cell: band-prune the O(occ²) full scan. `band_radius`
        // is cell-level (it depends only on occupancy/extent), so compute once
        // for the whole group; the per-query decision is whether the band is a
        // strict inner subset of that query's security coverage.
        let dense_radius = if center_len > crate::policy::DENSE_CELL_THRESHOLD {
            grid.dense_band_radius(cell as u32, crate::policy::DENSE_BAND_TARGET_COUNT)
        } else {
            None
        };

        match dense_radius {
            Some(r_claim) if r_claim > 0.0 => {
                self.center_bound.resize(num_queries, 0.0);
                self.band_mode.resize(num_queries, false);
                // Gather a band a hair wider than `r_claim` to absorb f32 error,
                // then keep points with `dot > band_bound = 1 - r_claim²/2`.
                // By the band's superset property every center point within
                // chord `r_claim` (hence `dot > band_bound`) is in the band, so
                // the coverage is complete down to `band_bound`; the shell
                // takeover backstops anything below it (rare for a dense cell,
                // where the cell closes within the band).
                let r_gather = r_claim * (1.0 + 1e-3);
                let band_bound = (1.0 - 0.5 * r_claim * r_claim).clamp(-1.0, 1.0);
                for qi in 0..num_queries {
                    let security = security_thresholds[qi];
                    if band_bound <= security {
                        // Band would cover the whole security region (no win)
                        // and would wrongly claim less coverage than the full
                        // scan; fall back to a directed full scan for this
                        // query (rare in a genuinely dense cell). center_bound
                        // and thresholds stay at their defaults (security/hi).
                        let hi = self.thresholds[qi];
                        for pos in (qi + 1)..center_len {
                            let dot = fp::dot3_f32(
                                xs[pos],
                                ys[pos],
                                zs[pos],
                                query_x[qi],
                                query_y[qi],
                                query_z[qi],
                            );
                            if dot <= security {
                                continue;
                            }
                            let slot = (center_soa_start + pos) as u32;
                            if dot > hi {
                                chunk0_keys[qi].push(make_desc_key(dot, slot));
                            } else {
                                self.center_tail_counts[qi] += 1;
                            }
                        }
                        continue;
                    }

                    self.center_bound[qi] = band_bound;
                    self.band_mode[qi] = true;
                    // The band covers only `(band_bound, 1]`. The model's
                    // hi/tail split assumes the tail extends down to `security`,
                    // which the band does NOT — so it is meaningless here: emit
                    // the whole band as chunk0 and disable the tail. chunk0
                    // exhaustion then reports `center_bound = band_bound` and the
                    // shell takeover covers everything below it. Lower the ring
                    // threshold to `band_bound` too so the shared ring pass
                    // collects every ring point above the floor into chunk0
                    // (regardless of where the model's `hi` sat).
                    self.tail_possible[qi] = false;
                    self.thresholds[qi] = band_bound;
                    grid.dense_band_slots(
                        cell as u32,
                        query_x[qi],
                        query_y[qi],
                        query_z[qi],
                        r_gather,
                        &mut self.band_scratch,
                    );
                    for idx in 0..self.band_scratch.len() {
                        let slot = self.band_scratch[idx];
                        let pos = slot as usize - center_soa_start;
                        // Directed intra-bin filter: later slots only (this
                        // also excludes self at pos == qi).
                        if pos <= qi {
                            continue;
                        }
                        let dot = fp::dot3_f32(
                            xs[pos],
                            ys[pos],
                            zs[pos],
                            query_x[qi],
                            query_y[qi],
                            query_z[qi],
                        );
                        if dot <= band_bound {
                            continue;
                        }
                        chunk0_keys[qi].push(make_desc_key(dot, slot));
                    }
                }
            }
            _ => {
                let hi_thresholds = &self.thresholds[..num_queries];
                let full_chunks = center_len / 8;
                let (x_chunks, _) = xs.as_chunks::<8>();
                let (y_chunks, _) = ys.as_chunks::<8>();
                let (z_chunks, _) = zs.as_chunks::<8>();
                for chunk in 0..full_chunks {
                    let i = chunk * 8;
                    let candidates = fp::PointChunk8::from_array_refs(
                        &x_chunks[chunk],
                        &y_chunks[chunk],
                        &z_chunks[chunk],
                    );

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
                            mask_bits &= u32::MAX << (rel + 1);
                            if mask_bits == 0 {
                                continue;
                            }
                        }

                        // Split into hi (above the tightened threshold -> chunk0)
                        // and the [security, t] band (-> tail) in one pass; the old
                        // post-hoc demotion loop is gone.
                        let hi_bits = dots.mask_gt(hi_thresholds[qi]) & mask_bits;
                        let band_bits = mask_bits & !hi_bits;
                        let mut hi = hi_bits;
                        let dots_arr = dots.to_array();
                        while hi != 0 {
                            let lane = hi.trailing_zeros() as usize;
                            let slot = (center_soa_start + i + lane) as u32;
                            chunk0_keys[qi].push(make_desc_key(dots_arr[lane], slot));
                            hi &= hi - 1;
                        }
                        self.center_tail_counts[qi] += band_bits.count_ones() as usize;
                    }
                }

                let rem = center_len % 8;
                if rem != 0 {
                    let i = full_chunks * 8;
                    debug_assert!(i < center_len);
                    let valid_bits = (1u32 << (rem as u32)) - 1;

                    let mut xbuf = [0.0f32; 8];
                    let mut ybuf = [0.0f32; 8];
                    let mut zbuf = [0.0f32; 8];
                    xbuf[..rem].copy_from_slice(&xs[i..]);
                    ybuf[..rem].copy_from_slice(&ys[i..]);
                    zbuf[..rem].copy_from_slice(&zs[i..]);

                    let candidates = fp::PointChunk8::from_arrays(xbuf, ybuf, zbuf);
                    for qi in 0..num_queries {
                        let dots = candidates.dots(query_x[qi], query_y[qi], query_z[qi]);
                        let mut mask_bits = dots.mask_gt(security_thresholds[qi]) & valid_bits;
                        if mask_bits == 0 {
                            continue;
                        }

                        // Directed intra-bin filter for the padded tail chunk:
                        // query qi may only emit candidate positions strictly after qi.
                        if qi >= i {
                            let rel = qi - i;
                            debug_assert!(rel < 8);
                            mask_bits &= !((1u32 << (rel + 1)) - 1);
                            if mask_bits == 0 {
                                continue;
                            }
                        }

                        let hi_bits = dots.mask_gt(hi_thresholds[qi]) & mask_bits;
                        let band_bits = mask_bits & !hi_bits;
                        let mut hi = hi_bits;
                        let dots_arr = dots.to_array();
                        while hi != 0 {
                            let lane = hi.trailing_zeros() as usize;
                            let slot = (center_soa_start + i + lane) as u32;
                            chunk0_keys[qi].push(make_desc_key(dots_arr[lane], slot));
                            hi &= hi - 1;
                        }
                        self.center_tail_counts[qi] += band_bits.count_ones() as usize;
                    }
                }
            }
        }
        for qi in 0..num_queries {
            if !tail_keys[qi].is_empty() {
                self.tail_possible[qi] = true;
            }
        }
        timings.add_center_pass(t.lap());

        // === Ring pass: collect "hi" candidates into chunk0.
        //
        // (A per-(ring cell, query) cap-prune was tried here and measured
        // as a net loss: ring cells are adjacent cells, whose caps almost
        // always overlap the threshold region, so the prune rarely fires.)
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
            let (x_chunks, _) = xs.as_chunks::<8>();
            let (y_chunks, _) = ys.as_chunks::<8>();
            let (z_chunks, _) = zs.as_chunks::<8>();
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let candidates = fp::PointChunk8::from_array_refs(
                    &x_chunks[chunk],
                    &y_chunks[chunk],
                    &z_chunks[chunk],
                );

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
                let (slot_base, valid_bits, candidates) = if full_chunks > 0 {
                    let i = range_len - 8;
                    let valid_bits = (0xffu32 << (8 - rem)) & 0xff;
                    (
                        soa_start + i,
                        valid_bits,
                        fp::PointChunk8::from_array_refs(
                            xs[i..].try_into().unwrap(),
                            ys[i..].try_into().unwrap(),
                            zs[i..].try_into().unwrap(),
                        ),
                    )
                } else {
                    debug_assert!(range_len < 8);
                    let valid_bits = (1u32 << (rem as u32)) - 1;

                    let mut xbuf = [0.0f32; 8];
                    let mut ybuf = [0.0f32; 8];
                    let mut zbuf = [0.0f32; 8];
                    xbuf[..rem].copy_from_slice(xs);
                    ybuf[..rem].copy_from_slice(ys);
                    zbuf[..rem].copy_from_slice(zs);

                    (
                        soa_start,
                        valid_bits,
                        fp::PointChunk8::from_arrays(xbuf, ybuf, zbuf),
                    )
                };

                for (qi, &query_slot) in queries.iter().enumerate() {
                    let dots = candidates.dots(query_x[qi], query_y[qi], query_z[qi]);
                    let mut mask_bits = dots.mask_gt(thresholds[qi]) & valid_bits;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr = dots.to_array();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (slot_base + lane) as u32;
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
        let keys = chunk0_keys.iter().map(Vec::len).sum::<usize>()
            + tail_keys.iter().map(Vec::len).sum::<usize>();
        let capacity = chunk0_keys.iter().map(Vec::capacity).sum::<usize>()
            + tail_keys.iter().map(Vec::capacity).sum::<usize>();
        timings.observe_key_storage(keys, capacity);
        timings.add_tail_possible_queries(
            self.tail_possible[..num_queries]
                .iter()
                .filter(|&&possible| possible)
                .count(),
        );
        timings.add_center_tail_keys(self.center_tail_counts[..num_queries].iter().sum::<usize>());

        PreparedPackedGroupStatus::Ready(PreparedPackedGroup {
            scratch: self,
            group,
            group_gen,
            tail_built_any: false,
        })
    }
}
