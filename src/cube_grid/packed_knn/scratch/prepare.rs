use super::super::timing::PackedLapTimer;
use super::helpers::{make_desc_key, outside_max_dot_xyz, security_planes_3x3_interior};
use super::*;
use crate::fp;
use crate::policy::{
    PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER, PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER,
    PACKED_HI_BUDGET,
};

#[inline]
fn finish_interior_security(
    s_min: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    ring2: &[u32],
    grid: &CubeMapGrid,
) -> f32 {
    // For the interior single-face case, the nearest outside point is reached
    // by crossing the closest boundary great circle. If we ever see a
    // non-positive signed distance (numerical issues), fall back to the cap
    // bound used by boundary cells.
    if s_min > 0.0 && s_min.is_finite() {
        let s = (s_min - crate::tolerances::GRID_PLANE_PAD).clamp(0.0, 1.0);
        (1.0 - s * s).max(0.0).sqrt()
    } else {
        outside_max_dot_xyz(qx, qy, qz, ring2, grid)
    }
}

#[inline]
fn scalar_interior_security(
    planes: &[glam::Vec3; 4],
    qx: f32,
    qy: f32,
    qz: f32,
    ring2: &[u32],
    grid: &CubeMapGrid,
) -> f32 {
    let mut s_min = 1.0f32;
    for n in planes {
        s_min = s_min.min(fp::dot3_f32(n.x, n.y, n.z, qx, qy, qz));
    }
    finish_interior_security(s_min, qx, qy, qz, ring2, grid)
}

fn append_interior_security_thresholds(
    out: &mut Vec<f32>,
    planes: &[glam::Vec3; 4],
    qx_src: &[f32],
    qy_src: &[f32],
    qz_src: &[f32],
    ring2: &[u32],
    grid: &CubeMapGrid,
) {
    debug_assert_eq!(qx_src.len(), qy_src.len());
    debug_assert_eq!(qx_src.len(), qz_src.len());

    let (x_chunks, x_rem) = qx_src.as_chunks::<8>();
    let (y_chunks, y_rem) = qy_src.as_chunks::<8>();
    let (z_chunks, z_rem) = qz_src.as_chunks::<8>();
    for chunk in 0..x_chunks.len() {
        let queries =
            fp::PointChunk8::from_array_refs(&x_chunks[chunk], &y_chunks[chunk], &z_chunks[chunk]);
        let mut s_min = [1.0f32; 8];
        for n in planes {
            let signed = queries.dots(n.x, n.y, n.z).to_array();
            for lane in 0..8 {
                s_min[lane] = s_min[lane].min(signed[lane]);
            }
        }
        #[cfg(all(target_feature = "avx2", not(feature = "simd_scalar")))]
        {
            let (mut thresholds, valid_mask) =
                fp::interior_security_thresholds8(s_min, crate::tolerances::GRID_PLANE_PAD);
            if valid_mask != 0xff {
                for lane in 0..8 {
                    if valid_mask & (1 << lane) == 0 {
                        thresholds[lane] = finish_interior_security(
                            s_min[lane],
                            x_chunks[chunk][lane],
                            y_chunks[chunk][lane],
                            z_chunks[chunk][lane],
                            ring2,
                            grid,
                        );
                    }
                }
            }
            out.extend_from_slice(&thresholds);
        }
        #[cfg(not(all(target_feature = "avx2", not(feature = "simd_scalar"))))]
        for lane in 0..8 {
            out.push(finish_interior_security(
                s_min[lane],
                x_chunks[chunk][lane],
                y_chunks[chunk][lane],
                z_chunks[chunk][lane],
                ring2,
                grid,
            ));
        }
    }

    // Preserve the scalar path for the final partial chunk.
    for i in 0..x_rem.len() {
        out.push(scalar_interior_security(
            planes, x_rem[i], y_rem[i], z_rem[i], ring2, grid,
        ));
    }
}

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

        let query_bin = group.query_bin();
        let layout = group.layout();
        let num_queries = group.query_count();
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
                let (bin_b, _) = layout.bin_local(n_start as u32);
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
                append_interior_security_thresholds(
                    &mut self.security_thresholds,
                    &planes,
                    qx_src,
                    qy_src,
                    qz_src,
                    ring2,
                    grid,
                );
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

                    // Candidate positions in this chunk are [i, i+7]. Since the
                    // directed filter requires candidate_pos > query_pos, the
                    // query at i+7 has no eligible lane in this chunk.
                    let qi_end = (i + 7).min(num_queries);
                    for (
                        qi,
                        (((((&qx, &qy), &qz), &security), &hi_threshold), (keys, tail_count)),
                    ) in query_x[..qi_end]
                        .iter()
                        .zip(&query_y[..qi_end])
                        .zip(&query_z[..qi_end])
                        .zip(&security_thresholds[..qi_end])
                        .zip(&hi_thresholds[..qi_end])
                        .zip(
                            chunk0_keys[..qi_end]
                                .iter_mut()
                                .zip(&mut self.center_tail_counts[..qi_end]),
                        )
                        .enumerate()
                    {
                        let dots = candidates.dots(qx, qy, qz);
                        let mut mask_bits = dots.mask_gt(security);
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
                        let hi_bits = dots.mask_gt(hi_threshold) & mask_bits;
                        let band_bits = mask_bits & !hi_bits;
                        if hi_bits != 0 {
                            let mut hi = hi_bits;
                            let dots_arr = dots.to_array();
                            while hi != 0 {
                                let lane = hi.trailing_zeros() as usize;
                                let slot = (center_soa_start + i + lane) as u32;
                                keys.push(make_desc_key(dots_arr[lane], slot));
                                hi &= hi - 1;
                            }
                        }
                        *tail_count += band_bits.count_ones() as usize;
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
                    // The final valid candidate is at i+rem-1, so queries at
                    // or beyond that position have no eligible lane here.
                    let qi_end = (i + rem - 1).min(num_queries);
                    for (
                        qi,
                        (((((&qx, &qy), &qz), &security), &hi_threshold), (keys, tail_count)),
                    ) in query_x[..qi_end]
                        .iter()
                        .zip(&query_y[..qi_end])
                        .zip(&query_z[..qi_end])
                        .zip(&security_thresholds[..qi_end])
                        .zip(&hi_thresholds[..qi_end])
                        .zip(
                            chunk0_keys[..qi_end]
                                .iter_mut()
                                .zip(&mut self.center_tail_counts[..qi_end]),
                        )
                        .enumerate()
                    {
                        let dots = candidates.dots(qx, qy, qz);
                        let mut mask_bits = dots.mask_gt(security) & valid_bits;
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

                        let hi_bits = dots.mask_gt(hi_threshold) & mask_bits;
                        let band_bits = mask_bits & !hi_bits;
                        if hi_bits != 0 {
                            let mut hi = hi_bits;
                            let dots_arr = dots.to_array();
                            while hi != 0 {
                                let lane = hi.trailing_zeros() as usize;
                                let slot = (center_soa_start + i + lane) as u32;
                                keys.push(make_desc_key(dots_arr[lane], slot));
                                hi &= hi - 1;
                            }
                        }
                        *tail_count += band_bits.count_ones() as usize;
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

                for ((((&qx, &qy), &qz), &threshold), keys) in query_x
                    .iter()
                    .zip(query_y)
                    .zip(query_z)
                    .zip(thresholds)
                    .zip(chunk0_keys.iter_mut())
                {
                    let dots = candidates.dots(qx, qy, qz);
                    let mut mask_bits = dots.mask_gt(threshold);
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr = dots.to_array();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (soa_start + i + lane) as u32;
                        let dot = dots_arr[lane];
                        keys.push(make_desc_key(dot, slot));
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

                for ((((&qx, &qy), &qz), &threshold), keys) in query_x
                    .iter()
                    .zip(query_y)
                    .zip(query_z)
                    .zip(thresholds)
                    .zip(chunk0_keys.iter_mut())
                {
                    let dots = candidates.dots(qx, qy, qz);
                    let mut mask_bits = dots.mask_gt(threshold) & valid_bits;
                    if mask_bits == 0 {
                        continue;
                    }

                    let dots_arr = dots.to_array();
                    while mask_bits != 0 {
                        let lane = mask_bits.trailing_zeros() as usize;
                        let slot = (slot_base + lane) as u32;
                        let dot = dots_arr[lane];
                        keys.push(make_desc_key(dot, slot));
                        mask_bits &= mask_bits - 1;
                    }
                }
            }
        }
        timings.add_ring_pass(t.lap());
        let chunk0_candidates = chunk0_keys.iter().map(Vec::len).sum::<usize>();
        let keys = chunk0_keys.iter().map(Vec::len).sum::<usize>()
            + tail_keys.iter().map(Vec::len).sum::<usize>();
        let capacity = chunk0_keys.iter().map(Vec::capacity).sum::<usize>()
            + tail_keys.iter().map(Vec::capacity).sum::<usize>();
        timings.observe_key_storage(keys, capacity);
        timings.add_chunk0_keys(chunk0_candidates);
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

#[cfg(test)]
mod security_tests {
    use super::*;

    fn interior_fixture() -> (Vec<glam::Vec3>, CubeMapGrid, usize) {
        let base = glam::Vec3::new(1.0, 0.23, 0.27).normalize();
        let points: Vec<glam::Vec3> = (0..17)
            .map(|i| {
                let y = (i as f32 - 8.0) * 1e-4;
                let z = ((i * 7 % 17) as f32 - 8.0) * 1e-4;
                (base + glam::Vec3::new(0.0, y, z)).normalize()
            })
            .collect();
        let grid = CubeMapGrid::new(&points, 8);
        let cell = grid.point_to_cell(base);
        let start = grid.cell_offsets[cell] as usize;
        let end = grid.cell_offsets[cell + 1] as usize;
        assert_eq!(end - start, points.len());
        assert!(security_planes_3x3_interior(cell, &grid).is_some());
        (points, grid, cell)
    }

    fn assert_vector_matches_scalar(
        grid: &CubeMapGrid,
        cell: usize,
        planes: &[glam::Vec3; 4],
        qx: &[f32],
        qy: &[f32],
        qz: &[f32],
    ) {
        let ring2 = grid.cell_ring2(cell);
        let mut vector = Vec::new();
        append_interior_security_thresholds(&mut vector, planes, qx, qy, qz, ring2, grid);
        let scalar: Vec<f32> = (0..qx.len())
            .map(|i| scalar_interior_security(planes, qx[i], qy[i], qz[i], ring2, grid))
            .collect();
        assert_eq!(
            vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn interior_security_chunks_match_scalar_bitwise_with_remainder() {
        let (_points, grid, cell) = interior_fixture();
        let start = grid.cell_offsets[cell] as usize;
        let end = grid.cell_offsets[cell + 1] as usize;
        let planes = security_planes_3x3_interior(cell, &grid).unwrap();
        assert_vector_matches_scalar(
            &grid,
            cell,
            &planes,
            &grid.cell_points_x[start..end],
            &grid.cell_points_y[start..end],
            &grid.cell_points_z[start..end],
        );
    }

    #[test]
    fn interior_security_fallbacks_match_scalar_bitwise() {
        let (_points, grid, cell) = interior_fixture();
        let qx = [1.0f32; 8];
        let qy = [0.0f32; 8];
        let qz = [0.0f32; 8];
        let signed_zero = [glam::Vec3::new(-0.0, -0.0, -0.0); 4];
        assert_vector_matches_scalar(&grid, cell, &signed_zero, &qx, &qy, &qz);

        let nonfinite = [glam::Vec3::new(f32::NEG_INFINITY, 0.0, 0.0); 4];
        assert_vector_matches_scalar(&grid, cell, &nonfinite, &qx, &qy, &qz);
    }

    #[test]
    fn boundary_cell_keeps_cap_security_path() {
        let (_points, grid, _) = interior_fixture();
        let boundary_cell = grid.point_to_cell(glam::Vec3::new(1.0, 0.99, 0.0).normalize());
        assert!(security_planes_3x3_interior(boundary_cell, &grid).is_none());
    }
}
