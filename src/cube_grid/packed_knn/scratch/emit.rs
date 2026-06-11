use super::super::timing::PackedLapTimer;
use super::helpers::{key_to_dot, key_to_idx, sort_keys_u64};
use super::*;
use crate::fp;

impl PackedKnnCellScratch {
    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn ensure_tail_directed_for(
        &mut self,
        qi: usize,
        group_queries: &[u32],
        group_gen: u32,
        tail_built_any: &mut bool,
        grid: &CubeMapGrid,
        slot_gen_map: &[u32],
        local_shift: u32,
        local_mask: u32,
        timings: &mut PackedKnnTimings,
    ) {
        // Tail candidates have already been partitioned into (hi, tail, unsafe) buckets
        // during `prepare_group_directed`. Bin/local decoding is only needed there.
        let _ = (slot_gen_map, local_shift, local_mask);

        let Some(gen) = self.tail_ready_gen.get(qi).copied() else {
            return;
        };
        if gen == group_gen {
            return;
        }
        if !*tail_built_any {
            *tail_built_any = true;
            timings.inc_tail_builds();
        }
        self.tail_ready_gen[qi] = group_gen;

        // Keep any precomputed center-tail candidates already stored in `tail_keys[qi]` and
        // append ring-tail candidates here.
        self.tail_pos[qi] = 0;
        debug_assert!(self.tail_possible.get(qi).copied().unwrap_or(false));

        let mut t_tail = PackedLapTimer::start();
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

            let query_slot = group_queries[qi];
            let query_slot_usize = query_slot as usize;
            let qx_s = grid.cell_points_x[query_slot_usize];
            let qy_s = grid.cell_points_y[query_slot_usize];
            let qz_s = grid.cell_points_z[query_slot_usize];

            let full_chunks = range_len / 8;
            for chunk in 0..full_chunks {
                let i = chunk * 8;
                let candidates = fp::PointChunk8::from_slices(&xs[i..], &ys[i..], &zs[i..]);
                let dots = candidates.dots(qx_s, qy_s, qz_s);

                let safe_bits = dots.mask_gt(self.security_thresholds[qi]);
                let hi_bits = dots.mask_gt(self.thresholds[qi]);

                let mut tail_bits = safe_bits & !hi_bits;
                if tail_bits == 0 {
                    continue;
                }

                let dots_arr = dots.to_array();
                while tail_bits != 0 {
                    let lane = tail_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    if slot != query_slot {
                        let dot = dots_arr[lane];
                        self.tail_keys[qi].push(super::helpers::make_desc_key(dot, slot));
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

                if slot == query_slot {
                    continue;
                }

                let dot = fp::dot3_f32(cx, cy, cz, qx_s, qy_s, qz_s);
                if dot > self.security_thresholds[qi] && dot <= self.thresholds[qi] {
                    self.tail_keys[qi].push(super::helpers::make_desc_key(dot, slot));
                }
            }
        }
        timings.add_ring_fallback(t_tail.lap());

        if self.tail_keys[qi].is_empty() {
            self.tail_possible[qi] = false;
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    pub(super) fn next_chunk(
        &mut self,
        qi: usize,
        group_gen: u32,
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
                let mut t = PackedLapTimer::start();
                let keys = &mut self.chunk0_keys.get_mut(qi)?;
                let start = *self.chunk0_pos.get(qi)?;
                if start >= keys.len() {
                    return None;
                }
                let remaining = &mut keys[start..];
                let n_target = k.min(out.len());
                if remaining.is_empty() {
                    return None;
                }
                timings.add_select_query_prep(t.lap());

                // If remaining candidates are close to the requested emission size,
                // skip partition and just sort the remainder.
                //
                // Important: this must scale with `n_target` (k can shrink after the first
                // packed chunk), otherwise we end up sorting large remainders when asking
                // for small k (e.g. k=8).
                if remaining.len() <= 2 * n_target {
                    let emit = remaining.len().min(n_target);
                    let sort_len = remaining.len();
                    sort_keys_u64(remaining);
                    timings.add_select_sort_sized(t.lap(), sort_len);
                    for (dst, key) in out[..emit].iter_mut().zip(remaining.iter()) {
                        *dst = key_to_idx(*key);
                    }
                    timings.add_select_scatter(t.lap());
                    let last_dot = key_to_dot(remaining[emit - 1]);
                    self.chunk0_pos[qi] = start + emit;
                    let has_more = self.chunk0_pos[qi] < keys.len();
                    let unseen_bound = if has_more {
                        last_dot
                    } else if self.tail_possible[qi] {
                        self.thresholds[qi]
                    } else {
                        self.security_thresholds[qi]
                    };
                    return Some(PackedChunk {
                        n: emit,
                        unseen_bound,
                    });
                }

                // Large chunk: partition to extract top n, then sort those.
                let n = n_target.min(remaining.len());
                if remaining.len() > n {
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t.lap());
                }
                sort_keys_u64(&mut remaining[..n]);
                timings.add_select_sort_sized(t.lap(), n);
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t.lap());
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
                    self.tail_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "tail stage requested before ensure_tail"
                );
                let mut t = PackedLapTimer::start();
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
                timings.add_select_query_prep(t.lap());
                if remaining.len() > n {
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t.lap());
                }
                sort_keys_u64(&mut remaining[..n]);
                timings.add_select_sort_sized(t.lap(), n);
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t.lap());
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
            PackedStage::ExpandR2 => {
                debug_assert!(
                    self.expand2_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "expand_r2 stage requested before ensure_expand_r2"
                );
                let mut t_stage = PackedLapTimer::start();
                let mut t = PackedLapTimer::start();
                let keys = &mut self.expand2_keys.get_mut(qi)?;
                let start = *self.expand2_pos.get(qi)?;
                if start >= keys.len() {
                    return None;
                }
                let remaining = &mut keys[start..];
                let n = k.min(out.len()).min(remaining.len());
                if n == 0 {
                    return None;
                }
                timings.add_select_query_prep(t.lap());
                if remaining.len() > n {
                    remaining.select_nth_unstable(n - 1);
                    timings.add_select_partition(t.lap());
                }
                sort_keys_u64(&mut remaining[..n]);
                timings.add_select_sort_sized(t.lap(), n);
                for (dst, key) in out[..n].iter_mut().zip(remaining[..n].iter()) {
                    *dst = key_to_idx(*key);
                }
                timings.add_select_scatter(t.lap());
                let last_dot = key_to_dot(remaining[n - 1]);
                self.expand2_pos[qi] = start + n;
                let has_more = self.expand2_pos[qi] < keys.len();
                let unseen_bound = if has_more {
                    last_dot
                } else {
                    self.security2[qi]
                };
                timings.add_expand_r2_select(t_stage.lap());
                Some(PackedChunk { n, unseen_bound })
            }
        }
    }
}
