use super::super::timing::PackedLapTimer;
use super::helpers::{key_to_dot, key_to_idx, sort_keys_u64};
use super::*;
use crate::fp;

impl PackedKnnCellScratch {
    #[cfg_attr(feature = "profiling", inline(never))]
    // Tail candidates were already partitioned into (hi, tail, unsafe) buckets
    // during `prepare_group_directed`; bin/local decoding lives there, so no
    // layout context is needed here.
    pub(super) fn ensure_tail_directed_for(
        &mut self,
        qi: usize,
        group_queries: &[u32],
        group_gen: u32,
        tail_built_any: &mut bool,
        grid: &CubeMapGrid,
        timings: &mut PackedKnnTimings,
    ) {
        let Some(gen) = self.tail_ready_gen.get(qi).copied() else {
            return;
        };
        if gen == group_gen {
            return;
        }
        timings.inc_tail_requested_queries();
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
        let query_slot = group_queries[qi];
        let query_slot_usize = query_slot as usize;
        let qx_s = grid.cell_points_x[query_slot_usize];
        let qy_s = grid.cell_points_y[query_slot_usize];
        let qz_s = grid.cell_points_z[query_slot_usize];
        let security_threshold = self.security_thresholds[qi];
        let threshold = self.thresholds[qi];
        let tail_keys = &mut self.tail_keys[qi];
        let old_len = tail_keys.len();
        let mut ring_dot_evaluations = 0usize;
        for r in &self.cell_ranges[1..] {
            if r.kind == PackedCellRangeKind::SameBinEarlier {
                continue;
            }

            let soa_start = r.soa_start;
            let soa_end = r.soa_end;
            let range_len = soa_end - soa_start;
            ring_dot_evaluations += range_len;
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
                let dots = candidates.dots(qx_s, qy_s, qz_s);

                let safe_bits = dots.mask_gt(security_threshold);
                let hi_bits = dots.mask_gt(threshold);

                let mut tail_bits = safe_bits & !hi_bits;
                if tail_bits == 0 {
                    continue;
                }

                let dots_arr = dots.to_array();
                while tail_bits != 0 {
                    let lane = tail_bits.trailing_zeros() as usize;
                    let slot = (soa_start + i + lane) as u32;
                    debug_assert_ne!(slot, query_slot);
                    let dot = dots_arr[lane];
                    tail_keys.push(super::helpers::make_desc_key(dot, slot));
                    tail_bits &= tail_bits - 1;
                }
            }

            let tail_start = full_chunks * 8;
            for i in tail_start..range_len {
                let cx = xs[i];
                let cy = ys[i];
                let cz = zs[i];
                let slot = (soa_start + i) as u32;
                debug_assert_ne!(slot, query_slot);

                let dot = fp::dot3_f32(cx, cy, cz, qx_s, qy_s, qz_s);
                if dot > security_threshold && dot <= threshold {
                    tail_keys.push(super::helpers::make_desc_key(dot, slot));
                }
            }
        }
        timings.add_ring_fallback(t_tail.lap());
        let added = tail_keys.len() - old_len;
        timings.add_ring_tail_rescan(added == 0, ring_dot_evaluations);
        let tail_empty = tail_keys.is_empty();
        let capacity = self.chunk0_keys[..group_queries.len()]
            .iter()
            .map(Vec::capacity)
            .sum::<usize>()
            + self.tail_keys[..group_queries.len()]
                .iter()
                .map(Vec::capacity)
                .sum::<usize>();
        timings.observe_key_storage(added, capacity);

        if tail_empty {
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
        let coverage_bound = if self.band_mode.get(qi).copied().unwrap_or(false) {
            self.center_bound[qi]
        } else {
            self.security_thresholds[qi]
        } + crate::tolerances::GRID_DOT_BOUND_PAD;

        let n_target = k.min(out.len());

        match stage {
            PackedStage::Chunk0 => {
                let run = emit_run::<true>(
                    self.chunk0_keys.get_mut(qi)?,
                    self.chunk0_pos.get_mut(qi)?,
                    n_target,
                    out,
                    timings,
                )?;
                let post_chunk_bound = if self.tail_possible[qi] {
                    self.thresholds[qi] + crate::tolerances::GRID_DOT_BOUND_PAD
                } else {
                    coverage_bound
                };
                let unseen_bound = if run.has_more {
                    run.last_dot.max(post_chunk_bound)
                } else if self.tail_possible[qi] {
                    post_chunk_bound
                } else {
                    coverage_bound
                };
                Some(PackedChunk {
                    n: run.n,
                    first_dot: run.first_dot,
                    unseen_bound,
                })
            }
            PackedStage::Tail => {
                debug_assert!(
                    self.tail_ready_gen.get(qi).copied().unwrap_or(0) == group_gen,
                    "tail stage requested before ensure_tail"
                );
                let run = emit_run::<false>(
                    self.tail_keys.get_mut(qi)?,
                    self.tail_pos.get_mut(qi)?,
                    n_target,
                    out,
                    timings,
                )?;
                let unseen_bound = if run.has_more {
                    run.last_dot.max(coverage_bound)
                } else {
                    coverage_bound
                };
                Some(PackedChunk {
                    n: run.n,
                    first_dot: run.first_dot,
                    unseen_bound,
                })
            }
        }
    }
}

/// One emitted run of `emit_run`; the caller maps `has_more` to its stage's
/// unseen-dot bound (Chunk0 falls through to tail/coverage bounds, Tail to
/// the coverage bound).
struct EmittedRun {
    n: usize,
    first_dot: f32,
    last_dot: f32,
    has_more: bool,
}

/// The partition→sort→scatter→advance sequence shared by the Chunk0
/// (small/large remainder) and Tail paths: take the top `n_target` of
/// `keys[*pos..]`, sort them ascending, scatter their slot indices into
/// `out`, and advance the cursor past what was emitted.
///
/// `WHOLE_SORT_SMALL` (the Chunk0 small-remainder path): when the remainder
/// is within 2× of `n_target`, skip the partition and sort it whole. This
/// must scale with `n_target` — k can shrink after the first packed chunk,
/// and partitioning is what keeps small-k asks from sorting large remainders
/// (e.g. k=8). Const-generic + inline(always) so each call site keeps its
/// pre-extraction codegen (an out-of-line call here measured +0.6%
/// instructions on the whole build).
#[inline(always)]
fn emit_run<const WHOLE_SORT_SMALL: bool>(
    keys: &mut [u64],
    pos: &mut usize,
    n_target: usize,
    out: &mut [u32],
    timings: &mut PackedKnnTimings,
) -> Option<EmittedRun> {
    let mut t = PackedLapTimer::start();
    let total = keys.len();
    let start = *pos;
    if start >= total {
        return None;
    }
    let remaining = &mut keys[start..];
    timings.add_select_query_prep(t.lap());

    let n = if WHOLE_SORT_SMALL && remaining.len() <= 2 * n_target {
        let sort_len = remaining.len();
        sort_keys_u64(remaining);
        timings.add_select_sort_sized(t.lap(), sort_len);
        remaining.len().min(n_target)
    } else {
        let n = n_target.min(remaining.len());
        if remaining.len() > n {
            remaining.select_nth_unstable(n - 1);
            timings.add_select_partition(t.lap());
        }
        sort_keys_u64(&mut remaining[..n]);
        timings.add_select_sort_sized(t.lap(), n);
        n
    };
    for (dst, key) in out[..n].iter_mut().zip(remaining.iter()) {
        *dst = key_to_idx(*key);
    }
    timings.add_select_scatter(t.lap());
    let first_dot = key_to_dot(remaining[0]);
    let last_dot = key_to_dot(remaining[n - 1]);
    *pos = start + n;
    Some(EmittedRun {
        n,
        first_dot,
        last_dot,
        has_more: *pos < total,
    })
}
