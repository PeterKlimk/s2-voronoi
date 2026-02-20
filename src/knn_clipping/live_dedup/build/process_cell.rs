//! `process_cell` implementation for live-dedup cell building.

use crate::cube_grid::packed_knn::{PackedKnnCellScratch, PackedKnnTimings, PackedStage};

use crate::knn_clipping::topo2d::Topo2DBuilder;
use crate::knn_clipping::TerminationConfig;

use super::super::edge_checks::unpack_edge_key;
use super::super::packed::{pack_ref, DEFERRED, INVALID_INDEX};
use super::super::types::{DeferredSlot, EdgeCheck};

use super::CellContext;

struct CellOrchestrator<'a, 'b, 'c> {
    cell_sub: &'a mut crate::knn_clipping::timing::CellSubAccum,
    ctx: &'a mut CellContext,
    shard_ctx: &'a mut super::ShardContext<'b>,
    grid_ctx: &'a super::GridContext<'c>,
    termination: TerminationConfig,
    termination_max_k_cap: Option<usize>,
    i: usize,

    cell_neighbors_processed: usize,
    terminated: bool,
    knn_exhausted: bool,
    full_scan_done: bool,
    did_full_scan_fallback: bool,
    used_knn: bool,

    worst_cos: f32,
    max_k_requested: usize,
    knn_stage: crate::knn_clipping::timing::KnnCellStage,
    reached_schedule_max_k: bool,

    did_packed: bool,
    packed_security: f32,
    packed_safe_exhausted: bool,
    packed_tail_used: bool,

    directed_ctx: crate::cube_grid::DirectedCtx<'c>,
    incoming_checks: Vec<EdgeCheck>,
    incoming_edgechecks: usize,
    edgecheck_seed_clips: usize,
    cell_start: u32,
}

impl<'a, 'b, 'c> CellOrchestrator<'a, 'b, 'c> {
    fn new(
        cell_sub: &'a mut crate::knn_clipping::timing::CellSubAccum,
        ctx: &'a mut CellContext,
        shard_ctx: &'a mut super::ShardContext<'b>,
        grid_ctx: &'a super::GridContext<'c>,
        termination: TerminationConfig,
        termination_max_k_cap: Option<usize>,
        i: usize,
    ) -> Self {
        let directed_ctx = crate::cube_grid::DirectedCtx::new(
            shard_ctx.bin.as_u8(),
            shard_ctx.local.as_u32(),
            &grid_ctx.assignment.slot_gen_map,
            grid_ctx.assignment.local_shift,
            grid_ctx.assignment.local_mask,
        );

        Self {
            cell_sub,
            ctx,
            shard_ctx,
            grid_ctx,
            termination,
            termination_max_k_cap,
            i,
            cell_neighbors_processed: 0,
            terminated: false,
            knn_exhausted: false,
            full_scan_done: false,
            did_full_scan_fallback: false,
            used_knn: false,
            worst_cos: 1.0,
            max_k_requested: 0,
            knn_stage: crate::knn_clipping::timing::KnnCellStage::Resume(0),
            reached_schedule_max_k: false,
            did_packed: false,
            packed_security: 0.0,
            packed_safe_exhausted: false,
            packed_tail_used: false,
            directed_ctx,
            incoming_checks: Vec::new(),
            incoming_edgechecks: 0,
            edgecheck_seed_clips: 0,
            cell_start: 0,
        }
    }

    fn run(
        mut self,
        packed: Option<(
            &mut PackedKnnCellScratch,
            &mut PackedKnnTimings,
            usize,
            usize,
            usize,
        )>,
    ) {
        self.phase_1_init();
        self.phase_2_edgecheck_seeds();
        self.phase_3_packed_knn_seeds(packed);
        self.phase_4_resumable_knn();
        self.phase_5_full_scan_fallback();
        self.phase_6_extract_output();
    }

    fn phase_1_init(&mut self) {
        let points = self.grid_ctx.points;
        self.ctx.builder.reset(self.i, points[self.i]);
        self.ctx.attempted_neighbors.clear();
        self.ctx.neighbor_slots.clear();

        self.cell_start = self.shard_ctx.shard.output.cell_indices.len() as u32;
        self.shard_ctx
            .shard
            .output
            .set_cell_start(self.shard_ctx.local, self.cell_start);

        self.incoming_checks = self
            .shard_ctx
            .shard
            .dedup
            .take_edge_checks(self.shard_ctx.local);
        self.incoming_edgechecks = self.incoming_checks.len();
    }

    fn phase_2_edgecheck_seeds(&mut self) {
        if self.incoming_checks.is_empty() {
            return;
        }
        let cell_idx = u32::try_from(self.i).expect("cell index must fit in u32");
        let t_clip = crate::knn_clipping::timing::Timer::start();

        for check in &self.incoming_checks {
            let (a, b) = unpack_edge_key(check.key);
            let neighbor_idx = if a == cell_idx { b } else { a } as usize;
            let neighbor_slot = self.grid_ctx.grid.point_index_to_slot(neighbor_idx);

            if !self.ctx.attempted_neighbors.insert(neighbor_idx) {
                continue;
            }

            let neighbor = self.grid_ctx.points[neighbor_idx];
            if self
                .ctx
                .builder
                .clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor, check.hp_eps)
                .is_err()
            {
                break;
            }
            self.cell_neighbors_processed += 1;
        }
        self.cell_sub.add_clip(t_clip.elapsed());
        self.edgecheck_seed_clips = self.cell_neighbors_processed;
    }

    fn phase_3_packed_knn_seeds(
        &mut self,
        packed: Option<(
            &mut PackedKnnCellScratch,
            &mut PackedKnnTimings,
            usize,
            usize,
            usize,
        )>,
    ) {
        let Some((packed_scratch, packed_timings, qi, packed_k0_base, packed_k1)) = packed else {
            return;
        };

        self.did_packed = true;
        self.packed_security = packed_scratch.security(qi);

        let max_neighbors = self.grid_ctx.points.len().saturating_sub(1);
        let max_k0 = packed_k0_base.min(max_neighbors).max(1);
        let k1 = packed_k1.min(max_neighbors).max(1);
        let seed_clips = self.cell_neighbors_processed;
        let mut k_cur = max_k0.saturating_sub(seed_clips).max(k1).min(max_k0);

        let point_indices = self.grid_ctx.grid.point_indices();
        let mut stage = PackedStage::Chunk0;

        loop {
            self.ctx.packed_chunk.clear();
            self.ctx.packed_chunk.resize(k_cur, u32::MAX);

            let Some(chunk) = packed_scratch.next_chunk(
                qi,
                stage,
                k_cur,
                &mut self.ctx.packed_chunk,
                packed_timings,
            ) else {
                if stage == PackedStage::Chunk0 && packed_scratch.tail_possible(qi) {
                    packed_scratch.ensure_tail_directed_for(
                        qi,
                        self.grid_ctx.grid,
                        &self.grid_ctx.assignment.slot_gen_map,
                        self.grid_ctx.assignment.local_shift,
                        self.grid_ctx.assignment.local_mask,
                        packed_timings,
                    );
                    stage = PackedStage::Tail;
                    self.packed_tail_used = true;
                    k_cur = k1;
                    continue;
                }
                self.packed_safe_exhausted = true;
                break;
            };

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let mut term_ready = false;
            for pos in 0..chunk.n {
                let neighbor_slot = self.ctx.packed_chunk[pos];
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if neighbor_idx == self.i {
                    continue;
                }

                self.ctx.attempted_neighbors.mark(neighbor_idx);

                let neighbor = self.grid_ctx.points[neighbor_idx];
                let clip_result = match self.ctx.builder.clip_with_slot_result(
                    neighbor_idx,
                    neighbor_slot,
                    neighbor,
                ) {
                    Ok(r) => r,
                    Err(_) => break,
                };
                match clip_result {
                    crate::knn_clipping::topo2d::types::ClipResult::Unchanged => {
                        term_ready = true;
                    }
                    _ => {
                        term_ready = false;
                    }
                }

                if self.ctx.builder.is_bounded()
                    && clip_result == crate::knn_clipping::topo2d::types::ClipResult::Unchanged
                {
                    let bound = if pos + 1 < chunk.n {
                        let next_slot = self.ctx.packed_chunk[pos + 1];
                        let next = point_indices[next_slot as usize] as usize;
                        self.grid_ctx.points[self.i].dot(self.grid_ctx.points[next])
                    } else {
                        chunk.unseen_bound
                    };
                    if self.ctx.builder.can_terminate(bound) {
                        self.terminated = true;
                        break;
                    }
                }
                self.cell_neighbors_processed += 1;
                let dot = self.grid_ctx.points[self.i].dot(neighbor);
                self.worst_cos = self.worst_cos.min(dot);
            }
            self.cell_sub.add_clip(t_clip.elapsed());

            if self.terminated || self.ctx.builder.is_failed() {
                break;
            }

            if self.ctx.builder.is_bounded()
                && term_ready
                && self.ctx.builder.can_terminate(chunk.unseen_bound)
            {
                self.terminated = true;
                break;
            }

            k_cur = k1;
        }
    }

    fn phase_4_resumable_knn(&mut self) {
        let max_neighbors = self.grid_ctx.points.len().saturating_sub(1);
        let resume_k = crate::knn_clipping::KNN_RESUME_K.min(max_neighbors);

        if !self.terminated && !self.knn_exhausted && !self.ctx.builder.is_failed() && resume_k > 0
        {
            self.used_knn = true;
            self.max_k_requested = resume_k;
            self.ctx.neighbor_slots.clear();
            let t_knn = crate::knn_clipping::timing::Timer::start();
            let status = self
                .grid_ctx
                .grid
                .find_k_nearest_resumable_slots_directed_ctx_into(
                    self.grid_ctx.points[self.i],
                    self.i,
                    resume_k,
                    resume_k,
                    &mut self.ctx.scratch,
                    &mut self.ctx.neighbor_slots,
                    self.directed_ctx,
                );
            self.cell_sub.add_knn(t_knn.elapsed());

            self.knn_stage = crate::knn_clipping::timing::KnnCellStage::Resume(resume_k);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let point_indices = self.grid_ctx.grid.point_indices();
            self.terminated = clip_resumable_knn_neighbors(
                &mut self.ctx.builder,
                &mut self.ctx.attempted_neighbors,
                self.grid_ctx.points,
                self.i,
                &self.ctx.neighbor_slots,
                point_indices,
                self.termination,
                &mut self.cell_neighbors_processed,
                &mut self.worst_cos,
            );
            self.cell_sub.add_clip(t_clip.elapsed());

            if self.terminated {
                self.knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
            } else if status == crate::cube_grid::KnnStatus::Exhausted {
                self.knn_exhausted = true;
            }
        }

        if !self.terminated && !self.knn_exhausted && !self.ctx.builder.is_failed() {
            for &k_stage in crate::knn_clipping::KNN_RESTART_KS.iter() {
                let k = k_stage.min(max_neighbors);
                if k == 0 || k <= self.max_k_requested {
                    continue;
                }
                self.used_knn = true;
                self.max_k_requested = k;
                self.ctx.neighbor_slots.clear();

                let t_knn = crate::knn_clipping::timing::Timer::start();
                let status = self
                    .grid_ctx
                    .grid
                    .find_k_nearest_resumable_slots_directed_ctx_into(
                        self.grid_ctx.points[self.i],
                        self.i,
                        k,
                        k,
                        &mut self.ctx.scratch,
                        &mut self.ctx.neighbor_slots,
                        self.directed_ctx,
                    );
                self.cell_sub.add_knn(t_knn.elapsed());

                self.knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(k_stage);

                let t_clip = crate::knn_clipping::timing::Timer::start();
                let point_indices = self.grid_ctx.grid.point_indices();
                self.terminated = clip_resumable_knn_neighbors(
                    &mut self.ctx.builder,
                    &mut self.ctx.attempted_neighbors,
                    self.grid_ctx.points,
                    self.i,
                    &self.ctx.neighbor_slots,
                    point_indices,
                    self.termination,
                    &mut self.cell_neighbors_processed,
                    &mut self.worst_cos,
                );
                self.cell_sub.add_clip(t_clip.elapsed());

                if self.terminated {
                    break;
                }

                self.knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
                if self.knn_exhausted {
                    break;
                }
            }
        }

        let max_knn_target = crate::knn_clipping::KNN_RESTART_MAX.min(max_neighbors);
        if self.max_k_requested >= max_knn_target {
            self.reached_schedule_max_k = true;
        }

        if !self.terminated && self.ctx.builder.is_bounded() {
            let bound = if self.used_knn {
                self.worst_cos
            } else if self.did_packed && self.packed_safe_exhausted {
                self.packed_security
            } else {
                self.worst_cos
            };
            if self.ctx.builder.can_terminate(bound) {
                self.terminated = true;
            }
        }

        if !self.terminated
            && self.reached_schedule_max_k
            && !self.ctx.builder.is_failed()
            && self.ctx.builder.is_bounded()
        {
            let mut k = self
                .max_k_requested
                .max(crate::knn_clipping::KNN_RESTART_MAX)
                .min(max_neighbors);
            let cap = self
                .termination_max_k_cap
                .unwrap_or(max_neighbors)
                .min(max_neighbors)
                .max(k);

            while !self.terminated && !self.ctx.builder.is_failed() && k < cap {
                let next_k = (k.saturating_mul(crate::knn_clipping::TERMINATION_GROW_MULTIPLIER))
                    .max(k + crate::knn_clipping::TERMINATION_GROW_MIN_STEP)
                    .min(cap);
                if next_k <= k {
                    break;
                }

                self.used_knn = true;
                self.ctx.neighbor_slots.clear();

                let t_knn = crate::knn_clipping::timing::Timer::start();
                let status = self
                    .grid_ctx
                    .grid
                    .find_k_nearest_resumable_slots_directed_ctx_into(
                        self.grid_ctx.points[self.i],
                        self.i,
                        next_k,
                        next_k,
                        &mut self.ctx.scratch,
                        &mut self.ctx.neighbor_slots,
                        self.directed_ctx,
                    );
                self.cell_sub.add_knn(t_knn.elapsed());
                self.knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(next_k);

                let t_clip = crate::knn_clipping::timing::Timer::start();
                let point_indices = self.grid_ctx.grid.point_indices();
                self.terminated = clip_resumable_knn_neighbors(
                    &mut self.ctx.builder,
                    &mut self.ctx.attempted_neighbors,
                    self.grid_ctx.points,
                    self.i,
                    &self.ctx.neighbor_slots,
                    point_indices,
                    self.termination,
                    &mut self.cell_neighbors_processed,
                    &mut self.worst_cos,
                );
                self.cell_sub.add_clip(t_clip.elapsed());

                self.knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
                k = next_k;

                let kth_dot = self
                    .ctx
                    .neighbor_slots
                    .last()
                    .map(|&slot| {
                        let j = point_indices[slot as usize] as usize;
                        self.grid_ctx.points[self.i].dot(self.grid_ctx.points[j])
                    })
                    .unwrap_or(self.worst_cos);
                if self.ctx.builder.can_terminate(kth_dot) {
                    self.terminated = true;
                }

                if k >= max_neighbors {
                    self.terminated = true;
                }
            }
        }
    }

    fn phase_5_full_scan_fallback(&mut self) {
        if !self.ctx.builder.is_bounded() && !self.ctx.builder.is_failed() {
            self.did_full_scan_fallback = true;
            let already_clipped: rustc_hash::FxHashSet<usize> =
                self.ctx.builder.neighbor_indices_iter().collect();
            let gen = self.grid_ctx.points[self.i];

            let mut sorted_indices: Vec<usize> = (0..self.grid_ctx.points.len())
                .filter(|&j| {
                    if j == self.i || already_clipped.contains(&j) {
                        return false;
                    }
                    if self.grid_ctx.assignment.generator_bin[j] == self.shard_ctx.bin
                        && self.grid_ctx.assignment.global_to_local[j].as_u32()
                            < self.shard_ctx.local.as_u32()
                    {
                        return false;
                    }
                    true
                })
                .collect();
            sorted_indices.sort_by(|&a, &b| {
                let da = gen.dot(self.grid_ctx.points[a]);
                let db = gen.dot(self.grid_ctx.points[b]);
                db.partial_cmp(&da).unwrap()
            });
            for p_idx in sorted_indices {
                if !self.ctx.attempted_neighbors.insert(p_idx) {
                    continue;
                }
                let slot = self.grid_ctx.grid.point_index_to_slot(p_idx);
                if self
                    .ctx
                    .builder
                    .clip_with_slot(p_idx, slot, self.grid_ctx.points[p_idx])
                    .is_err()
                {
                    break;
                }
                self.cell_neighbors_processed += 1;
                if self.ctx.builder.is_bounded() {
                    break;
                }
            }
            self.full_scan_done = true;
        }

        if !self.ctx.builder.is_bounded() || self.ctx.builder.is_failed() {
            let (active, total) = self.ctx.builder.count_active_planes();
            let gen = self.grid_ctx.points[self.i];
            let neighbor_indices: Vec<usize> = self.ctx.builder.neighbor_indices_iter().collect();

            panic!(
                "Cell {} construction failed: bounded={}, failure={:?}, \
                 planes={}, active={}, vertices={}, \
                 did_packed={}, did_knn={}, did_full_scan={}\n\
                 Generator pos: {:?}\n\
                 First 10 neighbor indices: {:?}",
                self.i,
                self.ctx.builder.is_bounded(),
                self.ctx.builder.failure(),
                total,
                active,
                self.ctx.builder.vertex_count(),
                self.did_packed,
                self.used_knn,
                self.full_scan_done,
                gen,
                &neighbor_indices[..neighbor_indices.len().min(10)],
            );
        }
    }

    fn phase_6_extract_output(&mut self) {
        let stage = if self.did_full_scan_fallback {
            crate::knn_clipping::timing::KnnCellStage::FullScanFallback
        } else if self.used_knn {
            self.knn_stage
        } else if self.did_packed {
            if self.packed_tail_used {
                crate::knn_clipping::timing::KnnCellStage::PackedTail
            } else {
                crate::knn_clipping::timing::KnnCellStage::PackedChunk0
            }
        } else {
            self.knn_stage
        };

        self.cell_sub.add_cell_stage(
            stage,
            self.knn_exhausted,
            self.cell_neighbors_processed,
            self.packed_tail_used,
            self.packed_safe_exhausted,
            self.used_knn,
            self.incoming_edgechecks,
            self.edgecheck_seed_clips,
        );

        let mut t_post = crate::knn_clipping::timing::LapTimer::start();
        self.ctx
            .builder
            .to_vertex_data_full(&mut self.ctx.output_buffer)
            .expect("to_vertex_data_full failed after bounded check");
        self.cell_sub.add_cert(t_post.lap());

        let cell_idx = self.i as u32;
        let incoming_checks = std::mem::take(&mut self.incoming_checks);
        self.ctx.edge_scratch.collect_and_resolve(
            cell_idx,
            self.shard_ctx,
            &self.ctx.output_buffer,
            self.grid_ctx.assignment,
            incoming_checks,
        );
        let collect_resolve_time = t_post.lap();
        self.cell_sub.add_edge_collect(collect_resolve_time / 2);
        self.cell_sub.add_edge_resolve(collect_resolve_time / 2);

        let count = self.ctx.output_buffer.vertices.len();
        let shard = &mut *self.shard_ctx.shard;
        let local = self.shard_ctx.local;
        let bin = self.shard_ctx.bin;

        shard.output.set_cell_count(
            local,
            u8::try_from(count).expect("cell vertex count exceeds u8 capacity"),
        );

        {
            let vertex_indices = &mut self.ctx.edge_scratch.vertex_indices;
            for ((key, pos), vi) in self
                .ctx
                .output_buffer
                .vertices
                .iter()
                .copied()
                .zip(vertex_indices.iter_mut())
            {
                #[cfg(feature = "timing")]
                {
                    shard.triplet_keys += 1;
                }
                let owner_bin = self.grid_ctx.assignment.generator_bin[key[0] as usize];
                if owner_bin == bin {
                    if *vi == INVALID_INDEX {
                        let new_idx = shard.output.vertices.len() as u32;
                        shard.output.vertices.push(pos);
                        shard.output.vertex_keys.push(key);
                        *vi = new_idx;
                    }
                    let v_idx = *vi;
                    debug_assert!(v_idx != INVALID_INDEX, "missing on-shard vertex index");
                    shard.output.cell_indices.push(pack_ref(bin, v_idx));
                } else {
                    debug_assert_eq!(*vi, INVALID_INDEX, "received index for off-shard owner");
                    let source_slot = shard.output.cell_indices.len() as u32;
                    shard.output.cell_indices.push(DEFERRED);
                    shard.output.deferred.push(DeferredSlot {
                        key,
                        pos,
                        source_bin: bin,
                        source_slot,
                    });
                }
            }
        }
        self.cell_sub.add_key_dedup(t_post.lap());

        self.ctx.edge_scratch.emit(
            shard,
            &self.ctx.output_buffer.vertices,
            self.cell_start,
            bin,
        );
        self.cell_sub.add_edge_emit(t_post.lap());

        debug_assert_eq!(
            shard.output.cell_indices.len() as u32 - self.cell_start,
            count as u32,
            "cell index stream mismatch"
        );
    }
}

pub(super) fn process_cell<'a, 'b, 'c>(
    cell_sub: &'a mut crate::knn_clipping::timing::CellSubAccum,
    ctx: &'a mut CellContext,
    shard_ctx: &'a mut super::ShardContext<'b>,
    grid_ctx: &'a super::GridContext<'c>,
    termination: TerminationConfig,
    termination_max_k_cap: Option<usize>,
    i: usize,
    packed: Option<(
        &mut PackedKnnCellScratch,
        &mut PackedKnnTimings,
        usize,
        usize,
        usize,
    )>,
) {
    let orchestrator = CellOrchestrator::new(
        cell_sub,
        ctx,
        shard_ctx,
        grid_ctx,
        termination,
        termination_max_k_cap,
        i,
    );
    orchestrator.run(packed);
}

#[inline]
fn clip_resumable_knn_neighbors(
    builder: &mut Topo2DBuilder,
    attempted_neighbors: &mut super::AttemptedNeighbors,
    points: &[glam::Vec3],
    generator_idx: usize,
    neighbor_slots: &[u32],
    point_indices: &[u32],
    termination: TerminationConfig,
    cell_neighbors_processed: &mut usize,
    worst_cos: &mut f32,
) -> bool {
    for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
        let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
        if !attempted_neighbors.insert(neighbor_idx) {
            continue;
        }

        let neighbor = points[neighbor_idx];
        if builder
            .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
            .is_err()
        {
            break;
        }

        *cell_neighbors_processed += 1;
        let dot = points[generator_idx].dot(neighbor);
        *worst_cos = (*worst_cos).min(dot);

        if builder.is_bounded()
            && termination.should_check(*cell_neighbors_processed)
            && builder.can_terminate({
                let mut bound = *worst_cos;
                for &next_slot in neighbor_slots.iter().skip(pos + 1) {
                    let next = point_indices[next_slot as usize] as usize;
                    if next != generator_idx {
                        bound = points[generator_idx].dot(points[next]);
                        break;
                    }
                }
                bound
            })
        {
            return true;
        }
    }

    false
}
