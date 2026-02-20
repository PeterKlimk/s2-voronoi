//! `process_cell` implementation for live-dedup cell building.

use crate::cube_grid::packed_knn::{PackedKnnCellScratch, PackedKnnTimings, PackedStage};

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
    i: usize,

    cell_neighbors_processed: usize,
    terminated: bool,
    knn_exhausted: bool,
    used_knn: bool,

    worst_cos: f32,
    knn_stage: crate::knn_clipping::timing::KnnCellStage,

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
            i,
            cell_neighbors_processed: 0,
            terminated: false,
            knn_exhausted: false,
            used_knn: false,
            worst_cos: 1.0,
            knn_stage: crate::knn_clipping::timing::KnnCellStage::DirectedCursor,
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
        self.phase_4_directed_cursor_fallback();
        self.phase_5_validate_success();
        self.phase_6_extract_output();
    }

    fn phase_1_init(&mut self) {
        let points = self.grid_ctx.points;
        self.ctx.builder.reset(self.i, points[self.i]);
        self.ctx.attempted_neighbors.clear();

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

    fn phase_4_directed_cursor_fallback(&mut self) {
        if self.terminated || self.ctx.builder.is_failed() {
            return;
        }

        if self.ctx.builder.is_bounded() {
            let bound = if self.did_packed && self.packed_safe_exhausted {
                self.packed_security
            } else {
                self.worst_cos
            };
            if self.ctx.builder.can_terminate(bound) {
                self.terminated = true;
                return;
            }
        }

        self.used_knn = true;
        self.knn_stage = crate::knn_clipping::timing::KnnCellStage::DirectedCursor;

        let grid = self.grid_ctx.grid;
        let points = self.grid_ctx.points;
        let i = self.i;
        let point_indices = grid.point_indices();
        let termination = self.termination;
        let directed_ctx = self.directed_ctx;
        let (builder, attempted_neighbors, scratch) = {
            let ctx = &mut self.ctx;
            (
                &mut ctx.builder,
                &mut ctx.attempted_neighbors,
                &mut ctx.scratch,
            )
        };

        let mut cursor = grid.directed_no_k_cursor(points[i], i, scratch, directed_ctx);
        while !self.terminated && !builder.is_failed() {
            let t_knn = crate::knn_clipping::timing::Timer::start();
            let Some(neighbor_slot) = cursor.pop_next_proven_slot() else {
                self.knn_exhausted = cursor.is_exhausted();
                break;
            };
            self.cell_sub.add_knn(t_knn.elapsed());

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
            if attempted_neighbors.insert(neighbor_idx) {
                let neighbor = points[neighbor_idx];
                if builder
                    .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
                    .is_err()
                {
                    self.cell_sub.add_clip(t_clip.elapsed());
                    break;
                }

                self.cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                self.worst_cos = self.worst_cos.min(dot);

                if builder.is_bounded()
                    && termination.should_check(self.cell_neighbors_processed)
                    && builder.can_terminate(cursor.remaining_dot_upper_bound())
                {
                    self.terminated = true;
                }
            }
            self.cell_sub.add_clip(t_clip.elapsed());
        }

        self.knn_exhausted |= cursor.is_exhausted();
        if !self.terminated && !builder.is_failed() && builder.is_bounded() {
            if builder.can_terminate(cursor.remaining_dot_upper_bound()) {
                self.terminated = true;
            }
        }
    }

    fn phase_5_validate_success(&mut self) {
        if !self.ctx.builder.is_bounded() || self.ctx.builder.is_failed() {
            let (active, total) = self.ctx.builder.count_active_planes();
            let gen = self.grid_ctx.points[self.i];
            let neighbor_indices: Vec<usize> = self.ctx.builder.neighbor_indices_iter().collect();

            panic!(
                "Cell {} construction failed: bounded={}, failure={:?}, \
                 planes={}, active={}, vertices={}, \
                 did_packed={}, did_cursor_fallback={}, knn_exhausted={}\n\
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
                self.knn_exhausted,
                gen,
                &neighbor_indices[..neighbor_indices.len().min(10)],
            );
        }
    }

    fn phase_6_extract_output(&mut self) {
        let stage = if self.used_knn {
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
    i: usize,
    packed: Option<(
        &mut PackedKnnCellScratch,
        &mut PackedKnnTimings,
        usize,
        usize,
        usize,
    )>,
) {
    let orchestrator = CellOrchestrator::new(cell_sub, ctx, shard_ctx, grid_ctx, termination, i);
    orchestrator.run(packed);
}
