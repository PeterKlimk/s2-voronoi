//! `process_cell` implementation for live-dedup cell building.

use crate::cube_grid::{DirectedNeighborBatchSource, DirectedNeighborStream, PackedQuery};

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
            packed_safe_exhausted: false,
            packed_tail_used: false,
            directed_ctx,
            incoming_checks: Vec::new(),
            incoming_edgechecks: 0,
            edgecheck_seed_clips: 0,
            cell_start: 0,
        }
    }

    fn run(mut self, packed: Option<PackedQuery<'_, 'c>>) {
        self.phase_1_init();
        self.phase_2_edgecheck_seeds();
        self.phase_3_neighbor_stream(packed);
        self.phase_4_validate_success();
        self.phase_5_extract_output();
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

    fn phase_3_neighbor_stream<'p>(&mut self, packed: Option<PackedQuery<'p, 'c>>) {
        let grid = self.grid_ctx.grid;
        let points = self.grid_ctx.points;
        let i = self.i;
        let point_indices = grid.point_indices();
        let termination = self.termination;
        let directed_ctx = self.directed_ctx;
        let (builder, attempted_neighbors, scratch, batch_buffer) = {
            let ctx = &mut self.ctx;
            (
                &mut ctx.builder,
                &mut ctx.attempted_neighbors,
                &mut ctx.scratch,
                &mut ctx.packed_chunk,
            )
        };

        let mut stream =
            DirectedNeighborStream::new(grid, points, i, scratch, directed_ctx, packed);

        while !self.terminated && !builder.is_failed() {
            let t_knn = crate::knn_clipping::timing::Timer::start();
            let Some(batch) = stream.next_batch(batch_buffer) else {
                break;
            };

            match batch.source {
                DirectedNeighborBatchSource::DirectedCursor => {
                    self.used_knn = true;
                    self.knn_stage = crate::knn_clipping::timing::KnnCellStage::DirectedCursor;
                    self.cell_sub.add_knn(t_knn.elapsed());
                }
                DirectedNeighborBatchSource::PackedExhausted => {
                    if builder.is_bounded() && builder.can_terminate(batch.unseen_bound) {
                        self.terminated = true;
                    }
                    continue;
                }
                DirectedNeighborBatchSource::PackedChunk0
                | DirectedNeighborBatchSource::PackedTail => {}
            }

            let t_clip = crate::knn_clipping::timing::Timer::start();
            for pos in 0..batch.n {
                let neighbor_slot = batch_buffer[pos];
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if neighbor_idx == i {
                    continue;
                }

                let should_clip = match batch.source {
                    DirectedNeighborBatchSource::DirectedCursor => {
                        attempted_neighbors.insert(neighbor_idx)
                    }
                    DirectedNeighborBatchSource::PackedChunk0
                    | DirectedNeighborBatchSource::PackedTail => {
                        attempted_neighbors.mark(neighbor_idx);
                        true
                    }
                    DirectedNeighborBatchSource::PackedExhausted => unreachable!(),
                };
                if !should_clip {
                    continue;
                }

                let neighbor = points[neighbor_idx];
                let clip_result =
                    match builder.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor) {
                        Ok(result) => result,
                        Err(_) => break,
                    };

                self.cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                self.worst_cos = self.worst_cos.min(dot);

                match batch.source {
                    DirectedNeighborBatchSource::DirectedCursor => {
                        if builder.is_bounded()
                            && termination.should_check(self.cell_neighbors_processed)
                            && builder.can_terminate(batch.unseen_bound)
                        {
                            self.terminated = true;
                            break;
                        }
                    }
                    DirectedNeighborBatchSource::PackedChunk0
                    | DirectedNeighborBatchSource::PackedTail => {
                        if builder.is_bounded()
                            && clip_result
                                == crate::knn_clipping::topo2d::types::ClipResult::Unchanged
                        {
                            let bound = if pos + 1 < batch.n {
                                let next_slot = batch_buffer[pos + 1];
                                let next = point_indices[next_slot as usize] as usize;
                                points[i].dot(points[next])
                            } else {
                                batch.unseen_bound
                            };
                            if builder.can_terminate(bound) {
                                self.terminated = true;
                                break;
                            }
                        }
                    }
                    DirectedNeighborBatchSource::PackedExhausted => unreachable!(),
                }
            }
            self.cell_sub.add_clip(t_clip.elapsed());
        }

        self.did_packed |= stream.did_packed();
        self.packed_tail_used |= stream.packed_tail_used();
        self.packed_safe_exhausted |= stream.packed_safe_exhausted();
        self.knn_exhausted |= stream.knn_exhausted();
    }

    fn phase_4_validate_success(&mut self) {
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

    fn phase_5_extract_output(&mut self) {
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
    packed: Option<PackedQuery<'_, 'c>>,
) {
    let orchestrator = CellOrchestrator::new(cell_sub, ctx, shard_ctx, grid_ctx, termination, i);
    orchestrator.run(packed);
}
