mod failure;
mod frontier;
#[cfg(test)]
mod tests;

use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream, PackedQuery,
};
use crate::knn_clipping::topo2d::{BuilderClipOutcome, BuilderFallbackRequest, BuilderStepOutcome};
use crate::policy::{KnnPolicy, TerminationPolicy};

use super::{CellBuildError, CellFailure, CellOutputBuffer};
use failure::{classify_terminal_failure, unexpected_failure_error};
use frontier::{maybe_terminate_or_advance_frontier, probe_frontier};

use glam::Vec3;

struct AttemptedNeighbors {
    seen_stamp: Vec<u32>,
    stamp: u32,
}

impl AttemptedNeighbors {
    #[inline]
    fn new(num_points: usize) -> Self {
        Self {
            seen_stamp: vec![0; num_points],
            stamp: 1,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.seen_stamp.fill(0);
            self.stamp = 1;
        }
    }

    #[inline]
    fn insert(&mut self, id: usize) -> bool {
        debug_assert!(id < self.seen_stamp.len(), "neighbor id out of bounds");
        if self.seen_stamp[id] == self.stamp {
            return false;
        }
        self.seen_stamp[id] = self.stamp;
        true
    }

    #[inline]
    fn mark(&mut self, id: usize) {
        debug_assert!(id < self.seen_stamp.len(), "neighbor id out of bounds");
        self.seen_stamp[id] = self.stamp;
    }
}

pub(crate) struct CellBuildContext {
    builder: crate::knn_clipping::topo2d::Topo2DBuilder,
    scratch: crate::cube_grid::CubeMapGridScratch,
    packed_chunk: Vec<u32>,
    output_buffer: CellOutputBuffer,
    attempted_neighbors: AttemptedNeighbors,
}

impl CellBuildContext {
    pub(crate) fn new(grid: &crate::cube_grid::CubeMapGrid, policy: KnnPolicy) -> Self {
        Self {
            builder: crate::knn_clipping::topo2d::Topo2DBuilder::new(0, Vec3::ZERO),
            scratch: grid.make_scratch(),
            packed_chunk: Vec::with_capacity(policy.packed().scratch_chunk_capacity()),
            output_buffer: CellOutputBuffer::default(),
            attempted_neighbors: AttemptedNeighbors::new(grid.point_indices().len()),
        }
    }

    pub(crate) fn output_buffer(&self) -> &CellOutputBuffer {
        &self.output_buffer
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SeedNeighbor {
    pub(crate) neighbor_idx: usize,
    pub(crate) neighbor_slot: u32,
    pub(crate) hp_eps: f32,
}

pub(crate) struct CellBuildRequest<'a, 'm, 'p, 'g, 's> {
    pub(crate) points: &'a [Vec3],
    pub(crate) grid: &'a crate::cube_grid::CubeMapGrid,
    pub(crate) generator_idx: usize,
    pub(crate) directed_ctx: crate::cube_grid::DirectedEligibility<'m>,
    pub(crate) termination: TerminationPolicy,
    pub(crate) packed: Option<PackedQuery<'p, 'g, 'm>>,
    pub(crate) seed_neighbors: &'s [SeedNeighbor],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CellBuildStats {
    knn_query: Duration,
    clipping: Duration,
    certification: Duration,
    neighbors_processed: usize,
    incoming_seed_neighbors: usize,
    edgecheck_seed_clips: usize,
    knn_exhausted: bool,
    used_knn: bool,
    did_packed: bool,
    packed_tail_used: bool,
    packed_expand_r2_used: bool,
    packed_safe_exhausted: bool,
    knn_stage: crate::knn_clipping::timing::KnnCellStage,
}

impl CellBuildStats {
    #[inline]
    pub(crate) fn record_into(&self, cell_sub: &mut crate::knn_clipping::timing::CellSubAccum) {
        cell_sub.add_knn(self.knn_query);
        cell_sub.add_clip(self.clipping);
        cell_sub.add_cert(self.certification);

        let stage = if self.used_knn {
            self.knn_stage
        } else if self.did_packed {
            if self.packed_expand_r2_used {
                crate::knn_clipping::timing::KnnCellStage::PackedExpandR2
            } else if self.packed_tail_used {
                crate::knn_clipping::timing::KnnCellStage::PackedTail
            } else {
                crate::knn_clipping::timing::KnnCellStage::PackedChunk0
            }
        } else {
            self.knn_stage
        };

        cell_sub.add_cell_stage(
            stage,
            self.knn_exhausted,
            self.neighbors_processed,
            self.packed_tail_used,
            self.packed_expand_r2_used,
            self.packed_safe_exhausted,
            self.used_knn,
            self.incoming_seed_neighbors,
            self.edgecheck_seed_clips,
        );
    }
}

fn fallback_detail(
    builder: &crate::knn_clipping::topo2d::Topo2DBuilder,
    failure: CellFailure,
    fallback_request: Option<BuilderFallbackRequest>,
) -> Option<String> {
    fallback_request
        .or_else(|| {
            crate::knn_clipping::topo2d::Topo2DBuilder::fallback_request_for_failure(failure)
        })
        .map(|request| {
            format!(
                "fallback trigger={:?}, replay_constraints={}, replay_generator_idx={}",
                request.trigger,
                builder.accepted_constraint_count(),
                builder.generator_idx()
            )
        })
}

pub(crate) fn build_cell_into<'a, 'm, 'p, 'g, 's>(
    ctx: &'a mut CellBuildContext,
    mut request: CellBuildRequest<'a, 'm, 'p, 'g, 's>,
) -> Result<CellBuildStats, CellBuildError> {
    let points = request.points;
    let grid = request.grid;
    let generator_idx = request.generator_idx;
    let point_indices = grid.point_indices();
    let termination = request.termination;

    let mut neighbors_processed = 0usize;
    let mut terminated = false;
    let mut knn_exhausted = false;
    let mut used_knn = false;
    let mut knn_stage = crate::knn_clipping::timing::KnnCellStage::DirectedCursor;
    let mut did_packed = false;
    let mut packed_safe_exhausted = false;
    let mut packed_tail_used = false;
    let mut packed_expand_r2_used = false;
    let mut edgecheck_seed_clips = 0usize;
    let mut knn_query_time = Duration::ZERO;
    let mut clipping_time = Duration::ZERO;
    let mut certification_time = Duration::ZERO;
    let mut last_neighbor_idx: Option<usize> = None;
    let mut last_neighbor_slot: Option<u32> = None;
    let mut last_batch_source: Option<DirectedNeighborBatchSource> = None;
    let mut last_clip_phase = "none";
    let mut fallback_request = None;

    ctx.builder.reset(generator_idx, points[generator_idx]);
    ctx.attempted_neighbors.clear();
    ctx.output_buffer.clear();

    if !request.seed_neighbors.is_empty() {
        let t_clip = crate::knn_clipping::timing::Timer::start();
        for seed in request.seed_neighbors {
            last_neighbor_idx = Some(seed.neighbor_idx);
            last_neighbor_slot = Some(seed.neighbor_slot);
            last_batch_source = None;
            last_clip_phase = "edgecheck_seed";

            if !ctx.attempted_neighbors.insert(seed.neighbor_idx) {
                continue;
            }

            let neighbor = points[seed.neighbor_idx];
            match ctx.builder.clip_with_slot_edgecheck_policy(
                seed.neighbor_idx,
                seed.neighbor_slot,
                neighbor,
                seed.hp_eps,
            ) {
                Ok(BuilderStepOutcome::Applied) => {}
                Ok(BuilderStepOutcome::NeedsFallback(request)) => {
                    fallback_request = Some(request);
                    ctx.builder.enter_fallback(points, request);
                }
                Err(_) => break,
            }
            neighbors_processed += 1;
        }
        clipping_time += t_clip.elapsed();
        edgecheck_seed_clips = neighbors_processed;
    }

    let mut stream = DirectedNeighborStream::new(
        grid,
        points,
        generator_idx,
        &mut ctx.scratch,
        request.directed_ctx,
        request.packed.take(),
    );

    while !terminated && !ctx.builder.is_failed() {
        let frontier = probe_frontier(
            &mut stream,
            &mut ctx.packed_chunk,
            &mut used_knn,
            &mut knn_stage,
            &mut knn_query_time,
        );

        match frontier {
            DirectedNeighborFrontier::ExactBatch(batch) => {
                if batch.source == DirectedNeighborBatchSource::PackedExpandR2 {
                    knn_stage = crate::knn_clipping::timing::KnnCellStage::PackedExpandR2;
                }

                let t_clip = crate::knn_clipping::timing::Timer::start();
                for pos in 0..batch.n {
                    let neighbor_slot = ctx.packed_chunk[pos];
                    let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                    if neighbor_idx == generator_idx {
                        continue;
                    }

                    let should_clip = match batch.source {
                        DirectedNeighborBatchSource::DirectedCursor => {
                            ctx.attempted_neighbors.insert(neighbor_idx)
                        }
                        DirectedNeighborBatchSource::PackedChunk0
                        | DirectedNeighborBatchSource::PackedTail
                        | DirectedNeighborBatchSource::PackedExpandR2 => {
                            ctx.attempted_neighbors.mark(neighbor_idx);
                            true
                        }
                    };
                    if !should_clip {
                        continue;
                    }

                    last_neighbor_idx = Some(neighbor_idx);
                    last_neighbor_slot = Some(neighbor_slot);
                    last_batch_source = Some(batch.source);
                    last_clip_phase = "stream";

                    let neighbor = points[neighbor_idx];
                    let clip_result = match ctx.builder.clip_with_slot_result_policy(
                        neighbor_idx,
                        neighbor_slot,
                        neighbor,
                    ) {
                        Ok(BuilderClipOutcome::Applied(result)) => result,
                        Ok(BuilderClipOutcome::NeedsFallback(request)) => {
                            fallback_request = Some(request);
                            ctx.builder.enter_fallback(points, request);
                            crate::knn_clipping::topo2d::types::ClipResult::Changed
                        }
                        Err(_) => break,
                    };

                    neighbors_processed += 1;

                    let should_check_termination = match batch.source {
                        DirectedNeighborBatchSource::DirectedCursor => {
                            termination.should_check(neighbors_processed)
                        }
                        DirectedNeighborBatchSource::PackedChunk0
                        | DirectedNeighborBatchSource::PackedTail
                        | DirectedNeighborBatchSource::PackedExpandR2 => {
                            clip_result == crate::knn_clipping::topo2d::types::ClipResult::Unchanged
                        }
                    };

                    if ctx.builder.is_bounded() && should_check_termination {
                        let bound = if pos + 1 < batch.n {
                            let next_slot = ctx.packed_chunk[pos + 1];
                            let next = point_indices[next_slot as usize] as usize;
                            points[generator_idx].dot(points[next])
                        } else {
                            batch.unseen_bound
                        };
                        if ctx.builder.can_terminate(bound) {
                            terminated = true;
                            break;
                        }
                    }
                }
                clipping_time += t_clip.elapsed();
                stream.advance_frontier();

                if !terminated && !ctx.builder.is_failed() && ctx.builder.is_bounded() {
                    terminated = maybe_terminate_or_advance_frontier(
                        &mut stream,
                        &mut ctx.packed_chunk,
                        &mut ctx.builder,
                        termination,
                        neighbors_processed,
                        &mut used_knn,
                        &mut knn_stage,
                        &mut knn_query_time,
                    );
                }
            }
            DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                let can_check =
                    !stream.is_cursor_stage() || termination.should_check(neighbors_processed);
                if ctx.builder.is_bounded()
                    && can_check
                    && ctx.builder.can_terminate(dot_upper_bound)
                {
                    terminated = true;
                } else {
                    stream.advance_frontier();
                }
            }
            DirectedNeighborFrontier::Exhausted => break,
        }
    }

    did_packed |= stream.did_packed();
    packed_tail_used |= stream.packed_tail_used();
    packed_expand_r2_used |= stream.packed_expand_r2_used();
    packed_safe_exhausted |= stream.packed_safe_exhausted();
    knn_exhausted |= stream.knn_exhausted();

    if !ctx.builder.is_bounded() || ctx.builder.is_failed() {
        if let Some(failure) = classify_terminal_failure(
            ctx.builder.is_bounded(),
            ctx.builder.failure(),
            knn_exhausted,
        ) {
            return Err(CellBuildError {
                generator_idx,
                failure,
                detail: fallback_detail(&ctx.builder, failure, fallback_request),
            });
        }
        return Err(unexpected_failure_error(
            ctx,
            points,
            generator_idx,
            neighbors_processed,
            did_packed,
            used_knn,
            knn_exhausted,
            last_clip_phase,
            last_batch_source,
            last_neighbor_idx,
            last_neighbor_slot,
            "validation",
            ctx.builder.failure(),
        ));
    }

    let t_cert = crate::knn_clipping::timing::Timer::start();
    if let Err(failure) = ctx.builder.to_vertex_data_full(&mut ctx.output_buffer) {
        return Err(unexpected_failure_error(
            ctx,
            points,
            generator_idx,
            neighbors_processed,
            did_packed,
            used_knn,
            knn_exhausted,
            last_clip_phase,
            last_batch_source,
            last_neighbor_idx,
            last_neighbor_slot,
            "vertex extraction",
            Some(failure),
        ));
    }
    certification_time += t_cert.elapsed();

    Ok(CellBuildStats {
        knn_query: knn_query_time,
        clipping: clipping_time,
        certification: certification_time,
        neighbors_processed,
        incoming_seed_neighbors: request.seed_neighbors.len(),
        edgecheck_seed_clips,
        knn_exhausted,
        used_knn,
        did_packed,
        packed_tail_used,
        packed_expand_r2_used,
        packed_safe_exhausted,
        knn_stage,
    })
}
