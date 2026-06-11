mod failure;
mod frontier;
#[cfg(test)]
mod tests;

use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream, PackedQuery,
};
use crate::knn_clipping::topo2d::{BuilderClipOutcome, BuilderFallbackRequest, BuilderStepOutcome};
use crate::policy::PackedNeighborPolicy;

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
    #[cfg(test)]
    force_fallback_after_neighbors_processed: Option<usize>,
}

impl CellBuildContext {
    pub(crate) fn new(grid: &crate::cube_grid::CubeMapGrid, policy: PackedNeighborPolicy) -> Self {
        Self {
            builder: crate::knn_clipping::topo2d::Topo2DBuilder::new(0, Vec3::ZERO),
            scratch: grid.make_scratch(),
            packed_chunk: Vec::with_capacity(policy.scratch_chunk_capacity()),
            output_buffer: CellOutputBuffer::default(),
            attempted_neighbors: AttemptedNeighbors::new(grid.point_indices().len()),
            #[cfg(test)]
            force_fallback_after_neighbors_processed: None,
        }
    }

    pub(crate) fn output_buffer(&self) -> &CellOutputBuffer {
        &self.output_buffer
    }
}

#[cfg(test)]
fn maybe_force_fallback(
    builder: &mut crate::knn_clipping::topo2d::Topo2DBuilder,
    force_fallback_after_neighbors_processed: &mut Option<usize>,
    points: &[Vec3],
    neighbors_processed: usize,
    fallback_request: &mut Option<BuilderFallbackRequest>,
) {
    if builder.is_fallback() {
        return;
    }
    let Some(target) = *force_fallback_after_neighbors_processed else {
        return;
    };
    if neighbors_processed < target {
        return;
    }

    let request = BuilderFallbackRequest {
        trigger: crate::knn_clipping::topo2d::builder::BuilderFallbackTrigger::ProjectionLimit,
    };
    *fallback_request = Some(request);
    builder.enter_fallback(points, request);
    *force_fallback_after_neighbors_processed = None;
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

/// Diagnostic trail of the most recent clip, for unexpected-failure reports.
pub(super) struct BuildTrace {
    pub(super) last_neighbor_idx: Option<usize>,
    pub(super) last_neighbor_slot: Option<u32>,
    pub(super) last_batch_source: Option<DirectedNeighborBatchSource>,
    pub(super) last_clip_phase: &'static str,
    fallback_request: Option<BuilderFallbackRequest>,
}

impl BuildTrace {
    fn new() -> Self {
        Self {
            last_neighbor_idx: None,
            last_neighbor_slot: None,
            last_batch_source: None,
            last_clip_phase: "none",
            fallback_request: None,
        }
    }
}

/// Counters and timings accumulated across the build phases.
pub(super) struct BuildCounters {
    pub(super) neighbors_processed: usize,
    edgecheck_seed_clips: usize,
    knn_query_time: Duration,
    clipping_time: Duration,
    certification_time: Duration,
    pub(super) used_knn: bool,
    knn_stage: crate::knn_clipping::timing::KnnCellStage,
    pub(super) knn_exhausted: bool,
    pub(super) did_packed: bool,
    packed_tail_used: bool,
    packed_expand_r2_used: bool,
    packed_safe_exhausted: bool,
    terminated: bool,
}

impl BuildCounters {
    fn new() -> Self {
        Self {
            neighbors_processed: 0,
            edgecheck_seed_clips: 0,
            knn_query_time: Duration::ZERO,
            clipping_time: Duration::ZERO,
            certification_time: Duration::ZERO,
            used_knn: false,
            knn_stage: crate::knn_clipping::timing::KnnCellStage::ShellExpand,
            knn_exhausted: false,
            did_packed: false,
            packed_tail_used: false,
            packed_expand_r2_used: false,
            packed_safe_exhausted: false,
            terminated: false,
        }
    }

    fn absorb_stream(&mut self, stream: &DirectedNeighborStream<'_, '_, '_, '_>) {
        self.did_packed |= stream.did_packed();
        self.packed_tail_used |= stream.packed_tail_used();
        self.packed_expand_r2_used |= stream.packed_expand_r2_used();
        self.packed_safe_exhausted |= stream.packed_safe_exhausted();
        self.knn_exhausted |= stream.knn_exhausted();
    }
}

/// The disjoint `CellBuildContext` borrows the stream-consumption phase needs
/// (the stream itself holds the context's scratch for its whole life, so the
/// remaining fields are threaded explicitly).
struct StreamPhase<'x> {
    builder: &'x mut crate::knn_clipping::topo2d::Topo2DBuilder,
    packed_chunk: &'x mut Vec<u32>,
    attempted_neighbors: &'x mut AttemptedNeighbors,
    #[cfg(test)]
    force_fallback_after_neighbors_processed: &'x mut Option<usize>,
}

/// Phase 1: clip edge-check seed constraints forwarded by earlier same-bin
/// cells (see "The stitching invariant" in docs/live_dedup.md).
fn clip_seed_neighbors(
    ctx: &mut CellBuildContext,
    points: &[Vec3],
    seed_neighbors: &[SeedNeighbor],
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    if seed_neighbors.is_empty() {
        return;
    }
    let t_clip = crate::knn_clipping::timing::Timer::start();
    for seed in seed_neighbors {
        trace.last_neighbor_idx = Some(seed.neighbor_idx);
        trace.last_neighbor_slot = Some(seed.neighbor_slot);
        trace.last_batch_source = None;
        trace.last_clip_phase = "edgecheck_seed";

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
                trace.fallback_request = Some(request);
                ctx.builder.enter_fallback(points, request);
            }
            Err(_) => break,
        }
        counters.neighbors_processed += 1;
        #[cfg(test)]
        maybe_force_fallback(
            &mut ctx.builder,
            &mut ctx.force_fallback_after_neighbors_processed,
            points,
            counters.neighbors_processed,
            &mut trace.fallback_request,
        );
    }
    counters.clipping_time += t_clip.elapsed();
    counters.edgecheck_seed_clips = counters.neighbors_processed;
}

/// Clip one exact batch; returns with `counters.terminated` set when the
/// builder's certificate fires mid-batch.
fn clip_batch(
    phase: &mut StreamPhase<'_>,
    batch: crate::cube_grid::DirectedNeighborBatch,
    points: &[Vec3],
    point_indices: &[u32],
    generator_idx: usize,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    let t_clip = crate::knn_clipping::timing::Timer::start();
    for pos in 0..batch.n {
        let neighbor_slot = phase.packed_chunk[pos];
        let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
        if neighbor_idx == generator_idx {
            continue;
        }

        let should_clip = match batch.source {
            // The takeover re-covers packed-served points; dedup on insertion.
            DirectedNeighborBatchSource::ShellExpand => {
                phase.attempted_neighbors.insert(neighbor_idx)
            }
            DirectedNeighborBatchSource::PackedChunk0
            | DirectedNeighborBatchSource::PackedTail
            | DirectedNeighborBatchSource::PackedExpandR2 => {
                phase.attempted_neighbors.mark(neighbor_idx);
                true
            }
        };
        if !should_clip {
            continue;
        }

        trace.last_neighbor_idx = Some(neighbor_idx);
        trace.last_neighbor_slot = Some(neighbor_slot);
        trace.last_batch_source = Some(batch.source);
        trace.last_clip_phase = "stream";

        let neighbor = points[neighbor_idx];
        let clip_result =
            match phase
                .builder
                .clip_with_slot_result_policy(neighbor_idx, neighbor_slot, neighbor)
            {
                Ok(BuilderClipOutcome::Applied(result)) => result,
                Ok(BuilderClipOutcome::NeedsFallback(request)) => {
                    trace.fallback_request = Some(request);
                    phase.builder.enter_fallback(points, request);
                    crate::knn_clipping::topo2d::types::ClipResult::Changed
                }
                Err(_) => break,
            };

        counters.neighbors_processed += 1;
        #[cfg(test)]
        maybe_force_fallback(
            phase.builder,
            phase.force_fallback_after_neighbors_processed,
            points,
            counters.neighbors_processed,
            &mut trace.fallback_request,
        );

        // All batch sources are sorted, so mid-batch bounds are sound; only
        // re-check when a clip left the polygon unchanged.
        let should_check_termination =
            clip_result == crate::knn_clipping::topo2d::types::ClipResult::Unchanged;

        if phase.builder.is_bounded() && should_check_termination {
            let bound = if pos + 1 < batch.n {
                let next_slot = phase.packed_chunk[pos + 1];
                let next = point_indices[next_slot as usize] as usize;
                let next_dot = points[generator_idx].dot(points[next]);
                // Shell layers are sorted within the layer, but the next
                // layer can contain closer points than this layer's tail;
                // the mid-batch bound must also cover them. (Packed batches
                // dominate their unseen set, so next_dot alone is sound for
                // those sources.)
                if batch.source == DirectedNeighborBatchSource::ShellExpand {
                    next_dot.max(batch.unseen_bound)
                } else {
                    next_dot
                }
            } else {
                batch.unseen_bound
            };
            if phase.builder.can_terminate(bound) {
                counters.terminated = true;
                break;
            }
        }
    }
    counters.clipping_time += t_clip.elapsed();
}

/// Phase 2: drive the neighbor stream to termination, failure, or exhaustion.
fn consume_stream(
    stream: &mut DirectedNeighborStream<'_, '_, '_, '_>,
    mut phase: StreamPhase<'_>,
    points: &[Vec3],
    point_indices: &[u32],
    generator_idx: usize,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    while !counters.terminated && !phase.builder.is_failed() {
        let frontier = probe_frontier(
            stream,
            phase.packed_chunk,
            &mut counters.used_knn,
            &mut counters.knn_stage,
            &mut counters.knn_query_time,
        );

        match frontier {
            DirectedNeighborFrontier::ExactBatch(batch) => {
                if batch.source == DirectedNeighborBatchSource::PackedExpandR2 {
                    counters.knn_stage = crate::knn_clipping::timing::KnnCellStage::PackedExpandR2;
                }

                clip_batch(
                    &mut phase,
                    batch,
                    points,
                    point_indices,
                    generator_idx,
                    trace,
                    counters,
                );
                stream.advance_frontier();

                if !counters.terminated && !phase.builder.is_failed() && phase.builder.is_bounded()
                {
                    counters.terminated = maybe_terminate_or_advance_frontier(
                        stream,
                        phase.packed_chunk,
                        phase.builder,
                        &mut counters.used_knn,
                        &mut counters.knn_stage,
                        &mut counters.knn_query_time,
                    );
                }
            }
            DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                // Only packed stages produce bounded-unknown frontiers; the
                // takeover always emits exact layers.
                if phase.builder.is_bounded() && phase.builder.can_terminate(dot_upper_bound) {
                    counters.terminated = true;
                } else {
                    stream.advance_frontier();
                }
            }
            DirectedNeighborFrontier::Exhausted => break,
        }
    }
}

/// Phase 3: classify terminal failure, or extract the finished cell.
fn finish_cell(
    ctx: &mut CellBuildContext,
    points: &[Vec3],
    generator_idx: usize,
    trace: &BuildTrace,
    counters: &mut BuildCounters,
) -> Result<(), CellBuildError> {
    if !ctx.builder.is_bounded() || ctx.builder.is_failed() {
        if let Some(failure) = classify_terminal_failure(
            ctx.builder.is_bounded(),
            ctx.builder.failure(),
            counters.knn_exhausted,
        ) {
            return Err(CellBuildError {
                generator_idx,
                failure,
                detail: fallback_detail(&ctx.builder, failure, trace.fallback_request),
            });
        }
        return Err(unexpected_failure_error(
            ctx,
            points,
            generator_idx,
            trace,
            counters,
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
            trace,
            counters,
            "vertex extraction",
            Some(failure),
        ));
    }
    counters.certification_time += t_cert.elapsed();
    Ok(())
}

pub(crate) fn build_cell_into<'a, 'm, 'p, 'g, 's>(
    ctx: &'a mut CellBuildContext,
    mut request: CellBuildRequest<'a, 'm, 'p, 'g, 's>,
) -> Result<CellBuildStats, CellBuildError> {
    let points = request.points;
    let grid = request.grid;
    let generator_idx = request.generator_idx;
    let point_indices = grid.point_indices();

    let mut trace = BuildTrace::new();
    let mut counters = BuildCounters::new();

    ctx.builder.reset(generator_idx, points[generator_idx]);
    ctx.attempted_neighbors.clear();
    ctx.output_buffer.clear();

    clip_seed_neighbors(
        ctx,
        points,
        request.seed_neighbors,
        &mut trace,
        &mut counters,
    );

    let mut stream = DirectedNeighborStream::new(
        grid,
        points,
        generator_idx,
        &mut ctx.scratch,
        request.directed_ctx,
        request.packed.take(),
    );
    consume_stream(
        &mut stream,
        StreamPhase {
            builder: &mut ctx.builder,
            packed_chunk: &mut ctx.packed_chunk,
            attempted_neighbors: &mut ctx.attempted_neighbors,
            #[cfg(test)]
            force_fallback_after_neighbors_processed: &mut ctx
                .force_fallback_after_neighbors_processed,
        },
        points,
        point_indices,
        generator_idx,
        &mut trace,
        &mut counters,
    );
    counters.absorb_stream(&stream);
    drop(stream);

    finish_cell(ctx, points, generator_idx, &trace, &mut counters)?;

    Ok(CellBuildStats {
        knn_query: counters.knn_query_time,
        clipping: counters.clipping_time,
        certification: counters.certification_time,
        neighbors_processed: counters.neighbors_processed,
        incoming_seed_neighbors: request.seed_neighbors.len(),
        edgecheck_seed_clips: counters.edgecheck_seed_clips,
        knn_exhausted: counters.knn_exhausted,
        used_knn: counters.used_knn,
        did_packed: counters.did_packed,
        packed_tail_used: counters.packed_tail_used,
        packed_expand_r2_used: counters.packed_expand_r2_used,
        packed_safe_exhausted: counters.packed_safe_exhausted,
        knn_stage: counters.knn_stage,
    })
}
