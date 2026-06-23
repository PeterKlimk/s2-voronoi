mod failure;
mod frontier;
#[cfg(test)]
mod tests;

use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream, PackedQuery,
};
use crate::knn_clipping::topo2d::types::MAX_POLY_VERTICES;
use crate::knn_clipping::topo2d::{
    BuilderClipOutcome, BuilderFallbackRequest, BuilderFallbackTrigger, BuilderStepOutcome,
};
use crate::policy::PackedNeighborPolicy;

use super::{CellBuildError, CellFailure, CellOutputBuffer};
use failure::{classify_terminal_failure, unexpected_failure_error};
use frontier::{maybe_terminate_or_advance_frontier, probe_frontier};

use glam::Vec3;

/// Per-cell "already attempted this neighbor" set, stamp-based to avoid an
/// O(n) clear per cell.
///
/// Keyed by the neighbor's **SOA slot**, not its global point index. Both
/// uniquely identify a point (slot↔index is a permutation), so dedup semantics
/// are identical — but slot order is the grid's spatial order, so a cell's
/// neighbors (and successive cells) touch a clustered region of `seen_stamp`,
/// which is far more cache-friendly than the spatially-scattered global-index
/// order. (`seen_stamp` is num_points entries; slots are in `[0, num_points)`.)
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
    fn insert(&mut self, slot: usize) -> bool {
        debug_assert!(slot < self.seen_stamp.len(), "neighbor slot out of bounds");
        if self.seen_stamp[slot] == self.stamp {
            return false;
        }
        self.seen_stamp[slot] = self.stamp;
        true
    }

    #[inline]
    fn mark(&mut self, slot: usize) {
        debug_assert!(slot < self.seen_stamp.len(), "neighbor slot out of bounds");
        self.seen_stamp[slot] = self.stamp;
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
            output_buffer: CellOutputBuffer::with_capacity(MAX_POLY_VERTICES),
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
    final_edges: usize,
    directional_shadow_checks: usize,
    directional_shadow_candidate_tests: usize,
    directional_shadow_hits: usize,
    directional_shadow_saved: usize,
    directional_support_candidate_tests: usize,
    directional_support_hits: usize,
    directional_support_saved: usize,
    directional_support_false_positive_hits: usize,
    fallback_projection: usize,
    fallback_polygon_cap: usize,
    fallback_all_constraints: usize,
    incoming_seed_neighbors: usize,
    edgecheck_seed_clips: usize,
    knn_exhausted: bool,
    used_knn: bool,
    did_packed: bool,
    packed_tail_used: bool,
    packed_safe_exhausted: bool,
    knn_stage: crate::knn_clipping::timing::KnnCellStage,
}

impl CellBuildStats {
    #[inline]
    pub(crate) fn record_into(&self, cell_sub: &mut crate::knn_clipping::timing::CellSubAccum) {
        cell_sub.add_knn(self.knn_query);
        cell_sub.add_clip(self.clipping);
        cell_sub.add_cert(self.certification);
        cell_sub.add_directional_shadow(
            self.directional_shadow_checks,
            self.directional_shadow_candidate_tests,
            self.directional_shadow_hits,
            self.directional_shadow_saved,
            self.directional_support_candidate_tests,
            self.directional_support_hits,
            self.directional_support_saved,
            self.directional_support_false_positive_hits,
        );
        cell_sub.add_fallbacks(
            self.fallback_projection,
            self.fallback_polygon_cap,
            self.fallback_all_constraints,
        );

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

        cell_sub.add_cell_stage(
            stage,
            self.knn_exhausted,
            self.neighbors_processed,
            self.final_edges,
            self.packed_tail_used,
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
    packed_safe_exhausted: bool,
    directional_shadow_checks: usize,
    directional_shadow_candidate_tests: usize,
    directional_shadow_hits: usize,
    directional_shadow_saved: usize,
    directional_support_candidate_tests: usize,
    directional_support_hits: usize,
    directional_support_saved: usize,
    directional_support_false_positive_hits: usize,
    fallback_projection: usize,
    fallback_polygon_cap: usize,
    fallback_all_constraints: usize,
    #[cfg(feature = "timing")]
    directional_shadow_terminated: bool,
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
            packed_safe_exhausted: false,
            directional_shadow_checks: 0,
            directional_shadow_candidate_tests: 0,
            directional_shadow_hits: 0,
            directional_shadow_saved: 0,
            directional_support_candidate_tests: 0,
            directional_support_hits: 0,
            directional_support_saved: 0,
            directional_support_false_positive_hits: 0,
            fallback_projection: 0,
            fallback_polygon_cap: 0,
            fallback_all_constraints: 0,
            #[cfg(feature = "timing")]
            directional_shadow_terminated: false,
            terminated: false,
        }
    }

    fn absorb_stream(&mut self, stream: &DirectedNeighborStream<'_, '_, '_, '_>) {
        self.did_packed |= stream.did_packed();
        self.packed_tail_used |= stream.packed_tail_used();
        self.packed_safe_exhausted |= stream.packed_safe_exhausted();
        self.knn_exhausted |= stream.knn_exhausted();
    }

    fn record_fallback(&mut self, request: BuilderFallbackRequest) {
        match request.trigger {
            BuilderFallbackTrigger::ProjectionLimit => self.fallback_projection += 1,
            BuilderFallbackTrigger::PolygonVertexLimit => self.fallback_polygon_cap += 1,
        }
    }
}

#[cfg(feature = "timing")]
fn audit_directional_batch_skip(
    builder: &mut crate::knn_clipping::topo2d::Topo2DBuilder,
    remaining_slots: &[u32],
    unseen_bound_after_batch: f32,
    pos_slots: &[crate::cube_grid::SlotPoint],
    counters: &mut BuildCounters,
) {
    if counters.directional_shadow_terminated || remaining_slots.is_empty() || !builder.is_bounded()
    {
        return;
    }
    counters.directional_shadow_checks += 1;

    // Conservative lower-bound prototype: if the existing scalar certificate
    // would pass after this exact batch, and every remaining known candidate in
    // the batch is all-inside against the current polygon, a direction-aware
    // known-batch certificate could skip those candidates and terminate here.
    if !builder.can_terminate(unseen_bound_after_batch) {
        return;
    }

    let mut support_all_unchanged = true;
    for &slot in remaining_slots {
        counters.directional_support_candidate_tests += 1;
        let neighbor = pos_slots[slot as usize].pos;
        if !builder.candidate_would_be_unchanged_support(neighbor) {
            support_all_unchanged = false;
            break;
        }
    }

    for &slot in remaining_slots {
        counters.directional_shadow_candidate_tests += 1;
        let neighbor = pos_slots[slot as usize].pos;
        if !builder.candidate_would_be_unchanged(neighbor) {
            if support_all_unchanged {
                counters.directional_support_false_positive_hits += 1;
            }
            return;
        }
    }

    counters.directional_shadow_hits += 1;
    counters.directional_shadow_saved += remaining_slots.len();
    if support_all_unchanged {
        counters.directional_support_hits += 1;
        counters.directional_support_saved += remaining_slots.len();
    }
    counters.directional_shadow_terminated = true;
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
/// cells (see "The stitching invariant" in docs/architecture.md).
fn clip_seed_neighbors(
    ctx: &mut CellBuildContext,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
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

        if !ctx.attempted_neighbors.insert(seed.neighbor_slot as usize) {
            continue;
        }

        let neighbor = pos_slots[seed.neighbor_slot as usize].pos;
        match ctx.builder.clip_with_slot_edgecheck_policy(
            seed.neighbor_idx,
            seed.neighbor_slot,
            neighbor,
            seed.hp_eps,
        ) {
            Ok(BuilderStepOutcome::Applied) => {}
            Ok(BuilderStepOutcome::NeedsFallback(request)) => {
                trace.fallback_request = Some(request);
                counters.record_fallback(request);
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
#[allow(clippy::too_many_arguments)]
fn clip_batch(
    phase: &mut StreamPhase<'_>,
    batch: crate::cube_grid::DirectedNeighborBatch,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
    generator_idx: usize,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    let t_clip = crate::knn_clipping::timing::Timer::start();
    let packed_chunk = &phase.packed_chunk[..batch.n];
    for pos in 0..batch.n {
        let neighbor_slot = packed_chunk[pos];
        // One fused load gets both the global index and the position (one cache
        // line) instead of two separate random by-slot loads.
        let slot_point = pos_slots[neighbor_slot as usize];
        let neighbor_idx = slot_point.idx as usize;
        if neighbor_idx == generator_idx {
            continue;
        }

        let should_clip = match batch.source {
            // The takeover re-covers packed-served points; dedup on insertion.
            DirectedNeighborBatchSource::ShellExpand => {
                phase.attempted_neighbors.insert(neighbor_slot as usize)
            }
            DirectedNeighborBatchSource::PackedChunk0 | DirectedNeighborBatchSource::PackedTail => {
                phase.attempted_neighbors.mark(neighbor_slot as usize);
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

        // Position from the fused record loaded above (spatial order → clustered,
        // cache-friendly); bit-identical to points[neighbor_idx].
        let neighbor = slot_point.pos;
        let clip_result =
            match phase
                .builder
                .clip_with_slot_result_policy(neighbor_idx, neighbor_slot, neighbor)
            {
                Ok(BuilderClipOutcome::Applied(result)) => result,
                Ok(BuilderClipOutcome::NeedsFallback(request)) => {
                    trace.fallback_request = Some(request);
                    counters.record_fallback(request);
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
                let next_slot = packed_chunk[pos + 1];
                // Next neighbor's position from the slot-ordered AoS (clustered,
                // and consistent with the main gather above — points[next] would
                // be a cold scattered read into the otherwise-unused points[]);
                // bit-identical to points[point_indices[next_slot]].
                let next_dot = points[generator_idx].dot(pos_slots[next_slot as usize].pos);
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
            #[cfg(feature = "timing")]
            audit_directional_batch_skip(
                phase.builder,
                &packed_chunk[pos + 1..batch.n],
                batch.unseen_bound,
                pos_slots,
                counters,
            );
        }
    }
    counters.clipping_time += t_clip.elapsed();
}

/// Phase 2: drive the neighbor stream to termination, failure, or exhaustion.
#[allow(clippy::too_many_arguments)]
fn consume_stream(
    stream: &mut DirectedNeighborStream<'_, '_, '_, '_>,
    mut phase: StreamPhase<'_>,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
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
                clip_batch(
                    &mut phase,
                    batch,
                    points,
                    pos_slots,
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
                        pos_slots,
                        counters,
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
            if failure == CellFailure::UnboundedAfterExhaustion {
                let t_cert = crate::knn_clipping::timing::Timer::start();
                if ctx
                    .builder
                    .to_vertex_data_from_all_constraints(points, &mut ctx.output_buffer)
                    .is_ok()
                {
                    counters.fallback_all_constraints += 1;
                    counters.certification_time += t_cert.elapsed();
                    return Ok(());
                }
                counters.certification_time += t_cert.elapsed();
            }
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
    let pos_slots = grid.point_pos_slots();

    let mut trace = BuildTrace::new();
    let mut counters = BuildCounters::new();

    ctx.builder.reset(generator_idx, points[generator_idx]);
    ctx.attempted_neighbors.clear();
    ctx.output_buffer.clear();

    clip_seed_neighbors(
        ctx,
        points,
        pos_slots,
        request.seed_neighbors,
        &mut trace,
        &mut counters,
    );

    {
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
            pos_slots,
            generator_idx,
            &mut trace,
            &mut counters,
        );
        counters.absorb_stream(&stream);
    }

    finish_cell(ctx, points, generator_idx, &trace, &mut counters)?;

    Ok(CellBuildStats {
        knn_query: counters.knn_query_time,
        clipping: counters.clipping_time,
        certification: counters.certification_time,
        neighbors_processed: counters.neighbors_processed,
        final_edges: ctx.output_buffer.vertices.len(),
        directional_shadow_checks: counters.directional_shadow_checks,
        directional_shadow_candidate_tests: counters.directional_shadow_candidate_tests,
        directional_shadow_hits: counters.directional_shadow_hits,
        directional_shadow_saved: counters.directional_shadow_saved,
        directional_support_candidate_tests: counters.directional_support_candidate_tests,
        directional_support_hits: counters.directional_support_hits,
        directional_support_saved: counters.directional_support_saved,
        directional_support_false_positive_hits: counters.directional_support_false_positive_hits,
        fallback_projection: counters.fallback_projection,
        fallback_polygon_cap: counters.fallback_polygon_cap,
        fallback_all_constraints: counters.fallback_all_constraints,
        incoming_seed_neighbors: request.seed_neighbors.len(),
        edgecheck_seed_clips: counters.edgecheck_seed_clips,
        knn_exhausted: counters.knn_exhausted,
        used_knn: counters.used_knn,
        did_packed: counters.did_packed,
        packed_tail_used: counters.packed_tail_used,
        packed_safe_exhausted: counters.packed_safe_exhausted,
        knn_stage: counters.knn_stage,
    })
}
