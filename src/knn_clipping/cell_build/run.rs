use std::time::Duration;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream, PackedQuery,
};
use crate::policy::{KnnPolicy, TerminationPolicy};

use super::{CellBuildError, CellFailure, CellOutputBuffer};

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

#[inline]
fn probe_frontier<'a, 'm, 'p>(
    stream: &mut DirectedNeighborStream<'a, 'm, 'p>,
    packed_chunk: &mut Vec<u32>,
    used_knn: &mut bool,
    knn_stage: &mut crate::knn_clipping::timing::KnnCellStage,
    knn_query_time: &mut Duration,
) -> DirectedNeighborFrontier {
    let t_knn = crate::knn_clipping::timing::Timer::start();
    let cursor_stage_before = stream.is_cursor_stage();
    let frontier = stream.frontier(packed_chunk);
    let frontier_is_cursor = match frontier {
        DirectedNeighborFrontier::ExactBatch(batch) => {
            batch.source == DirectedNeighborBatchSource::DirectedCursor
        }
        DirectedNeighborFrontier::UnknownButBounded { .. }
        | DirectedNeighborFrontier::Exhausted => cursor_stage_before,
    };
    if frontier_is_cursor {
        *used_knn = true;
        *knn_stage = crate::knn_clipping::timing::KnnCellStage::DirectedCursor;
        *knn_query_time += t_knn.elapsed();
    }
    frontier
}

#[inline]
fn maybe_terminate_or_advance_frontier<'a, 'm, 'p>(
    stream: &mut DirectedNeighborStream<'a, 'm, 'p>,
    packed_chunk: &mut Vec<u32>,
    builder: &mut crate::knn_clipping::topo2d::Topo2DBuilder,
    termination: TerminationPolicy,
    neighbors_processed: usize,
    used_knn: &mut bool,
    knn_stage: &mut crate::knn_clipping::timing::KnnCellStage,
    knn_query_time: &mut Duration,
) -> bool {
    let frontier = probe_frontier(stream, packed_chunk, used_knn, knn_stage, knn_query_time);
    let can_check_cursor = termination.should_check(neighbors_processed);

    match frontier {
        DirectedNeighborFrontier::ExactBatch(batch) => {
            let can_check =
                batch.source != DirectedNeighborBatchSource::DirectedCursor || can_check_cursor;
            can_check && builder.can_terminate(batch.first_dot)
        }
        DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
            let can_check = !stream.is_cursor_stage() || can_check_cursor;
            if can_check && builder.can_terminate(dot_upper_bound) {
                true
            } else {
                stream.advance_frontier();
                false
            }
        }
        DirectedNeighborFrontier::Exhausted => true,
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

pub(crate) struct CellBuildRequest<'a, 'm, 'p, 's> {
    pub(crate) points: &'a [Vec3],
    pub(crate) grid: &'a crate::cube_grid::CubeMapGrid,
    pub(crate) generator_idx: usize,
    pub(crate) directed_ctx: crate::cube_grid::DirectedCtx<'m>,
    pub(crate) termination: TerminationPolicy,
    pub(crate) packed: Option<PackedQuery<'p, 'm>>,
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

pub(crate) fn build_cell_into<'a, 'm, 'p, 's>(
    ctx: &'a mut CellBuildContext,
    mut request: CellBuildRequest<'a, 'm, 'p, 's>,
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
            if ctx
                .builder
                .clip_with_slot_edgecheck(
                    seed.neighbor_idx,
                    seed.neighbor_slot,
                    neighbor,
                    seed.hp_eps,
                )
                .is_err()
            {
                break;
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
                    let clip_result = match ctx.builder.clip_with_slot_result(
                        neighbor_idx,
                        neighbor_slot,
                        neighbor,
                    ) {
                        Ok(result) => result,
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
            });
        }
        panic_unexpected_failure(
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
        );
    }

    let t_cert = crate::knn_clipping::timing::Timer::start();
    if let Err(failure) = ctx.builder.to_vertex_data_full(&mut ctx.output_buffer) {
        panic_unexpected_failure(
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
        );
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

fn classify_terminal_failure(
    bounded: bool,
    failure: Option<CellFailure>,
    knn_exhausted: bool,
) -> Option<CellFailure> {
    match failure {
        Some(CellFailure::ProjectionInvalid) => return Some(CellFailure::ProjectionInvalid),
        Some(CellFailure::TooManyVertices) => return Some(CellFailure::TooManyVertices),
        _ => {}
    }
    if !bounded && knn_exhausted {
        return Some(CellFailure::UnboundedAfterExhaustion);
    }
    None
}

fn panic_unexpected_failure(
    ctx: &CellBuildContext,
    points: &[Vec3],
    generator_idx: usize,
    neighbors_processed: usize,
    did_packed: bool,
    used_knn: bool,
    knn_exhausted: bool,
    last_clip_phase: &str,
    last_batch_source: Option<DirectedNeighborBatchSource>,
    last_neighbor_idx: Option<usize>,
    last_neighbor_slot: Option<u32>,
    context: &str,
    explicit_failure: Option<CellFailure>,
) -> ! {
    let (active, total) = ctx.builder.count_active_planes();
    let gen = points[generator_idx];
    let neighbor_indices: Vec<usize> = ctx.builder.neighbor_indices_iter().collect();
    let failure = explicit_failure.or(ctx.builder.failure());

    panic!(
        "Cell {} unexpected {} failure: bounded={}, failure={:?}, \
         planes={}, active={}, vertices={}, neighbors_processed={}, \
         did_packed={}, did_cursor_fallback={}, knn_exhausted={}, \
         last_clip_phase={}, last_batch_source={:?}, last_neighbor_idx={:?}, last_neighbor_slot={:?}\n\
         Generator pos: {:?}\n\
         First 10 neighbor indices: {:?}",
        generator_idx,
        context,
        ctx.builder.is_bounded(),
        failure,
        total,
        active,
        ctx.builder.vertex_count(),
        neighbors_processed,
        did_packed,
        used_knn,
        knn_exhausted,
        last_clip_phase,
        last_batch_source,
        last_neighbor_idx,
        last_neighbor_slot,
        gen,
        &neighbor_indices[..neighbor_indices.len().min(10)],
    );
}

#[cfg(test)]
mod tests {
    use super::{build_cell_into, classify_terminal_failure, CellBuildContext, CellBuildRequest};
    use crate::cube_grid::{CubeMapGrid, DirectedCtx};
    use crate::knn_clipping::cell_build::CellFailure;
    use crate::knn_clipping::TerminationConfig;
    use glam::Vec3;

    fn octahedron_points() -> Vec<Vec3> {
        vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
    }

    #[test]
    fn projection_invalid_stays_distinct_from_exhausted_unbounded() {
        assert_eq!(
            classify_terminal_failure(false, Some(CellFailure::ProjectionInvalid), true),
            Some(CellFailure::ProjectionInvalid)
        );
        assert_eq!(
            classify_terminal_failure(false, None, true),
            Some(CellFailure::UnboundedAfterExhaustion)
        );
    }

    #[test]
    fn too_many_vertices_is_a_structured_failure() {
        assert_eq!(
            classify_terminal_failure(true, Some(CellFailure::TooManyVertices), false),
            Some(CellFailure::TooManyVertices)
        );
    }

    #[test]
    fn bounded_nonfailed_cell_has_no_terminal_failure() {
        assert_eq!(classify_terminal_failure(true, None, true), None);
        assert_eq!(classify_terminal_failure(true, None, false), None);
    }

    #[test]
    fn direct_cursor_builds_normal_cell() {
        let points = octahedron_points();
        let grid = CubeMapGrid::new(&points, 4);
        let policy = TerminationConfig::default().knn_policy(points.len());
        let mut ctx = CellBuildContext::new(&grid, policy);
        let fake_slot_map = vec![0u32; points.len()];
        let directed_ctx = DirectedCtx::new(u8::MAX, 0, &fake_slot_map, 0, 0);

        let stats = build_cell_into(
            &mut ctx,
            CellBuildRequest {
                points: &points,
                grid: &grid,
                generator_idx: 0,
                directed_ctx,
                termination: policy.termination(),
                packed: None,
                seed_neighbors: &[],
            },
        )
        .expect("cell build should succeed");

        assert!(ctx.output_buffer().vertices.len() >= 3);
        assert!(!stats.knn_exhausted || !stats.did_packed);
    }
}
