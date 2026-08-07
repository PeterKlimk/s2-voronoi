mod failure;
mod frontier;
mod telemetry_detail;
#[cfg(test)]
mod tests;

use crate::cube_grid::{
    DirectedNeighborBatchSource, DirectedNeighborFrontier, DirectedNeighborStream, PackedQuery,
};
use crate::knn_clipping::topo2d::types::MAX_POLY_VERTICES;
use crate::knn_clipping::topo2d::{BuilderFallbackTrigger, BuilderStepOutcome};
use crate::live_dedup::EdgeCheck;
use crate::policy::PackedNeighborPolicy;

use crate::live_dedup::{CellBuildError, CellFailure, CellOutputBuffer};
use failure::{classify_terminal_failure, unexpected_failure_error};
use frontier::{complete_exact_bound, maybe_terminate_or_advance_frontier, probe_frontier};
use telemetry_detail::{BuildTelemetryDetail, CellTelemetryDetail};

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
    // Keep worker setup folded into the shard driver. This and the phase
    // annotations below pin the release-codegen shape; unrelated cold-pipeline
    // growth can otherwise make LLVM outline the per-generator driver. See
    // docs/performance.md#source-pinned-performance-decisions.
    #[inline(always)]
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

    #[cfg(test)]
    pub(crate) fn output_buffer(&self) -> &CellOutputBuffer {
        &self.output_buffer
    }

    pub(crate) fn output_buffer_mut(&mut self) -> &mut CellOutputBuffer {
        &mut self.output_buffer
    }
}

#[cfg(test)]
fn maybe_force_fallback(
    builder: &mut crate::knn_clipping::topo2d::Topo2DBuilder,
    force_fallback_after_neighbors_processed: &mut Option<usize>,
    points: &[Vec3],
    neighbors_processed: usize,
    fallback_trigger: &mut Option<BuilderFallbackTrigger>,
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

    let trigger = BuilderFallbackTrigger::ProjectionLimit;
    *fallback_trigger = Some(trigger);
    let entered = builder.try_enter_fallback(points, trigger);
    debug_assert!(entered);
    *force_fallback_after_neighbors_processed = None;
}

pub(crate) struct CellBuildRequest<'a, 'm, 'p, 'g, 's> {
    pub(crate) points: &'a [Vec3],
    pub(crate) grid: &'a crate::cube_grid::CubeMapGrid,
    pub(crate) generator_idx: usize,
    pub(crate) generator: Vec3,
    pub(crate) directed_ctx: crate::cube_grid::DirectedEligibility<'m>,
    pub(crate) packed: Option<PackedQuery<'p, 'g, 'm>>,
    pub(crate) incoming_checks: &'s [EdgeCheck],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CellBuildStats {
    neighbors_processed: usize,
    final_edges: usize,
    telemetry_detail: CellTelemetryDetail,
    fallback_code: u8,
    recovered_all_constraints: bool,
    incoming_seed_neighbors: usize,
    edgecheck_seed_clips: usize,
    knn_exhausted: bool,
    used_knn: bool,
    did_packed: bool,
    packed_tail_used: bool,
    packed_safe_exhausted: bool,
    knn_stage: crate::telemetry::KnnCellStage,
    #[cfg(test)]
    termination_checkpoint: Option<TerminationCheckpoint>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TerminationCheckpoint {
    PackedPreBatch,
    PackedMidBatch,
    PackedPostBatch,
    Shell,
}

impl CellBuildStats {
    #[inline]
    pub(crate) fn record_into(&self, cell_telemetry: &mut crate::telemetry::CellTelemetryAccum) {
        let (fallback_projection, fallback_polygon_cap, mut fallback_all_constraints) =
            fallback_counts_from_code(self.fallback_code);
        fallback_all_constraints += usize::from(self.recovered_all_constraints);
        cell_telemetry.add_fallbacks(
            fallback_projection,
            fallback_polygon_cap,
            fallback_all_constraints,
        );
        self.telemetry_detail
            .record_into(cell_telemetry, self.neighbors_processed);

        let stage = if self.used_knn {
            self.knn_stage
        } else if self.did_packed {
            if self.packed_tail_used {
                crate::telemetry::KnnCellStage::PackedTail
            } else {
                crate::telemetry::KnnCellStage::PackedChunk0
            }
        } else {
            self.knn_stage
        };

        cell_telemetry.add_cell_stage(
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
    fallback_trigger: Option<BuilderFallbackTrigger>,
) -> Option<String> {
    fallback_trigger
        .or_else(|| {
            crate::knn_clipping::topo2d::Topo2DBuilder::fallback_trigger_for_failure(failure)
        })
        .map(|trigger| {
            format!(
                "fallback trigger={:?}, replay_constraints={}, replay_generator_idx={}",
                trigger,
                builder.accepted_constraint_count(),
                builder.generator_idx()
            )
        })
}

/// Compact classification of the most recent clip attempt.
#[derive(Clone, Copy)]
#[repr(u8)]
enum LastClipKind {
    None,
    EdgecheckSeed,
    PackedChunk0,
    PackedTail,
    ShellExpand,
}

impl LastClipKind {
    #[inline]
    fn from_batch_source(source: DirectedNeighborBatchSource) -> Self {
        match source {
            DirectedNeighborBatchSource::PackedChunk0 => Self::PackedChunk0,
            DirectedNeighborBatchSource::PackedTail => Self::PackedTail,
            DirectedNeighborBatchSource::ShellExpand => Self::ShellExpand,
        }
    }

    fn phase(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::EdgecheckSeed => "edgecheck_seed",
            Self::PackedChunk0 | Self::PackedTail | Self::ShellExpand => "stream",
        }
    }

    fn batch_source(self) -> Option<DirectedNeighborBatchSource> {
        match self {
            Self::None | Self::EdgecheckSeed => None,
            Self::PackedChunk0 => Some(DirectedNeighborBatchSource::PackedChunk0),
            Self::PackedTail => Some(DirectedNeighborBatchSource::PackedTail),
            Self::ShellExpand => Some(DirectedNeighborBatchSource::ShellExpand),
        }
    }
}

/// Diagnostic trail of the most recent clip, for unexpected-failure reports.
pub(super) struct BuildTrace {
    // Both values originate as u32 grid identifiers. Keeping them in one word
    // avoids publishing two independent Option fields on every clip attempt.
    last_neighbor: u64,
    last_clip_kind: LastClipKind,
    fallback_trigger: Option<BuilderFallbackTrigger>,
    // One-shot state: success installs FallbackBuilder and rejection stops the
    // stream, so no cell can record a second ordinary fallback transition.
    fallback_code: u8,
}

#[inline]
fn fallback_code(trigger: BuilderFallbackTrigger) -> u8 {
    match trigger {
        BuilderFallbackTrigger::ProjectionLimit => 1,
        BuilderFallbackTrigger::PolygonVertexLimit => 2,
        BuilderFallbackTrigger::ClippedAway => 3,
    }
}

#[inline]
fn fallback_counts_from_code(code: u8) -> (usize, usize, usize) {
    match code {
        1 => (1, 0, 0),
        2 => (0, 1, 0),
        3 => (0, 0, 1),
        _ => (0, 0, 0),
    }
}

impl BuildTrace {
    fn new() -> Self {
        Self {
            last_neighbor: 0,
            last_clip_kind: LastClipKind::None,
            fallback_trigger: None,
            fallback_code: 0,
        }
    }

    #[inline(always)]
    fn record_edgecheck(&mut self, neighbor_idx: u32, neighbor_slot: u32) {
        self.last_neighbor = (u64::from(neighbor_idx) << 32) | u64::from(neighbor_slot);
        self.last_clip_kind = LastClipKind::EdgecheckSeed;
    }

    #[inline(always)]
    fn record_stream(&mut self, neighbor_idx: u32, neighbor_slot: u32, kind: LastClipKind) {
        self.last_neighbor = (u64::from(neighbor_idx) << 32) | u64::from(neighbor_slot);
        self.last_clip_kind = kind;
    }

    #[inline]
    fn record_fallback(&mut self, trigger: BuilderFallbackTrigger) {
        debug_assert_eq!(self.fallback_code, 0, "fallback can be entered only once");
        self.fallback_trigger = Some(trigger);
        self.fallback_code = fallback_code(trigger);
    }

    pub(super) fn last_neighbor_idx(&self) -> Option<usize> {
        (!matches!(self.last_clip_kind, LastClipKind::None))
            .then_some((self.last_neighbor >> 32) as usize)
    }

    pub(super) fn last_neighbor_slot(&self) -> Option<u32> {
        (!matches!(self.last_clip_kind, LastClipKind::None)).then_some(self.last_neighbor as u32)
    }

    pub(super) fn last_clip_phase(&self) -> &'static str {
        self.last_clip_kind.phase()
    }

    pub(super) fn last_batch_source(&self) -> Option<DirectedNeighborBatchSource> {
        self.last_clip_kind.batch_source()
    }
}

/// Algorithmic counters accumulated across the build phases.
pub(super) struct BuildCounters {
    pub(super) neighbors_processed: usize,
    edgecheck_seed_clips: usize,
    pub(super) used_knn: bool,
    knn_stage: crate::telemetry::KnnCellStage,
    pub(super) knn_exhausted: bool,
    pub(super) did_packed: bool,
    packed_tail_used: bool,
    packed_safe_exhausted: bool,
    telemetry_detail: BuildTelemetryDetail,
    recovered_all_constraints: bool,
    terminated: bool,
    #[cfg(test)]
    termination_checkpoint: Option<TerminationCheckpoint>,
}

impl BuildCounters {
    fn new() -> Self {
        Self {
            neighbors_processed: 0,
            edgecheck_seed_clips: 0,
            used_knn: false,
            knn_stage: crate::telemetry::KnnCellStage::ShellExpand,
            knn_exhausted: false,
            did_packed: false,
            packed_tail_used: false,
            packed_safe_exhausted: false,
            telemetry_detail: BuildTelemetryDetail::new(),
            recovered_all_constraints: false,
            terminated: false,
            #[cfg(test)]
            termination_checkpoint: None,
        }
    }

    #[cfg(test)]
    fn record_termination_checkpoint(&mut self, checkpoint: TerminationCheckpoint) {
        debug_assert!(self.termination_checkpoint.is_none());
        self.termination_checkpoint = Some(checkpoint);
    }

    fn absorb_stream(&mut self, stream: &DirectedNeighborStream<'_, '_, '_, '_>) {
        self.did_packed |= stream.did_packed();
        self.packed_tail_used |= stream.packed_tail_used();
        self.packed_safe_exhausted |= stream.packed_safe_exhausted();
        self.knn_exhausted |= stream.knn_exhausted();
    }
}

#[cfg(test)]
#[inline]
fn fallback_counts(trace: &BuildTrace, counters: &BuildCounters) -> (usize, usize, usize) {
    let (projection, polygon_cap, mut all_constraints) =
        fallback_counts_from_code(trace.fallback_code);
    all_constraints += usize::from(counters.recovered_all_constraints);
    (projection, polygon_cap, all_constraints)
}

/// Rebuild an actually exhausted, still-synthetic cell from an unrestricted
/// spherical constraint stream. The initial spherical seed is formed only
/// from real constraints; after that, ordinary spherical clipping is safe
/// because every later halfspace can only shrink the real spherical polygon.
fn recover_unbounded_after_exhaustion(
    ctx: &mut CellBuildContext,
    grid: &crate::cube_grid::CubeMapGrid,
    generator_idx: usize,
    generator: Vec3,
    counters: &mut BuildCounters,
) -> bool {
    counters.telemetry_detail.invalidate_progress_tail();
    let pos_slots = grid.point_pos_slots();
    let mut seed_slots = Vec::new();
    let mut seeded = false;
    let mut frontier = grid.unrestricted_shell_frontier(generator, generator_idx, &mut ctx.scratch);

    while let Some(batch) = frontier.frontier(&mut ctx.packed_chunk) {
        let slots = &ctx.packed_chunk[..batch.n];
        counters.neighbors_processed += slots.len();

        if !seeded {
            seed_slots.extend_from_slice(slots);
            if seed_slots.len() >= 3 {
                seeded = ctx
                    .builder
                    .try_restart_spherical_from_neighbors(seed_slots.iter().map(|&slot| {
                        let point = pos_slots[slot as usize];
                        (point.idx as usize, slot, point.pos)
                    }));
            }
        } else {
            for &slot in slots {
                let point = pos_slots[slot as usize];
                if ctx
                    .builder
                    .clip_with_slot_result_policy(point.idx as usize, slot, point.pos)
                    .is_err()
                {
                    return false;
                }
            }
        }

        frontier.advance();
    }

    seeded
        && ctx.builder.is_bounded()
        && ctx
            .builder
            .to_vertex_data_full(&mut ctx.output_buffer)
            .is_ok()
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
// These phase seams are organizational; all three belong to one hot
// per-generator operation and should remain flattened into its caller.
#[inline(always)]
fn clip_seed_neighbors(
    ctx: &mut CellBuildContext,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
    incoming_checks: &[EdgeCheck],
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    if incoming_checks.is_empty() {
        return;
    }
    for check in incoming_checks {
        let neighbor_slot = check.neighbor_slot;
        let neighbor_point = pos_slots[neighbor_slot as usize];
        let neighbor_idx = neighbor_point.idx as usize;
        trace.record_edgecheck(neighbor_point.idx, neighbor_slot);

        if !ctx.attempted_neighbors.insert(neighbor_slot as usize) {
            continue;
        }

        let neighbor = neighbor_point.pos;
        let fallback_rejected =
            match ctx
                .builder
                .clip_with_slot_edgecheck_policy(neighbor_idx, neighbor_slot, neighbor)
            {
                Ok(BuilderStepOutcome::Applied) => false,
                Ok(BuilderStepOutcome::NeedsFallback(trigger)) => {
                    trace.record_fallback(trigger);
                    !ctx.builder.try_enter_fallback(points, trigger)
                }
                Err(_) => break,
            };
        counters.neighbors_processed += 1;
        // Forwarded edge checks are construction seeds rather than a proximity
        // tail. Treat each accepted seed as progress; the subsequent stream
        // tail remains exact.
        counters
            .telemetry_detail
            .record_progress(counters.neighbors_processed);
        if fallback_rejected {
            break;
        }
        #[cfg(test)]
        maybe_force_fallback(
            &mut ctx.builder,
            &mut ctx.force_fallback_after_neighbors_processed,
            points,
            counters.neighbors_processed,
            &mut trace.fallback_trigger,
        );
    }
    counters.edgecheck_seed_clips = counters.neighbors_processed;
}

/// Clip one exact batch; returns with `counters.terminated` set when the
/// builder's certificate fires mid-batch.
// There is one production call site. Keeping the source-specialized loops
// visible there avoids a separate outlined copy and reduces total code size.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn clip_batch(
    phase: &mut StreamPhase<'_>,
    batch: crate::cube_grid::DirectedNeighborBatch,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
    generator_idx: usize,
    generator: Vec3,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    match batch.source {
        DirectedNeighborBatchSource::ShellExpand => clip_batch_source::<true>(
            phase,
            batch,
            points,
            pos_slots,
            generator_idx,
            generator,
            trace,
            counters,
        ),
        DirectedNeighborBatchSource::PackedChunk0 | DirectedNeighborBatchSource::PackedTail => {
            clip_batch_source::<false>(
                phase,
                batch,
                points,
                pos_slots,
                generator_idx,
                generator,
                trace,
                counters,
            )
        }
    }
}

#[inline(always)]
fn should_clip_neighbor<const SHELL: bool>(
    attempted_neighbors: &mut AttemptedNeighbors,
    neighbor_slot: usize,
) -> bool {
    if SHELL {
        // The takeover re-covers packed-served points; dedup on insertion.
        attempted_neighbors.insert(neighbor_slot)
    } else {
        attempted_neighbors.mark(neighbor_slot);
        true
    }
}

/// Result of one direct builder clip without the outer builder-mode branch.
enum DirectClipOutcome {
    Applied(crate::knn_clipping::topo2d::types::ClipResult),
    NeedsFallback(BuilderFallbackTrigger),
    Failed,
}

trait DirectStreamBuilder {
    #[cfg(test)]
    const GNOMONIC: bool;

    fn clip_stream_neighbor(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> DirectClipOutcome;
    fn stream_is_bounded(&self) -> bool;
    fn stream_can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool;
}

impl DirectStreamBuilder for crate::knn_clipping::topo2d::builder::GnomonicBuilder {
    #[cfg(test)]
    const GNOMONIC: bool = true;

    #[inline(always)]
    fn clip_stream_neighbor(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> DirectClipOutcome {
        match self.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor) {
            Ok(result) => DirectClipOutcome::Applied(result),
            Err(failure) => {
                match crate::knn_clipping::topo2d::Topo2DBuilder::fallback_trigger_for_failure(
                    failure,
                ) {
                    Some(trigger) => DirectClipOutcome::NeedsFallback(trigger),
                    None => DirectClipOutcome::Failed,
                }
            }
        }
    }

    #[inline(always)]
    fn stream_is_bounded(&self) -> bool {
        self.is_bounded()
    }

    #[inline(always)]
    fn stream_can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool {
        self.can_terminate(max_unseen_dot_bound)
    }
}

impl DirectStreamBuilder for crate::knn_clipping::topo2d::builder::FallbackBuilder {
    #[cfg(test)]
    const GNOMONIC: bool = false;

    #[inline(always)]
    fn clip_stream_neighbor(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> DirectClipOutcome {
        match self.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor) {
            Ok(result) => DirectClipOutcome::Applied(result),
            Err(_) => DirectClipOutcome::Failed,
        }
    }

    #[inline(always)]
    fn stream_is_bounded(&self) -> bool {
        self.is_bounded()
    }

    #[inline(always)]
    fn stream_can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool {
        self.can_terminate(max_unseen_dot_bound)
    }
}

enum DirectBatchStop {
    Complete(usize),
    SwitchToFallback {
        next_pos: usize,
        prefix_consumed: usize,
        trigger: BuilderFallbackTrigger,
        forced: bool,
    },
}

/// Run one contiguous batch segment against a builder whose mode is fixed for
/// the whole call. The ordinary gnomonic path therefore carries no
/// per-neighbor `BuilderImpl` discriminant branch.
#[allow(clippy::too_many_arguments)]
fn clip_batch_direct<const SHELL: bool, B: DirectStreamBuilder>(
    builder: &mut B,
    packed_chunk: &[u32],
    attempted_neighbors: &mut AttemptedNeighbors,
    batch: crate::cube_grid::DirectedNeighborBatch,
    pos_slots: &[crate::cube_grid::SlotPoint],
    generator_idx: usize,
    generator: Vec3,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
    start_pos: usize,
    #[cfg(test)] force_fallback_after_neighbors_processed: &mut Option<usize>,
) -> DirectBatchStop {
    let last_clip_kind = LastClipKind::from_batch_source(batch.source);
    let mut prefix_consumed = start_pos;
    for pos in start_pos..batch.n {
        prefix_consumed = pos + 1;
        let neighbor_slot = packed_chunk[pos];
        // One fused load gets both the global index and position from the
        // slot-ordered AoS instead of two scattered by-slot loads.
        let slot_point = pos_slots[neighbor_slot as usize];
        let neighbor_idx = slot_point.idx as usize;
        if SHELL {
            if neighbor_idx == generator_idx {
                continue;
            }
        } else {
            debug_assert_ne!(
                neighbor_idx, generator_idx,
                "packed neighbor batch must exclude its generator"
            );
        }

        if !should_clip_neighbor::<SHELL>(attempted_neighbors, neighbor_slot as usize) {
            continue;
        }

        trace.record_stream(slot_point.idx, neighbor_slot, last_clip_kind);
        let clip_result =
            match builder.clip_stream_neighbor(neighbor_idx, neighbor_slot, slot_point.pos) {
                DirectClipOutcome::Applied(result) => result,
                DirectClipOutcome::NeedsFallback(trigger) => {
                    trace.record_fallback(trigger);
                    counters.neighbors_processed += 1;
                    counters
                        .telemetry_detail
                        .record_progress(counters.neighbors_processed);
                    return DirectBatchStop::SwitchToFallback {
                        next_pos: pos + 1,
                        prefix_consumed,
                        trigger,
                        forced: false,
                    };
                }
                DirectClipOutcome::Failed => return DirectBatchStop::Complete(prefix_consumed),
            };

        counters.neighbors_processed += 1;
        if clip_result == crate::knn_clipping::topo2d::types::ClipResult::Changed {
            counters
                .telemetry_detail
                .record_progress(counters.neighbors_processed);
        }

        #[cfg(test)]
        if B::GNOMONIC {
            if let Some(target) = *force_fallback_after_neighbors_processed {
                if counters.neighbors_processed >= target {
                    let trigger = BuilderFallbackTrigger::ProjectionLimit;
                    trace.fallback_trigger = Some(trigger);
                    *force_fallback_after_neighbors_processed = None;
                    return DirectBatchStop::SwitchToFallback {
                        next_pos: pos + 1,
                        prefix_consumed,
                        trigger,
                        forced: true,
                    };
                }
            }
        }

        if builder.stream_is_bounded()
            && clip_result == crate::knn_clipping::topo2d::types::ClipResult::Unchanged
        {
            let bound = if pos + 1 < batch.n {
                let next_slot = packed_chunk[pos + 1];
                // The next position comes from the same clustered AoS; this
                // exact successor plus `unseen_bound` covers the full remainder.
                let next = pos_slots[next_slot as usize].pos;
                let next_dot = crate::fp::dot3_f32(
                    generator.x,
                    generator.y,
                    generator.z,
                    next.x,
                    next.y,
                    next.z,
                );
                complete_exact_bound(next_dot, batch.unseen_bound)
            } else {
                batch.unseen_bound
            };
            if builder.stream_can_terminate(bound) {
                #[cfg(test)]
                counters.record_termination_checkpoint(match batch.source {
                    DirectedNeighborBatchSource::ShellExpand => TerminationCheckpoint::Shell,
                    DirectedNeighborBatchSource::PackedChunk0
                    | DirectedNeighborBatchSource::PackedTail => {
                        if pos + 1 < batch.n {
                            TerminationCheckpoint::PackedMidBatch
                        } else {
                            TerminationCheckpoint::PackedPostBatch
                        }
                    }
                });
                counters.terminated = true;
                break;
            }
        }
    }
    DirectBatchStop::Complete(prefix_consumed)
}

/// Source-specialized batch loop. Builder mode is selected once per segment;
/// a rare gnomonic fallback request transfers the unconsumed suffix to the
/// fallback specialization.
#[allow(clippy::too_many_arguments)]
fn clip_batch_source<const SHELL: bool>(
    phase: &mut StreamPhase<'_>,
    batch: crate::cube_grid::DirectedNeighborBatch,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
    generator_idx: usize,
    generator: Vec3,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    let packed_chunk = &phase.packed_chunk[..batch.n];
    let mut start_pos = 0usize;
    let prefix_consumed;

    loop {
        let stop = if phase.builder.is_fallback() {
            let builder = phase
                .builder
                .fallback_builder_mut()
                .expect("fallback mode must expose fallback builder");
            clip_batch_direct::<SHELL, _>(
                builder,
                packed_chunk,
                phase.attempted_neighbors,
                batch,
                pos_slots,
                generator_idx,
                generator,
                trace,
                counters,
                start_pos,
                #[cfg(test)]
                phase.force_fallback_after_neighbors_processed,
            )
        } else {
            let builder = phase
                .builder
                .gnomonic_builder_mut()
                .expect("gnomonic mode must expose gnomonic builder");
            clip_batch_direct::<SHELL, _>(
                builder,
                packed_chunk,
                phase.attempted_neighbors,
                batch,
                pos_slots,
                generator_idx,
                generator,
                trace,
                counters,
                start_pos,
                #[cfg(test)]
                phase.force_fallback_after_neighbors_processed,
            )
        };

        match stop {
            DirectBatchStop::Complete(prefix) => {
                prefix_consumed = prefix;
                break;
            }
            DirectBatchStop::SwitchToFallback {
                next_pos,
                prefix_consumed: prefix,
                trigger,
                forced,
            } => {
                let entered = phase.builder.try_enter_fallback(points, trigger);
                if forced {
                    debug_assert!(entered);
                }
                if !entered {
                    prefix_consumed = prefix;
                    break;
                }
                start_pos = next_pos;
            }
        }
    }

    counters
        .telemetry_detail
        .record_packed_batch_usage(batch.source, batch.n, prefix_consumed);
    counters.telemetry_detail.record_shell_batch::<SHELL>(
        batch.n,
        prefix_consumed,
        counters.terminated,
    );
}

/// Phase 2: drive the neighbor stream to termination, failure, or exhaustion.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn consume_stream(
    stream: &mut DirectedNeighborStream<'_, '_, '_, '_>,
    mut phase: StreamPhase<'_>,
    points: &[Vec3],
    pos_slots: &[crate::cube_grid::SlotPoint],
    generator_idx: usize,
    generator: Vec3,
    trace: &mut BuildTrace,
    counters: &mut BuildCounters,
) {
    while !counters.terminated && !phase.builder.is_failed() {
        let frontier = probe_frontier(
            stream,
            phase.packed_chunk,
            &mut counters.used_knn,
            &mut counters.knn_stage,
        );

        match frontier {
            DirectedNeighborFrontier::ExactBatch(batch) => {
                clip_batch(
                    &mut phase,
                    batch,
                    points,
                    pos_slots,
                    generator_idx,
                    generator,
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
                        counters,
                    );
                }
            }
            DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                // Only packed stages produce bounded-unknown frontiers; the
                // takeover always emits exact layers.
                if phase.builder.is_bounded() && phase.builder.can_terminate(dot_upper_bound) {
                    #[cfg(test)]
                    counters.record_termination_checkpoint(TerminationCheckpoint::PackedPostBatch);
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
#[inline(always)]
fn finish_cell(
    ctx: &mut CellBuildContext,
    points: &[Vec3],
    grid: &crate::cube_grid::CubeMapGrid,
    generator_idx: usize,
    generator: Vec3,
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
                let recovered = recover_unbounded_after_exhaustion(
                    ctx,
                    grid,
                    generator_idx,
                    generator,
                    counters,
                );
                if recovered {
                    counters.recovered_all_constraints = true;
                    return Ok(());
                }
            }
            return Err(CellBuildError {
                generator_idx,
                failure,
                detail: fallback_detail(&ctx.builder, failure, trace.fallback_trigger),
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
    Ok(())
}

#[inline(always)]
pub(crate) fn build_cell_into<'a, 'm, 'p, 'g, 's>(
    ctx: &'a mut CellBuildContext,
    mut request: CellBuildRequest<'a, 'm, 'p, 'g, 's>,
) -> Result<CellBuildStats, CellBuildError> {
    let points = request.points;
    let grid = request.grid;
    let generator_idx = request.generator_idx;
    let generator = request.generator;
    let pos_slots = grid.point_pos_slots();

    let mut trace = BuildTrace::new();
    let mut counters = BuildCounters::new();

    ctx.builder.reset(generator_idx, generator);
    ctx.attempted_neighbors.clear();
    // Every successful finish path clears the reusable output before writing:
    // gnomonic extraction, spherical fallback extraction, and all-constraints
    // exhaustion recovery. Error results return before the driver can consume
    // this buffer, so clearing here would only duplicate successful-path work.

    clip_seed_neighbors(
        ctx,
        points,
        pos_slots,
        request.incoming_checks,
        &mut trace,
        &mut counters,
    );

    {
        let mut stream = DirectedNeighborStream::new(
            grid,
            generator,
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
            generator,
            &mut trace,
            &mut counters,
        );
        counters.absorb_stream(&stream);
    }

    finish_cell(
        ctx,
        points,
        grid,
        generator_idx,
        generator,
        &trace,
        &mut counters,
    )?;

    Ok(CellBuildStats {
        neighbors_processed: counters.neighbors_processed,
        final_edges: ctx.output_buffer.vertices.len(),
        telemetry_detail: counters
            .telemetry_detail
            .finish(counters.neighbors_processed),
        fallback_code: trace.fallback_code,
        recovered_all_constraints: counters.recovered_all_constraints,
        incoming_seed_neighbors: request.incoming_checks.len(),
        edgecheck_seed_clips: counters.edgecheck_seed_clips,
        knn_exhausted: counters.knn_exhausted,
        used_knn: counters.used_knn,
        did_packed: counters.did_packed,
        packed_tail_used: counters.packed_tail_used,
        packed_safe_exhausted: counters.packed_safe_exhausted,
        knn_stage: counters.knn_stage,
        #[cfg(test)]
        termination_checkpoint: counters.termination_checkpoint,
    })
}
