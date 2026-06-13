//! Planar pipeline driver: per-bin parallel cell construction through the
//! shared live-dedup engine, then assembly and edge reconciliation.
//!
//! The planar sibling of the spherical driver in
//! `knn_clipping::live_dedup::build`: the same per-bin loop, edge-check
//! seeding, and shard emission (shared via `emit_cell_output`), driving
//! `PlaneCellBuilder` + `PlaneNeighborStream` instead of the gnomonic
//! builder + cube-grid stream. No packed SIMD stage yet — every query runs
//! the shell-expansion path.
//!
//! Lives in `plane_clipping` so module dependencies run one way
//! (`plane_clipping -> knn_clipping::live_dedup`'s pub(crate) seam); the
//! dedup engine itself never references planar types, and positions flow
//! through it as native `Vec2` via the engine's position-generic seam.

use glam::Vec2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::diagram::VoronoiCell;
use crate::knn_clipping::cell_build::{CellBuildError, CellFailure, CellOutputBuffer};
use crate::knn_clipping::edge_reconcile;
use crate::knn_clipping::timing::{CellSubAccum, KnnCellStage, LapTimer, Timer, TimingBuilder};
use crate::live_dedup::{
    self, assign_bins_with, checked_local_id, checked_u32, emit_cell_output, target_bin_count,
    unpack_edge_key, BinAssignment, BinId, BuildCellsError, EdgeScratch, PackedLayoutCapacityError,
    ShardContext, ShardState, ShardedCellsData,
};
use crate::packed_layout::PackedSlotLayout;
use crate::plane_clipping::PlaneCellBuilder;
use crate::plane_grid::packed::{
    PlanePackedGroupInput, PlanePackedQuery, PlanePackedScratch, PlanePackedTimings,
    PlanePreparedGroupStatus,
};
use crate::plane_grid::{
    PlaneGrid, PlaneGridScratch, PlaneNeighborBatchSource, PlaneNeighborFrontier,
    PlaneNeighborStream,
};
use crate::policy::PackedNeighborPolicy;

pub(crate) struct PlaneCellsOutput {
    pub(crate) vertices: Vec<Vec2>,
    pub(crate) cells: Vec<VoronoiCell>,
    pub(crate) cell_indices: Vec<u32>,
}

/// Build, dedup, assemble, and edge-reconcile all planar cells.
///
/// `points` are normalized coordinates inside `[0, domain.x] x [0, domain.y]`
/// (the caller maps the user rect); positions come back as normalized `Vec2`.
pub(crate) fn compute_plane_cells(
    points: &[Vec2],
    grid: &PlaneGrid,
    domain: Vec2,
    tb: &mut TimingBuilder,
) -> Result<PlaneCellsOutput, crate::VoronoiError> {
    let t = Timer::start();
    let sharded = build_cells_sharded_plane(points, grid, domain)
        .map_err(|err| map_plane_build_error(err, points))?;
    #[cfg_attr(not(feature = "timing"), allow(clippy::clone_on_copy))]
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());

    let t = Timer::start();
    let assembly = live_dedup::assemble_sharded_live_dedup(sharded)?;
    #[allow(clippy::clone_on_copy)] // real DedupSubPhases is not Copy
    tb.set_dedup(t.elapsed(), assembly.dedup_sub.clone());

    let t = Timer::start();
    let records: Vec<live_dedup::EdgeRecord> = assembly
        .unresolved_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();
    #[cfg(feature = "p5_shadow")]
    crate::knn_clipping::p5_shadow::record_plane_unresolved(records.iter().map(|r| r.key.as_u64()));
    let mut cells = assembly.cells;
    let mut cell_indices = assembly.cell_indices;
    edge_reconcile::reconcile_unresolved_edges(
        &records,
        &assembly.vertices,
        &mut cells,
        &mut cell_indices,
        &assembly.vertex_keys,
        crate::tolerances::PLANE_RECONCILE_DEGENERATE_LEN_EPS,
        edge_reconcile::repair_apply_from_env(),
    )?;
    tb.set_edge_reconcile(t.elapsed());

    Ok(PlaneCellsOutput {
        vertices: assembly.vertices,
        cells,
        cell_indices,
    })
}

/// Square block bin layout over the res x res plane grid, with the same
/// packed (bin, local) layout and cell-major local ordering as the sphere.
fn assign_bins_plane(
    num_points: usize,
    grid: &PlaneGrid,
) -> Result<BinAssignment, PackedLayoutCapacityError> {
    let res = grid.res();
    let target_bins = target_bin_count(1);

    let mut bin_res = (target_bins as f64).sqrt().ceil() as usize;
    bin_res = bin_res.clamp(1, res.max(1));
    let bin_stride = res.div_ceil(bin_res).max(1);
    let bin_res = res.div_ceil(bin_stride);
    let num_bins = bin_res * bin_res;

    let bin_for_cell = move |cell: usize| -> usize {
        let ix = cell % res;
        let iy = cell / res;
        let bu = (ix / bin_stride).min(bin_res - 1);
        let bv = (iy / bin_stride).min(bin_res - 1);
        bv * bin_res + bu
    };

    assign_bins_with(
        num_points,
        res * res,
        grid.cell_offsets(),
        grid.point_indices(),
        num_bins,
        bin_for_cell,
    )
}

/// Per-worker reusable state for planar cell construction.
struct PlaneWorker {
    builder: PlaneCellBuilder,
    output_buffer: CellOutputBuffer<Vec2>,
    batch: Vec<u32>,
    /// Squared distances parallel to `batch`, sorted ascending within the
    /// ring (the per-emission termination bounds, straight from the
    /// frontier's sort keys).
    batch_dists: Vec<f32>,
    /// Sorted neighbor ids already clipped as edge-check seeds; stream
    /// re-emissions of these are skipped (re-clipping with a different
    /// epsilon could disagree with the seeded decision).
    seed_ids: Vec<u32>,
    /// Frontier scratch, reused across the bin's cells.
    grid_scratch: PlaneGridScratch,
}

fn build_cells_sharded_plane(
    points: &[Vec2],
    grid: &PlaneGrid,
    domain: Vec2,
) -> Result<ShardedCellsData<Vec2>, BuildCellsError> {
    let assignment =
        assign_bins_plane(points.len(), grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    // Wall ids occupy n..n+4 in key space.
    checked_u32(points.len() + 4, "virtual wall ids")?;
    let wall_base = points.len() as u32;

    let per_bin: Result<Vec<(ShardState<Vec2>, CellSubAccum)>, BuildCellsError> =
        maybe_par_into_iter!(0..num_bins)
            .map(|bin_usize| {
                let bin = BinId::from_usize(bin_usize);
                let my_generators = &assignment.bin_generators[bin_usize];
                let mut shard = ShardState::new(my_generators.len());
                let mut sub_accum = CellSubAccum::new();
                let mut edge_scratch = EdgeScratch::new();
                let mut worker = PlaneWorker {
                    builder: PlaneCellBuilder::new(0, Vec2::ZERO, wall_base, domain),
                    output_buffer: CellOutputBuffer::default(),
                    batch: Vec::new(),
                    batch_dists: Vec::new(),
                    seed_ids: Vec::new(),
                    grid_scratch: grid.make_scratch(),
                };

                let vertex_capacity = my_generators.len().saturating_mul(6);
                shard.output.vertices.reserve(vertex_capacity);
                shard.output.vertex_keys.reserve(vertex_capacity);
                shard.output.cell_indices.reserve(vertex_capacity);
                shard.output.deferred_slots.reserve(my_generators.len());

                let layout = PackedSlotLayout::new(
                    &assignment.slot_gen_map,
                    assignment.local_shift,
                    assignment.local_mask,
                );

                let mut packed_scratch = PlanePackedScratch::new();
                #[cfg_attr(
                    not(feature = "timing"),
                    allow(clippy::default_constructed_unit_structs)
                )]
                let mut packed_timings = PlanePackedTimings::default();
                let packed_policy = PackedNeighborPolicy::for_point_count(points.len(), false);
                let packed_queries_all: Vec<u32> = my_generators
                    .iter()
                    .map(|&i| grid.point_index_to_slot(i))
                    .collect();

                // Cell-grouped runs: every cell's queries are prepared as one
                // packed group (locals are assigned in cell-major order, so
                // each run is contiguous in my_generators).
                let mut cursor = 0usize;
                while cursor < my_generators.len() {
                    let cell = grid.point_index_to_cell(my_generators[cursor]);
                    let group_start = cursor;
                    while cursor < my_generators.len()
                        && grid.point_index_to_cell(my_generators[cursor]) == cell
                    {
                        cursor += 1;
                    }

                    let queries = &packed_queries_all[group_start..cursor];
                    let query_local_start = checked_u32(group_start, "packed query local start")?;
                    let group = PlanePackedGroupInput::new(
                        cell,
                        bin.as_u8(),
                        queries,
                        query_local_start,
                        layout,
                    );

                    match packed_scratch.prepare_group(grid, group, &mut packed_timings) {
                        PlanePreparedGroupStatus::Ready(mut prepared) => {
                            for (offset, &global) in
                                my_generators[group_start..cursor].iter().enumerate()
                            {
                                let local_idx = group_start + offset;
                                let local =
                                    checked_local_id(local_idx, "shard-local generator index")?;
                                let mut shard_ctx = ShardContext {
                                    shard: &mut shard,
                                    bin,
                                    local,
                                };
                                let packed = PlanePackedQuery::new(
                                    &mut prepared,
                                    &mut packed_timings,
                                    offset,
                                    packed_policy,
                                );
                                build_and_emit_cell_plane(
                                    &mut sub_accum,
                                    &mut worker,
                                    &mut edge_scratch,
                                    &mut shard_ctx,
                                    points,
                                    grid,
                                    &assignment,
                                    layout,
                                    global,
                                    Some(packed),
                                )?;
                            }
                        }
                        PlanePreparedGroupStatus::SlowPath => {
                            for (offset, &global) in
                                my_generators[group_start..cursor].iter().enumerate()
                            {
                                let local_idx = group_start + offset;
                                let local =
                                    checked_local_id(local_idx, "shard-local generator index")?;
                                let mut shard_ctx = ShardContext {
                                    shard: &mut shard,
                                    bin,
                                    local,
                                };
                                build_and_emit_cell_plane(
                                    &mut sub_accum,
                                    &mut worker,
                                    &mut edge_scratch,
                                    &mut shard_ctx,
                                    points,
                                    grid,
                                    &assignment,
                                    layout,
                                    global,
                                    None,
                                )?;
                            }
                        }
                    }
                    #[cfg(feature = "timing")]
                    {
                        sub_accum.add_packed_knn(packed_timings.total());
                        sub_accum.add_packed_knn_breakdown(&packed_timings);
                        packed_timings.clear();
                    }
                }

                Ok((shard, sub_accum))
            })
            .collect();
    let per_bin = per_bin?;

    let mut shards: Vec<ShardState<Vec2>> = Vec::with_capacity(num_bins);
    let mut merged_sub = CellSubAccum::new();
    for (shard, sub) in per_bin {
        merged_sub.merge(&sub);
        shards.push(shard);
    }

    Ok(ShardedCellsData::from_parts(assignment, shards, merged_sub))
}

#[inline]
fn cell_build_error(generator_idx: usize, failure: CellFailure) -> BuildCellsError {
    BuildCellsError::CellBuild(CellBuildError {
        generator_idx,
        failure,
        detail: None,
    })
}

#[allow(clippy::too_many_arguments)] // driver seam, mirrors the spherical build_and_emit_cell
fn build_and_emit_cell_plane<'p, 'g>(
    cell_sub: &mut CellSubAccum,
    worker: &mut PlaneWorker,
    edge_scratch: &mut EdgeScratch,
    shard_ctx: &mut ShardContext<'_, Vec2>,
    points: &[Vec2],
    grid: &PlaneGrid,
    assignment: &BinAssignment,
    layout: PackedSlotLayout<'_>,
    generator_idx: usize,
    packed: Option<PlanePackedQuery<'_, 'p, 'g>>,
) -> Result<(), BuildCellsError> {
    let cell_start = checked_u32(
        shard_ctx.shard.output.cell_indices.len(),
        "cell index start",
    )?;
    shard_ctx
        .shard
        .output
        .set_cell_start(shard_ctx.local, cell_start);

    let incoming_checks = shard_ctx.shard.dedup.take_edge_checks(shard_ctx.local);
    let cell_idx = checked_u32(generator_idx, "generator index")?;

    let builder = &mut worker.builder;
    builder.reset(generator_idx, points[generator_idx]);

    // Seed with incoming edge-check constraints, reusing the opposite side's
    // epsilon so both cells make the same near-degenerate decisions.
    worker.seed_ids.clear();
    for check in &incoming_checks {
        let (a, b) = unpack_edge_key(check.key);
        let neighbor_idx = if a == cell_idx { b } else { a } as usize;
        let neighbor_slot = grid.point_index_to_slot(neighbor_idx);
        builder
            .clip_with_slot_edgecheck(
                neighbor_idx,
                neighbor_slot,
                points[neighbor_idx],
                check.hp_eps,
            )
            .map_err(|f| cell_build_error(generator_idx, f))?;
        worker.seed_ids.push(neighbor_idx as u32);
    }
    worker.seed_ids.sort_unstable();

    let directed_ctx = crate::cube_grid::DirectedEligibility::from_layout(
        shard_ctx.bin.as_u8(),
        shard_ctx.local.as_u32(),
        layout,
    );
    let mut stream = PlaneNeighborStream::new(
        grid,
        points,
        generator_idx,
        &mut worker.grid_scratch,
        directed_ctx,
        packed,
    );
    let point_indices = grid.point_indices();

    // Coarse knn/clip lap attribution per batch (zero-sized without the
    // `timing` feature): frontier/advance time -> knn, clip loop -> clip.
    let mut lap = LapTimer::start();
    let mut neighbors_processed = 0usize;
    let mut tail_used = false;
    let mut expand_used = false;
    let mut used_knn = false;
    'stream: loop {
        match stream.frontier(&mut worker.batch, &mut worker.batch_dists) {
            PlaneNeighborFrontier::ExactBatch(batch) => {
                cell_sub.add_knn(lap.lap());
                match batch.source {
                    PlaneNeighborBatchSource::PackedTail => tail_used = true,
                    PlaneNeighborBatchSource::PackedExpandR2 => expand_used = true,
                    PlaneNeighborBatchSource::ShellExpand => used_knn = true,
                    PlaneNeighborBatchSource::PackedChunk0 => {}
                }
                for pos in 0..batch.n {
                    let slot = worker.batch[pos];
                    let neighbor_idx = point_indices[slot as usize] as usize;
                    if worker
                        .seed_ids
                        .binary_search(&(neighbor_idx as u32))
                        .is_ok()
                    {
                        continue;
                    }
                    // Packed -> takeover overlap can re-emit a neighbor; the
                    // re-clip is an Unchanged no-op (same plane, same eps).
                    builder
                        .clip_with_slot(neighbor_idx, slot, points[neighbor_idx])
                        .map_err(|f| cell_build_error(generator_idx, f))?;
                    neighbors_processed += 1;

                    // Remaining-unseen lower bound: the rest of this batch is
                    // sorted ascending (batch_dists are the frontier's own
                    // sort keys), and unseen_bound covers everything beyond
                    // it (shell layers re-cover, so always combine).
                    let bound = if pos + 1 < batch.n {
                        worker.batch_dists[pos + 1].min(batch.unseen_bound)
                    } else {
                        batch.unseen_bound
                    };
                    if builder.can_terminate(bound) {
                        cell_sub.add_clip(lap.lap());
                        break 'stream;
                    }
                }
                cell_sub.add_clip(lap.lap());
                stream.advance_frontier();
            }
            PlaneNeighborFrontier::UnknownButBounded { dist_lower_bound } => {
                cell_sub.add_knn(lap.lap());
                if builder.can_terminate(dist_lower_bound) {
                    break 'stream;
                }
                stream.advance_frontier();
            }
            PlaneNeighborFrontier::Exhausted => {
                cell_sub.add_knn(lap.lap());
                break;
            }
        }
    }

    let stage = if used_knn {
        KnnCellStage::ShellExpand
    } else if expand_used {
        KnnCellStage::PackedExpandR2
    } else if tail_used {
        KnnCellStage::PackedTail
    } else {
        KnnCellStage::PackedChunk0
    };
    cell_sub.add_cell_stage(
        stage,
        stream.knn_exhausted(),
        neighbors_processed,
        tail_used,
        expand_used,
        stream.packed_safe_exhausted(),
        used_knn,
        incoming_checks.len(),
        worker.seed_ids.len(),
    );

    builder
        .to_vertex_data(&mut worker.output_buffer)
        .map_err(|f| cell_build_error(generator_idx, f))?;

    emit_cell_output(
        cell_sub,
        edge_scratch,
        shard_ctx,
        assignment,
        cell_idx,
        cell_start,
        &worker.output_buffer,
        incoming_checks,
    )
}

/// Count neighbors of `generator_idx` within the planar coincidence
/// distance (normalized units). Error-path only; the O(n) scan is fine.
fn count_coincident_neighbors(points: &[Vec2], generator_idx: usize) -> usize {
    let Some(&g) = points.get(generator_idx) else {
        return 0;
    };
    let limit_sq =
        crate::tolerances::PLANE_COINCIDENT_DIST * crate::tolerances::PLANE_COINCIDENT_DIST;
    points
        .iter()
        .enumerate()
        .filter(|&(i, p)| i != generator_idx && (*p - g).length_squared() <= limit_sq)
        .count()
}

fn map_plane_build_error(err: BuildCellsError, points: &[Vec2]) -> crate::VoronoiError {
    match err {
        BuildCellsError::CellBuild(err) => match err.failure {
            // Verify before classifying (mirrors the sphere's
            // classify_coincident_clipped_away): a clipped-away planar cell
            // is degenerate input only if the generator really has
            // near-coincident neighbors — otherwise it signals an internal
            // clipping failure and must not send the user hunting their data.
            CellFailure::ClippedAway => {
                let coincident_pairs = count_coincident_neighbors(points, err.generator_idx);
                if coincident_pairs > 0 {
                    crate::VoronoiError::DegenerateInput {
                        coincident_pairs,
                        message: format!(
                            "generator {} was fully clipped away ({} neighbor(s) within \
                             the planar coincidence distance)",
                            err.generator_idx, coincident_pairs
                        ),
                    }
                } else {
                    crate::VoronoiError::ComputationFailed(format!(
                        "planar cell construction failed for generator {}: cell clipped \
                         away without coincident neighbors (internal error)",
                        err.generator_idx
                    ))
                }
            }
            failure => crate::VoronoiError::ComputationFailed(format!(
                "planar cell construction failed for generator {}: {:?}",
                err.generator_idx, failure
            )),
        },
        BuildCellsError::PackedLayoutCapacity(err) => {
            crate::VoronoiError::RepresentationLimit(format!(
                "packed bin/local layout capacity exceeded in bin {}: population {} exceeds \
                 local mask {} (num_bins={}, local_shift={})",
                err.bin, err.local_population, err.local_mask, err.num_bins, err.local_shift
            ))
        }
        BuildCellsError::RepresentationLimit(message) => {
            crate::VoronoiError::RepresentationLimit(message)
        }
    }
}
