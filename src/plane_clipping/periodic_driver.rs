//! Periodic pipeline driver: per-bin parallel toroidal cell construction
//! through the shared live-dedup engine (packed SIMD stage + shell-expansion
//! takeover, like the bounded sibling).

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
use crate::plane_clipping::periodic_builder::PeriodicCellBuilder;
use crate::plane_grid::packed::{
    PlanePackedGroupInput, PlanePackedQuery, PlanePackedScratch, PlanePackedTimings,
    PlanePreparedGroupStatus,
};
use crate::plane_grid::periodic::{
    PeriodicGrid, PeriodicGridScratch, PeriodicNeighborFrontier, PeriodicNeighborStream,
};
use crate::plane_grid::PlaneNeighborBatchSource;
use crate::policy::PackedNeighborPolicy;

pub(crate) struct PeriodicCellsOutput {
    pub(crate) vertices: Vec<Vec2>,
    pub(crate) cells: Vec<VoronoiCell>,
    pub(crate) cell_indices: Vec<u32>,
    /// Post-repair unpaired edges as generator pairs (empty on a valid
    /// torus result). The caller's plain path errors on these; the report
    /// path surfaces them.
    pub(crate) residual: Vec<(u32, u32)>,
}

/// Build, dedup, assemble, and edge-reconcile all toroidal cells.
///
/// `points` are normalized coordinates inside `[0, px) x [0, py)`; vertex
/// positions come back canonically wrapped into the same domain.
pub(crate) fn compute_periodic_cells(
    points: &[Vec2],
    grid: &PeriodicGrid,
    tb: &mut TimingBuilder,
) -> Result<PeriodicCellsOutput, crate::VoronoiError> {
    let t = Timer::start();
    let sharded = build_cells_sharded_periodic(points, grid).map_err(map_periodic_build_error)?;
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
    let mut cells = assembly.cells;
    let mut cell_indices = assembly.cell_indices;
    // Torus topology: no boundary, every interior edge must pair.
    // Note: reconcile distances are raw Euclidean on wrapped positions, so
    // an epsilon edge exactly straddling the wrap seam is not auto-merged;
    // such a residual makes the output an invalid subdivision, so the plain
    // path fails loud rather than returning it silently (was a documented
    // "honestly reported" finding only visible via validate_plane).
    let residual = edge_reconcile::reconcile_unresolved_edges(
        &records,
        &assembly.vertices,
        &mut cells,
        &mut cell_indices,
        &assembly.vertex_keys,
        crate::tolerances::PLANE_RECONCILE_DEGENERATE_LEN_EPS,
        edge_reconcile::repair_apply_from_env(),
        |_, _| false,
    )?;

    tb.set_edge_reconcile(t.elapsed());

    Ok(PeriodicCellsOutput {
        residual,
        vertices: assembly.vertices,
        cells,
        cell_indices,
    })
}

/// Square block bin layout over the wrapped res x res grid (blocks need no
/// wrap awareness — only neighbor *queries* wrap).
fn assign_bins_periodic(
    num_points: usize,
    grid: &PeriodicGrid,
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

struct PeriodicWorker {
    builder: PeriodicCellBuilder,
    output_buffer: CellOutputBuffer<Vec2>,
    batch: Vec<u32>,
    batch_dists: Vec<f32>,
    seed_ids: Vec<u32>,
    grid_scratch: PeriodicGridScratch,
}

fn build_cells_sharded_periodic(
    points: &[Vec2],
    grid: &PeriodicGrid,
) -> Result<ShardedCellsData<Vec2>, BuildCellsError> {
    let assignment =
        assign_bins_periodic(points.len(), grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    checked_u32(points.len(), "generator count")?;
    let (px, py) = grid.periods();

    let per_bin: Result<Vec<(ShardState<Vec2>, CellSubAccum)>, BuildCellsError> =
        maybe_par_into_iter!(0..num_bins)
            .map(|bin_usize| {
                let bin = BinId::from_usize(bin_usize);
                let my_generators = &assignment.bin_generators[bin_usize];
                let mut shard = ShardState::new(my_generators.len());
                let mut sub_accum = CellSubAccum::new();
                let mut edge_scratch = EdgeScratch::new();
                let mut worker = PeriodicWorker {
                    builder: PeriodicCellBuilder::new(0, Vec2::ZERO, px, py),
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
                // each run is contiguous in my_generators) — same protocol as
                // the bounded driver; tiny wrapped grids take the SlowPath.
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
                                build_and_emit_cell_periodic(
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
                                build_and_emit_cell_periodic(
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

#[allow(clippy::too_many_arguments)] // driver seam, mirrors the other drivers
fn build_and_emit_cell_periodic<'p, 'g>(
    cell_sub: &mut CellSubAccum,
    worker: &mut PeriodicWorker,
    edge_scratch: &mut EdgeScratch,
    shard_ctx: &mut ShardContext<'_, Vec2>,
    points: &[Vec2],
    grid: &PeriodicGrid,
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
    let mut stream = PeriodicNeighborStream::new(
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
    // The frontier loop's Exhausted arm is a non-trivial break (mirrors the
    // sibling drivers).
    #[allow(clippy::while_let_loop)]
    'stream: loop {
        match stream.frontier(&mut worker.batch, &mut worker.batch_dists) {
            PeriodicNeighborFrontier::ExactBatch(batch) => {
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
                    // Skip seeds AND already-clipped neighbors. Unlike the
                    // bounded driver, a packed -> takeover re-emit must NOT
                    // be re-clipped here: the periodic builder seeds
                    // unbounded (1e6 sentinel coordinates), so early clip
                    // intersections carry interpolation error far above the
                    // clip epsilon, and re-clipping the bit-identical plane
                    // can cut a phantom sliver off the drifted polygon.
                    match worker.seed_ids.binary_search(&(neighbor_idx as u32)) {
                        Ok(_) => continue,
                        Err(insert_at) => {
                            worker.seed_ids.insert(insert_at, neighbor_idx as u32);
                        }
                    }
                    builder
                        .clip_with_slot(neighbor_idx, slot, points[neighbor_idx])
                        .map_err(|f| cell_build_error(generator_idx, f))?;
                    neighbors_processed += 1;

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
            PeriodicNeighborFrontier::UnknownButBounded { dist_lower_bound } => {
                cell_sub.add_knn(lap.lap());
                if builder.can_terminate(dist_lower_bound) {
                    break 'stream;
                }
                stream.advance_frontier();
            }
            PeriodicNeighborFrontier::Exhausted => {
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

fn map_periodic_build_error(err: BuildCellsError) -> crate::VoronoiError {
    match err {
        BuildCellsError::CellBuild(err) => match err.failure {
            // The half-period guard / unbounded exhaustion: the torus is
            // underpopulated for nearest-image clipping.
            CellFailure::UnboundedAfterExhaustion => crate::VoronoiError::UnsupportedGeometry {
                generator_index: err.generator_idx,
                message: "cell exceeds the half-period guard (too few generators for a \
                          periodic domain of this size); add generators or use the bounded \
                          compute_plane"
                    .to_string(),
            },
            CellFailure::ClippedAway => crate::VoronoiError::DegenerateInput {
                coincident_pairs: 1,
                message: format!(
                    "generator {} was fully clipped away (near-coincident generators)",
                    err.generator_idx
                ),
            },
            failure => crate::VoronoiError::ComputationFailed(format!(
                "periodic cell construction failed for generator {}: {:?}",
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
