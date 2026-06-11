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
//! dedup engine itself never references planar types. Positions enter the
//! shard layer as `Vec3 {x, y, 0}` — the documented temporary embedding,
//! pending position-type genericization.

use glam::{Vec2, Vec3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::diagram::VoronoiCell;
use crate::knn_clipping::cell_build::{CellBuildError, CellFailure, CellOutputBuffer};
use crate::knn_clipping::edge_reconcile;
use crate::knn_clipping::live_dedup::{
    self, assign_bins_with, checked_local_id, checked_u32, emit_cell_output, target_bin_count,
    unpack_edge_key, BinAssignment, BinId, BuildCellsError, LiveDedupCellScratch,
    PackedLayoutCapacityError, ShardContext, ShardState, ShardedCellsData,
};
use crate::knn_clipping::timing::CellSubAccum;
use crate::packed_layout::PackedSlotLayout;
use crate::plane_clipping::PlaneCellBuilder;
use crate::plane_grid::{PlaneGrid, PlaneGridScratch, PlaneNeighborFrontier, PlaneNeighborStream};

pub(crate) struct PlaneCellsOutput {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) cells: Vec<VoronoiCell>,
    pub(crate) cell_indices: Vec<u32>,
}

/// Build, dedup, assemble, and edge-reconcile all planar cells.
///
/// `points` are normalized coordinates inside `[0, domain.x] x [0, domain.y]`
/// (the caller maps the user rect); positions come back as `Vec3 {x, y, 0}`.
pub(crate) fn compute_plane_cells(
    points: &[Vec2],
    grid: &PlaneGrid,
    domain: Vec2,
) -> Result<PlaneCellsOutput, crate::VoronoiError> {
    let sharded = build_cells_sharded_plane(points, grid, domain)
        .map_err(|err| map_plane_build_error(err, points))?;
    let assembly = live_dedup::assemble_sharded_live_dedup(sharded)?;

    let records: Vec<live_dedup::EdgeRecord> = assembly
        .unresolved_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();
    let mut cells = assembly.cells;
    let mut cell_indices = assembly.cell_indices;
    if let Some((reconciled_cells, reconciled_indices)) =
        edge_reconcile::reconcile_unresolved_edges(
            &records,
            &assembly.vertices,
            &cells,
            &cell_indices,
            &assembly.vertex_keys,
            crate::tolerances::PLANE_RECONCILE_DEGENERATE_LEN_EPS,
        )?
    {
        cells = reconciled_cells;
        cell_indices = reconciled_indices;
    }

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
    output_buffer: CellOutputBuffer,
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
) -> Result<ShardedCellsData, BuildCellsError> {
    let assignment =
        assign_bins_plane(points.len(), grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    // Wall ids occupy n..n+4 in key space.
    checked_u32(points.len() + 4, "virtual wall ids")?;
    let wall_base = points.len() as u32;

    let per_bin: Result<Vec<(ShardState, CellSubAccum)>, BuildCellsError> =
        maybe_par_into_iter!(0..num_bins)
            .map(|bin_usize| {
                let bin = BinId::from_usize(bin_usize);
                let my_generators = &assignment.bin_generators[bin_usize];
                let mut shard = ShardState::new(my_generators.len());
                let mut sub_accum = CellSubAccum::new();
                let mut live_ctx = LiveDedupCellScratch::new(points.len());
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

                for (local_idx, &global) in my_generators.iter().enumerate() {
                    let local = checked_local_id(local_idx, "shard-local generator index")?;
                    let mut shard_ctx = ShardContext {
                        shard: &mut shard,
                        bin,
                        local,
                    };
                    build_and_emit_cell_plane(
                        &mut sub_accum,
                        &mut worker,
                        &mut live_ctx,
                        &mut shard_ctx,
                        points,
                        grid,
                        &assignment,
                        layout,
                        global,
                    )?;
                }

                Ok((shard, sub_accum))
            })
            .collect();
    let per_bin = per_bin?;

    let mut shards: Vec<ShardState> = Vec::with_capacity(num_bins);
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
fn build_and_emit_cell_plane(
    cell_sub: &mut CellSubAccum,
    worker: &mut PlaneWorker,
    live_ctx: &mut LiveDedupCellScratch,
    shard_ctx: &mut ShardContext<'_>,
    points: &[Vec2],
    grid: &PlaneGrid,
    assignment: &BinAssignment,
    layout: PackedSlotLayout<'_>,
    generator_idx: usize,
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
    );
    let point_indices = grid.point_indices();

    'stream: while let PlaneNeighborFrontier::ExactBatch(batch) =
        stream.frontier(&mut worker.batch, &mut worker.batch_dists)
    {
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
            builder
                .clip_with_slot(neighbor_idx, slot, points[neighbor_idx])
                .map_err(|f| cell_build_error(generator_idx, f))?;

            // Remaining-unseen lower bound: the rest of this ring is sorted
            // ascending (batch_dists are the frontier's own sort keys), and
            // unseen_bound covers everything beyond it (shell layers
            // re-cover, so always combine).
            let bound = if pos + 1 < batch.n {
                worker.batch_dists[pos + 1].min(batch.unseen_bound)
            } else {
                batch.unseen_bound
            };
            if builder.can_terminate(bound) {
                break 'stream;
            }
        }
        stream.advance_frontier();
    }

    builder
        .to_vertex_data_v3(&mut worker.output_buffer)
        .map_err(|f| cell_build_error(generator_idx, f))?;

    emit_cell_output(
        cell_sub,
        live_ctx,
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
