//! Planar cell-construction driver for live dedup.
//!
//! The planar sibling of the spherical driver in `build/mod.rs`: the same
//! per-bin parallel loop, edge-check seeding, and shard emission (shared via
//! `emit_cell_output`), driving `PlaneCellBuilder` + `PlaneNeighborStream`
//! instead of the gnomonic builder + cube-grid stream. No packed SIMD stage
//! yet — every query runs the shell-expansion path.
//!
//! Positions enter the shard layer as `Vec3 {x, y, 0}` (the documented
//! temporary embedding, pending position-type genericization). This module
//! living inside `live_dedup` is deliberate: the dedup layer is the
//! geometry-agnostic engine and each geometry brings a driver; promoting
//! `live_dedup` out of `knn_clipping` is the follow-up refactor.

use glam::Vec2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::binning::assign_bins_plane;
use super::super::shard::ShardState;
use super::super::types::BinId;
use super::super::{BuildCellsError, ShardedCellsData};
use super::{checked_local_id, checked_u32, emit_cell_output, LiveDedupCellScratch, ShardContext};
use crate::cube_grid::DirectedEligibility;
use crate::knn_clipping::cell_build::{CellBuildError, CellFailure, CellOutputBuffer};
use crate::knn_clipping::live_dedup::edge_checks::unpack_edge_key;
use crate::knn_clipping::timing::CellSubAccum;
use crate::packed_layout::PackedSlotLayout;
use crate::plane_clipping::PlaneCellBuilder;
use crate::plane_grid::{PlaneGrid, PlaneNeighborFrontier, PlaneNeighborStream};

/// Per-worker reusable state for planar cell construction.
struct PlaneWorker {
    builder: PlaneCellBuilder,
    output_buffer: CellOutputBuffer,
    batch: Vec<u32>,
    /// Sorted neighbor ids already clipped as edge-check seeds; stream
    /// re-emissions of these are skipped (re-clipping with a different
    /// epsilon could disagree with the seeded decision).
    seed_ids: Vec<u32>,
}

pub(in crate::knn_clipping) fn build_cells_sharded_plane(
    points: &[Vec2],
    grid: &PlaneGrid,
    domain: Vec2,
) -> Result<ShardedCellsData, BuildCellsError> {
    let assignment =
        assign_bins_plane(points.len(), grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    // Wall ids occupy n..n+4 in key space.
    let wall_base = checked_u32(points.len(), "generator count (wall id base)")?;
    checked_u32(points.len() + 4, "virtual wall ids")?;

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
                    seed_ids: Vec::new(),
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

    Ok(ShardedCellsData {
        assignment,
        shards,
        cell_sub: merged_sub,
    })
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
    assignment: &super::super::binning::BinAssignment,
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

    let directed_ctx =
        DirectedEligibility::from_layout(shard_ctx.bin.as_u8(), shard_ctx.local.as_u32(), layout);
    let mut stream = PlaneNeighborStream::new(grid, points, generator_idx, directed_ctx);
    let generator = points[generator_idx];
    let point_indices = grid.point_indices();

    'stream: while let PlaneNeighborFrontier::ExactBatch(batch) = stream.frontier(&mut worker.batch)
    {
        {
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

                    // Remaining-unseen lower bound: the rest of this ring is
                    // sorted ascending, and unseen_bound covers everything
                    // beyond it (shell layers re-cover, so always combine).
                    let bound = if pos + 1 < batch.n {
                        let next = point_indices[worker.batch[pos + 1] as usize] as usize;
                        (points[next] - generator)
                            .length_squared()
                            .min(batch.unseen_bound)
                    } else {
                        batch.unseen_bound
                    };
                    if builder.can_terminate(bound) {
                        break 'stream;
                    }
                }
                stream.advance_frontier();
            }
        }
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
