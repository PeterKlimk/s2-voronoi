//! Spherical pipeline driver: per-bin parallel cell construction through
//! the shared live-dedup engine (the planar sibling is
//! `plane_clipping::driver`).

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cube_grid::packed_knn::{
    PackedGroupInput, PackedKnnCellScratch, PackedKnnTimings, PreparedPackedGroup,
    PreparedPackedGroupStatus,
};
use crate::cube_grid::{CubeMapGrid, PackedQuery};
use crate::knn_clipping::cell_build::{build_cell_into, CellBuildContext, CellBuildRequest};
use crate::knn_clipping::TerminationConfig;
use crate::live_dedup::{
    assign_bins, checked_local_id, checked_u32, emit_cell_output, BinId, BuildCellsError,
    EdgeScratch, ShardContext, ShardState, ShardedCellsData, VertexData,
};
use crate::packed_layout::PackedSlotLayout;

pub(super) struct GridContext<'a> {
    pub(super) points: &'a [Vec3],
    pub(super) grid: &'a CubeMapGrid,
    pub(super) assignment: &'a crate::live_dedup::BinAssignment,
}

struct SphereCellScratch {
    edge_scratch: EdgeScratch,
}

impl SphereCellScratch {
    fn new() -> Self {
        Self {
            edge_scratch: EdgeScratch::new(),
        }
    }
}

#[inline(always)]
fn stored_x_outside_zero_hint(a: Vec3, b: Vec3) -> u32 {
    ((a.x - b.x).abs() > crate::tolerances::OUTPUT_RESOLUTION_ZERO_HINT_X_EPS) as u32
}

#[inline(never)]
fn has_zero_edge_candidate(vertices: &[VertexData]) -> bool {
    debug_assert!(vertices.len() >= 3);
    let mut all_edges_outside_hint = 1u32;
    let ptr = vertices.as_ptr();
    // SAFETY: successful final extraction has at least three initialized
    // vertices. Read the contiguous cycle once, then compare its closing edge.
    unsafe {
        let first = (*ptr).1;
        let mut previous = first;
        for i in 1..vertices.len() {
            let current = (*ptr.add(i)).1;
            all_edges_outside_hint &= stored_x_outside_zero_hint(previous, current);
            previous = current;
        }
        all_edges_outside_hint &= stored_x_outside_zero_hint(previous, first);
    }
    all_edges_outside_hint == 0
}

pub(crate) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
) -> Result<ShardedCellsData, BuildCellsError> {
    let policy = termination.packed_policy(points.len());
    // Legacy config compatibility: no-k fallback ignores this cap.

    let assignment = assign_bins(points, grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    let packed_policy = policy;

    let per_bin: Result<Vec<(ShardState, crate::timing::CellSubAccum)>, BuildCellsError> =
        maybe_par_into_iter!(0..num_bins)
            .map(|bin_usize| {
                use crate::timing::CellSubAccum;

                let bin = BinId::from_usize(bin_usize);
                let my_generators = &assignment.bin_generators[bin_usize];
                let mut shard = ShardState::new(my_generators.len());

                let mut sub_accum = CellSubAccum::new();
                let mut build_ctx = CellBuildContext::new(grid, policy);
                let mut live_ctx = SphereCellScratch::new();
                let vertex_capacity = my_generators.len().saturating_mul(6);
                shard.output.vertices.reserve(vertex_capacity);
                shard.output.vertex_keys.reserve(vertex_capacity);
                shard.output.vertex_incidence.reserve(vertex_capacity);
                shard
                    .output
                    .cell_indices
                    .reserve(my_generators.len().saturating_mul(6));
                // Conservative estimate for off-shard vertices.
                shard.output.deferred_slots.reserve(my_generators.len());

                let mut packed_scratch = PackedKnnCellScratch::new();

                #[cfg_attr(
                    not(feature = "timing"),
                    allow(clippy::default_constructed_unit_structs)
                )]
                let mut packed_timings = PackedKnnTimings::default();

                #[cfg(debug_assertions)]
                {
                    for &i in my_generators {
                        debug_assert_eq!(
                            assignment.generator_bin[i], bin,
                            "cell assigned to wrong bin"
                        );
                    }
                }

                let grid_ctx = GridContext {
                    points,
                    grid,
                    assignment: &assignment,
                };
                let packed_layout = PackedSlotLayout::new(
                    &assignment.slot_gen_map,
                    assignment.local_shift,
                    assignment.local_mask,
                );

                let mut cursor = 0usize;
                while cursor < my_generators.len() {
                    let cell = grid.point_index_to_cell(my_generators[cursor]) as u32;
                    let start = cursor;
                    while cursor < my_generators.len()
                        && grid.point_index_to_cell(my_generators[cursor]) as u32 == cell
                    {
                        cursor += 1;
                    }
                    let group_start = start;

                    if packed_policy.enabled() {
                        let query_slot_start = grid.cell_offsets()[cell as usize];
                        let query_local_start =
                            checked_u32(group_start, "packed query local start")?;
                        let group = PackedGroupInput::new(
                            cell as usize,
                            bin.as_u8(),
                            query_slot_start,
                            cursor - group_start,
                            query_local_start,
                            packed_layout,
                        );

                        #[cfg(not(feature = "timing"))]
                        let t_packed = crate::timing::Timer::start();
                        let prepared =
                            packed_scratch.prepare_group_directed(grid, group, &mut packed_timings);
                        #[cfg(not(feature = "timing"))]
                        let packed_elapsed = t_packed.elapsed();

                        match prepared {
                            PreparedPackedGroupStatus::Ready(mut prepared) => {
                                emit_generator_group(
                                    &mut sub_accum,
                                    &mut build_ctx,
                                    &mut live_ctx,
                                    &mut shard,
                                    bin,
                                    &grid_ctx,
                                    &my_generators[group_start..cursor],
                                    group_start,
                                    query_slot_start,
                                    Some((&mut prepared, &mut packed_timings, packed_policy)),
                                )?;
                                #[cfg(feature = "timing")]
                                prepared.record_tail_usage(&mut packed_timings);
                            }
                            PreparedPackedGroupStatus::SlowPath => {
                                emit_generator_group(
                                    &mut sub_accum,
                                    &mut build_ctx,
                                    &mut live_ctx,
                                    &mut shard,
                                    bin,
                                    &grid_ctx,
                                    &my_generators[group_start..cursor],
                                    group_start,
                                    query_slot_start,
                                    None,
                                )?;
                            }
                        }

                        #[cfg(feature = "timing")]
                        {
                            sub_accum.add_packed_knn(packed_timings.total());
                            sub_accum.add_packed_knn_breakdown(&packed_timings);
                        }
                        #[cfg(not(feature = "timing"))]
                        sub_accum.add_packed_knn(packed_elapsed);
                    } else {
                        emit_generator_group(
                            &mut sub_accum,
                            &mut build_ctx,
                            &mut live_ctx,
                            &mut shard,
                            bin,
                            &grid_ctx,
                            &my_generators[group_start..cursor],
                            group_start,
                            grid.cell_offsets()[cell as usize],
                            None,
                        )?;
                    }
                }

                Ok((shard, sub_accum))
            })
            .collect();
    let per_bin = per_bin?;

    let mut shards: Vec<ShardState> = Vec::with_capacity(num_bins);
    let mut merged_sub = crate::timing::CellSubAccum::new();
    for (shard, sub) in per_bin {
        merged_sub.merge(&sub);
        shards.push(shard);
    }

    Ok(ShardedCellsData::from_parts(assignment, shards, merged_sub))
}

/// Emit every cell of one same-grid-cell generator group. The call sites
/// (packed ready, packed slow-path, packed disabled) differ only in whether a
/// prepared packed batch feeds each cell's `PackedQuery`; the per-cell setup
/// lives here once.
#[allow(clippy::too_many_arguments)]
fn emit_generator_group<'c>(
    sub_accum: &mut crate::timing::CellSubAccum,
    build_ctx: &mut CellBuildContext,
    live_ctx: &mut SphereCellScratch,
    shard: &mut ShardState,
    bin: BinId,
    grid_ctx: &GridContext<'c>,
    generators: &[usize],
    group_start: usize,
    query_slot_start: u32,
    mut packed: Option<(
        &mut PreparedPackedGroup<'_, 'c>,
        &mut PackedKnnTimings,
        crate::policy::PackedNeighborPolicy,
    )>,
) -> Result<(), BuildCellsError> {
    for (offset, &global) in generators.iter().enumerate() {
        let local = checked_local_id(group_start + offset, "shard-local generator index")?;
        let mut shard_ctx = ShardContext {
            shard: &mut *shard,
            bin,
            local,
        };
        let query = packed.as_mut().map(|(prepared, timings, policy)| {
            PackedQuery::new(prepared, timings, offset, *policy)
        });
        let query_slot = query_slot_start + offset as u32;
        build_and_emit_cell(
            sub_accum,
            &mut *build_ctx,
            &mut *live_ctx,
            &mut shard_ctx,
            grid_ctx,
            global,
            query_slot,
            query,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_and_emit_cell<'a, 'b, 'c>(
    cell_sub: &'a mut crate::timing::CellSubAccum,
    build_ctx: &'a mut CellBuildContext,
    live_ctx: &'a mut SphereCellScratch,
    shard_ctx: &'a mut ShardContext<'b>,
    grid_ctx: &'a GridContext<'c>,
    generator_idx: usize,
    query_slot: u32,
    packed: Option<PackedQuery<'_, '_, 'c>>,
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
    let generator_slot = grid_ctx.grid.point_pos_slots()[query_slot as usize];
    debug_assert_eq!(
        generator_slot.idx as usize, generator_idx,
        "cell-major generator order must match the grid's contiguous slot order"
    );
    debug_assert_eq!(
        generator_slot.pos.to_array().map(f32::to_bits),
        grid_ctx.points[generator_idx].to_array().map(f32::to_bits),
        "slot-native generator position must be bit-identical to the canonical point"
    );

    let directed_ctx = crate::cube_grid::DirectedEligibility::from_layout(
        shard_ctx.bin.as_u8(),
        shard_ctx.local.as_u32(),
        PackedSlotLayout::new(
            &grid_ctx.assignment.slot_gen_map,
            grid_ctx.assignment.local_shift,
            grid_ctx.assignment.local_mask,
        ),
    );
    let stats = build_cell_into(
        build_ctx,
        CellBuildRequest {
            points: grid_ctx.points,
            grid: grid_ctx.grid,
            generator_idx,
            generator: generator_slot.pos,
            directed_ctx,
            packed,
            incoming_checks: &incoming_checks,
        },
    )
    .map_err(BuildCellsError::CellBuild)?;
    stats.record_into(cell_sub);

    let output_buffer = build_ctx.output_buffer();
    emit_cell_output(
        cell_sub,
        &mut live_ctx.edge_scratch,
        shard_ctx,
        grid_ctx.assignment,
        cell_idx,
        cell_start,
        output_buffer,
        incoming_checks,
    )?;
    let exact_zero_edge_hint = has_zero_edge_candidate(&output_buffer.vertices);
    if exact_zero_edge_hint {
        shard_ctx
            .shard
            .output
            .exact_zero_edge_hint_cells
            .push(cell_idx);
    }
    Ok(())
}

#[cfg(test)]
mod resolution_hint_tests {
    use super::has_zero_edge_candidate;
    use glam::Vec3;

    fn vertex(id: u32, x: f32) -> ([u32; 3], Vec3) {
        ([id, id + 10, id + 20], Vec3::new(x, 0.0, 1.0))
    }

    #[test]
    fn widened_hint_includes_its_bound_and_rejects_separated_cycle() {
        let bound = crate::tolerances::OUTPUT_RESOLUTION_ZERO_HINT_X_EPS;
        assert!(has_zero_edge_candidate(&[
            vertex(0, 0.0),
            vertex(1, bound),
            vertex(2, 1.0),
        ]));
        let representative_bound = crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS;
        assert!(has_zero_edge_candidate(&[
            vertex(0, representative_bound),
            vertex(1, -representative_bound),
            vertex(2, 1.0),
        ]));
        assert!(has_zero_edge_candidate(&[
            vertex(0, 0.0),
            vertex(1, -0.0),
            vertex(2, 1.0),
        ]));
        let just_outside = f32::from_bits(bound.to_bits() + 1);
        assert!(!has_zero_edge_candidate(&[
            vertex(0, 0.0),
            vertex(1, just_outside),
            vertex(2, -1.0),
        ]));
        assert!(!has_zero_edge_candidate(&[
            vertex(0, -1.0),
            vertex(1, 0.0),
            vertex(2, 1.0),
        ]));
    }
}
