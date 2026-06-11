//! Cell construction for live dedup.
//!
//! The spherical driver lives here; the planar driver (plane_clipping::
//! driver) reuses the pub(crate) emission seam (`emit_cell_output`,
//! `LiveDedupCellScratch`, `ShardContext`) so the dedup layer stays
//! geometry-free.

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::binning::assign_bins;
use super::edge_checks::{collect_and_resolve_cell_edges, unpack_edge_key};
use super::packed::{pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::ShardState;
use super::types::{
    BinId, DeferredSlot, EdgeCheck, EdgeCheckOverflow, EdgeOverflowLocal, EdgeToLater, LocalId,
};
use super::{BuildCellsError, ShardedCellsData};
use crate::cube_grid::packed_knn::{
    PackedGroupInput, PackedKnnCellScratch, PackedKnnTimings, PreparedPackedGroupStatus,
};
use crate::cube_grid::{CubeMapGrid, PackedQuery};
use crate::knn_clipping::cell_build::{
    build_cell_into, CellBuildContext, CellBuildRequest, CellOutputBuffer, SeedNeighbor, VertexData,
};
use crate::knn_clipping::TerminationConfig;
use crate::packed_layout::PackedSlotLayout;

pub(crate) struct EdgeScratch {
    edges_to_later: Vec<EdgeToLater>,
    edges_overflow: Vec<EdgeOverflowLocal>,
    vertex_indices: Vec<u32>,
}

impl EdgeScratch {
    fn new() -> Self {
        Self {
            edges_to_later: Vec::new(),
            edges_overflow: Vec::new(),
            vertex_indices: Vec::new(),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    fn collect_and_resolve(
        &mut self,
        cell_idx: u32,
        shard_ctx: &mut ShardContext<'_>,
        output_buffer: &CellOutputBuffer,
        assignment: &super::binning::BinAssignment,
        incoming_checks: Vec<EdgeCheck>,
    ) {
        self.vertex_indices.clear();
        self.vertex_indices
            .resize(output_buffer.vertices.len(), INVALID_INDEX);
        collect_and_resolve_cell_edges(
            cell_idx,
            shard_ctx,
            output_buffer,
            assignment,
            incoming_checks,
            &mut self.vertex_indices,
            &mut self.edges_to_later,
            &mut self.edges_overflow,
        );
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    fn emit(
        &mut self,
        shard: &mut ShardState,
        cell_vertices: &[VertexData],
        cell_start: u32,
        bin: BinId,
    ) {
        use super::edge_checks::{third_for_edge_endpoint, unpack_edge_key};

        for entry in self.edges_to_later.drain(..) {
            let locals = entry.locals;
            let (a, b) = unpack_edge_key(entry.key);
            let thirds = [
                third_for_edge_endpoint(cell_vertices[locals[0] as usize].0, a, b),
                third_for_edge_endpoint(cell_vertices[locals[1] as usize].0, a, b),
            ];
            shard.dedup.push_edge_check(
                entry.local_b,
                EdgeCheck {
                    key: entry.key,
                    hp_eps: entry.hp_eps,
                    thirds,
                    indices: [
                        self.vertex_indices[locals[0] as usize],
                        self.vertex_indices[locals[1] as usize],
                    ],
                },
            );
        }

        for entry in self.edges_overflow.drain(..) {
            let locals = entry.locals;
            let (a, b) = unpack_edge_key(entry.key);
            let thirds = [
                third_for_edge_endpoint(cell_vertices[locals[0] as usize].0, a, b),
                third_for_edge_endpoint(cell_vertices[locals[1] as usize].0, a, b),
            ];
            shard.output.edge_check_overflow.push(EdgeCheckOverflow {
                key: entry.key,
                side: entry.side,
                source_bin: bin,
                thirds,
                indices: [
                    self.vertex_indices[locals[0] as usize],
                    self.vertex_indices[locals[1] as usize],
                ],
                slots: [cell_start + locals[0] as u32, cell_start + locals[1] as u32],
            });
        }
    }
}

pub(crate) struct LiveDedupCellScratch {
    edge_scratch: EdgeScratch,
    seed_neighbors: Vec<SeedNeighbor>,
}

impl LiveDedupCellScratch {
    pub(crate) fn new(num_points: usize) -> Self {
        Self {
            edge_scratch: EdgeScratch::new(),
            seed_neighbors: Vec::with_capacity(num_points.min(16)),
        }
    }
}

pub(super) struct GridContext<'a> {
    pub(super) points: &'a [Vec3],
    pub(super) grid: &'a CubeMapGrid,
    pub(super) assignment: &'a super::binning::BinAssignment,
}

pub(crate) struct ShardContext<'a> {
    pub(crate) shard: &'a mut ShardState,
    pub(crate) bin: BinId,
    pub(crate) local: LocalId,
}

#[inline]
fn representation_limit(message: impl Into<String>) -> BuildCellsError {
    BuildCellsError::RepresentationLimit(message.into())
}

#[inline]
pub(crate) fn checked_u32(value: usize, context: &str) -> Result<u32, BuildCellsError> {
    u32::try_from(value)
        .map_err(|_| representation_limit(format!("{context} exceeds u32 capacity")))
}

#[inline]
fn checked_u8(value: usize, context: &str) -> Result<u8, BuildCellsError> {
    u8::try_from(value).map_err(|_| representation_limit(format!("{context} exceeds u8 capacity")))
}

#[inline]
pub(crate) fn checked_local_id(value: usize, context: &str) -> Result<LocalId, BuildCellsError> {
    checked_u32(value, context).map(LocalId::from)
}

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
) -> Result<ShardedCellsData, BuildCellsError> {
    let policy = termination.packed_policy(points.len());
    // Legacy config compatibility: no-k fallback ignores this cap.

    let assignment = assign_bins(points, grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    let packed_policy = policy;

    let per_bin: Result<
        Vec<(ShardState, crate::knn_clipping::timing::CellSubAccum)>,
        BuildCellsError,
    > = maybe_par_into_iter!(0..num_bins)
        .map(|bin_usize| {
            use crate::knn_clipping::timing::CellSubAccum;

            let bin = BinId::from_usize(bin_usize);
            let my_generators = &assignment.bin_generators[bin_usize];
            let mut shard = ShardState::new(my_generators.len());

            let mut sub_accum = CellSubAccum::new();
            let mut build_ctx = CellBuildContext::new(grid, policy);
            let mut live_ctx = LiveDedupCellScratch::new(grid.point_indices().len());
            let vertex_capacity = my_generators.len().saturating_mul(6);
            shard.output.vertices.reserve(vertex_capacity);
            shard.output.vertex_keys.reserve(vertex_capacity);
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

            let packed_queries_all: Vec<u32> = my_generators
                .iter()
                .map(|&i| grid.point_index_to_slot(i))
                .collect();
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
                    let queries = &packed_queries_all[group_start..cursor];
                    let query_local_start = checked_u32(group_start, "packed query local start")?;
                    let group = PackedGroupInput::new(
                        cell as usize,
                        bin.as_u8(),
                        queries,
                        query_local_start,
                        packed_layout,
                    );

                    #[cfg(not(feature = "timing"))]
                    let t_packed = crate::knn_clipping::timing::Timer::start();
                    let prepared =
                        packed_scratch.prepare_group_directed(grid, group, &mut packed_timings);
                    #[cfg(not(feature = "timing"))]
                    let packed_elapsed = t_packed.elapsed();

                    match prepared {
                        PreparedPackedGroupStatus::Ready(mut prepared) => {
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
                                build_and_emit_cell(
                                    &mut sub_accum,
                                    &mut build_ctx,
                                    &mut live_ctx,
                                    &mut shard_ctx,
                                    &grid_ctx,
                                    global,
                                    Some(PackedQuery::new(
                                        &mut prepared,
                                        &mut packed_timings,
                                        offset,
                                        packed_policy,
                                    )),
                                )?;
                            }
                        }
                        PreparedPackedGroupStatus::SlowPath => {
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
                                build_and_emit_cell(
                                    &mut sub_accum,
                                    &mut build_ctx,
                                    &mut live_ctx,
                                    &mut shard_ctx,
                                    &grid_ctx,
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
                    }
                    #[cfg(not(feature = "timing"))]
                    sub_accum.add_packed_knn(packed_elapsed);
                } else {
                    for (offset, &global) in my_generators[group_start..cursor].iter().enumerate() {
                        let local_idx = group_start + offset;
                        let local = checked_local_id(local_idx, "shard-local generator index")?;
                        let mut shard_ctx = ShardContext {
                            shard: &mut shard,
                            bin,
                            local,
                        };
                        build_and_emit_cell(
                            &mut sub_accum,
                            &mut build_ctx,
                            &mut live_ctx,
                            &mut shard_ctx,
                            &grid_ctx,
                            global,
                            None,
                        )?;
                    }
                }
            }

            Ok((shard, sub_accum))
        })
        .collect();
    let per_bin = per_bin?;

    let mut shards: Vec<ShardState> = Vec::with_capacity(num_bins);
    let mut merged_sub = crate::knn_clipping::timing::CellSubAccum::new();
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

fn build_and_emit_cell<'a, 'b, 'c>(
    cell_sub: &'a mut crate::knn_clipping::timing::CellSubAccum,
    build_ctx: &'a mut CellBuildContext,
    live_ctx: &'a mut LiveDedupCellScratch,
    shard_ctx: &'a mut ShardContext<'b>,
    grid_ctx: &'a GridContext<'c>,
    generator_idx: usize,
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
    live_ctx.seed_neighbors.clear();
    let cell_idx = checked_u32(generator_idx, "generator index")?;
    for check in &incoming_checks {
        let (a, b) = unpack_edge_key(check.key);
        let neighbor_idx = if a == cell_idx { b } else { a } as usize;
        let neighbor_slot = grid_ctx.grid.point_index_to_slot(neighbor_idx);
        live_ctx.seed_neighbors.push(SeedNeighbor {
            neighbor_idx,
            neighbor_slot,
            hp_eps: check.hp_eps,
        });
    }

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
            directed_ctx,
            packed,
            seed_neighbors: &live_ctx.seed_neighbors,
        },
    )
    .map_err(BuildCellsError::CellBuild)?;
    stats.record_into(cell_sub);

    let output_buffer = build_ctx.output_buffer();
    emit_cell_output(
        cell_sub,
        live_ctx,
        shard_ctx,
        grid_ctx.assignment,
        cell_idx,
        cell_start,
        output_buffer,
        incoming_checks,
    )
}

/// Emit one built cell's output into its shard: resolve/record edge checks,
/// dedup vertices by owner bin (deferring off-shard owners), and forward
/// edge checks to later cells. Geometry-free; shared by the spherical and
/// planar drivers.
#[allow(clippy::too_many_arguments)] // internal seam shared by two drivers
pub(crate) fn emit_cell_output(
    cell_sub: &mut crate::knn_clipping::timing::CellSubAccum,
    live_ctx: &mut LiveDedupCellScratch,
    shard_ctx: &mut ShardContext<'_>,
    assignment: &super::binning::BinAssignment,
    cell_idx: u32,
    cell_start: u32,
    output_buffer: &CellOutputBuffer,
    incoming_checks: Vec<EdgeCheck>,
) -> Result<(), BuildCellsError> {
    let mut t_post = crate::knn_clipping::timing::LapTimer::start();
    live_ctx.edge_scratch.collect_and_resolve(
        cell_idx,
        shard_ctx,
        output_buffer,
        assignment,
        incoming_checks,
    );
    let collect_resolve_time = t_post.lap();
    cell_sub.add_edge_collect(collect_resolve_time / 2);
    cell_sub.add_edge_resolve(collect_resolve_time / 2);

    let count = output_buffer.vertices.len();
    let shard = &mut *shard_ctx.shard;
    let local = shard_ctx.local;
    let bin = shard_ctx.bin;

    let cell_count = checked_u8(count, "cell vertex count")?;
    shard.output.set_cell_count(local, cell_count);

    {
        let vertex_indices = &mut live_ctx.edge_scratch.vertex_indices;
        for ((key, pos), vi) in output_buffer
            .vertices
            .iter()
            .copied()
            .zip(vertex_indices.iter_mut())
        {
            #[cfg(feature = "timing")]
            {
                shard.triplet_keys += 1;
            }
            let owner_bin = assignment.generator_bin[key[0] as usize];
            if owner_bin == bin {
                if *vi == INVALID_INDEX {
                    let new_idx = checked_u32(shard.output.vertices.len(), "shard vertex index")?;
                    shard.output.vertices.push(pos);
                    shard.output.vertex_keys.push(key);
                    *vi = new_idx;
                }
                let v_idx = *vi;
                debug_assert!(v_idx != INVALID_INDEX, "missing on-shard vertex index");
                shard.output.cell_indices.push(pack_ref(bin, v_idx));
            } else {
                debug_assert_eq!(*vi, INVALID_INDEX, "received index for off-shard owner");
                let source_slot =
                    checked_u32(shard.output.cell_indices.len(), "deferred source slot")?;
                shard.output.cell_indices.push(DEFERRED);
                shard.output.deferred_slots.push(DeferredSlot {
                    key,
                    pos,
                    source_bin: bin,
                    source_slot,
                });
            }
        }
    }
    cell_sub.add_key_dedup(t_post.lap());

    live_ctx
        .edge_scratch
        .emit(shard, &output_buffer.vertices, cell_start, bin);
    cell_sub.add_edge_emit(t_post.lap());

    debug_assert_eq!(
        shard.output.cell_indices.len() as u32 - cell_start,
        count as u32,
        "cell index stream mismatch"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{checked_local_id, checked_u32, checked_u8, BuildCellsError};

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checked_u32_reports_representation_limit() {
        let err = checked_u32((u32::MAX as usize) + 1, "generator index")
            .expect_err("value above u32::MAX should fail");
        match err {
            BuildCellsError::RepresentationLimit(msg) => {
                assert!(msg.contains("generator index"));
                assert!(msg.contains("u32"));
            }
            _ => panic!("expected representation limit"),
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checked_local_id_reports_representation_limit() {
        let err = checked_local_id((u32::MAX as usize) + 1, "shard-local generator index")
            .expect_err("local id above u32::MAX should fail");
        match err {
            BuildCellsError::RepresentationLimit(msg) => {
                assert!(msg.contains("shard-local generator index"));
                assert!(msg.contains("u32"));
            }
            _ => panic!("expected representation limit"),
        }
    }

    #[test]
    fn checked_u8_reports_representation_limit() {
        let err =
            checked_u8(256, "cell vertex count").expect_err("value above u8::MAX should fail");
        match err {
            BuildCellsError::RepresentationLimit(msg) => {
                assert!(msg.contains("cell vertex count"));
                assert!(msg.contains("u8"));
            }
            _ => panic!("expected representation limit"),
        }
    }
}
