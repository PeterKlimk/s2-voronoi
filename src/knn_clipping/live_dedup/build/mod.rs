//! Cell construction for live dedup.

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
    DirectedCellGroup, PackedKnnCellScratch, PackedKnnTimings, PreparedPackedGroupStatus,
};
use crate::cube_grid::{CubeMapGrid, PackedQuery};
use crate::knn_clipping::cell_build::{
    build_cell_into, CellBuildContext, CellBuildError, CellBuildRequest, CellOutputBuffer,
    SeedNeighbor, VertexData,
};
use crate::knn_clipping::TerminationConfig;

struct EdgeScratch {
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

struct LiveDedupCellScratch {
    edge_scratch: EdgeScratch,
    seed_neighbors: Vec<SeedNeighbor>,
}

impl LiveDedupCellScratch {
    fn new(num_points: usize) -> Self {
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

pub(super) struct ShardContext<'a> {
    pub(super) shard: &'a mut ShardState,
    pub(super) bin: BinId,
    pub(super) local: LocalId,
}

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
) -> Result<ShardedCellsData, BuildCellsError> {
    let policy = termination.knn_policy(points.len());
    // Legacy config compatibility: no-k fallback ignores this cap.
    let _ = policy.termination().max_k_cap();

    let assignment = assign_bins(points, grid).map_err(BuildCellsError::PackedLayoutCapacity)?;
    let num_bins = assignment.num_bins;
    let packed_policy = policy.packed();
    let termination_policy = policy.termination();

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
            let packed_query_locals_all: Vec<u32> = (0..my_generators.len())
                .map(|local_idx| u32::try_from(local_idx).expect("local id must fit in u32"))
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
                    let query_locals = &packed_query_locals_all[group_start..cursor];
                    let group = DirectedCellGroup::new(
                        cell as usize,
                        bin.as_u8(),
                        queries,
                        query_locals,
                        &assignment.slot_gen_map,
                        assignment.local_shift,
                        assignment.local_mask,
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
                                let local = LocalId::from_usize(local_idx);
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
                                    termination_policy,
                                    global,
                                    Some(PackedQuery::new(
                                        &mut prepared,
                                        &mut packed_timings,
                                        offset,
                                        packed_policy,
                                    )),
                                )
                                .map_err(BuildCellsError::CellBuild)?;
                            }
                        }
                        PreparedPackedGroupStatus::SlowPath => {
                            for (offset, &global) in
                                my_generators[group_start..cursor].iter().enumerate()
                            {
                                let local_idx = group_start + offset;
                                let local = LocalId::from_usize(local_idx);
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
                                    termination_policy,
                                    global,
                                    None,
                                )
                                .map_err(BuildCellsError::CellBuild)?;
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
                        let local = LocalId::from_usize(local_idx);
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
                            termination_policy,
                            global,
                            None,
                        )
                        .map_err(BuildCellsError::CellBuild)?;
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
    termination: crate::policy::TerminationPolicy,
    generator_idx: usize,
    packed: Option<PackedQuery<'_, '_, 'c>>,
) -> Result<(), CellBuildError> {
    let cell_start = shard_ctx.shard.output.cell_indices.len() as u32;
    shard_ctx
        .shard
        .output
        .set_cell_start(shard_ctx.local, cell_start);

    let incoming_checks = shard_ctx.shard.dedup.take_edge_checks(shard_ctx.local);
    live_ctx.seed_neighbors.clear();
    let cell_idx = u32::try_from(generator_idx).expect("cell index must fit in u32");
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

    let directed_ctx = crate::cube_grid::DirectedEligibility::new(
        shard_ctx.bin.as_u8(),
        shard_ctx.local.as_u32(),
        &grid_ctx.assignment.slot_gen_map,
        grid_ctx.assignment.local_shift,
        grid_ctx.assignment.local_mask,
    );

    let stats = build_cell_into(
        build_ctx,
        CellBuildRequest {
            points: grid_ctx.points,
            grid: grid_ctx.grid,
            generator_idx,
            directed_ctx,
            termination,
            packed,
            seed_neighbors: &live_ctx.seed_neighbors,
        },
    )?;
    stats.record_into(cell_sub);

    let mut t_post = crate::knn_clipping::timing::LapTimer::start();
    let output_buffer = build_ctx.output_buffer();
    live_ctx.edge_scratch.collect_and_resolve(
        cell_idx,
        shard_ctx,
        output_buffer,
        grid_ctx.assignment,
        incoming_checks,
    );
    let collect_resolve_time = t_post.lap();
    cell_sub.add_edge_collect(collect_resolve_time / 2);
    cell_sub.add_edge_resolve(collect_resolve_time / 2);

    let count = output_buffer.vertices.len();
    let shard = &mut *shard_ctx.shard;
    let local = shard_ctx.local;
    let bin = shard_ctx.bin;

    shard.output.set_cell_count(
        local,
        u8::try_from(count).expect("cell vertex count exceeds u8 capacity"),
    );

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
            let owner_bin = grid_ctx.assignment.generator_bin[key[0] as usize];
            if owner_bin == bin {
                if *vi == INVALID_INDEX {
                    let new_idx = shard.output.vertices.len() as u32;
                    shard.output.vertices.push(pos);
                    shard.output.vertex_keys.push(key);
                    *vi = new_idx;
                }
                let v_idx = *vi;
                debug_assert!(v_idx != INVALID_INDEX, "missing on-shard vertex index");
                shard.output.cell_indices.push(pack_ref(bin, v_idx));
            } else {
                debug_assert_eq!(*vi, INVALID_INDEX, "received index for off-shard owner");
                let source_slot = shard.output.cell_indices.len() as u32;
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
