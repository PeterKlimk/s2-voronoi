//! Cell construction for live dedup.

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod process_cell;
use process_cell::process_cell;

use super::binning::assign_bins;
use super::edge_checks::collect_and_resolve_cell_edges;
use super::packed::INVALID_INDEX;
use super::shard::ShardState;
use super::types::{BinId, EdgeCheck, EdgeCheckOverflow, EdgeOverflowLocal, EdgeToLater, LocalId};
use super::ShardedCellsData;
use crate::cube_grid::packed_knn::{PackedKnnCellScratch, PackedKnnCellStatus, PackedKnnTimings};
use crate::cube_grid::CubeMapGrid;
use crate::knn_clipping::cell_builder::{CellOutputBuffer, VertexData};
use crate::knn_clipping::topo2d::Topo2DBuilder;
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

struct AttemptedNeighbors {
    seen_stamp: Vec<u32>,
    stamp: u32,
}

impl AttemptedNeighbors {
    fn new(num_points: usize) -> Self {
        Self {
            seen_stamp: vec![0; num_points],
            stamp: 1,
        }
    }

    fn clear(&mut self) {
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.seen_stamp.fill(0);
            self.stamp = 1;
        }
    }

    fn insert(&mut self, id: usize) -> bool {
        debug_assert!(id < self.seen_stamp.len(), "neighbor id out of bounds");
        if self.seen_stamp[id] == self.stamp {
            return false;
        }
        self.seen_stamp[id] = self.stamp;
        true
    }

    fn mark(&mut self, id: usize) {
        debug_assert!(id < self.seen_stamp.len(), "neighbor id out of bounds");
        self.seen_stamp[id] = self.stamp;
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

struct CellContext {
    builder: Topo2DBuilder,
    scratch: crate::cube_grid::CubeMapGridScratch,
    packed_chunk: Vec<u32>,
    output_buffer: CellOutputBuffer,
    edge_scratch: EdgeScratch,
    attempted_neighbors: AttemptedNeighbors,
}

impl CellContext {
    fn new(grid: &CubeMapGrid) -> Self {
        Self {
            builder: Topo2DBuilder::new(0, Vec3::ZERO),
            scratch: grid.make_scratch(),
            packed_chunk: Vec::with_capacity(
                crate::knn_clipping::PACKED_K0.max(crate::knn_clipping::PACKED_K1),
            ),
            output_buffer: CellOutputBuffer::default(),
            edge_scratch: EdgeScratch::new(),
            attempted_neighbors: AttemptedNeighbors::new(grid.point_indices().len()),
        }
    }
}

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
) -> ShardedCellsData {
    // Legacy config compatibility: no-k fallback ignores this cap.
    let _ = termination.max_k_cap;

    let assignment = assign_bins(points, grid);
    let num_bins = assignment.num_bins;
    // Packed-kNN uses a "big first chunk" (`packed_k0_base`) then fixed-size chunks (`packed_k1`).
    let packed_k0_base = crate::knn_clipping::PACKED_K0.min(points.len().saturating_sub(1));
    let packed_k1 = crate::knn_clipping::PACKED_K1.min(points.len().saturating_sub(1));

    let per_bin: Vec<(ShardState, crate::knn_clipping::timing::CellSubAccum)> =
        maybe_par_into_iter!(0..num_bins)
            .map(|bin_usize| {
                use crate::knn_clipping::timing::CellSubAccum;

                let bin = BinId::from_usize(bin_usize);
                let my_generators = &assignment.bin_generators[bin_usize];
                let mut shard = ShardState::new(my_generators.len());

                let mut sub_accum = CellSubAccum::new();
                let mut ctx = CellContext::new(grid);
                let vertex_capacity = my_generators.len().saturating_mul(6);
                shard.output.vertices.reserve(vertex_capacity);
                shard.output.vertex_keys.reserve(vertex_capacity);
                shard
                    .output
                    .cell_indices
                    .reserve(my_generators.len().saturating_mul(6));
                // Conservative estimate for off-shard vertices
                shard.output.deferred.reserve(my_generators.len());
                shard
                    .dedup
                    .support_data
                    .reserve(my_generators.len().saturating_mul(2));

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

                    if packed_k0_base > 0 {
                        let queries = &packed_queries_all[group_start..cursor];
                        let query_locals = &packed_query_locals_all[group_start..cursor];

                        #[cfg(not(feature = "timing"))]
                        let t_packed = crate::knn_clipping::timing::Timer::start();
                        let status = packed_scratch.prepare_group_directed(
                            grid,
                            cell as usize,
                            queries,
                            query_locals,
                            bin.as_u8(),
                            &assignment.slot_gen_map,
                            assignment.local_shift,
                            assignment.local_mask,
                            &mut packed_timings,
                        );
                        #[cfg(not(feature = "timing"))]
                        let packed_elapsed = t_packed.elapsed();

                        match status {
                            PackedKnnCellStatus::Ok => {
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
                                    process_cell(
                                        &mut sub_accum,
                                        &mut ctx,
                                        &mut shard_ctx,
                                        &grid_ctx,
                                        termination,
                                        global,
                                        Some((
                                            &mut packed_scratch,
                                            &mut packed_timings,
                                            offset,
                                            packed_k0_base,
                                            packed_k1,
                                        )),
                                    );
                                }
                            }
                            PackedKnnCellStatus::SlowPath => {
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
                                    process_cell(
                                        &mut sub_accum,
                                        &mut ctx,
                                        &mut shard_ctx,
                                        &grid_ctx,
                                        termination,
                                        global,
                                        None,
                                    );
                                }
                            }
                        }

                        #[cfg(feature = "timing")]
                        {
                            sub_accum.add_packed_knn(packed_timings.total());
                        }
                        #[cfg(not(feature = "timing"))]
                        sub_accum.add_packed_knn(packed_elapsed);
                    } else {
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
                            process_cell(
                                &mut sub_accum,
                                &mut ctx,
                                &mut shard_ctx,
                                &grid_ctx,
                                termination,
                                global,
                                None,
                            );
                        }
                    }
                }

                (shard, sub_accum)
            })
            .collect();

    let mut shards: Vec<ShardState> = Vec::with_capacity(num_bins);
    let mut merged_sub = crate::knn_clipping::timing::CellSubAccum::new();
    for (shard, sub) in per_bin {
        merged_sub.merge(&sub);
        shards.push(shard);
    }

    ShardedCellsData {
        assignment,
        shards,
        cell_sub: merged_sub,
    }
}
