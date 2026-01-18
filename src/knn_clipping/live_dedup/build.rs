//! Cell construction for live dedup.

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::binning::assign_bins;
use super::edge_checks::collect_and_resolve_cell_edges;
use super::packed::{pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::ShardState;
use super::types::{
    BinId, DeferredSlot, EdgeCheck, EdgeCheckOverflow, EdgeOverflowLocal, EdgeToLater, LocalId,
    PackedSeed,
};
use super::ShardedCellsData;
use crate::cube_grid::packed_knn::{
    packed_knn_cell_stream, PackedKnnCellScratch, PackedKnnCellStatus, PackedKnnTimings,
};
use crate::knn_clipping::cell_builder::VertexData;
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

    fn collect_and_resolve(
        &mut self,
        cell_idx: u32,
        bin: BinId,
        local: LocalId,
        cell_vertices: &[VertexData],
        edge_neighbor_slots: &[u32],
        edge_neighbor_globals: &[u32],
        assignment: &super::binning::BinAssignment,
        shard: &mut ShardState,
    ) {
        self.vertex_indices.clear();
        self.vertex_indices
            .resize(cell_vertices.len(), INVALID_INDEX);
        collect_and_resolve_cell_edges(
            cell_idx,
            bin,
            local,
            cell_vertices,
            edge_neighbor_slots,
            edge_neighbor_globals,
            assignment,
            shard,
            &mut self.vertex_indices,
            &mut self.edges_to_later,
            &mut self.edges_overflow,
        );
    }

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

struct CellContext {
    builder: Topo2DBuilder,
    scratch: crate::cube_grid::CubeMapGridScratch,
    neighbor_slots: Vec<u32>,
    cell_vertices: Vec<VertexData>,
    edge_neighbor_slots: Vec<u32>,
    edge_neighbor_globals: Vec<u32>,
    edge_scratch: EdgeScratch,
}

impl CellContext {
    fn new(knn: &crate::knn_clipping::CubeMapGridKnn) -> Self {
        Self {
            builder: Topo2DBuilder::new(0, Vec3::ZERO),
            scratch: knn.make_scratch(),
            neighbor_slots: Vec::with_capacity(crate::knn_clipping::KNN_RESTART_MAX),
            cell_vertices: Vec::new(),
            edge_neighbor_slots: Vec::new(),
            edge_neighbor_globals: Vec::new(),
            edge_scratch: EdgeScratch::new(),
        }
    }
}

fn process_cell(
    cell_sub: &mut crate::knn_clipping::timing::CellSubAccum,
    ctx: &mut CellContext,
    shard: &mut ShardState,
    points: &[Vec3],
    knn: &crate::knn_clipping::CubeMapGridKnn,
    assignment: &super::binning::BinAssignment,
    termination: TerminationConfig,
    termination_max_k_cap: Option<usize>,
    bin: BinId,
    i: usize,
    local: LocalId,
    packed: Option<PackedSeed<'_>>,
) {
    let builder = &mut ctx.builder;
    let scratch = &mut ctx.scratch;
    let neighbor_slots = &mut ctx.neighbor_slots;
    let cell_vertices = &mut ctx.cell_vertices;
    let edge_neighbor_slots = &mut ctx.edge_neighbor_slots;
    let edge_neighbor_globals = &mut ctx.edge_neighbor_globals;
    let edge_scratch = &mut ctx.edge_scratch;

    builder.reset(i, points[i]);
    neighbor_slots.clear();

    // === Phase 1: Initialization ===

    let cell_start = shard.output.cell_indices.len() as u32;
    shard.output.set_cell_start(local, cell_start);

    let mut cell_neighbors_processed = 0usize;
    let mut terminated = false;
    let mut knn_exhausted = false;
    let mut full_scan_done = false;
    let mut did_full_scan_fallback = false;
    let mut used_knn = false;

    let mut worst_cos = 1.0f32;
    let max_neighbors = points.len().saturating_sub(1);
    let mut max_k_requested = 0usize;
    let mut knn_stage =
        crate::knn_clipping::timing::KnnCellStage::Resume(crate::knn_clipping::KNN_RESUME_K);
    let mut reached_schedule_max_k = false;

    let mut did_packed = false;
    let mut packed_count = 0usize;
    let mut packed_security = 0.0f32;
    let mut packed_k_local = 0usize;

    // === Phase 2: Packed kNN Seeds ===
    if let Some(seed) = packed {
        did_packed = true;
        packed_count = seed.count;
        packed_security = seed.security;
        packed_k_local = seed.k;

        if packed_count > 0 {
            let t_clip = crate::knn_clipping::timing::Timer::start();
            let grid = knn.grid();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in seed.neighbors.iter().enumerate() {
                // Convert slot to global index
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if neighbor_idx == i {
                    continue;
                }
                #[cfg(debug_assertions)]
                debug_assert!(
                    !builder.has_neighbor(neighbor_idx),
                    "packed kNN returned duplicate neighbor {} for cell {}",
                    neighbor_idx,
                    i
                );
                let neighbor = points[neighbor_idx];
                if builder
                    .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
                    .is_err()
                {
                    break;
                }
                cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                worst_cos = worst_cos.min(dot);

                if builder.is_bounded() {
                    if termination.should_check(cell_neighbors_processed)
                        && builder.can_terminate({
                            let mut bound = worst_cos;
                            for &next_slot in seed.neighbors.iter().skip(pos + 1) {
                                let next = point_indices[next_slot as usize] as usize;
                                if next != i {
                                    bound = points[i].dot(points[next]);
                                    break;
                                }
                            }
                            bound
                        })
                    {
                        terminated = true;
                        break;
                    }
                }
            }
            cell_sub.add_clip(t_clip.elapsed());
        }

        if !terminated && builder.is_bounded() {
            let bound = if packed_count == packed_k_local {
                worst_cos
            } else {
                packed_security
            };
            if builder.can_terminate(bound) {
                terminated = true;
            }
        }
    }

    // === Phase 3: Resumable kNN Scan ===
    let resume_k = crate::knn_clipping::KNN_RESUME_K.min(max_neighbors);
    if !terminated && !knn_exhausted && !builder.is_failed() && resume_k > 0 {
        used_knn = true;
        max_k_requested = resume_k;
        neighbor_slots.clear();
        let t_knn = crate::knn_clipping::timing::Timer::start();
        let status =
            knn.knn_resumable_slots_into(points[i], i, resume_k, resume_k, scratch, neighbor_slots);
        cell_sub.add_knn(t_knn.elapsed());

        // Track which resume stage we're at
        knn_stage = crate::knn_clipping::timing::KnnCellStage::Resume(resume_k);

        let t_clip = crate::knn_clipping::timing::Timer::start();
        let grid = knn.grid();
        let point_indices = grid.point_indices();
        for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
            let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
            if did_packed && builder.has_neighbor(neighbor_idx) {
                continue;
            }
            #[cfg(debug_assertions)]
            debug_assert!(
                !builder.has_neighbor(neighbor_idx),
                "kNN resume returned duplicate neighbor {} for cell {}",
                neighbor_idx,
                i
            );
            let neighbor = points[neighbor_idx];
            if builder
                .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
                .is_err()
            {
                break;
            }
            cell_neighbors_processed += 1;
            let dot = points[i].dot(neighbor);
            worst_cos = worst_cos.min(dot);

            if builder.is_bounded() {
                if termination.should_check(cell_neighbors_processed)
                    && builder.can_terminate({
                        let mut bound = worst_cos;
                        for &next_slot in neighbor_slots.iter().skip(pos + 1) {
                            let next = point_indices[next_slot as usize] as usize;
                            if next != i {
                                bound = points[i].dot(points[next]);
                                break;
                            }
                        }
                        bound
                    })
                {
                    terminated = true;
                    break;
                }
            }
        }
        cell_sub.add_clip(t_clip.elapsed());

        if terminated {
            knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
        } else if status == crate::cube_grid::KnnStatus::Exhausted {
            knn_exhausted = true;
        }
    }

    if !terminated && !knn_exhausted && !builder.is_failed() {
        for &k_stage in crate::knn_clipping::KNN_RESTART_KS.iter() {
            let k = k_stage.min(max_neighbors);
            if k == 0 || k <= max_k_requested {
                continue;
            }
            used_knn = true;
            max_k_requested = k;
            neighbor_slots.clear();

            let t_knn = crate::knn_clipping::timing::Timer::start();
            let status = knn.knn_resumable_slots_into(points[i], i, k, k, scratch, neighbor_slots);
            cell_sub.add_knn(t_knn.elapsed());

            // Track which restart stage we're at
            knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(k_stage);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let grid = knn.grid();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if builder.has_neighbor(neighbor_idx) {
                    continue;
                }
                let neighbor = points[neighbor_idx];
                if builder
                    .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
                    .is_err()
                {
                    break;
                }
                cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                worst_cos = worst_cos.min(dot);

                if builder.is_bounded() {
                    if termination.should_check(cell_neighbors_processed)
                        && builder.can_terminate({
                            let mut bound = worst_cos;
                            for &next_slot in neighbor_slots.iter().skip(pos + 1) {
                                let next = point_indices[next_slot as usize] as usize;
                                if next != i {
                                    bound = points[i].dot(points[next]);
                                    break;
                                }
                            }
                            bound
                        })
                    {
                        terminated = true;
                        break;
                    }
                }
            }
            cell_sub.add_clip(t_clip.elapsed());

            if terminated {
                break;
            }

            knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
            if knn_exhausted {
                break;
            }
        }
    }

    let max_knn_target = crate::knn_clipping::KNN_RESTART_MAX.min(max_neighbors);
    if max_k_requested >= max_knn_target {
        reached_schedule_max_k = true;
    }

    // Final termination check at the end
    if !terminated && builder.is_bounded() {
        let bound = if used_knn {
            worst_cos
        } else if did_packed && packed_count < packed_k_local {
            packed_security
        } else {
            worst_cos
        };
        if builder.can_terminate(bound) {
            terminated = true;
        }
    }

    // If termination is enabled and the cell is bounded but still not proven,
    // keep requesting more neighbors until `can_terminate()` succeeds.
    //
    // Important: without this, the algorithm can accept "bounded but unproven" cells,
    // which causes asymmetric edges at high densities.
    if !terminated && reached_schedule_max_k && !builder.is_failed() && builder.is_bounded() {
        let mut k = max_k_requested
            .max(crate::knn_clipping::KNN_RESTART_MAX)
            .min(max_neighbors);
        let cap = termination_max_k_cap
            .unwrap_or(max_neighbors)
            .min(max_neighbors)
            .max(k);

        const K_STEP_MIN: usize = 32;

        while !terminated && !builder.is_failed() && k < cap {
            let next_k = (k.saturating_mul(2)).max(k + K_STEP_MIN).min(cap);
            if next_k <= k {
                break;
            }

            used_knn = true;
            neighbor_slots.clear();

            let t_knn = crate::knn_clipping::timing::Timer::start();
            let status =
                knn.knn_resumable_slots_into(points[i], i, next_k, next_k, scratch, neighbor_slots);
            cell_sub.add_knn(t_knn.elapsed());
            knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(next_k);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let grid = knn.grid();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if builder.has_neighbor(neighbor_idx) {
                    continue;
                }
                let neighbor = points[neighbor_idx];
                if builder
                    .clip_with_slot(neighbor_idx, neighbor_slot, neighbor)
                    .is_err()
                {
                    break;
                }
                cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                worst_cos = worst_cos.min(dot);

                if termination.should_check(cell_neighbors_processed)
                    && builder.can_terminate({
                        let mut bound = worst_cos;
                        for &next_slot in neighbor_slots.iter().skip(pos + 1) {
                            let next = point_indices[next_slot as usize] as usize;
                            if next != i {
                                bound = points[i].dot(points[next]);
                                break;
                            }
                        }
                        bound
                    })
                {
                    terminated = true;
                    break;
                }
            }
            cell_sub.add_clip(t_clip.elapsed());

            knn_exhausted = status == crate::cube_grid::KnnStatus::Exhausted;
            k = next_k;

            // Conservative bound on unseen: the k-th neighbor dot.
            // (kNN results are sorted by distance; for unit vectors, distance order
            // matches dot order.)
            let kth_dot = neighbor_slots
                .last()
                .map(|&slot| {
                    let j = point_indices[slot as usize] as usize;
                    points[i].dot(points[j])
                })
                .unwrap_or(worst_cos);
            if builder.can_terminate(kth_dot) {
                terminated = true;
            }

            // If we've effectively clipped against all possible neighbors, there's
            // nothing left unseen.
            if k >= max_neighbors {
                terminated = true;
            }

            // If we brute-forced and still can't terminate, we'll continue growing k
            // until cap (possibly all points).
        }
    }

    // === Phase 4: Full Scan Fallback ===
    // Full scan fallback if cell is not bounded after kNN
    // With Topo2DBuilder, we just need to keep adding neighbors until bounded
    if !builder.is_bounded() && !builder.is_failed() {
        did_full_scan_fallback = true;
        let already_clipped: rustc_hash::FxHashSet<usize> =
            builder.neighbor_indices_iter().collect();
        let gen = points[i];
        let mut sorted_indices: Vec<usize> = (0..points.len())
            .filter(|&j| j != i && !already_clipped.contains(&j))
            .collect();
        sorted_indices.sort_by(|&a, &b| {
            let da = gen.dot(points[a]);
            let db = gen.dot(points[b]);
            db.partial_cmp(&da).unwrap()
        });
        for p_idx in sorted_indices {
            if builder.clip(p_idx, points[p_idx]).is_err() {
                break;
            }
            cell_neighbors_processed += 1;
            // Check if now bounded
            if builder.is_bounded() {
                break;
            }
        }
        full_scan_done = true;
    }

    // If still not bounded or failed, panic with diagnostics
    if !builder.is_bounded() || builder.is_failed() {
        let (active, total) = builder.count_active_planes();
        let gen = points[i];
        let neighbor_indices: Vec<usize> = builder.neighbor_indices_iter().collect();

        panic!(
            "Cell {} construction failed: bounded={}, failure={:?}, \
             planes={}, active={}, vertices={}, \
             did_packed={}, did_knn={}, did_full_scan={}\n\
             Generator pos: {:?}\n\
             First 10 neighbor indices: {:?}",
            i,
            builder.is_bounded(),
            builder.failure(),
            total,
            active,
            builder.vertex_count(),
            did_packed,
            used_knn,
            full_scan_done,
            gen,
            &neighbor_indices[..neighbor_indices.len().min(10)],
        );
    }

    let knn_stage = if did_full_scan_fallback {
        crate::knn_clipping::timing::KnnCellStage::FullScanFallback
    } else {
        knn_stage
    };
    cell_sub.add_cell_stage(knn_stage, knn_exhausted, cell_neighbors_processed);

    // === Phase 5: Output Extraction ===
    let t_cert = crate::knn_clipping::timing::Timer::start();
    builder
        .to_vertex_data_full(cell_vertices, edge_neighbor_globals, edge_neighbor_slots)
        .expect("to_vertex_data_full failed after bounded check");
    cell_sub.add_cert(t_cert.elapsed());

    let cell_idx = i as u32;
    let t_edge_collect = crate::knn_clipping::timing::Timer::start();
    edge_scratch.collect_and_resolve(
        cell_idx,
        bin,
        local,
        cell_vertices,
        edge_neighbor_slots,
        edge_neighbor_globals,
        assignment,
        shard,
    );
    let collect_resolve_time = t_edge_collect.elapsed();
    // Split time between collect and resolve for backward-compatible timing
    cell_sub.add_edge_collect(collect_resolve_time / 2);
    cell_sub.add_edge_resolve(collect_resolve_time / 2);

    let count = cell_vertices.len();
    shard.output.set_cell_count(
        local,
        u8::try_from(count).expect("cell vertex count exceeds u8 capacity"),
    );

    let t_keys = crate::knn_clipping::timing::Timer::start();
    {
        let vertex_indices = &mut edge_scratch.vertex_indices;
        for ((key, pos), vi) in cell_vertices.iter().copied().zip(vertex_indices.iter_mut()) {
            #[cfg(feature = "timing")]
            {
                shard.triplet_keys += 1;
            }
            let owner_bin = assignment.generator_bin[key[0] as usize];
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
                shard.output.deferred.push(DeferredSlot {
                    key,
                    pos,
                    source_bin: bin,
                    source_slot,
                });
            }
        }
    }
    cell_sub.add_key_dedup(t_keys.elapsed());

    let t_edge_emit = crate::knn_clipping::timing::Timer::start();
    edge_scratch.emit(shard, cell_vertices, cell_start, bin);
    cell_sub.add_edge_emit(t_edge_emit.elapsed());

    debug_assert_eq!(
        shard.output.cell_indices.len() as u32 - cell_start,
        count as u32,
        "cell index stream mismatch"
    );
}

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    knn: &crate::knn_clipping::CubeMapGridKnn,
    termination: TerminationConfig,
) -> ShardedCellsData {
    // If termination is enabled but not proven after the kNN schedule, we keep requesting
    // more neighbors until the termination check succeeds.
    //
    // This makes correctness independent of any fixed k cap.
    let termination_max_k_cap = termination.max_k_cap;

    let assignment = assign_bins(points, knn.grid());
    let num_bins = assignment.num_bins;
    let packed_k = crate::knn_clipping::KNN_RESUME_K.min(points.len().saturating_sub(1));

    let per_bin: Vec<(ShardState, crate::knn_clipping::timing::CellSubAccum)> =
        maybe_par_into_iter!(0..num_bins)
            .map(|bin_usize| {
                use crate::knn_clipping::timing::CellSubAccum;

                let bin = BinId::from_usize(bin_usize);
                let my_generators = &assignment.bin_generators[bin_usize];
                let mut shard = ShardState::new(my_generators.len());

                let mut sub_accum = CellSubAccum::new();
                let mut ctx = CellContext::new(knn);
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

                let grid = knn.grid();
                let mut packed_scratch = PackedKnnCellScratch::new();
                let mut packed_timings = PackedKnnTimings::default();

                let packed_queries_all: Vec<u32> = my_generators
                    .iter()
                    .map(|&i| u32::try_from(i).expect("point index must fit in u32"))
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

                    if packed_k > 0 {
                        let queries = &packed_queries_all[group_start..cursor];

                        // Grouped queries by cell to improve cache locality during kNN.
                        // NOTE: `packed_knn_cell_stream` invokes the callback per query.
                        // The callback builds the Voronoi cell and is separately timed (clipping,
                        // certification, key_dedup, and any fallback knn work). If we time the whole
                        // call naively, we'd double-count that work under `packed_knn`.
                        let t_packed = crate::knn_clipping::timing::Timer::start();
                        let status = packed_knn_cell_stream(
                            grid,
                            points,
                            cell as usize,
                            queries,
                            packed_k,
                            &mut packed_scratch,
                            &mut packed_timings,
                            |qi, query_idx, neighbors, count, security| {
                                let local = LocalId::from_usize(group_start + qi);
                                let seed = PackedSeed {
                                    neighbors,
                                    count,
                                    security,
                                    k: packed_k,
                                };
                                process_cell(
                                    &mut sub_accum,
                                    &mut ctx,
                                    &mut shard,
                                    points,
                                    knn,
                                    &assignment,
                                    termination,
                                    termination_max_k_cap,
                                    bin,
                                    query_idx as usize,
                                    local,
                                    Some(seed),
                                );
                            },
                        );
                        let packed_elapsed = t_packed.elapsed();

                        if status == PackedKnnCellStatus::SlowPath {
                            for local_idx in group_start..cursor {
                                let global = my_generators[local_idx];
                                let local = LocalId::from_usize(local_idx);
                                process_cell(
                                    &mut sub_accum,
                                    &mut ctx,
                                    &mut shard,
                                    points,
                                    knn,
                                    &assignment,
                                    termination,
                                    termination_max_k_cap,
                                    bin,
                                    global,
                                    local,
                                    None,
                                );
                            }
                        }

                        // Attribute only the packed k-NN overhead to `packed_knn`, excluding the work
                        // done inside `process_cell` (which has its own sub-phase timers).
                        #[cfg(feature = "timing")]
                        {
                            let overhead_total =
                                packed_elapsed.saturating_sub(packed_timings.callback);
                            sub_accum.add_packed_knn(overhead_total);

                            sub_accum.add_packed_knn_setup(packed_timings.setup);
                            sub_accum.add_packed_knn_query_cache(packed_timings.query_cache);
                            sub_accum.add_packed_knn_security_thresholds(
                                packed_timings.security_thresholds,
                            );
                            sub_accum.add_packed_knn_center_pass(packed_timings.center_pass);
                            sub_accum
                                .add_packed_knn_ring_thresholds(packed_timings.ring_thresholds);
                            sub_accum.add_packed_knn_ring_pass(packed_timings.ring_pass);
                            sub_accum.add_packed_knn_ring_fallback(packed_timings.ring_fallback);
                            sub_accum.add_packed_knn_select_sort(packed_timings.select_sort);

                            let measured = packed_timings.total();
                            sub_accum.add_packed_knn_other(overhead_total.saturating_sub(measured));
                        }
                        #[cfg(not(feature = "timing"))]
                        sub_accum.add_packed_knn(packed_elapsed);
                    } else {
                        for local_idx in group_start..cursor {
                            let global = my_generators[local_idx];
                            let local = LocalId::from_usize(local_idx);
                            process_cell(
                                &mut sub_accum,
                                &mut ctx,
                                &mut shard,
                                points,
                                knn,
                                &assignment,
                                termination,
                                termination_max_k_cap,
                                bin,
                                global,
                                local,
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
