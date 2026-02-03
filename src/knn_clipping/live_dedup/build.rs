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
use super::ShardedCellsData;
use crate::cube_grid::packed_knn::{
    PackedKnnCellScratch, PackedKnnCellStatus, PackedKnnTimings, PackedStage,
};
use crate::cube_grid::CubeMapGrid;
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
        edge_neighbor_eps: &[f32],
        assignment: &super::binning::BinAssignment,
        shard: &mut ShardState,
        incoming_checks: Vec<EdgeCheck>,
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
            edge_neighbor_eps,
            assignment,
            shard,
            incoming_checks,
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

#[cfg(debug_assertions)]
struct AttemptedNeighbors {
    set: rustc_hash::FxHashSet<usize>,
}

#[cfg(debug_assertions)]
impl AttemptedNeighbors {
    fn new() -> Self {
        Self {
            set: rustc_hash::FxHashSet::default(),
        }
    }

    fn clear(&mut self) {
        self.set.clear();
    }

    fn insert(&mut self, idx: usize) -> bool {
        self.set.insert(idx)
    }
}

#[cfg(not(debug_assertions))]
struct AttemptedNeighbors;

#[cfg(not(debug_assertions))]
impl AttemptedNeighbors {
    fn new() -> Self {
        Self
    }

    fn clear(&mut self) {}

    fn insert(&mut self, _idx: usize) -> bool {
        true
    }
}

struct CellContext {
    builder: Topo2DBuilder,
    scratch: crate::cube_grid::CubeMapGridScratch,
    neighbor_slots: Vec<u32>,
    packed_chunk: Vec<u32>,
    cell_vertices: Vec<VertexData>,
    edge_neighbor_slots: Vec<u32>,
    edge_neighbor_globals: Vec<u32>,
    edge_neighbor_eps: Vec<f32>,
    edge_scratch: EdgeScratch,
    attempted_neighbors: AttemptedNeighbors,
}

impl CellContext {
    fn new(grid: &CubeMapGrid) -> Self {
        Self {
            builder: Topo2DBuilder::new(0, Vec3::ZERO),
            scratch: grid.make_scratch(),
            neighbor_slots: Vec::with_capacity(crate::knn_clipping::KNN_RESTART_MAX),
            packed_chunk: Vec::with_capacity(crate::knn_clipping::KNN_RESTART_MAX),
            cell_vertices: Vec::new(),
            edge_neighbor_slots: Vec::new(),
            edge_neighbor_globals: Vec::new(),
            edge_neighbor_eps: Vec::new(),
            edge_scratch: EdgeScratch::new(),
            attempted_neighbors: AttemptedNeighbors::new(),
        }
    }
}

fn process_cell(
    cell_sub: &mut crate::knn_clipping::timing::CellSubAccum,
    ctx: &mut CellContext,
    shard: &mut ShardState,
    points: &[Vec3],
    grid: &CubeMapGrid,
    assignment: &super::binning::BinAssignment,
    termination: TerminationConfig,
    termination_max_k_cap: Option<usize>,
    use_dedicated_cert_full: bool,
    bin: BinId,
    i: usize,
    local: LocalId,
    packed: Option<(
        &mut PackedKnnCellScratch,
        &mut PackedKnnTimings,
        usize,
        usize,
        usize,
    )>,
) {
    let builder = &mut ctx.builder;
    let scratch = &mut ctx.scratch;
    let neighbor_slots = &mut ctx.neighbor_slots;
    let packed_chunk = &mut ctx.packed_chunk;
    let cell_vertices = &mut ctx.cell_vertices;
    let edge_neighbor_slots = &mut ctx.edge_neighbor_slots;
    let edge_neighbor_globals = &mut ctx.edge_neighbor_globals;
    let edge_neighbor_eps = &mut ctx.edge_neighbor_eps;
    let edge_scratch = &mut ctx.edge_scratch;
    let attempted_neighbors = &mut ctx.attempted_neighbors;

    builder.reset(i, points[i]);
    attempted_neighbors.clear();
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
    let mut knn_stage = crate::knn_clipping::timing::KnnCellStage::Resume(0);
    let mut reached_schedule_max_k = false;

    let mut did_packed = false;
    let mut packed_security = 0.0f32;
    let mut packed_safe_exhausted = false;
    let mut packed_tail_used = false;

    // Take incoming edge checks once. We'll use them both for geometry seeding and for
    // resolving edges to earlier neighbors later in the pipeline.
    let incoming_checks = shard.dedup.take_edge_checks(local);
    let incoming_edgechecks = incoming_checks.len();

    // === Phase 2: Edgecheck Geometry Seeds ===
    //
    // Incoming edge checks encode (symmetric) edges where the earlier-local side already
    // discovered adjacency. Treat them as mandatory earlier-neighbor candidates for clipping.
    //
    // Note: we do not attempt termination checks here; termination bounds come from kNN stages.
    if !incoming_checks.is_empty() {
        let cell_idx = u32::try_from(i).expect("cell index must fit in u32");
        let t_clip = crate::knn_clipping::timing::Timer::start();
        for check in &incoming_checks {
            let (a, b) = unpack_edge_key(check.key);

            debug_assert!(
                a == cell_idx || b == cell_idx,
                "incoming edge check does not reference current cell"
            );

            debug_assert!(a != b, "incoming edge check has identical endpoints");

            let neighbor_idx = if a == cell_idx { b } else { a } as usize;

            if !attempted_neighbors.insert(neighbor_idx) {
                continue;
            }

            debug_assert_eq!(
                assignment.generator_bin[neighbor_idx], bin,
                "incoming edge check neighbor must be in same bin"
            );

            debug_assert!(
                assignment.global_to_local[neighbor_idx].as_u32() < local.as_u32(),
                "incoming edge check neighbor must be earlier-local"
            );

            // Important: do not update `worst_cos` here. These neighbors are not ordered
            // (and may be arbitrarily far), so using their dot products as termination
            // bounds would be unsound.
            let neighbor_slot = grid.point_index_to_slot(neighbor_idx);
            let neighbor = points[neighbor_idx];
            if builder
                .clip_with_slot_edgecheck(neighbor_idx, neighbor_slot, neighbor, check.hp_eps)
                .is_err()
            {
                break;
            }
            cell_neighbors_processed += 1;
        }
        cell_sub.add_clip(t_clip.elapsed());
    }
    let edgecheck_seed_clips = cell_neighbors_processed;

    // === Phase 3: Packed kNN Seeds ===
    if let Some((packed_scratch, packed_timings, qi, packed_k0_base, packed_k1)) = packed {
        did_packed = true;
        packed_security = packed_scratch.security(qi);

        let max_k0 = packed_k0_base.min(max_neighbors).max(1);
        let k1 = packed_k1.min(max_neighbors).max(1);
        let seed_clips = cell_neighbors_processed;
        let mut k_cur = max_k0.saturating_sub(seed_clips).max(k1).min(max_k0);

        let point_indices = grid.point_indices();
        let mut stage = PackedStage::Chunk0;
        #[cfg(feature = "timing")]
        let mut recorded_chunk0_overfetch = false;
        #[cfg(feature = "timing")]
        let mut recorded_chunk0_term_excess = false;
        #[cfg(feature = "timing")]
        let mut chunk0_first_emitted = 0usize;

        loop {
            #[cfg(feature = "timing")]
            if !recorded_chunk0_overfetch && stage == PackedStage::Chunk0 {
                let chunk0_len = packed_scratch.chunk0_len(qi);
                cell_sub.add_packed_knn_chunk0_overfetch_sample(
                    incoming_edgechecks,
                    k_cur,
                    chunk0_len,
                );
                recorded_chunk0_overfetch = true;
            }

            packed_chunk.clear();
            packed_chunk.resize(k_cur, u32::MAX);

            let Some(chunk) =
                packed_scratch.next_chunk(qi, stage, k_cur, packed_chunk, packed_timings)
            else {
                if stage == PackedStage::Chunk0 && packed_scratch.tail_possible(qi) {
                    packed_scratch.ensure_tail_directed_for(
                        qi,
                        grid,
                        &assignment.slot_gen_map,
                        assignment.local_shift,
                        assignment.local_mask,
                        packed_timings,
                    );
                    stage = PackedStage::Tail;
                    packed_tail_used = true;
                    k_cur = k1;
                    continue;
                }
                packed_safe_exhausted = true;
                break;
            };

            #[cfg(feature = "timing")]
            let is_first_chunk0 = stage == PackedStage::Chunk0 && !recorded_chunk0_term_excess;
            #[cfg(feature = "timing")]
            if is_first_chunk0 {
                chunk0_first_emitted = chunk.n;
            }

            let t_clip = crate::knn_clipping::timing::Timer::start();
            #[cfg(feature = "timing")]
            let mut used_slots_in_chunk = 0usize;
            #[cfg(feature = "timing")]
            let mut terminated_in_chunk = false;
            let mut term_ready = false;
            for pos in 0..chunk.n {
                #[cfg(feature = "timing")]
                {
                    used_slots_in_chunk = pos + 1;
                }
                let neighbor_slot = packed_chunk[pos];
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if neighbor_idx == i {
                    continue;
                }
                if !attempted_neighbors.insert(neighbor_idx) {
                    continue;
                }

                let neighbor = points[neighbor_idx];
                let clip_result =
                    match builder.clip_with_slot_result(neighbor_idx, neighbor_slot, neighbor) {
                        Ok(r) => r,
                        Err(_) => {
                            break;
                        }
                    };
                match clip_result {
                    crate::knn_clipping::topo2d::types::ClipResult::Unchanged => {
                        term_ready = true;
                    }
                    crate::knn_clipping::topo2d::types::ClipResult::Changed => {
                        term_ready = false;
                    }
                    crate::knn_clipping::topo2d::types::ClipResult::TooManyVertices => {
                        // This should have returned an error already.
                        term_ready = false;
                    }
                }

                // Only run termination checks when the clip is unchanged. If the clip changed,
                // the termination threshold needs to be recomputed (min_cos / sqrt), and doing
                // that on every changed clip is often wasted work.
                if builder.is_bounded()
                    && clip_result == crate::knn_clipping::topo2d::types::ClipResult::Unchanged
                {
                    let bound = if pos + 1 < chunk.n {
                        let next_slot = packed_chunk[pos + 1];
                        let next = point_indices[next_slot as usize] as usize;
                        points[i].dot(points[next])
                    } else {
                        chunk.unseen_bound
                    };
                    if builder.can_terminate(bound) {
                        terminated = true;
                        #[cfg(feature = "timing")]
                        {
                            terminated_in_chunk = true;
                        }
                        break;
                    }
                }
                cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                worst_cos = worst_cos.min(dot);
            }
            cell_sub.add_clip(t_clip.elapsed());

            #[cfg(feature = "timing")]
            if is_first_chunk0 {
                cell_sub.add_packed_knn_chunk0_termination_excess_sample(
                    incoming_edgechecks,
                    chunk0_first_emitted,
                    used_slots_in_chunk,
                    terminated_in_chunk,
                );
                recorded_chunk0_term_excess = true;
            }

            if terminated || builder.is_failed() {
                break;
            }

            if builder.is_bounded() && term_ready && builder.can_terminate(chunk.unseen_bound) {
                terminated = true;
                break;
            }

            k_cur = k1;
        }
    }

    // === Phase 4: Resumable kNN Scan ===
    let resume_k = crate::knn_clipping::KNN_RESUME_K.min(max_neighbors);
    if !terminated && !knn_exhausted && !builder.is_failed() && resume_k > 0 {
        used_knn = true;
        max_k_requested = resume_k;
        neighbor_slots.clear();
        let t_knn = crate::knn_clipping::timing::Timer::start();
        let status = grid.find_k_nearest_resumable_slots_directed_into(
            points[i],
            i,
            resume_k,
            resume_k,
            scratch,
            neighbor_slots,
            bin.as_u8(),
            local.as_u32(),
            &assignment.slot_gen_map,
            assignment.local_shift,
            assignment.local_mask,
        );
        cell_sub.add_knn(t_knn.elapsed());

        // Track which resume stage we're at
        knn_stage = crate::knn_clipping::timing::KnnCellStage::Resume(resume_k);

        let t_clip = crate::knn_clipping::timing::Timer::start();
        let point_indices = grid.point_indices();
        for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
            let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
            if did_packed && builder.has_neighbor(neighbor_idx) {
                continue;
            }
            if !attempted_neighbors.insert(neighbor_idx) {
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

            if builder.is_bounded()
                && termination.should_check(cell_neighbors_processed)
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
            let status = grid.find_k_nearest_resumable_slots_directed_into(
                points[i],
                i,
                k,
                k,
                scratch,
                neighbor_slots,
                bin.as_u8(),
                local.as_u32(),
                &assignment.slot_gen_map,
                assignment.local_shift,
                assignment.local_mask,
            );
            cell_sub.add_knn(t_knn.elapsed());

            // Track which restart stage we're at
            knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(k_stage);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if builder.has_neighbor(neighbor_idx) {
                    continue;
                }
                if !attempted_neighbors.insert(neighbor_idx) {
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

                if builder.is_bounded()
                    && termination.should_check(cell_neighbors_processed)
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
        } else if did_packed && packed_safe_exhausted {
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

        while !terminated && !builder.is_failed() && k < cap {
            let next_k = (k.saturating_mul(crate::knn_clipping::TERMINATION_GROW_MULTIPLIER))
                .max(k + crate::knn_clipping::TERMINATION_GROW_MIN_STEP)
                .min(cap);
            if next_k <= k {
                break;
            }

            used_knn = true;
            neighbor_slots.clear();

            let t_knn = crate::knn_clipping::timing::Timer::start();
            let status = grid.find_k_nearest_resumable_slots_directed_into(
                points[i],
                i,
                next_k,
                next_k,
                scratch,
                neighbor_slots,
                bin.as_u8(),
                local.as_u32(),
                &assignment.slot_gen_map,
                assignment.local_shift,
                assignment.local_mask,
            );
            cell_sub.add_knn(t_knn.elapsed());
            knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(next_k);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if builder.has_neighbor(neighbor_idx) {
                    continue;
                }
                if !attempted_neighbors.insert(neighbor_idx) {
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
            .filter(|&j| {
                if j == i || already_clipped.contains(&j) {
                    return false;
                }
                // Directed within-bin filtering: never consider earlier locals here.
                // Earlier-local within-bin adjacency must come from incoming edgechecks.
                if assignment.generator_bin[j] == bin
                    && assignment.global_to_local[j].as_u32() < local.as_u32()
                {
                    return false;
                }
                true
            })
            .collect();
        sorted_indices.sort_by(|&a, &b| {
            let da = gen.dot(points[a]);
            let db = gen.dot(points[b]);
            db.partial_cmp(&da).unwrap()
        });
        for p_idx in sorted_indices {
            if !attempted_neighbors.insert(p_idx) {
                continue;
            }
            let slot = grid.point_index_to_slot(p_idx);
            if builder.clip_with_slot(p_idx, slot, points[p_idx]).is_err() {
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

    let stage = if did_full_scan_fallback {
        crate::knn_clipping::timing::KnnCellStage::FullScanFallback
    } else if used_knn {
        knn_stage
    } else if did_packed {
        if packed_tail_used {
            crate::knn_clipping::timing::KnnCellStage::PackedTail
        } else {
            crate::knn_clipping::timing::KnnCellStage::PackedChunk0
        }
    } else {
        knn_stage
    };
    cell_sub.add_cell_stage(
        stage,
        knn_exhausted,
        cell_neighbors_processed,
        packed_tail_used,
        packed_safe_exhausted,
        used_knn,
        incoming_edgechecks,
        edgecheck_seed_clips,
    );

    // === Phase 5: Output Extraction ===
    //
    // Sequential sub-phases: use a lap timer to reduce overhead (one `Instant::now()` per lap).
    let mut t_post = crate::knn_clipping::timing::LapTimer::start();
    if use_dedicated_cert_full {
        builder
            .to_vertex_data_full_dedicated(
                cell_vertices,
                edge_neighbor_globals,
                edge_neighbor_slots,
                edge_neighbor_eps,
            )
            .expect("to_vertex_data_full_dedicated failed after bounded check");
    } else {
        builder
            .to_vertex_data_full(
                cell_vertices,
                edge_neighbor_globals,
                edge_neighbor_slots,
                edge_neighbor_eps,
            )
            .expect("to_vertex_data_full failed after bounded check");
    }
    cell_sub.add_cert(t_post.lap());

    let cell_idx = i as u32;
    edge_scratch.collect_and_resolve(
        cell_idx,
        bin,
        local,
        cell_vertices,
        edge_neighbor_slots,
        edge_neighbor_globals,
        edge_neighbor_eps,
        assignment,
        shard,
        incoming_checks,
    );
    let collect_resolve_time = t_post.lap();
    // Split time between collect and resolve for backward-compatible timing
    cell_sub.add_edge_collect(collect_resolve_time / 2);
    cell_sub.add_edge_resolve(collect_resolve_time / 2);

    let count = cell_vertices.len();
    shard.output.set_cell_count(
        local,
        u8::try_from(count).expect("cell vertex count exceeds u8 capacity"),
    );

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
    cell_sub.add_key_dedup(t_post.lap());

    edge_scratch.emit(shard, cell_vertices, cell_start, bin);
    cell_sub.add_edge_emit(t_post.lap());

    debug_assert_eq!(
        shard.output.cell_indices.len() as u32 - cell_start,
        count as u32,
        "cell index stream mismatch"
    );
}

pub(super) fn build_cells_sharded_live_dedup(
    points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
) -> ShardedCellsData {
    // If termination is enabled but not proven after the kNN schedule, we keep requesting
    // more neighbors until the termination check succeeds.
    //
    // This makes correctness independent of any fixed k cap.
    let termination_max_k_cap = termination.max_k_cap;

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
                let use_dedicated_cert_full =
                    crate::knn_clipping::topo2d::builder::use_dedicated_cert_full_env();
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
                                #[cfg(feature = "timing")]
                                {
                                    let planes_used = {
                                        let res = grid.res();
                                        if res < 3 {
                                            false
                                        } else {
                                            let (_, iu, iv) = crate::cube_grid::cell_to_face_ij(
                                                cell as usize,
                                                res,
                                            );
                                            iu >= 1 && iv >= 1 && iu + 1 < res && iv + 1 < res
                                        }
                                    };
                                    let q = u64::try_from(queries.len()).unwrap_or(0);
                                    sub_accum.add_packed_security_queries(
                                        if planes_used { q } else { 0 },
                                        if planes_used { 0 } else { q },
                                    );
                                    let tail_possible = (0..queries.len())
                                        .filter(|&qi| packed_scratch.tail_possible(qi))
                                        .count()
                                        as u64;
                                    sub_accum.add_packed_tail_possible_queries(q, tail_possible);
                                    sub_accum.add_packed_groups(1);
                                }
                                for (offset, &global) in
                                    my_generators[group_start..cursor].iter().enumerate()
                                {
                                    let local_idx = group_start + offset;
                                    let local = LocalId::from_usize(local_idx);
                                    process_cell(
                                        &mut sub_accum,
                                        &mut ctx,
                                        &mut shard,
                                        points,
                                        grid,
                                        &assignment,
                                        termination,
                                        termination_max_k_cap,
                                        use_dedicated_cert_full,
                                        bin,
                                        global,
                                        local,
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
                                    process_cell(
                                        &mut sub_accum,
                                        &mut ctx,
                                        &mut shard,
                                        points,
                                        grid,
                                        &assignment,
                                        termination,
                                        termination_max_k_cap,
                                        use_dedicated_cert_full,
                                        bin,
                                        global,
                                        local,
                                        None,
                                    );
                                }
                            }
                        }

                        #[cfg(feature = "timing")]
                        {
                            let total = packed_timings.total();
                            sub_accum.add_packed_knn(total);

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
                            sub_accum.add_packed_tail_build_groups(packed_timings.tail_builds);
                            sub_accum.add_packed_knn_select_prep(packed_timings.select_prep);
                            sub_accum
                                .add_packed_knn_select_query_prep(packed_timings.select_query_prep);
                            sub_accum
                                .add_packed_knn_select_partition(packed_timings.select_partition);
                            sub_accum.add_packed_knn_select_sort(packed_timings.select_sort);
                            sub_accum.add_packed_knn_sort_len_hist(
                                &packed_timings.sort_len_counts,
                                &packed_timings.sort_len_nanos,
                            );
                            sub_accum.add_packed_knn_sort_len_exact_hist(
                                &packed_timings.sort_len_exact_counts,
                                &packed_timings.sort_len_exact_nanos,
                            );
                            sub_accum.add_packed_knn_select_scatter(packed_timings.select_scatter);

                            // `packed_knn` is defined as the sum of packed subcomponents.
                        }
                        #[cfg(not(feature = "timing"))]
                        sub_accum.add_packed_knn(packed_elapsed);
                    } else {
                        for (offset, &global) in
                            my_generators[group_start..cursor].iter().enumerate()
                        {
                            let local_idx = group_start + offset;
                            let local = LocalId::from_usize(local_idx);
                            process_cell(
                                &mut sub_accum,
                                &mut ctx,
                                &mut shard,
                                points,
                                grid,
                                &assignment,
                                termination,
                                termination_max_k_cap,
                                use_dedicated_cert_full,
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
