//! `process_cell` implementation for live-dedup cell building.

use glam::Vec3;

use crate::cube_grid::packed_knn::{PackedKnnCellScratch, PackedKnnTimings, PackedStage};
use crate::cube_grid::CubeMapGrid;
use crate::knn_clipping::TerminationConfig;
use crate::knn_clipping::cell_builder::VertexData;
use crate::knn_clipping::topo2d::Topo2DBuilder;

use super::super::edge_checks::unpack_edge_key;
use super::super::packed::{pack_ref, DEFERRED, INVALID_INDEX};
use super::super::shard::ShardState;
use super::super::types::{BinId, DeferredSlot, EdgeCheck, LocalId};

use super::CellContext;

pub(super) fn process_cell(
    cell_sub: &mut crate::knn_clipping::timing::CellSubAccum,
    ctx: &mut CellContext,
    shard: &mut ShardState,
    points: &[Vec3],
    grid: &CubeMapGrid,
    assignment: &super::super::binning::BinAssignment,
    termination: TerminationConfig,
    termination_max_k_cap: Option<usize>,
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

    let directed_ctx = crate::cube_grid::DirectedCtx::new(
        bin.as_u8(),
        local.as_u32(),
        &assignment.slot_gen_map,
        assignment.local_shift,
        assignment.local_mask,
    );

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
            let neighbor_slot = grid.point_index_to_slot(neighbor_idx);

            if !attempted_neighbors.insert(neighbor_slot as usize) {
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

        loop {
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

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let mut term_ready = false;
            for pos in 0..chunk.n {
                let neighbor_slot = packed_chunk[pos];
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if neighbor_idx == i {
                    continue;
                }
                // Packed query output is unique by slot; just mark as seen for later stages.
                attempted_neighbors.mark(neighbor_slot as usize);

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
                        break;
                    }
                }
                cell_neighbors_processed += 1;
                let dot = points[i].dot(neighbor);
                worst_cos = worst_cos.min(dot);
            }
            cell_sub.add_clip(t_clip.elapsed());

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
        let status = grid.find_k_nearest_resumable_slots_directed_ctx_into(
            points[i],
            i,
            resume_k,
            resume_k,
            scratch,
            neighbor_slots,
            directed_ctx,
        );
        cell_sub.add_knn(t_knn.elapsed());

        // Track which resume stage we're at
        knn_stage = crate::knn_clipping::timing::KnnCellStage::Resume(resume_k);

        let t_clip = crate::knn_clipping::timing::Timer::start();
        let point_indices = grid.point_indices();
        for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
            let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
            if !attempted_neighbors.insert(neighbor_slot as usize) {
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
            let status = grid.find_k_nearest_resumable_slots_directed_ctx_into(
                points[i],
                i,
                k,
                k,
                scratch,
                neighbor_slots,
                directed_ctx,
            );
            cell_sub.add_knn(t_knn.elapsed());

            // Track which restart stage we're at
            knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(k_stage);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if !attempted_neighbors.insert(neighbor_slot as usize) {
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
            let status = grid.find_k_nearest_resumable_slots_directed_ctx_into(
                points[i],
                i,
                next_k,
                next_k,
                scratch,
                neighbor_slots,
                directed_ctx,
            );
            cell_sub.add_knn(t_knn.elapsed());
            knn_stage = crate::knn_clipping::timing::KnnCellStage::Restart(next_k);

            let t_clip = crate::knn_clipping::timing::Timer::start();
            let point_indices = grid.point_indices();
            for (pos, &neighbor_slot) in neighbor_slots.iter().enumerate() {
                let neighbor_idx = point_indices[neighbor_slot as usize] as usize;
                if !attempted_neighbors.insert(neighbor_slot as usize) {
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
            let slot = grid.point_index_to_slot(p_idx);
            if !attempted_neighbors.insert(slot as usize) {
                continue;
            }
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

    extract_output(
        cell_sub,
        builder,
        edge_scratch,
        shard,
        assignment,
        incoming_checks,
        cell_vertices,
        edge_neighbor_globals,
        edge_neighbor_slots,
        edge_neighbor_eps,
        cell_start,
        bin,
        i,
        local,
    );
}

#[inline]
fn extract_output(
    cell_sub: &mut crate::knn_clipping::timing::CellSubAccum,
    builder: &mut Topo2DBuilder,
    edge_scratch: &mut super::EdgeScratch,
    shard: &mut ShardState,
    assignment: &super::super::binning::BinAssignment,
    incoming_checks: Vec<EdgeCheck>,
    cell_vertices: &mut Vec<VertexData>,
    edge_neighbor_globals: &mut Vec<u32>,
    edge_neighbor_slots: &mut Vec<u32>,
    edge_neighbor_eps: &mut Vec<f32>,
    cell_start: u32,
    bin: BinId,
    i: usize,
    local: LocalId,
) {
    // === Phase 5: Output Extraction ===
    //
    // Sequential sub-phases: use a lap timer to reduce overhead (one `Instant::now()` per lap).
    let mut t_post = crate::knn_clipping::timing::LapTimer::start();
    builder
        .to_vertex_data_full(
            cell_vertices,
            edge_neighbor_globals,
            edge_neighbor_slots,
            edge_neighbor_eps,
        )
        .expect("to_vertex_data_full failed after bounded check");
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
