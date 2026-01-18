//! Assembly helpers for live dedup.

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::edge_checks::resolve_edge_check_overflow;
use super::packed::{pack_ref, unpack_ref, DEFERRED};
use super::shard::ShardFinal;
use super::types::{BadEdgeRecord, BinId, DeferredSlot, EdgeCheckOverflow, SupportOverflow};
use super::with_two_mut;
use super::ShardedCellsData;
use crate::knn_clipping::cell_builder::VertexKey;
use crate::knn_clipping::timing::{DedupSubPhases, Timer};
use crate::VoronoiCell;

pub(super) fn assemble_sharded_live_dedup(
    mut data: ShardedCellsData,
) -> (
    Vec<Vec3>,
    Vec<VertexKey>,
    Vec<BadEdgeRecord>,
    Vec<VoronoiCell>,
    Vec<u32>,
    DedupSubPhases,
) {
    let t0 = Timer::start();

    let num_bins = data.assignment.num_bins;

    // Phase 3: collect overflow by target bin
    let mut support_by_target: Vec<Vec<SupportOverflow>> =
        (0..num_bins).map(|_| Vec::new()).collect();
    for shard in &mut data.shards {
        for entry in shard.dedup.support_overflow.drain(..) {
            support_by_target[entry.target_bin.as_usize()].push(entry);
        }
    }

    #[allow(unused_variables)]
    let overflow_collect_time = t0.elapsed();
    let t1 = Timer::start();

    // Phase 3: overflow flush (V1: single-threaded)
    for target_idx in 0..num_bins {
        let target_bin = BinId::from_usize(target_idx);
        // Support sets
        for entry in support_by_target[target_idx].drain(..) {
            let source = entry.source_bin.as_usize();
            debug_assert_ne!(
                entry.source_bin, target_bin,
                "overflow should not target same bin"
            );
            let (source_shard, target_shard) = with_two_mut(&mut data.shards, source, target_idx);

            let idx = target_shard.dedup_support_owned(entry.support, entry.pos);
            source_shard.output.cell_indices[entry.source_slot as usize] =
                pack_ref(target_bin, idx);
        }
    }

    let mut bad_edges: Vec<BadEdgeRecord> = Vec::new();
    let mut edge_check_overflow: Vec<EdgeCheckOverflow> = Vec::new();
    let mut deferred_slots: Vec<DeferredSlot> = Vec::new();
    for shard in &mut data.shards {
        bad_edges.append(&mut shard.output.bad_edges);
        edge_check_overflow.append(&mut shard.output.edge_check_overflow);
        deferred_slots.append(&mut shard.output.deferred);
    }

    let (edge_checks_overflow_sort_time, edge_checks_overflow_match_time) =
        resolve_edge_check_overflow(&mut data.shards, &mut edge_check_overflow, &mut bad_edges);
    #[allow(unused_variables)]
    let edge_checks_overflow_time =
        edge_checks_overflow_sort_time + edge_checks_overflow_match_time;

    let patch_slot = |slot: &mut u64, owner_bin: BinId, idx: u32| {
        let packed = pack_ref(owner_bin, idx);
        if *slot == DEFERRED {
            *slot = packed;
        } else {
            debug_assert_eq!(*slot, packed, "edge check index mismatch");
        }
    };

    let t_deferred = Timer::start();
    let mut fallback_map: FxHashMap<VertexKey, (BinId, u32)> = FxHashMap::default();
    for entry in deferred_slots {
        let source_bin = entry.source_bin.as_usize();
        let source_slot = entry.source_slot as usize;
        if data.shards[source_bin].output.cell_indices[source_slot] != DEFERRED {
            continue;
        }

        let owner_bin = data.assignment.generator_bin[entry.key[0] as usize];
        let idx = if let Some(&(bin, idx)) = fallback_map.get(&entry.key) {
            debug_assert_eq!(bin, owner_bin, "fallback owner bin mismatch");
            idx
        } else {
            let new_idx = {
                let owner_shard = &mut data.shards[owner_bin.as_usize()];
                let new_idx = owner_shard.output.vertices.len() as u32;
                owner_shard.output.vertices.push(entry.pos);
                owner_shard.output.vertex_keys.push(entry.key);
                new_idx
            };
            fallback_map.insert(entry.key, (owner_bin, new_idx));
            new_idx
        };

        let slot = &mut data.shards[source_bin].output.cell_indices[source_slot];
        patch_slot(slot, owner_bin, idx);
    }
    #[allow(unused_variables)]
    let deferred_fallback_time = t_deferred.elapsed();

    #[cfg(debug_assertions)]
    for shard in &data.shards {
        debug_assert!(
            !shard.output.cell_indices.contains(&DEFERRED),
            "unresolved deferred indices remain after overflow flush"
        );
    }

    #[allow(unused_variables)]
    let overflow_flush_time = t1.elapsed();

    // Convert to ShardFinal, dropping dedup structures to reduce memory pressure
    let finals: Vec<ShardFinal> = std::mem::take(&mut data.shards)
        .into_iter()
        .map(|s| s.into_final())
        .collect();

    let t2 = Timer::start();

    // Phase 4: concatenate vertices
    let mut vertex_offsets: Vec<u32> = vec![0; num_bins];
    let mut total_vertices = 0usize;
    for (bin, shard) in finals.iter().enumerate() {
        vertex_offsets[bin] =
            u32::try_from(total_vertices).expect("total vertex count exceeds u32 capacity");
        total_vertices += shard.output.vertices.len();
    }

    #[cfg(feature = "parallel")]
    let (all_vertices, all_vertex_keys) = {
        let mut all_vertices = Vec::<Vec3>::with_capacity(total_vertices);
        let mut all_vertex_keys = Vec::<VertexKey>::with_capacity(total_vertices);

        // Safety: We will write to every element in the parallel loop below.
        unsafe {
            all_vertices.set_len(total_vertices);
            all_vertex_keys.set_len(total_vertices);
        }

        let vertices_ptr = all_vertices.as_mut_ptr() as usize;
        let keys_ptr = all_vertex_keys.as_mut_ptr() as usize;

        finals
            .par_iter()
            .zip(vertex_offsets.par_iter())
            .for_each(|(shard, &offset)| {
                let count = shard.output.vertices.len();
                debug_assert_eq!(
                    count,
                    shard.output.vertex_keys.len(),
                    "vertex keys out of sync with vertex positions"
                );

                if count > 0 {
                    let offset = offset as usize;
                    unsafe {
                        let v_dst = (vertices_ptr as *mut Vec3).add(offset);
                        std::ptr::copy_nonoverlapping(shard.output.vertices.as_ptr(), v_dst, count);

                        let k_dst = (keys_ptr as *mut VertexKey).add(offset);
                        std::ptr::copy_nonoverlapping(
                            shard.output.vertex_keys.as_ptr(),
                            k_dst,
                            count,
                        );
                    }
                }
            });

        (all_vertices, all_vertex_keys)
    };

    #[cfg(not(feature = "parallel"))]
    let (all_vertices, all_vertex_keys) = {
        let mut all_vertices = Vec::with_capacity(total_vertices);
        let mut all_vertex_keys = Vec::with_capacity(total_vertices);
        for shard in &finals {
            debug_assert_eq!(
                shard.output.vertices.len(),
                shard.output.vertex_keys.len(),
                "vertex keys out of sync with vertex positions"
            );
            all_vertices.extend_from_slice(&shard.output.vertices);
            all_vertex_keys.extend_from_slice(&shard.output.vertex_keys);
        }
        (all_vertices, all_vertex_keys)
    };

    let num_cells = data.assignment.generator_bin.len();
    #[allow(unused_variables)]
    let concat_vertices_time = t2.elapsed();
    let t3 = Timer::start();

    // Phase 4: emit cells in generator index order (prefix-sum + direct fill).
    let mut cell_starts_global: Vec<u32> = vec![0; num_cells + 1];
    let mut total_cell_indices = 0u32;
    for gen_idx in 0..num_cells {
        let bin = data.assignment.generator_bin[gen_idx].as_usize();
        let local = data.assignment.global_to_local[gen_idx];
        let count = finals[bin].output.cell_count(local) as u32;
        total_cell_indices = total_cell_indices
            .checked_add(count)
            .expect("cell index buffer exceeds u32 capacity");
        cell_starts_global[gen_idx + 1] = total_cell_indices;
    }

    // Avoid redundant initialization passes in release builds.
    // In debug builds, use sentinels to assert full coverage.
    #[cfg(debug_assertions)]
    let mut cells: Vec<VoronoiCell> = vec![VoronoiCell::new(u32::MAX, u16::MAX); num_cells];
    #[cfg(not(debug_assertions))]
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_cells);

    #[cfg(debug_assertions)]
    let mut cell_indices: Vec<u32> = vec![u32::MAX; total_cell_indices as usize];
    #[cfg(not(debug_assertions))]
    let mut cell_indices: Vec<u32> = Vec::with_capacity(total_cell_indices as usize);

    #[cfg(debug_assertions)]
    {
        let expected_indices: usize = finals
            .iter()
            .map(|shard| shard.output.cell_indices.len())
            .sum();
        debug_assert_eq!(
            expected_indices,
            cell_indices.len(),
            "cell index count mismatch after prefix sum"
        );
        debug_assert_eq!(
            cell_starts_global[num_cells], total_cell_indices,
            "prefix sum final total mismatch"
        );
        debug_assert_eq!(cell_starts_global[0], 0, "prefix sum must start at 0");
        debug_assert!(
            cell_starts_global.windows(2).all(|w| w[0] <= w[1]),
            "prefix sum must be non-decreasing"
        );
    }

    let cell_indices_ptr: usize = {
        #[cfg(debug_assertions)]
        {
            cell_indices.as_mut_ptr() as usize
        }
        #[cfg(not(debug_assertions))]
        {
            cell_indices.spare_capacity_mut().as_mut_ptr() as usize
        }
    };
    let cells_ptr: usize = {
        #[cfg(debug_assertions)]
        {
            cells.as_mut_ptr() as usize
        }
        #[cfg(not(debug_assertions))]
        {
            cells.spare_capacity_mut().as_mut_ptr() as usize
        }
    };

    maybe_par_into_iter!(0..num_cells).for_each(|gen_idx| {
        let bin = data.assignment.generator_bin[gen_idx].as_usize();
        let local = data.assignment.global_to_local[gen_idx];
        let shard = &finals[bin];
        let start = shard.output.cell_start(local) as usize;
        let count = shard.output.cell_count(local) as usize;

        let dst_start = cell_starts_global[gen_idx] as usize;

        #[cfg(debug_assertions)]
        {
            let local_idx = local.as_usize();
            debug_assert!(bin < num_bins, "generator bin out of range");
            debug_assert!(
                local_idx < shard.output.cell_counts.len(),
                "local index out of range"
            );
            debug_assert!(
                start + count <= shard.output.cell_indices.len(),
                "src range OOB"
            );
            debug_assert!(
                dst_start + count <= total_cell_indices as usize,
                "dst range OOB"
            );
        }

        let src = &shard.output.cell_indices[start..start + count];
        // Safety: pointers are valid for the buffers; each cell writes a disjoint range.
        unsafe {
            let dst = (cell_indices_ptr as *mut u32).add(dst_start);
            for (i, &packed) in src.iter().enumerate() {
                debug_assert_ne!(packed, DEFERRED, "deferred index leaked to assembly");
                let (vbin, local) = unpack_ref(packed);
                #[cfg(debug_assertions)]
                {
                    debug_assert!(vbin.as_usize() < num_bins, "packed vertex bin out of range");
                    debug_assert!(
                        (local as usize) < finals[vbin.as_usize()].output.vertices.len(),
                        "packed vertex local index out of range"
                    );
                }
                dst.add(i).write(vertex_offsets[vbin.as_usize()] + local);
            }

            let count_u16 = u16::from(shard.output.cell_count(local));
            (cells_ptr as *mut VoronoiCell)
                .add(gen_idx)
                .write(VoronoiCell::new(cell_starts_global[gen_idx], count_u16));
        }
    });

    #[cfg(not(debug_assertions))]
    unsafe {
        cells.set_len(num_cells);
        cell_indices.set_len(total_cell_indices as usize);
    }

    #[cfg(debug_assertions)]
    {
        debug_assert!(
            !cell_indices.contains(&u32::MAX),
            "unwritten cell indices remain after assembly"
        );
        debug_assert!(
            !cells.iter().any(|c| c.vertex_start() == u32::MAX as usize),
            "unwritten cells remain after assembly (vertex_start sentinel)"
        );
        debug_assert!(
            !cells.iter().any(|c| c.vertex_count() == u16::MAX as usize),
            "unwritten cells remain after assembly (vertex_count sentinel)"
        );
    }
    #[allow(unused_variables)]
    let emit_cells_time = t3.elapsed();

    #[cfg(feature = "timing")]
    let sub_phases = DedupSubPhases {
        overflow_collect: overflow_collect_time,
        overflow_flush: overflow_flush_time,
        edge_checks_overflow: edge_checks_overflow_time,
        edge_checks_overflow_sort: edge_checks_overflow_sort_time,
        edge_checks_overflow_match: edge_checks_overflow_match_time,
        deferred_fallback: deferred_fallback_time,
        concat_vertices: concat_vertices_time,
        emit_cells: emit_cells_time,
        triplet_keys: finals.iter().map(|s| s.triplet_keys).sum(),
        support_keys: finals.iter().map(|s| s.support_keys).sum(),
        bad_edges_count: bad_edges.len() as u64,
    };

    #[cfg(not(feature = "timing"))]
    let sub_phases = DedupSubPhases;

    (
        all_vertices,
        all_vertex_keys,
        bad_edges,
        cells,
        cell_indices,
        sub_phases,
    )
}
