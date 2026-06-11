//! Assembly helpers for live dedup.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::edge_checks::resolve_edge_check_overflow;
use super::packed::{pack_ref, unpack_ref, DEFERRED};
use super::shard::ShardFinal;
use super::types::{BinId, DeferredSlot, EdgeCheckOverflow, UnresolvedEdgeMismatch};
use super::ShardedCellsData;
use crate::diagram::VoronoiCell;
use crate::knn_clipping::cell_build::VertexKey;
use crate::timing::{DedupSubPhases, Timer};

fn patch_deferred_slots_with_fallback<P: super::types::VertexPosition>(
    shards: &mut [super::shard::ShardState<P>],
    generator_bin: &[BinId],
    deferred_slots: Vec<DeferredSlot<P>>,
) -> Result<(), crate::VoronoiError> {
    let patch_slot = |slot: &mut u64, owner_bin: BinId, idx: u32| {
        let packed = pack_ref(owner_bin, idx);
        if *slot == DEFERRED {
            *slot = packed;
        } else {
            debug_assert_eq!(*slot, packed, "edge check index mismatch");
        }
    };

    let mut fallback_map: FxHashMap<VertexKey, (BinId, u32)> = FxHashMap::default();
    for entry in deferred_slots {
        let source_bin = entry.source_bin.as_usize();
        let source_slot = entry.source_slot as usize;
        if shards[source_bin].output.cell_indices[source_slot] != DEFERRED {
            continue;
        }

        let owner_bin = generator_bin[entry.key[0] as usize];
        let idx = if let Some(&(bin, idx)) = fallback_map.get(&entry.key) {
            debug_assert_eq!(bin, owner_bin, "fallback owner bin mismatch");
            idx
        } else {
            let new_idx = {
                let owner_shard = &mut shards[owner_bin.as_usize()];
                let new_idx = u32::try_from(owner_shard.output.vertices.len()).map_err(|_| {
                    crate::VoronoiError::RepresentationLimit(
                        "deferred fallback vertex index exceeds u32 capacity".to_string(),
                    )
                })?;
                owner_shard.output.vertices.push(entry.pos);
                owner_shard.output.vertex_keys.push(entry.key);
                new_idx
            };
            fallback_map.insert(entry.key, (owner_bin, new_idx));
            new_idx
        };

        let slot = &mut shards[source_bin].output.cell_indices[source_slot];
        patch_slot(slot, owner_bin, idx);
    }
    Ok(())
}

pub(super) fn assemble_sharded_live_dedup<P: super::types::VertexPosition>(
    mut data: ShardedCellsData<P>,
) -> Result<super::AssemblyResult<P>, crate::VoronoiError> {
    let t0 = Timer::start();

    let num_bins = data.assignment.num_bins;

    #[allow(unused_variables)]
    let overflow_collect_time = t0.elapsed();
    let t1 = Timer::start();

    let mut unresolved_edges: Vec<UnresolvedEdgeMismatch> = Vec::new();
    let mut edge_check_overflow: Vec<EdgeCheckOverflow> = Vec::new();
    let mut deferred_slots: Vec<DeferredSlot<P>> = Vec::new();
    for shard in &mut data.shards {
        unresolved_edges.append(&mut shard.output.unresolved_edges);
        edge_check_overflow.append(&mut shard.output.edge_check_overflow);
        deferred_slots.append(&mut shard.output.deferred_slots);
    }

    let overflow_timing = resolve_edge_check_overflow(
        &mut data.shards,
        &mut edge_check_overflow,
        &mut unresolved_edges,
    );
    #[allow(unused_variables)]
    let edge_checks_overflow_time = overflow_timing.sort + overflow_timing.match_;

    let t_deferred = Timer::start();
    patch_deferred_slots_with_fallback(
        &mut data.shards,
        &data.assignment.generator_bin,
        deferred_slots,
    )?;
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
    let finals: Vec<ShardFinal<P>> = std::mem::take(&mut data.shards)
        .into_iter()
        .map(|s| s.into_final())
        .collect();

    let t2 = Timer::start();

    // Phase 4: concatenate vertices
    let mut vertex_offsets: Vec<u32> = vec![0; num_bins];
    let mut total_vertices = 0usize;
    for (bin, shard) in finals.iter().enumerate() {
        vertex_offsets[bin] = u32::try_from(total_vertices).map_err(|_| {
            crate::VoronoiError::RepresentationLimit(
                "assembled vertex offsets exceed u32 capacity".to_string(),
            )
        })?;
        total_vertices = total_vertices
            .checked_add(shard.output.vertices.len())
            .ok_or_else(|| {
                crate::VoronoiError::RepresentationLimit(
                    "assembled vertex buffer exceeds usize capacity".to_string(),
                )
            })?;
    }
    if total_vertices > u32::MAX as usize {
        return Err(crate::VoronoiError::RepresentationLimit(
            "assembled vertex buffer exceeds u32 capacity".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    let (all_vertices, all_vertex_keys) = {
        let mut all_vertices = Vec::<P>::with_capacity(total_vertices);
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
                        let v_dst = (vertices_ptr as *mut P).add(offset);
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
        total_cell_indices = total_cell_indices.checked_add(count).ok_or_else(|| {
            crate::VoronoiError::RepresentationLimit(
                "assembled cell index buffer exceeds u32 capacity".to_string(),
            )
        })?;
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
        triplet_keys: finals.iter().map(|s| s.triplet_keys).sum(),
        unresolved_edges_count: unresolved_edges.len() as u64,
    };

    #[cfg(not(feature = "timing"))]
    let sub_phases = DedupSubPhases;

    Ok(super::AssemblyResult {
        vertices: all_vertices,
        vertex_keys: all_vertex_keys,
        unresolved_edges,
        cells,
        cell_indices,
        dedup_sub: sub_phases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn_clipping::edge_reconcile::{
        edge_segments_for_neighbor, reconcile_unresolved_edges,
    };
    use crate::knn_clipping::live_dedup::binning::BinAssignment;
    use crate::knn_clipping::live_dedup::packed::pack_edge;
    use crate::knn_clipping::live_dedup::shard::ShardState;
    use crate::knn_clipping::live_dedup::types::{EdgeCheckOverflow, LocalId};
    use crate::knn_clipping::live_dedup::{EdgeRecord, ShardedCellsData};
    use glam::Vec3;
    use std::collections::BTreeSet;

    fn bin(value: usize) -> BinId {
        BinId::from_usize(value)
    }

    #[test]
    fn deferred_fallback_allocates_once_per_owner_key() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        shards[0].output.cell_indices = vec![DEFERRED, DEFERRED];
        let generator_bin = vec![bin(1), bin(0), bin(0)];
        let key = [0, 1, 2];
        let pos = Vec3::new(0.0, 0.0, 1.0);

        patch_deferred_slots_with_fallback(
            &mut shards,
            &generator_bin,
            vec![
                DeferredSlot {
                    key,
                    pos,
                    source_bin: bin(0),
                    source_slot: 0,
                },
                DeferredSlot {
                    key,
                    pos,
                    source_bin: bin(0),
                    source_slot: 1,
                },
            ],
        )
        .expect("fallback patching should succeed without capacity overflow");

        assert_eq!(shards[1].output.vertices.len(), 1);
        assert_eq!(shards[1].output.vertex_keys, vec![key]);
        assert_eq!(shards[0].output.cell_indices[0], pack_ref(bin(1), 0));
        assert_eq!(shards[0].output.cell_indices[1], pack_ref(bin(1), 0));
    }

    #[test]
    fn overflow_matching_patches_cross_bin_slots_before_fallback() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        shards[0].output.cell_indices = vec![DEFERRED, DEFERRED];
        shards[1].output.cell_indices = vec![DEFERRED, DEFERRED];

        let edge_key = pack_edge(0, 1);
        let mut unresolved = Vec::new();
        let mut overflow = vec![
            EdgeCheckOverflow {
                key: edge_key,
                side: 0,
                source_bin: bin(0),
                thirds: [2, 3],
                indices: [10, 11],
                slots: [0, 1],
            },
            EdgeCheckOverflow {
                key: edge_key,
                side: 1,
                source_bin: bin(1),
                thirds: [3, 2],
                indices: [20, 21],
                slots: [0, 1],
            },
        ];

        resolve_edge_check_overflow(&mut shards, &mut overflow, &mut unresolved);

        assert!(
            unresolved.is_empty(),
            "full reverse-winding match should not remain unresolved"
        );
        assert_eq!(shards[0].output.cell_indices[0], pack_ref(bin(1), 21));
        assert_eq!(shards[0].output.cell_indices[1], pack_ref(bin(1), 20));
        assert_eq!(shards[1].output.cell_indices[0], pack_ref(bin(0), 11));
        assert_eq!(shards[1].output.cell_indices[1], pack_ref(bin(0), 10));
    }

    #[test]
    fn overflow_mismatch_is_reported_unresolved() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        let edge_key = pack_edge(0, 1);
        let mut unresolved = Vec::new();
        let mut overflow = vec![
            EdgeCheckOverflow {
                key: edge_key,
                side: 0,
                source_bin: bin(0),
                thirds: [2, 3],
                indices: [10, 11],
                slots: [0, 1],
            },
            EdgeCheckOverflow {
                key: edge_key,
                side: 1,
                source_bin: bin(1),
                thirds: [9, 8],
                indices: [20, 21],
                slots: [0, 1],
            },
        ];

        resolve_edge_check_overflow(&mut shards, &mut overflow, &mut unresolved);

        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].key, edge_key);
    }

    #[test]
    fn assembly_then_reconcile_handles_overflow_fallback_and_unresolved_edge() {
        let mut shard0 = ShardState::new(3);
        let mut shard1 = ShardState::new(3);

        shard0.output.vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        shard0.output.vertex_keys = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3]];
        shard0.output.cell_indices = vec![
            pack_ref(bin(0), 0),
            pack_ref(bin(0), 1),
            pack_ref(bin(0), 2),
        ];
        shard0.output.set_cell_start(LocalId::from_usize(0), 0);
        shard0.output.set_cell_count(LocalId::from_usize(0), 3);

        shard1.output.vertices = vec![
            Vec3::new(1.0 + 1.0e-5, 2.0e-6, 0.0),
            Vec3::new(2.0e-6, 1.0 + 1.0e-5, 0.0),
        ];
        shard1.output.vertex_keys = vec![[0, 1, 4], [0, 1, 5]];
        shard1.output.cell_indices = vec![pack_ref(bin(1), 0), pack_ref(bin(1), 1), DEFERRED];
        shard1.output.set_cell_start(LocalId::from_usize(0), 0);
        shard1.output.set_cell_count(LocalId::from_usize(0), 3);
        shard1.output.deferred_slots.push(DeferredSlot {
            key: [0, 4, 5],
            pos: Vec3::new(-1.0, 0.0, 0.0),
            source_bin: bin(1),
            source_slot: 2,
        });
        let edge_key = pack_edge(0, 1);
        shard0.output.edge_check_overflow.push(EdgeCheckOverflow {
            key: edge_key,
            side: 0,
            source_bin: bin(0),
            thirds: [2, 3],
            indices: [0, 1],
            slots: [0, 1],
        });
        shard1.output.edge_check_overflow.push(EdgeCheckOverflow {
            key: edge_key,
            side: 1,
            source_bin: bin(1),
            thirds: [9, 8],
            indices: [0, 1],
            slots: [0, 1],
        });

        let assignment = BinAssignment {
            generator_bin: vec![bin(0), bin(1), bin(0), bin(0), bin(1), bin(1)],
            global_to_local: vec![
                LocalId::from_usize(0),
                LocalId::from_usize(0),
                LocalId::from_usize(1),
                LocalId::from_usize(2),
                LocalId::from_usize(1),
                LocalId::from_usize(2),
            ],
            slot_gen_map: Vec::new(),
            local_shift: 0,
            local_mask: 0,
            bin_generators: vec![vec![0, 2, 3], vec![1, 4, 5]],
            num_bins: 2,
        };
        let sharded = ShardedCellsData {
            assignment,
            shards: vec![shard0, shard1],
            cell_sub: crate::timing::CellSubAccum::new(),
        };

        let assembled = assemble_sharded_live_dedup(sharded).expect("assembly should succeed");
        assert_eq!(assembled.unresolved_edges.len(), 1);
        assert_eq!(assembled.unresolved_edges[0].key, edge_key);
        assert_eq!(assembled.cells.len(), 6);
        assert_eq!(assembled.cells[0].vertex_count(), 3);
        assert_eq!(assembled.cells[1].vertex_count(), 3);

        let cell1_start = assembled.cells[1].vertex_start();
        let cell1_indices = &assembled.cell_indices[cell1_start..cell1_start + 3];
        let fallback_global = cell1_indices[2] as usize;
        assert_eq!(
            assembled.vertex_keys[fallback_global],
            [0, 4, 5],
            "deferred slot should be patched through fallback ownership before reconciliation"
        );

        let reconcile_input: Vec<EdgeRecord> = assembled
            .unresolved_edges
            .iter()
            .map(|edge| EdgeRecord { key: edge.key })
            .collect();
        let (cells, cell_indices) = reconcile_unresolved_edges(
            &reconcile_input,
            &assembled.vertices,
            &assembled.cells,
            &assembled.cell_indices,
            &assembled.vertex_keys,
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        )
        .expect("reconciliation should succeed without capacity overflow")
        .expect("expected unresolved shared-edge mismatch to be reconciled");

        let seg_a = edge_segments_for_neighbor(0, 1, &cells, &cell_indices, &assembled.vertex_keys)
            .expect("edge segments should resolve after reconciliation");
        let seg_b = edge_segments_for_neighbor(1, 0, &cells, &cell_indices, &assembled.vertex_keys)
            .expect("edge segments should resolve after reconciliation");
        assert_eq!(seg_a.len(), 1);
        assert_eq!(seg_b.len(), 1);
        let set_a = BTreeSet::from([seg_a[0].0, seg_a[0].1]);
        let set_b = BTreeSet::from([seg_b[0].0, seg_b[0].1]);
        assert_eq!(
            set_a, set_b,
            "post-assembly reconciliation should make both cells share the same edge endpoints"
        );
    }
}
