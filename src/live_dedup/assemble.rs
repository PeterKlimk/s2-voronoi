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
) -> Result<bool, crate::VoronoiError> {
    let patch_slot = |slot: &mut u64, owner_bin: BinId, idx: u32| {
        let packed = pack_ref(owner_bin, idx);
        if *slot == DEFERRED {
            *slot = packed;
        } else {
            debug_assert_eq!(*slot, packed, "edge check index mismatch");
        }
    };

    let mut fallback_map: FxHashMap<VertexKey, (BinId, u32)> = FxHashMap::default();
    let mut resolution_drift_exceeded = false;
    for entry in deferred_slots {
        let source_bin = entry.source_bin.as_usize();
        let source_slot = entry.source_slot as usize;
        let existing = shards[source_bin].output.cell_indices[source_slot];
        if existing != DEFERRED {
            let (representative_bin, representative_local) = unpack_ref(existing);
            let representative = shards[representative_bin.as_usize()].output.vertices
                [representative_local as usize];
            let delta = representative.resolution_axis_delta(entry.pos);
            resolution_drift_exceeded |= !delta.is_finite()
                || delta > f64::from(crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS);
            shards[representative_bin.as_usize()]
                .output
                .add_vertex_incidence(representative_local);
            continue;
        }

        let owner_bin = generator_bin[entry.key[0] as usize];
        let (idx, is_new) = if let Some(&(bin, idx)) = fallback_map.get(&entry.key) {
            debug_assert_eq!(bin, owner_bin, "fallback owner bin mismatch");
            let representative = shards[owner_bin.as_usize()].output.vertices[idx as usize];
            let delta = representative.resolution_axis_delta(entry.pos);
            resolution_drift_exceeded |= !delta.is_finite()
                || delta > f64::from(crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS);
            (idx, false)
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
                owner_shard.output.vertex_incidence.push(1);
                new_idx
            };
            fallback_map.insert(entry.key, (owner_bin, new_idx));
            (new_idx, true)
        };

        if !is_new {
            shards[owner_bin.as_usize()]
                .output
                .add_vertex_incidence(idx);
        }

        let slot = &mut shards[source_bin].output.cell_indices[source_slot];
        patch_slot(slot, owner_bin, idx);
    }
    Ok(resolution_drift_exceeded)
}

struct CollectedShardBookkeeping<P> {
    unresolved_edges: Vec<UnresolvedEdgeMismatch>,
    edge_check_overflow: Vec<EdgeCheckOverflow>,
    deferred_slots: Vec<DeferredSlot<P>>,
}

fn collect_shard_bookkeeping<P: super::types::VertexPosition>(
    shards: &mut [super::shard::ShardState<P>],
) -> CollectedShardBookkeeping<P> {
    let unresolved_total: usize = shards
        .iter()
        .map(|shard| shard.output.unresolved_edges.len())
        .sum();
    let overflow_total: usize = shards
        .iter()
        .map(|shard| shard.output.edge_check_overflow.len())
        .sum();
    let deferred_total: usize = shards
        .iter()
        .map(|shard| shard.output.deferred_slots.len())
        .sum();

    let mut unresolved_edges = Vec::new();
    if unresolved_total != 0 {
        unresolved_edges.reserve_exact(unresolved_total);
    }
    let mut edge_check_overflow = Vec::new();
    edge_check_overflow.reserve_exact(overflow_total);
    let mut deferred_slots = Vec::new();
    deferred_slots.reserve_exact(deferred_total);

    for shard in shards {
        unresolved_edges.append(&mut shard.output.unresolved_edges);
        edge_check_overflow.append(&mut shard.output.edge_check_overflow);
        deferred_slots.append(&mut shard.output.deferred_slots);
    }

    CollectedShardBookkeeping {
        unresolved_edges,
        edge_check_overflow,
        deferred_slots,
    }
}

pub(super) fn assemble_sharded_live_dedup<P: super::types::VertexPosition>(
    mut data: ShardedCellsData<P>,
) -> Result<super::AssemblyResult<P>, crate::VoronoiError> {
    let t0 = Timer::start();

    let num_bins = data.assignment.num_bins;

    #[allow(unused_variables)]
    let overflow_collect_time = t0.elapsed();
    let t1 = Timer::start();

    let CollectedShardBookkeeping {
        mut unresolved_edges,
        edge_check_overflow,
        deferred_slots,
    } = collect_shard_bookkeeping(&mut data.shards);

    let overflow_timing = resolve_edge_check_overflow(
        &mut data.shards,
        &edge_check_overflow,
        &mut unresolved_edges,
    );
    #[allow(unused_variables)]
    let edge_checks_overflow_time = overflow_timing.sort + overflow_timing.match_;

    // Dev-only: tally unresolved-edge origins to see which path inflates the
    // residual (within-bin vs cross-bin). See docs/correctness.md.
    if std::env::var("VORONOI_MESH_UNPAIRED_ORIGINS").is_ok() {
        use super::types::UnresolvedEdgeOrigin as O;
        let mut c = [0usize; 10];
        for e in &unresolved_edges {
            let i = match e.origin {
                O::InBinMissingCheck => 0,
                O::InBinThirdsMismatch => 1,
                O::InBinDuplicateSide => 2,
                O::InBinUnconsumedCheck => 3,
                O::CrossBinThirdsMismatch => 4,
                O::CrossBinSingleSided => 5,
                O::CrossBinDuplicateSide => 6,
                O::CrossBinSlotConflict => 7,
                O::PostRepairUnpaired => 8,
                O::EndpointKeyMismatch => 9,
            };
            c[i] += 1;
        }
        eprintln!(
            "[origins] total={} | InBin(miss={} thirds={} dup={} unconsumed={}) \
             CrossBin(thirds={} single={} dup={} slot={}) endpoint_key={}",
            unresolved_edges.len(),
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            c[6],
            c[7],
            c[9]
        );
    }

    let t_deferred = Timer::start();
    let deferred_resolution_drift_exceeded = patch_deferred_slots_with_fallback(
        &mut data.shards,
        &data.assignment.generator_bin,
        deferred_slots,
    )?;
    let resolution_drift_exceeded = deferred_resolution_drift_exceeded
        || data
            .shards
            .iter()
            .any(|shard| shard.output.resolution_drift_exceeded);
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
    let mut finals: Vec<ShardFinal<P>> = std::mem::take(&mut data.shards)
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

    // Positions are always needed by the diagram, so concatenate them. Vertex
    // *keys* are only consulted by edge reconciliation (for at most the defect
    // region), so they are NOT concatenated — kept per-shard in
    // `ShardedVertexKeys` below.
    // `P: Copy`, and the parallel scatter below writes every slot exactly once
    // via the partitioned `vertex_offsets`. Keep the Vec length at zero until
    // the scatter completes so no uninitialized `P` is ever exposed as a value.
    #[cfg(feature = "parallel")]
    let all_vertices = {
        let mut all_vertices = Vec::<P>::with_capacity(total_vertices);
        let vertices_ptr = all_vertices.spare_capacity_mut().as_mut_ptr() as usize;
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
                    unsafe {
                        let v_dst = (vertices_ptr as *mut P).add(offset as usize);
                        std::ptr::copy_nonoverlapping(shard.output.vertices.as_ptr(), v_dst, count);
                    }
                }
            });
        // SAFETY: `vertex_offsets` is the prefix sum of all shard lengths, so
        // each copy targets a disjoint range and their union is `0..total_vertices`.
        // Rayon has joined all workers, and thus every element is initialized.
        unsafe {
            all_vertices.set_len(total_vertices);
        }
        all_vertices
    };

    #[cfg(not(feature = "parallel"))]
    let all_vertices = {
        let mut all_vertices = Vec::with_capacity(total_vertices);
        for shard in &finals {
            all_vertices.extend_from_slice(&shard.output.vertices);
        }
        all_vertices
    };

    // Move the per-shard key vecs out of the (about-to-be-dropped) finals into
    // the sharded accessor — O(num_bins), zero copy. `offsets` is the
    // prefix-sum (vertex_offsets + total) so a global vid maps to `(bin, local)`.
    let all_vertex_keys = {
        let mut offsets = vertex_offsets.clone();
        offsets.push(total_vertices as u32);
        let shard_keys: Vec<Vec<VertexKey>> = finals
            .iter_mut()
            .map(|s| std::mem::take(&mut s.output.vertex_keys))
            .collect();
        super::ShardedVertexKeys::new(offsets, shard_keys)
    };

    let num_cells = data.assignment.generator_bin.len();
    #[allow(unused_variables)]
    let concat_vertices_time = t2.elapsed();
    let t3 = Timer::start();

    // Phase 4: emit cells in generator index order (prefix-sum + direct fill).
    // Avoid redundant initialization passes in release builds. In debug builds, use sentinels to
    // assert full coverage.
    #[cfg(debug_assertions)]
    let mut cells: Vec<VoronoiCell> = vec![VoronoiCell::new(u32::MAX, u16::MAX); num_cells];
    #[cfg(not(debug_assertions))]
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_cells);

    let mut total_cell_indices = 0u32;
    // The same index addresses initialized debug entries and release spare capacity below.
    #[allow(clippy::needless_range_loop)]
    for gen_idx in 0..num_cells {
        let (bin, local) = data.assignment.generator_bin_local(gen_idx);
        let bin = bin.as_usize();
        let count = u16::from(finals[bin].output.cell_count(local));
        let start = total_cell_indices;
        total_cell_indices = total_cell_indices
            .checked_add(u32::from(count))
            .ok_or_else(|| {
                crate::VoronoiError::RepresentationLimit(
                    "assembled cell index buffer exceeds u32 capacity".to_string(),
                )
            })?;
        #[cfg(debug_assertions)]
        {
            cells[gen_idx] = VoronoiCell::new(start, count);
        }
        #[cfg(not(debug_assertions))]
        {
            cells.spare_capacity_mut()[gen_idx].write(VoronoiCell::new(start, count));
        }
    }

    #[cfg(not(debug_assertions))]
    unsafe {
        // Every spare-capacity entry was initialized in the checked prefix loop above. On an early
        // error, the Vec retains length zero and VoronoiCell has no drop state.
        cells.set_len(num_cells);
    }

    let incidence_summary = {
        let mut used_vertices = 0usize;
        let mut low_incidence = false;
        for shard in &finals {
            debug_assert_eq!(
                shard.output.vertex_incidence.len(),
                shard.output.vertices.len(),
                "vertex incidence out of sync with positions"
            );
            for &count in &shard.output.vertex_incidence {
                used_vertices += usize::from(count != 0);
                low_incidence |= count == 1 || count == 2;
            }
        }
        super::IncidenceSummary {
            used_vertices,
            live_half_edges: total_cell_indices as usize,
            low_incidence,
        }
    };

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
        if let Some(last) = cells.last() {
            debug_assert_eq!(cells[0].vertex_start(), 0, "prefix sum must start at 0");
            debug_assert!(
                cells
                    .windows(2)
                    .all(|w| w[0].vertex_start() <= w[1].vertex_start()),
                "prefix sum must be non-decreasing"
            );
            debug_assert_eq!(
                last.vertex_start() + last.vertex_count(),
                total_cell_indices as usize,
                "prefix sum final total mismatch"
            );
        } else {
            debug_assert_eq!(total_cell_indices, 0);
        }
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
    // Capture slices by value so the parallel closure carries their data pointers directly instead
    // of reloading them through references to the owning Vecs in the per-vertex scatter loop.
    let assignment = &data.assignment;
    let finals_ref = finals.as_slice();
    let cells_ref = cells.as_slice();
    let vertex_offsets = vertex_offsets.as_slice();
    maybe_par_into_iter!(0..num_cells).for_each(move |gen_idx| {
        let (bin, local) = assignment.generator_bin_local(gen_idx);
        let bin = bin.as_usize();
        let shard = &finals_ref[bin];
        let start = shard.output.cell_start(local) as usize;
        let cell = &cells_ref[gen_idx];
        let count = cell.vertex_count();
        let dst_start = cell.vertex_start();

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
                        (local as usize) < finals_ref[vbin.as_usize()].output.vertices.len(),
                        "packed vertex local index out of range"
                    );
                }
                let global = vertex_offsets[vbin.as_usize()] + local;
                dst.add(i).write(global);
            }
        }
    });

    #[cfg(not(debug_assertions))]
    unsafe {
        cell_indices.set_len(total_cell_indices as usize);
    }

    let mut exact_zero_edge_hint_cells = Vec::new();
    for shard in &finals {
        exact_zero_edge_hint_cells.extend_from_slice(&shard.output.exact_zero_edge_hint_cells);
    }
    let exact_zero_edge_hint_cell_count = exact_zero_edge_hint_cells.len();
    let mut exact_zero_edge_candidates = Vec::new();
    for cell_idx in exact_zero_edge_hint_cells {
        let cell = &cells[cell_idx as usize];
        let span = &cell_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
        for edge_idx in 0..span.len() {
            let a = span[edge_idx];
            let b = span[(edge_idx + 1) % span.len()];
            if a == b {
                continue;
            }
            let pa = all_vertices[a as usize];
            let pb = all_vertices[b as usize];
            if pa.dist_sq_f64(pb) == 0.0 {
                exact_zero_edge_candidates.push((a.min(b), a.max(b)));
            }
        }
    }
    exact_zero_edge_candidates.sort_unstable();
    exact_zero_edge_candidates.dedup();

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
        exact_zero_edge_candidates,
        exact_zero_edge_hint_cells: exact_zero_edge_hint_cell_count,
        resolution_drift_exceeded,
        incidence_summary,
        dedup_sub: sub_phases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn_clipping::edge_reconcile::{
        edge_segments_for_neighbor, reconcile_unresolved_edges, RepairApply, VertexKeys,
    };
    use crate::knn_clipping::live_dedup::binning::BinAssignment;
    use crate::knn_clipping::live_dedup::packed::pack_edge;
    use crate::knn_clipping::live_dedup::shard::ShardState;
    use crate::knn_clipping::live_dedup::types::{
        EdgeCheckOverflow, LocalId, UnresolvedEdgeOrigin,
    };
    use crate::knn_clipping::live_dedup::{EdgeRecord, ShardedCellsData};
    use glam::Vec3;
    use std::collections::BTreeSet;

    fn bin(value: usize) -> BinId {
        BinId::from_usize(value)
    }

    #[test]
    fn shard_bookkeeping_collection_reserves_drains_and_preserves_order() {
        let mut shards = vec![ShardState::<Vec3>::new(0), ShardState::<Vec3>::new(0)];
        for (ordinal, shard) in shards.iter_mut().enumerate() {
            let key = pack_edge(ordinal as u32, ordinal as u32 + 10);
            shard.output.unresolved_edges.push(UnresolvedEdgeMismatch {
                key,
                origin: UnresolvedEdgeOrigin::InBinMissingCheck,
            });
            shard.output.edge_check_overflow.push(EdgeCheckOverflow {
                key,
                side: ordinal as u8,
                source_bin: bin(ordinal),
                thirds: [1, 2],
                indices: [3, 4],
                slots: [5, 6],
            });
            shard.output.deferred_slots.push(DeferredSlot {
                key: [ordinal as u32, 20, 30],
                pos: Vec3::new(ordinal as f32, 0.0, 1.0),
                source_bin: bin(ordinal),
                source_slot: ordinal as u32,
            });
        }

        let collected = collect_shard_bookkeeping(&mut shards);

        assert_eq!(collected.unresolved_edges.len(), 2);
        assert_eq!(collected.edge_check_overflow.len(), 2);
        assert_eq!(collected.deferred_slots.len(), 2);
        assert!(collected.unresolved_edges.capacity() >= 2);
        assert!(collected.edge_check_overflow.capacity() >= 2);
        assert!(collected.deferred_slots.capacity() >= 2);
        assert_eq!(collected.unresolved_edges[0].key, pack_edge(0, 10));
        assert_eq!(collected.unresolved_edges[1].key, pack_edge(1, 11));
        assert_eq!(collected.edge_check_overflow[0].source_bin, bin(0));
        assert_eq!(collected.edge_check_overflow[1].source_bin, bin(1));
        assert_eq!(collected.deferred_slots[0].key[0], 0);
        assert_eq!(collected.deferred_slots[1].key[0], 1);
        for shard in &shards {
            assert!(shard.output.unresolved_edges.is_empty());
            assert!(shard.output.edge_check_overflow.is_empty());
            assert!(shard.output.deferred_slots.is_empty());
        }

        let empty_unresolved = collect_shard_bookkeeping(&mut shards);
        assert_eq!(empty_unresolved.unresolved_edges.capacity(), 0);
    }

    #[test]
    fn deferred_fallback_allocates_once_per_owner_key() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        shards[0].output.cell_indices = vec![DEFERRED, DEFERRED];
        let generator_bin = vec![bin(1), bin(0), bin(0)];
        let key = [0, 1, 2];
        let pos = Vec3::new(0.0, 0.0, 1.0);

        let drift_exceeded = patch_deferred_slots_with_fallback(
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

        assert!(!drift_exceeded);
        assert_eq!(shards[1].output.vertices.len(), 1);
        assert_eq!(shards[1].output.vertex_keys, vec![key]);
        assert_eq!(shards[1].output.vertex_incidence, vec![2]);
        assert_eq!(shards[0].output.cell_indices[0], pack_ref(bin(1), 0));
        assert_eq!(shards[0].output.cell_indices[1], pack_ref(bin(1), 0));
    }

    #[test]
    fn deferred_patch_reports_representative_drift_beyond_guard() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        shards[0].output.cell_indices = vec![pack_ref(bin(1), 0)];
        shards[1].output.vertices = vec![Vec3::ZERO];
        shards[1].output.vertex_keys = vec![[0, 1, 2]];
        shards[1].output.vertex_incidence = vec![0];
        let eps = crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS;

        let drift_exceeded = patch_deferred_slots_with_fallback(
            &mut shards,
            &[bin(1), bin(0), bin(0)],
            vec![DeferredSlot {
                key: [0, 1, 2],
                pos: Vec3::new(f32::from_bits(eps.to_bits() + 1), 0.0, 0.0),
                source_bin: bin(0),
                source_slot: 0,
            }],
        )
        .expect("prepatched deferred slot should be checked");

        assert!(drift_exceeded);
    }

    #[test]
    fn overflow_matching_patches_cross_bin_slots_before_fallback() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        shards[0].output.cell_indices = vec![DEFERRED, DEFERRED];
        shards[1].output.cell_indices = vec![DEFERRED, DEFERRED];

        let edge_key = pack_edge(0, 1);
        let mut unresolved = Vec::new();
        let overflow = vec![
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

        resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

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
        let overflow = vec![
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

        resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].key, edge_key);
    }

    #[test]
    fn overflow_duplicate_runs_do_not_patch_an_arbitrary_pair() {
        for sides in [[0u8, 0, 1], [0u8, 1, 1]] {
            let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
            shards[0].output.cell_indices = vec![DEFERRED; 4];
            shards[1].output.cell_indices = vec![DEFERRED; 4];
            let edge_key = pack_edge(0, 1);
            let mut side_counts = [0usize; 2];
            let overflow: Vec<EdgeCheckOverflow> = sides
                .into_iter()
                .map(|side| {
                    let ordinal = side_counts[side as usize];
                    side_counts[side as usize] += 1;
                    EdgeCheckOverflow {
                        key: edge_key,
                        side,
                        source_bin: bin(side as usize),
                        thirds: if side == 0 { [2, 3] } else { [3, 2] },
                        indices: [10 + ordinal as u32 * 2, 11 + ordinal as u32 * 2],
                        slots: [ordinal as u32 * 2, ordinal as u32 * 2 + 1],
                    }
                })
                .collect();
            let mut unresolved = Vec::new();

            resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

            assert_eq!(unresolved.len(), 1, "sides={sides:?}");
            assert_eq!(
                unresolved[0].origin,
                UnresolvedEdgeOrigin::CrossBinDuplicateSide,
                "sides={sides:?}"
            );
            assert!(
                shards
                    .iter()
                    .all(|shard| shard.output.cell_indices.iter().all(|&v| v == DEFERRED)),
                "ambiguous run must be left to vertex-key fallback; sides={sides:?}"
            );
        }
    }

    #[test]
    fn overflow_duplicate_run_without_opposite_side_reports_both_defects() {
        let mut shards = vec![ShardState::<Vec3>::new(1), ShardState::<Vec3>::new(1)];
        shards[0].output.cell_indices = vec![DEFERRED; 6];
        let edge_key = pack_edge(0, 1);
        let overflow: Vec<EdgeCheckOverflow> = (0..3)
            .map(|ordinal| EdgeCheckOverflow {
                key: edge_key,
                side: 0,
                source_bin: bin(0),
                thirds: [2, 3],
                indices: [10 + ordinal * 2, 11 + ordinal * 2],
                slots: [ordinal * 2, ordinal * 2 + 1],
            })
            .collect();
        let mut unresolved = Vec::new();

        resolve_edge_check_overflow(&mut shards, &overflow, &mut unresolved);

        let origins: BTreeSet<_> = unresolved.iter().map(|entry| entry.origin).collect();
        assert_eq!(
            origins,
            BTreeSet::from([
                UnresolvedEdgeOrigin::CrossBinDuplicateSide,
                UnresolvedEdgeOrigin::CrossBinSingleSided,
            ])
        );
    }

    #[test]
    fn assembly_then_reconcile_handles_overflow_fallback_and_unresolved_edge() {
        let mut shard0 = ShardState::new(3);
        let mut shard1 = ShardState::new(3);
        shard0.output.resolution_drift_exceeded = true;

        shard0.output.vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        shard0.output.vertex_keys = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3]];
        shard0.output.vertex_incidence = vec![0; 3];
        shard0.output.cell_indices = vec![
            pack_ref(bin(0), 0),
            pack_ref(bin(0), 1),
            pack_ref(bin(0), 2),
        ];
        shard0.output.set_cell_start(LocalId::from_usize(0), 0);
        shard0.output.set_cell_count(LocalId::from_usize(0), 3);

        shard1.output.vertices = vec![
            Vec3::new(1.0 + 2.0e-7, 4.0e-8, 0.0),
            Vec3::new(4.0e-8, 1.0 + 2.0e-7, 0.0),
        ];
        shard1.output.vertex_keys = vec![[0, 1, 4], [0, 1, 5]];
        shard1.output.vertex_incidence = vec![0; 2];
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
            generator_layout: vec![0, 1u32 << 31, 1, 2, (1u32 << 31) | 1, (1u32 << 31) | 2],
            slot_gen_map: Vec::new(),
            local_shift: 31,
            local_mask: (1u32 << 31) - 1,
            bin_generators: vec![vec![0, 2, 3], vec![1, 4, 5]],
            num_bins: 2,
        };
        let sharded = ShardedCellsData {
            assignment,
            shards: vec![shard0, shard1],
            cell_sub: crate::timing::CellSubAccum::new(),
        };

        let assembled = assemble_sharded_live_dedup(sharded).expect("assembly should succeed");
        assert!(assembled.resolution_drift_exceeded);
        assert_eq!(assembled.unresolved_edges.len(), 1);
        assert_eq!(assembled.unresolved_edges[0].key, edge_key);
        assert_eq!(assembled.cells.len(), 6);
        assert_eq!(assembled.cells[0].vertex_count(), 3);
        assert_eq!(assembled.cells[1].vertex_count(), 3);

        let cell1_start = assembled.cells[1].vertex_start();
        let cell1_indices = &assembled.cell_indices[cell1_start..cell1_start + 3];
        let fallback_global = cell1_indices[2] as usize;
        assert_eq!(
            assembled.vertex_keys.get(fallback_global as u32),
            Some([0, 4, 5]),
            "deferred slot should be patched through fallback ownership before reconciliation"
        );

        let reconcile_input: Vec<EdgeRecord> = assembled
            .unresolved_edges
            .iter()
            .map(|edge| EdgeRecord { key: edge.key })
            .collect();
        let mut cells = assembled.cells.clone();
        let mut cell_indices = assembled.cell_indices.clone();
        let spans_before: Vec<Vec<u32>> = cells
            .iter()
            .map(|c| cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()].to_vec())
            .collect();
        let _residual = reconcile_unresolved_edges(
            &reconcile_input,
            &assembled.vertices,
            &mut cells,
            &mut cell_indices,
            VertexKeys::Sharded(&assembled.vertex_keys),
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
            RepairApply::InPlace,
            |_, _| false,
        )
        .expect("reconciliation should succeed without capacity overflow");
        let spans_after: Vec<Vec<u32>> = cells
            .iter()
            .map(|c| cell_indices[c.vertex_start()..c.vertex_start() + c.vertex_count()].to_vec())
            .collect();
        assert_ne!(
            spans_before, spans_after,
            "expected unresolved shared-edge mismatch to be reconciled"
        );

        let seg_a = edge_segments_for_neighbor(
            0,
            1,
            &cells,
            &cell_indices,
            VertexKeys::Sharded(&assembled.vertex_keys),
        )
        .expect("edge segments should resolve after reconciliation");
        let seg_b = edge_segments_for_neighbor(
            1,
            0,
            &cells,
            &cell_indices,
            VertexKeys::Sharded(&assembled.vertex_keys),
        )
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
