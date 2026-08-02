//! Assembly helpers for live dedup.

mod telemetry;

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::edge_checks::resolve_edge_check_overflow;
#[cfg(debug_assertions)]
use super::packed::INVALID_INDEX;
use super::shard::ShardFinal;
use super::types::{BinId, DeferredSlot, EdgeCheckOverflow, EdgeMismatch};
use super::ShardedCellsData;
use crate::diagram::VoronoiCell;
use crate::live_dedup::VertexKey;
use crate::timing::{DedupSubPhases, Timer};

const SHARD_ORDER_SAMPLES_PER_BIN: usize = 32;
#[cfg(feature = "parallel")]
const PAR_ZERO_HINT_CELL_THRESHOLD: usize = 4_096;

#[inline]
fn resolution_axis_delta(a: Vec3, b: Vec3) -> f64 {
    (f64::from(a.x) - f64::from(b.x)).abs()
}

#[inline]
fn dist_sq_f64(a: Vec3, b: Vec3) -> f64 {
    let dx = f64::from(a.x) - f64::from(b.x);
    let dy = f64::from(a.y) - f64::from(b.y);
    let dz = f64::from(a.z) - f64::from(b.z);
    dx * dx + dy * dy + dz * dz
}

/// Exact stored-zero evidence confirmed from final post-patch cell cycles.
struct ConfirmedZeroEdgeHints {
    candidates: Vec<(u32, u32)>,
    hinted_cells: Vec<u32>,
}

#[inline(always)]
fn append_exact_zero_edges_for_cell(
    cell_idx: u32,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    candidates: &mut Vec<(u32, u32)>,
) {
    let cell = &cells[cell_idx as usize];
    let span = &cell_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
    for edge_idx in 0..span.len() {
        let a = span[edge_idx];
        let b = span[(edge_idx + 1) % span.len()];
        if a == b {
            continue;
        }
        let pa = vertices[a as usize];
        let pb = vertices[b as usize];
        if dist_sq_f64(pa, pb) == 0.0 {
            candidates.push((a.min(b), a.max(b)));
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(never)]
fn confirm_exact_zero_edges_parallel(
    hinted_cells: &[u32],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Vec<(u32, u32)> {
    hinted_cells
        .par_iter()
        .fold(Vec::new, |mut candidates, &cell_idx| {
            append_exact_zero_edges_for_cell(
                cell_idx,
                vertices,
                cells,
                cell_indices,
                &mut candidates,
            );
            candidates
        })
        .reduce(Vec::new, |mut left, mut right| {
            left.append(&mut right);
            left
        })
}

fn confirm_exact_zero_edge_hints(
    finals: &[ShardFinal],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> ConfirmedZeroEdgeHints {
    let mut hinted_cells = Vec::new();
    for shard in finals {
        hinted_cells.extend_from_slice(&shard.output.exact_zero_edge_hint_cells);
    }
    hinted_cells.sort_unstable();
    hinted_cells.dedup();

    #[cfg(feature = "parallel")]
    let mut candidates =
        if hinted_cells.len() >= PAR_ZERO_HINT_CELL_THRESHOLD && rayon::current_num_threads() > 1 {
            confirm_exact_zero_edges_parallel(&hinted_cells, vertices, cells, cell_indices)
        } else {
            let mut candidates = Vec::new();
            for &cell_idx in &hinted_cells {
                append_exact_zero_edges_for_cell(
                    cell_idx,
                    vertices,
                    cells,
                    cell_indices,
                    &mut candidates,
                );
            }
            candidates
        };
    #[cfg(not(feature = "parallel"))]
    let mut candidates = {
        let mut candidates = Vec::new();
        for &cell_idx in &hinted_cells {
            append_exact_zero_edges_for_cell(
                cell_idx,
                vertices,
                cells,
                cell_indices,
                &mut candidates,
            );
        }
        candidates
    };

    candidates.sort_unstable();
    candidates.dedup();
    ConfirmedZeroEdgeHints {
        candidates,
        hinted_cells,
    }
}

/// Choose shard-order scatter only when spatial order has little correlation
/// with the caller's generator order. In that regime generator-order scatter
/// jumps through each shard's source arrays; shard order makes those reads
/// sequential. When the orders correlate, retain sequential destination
/// writes instead.
fn prefer_shard_order_scatter(bin_generators: &[Vec<usize>], num_cells: usize) -> bool {
    let mut abs_delta = 0u64;
    let mut samples = 0u64;
    for generators in bin_generators {
        let pair_count = generators.len().saturating_sub(1);
        let sample_count = pair_count.min(SHARD_ORDER_SAMPLES_PER_BIN);
        for sample in 0..sample_count {
            let i = sample * pair_count / sample_count;
            abs_delta += generators[i + 1].abs_diff(generators[i]) as u64;
            samples += 1;
        }
    }
    // The one-percent classifier keeps spatially correlated inputs on
    // contiguous destination writes and scrambled inputs on contiguous
    // shard-source reads. See
    // docs/performance.md#source-pinned-performance-decisions.
    crate::spatial_order::classify_spatial_correlation(abs_delta, samples, num_cells).is_scrambled()
}

#[inline(always)]
unsafe fn scatter_local_indices(
    src: &[u32],
    cell_indices_ptr: usize,
    dst_start: usize,
    vertex_offset: u32,
    _owner_vertex_count: usize,
) {
    // SAFETY: the caller assigns one disjoint destination span per cell and
    // keeps both source and destination allocations alive through the copy.
    let dst = unsafe { (cell_indices_ptr as *mut u32).add(dst_start) };
    for (i, &local) in src.iter().enumerate() {
        #[cfg(debug_assertions)]
        debug_assert!(
            local == INVALID_INDEX || (local as usize) < _owner_vertex_count,
            "primary local vertex index out of range"
        );
        // Foreign slots carry INVALID_INDEX here and are overwritten by the
        // sparse sidecar immediately after this bulk pass.
        let global = vertex_offset.wrapping_add(local);
        unsafe {
            dst.add(i).write(global);
        }
    }
}

fn patch_deferred_slots_with_fallback(
    shards: &mut [super::shard::ShardState],
    generator_bin: &[BinId],
    deferred_slots: Vec<DeferredSlot>,
) -> Result<bool, crate::VoronoiError> {
    let mut fallback_map: FxHashMap<VertexKey, (BinId, u32)> = FxHashMap::default();
    let mut resolution_drift_exceeded = false;
    for entry in deferred_slots {
        let source_bin = entry.source_bin.as_usize();
        if let Some((representative_bin, representative_local)) = shards[source_bin]
            .output
            .logical_reference(entry.source_bin, entry.source_slot)
        {
            let representative = shards[representative_bin.as_usize()].output.vertices
                [representative_local as usize];
            let delta = resolution_axis_delta(representative, entry.pos);
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
            let delta = resolution_axis_delta(representative, entry.pos);
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

        let conflict = shards[source_bin].output.patch_reference(
            entry.source_bin,
            entry.source_slot,
            entry.source_cell,
            entry.source_offset,
            owner_bin,
            idx,
        );
        debug_assert!(!conflict, "deferred fallback reference conflict");
    }
    Ok(resolution_drift_exceeded)
}

struct CollectedShardBookkeeping {
    edge_mismatches: Vec<EdgeMismatch>,
    edge_check_overflow: Vec<EdgeCheckOverflow>,
    deferred_slots: Vec<DeferredSlot>,
}

fn collect_shard_bookkeeping(shards: &mut [super::shard::ShardState]) -> CollectedShardBookkeeping {
    let unresolved_total: usize = shards
        .iter()
        .map(|shard| shard.output.edge_mismatches.len())
        .sum();
    let overflow_total: usize = shards
        .iter()
        .map(|shard| shard.output.edge_check_overflow.len())
        .sum();
    let deferred_total: usize = shards
        .iter()
        .map(|shard| shard.output.deferred_slots.len())
        .sum();

    let mut edge_mismatches = Vec::new();
    if unresolved_total != 0 {
        edge_mismatches.reserve_exact(unresolved_total);
    }
    let mut edge_check_overflow = Vec::new();
    edge_check_overflow.reserve_exact(overflow_total);
    let mut deferred_slots = Vec::new();
    deferred_slots.reserve_exact(deferred_total);

    for shard in shards {
        edge_mismatches.append(&mut shard.output.edge_mismatches);
        edge_check_overflow.append(&mut shard.output.edge_check_overflow);
        deferred_slots.append(&mut shard.output.deferred_slots);
    }

    CollectedShardBookkeeping {
        edge_mismatches,
        edge_check_overflow,
        deferred_slots,
    }
}

struct ConcatenatedVertices {
    positions: Vec<Vec3>,
    keys: super::ShardedVertexKeys,
    offsets: Vec<u32>,
}

#[inline(always)]
fn concatenate_vertices(
    finals: &mut [ShardFinal],
    num_bins: usize,
) -> Result<ConcatenatedVertices, crate::VoronoiError> {
    let mut offsets: Vec<u32> = vec![0; num_bins];
    let mut total_vertices = 0usize;
    for (bin, shard) in finals.iter().enumerate() {
        offsets[bin] = u32::try_from(total_vertices).map_err(|_| {
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
    // keys are sparse-use reconciliation provenance and remain sharded.
    #[cfg(feature = "parallel")]
    let positions = {
        let mut positions = Vec::<Vec3>::with_capacity(total_vertices);
        let vertices_ptr = positions.spare_capacity_mut().as_mut_ptr() as usize;
        finals
            .par_iter()
            .zip(offsets.par_iter())
            .for_each(|(shard, &offset)| {
                let count = shard.output.vertices.len();
                debug_assert_eq!(
                    count,
                    shard.output.vertex_keys.len(),
                    "vertex keys out of sync with vertex positions"
                );
                if count > 0 {
                    unsafe {
                        let v_dst = (vertices_ptr as *mut Vec3).add(offset as usize);
                        std::ptr::copy_nonoverlapping(shard.output.vertices.as_ptr(), v_dst, count);
                    }
                }
            });
        // SAFETY: `offsets` is the prefix sum of all shard lengths, so each
        // copy targets a disjoint range whose union is `0..total_vertices`.
        unsafe {
            positions.set_len(total_vertices);
        }
        positions
    };

    #[cfg(not(feature = "parallel"))]
    let positions = {
        let mut positions = Vec::with_capacity(total_vertices);
        for shard in finals.iter() {
            positions.extend_from_slice(&shard.output.vertices);
        }
        positions
    };

    let keys = {
        let mut key_offsets = offsets.clone();
        key_offsets.push(total_vertices as u32);
        let shard_keys: Vec<Vec<VertexKey>> = finals
            .iter_mut()
            .map(|shard| std::mem::take(&mut shard.output.vertex_keys))
            .collect();
        super::ShardedVertexKeys::new(key_offsets, shard_keys)
    };

    Ok(ConcatenatedVertices {
        positions,
        keys,
        offsets,
    })
}

struct CellPrefixes {
    cells: Vec<VoronoiCell>,
    total_indices: u32,
}

#[cfg(feature = "parallel")]
// This alternative setup runs once per diagram; keep it out of the common
// generator-ordered prefix path while its workers perform the bulk work.
#[cold]
#[inline(never)]
fn emit_cell_prefixes_shard_order_parallel(
    finals: &[ShardFinal],
    assignment: &super::BinAssignment,
) -> Result<CellPrefixes, crate::VoronoiError> {
    let num_cells = assignment.generator_bin.len();
    #[cfg(debug_assertions)]
    let mut cells = vec![VoronoiCell::new(u32::MAX, u16::MAX); num_cells];
    #[cfg(not(debug_assertions))]
    let mut cells = Vec::<VoronoiCell>::with_capacity(num_cells);
    let cells_ptr = {
        #[cfg(debug_assertions)]
        {
            cells.as_mut_ptr() as usize
        }
        #[cfg(not(debug_assertions))]
        {
            cells.spare_capacity_mut().as_mut_ptr() as usize
        }
    };

    let mut bin_bases = Vec::with_capacity(assignment.num_bins);
    let mut total_indices = 0u32;
    for shard in finals {
        bin_bases.push(total_indices);
        let count = u32::try_from(shard.output.cell_indices.len()).map_err(|_| {
            crate::VoronoiError::RepresentationLimit(
                "assembled shard index buffer exceeds u32 capacity".to_string(),
            )
        })?;
        total_indices = total_indices.checked_add(count).ok_or_else(|| {
            crate::VoronoiError::RepresentationLimit(
                "assembled cell index buffer exceeds u32 capacity".to_string(),
            )
        })?;
    }

    (0..assignment.num_bins).into_par_iter().for_each(|bin| {
        let shard = &finals[bin];
        let mut start = bin_bases[bin];
        for (local, &gen_idx) in assignment.bin_generators[bin].iter().enumerate() {
            let count = u16::from(
                shard
                    .output
                    .cell_count(super::types::LocalId::from_usize(local)),
            );
            // SAFETY: every generator belongs to exactly one bin, so workers
            // initialize disjoint cell records exactly once.
            unsafe {
                (cells_ptr as *mut VoronoiCell)
                    .add(gen_idx)
                    .write(VoronoiCell::new(start, count));
            }
            start += u32::from(count);
        }
        debug_assert_eq!(
            start - bin_bases[bin],
            shard.output.cell_indices.len() as u32
        );
    });

    #[cfg(not(debug_assertions))]
    unsafe {
        cells.set_len(num_cells);
    }
    Ok(CellPrefixes {
        cells,
        total_indices,
    })
}

#[cfg(feature = "parallel")]
fn emit_cell_prefixes_parallel(
    finals: &[ShardFinal],
    assignment: &super::BinAssignment,
    shard_order_spans: bool,
) -> Result<CellPrefixes, crate::VoronoiError> {
    let num_cells = assignment.generator_bin.len();
    if shard_order_spans {
        return emit_cell_prefixes_shard_order_parallel(finals, assignment);
    }
    let chunk_count = (rayon::current_num_threads() * 4).min(num_cells);
    let chunk_len = num_cells.div_ceil(chunk_count);

    #[cfg(debug_assertions)]
    let mut cells: Vec<VoronoiCell> = vec![VoronoiCell::new(u32::MAX, u16::MAX); num_cells];
    #[cfg(not(debug_assertions))]
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_cells);
    let cells_ptr = {
        #[cfg(debug_assertions)]
        {
            cells.as_mut_ptr() as usize
        }
        #[cfg(not(debug_assertions))]
        {
            cells.spare_capacity_mut().as_mut_ptr() as usize
        }
    };

    let chunk_totals: Result<Vec<u32>, crate::VoronoiError> = (0..chunk_count)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * chunk_len;
            let end = (start + chunk_len).min(num_cells);
            let mut local_indices = 0u32;
            for gen_idx in start..end {
                let (bin, local) = assignment.generator_bin_local(gen_idx);
                let count = u16::from(finals[bin.as_usize()].output.cell_count(local));
                let cell = VoronoiCell::new(local_indices, count);
                local_indices = local_indices.checked_add(u32::from(count)).ok_or_else(|| {
                    crate::VoronoiError::RepresentationLimit(
                        "assembled cell index chunk exceeds u32 capacity".to_string(),
                    )
                })?;
                // SAFETY: chunks cover disjoint generator ranges within the
                // allocation, and every entry in this range is written once.
                unsafe {
                    (cells_ptr as *mut VoronoiCell).add(gen_idx).write(cell);
                }
            }
            Ok(local_indices)
        })
        .collect();
    let chunk_totals = chunk_totals?;

    #[cfg(not(debug_assertions))]
    unsafe {
        // Every cell entry was initialized by exactly one successful chunk.
        cells.set_len(num_cells);
    }

    let mut chunk_bases = Vec::with_capacity(chunk_totals.len());
    let mut total_indices = 0u32;
    for count in chunk_totals {
        chunk_bases.push(total_indices);
        total_indices = total_indices.checked_add(count).ok_or_else(|| {
            crate::VoronoiError::RepresentationLimit(
                "assembled cell index buffer exceeds u32 capacity".to_string(),
            )
        })?;
    }

    cells
        .par_chunks_mut(chunk_len)
        .zip(chunk_bases.par_iter())
        .for_each(|(chunk, &base)| {
            for cell in chunk {
                *cell = VoronoiCell::new(
                    base + cell.vertex_start() as u32,
                    cell.vertex_count() as u16,
                );
            }
        });

    Ok(CellPrefixes {
        cells,
        total_indices,
    })
}

#[inline(always)]
fn emit_cell_prefixes(
    finals: &[ShardFinal],
    assignment: &super::BinAssignment,
    #[allow(unused_variables)] shard_order_spans: bool,
) -> Result<CellPrefixes, crate::VoronoiError> {
    let num_cells = assignment.generator_bin.len();
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && num_cells >= 65_536 {
        return emit_cell_prefixes_parallel(finals, assignment, shard_order_spans);
    }
    // Avoid redundant initialization in release builds. Debug builds retain
    // sentinels so tests and assertions can prove complete coverage.
    #[cfg(debug_assertions)]
    let mut cells: Vec<VoronoiCell> = vec![VoronoiCell::new(u32::MAX, u16::MAX); num_cells];
    #[cfg(not(debug_assertions))]
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(num_cells);

    let mut total_indices = 0u32;
    // The same index addresses initialized debug entries and release spare
    // capacity below.
    #[allow(clippy::needless_range_loop)]
    for gen_idx in 0..num_cells {
        let (bin, local) = assignment.generator_bin_local(gen_idx);
        let count = u16::from(finals[bin.as_usize()].output.cell_count(local));
        let start = total_indices;
        total_indices = total_indices.checked_add(u32::from(count)).ok_or_else(|| {
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
        // Every spare-capacity entry was initialized in the checked prefix
        // loop. On an early error the Vec retains length zero.
        cells.set_len(num_cells);
    }
    Ok(CellPrefixes {
        cells,
        total_indices,
    })
}

#[inline(always)]
fn summarize_incidence(
    finals: &[ShardFinal],
    total_cell_indices: u32,
    collect_low_vertices: bool,
) -> super::IncidenceSummary {
    let mut used_vertices = 0usize;
    let mut low_incidence = false;
    let mut low_incidence_vertices = Vec::new();
    if collect_low_vertices {
        let mut base = 0u32;
        for shard in finals {
            debug_assert_eq!(
                shard.output.vertex_incidence.len(),
                shard.output.vertices.len(),
                "vertex incidence out of sync with positions"
            );
            for (local, &count) in shard.output.vertex_incidence.iter().enumerate() {
                used_vertices += usize::from(count != 0);
                low_incidence |= count == 1 || count == 2;
                if count == 1 || count == 2 {
                    low_incidence_vertices.push(base + local as u32);
                }
            }
            base += shard.output.vertex_incidence.len() as u32;
        }
    } else {
        for shard in finals {
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
    }
    super::IncidenceSummary {
        used_vertices,
        live_half_edges: total_cell_indices as usize,
        low_incidence,
        low_incidence_vertices,
    }
}

#[inline(always)]
fn scatter_cell_indices(
    finals: &[ShardFinal],
    cells: &[VoronoiCell],
    assignment: &super::BinAssignment,
    vertex_offsets: &[u32],
    total_cell_indices: u32,
    #[allow(unused_variables)] shard_order_spans: bool,
) -> (Vec<u32>, bool) {
    let num_bins = assignment.num_bins;
    let num_cells = assignment.generator_bin.len();
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
        if cells.is_empty() {
            debug_assert_eq!(total_cell_indices, 0);
        } else if !shard_order_spans {
            let last = cells.last().unwrap();
            debug_assert_eq!(cells[0].vertex_start(), 0, "prefix sum must start at 0");
            debug_assert!(
                cells
                    .windows(2)
                    .all(|window| window[0].vertex_start() <= window[1].vertex_start()),
                "prefix sum must be non-decreasing"
            );
            debug_assert_eq!(
                last.vertex_start() + last.vertex_count(),
                total_cell_indices as usize,
                "prefix sum final total mismatch"
            );
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
    // Capture slices by value so the parallel closure carries their data
    // pointers directly instead of reloading through owning Vec references.
    let bin_generators = assignment.bin_generators.as_slice();
    let scatter_by_shard = prefer_shard_order_scatter(bin_generators, num_cells);
    if scatter_by_shard {
        maybe_par_into_iter!(0..num_bins).for_each(move |bin| {
            let shard = &finals[bin];
            let generators = &bin_generators[bin];
            debug_assert_eq!(generators.len(), shard.output.cell_starts.len());
            debug_assert_eq!(generators.len(), shard.output.cell_counts.len());

            // Shard streams are local-id ordered. Their source reads stay
            // sequential; adaptive span assignment can make destinations
            // sequential as well.
            for (local_idx, &gen_idx) in generators.iter().enumerate() {
                let start = shard.output.cell_starts[local_idx] as usize;
                let count = shard.output.cell_counts[local_idx] as usize;
                let cell = &cells[gen_idx];
                let dst_start = cell.vertex_start();

                #[cfg(debug_assertions)]
                {
                    debug_assert_eq!(count, cell.vertex_count(), "cell count mismatch");
                    debug_assert!(gen_idx < num_cells, "generator index out of range");
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
                unsafe {
                    scatter_local_indices(
                        src,
                        cell_indices_ptr,
                        dst_start,
                        vertex_offsets[bin],
                        shard.output.vertices.len(),
                    );
                }
            }
        });
    } else {
        maybe_par_into_iter!(0..num_cells).for_each(move |gen_idx| {
            let (bin, local) = assignment.generator_bin_local(gen_idx);
            let bin = bin.as_usize();
            let shard = &finals[bin];
            let start = shard.output.cell_start(local) as usize;
            let cell = &cells[gen_idx];
            let count = cell.vertex_count();
            let dst_start = cell.vertex_start();

            #[cfg(debug_assertions)]
            {
                debug_assert!(bin < num_bins, "generator bin out of range");
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
            unsafe {
                scatter_local_indices(
                    src,
                    cell_indices_ptr,
                    dst_start,
                    vertex_offsets[bin],
                    shard.output.vertices.len(),
                );
            }
        });
    }

    #[cfg(not(debug_assertions))]
    unsafe {
        cell_indices.set_len(total_cell_indices as usize);
    }
    (cell_indices, scatter_by_shard)
}

#[inline(always)]
fn patch_reference_overrides(
    finals: &[ShardFinal],
    cells: &[VoronoiCell],
    cell_indices: &mut [u32],
    vertex_offsets: &[u32],
    num_bins: usize,
) {
    for shard in finals {
        for entry in &shard.output.reference_overrides {
            let cell = &cells[entry.source_cell as usize];
            debug_assert!((entry.source_offset as usize) < cell.vertex_count());
            let dst = cell.vertex_start() + entry.source_offset as usize;
            let owner_bin = entry.owner_bin.as_usize();
            debug_assert!(owner_bin < num_bins);
            debug_assert!(
                (entry.owner_local as usize) < finals[owner_bin].output.vertices.len(),
                "override local vertex index out of range"
            );
            cell_indices[dst] = vertex_offsets[owner_bin] + entry.owner_local;
        }
    }
}

struct ResolutionHints {
    edge_hint_cells: Vec<u32>,
    exact_zero_edge_candidates: Vec<(u32, u32)>,
    exact_zero_edge_hint_cell_count: usize,
}

#[inline(always)]
fn collect_resolution_hints(
    finals: &[ShardFinal],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> ResolutionHints {
    let mut edge_hint_cells = Vec::new();
    for shard in finals {
        edge_hint_cells.extend_from_slice(&shard.output.resolution_edge_hint_cells);
    }
    edge_hint_cells.sort_unstable();
    edge_hint_cells.dedup();
    let ConfirmedZeroEdgeHints {
        candidates: exact_zero_edge_candidates,
        hinted_cells: exact_zero_edge_hint_cells,
    } = confirm_exact_zero_edge_hints(finals, vertices, cells, cell_indices);
    ResolutionHints {
        edge_hint_cells,
        exact_zero_edge_candidates,
        exact_zero_edge_hint_cell_count: exact_zero_edge_hint_cells.len(),
    }
}

#[cfg(feature = "timing")]
fn shard_order_stats(bin_generators: &[Vec<usize>]) -> (u64, u64, u64) {
    bin_generators.iter().fold(
        (0u64, 0u64, 0u64),
        |(descents, pairs, abs_delta), generators| {
            let local_descents = generators.windows(2).filter(|w| w[1] < w[0]).count() as u64;
            let local_abs_delta = generators
                .windows(2)
                .map(|w| w[1].abs_diff(w[0]) as u64)
                .sum::<u64>();
            (
                descents + local_descents,
                pairs + generators.len().saturating_sub(1) as u64,
                abs_delta + local_abs_delta,
            )
        },
    )
}

pub(super) fn assemble_sharded_live_dedup(
    mut data: ShardedCellsData,
) -> Result<super::AssemblyResult, crate::VoronoiError> {
    let num_bins = data.assignment.num_bins;

    let t_bookkeeping = Timer::start();
    let CollectedShardBookkeeping {
        mut edge_mismatches,
        edge_check_overflow,
        deferred_slots,
    } = collect_shard_bookkeeping(&mut data.shards);
    #[allow(unused_variables)]
    let bookkeeping_time = t_bookkeeping.elapsed();

    let t_overflow = Timer::start();
    let overflow_timing =
        resolve_edge_check_overflow(&mut data.shards, &edge_check_overflow, &mut edge_mismatches);
    #[allow(unused_variables)]
    let edge_check_overflow_time = t_overflow.elapsed();
    // Keep the existing nested measurements live for profiling builds even
    // though this attribution reports the enclosing wall-clock phase.
    let _overflow_detail_time = overflow_timing.sort + overflow_timing.match_;

    // `ComputeReport` already records a clean result, so keep that path free
    // of both the environment lookup and an all-zero diagnostic line.
    if !edge_mismatches.is_empty() {
        telemetry::maybe_emit_edge_mismatch_origins(&edge_mismatches);
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
    for shard in &mut data.shards {
        shard.output.finish_reference_patching();
    }
    #[allow(unused_variables)]
    let deferred_fallback_time = t_deferred.elapsed();

    // Convert to ShardFinal, dropping dedup structures to reduce memory pressure
    let t_finalize = Timer::start();
    let mut finals: Vec<ShardFinal> = std::mem::take(&mut data.shards)
        .into_iter()
        .map(|s| s.into_final())
        .collect();
    #[allow(unused_variables)]
    let finalize_shards_time = t_finalize.elapsed();

    let t2 = Timer::start();
    let ConcatenatedVertices {
        positions: all_vertices,
        keys: all_vertex_keys,
        offsets: vertex_offsets,
    } = concatenate_vertices(&mut finals, num_bins)?;
    #[allow(unused_variables)]
    let concat_vertices_time = t2.elapsed();
    #[cfg(feature = "parallel")]
    let shard_order_spans = rayon::current_num_threads() > 1
        && data.assignment.generator_bin.len() >= 65_536
        && prefer_shard_order_scatter(
            &data.assignment.bin_generators,
            data.assignment.generator_bin.len(),
        );
    #[cfg(not(feature = "parallel"))]
    let shard_order_spans = false;
    let t_cell_prefixes = Timer::start();
    let CellPrefixes {
        cells,
        total_indices: total_cell_indices,
    } = emit_cell_prefixes(&finals, &data.assignment, shard_order_spans)?;
    #[allow(unused_variables)]
    let emit_cell_prefixes_time = t_cell_prefixes.elapsed();

    let t_incidence = Timer::start();
    let incidence_summary =
        summarize_incidence(&finals, total_cell_indices, !edge_mismatches.is_empty());
    #[allow(unused_variables)]
    let incidence_summary_time = t_incidence.elapsed();

    let t_cell_indices = Timer::start();
    #[allow(unused_variables)]
    let (mut cell_indices, scatter_by_shard) = scatter_cell_indices(
        &finals,
        &cells,
        &data.assignment,
        &vertex_offsets,
        total_cell_indices,
        shard_order_spans,
    );
    #[allow(unused_variables)]
    let scatter_cell_indices_time = t_cell_indices.elapsed();

    #[cfg(feature = "timing")]
    let (shard_order_descents, shard_order_pairs, shard_order_abs_delta) =
        shard_order_stats(&data.assignment.bin_generators);

    // The common scatter above reads one narrow local id and performs no
    // owner branch. Patch the sparse foreign references by their final cell
    // identity after all disjoint bulk writes have completed.
    let t_overrides = Timer::start();
    patch_reference_overrides(
        &finals,
        &cells,
        &mut cell_indices,
        &vertex_offsets,
        num_bins,
    );
    #[allow(unused_variables)]
    let patch_reference_overrides_time = t_overrides.elapsed();

    #[cfg(debug_assertions)]
    debug_assert!(
        !cell_indices.contains(&u32::MAX),
        "unresolved foreign cell reference after sparse patch"
    );

    let t_zero_hints = Timer::start();
    let ResolutionHints {
        edge_hint_cells: resolution_edge_hint_cells,
        exact_zero_edge_candidates,
        exact_zero_edge_hint_cell_count,
    } = collect_resolution_hints(&finals, &all_vertices, &cells, &cell_indices);
    #[allow(unused_variables)]
    let exact_zero_hints_time = t_zero_hints.elapsed();

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
    #[cfg(feature = "timing")]
    let sub_phases = DedupSubPhases {
        bookkeeping: bookkeeping_time,
        edge_check_overflow: edge_check_overflow_time,
        edge_check_overflow_sort: overflow_timing.sort,
        edge_check_overflow_match: overflow_timing.match_,
        edge_check_overflow_records: edge_check_overflow.len() as u64,
        deferred_patching: deferred_fallback_time,
        finalize_shards: finalize_shards_time,
        concat_vertices: concat_vertices_time,
        emit_cell_prefixes: emit_cell_prefixes_time,
        incidence_summary: incidence_summary_time,
        scatter_cell_indices: scatter_cell_indices_time,
        patch_reference_overrides: patch_reference_overrides_time,
        exact_zero_hints: exact_zero_hints_time,
        shard_order_descents,
        shard_order_pairs,
        shard_order_abs_delta,
        scatter_by_shard,
        triplet_keys: finals.iter().map(|s| s.triplet_keys).sum(),
        edge_mismatches_count: edge_mismatches.len() as u64,
        primary_cell_references: finals
            .iter()
            .map(|s| s.output.cell_indices.len() as u64)
            .sum(),
        reference_overrides: finals
            .iter()
            .map(|s| s.output.reference_overrides.len() as u64)
            .sum(),
    };

    #[cfg(not(feature = "timing"))]
    let sub_phases = DedupSubPhases;

    Ok(super::AssemblyResult {
        vertices: all_vertices,
        vertex_keys: all_vertex_keys,
        edge_mismatches,
        cells,
        cell_indices,
        exact_zero_edge_candidates,
        resolution_edge_hint_cells,
        exact_zero_edge_hint_cells: exact_zero_edge_hint_cell_count,
        resolution_drift_exceeded,
        incidence_summary,
        dedup_sub: sub_phases,
    })
}

#[cfg(test)]
mod tests;
