//! The geometry-agnostic emission seam: per-cell shard emission shared by
//! the spherical (`knn_clipping::driver`) and planar
//! (`plane_clipping::driver`) drivers.

use glam::Vec3;

use super::binning::BinAssignment;
use super::edge_checks::{
    collect_and_resolve_cell_edges, thirds_for_emit, OUTGOING_NONE, OUTGOING_OVERFLOW_SIDE0,
    OUTGOING_OVERFLOW_SIDE1,
};
use super::packed::{pack_edge, pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::ShardState;
use super::types::{BinId, DeferredSlot, EdgeCheck, EdgeCheckOverflow, LocalId};
use super::{BuildCellsError, CellOutputBuffer, VertexData};

#[inline(always)]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // unordered (NaN) must fail the certificate
fn exceeds_resolution_drift<P: super::types::VertexPosition>(representative: P, local: P) -> bool {
    let delta = representative.resolution_axis_delta(local);
    // `resolution_axis_delta` is non-negative. An ordered `<=` accepts exactly
    // the finite in-range values; negating it also rejects NaN and infinity
    // without a separate floating-point classification.
    !(delta <= f64::from(crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS))
}

pub(crate) struct EdgeScratch {
    outgoing: Vec<u32>,
    vertex_indices: Vec<u32>,
}

#[inline(always)]
fn assert_endpoint_lengths<P>(cell_vertices: &[VertexData<P>], vertex_indices_len: usize) -> usize {
    let vertex_count = cell_vertices.len();
    assert_eq!(
        vertex_indices_len, vertex_count,
        "edge endpoint arrays out of sync"
    );
    vertex_count
}

impl EdgeScratch {
    pub(crate) fn new() -> Self {
        Self {
            outgoing: Vec::new(),
            vertex_indices: Vec::new(),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    fn collect_and_resolve<P: super::types::VertexPosition>(
        &mut self,
        cell_idx: u32,
        shard_ctx: &mut ShardContext<'_, P>,
        output_buffer: &CellOutputBuffer<P>,
        slot_points: &[crate::cube_grid::SlotPoint],
        assignment: &BinAssignment,
        incoming_checks: Vec<EdgeCheck>,
    ) {
        self.vertex_indices.clear();
        self.vertex_indices
            .resize(output_buffer.vertices.len(), INVALID_INDEX);
        self.outgoing.clear();
        self.outgoing
            .resize(output_buffer.vertices.len(), OUTGOING_NONE);
        collect_and_resolve_cell_edges(
            cell_idx,
            shard_ctx,
            output_buffer,
            slot_points,
            assignment,
            incoming_checks,
            &mut self.vertex_indices,
            &mut self.outgoing,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn emit_outgoing_edge<P: super::types::VertexPosition>(
    tag: u32,
    shard: &mut ShardState<P>,
    cell_idx: u32,
    neighbor: u32,
    cell_slot: u32,
    cell_start: u32,
    edge_local: usize,
    vertex_count: usize,
    endpoint_keys: [crate::knn_clipping::cell_build::VertexKey; 2],
    endpoint_indices: [u32; 2],
    bin: BinId,
    keys_verified: bool,
) {
    if tag == OUTGOING_NONE {
        return;
    }
    let key = pack_edge(cell_idx, neighbor);
    let thirds = thirds_for_emit(
        keys_verified,
        &mut shard.output.unresolved_edges,
        key,
        endpoint_keys,
    );
    if tag == OUTGOING_OVERFLOW_SIDE0 || tag == OUTGOING_OVERFLOW_SIDE1 {
        let next = if edge_local + 1 == vertex_count {
            0
        } else {
            edge_local + 1
        };
        shard.output.edge_check_overflow.push(EdgeCheckOverflow {
            key,
            side: u8::from(tag == OUTGOING_OVERFLOW_SIDE1),
            source_bin: bin,
            thirds,
            indices: endpoint_indices,
            slots: [cell_start + edge_local as u32, cell_start + next as u32],
        });
    } else {
        shard.dedup.push_edge_check(
            LocalId::from(tag),
            EdgeCheck {
                neighbor_slot: cell_slot,
                thirds,
                indices: endpoint_indices,
            },
        );
    }
}

pub(crate) struct ShardContext<'a, P = Vec3> {
    pub(crate) shard: &'a mut ShardState<P>,
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
pub(crate) fn checked_u8(value: usize, context: &str) -> Result<u8, BuildCellsError> {
    u8::try_from(value).map_err(|_| representation_limit(format!("{context} exceeds u8 capacity")))
}

#[inline]
pub(crate) fn checked_local_id(value: usize, context: &str) -> Result<LocalId, BuildCellsError> {
    checked_u32(value, context).map(LocalId::from)
}

/// Emit one built cell's output into its shard: resolve/record edge checks,
/// dedup vertices by owner bin (deferring off-shard owners), and forward
/// edge checks to later cells. Geometry-free; shared by the spherical and
/// planar drivers.
#[allow(clippy::too_many_arguments)] // internal seam shared by two drivers
pub(crate) fn emit_cell_output<P: super::types::VertexPosition>(
    cell_sub: &mut crate::timing::CellSubAccum,
    scratch: &mut EdgeScratch,
    shard_ctx: &mut ShardContext<'_, P>,
    assignment: &BinAssignment,
    cell_idx: u32,
    cell_slot: u32,
    cell_start: u32,
    output_buffer: &CellOutputBuffer<P>,
    slot_points: &[crate::cube_grid::SlotPoint],
    incoming_checks: Vec<EdgeCheck>,
) -> Result<(), BuildCellsError> {
    debug_assert_eq!(
        slot_points[cell_slot as usize].idx, cell_idx,
        "forwarded edge-check slot must identify its source generator"
    );
    let mut t_post = crate::timing::LapTimer::start();
    scratch.collect_and_resolve(
        cell_idx,
        shard_ctx,
        output_buffer,
        slot_points,
        assignment,
        incoming_checks,
    );
    let collect_resolve_time = t_post.lap();
    cell_sub.add_edge_collect(collect_resolve_time / 2);
    cell_sub.add_edge_resolve(collect_resolve_time / 2);

    let count = assert_endpoint_lengths(&output_buffer.vertices, scratch.vertex_indices.len());
    let shard = &mut *shard_ctx.shard;
    let local = shard_ctx.local;
    let bin = shard_ctx.bin;

    let cell_count = checked_u8(count, "cell vertex count")?;
    shard.output.set_cell_count(local, cell_count);

    {
        let vertex_indices = &mut scratch.vertex_indices;
        let mut first_key = [0u32; 3];
        let mut first_index = INVALID_INDEX;
        let mut previous_key = [0u32; 3];
        let mut previous_index = INVALID_INDEX;
        for (i, ((key, pos), vi)) in output_buffer
            .vertices
            .iter()
            .copied()
            .zip(vertex_indices.iter_mut())
            .enumerate()
        {
            #[cfg(feature = "timing")]
            {
                shard.triplet_keys += 1;
            }
            // Native AVX2 codegen benefits from testing the resolved index
            // before the owner-map load. Generic x86 codegen regresses badly
            // from this branch layout, so retain its owner-first shape below.
            #[cfg(target_feature = "avx2")]
            let needs_owner_lookup = {
                // A resolved index is necessarily local to this shard: in-bin
                // edge checks carry shard-local ids, while cross-bin/deferred
                // endpoints retain INVALID_INDEX until assembly.
                if *vi != INVALID_INDEX {
                    debug_assert!(
                        (*vi as usize) < shard.output.vertices.len(),
                        "resolved vertex index outside its shard"
                    );
                    let representative =
                        unsafe { *shard.output.vertices.get_unchecked(*vi as usize) };
                    shard.output.resolution_drift_exceeded |=
                        exceeds_resolution_drift(representative, pos);
                    shard.output.add_vertex_incidence(*vi);
                    shard.output.cell_indices.push(pack_ref(bin, *vi));
                    false
                } else {
                    true
                }
            };
            #[cfg(not(target_feature = "avx2"))]
            let needs_owner_lookup = true;

            if needs_owner_lookup {
                let owner_bin = assignment.generator_bin[key[0] as usize];
                if owner_bin == bin {
                    #[cfg(target_feature = "avx2")]
                    {
                        let new_idx =
                            checked_u32(shard.output.vertices.len(), "shard vertex index")?;
                        shard.output.vertices.push(pos);
                        shard.output.vertex_keys.push(key);
                        shard.output.vertex_incidence.push(1);
                        *vi = new_idx;
                    }
                    #[cfg(not(target_feature = "avx2"))]
                    if *vi == INVALID_INDEX {
                        let new_idx =
                            checked_u32(shard.output.vertices.len(), "shard vertex index")?;
                        shard.output.vertices.push(pos);
                        shard.output.vertex_keys.push(key);
                        shard.output.vertex_incidence.push(1);
                        *vi = new_idx;
                    } else {
                        let representative =
                            unsafe { *shard.output.vertices.get_unchecked(*vi as usize) };
                        shard.output.resolution_drift_exceeded |=
                            exceeds_resolution_drift(representative, pos);
                        shard.output.add_vertex_incidence(*vi);
                    }
                    let v_idx = *vi;
                    debug_assert_ne!(v_idx, INVALID_INDEX, "missing on-shard vertex index");
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

            let current_index = *vi;
            if i == 0 {
                first_key = key;
                first_index = current_index;
            } else {
                emit_outgoing_edge(
                    scratch.outgoing[i - 1],
                    shard,
                    cell_idx,
                    output_buffer.edge_neighbor_globals[i - 1],
                    cell_slot,
                    cell_start,
                    i - 1,
                    count,
                    [previous_key, key],
                    [previous_index, current_index],
                    bin,
                    output_buffer.edge_keys_verified,
                );
            }
            previous_key = key;
            previous_index = current_index;
        }
        emit_outgoing_edge(
            scratch.outgoing[count - 1],
            shard,
            cell_idx,
            output_buffer.edge_neighbor_globals[count - 1],
            cell_slot,
            cell_start,
            count - 1,
            count,
            [previous_key, first_key],
            [previous_index, first_index],
            bin,
            output_buffer.edge_keys_verified,
        );
    }
    let dedup_emit_time = t_post.lap();
    // Vertex dedup and outgoing edge emission are deliberately interleaved so
    // an edge is forwarded as soon as its second endpoint index is known.
    cell_sub.add_key_dedup(dedup_emit_time / 2);
    cell_sub.add_edge_emit(dedup_emit_time / 2);

    debug_assert_eq!(
        shard.output.cell_indices.len() as u32 - cell_start,
        count as u32,
        "cell index stream mismatch"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        assert_endpoint_lengths, checked_local_id, checked_u32, checked_u8,
        exceeds_resolution_drift, BuildCellsError,
    };
    use glam::Vec3;

    #[test]
    fn resolution_drift_guard_is_inclusive_and_rejects_non_finite() {
        let eps = crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS;
        let origin = Vec3::ZERO;
        assert!(!exceeds_resolution_drift(
            origin,
            Vec3::new(eps, 100.0, -100.0)
        ));
        assert!(exceeds_resolution_drift(
            origin,
            Vec3::new(f32::from_bits(eps.to_bits() + 1), 0.0, 0.0)
        ));
        assert!(exceeds_resolution_drift(
            origin,
            Vec3::new(f32::NAN, 0.0, 0.0)
        ));
        assert!(exceeds_resolution_drift(
            origin,
            Vec3::new(f32::INFINITY, 0.0, 0.0)
        ));
        assert!(!exceeds_resolution_drift(
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(-0.0, -1.0, -1.0)
        ));

        // At 0.5, 16 upward f32 ULPs remain inside the bound and 17 are
        // outside. This pins the f64-over-stored-f32 subtraction away from
        // zero, where cancellation could otherwise obscure the boundary.
        let offset = 0.5f32;
        let inside = f32::from_bits(offset.to_bits() + 16);
        let outside = f32::from_bits(offset.to_bits() + 17);
        assert!(!exceeds_resolution_drift(
            Vec3::new(offset, 0.0, 0.0),
            Vec3::new(inside, 0.0, 0.0)
        ));
        assert!(exceeds_resolution_drift(
            Vec3::new(offset, 0.0, 0.0),
            Vec3::new(outside, 0.0, 0.0)
        ));
    }

    #[test]
    #[should_panic(expected = "edge endpoint arrays out of sync")]
    fn endpoint_length_mismatch_panics_before_emit() {
        let vertices = [([0, 1, 2], glam::Vec3::ZERO)];
        assert_endpoint_lengths(&vertices, 0);
    }

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
