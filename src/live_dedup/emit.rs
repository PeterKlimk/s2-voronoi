//! The geometry-agnostic emission seam: per-cell shard emission shared by
//! the spherical (`knn_clipping::driver`) and planar
//! (`plane_clipping::driver`) drivers.

use glam::Vec3;

use super::binning::BinAssignment;
use super::edge_checks::collect_and_resolve_cell_edges;
use super::packed::{pack_ref, DEFERRED, INVALID_INDEX};
use super::shard::ShardState;
use super::types::{
    BinId, DeferredSlot, EdgeCheck, EdgeCheckOverflow, EdgeOverflowLocal, EdgeToLater, LocalId,
};
use super::{BuildCellsError, CellOutputBuffer, VertexData};

#[inline(always)]
fn exceeds_resolution_drift<P: super::types::VertexPosition>(representative: P, local: P) -> bool {
    let delta = representative.resolution_axis_delta(local);
    !delta.is_finite()
        || delta > f64::from(crate::tolerances::OUTPUT_RESOLUTION_REPRESENTATIVE_X_EPS)
}

pub(crate) struct EdgeScratch {
    edges_to_later: Vec<EdgeToLater>,
    edges_overflow: Vec<EdgeOverflowLocal>,
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
            edges_to_later: Vec::new(),
            edges_overflow: Vec::new(),
            vertex_indices: Vec::new(),
        }
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    fn collect_and_resolve<P: super::types::VertexPosition>(
        &mut self,
        cell_idx: u32,
        shard_ctx: &mut ShardContext<'_, P>,
        output_buffer: &CellOutputBuffer<P>,
        assignment: &BinAssignment,
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
    fn emit<P: super::types::VertexPosition>(
        &mut self,
        shard: &mut ShardState<P>,
        cell_vertices: &[VertexData<P>],
        cell_idx: u32,
        cell_start: u32,
        bin: BinId,
        keys_verified: bool,
    ) {
        use super::edge_checks::thirds_for_emit;

        let vertex_count = assert_endpoint_lengths(cell_vertices, self.vertex_indices.len());

        // These scratch records are Copy and own no resources. Iterating by
        // copy avoids Drain's per-element/unwind bookkeeping; successful
        // emission clears the reusable buffer below.
        for entry in self.edges_to_later.iter().copied() {
            let locals = entry.locals;
            let a = locals[0] as usize;
            let b = locals[1] as usize;
            debug_assert!(a < vertex_count && b < vertex_count);
            // The sole record producer creates both locals from `i` and its
            // cyclic successor in `0..vertex_count`; lengths were checked once
            // above. Keep repeated bounds checks out of the forwarding loops.
            let keys = unsafe {
                [
                    cell_vertices.get_unchecked(a).0,
                    cell_vertices.get_unchecked(b).0,
                ]
            };
            let indices = unsafe {
                [
                    *self.vertex_indices.get_unchecked(a),
                    *self.vertex_indices.get_unchecked(b),
                ]
            };
            let thirds = thirds_for_emit(
                keys_verified,
                &mut shard.output.unresolved_edges,
                entry.key,
                keys,
            );
            shard.dedup.push_edge_check(
                entry.local_b,
                EdgeCheck {
                    neighbor_idx: cell_idx,
                    thirds,
                    indices,
                },
            );
        }
        self.edges_to_later.clear();

        for entry in self.edges_overflow.iter().copied() {
            let locals = entry.locals;
            let a = locals[0] as usize;
            let b = locals[1] as usize;
            debug_assert!(a < vertex_count && b < vertex_count);
            // Same producer/range proof as the ordinary forwarding loop.
            let keys = unsafe {
                [
                    cell_vertices.get_unchecked(a).0,
                    cell_vertices.get_unchecked(b).0,
                ]
            };
            let indices = unsafe {
                [
                    *self.vertex_indices.get_unchecked(a),
                    *self.vertex_indices.get_unchecked(b),
                ]
            };
            let thirds = thirds_for_emit(
                keys_verified,
                &mut shard.output.unresolved_edges,
                entry.key,
                keys,
            );
            shard.output.edge_check_overflow.push(EdgeCheckOverflow {
                key: entry.key,
                side: entry.side,
                source_bin: bin,
                thirds,
                indices,
                slots: [cell_start + locals[0] as u32, cell_start + locals[1] as u32],
            });
        }
        self.edges_overflow.clear();
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
    cell_start: u32,
    output_buffer: &CellOutputBuffer<P>,
    incoming_checks: Vec<EdgeCheck>,
) -> Result<(), BuildCellsError> {
    let mut t_post = crate::timing::LapTimer::start();
    scratch.collect_and_resolve(
        cell_idx,
        shard_ctx,
        output_buffer,
        assignment,
        incoming_checks,
    );
    let collect_resolve_time = t_post.lap();
    cell_sub.add_edge_collect(collect_resolve_time / 2);
    cell_sub.add_edge_resolve(collect_resolve_time / 2);

    let count = output_buffer.vertices.len();
    let shard = &mut *shard_ctx.shard;
    let local = shard_ctx.local;
    let bin = shard_ctx.bin;

    let cell_count = checked_u8(count, "cell vertex count")?;
    shard.output.set_cell_count(local, cell_count);

    {
        let vertex_indices = &mut scratch.vertex_indices;
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
            // Native AVX2 codegen benefits from testing the resolved index
            // before the owner-map load. Generic x86 codegen regresses badly
            // from this branch layout, so retain its owner-first shape below.
            #[cfg(target_feature = "avx2")]
            {
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
                    continue;
                }
            }

            let owner_bin = assignment.generator_bin[key[0] as usize];
            if owner_bin == bin {
                #[cfg(target_feature = "avx2")]
                {
                    let new_idx = checked_u32(shard.output.vertices.len(), "shard vertex index")?;
                    shard.output.vertices.push(pos);
                    shard.output.vertex_keys.push(key);
                    shard.output.vertex_incidence.push(1);
                    *vi = new_idx;
                }
                #[cfg(not(target_feature = "avx2"))]
                if *vi == INVALID_INDEX {
                    let new_idx = checked_u32(shard.output.vertices.len(), "shard vertex index")?;
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
    }
    cell_sub.add_key_dedup(t_post.lap());

    scratch.emit(
        shard,
        &output_buffer.vertices,
        cell_idx,
        cell_start,
        bin,
        output_buffer.edge_keys_verified,
    );
    cell_sub.add_edge_emit(t_post.lap());

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
