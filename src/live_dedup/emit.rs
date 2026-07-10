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

pub(crate) struct EdgeScratch {
    edges_to_later: Vec<EdgeToLater>,
    edges_overflow: Vec<EdgeOverflowLocal>,
    vertex_indices: Vec<u32>,
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
        cell_start: u32,
        bin: BinId,
        keys_verified: bool,
    ) {
        use super::edge_checks::thirds_for_emit;

        for entry in self.edges_to_later.drain(..) {
            let locals = entry.locals;
            let thirds = thirds_for_emit(
                keys_verified,
                &mut shard.output.unresolved_edges,
                entry.key,
                [
                    cell_vertices[locals[0] as usize].0,
                    cell_vertices[locals[1] as usize].0,
                ],
            );
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
            let thirds = thirds_for_emit(
                keys_verified,
                &mut shard.output.unresolved_edges,
                entry.key,
                [
                    cell_vertices[locals[0] as usize].0,
                    cell_vertices[locals[1] as usize].0,
                ],
            );
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
            let owner_bin = assignment.generator_bin[key[0] as usize];
            if owner_bin == bin {
                if *vi == INVALID_INDEX {
                    let new_idx = checked_u32(shard.output.vertices.len(), "shard vertex index")?;
                    shard.output.vertices.push(pos);
                    shard.output.vertex_keys.push(key);
                    *vi = new_idx;
                }
                let v_idx = *vi;
                debug_assert!(v_idx != INVALID_INDEX, "missing on-shard vertex index");
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
    use super::{checked_local_id, checked_u32, checked_u8, BuildCellsError};

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
