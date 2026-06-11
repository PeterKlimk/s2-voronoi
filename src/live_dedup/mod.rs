//! Live vertex deduplication during cell construction using sharded ownership.
//!
//! V1 design:
//! - Parallel cell building by spatial bin
//! - Single-threaded overflow flush (simplifies correctness)
//! - Per-cell duplicate index checks handled by validation (not in hot path)

use crate::diagram::VoronoiCell;

mod assemble;
mod binning;
mod cell_output;
mod edge_checks;
mod emit;
mod packed;
mod shard;
mod types;

pub use cell_output::{CellBuildError, CellFailure, CellOutputBuffer, VertexData, VertexKey};

pub(crate) use binning::BinAssignment;
pub(crate) use binning::PackedLayoutCapacityError;
pub(crate) use binning::{assign_bins, assign_bins_with, target_bin_count};
pub(crate) use edge_checks::unpack_edge_key;
pub(crate) use emit::{checked_local_id, checked_u32, emit_cell_output, EdgeScratch, ShardContext};
pub(crate) use shard::ShardState;
pub(crate) use types::BinId;
pub(crate) use types::{EdgeRecord, UnresolvedEdgeMismatch, VertexPosition};

/// Result of assembling sharded live-dedup data into global arrays.
pub(crate) struct AssemblyResult<P = glam::Vec3> {
    /// All Voronoi vertex positions (global, concatenated from shards).
    pub vertices: Vec<P>,
    /// Vertex keys (triplet of generator indices), parallel to `vertices`.
    pub vertex_keys: Vec<VertexKey>,
    /// Unresolved shared-edge mismatches that survived live dedup assembly.
    ///
    /// Assembly first tries to reconcile cross-bin edge checks and patch deferred vertex slots.
    /// Entries that remain here feed the narrow post-pass reconciliation in
    /// `edge_reconcile.rs`; they are not a generic record of arbitrary topology failures.
    pub unresolved_edges: Vec<UnresolvedEdgeMismatch>,
    /// Per-cell storage (one per generator).
    pub cells: Vec<VoronoiCell>,
    /// Flattened vertex indices for all cells.
    pub cell_indices: Vec<u32>,
    /// Timing sub-phases for the dedup stage.
    pub dedup_sub: crate::timing::DedupSubPhases,
}

pub(crate) struct ShardedCellsData<P = glam::Vec3> {
    assignment: BinAssignment,
    shards: Vec<ShardState<P>>,
    pub(super) cell_sub: crate::timing::CellSubAccum,
}

impl<P: VertexPosition> ShardedCellsData<P> {
    /// Assemble from a geometry driver's output (the planar driver lives in
    /// `plane_clipping` and builds shards through the pub(crate) seam).
    pub(crate) fn from_parts(
        assignment: BinAssignment,
        shards: Vec<ShardState<P>>,
        cell_sub: crate::timing::CellSubAccum,
    ) -> Self {
        Self {
            assignment,
            shards,
            cell_sub,
        }
    }
}

pub(crate) enum BuildCellsError {
    CellBuild(CellBuildError),
    PackedLayoutCapacity(PackedLayoutCapacityError),
    RepresentationLimit(String),
}

fn with_two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);
    if i < j {
        let (a, b) = v.split_at_mut(j);
        (&mut a[i], &mut b[0])
    } else {
        let (a, b) = v.split_at_mut(i);
        (&mut b[0], &mut a[j])
    }
}

pub(crate) fn assemble_sharded_live_dedup<P: VertexPosition>(
    data: ShardedCellsData<P>,
) -> Result<AssemblyResult<P>, crate::VoronoiError> {
    assemble::assemble_sharded_live_dedup(data)
}
