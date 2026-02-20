//! Live vertex deduplication during cell construction using sharded ownership.
//!
//! V1 design:
//! - Parallel cell building by spatial bin
//! - Single-threaded overflow flush (simplifies correctness)
//! - Per-cell duplicate index checks handled by validation (not in hot path)

use super::cell_builder::VertexKey;
use super::TerminationConfig;

mod assemble;
mod binning;
mod build;
mod edge_checks;
mod packed;
mod shard;
mod types;

use binning::BinAssignment;
use shard::ShardState;
use types::BadEdgeRecord;

pub(super) use types::EdgeRecord;

/// Result of assembling sharded live-dedup data into global arrays.
pub(super) struct AssemblyResult {
    /// All Voronoi vertex positions (global, concatenated from shards).
    pub vertices: Vec<glam::Vec3>,
    /// Vertex keys (triplet of generator indices), parallel to `vertices`.
    pub vertex_keys: Vec<VertexKey>,
    /// Edges that could not be fully resolved during dedup.
    pub bad_edges: Vec<BadEdgeRecord>,
    /// Per-cell storage (one per generator).
    pub cells: Vec<crate::VoronoiCell>,
    /// Flattened vertex indices for all cells.
    pub cell_indices: Vec<u32>,
    /// Timing sub-phases for the dedup stage.
    pub dedup_sub: super::timing::DedupSubPhases,
}

pub(super) struct ShardedCellsData {
    assignment: BinAssignment,
    shards: Vec<ShardState>,
    pub(super) cell_sub: super::timing::CellSubAccum,
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

pub(super) fn build_cells_sharded_live_dedup(
    points: &[glam::Vec3],
    grid: &crate::cube_grid::CubeMapGrid,
    termination: TerminationConfig,
) -> ShardedCellsData {
    build::build_cells_sharded_live_dedup(points, grid, termination)
}

pub(super) fn assemble_sharded_live_dedup(data: ShardedCellsData) -> AssemblyResult {
    assemble::assemble_sharded_live_dedup(data)
}
