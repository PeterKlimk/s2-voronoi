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

pub(super) type AssembledLiveDedup = (
    Vec<glam::Vec3>,
    Vec<VertexKey>,
    Vec<BadEdgeRecord>,
    Vec<crate::VoronoiCell>,
    Vec<u32>,
    super::timing::DedupSubPhases,
);

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

pub(super) fn assemble_sharded_live_dedup(
    data: ShardedCellsData,
) -> AssembledLiveDedup {
    assemble::assemble_sharded_live_dedup(data)
}
