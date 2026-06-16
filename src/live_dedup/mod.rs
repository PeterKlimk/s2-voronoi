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
pub(crate) use packed::pack_edge;
pub(crate) use shard::ShardState;
pub(crate) use types::BinId;
pub use types::UnresolvedEdgeOrigin;
pub(crate) use types::{EdgeKey, EdgeRecord, UnresolvedEdgeMismatch, VertexPosition};

/// Per-shard vertex keys kept un-concatenated.
///
/// `vertex_keys` is consumed only by edge reconciliation, which — after the
/// same-key dup-scan was localized — touches keys for at most the defect
/// region (typically nothing). Concatenating all shards' keys into one flat
/// array is then O(V) copy + a 12·V-byte allocation that is essentially never
/// read. Instead we **move** the per-shard key vecs here (O(num_bins), zero
/// copy) and look up the handful reconciliation needs by `(bin, local)`.
pub(crate) struct ShardedVertexKeys {
    /// Prefix-sum starts, length `num_bins + 1`; bin `b` owns global vertex
    /// ids `[offsets[b], offsets[b+1])`.
    offsets: Vec<u32>,
    /// Per-bin key arrays in global slot order, moved out of the shards.
    shards: Vec<Vec<VertexKey>>,
}

impl ShardedVertexKeys {
    /// `offsets` is the prefix sum (len `shards.len() + 1`) of per-shard key
    /// counts; `shards[b].len() == offsets[b+1] - offsets[b]`.
    pub(crate) fn new(offsets: Vec<u32>, shards: Vec<Vec<VertexKey>>) -> Self {
        debug_assert_eq!(offsets.len(), shards.len() + 1, "offsets/shards length");
        Self { offsets, shards }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0) as usize
    }

    /// Key for global vertex id `vid`, or `None` if out of range. The bin is
    /// found by `partition_point` on the prefix-sum offsets, which is correct
    /// even with empty bins (duplicate offsets); `num_bins` is ~2× threads so
    /// the search is trivial.
    #[inline]
    pub(crate) fn get(&self, vid: u32) -> Option<VertexKey> {
        if self.shards.is_empty() || vid as usize >= self.len() {
            return None;
        }
        let bin = self.offsets.partition_point(|&o| o <= vid) - 1;
        let local = (vid - self.offsets[bin]) as usize;
        self.shards[bin].get(local).copied()
    }

    /// Visit every `(global_vid, key)` in global slot order (used only by the
    /// global-scan escape path and the debug oracle).
    pub(crate) fn for_each(&self, mut f: impl FnMut(u32, VertexKey)) {
        for (bin, keys) in self.shards.iter().enumerate() {
            let base = self.offsets[bin];
            for (local, &k) in keys.iter().enumerate() {
                f(base + local as u32, k);
            }
        }
    }
}

/// Result of assembling sharded live-dedup data into global arrays.
pub(crate) struct AssemblyResult<P = glam::Vec3> {
    /// All Voronoi vertex positions (global, concatenated from shards).
    pub vertices: Vec<P>,
    /// Vertex keys (triplet of generator indices), parallel to `vertices`, kept
    /// un-concatenated — see `ShardedVertexKeys`.
    pub vertex_keys: ShardedVertexKeys,
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

#[cfg(test)]
mod sharded_keys_tests {
    use super::ShardedVertexKeys;

    #[test]
    fn get_handles_empty_bins_and_bounds() {
        // bins: [k0,k1] | (empty) | [k2]  -> offsets are a prefix sum with a
        // duplicate (the empty bin), the case partition_point must skip.
        let keys = ShardedVertexKeys::new(
            vec![0, 2, 2, 3],
            vec![vec![[0, 0, 0], [1, 1, 1]], vec![], vec![[2, 2, 2]]],
        );
        assert_eq!(keys.len(), 3);
        assert_eq!(keys.get(0), Some([0, 0, 0]));
        assert_eq!(keys.get(1), Some([1, 1, 1]));
        // vid 2 lives in bin 2, NOT the empty bin 1 with the same offset.
        assert_eq!(keys.get(2), Some([2, 2, 2]));
        assert_eq!(keys.get(3), None); // out of range
        assert_eq!(keys.get(99), None);
    }

    #[test]
    fn get_matches_a_flat_concat() {
        // Differential: sharded lookup must equal the flat concatenation it
        // replaces, for every global vid (incl. across a trailing empty bin).
        let shards = vec![
            vec![[10, 0, 0], [11, 0, 0], [12, 0, 0]],
            vec![[20, 0, 0]],
            vec![],
            vec![[30, 0, 0], [31, 0, 0]],
        ];
        let mut offsets = vec![0u32];
        let mut flat = Vec::new();
        for s in &shards {
            flat.extend_from_slice(s);
            offsets.push(flat.len() as u32);
        }
        let keys = ShardedVertexKeys::new(offsets, shards);
        assert_eq!(keys.len(), flat.len());
        for (vid, &k) in flat.iter().enumerate() {
            assert_eq!(keys.get(vid as u32), Some(k), "vid {vid}");
        }
        assert_eq!(keys.get(flat.len() as u32), None);
    }
}
