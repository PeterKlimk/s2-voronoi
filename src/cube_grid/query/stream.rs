use crate::cube_grid::packed_knn::{
    DirectedCellGroup, PackedKnnCellScratch, PackedKnnTimings, PackedStage,
};

use super::directed::DirectedNoKCursor;
use super::{CubeMapGrid, CubeMapGridScratch, DirectedCtx};
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectedNeighborBatchSource {
    PackedChunk0,
    PackedTail,
    PackedExhausted,
    DirectedCursor,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectedNeighborBatch {
    pub(crate) n: usize,
    pub(crate) unseen_bound: f32,
    pub(crate) source: DirectedNeighborBatchSource,
}

pub(crate) struct PackedQuery<'a, 'g> {
    scratch: &'a mut PackedKnnCellScratch,
    timings: &'a mut PackedKnnTimings,
    group: DirectedCellGroup<'g>,
    query_index: usize,
    chunk0_size: usize,
    chunk_size: usize,
}

impl<'a, 'g> PackedQuery<'a, 'g> {
    pub(crate) fn new(
        scratch: &'a mut PackedKnnCellScratch,
        timings: &'a mut PackedKnnTimings,
        group: DirectedCellGroup<'g>,
        query_index: usize,
        chunk0_size: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            scratch,
            timings,
            group,
            query_index,
            chunk0_size,
            chunk_size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStage {
    PackedChunk0,
    PackedTail,
    Cursor,
    Done,
}

pub(crate) struct DirectedNeighborStream<'a, 'm, 'p> {
    grid: &'a CubeMapGrid,
    cursor: DirectedNoKCursor<'a, 'm>,
    packed: Option<PackedQuery<'p, 'm>>,
    stage: StreamStage,
    did_packed: bool,
    packed_tail_used: bool,
    packed_safe_exhausted: bool,
    used_cursor: bool,
    knn_exhausted: bool,
}

impl<'a, 'm, 'p> DirectedNeighborStream<'a, 'm, 'p> {
    pub(crate) fn new(
        grid: &'a CubeMapGrid,
        points: &'a [Vec3],
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        directed_ctx: DirectedCtx<'m>,
        packed: Option<PackedQuery<'p, 'm>>,
    ) -> Self {
        let cursor = grid.directed_no_k_cursor(points[query_idx], query_idx, scratch, directed_ctx);
        let did_packed = packed.is_some();
        let stage = if did_packed {
            StreamStage::PackedChunk0
        } else {
            StreamStage::Cursor
        };

        Self {
            grid,
            cursor,
            packed,
            stage,
            did_packed,
            packed_tail_used: false,
            packed_safe_exhausted: false,
            used_cursor: false,
            knn_exhausted: false,
        }
    }

    pub(crate) fn next_batch(&mut self, out: &mut Vec<u32>) -> Option<DirectedNeighborBatch> {
        out.clear();

        loop {
            match self.stage {
                StreamStage::PackedChunk0 | StreamStage::PackedTail => {
                    let packed = self
                        .packed
                        .as_mut()
                        .expect("packed stage requires packed query state");
                    let (stage, source, k) = match self.stage {
                        StreamStage::PackedChunk0 => (
                            PackedStage::Chunk0,
                            DirectedNeighborBatchSource::PackedChunk0,
                            packed.chunk0_size,
                        ),
                        StreamStage::PackedTail => (
                            PackedStage::Tail,
                            DirectedNeighborBatchSource::PackedTail,
                            packed.chunk_size,
                        ),
                        _ => unreachable!("unexpected packed stage"),
                    };

                    out.resize(k, u32::MAX);
                    if let Some(chunk) =
                        packed
                            .scratch
                            .next_chunk(packed.query_index, stage, k, out, packed.timings)
                    {
                        out.truncate(chunk.n);
                        return Some(DirectedNeighborBatch {
                            n: chunk.n,
                            unseen_bound: chunk.unseen_bound,
                            source,
                        });
                    }

                    if self.stage == StreamStage::PackedChunk0
                        && packed.scratch.tail_possible(packed.query_index)
                    {
                        packed.scratch.ensure_tail_directed_for(
                            packed.query_index,
                            self.grid,
                            packed.group.slot_gen_map(),
                            packed.group.local_shift(),
                            packed.group.local_mask(),
                            packed.timings,
                        );
                        self.stage = StreamStage::PackedTail;
                        self.packed_tail_used = true;
                        continue;
                    }

                    self.packed_safe_exhausted = true;
                    self.stage = StreamStage::Cursor;
                    return Some(DirectedNeighborBatch {
                        n: 0,
                        unseen_bound: packed.scratch.security(packed.query_index),
                        source: DirectedNeighborBatchSource::PackedExhausted,
                    });
                }
                StreamStage::Cursor => {
                    self.used_cursor = true;
                    if let Some(slot) = self.cursor.pop_next_proven_slot() {
                        out.push(slot);
                        return Some(DirectedNeighborBatch {
                            n: 1,
                            unseen_bound: self.cursor.remaining_dot_upper_bound(),
                            source: DirectedNeighborBatchSource::DirectedCursor,
                        });
                    }

                    self.knn_exhausted = self.cursor.is_exhausted();
                    self.stage = StreamStage::Done;
                    return None;
                }
                StreamStage::Done => return None,
            }
        }
    }

    #[inline]
    pub(crate) fn did_packed(&self) -> bool {
        self.did_packed
    }

    #[inline]
    pub(crate) fn packed_tail_used(&self) -> bool {
        self.packed_tail_used
    }

    #[inline]
    pub(crate) fn packed_safe_exhausted(&self) -> bool {
        self.packed_safe_exhausted
    }

    #[inline]
    pub(crate) fn knn_exhausted(&self) -> bool {
        self.knn_exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube_grid::packed_knn::{DirectedCellGroup, PackedKnnCellStatus};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const QUERY_BIN: u8 = 0;
    const LOCAL_SHIFT: u32 = 24;
    const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;

    fn random_unit_points(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n);
        while points.len() < n {
            let p = Vec3::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let len_sq = p.length_squared();
            if len_sq <= 1e-12 {
                continue;
            }
            points.push(p / len_sq.sqrt());
        }
        points
    }

    fn fullest_cell(grid: &CubeMapGrid) -> usize {
        let mut best_cell = 0usize;
        let mut best_len = 0usize;
        for cell in 0..(grid.cell_offsets().len() - 1) {
            let len = grid.cell_points(cell).len();
            if len > best_len {
                best_len = len;
                best_cell = cell;
            }
        }
        assert!(best_len > 0, "expected at least one non-empty cell");
        best_cell
    }

    fn directed_bruteforce_slots(
        grid: &CubeMapGrid,
        points: &[Vec3],
        query_idx: usize,
        query_local: u32,
    ) -> Vec<u32> {
        let query = points[query_idx];
        let mut candidates = Vec::with_capacity(points.len().saturating_sub(1));
        for (slot, &neighbor_idx_u32) in grid.point_indices().iter().enumerate() {
            let neighbor_idx = neighbor_idx_u32 as usize;
            if neighbor_idx == query_idx {
                continue;
            }
            if (slot as u32) < query_local {
                continue;
            }
            let dot = query.dot(points[neighbor_idx]);
            let dist_sq = (2.0 - 2.0 * dot).max(0.0);
            candidates.push((dist_sq, slot as u32));
        }
        candidates.sort_unstable_by(|&(da, sa), &(db, sb)| da.total_cmp(&db).then(sa.cmp(&sb)));
        candidates.into_iter().map(|(_, slot)| slot).collect()
    }

    #[test]
    fn directed_neighbor_stream_matches_bruteforce_order() {
        const N: usize = 320;
        const RES: usize = 10;

        for &seed in &[5u64, 29, 777] {
            let points = random_unit_points(N, seed);
            let grid = CubeMapGrid::new(&points, RES);
            let cell = fullest_cell(&grid);
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
            let query_locals = queries.clone();
            let mut slot_gen_map = vec![0u32; points.len()];
            for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
                *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
            }

            let group = DirectedCellGroup::new(
                cell,
                QUERY_BIN,
                &queries,
                &query_locals,
                &slot_gen_map,
                LOCAL_SHIFT,
                LOCAL_MASK,
            );
            let mut packed_scratch = PackedKnnCellScratch::new();
            let mut packed_timings = PackedKnnTimings::default();
            assert_eq!(
                packed_scratch.prepare_group_directed(&grid, group, &mut packed_timings),
                PackedKnnCellStatus::Ok
            );

            for qi in 0..queries.len() {
                let query_slot = queries[qi];
                let query_idx = grid.point_indices()[query_slot as usize] as usize;
                let query_local = query_locals[qi];
                let ctx = DirectedCtx::new(
                    QUERY_BIN,
                    query_local,
                    &slot_gen_map,
                    LOCAL_SHIFT,
                    LOCAL_MASK,
                );
                let mut grid_scratch = grid.make_scratch();
                let packed =
                    PackedQuery::new(&mut packed_scratch, &mut packed_timings, group, qi, 16, 8);
                let mut stream = DirectedNeighborStream::new(
                    &grid,
                    &points,
                    query_idx,
                    &mut grid_scratch,
                    ctx,
                    Some(packed),
                );

                let mut batch = Vec::new();
                let mut emitted = Vec::with_capacity(points.len().saturating_sub(1));
                let mut seen = vec![false; points.len()];
                while let Some(result) = stream.next_batch(&mut batch) {
                    for &slot in &batch[..result.n] {
                        let neighbor_idx = grid.point_indices()[slot as usize] as usize;
                        let should_emit = match result.source {
                            DirectedNeighborBatchSource::DirectedCursor => {
                                let fresh = !seen[neighbor_idx];
                                seen[neighbor_idx] = true;
                                fresh
                            }
                            DirectedNeighborBatchSource::PackedChunk0
                            | DirectedNeighborBatchSource::PackedTail => {
                                seen[neighbor_idx] = true;
                                true
                            }
                            DirectedNeighborBatchSource::PackedExhausted => false,
                        };
                        if should_emit {
                            emitted.push(slot);
                        }
                    }
                }

                let expected = directed_bruteforce_slots(&grid, &points, query_idx, query_local);
                assert_eq!(
                    emitted, expected,
                    "stream order mismatch for seed={seed}, qi={qi}"
                );
            }
        }
    }
}
