use crate::cube_grid::packed_knn::{
    PackedNeighborBatchSource, PackedNeighborFrontier, PackedQuery,
};

use super::directed::DirectedNoKCursor;
use super::{CubeMapGrid, CubeMapGridScratch, DirectedEligibility};
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectedNeighborBatchSource {
    PackedChunk0,
    PackedTail,
    PackedExpandR2,
    DirectedCursor,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectedNeighborBatch {
    pub(crate) n: usize,
    pub(crate) first_dot: f32,
    pub(crate) unseen_bound: f32,
    pub(crate) source: DirectedNeighborBatchSource,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DirectedNeighborFrontier {
    ExactBatch(DirectedNeighborBatch),
    UnknownButBounded { dot_upper_bound: f32 },
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamStage {
    Packed,
    Cursor,
    Done,
}

#[derive(Debug, Clone)]
enum CachedFrontier {
    ExactBatch {
        batch: DirectedNeighborBatch,
        slots: Vec<u32>,
    },
    UnknownButBounded {
        dot_upper_bound: f32,
    },
    Exhausted,
}

pub(crate) struct DirectedNeighborStream<'a, 'm, 'p, 'g> {
    grid: &'a CubeMapGrid,
    query: Vec3,
    cursor: DirectedNoKCursor<'a, 'm>,
    packed: Option<PackedQuery<'p, 'g, 'm>>,
    stage: StreamStage,
    cached_frontier: Option<CachedFrontier>,
    did_packed: bool,
    packed_safe_exhausted: bool,
    used_cursor: bool,
    knn_exhausted: bool,
}

impl<'a, 'm, 'p, 'g> DirectedNeighborStream<'a, 'm, 'p, 'g> {
    #[inline]
    fn slot_dot(&self, slot: u32) -> f32 {
        let slot = slot as usize;
        crate::fp::dot3_f32(
            self.query.x,
            self.query.y,
            self.query.z,
            self.grid.cell_points_x[slot],
            self.grid.cell_points_y[slot],
            self.grid.cell_points_z[slot],
        )
    }

    pub(crate) fn new(
        grid: &'a CubeMapGrid,
        points: &'a [Vec3],
        query_idx: usize,
        scratch: &'a mut CubeMapGridScratch,
        directed_ctx: DirectedEligibility<'m>,
        packed: Option<PackedQuery<'p, 'g, 'm>>,
    ) -> Self {
        let cursor = grid.directed_no_k_cursor(points[query_idx], query_idx, scratch, directed_ctx);
        let did_packed = packed.is_some();
        let stage = if did_packed {
            StreamStage::Packed
        } else {
            StreamStage::Cursor
        };

        Self {
            grid,
            query: points[query_idx],
            cursor,
            packed,
            stage,
            cached_frontier: None,
            did_packed,
            packed_safe_exhausted: false,
            used_cursor: false,
            knn_exhausted: false,
        }
    }

    pub(crate) fn frontier(&mut self, out: &mut Vec<u32>) -> DirectedNeighborFrontier {
        out.clear();

        if let Some(cached) = &self.cached_frontier {
            match cached {
                CachedFrontier::ExactBatch { batch, slots } => {
                    out.extend_from_slice(slots);
                    return DirectedNeighborFrontier::ExactBatch(*batch);
                }
                CachedFrontier::UnknownButBounded { dot_upper_bound } => {
                    return DirectedNeighborFrontier::UnknownButBounded {
                        dot_upper_bound: *dot_upper_bound,
                    };
                }
                CachedFrontier::Exhausted => return DirectedNeighborFrontier::Exhausted,
            }
        }

        loop {
            match self.stage {
                StreamStage::Packed => {
                    let packed = self
                        .packed
                        .as_mut()
                        .expect("packed stage requires packed query state");
                    match packed.frontier(self.grid, out) {
                        PackedNeighborFrontier::ExactBatch(batch) => {
                            let source = match batch.source {
                                PackedNeighborBatchSource::Chunk0 => {
                                    DirectedNeighborBatchSource::PackedChunk0
                                }
                                PackedNeighborBatchSource::Tail => {
                                    DirectedNeighborBatchSource::PackedTail
                                }
                                PackedNeighborBatchSource::ExpandR2 => {
                                    DirectedNeighborBatchSource::PackedExpandR2
                                }
                            };
                            let batch = DirectedNeighborBatch {
                                n: batch.n,
                                first_dot: batch.first_dot,
                                unseen_bound: batch.unseen_bound,
                                source,
                            };
                            self.cached_frontier = Some(CachedFrontier::ExactBatch {
                                batch,
                                slots: out.clone(),
                            });
                            return DirectedNeighborFrontier::ExactBatch(batch);
                        }
                        PackedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                            self.cached_frontier =
                                Some(CachedFrontier::UnknownButBounded { dot_upper_bound });
                            return DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound };
                        }
                        PackedNeighborFrontier::Exhausted => {
                            self.packed_safe_exhausted |= packed.safe_exhausted();
                            self.stage = StreamStage::Cursor;
                            continue;
                        }
                    }
                }
                StreamStage::Cursor => {
                    self.used_cursor = true;
                    if let Some(slot) = self.cursor.pop_next_proven_slot() {
                        let unseen_bound = self.cursor.remaining_dot_upper_bound();
                        out.push(slot);
                        let batch = DirectedNeighborBatch {
                            n: 1,
                            first_dot: self.slot_dot(slot),
                            unseen_bound,
                            source: DirectedNeighborBatchSource::DirectedCursor,
                        };
                        self.cached_frontier = Some(CachedFrontier::ExactBatch {
                            batch,
                            slots: vec![slot],
                        });
                        return DirectedNeighborFrontier::ExactBatch(batch);
                    }

                    self.knn_exhausted = self.cursor.is_exhausted();
                    if self.knn_exhausted {
                        self.stage = StreamStage::Done;
                        self.cached_frontier = Some(CachedFrontier::Exhausted);
                        return DirectedNeighborFrontier::Exhausted;
                    }

                    let dot_upper_bound = self.cursor.remaining_dot_upper_bound();
                    self.cached_frontier =
                        Some(CachedFrontier::UnknownButBounded { dot_upper_bound });
                    return DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound };
                }
                StreamStage::Done => {
                    self.cached_frontier = Some(CachedFrontier::Exhausted);
                    return DirectedNeighborFrontier::Exhausted;
                }
            }
        }
    }

    pub(crate) fn advance_frontier(&mut self) {
        let cached = self.cached_frontier.take();
        match cached {
            Some(CachedFrontier::ExactBatch { .. })
            | Some(CachedFrontier::UnknownButBounded { .. }) => match self.stage {
                StreamStage::Packed => {
                    let packed = self
                        .packed
                        .as_mut()
                        .expect("packed stage requires packed query state");
                    packed.advance_frontier(self.grid);
                    if packed.is_exhausted() {
                        self.packed_safe_exhausted |= packed.safe_exhausted();
                        self.stage = StreamStage::Cursor;
                    }
                }
                StreamStage::Cursor | StreamStage::Done => {}
            },
            Some(CachedFrontier::Exhausted) | None => {}
        }
    }

    #[inline]
    pub(crate) fn did_packed(&self) -> bool {
        self.did_packed
    }

    #[inline]
    pub(crate) fn packed_tail_used(&self) -> bool {
        self.packed
            .as_ref()
            .map(|packed| packed.tail_used())
            .unwrap_or(false)
    }

    #[inline]
    pub(crate) fn packed_expand_r2_used(&self) -> bool {
        self.packed
            .as_ref()
            .map(|packed| packed.expand_r2_used())
            .unwrap_or(false)
    }

    #[inline]
    pub(crate) fn packed_safe_exhausted(&self) -> bool {
        self.packed_safe_exhausted
    }

    #[inline]
    pub(crate) fn knn_exhausted(&self) -> bool {
        self.knn_exhausted
    }

    #[inline]
    pub(crate) fn is_cursor_stage(&self) -> bool {
        self.stage == StreamStage::Cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube_grid::packed_knn::{
        PackedGroupInput, PackedKnnCellScratch, PackedKnnTimings, PreparedPackedGroupStatus,
    };
    use crate::packed_layout::PackedSlotLayout;
    use crate::policy::PackedNeighborPolicy;
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
            let mut slot_gen_map = vec![0u32; points.len()];
            for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
                *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
            }
            let layout = PackedSlotLayout::new(&slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);

            let group = PackedGroupInput::new(cell, QUERY_BIN, &queries, start as u32, layout);
            for &expand_r2_enabled in &[false, true] {
                let mut packed_scratch = PackedKnnCellScratch::new();
                let mut packed_timings = PackedKnnTimings::default();
                let PreparedPackedGroupStatus::Ready(mut prepared) =
                    packed_scratch.prepare_group_directed(&grid, group, &mut packed_timings)
                else {
                    panic!("packed prepare unexpectedly fell back to slow path");
                };

                for qi in 0..queries.len() {
                    let query_slot = queries[qi];
                    let query_idx = grid.point_indices()[query_slot as usize] as usize;
                    let query_local = queries[qi];
                    let ctx = DirectedEligibility::from_layout(QUERY_BIN, query_local, layout);
                    let mut grid_scratch = grid.make_scratch();
                    let packed = PackedQuery::new(
                        &mut prepared,
                        &mut packed_timings,
                        qi,
                        PackedNeighborPolicy::for_point_count(points.len(), expand_r2_enabled),
                    );
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
                    loop {
                        match stream.frontier(&mut batch) {
                            DirectedNeighborFrontier::ExactBatch(result) => {
                                for &slot in &batch[..result.n] {
                                    let neighbor_idx = grid.point_indices()[slot as usize] as usize;
                                    let should_emit = match result.source {
                                        DirectedNeighborBatchSource::DirectedCursor => {
                                            let fresh = !seen[neighbor_idx];
                                            seen[neighbor_idx] = true;
                                            fresh
                                        }
                                        DirectedNeighborBatchSource::PackedChunk0
                                        | DirectedNeighborBatchSource::PackedTail
                                        | DirectedNeighborBatchSource::PackedExpandR2 => {
                                            seen[neighbor_idx] = true;
                                            true
                                        }
                                    };
                                    if should_emit {
                                        emitted.push(slot);
                                    }
                                }
                                stream.advance_frontier();
                            }
                            DirectedNeighborFrontier::UnknownButBounded { .. } => {
                                stream.advance_frontier();
                            }
                            DirectedNeighborFrontier::Exhausted => break,
                        }
                    }

                    let expected =
                        directed_bruteforce_slots(&grid, &points, query_idx, query_local);
                    assert_eq!(
                        emitted, expected,
                        "stream order mismatch for seed={seed}, qi={qi}, expand_r2_enabled={expand_r2_enabled}"
                    );
                }
            }
        }
    }

    #[test]
    fn repeated_frontier_calls_do_not_advance_stage() {
        const N: usize = 320;
        const RES: usize = 10;

        let points = random_unit_points(N, 29);
        let grid = CubeMapGrid::new(&points, RES);
        let cell = fullest_cell(&grid);
        let start = grid.cell_offsets()[cell] as usize;
        let end = grid.cell_offsets()[cell + 1] as usize;
        let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
        let mut slot_gen_map = vec![0u32; points.len()];
        for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
            *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
        }
        let layout = PackedSlotLayout::new(&slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);

        let group = PackedGroupInput::new(cell, QUERY_BIN, &queries, start as u32, layout);
        let mut packed_scratch = PackedKnnCellScratch::new();
        let mut packed_timings = PackedKnnTimings::default();
        let PreparedPackedGroupStatus::Ready(mut prepared) =
            packed_scratch.prepare_group_directed(&grid, group, &mut packed_timings)
        else {
            panic!("packed prepare unexpectedly fell back to slow path");
        };

        let qi = 0usize;
        let query_slot = queries[qi];
        let query_idx = grid.point_indices()[query_slot as usize] as usize;
        let ctx = DirectedEligibility::from_layout(QUERY_BIN, queries[qi], layout);
        let mut grid_scratch = grid.make_scratch();
        let packed = PackedQuery::new(
            &mut prepared,
            &mut packed_timings,
            qi,
            PackedNeighborPolicy::for_point_count(points.len(), true),
        );
        let mut stream = DirectedNeighborStream::new(
            &grid,
            &points,
            query_idx,
            &mut grid_scratch,
            ctx,
            Some(packed),
        );

        let mut batch = Vec::new();
        let first = stream.frontier(&mut batch);
        let first_slots = batch.clone();
        let second = stream.frontier(&mut batch);
        assert_eq!(
            std::mem::discriminant(&first),
            std::mem::discriminant(&second),
            "repeated frontier call should return the same frontier kind without advancing"
        );
        assert_eq!(
            batch, first_slots,
            "repeated frontier call should return the same exact batch without advancing"
        );

        stream.advance_frontier();
        let third = stream.frontier(&mut batch);
        match (first, third) {
            (
                DirectedNeighborFrontier::ExactBatch(first_batch),
                DirectedNeighborFrontier::ExactBatch(third_batch),
            ) => {
                assert!(
                    third_batch.first_dot <= first_batch.unseen_bound + 1e-6,
                    "advanced frontier exact batch should not outrank previous unseen bound: first={:?}, third={:?}",
                    first_batch,
                    third_batch,
                );
            }
            (
                DirectedNeighborFrontier::ExactBatch(first_batch),
                DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound },
            ) => {
                assert!(
                    dot_upper_bound <= first_batch.unseen_bound + 1e-6,
                    "advanced frontier bound should not exceed previous unseen bound: first={:?}, bound={}",
                    first_batch,
                    dot_upper_bound,
                );
            }
            (DirectedNeighborFrontier::ExactBatch(_), DirectedNeighborFrontier::Exhausted) => {}
            _ => panic!("expected initial frontier to be an exact batch"),
        }
    }

    #[test]
    fn frontier_certificates_remain_conservative_against_bruteforce() {
        const N: usize = 320;
        const RES: usize = 10;

        for &seed in &[5u64, 29, 777] {
            let points = random_unit_points(N, seed);
            let grid = CubeMapGrid::new(&points, RES);
            let cell = fullest_cell(&grid);
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
            let mut slot_gen_map = vec![0u32; points.len()];
            for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
                *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | slot as u32;
            }
            let layout = PackedSlotLayout::new(&slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);

            let group = PackedGroupInput::new(cell, QUERY_BIN, &queries, start as u32, layout);
            for &expand_r2_enabled in &[false, true] {
                let mut packed_scratch = PackedKnnCellScratch::new();
                let mut packed_timings = PackedKnnTimings::default();
                let PreparedPackedGroupStatus::Ready(mut prepared) =
                    packed_scratch.prepare_group_directed(&grid, group, &mut packed_timings)
                else {
                    panic!("packed prepare unexpectedly fell back to slow path");
                };

                for qi in 0..queries.len() {
                    let query_slot = queries[qi];
                    let query_idx = grid.point_indices()[query_slot as usize] as usize;
                    let query_local = queries[qi];
                    let ctx = DirectedEligibility::from_layout(QUERY_BIN, query_local, layout);
                    let mut grid_scratch = grid.make_scratch();
                    let packed = PackedQuery::new(
                        &mut prepared,
                        &mut packed_timings,
                        qi,
                        PackedNeighborPolicy::for_point_count(points.len(), expand_r2_enabled),
                    );
                    let mut stream = DirectedNeighborStream::new(
                        &grid,
                        &points,
                        query_idx,
                        &mut grid_scratch,
                        ctx,
                        Some(packed),
                    );

                    let expected =
                        directed_bruteforce_slots(&grid, &points, query_idx, query_local);
                    let mut seen = vec![false; points.len()];
                    let mut batch = Vec::new();

                    loop {
                        let best_unseen_dot = expected
                            .iter()
                            .find_map(|&slot| {
                                let neighbor_idx = grid.point_indices()[slot as usize] as usize;
                                (!seen[neighbor_idx])
                                    .then(|| points[query_idx].dot(points[neighbor_idx]))
                            })
                            .unwrap_or(-1.0);
                        match stream.frontier(&mut batch) {
                            DirectedNeighborFrontier::ExactBatch(result) => {
                                assert!(
                                    result.first_dot + 1e-6 >= best_unseen_dot,
                                    "frontier exact batch first dot should cover the best unseen neighbor"
                                );
                                for &slot in &batch[..result.n] {
                                    let neighbor_idx = grid.point_indices()[slot as usize] as usize;
                                    match result.source {
                                        DirectedNeighborBatchSource::DirectedCursor => {
                                            seen[neighbor_idx] = true;
                                        }
                                        DirectedNeighborBatchSource::PackedChunk0
                                        | DirectedNeighborBatchSource::PackedTail
                                        | DirectedNeighborBatchSource::PackedExpandR2 => {
                                            seen[neighbor_idx] = true;
                                        }
                                    }
                                }
                                stream.advance_frontier();
                            }
                            DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                                assert!(
                                    best_unseen_dot <= dot_upper_bound + 1e-6,
                                    "frontier bound underestimated best unseen neighbor for seed={seed}, qi={qi}, expand_r2_enabled={expand_r2_enabled}"
                                );
                                stream.advance_frontier();
                            }
                            DirectedNeighborFrontier::Exhausted => {
                                assert_eq!(
                                    seen.iter().filter(|seen| **seen).count(),
                                    expected.len(),
                                    "frontier exhausted before emitting all neighbors for seed={seed}, qi={qi}, expand_r2_enabled={expand_r2_enabled}"
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}
