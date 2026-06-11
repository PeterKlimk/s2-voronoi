//! Planar packed-kNN stage: batched, SIMD-filtered directed neighbor
//! candidates for one grid cell's queries at a time.
//!
//! The planar port of `cube_grid::packed_knn`. Per cell, all queries are
//! prepared as one group (3x3 SoA scan, 8-wide squared distances via
//! `fp::PlaneChunk8`), then each query drains its sorted candidates in
//! stages — Chunk0 (tightened "hi" set), Tail (the [threshold, security)
//! band, built lazily), ExpandR2 (the 5x5 band, built lazily) — before the
//! shell-expansion takeover re-covers everything else. Certificates are
//! squared-distance LOWER bounds (smaller is closer), from the exact box
//! geometry.

mod scratch;
mod timing;

use super::PlaneGrid;
use crate::packed_layout::PackedSlotLayout;
use crate::policy::PackedNeighborPolicy;

pub use scratch::PlanePackedScratch;
pub(crate) use scratch::{PlanePreparedGroup, PlanePreparedGroupStatus};
pub use timing::PlanePackedTimings;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanePackedGroupInput<'a> {
    cell: usize,
    query_bin: u8,
    queries: &'a [u32],
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    query_local_start: u32,
    layout: PackedSlotLayout<'a>,
}

impl<'a> PlanePackedGroupInput<'a> {
    #[inline]
    pub(crate) fn new(
        cell: usize,
        query_bin: u8,
        queries: &'a [u32],
        query_local_start: u32,
        layout: PackedSlotLayout<'a>,
    ) -> Self {
        Self {
            cell,
            query_bin,
            queries,
            query_local_start,
            layout,
        }
    }

    #[inline]
    pub(crate) fn cell(self) -> usize {
        self.cell
    }

    #[inline]
    pub(crate) fn query_bin(self) -> u8 {
        self.query_bin
    }

    #[inline]
    pub(crate) fn queries(self) -> &'a [u32] {
        self.queries
    }

    #[inline]
    pub(crate) fn slot_gen_map(self) -> &'a [u32] {
        self.layout.slot_gen_map()
    }

    #[inline]
    pub(crate) fn local_shift(self) -> u32 {
        self.layout.local_shift()
    }

    #[inline]
    pub(crate) fn local_mask(self) -> u32 {
        self.layout.local_mask()
    }

    /// live_dedup contract: a group is one complete center-cell run in slot
    /// order with contiguous locals.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_matches_grid(self, grid: &PlaneGrid) {
        let start = grid.cell_offsets()[self.cell] as usize;
        let end = grid.cell_offsets()[self.cell + 1] as usize;
        debug_assert_eq!(
            self.queries.len(),
            end - start,
            "planar packed group must cover the full center cell"
        );
        debug_assert!(
            self.queries
                .iter()
                .enumerate()
                .all(|(offset, &slot)| slot as usize == start + offset),
            "planar packed group queries must be the center cell in slot order"
        );
        debug_assert!(
            self.queries.iter().enumerate().all(|(offset, &slot)| {
                let (bin, local) = self.layout.bin_local(slot);
                bin == self.query_bin && local == self.query_local_start + offset as u32
            }),
            "planar packed group (slot -> bin,local) mapping must match query inputs"
        );
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    pub(crate) fn debug_assert_matches_grid(self, _grid: &PlaneGrid) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanePackedStage {
    Chunk0,
    Tail,
    ExpandR2,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanePackedChunk {
    pub(crate) n: usize,
    /// Lower bound on the squared distance of every eligible candidate not
    /// yet emitted by the packed stages.
    pub(crate) unseen_bound: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanePackedBatchSource {
    Chunk0,
    Tail,
    ExpandR2,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanePackedBatch {
    pub(crate) n: usize,
    pub(crate) first_dist_sq: f32,
    pub(crate) unseen_bound: f32,
    pub(crate) source: PlanePackedBatchSource,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PlanePackedFrontier {
    ExactBatch(PlanePackedBatch),
    /// No batch materialized at this stage boundary, but everything unseen
    /// is at least this far away; `advance_frontier` moves to the next stage.
    UnknownButBounded {
        dist_lower_bound: f32,
    },
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryStage {
    Chunk0,
    Tail,
    ExpandR2,
    Exhausted,
}

#[derive(Debug, Clone)]
enum CachedFrontier {
    ExactBatch {
        batch: PlanePackedBatch,
        slots: Vec<u32>,
        dists: Vec<f32>,
    },
    UnknownButBounded {
        dist_lower_bound: f32,
    },
    Exhausted,
}

/// One query's view of a prepared group: drains the packed stages with the
/// same frontier/advance protocol as the shell frontier.
pub(crate) struct PlanePackedQuery<'a, 'p, 'g> {
    prepared: &'a mut PlanePreparedGroup<'p, 'g>,
    timings: &'a mut PlanePackedTimings,
    query_index: usize,
    policy: PackedNeighborPolicy,
    stage: QueryStage,
    cached_frontier: Option<CachedFrontier>,
    safe_exhausted: bool,
}

impl<'a, 'p, 'g> PlanePackedQuery<'a, 'p, 'g> {
    pub(crate) fn new(
        prepared: &'a mut PlanePreparedGroup<'p, 'g>,
        timings: &'a mut PlanePackedTimings,
        query_index: usize,
        policy: PackedNeighborPolicy,
    ) -> Self {
        Self {
            prepared,
            timings,
            query_index,
            policy,
            stage: QueryStage::Chunk0,
            cached_frontier: None,
            safe_exhausted: false,
        }
    }

    /// Current frontier; repeated calls return the same value until
    /// [`Self::advance_frontier`]. Fills `out`/`out_dists` for exact batches.
    pub(crate) fn frontier(
        &mut self,
        out: &mut Vec<u32>,
        out_dists: &mut Vec<f32>,
    ) -> PlanePackedFrontier {
        out.clear();
        out_dists.clear();

        if let Some(cached) = &self.cached_frontier {
            match cached {
                CachedFrontier::ExactBatch {
                    batch,
                    slots,
                    dists,
                } => {
                    out.extend_from_slice(slots);
                    out_dists.extend_from_slice(dists);
                    return PlanePackedFrontier::ExactBatch(*batch);
                }
                CachedFrontier::UnknownButBounded { dist_lower_bound } => {
                    return PlanePackedFrontier::UnknownButBounded {
                        dist_lower_bound: *dist_lower_bound,
                    };
                }
                CachedFrontier::Exhausted => return PlanePackedFrontier::Exhausted,
            }
        }

        let (stage, source, k) = match self.stage {
            QueryStage::Chunk0 => (
                PlanePackedStage::Chunk0,
                PlanePackedBatchSource::Chunk0,
                self.policy.chunk0_size(),
            ),
            QueryStage::Tail => (
                PlanePackedStage::Tail,
                PlanePackedBatchSource::Tail,
                self.policy.chunk_size(),
            ),
            QueryStage::ExpandR2 => (
                PlanePackedStage::ExpandR2,
                PlanePackedBatchSource::ExpandR2,
                self.policy.chunk_size(),
            ),
            QueryStage::Exhausted => {
                self.cached_frontier = Some(CachedFrontier::Exhausted);
                return PlanePackedFrontier::Exhausted;
            }
        };

        if let Some(chunk) =
            self.prepared
                .next_chunk(self.query_index, stage, k, out, out_dists, self.timings)
        {
            let batch = PlanePackedBatch {
                n: chunk.n,
                first_dist_sq: out_dists.first().copied().unwrap_or(f32::INFINITY),
                unseen_bound: chunk.unseen_bound,
                source,
            };
            self.cached_frontier = Some(CachedFrontier::ExactBatch {
                batch,
                slots: out.clone(),
                dists: out_dists.clone(),
            });
            return PlanePackedFrontier::ExactBatch(batch);
        }

        let dist_lower_bound =
            if self.stage == QueryStage::Chunk0 && self.prepared.tail_possible(self.query_index) {
                self.prepared.tail_lower_bound(self.query_index)
            } else {
                self.prepared.resume_security(self.query_index)
            };
        self.cached_frontier = Some(CachedFrontier::UnknownButBounded { dist_lower_bound });
        PlanePackedFrontier::UnknownButBounded { dist_lower_bound }
    }

    pub(crate) fn advance_frontier(&mut self, grid: &PlaneGrid) {
        let cached = self.cached_frontier.take();
        match cached {
            Some(CachedFrontier::ExactBatch { .. }) => {}
            Some(CachedFrontier::UnknownButBounded { .. }) => self.advance_stage(grid),
            Some(CachedFrontier::Exhausted) | None => {}
        }
    }

    fn advance_stage(&mut self, grid: &PlaneGrid) {
        if self.stage == QueryStage::Chunk0 && self.prepared.tail_possible(self.query_index) {
            self.prepared
                .ensure_tail_for(self.query_index, grid, self.timings);
            self.stage = QueryStage::Tail;
            return;
        }

        if self.stage != QueryStage::ExpandR2
            && self.policy.expand_r2_enabled()
            && self
                .prepared
                .ensure_expand_r2_band_for(self.query_index, grid, self.timings)
        {
            self.stage = QueryStage::ExpandR2;
            return;
        }

        self.safe_exhausted = true;
        self.stage = QueryStage::Exhausted;
    }

    /// True when the packed stages ended with a certificate covering
    /// everything unseen (vs. handing off mid-stream).
    #[inline]
    pub(crate) fn safe_exhausted(&self) -> bool {
        self.safe_exhausted
    }

    #[inline]
    pub(crate) fn is_exhausted(&self) -> bool {
        self.stage == QueryStage::Exhausted
    }
}
