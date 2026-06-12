// The loop index addresses parallel per-cell arrays (mirrors the cube
// suite); the frontier loop has a non-trivial Exhausted arm.
#![allow(clippy::needless_range_loop, clippy::while_let_loop)]
//! Contract tests for the planar neighbor-source layer, in isolation from
//! the (future) planar Voronoi pipeline.
//!
//! Mirrors `cube_grid::tests::nn_contract` — the durable contract is the
//! same one the cell builder relies on:
//!
//! 1. **Eligible set**: across its lifetime the stream emits exactly the
//!    directed-eligible points (other-bin points; same-bin points in later
//!    cells; same-bin same-cell points with local >= query local, excluding
//!    self). Duplicate emissions are permitted and deduplicated by the
//!    consumer.
//! 2. **Certificates**: every frontier's unseen bound conservatively
//!    lower-bounds the squared distance of every eligible point not yet
//!    emitted.
//!
//! Emission *order* is deliberately not part of the contract. Scenarios
//! stress the planar grid specifically: domain corners and edges (clamped,
//! not wrapped), exact wall coordinates, single-cell and coarse grids,
//! degenerate one-dimensional distributions, clusters, bimodal densities,
//! duplicates, and tiny inputs.

use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::packed::{
    PlanePackedGroupInput, PlanePackedQuery, PlanePackedScratch, PlanePackedTimings,
    PlanePreparedGroupStatus,
};
use super::{PlaneGrid, PlaneNeighborFrontier, PlaneNeighborStream};
use crate::cube_grid::DirectedEligibility;
use crate::packed_layout::PackedSlotLayout;
use crate::policy::PackedNeighborPolicy;

const LOCAL_SHIFT: u32 = 24;
const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;
const DIST_SQ_TOL: f32 = 1e-6;

struct Harness {
    points: Vec<Vec2>,
    grid: PlaneGrid,
    slot_gen_map: Vec<u32>,
    cell_of_slot: Vec<usize>,
    bin_of_cell: Vec<u8>,
}

impl Harness {
    fn new(points: Vec<Vec2>, res: usize, num_bins: u8) -> Self {
        assert!(num_bins >= 1);
        let grid = PlaneGrid::new(&points, res);
        let num_cells = grid.cell_offsets().len() - 1;

        // Contiguous cell ranges per bin (the layout invariant the production
        // binning also guarantees).
        let bin_of_cell: Vec<u8> = (0..num_cells)
            .map(|cell| ((cell * num_bins as usize) / num_cells) as u8)
            .collect();

        let mut cell_of_slot = vec![0usize; points.len()];
        let mut slot_gen_map = vec![0u32; points.len()];
        for cell in 0..num_cells {
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            for slot in start..end {
                cell_of_slot[slot] = cell;
                // Local id = global slot: preserves slot order within bins.
                slot_gen_map[slot] = ((bin_of_cell[cell] as u32) << LOCAL_SHIFT) | slot as u32;
            }
        }

        Harness {
            points,
            grid,
            slot_gen_map,
            cell_of_slot,
            bin_of_cell,
        }
    }

    fn layout(&self) -> PackedSlotLayout<'_> {
        PackedSlotLayout::new(&self.slot_gen_map, LOCAL_SHIFT, LOCAL_MASK)
    }

    fn slot_dist_sq(&self, query_slot: u32, slot: u32) -> f32 {
        let qi = self.grid.point_indices()[query_slot as usize] as usize;
        let pi = self.grid.point_indices()[slot as usize] as usize;
        let (dx, dy) = (
            self.points[pi].x - self.points[qi].x,
            self.points[pi].y - self.points[qi].y,
        );
        dx * dx + dy * dy
    }

    /// Directed-eligible slots for a query, per the cell-mode rules.
    fn brute_eligible(&self, query_slot: u32) -> Vec<u32> {
        let start_cell = self.cell_of_slot[query_slot as usize];
        let qbin = self.bin_of_cell[start_cell];
        let qlocal = query_slot; // local == slot in this layout
        (0..self.points.len() as u32)
            .filter(|&slot| {
                if slot == query_slot {
                    return false;
                }
                let cell = self.cell_of_slot[slot as usize];
                let bin = self.bin_of_cell[cell];
                if bin != qbin {
                    return true;
                }
                match cell.cmp(&start_cell) {
                    std::cmp::Ordering::Less => false,
                    std::cmp::Ordering::Equal => slot >= qlocal,
                    std::cmp::Ordering::Greater => true,
                }
            })
            .collect()
    }

    /// Drive a stream to exhaustion, asserting the certificate contract at
    /// every frontier; returns the deduplicated emitted slot set.
    fn collect_and_check(
        &self,
        name: &str,
        query_slot: u32,
        mut stream: PlaneNeighborStream<'_, '_, '_, '_>,
    ) -> Vec<u32> {
        let eligible = self.brute_eligible(query_slot);
        let mut unseen: std::collections::HashSet<u32> = eligible.iter().copied().collect();
        let mut emitted: Vec<u32> = Vec::new();
        let mut emitted_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut batch: Vec<u32> = Vec::new();
        let mut batch_dists: Vec<f32> = Vec::new();

        let nearest_unseen = |unseen: &std::collections::HashSet<u32>| -> f32 {
            // INFINITY when empty: any bound (including INFINITY) covers it.
            unseen
                .iter()
                .map(|&slot| self.slot_dist_sq(query_slot, slot))
                .fold(f32::INFINITY, f32::min)
        };

        loop {
            match stream.frontier(&mut batch, &mut batch_dists) {
                PlaneNeighborFrontier::ExactBatch(result) => {
                    // Per-emission bounds are sound iff every reported dist
                    // is the slot's true distance — assert that.
                    for (k, &slot) in batch[..result.n].iter().enumerate() {
                        assert!(
                            slot != query_slot,
                            "{name}: stream emitted the query slot itself"
                        );
                        let true_d = self.slot_dist_sq(query_slot, slot);
                        assert!(
                            (batch_dists[k] - true_d).abs() <= DIST_SQ_TOL,
                            "{name}: reported dist {} != true dist {} (slot {slot})",
                            batch_dists[k],
                            true_d
                        );
                        if emitted_set.insert(slot) {
                            emitted.push(slot);
                        }
                        unseen.remove(&slot);
                    }
                    assert!(
                        nearest_unseen(&unseen) >= result.unseen_bound - DIST_SQ_TOL,
                        "{name}: batch unseen_bound {} overstates an unseen eligible \
                         point at dist_sq {} (query slot {query_slot})",
                        result.unseen_bound,
                        nearest_unseen(&unseen)
                    );
                    stream.advance_frontier();
                }
                PlaneNeighborFrontier::UnknownButBounded { dist_lower_bound } => {
                    assert!(
                        nearest_unseen(&unseen) >= dist_lower_bound - DIST_SQ_TOL,
                        "{name}: bounded frontier {} overstates an unseen eligible \
                         point at dist_sq {} (query slot {query_slot})",
                        dist_lower_bound,
                        nearest_unseen(&unseen)
                    );
                    stream.advance_frontier();
                }
                PlaneNeighborFrontier::Exhausted => break,
            }
        }

        let mut expected = eligible.clone();
        expected.sort_unstable();
        let mut got = emitted.clone();
        got.sort_unstable();
        assert_eq!(
            got, expected,
            "{name}: emitted eligible set mismatch for query slot {query_slot}"
        );
        emitted
    }

    /// Run the contract for every (sampled) query through the takeover-only
    /// path and, per center cell, the packed path.
    fn check_all(&self, name: &str) {
        let n = self.points.len();
        let stride = (n / 128).max(1);

        // Takeover-only path (packed = None).
        for slot in (0..n as u32).step_by(stride) {
            let query_idx = self.grid.point_indices()[slot as usize] as usize;
            let ctx = DirectedEligibility::from_layout(
                self.bin_of_cell[self.cell_of_slot[slot as usize]],
                slot,
                self.layout(),
            );
            let mut scratch = self.grid.make_scratch();
            let stream = PlaneNeighborStream::new(
                &self.grid,
                &self.points,
                query_idx,
                &mut scratch,
                ctx,
                None,
            );
            self.collect_and_check(&format!("{name}/takeover"), slot, stream);
        }

        // Packed path: per center cell, complete slot-order runs.
        let num_cells = self.grid.cell_offsets().len() - 1;
        for cell in 0..num_cells {
            let start = self.grid.cell_offsets()[cell] as usize;
            let end = self.grid.cell_offsets()[cell + 1] as usize;
            if start == end {
                continue;
            }
            let queries: Vec<u32> = (start..end).map(|s| s as u32).collect();
            let group = PlanePackedGroupInput::new(
                cell,
                self.bin_of_cell[cell],
                &queries,
                start as u32,
                self.layout(),
            );
            for &expand_r2 in &[false, true] {
                let mut packed_scratch = PlanePackedScratch::new();
                let mut timings = PlanePackedTimings;
                let PlanePreparedGroupStatus::Ready(mut prepared) =
                    packed_scratch.prepare_group(&self.grid, group, &mut timings)
                else {
                    // SlowPath groups are exercised by the takeover-only pass.
                    continue;
                };
                for (qi, &slot) in queries.iter().enumerate() {
                    let query_idx = self.grid.point_indices()[slot as usize] as usize;
                    let ctx = DirectedEligibility::from_layout(
                        self.bin_of_cell[cell],
                        slot,
                        self.layout(),
                    );
                    let mut scratch = self.grid.make_scratch();
                    let packed = PlanePackedQuery::new(
                        &mut prepared,
                        &mut timings,
                        qi,
                        PackedNeighborPolicy::for_point_count(self.points.len(), expand_r2),
                    );
                    let stream = PlaneNeighborStream::new(
                        &self.grid,
                        &self.points,
                        query_idx,
                        &mut scratch,
                        ctx,
                        Some(packed),
                    );
                    self.collect_and_check(&format!("{name}/packed_r2={expand_r2}"), slot, stream);
                }
            }
        }
    }
}

// === Point-set constructors ===

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn uniform(n: usize, seed: u64) -> Vec<Vec2> {
    let mut r = rng(seed);
    (0..n)
        .map(|_| Vec2::new(r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)))
        .collect()
}

fn jitter(base: Vec2, scale: f32, r: &mut ChaCha8Rng) -> Vec2 {
    Vec2::new(
        (base.x + r.gen_range(-scale..scale)).clamp(0.0, 1.0),
        (base.y + r.gen_range(-scale..scale)).clamp(0.0, 1.0),
    )
}

// === Scenarios ===

#[test]
fn plane_nn_contract_uniform() {
    Harness::new(uniform(320, 5), 10, 1).check_all("uniform");
    Harness::new(uniform(320, 29), 10, 3).check_all("uniform_3bins");
}

#[test]
fn plane_nn_contract_coarse_grids() {
    // res=1: a single cell holding everything.
    Harness::new(uniform(60, 7), 1, 1).check_all("res1");
    Harness::new(uniform(200, 11), 2, 2).check_all("res2_2bins");
    Harness::new(uniform(150, 13), 3, 1).check_all("res3");
}

#[test]
fn plane_nn_contract_corner_and_edge_clusters() {
    // Jittered clusters at the 4 domain corners and 4 edge midpoints: rings
    // clip against the domain edge instead of wrapping.
    let mut r = rng(17);
    let anchors = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.5, 1.0),
        Vec2::new(0.0, 0.5),
        Vec2::new(1.0, 0.5),
    ];
    let mut points = Vec::new();
    for &anchor in &anchors {
        for _ in 0..12 {
            points.push(jitter(anchor, 1e-3, &mut r));
        }
    }
    Harness::new(points, 4, 1).check_all("corner_edge_clusters");
}

#[test]
fn plane_nn_contract_exact_boundary_coordinates() {
    // Points with exact 0.0 / 1.0 / wall-fraction coordinates, including the
    // four exact domain corners.
    let mut points = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.5, 0.5),
        Vec2::new(0.0, 0.5),
        Vec2::new(1.0, 0.5),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.5, 1.0),
        Vec2::new(0.25, 0.75),
        Vec2::new(0.75, 0.25),
    ];
    points.extend(uniform(40, 19));
    Harness::new(points.clone(), 3, 1).check_all("exact_boundary_res3");
    Harness::new(points, 4, 2).check_all("exact_boundary_res4");
}

#[test]
fn plane_nn_contract_near_wall_classification() {
    // Points within ~1e-6 of interior cell walls (and exactly on them): the
    // contract must hold regardless of which side classification puts them on.
    const RES: usize = 8;
    let mut points = Vec::new();
    let mut r = rng(41);
    for k in 1..RES {
        let wall = k as f32 / RES as f32;
        for delta in [-1e-6f32, 0.0, 1e-6] {
            points.push(Vec2::new(wall + delta, r.gen_range(0.1f32..0.9)));
            points.push(Vec2::new(r.gen_range(0.1f32..0.9), wall + delta));
        }
    }
    Harness::new(points, RES, 1).check_all("near_walls");
}

#[test]
fn plane_nn_contract_degenerate_axis() {
    // All points on one horizontal line (single row of cells active).
    let mut r = rng(53);
    let row: Vec<Vec2> = (0..120)
        .map(|_| Vec2::new(r.gen_range(0.0f32..1.0), 0.5))
        .collect();
    Harness::new(row, 8, 1).check_all("one_row");

    // All points in a single grid cell of a fine grid (deep empty-ring
    // traversal for nothing, then exhaustion).
    let cluster: Vec<Vec2> = (0..80)
        .map(|_| {
            Vec2::new(
                0.7 + r.gen_range(0.0f32..0.01),
                0.2 + r.gen_range(0.0f32..0.01),
            )
        })
        .collect();
    Harness::new(cluster, 16, 1).check_all("one_cell");
}

#[test]
fn plane_nn_contract_clustered_and_bimodal() {
    let mut r = rng(43);
    // Tight cluster plus far anchors.
    let center = Vec2::new(0.6, 0.4);
    let mut clustered: Vec<Vec2> = (0..200).map(|_| jitter(center, 0.005, &mut r)).collect();
    clustered.extend([
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.05, 0.95),
        Vec2::new(0.95, 0.05),
    ]);
    Harness::new(clustered, 6, 1).check_all("clustered");

    // Bimodal: dense blob + isolated far points whose queries must traverse
    // many empty rings (the deep-ring regime).
    let mut bimodal: Vec<Vec2> = (0..300)
        .map(|_| jitter(Vec2::new(0.1, 0.1), 0.03, &mut r))
        .collect();
    for _ in 0..12 {
        bimodal.push(jitter(Vec2::new(0.9, 0.9), 0.08, &mut r));
    }
    Harness::new(bimodal, 12, 2).check_all("bimodal");
}

#[test]
fn plane_nn_contract_duplicate_positions() {
    let mut points = uniform(100, 47);
    for i in 0..5 {
        points.push(points[i * 13]);
    }
    Harness::new(points, 4, 1).check_all("duplicates");
}

#[test]
fn plane_nn_contract_tiny_inputs() {
    let mut r = rng(37);
    let two = vec![
        Vec2::new(r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)),
        Vec2::new(r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)),
    ];
    Harness::new(two, 4, 1).check_all("n2");

    let three: Vec<Vec2> = (0..3)
        .map(|_| Vec2::new(r.gen_range(0.0f32..1.0), r.gen_range(0.0f32..1.0)))
        .collect();
    Harness::new(three, 4, 1).check_all("n3");
}

#[test]
fn plane_nn_contract_stream_idempotent_frontier() {
    // Repeated frontier calls return the same batch until advance_frontier.
    let points = uniform(64, 61);
    let h = Harness::new(points, 4, 1);
    let ctx = DirectedEligibility::from_layout(0, 0, h.layout());
    let query_idx = h.grid.point_indices()[0] as usize;
    let mut scratch = h.grid.make_scratch();
    let mut stream =
        PlaneNeighborStream::new(&h.grid, &h.points, query_idx, &mut scratch, ctx, None);

    let mut batch = Vec::new();
    let mut dists = Vec::new();
    let PlaneNeighborFrontier::ExactBatch(first) = stream.frontier(&mut batch, &mut dists) else {
        panic!("expected an exact batch from a populated grid");
    };
    let first_slots = batch.clone();
    let PlaneNeighborFrontier::ExactBatch(second) = stream.frontier(&mut batch, &mut dists) else {
        panic!("repeated frontier call changed frontier kind");
    };
    assert_eq!(first.n, second.n);
    assert_eq!(batch, first_slots);
    assert!(
        dists.windows(2).all(|w| w[0] <= w[1]),
        "batch distances must be sorted ascending"
    );

    stream.advance_frontier();
    if let PlaneNeighborFrontier::ExactBatch(third) = stream.frontier(&mut batch, &mut dists) {
        assert!(
            third.first_dist_sq >= first.unseen_bound - DIST_SQ_TOL,
            "advanced batch outranks the previous certificate: first={:?}, third={:?}",
            first,
            third,
        );
    }
}

#[test]
fn plane_nn_contract_exhaustion_flag() {
    let points = uniform(30, 67);
    let h = Harness::new(points, 3, 1);
    let ctx = DirectedEligibility::from_layout(0, 0, h.layout());
    let query_idx = h.grid.point_indices()[0] as usize;
    let mut scratch = h.grid.make_scratch();
    let mut stream =
        PlaneNeighborStream::new(&h.grid, &h.points, query_idx, &mut scratch, ctx, None);
    let mut batch = Vec::new();
    let mut dists = Vec::new();
    while !matches!(
        stream.frontier(&mut batch, &mut dists),
        PlaneNeighborFrontier::Exhausted
    ) {
        stream.advance_frontier();
    }
    assert!(stream.knn_exhausted());
}
