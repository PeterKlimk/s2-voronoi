//! Contract tests for the neighbor-source layer, in isolation from the
//! Voronoi pipeline.
//!
//! The durable contract (what the cell builder actually relies on):
//!
//! 1. **Eligible set**: across its lifetime the stream emits exactly the
//!    directed-eligible points (other-bin points; same-bin points in later
//!    cells; same-bin same-cell points with local >= query local, excluding
//!    self). Duplicate emissions across stage boundaries are permitted and
//!    deduplicated by the consumer.
//! 2. **Certificates**: every frontier's unseen bound conservatively bounds
//!    the dot product of every eligible point not yet emitted.
//!
//! Emission *order* is deliberately not part of this contract (only
//! approximate nearest-first matters for clipping efficiency), so a frontier
//! rework that changes ordering must still pass this suite unchanged.
//!
//! Scenarios stress the cube-grid geometry specifically: corners (7-cell
//! neighborhoods), face-edge seams, exact symmetric coordinates, cross-face
//! traversal, antipodal pairs, near-wall classification, clusters, bimodal
//! densities, and tiny inputs.

use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::super::packed_knn::{
    PackedGroupInput, PackedKnnCellScratch, PackedKnnTimings, PackedQuery,
    PreparedPackedGroupStatus,
};
use super::super::projection::{face_uv_to_3d, st_to_uv};
use super::super::query::{DirectedEligibility, DirectedNeighborFrontier, DirectedNeighborStream};
use super::super::CubeMapGrid;
use crate::packed_layout::PackedSlotLayout;
use crate::policy::PackedNeighborPolicy;

const LOCAL_SHIFT: u32 = 24;
const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;
const DOT_TOL: f32 = 1e-6;

struct Harness {
    points: Vec<Vec3>,
    grid: CubeMapGrid,
    slot_gen_map: Vec<u32>,
    cell_of_slot: Vec<usize>,
    bin_of_cell: Vec<u8>,
}

impl Harness {
    fn new(points: Vec<Vec3>, res: usize, num_bins: u8) -> Self {
        assert!(num_bins >= 1);
        let grid = CubeMapGrid::new(&points, res);
        let num_cells = grid.cell_offsets().len() - 1;

        // Contiguous cell ranges per bin (cells within a bin share it, which
        // is the layout invariant the production binning also guarantees).
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

    fn slot_dot(&self, query_slot: u32, slot: u32) -> f32 {
        let qi = self.grid.point_indices()[query_slot as usize] as usize;
        let pi = self.grid.point_indices()[slot as usize] as usize;
        self.points[qi].dot(self.points[pi])
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
        mut stream: DirectedNeighborStream<'_, '_, '_, '_>,
    ) -> Vec<u32> {
        let eligible = self.brute_eligible(query_slot);
        let mut unseen: std::collections::HashSet<u32> = eligible.iter().copied().collect();
        let mut emitted: Vec<u32> = Vec::new();
        let mut emitted_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut batch: Vec<u32> = Vec::new();

        let best_unseen = |unseen: &std::collections::HashSet<u32>| -> f32 {
            // NEG_INFINITY when empty: any bound (including -inf) covers it.
            unseen
                .iter()
                .map(|&slot| self.slot_dot(query_slot, slot))
                .fold(f32::NEG_INFINITY, f32::max)
        };

        loop {
            match stream.frontier(&mut batch) {
                DirectedNeighborFrontier::ExactBatch(result) => {
                    for &slot in &batch[..result.n] {
                        assert!(
                            slot != query_slot,
                            "{name}: stream emitted the query slot itself"
                        );
                        if emitted_set.insert(slot) {
                            emitted.push(slot);
                        }
                        unseen.remove(&slot);
                    }
                    assert!(
                        best_unseen(&unseen) <= result.unseen_bound + DOT_TOL,
                        "{name}: batch unseen_bound {} fails to cover an unseen eligible \
                         point with dot {} (query slot {query_slot})",
                        result.unseen_bound,
                        best_unseen(&unseen)
                    );
                    stream.advance_frontier();
                }
                DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                    assert!(
                        best_unseen(&unseen) <= dot_upper_bound + DOT_TOL,
                        "{name}: bounded frontier {} fails to cover an unseen eligible \
                         point with dot {} (query slot {query_slot})",
                        dot_upper_bound,
                        best_unseen(&unseen)
                    );
                    stream.advance_frontier();
                }
                DirectedNeighborFrontier::Exhausted => break,
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

    /// Run the contract for every (sampled) query through the cursor-only
    /// path and the packed path.
    fn check_all(&self, name: &str) {
        let n = self.points.len();
        let stride = (n / 128).max(1);

        // Cursor-only path (packed = None).
        for slot in (0..n as u32).step_by(stride) {
            let query_idx = self.grid.point_indices()[slot as usize] as usize;
            let ctx = DirectedEligibility::from_layout(
                self.bin_of_cell[self.cell_of_slot[slot as usize]],
                slot,
                self.layout(),
            );
            let mut scratch = self.grid.make_scratch();
            let stream = DirectedNeighborStream::new(
                &self.grid,
                &self.points,
                query_idx,
                &mut scratch,
                ctx,
                None,
            );
            self.collect_and_check(&format!("{name}/cursor"), slot, stream);
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
            let group = PackedGroupInput::new(
                cell,
                self.bin_of_cell[cell],
                &queries,
                start as u32,
                self.layout(),
            );
            for &expand_r2 in &[false, true] {
                let mut packed_scratch = PackedKnnCellScratch::new();
                let mut timings = PackedKnnTimings::default();
                let PreparedPackedGroupStatus::Ready(mut prepared) =
                    packed_scratch.prepare_group_directed(&self.grid, group, &mut timings)
                else {
                    // SlowPath groups are exercised by the cursor-only pass.
                    continue;
                };
                for qi in 0..queries.len() {
                    let slot = queries[qi];
                    let query_idx = self.grid.point_indices()[slot as usize] as usize;
                    let ctx = DirectedEligibility::from_layout(
                        self.bin_of_cell[cell],
                        slot,
                        self.layout(),
                    );
                    let mut scratch = self.grid.make_scratch();
                    let packed = PackedQuery::new(
                        &mut prepared,
                        &mut timings,
                        qi,
                        PackedNeighborPolicy::for_point_count(n, expand_r2),
                    );
                    let stream = DirectedNeighborStream::new(
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

fn random_unit(rng: &mut ChaCha8Rng) -> Vec3 {
    loop {
        let p = Vec3::new(
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0f32..1.0),
        );
        let len_sq = p.length_squared();
        if len_sq > 1e-6 && len_sq <= 1.0 {
            return p / len_sq.sqrt();
        }
    }
}

fn uniform(n: usize, seed: u64) -> Vec<Vec3> {
    let mut r = rng(seed);
    (0..n).map(|_| random_unit(&mut r)).collect()
}

fn jitter(base: Vec3, scale: f32, r: &mut ChaCha8Rng) -> Vec3 {
    (base
        + Vec3::new(
            r.gen_range(-scale..scale),
            r.gen_range(-scale..scale),
            r.gen_range(-scale..scale),
        ))
    .normalize()
}

/// The 26 symmetric positions: 8 corners, 12 edge midlines, 6 face centers.
fn symmetric_positions() -> Vec<Vec3> {
    let inv3 = 1.0f32 / 3.0f32.sqrt();
    let inv2 = 1.0f32 / 2.0f32.sqrt();
    let mut out = Vec::new();
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                out.push(Vec3::new(sx * inv3, sy * inv3, sz * inv3));
            }
        }
    }
    for s in [-1.0f32, 1.0] {
        for t in [-1.0f32, 1.0] {
            out.push(Vec3::new(s * inv2, t * inv2, 0.0));
            out.push(Vec3::new(s * inv2, 0.0, t * inv2));
            out.push(Vec3::new(0.0, s * inv2, t * inv2));
        }
    }
    for s in [-1.0f32, 1.0] {
        out.push(Vec3::new(s, 0.0, 0.0));
        out.push(Vec3::new(0.0, s, 0.0));
        out.push(Vec3::new(0.0, 0.0, s));
    }
    out
}

// === Scenarios ===

#[test]
fn nn_contract_uniform() {
    Harness::new(uniform(320, 5), 10, 1).check_all("uniform");
    Harness::new(uniform(320, 29), 10, 3).check_all("uniform_3bins");
}

#[test]
fn nn_contract_coarse_grids() {
    // res=1: six cells total, every query crosses faces.
    Harness::new(uniform(60, 7), 1, 1).check_all("res1");
    Harness::new(uniform(200, 11), 2, 2).check_all("res2_2bins");
    Harness::new(uniform(150, 13), 3, 1).check_all("res3");
}

#[test]
fn nn_contract_corner_clusters() {
    // Jittered clusters at the 8 cube corners: 7-cell neighborhoods and
    // three-face stitching.
    let mut r = rng(17);
    let inv3 = 1.0f32 / 3.0f32.sqrt();
    let mut points = Vec::new();
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = Vec3::new(sx * inv3, sy * inv3, sz * inv3);
                for _ in 0..12 {
                    points.push(jitter(corner, 1e-3, &mut r));
                }
            }
        }
    }
    Harness::new(points, 4, 1).check_all("corner_clusters");
}

#[test]
fn nn_contract_exact_symmetric_positions() {
    // Exact seam coordinates (the known-degenerate regime for the clipper;
    // the NN layer must handle them exactly).
    let mut points = symmetric_positions();
    points.extend(uniform(40, 19));
    Harness::new(points.clone(), 3, 1).check_all("exact_seams_res3");
    Harness::new(points, 4, 2).check_all("exact_seams_res4");
}

#[test]
fn nn_contract_single_face_and_sparse() {
    // All points on the +x face: five faces empty, shells must cross.
    let mut r = rng(23);
    let one_face: Vec<Vec3> = (0..150)
        .map(|_| Vec3::new(1.0, r.gen_range(-0.9f32..0.9), r.gen_range(-0.9f32..0.9)).normalize())
        .collect();
    Harness::new(one_face, 6, 1).check_all("one_face");

    // One point per face: maximal traversal between any pair.
    let one_per_face = vec![
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    Harness::new(one_per_face, 4, 1).check_all("one_per_face");

    // Antipodal pairs: opposite-face traversal.
    let mut r = rng(31);
    let mut antipodal = Vec::new();
    for _ in 0..5 {
        let p = random_unit(&mut r);
        antipodal.push(p);
        antipodal.push(-p);
    }
    Harness::new(antipodal, 4, 1).check_all("antipodal");
}

#[test]
fn nn_contract_tiny_inputs() {
    let mut r = rng(37);
    let two = vec![random_unit(&mut r), random_unit(&mut r)];
    Harness::new(two, 4, 1).check_all("n2");

    let three: Vec<Vec3> = (0..3).map(|_| random_unit(&mut r)).collect();
    Harness::new(three, 4, 1).check_all("n3");
}

#[test]
fn nn_contract_near_wall_classification() {
    // Points within ~1e-6 of cell walls (and exactly on them): the contract
    // must hold regardless of which side classification puts them on.
    const RES: usize = 8;
    let mut points = Vec::new();
    let mut r = rng(41);
    for face in 0..6 {
        for k in 1..RES {
            let wall_uv = st_to_uv(k as f32 / RES as f32);
            for delta in [-1e-6f32, 0.0, 1e-6] {
                let v = r.gen_range(-0.8f32..0.8);
                points.push(face_uv_to_3d(face, wall_uv + delta, v).normalize());
            }
        }
    }
    Harness::new(points, RES, 1).check_all("near_walls");
}

#[test]
fn nn_contract_clustered_and_bimodal() {
    let mut r = rng(43);
    // Tight cap cluster plus axis anchors.
    let center = Vec3::new(0.6, 0.64, 0.48).normalize();
    let mut clustered: Vec<Vec3> = (0..200).map(|_| jitter(center, 0.01, &mut r)).collect();
    clustered.extend([
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ]);
    Harness::new(clustered, 6, 1).check_all("clustered");

    // Bimodal: dense cap + isolated far points whose queries must traverse
    // many empty cells (the deep-shell regime).
    let mut bimodal: Vec<Vec3> = (0..300).map(|_| jitter(Vec3::Z, 0.05, &mut r)).collect();
    for _ in 0..12 {
        bimodal.push(jitter(Vec3::NEG_Z, 0.5, &mut r));
    }
    Harness::new(bimodal, 8, 2).check_all("bimodal");
}

#[test]
fn nn_contract_duplicate_positions() {
    let mut points = uniform(100, 47);
    for i in 0..5 {
        points.push(points[i * 13]);
    }
    Harness::new(points, 4, 1).check_all("duplicates");
}
