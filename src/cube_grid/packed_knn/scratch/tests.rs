// PackedKnnTimings is only a unit struct without the `timing` feature, and
// the qi loops index several parallel arrays.
#![allow(clippy::default_constructed_unit_structs, clippy::needless_range_loop)]
use super::*;
use crate::packed_layout::PackedSlotLayout;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

const QUERY_BIN: u8 = 0;
const LOCAL_SHIFT: u32 = 24;
const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;

fn random_unit_points(n: usize, seed: u64) -> Vec<glam::Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n);
    while points.len() < n {
        let p = glam::Vec3::new(
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

fn expected_safe_slots(
    grid: &CubeMapGrid,
    points: &[glam::Vec3],
    query_idx: usize,
    query_local: u32,
    security: f32,
) -> Vec<u32> {
    let query = points[query_idx];
    let mut candidates = Vec::new();
    for (slot, &neighbor_idx_u32) in grid.point_indices().iter().enumerate() {
        let neighbor_idx = neighbor_idx_u32 as usize;
        if neighbor_idx == query_idx {
            continue;
        }
        if (slot as u32) < query_local {
            continue;
        }
        let dot = query.dot(points[neighbor_idx]);
        if dot > security {
            candidates.push((dot, slot as u32));
        }
    }
    candidates.sort_unstable_by(|&(da, sa), &(db, sb)| db.total_cmp(&da).then(sa.cmp(&sb)));
    candidates.into_iter().map(|(_, slot)| slot).collect()
}

fn cell_neighbor_depths(grid: &CubeMapGrid, start_cell: usize, max_depth: u8) -> Vec<u8> {
    let num_cells = grid.cell_offsets().len() - 1;
    let mut depth = vec![u8::MAX; num_cells];
    let mut queue = VecDeque::new();
    depth[start_cell] = 0;
    queue.push_back(start_cell);

    while let Some(cell) = queue.pop_front() {
        let d = depth[cell];
        if d == max_depth {
            continue;
        }
        for &ncell in grid.cell_neighbors(cell) {
            if ncell == u32::MAX {
                continue;
            }
            let next = ncell as usize;
            if depth[next] != u8::MAX {
                continue;
            }
            depth[next] = d + 1;
            queue.push_back(next);
        }
    }

    depth
}

#[test]
fn packed_chunks_match_safe_bruteforce_order_and_bounds() {
    const N: usize = 384;
    const RES: usize = 10;
    const EPS: f32 = 1e-5;

    for &seed in &[11u64, 37] {
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
        let mut scratch = PackedKnnCellScratch::new();
        let mut timings = PackedKnnTimings::default();
        let PreparedPackedGroupStatus::Ready(mut prepared) =
            scratch.prepare_group_directed(&grid, group, &mut timings)
        else {
            panic!("packed prepare unexpectedly fell back to slow path");
        };

        for qi in 0..queries.len() {
            let query_slot = queries[qi];
            let query_idx = grid.point_indices()[query_slot as usize] as usize;
            let security = prepared.security(qi);
            let expected = expected_safe_slots(&grid, &points, query_idx, queries[qi], security);
            let mut emitted = Vec::new();
            let mut prev_bound = 1.0f32;
            let mut stage = PackedStage::Chunk0;

            loop {
                let k = match stage {
                    PackedStage::Chunk0 => 16,
                    PackedStage::Tail => 8,
                    PackedStage::ExpandR2 => 8,
                };
                let mut out = vec![u32::MAX; k];
                let chunk = prepared.next_chunk(qi, stage, k, &mut out, &mut timings);
                match chunk {
                    Some(chunk) => {
                        assert!(
                            chunk.unseen_bound <= prev_bound + EPS,
                            "unseen bound increased for seed={seed}, qi={qi}"
                        );
                        prev_bound = chunk.unseen_bound;
                        emitted.extend_from_slice(&out[..chunk.n]);

                        if let Some(&next_slot) = expected.get(emitted.len()) {
                            let next_idx = grid.point_indices()[next_slot as usize] as usize;
                            let next_dot = points[query_idx].dot(points[next_idx]);
                            assert!(
                                next_dot <= chunk.unseen_bound + EPS,
                                "unseen bound was not conservative for seed={seed}, qi={qi}"
                            );
                        }
                    }
                    None if stage == PackedStage::Chunk0 && prepared.tail_possible(qi) => {
                        prepared.ensure_tail_directed_for(qi, &grid, &mut timings);
                        stage = PackedStage::Tail;
                    }
                    None => break,
                }
            }

            assert_eq!(
                emitted, expected,
                "safe packed order mismatch for seed={seed}, qi={qi}"
            );
        }
    }
}

#[test]
fn expand_r2_security_is_conservative() {
    const N: usize = 384;
    const RES: usize = 10;
    const EPS: f32 = 1e-5;

    for &seed in &[13u64, 41, 97] {
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
        let mut scratch = PackedKnnCellScratch::new();
        let mut timings = PackedKnnTimings::default();
        let PreparedPackedGroupStatus::Ready(mut prepared) =
            scratch.prepare_group_directed(&grid, group, &mut timings)
        else {
            panic!("packed prepare unexpectedly fell back to slow path");
        };

        let depth = cell_neighbor_depths(&grid, cell, 3);
        for qi in 0..queries.len() {
            let query_slot = queries[qi] as usize;
            let query_idx = grid.point_indices()[query_slot] as usize;
            let security2 = prepared.ensure_security2_for(qi, &grid, &mut timings);

            let mut brute_max = f32::NEG_INFINITY;
            for &neighbor_idx_u32 in grid.point_indices() {
                let neighbor_idx = neighbor_idx_u32 as usize;
                let neighbor_cell = grid.point_index_to_cell(neighbor_idx);
                if depth[neighbor_cell] <= 2 {
                    continue;
                }
                let dot = points[query_idx].dot(points[neighbor_idx]);
                brute_max = brute_max.max(dot);
            }

            assert!(
                brute_max <= security2 + EPS,
                "security2 not conservative for seed={seed}, qi={qi}: brute_max={brute_max}, security2={security2}"
            );
        }
    }
}

#[test]
fn expand_r2_band_matches_bruteforce_order_and_bounds() {
    const N: usize = 384;
    const RES: usize = 10;
    const EPS: f32 = 1e-5;

    for &seed in &[19u64, 73] {
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
        let mut scratch = PackedKnnCellScratch::new();
        let mut timings = PackedKnnTimings::default();
        let PreparedPackedGroupStatus::Ready(mut prepared) =
            scratch.prepare_group_directed(&grid, group, &mut timings)
        else {
            panic!("packed prepare unexpectedly fell back to slow path");
        };

        for qi in 0..queries.len() {
            let query_slot = queries[qi];
            let query_idx = grid.point_indices()[query_slot as usize] as usize;
            assert!(
                prepared.ensure_expand_r2_band_directed_for(qi, &grid, &mut timings),
                "cold r=2 expansion unexpectedly exceeded cap for seed={seed}, qi={qi}"
            );

            let security2 = prepared.resume_security(qi);
            let expected = expected_safe_slots(&grid, &points, query_idx, queries[qi], security2);
            let mut emitted = Vec::new();
            let mut prev_bound = 1.0f32;
            let mut stage = PackedStage::Chunk0;

            loop {
                let k = match stage {
                    PackedStage::Chunk0 => 16,
                    PackedStage::Tail => 8,
                    PackedStage::ExpandR2 => 8,
                };
                let mut out = vec![u32::MAX; k];
                let chunk = prepared.next_chunk(qi, stage, k, &mut out, &mut timings);
                match chunk {
                    Some(chunk) => {
                        assert!(
                            chunk.unseen_bound <= prev_bound + EPS,
                            "unseen bound increased for seed={seed}, qi={qi}"
                        );
                        prev_bound = chunk.unseen_bound;
                        emitted.extend_from_slice(&out[..chunk.n]);

                        if let Some(&next_slot) = expected.get(emitted.len()) {
                            let next_idx = grid.point_indices()[next_slot as usize] as usize;
                            let next_dot = points[query_idx].dot(points[next_idx]);
                            assert!(
                                next_dot <= chunk.unseen_bound + EPS,
                                "unseen bound was not conservative for seed={seed}, qi={qi}"
                            );
                        }
                    }
                    None if stage == PackedStage::Chunk0 && prepared.tail_possible(qi) => {
                        prepared.ensure_tail_directed_for(qi, &grid, &mut timings);
                        stage = PackedStage::Tail;
                    }
                    None if stage != PackedStage::ExpandR2 => {
                        stage = PackedStage::ExpandR2;
                    }
                    None => break,
                }
            }

            assert_eq!(
                emitted, expected,
                "expanded packed order mismatch for seed={seed}, qi={qi}"
            );
        }
    }
}
