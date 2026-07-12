// PackedKnnTimings is only a unit struct without the `timing` feature, and
// the qi loops index several parallel arrays.
#![allow(clippy::default_constructed_unit_structs, clippy::needless_range_loop)]
use super::*;
use crate::packed_layout::PackedSlotLayout;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

#[test]
fn out_of_range_group_uses_slow_path_after_valid_prepare() {
    let points = random_unit_points(64, 91);
    let grid = CubeMapGrid::new(&points, 4);
    let cell = fullest_cell(&grid);
    let start = grid.cell_offsets()[cell] as usize;
    let end = grid.cell_offsets()[cell + 1] as usize;
    let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
    let slot_gen_map: Vec<u32> = (0..points.len() as u32).collect();
    let layout = PackedSlotLayout::new(&slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);
    let mut scratch = PackedKnnCellScratch::new();
    let mut timings = PackedKnnTimings::default();

    let valid = PackedGroupInput::new(cell, QUERY_BIN, &queries, start as u32, layout);
    assert!(matches!(
        scratch.prepare_group_directed(&grid, valid, &mut timings),
        PreparedPackedGroupStatus::Ready(_)
    ));

    let invalid = PackedGroupInput::new(6 * grid.res * grid.res, QUERY_BIN, &[], 0, layout);
    assert!(matches!(
        scratch.prepare_group_directed(&grid, invalid, &mut timings),
        PreparedPackedGroupStatus::SlowPath
    ));
}

#[test]
fn aggregate_candidate_work_uses_slow_path() {
    let points = vec![glam::Vec3::Z; 1_100];
    let grid = CubeMapGrid::new(&points, 4);
    let cell = fullest_cell(&grid);
    let start = grid.cell_offsets()[cell] as usize;
    let end = grid.cell_offsets()[cell + 1] as usize;
    let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
    let slot_gen_map: Vec<u32> = (0..points.len() as u32).collect();
    let layout = PackedSlotLayout::new(&slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);
    let group = PackedGroupInput::new(cell, QUERY_BIN, &queries, start as u32, layout);
    let mut scratch = PackedKnnCellScratch::new();
    let mut timings = PackedKnnTimings::default();

    assert!(matches!(
        scratch.prepare_group_directed(&grid, group, &mut timings),
        PreparedPackedGroupStatus::SlowPath
    ));
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
fn directed_center_chunk_boundaries_match_safe_bruteforce() {
    const EPS: f32 = 1e-5;

    for n in 1..=40usize {
        // Keep every point in one grid cell while mixing exact duplicates and
        // f32-near-equal normalized positions. This exercises every modulo-8
        // center remainder without relying on geometric separation.
        let points: Vec<_> = (0..n)
            .map(|i| {
                if i % 7 == 0 {
                    glam::Vec3::new(0.1, 0.15, 1.0).normalize()
                } else {
                    let x = 0.1 + ((i % 5) as f32 - 2.0) * 2.0e-6;
                    let y = 0.15 + ((i % 3) as f32 - 1.0) * 3.0e-6;
                    glam::Vec3::new(x, y, 1.0).normalize()
                }
            })
            .collect();
        let grid = CubeMapGrid::new(&points, 4);
        let cell = fullest_cell(&grid);
        let start = grid.cell_offsets()[cell] as usize;
        let end = grid.cell_offsets()[cell + 1] as usize;
        assert_eq!(end - start, n, "fixture split across cells for n={n}");

        let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
        let mut slot_gen_map = vec![0u32; n];
        for (slot, &generator) in grid.point_indices()[start..end].iter().enumerate() {
            slot_gen_map[generator as usize] =
                ((QUERY_BIN as u32) << LOCAL_SHIFT) | (start + slot) as u32;
        }
        let layout = PackedSlotLayout::new(&slot_gen_map, LOCAL_SHIFT, LOCAL_MASK);
        let group = PackedGroupInput::new(cell, QUERY_BIN, &queries, start as u32, layout);
        let mut scratch = PackedKnnCellScratch::new();
        let mut timings = PackedKnnTimings::default();
        let PreparedPackedGroupStatus::Ready(mut prepared) =
            scratch.prepare_group_directed(&grid, group, &mut timings)
        else {
            panic!("packed prepare unexpectedly fell back for n={n}");
        };

        for qi in 0..n {
            let query_slot = queries[qi];
            let query_idx = grid.point_indices()[query_slot as usize] as usize;
            let expected =
                expected_safe_slots(&grid, &points, query_idx, query_slot, prepared.security(qi));
            let mut emitted = Vec::new();
            let mut prev_bound = 1.0f32;
            let mut stage = PackedStage::Chunk0;

            loop {
                let k = match stage {
                    PackedStage::Chunk0 => 16,
                    PackedStage::Tail => 8,
                };
                let mut out = vec![u32::MAX; k];
                match prepared.next_chunk(qi, stage, k, &mut out, &mut timings) {
                    Some(chunk) => {
                        assert!(
                            chunk.unseen_bound <= prev_bound + EPS,
                            "unseen bound increased for n={n}, qi={qi}"
                        );
                        prev_bound = chunk.unseen_bound;
                        emitted.extend_from_slice(&out[..chunk.n]);
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
                "directed center mismatch for n={n}, qi={qi}"
            );
        }
    }
}
