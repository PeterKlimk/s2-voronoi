use super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

fn directed_bruteforce_slots(
    grid: &CubeMapGrid,
    points: &[Vec3],
    query_idx: usize,
    query_bin: u8,
    query_local: u32,
    slot_gen_map: &[u32],
    local_shift: u32,
    local_mask: u32,
) -> Vec<u32> {
    let query = points[query_idx];
    let point_indices = grid.point_indices();
    let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(points.len().saturating_sub(1));

    for slot in 0..point_indices.len() {
        let neighbor_idx = point_indices[slot] as usize;
        if neighbor_idx == query_idx {
            continue;
        }

        let packed = slot_gen_map[slot];
        let bin_b = (packed >> local_shift) as u8;
        let local_b = packed & local_mask;
        if bin_b == query_bin && local_b < query_local {
            continue;
        }

        let dot = fp::dot3_f32(
            query.x,
            query.y,
            query.z,
            points[neighbor_idx].x,
            points[neighbor_idx].y,
            points[neighbor_idx].z,
        );
        let dist_sq = (2.0 - 2.0 * dot).max(0.0);
        candidates.push((dist_sq, slot as u32));
    }

    candidates.sort_unstable_by(|&(da, sa), &(db, sb)| da.total_cmp(&db).then(sa.cmp(&sb)));
    candidates.into_iter().map(|(_, slot)| slot).collect()
}

fn assert_cursor_matches_bruteforce_for_map(
    grid: &CubeMapGrid,
    points: &[Vec3],
    slot_gen_map: &[u32],
    local_shift: u32,
    local_mask: u32,
    query_stride: usize,
    seed: u64,
) {
    for query_idx in (0..points.len()).step_by(query_stride) {
        let query_slot = grid.point_index_to_slot(query_idx) as usize;
        let query_packed = slot_gen_map[query_slot];
        let query_bin = (query_packed >> local_shift) as u8;
        let query_local = query_packed & local_mask;
        let ctx = DirectedCtx::new(
            query_bin,
            query_local,
            slot_gen_map,
            local_shift,
            local_mask,
        );
        let mut scratch = grid.make_scratch();
        let mut cursor = grid.directed_no_k_cursor(points[query_idx], query_idx, &mut scratch, ctx);

        let mut emitted = Vec::with_capacity(points.len().saturating_sub(1));
        while let Some(slot) = cursor.pop_next_proven_slot() {
            emitted.push(slot);
        }
        assert!(
            cursor.is_exhausted(),
            "cursor should exhaust for query_idx={query_idx}, seed={seed}"
        );

        let expected = directed_bruteforce_slots(
            grid,
            points,
            query_idx,
            query_bin,
            query_local,
            slot_gen_map,
            local_shift,
            local_mask,
        );

        assert_eq!(
            emitted, expected,
            "directed cursor order mismatch for seed={seed}, query_idx={query_idx}"
        );
    }
}

#[test]
fn directed_no_k_cursor_matches_directed_bruteforce_order() {
    const N: usize = 256;
    const RES: usize = 10;
    const QUERY_STRIDE: usize = 7;
    const QUERY_BIN: u8 = 0;
    const LOCAL_SHIFT: u32 = 24;
    const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;

    for &seed in &[3u64, 17, 1234] {
        let points = random_unit_points(N, seed);
        let grid = CubeMapGrid::new(&points, RES);

        let mut slot_gen_map = vec![0u32; points.len()];
        for (slot, packed) in slot_gen_map.iter_mut().enumerate() {
            *packed = ((QUERY_BIN as u32) << LOCAL_SHIFT) | (slot as u32);
        }

        assert_cursor_matches_bruteforce_for_map(
            &grid,
            &points,
            &slot_gen_map,
            LOCAL_SHIFT,
            LOCAL_MASK,
            QUERY_STRIDE,
            seed,
        );
    }
}

#[test]
fn directed_no_k_cursor_matches_directed_bruteforce_order_mixed_bins() {
    const N: usize = 320;
    const RES: usize = 12;
    const QUERY_STRIDE: usize = 9;
    const LOCAL_SHIFT: u32 = 24;
    const LOCAL_MASK: u32 = (1u32 << LOCAL_SHIFT) - 1;
    const BIN_COUNT: usize = 7;

    for &seed in &[5u64, 29, 777] {
        let points = random_unit_points(N, seed);
        let grid = CubeMapGrid::new(&points, RES);

        let mut slot_gen_map = vec![0u32; points.len()];
        let mut next_local = [0u32; BIN_COUNT];
        let num_cells = grid.cell_offsets().len() - 1;
        for cell in 0..num_cells {
            let bin = ((cell * 3 + cell / 11) % BIN_COUNT) as u8;
            let start = grid.cell_offsets()[cell] as usize;
            let end = grid.cell_offsets()[cell + 1] as usize;
            let local_base = next_local[bin as usize];
            for (offset, slot) in (start..end).enumerate() {
                let local = local_base + offset as u32;
                slot_gen_map[slot] = ((bin as u32) << LOCAL_SHIFT) | local;
            }
            next_local[bin as usize] += (end - start) as u32;
        }

        assert_cursor_matches_bruteforce_for_map(
            &grid,
            &points,
            &slot_gen_map,
            LOCAL_SHIFT,
            LOCAL_MASK,
            QUERY_STRIDE,
            seed,
        );
    }
}
