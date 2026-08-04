//! Bin layout and generator assignment helpers.

use glam::Vec3;

use crate::cube_grid::{cell_to_face_ij, CubeMapGrid};

use super::types::{BinId, LocalId};

pub(crate) struct BinAssignment {
    pub(crate) generator_bin: Vec<BinId>,
    /// Packed `(bin, local)` assignment indexed by generator id.
    pub(crate) generator_layout: Vec<u32>,
    /// Packed slot_gen_map: each entry is `(bin << local_shift) | local`. Indexed by slot (SOA index).
    pub(crate) slot_gen_map: Vec<u32>,
    /// Precomputed shift for extracting bin from packed gen_map/slot_gen_map.
    pub(crate) local_shift: u32,
    /// Precomputed mask for extracting local from packed gen_map/slot_gen_map.
    pub(crate) local_mask: u32,
    pub(crate) bin_generators: Vec<Vec<usize>>,
    /// Bin-ordered non-empty grid cells. `bin_cell_offsets` indexes the run for
    /// each bin. This reuses the former per-generator point-to-cell storage.
    pub(crate) bin_cells: Vec<u32>,
    pub(crate) bin_cell_offsets: Vec<u32>,
    pub(crate) num_bins: usize,
}

impl BinAssignment {
    pub(crate) fn generator_bin_local(&self, generator: usize) -> (BinId, LocalId) {
        let packed = self.generator_layout[generator];
        (
            BinId::from((packed >> self.local_shift) as u8),
            LocalId::from(packed & self.local_mask),
        )
    }

    pub(crate) fn bin_cells(&self, bin: usize) -> &[u32] {
        let start = self.bin_cell_offsets[bin] as usize;
        let end = self.bin_cell_offsets[bin + 1] as usize;
        &self.bin_cells[start..end]
    }

    #[cfg(any(debug_assertions, test))]
    pub(crate) fn generator_local(&self, generator: usize) -> LocalId {
        LocalId::from(self.generator_layout[generator] & self.local_mask)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PackedLayoutCapacityError {
    pub(crate) bin: usize,
    pub(crate) local_population: usize,
    pub(crate) num_bins: usize,
    pub(crate) local_shift: u32,
    pub(crate) local_mask: u32,
}

struct BinLayout {
    bin_res: usize,
    bin_stride: usize,
    num_bins: usize,
}

const IMBALANCED_BIN_TARGET: usize = 216;

#[inline]
fn default_target_bin_count(threads: usize) -> usize {
    // The 24-shard layout keeps the largest-first queue at least three tasks
    // deep per worker through eight workers. Above that, use the 96-shard
    // layout so uneven shard costs leave enough work to fill the closing wave.
    if threads > 8 {
        96
    } else {
        threads.saturating_mul(2)
    }
}

/// Target shard count from threads with the `VORONOI_MESH_BIN_COUNT` override,
/// clamped to `[6, 96]` so every cube face can own at least one shard.
fn target_bin_count_with_override(threads: usize) -> (usize, bool) {
    match std::env::var("VORONOI_MESH_BIN_COUNT") {
        Ok(var) => (
            var.parse()
                .unwrap_or_else(|_| default_target_bin_count(threads))
                .clamp(6, 96),
            true,
        ),
        Err(_) => (default_target_bin_count(threads).clamp(6, 96), false),
    }
}

fn choose_bin_layout_for_target(grid_res: usize, target_bins: usize) -> BinLayout {
    let target_per_face = (target_bins as f64 / 6.0).max(1.0);
    let mut bin_res = target_per_face.sqrt().ceil() as usize;
    bin_res = bin_res.clamp(1, grid_res.max(1));

    let mut bin_stride = grid_res.div_ceil(bin_res);
    bin_stride = bin_stride.max(1);
    bin_res = grid_res.div_ceil(bin_stride);

    BinLayout {
        bin_res,
        bin_stride,
        num_bins: 6 * bin_res * bin_res,
    }
}

#[inline]
fn bin_for_cell(cell: usize, grid_res: usize, layout: &BinLayout) -> usize {
    let (face, iu, iv) = cell_to_face_ij(cell, grid_res);
    let bu = (iu / layout.bin_stride).min(layout.bin_res - 1);
    let bv = (iv / layout.bin_stride).min(layout.bin_res - 1);
    face * layout.bin_res * layout.bin_res + bv * layout.bin_res + bu
}

fn max_bin_population(grid: &CubeMapGrid, layout: &BinLayout) -> usize {
    let mut populations = vec![0usize; layout.num_bins];
    for (cell, offsets) in grid.cell_offsets().windows(2).enumerate() {
        populations[bin_for_cell(cell, grid.res(), layout)] += (offsets[1] - offsets[0]) as usize;
    }
    populations.into_iter().max().unwrap_or(0)
}

#[inline]
fn should_refine_imbalanced_layout(
    point_count: usize,
    coarse_bins: usize,
    coarse_max: usize,
    fine_max: usize,
) -> bool {
    // Require a coarse bin holding at least seven times the mean population,
    // then require the finer layout to reduce that absolute maximum by 10%.
    // Integer products make the scheduling decision reproducible.
    (coarse_max as u128) * (coarse_bins as u128) >= (point_count as u128) * 7
        && fine_max.saturating_mul(10) <= coarse_max.saturating_mul(9)
}

#[inline]
fn is_severely_imbalanced(point_count: usize, coarse_bins: usize, coarse_max: usize) -> bool {
    (coarse_max as u128) * (coarse_bins as u128) >= (point_count as u128) * 7
}

fn validate_local_capacity(
    bin: usize,
    local_population: usize,
    num_bins: usize,
    local_shift: u32,
    local_mask: u32,
) -> Result<(), PackedLayoutCapacityError> {
    if (local_population as u32) <= local_mask {
        return Ok(());
    }
    Err(PackedLayoutCapacityError {
        bin,
        local_population,
        num_bins,
        local_shift,
        local_mask,
    })
}

fn max_local_population(num_bins: usize) -> usize {
    let bin_bits = if num_bins <= 1 {
        1
    } else {
        32 - (num_bins as u32 - 1).leading_zeros()
    };
    ((1u64 << (32 - bin_bits)) - 1) as usize
}

pub(crate) fn assign_bins(
    points: &[Vec3],
    grid: &CubeMapGrid,
    point_cell_storage: Vec<u32>,
) -> Result<BinAssignment, PackedLayoutCapacityError> {
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    let (target_bins, explicit_override) = target_bin_count_with_override(threads);
    let layout = choose_bin_layout_for_target(grid.res(), target_bins);

    // Construct per-bin generator order directly from the grid's cell-major layout.
    //
    // This preserves the exact `(grid.point_index_to_cell(g), g)` order without a per-bin sort,
    // keeping `LocalId` as the processing rank for edge-check scheduling.
    let res = grid.res();
    let assign = |layout: &BinLayout, storage| {
        assign_bins_with(
            points.len(),
            6 * res * res,
            grid.cell_offsets(),
            grid.point_indices(),
            layout.num_bins,
            |cell| bin_for_cell(cell, res, layout),
            storage,
        )
    };
    let assignment = assign(&layout, point_cell_storage)?;

    // Preserve explicit diagnostic overrides. The adaptive layout applies only
    // to the ordinary high-core 96-bin policy, where severe population skew
    // otherwise leaves one or two construction tasks on the critical tail.
    // The completed coarse assignment supplies the first gate for free, so
    // ordinary inputs perform no additional grid scan.
    let coarse_max = assignment
        .bin_generators
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    if !explicit_override
        && layout.num_bins == 96
        && is_severely_imbalanced(points.len(), layout.num_bins, coarse_max)
    {
        let fine = choose_bin_layout_for_target(grid.res(), IMBALANCED_BIN_TARGET);
        let fine_max = max_bin_population(grid, &fine);
        if should_refine_imbalanced_layout(points.len(), layout.num_bins, coarse_max, fine_max)
            && fine_max <= max_local_population(fine.num_bins)
        {
            return assign(&fine, assignment.bin_cells);
        }
    }

    Ok(assignment)
}

/// Grid-agnostic assignment core over a CSR (cell_offsets, point_indices)
/// layout: locals are assigned in cell-major order, the invariant the
/// directed edge-check scheduling relies on.
pub(crate) fn assign_bins_with(
    n: usize,
    num_cells: usize,
    cell_offsets: &[u32],
    point_indices: &[u32],
    num_bins: usize,
    bin_for_cell: impl Fn(usize) -> usize,
    mut point_cell_storage: Vec<u32>,
) -> Result<BinAssignment, PackedLayoutCapacityError> {
    // Compute bit layout for packed gen_map.
    // bin_bits: minimum bits needed to represent num_bins - 1
    // local_bits: remaining bits for local_id
    let bin_bits = if num_bins <= 1 {
        1
    } else {
        32 - (num_bins as u32 - 1).leading_zeros()
    };
    let local_shift = 32 - bin_bits;
    let local_mask = (1u32 << local_shift) - 1;

    let cell_points = |cell: usize| -> &[u32] {
        &point_indices[cell_offsets[cell] as usize..cell_offsets[cell + 1] as usize]
    };

    // Pre-count to avoid reallocations while building the per-bin generator lists.
    let mut counts: Vec<usize> = vec![0; num_bins];
    let mut cell_counts: Vec<usize> = vec![0; num_bins];
    for cell in 0..num_cells {
        let b = bin_for_cell(cell);
        let len = cell_points(cell).len();
        counts[b] += len;
        cell_counts[b] += usize::from(len != 0);
    }

    let mut bin_cell_offsets = Vec::with_capacity(num_bins + 1);
    bin_cell_offsets.push(0u32);
    for count in cell_counts {
        let next = bin_cell_offsets
            .last()
            .copied()
            .unwrap()
            .checked_add(count as u32)
            .expect("non-empty cell count exceeds u32");
        bin_cell_offsets.push(next);
    }
    let nonempty_cells = bin_cell_offsets[num_bins] as usize;
    point_cell_storage.clear();
    point_cell_storage.resize(nonempty_cells, u32::MAX);
    let mut next_bin_cell = bin_cell_offsets[..num_bins].to_vec();

    let mut bin_generators: Vec<Vec<usize>> = (0..num_bins)
        .map(|b| Vec::with_capacity(counts[b]))
        .collect();

    let mut generator_bin: Vec<BinId> = vec![BinId::from(u8::MAX); n];
    let mut generator_layout: Vec<u32> = vec![u32::MAX; n];
    // Cells and their point spans are visited in CSR order, so the packed slot
    // map is produced sequentially. Build it directly instead of initializing
    // an n-element sentinel buffer that every entry immediately overwrites.
    let mut slot_gen_map: Vec<u32> = Vec::with_capacity(n);

    let mut visited = 0usize;
    for (cell, win) in cell_offsets.windows(2).enumerate() {
        let b_usize = bin_for_cell(cell);
        let b = BinId::from_usize(b_usize);
        // Points of a cell occupy contiguous slots cell_start.. in this order,
        // so we can fill slot_gen_map inline here from each point's own (bin,
        // local) — no separate O(n) pass, no generator assignment
        // read-back.
        let cell_start = win[0] as usize;
        let cell_end = win[1] as usize;
        if cell_start != cell_end {
            let dst = next_bin_cell[b_usize] as usize;
            point_cell_storage[dst] = cell as u32;
            next_bin_cell[b_usize] += 1;
        }
        for (offset, &g_u32) in point_indices[cell_start..cell_end].iter().enumerate() {
            let g = g_u32 as usize;
            debug_assert!(g < n, "grid returned out-of-range point index");

            let local_usize = bin_generators[b_usize].len();
            validate_local_capacity(b_usize, local_usize, num_bins, local_shift, local_mask)?;
            let local = LocalId::from_usize(local_usize);
            bin_generators[b_usize].push(g);

            generator_bin[g] = b;
            let packed = ((b.as_u8() as u32) << local_shift) | local.as_u32();
            generator_layout[g] = packed;

            // Pack: (bin << local_shift) | local
            debug_assert!(
                (local_usize as u32) <= local_mask,
                "local_id {} exceeds {} bits (max {})",
                local_usize,
                local_shift,
                local_mask
            );
            debug_assert_eq!(slot_gen_map.len(), cell_start + offset);
            slot_gen_map.push(packed);
            visited += 1;
        }
    }

    debug_assert_eq!(
        visited, n,
        "grid cells did not cover all points (visited={}, n={})",
        visited, n
    );
    debug_assert!(
        !generator_bin.iter().any(|&b| b == BinId::from(u8::MAX)),
        "unassigned generator bin entries"
    );
    debug_assert!(
        !generator_layout.contains(&u32::MAX),
        "unassigned generator_layout entries"
    );

    // slot_gen_map is now filled inline during the scatter pass above
    // (fused — no separate read-back pass).
    debug_assert_eq!(slot_gen_map.len(), n, "incomplete slot_gen_map");

    Ok(BinAssignment {
        generator_bin,
        generator_layout,
        slot_gen_map,
        local_shift,
        local_mask,
        bin_generators,
        bin_cells: point_cell_storage,
        bin_cell_offsets,
        num_bins,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        assign_bins_with, default_target_bin_count, max_local_population,
        should_refine_imbalanced_layout, validate_local_capacity,
    };

    #[test]
    fn imbalanced_layout_requires_both_overload_and_maximum_reduction() {
        assert!(should_refine_imbalanced_layout(1_000, 100, 70, 63));
        assert!(!should_refine_imbalanced_layout(1_000, 100, 69, 60));
        assert!(!should_refine_imbalanced_layout(1_000, 100, 70, 64));
    }

    #[test]
    fn finer_layout_local_capacity_is_explicit() {
        assert_eq!(max_local_population(96), (1 << 25) - 1);
        assert_eq!(max_local_population(216), (1 << 24) - 1);
    }

    #[test]
    fn high_core_default_increases_shard_granularity() {
        assert_eq!(default_target_bin_count(1), 2);
        assert_eq!(default_target_bin_count(8), 16);
        assert_eq!(default_target_bin_count(9), 96);
        assert_eq!(default_target_bin_count(11), 96);
        assert_eq!(default_target_bin_count(12), 96);
        assert_eq!(default_target_bin_count(16), 96);
        assert_eq!(default_target_bin_count(32), 96);
    }

    #[test]
    fn packed_local_capacity_accepts_values_within_mask() {
        assert!(validate_local_capacity(3, 255, 96, 8, 255).is_ok());
    }

    #[test]
    fn packed_local_capacity_rejects_values_above_mask() {
        let err = validate_local_capacity(7, 256, 96, 8, 255).unwrap_err();
        assert_eq!(err.bin, 7);
        assert_eq!(err.local_population, 256);
        assert_eq!(err.num_bins, 96);
        assert_eq!(err.local_shift, 8);
        assert_eq!(err.local_mask, 255);
    }

    #[test]
    fn generator_layout_round_trips_bin_and_local() {
        let assignment =
            assign_bins_with(4, 2, &[0, 2, 4], &[2, 0, 3, 1], 2, |cell| cell, Vec::new()).unwrap();

        assert_eq!(assignment.bin_cells, [0, 1]);
        assert_eq!(assignment.bin_cell_offsets, [0, 1, 2]);

        let expected = [(0, 1), (1, 1), (0, 0), (1, 0)];
        for (generator, &(bin, local)) in expected.iter().enumerate() {
            let (actual_bin, actual_local) = assignment.generator_bin_local(generator);
            assert_eq!(actual_bin.as_usize(), bin);
            assert_eq!(actual_local.as_usize(), local);
            assert_eq!(assignment.generator_local(generator), actual_local);
        }
    }
}
