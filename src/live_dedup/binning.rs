//! Bin layout and generator assignment helpers.

use glam::Vec3;

use crate::cube_grid::{cell_to_face_ij, CubeMapGrid};

use super::types::{BinId, LocalId};

pub(crate) struct BinAssignment {
    pub(crate) generator_bin: Vec<BinId>,
    pub(crate) global_to_local: Vec<LocalId>,
    /// Packed slot_gen_map: each entry is `(bin << local_shift) | local`. Indexed by slot (SOA index).
    pub(crate) slot_gen_map: Vec<u32>,
    /// Precomputed shift for extracting bin from packed gen_map/slot_gen_map.
    pub(crate) local_shift: u32,
    /// Precomputed mask for extracting local from packed gen_map/slot_gen_map.
    pub(crate) local_mask: u32,
    pub(crate) bin_generators: Vec<Vec<usize>>,
    pub(crate) num_bins: usize,
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

/// Target shard count from threads (x2) with the `S2_BIN_COUNT` override,
/// clamped to `[min_bins, 96]`. Shared by the spherical (min 6 — one per
/// face) and planar (min 1) layouts so the env knob has one parser.
pub(crate) fn target_bin_count(min_bins: usize) -> usize {
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "parallel"))]
    let threads = 1;

    if let Ok(var) = std::env::var("S2_BIN_COUNT") {
        var.parse().unwrap_or(threads * 2)
    } else {
        threads * 2
    }
    .clamp(min_bins, 96)
}

fn choose_bin_layout(grid_res: usize) -> BinLayout {
    let target_bins = target_bin_count(6);
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

pub(crate) fn assign_bins(
    points: &[Vec3],
    grid: &CubeMapGrid,
) -> Result<BinAssignment, PackedLayoutCapacityError> {
    let layout = choose_bin_layout(grid.res());

    // Construct per-bin generator order directly from the grid's cell-major layout.
    //
    // This preserves the exact `(grid.point_index_to_cell(g), g)` order without a per-bin sort,
    // keeping `LocalId` as the processing rank for edge-check scheduling.
    let res = grid.res();
    let bin_for_cell = |cell: usize| -> usize {
        let (face, iu, iv) = cell_to_face_ij(cell, res);
        let bu = (iu / layout.bin_stride).min(layout.bin_res - 1);
        let bv = (iv / layout.bin_stride).min(layout.bin_res - 1);
        face * layout.bin_res * layout.bin_res + bv * layout.bin_res + bu
    };

    assign_bins_with(
        points.len(),
        6 * res * res,
        grid.cell_offsets(),
        grid.point_indices(),
        layout.num_bins,
        bin_for_cell,
    )
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
    for cell in 0..num_cells {
        let b = bin_for_cell(cell);
        counts[b] += cell_points(cell).len();
    }

    let mut bin_generators: Vec<Vec<usize>> = (0..num_bins)
        .map(|b| Vec::with_capacity(counts[b]))
        .collect();

    let mut generator_bin: Vec<BinId> = vec![BinId::from(u8::MAX); n];
    let mut global_to_local: Vec<LocalId> = vec![LocalId::from(u32::MAX); n];
    let mut slot_gen_map: Vec<u32> = vec![u32::MAX; n];

    let mut visited = 0usize;
    for (cell, win) in cell_offsets.windows(2).enumerate() {
        let b_usize = bin_for_cell(cell);
        let b = BinId::from_usize(b_usize);
        // Points of a cell occupy contiguous slots cell_start.. in this order,
        // so we can fill slot_gen_map inline here from each point's own (bin,
        // local) — no separate O(n) pass, no generator_bin/global_to_local
        // read-back.
        let cell_start = win[0] as usize;
        let cell_end = win[1] as usize;
        for (offset, &g_u32) in point_indices[cell_start..cell_end].iter().enumerate() {
            let g = g_u32 as usize;
            debug_assert!(g < n, "grid returned out-of-range point index");

            let local_usize = bin_generators[b_usize].len();
            validate_local_capacity(b_usize, local_usize, num_bins, local_shift, local_mask)?;
            let local = LocalId::from_usize(local_usize);
            bin_generators[b_usize].push(g);

            generator_bin[g] = b;
            global_to_local[g] = local;

            // Pack: (bin << local_shift) | local
            debug_assert!(
                (local_usize as u32) <= local_mask,
                "local_id {} exceeds {} bits (max {})",
                local_usize,
                local_shift,
                local_mask
            );
            slot_gen_map[cell_start + offset] =
                ((b.as_u8() as u32) << local_shift) | local.as_u32();
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
        !global_to_local
            .iter()
            .any(|&l| l == LocalId::from(u32::MAX)),
        "unassigned global_to_local entries"
    );

    // slot_gen_map is now filled inline during the scatter pass above
    // (fused — no separate read-back pass).
    debug_assert!(
        !slot_gen_map.contains(&u32::MAX),
        "unassigned slot_gen_map entries"
    );

    Ok(BinAssignment {
        generator_bin,
        global_to_local,
        slot_gen_map,
        local_shift,
        local_mask,
        bin_generators,
        num_bins,
    })
}

#[cfg(test)]
mod tests {
    use super::validate_local_capacity;

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
}
