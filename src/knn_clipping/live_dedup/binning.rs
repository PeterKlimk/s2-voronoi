//! Bin layout and generator assignment helpers.

use glam::Vec3;

use crate::cube_grid::{cell_to_face_ij, CubeMapGrid};

use super::types::{BinId, LocalId};

pub(super) struct BinAssignment {
    pub(super) generator_bin: Vec<BinId>,
    pub(super) global_to_local: Vec<LocalId>,
    /// Packed gen_map: each entry is `(bin << local_shift) | local`.
    pub(super) gen_map: Vec<u32>,
    /// Precomputed shift for extracting bin from packed gen_map.
    pub(super) local_shift: u32,
    /// Precomputed mask for extracting local from packed gen_map.
    pub(super) local_mask: u32,
    pub(super) bin_generators: Vec<Vec<usize>>,
    pub(super) num_bins: usize,
}

impl BinAssignment {
    /// Unpack a gen_map entry into (bin, local).
    #[inline]
    pub(super) fn unpack(&self, packed: u32) -> (BinId, LocalId) {
        let bin = (packed >> self.local_shift) as u8;
        let local = packed & self.local_mask;
        (BinId::from(bin), LocalId::from(local))
    }
}

struct BinLayout {
    bin_res: usize,
    bin_stride: usize,
    num_bins: usize,
}

fn choose_bin_layout(grid_res: usize) -> BinLayout {
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    let target_bins = (threads * 2).clamp(6, 96);
    let target_per_face = (target_bins as f64 / 6.0).max(1.0);
    let mut bin_res = target_per_face.sqrt().ceil() as usize;
    bin_res = bin_res.clamp(1, grid_res.max(1));

    let mut bin_stride = (grid_res + bin_res - 1) / bin_res;
    bin_stride = bin_stride.max(1);
    bin_res = (grid_res + bin_stride - 1) / bin_stride;

    BinLayout {
        bin_res,
        bin_stride,
        num_bins: 6 * bin_res * bin_res,
    }
}

pub(super) fn assign_bins(points: &[Vec3], grid: &CubeMapGrid) -> BinAssignment {
    let n = points.len();
    let layout = choose_bin_layout(grid.res());
    let num_bins = layout.num_bins;

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

    // Construct per-bin generator order directly from the grid's cell-major layout.
    //
    // This preserves the exact `(grid.point_index_to_cell(g), g)` order without a per-bin sort,
    // keeping `LocalId` as the processing rank for edge-check scheduling.
    let res = grid.res();
    let num_cells = 6 * res * res;

    let bin_for_cell = |cell: usize| -> usize {
        let (face, iu, iv) = cell_to_face_ij(cell, res);
        let bu = (iu / layout.bin_stride).min(layout.bin_res - 1);
        let bv = (iv / layout.bin_stride).min(layout.bin_res - 1);
        face * layout.bin_res * layout.bin_res + bv * layout.bin_res + bu
    };

    // Pre-count to avoid reallocations while building the per-bin generator lists.
    let mut counts: Vec<usize> = vec![0; num_bins];
    for cell in 0..num_cells {
        let b = bin_for_cell(cell);
        counts[b] += grid.cell_points(cell).len();
    }

    let mut bin_generators: Vec<Vec<usize>> = (0..num_bins)
        .map(|b| Vec::with_capacity(counts[b]))
        .collect();

    let mut generator_bin: Vec<BinId> = vec![BinId::from(u8::MAX); n];
    let mut global_to_local: Vec<LocalId> = vec![LocalId::from(u32::MAX); n];
    let mut gen_map: Vec<u32> = vec![u32::MAX; n];

    let mut visited = 0usize;
    for cell in 0..num_cells {
        let b_usize = bin_for_cell(cell);
        let b = BinId::from_usize(b_usize);
        for &g_u32 in grid.cell_points(cell) {
            let g = g_u32 as usize;
            debug_assert!(g < n, "grid returned out-of-range point index");

            let local_usize = bin_generators[b_usize].len();
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
            gen_map[g] = (b_usize as u32) << local_shift | (local_usize as u32);
            visited += 1;
        }
    }

    debug_assert_eq!(
        visited, n,
        "grid cell_points did not cover all points (visited={}, n={})",
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
    debug_assert!(
        !gen_map.iter().any(|&m| m == u32::MAX),
        "unassigned gen_map entries"
    );

    BinAssignment {
        generator_bin,
        global_to_local,
        gen_map,
        local_shift,
        local_mask,
        bin_generators,
        num_bins,
    }
}
