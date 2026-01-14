//! Bin layout and generator assignment helpers.

use glam::Vec3;

use crate::cube_grid::{cell_to_face_ij, CubeMapGrid};

use super::types::{BinId, LocalId};

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GenMap {
    pub(super) bin: BinId,
    pub(super) local: LocalId,
}

pub(super) struct BinAssignment {
    pub(super) generator_bin: Vec<BinId>,
    pub(super) global_to_local: Vec<LocalId>,
    pub(super) gen_map: Vec<GenMap>,
    pub(super) bin_generators: Vec<Vec<usize>>,
    pub(super) num_bins: usize,
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

    let mut generator_bin: Vec<BinId> = Vec::with_capacity(n);
    let mut counts: Vec<usize> = vec![0; num_bins];
    for i in 0..n {
        let cell = grid.point_index_to_cell(i);
        let (face, iu, iv) = cell_to_face_ij(cell, grid.res());
        let bu = (iu / layout.bin_stride).min(layout.bin_res - 1);
        let bv = (iv / layout.bin_stride).min(layout.bin_res - 1);
        let b = face * layout.bin_res * layout.bin_res + bv * layout.bin_res + bu;
        generator_bin.push(BinId::from_usize(b));
        counts[b] += 1;
    }

    let mut bin_generators: Vec<Vec<usize>> = (0..num_bins)
        .map(|b| Vec::with_capacity(counts[b]))
        .collect();
    for (i, &b) in generator_bin.iter().enumerate() {
        bin_generators[b.as_usize()].push(i);
    }

    for generators in &mut bin_generators {
        generators.sort_unstable_by_key(|&g| (grid.point_index_to_cell(g), g));
    }

    let mut global_to_local: Vec<LocalId> = vec![LocalId::from(0); n];
    for generators in &bin_generators {
        for (local_idx, &global_idx) in generators.iter().enumerate() {
            global_to_local[global_idx] = LocalId::from_usize(local_idx);
        }
    }

    let mut gen_map: Vec<GenMap> = vec![
        GenMap {
            bin: BinId::from(0),
            local: LocalId::from(0)
        };
        n
    ];
    for (global_idx, &bin) in generator_bin.iter().enumerate() {
        gen_map[global_idx] = GenMap {
            bin,
            local: global_to_local[global_idx],
        };
    }

    BinAssignment {
        generator_bin,
        global_to_local,
        gen_map,
        bin_generators,
        num_bins,
    }
}
