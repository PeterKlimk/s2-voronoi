//! Compute entry points for the kNN + clipping Voronoi backend.

use glam::Vec3;

use super::edge_repair;
use super::live_dedup;
use super::timing::{Timer, TimingBuilder};
use super::{
    constants, merge_close_points, MergeResult, TerminationConfig, KNN_GRID_TARGET_DENSITY,
};
use crate::cube_grid::CubeMapGrid;
#[cfg(feature = "timing")]
use crate::cube_grid::CubeMapGridBuildTimings;
use crate::VoronoiConfig;

pub(super) fn compute_voronoi_knn_clipping_owned_core(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_threshold: Option<f32>,
    skip_preprocess: bool,
) -> crate::SphericalVoronoi {
    let mut tb = TimingBuilder::new();

    // Preprocessing: merge close points
    let t = Timer::start();
    let (effective_points, merge_result): (Option<Vec<Vec3>>, Option<MergeResult>) =
        if skip_preprocess {
            (None, None)
        } else {
            let threshold = preprocess_threshold
                .unwrap_or_else(|| constants::merge_threshold_for_density(points.len()));
            let mut result = merge_close_points(&points, threshold);
            if result.num_merged > 0 {
                let pts = std::mem::take(&mut result.effective_points);
                (Some(pts), Some(result))
            } else {
                // No merges: use the original owned points (avoid an extra full copy).
                (None, None)
            }
        };
    tb.set_preprocess(t.elapsed());
    let needs_remap = merge_result.is_some();

    let effective_points_ref: &[Vec3] = match &effective_points {
        Some(v) => v.as_slice(),
        None => points.as_slice(),
    };

    // Build cube-map grid on effective points (this is the timed grid build).
    let t = Timer::start();
    let n = effective_points_ref.len();
    let target = KNN_GRID_TARGET_DENSITY.max(1.0);
    let res = ((n as f64 / (6.0 * target)).sqrt() as usize).max(4);
    #[cfg(feature = "timing")]
    let mut grid_build_timings = CubeMapGridBuildTimings::default();
    #[cfg(feature = "timing")]
    let grid =
        CubeMapGrid::new_with_build_timings(effective_points_ref, res, &mut grid_build_timings);
    #[cfg(not(feature = "timing"))]
    let grid = CubeMapGrid::new(effective_points_ref, res);
    tb.set_knn_build(t.elapsed());
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(grid_build_timings.clone());

    // Build cells using sharded live dedup
    let t = Timer::start();
    let sharded =
        live_dedup::build_cells_sharded_live_dedup(effective_points_ref, &grid, termination);

    #[cfg_attr(not(feature = "timing"), allow(clippy::clone_on_copy))]
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());

    let t = Timer::start();
    let assembled = live_dedup::assemble_sharded_live_dedup(sharded);
    tb.set_dedup(t.elapsed(), assembled.dedup_sub);

    let repair_edges_storage: Vec<live_dedup::EdgeRecord> = assembled
        .bad_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();

    let t = Timer::start();
    let (mut eff_cells, mut eff_cell_indices) = (assembled.cells, assembled.cell_indices);
    if let Some((cells, indices)) = edge_repair::repair_bad_edges(
        &repair_edges_storage,
        &assembled.vertices,
        &eff_cells,
        &eff_cell_indices,
        &assembled.vertex_keys,
    ) {
        eff_cells = cells;
        eff_cell_indices = indices;
    }
    tb.set_edge_repair(t.elapsed());

    // Remap cells back to original point indices if we merged
    let t = Timer::start();
    let (cells, cell_indices) = if needs_remap {
        use crate::VoronoiCell;
        let merge_result = merge_result.as_ref().unwrap();

        // Each original point maps to an effective point's cell
        let mut new_cells = Vec::with_capacity(points.len());
        let mut new_cell_indices: Vec<u32> = Vec::new();

        for orig_idx in 0..points.len() {
            let eff_idx = merge_result.original_to_effective[orig_idx];
            let eff_cell = &eff_cells[eff_idx];

            let start = u32::try_from(new_cell_indices.len())
                .expect("cell index buffer exceeds u32 capacity");
            let eff_start = eff_cell.vertex_start();
            let eff_end = eff_start + eff_cell.vertex_count();
            new_cell_indices.extend_from_slice(&eff_cell_indices[eff_start..eff_end]);

            let count_u16 =
                u16::try_from(eff_cell.vertex_count()).expect("cell vertex count exceeds u16");
            new_cells.push(VoronoiCell::new(start, count_u16));
        }
        (new_cells, new_cell_indices)
    } else {
        (eff_cells, eff_cell_indices)
    };

    let voronoi =
        crate::SphericalVoronoi::from_raw_parts(points, assembled.vertices, cells, cell_indices);
    tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = tb.finish();
    timings.report(voronoi.num_cells());

    voronoi
}

pub fn compute_voronoi_knn_clipping_with_config_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> crate::SphericalVoronoi {
    let termination = TerminationConfig {
        max_k_cap: config.termination_max_k,
        ..Default::default()
    };
    compute_voronoi_knn_clipping_owned_core(
        points,
        termination,
        config.preprocess_threshold,
        !config.preprocess,
    )
}
