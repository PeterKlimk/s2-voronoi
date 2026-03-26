//! Compute entry points for the kNN + clipping Voronoi backend.

use glam::Vec3;

use super::edge_reconcile;
use super::live_dedup;
use super::timing::{Timer, TimingBuilder};
use super::{
    cell_build::{CellBuildError, CellFailure},
    constants, merge_close_points, MergeResult, TerminationConfig, KNN_GRID_TARGET_DENSITY,
};
use crate::cube_grid::CubeMapGrid;
#[cfg(feature = "timing")]
use crate::cube_grid::CubeMapGridBuildTimings;
use crate::diagram::VoronoiCell;
use crate::{ComputeOutput, ComputeReport, PreprocessMode, PreprocessReport, VoronoiConfig};

pub(super) fn compute_voronoi_knn_clipping_owned_core(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    let mut tb = TimingBuilder::new();

    let t = Timer::start();
    let (effective_points, merge_result, _preprocess_report) =
        preprocess_effective_points(&points, preprocess_mode);
    tb.set_preprocess(t.elapsed());

    let effective_points_ref: &[Vec3] = match &effective_points {
        Some(v) => v.as_slice(),
        None => points.as_slice(),
    };

    let grid = build_query_grid(effective_points_ref, &mut tb);
    let sharded = construct_cell_shards(effective_points_ref, &grid, termination, &mut tb)?;
    let assembled = assemble_shards(sharded, &mut tb);
    let live_dedup::AssemblyResult {
        vertices,
        vertex_keys,
        unresolved_edges,
        cells,
        cell_indices,
        dedup_sub: _,
    } = assembled;
    let (eff_cells, eff_cell_indices) = reconcile_edges(
        &vertices,
        &vertex_keys,
        &unresolved_edges,
        cells,
        cell_indices,
        &mut tb,
    );

    let t = Timer::start();
    let (cells, cell_indices) = remap_cells_to_original_indices(
        &points,
        merge_result.as_ref(),
        eff_cells,
        eff_cell_indices,
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(points, vertices, cells, cell_indices);
    tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = tb.finish();
    timings.report(diagram.num_cells());

    Ok(diagram)
}

pub fn compute_voronoi_knn_clipping_with_config_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    let termination = TerminationConfig {
        packed_expand_r2: config.packed_knn_expand_r2,
        max_k_cap: config.termination_max_k,
        ..Default::default()
    };
    compute_voronoi_knn_clipping_owned_core(points, termination, config.preprocess_mode)
}

pub fn compute_voronoi_knn_clipping_with_report_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<ComputeOutput, crate::VoronoiError> {
    let termination = TerminationConfig {
        packed_expand_r2: config.packed_knn_expand_r2,
        max_k_cap: config.termination_max_k,
        ..Default::default()
    };
    compute_voronoi_knn_clipping_report_core(points, termination, config.preprocess_mode)
}

fn compute_voronoi_knn_clipping_report_core(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
) -> Result<ComputeOutput, crate::VoronoiError> {
    let mut tb = TimingBuilder::new();

    let t = Timer::start();
    let (effective_points, merge_result, preprocess_report) =
        preprocess_effective_points(&points, preprocess_mode);
    tb.set_preprocess(t.elapsed());

    let effective_points_ref: &[Vec3] = match &effective_points {
        Some(v) => v.as_slice(),
        None => points.as_slice(),
    };

    let grid = build_query_grid(effective_points_ref, &mut tb);
    let sharded = construct_cell_shards(effective_points_ref, &grid, termination, &mut tb)?;
    let assembled = assemble_shards(sharded, &mut tb);
    let live_dedup::AssemblyResult {
        vertices,
        vertex_keys,
        unresolved_edges,
        cells,
        cell_indices,
        dedup_sub: _,
    } = assembled;
    let (eff_cells, eff_cell_indices) = reconcile_edges(
        &vertices,
        &vertex_keys,
        &unresolved_edges,
        cells,
        cell_indices,
        &mut tb,
    );

    let effective_diagram = if merge_result.is_some() {
        Some(crate::SphericalVoronoi::from_raw_parts(
            effective_points_ref.to_vec(),
            vertices.clone(),
            eff_cells.clone(),
            eff_cell_indices.clone(),
        ))
    } else {
        None
    };
    let effective_validation = effective_diagram.as_ref().map(crate::validation::validate);

    let t = Timer::start();
    let (cells, cell_indices) = remap_cells_to_original_indices(
        &points,
        merge_result.as_ref(),
        eff_cells,
        eff_cell_indices,
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(points, vertices, cells, cell_indices);
    let returned_validation = crate::validation::validate(&diagram);
    tb.set_assemble(t.elapsed());

    let timings = tb.finish();
    timings.report(diagram.num_cells());

    Ok(ComputeOutput {
        diagram,
        effective_diagram,
        report: ComputeReport {
            preprocess: preprocess_report,
            returned_validation,
            effective_validation,
        },
    })
}

fn map_cell_build_error(err: CellBuildError) -> crate::VoronoiError {
    match err.failure {
        CellFailure::ProjectionInvalid => crate::VoronoiError::UnsupportedGeometry {
            generator_index: err.generator_idx,
            message:
                "cell extends to the generator hemisphere boundary; gnomonic projection is invalid"
                    .to_string(),
        },
        CellFailure::UnboundedAfterExhaustion => crate::VoronoiError::ComputationFailed(format!(
            "cell {} exhausted the neighbor stream before reaching a bounded polygon",
            err.generator_idx
        )),
        CellFailure::TooManyVertices => crate::VoronoiError::ComputationFailed(format!(
            "cell {} exceeded the clipping vertex budget",
            err.generator_idx
        )),
        other => crate::VoronoiError::ComputationFailed(format!(
            "cell {} failed during construction with {:?}",
            err.generator_idx, other
        )),
    }
}

fn preprocess_effective_points(
    points: &[Vec3],
    preprocess_mode: PreprocessMode,
) -> (Option<Vec<Vec3>>, Option<MergeResult>, PreprocessReport) {
    let threshold = match preprocess_mode {
        PreprocessMode::Disabled => {
            return (
                None,
                None,
                PreprocessReport {
                    requested_mode: preprocess_mode,
                    threshold_used: None,
                    original_points: points.len(),
                    effective_points: points.len(),
                    num_merged: 0,
                },
            );
        }
        PreprocessMode::MergeDensity => constants::merge_threshold_for_density(points.len()),
        PreprocessMode::MergeWithin(threshold) => threshold,
    };
    let mut result = merge_close_points(points, threshold);
    let report = PreprocessReport {
        requested_mode: preprocess_mode,
        threshold_used: Some(threshold),
        original_points: points.len(),
        effective_points: result.effective_points.len(),
        num_merged: result.num_merged,
    };
    if result.num_merged > 0 {
        let pts = std::mem::take(&mut result.effective_points);
        (Some(pts), Some(result), report)
    } else {
        (None, None, report)
    }
}

fn build_query_grid(
    effective_points: &[Vec3],
    tb: &mut TimingBuilder,
) -> crate::cube_grid::CubeMapGrid {
    let t = Timer::start();
    let n = effective_points.len();
    let target = KNN_GRID_TARGET_DENSITY.max(1.0);
    let res = ((n as f64 / (6.0 * target)).sqrt() as usize).max(4);
    #[cfg(feature = "timing")]
    let mut grid_build_timings = CubeMapGridBuildTimings::default();
    #[cfg(feature = "timing")]
    let grid = CubeMapGrid::new_with_build_timings(effective_points, res, &mut grid_build_timings);
    #[cfg(not(feature = "timing"))]
    let grid = CubeMapGrid::new(effective_points, res);
    tb.set_knn_build(t.elapsed());
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(grid_build_timings.clone());
    grid
}

fn construct_cell_shards(
    effective_points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
    tb: &mut TimingBuilder,
) -> Result<live_dedup::ShardedCellsData, crate::VoronoiError> {
    let t = Timer::start();
    let sharded = live_dedup::build_cells_sharded_live_dedup(effective_points, grid, termination)
        .map_err(map_cell_build_error)?;
    #[cfg_attr(not(feature = "timing"), allow(clippy::clone_on_copy))]
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());
    Ok(sharded)
}

fn assemble_shards(
    sharded: live_dedup::ShardedCellsData,
    tb: &mut TimingBuilder,
) -> live_dedup::AssemblyResult {
    let t = Timer::start();
    let assembled = live_dedup::assemble_sharded_live_dedup(sharded);
    tb.set_dedup(t.elapsed(), assembled.dedup_sub);
    assembled
}

fn reconcile_edges(
    vertices: &[Vec3],
    vertex_keys: &[crate::knn_clipping::cell_build::VertexKey],
    unresolved_edges: &[live_dedup::UnresolvedEdgeMismatch],
    mut cells: Vec<VoronoiCell>,
    mut cell_indices: Vec<u32>,
    tb: &mut TimingBuilder,
) -> (Vec<VoronoiCell>, Vec<u32>) {
    let repair_edges_storage: Vec<live_dedup::EdgeRecord> = unresolved_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();

    let t = Timer::start();
    if let Some((reconciled_cells, reconciled_indices)) = edge_reconcile::reconcile_unresolved_edges(
        &repair_edges_storage,
        vertices,
        &cells,
        &cell_indices,
        vertex_keys,
    ) {
        cells = reconciled_cells;
        cell_indices = reconciled_indices;
    }
    tb.set_edge_reconcile(t.elapsed());
    (cells, cell_indices)
}

fn remap_cells_to_original_indices(
    points: &[Vec3],
    merge_result: Option<&MergeResult>,
    eff_cells: Vec<VoronoiCell>,
    eff_cell_indices: Vec<u32>,
) -> (Vec<VoronoiCell>, Vec<u32>) {
    if let Some(merge_result) = merge_result {
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
    }
}

#[cfg(test)]
mod tests {
    use super::map_cell_build_error;
    use crate::knn_clipping::cell_build::{CellBuildError, CellFailure};
    use crate::VoronoiError;

    #[test]
    fn map_projection_invalid_to_unsupported_geometry() {
        let err = map_cell_build_error(CellBuildError {
            generator_idx: 7,
            failure: CellFailure::ProjectionInvalid,
        });
        assert!(matches!(
            err,
            VoronoiError::UnsupportedGeometry {
                generator_index: 7,
                ..
            }
        ));
    }

    #[test]
    fn map_unbounded_after_exhaustion_to_computation_failed() {
        let err = map_cell_build_error(CellBuildError {
            generator_idx: 11,
            failure: CellFailure::UnboundedAfterExhaustion,
        });
        match err {
            VoronoiError::ComputationFailed(msg) => {
                assert!(msg.contains("11"));
                assert!(msg.contains("bounded polygon"));
            }
            other => panic!("expected ComputationFailed, got {:?}", other),
        }
    }

    #[test]
    fn map_too_many_vertices_to_computation_failed() {
        let err = map_cell_build_error(CellBuildError {
            generator_idx: 13,
            failure: CellFailure::TooManyVertices,
        });
        match err {
            VoronoiError::ComputationFailed(msg) => {
                assert!(msg.contains("13"));
                assert!(msg.contains("vertex budget"));
            }
            other => panic!("expected ComputationFailed, got {:?}", other),
        }
    }
}
