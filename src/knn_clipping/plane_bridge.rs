//! Bridge from the planar pipeline to the shared live-dedup engine.
//!
//! Temporary seam: `live_dedup` (and the edge reconcile post-pass) live
//! inside `knn_clipping` for historical reasons but are geometry-agnostic;
//! this module exposes the one entry the planar compute path needs. It goes
//! away when `live_dedup` is promoted to a crate-level module.

use glam::{Vec2, Vec3};

use super::{edge_reconcile, live_dedup};
use crate::diagram::VoronoiCell;
use crate::plane_grid::PlaneGrid;

pub(crate) struct PlaneCellsOutput {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) cells: Vec<VoronoiCell>,
    pub(crate) cell_indices: Vec<u32>,
}

/// Build, dedup, assemble, and edge-reconcile all planar cells.
///
/// `points` are normalized coordinates inside `[0, domain.x] x [0, domain.y]`
/// (the caller maps the user rect); positions come back as `Vec3 {x, y, 0}`.
pub(crate) fn compute_plane_cells(
    points: &[Vec2],
    grid: &PlaneGrid,
    domain: Vec2,
) -> Result<PlaneCellsOutput, crate::VoronoiError> {
    let sharded = live_dedup::build_cells_sharded_plane(points, grid, domain)
        .map_err(map_plane_build_error)?;
    let assembly = live_dedup::assemble_sharded_live_dedup(sharded)?;

    let records: Vec<live_dedup::EdgeRecord> = assembly
        .unresolved_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();
    let mut cells = assembly.cells;
    let mut cell_indices = assembly.cell_indices;
    if let Some((reconciled_cells, reconciled_indices)) =
        edge_reconcile::reconcile_unresolved_edges(
            &records,
            &assembly.vertices,
            &cells,
            &cell_indices,
            &assembly.vertex_keys,
        )?
    {
        cells = reconciled_cells;
        cell_indices = reconciled_indices;
    }

    Ok(PlaneCellsOutput {
        vertices: assembly.vertices,
        cells,
        cell_indices,
    })
}

fn map_plane_build_error(err: live_dedup::BuildCellsError) -> crate::VoronoiError {
    match err {
        live_dedup::BuildCellsError::CellBuild(err) => match err.failure {
            // With exact duplicates welded upstream, a clipped-away planar
            // cell means epsilon-coincident generators.
            crate::knn_clipping::cell_build::CellFailure::ClippedAway => {
                crate::VoronoiError::DegenerateInput {
                    coincident_pairs: 1,
                    message: format!(
                        "generator {} was fully clipped away (near-coincident generators)",
                        err.generator_idx
                    ),
                }
            }
            failure => crate::VoronoiError::ComputationFailed(format!(
                "planar cell construction failed for generator {}: {:?}",
                err.generator_idx, failure
            )),
        },
        live_dedup::BuildCellsError::PackedLayoutCapacity(err) => {
            crate::VoronoiError::RepresentationLimit(format!(
                "packed bin/local layout capacity exceeded in bin {}: population {} exceeds local mask {} (num_bins={}, local_shift={})",
                err.bin, err.local_population, err.local_mask, err.num_bins, err.local_shift
            ))
        }
        live_dedup::BuildCellsError::RepresentationLimit(message) => {
            crate::VoronoiError::RepresentationLimit(message)
        }
    }
}
