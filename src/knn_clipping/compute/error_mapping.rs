use glam::Vec3;

use crate::knn_clipping::preprocess::MergeResult;
use crate::live_dedup::{BuildCellsError, CellBuildError, CellFailure};

pub(super) fn map_cell_build_error(
    err: CellBuildError,
    effective_points: &[Vec3],
    merge_result: Option<&MergeResult>,
) -> crate::VoronoiError {
    let detail_suffix = err
        .detail
        .as_deref()
        .map(|detail| format!(" ({detail})"))
        .unwrap_or_default();

    match err.failure {
        CellFailure::ProjectionInvalid => crate::VoronoiError::UnsupportedGeometry {
            generator_index: err.generator_idx,
            message: format!(
                "cell extends to the generator hemisphere boundary; gnomonic projection is invalid{}",
                detail_suffix
            ),
        },
        CellFailure::UnboundedAfterExhaustion => crate::VoronoiError::ComputationFailed(format!(
            "cell {} exhausted the neighbor stream before reaching a bounded polygon{}",
            err.generator_idx, detail_suffix
        )),
        CellFailure::TooManyVertices => crate::VoronoiError::ComputationFailed(format!(
            "cell {} exceeded the clipping vertex budget{}",
            err.generator_idx, detail_suffix
        )),
        CellFailure::ClippedAway => {
            if let Some(degenerate) =
                classify_coincident_clipped_away(&err, effective_points, merge_result)
            {
                return degenerate;
            }
            crate::VoronoiError::ComputationFailed(format!(
                "cell {} failed during construction with ClippedAway{}",
                err.generator_idx, detail_suffix
            ))
        }
        other => crate::VoronoiError::ComputationFailed(format!(
            "cell {} failed during construction with {:?}{}",
            err.generator_idx, other, detail_suffix
        )),
    }
}

/// Classify a `ClippedAway` failure caused by sub-weld-radius coincidence.
///
/// A cell can only be clipped to nothing when other generators sit within the
/// resolvability scale of its generator (welding is disabled or the requested
/// radius is below the weld radius). Such inputs get an actionable
/// `DegenerateInput` naming the coincident generators instead of a generic
/// computation failure. Emitting a degenerate cell instead is not an option:
/// the neighbors were already clipped against this generator's bisectors, so
/// their boundaries would carry edges pairing against a missing cell.
fn classify_coincident_clipped_away(
    err: &CellBuildError,
    effective_points: &[Vec3],
    merge_result: Option<&MergeResult>,
) -> Option<crate::VoronoiError> {
    let generator = *effective_points.get(err.generator_idx)?;
    let radius_sq = crate::tolerances::weld_radius() * crate::tolerances::weld_radius();
    let coincident: Vec<usize> = effective_points
        .iter()
        .enumerate()
        .filter(|&(i, p)| i != err.generator_idx && (*p - generator).length_squared() < radius_sq)
        .map(|(i, _)| original_index_for_effective(i, merge_result))
        .collect();
    if coincident.is_empty() {
        return None;
    }

    let generator_original = original_index_for_effective(err.generator_idx, merge_result);
    Some(crate::VoronoiError::DegenerateInput {
        coincident_pairs: coincident.len(),
        message: format!(
            "generator {} is within the weld radius ({:.1e}) of generator(s) {:?} and its cell \
             is below representable scale; enable welding (PreprocessMode::Weld, the default) \
             or merge these points",
            generator_original,
            crate::tolerances::weld_radius(),
            coincident
        ),
    })
}

/// First original input index mapping to an effective index (identity when no
/// welds occurred). O(n) scan; only used on terminal error paths.
fn original_index_for_effective(effective_idx: usize, merge_result: Option<&MergeResult>) -> usize {
    match merge_result {
        Some(mr) => mr
            .original_to_effective
            .iter()
            .position(|&e| e as usize == effective_idx)
            .unwrap_or(effective_idx),
        None => effective_idx,
    }
}

pub(super) fn map_build_cells_error(
    err: BuildCellsError,
    effective_points: &[Vec3],
    merge_result: Option<&MergeResult>,
) -> crate::VoronoiError {
    match err {
        BuildCellsError::CellBuild(err) => {
            map_cell_build_error(err, effective_points, merge_result)
        }
        BuildCellsError::PackedLayoutCapacity(err) => {
            crate::VoronoiError::RepresentationLimit(format!(
                "packed bin/local layout capacity exceeded in bin {}: population {} exceeds local mask {} (num_bins={}, local_shift={})",
                err.bin, err.local_population, err.local_mask, err.num_bins, err.local_shift
            ))
        }
        BuildCellsError::RepresentationLimit(message) => {
            crate::VoronoiError::RepresentationLimit(message)
        }
    }
}
