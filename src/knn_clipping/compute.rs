//! Compute entry points for the kNN + clipping Voronoi backend.

use glam::{DVec3, Vec3};

use super::edge_reconcile;
use super::escalate;
use super::live_dedup;
use super::timing::{Timer, TimingBuilder};
use super::{
    cell_build::{CellBuildError, CellFailure},
    merge_close_points, MergeResult, TerminationConfig,
};
use crate::cube_grid::CubeMapGrid;
#[cfg(feature = "timing")]
use crate::cube_grid::CubeMapGridBuildTimings;
use crate::diagram::VoronoiCell;
use crate::{
    ComputeOutput, ComputeReport, DegenerateMode, DegenerateReport, PreprocessMode,
    PreprocessReport, RepairMode, VoronoiConfig,
};

/// Per-seed neighbor count for the escalation local set (brute-force gather
/// scaffold; production will pull from the grid / considered-neighbor set).
const ESCALATE_GATHER_K: usize = 96;
/// Grow-until-clean round cap per defect component.
const ESCALATE_MAX_ROUNDS: usize = 12;

pub(super) fn compute_voronoi_knn_clipping_owned_core(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
    repair_mode: RepairMode,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    validate_generator_capacity(points.len())?;
    validate_generator_finiteness(&points)?;
    let mut points = points;
    canonicalize_unit_points(&mut points);
    let mut tb = TimingBuilder::new();

    let (effective_points, merge_result, _preprocess_report, grid) =
        prepare_points_and_grid(&points, preprocess_mode, &mut tb);

    let effective_points_ref: &[Vec3] = match &effective_points {
        Some(v) => v.as_slice(),
        None => points.as_slice(),
    };

    let sharded = construct_cell_shards(
        effective_points_ref,
        &grid,
        termination,
        merge_result.as_ref(),
        &mut tb,
    )?;
    let assembled = assemble_shards(sharded, &mut tb)?;
    let live_dedup::AssemblyResult {
        mut vertices,
        vertex_keys,
        unresolved_edges,
        cells,
        cell_indices,
        dedup_sub: _,
    } = assembled;
    let (mut eff_cells, mut eff_cell_indices, post_repair_unpaired) = reconcile_edges(
        &mut vertices,
        &vertex_keys,
        &unresolved_edges,
        cells,
        cell_indices,
        &mut tb,
    )?;
    let repair_accepted = maybe_repair_effective(
        effective_points_ref,
        &grid,
        &mut vertices,
        &vertex_keys,
        &mut eff_cells,
        &mut eff_cell_indices,
        &post_repair_unpaired,
        repair_mode,
    );
    // Plain path: if local repair did not accept a strictly-valid replacement,
    // a residual is provably-invalid output and there is no report channel to
    // surface it, so fail loud rather than ship it.
    if !repair_accepted && !post_repair_unpaired.is_empty() {
        return Err(edge_reconcile::residual_error(&post_repair_unpaired));
    }

    let t = Timer::start();
    let (cells, cell_indices, weld_map) = remap_cells_to_original_indices(
        &points,
        merge_result.as_ref(),
        eff_cells,
        eff_cell_indices,
    );

    let diagram =
        crate::SphericalVoronoi::from_raw_parts(points, vertices, cells, cell_indices, weld_map);
    tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = tb.finish();
    timings.report(diagram.num_cells());

    crate::validation::verify_sphere_if_enabled(&diagram)?;
    Ok(diagram)
}

pub fn compute_voronoi_knn_clipping_with_config_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    let termination = TerminationConfig::default();
    if !matches!(config.degenerate_mode, DegenerateMode::PerturbGreatCircle) {
        return compute_voronoi_knn_clipping_owned_core(
            points,
            termination,
            config.preprocess_mode,
            config.repair_mode,
        );
    }

    match compute_voronoi_knn_clipping_owned_core(
        points.clone(),
        termination,
        config.preprocess_mode,
        config.repair_mode,
    ) {
        Ok(diagram) => Ok(diagram),
        Err(err) => match maybe_perturb_rank2_great_circle(&points, &err) {
            Some(perturbed) => compute_voronoi_knn_clipping_owned_core(
                perturbed,
                termination,
                config.preprocess_mode,
                config.repair_mode,
            ),
            None => Err(err),
        },
    }
}

pub fn compute_voronoi_knn_clipping_with_report_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<ComputeOutput, crate::VoronoiError> {
    let termination = TerminationConfig::default();
    if !matches!(config.degenerate_mode, DegenerateMode::PerturbGreatCircle) {
        return compute_voronoi_knn_clipping_report_core(
            points,
            termination,
            config.preprocess_mode,
            config.repair_mode,
            DegenerateReport {
                requested_mode: config.degenerate_mode,
                perturbation_applied: false,
            },
        );
    }

    match compute_voronoi_knn_clipping_report_core(
        points.clone(),
        termination,
        config.preprocess_mode,
        config.repair_mode,
        DegenerateReport {
            requested_mode: config.degenerate_mode,
            perturbation_applied: false,
        },
    ) {
        Ok(output) => Ok(output),
        Err(err) => match maybe_perturb_rank2_great_circle(&points, &err) {
            Some(perturbed) => compute_voronoi_knn_clipping_report_core(
                perturbed,
                termination,
                config.preprocess_mode,
                config.repair_mode,
                DegenerateReport {
                    requested_mode: config.degenerate_mode,
                    perturbation_applied: true,
                },
            ),
            None => Err(err),
        },
    }
}

fn compute_voronoi_knn_clipping_report_core(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
    repair_mode: RepairMode,
    degenerate_report: DegenerateReport,
) -> Result<ComputeOutput, crate::VoronoiError> {
    validate_generator_capacity(points.len())?;
    validate_generator_finiteness(&points)?;
    let mut points = points;
    canonicalize_unit_points(&mut points);
    let mut tb = TimingBuilder::new();

    let (effective_points, merge_result, preprocess_report, grid) =
        prepare_points_and_grid(&points, preprocess_mode, &mut tb);

    let effective_points_ref: &[Vec3] = match &effective_points {
        Some(v) => v.as_slice(),
        None => points.as_slice(),
    };

    let sharded = construct_cell_shards(
        effective_points_ref,
        &grid,
        termination,
        merge_result.as_ref(),
        &mut tb,
    )?;
    let assembled = assemble_shards(sharded, &mut tb)?;
    let live_dedup::AssemblyResult {
        mut vertices,
        vertex_keys,
        unresolved_edges,
        cells,
        cell_indices,
        dedup_sub: _,
    } = assembled;
    let (mut eff_cells, mut eff_cell_indices, post_repair_unpaired) = reconcile_edges(
        &mut vertices,
        &vertex_keys,
        &unresolved_edges,
        cells,
        cell_indices,
        &mut tb,
    )?;
    let repair_accepted = maybe_repair_effective(
        effective_points_ref,
        &grid,
        &mut vertices,
        &vertex_keys,
        &mut eff_cells,
        &mut eff_cell_indices,
        &post_repair_unpaired,
        repair_mode,
    );
    // Surface output-invariant residuals alongside the detection records. If
    // local repair was accepted, the returned diagram is strictly valid and
    // these residuals no longer survive to output.
    let mut unresolved_edges = unresolved_edges;
    let pre_repair_edge_mismatches: Vec<(u32, u32, live_dedup::UnresolvedEdgeOrigin)> =
        unresolved_edges
            .iter()
            .map(|m| {
                let (a, b) = edge_reconcile::unpack_edge(m.key.as_u64());
                (a.min(b), a.max(b), m.origin)
            })
            .collect();
    let post_repair_unpaired_edges: Vec<(u32, u32)> = if repair_accepted {
        Vec::new()
    } else {
        post_repair_unpaired
            .iter()
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect()
    };
    if !repair_accepted {
        for &(a, b) in &post_repair_unpaired {
            unresolved_edges.push(live_dedup::UnresolvedEdgeMismatch {
                key: live_dedup::pack_edge(a, b),
                origin: live_dedup::UnresolvedEdgeOrigin::PostRepairUnpaired,
            });
        }
    }

    let effective_diagram = if merge_result.is_some() {
        Some(crate::SphericalVoronoi::from_raw_parts(
            effective_points_ref.to_vec(),
            vertices.clone(),
            eff_cells.clone(),
            eff_cell_indices.clone(),
            None,
        ))
    } else {
        None
    };
    let effective_validation = effective_diagram.as_ref().map(crate::validation::validate);

    let t = Timer::start();
    let (cells, cell_indices, weld_map) = remap_cells_to_original_indices(
        &points,
        merge_result.as_ref(),
        eff_cells,
        eff_cell_indices,
    );

    let diagram =
        crate::SphericalVoronoi::from_raw_parts(points, vertices, cells, cell_indices, weld_map);
    let returned_validation = crate::validation::validate(&diagram);
    tb.set_assemble(t.elapsed());

    let timings = tb.finish();
    timings.report(diagram.num_cells());

    Ok(ComputeOutput {
        diagram,
        effective_diagram,
        report: ComputeReport {
            preprocess: preprocess_report,
            degenerate: degenerate_report,
            returned_validation,
            effective_validation,
            pre_repair_edge_mismatches,
            post_repair_unpaired_edges,
            unresolved_edge_pairs: unresolved_edges
                .iter()
                .map(|m| {
                    let (a, b) = edge_reconcile::unpack_edge(m.key.as_u64());
                    (a.min(b), a.max(b), m.origin)
                })
                .collect(),
        },
    })
}

/// True if any vertex referenced by a live cell has degree 1 or 2 (a real
/// sub-3-incidence defect the repair should examine).
///
/// Counts incidence over each cell's *live* window `[vertex_start ..
/// vertex_start + vertex_count)`, NOT the raw `cell_indices` buffer. Edge
/// reconciliation shrinks a cell's `vertex_count` in place without compacting
/// the backing buffer (see `apply_merges_in_place` /
/// `drop_degenerate_collinear_vertices` in `edge_reconcile`), so the buffer can
/// retain stale tail slots that no live cell references. Scanning the whole
/// buffer counts those stale slots as phantom degree-1/2 vertices and trips a
/// no-op repair (a single stale slot cost ~13s of acceptance-gate work at
/// 2.5M). Counting live windows matches the validators (`validate_impl`,
/// `verify_sphere_fast`) and the repair's own `low_incidence_gens`.
fn has_low_incidence_vertices(
    vertex_count: usize,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> bool {
    let mut cnt = vec![0u32; vertex_count];
    for cell in cells {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        for &v in &cell_indices[start..end] {
            cnt[v as usize] += 1;
        }
    }
    cnt.iter().any(|&c| c == 1 || c == 2)
}

/// Try the configured local repair and commit it only if whole-diagram strict
/// validation succeeds. Returns true when the repaired effective diagram was
/// accepted.
#[allow(clippy::too_many_arguments)] // cohesive repair-entry state; splitting would obscure it
fn maybe_repair_effective(
    effective_points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &mut Vec<Vec3>,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    eff_cells: &mut Vec<VoronoiCell>,
    eff_cell_indices: &mut Vec<u32>,
    post_repair_unpaired: &[(u32, u32)],
    repair_mode: RepairMode,
) -> bool {
    // A0 probes need the fast assembled state, not the repaired one.
    if std::env::var("S2_ESCALATE_PROBE_A0").is_ok() {
        escalate::stash_a0_fast(effective_points, vertex_keys, eff_cells, eff_cell_indices);
        return false;
    }

    let repair_enabled =
        !matches!(repair_mode, RepairMode::Disabled) || escalate::escalation_enabled();
    if !repair_enabled {
        return false;
    }

    let defect_pairs: Vec<(u32, u32)> = post_repair_unpaired
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    let has_low_incidence = has_low_incidence_vertices(vertices.len(), eff_cells, eff_cell_indices);
    if defect_pairs.is_empty() && !has_low_incidence {
        return false;
    }

    // Local-neighbor gather index for the repair: O(local) shell-frontier kNN per
    // seed instead of the old O(n) brute force (a closure of thousands on a
    // dense-defect input made the brute force minutes-long). Reuse the construction
    // `grid` (occupancy-tuned; `compact_welded` keeps it bit-equivalent to a fresh
    // effective-point build) rather than rebuilding. Feed the gather an all-emit
    // (all-zero) eligibility layout: repair wants every nearby generator, not the
    // directed construction subset, and an all-zero map decodes to bin 0 (!= the
    // query bin) at any n, so it stays all-emit even past 2^24 slots.
    let mut repair_scratch = grid.make_scratch();
    let repair_slot_map: Vec<u32> = vec![0u32; effective_points.len()];

    let mut work = escalate::WorkingDiagram::from_assembled(
        vertices,
        vertex_keys,
        eff_cells,
        eff_cell_indices,
    );
    #[cfg(feature = "escalate_probe")]
    let stats = if std::env::var("S2_ESCALATE_DELAUNATOR").is_ok() {
        escalate::repair_delaunator(
            effective_points,
            &mut work,
            &defect_pairs,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    } else if matches!(repair_mode, RepairMode::LocalProjected) {
        escalate::repair_local_exact(
            effective_points,
            grid,
            &mut repair_scratch,
            &repair_slot_map,
            &mut work,
            &defect_pairs,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    } else {
        escalate::repair_local_hull(
            effective_points,
            grid,
            &mut repair_scratch,
            &repair_slot_map,
            &mut work,
            &defect_pairs,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    };
    #[cfg(not(feature = "escalate_probe"))]
    let stats = if matches!(repair_mode, RepairMode::LocalProjected) {
        escalate::repair_local_exact(
            effective_points,
            grid,
            &mut repair_scratch,
            &repair_slot_map,
            &mut work,
            &defect_pairs,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    } else {
        escalate::repair_local_hull(
            effective_points,
            grid,
            &mut repair_scratch,
            &repair_slot_map,
            &mut work,
            &defect_pairs,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    };
    // No splices means the repair did not modify `work` (a `splice_generator`
    // call is the only mutation, tracked 1:1 by `spliced_generators`). Skip the
    // flatten + full-diagram clone + validate of an unchanged diagram, which is
    // the dominant cost of a no-op repair (~12.6s of a 15s tail at 2.5M).
    if stats.spliced_generators == 0 {
        return false;
    }

    let (new_vertices, new_cells, new_cell_indices) = work.into_flat();

    // Whole-diagram never-worse gate: accept only if the repaired diagram is
    // strictly valid. Validate the effective arrays in place via
    // `verify_sphere_effective_strict` (same strict contract as `validate`,
    // pinned by the `effective_strict_matches_fast` differential test) rather
    // than cloning all of `effective_points`/vertices/cells/indices into a
    // temporary `SphericalVoronoi` — the clone was the dominant cost of a
    // committed repair. The validate itself stays whole-diagram (the repair's
    // blast radius is vertex-triple-identity-wide, so a local gate is unsound).
    if crate::validation::verify_sphere_effective_strict(
        &new_vertices,
        &new_cells,
        &new_cell_indices,
    )
    .is_err()
    {
        return false;
    }

    *vertices = new_vertices;
    *eff_cells = new_cells;
    *eff_cell_indices = new_cell_indices;
    true
}

fn map_cell_build_error(
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

fn map_build_cells_error(
    err: live_dedup::BuildCellsError,
    effective_points: &[Vec3],
    merge_result: Option<&MergeResult>,
) -> crate::VoronoiError {
    match err {
        live_dedup::BuildCellsError::CellBuild(err) => {
            map_cell_build_error(err, effective_points, merge_result)
        }
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

fn validate_generator_capacity(num_points: usize) -> Result<(), crate::VoronoiError> {
    if u32::try_from(num_points).is_ok() {
        return Ok(());
    }
    Err(crate::VoronoiError::RepresentationLimit(format!(
        "generator count {} exceeds u32-backed index capacity",
        num_points
    )))
}

/// Reject inputs containing non-finite components with an index-bearing error.
fn validate_generator_finiteness(points: &[Vec3]) -> Result<(), crate::VoronoiError> {
    #[cfg(feature = "parallel")]
    let first_bad = {
        use rayon::prelude::*;
        points.par_iter().position_first(|p| !p.is_finite())
    };
    #[cfg(not(feature = "parallel"))]
    let first_bad = points.iter().position(|p| !p.is_finite());

    match first_bad {
        None => Ok(()),
        Some(point_index) => Err(crate::VoronoiError::InvalidInput {
            point_index,
            message: format!(
                "point has a non-finite component: ({}, {}, {})",
                points[point_index].x, points[point_index].y, points[point_index].z
            ),
        }),
    }
}

/// Preprocess (weld) and build the query grid in one step.
///
/// The grid is built on the raw points and doubles as the weld detector
/// (`CubeMapGrid::collect_weld_pairs`); on welds the grid is compacted in
/// place to the effective points instead of being rebuilt, so the zero-weld
/// common case pays only the detection scan and the weld case pays linear
/// sweeps. The standalone quantized-key detector remains only for
/// `MergeWithin` radii too large for grid adjacency. The resolution policy
/// sees the raw count; welds are far too few to shift it.
/// P5 stage 0: canonicalize input points once at entry — f64-normalize and
/// round back to f32 — so every consumer (grid, weld, charts, certificates,
/// and the P5 canonical predicates) sees identical bits per generator. The
/// per-builder f64 renormalization is gone; without this pass the pipeline
/// solved a ~1-ulp-perturbed, asymmetrically-treated point set (stage-1
/// shadow findings, docs/p5-consistency-design.md). Out-of-band lengths
/// (contract-violating inputs) are left untouched and fail downstream as
/// before, rather than being turned into NaNs here.
fn canonicalize_unit_points(points: &mut [Vec3]) {
    fn canonicalize_chunk(chunk: &mut [Vec3]) {
        for p in chunk.iter_mut() {
            let v = glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64);
            let len_sq = v.length_squared();
            if (0.25..=4.0).contains(&len_sq) {
                let n = v / len_sq.sqrt();
                *p = Vec3::new(n.x as f32, n.y as f32, n.z as f32);
            }
        }
    }
    // ~10ns/point scalar (f64 sqrt + div); parallel chunks so the default
    // build pays ~nothing. Measured ST cost: ~20ms at 2M (the bulk of stage
    // 0's +0.5-0.8% single-threaded total).
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        points.par_chunks_mut(1 << 16).for_each(canonicalize_chunk);
    }
    #[cfg(not(feature = "parallel"))]
    canonicalize_chunk(points);
}

#[derive(Debug, Clone, Copy)]
struct Rank2GreatCircle {
    normal: DVec3,
}

fn maybe_perturb_rank2_great_circle(
    points: &[Vec3],
    err: &crate::VoronoiError,
) -> Option<Vec<Vec3>> {
    if !matches!(
        err,
        crate::VoronoiError::UnsupportedGeometry { .. } | crate::VoronoiError::ComputationFailed(_)
    ) {
        return None;
    }
    let mut canonical = points.to_vec();
    canonicalize_unit_points(&mut canonical);
    let class = classify_rank2_great_circle(&canonical)?;
    Some(perturb_great_circle_points(&canonical, class.normal))
}

fn classify_rank2_great_circle(points: &[Vec3]) -> Option<Rank2GreatCircle> {
    if points.len() < 4 {
        return None;
    }

    let mut best_cross = DVec3::ZERO;
    let mut best_len2 = 0.0f64;
    for i in 0..points.len() {
        let a = dvec(points[i]);
        for &b32 in &points[i + 1..] {
            let cross = a.cross(dvec(b32));
            let len2 = cross.length_squared();
            if len2 > best_len2 {
                best_len2 = len2;
                best_cross = cross;
            }
        }
    }
    if best_len2 < 0.25 {
        return None;
    }

    let normal = best_cross / best_len2.sqrt();
    let mut max_abs_dot = 0.0f64;
    let mut sum_dot2 = 0.0f64;
    for &p in points {
        let d = normal.dot(dvec(p)).abs();
        max_abs_dot = max_abs_dot.max(d);
        sum_dot2 += d * d;
    }
    let rms_dot = (sum_dot2 / points.len() as f64).sqrt();
    if max_abs_dot > 2.0e-6 || rms_dot > 5.0e-7 {
        return None;
    }

    if !covers_great_circle(points, normal) {
        return None;
    }

    Some(Rank2GreatCircle { normal })
}

fn covers_great_circle(points: &[Vec3], normal: DVec3) -> bool {
    let seed = if normal.x.abs() < 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let e1 = normal.cross(seed).normalize();
    let e2 = normal.cross(e1).normalize();
    let mut angles: Vec<f64> = points
        .iter()
        .map(|&p| {
            let p = dvec(p);
            p.dot(e2).atan2(p.dot(e1)).rem_euclid(std::f64::consts::TAU)
        })
        .collect();
    angles.sort_by(|a, b| a.total_cmp(b));

    let mut max_gap = 0.0f64;
    for w in angles.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    if let (Some(first), Some(last)) = (angles.first(), angles.last()) {
        max_gap = max_gap.max(first + std::f64::consts::TAU - last);
    }

    // A full great-circle set has no empty semicircle. Smaller arcs are better
    // treated as hemisphere/large-cell fallback cases, not SoS perturbation.
    max_gap < std::f64::consts::PI
}

fn perturb_great_circle_points(points: &[Vec3], normal: DVec3) -> Vec<Vec3> {
    // This is a realized robust-mode joggle, not a symbolic-only SoS epsilon.
    // The current f32 topology/validation path still sees near-antipodal pole
    // edges for microscopic offsets on exact great-circle fixtures; 1e-2 rad is
    // the already-tested small-jitter regime for these inputs.
    let scale = 1.0e-2f64;
    points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let amp = scale * stable_signed_unit(i as u64);
            let q = (dvec(p) + normal * amp).normalize();
            Vec3::new(q.x as f32, q.y as f32, q.z as f32)
        })
        .collect()
}

fn stable_signed_unit(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = ((x >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let signed = 2.0 * unit - 1.0;
    if signed.abs() < 0.125 {
        if signed < 0.0 {
            -0.125
        } else {
            0.125
        }
    } else {
        signed
    }
}

#[inline]
fn dvec(p: Vec3) -> DVec3 {
    DVec3::new(p.x as f64, p.y as f64, p.z as f64)
}

fn prepare_points_and_grid(
    points: &[Vec3],
    preprocess_mode: PreprocessMode,
    tb: &mut TimingBuilder,
) -> (
    Option<Vec<Vec3>>,
    Option<MergeResult>,
    PreprocessReport,
    CubeMapGrid,
) {
    let threshold = match preprocess_mode {
        PreprocessMode::Disabled => None,
        PreprocessMode::Weld => Some(crate::tolerances::weld_radius()),
        PreprocessMode::MergeWithin(threshold) => Some(threshold),
    };

    let mut grid = build_query_grid(points, tb);

    let t = Timer::start();
    let mut effective_points = None;
    let mut merge_result = None;
    if let Some(threshold) = threshold {
        if threshold <= grid.max_grid_weld_threshold() {
            let pairs = grid.collect_weld_pairs(threshold);
            if !pairs.is_empty() {
                let (mut result, kept) = super::preprocess::merge_result_from_pairs(points, &pairs);
                grid.compact_welded(
                    &kept,
                    &result.original_to_effective,
                    result.effective_points.len(),
                );
                let pts = std::mem::take(&mut result.effective_points);
                effective_points = Some(pts);
                merge_result = Some(result);
            }
        } else {
            // Radius too large for grid adjacency (large `MergeWithin`):
            // standalone detector, then rebuild the grid on the survivors.
            let mut result = merge_close_points(points, threshold);
            if result.num_merged > 0 {
                let pts = std::mem::take(&mut result.effective_points);
                grid = build_query_grid(&pts, tb);
                effective_points = Some(pts);
                merge_result = Some(result);
            }
        }
    }
    tb.set_preprocess(t.elapsed());

    let report = PreprocessReport {
        requested_mode: preprocess_mode,
        threshold_used: threshold,
        original_points: points.len(),
        effective_points: effective_points.as_ref().map_or(points.len(), |p| p.len()),
        num_merged: merge_result.as_ref().map_or(0, |m| m.num_merged),
    };
    (effective_points, merge_result, report, grid)
}

fn max_cell_occupancy(grid: &crate::cube_grid::CubeMapGrid) -> usize {
    grid.cell_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0)
}

/// `Σocc²/n`: the occupancy-rebuild trigger signal (see
/// `policy::GRID_REBUILD_SUMSQ_PER_N`). One cheap pass over the CSR offsets;
/// equals the target density for uniform input, rising with concentration.
fn cell_sum_sq_per_n(grid: &crate::cube_grid::CubeMapGrid, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = grid
        .cell_offsets()
        .windows(2)
        .map(|w| {
            let c = (w[1] - w[0]) as f64;
            c * c
        })
        .sum();
    sum_sq / n as f64
}

fn build_query_grid(
    effective_points: &[Vec3],
    tb: &mut TimingBuilder,
) -> crate::cube_grid::CubeMapGrid {
    let t = Timer::start();
    let n = effective_points.len();
    #[cfg(feature = "timing")]
    let mut grid_build_timings = CubeMapGridBuildTimings::default();

    let build = |res: usize, #[cfg(feature = "timing")] timings: &mut CubeMapGridBuildTimings| {
        #[cfg(feature = "timing")]
        {
            CubeMapGrid::new_with_build_timings(effective_points, res, timings)
        }
        #[cfg(not(feature = "timing"))]
        {
            CubeMapGrid::new(effective_points, res)
        }
    };

    let mut res = crate::policy::knn_grid_resolution(n);
    #[cfg(feature = "timing")]
    let grid = build(res, &mut grid_build_timings);
    #[cfg(not(feature = "timing"))]
    let grid = build(res);
    let mut max_occupancy = max_cell_occupancy(&grid);
    let sum_sq_per_n = cell_sum_sq_per_n(&grid, n);

    // Occupancy feedback: a catastrophically concentrated input (Σocc²/n over
    // the threshold) makes the per-cell candidate scan O(occ²)-infeasible; one
    // global re-grid at higher resolution (within the memory budget) restores
    // tractable per-cell work. Fires only in that regime — modest clusters
    // degrade gracefully and a re-grid would be a net pessimization there.
    let mut rebuilt = false;
    let grid =
        match crate::policy::grid_occupancy_rebuild_resolution(res, n, max_occupancy, sum_sq_per_n)
        {
            Some(new_res) => {
                res = new_res;
                rebuilt = true;
                #[cfg(feature = "timing")]
                let regrid = build(new_res, &mut grid_build_timings);
                #[cfg(not(feature = "timing"))]
                let regrid = build(new_res);
                max_occupancy = max_cell_occupancy(&regrid);
                regrid
            }
            None => grid,
        };

    // Gate the dense-cell band-prune on a rebuild having fired. The band only
    // wins on deep-certificate, un-splittable concentration (cap-like), which
    // is exactly the regime that triggers the occupancy rebuild and survives
    // it (a cell still over the dense threshold). Moderate clusters that never
    // trip the rebuild close fast in the packed path, where the band + takeover
    // is a measured net loss (clustered 500k ~ -13%); disable it there. Scale-
    // invariant, unlike a fixed occupancy threshold (clustered occ grows with
    // n). See docs/punch1-center-cell-integration.md.
    let mut grid = grid;
    if !rebuilt {
        grid.clear_dense_index();
    }

    tb.set_knn_build(t.elapsed());
    tb.set_grid_stats(res, max_occupancy as u64, rebuilt);
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(grid_build_timings.clone());
    grid
}

fn construct_cell_shards(
    effective_points: &[Vec3],
    grid: &CubeMapGrid,
    termination: TerminationConfig,
    merge_result: Option<&MergeResult>,
    tb: &mut TimingBuilder,
) -> Result<live_dedup::ShardedCellsData, crate::VoronoiError> {
    let t = Timer::start();
    let sharded =
        super::driver::build_cells_sharded_live_dedup(effective_points, grid, termination)
            .map_err(|err| map_build_cells_error(err, effective_points, merge_result))?;
    #[cfg_attr(not(feature = "timing"), allow(clippy::clone_on_copy))]
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());
    Ok(sharded)
}

fn assemble_shards(
    sharded: live_dedup::ShardedCellsData,
    tb: &mut TimingBuilder,
) -> Result<live_dedup::AssemblyResult, crate::VoronoiError> {
    let t = Timer::start();
    let assembled = live_dedup::assemble_sharded_live_dedup(sharded)?;
    // clone is required under the timing feature (real DedupSubPhases is
    // not Copy); the stub is Copy, hence the allow.
    #[allow(clippy::clone_on_copy)]
    tb.set_dedup(t.elapsed(), assembled.dedup_sub.clone());
    Ok(assembled)
}

/// Reconciled cell arrays plus the post-repair output-invariant residuals
/// (cell pairs whose shared edge stayed unpaired after both repair passes).
type ReconciledWithResiduals = (Vec<VoronoiCell>, Vec<u32>, Vec<(u32, u32)>);

fn reconcile_edges(
    vertices: &mut Vec<Vec3>,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    unresolved_edges: &[live_dedup::UnresolvedEdgeMismatch],
    mut cells: Vec<VoronoiCell>,
    mut cell_indices: Vec<u32>,
    tb: &mut TimingBuilder,
) -> Result<ReconciledWithResiduals, crate::VoronoiError> {
    let repair_edges_storage: Vec<live_dedup::EdgeRecord> = unresolved_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();

    let t = Timer::start();
    // The sphere has no boundary: every interior edge must pair.
    let post_repair_unpaired = edge_reconcile::reconcile_unresolved_edges(
        &repair_edges_storage,
        vertices.as_slice(),
        &mut cells,
        &mut cell_indices,
        edge_reconcile::VertexKeys::Sharded(vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        edge_reconcile::repair_apply_from_env(),
        |_, _| false,
    )?;
    // The simple cross-bin stitch above is the only repair pass: any surviving
    // unpaired interior edge is surfaced as a residual error by the caller
    // (valid-or-error contract — see docs/reclip-hull-snap-experiment-2026-06.md
    // for why post-hoc Tier-2 repair was investigated and dropped).
    tb.set_edge_reconcile(t.elapsed());
    Ok((cells, cell_indices, post_repair_unpaired))
}

/// Map effective cells back to original input indices.
///
/// Welded twins alias their canonical cell's `(start, len)` range in the
/// shared index buffer rather than receiving copied boundaries, and the weld
/// map records the canonical (smallest) original index per cell so consumers
/// and validation can account for shared cells explicitly.
fn remap_cells_to_original_indices(
    points: &[Vec3],
    merge_result: Option<&MergeResult>,
    eff_cells: Vec<VoronoiCell>,
    eff_cell_indices: Vec<u32>,
) -> (Vec<VoronoiCell>, Vec<u32>, Option<Vec<u32>>) {
    if let Some(merge_result) = merge_result {
        let mut eff_to_canonical: Vec<u32> = vec![u32::MAX; eff_cells.len()];
        let mut new_cells = Vec::with_capacity(points.len());
        let mut weld_map = Vec::with_capacity(points.len());

        for orig_idx in 0..points.len() {
            let eff_idx = merge_result.original_to_effective[orig_idx] as usize;
            if eff_to_canonical[eff_idx] == u32::MAX {
                eff_to_canonical[eff_idx] = orig_idx as u32;
            }
            weld_map.push(eff_to_canonical[eff_idx]);
            new_cells.push(eff_cells[eff_idx]);
        }
        (new_cells, eff_cell_indices, Some(weld_map))
    } else {
        (eff_cells, eff_cell_indices, None)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_query_grid, cell_sum_sq_per_n, map_build_cells_error, map_cell_build_error,
        max_cell_occupancy, validate_generator_capacity,
    };
    use crate::knn_clipping::cell_build::{CellBuildError, CellFailure};
    use crate::knn_clipping::live_dedup::{BuildCellsError, PackedLayoutCapacityError};
    use crate::VoronoiError;
    use glam::Vec3;

    #[test]
    fn map_projection_invalid_to_unsupported_geometry() {
        let err = map_cell_build_error(
            CellBuildError {
                generator_idx: 7,
                failure: CellFailure::ProjectionInvalid,
                detail: None,
            },
            &[],
            None,
        );
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
        let err = map_cell_build_error(
            CellBuildError {
                generator_idx: 11,
                failure: CellFailure::UnboundedAfterExhaustion,
                detail: None,
            },
            &[],
            None,
        );
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
        let err = map_cell_build_error(
            CellBuildError {
                generator_idx: 13,
                failure: CellFailure::TooManyVertices,
                detail: None,
            },
            &[],
            None,
        );
        match err {
            VoronoiError::ComputationFailed(msg) => {
                assert!(msg.contains("13"));
                assert!(msg.contains("vertex budget"));
            }
            other => panic!("expected ComputationFailed, got {:?}", other),
        }
    }

    #[test]
    fn map_cell_build_error_appends_detail_when_present() {
        let err = map_cell_build_error(
            CellBuildError {
                generator_idx: 17,
                failure: CellFailure::NoValidSeed,
                detail: Some("unexpected vertex extraction failure".to_string()),
            },
            &[],
            None,
        );
        match err {
            VoronoiError::ComputationFailed(msg) => {
                assert!(msg.contains("17"));
                assert!(msg.contains("NoValidSeed"));
                assert!(msg.contains("unexpected vertex extraction failure"));
            }
            other => panic!("expected ComputationFailed, got {:?}", other),
        }
    }

    #[test]
    fn map_packed_layout_capacity_to_representation_limit() {
        let err = map_build_cells_error(
            BuildCellsError::PackedLayoutCapacity(PackedLayoutCapacityError {
                bin: 5,
                local_population: 4096,
                num_bins: 96,
                local_shift: 8,
                local_mask: 255,
            }),
            &[],
            None,
        );
        match err {
            VoronoiError::RepresentationLimit(msg) => {
                assert!(msg.contains("bin 5"));
                assert!(msg.contains("4096"));
                assert!(msg.contains("255"));
                assert!(msg.contains("96"));
            }
            other => panic!("expected RepresentationLimit, got {:?}", other),
        }
    }

    #[test]
    fn map_build_cells_representation_limit_to_public_representation_limit() {
        let err = map_build_cells_error(
            BuildCellsError::RepresentationLimit(
                "cell vertex count exceeds u8 capacity".to_string(),
            ),
            &[],
            None,
        );
        match err {
            VoronoiError::RepresentationLimit(msg) => {
                assert!(msg.contains("cell vertex count"));
                assert!(msg.contains("u8"));
            }
            other => panic!("expected RepresentationLimit, got {:?}", other),
        }
    }

    #[test]
    fn map_clipped_away_with_coincident_neighbor_to_degenerate_input() {
        let g = glam::Vec3::new(1.0, 0.0, 0.0);
        let twin = glam::Vec3::new(1.0, 5e-7, 0.0);
        let far = glam::Vec3::new(0.0, 1.0, 0.0);
        let err = map_cell_build_error(
            CellBuildError {
                generator_idx: 0,
                failure: CellFailure::ClippedAway,
                detail: None,
            },
            &[g, twin, far],
            None,
        );
        match err {
            VoronoiError::DegenerateInput {
                coincident_pairs,
                message,
            } => {
                assert_eq!(coincident_pairs, 1);
                assert!(message.contains("generator 0"));
                assert!(message.contains("[1]"));
                assert!(message.contains("Weld"));
            }
            other => panic!("expected DegenerateInput, got {:?}", other),
        }
    }

    #[test]
    fn map_clipped_away_without_coincidence_stays_computation_failed() {
        let err = map_cell_build_error(
            CellBuildError {
                generator_idx: 0,
                failure: CellFailure::ClippedAway,
                detail: None,
            },
            &[
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            ],
            None,
        );
        match err {
            VoronoiError::ComputationFailed(msg) => assert!(msg.contains("ClippedAway")),
            other => panic!("expected ComputationFailed, got {:?}", other),
        }
    }

    #[test]
    fn clustered_input_triggers_occupancy_rebuild() {
        use crate::cube_grid::CubeMapGrid;
        use crate::knn_clipping::timing::TimingBuilder;

        // Deterministic golden-angle spiral cluster in a ~0.1 rad cap around
        // +Z: a density-derived grid packs thousands of points per cell.
        let n = 20_000usize;
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let points: Vec<Vec3> = (0..n)
            .map(|i| {
                let r = 0.1 * ((i as f32 + 0.5) / n as f32).sqrt();
                let theta = golden * i as f32;
                Vec3::new(r * theta.cos(), r * theta.sin(), 1.0).normalize()
            })
            .collect();

        let naive_res = crate::policy::knn_grid_resolution(n);
        let naive_grid = CubeMapGrid::new(&points, naive_res);
        let naive_occupancy = max_cell_occupancy(&naive_grid);
        // The trigger is the catastrophic-work signal Σocc²/n: this fully
        // concentrated fixture must clear it (all points pile into a few cells).
        let naive_sum_sq_per_n = cell_sum_sq_per_n(&naive_grid, n);
        assert!(
            naive_sum_sq_per_n > crate::policy::GRID_REBUILD_SUMSQ_PER_N,
            "fixture must be catastrophically concentrated (Σocc²/n {naive_sum_sq_per_n:.0})"
        );

        let mut tb = TimingBuilder::new();
        let grid = build_query_grid(&points, &mut tb);
        let rebuilt_occupancy = max_cell_occupancy(&grid);
        assert!(
            grid.res() > naive_res,
            "occupancy feedback must raise the resolution ({} -> {})",
            naive_res,
            grid.res()
        );
        assert!(
            rebuilt_occupancy < naive_occupancy / 4,
            "rebuild must materially reduce the fullest cell ({naive_occupancy} -> {rebuilt_occupancy})"
        );
        // Memory budget: total cells stay O(n).
        let cells = 6 * grid.res() * grid.res();
        assert!(cells as f64 <= crate::policy::GRID_MAX_CELLS_PER_POINT * n as f64 * 1.1);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn reject_generator_counts_above_u32_capacity() {
        let err = validate_generator_capacity((u32::MAX as usize) + 1)
            .expect_err("generator count above u32::MAX should fail");
        match err {
            VoronoiError::RepresentationLimit(msg) => {
                assert!(msg.contains("generator count"));
                assert!(msg.contains("u32"));
            }
            other => panic!("expected RepresentationLimit, got {:?}", other),
        }
    }
}
