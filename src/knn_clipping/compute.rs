//! Compute entry points for the kNN + clipping Voronoi backend.

mod coplanar;
mod error_mapping;

use glam::Vec3;

use super::edge_reconcile;
use super::local_rebuild;
use super::output_resolution;
use super::preprocess::{try_merge_close_points, MergeResult};
use crate::cell_layout::LiveCellLayout;
use crate::cube_grid::CubeMapGrid;
#[cfg(feature = "timing")]
use crate::cube_grid::CubeMapGridBuildTimings;
use crate::diagram::VoronoiCell;
use crate::live_dedup;
use crate::timing::{Timer, TimingBuilder};
use crate::{
    CellKillingPolicy, ComputeOutput, ComputeReport, DegenerateMode, DegenerateReport,
    LocalRebuildMode, LocalRebuildStatus, PreprocessMode, PreprocessReport, VoronoiConfig,
};

use coplanar::maybe_perturb_coplanar;
use error_mapping::map_build_cells_error;

/// Per-seed neighbor count for the local rebuild's grid kNN gather (the 2-ring gather
/// collects each seed's `k + 1` nearest via the shell frontier).
const LOCAL_REBUILD_GATHER_K: usize = 32;
/// Grow-until-clean round cap per defect component.
const LOCAL_REBUILD_MAX_ROUNDS: usize = 12;

/// Everything the shared pipeline produces before the plain and report paths
/// diverge: canonicalized inputs, the reconciled effective arrays, and the
/// local rebuild outcome. `TimingBuilder` rides along so the caller's final remap
/// lands in the same timing report.
struct PipelineState {
    points: Vec<Vec3>,
    effective_input: EffectiveInput,
    preprocess_report: PreprocessReport,
    geometry: EffectiveGeometry,
    edge_mismatches: Vec<live_dedup::EdgeMismatch>,
    residual_unpaired: Vec<(u32, u32)>,
    local_rebuild_seed_pairs: Vec<(u32, u32)>,
    local_rebuild: LocalRebuildOutcome,
    output_resolution: crate::OutputResolutionReport,
    positive_resolution: Option<output_resolution::PositiveResolutionReport>,
    cell_killing_generators: Vec<usize>,
    tb: TimingBuilder,
}

/// Coherent effective-space diagram storage from assembly through final
/// output resolution. Later phases may leave positions unreferenced, but the
/// cell spans and index buffer always describe this vertex-id space together.
struct EffectiveGeometry {
    vertices: Vec<Vec3>,
    cells: Vec<VoronoiCell>,
    cell_indices: Vec<u32>,
}

impl PipelineState {
    #[cfg(test)]
    fn effective_points_ref(&self) -> &[Vec3] {
        self.effective_input.points(&self.points)
    }

    fn merge_result(&self) -> Option<&MergeResult> {
        self.effective_input.merge_result()
    }
}

/// The point set selected by preprocessing. Identity input borrows the
/// canonicalized original points; only an actual merge owns representatives
/// and the map needed to expand their cells back to original generators.
enum EffectiveInput {
    Identity,
    Merged(MergeResult),
}

impl EffectiveInput {
    fn points<'a>(&'a self, original_points: &'a [Vec3]) -> &'a [Vec3] {
        match self {
            Self::Identity => original_points,
            Self::Merged(result) => &result.effective_points,
        }
    }

    fn merge_result(&self) -> Option<&MergeResult> {
        match self {
            Self::Identity => None,
            Self::Merged(result) => Some(result),
        }
    }

    fn effective_len(&self, original_len: usize) -> usize {
        match self {
            Self::Identity => original_len,
            Self::Merged(result) => result.effective_points.len(),
        }
    }

    fn num_merged(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Merged(result) => result.num_merged,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolutionDiscoveryMode {
    CertifiedHint,
    ExhaustiveDriftFallback,
}

impl ResolutionDiscoveryMode {
    const fn from_drift(resolution_drift_exceeded: bool) -> Self {
        if resolution_drift_exceeded {
            Self::ExhaustiveDriftFallback
        } else {
            Self::CertifiedHint
        }
    }

    const fn drift_fallback(self) -> bool {
        matches!(self, Self::ExhaustiveDriftFallback)
    }

    const fn certified_hint(self) -> bool {
        matches!(self, Self::CertifiedHint)
    }
}

fn canonicalize_pipeline_exact_zero_edges(
    vertices: &[Vec3],
    vertex_keys: &live_dedup::ShardedVertexKeys,
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    hinted_candidates: Vec<(u32, u32)>,
    mutation_scan_cells: &[u32],
    mode: ResolutionDiscoveryMode,
) -> Result<output_resolution::CanonicalizationOutcome, crate::VoronoiError> {
    let (exact_zero_candidates, localized_candidate_cells) = if mode.certified_hint() {
        // Construction hints name pre-reconciliation edges. Re-scan their
        // degree-local incident cells in the terminal diagram so a local rebuild
        // cannot leave a stale candidate, and add the complete footprint of
        // every post-assembly mutation. Untouched cells retain the original
        // construction certificate.
        let mut discovery_cells: Vec<usize> = mutation_scan_cells
            .iter()
            .map(|&cell| cell as usize)
            .collect();
        discovery_cells.reserve(hinted_candidates.len() * 6);
        let mut complete = true;
        for &(a, b) in &hinted_candidates {
            for vertex in [a, b] {
                if let Some(key) = vertex_keys.get(vertex) {
                    discovery_cells.extend(key.map(|generator| generator as usize));
                } else {
                    complete = false;
                    break;
                }
            }
            if !complete {
                break;
            }
        }
        if complete {
            discovery_cells.sort_unstable();
            discovery_cells.dedup();
            let candidates = output_resolution::collect_zero_edges_in_cells(
                vertices,
                cells,
                cell_indices,
                &discovery_cells,
            )?;

            // A rebuilt/minted endpoint may not exist in the assembly key
            // store. In that rare case candidate discovery is still local and
            // complete, but quotient classification conservatively considers
            // every cell. Otherwise include every key owner so all references
            // rewritten by a contraction are in scope.
            let mut candidate_cells = discovery_cells;
            for &(a, b) in &candidates {
                for vertex in [a, b] {
                    if let Some(key) = vertex_keys.get(vertex) {
                        candidate_cells.extend(key.map(|generator| generator as usize));
                    } else {
                        complete = false;
                        break;
                    }
                }
                if !complete {
                    break;
                }
            }
            if complete {
                candidate_cells.sort_unstable();
                candidate_cells.dedup();
                (Some(candidates), Some(candidate_cells))
            } else {
                (Some(candidates), None)
            }
        } else {
            // Missing provenance invalidates localization. Fall back to the
            // terminal whole-diagram scan rather than guess.
            (None, None)
        }
    } else {
        (None, None)
    };

    output_resolution::canonicalize_exact_zero_edges(
        vertices,
        cells,
        cell_indices,
        exact_zero_candidates,
        localized_candidate_cells,
    )
}

struct ResolutionView<'a> {
    vertices: &'a [Vec3],
    vertex_keys: &'a live_dedup::ShardedVertexKeys,
    cells: &'a [VoronoiCell],
    cell_indices: &'a [u32],
}

struct PositiveResolutionCover {
    candidates: Vec<(f64, u32, u32)>,
    component_cells: Vec<usize>,
    certificate_cells: Vec<usize>,
}

fn positive_resolution_cover(
    view: ResolutionView<'_>,
    hinted_cells: &[u32],
    mutation_cells: &[u32],
    exact_changed_cells: &[usize],
    mode: ResolutionDiscoveryMode,
    threshold: f64,
) -> Result<PositiveResolutionCover, crate::VoronoiError> {
    let mut discovery_cells = if mode.certified_hint() {
        hinted_cells
            .iter()
            .chain(mutation_cells)
            .map(|&cell| cell as usize)
            .chain(exact_changed_cells.iter().copied())
            .collect::<Vec<_>>()
    } else {
        (0..view.cells.len()).collect()
    };
    discovery_cells.sort_unstable();
    discovery_cells.dedup();
    let candidates = output_resolution::collect_positive_edges_in_cells(
        view.vertices,
        view.cells,
        view.cell_indices,
        &discovery_cells,
        threshold * threshold,
    )?;
    if candidates.is_empty() {
        return Ok(PositiveResolutionCover {
            candidates,
            component_cells: discovery_cells.clone(),
            certificate_cells: discovery_cells,
        });
    }

    let mut component_cells = discovery_cells;
    let mut complete = true;
    for &(_, a, b) in &candidates {
        for vertex in [a, b] {
            if let Some(key) = view.vertex_keys.get(vertex) {
                component_cells.extend(key.map(|generator| generator as usize));
            } else {
                complete = false;
                break;
            }
        }
        if !complete {
            break;
        }
    }
    if !complete {
        let exhaustive: Vec<usize> = (0..view.cells.len()).collect();
        return Ok(PositiveResolutionCover {
            candidates,
            component_cells: exhaustive.clone(),
            certificate_cells: exhaustive,
        });
    }
    component_cells.sort_unstable();
    component_cells.dedup();

    let mut certificate_cells = component_cells.clone();
    for &cell_idx in &component_cells {
        let cell = view.cells.get(cell_idx).ok_or_else(|| {
            crate::VoronoiError::ComputationFailed(format!(
                "positive resolution referenced out-of-range cell {cell_idx}"
            ))
        })?;
        let span = view
            .cell_indices
            .get(cell.vertex_start()..cell.vertex_start() + cell.vertex_count())
            .ok_or_else(|| {
                crate::VoronoiError::ComputationFailed(
                    "positive-resolution certificate cell span is out of range".into(),
                )
            })?;
        for &vertex in span {
            if let Some(key) = view.vertex_keys.get(vertex) {
                certificate_cells.extend(key.map(|generator| generator as usize));
            } else {
                complete = false;
                break;
            }
        }
        if !complete {
            break;
        }
    }
    if !complete {
        certificate_cells = (0..view.cells.len()).collect();
    } else {
        certificate_cells.sort_unstable();
        certificate_cells.dedup();
    }
    Ok(PositiveResolutionCover {
        candidates,
        component_cells,
        certificate_cells,
    })
}

/// The shared front of both compute paths: validate/canonicalize → preprocess
/// and grid → per-cell construction → assemble → reconcile → optional local
/// rebuild → output resolution. The plain path fails loud on residuals; the
/// report path surfaces them in `ComputeReport`.
fn run_core_pipeline(
    points: Vec<Vec3>,
    preprocess_mode: PreprocessMode,
    local_rebuild_mode: LocalRebuildMode,
    positive_chord_threshold: Option<f32>,
    workspace: Option<&super::driver::BuildWorkspace>,
) -> Result<PipelineState, crate::VoronoiError> {
    validate_generator_capacity(points.len())?;
    let mut points = points;
    validate_and_canonicalize_unit_points(&mut points)?;
    validate_preprocess_mode(preprocess_mode)?;
    let mut tb = TimingBuilder::new();

    let PreparedPointsAndGrid {
        effective_input,
        report: preprocess_report,
        mut grid,
        occupancy_rebuilt,
    } = prepare_points_and_grid(&points, preprocess_mode, workspace, &mut tb)?;

    let effective_points_ref = effective_input.points(&points);

    let point_cell_storage = grid.take_point_cells();
    let construction_policy = CellConstructionPolicy {
        positive_chord_threshold,
        occupancy_rebuilt,
    };

    let sharded = construct_cell_shards(
        effective_points_ref,
        &grid,
        point_cell_storage,
        effective_input.merge_result(),
        construction_policy,
        workspace,
        &mut tb,
    )?;
    let assembled = assemble_shards(sharded, &mut tb)?;
    let live_dedup::AssemblyResult {
        vertices,
        vertex_keys: assembly_vertex_keys,
        edge_mismatches,
        cells,
        cell_indices,
        exact_zero_edge_candidates,
        resolution_edge_hint_cells,
        exact_zero_edge_hint_cells,
        resolution_drift_exceeded,
        incidence_summary,
        dedup_sub: _,
    } = assembled;
    let mut geometry = EffectiveGeometry {
        vertices,
        cells,
        cell_indices,
    };
    let reconcile_result = reconcile_edges(
        &mut geometry,
        &assembly_vertex_keys,
        &edge_mismatches,
        &mut tb,
    )?;
    let edge_reconcile::ReconcileResult {
        residual_pairs: residual_unpaired,
        local_rebuild_seed_pairs,
        merge_affected_cells,
        resolution_scan_cells: reconcile_resolution_scan_cells,
        changed_cell_snapshots,
        ..
    } = reconcile_result;
    // This is part of the plain-return safety gate, not merely a local rebuild
    // trigger. Compute it even when local rebuild is disabled so that mode cannot
    // suppress a known-invalid low-incidence output.
    let t_low_incidence = std::time::Instant::now();
    let topology = if reconcile_resolution_scan_cells.is_empty() {
        let incremental = TopologySummary {
            used_vertices: incidence_summary.used_vertices,
            live_half_edges: incidence_summary.live_half_edges,
            low_incidence: incidence_summary.low_incidence,
        };
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            incremental,
            summarize_topology_scalar(
                geometry.vertices.len(),
                &geometry.cells,
                &geometry.cell_indices,
            ),
            "owner-local incidence summary diverged from the live-window scan"
        );
        incremental
    } else if let Some(incremental) = summarize_topology_after_reconcile(
        &incidence_summary,
        &assembly_vertex_keys,
        &changed_cell_snapshots,
        &geometry.cells,
        &geometry.cell_indices,
    ) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            incremental,
            summarize_topology_scalar(
                geometry.vertices.len(),
                &geometry.cells,
                &geometry.cell_indices,
            ),
            "defect-local topology summary diverged from the live-window scan"
        );
        incremental
    } else {
        summarize_topology(
            geometry.vertices.len(),
            &geometry.cells,
            &geometry.cell_indices,
        )
    };
    let low_incidence_scan_time = t_low_incidence.elapsed();
    let LocalRebuildResult {
        outcome: local_rebuild,
        resolution_scan_cells: local_rebuild_resolution_scan_cells,
    } = maybe_rebuild_effective(
        effective_points_ref,
        &grid,
        &mut geometry,
        &assembly_vertex_keys,
        &residual_unpaired,
        &local_rebuild_seed_pairs,
        &merge_affected_cells,
        topology,
        low_incidence_scan_time,
        local_rebuild_mode,
    );
    let reconcile_resolution_scan_cell_count = reconcile_resolution_scan_cells.len();
    let local_rebuild_resolution_scan_cell_count = local_rebuild_resolution_scan_cells.len();
    let mut mutation_scan_cells = reconcile_resolution_scan_cells;
    mutation_scan_cells.extend(local_rebuild_resolution_scan_cells);
    mutation_scan_cells.sort_unstable();
    mutation_scan_cells.dedup();

    let resolution_mode = ResolutionDiscoveryMode::from_drift(resolution_drift_exceeded);
    let hinted_candidate_count = exact_zero_edge_candidates.len();
    let resolution_outcome = canonicalize_pipeline_exact_zero_edges(
        &geometry.vertices,
        &assembly_vertex_keys,
        &mut geometry.cells,
        &mut geometry.cell_indices,
        exact_zero_edge_candidates,
        &mutation_scan_cells,
        resolution_mode,
    )?;
    let positive_resolution = if let Some(threshold) = positive_chord_threshold {
        if resolution_outcome.report.exact_zero_edges_remaining == 0 {
            let cover = positive_resolution_cover(
                ResolutionView {
                    vertices: &geometry.vertices,
                    vertex_keys: &assembly_vertex_keys,
                    cells: &geometry.cells,
                    cell_indices: &geometry.cell_indices,
                },
                &resolution_edge_hint_cells,
                &mutation_scan_cells,
                &resolution_outcome.changed_cells,
                resolution_mode,
                f64::from(threshold),
            )?;
            Some(output_resolution::simplify_positive_edges(
                &geometry.vertices,
                &mut geometry.cells,
                &mut geometry.cell_indices,
                cover.candidates,
                &cover.component_cells,
                &cover.certificate_cells,
                f64::from(threshold),
                resolution_edge_hint_cells.len(),
            )?)
        } else {
            None
        }
    } else {
        None
    };
    tb.set_output_resolution_discovery(
        resolution_mode.drift_fallback(),
        reconcile_resolution_scan_cell_count,
        local_rebuild_resolution_scan_cell_count,
        exact_zero_edge_hint_cells,
        hinted_candidate_count,
        resolution_outcome.report.exact_zero_edges_detected,
    );
    Ok(PipelineState {
        points,
        effective_input,
        preprocess_report,
        geometry,
        edge_mismatches,
        residual_unpaired,
        local_rebuild_seed_pairs,
        local_rebuild,
        output_resolution: resolution_outcome.report,
        positive_resolution,
        cell_killing_generators: resolution_outcome.cell_killing_generators,
        tb,
    })
}

fn enforce_cell_killing_policy(
    state: &PipelineState,
    policy: CellKillingPolicy,
) -> Result<(), crate::VoronoiError> {
    if state.cell_killing_generators.is_empty() {
        return Ok(());
    }

    match policy {
        CellKillingPolicy::Preserve => return Ok(()),
        CellKillingPolicy::Error => {}
    }

    let generator_indices = if let Some(merge) = state.merge_result() {
        merge
            .original_to_effective
            .iter()
            .enumerate()
            .filter_map(|(original, &effective)| {
                state
                    .cell_killing_generators
                    .binary_search(&(effective as usize))
                    .is_ok()
                    .then_some(original)
            })
            .collect()
    } else {
        state.cell_killing_generators.clone()
    };

    Err(crate::VoronoiError::CellEliminationRequired {
        generator_indices,
        remaining_exact_zero_edges: state.output_resolution.exact_zero_edges_remaining,
    })
}

fn validate_preprocess_mode(mode: PreprocessMode) -> Result<(), crate::VoronoiError> {
    let PreprocessMode::MergeWithin(threshold) = mode else {
        return Ok(());
    };
    if !threshold.is_finite() || threshold <= 0.0 || threshold * threshold == 0.0 {
        return Err(crate::VoronoiError::InvalidConfiguration(format!(
            "MergeWithin threshold must be finite, positive, and large enough for its squared f32 distance to be nonzero; got {threshold:?}"
        )));
    }
    Ok(())
}

pub(super) fn compute_voronoi_knn_clipping_owned_core(
    points: Vec<Vec3>,
    preprocess_mode: PreprocessMode,
    local_rebuild_mode: LocalRebuildMode,
    cell_killing_policy: CellKillingPolicy,
    workspace: Option<&super::driver::BuildWorkspace>,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    let mut state =
        run_core_pipeline(points, preprocess_mode, local_rebuild_mode, None, workspace)?;
    check_plain_return_signals(
        state.local_rebuild,
        &state.residual_unpaired,
        &state.local_rebuild_seed_pairs,
    )?;
    enforce_cell_killing_policy(&state, cell_killing_policy)?;

    let t = Timer::start();
    let (cells, cell_indices, weld_map) = remap_cells_to_original_indices(
        &state.points,
        state.effective_input.merge_result(),
        state.geometry.cells,
        state.geometry.cell_indices,
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(
        state.points,
        state.geometry.vertices,
        cells,
        cell_indices,
        weld_map,
    );
    state.tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = state.tb.finish();
    timings.report(diagram.num_cells());

    crate::validation::verify_sphere_if_enabled(&diagram)?;
    Ok(diagram)
}

/// Run `attempt` once and — when the config opts into
/// `DegenerateMode::PerturbCoplanar` — retry a certified affine-circle or
/// conservatively detected full-great-circle failure once on perturbed points.
/// `attempt` receives `perturbation_applied`.
fn with_coplanar_perturb_retry<T>(
    points: Vec<Vec3>,
    degenerate_mode: DegenerateMode,
    attempt: impl Fn(Vec<Vec3>, bool) -> Result<T, crate::VoronoiError>,
) -> Result<T, crate::VoronoiError> {
    if !matches!(degenerate_mode, DegenerateMode::PerturbCoplanar) {
        return attempt(points, false);
    }

    match attempt(points.clone(), false) {
        Ok(value) => Ok(value),
        Err(err) => match maybe_perturb_coplanar(&points, &err) {
            Some(perturbed) => attempt(perturbed, true),
            None => Err(err),
        },
    }
}

pub(crate) fn compute_voronoi_knn_clipping_with_config_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    with_coplanar_perturb_retry(points, config.degenerate_mode, |points, _| {
        compute_voronoi_knn_clipping_owned_core(
            points,
            config.preprocess_mode,
            config.local_rebuild_mode,
            config.cell_killing_policy,
            None,
        )
    })
}

pub(crate) fn compute_voronoi_knn_clipping_with_workspace_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
    workspace: &super::driver::BuildWorkspace,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    with_coplanar_perturb_retry(points, config.degenerate_mode, |points, _| {
        compute_voronoi_knn_clipping_owned_core(
            points,
            config.preprocess_mode,
            config.local_rebuild_mode,
            config.cell_killing_policy,
            Some(workspace),
        )
    })
}

pub(crate) fn compute_voronoi_knn_clipping_with_report_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<ComputeOutput, crate::VoronoiError> {
    with_coplanar_perturb_retry(
        points,
        config.degenerate_mode,
        |points, perturbation_applied| {
            compute_voronoi_knn_clipping_report_core(
                points,
                config.preprocess_mode,
                config.local_rebuild_mode,
                config.cell_killing_policy,
                None,
                DegenerateReport {
                    requested_mode: config.degenerate_mode,
                    perturbation_applied,
                },
            )
            .map(|(output, _)| output)
        },
    )
}

pub(crate) fn compute_voronoi_knn_clipping_simplified_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
    threshold: f32,
) -> Result<(ComputeOutput, output_resolution::PositiveResolutionReport), crate::VoronoiError> {
    with_coplanar_perturb_retry(
        points,
        config.degenerate_mode,
        |points, perturbation_applied| {
            let (output, report) = compute_voronoi_knn_clipping_report_core(
                points,
                config.preprocess_mode,
                config.local_rebuild_mode,
                CellKillingPolicy::Error,
                Some(threshold),
                DegenerateReport {
                    requested_mode: config.degenerate_mode,
                    perturbation_applied,
                },
            )?;
            let report = report.ok_or_else(|| {
                crate::VoronoiError::ComputationFailed(
                    "construction-aware simplification did not produce a resolution report".into(),
                )
            })?;
            Ok((output, report))
        },
    )
}

fn compute_voronoi_knn_clipping_report_core(
    points: Vec<Vec3>,
    preprocess_mode: PreprocessMode,
    local_rebuild_mode: LocalRebuildMode,
    cell_killing_policy: CellKillingPolicy,
    positive_chord_threshold: Option<f32>,
    degenerate_report: DegenerateReport,
) -> Result<
    (
        ComputeOutput,
        Option<output_resolution::PositiveResolutionReport>,
    ),
    crate::VoronoiError,
> {
    let mut state = run_core_pipeline(
        points,
        preprocess_mode,
        local_rebuild_mode,
        positive_chord_threshold,
        None,
    )?;
    enforce_cell_killing_policy(&state, cell_killing_policy)?;
    let local_rebuild_accepted = state.local_rebuild.accepted();
    // Surface output-invariant residuals alongside the detection records. If
    // local rebuild was accepted, the returned diagram is strictly valid and
    // these residuals no longer survive to output.
    let assembly_edge_mismatch_count = state.edge_mismatches.len();
    let residual_unpaired_edges: Vec<(u32, u32)> = if local_rebuild_accepted {
        Vec::new()
    } else {
        state
            .residual_unpaired
            .iter()
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect()
    };
    let residual_reconciliation_pairs = if local_rebuild_accepted {
        Vec::new()
    } else {
        state.local_rebuild_seed_pairs.clone()
    };
    if !local_rebuild_accepted {
        for &(a, b) in &state.residual_unpaired {
            state.edge_mismatches.push(live_dedup::EdgeMismatch {
                key: live_dedup::pack_edge(a, b),
                origin: live_dedup::EdgeMismatchOrigin::PostReconciliationUnpaired,
            });
        }
    }

    let effective_diagram = state.merge_result().map(|merge| {
        crate::SphericalVoronoi::from_raw_parts(
            merge.effective_points.clone(),
            state.geometry.vertices.clone(),
            state.geometry.cells.clone(),
            state.geometry.cell_indices.clone(),
            None,
        )
    });
    let effective_validation = effective_diagram.as_ref().map(crate::validation::validate);

    let t = Timer::start();
    let (cells, cell_indices, weld_map) = remap_cells_to_original_indices(
        &state.points,
        state.effective_input.merge_result(),
        state.geometry.cells,
        state.geometry.cell_indices,
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(
        state.points,
        state.geometry.vertices,
        cells,
        cell_indices,
        weld_map,
    );
    let returned_validation = crate::validation::validate(&diagram);
    state.tb.set_assemble(t.elapsed());

    let timings = state.tb.finish();
    timings.report(diagram.num_cells());

    let positive_resolution = state.positive_resolution.take();
    Ok((
        ComputeOutput {
            diagram,
            effective_diagram,
            report: ComputeReport {
                preprocess: state.preprocess_report,
                degenerate: degenerate_report,
                returned_validation,
                effective_validation,
                assembly_edge_mismatch_count,
                local_rebuild: state.local_rebuild.report_status(),
                output_resolution: state.output_resolution,
                residual_unpaired_edges,
                residual_reconciliation_pairs,
                reconciliation_edge_records: state
                    .edge_mismatches
                    .iter()
                    .map(|m| {
                        let (a, b) = edge_reconcile::unpack_edge(m.key.as_u64());
                        (a.min(b), a.max(b), m.origin)
                    })
                    .collect(),
            },
        },
        positive_resolution,
    ))
}

/// Cheap topology facts collected by the incidence pass already required for
/// the local rebuild trigger and plain-return safety gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TopologySummary {
    used_vertices: usize,
    live_half_edges: usize,
    low_incidence: bool,
}

impl TopologySummary {
    /// Euler characteristic implied by exact edge agreement (`E = H / 2`).
    /// An odd half-edge count cannot describe a closed paired subdivision.
    fn paired_euler_characteristic(self, num_cells: usize) -> Option<i128> {
        if !self.live_half_edges.is_multiple_of(2) {
            return None;
        }
        Some(self.used_vertices as i128 - (self.live_half_edges / 2) as i128 + num_cells as i128)
    }

    fn has_sphere_euler(self, num_cells: usize) -> bool {
        self.paired_euler_characteristic(num_cells) == Some(2)
    }
}

/// Update construction-time incidence from the sparse set of cycles observed
/// before reconciliation. The exact old incidence of each delta-touched
/// vertex is recovered from the at-most-three cells in its construction key;
/// a snapshotted cycle supplies the old span and an unsnapshotted cell is, by
/// definition, unchanged. Thus no whole-diagram count ledger has to survive
/// assembly.
///
/// A pre-existing low-incidence boolean is ambiguous: reconciliation might
/// repair its last source or leave another untouched source behind. Assembly
/// therefore retains the sparse ids behind that flag and this routine checks
/// them exactly alongside the mutation delta. Missing provenance falls back
/// to the full scan.
fn summarize_topology_after_reconcile(
    baseline: &live_dedup::IncidenceSummary,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    snapshots: &[edge_reconcile::CellCycleSnapshot],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Option<TopologySummary> {
    if snapshots.is_empty()
        || (baseline.low_incidence && baseline.low_incidence_vertices.is_empty())
    {
        return None;
    }

    use rustc_hash::{FxHashMap, FxHashSet};

    let mut delta: FxHashMap<u32, i32> = FxHashMap::default();
    let mut original_cycles: FxHashMap<u32, &[u32]> = FxHashMap::default();
    let mut seen_cells: FxHashSet<u32> = FxHashSet::default();
    let mut half_edge_delta = 0i64;
    for snapshot in snapshots {
        if !seen_cells.insert(snapshot.cell) {
            return None;
        }
        original_cycles.insert(snapshot.cell, &snapshot.vertices);
        let cell = cells.get(snapshot.cell as usize)?;
        let start = cell.vertex_start();
        let end = start.checked_add(cell.vertex_count())?;
        let final_cycle = cell_indices.get(start..end)?;

        half_edge_delta = half_edge_delta
            .checked_add(i64::try_from(final_cycle.len()).ok()?)?
            .checked_sub(i64::try_from(snapshot.vertices.len()).ok()?)?;
        for &vertex in &snapshot.vertices {
            *delta.entry(vertex).or_default() -= 1;
        }
        for &vertex in final_cycle {
            *delta.entry(vertex).or_default() += 1;
        }
    }
    for &vertex in &baseline.low_incidence_vertices {
        delta.entry(vertex).or_default();
    }

    let mut used_vertices = i64::try_from(baseline.used_vertices).ok()?;
    let mut low_incidence = false;
    for (vertex, change) in delta {
        let key = vertex_keys.get(vertex)?;
        let mut old = 0i64;
        for (slot, &owner) in key.iter().enumerate() {
            if key[..slot].contains(&owner) {
                continue;
            }
            let cycle = if let Some(&original) = original_cycles.get(&owner) {
                original
            } else {
                let cell = cells.get(owner as usize)?;
                let start = cell.vertex_start();
                let end = start.checked_add(cell.vertex_count())?;
                cell_indices.get(start..end)?
            };
            old += i64::try_from(cycle.iter().filter(|&&v| v == vertex).count()).ok()?;
        }
        let new = old.checked_add(i64::from(change))?;
        if new < 0 {
            return None;
        }
        used_vertices += i64::from(new != 0) - i64::from(old != 0);
        low_incidence |= new == 1 || new == 2;
    }

    let live_half_edges = i64::try_from(baseline.live_half_edges)
        .ok()?
        .checked_add(half_edge_delta)?;
    Some(TopologySummary {
        used_vertices: usize::try_from(used_vertices).ok()?,
        live_half_edges: usize::try_from(live_half_edges).ok()?,
        low_incidence,
    })
}

/// Summarize referenced vertices and live half-edges, including whether any
/// referenced vertex has degree 1 or 2 (a real defect the local rebuild should
/// examine).
///
/// Counts incidence over each cell's *live* window `[vertex_start ..
/// vertex_start + vertex_count)`, NOT the raw `cell_indices` buffer. Edge
/// reconciliation shrinks a cell's `vertex_count` in place without compacting
/// the backing buffer (see `apply_merges_in_place` /
/// `drop_degenerate_collinear_vertices` in `edge_reconcile`), so the buffer can
/// retain stale tail slots that no live cell references. Scanning the whole
/// buffer counts those stale slots as phantom degree-1/2 vertices and trips a
/// no-op local rebuild. Counting live windows matches the validators (`validate_impl`,
/// `verify_sphere_fast`) and the local rebuild's own `low_incidence_gens`.
/// Construction supplies the common-path summary, and defect-local
/// reconciliation updates it from captured cell cycles. This whole-diagram
/// scan is the conservative escape path for incomplete provenance. When it is
/// needed, multi-threaded builds use exact shared atomic counters read only
/// after the chunk-parallel scan; a one-thread Rayon pool uses the same plain
/// counter path as a build without `parallel`. The scale evidence is recorded
/// in `docs/performance.md#source-pinned-performance-decisions`.
fn summarize_topology_scalar(
    vertex_count: usize,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> TopologySummary {
    let mut cnt = vec![0u32; vertex_count];
    let mut live_half_edges = 0usize;
    let layout = LiveCellLayout::new(cells, cell_indices);
    for cell in cells {
        live_half_edges += cell.vertex_count();
        for &v in layout.span_for(cell) {
            cnt[v as usize] += 1;
        }
    }
    let mut used_vertices = 0usize;
    let mut low_incidence = false;
    for count in cnt {
        used_vertices += usize::from(count != 0);
        low_incidence |= count == 1 || count == 2;
    }
    TopologySummary {
        used_vertices,
        live_half_edges,
        low_incidence,
    }
}

fn summarize_topology(
    vertex_count: usize,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> TopologySummary {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
        let threads = rayon::current_num_threads().max(1);
        if threads == 1 {
            return summarize_topology_scalar(vertex_count, cells, cell_indices);
        }
        // Ordinary spherical vertices have incidence three. Keep the shared
        // counter footprint compact and detect the exceptional u8 wrap after
        // the scan by comparing its reconstructed half-edge total with the
        // exact total accumulated from the cells. A mismatch falls back to the
        // exact scalar u32 scan below.
        let cnt: Vec<AtomicU8> = (0..vertex_count).map(|_| AtomicU8::new(0)).collect();
        let chunk = cells.len().div_ceil(threads * 4).max(1024);
        let layout = LiveCellLayout::new(cells, cell_indices);
        let live_half_edges = cells
            .par_chunks(chunk)
            .map(|cells_chunk| {
                let mut half_edges = 0usize;
                for cell in cells_chunk {
                    half_edges += cell.vertex_count();
                    for &v in layout.span_for(cell) {
                        cnt[v as usize].fetch_add(1, Relaxed);
                    }
                }
                half_edges
            })
            .sum();
        let (used_vertices, low_incidence, counted_half_edges) = cnt
            .par_iter()
            .map(|c| {
                let count = c.load(Relaxed);
                (
                    usize::from(count != 0),
                    count == 1 || count == 2,
                    usize::from(count),
                )
            })
            .reduce(|| (0, false, 0), |a, b| (a.0 + b.0, a.1 || b.1, a.2 + b.2));
        if counted_half_edges != live_half_edges {
            return summarize_topology_scalar(vertex_count, cells, cell_indices);
        }
        TopologySummary {
            used_vertices,
            live_half_edges,
            low_incidence,
        }
    }
    #[cfg(not(feature = "parallel"))]
    summarize_topology_scalar(vertex_count, cells, cell_indices)
}

/// Outcome of local-rebuild processing, for the caller's fail-loud decision.
#[derive(Clone, Copy)]
struct LocalRebuildOutcome {
    status: LocalRebuildStatus,
    /// Detection found a low-incidence (degree-1/2) vertex defect. Such a
    /// vertex is strictly-invalid output even when every edge pairs (it fails
    /// `verify_sphere_effective_strict`'s "low-incidence vertex" check), so
    /// when the local rebuild was not accepted the plain path must fail loud on it —
    /// there is no unpaired-edge residual to trip the existing guard.
    low_incidence_defect: bool,
    /// The cheap `V - H/2 + F` check failed (or `H` was odd). This catches
    /// global topology defects at no additional traversal once exact edge
    /// agreement is supplied by construction.
    euler_defect: bool,
}

impl LocalRebuildOutcome {
    const fn new(
        status: LocalRebuildStatus,
        low_incidence_defect: bool,
        euler_defect: bool,
    ) -> LocalRebuildOutcome {
        LocalRebuildOutcome {
            status,
            low_incidence_defect,
            euler_defect,
        }
    }

    const fn accepted(self) -> bool {
        self.status.accepted()
    }

    const fn report_status(self) -> LocalRebuildStatus {
        self.status
    }
}

struct LocalRebuildResult {
    outcome: LocalRebuildOutcome,
    /// Cells whose final cycles were replaced by an accepted Hull3d splice.
    /// Newly minted vertices are referenced only from these cells.
    resolution_scan_cells: Vec<u32>,
}

impl LocalRebuildResult {
    fn unchanged(outcome: LocalRebuildOutcome) -> Self {
        Self {
            outcome,
            resolution_scan_cells: Vec::new(),
        }
    }
}

/// One fully materialized local-rebuild proposal. The replacement cells and
/// indices travel with the minted positions they reference and the exact
/// footprint needed by terminal output resolution.
struct LocalRebuildCandidate {
    minted_vertices: Vec<Vec3>,
    cells: Vec<VoronoiCell>,
    cell_indices: Vec<u32>,
    resolution_scan_cells: Vec<u32>,
}

impl LocalRebuildCandidate {
    fn from_work(work: local_rebuild::WorkingDiagram<'_>) -> Self {
        let resolution_scan_cells = work.overridden_cells();
        let (minted_vertices, cells, mut cell_indices) = work.into_flat();

        // The in-place and rebuild reconciliation oracles can present the same
        // cyclic boundary with different starting slots. Hull3d preserves that
        // arbitrary rotation when it splices a neighborhood. Canonicalize only
        // this cold rebuilt output so semantically identical local rebuild
        // backends remain byte-for-byte differential oracles; winding is
        // unchanged.
        canonicalize_cell_cycle_starts(&cells, &mut cell_indices);

        Self {
            minted_vertices,
            cells,
            cell_indices,
            resolution_scan_cells,
        }
    }

    /// Append the candidate's new positions for whole-diagram validation,
    /// then either roll that append back or install both replacement arrays.
    fn try_commit(
        self,
        effective_points: &[Vec3],
        geometry: &mut EffectiveGeometry,
        debug: bool,
        materialization_started: std::time::Instant,
    ) -> Option<Vec<u32>> {
        let Self {
            minted_vertices,
            cells,
            cell_indices,
            resolution_scan_cells,
        } = self;
        let base_vertex_count = geometry.vertices.len();
        geometry.vertices.extend(minted_vertices);
        let flat_elapsed = materialization_started.elapsed();
        let t_gate = std::time::Instant::now();

        // Whole-diagram never-worse gate: accept only if the rebuilt diagram is
        // strictly valid. Validate the effective arrays in place via
        // `verify_sphere_effective_strict` (same strict contract as `validate`,
        // pinned by the `effective_strict_matches_fast` differential test)
        // rather than cloning all arrays into a temporary `SphericalVoronoi`.
        // The gate must remain whole-diagram: minted triple identities can
        // affect cells beyond the immediate splice footprint.
        let gate = crate::validation::verify_sphere_effective_strict(
            effective_points,
            &geometry.vertices,
            LiveCellLayout::new(&cells, &cell_indices),
        );
        if debug {
            eprintln!(
                "local rebuild commit: into_flat {:?}, gate {:?} ({} verts, {} cells, gate {})",
                flat_elapsed,
                t_gate.elapsed(),
                geometry.vertices.len(),
                cells.len(),
                if gate.is_ok() { "accepted" } else { "rejected" },
            );
            if let Err(err) = &gate {
                eprintln!("  local rebuild gate rejection: {err}");
            }
        }
        if gate.is_err() {
            geometry.vertices.truncate(base_vertex_count);
            return None;
        }

        geometry.cells = cells;
        geometry.cell_indices = cell_indices;
        Some(resolution_scan_cells)
    }
}

/// Reject defect signals that cannot be surfaced by the plain compute API.
///
/// Kept as one pure decision seam so fault-injection tests can pin the exact
/// production return policy independently of local rebuild mechanics.
fn check_plain_return_signals(
    local_rebuild: LocalRebuildOutcome,
    residual_unpaired: &[(u32, u32)],
    local_rebuild_seed_pairs: &[(u32, u32)],
) -> Result<(), crate::VoronoiError> {
    // A committed local rebuild has already passed whole-diagram strict validation,
    // so pre-rebuild signals no longer describe the returned geometry.
    if local_rebuild.accepted() {
        return Ok(());
    }
    if !residual_unpaired.is_empty() {
        return Err(edge_reconcile::residual_error(residual_unpaired));
    }
    if !local_rebuild_seed_pairs.is_empty() {
        return Err(edge_reconcile::reconciliation_rejection_error(
            local_rebuild_seed_pairs,
        ));
    }
    // A low-incidence (degree-1/2) defect can exist with every edge paired,
    // so it needs a signal independent of the edge-residual checks above.
    if local_rebuild.low_incidence_defect {
        return Err(crate::VoronoiError::ComputationFailed(
            "post-assembly local rebuild could not resolve a residual low-incidence \
             (degree-1/2) vertex defect — output is not a valid subdivision. \
             Use compute_with_report to inspect, or report this input."
                .to_string(),
        ));
    }
    if local_rebuild.euler_defect {
        return Err(crate::VoronoiError::ComputationFailed(
            "post-assembly topology summary failed the spherical Euler check; \
             output is not a single valid spherical subdivision. Use \
             compute_with_report to inspect, or report this input."
                .to_string(),
        ));
    }
    Ok(())
}

/// Try the configured local rebuild and commit it only if whole-diagram strict
/// validation succeeds. Reports both the public outcome and the exact local
/// footprint whose final cycles changed on an accepted splice.
#[derive(Clone, Copy)]
struct LocalRebuildDiagnostics {
    debug: bool,
}

impl LocalRebuildDiagnostics {
    /// Snapshot internal diagnostics once per actual rebuild attempt. The
    /// caller deliberately constructs this only after the trigger check, so
    /// disabled and clean computations perform no diagnostic environment read.
    fn read_from_env() -> Self {
        Self {
            debug: std::env::var("VORONOI_MESH_LOCAL_REBUILD_DEBUG").is_ok(),
        }
    }
}

#[allow(clippy::too_many_arguments)] // cohesive rebuild-entry state; splitting would obscure it
fn maybe_rebuild_effective(
    effective_points: &[Vec3],
    grid: &CubeMapGrid,
    geometry: &mut EffectiveGeometry,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    residual_unpaired: &[(u32, u32)],
    local_rebuild_seed_pairs: &[(u32, u32)],
    merge_affected_cells: &[u32],
    topology: TopologySummary,
    low_incidence_scan_time: std::time::Duration,
    local_rebuild_mode: LocalRebuildMode,
) -> LocalRebuildResult {
    let has_low_incidence = topology.low_incidence;
    let euler_defect = !topology.has_sphere_euler(geometry.cells.len());
    let local_rebuild_enabled = !matches!(local_rebuild_mode, LocalRebuildMode::Disabled);
    if !local_rebuild_enabled {
        return LocalRebuildResult::unchanged(LocalRebuildOutcome::new(
            LocalRebuildStatus::Disabled,
            has_low_incidence,
            euler_defect,
        ));
    }

    let mut defect_pairs: Vec<(u32, u32)> = residual_unpaired
        .iter()
        .chain(local_rebuild_seed_pairs)
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    defect_pairs.sort_unstable();
    defect_pairs.dedup();
    if defect_pairs.is_empty() && !has_low_incidence {
        return LocalRebuildResult::unchanged(LocalRebuildOutcome::new(
            LocalRebuildStatus::NotTriggered,
            false,
            euler_defect,
        ));
    }
    let diagnostics = LocalRebuildDiagnostics::read_from_env();
    if diagnostics.debug {
        eprintln!(
            "local rebuild trigger: low-incidence scan {:?} (defect_pairs={}, unpaired={}, no_chain={}, low_incidence={})",
            low_incidence_scan_time,
            defect_pairs.len(),
            residual_unpaired.len(),
            local_rebuild_seed_pairs.len(),
            has_low_incidence,
        );
    }
    let outcome = |status| LocalRebuildOutcome::new(status, has_low_incidence, euler_defect);

    // Local-neighbor gathering uses an O(local) shell frontier per seed. Reuse
    // the occupancy-tuned construction grid (`compact_welded` keeps it
    // bit-equivalent to a fresh effective-point build) rather than rebuilding
    // or scanning all generators. Local rebuild needs the unrestricted frontier,
    // not the directed construction subset. See
    // docs/performance.md#source-pinned-performance-decisions.
    let mut rebuild_scratch = grid.make_scratch();

    let base_layout = LiveCellLayout::new(&geometry.cells, &geometry.cell_indices);
    let mut work = local_rebuild::WorkingDiagram::from_reconciled(
        &geometry.vertices,
        vertex_keys,
        base_layout,
    );
    let stats = local_rebuild::rebuild_with_local_hull(
        effective_points,
        grid,
        &mut rebuild_scratch,
        &mut work,
        &defect_pairs,
        merge_affected_cells,
        LOCAL_REBUILD_GATHER_K,
        LOCAL_REBUILD_MAX_ROUNDS,
        diagnostics.debug,
    );
    // No splices means the local rebuild did not modify `work` (a `splice_generator`
    // call is the only mutation, tracked 1:1 by `spliced_generators`). Skip the
    // flatten + full-diagram clone + validation of an unchanged diagram. See
    // docs/performance.md#source-pinned-performance-decisions.
    if stats.spliced_generators == 0 {
        return LocalRebuildResult::unchanged(outcome(LocalRebuildStatus::Rejected));
    }

    // Materialize the overlay into one owned transaction before mutably
    // borrowing the geometry it was built over. The base vertex array is
    // extended in place — and truncated back on rejection — so an accepted
    // local rebuild never copies the base positions.
    let t_flat = std::time::Instant::now();
    let candidate = LocalRebuildCandidate::from_work(work);
    let Some(resolution_scan_cells) =
        candidate.try_commit(effective_points, geometry, diagnostics.debug, t_flat)
    else {
        return LocalRebuildResult::unchanged(outcome(LocalRebuildStatus::Rejected));
    };

    LocalRebuildResult {
        outcome: outcome(LocalRebuildStatus::Accepted),
        resolution_scan_cells,
    }
}

fn canonicalize_cell_cycle_starts(cells: &[VoronoiCell], cell_indices: &mut [u32]) {
    for cell in cells {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        let span = &mut cell_indices[start..end];
        if let Some((offset, _)) = span.iter().enumerate().min_by_key(|&(_, vertex)| vertex) {
            span.rotate_left(offset);
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

/// Preprocess (weld) and build the query grid in one step.
///
/// The grid is built on the raw points and doubles as the weld detector
/// (`CubeMapGrid::collect_weld_pairs`); on welds the grid is compacted in
/// place to the effective points instead of being rebuilt, so the zero-weld
/// common case pays only the detection scan and the weld case pays linear
/// sweeps. The standalone quantized-key detector remains only for
/// `MergeWithin` radii too large for grid adjacency. The resolution policy
/// sees the raw count; welds are far too few to shift it.
/// Canonicalize input points once at entry — f64-normalize and
/// round back to f32 — so every consumer (grid, weld, charts, certificates,
/// and local rebuild) sees identical bits per generator rather than
/// independently renormalizing. Out-of-band lengths (contract-violating
/// inputs) are left untouched to fail downstream instead of being turned into
/// NaNs here.
fn canonicalize_and_find_first_non_finite(points: &mut [Vec3]) -> Option<usize> {
    fn canonicalize_chunk(chunk: &mut [Vec3]) -> Option<usize> {
        let mut first_bad = None;
        for (i, p) in chunk.iter_mut().enumerate() {
            if !p.is_finite() {
                first_bad.get_or_insert(i);
                continue;
            }
            let v = glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64);
            let len_sq = v.length_squared();
            if (0.25..=4.0).contains(&len_sq) {
                *p = crate::types::canonical_vec3_from_dvec3(v);
            }
            #[cfg(feature = "profiling")]
            crate::point_audit::record_vec3_from_dvec3(
                crate::point_audit::PointProducer::CanonicalGenerator,
                *p,
                v,
            );
        }
        first_bad
    }
    // Parallel chunks keep this required normalization pass off the default
    // build's serial critical path. See
    // docs/performance.md#source-pinned-performance-decisions.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const CHUNK: usize = 1 << 16;
        points
            .par_chunks_mut(CHUNK)
            .enumerate()
            .filter_map(|(chunk_idx, chunk)| {
                canonicalize_chunk(chunk).map(|i| chunk_idx * CHUNK + i)
            })
            .min()
    }
    #[cfg(not(feature = "parallel"))]
    canonicalize_chunk(points)
}

/// Reject non-finite generators while canonicalizing valid ones in the same
/// traversal. Invalid points are left untouched so the public error retains
/// the exact original component formatting; the minimum global index makes
/// the parallel result identical to the serial first-invalid result.
fn validate_and_canonicalize_unit_points(points: &mut [Vec3]) -> Result<(), crate::VoronoiError> {
    match canonicalize_and_find_first_non_finite(points) {
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

fn canonicalize_unit_points(points: &mut [Vec3]) {
    let _ = canonicalize_and_find_first_non_finite(points);
}

struct PreparedPointsAndGrid {
    effective_input: EffectiveInput,
    report: PreprocessReport,
    grid: CubeMapGrid,
    occupancy_rebuilt: bool,
}

fn prepare_points_and_grid(
    points: &[Vec3],
    preprocess_mode: PreprocessMode,
    workspace: Option<&super::driver::BuildWorkspace>,
    tb: &mut TimingBuilder,
) -> Result<PreparedPointsAndGrid, crate::VoronoiError> {
    let threshold = match preprocess_mode {
        PreprocessMode::Disabled => None,
        PreprocessMode::Weld => Some(crate::tolerances::weld_radius()),
        PreprocessMode::MergeWithin(threshold) => Some(threshold),
    };

    let (mut grid, mut dense_index_eligible) =
        build_query_grid(points, tb, threshold.is_some(), workspace);

    let t = Timer::start();
    let mut effective_input = EffectiveInput::Identity;
    if let Some(threshold) = threshold {
        if threshold <= grid.max_grid_weld_threshold() {
            let pairs = grid
                .collect_weld_pairs_and_finalize_slot_points(threshold)
                .map_err(|coincident_pairs| crate::VoronoiError::DegenerateInput {
                    coincident_pairs,
                    message: format!(
                        "weld detection exceeded the retained-pair budget of {}; reduce the merge threshold or deduplicate the input",
                        crate::cube_grid::MAX_RETAINED_WELD_PAIRS
                    ),
                })?;
            tb.set_weld_pair_stats(pairs.len(), pairs.capacity());
            if !pairs.is_empty() {
                let (result, kept) = super::preprocess::merge_result_from_pairs(points, &pairs);
                grid.compact_welded(
                    &kept,
                    &result.original_to_effective,
                    result.effective_points.len(),
                );
                effective_input = EffectiveInput::Merged(result);
            }
        } else {
            // Radius too large for grid adjacency (large `MergeWithin`):
            // standalone detector, then rebuild the grid on the survivors.
            let result = try_merge_close_points(points, threshold).map_err(|coincident_pairs| {
                crate::VoronoiError::DegenerateInput {
                    coincident_pairs,
                    message: format!(
                        "standalone weld detection exceeded the retained-pair budget of {}; reduce the merge threshold or deduplicate the input",
                        crate::cube_grid::MAX_RETAINED_WELD_PAIRS
                    ),
                }
            })?;
            if result.num_merged > 0 {
                (grid, dense_index_eligible) =
                    build_query_grid(&result.effective_points, tb, true, workspace);
                effective_input = EffectiveInput::Merged(result);
            }
            grid.finalize_slot_points();
        }
    }
    tb.set_preprocess(t.elapsed());

    // Every grid built above is provisional: occupancy feedback may replace
    // it, and preprocessing may compact or rebuild it. Materialize the
    // optional side index once, on the retained slot/cell layout, and only in
    // the deep-concentration regime where the packed band path is enabled.
    if dense_index_eligible {
        let t_dense = Timer::start();
        grid.build_dense_index();
        tb.add_knn_build(t_dense.elapsed());
    }

    let report = PreprocessReport {
        requested_mode: preprocess_mode,
        threshold_used: threshold,
        original_points: points.len(),
        effective_points: effective_input.effective_len(points.len()),
        num_merged: effective_input.num_merged(),
    };
    Ok(PreparedPointsAndGrid {
        effective_input,
        report,
        grid,
        occupancy_rebuilt: dense_index_eligible,
    })
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
    defer_point_views: bool,
    workspace: Option<&super::driver::BuildWorkspace>,
) -> (crate::cube_grid::CubeMapGrid, bool) {
    let t = Timer::start();
    let n = effective_points.len();
    #[cfg(feature = "timing")]
    let mut grid_build_timings = CubeMapGridBuildTimings::default();

    let build = |res: usize, #[cfg(feature = "timing")] timings: &mut CubeMapGridBuildTimings| {
        let cached_topology = workspace.and_then(|workspace| workspace.grid_topology(res));
        #[cfg(feature = "timing")]
        let grid = CubeMapGrid::new_deferred_with_cached_topology_and_build_timings(
            effective_points,
            res,
            defer_point_views,
            cached_topology,
            timings,
        );
        #[cfg(not(feature = "timing"))]
        let grid = CubeMapGrid::new_deferred_with_cached_topology(
            effective_points,
            res,
            defer_point_views,
            cached_topology,
        );
        if let Some(workspace) = workspace {
            workspace.retain_grid_topology(res, grid.topology_arc());
        }
        grid
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

    // Gate the dense-cell band-prune on a rebuild having fired. The caller
    // materializes the side index only after preprocessing selects this grid's
    // final slot/cell layout. The band only
    // wins on deep-certificate, un-splittable concentration (cap-like), which
    // is exactly the regime that triggers the occupancy rebuild and survives
    // it (a cell still over the dense threshold). Moderate clusters that never
    // trip the rebuild close fast in the packed path, so the band and takeover
    // are disabled there. The gate is scale-invariant, unlike a fixed occupancy
    // threshold. See docs/performance.md#source-pinned-performance-decisions.
    tb.set_knn_build(t.elapsed());
    tb.set_grid_stats(res, max_occupancy as u64, rebuilt);
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(grid_build_timings);
    (grid, rebuilt)
}

#[derive(Clone, Copy)]
struct CellConstructionPolicy {
    positive_chord_threshold: Option<f32>,
    occupancy_rebuilt: bool,
}

fn construct_cell_shards(
    effective_points: &[Vec3],
    grid: &CubeMapGrid,
    point_cell_storage: Vec<u32>,
    merge_result: Option<&MergeResult>,
    policy: CellConstructionPolicy,
    workspace: Option<&super::driver::BuildWorkspace>,
    tb: &mut TimingBuilder,
) -> Result<live_dedup::ShardedCellsData, crate::VoronoiError> {
    let t = Timer::start();
    let sharded = super::driver::build_cells_sharded_live_dedup(
        effective_points,
        grid,
        point_cell_storage,
        policy.positive_chord_threshold,
        policy.occupancy_rebuilt,
        workspace,
    )
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

fn reconcile_edges(
    geometry: &mut EffectiveGeometry,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    edge_mismatches: &[live_dedup::EdgeMismatch],
    tb: &mut TimingBuilder,
) -> Result<edge_reconcile::ReconcileResult, crate::VoronoiError> {
    // Snapshot all reconciliation diagnostics/oracles once, but only after a
    // mismatch exists. `ComputeReport` already records that a zero-record case
    // was clean, so that path remains free of environment lookups.
    let reconcile_options = if edge_mismatches.is_empty() {
        edge_reconcile::ReconcileOptions::default()
    } else {
        let options = edge_reconcile::ReconcileOptions::read_from_env();
        if options.emit_telemetry() {
            edge_reconcile::emit_primary_reconcile_telemetry(
                edge_mismatches,
                geometry.vertices.as_slice(),
                &geometry.cells,
                &geometry.cell_indices,
                edge_reconcile::VertexKeys::Sharded(vertex_keys),
                crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
                options,
            );
        }
        options
    };

    let reconciliation_edge_storage: Vec<live_dedup::EdgeRecord> = edge_mismatches
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();

    let t = Timer::start();
    // The sphere has no boundary: every interior edge must pair.
    let reconcile_result = edge_reconcile::reconcile_edge_mismatches(
        &reconciliation_edge_storage,
        geometry.vertices.as_slice(),
        &mut geometry.cells,
        &mut geometry.cell_indices,
        edge_reconcile::VertexKeys::Sharded(vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        reconcile_options,
    )?;
    // The simple cross-bin stitch above is the only local rebuild pass: any surviving
    // unpaired interior edge is surfaced as a residual error by the caller
    // (valid-or-error contract — see docs/correctness.md; the dropped
    // post-hoc Tier-2 local rebuild investigation lives in git history).
    tb.set_edge_reconcile(
        t.elapsed(),
        reconcile_result.merge_safety_scan_cells,
        reconcile_result.merge_safety_global_fallbacks,
    );
    Ok(reconcile_result)
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
mod tests;
