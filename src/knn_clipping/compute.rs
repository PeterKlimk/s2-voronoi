//! Compute entry points for the kNN + clipping Voronoi backend.

use glam::{DVec3, Vec3};

use super::edge_reconcile;
use super::escalate;
use super::live_dedup;
use super::output_resolution;
use super::timing::{Timer, TimingBuilder};
use super::{
    cell_build::{CellBuildError, CellFailure},
    try_merge_close_points, MergeResult, TerminationConfig,
};
use crate::cube_grid::CubeMapGrid;
#[cfg(feature = "timing")]
use crate::cube_grid::CubeMapGridBuildTimings;
use crate::diagram::VoronoiCell;
use crate::{
    CellKillingPolicy, ComputeOutput, ComputeReport, DegenerateMode, DegenerateReport,
    PreprocessMode, PreprocessReport, RepairMode, VoronoiConfig,
};

/// Per-seed neighbor count for the repair's grid kNN gather (the 2-ring gather
/// collects each seed's `k + 1` nearest via the shell frontier).
const ESCALATE_GATHER_K: usize = 32;
/// Grow-until-clean round cap per defect component.
const ESCALATE_MAX_ROUNDS: usize = 12;

/// Everything the shared pipeline produces before the plain and report paths
/// diverge: canonicalized inputs, the reconciled effective arrays, and the
/// repair outcome. `TimingBuilder` rides along so the caller's final remap
/// lands in the same timing report.
struct PipelineState {
    points: Vec<Vec3>,
    effective_points: Option<Vec<Vec3>>,
    merge_result: Option<MergeResult>,
    preprocess_report: PreprocessReport,
    vertices: Vec<Vec3>,
    eff_cells: Vec<VoronoiCell>,
    eff_cell_indices: Vec<u32>,
    unresolved_edges: Vec<live_dedup::UnresolvedEdgeMismatch>,
    post_repair_unpaired: Vec<(u32, u32)>,
    reconciliation_escalations: Vec<(u32, u32)>,
    repair: RepairOutcome,
    output_resolution: crate::OutputResolutionReport,
    cell_killing_generators: Vec<usize>,
    tb: TimingBuilder,
}

impl PipelineState {
    fn effective_points_ref(&self) -> &[Vec3] {
        match &self.effective_points {
            Some(v) => v.as_slice(),
            None => self.points.as_slice(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolutionDiscoveryDecision {
    certified_hint: bool,
    drift_fallback: bool,
}

fn resolution_discovery_decision(resolution_drift_exceeded: bool) -> ResolutionDiscoveryDecision {
    ResolutionDiscoveryDecision {
        certified_hint: !resolution_drift_exceeded,
        drift_fallback: resolution_drift_exceeded,
    }
}

fn canonicalize_pipeline_exact_zero_edges(
    vertices: &[Vec3],
    vertex_keys: &live_dedup::ShardedVertexKeys,
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    hinted_candidates: Vec<(u32, u32)>,
    mutation_scan_cells: &[u32],
    decision: ResolutionDiscoveryDecision,
) -> Result<output_resolution::CanonicalizationOutcome, crate::VoronoiError> {
    let (exact_zero_candidates, localized_candidate_cells) = if decision.certified_hint {
        // Construction hints name pre-reconciliation edges. Re-scan their
        // degree-local incident cells in the terminal diagram so a repair
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

            // A repaired/minted endpoint may not exist in the assembly key
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

/// The shared front of both compute paths: validate → canonicalize → grid →
/// per-cell shards → assemble → reconcile → repair. The plain path fails loud
/// on residuals; the report path surfaces them in `ComputeReport`.
fn run_core_pipeline(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
    repair_mode: RepairMode,
) -> Result<PipelineState, crate::VoronoiError> {
    validate_generator_capacity(points.len())?;
    let mut points = points;
    validate_and_canonicalize_unit_points(&mut points)?;
    validate_preprocess_mode(preprocess_mode)?;
    let mut tb = TimingBuilder::new();

    let (effective_points, merge_result, preprocess_report, grid) =
        prepare_points_and_grid(&points, preprocess_mode, &mut tb)?;

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
        exact_zero_edge_candidates,
        exact_zero_edge_hint_cells,
        resolution_drift_exceeded,
        dedup_sub: _,
    } = assembled;
    let (mut eff_cells, mut eff_cell_indices, reconcile_result) = reconcile_edges(
        &mut vertices,
        &vertex_keys,
        &unresolved_edges,
        cells,
        cell_indices,
        &mut tb,
    )?;
    let edge_reconcile::ReconcileResult {
        residual_pairs: post_repair_unpaired,
        escalation_pairs: reconciliation_escalations,
        merge_affected_cells,
        resolution_scan_cells: reconcile_resolution_scan_cells,
    } = reconcile_result;
    // This is part of the plain-return safety gate, not merely a repair
    // trigger. Compute it even when repair is disabled so that mode cannot
    // suppress a known-invalid low-incidence output.
    let t_low_incidence = std::time::Instant::now();
    let topology = summarize_topology(vertices.len(), &eff_cells, &eff_cell_indices);
    let low_incidence_scan_time = t_low_incidence.elapsed();
    let RepairResult {
        outcome: repair,
        resolution_scan_cells: repair_resolution_scan_cells,
    } = maybe_repair_effective(
        effective_points_ref,
        &grid,
        &mut vertices,
        &vertex_keys,
        &mut eff_cells,
        &mut eff_cell_indices,
        &post_repair_unpaired,
        &reconciliation_escalations,
        &merge_affected_cells,
        topology,
        low_incidence_scan_time,
        repair_mode,
    );
    let reconcile_resolution_scan_cell_count = reconcile_resolution_scan_cells.len();
    let repair_resolution_scan_cell_count = repair_resolution_scan_cells.len();
    let mut mutation_scan_cells = reconcile_resolution_scan_cells;
    mutation_scan_cells.extend(repair_resolution_scan_cells);
    mutation_scan_cells.sort_unstable();
    mutation_scan_cells.dedup();

    let resolution_decision = resolution_discovery_decision(resolution_drift_exceeded);
    let hinted_candidate_count = exact_zero_edge_candidates.len();
    let resolution_outcome = canonicalize_pipeline_exact_zero_edges(
        &vertices,
        &vertex_keys,
        &mut eff_cells,
        &mut eff_cell_indices,
        exact_zero_edge_candidates,
        &mutation_scan_cells,
        resolution_decision,
    )?;
    tb.set_output_resolution_discovery(
        resolution_decision.certified_hint,
        resolution_decision.drift_fallback,
        reconcile_resolution_scan_cell_count,
        repair_resolution_scan_cell_count,
        exact_zero_edge_hint_cells,
        hinted_candidate_count,
        resolution_outcome.report.exact_zero_edges_detected,
    );
    Ok(PipelineState {
        points,
        effective_points,
        merge_result,
        preprocess_report,
        vertices,
        eff_cells,
        eff_cell_indices,
        unresolved_edges,
        post_repair_unpaired,
        reconciliation_escalations,
        repair,
        output_resolution: resolution_outcome.report,
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

    let generator_indices = if let Some(merge) = &state.merge_result {
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
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
    repair_mode: RepairMode,
    cell_killing_policy: CellKillingPolicy,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    let mut state = run_core_pipeline(points, termination, preprocess_mode, repair_mode)?;
    check_plain_return_signals(
        state.repair,
        &state.post_repair_unpaired,
        &state.reconciliation_escalations,
    )?;
    enforce_cell_killing_policy(&state, cell_killing_policy)?;

    let t = Timer::start();
    let (cells, cell_indices, weld_map) = remap_cells_to_original_indices(
        &state.points,
        state.merge_result.as_ref(),
        state.eff_cells,
        state.eff_cell_indices,
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(
        state.points,
        state.vertices,
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

pub fn compute_voronoi_knn_clipping_with_config_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<crate::SphericalVoronoi, crate::VoronoiError> {
    let termination = TerminationConfig::default();
    with_coplanar_perturb_retry(points, config.degenerate_mode, |points, _| {
        compute_voronoi_knn_clipping_owned_core(
            points,
            termination,
            config.preprocess_mode,
            config.repair_mode,
            config.cell_killing_policy,
        )
    })
}

pub fn compute_voronoi_knn_clipping_with_report_owned(
    points: Vec<Vec3>,
    config: &VoronoiConfig,
) -> Result<ComputeOutput, crate::VoronoiError> {
    let termination = TerminationConfig::default();
    with_coplanar_perturb_retry(
        points,
        config.degenerate_mode,
        |points, perturbation_applied| {
            compute_voronoi_knn_clipping_report_core(
                points,
                termination,
                config.preprocess_mode,
                config.repair_mode,
                config.cell_killing_policy,
                DegenerateReport {
                    requested_mode: config.degenerate_mode,
                    perturbation_applied,
                },
            )
        },
    )
}

fn compute_voronoi_knn_clipping_report_core(
    points: Vec<Vec3>,
    termination: TerminationConfig,
    preprocess_mode: PreprocessMode,
    repair_mode: RepairMode,
    cell_killing_policy: CellKillingPolicy,
    degenerate_report: DegenerateReport,
) -> Result<ComputeOutput, crate::VoronoiError> {
    let mut state = run_core_pipeline(points, termination, preprocess_mode, repair_mode)?;
    enforce_cell_killing_policy(&state, cell_killing_policy)?;
    let repair_accepted = state.repair.accepted;
    // Surface output-invariant residuals alongside the detection records. If
    // local repair was accepted, the returned diagram is strictly valid and
    // these residuals no longer survive to output.
    let pre_repair_edge_mismatches: Vec<(u32, u32, live_dedup::UnresolvedEdgeOrigin)> = state
        .unresolved_edges
        .iter()
        .map(|m| {
            let (a, b) = edge_reconcile::unpack_edge(m.key.as_u64());
            (a.min(b), a.max(b), m.origin)
        })
        .collect();
    let post_repair_unpaired_edges: Vec<(u32, u32)> = if repair_accepted {
        Vec::new()
    } else {
        state
            .post_repair_unpaired
            .iter()
            .map(|&(a, b)| (a.min(b), a.max(b)))
            .collect()
    };
    let post_repair_escalation_pairs = if repair_accepted {
        Vec::new()
    } else {
        state.reconciliation_escalations.clone()
    };
    if !repair_accepted {
        for &(a, b) in &state.post_repair_unpaired {
            state
                .unresolved_edges
                .push(live_dedup::UnresolvedEdgeMismatch {
                    key: live_dedup::pack_edge(a, b),
                    origin: live_dedup::UnresolvedEdgeOrigin::PostRepairUnpaired,
                });
        }
    }

    let effective_diagram = if state.merge_result.is_some() {
        Some(crate::SphericalVoronoi::from_raw_parts(
            state.effective_points_ref().to_vec(),
            state.vertices.clone(),
            state.eff_cells.clone(),
            state.eff_cell_indices.clone(),
            None,
        ))
    } else {
        None
    };
    let effective_validation = effective_diagram.as_ref().map(crate::validation::validate);

    let t = Timer::start();
    let (cells, cell_indices, weld_map) = remap_cells_to_original_indices(
        &state.points,
        state.merge_result.as_ref(),
        state.eff_cells,
        state.eff_cell_indices,
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(
        state.points,
        state.vertices,
        cells,
        cell_indices,
        weld_map,
    );
    let returned_validation = crate::validation::validate(&diagram);
    state.tb.set_assemble(t.elapsed());

    let timings = state.tb.finish();
    timings.report(diagram.num_cells());

    Ok(ComputeOutput {
        diagram,
        effective_diagram,
        report: ComputeReport {
            preprocess: state.preprocess_report,
            degenerate: degenerate_report,
            returned_validation,
            effective_validation,
            pre_repair_edge_mismatch_count: pre_repair_edge_mismatches.len(),
            repair: crate::RepairReport {
                attempted: state.repair.attempted,
                accepted: state.repair.accepted,
            },
            output_resolution: state.output_resolution,
            pre_repair_edge_mismatches,
            post_repair_unpaired_edges,
            post_repair_escalation_pairs,
            unresolved_edge_pairs: state
                .unresolved_edges
                .iter()
                .map(|m| {
                    let (a, b) = edge_reconcile::unpack_edge(m.key.as_u64());
                    (a.min(b), a.max(b), m.origin)
                })
                .collect(),
        },
    })
}

/// Cheap topology facts collected by the incidence pass already required for
/// the repair trigger and plain-return safety gate.
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

/// Summarize referenced vertices and live half-edges, including whether any
/// referenced vertex has degree 1 or 2 (a real defect the repair should
/// examine).
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
/// This scan runs on EVERY build as a plain-return safety signal (and, when
/// enabled, a repair trigger). It cannot piggyback on reconcile, which
/// early-returns when no edge is unresolved. Multi-threaded builds use exact
/// shared atomic counters, read only after the chunk-parallel scan; a one-thread
/// Rayon pool uses the same plain-counter path as a build without `parallel`.
fn summarize_topology_scalar(
    vertex_count: usize,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> TopologySummary {
    let mut cnt = vec![0u32; vertex_count];
    let mut live_half_edges = 0usize;
    for cell in cells {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        live_half_edges += cell.vertex_count();
        for &v in &cell_indices[start..end] {
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
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
        let threads = rayon::current_num_threads().max(1);
        if threads == 1 {
            return summarize_topology_scalar(vertex_count, cells, cell_indices);
        }
        // (u32: cannot saturate — total increments are bounded by
        // `cell_indices.len()`.)
        let cnt: Vec<AtomicU32> = (0..vertex_count).map(|_| AtomicU32::new(0)).collect();
        let chunk = cells.len().div_ceil(threads * 4).max(1024);
        let live_half_edges = cells
            .par_chunks(chunk)
            .map(|cells_chunk| {
                let mut half_edges = 0usize;
                for cell in cells_chunk {
                    let start = cell.vertex_start();
                    let end = start + cell.vertex_count();
                    half_edges += cell.vertex_count();
                    for &v in &cell_indices[start..end] {
                        cnt[v as usize].fetch_add(1, Relaxed);
                    }
                }
                half_edges
            })
            .sum();
        let (used_vertices, low_incidence) = cnt
            .par_iter()
            .map(|c| {
                let count = c.load(Relaxed);
                (usize::from(count != 0), count == 1 || count == 2)
            })
            .reduce(|| (0, false), |a, b| (a.0 + b.0, a.1 || b.1));
        TopologySummary {
            used_vertices,
            live_half_edges,
            low_incidence,
        }
    }
    #[cfg(not(feature = "parallel"))]
    summarize_topology_scalar(vertex_count, cells, cell_indices)
}

/// Outcome of the repair attempt, for the caller's fail-loud decision.
#[derive(Clone, Copy)]
struct RepairOutcome {
    /// A repair pass ran: defects were detected and the configured mode is
    /// enabled. False on clean builds and when repair is disabled.
    attempted: bool,
    /// The repaired effective diagram passed the strict gate and was committed.
    accepted: bool,
    /// Detection found a low-incidence (degree-1/2) vertex defect. Such a
    /// vertex is strictly-invalid output even when every edge pairs (it fails
    /// `verify_sphere_effective_strict`'s "low-incidence vertex" check), so
    /// when the repair was not accepted the plain path must fail loud on it —
    /// there is no unpaired-edge residual to trip the existing guard.
    low_incidence_defect: bool,
    /// The cheap `V - H/2 + F` check failed (or `H` was odd). This catches
    /// global topology defects at no additional traversal once exact edge
    /// agreement is supplied by construction.
    euler_defect: bool,
}

impl RepairOutcome {
    const fn not_attempted(low_incidence_defect: bool, euler_defect: bool) -> RepairOutcome {
        RepairOutcome {
            attempted: false,
            accepted: false,
            low_incidence_defect,
            euler_defect,
        }
    }
}

struct RepairResult {
    outcome: RepairOutcome,
    /// Cells whose final cycles were replaced by an accepted Local3d splice.
    /// Newly minted vertices are referenced only from these cells.
    resolution_scan_cells: Vec<u32>,
}

impl RepairResult {
    fn unchanged(outcome: RepairOutcome) -> Self {
        Self {
            outcome,
            resolution_scan_cells: Vec::new(),
        }
    }
}

/// Reject defect signals that cannot be surfaced by the plain compute API.
///
/// Kept as one pure decision seam so fault-injection tests can pin the exact
/// production return policy independently of repair mechanics.
fn check_plain_return_signals(
    repair: RepairOutcome,
    post_repair_unpaired: &[(u32, u32)],
    reconciliation_escalations: &[(u32, u32)],
) -> Result<(), crate::VoronoiError> {
    // A committed repair has already passed whole-diagram strict validation,
    // so pre-repair signals no longer describe the returned geometry.
    if repair.accepted {
        return Ok(());
    }
    if !post_repair_unpaired.is_empty() {
        return Err(edge_reconcile::residual_error(post_repair_unpaired));
    }
    if !reconciliation_escalations.is_empty() {
        return Err(edge_reconcile::escalation_error(reconciliation_escalations));
    }
    // A low-incidence (degree-1/2) defect can exist with every edge paired,
    // so it needs a signal independent of the edge-residual checks above.
    if repair.low_incidence_defect {
        return Err(crate::VoronoiError::ComputationFailed(
            "post-assembly repair could not resolve a residual low-incidence \
             (degree-1/2) vertex defect — output is not a valid subdivision. \
             Use compute_with_report to inspect, or report this input."
                .to_string(),
        ));
    }
    if repair.euler_defect {
        return Err(crate::VoronoiError::ComputationFailed(
            "post-assembly topology summary failed the spherical Euler check; \
             output is not a single valid spherical subdivision. Use \
             compute_with_report to inspect, or report this input."
                .to_string(),
        ));
    }
    Ok(())
}

/// Try the configured local repair and commit it only if whole-diagram strict
/// validation succeeds. Reports both the public repair outcome and the exact
/// local footprint whose final cycles changed on an accepted splice.
#[allow(clippy::too_many_arguments)] // cohesive repair-entry state; splitting would obscure it
fn maybe_repair_effective(
    effective_points: &[Vec3],
    grid: &CubeMapGrid,
    vertices: &mut Vec<Vec3>,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    eff_cells: &mut Vec<VoronoiCell>,
    eff_cell_indices: &mut Vec<u32>,
    post_repair_unpaired: &[(u32, u32)],
    reconciliation_escalations: &[(u32, u32)],
    merge_affected_cells: &[u32],
    topology: TopologySummary,
    low_incidence_scan_time: std::time::Duration,
    repair_mode: RepairMode,
) -> RepairResult {
    let has_low_incidence = topology.low_incidence;
    let euler_defect = !topology.has_sphere_euler(eff_cells.len());
    // A0 probes need the fast assembled state, not the repaired one.
    #[cfg(feature = "escalate_probe")]
    if std::env::var("VORONOI_MESH_ESCALATE_PROBE_A0").is_ok() {
        escalate::stash_a0_fast(effective_points, vertex_keys, eff_cells, eff_cell_indices);
        return RepairResult::unchanged(RepairOutcome::not_attempted(
            has_low_incidence,
            euler_defect,
        ));
    }

    let repair_enabled =
        !matches!(repair_mode, RepairMode::Disabled) || escalate::escalation_enabled();
    if !repair_enabled {
        return RepairResult::unchanged(RepairOutcome::not_attempted(
            has_low_incidence,
            euler_defect,
        ));
    }

    let mut defect_pairs: Vec<(u32, u32)> = post_repair_unpaired
        .iter()
        .chain(reconciliation_escalations)
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    defect_pairs.sort_unstable();
    defect_pairs.dedup();
    if std::env::var("VORONOI_MESH_ESCALATE_DEBUG").is_ok() {
        eprintln!(
            "repair trigger: low-incidence scan {:?} (defect_pairs={}, unpaired={}, no_chain={}, low_incidence={})",
            low_incidence_scan_time,
            defect_pairs.len(),
            post_repair_unpaired.len(),
            reconciliation_escalations.len(),
            has_low_incidence,
        );
    }
    if defect_pairs.is_empty() && !has_low_incidence {
        return RepairResult::unchanged(RepairOutcome::not_attempted(false, euler_defect));
    }
    let outcome = |accepted: bool| RepairOutcome {
        attempted: true,
        accepted,
        low_incidence_defect: has_low_incidence,
        euler_defect,
    };

    // Local-neighbor gather index for the repair: O(local) shell-frontier kNN per
    // seed instead of the old O(n) brute force (a closure of thousands on a
    // dense-defect input made the brute force minutes-long). Reuse the construction
    // `grid` (occupancy-tuned; `compact_welded` keeps it bit-equivalent to a fresh
    // effective-point build) rather than rebuilding. Repair uses the grid's
    // unrestricted shell frontier because it wants every nearby generator, not
    // the directed construction subset.
    let mut repair_scratch = grid.make_scratch();

    let mut work = escalate::WorkingDiagram::from_assembled(
        vertices,
        vertex_keys,
        eff_cells,
        eff_cell_indices,
    );
    #[cfg(feature = "escalate_probe")]
    let stats = if std::env::var("VORONOI_MESH_ESCALATE_DELAUNATOR").is_ok() {
        escalate::repair_delaunator(
            effective_points,
            &mut work,
            &defect_pairs,
            merge_affected_cells,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    } else if matches!(repair_mode, RepairMode::LocalProjected) {
        escalate::repair_local_exact(
            effective_points,
            grid,
            &mut repair_scratch,
            &mut work,
            &defect_pairs,
            merge_affected_cells,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    } else {
        escalate::repair_local_hull(
            effective_points,
            grid,
            &mut repair_scratch,
            &mut work,
            &defect_pairs,
            merge_affected_cells,
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
            &mut work,
            &defect_pairs,
            merge_affected_cells,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    } else {
        escalate::repair_local_hull(
            effective_points,
            grid,
            &mut repair_scratch,
            &mut work,
            &defect_pairs,
            merge_affected_cells,
            ESCALATE_GATHER_K,
            ESCALATE_MAX_ROUNDS,
        )
    };
    // No splices means the repair did not modify `work` (a `splice_generator`
    // call is the only mutation, tracked 1:1 by `spliced_generators`). Skip the
    // flatten + full-diagram clone + validate of an unchanged diagram, which is
    // the dominant cost of a no-op repair (~12.6s of a 15s tail at 2.5M).
    if stats.spliced_generators == 0 {
        return RepairResult::unchanged(outcome(false));
    }

    // Materialize the overlay: minted vertex positions (vids past the base
    // length) plus the full cell arrays. The base vertex array is extended in
    // place — and truncated back on rejection — so an accepted repair never
    // copies the base positions.
    let t_flat = std::time::Instant::now();
    let resolution_scan_cells = work.overridden_cells();
    let (minted_vertices, new_cells, mut new_cell_indices) = work.into_flat();
    // The in-place and rebuild reconciliation oracles can present the same
    // cyclic boundary with different starting slots. Local3d preserves that
    // arbitrary rotation when it splices a neighborhood. Canonicalize only
    // this cold repaired output so semantically identical repair backends
    // remain byte-for-byte differential oracles; winding is unchanged.
    canonicalize_cell_cycle_starts(&new_cells, &mut new_cell_indices);
    let base_vertex_count = vertices.len();
    vertices.extend(minted_vertices);
    let flat_elapsed = t_flat.elapsed();
    let t_gate = std::time::Instant::now();

    // Whole-diagram never-worse gate: accept only if the repaired diagram is
    // strictly valid. Validate the effective arrays in place via
    // `verify_sphere_effective_strict` (same strict contract as `validate`,
    // pinned by the `effective_strict_matches_fast` differential test) rather
    // than cloning all of `effective_points`/vertices/cells/indices into a
    // temporary `SphericalVoronoi` — the clone was the dominant cost of a
    // committed repair. The validate itself stays whole-diagram (the repair's
    // blast radius is vertex-triple-identity-wide, so a local gate is unsound).
    let gate = crate::validation::verify_sphere_effective_strict(
        effective_points,
        vertices,
        &new_cells,
        &new_cell_indices,
    );
    if std::env::var("VORONOI_MESH_ESCALATE_DEBUG").is_ok() {
        eprintln!(
            "repair commit: into_flat {:?}, gate {:?} ({} verts, {} cells, gate {})",
            flat_elapsed,
            t_gate.elapsed(),
            vertices.len(),
            new_cells.len(),
            if gate.is_ok() { "accepted" } else { "rejected" },
        );
        if let Err(err) = &gate {
            eprintln!("  repair gate rejection: {err}");
        }
    }
    if gate.is_err() {
        vertices.truncate(base_vertex_count);
        return RepairResult::unchanged(outcome(false));
    }

    *eff_cells = new_cells;
    *eff_cell_indices = new_cell_indices;
    RepairResult {
        outcome: outcome(true),
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
/// and repair) sees identical bits per generator. The
/// per-builder f64 renormalization is gone; without this pass the pipeline
/// solved a ~1-ulp-perturbed, asymmetrically-treated point set. Out-of-band lengths
/// (contract-violating inputs) are left untouched and fail downstream as
/// before, rather than being turned into NaNs here.
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
                let n = v / len_sq.sqrt();
                *p = Vec3::new(n.x as f32, n.y as f32, n.z as f32);
            }
        }
        first_bad
    }
    // ~10ns/point scalar (f64 sqrt + div); parallel chunks so the default
    // build pays ~nothing. Measured ST cost: ~20ms at 2M (the bulk of stage
    // 0's +0.5-0.8% single-threaded total).
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

#[derive(Debug, Clone, Copy)]
struct CoplanarClass {
    normal: DVec3,
}

fn maybe_perturb_coplanar(points: &[Vec3], err: &crate::VoronoiError) -> Option<Vec<Vec3>> {
    if !matches!(
        err,
        crate::VoronoiError::UnsupportedGeometry { .. } | crate::VoronoiError::ComputationFailed(_)
    ) {
        return None;
    }
    let mut canonical = points.to_vec();
    canonicalize_unit_points(&mut canonical);
    let class = classify_exact_affine_circle(&canonical)
        .or_else(|| classify_near_great_circle(&canonical))?;
    Some(perturb_coplanar_points(&canonical, class.normal))
}

/// Certify affine coplanarity in the actual canonical f32 model. The seed
/// plane is selected with bounded linear sweeps for a stable perturbation
/// direction; `orient3d == 0` then decides coplanarity exactly for those
/// binary input coordinates. No tolerance can turn a merely near-coplanar
/// ordinary input into this class.
fn classify_exact_affine_circle(points: &[Vec3]) -> Option<CoplanarClass> {
    if points.len() < 4 {
        return None;
    }
    let ([a, b, c], normal) = stable_affine_plane(points)?;
    let a = robust_coord(dvec(points[a]));
    let b = robust_coord(dvec(points[b]));
    let c = robust_coord(dvec(points[c]));
    if points
        .iter()
        .all(|&p| robust::orient3d(a, b, c, robust_coord(dvec(p))) == 0.0)
    {
        Some(CoplanarClass { normal })
    } else {
        None
    }
}

/// Choose a well-spread affine plane seed in a fixed number of linear sweeps.
/// Returns `None` only when fewer than three distinct, non-collinear points are
/// available or the input is non-finite.
fn stable_affine_plane(points: &[Vec3]) -> Option<([usize; 3], DVec3)> {
    fn farthest_from(points: &[Vec3], pivot: usize) -> Option<usize> {
        let a = dvec(points[pivot]);
        let mut best = None;
        let mut best_distance2 = 0.0f64;
        for (i, &p) in points.iter().enumerate() {
            let distance2 = (dvec(p) - a).length_squared();
            if distance2.is_finite() && distance2 > best_distance2 {
                best_distance2 = distance2;
                best = Some(i);
            }
        }
        best
    }

    let mut a = 0usize;
    let mut b = farthest_from(points, a)?;
    a = farthest_from(points, b)?;
    b = farthest_from(points, a)?;

    let pa = dvec(points[a]);
    let ab = dvec(points[b]) - pa;
    let mut c = None;
    let mut best_cross = DVec3::ZERO;
    let mut best_area2 = 0.0f64;
    for (i, &p) in points.iter().enumerate() {
        let cross = ab.cross(dvec(p) - pa);
        let area2 = cross.length_squared();
        if area2.is_finite() && area2 > best_area2 {
            best_area2 = area2;
            best_cross = cross;
            c = Some(i);
        }
    }
    let c = c?;
    Some(([a, b, c], best_cross / best_area2.sqrt()))
}

#[inline]
fn robust_coord(p: DVec3) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

/// Compatibility classifier for nominal great-circle input whose canonical
/// f32 rounding prevents exact affine certification. Unlike the exact path,
/// this tolerance classifier requires full-circle coverage so an ordinary
/// large cell in a hemisphere cannot be misclassified as a degeneracy.
fn classify_near_great_circle(points: &[Vec3]) -> Option<CoplanarClass> {
    if points.len() < 4 {
        return None;
    }

    let normal = stable_rank2_normal(points)?;
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

    Some(CoplanarClass { normal })
}

/// Find a numerically stable candidate normal in a fixed number of linear
/// sweeps. The old implementation searched every pair for the largest cross
/// product, making *any* failed million-point build fall into an O(n²)
/// great-circle probe before it could return the original error.
///
/// This selection is deliberately conservative: failure to find a pair with
/// enough angular separation merely declines the perturbation retry. It cannot
/// create a false rank-2 classification because `classify_near_great_circle`
/// subsequently checks every point against the candidate plane and verifies
/// full-circle coverage. Re-pivoting at the farthest point handles ordered
/// two-arc inputs where no pair involving `points[0]` is sufficiently stable.
fn stable_rank2_normal(points: &[Vec3]) -> Option<DVec3> {
    const SWEEPS: usize = 3;
    const MIN_CROSS_LEN2: f64 = 0.25;

    let mut pivot = 0usize;
    let mut best_cross = DVec3::ZERO;
    let mut best_len2 = 0.0f64;
    for _ in 0..SWEEPS {
        let a = dvec(points[pivot]);
        let mut next_pivot = pivot;
        let mut sweep_best_len2 = 0.0f64;
        for (i, &b32) in points.iter().enumerate() {
            let cross = a.cross(dvec(b32));
            let len2 = cross.length_squared();
            if len2 > sweep_best_len2 {
                sweep_best_len2 = len2;
                next_pivot = i;
            }
            if len2 > best_len2 {
                best_len2 = len2;
                best_cross = cross;
            }
        }
        if next_pivot == pivot {
            break;
        }
        pivot = next_pivot;
    }
    if best_len2 < MIN_CROSS_LEN2 {
        return None;
    }
    Some(best_cross / best_len2.sqrt())
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

fn perturb_coplanar_points(points: &[Vec3], normal: DVec3) -> Vec<Vec3> {
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

type PreparedPointsAndGrid = (
    Option<Vec<Vec3>>,
    Option<MergeResult>,
    PreprocessReport,
    CubeMapGrid,
);

fn prepare_points_and_grid(
    points: &[Vec3],
    preprocess_mode: PreprocessMode,
    tb: &mut TimingBuilder,
) -> Result<PreparedPointsAndGrid, crate::VoronoiError> {
    let threshold = match preprocess_mode {
        PreprocessMode::Disabled => None,
        PreprocessMode::Weld => Some(crate::tolerances::weld_radius()),
        PreprocessMode::MergeWithin(threshold) => Some(threshold),
    };

    let (mut grid, mut dense_index_eligible) = build_query_grid(points, tb);

    let t = Timer::start();
    let mut effective_points = None;
    let mut merge_result = None;
    if let Some(threshold) = threshold {
        if threshold <= grid.max_grid_weld_threshold() {
            let pairs = grid.collect_weld_pairs(threshold).map_err(|coincident_pairs| {
                crate::VoronoiError::DegenerateInput {
                    coincident_pairs,
                    message: format!(
                        "weld detection exceeded the retained-pair budget of {}; reduce the merge threshold or deduplicate the input",
                        crate::cube_grid::MAX_RETAINED_WELD_PAIRS
                    ),
                }
            })?;
            tb.set_weld_pair_stats(pairs.len(), pairs.capacity());
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
            let mut result = try_merge_close_points(points, threshold).map_err(|coincident_pairs| {
                crate::VoronoiError::DegenerateInput {
                    coincident_pairs,
                    message: format!(
                        "standalone weld detection exceeded the retained-pair budget of {}; reduce the merge threshold or deduplicate the input",
                        crate::cube_grid::MAX_RETAINED_WELD_PAIRS
                    ),
                }
            })?;
            if result.num_merged > 0 {
                let pts = std::mem::take(&mut result.effective_points);
                (grid, dense_index_eligible) = build_query_grid(&pts, tb);
                effective_points = Some(pts);
                merge_result = Some(result);
            }
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
        effective_points: effective_points.as_ref().map_or(points.len(), |p| p.len()),
        num_merged: merge_result.as_ref().map_or(0, |m| m.num_merged),
    };
    Ok((effective_points, merge_result, report, grid))
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
) -> (crate::cube_grid::CubeMapGrid, bool) {
    let t = Timer::start();
    let n = effective_points.len();
    #[cfg(feature = "timing")]
    let mut grid_build_timings = CubeMapGridBuildTimings::default();

    let build = |res: usize, #[cfg(feature = "timing")] timings: &mut CubeMapGridBuildTimings| {
        #[cfg(feature = "timing")]
        {
            CubeMapGrid::new_deferred_dense_with_build_timings(effective_points, res, timings)
        }
        #[cfg(not(feature = "timing"))]
        {
            CubeMapGrid::new_deferred_dense(effective_points, res)
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

    // Gate the dense-cell band-prune on a rebuild having fired. The caller
    // materializes the side index only after preprocessing selects this grid's
    // final slot/cell layout. The band only
    // wins on deep-certificate, un-splittable concentration (cap-like), which
    // is exactly the regime that triggers the occupancy rebuild and survives
    // it (a cell still over the dense threshold). Moderate clusters that never
    // trip the rebuild close fast in the packed path, where the band + takeover
    // is a measured net loss (clustered 500k ~ -13%); disable it there. Scale-
    // invariant, unlike a fixed occupancy threshold (clustered occ grows with
    // n).
    tb.set_knn_build(t.elapsed());
    tb.set_grid_stats(res, max_occupancy as u64, rebuilt);
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(grid_build_timings.clone());
    (grid, rebuilt)
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

/// Reconciled cell arrays plus the reconcile outcome: the post-repair
/// output-invariant residuals (cell pairs whose shared edge stayed unpaired
/// after both repair passes) and the merge-rewritten cells the local repair's
/// residual scan must keep in scope.
type ReconciledWithResiduals = (Vec<VoronoiCell>, Vec<u32>, edge_reconcile::ReconcileResult);

fn reconcile_edges(
    vertices: &mut Vec<Vec3>,
    vertex_keys: &live_dedup::ShardedVertexKeys,
    unresolved_edges: &[live_dedup::UnresolvedEdgeMismatch],
    mut cells: Vec<VoronoiCell>,
    mut cell_indices: Vec<u32>,
    tb: &mut TimingBuilder,
) -> Result<ReconciledWithResiduals, crate::VoronoiError> {
    // Keep the clean production path free of even an environment lookup.
    // `ComputeReport` already records that a zero-record case was clean; the
    // detailed reconciliation telemetry is useful only on defect runs.
    if !unresolved_edges.is_empty() {
        edge_reconcile::emit_primary_reconcile_telemetry(
            unresolved_edges,
            vertices.as_slice(),
            &cells,
            &cell_indices,
            edge_reconcile::VertexKeys::Sharded(vertex_keys),
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        );
    }

    let repair_edges_storage: Vec<live_dedup::EdgeRecord> = unresolved_edges
        .iter()
        .map(|b| live_dedup::EdgeRecord { key: b.key })
        .collect();

    let t = Timer::start();
    // The sphere has no boundary: every interior edge must pair.
    let reconcile_result = edge_reconcile::reconcile_unresolved_edges(
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
    // (valid-or-error contract — see docs/correctness.md; the dropped
    // post-hoc Tier-2 repair investigation lives in git history).
    tb.set_edge_reconcile(t.elapsed());
    Ok((cells, cell_indices, reconcile_result))
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
        build_query_grid, canonicalize_pipeline_exact_zero_edges, cell_sum_sq_per_n,
        check_plain_return_signals, classify_exact_affine_circle, classify_near_great_circle,
        map_build_cells_error, map_cell_build_error, max_cell_occupancy,
        resolution_discovery_decision, run_core_pipeline, stable_rank2_normal, summarize_topology,
        validate_and_canonicalize_unit_points, validate_generator_capacity, RepairOutcome,
    };
    use crate::diagram::VoronoiCell;
    use crate::knn_clipping::cell_build::{CellBuildError, CellFailure};
    use crate::knn_clipping::live_dedup::{
        BuildCellsError, PackedLayoutCapacityError, ShardedVertexKeys,
    };
    use crate::{PreprocessMode, RepairMode, VoronoiError};
    use glam::Vec3;

    fn zero_edge_cube_fixture() -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>, ShardedVertexKeys) {
        let unit = |x: f32, y: f32, z: f32| Vec3::new(x, y, z).normalize();
        let mut vertices = vec![
            unit(-1.0, -1.0, -1.0),
            unit(1.0, -1.0, -1.0),
            unit(1.0, 1.0, -1.0),
            unit(-1.0, 1.0, -1.0),
            unit(-1.0, -1.0, 1.0),
            unit(1.0, -1.0, 1.0),
            unit(1.0, 1.0, 1.0),
            unit(-1.0, 1.0, 1.0),
        ];
        vertices[1] = vertices[0];
        let cycles: [&[u32]; 6] = [
            &[0, 3, 2, 1],
            &[4, 5, 6, 7],
            &[0, 1, 5, 4],
            &[3, 7, 6, 2],
            &[0, 4, 7, 3],
            &[1, 2, 6, 5],
        ];
        let mut cells = Vec::new();
        let mut indices = Vec::new();
        for cycle in cycles {
            cells.push(VoronoiCell::new(indices.len() as u32, cycle.len() as u16));
            indices.extend_from_slice(cycle);
        }
        let keys = ShardedVertexKeys::new(
            vec![0, 8],
            vec![vec![
                [0, 2, 4],
                [0, 2, 5],
                [0, 3, 5],
                [0, 3, 4],
                [1, 2, 4],
                [1, 2, 5],
                [1, 3, 5],
                [1, 3, 4],
            ]],
        );
        (vertices, cells, indices, keys)
    }

    fn disabled_weld_cell_killing_points() -> Vec<Vec3> {
        fn displaced(mut b: [f64; 3], theta: f64, phi: f64) -> Vec3 {
            let bl = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            for x in &mut b {
                *x /= bl;
            }
            let el = (b[0] * b[0] + b[1] * b[1]).sqrt();
            let e = [-b[1] / el, b[0] / el, 0.0];
            let f = [
                b[1] * e[2] - b[2] * e[1],
                b[2] * e[0] - b[0] * e[2],
                b[0] * e[1] - b[1] * e[0],
            ];
            let c = theta.cos();
            let s = theta.sin();
            Vec3::new(
                (c * b[0] + s * (phi.cos() * e[0] + phi.sin() * f[0])) as f32,
                (c * b[1] + s * (phi.cos() * e[1] + phi.sin() * f[1])) as f32,
                (c * b[2] + s * (phi.cos() * e[2] + phi.sin() * f[2])) as f32,
            )
            .normalize()
        }

        let base = [-0.61, -0.27, 0.74];
        let theta = 9.0e-8;
        let phase = 3.0 * 0.071;
        let ring = 8;
        let mut points = vec![displaced(base, 0.0, 0.0)];
        for k in 0..ring {
            points.push(displaced(
                base,
                theta,
                phase + std::f64::consts::TAU * k as f64 / ring as f64,
            ));
        }
        let local = points.clone();
        points.extend(local.into_iter().map(|point| -point));
        points
    }

    #[test]
    fn exact_zero_elision_rebuilds_a_strict_compact_mesh() {
        let points = disabled_weld_cell_killing_points();
        let state = run_core_pipeline(
            points.clone(),
            super::TerminationConfig::default(),
            PreprocessMode::Disabled,
            RepairMode::Local3d,
        )
        .expect("cell-killing fixture should reach output resolution");
        assert_eq!(state.cell_killing_generators, [1, 10]);
        assert_eq!(state.output_resolution.cell_killing_components_preserved, 3);

        let elision = super::output_resolution::elide_exact_zero_cells_for_mesh(
            state.effective_points_ref(),
            &state.vertices,
            &state.eff_cells,
            &state.eff_cell_indices,
        )
        .expect("global exact-zero elision quotient should be a valid cell mesh");
        assert_eq!(elision.zero_edges_before, 3);
        assert_eq!(elision.zero_components_before, 3);
        assert_eq!(elision.effective_cells_elided, 2);
        assert_eq!(elision.degree_two_vertices_suppressed, 2);
        assert!(
            elision.max_suppression_cross_track_radians.is_finite()
                && elision.max_suppression_cross_track_radians <= 1.0e-6,
            "forced boundary merge moved off its replacement great circle by {:.3e} rad",
            elision.max_suppression_cross_track_radians,
        );
        assert_eq!(elision.diagram.num_cells(), points.len() - 2);
        assert_eq!(elision.effective_to_cell[1], None);
        assert_eq!(elision.effective_to_cell[10], None);
        assert_eq!(
            elision
                .effective_to_cell
                .iter()
                .filter(|cell| cell.is_none())
                .count(),
            2
        );
        assert_eq!(elision.cell_to_effective.len(), elision.diagram.num_cells());
        assert!(elision.diagram.build_adjacency().is_complete());
        let validation = crate::validation::validate(&elision.diagram);
        assert!(validation.is_strictly_valid(), "{}", validation.headline());
        assert_eq!(validation.zero_length_edges, 0);

        let mut welded_points = points;
        welded_points.push(welded_points[1]);
        let welded_state = run_core_pipeline(
            welded_points.clone(),
            super::TerminationConfig::default(),
            PreprocessMode::MergeWithin(1.0e-10),
            RepairMode::Local3d,
        )
        .expect("welded extension should reach output resolution");
        let merge = welded_state
            .merge_result
            .as_ref()
            .expect("duplicate generator should be welded");
        let welded_elision = super::output_resolution::elide_exact_zero_cells_for_mesh(
            welded_state.effective_points_ref(),
            &welded_state.vertices,
            &welded_state.eff_cells,
            &welded_state.eff_cell_indices,
        )
        .expect("welded effective mesh should admit the same quotient");
        let original_to_cell: Vec<Option<u32>> = merge
            .original_to_effective
            .iter()
            .map(|&effective| welded_elision.effective_to_cell[effective as usize])
            .collect();
        let elided_originals: Vec<usize> = original_to_cell
            .iter()
            .enumerate()
            .filter_map(|(original, cell)| cell.is_none().then_some(original))
            .collect();
        assert_eq!(elided_originals, [1, 10, 18]);
        assert_eq!(welded_elision.diagram.num_cells(), 16);
        assert_eq!(welded_elision.effective_cells_elided, 2);
        assert_eq!(welded_elision.degree_two_vertices_suppressed, 2);
    }

    #[test]
    fn resolution_discovery_decision_falls_back_only_on_global_drift() {
        for drift in [false, true] {
            let decision = resolution_discovery_decision(drift);
            assert_eq!(decision.certified_hint, !drift);
            assert_eq!(decision.drift_fallback, drift);
        }
    }

    #[test]
    fn drift_violation_forces_exhaustive_zero_edge_discovery() {
        let decision = resolution_discovery_decision(true);
        assert!(!decision.certified_hint);
        assert!(decision.drift_fallback);

        let (vertices, mut exhaustive_cells, mut exhaustive_indices, keys) =
            zero_edge_cube_fixture();
        let report = canonicalize_pipeline_exact_zero_edges(
            &vertices,
            &keys,
            &mut exhaustive_cells,
            &mut exhaustive_indices,
            Vec::new(),
            &[],
            decision,
        )
        .expect("drift fallback should run exhaustive discovery");
        assert_eq!(report.report.exact_zero_edges_detected, 1);
        assert_eq!(report.report.exact_zero_edges_contracted, 1);
        assert_eq!(report.report.exact_zero_edges_remaining, 0);
        assert!(report.cell_killing_generators.is_empty());

        let (_, mut hinted_cells, mut hinted_indices, hinted_keys) = zero_edge_cube_fixture();
        let hinted_report = canonicalize_pipeline_exact_zero_edges(
            &vertices,
            &hinted_keys,
            &mut hinted_cells,
            &mut hinted_indices,
            vec![(0, 1)],
            &[],
            resolution_discovery_decision(false),
        )
        .expect("certified candidate should produce the same quotient");
        assert_eq!(hinted_report, report);
        assert_eq!(hinted_cells.len(), exhaustive_cells.len());
        for (hinted, exhaustive) in hinted_cells.iter().zip(&exhaustive_cells) {
            assert_eq!(hinted.vertex_count(), exhaustive.vertex_count());
            let hs = hinted.vertex_start();
            let es = exhaustive.vertex_start();
            assert_eq!(
                &hinted_indices[hs..hs + hinted.vertex_count()],
                &exhaustive_indices[es..es + exhaustive.vertex_count()]
            );
        }

        // Model a post-construction repair which creates the zero edge in one
        // rewritten cell. It was absent from construction hints, but the local
        // mutation footprint is sufficient to discover the same quotient.
        let (_, mut repaired_cells, mut repaired_indices, repaired_keys) = zero_edge_cube_fixture();
        let repaired_report = canonicalize_pipeline_exact_zero_edges(
            &vertices,
            &repaired_keys,
            &mut repaired_cells,
            &mut repaired_indices,
            Vec::new(),
            &[0],
            resolution_discovery_decision(false),
        )
        .expect("local mutation scan should discover an unhinted zero edge");
        assert_eq!(repaired_report, report);
        for (repaired, exhaustive) in repaired_cells.iter().zip(&exhaustive_cells) {
            assert_eq!(repaired.vertex_count(), exhaustive.vertex_count());
            let rs = repaired.vertex_start();
            let es = exhaustive.vertex_start();
            assert_eq!(
                &repaired_indices[rs..rs + repaired.vertex_count()],
                &exhaustive_indices[es..es + exhaustive.vertex_count()]
            );
        }
    }

    #[test]
    fn low_incidence_scan_counts_only_live_cell_windows() {
        let cells = [
            VoronoiCell::new(0, 1),
            VoronoiCell::new(2, 1),
            VoronoiCell::new(4, 1),
        ];
        // Vertex 0 is live in all three cells. Vertex 1 exists only in the
        // stale tail slot following each live span and must not be counted.
        let indices = [0, 1, 0, 1, 0, 1];
        let summary = summarize_topology(2, &cells, &indices);
        assert_eq!(summary.used_vertices, 1);
        assert_eq!(summary.live_half_edges, 3);
        assert!(!summary.low_incidence);

        let low_cells = [VoronoiCell::new(0, 1), VoronoiCell::new(2, 1)];
        let summary = summarize_topology(2, &low_cells, &indices[..4]);
        assert_eq!(summary.used_vertices, 1);
        assert_eq!(summary.live_half_edges, 2);
        assert!(summary.low_incidence);
    }

    #[test]
    fn disabled_repair_cannot_hide_low_incidence_from_plain_return_gate() {
        let cells = [VoronoiCell::new(0, 1), VoronoiCell::new(1, 1)];
        let indices = [0, 0];
        let topology = summarize_topology(1, &cells, &indices);
        assert!(topology.low_incidence);

        // This is the outcome produced by RepairMode::Disabled: no repair was
        // attempted, but the independently-computed safety signal survives.
        let repair = RepairOutcome::not_attempted(
            topology.low_incidence,
            !topology.has_sphere_euler(cells.len()),
        );
        assert!(!repair.attempted);
        assert!(!repair.accepted);
        let err = check_plain_return_signals(repair, &[], &[])
            .expect_err("known-invalid output must not escape when repair is disabled");
        assert!(matches!(err, VoronoiError::ComputationFailed(_)));
    }

    #[test]
    fn accepted_strict_repair_supersedes_pre_repair_signals() {
        let repair = RepairOutcome {
            attempted: true,
            accepted: true,
            low_incidence_defect: true,
            euler_defect: true,
        };
        check_plain_return_signals(repair, &[(1, 2)], &[(2, 3)])
            .expect("accepted repair was already strictly validated");
    }

    fn fib_sphere(n: usize) -> Vec<[f32; 3]> {
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let theta = golden * i as f32;
                let v = Vec3::new(theta.cos() * r, y, theta.sin() * r).normalize();
                [v.x, v.y, v.z]
            })
            .collect()
    }

    fn effective_arrays(
        diagram: &crate::SphericalVoronoi,
    ) -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
        let vertices = diagram
            .vertices()
            .iter()
            .map(|v| Vec3::new(v.x, v.y, v.z))
            .collect();
        let cells = (0..diagram.num_cells())
            .map(|i| VoronoiCell::new(diagram.cell_start(i), diagram.cell(i).len() as u16))
            .collect();
        (vertices, cells, diagram.cell_indices_raw().to_vec())
    }

    fn unaccepted_outcome(
        vertices: &[Vec3],
        cells: &[VoronoiCell],
        cell_indices: &[u32],
    ) -> RepairOutcome {
        let topology = summarize_topology(vertices.len(), cells, cell_indices);
        RepairOutcome::not_attempted(
            topology.low_incidence,
            !topology.has_sphere_euler(cells.len()),
        )
    }

    fn assert_signal_free_gap(
        name: &str,
        vertices: &[Vec3],
        cells: &[VoronoiCell],
        cell_indices: &[u32],
    ) {
        let generators = vec![Vec3::Z; cells.len()];
        let strict = crate::validation::verify_sphere_effective_strict(
            &generators,
            vertices,
            cells,
            cell_indices,
        );
        assert!(strict.is_err(), "{name}: injected defect must be invalid");

        let repair = unaccepted_outcome(vertices, cells, cell_indices);
        let gate = check_plain_return_signals(repair, &[], &[]);
        assert!(
            gate.is_ok(),
            "{name}: expected this mutation to isolate a missing certificate; \
             existing return signals rejected it instead: {gate:?}"
        );
    }

    fn assert_low_incidence_signal_catches(
        name: &str,
        vertices: &[Vec3],
        cells: &[VoronoiCell],
        cell_indices: &[u32],
    ) {
        let generators = vec![Vec3::Z; cells.len()];
        assert!(
            crate::validation::verify_sphere_effective_strict(
                &generators,
                vertices,
                cells,
                cell_indices,
            )
            .is_err(),
            "{name}: injected defect must be invalid"
        );
        let topology = summarize_topology(vertices.len(), cells, cell_indices);
        assert!(
            topology.low_incidence,
            "{name}: fixture must be low-incidence"
        );
        let repair = RepairOutcome::not_attempted(
            topology.low_incidence,
            !topology.has_sphere_euler(cells.len()),
        );
        assert!(
            check_plain_return_signals(repair, &[], &[]).is_err(),
            "{name}: low-incidence mutation must be rejected by the plain gate"
        );
    }

    fn assert_euler_summary_catches(
        name: &str,
        vertices: &[Vec3],
        cells: &[VoronoiCell],
        cell_indices: &[u32],
    ) {
        let generators = vec![Vec3::Z; cells.len()];
        assert!(
            crate::validation::verify_sphere_effective_strict(
                &generators,
                vertices,
                cells,
                cell_indices,
            )
            .is_err(),
            "{name}: injected defect must be invalid"
        );
        let topology = summarize_topology(vertices.len(), cells, cell_indices);
        assert!(
            !topology.has_sphere_euler(cells.len()),
            "{name}: fixture must fail the paired Euler summary"
        );
        assert!(
            check_plain_return_signals(
                RepairOutcome::not_attempted(topology.low_incidence, true),
                &[],
                &[],
            )
            .is_err(),
            "{name}: Euler summary must reject the mutation"
        );
    }

    /// Mutate a known-valid output after assembly while deliberately supplying
    /// no edge-reconciliation signal. These are not claims that the production
    /// pipeline naturally emits each state: they identify the exact properties
    /// whose safety currently rests on construction/detection completeness.
    #[test]
    fn fault_injection_maps_plain_gate_coverage_and_gaps() {
        let good = crate::compute(&fib_sphere(64)).expect("valid baseline");
        let base_generators: Vec<Vec3> = good
            .generators()
            .iter()
            .map(|g| Vec3::new(g.x, g.y, g.z))
            .collect();
        let (base_vertices, base_cells, base_indices) = effective_arrays(&good);
        crate::validation::verify_sphere_effective_strict(
            &base_generators,
            &base_vertices,
            &base_cells,
            &base_indices,
        )
        .expect("baseline must be strictly valid");

        // Reversing one cell preserves every undirected edge use and every
        // incidence count, but makes all of its shared pairs same-direction.
        let (vertices, cells, mut indices) = (
            base_vertices.clone(),
            base_cells.clone(),
            base_indices.clone(),
        );
        let start = cells[0].vertex_start();
        let end = start + cells[0].vertex_count();
        indices[start..end].reverse();
        assert_signal_free_gap("same-direction pairs", &vertices, &cells, &indices);

        // Repeating a non-adjacent vertex removes one ordinary degree-3 use;
        // the existing low-incidence signal catches this class.
        let (vertices, cells, mut indices) = (
            base_vertices.clone(),
            base_cells.clone(),
            base_indices.clone(),
        );
        let cell = cells
            .iter()
            .find(|cell| cell.vertex_count() >= 4)
            .copied()
            .expect("baseline cell with four vertices");
        let start = cell.vertex_start();
        indices[start + 2] = indices[start];
        assert_low_incidence_signal_catches("duplicate vertex", &vertices, &cells, &indices);

        // A direct self-loop is also a repeated vertex and removes the former
        // endpoint's degree-3 use, so it reaches the same existing signal.
        let (vertices, cells, mut indices) = (
            base_vertices.clone(),
            base_cells.clone(),
            base_indices.clone(),
        );
        let start = cells[0].vertex_start();
        indices[start + 1] = indices[start];
        assert_low_incidence_signal_catches("self-loop", &vertices, &cells, &indices);

        // Add another use of an existing edge. All referenced vertices already
        // have degree >= 3, so the incidence signal remains clean.
        let (vertices, mut cells, mut indices) = (
            base_vertices.clone(),
            base_cells.clone(),
            base_indices.clone(),
        );
        let first = &indices[base_cells[0].vertex_start()
            ..base_cells[0].vertex_start() + base_cells[0].vertex_count()];
        let a = first[0];
        let b = first[1];
        let x = (0..vertices.len() as u32)
            .find(|&v| v != a && v != b && !first.contains(&v))
            .expect("unrelated existing vertex");
        let start = indices.len();
        indices.extend_from_slice(&[a, b, x]);
        cells.push(VoronoiCell::new(start as u32, 3));
        assert_euler_summary_catches("overused edge", &vertices, &cells, &indices);

        // Duplicate a face span. Counts only increase, so low incidence cannot
        // reveal the duplicate.
        let (vertices, mut cells, mut indices) = (
            base_vertices.clone(),
            base_cells.clone(),
            base_indices.clone(),
        );
        let source = base_cells[0];
        let span =
            &base_indices[source.vertex_start()..source.vertex_start() + source.vertex_count()];
        let start = indices.len();
        indices.extend_from_slice(span);
        cells.push(VoronoiCell::new(start as u32, span.len() as u16));
        assert_euler_summary_catches("duplicate cell", &vertices, &cells, &indices);

        // Two disjoint copies are locally well-formed spheres. Their union has
        // two components and Euler characteristic 4, with no low incidence.
        let mut vertices = base_vertices.clone();
        let vertex_offset = vertices.len() as u32;
        vertices.extend_from_slice(&base_vertices);
        let mut cells = base_cells.clone();
        let mut indices = base_indices.clone();
        for cell in &base_cells {
            let span =
                &base_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
            let start = indices.len();
            indices.extend(span.iter().map(|&v| v + vertex_offset));
            cells.push(VoronoiCell::new(start as u32, span.len() as u16));
        }
        assert_euler_summary_catches("disconnected/Euler", &vertices, &cells, &indices);

        // Geometry-only corruption leaves all topological signals unchanged.
        let (mut vertices, cells, indices) = (
            base_vertices.clone(),
            base_cells.clone(),
            base_indices.clone(),
        );
        let span =
            &indices[cells[0].vertex_start()..cells[0].vertex_start() + cells[0].vertex_count()];
        vertices[span[1] as usize] = -vertices[span[0] as usize];
        assert_signal_free_gap("antipodal edge", &vertices, &cells, &indices);

        // Weld maps are created after the effective-space gate. An arbitrary
        // corrupt alias is strictly invalid but has no pre-remap repair signal;
        // its production safety rests on `remap_cells_to_original_indices`.
        let generators = good
            .generators()
            .iter()
            .map(|g| Vec3::new(g.x, g.y, g.z))
            .collect();
        let mut weld_map: Vec<u32> = (0..base_cells.len() as u32).collect();
        weld_map[1] = 0;
        let bad_weld = crate::SphericalVoronoi::from_raw_parts(
            generators,
            base_vertices,
            base_cells,
            base_indices,
            Some(weld_map),
        );
        assert!(
            !crate::validation::validate(&bad_weld).is_strictly_valid(),
            "corrupt weld alias must fail strict validation"
        );
        assert!(
            check_plain_return_signals(RepairOutcome::not_attempted(false, false), &[], &[])
                .is_ok(),
            "weld-map validity is not represented by a pre-remap repair signal"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn one_thread_scalar_low_incidence_matches_atomic_path() {
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool");
        let two_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-thread pool");

        let cases = [
            (
                2,
                vec![
                    VoronoiCell::new(0, 1),
                    VoronoiCell::new(2, 1),
                    VoronoiCell::new(4, 1),
                ],
                vec![0, 1, 0, 1, 0, 1],
            ),
            (
                3,
                vec![VoronoiCell::new(0, 2), VoronoiCell::new(2, 2)],
                vec![0, 1, 0, 2],
            ),
            (
                1,
                vec![
                    VoronoiCell::new(0, 1),
                    VoronoiCell::new(1, 1),
                    VoronoiCell::new(2, 1),
                    VoronoiCell::new(3, 1),
                ],
                vec![0, 0, 0, 0],
            ),
        ];

        for (vertex_count, cells, indices) in cases {
            let scalar = one_thread.install(|| summarize_topology(vertex_count, &cells, &indices));
            let atomic = two_threads.install(|| summarize_topology(vertex_count, &cells, &indices));
            assert_eq!(scalar, atomic);
        }
    }

    #[test]
    fn fused_validation_reports_non_finite_at_start_middle_and_end() {
        let invalids = [
            (0usize, Vec3::new(f32::NAN, 2.0, 3.0)),
            (2usize, Vec3::new(1.0, f32::INFINITY, 3.0)),
            (4usize, Vec3::new(1.0, 2.0, f32::NEG_INFINITY)),
        ];

        for (bad_idx, bad) in invalids {
            let mut points = vec![Vec3::new(0.5, 0.5, 0.5); 5];
            points[bad_idx] = bad;
            let err = validate_and_canonicalize_unit_points(&mut points)
                .expect_err("non-finite generator must be rejected");
            match err {
                VoronoiError::InvalidInput {
                    point_index,
                    message,
                } => {
                    assert_eq!(point_index, bad_idx);
                    assert_eq!(
                        message,
                        format!(
                            "point has a non-finite component: ({}, {}, {})",
                            bad.x, bad.y, bad.z
                        )
                    );
                    assert_eq!(
                        points[bad_idx].to_array().map(f32::to_bits),
                        bad.to_array().map(f32::to_bits)
                    );
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[test]
    fn fused_validation_chooses_first_invalid_across_parallel_chunks() {
        const CHUNK: usize = 1 << 16;
        let first_bad = CHUNK + 7;
        let mut points = vec![Vec3::new(0.5, 0.5, 0.5); 2 * CHUNK + 3];
        points[first_bad] = Vec3::new(1.0, f32::NAN, 3.0);
        points[2 * CHUNK + 1] = Vec3::new(f32::INFINITY, 2.0, 3.0);

        let err = validate_and_canonicalize_unit_points(&mut points)
            .expect_err("non-finite generators must be rejected");
        assert!(matches!(
            err,
            VoronoiError::InvalidInput {
                point_index,
                ..
            } if point_index == first_bad
        ));
    }

    #[test]
    fn fused_validation_preserves_canonicalization_bits() {
        let mut points = vec![
            Vec3::new(0.3, -0.7, 0.2),
            Vec3::new(-0.4, 0.6, 0.8),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let expected: Vec<Vec3> = points
            .iter()
            .map(|p| {
                let v = glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64);
                let len_sq = v.length_squared();
                if (0.25..=4.0).contains(&len_sq) {
                    let n = v / len_sq.sqrt();
                    Vec3::new(n.x as f32, n.y as f32, n.z as f32)
                } else {
                    *p
                }
            })
            .collect();

        validate_and_canonicalize_unit_points(&mut points).expect("finite points must pass");
        for (got, expected) in points.iter().zip(&expected) {
            assert_eq!(
                got.to_array().map(f32::to_bits),
                expected.to_array().map(f32::to_bits)
            );
        }
    }

    #[test]
    fn stable_rank2_normal_repivots_for_two_arc_ordering() {
        let points: Vec<Vec3> = [0.0f32, 20.0, 160.0, 180.0, 200.0, 340.0]
            .into_iter()
            .map(|degrees| {
                let angle = degrees.to_radians();
                Vec3::new(angle.cos(), angle.sin(), 0.0)
            })
            .collect();

        // No pair involving the first point clears the stability threshold:
        // the second sweep must re-pivot onto an arc endpoint.
        let first = points[0];
        assert!(
            points
                .iter()
                .map(|&p| first.cross(p).length_squared())
                .fold(0.0f32, f32::max)
                < 0.25
        );
        let normal = stable_rank2_normal(&points).expect("stable pair should be found");
        assert!(normal.z.abs() > 0.999_999);
    }

    #[test]
    fn stable_rank2_normal_large_nonplanar_probe_is_linear() {
        // Large enough that the former all-pairs implementation is
        // impractical even as a unit test. The candidate plane is rejected by
        // the caller's all-point plane check; this test pins bounded pair
        // selection itself.
        let points: Vec<Vec3> = (0..100_000)
            .map(|i| match i % 3 {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            })
            .collect();
        assert!(stable_rank2_normal(&points).is_some());
        assert!(classify_near_great_circle(&points).is_none());
    }

    #[test]
    fn rank2_classifier_scales_to_large_great_circle() {
        let n = 100_000usize;
        let points: Vec<Vec3> = (0..n)
            .map(|i| {
                let angle = std::f32::consts::TAU * i as f32 / n as f32;
                Vec3::new(angle.cos(), angle.sin(), 0.0)
            })
            .collect();
        let class = classify_near_great_circle(&points)
            .expect("full great-circle fixture should be classified as rank 2");
        assert!(class.normal.z.abs() > 0.999_999);
    }

    #[test]
    fn exact_affine_circle_classifier_uses_exact_canonical_model() {
        let coplanar = [
            Vec3::new(0.8, 0.0, 0.6),
            Vec3::new(0.0, 0.8, 0.6),
            Vec3::new(-0.8, 0.0, 0.6),
            Vec3::new(0.0, -0.8, 0.6),
        ];
        let class = classify_exact_affine_circle(&coplanar)
            .expect("constant-z canonical points are exactly affinely coplanar");
        assert!(class.normal.z.abs() > 0.999_999);

        let mut noncoplanar = coplanar;
        noncoplanar[3].z = f32::from_bits(noncoplanar[3].z.to_bits() + 1);
        assert!(
            classify_exact_affine_circle(&noncoplanar).is_none(),
            "one f32 ulp off the plane must not tolerance-classify as exact"
        );
    }

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
        let (grid, dense_index_eligible) = build_query_grid(&points, &mut tb);
        assert!(dense_index_eligible);
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

    #[test]
    fn dense_index_is_deferred_until_retained_grid_finalization() {
        use crate::knn_clipping::timing::TimingBuilder;

        // A sub-cell cap remains dense even after occupancy feedback reaches
        // its resolution/memory limit, so the retained grid genuinely needs
        // the side index. A small spiral avoids duplicate positions while
        // keeping every point in the same final cell.
        let n = 5_000usize;
        let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let points: Vec<Vec3> = (0..n)
            .map(|i| {
                let r = 1.0e-4 * ((i as f32 + 0.5) / n as f32).sqrt();
                let theta = golden * i as f32;
                Vec3::new(r * theta.cos(), r * theta.sin(), 1.0).normalize()
            })
            .collect();

        let mut tb = TimingBuilder::new();
        let (mut grid, dense_index_eligible) = build_query_grid(&points, &mut tb);
        assert!(dense_index_eligible, "sub-cell cap must trigger regridding");
        let dense_cell = grid.point_index_to_cell(0) as u32;
        assert!(grid.cell_points(dense_cell as usize).len() > crate::policy::DENSE_CELL_THRESHOLD);
        assert_eq!(
            grid.dense_band_radius(dense_cell, 64),
            None,
            "provisional grid must not build the dense side index"
        );

        grid.build_dense_index();
        assert!(
            grid.dense_band_radius(dense_cell, 64).is_some(),
            "retained grid finalization must materialize the dense side index"
        );
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
