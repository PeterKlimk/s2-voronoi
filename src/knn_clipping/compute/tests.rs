use super::coplanar::{
    classify_exact_affine_circle, classify_near_great_circle, stable_rank2_normal,
};
use super::error_mapping::{map_build_cells_error, map_cell_build_error};
use super::{
    build_query_grid, canonicalize_pipeline_exact_zero_edges, cell_sum_sq_per_n,
    check_plain_return_signals, max_cell_occupancy, prepare_points_and_grid, run_core_pipeline,
    summarize_topology, summarize_topology_after_reconcile, validate_and_canonicalize_unit_points,
    validate_generator_capacity, EffectiveGeometry, EffectiveInput, LocalRebuildCandidate,
    LocalRebuildOutcome, ResolutionDiscoveryMode,
};
use crate::cell_layout::LiveCellLayout;
use crate::diagram::VoronoiCell;
use crate::knn_clipping::edge_reconcile::CellCycleSnapshot;
use crate::live_dedup::{
    BuildCellsError, IncidenceSummary, PackedLayoutCapacityError, ShardedVertexKeys,
};
use crate::live_dedup::{CellBuildError, CellFailure};
use crate::test_support::{effective_arrays, effective_generators, fib_sphere};
use crate::timing::TimingBuilder;
use crate::{LocalRebuildMode, LocalRebuildStatus, PreprocessMode, VoronoiConfig, VoronoiError};
use glam::Vec3;

fn separated_preprocess_points() -> Vec<Vec3> {
    let unit = |x: f32, y: f32, z: f32| Vec3::new(x, y, z).normalize();
    vec![
        unit(1.0, 1.0, 1.0),
        unit(-1.0, -1.0, 1.0),
        unit(-1.0, 1.0, -1.0),
        unit(1.0, -1.0, -1.0),
    ]
}

#[test]
fn prepared_input_owns_only_actual_merges() {
    let points = separated_preprocess_points();

    for mode in [PreprocessMode::Disabled, PreprocessMode::Weld] {
        let mut tb = TimingBuilder::new();
        let prepared = prepare_points_and_grid(&points, mode, &mut tb)
            .expect("separated points should prepare without merging");
        assert!(matches!(
            &prepared.effective_input,
            EffectiveInput::Identity
        ));
        assert!(std::ptr::eq(
            prepared.effective_input.points(&points).as_ptr(),
            points.as_ptr()
        ));
        assert_eq!(prepared.report.requested_mode, mode);
        assert_eq!(prepared.report.original_points, points.len());
        assert_eq!(prepared.report.effective_points, points.len());
        assert_eq!(prepared.report.num_merged, 0);
    }

    let mut duplicated = points;
    duplicated.push(duplicated[0]);
    let mut tb = TimingBuilder::new();
    let prepared = prepare_points_and_grid(&duplicated, PreprocessMode::Weld, &mut tb)
        .expect("exact duplicate should prepare as one merged input");
    let EffectiveInput::Merged(merge) = &prepared.effective_input else {
        panic!("actual merge must own one complete merge result");
    };
    assert_eq!(merge.effective_points.len(), duplicated.len() - 1);
    assert_eq!(merge.original_to_effective.len(), duplicated.len());
    assert_eq!(
        merge.original_to_effective[0],
        merge.original_to_effective[duplicated.len() - 1]
    );
    assert_eq!(merge.num_merged, 1);
    assert_eq!(prepared.report.original_points, duplicated.len());
    assert_eq!(prepared.report.effective_points, duplicated.len() - 1);
    assert_eq!(prepared.report.num_merged, 1);
}

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
        PreprocessMode::Disabled,
        LocalRebuildMode::Hull3d,
        None,
    )
    .expect("cell-killing fixture should reach output resolution");
    assert_eq!(state.cell_killing_generators, [1, 10]);
    assert_eq!(state.output_resolution.cell_killing_components_preserved, 3);

    let elision = super::output_resolution::elide_exact_zero_cells_for_mesh(
        state.effective_points_ref(),
        &state.geometry.vertices,
        &state.geometry.cells,
        &state.geometry.cell_indices,
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
        PreprocessMode::MergeWithin(1.0e-10),
        LocalRebuildMode::Hull3d,
        None,
    )
    .expect("welded extension should reach output resolution");
    let merge = welded_state
        .merge_result()
        .expect("duplicate generator should be welded");
    let welded_elision = super::output_resolution::elide_exact_zero_cells_for_mesh(
        welded_state.effective_points_ref(),
        &welded_state.geometry.vertices,
        &welded_state.geometry.cells,
        &welded_state.geometry.cell_indices,
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
fn resolution_discovery_mode_falls_back_only_on_global_drift() {
    assert_eq!(
        ResolutionDiscoveryMode::from_drift(false),
        ResolutionDiscoveryMode::CertifiedHint
    );
    assert_eq!(
        ResolutionDiscoveryMode::from_drift(true),
        ResolutionDiscoveryMode::ExhaustiveDriftFallback
    );
}

#[test]
fn drift_violation_forces_exhaustive_zero_edge_discovery() {
    let mode = ResolutionDiscoveryMode::from_drift(true);
    assert_eq!(mode, ResolutionDiscoveryMode::ExhaustiveDriftFallback);

    let (vertices, mut exhaustive_cells, mut exhaustive_indices, keys) = zero_edge_cube_fixture();
    let report = canonicalize_pipeline_exact_zero_edges(
        &vertices,
        &keys,
        &mut exhaustive_cells,
        &mut exhaustive_indices,
        Vec::new(),
        &[],
        mode,
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
        ResolutionDiscoveryMode::CertifiedHint,
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

    // Model a post-construction local rebuild which creates the zero edge in one
    // rewritten cell. It was absent from construction hints, but the local
    // mutation footprint is sufficient to discover the same quotient.
    let (_, mut rebuilt_cells, mut rebuilt_indices, rebuilt_keys) = zero_edge_cube_fixture();
    let rebuilt_report = canonicalize_pipeline_exact_zero_edges(
        &vertices,
        &rebuilt_keys,
        &mut rebuilt_cells,
        &mut rebuilt_indices,
        Vec::new(),
        &[0],
        ResolutionDiscoveryMode::CertifiedHint,
    )
    .expect("local mutation scan should discover an unhinted zero edge");
    assert_eq!(rebuilt_report, report);
    for (rebuilt, exhaustive) in rebuilt_cells.iter().zip(&exhaustive_cells) {
        assert_eq!(rebuilt.vertex_count(), exhaustive.vertex_count());
        let rs = rebuilt.vertex_start();
        let es = exhaustive.vertex_start();
        assert_eq!(
            &rebuilt_indices[rs..rs + rebuilt.vertex_count()],
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
fn disabled_local_rebuild_cannot_hide_low_incidence_from_plain_return_gate() {
    let cells = [VoronoiCell::new(0, 1), VoronoiCell::new(1, 1)];
    let indices = [0, 0];
    let topology = summarize_topology(1, &cells, &indices);
    assert!(topology.low_incidence);

    // This is the outcome produced by LocalRebuildMode::Disabled: no local rebuild was
    // attempted, but the independently-computed safety signal survives.
    let local_rebuild = LocalRebuildOutcome::new(
        LocalRebuildStatus::Disabled,
        topology.low_incidence,
        !topology.has_sphere_euler(cells.len()),
    );
    assert_eq!(local_rebuild.report_status(), LocalRebuildStatus::Disabled);
    let err = check_plain_return_signals(local_rebuild, &[], &[])
        .expect_err("known-invalid output must not escape when local_rebuild is disabled");
    assert!(matches!(err, VoronoiError::ComputationFailed(_)));
}

#[test]
fn accepted_strict_local_rebuild_supersedes_pre_local_rebuild_signals() {
    let local_rebuild = LocalRebuildOutcome::new(LocalRebuildStatus::Accepted, true, true);
    check_plain_return_signals(local_rebuild, &[(1, 2)], &[(2, 3)])
        .expect("accepted local_rebuild was already strictly validated");
}

#[test]
fn local_rebuild_status_truth_table_and_ordinary_paths() {
    for (status, attempted, accepted) in [
        (LocalRebuildStatus::NotTriggered, false, false),
        (LocalRebuildStatus::Disabled, false, false),
        (LocalRebuildStatus::Rejected, true, false),
        (LocalRebuildStatus::Accepted, true, true),
    ] {
        assert_eq!(status.attempted(), attempted, "status={status:?}");
        assert_eq!(status.accepted(), accepted, "status={status:?}");
    }

    let points = fib_sphere(64);
    let clean = crate::compute_with_report(&points, VoronoiConfig::default()).expect("clean");
    assert_eq!(clean.report.local_rebuild, LocalRebuildStatus::NotTriggered);

    let disabled = crate::compute_with_report(
        &points,
        VoronoiConfig::default().with_local_rebuild_mode(LocalRebuildMode::Disabled),
    )
    .expect("disabled");
    assert_eq!(disabled.report.local_rebuild, LocalRebuildStatus::Disabled);
}

#[test]
fn local_rebuild_candidate_commits_positions_arrays_and_footprint_together() {
    let diagram = crate::compute(&fib_sphere(64)).expect("valid baseline");
    let generators = effective_generators(&diagram);
    let (vertices, cells, indices) = effective_arrays(&diagram);
    let replacement_vertex = indices[0];
    let minted_vertex = vertices[replacement_vertex as usize];
    let minted_vertex_id = vertices.len() as u32;

    let mut replacement_indices = indices.clone();
    for vertex in &mut replacement_indices {
        if *vertex == replacement_vertex {
            *vertex = minted_vertex_id;
        }
    }
    let candidate = LocalRebuildCandidate {
        minted_vertices: vec![minted_vertex],
        cells: cells.clone(),
        cell_indices: replacement_indices.clone(),
        resolution_scan_cells: vec![2, 7, 11],
    };
    let replacement_cells_ptr = candidate.cells.as_ptr();
    let replacement_indices_ptr = candidate.cell_indices.as_ptr();
    let mut geometry = EffectiveGeometry {
        vertices,
        cells,
        cell_indices: indices,
    };

    let footprint = candidate
        .try_commit(&generators, &mut geometry, false, std::time::Instant::now())
        .expect("equivalent replacement should pass the whole-diagram gate");

    assert_eq!(footprint, [2, 7, 11]);
    assert_eq!(geometry.vertices.len(), minted_vertex_id as usize + 1);
    assert_eq!(geometry.vertices.last(), Some(&minted_vertex));
    assert_eq!(geometry.cells.as_ptr(), replacement_cells_ptr);
    assert_eq!(geometry.cell_indices.as_ptr(), replacement_indices_ptr);
    assert_eq!(geometry.cell_indices, replacement_indices);
    assert!(!geometry.cell_indices.contains(&replacement_vertex));
}

#[test]
fn local_rebuild_candidate_rejection_restores_base_geometry() {
    let diagram = crate::compute(&fib_sphere(64)).expect("valid baseline");
    let generators = effective_generators(&diagram);
    let (vertices, cells, indices) = effective_arrays(&diagram);
    let mut invalid_indices = indices.clone();
    invalid_indices[0] = u32::MAX;
    let candidate = LocalRebuildCandidate {
        minted_vertices: vec![vertices[0]],
        cells: cells.clone(),
        cell_indices: invalid_indices,
        resolution_scan_cells: vec![0],
    };
    let mut geometry = EffectiveGeometry {
        vertices: vertices.clone(),
        cells,
        cell_indices: indices.clone(),
    };
    let base_cells_ptr = geometry.cells.as_ptr();
    let base_indices_ptr = geometry.cell_indices.as_ptr();

    let footprint =
        candidate.try_commit(&generators, &mut geometry, false, std::time::Instant::now());

    assert!(footprint.is_none());
    assert_eq!(geometry.vertices, vertices);
    assert_eq!(geometry.cells.as_ptr(), base_cells_ptr);
    assert_eq!(geometry.cell_indices.as_ptr(), base_indices_ptr);
    assert_eq!(geometry.cell_indices, indices);
}

fn unaccepted_outcome(
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> LocalRebuildOutcome {
    let topology = summarize_topology(vertices.len(), cells, cell_indices);
    LocalRebuildOutcome::new(
        LocalRebuildStatus::NotTriggered,
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
        LiveCellLayout::new(cells, cell_indices),
    );
    assert!(strict.is_err(), "{name}: injected defect must be invalid");

    let local_rebuild = unaccepted_outcome(vertices, cells, cell_indices);
    let gate = check_plain_return_signals(local_rebuild, &[], &[]);
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
            LiveCellLayout::new(cells, cell_indices),
        )
        .is_err(),
        "{name}: injected defect must be invalid"
    );
    let topology = summarize_topology(vertices.len(), cells, cell_indices);
    assert!(
        topology.low_incidence,
        "{name}: fixture must be low-incidence"
    );
    let local_rebuild = LocalRebuildOutcome::new(
        LocalRebuildStatus::NotTriggered,
        topology.low_incidence,
        !topology.has_sphere_euler(cells.len()),
    );
    assert!(
        check_plain_return_signals(local_rebuild, &[], &[]).is_err(),
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
            LiveCellLayout::new(cells, cell_indices),
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
            LocalRebuildOutcome::new(
                LocalRebuildStatus::NotTriggered,
                topology.low_incidence,
                true,
            ),
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
        .map(|g| Vec3::from_array(g.to_array()))
        .collect();
    let (base_vertices, base_cells, base_indices) = effective_arrays(&good);
    crate::validation::verify_sphere_effective_strict(
        &base_generators,
        &base_vertices,
        LiveCellLayout::new(&base_cells, &base_indices),
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
    let first = &indices
        [base_cells[0].vertex_start()..base_cells[0].vertex_start() + base_cells[0].vertex_count()];
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
    let span = &base_indices[source.vertex_start()..source.vertex_start() + source.vertex_count()];
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
        let span = &base_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
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
    let span = &indices[cells[0].vertex_start()..cells[0].vertex_start() + cells[0].vertex_count()];
    vertices[span[1] as usize] = -vertices[span[0] as usize];
    assert_signal_free_gap("antipodal edge", &vertices, &cells, &indices);

    // Weld maps are created after the effective-space gate. An arbitrary
    // corrupt alias is strictly invalid but has no pre-remap local rebuild signal;
    // its production safety rests on `remap_cells_to_original_indices`.
    let generators = good
        .generators()
        .iter()
        .map(|g| Vec3::from_array(g.to_array()))
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
        check_plain_return_signals(
            LocalRebuildOutcome::new(LocalRebuildStatus::NotTriggered, false, false),
            &[],
            &[],
        )
        .is_ok(),
        "weld-map validity is not represented by a pre-remap local_rebuild signal"
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

    // Exercise the compact-counter overflow path explicitly. Without the
    // half-edge checksum and exact fallback, 256 references wrap to zero and
    // incorrectly make the live vertex look unused.
    let overflow_cells: Vec<VoronoiCell> =
        (0..256).map(|start| VoronoiCell::new(start, 1)).collect();
    let overflow_indices = vec![0; 256];
    let scalar = one_thread.install(|| summarize_topology(1, &overflow_cells, &overflow_indices));
    let atomic = two_threads.install(|| summarize_topology(1, &overflow_cells, &overflow_indices));
    assert_eq!(scalar, atomic);
}

#[test]
fn reconciliation_topology_delta_detects_a_new_low_incidence_vertex() {
    let vertex_keys =
        ShardedVertexKeys::new(vec![0, 3], vec![vec![[0, 1, 2], [0, 1, 2], [0, 1, 2]]]);
    let baseline = IncidenceSummary {
        used_vertices: 3,
        live_half_edges: 9,
        low_incidence: false,
        low_incidence_vertices: Vec::new(),
    };
    let snapshots = [CellCycleSnapshot {
        cell: 0,
        vertices: vec![0, 1, 2],
    }];
    let cells = [
        VoronoiCell::new(0, 2),
        VoronoiCell::new(2, 3),
        VoronoiCell::new(5, 3),
    ];
    let indices = [0, 1, 0, 1, 2, 0, 1, 2];

    assert_eq!(
        summarize_topology_after_reconcile(&baseline, &vertex_keys, &snapshots, &cells, &indices,),
        Some(super::TopologySummary {
            used_vertices: 3,
            live_half_edges: 8,
            low_incidence: true,
        })
    );
}

#[test]
fn reconciliation_topology_delta_falls_back_for_ambiguous_baseline() {
    let vertex_keys = ShardedVertexKeys::new(vec![0, 1], vec![vec![[0, 0, 0]]]);
    let cells = [VoronoiCell::new(0, 1)];
    let indices = [0];
    let snapshots = [CellCycleSnapshot {
        cell: 0,
        vertices: vec![0],
    }];

    let baseline = IncidenceSummary {
        used_vertices: 1,
        live_half_edges: 2,
        low_incidence: true,
        low_incidence_vertices: Vec::new(),
    };
    assert_eq!(
        summarize_topology_after_reconcile(&baseline, &vertex_keys, &snapshots, &cells, &indices,),
        None
    );
}

#[test]
fn reconciliation_topology_delta_resolves_sparse_low_incidence_hint() {
    let vertex_keys =
        ShardedVertexKeys::new(vec![0, 3], vec![vec![[0, 1, 2], [0, 1, 2], [0, 1, 2]]]);
    let baseline = IncidenceSummary {
        used_vertices: 3,
        live_half_edges: 9,
        low_incidence: true,
        low_incidence_vertices: vec![2],
    };
    let snapshots = [CellCycleSnapshot {
        cell: 0,
        vertices: vec![0, 1, 2],
    }];
    let cells = [
        VoronoiCell::new(0, 3),
        VoronoiCell::new(3, 3),
        VoronoiCell::new(6, 3),
    ];
    let indices = [0, 1, 2, 0, 1, 2, 0, 1, 2];

    assert_eq!(
        summarize_topology_after_reconcile(&baseline, &vertex_keys, &snapshots, &cells, &indices,),
        Some(super::TopologySummary {
            used_vertices: 3,
            live_half_edges: 9,
            low_incidence: false,
        })
    );
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
        BuildCellsError::RepresentationLimit("cell vertex count exceeds u8 capacity".to_string()),
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
    use crate::timing::TimingBuilder;

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
    let (grid, dense_index_eligible) = build_query_grid(&points, &mut tb, false);
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
    use crate::timing::TimingBuilder;

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
    let (mut grid, dense_index_eligible) = build_query_grid(&points, &mut tb, false);
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
