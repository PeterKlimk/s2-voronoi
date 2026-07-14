use super::failure::classify_terminal_failure;
use super::fallback_detail;
use super::{
    build_cell_into, clip_batch, clip_seed_neighbors, consume_stream, finish_cell, probe_frontier,
    should_clip_neighbor, AttemptedNeighbors, BuildCounters, BuildTrace, CellBuildContext,
    CellBuildRequest, StreamPhase, TerminationCheckpoint,
};
use crate::cube_grid::packed_knn::{
    PackedGroupInput, PackedKnnCellScratch, PackedKnnTimings, PreparedPackedGroupStatus,
};
use crate::cube_grid::{
    CubeMapGrid, DirectedEligibility, DirectedNeighborFrontier, DirectedNeighborStream,
};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::Topo2DBuilder;
use crate::knn_clipping::TerminationConfig;
use crate::packed_layout::PackedSlotLayout;
use crate::policy::PackedNeighborPolicy;
use glam::Vec3;

fn octahedron_points() -> Vec<Vec3> {
    vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
}

fn poison_output_buffer(ctx: &mut CellBuildContext) {
    ctx.output_buffer.clear();
    ctx.output_buffer
        .vertices
        .push(([u32::MAX; 3], Vec3::new(0.25, -0.5, 0.75)));
    ctx.output_buffer.edge_neighbor_globals.push(u32::MAX - 1);
    ctx.output_buffer.edge_neighbor_slots.push(u32::MAX - 2);
    ctx.output_buffer.edge_keys_verified = true;
}

fn assert_output_replaced(buffer: &crate::live_dedup::CellOutputBuffer) {
    let n = buffer.vertices.len();
    assert!(n >= 3);
    assert_eq!(buffer.edge_neighbor_globals.len(), n);
    assert_eq!(buffer.edge_neighbor_slots.len(), n);
    assert!(buffer.vertices.iter().all(|(key, _)| *key != [u32::MAX; 3]));
    assert!(buffer
        .edge_neighbor_globals
        .iter()
        .all(|&neighbor| neighbor != u32::MAX - 1));
    assert!(buffer
        .edge_neighbor_slots
        .iter()
        .all(|&slot| slot != u32::MAX - 2));
}

fn assert_output_still_poisoned(buffer: &crate::live_dedup::CellOutputBuffer) {
    assert_eq!(buffer.vertices.len(), 1);
    assert_eq!(buffer.vertices[0].0, [u32::MAX; 3]);
    assert_eq!(buffer.edge_neighbor_globals, [u32::MAX - 1]);
    assert_eq!(buffer.edge_neighbor_slots, [u32::MAX - 2]);
    assert!(buffer.edge_keys_verified);
}

#[test]
fn extraction_writers_replace_poison_and_errors_do_not_consume_it() {
    let points = octahedron_points();
    let grid = CubeMapGrid::new(&points, 4);
    let policy = TerminationConfig::default().packed_policy(points.len());
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);

    // Common gnomonic writer.
    let mut ctx = CellBuildContext::new(&grid, policy);
    poison_output_buffer(&mut ctx);
    build_cell_into(
        &mut ctx,
        CellBuildRequest {
            points: &points,
            grid: &grid,
            generator_idx: 0,
            directed_ctx,
            packed: None,
            incoming_checks: &[],
        },
    )
    .expect("gnomonic cell should replace poisoned output");
    assert_output_replaced(ctx.output_buffer());

    // Exhaustion recovery uses the all-constraints writer. Invoke that writer
    // on the accepted constraints from the same finished cell so the test does
    // not depend on constructing a rare recoverable-exhaustion distribution.
    poison_output_buffer(&mut ctx);
    ctx.builder
        .to_vertex_data_from_all_constraints(&points, &mut ctx.output_buffer)
        .expect("all-constraints writer should replace poisoned output");
    assert_output_replaced(ctx.output_buffer());

    // Projection fallback writer, forced through the existing test hook.
    let mut fallback_ctx = CellBuildContext::new(&grid, policy);
    fallback_ctx.force_fallback_after_neighbors_processed = Some(2);
    poison_output_buffer(&mut fallback_ctx);
    build_cell_into(
        &mut fallback_ctx,
        CellBuildRequest {
            points: &points,
            grid: &grid,
            generator_idx: 0,
            directed_ctx,
            packed: None,
            incoming_checks: &[],
        },
    )
    .expect("fallback cell should replace poisoned output");
    assert!(fallback_ctx.builder.is_fallback());
    assert_output_replaced(fallback_ctx.output_buffer());

    // A terminal failure may leave stale data in the reusable context, but the
    // Err return prevents the production driver from reading or emitting it.
    let failing_points = vec![Vec3::Z];
    let failing_grid = CubeMapGrid::new(&failing_points, 4);
    let failing_policy = TerminationConfig::default().packed_policy(failing_points.len());
    let failing_slot_map = vec![0u32; failing_points.len()];
    let failing_directed = DirectedEligibility::new(u8::MAX, 0, &failing_slot_map, 0, 0);
    let mut failing_ctx = CellBuildContext::new(&failing_grid, failing_policy);
    poison_output_buffer(&mut failing_ctx);
    let result = build_cell_into(
        &mut failing_ctx,
        CellBuildRequest {
            points: &failing_points,
            grid: &failing_grid,
            generator_idx: 0,
            directed_ctx: failing_directed,
            packed: None,
            incoming_checks: &[],
        },
    );
    assert!(result.is_err());
    assert_output_still_poisoned(failing_ctx.output_buffer());
}

#[test]
fn source_specialized_attempted_neighbor_semantics() {
    let mut attempted = AttemptedNeighbors::new(4);

    // Both packed sources use this specialization: every occurrence is
    // clipped, and each slot is marked for a later shell takeover.
    assert!(should_clip_neighbor::<false>(&mut attempted, 1));
    assert!(should_clip_neighbor::<false>(&mut attempted, 1));
    assert!(!should_clip_neighbor::<true>(&mut attempted, 1));

    // Shell batches deduplicate both within the shell and against marks left
    // by either packed source.
    assert!(should_clip_neighbor::<true>(&mut attempted, 2));
    assert!(!should_clip_neighbor::<true>(&mut attempted, 2));

    attempted.clear();
    assert!(should_clip_neighbor::<true>(&mut attempted, 1));
}

fn great_circle_points(n: usize, jitter: f32) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let theta = std::f32::consts::TAU * i as f32 / n as f32;
            let z = if jitter > 0.0 {
                jitter * (0.37 * i as f32).sin()
            } else {
                0.0
            };
            Vec3::new(theta.cos(), theta.sin(), z).normalize()
        })
        .collect()
}

fn hemisphere_points(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let t = i as f32 + 0.5;
            let z = (t / n as f32).min(0.999);
            let theta = std::f32::consts::TAU * t * 0.618_034;
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

fn fibonacci_points(n: usize) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let r = (1.0 - y * y).sqrt();
            let theta = golden * i as f32;
            Vec3::new(r * theta.cos(), y, r * theta.sin())
        })
        .collect()
}

fn pole_with_latitude_ring(n: usize, z: f32) -> Vec<Vec3> {
    let r = (1.0 - z * z).sqrt();
    let mut points = Vec::with_capacity(n + 2);
    points.push(Vec3::Z);
    points.push(-Vec3::Z);
    for i in 0..n {
        let theta = std::f32::consts::TAU * i as f32 / n as f32;
        points.push(Vec3::new(r * theta.cos(), r * theta.sin(), z));
    }
    points
}

fn cap_fibonacci_points(n: usize, radius_rad: f32) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let cos_min = radius_rad.cos();
    (0..n)
        .map(|i| {
            let t = (i as f32 + 0.5) / n as f32;
            let z = 1.0 - t * (1.0 - cos_min);
            let r = (1.0 - z * z).sqrt();
            let theta = golden * i as f32;
            Vec3::new(r * theta.cos(), r * theta.sin(), z).normalize()
        })
        .collect()
}

fn cap_with_antipode(n: usize, radius_rad: f32) -> Vec<Vec3> {
    let mut points = cap_fibonacci_points(n.saturating_sub(1), radius_rad);
    points.push(-Vec3::Z);
    points
}

fn cap_with_octahedral_anchors(n: usize, radius_rad: f32) -> Vec<Vec3> {
    let anchors = octahedron_points();
    let mut points = anchors;
    points.extend(cap_fibonacci_points(
        n.saturating_sub(points.len()),
        radius_rad,
    ));
    points
}

#[derive(Debug, Clone)]
struct ProbeCell {
    generator: usize,
    ok: bool,
    failure: Option<CellFailure>,
    neighbors_processed: usize,
    final_edges: usize,
    knn_exhausted: bool,
    bounded: bool,
    fallback_projection: usize,
    fallback_polygon_cap: usize,
    fallback_all_constraints: usize,
    spherical_extract_vertices: Option<usize>,
    spherical_extract_edges: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CellSignature {
    vertex_keys: Vec<[u32; 3]>,
    edge_neighbors: Vec<u32>,
}

#[derive(Debug, Clone)]
struct EarlyExtractHit {
    neighbors_processed: usize,
    accepted_constraints: usize,
    edges: usize,
}

#[derive(Debug, Clone)]
struct EarlyProbeCell {
    generator: usize,
    ok: bool,
    failure: Option<CellFailure>,
    neighbors_processed: usize,
    final_edges: usize,
    knn_exhausted: bool,
    fallback_all_constraints: usize,
    first_success: Option<EarlyExtractHit>,
    first_final_match: Option<EarlyExtractHit>,
    successes: usize,
}

fn probe_cell(points: &[Vec3], grid: &CubeMapGrid, generator_idx: usize) -> ProbeCell {
    let policy = TerminationConfig::default().packed_policy(points.len());
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);
    let mut ctx = CellBuildContext::new(grid, policy);
    let pos_slots = grid.point_pos_slots();
    let mut trace = BuildTrace::new();
    let mut counters = BuildCounters::new();

    ctx.builder.reset(generator_idx, points[generator_idx]);
    ctx.attempted_neighbors.clear();
    ctx.output_buffer.clear();

    clip_seed_neighbors(
        &mut ctx,
        points,
        grid,
        pos_slots,
        &[],
        &mut trace,
        &mut counters,
    );

    {
        let mut stream = DirectedNeighborStream::new(
            grid,
            points,
            generator_idx,
            &mut ctx.scratch,
            directed_ctx,
            None,
        );
        consume_stream(
            &mut stream,
            StreamPhase {
                builder: &mut ctx.builder,
                packed_chunk: &mut ctx.packed_chunk,
                attempted_neighbors: &mut ctx.attempted_neighbors,
                force_fallback_after_neighbors_processed: &mut ctx
                    .force_fallback_after_neighbors_processed,
            },
            points,
            pos_slots,
            generator_idx,
            &mut trace,
            &mut counters,
        );
        counters.absorb_stream(&stream);
    }

    let result = finish_cell(&mut ctx, points, grid, generator_idx, &trace, &mut counters);
    let spherical_extract = if result.is_err() {
        let mut buffer = crate::live_dedup::CellOutputBuffer::default();
        ctx.builder
            .to_vertex_data_from_all_constraints(points, &mut buffer)
            .ok()
            .map(|()| (buffer.vertices.len(), buffer.edge_neighbor_globals.len()))
    } else {
        None
    };
    ProbeCell {
        generator: generator_idx,
        ok: result.is_ok(),
        failure: result.err().map(|err| err.failure),
        neighbors_processed: counters.neighbors_processed,
        final_edges: ctx.output_buffer.vertices.len(),
        knn_exhausted: counters.knn_exhausted,
        bounded: ctx.builder.is_bounded(),
        fallback_projection: counters.fallback_projection,
        fallback_polygon_cap: counters.fallback_polygon_cap,
        fallback_all_constraints: counters.fallback_all_constraints,
        spherical_extract_vertices: spherical_extract.map(|(vertices, _)| vertices),
        spherical_extract_edges: spherical_extract.map(|(_, edges)| edges),
    }
}

fn signature(buffer: &crate::live_dedup::CellOutputBuffer) -> CellSignature {
    let mut vertex_keys: Vec<[u32; 3]> = buffer.vertices.iter().map(|(key, _)| *key).collect();
    vertex_keys.sort_unstable();
    let mut edge_neighbors = buffer.edge_neighbor_globals.clone();
    edge_neighbors.sort_unstable();
    CellSignature {
        vertex_keys,
        edge_neighbors,
    }
}

fn early_probe_thresholds(n: usize) -> Vec<usize> {
    let mut thresholds = vec![
        8usize, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
    ];
    let tail = n.saturating_sub(1);
    for divisor in [8usize, 4, 2] {
        thresholds.push((tail / divisor).max(1));
    }
    thresholds.push(tail);
    thresholds.retain(|&x| x > 0 && x <= tail);
    thresholds.sort_unstable();
    thresholds.dedup();
    thresholds
}

fn maybe_record_early_extract(
    points: &[Vec3],
    builder: &Topo2DBuilder,
    counters: &BuildCounters,
    thresholds: &[usize],
    next_threshold: &mut usize,
    successes: &mut Vec<(EarlyExtractHit, CellSignature)>,
) {
    if *next_threshold >= thresholds.len()
        || counters.neighbors_processed < thresholds[*next_threshold]
        || builder.is_bounded()
        || builder.is_failed()
    {
        return;
    }
    while *next_threshold + 1 < thresholds.len()
        && counters.neighbors_processed >= thresholds[*next_threshold + 1]
    {
        *next_threshold += 1;
    }
    *next_threshold += 1;

    let mut buffer = crate::live_dedup::CellOutputBuffer::default();
    if builder
        .to_vertex_data_from_all_constraints(points, &mut buffer)
        .is_err()
    {
        return;
    }

    successes.push((
        EarlyExtractHit {
            neighbors_processed: counters.neighbors_processed,
            accepted_constraints: builder.accepted_constraint_count(),
            edges: buffer.vertices.len(),
        },
        signature(&buffer),
    ));
}

fn probe_early_extraction_cell(
    points: &[Vec3],
    grid: &CubeMapGrid,
    generator_idx: usize,
) -> EarlyProbeCell {
    let policy = TerminationConfig::default().packed_policy(points.len());
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);
    let mut ctx = CellBuildContext::new(grid, policy);
    let pos_slots = grid.point_pos_slots();
    let mut trace = BuildTrace::new();
    let mut counters = BuildCounters::new();
    let thresholds = early_probe_thresholds(points.len());
    let mut next_threshold = 0usize;
    let mut successes = Vec::new();

    ctx.builder.reset(generator_idx, points[generator_idx]);
    ctx.attempted_neighbors.clear();
    ctx.output_buffer.clear();

    clip_seed_neighbors(
        &mut ctx,
        points,
        grid,
        pos_slots,
        &[],
        &mut trace,
        &mut counters,
    );

    {
        let mut stream = DirectedNeighborStream::new(
            grid,
            points,
            generator_idx,
            &mut ctx.scratch,
            directed_ctx,
            None,
        );
        while !counters.terminated && !ctx.builder.is_failed() {
            let frontier = probe_frontier(
                &mut stream,
                &mut ctx.packed_chunk,
                &mut counters.used_knn,
                &mut counters.knn_stage,
                &mut counters.knn_query_time,
            );

            match frontier {
                DirectedNeighborFrontier::ExactBatch(batch) => {
                    clip_batch(
                        &mut StreamPhase {
                            builder: &mut ctx.builder,
                            packed_chunk: &mut ctx.packed_chunk,
                            attempted_neighbors: &mut ctx.attempted_neighbors,
                            force_fallback_after_neighbors_processed: &mut ctx
                                .force_fallback_after_neighbors_processed,
                        },
                        batch,
                        points,
                        pos_slots,
                        generator_idx,
                        &mut trace,
                        &mut counters,
                    );
                    stream.advance_frontier();
                    maybe_record_early_extract(
                        points,
                        &ctx.builder,
                        &counters,
                        &thresholds,
                        &mut next_threshold,
                        &mut successes,
                    );

                    if !counters.terminated && !ctx.builder.is_failed() && ctx.builder.is_bounded()
                    {
                        counters.terminated = super::maybe_terminate_or_advance_frontier(
                            &mut stream,
                            &mut ctx.packed_chunk,
                            &mut ctx.builder,
                            pos_slots,
                            &mut counters,
                        );
                    }
                }
                DirectedNeighborFrontier::UnknownButBounded { dot_upper_bound } => {
                    if ctx.builder.is_bounded() && ctx.builder.can_terminate(dot_upper_bound) {
                        counters.terminated = true;
                    } else {
                        stream.advance_frontier();
                    }
                }
                DirectedNeighborFrontier::Exhausted => break,
            }
        }
        counters.absorb_stream(&stream);
    }

    let result = finish_cell(&mut ctx, points, grid, generator_idx, &trace, &mut counters);
    let final_signature = result.as_ref().ok().map(|_| signature(&ctx.output_buffer));
    let first_success = successes.first().map(|(hit, _)| hit.clone());
    let first_final_match = final_signature.and_then(|final_signature| {
        successes
            .iter()
            .find(|(_, candidate)| *candidate == final_signature)
            .map(|(hit, _)| hit.clone())
    });

    EarlyProbeCell {
        generator: generator_idx,
        ok: result.is_ok(),
        failure: result.err().map(|err| err.failure),
        neighbors_processed: counters.neighbors_processed,
        final_edges: ctx.output_buffer.vertices.len(),
        knn_exhausted: counters.knn_exhausted,
        fallback_all_constraints: counters.fallback_all_constraints,
        first_success,
        first_final_match,
        successes: successes.len(),
    }
}

fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn summarize_usize(xs: &mut [usize]) -> String {
    if xs.is_empty() {
        return "n=0".to_string();
    }
    xs.sort_unstable();
    let sum: usize = xs.iter().sum();
    format!(
        "n={} min={} p50={} p90={} p99={} max={} mean={:.1}",
        xs.len(),
        xs[0],
        percentile(xs, 0.50),
        percentile(xs, 0.90),
        percentile(xs, 0.99),
        xs[xs.len() - 1],
        sum as f64 / xs.len() as f64
    )
}

fn summarize_hits<'a>(
    cells: impl Iterator<Item = &'a EarlyProbeCell>,
    pick: impl Fn(&'a EarlyProbeCell) -> Option<&'a EarlyExtractHit>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut neighbors = Vec::new();
    let mut constraints = Vec::new();
    let mut edges = Vec::new();
    for hit in cells.filter_map(pick) {
        neighbors.push(hit.neighbors_processed);
        constraints.push(hit.accepted_constraints);
        edges.push(hit.edges);
    }
    (neighbors, constraints, edges)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_target_indices(n: usize) -> Vec<usize> {
    if let Ok(raw) = std::env::var("VORONOI_MESH_PROBE_TARGETS") {
        let mut targets: Vec<usize> = raw
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .filter(|&idx| idx < n)
            .collect();
        targets.sort_unstable();
        targets.dedup();
        return targets;
    }

    let mut targets: Vec<usize> = (0..n.min(12)).collect();
    for idx in [n / 4, n / 2, 3 * n / 4, n.saturating_sub(1)] {
        if idx < n {
            targets.push(idx);
        }
    }
    targets.sort_unstable();
    targets.dedup();
    targets
}

#[test]
#[ignore = "diagnostic: checkpointed all-constraints extraction before exhaustion"]
fn probe_early_all_constraints_trigger_points() {
    let mut cases = vec![
        ("fib_100", fibonacci_points(100)),
        ("fib_500", fibonacci_points(500)),
        ("great_circle_50", great_circle_points(50, 0.0)),
        ("great_circle_jitter_50", great_circle_points(50, 0.01)),
        ("hemisphere_100", hemisphere_points(100)),
        ("hemisphere_500", hemisphere_points(500)),
        ("latitude_ring_64", pole_with_latitude_ring(64, 0.5)),
    ];
    if std::env::var_os("VORONOI_MESH_PROBE_LARGE").is_some() {
        cases.extend([
            ("fib_2k", fibonacci_points(2_000)),
            ("great_circle_200", great_circle_points(200, 0.0)),
            ("great_circle_jitter_200", great_circle_points(200, 0.01)),
            ("hemisphere_2k", hemisphere_points(2_000)),
            ("latitude_ring_256", pole_with_latitude_ring(256, 0.5)),
        ]);
    }

    for (name, points) in cases {
        let grid = CubeMapGrid::new(&points, crate::policy::knn_grid_resolution(points.len()));
        let mut cells: Vec<EarlyProbeCell> = (0..points.len())
            .map(|generator_idx| probe_early_extraction_cell(&points, &grid, generator_idx))
            .collect();
        let ok = cells.iter().filter(|c| c.ok).count();
        let err = cells.len() - ok;
        let exhausted = cells.iter().filter(|c| c.knn_exhausted).count();
        let all_constraints: usize = cells.iter().map(|c| c.fallback_all_constraints).sum();
        let first_success_count = cells.iter().filter(|c| c.first_success.is_some()).count();
        let first_match_count = cells
            .iter()
            .filter(|c| c.first_final_match.is_some())
            .count();
        let total_successes: usize = cells.iter().map(|c| c.successes).sum();
        let mut failure_counts = std::collections::BTreeMap::new();
        for failure in cells.iter().filter_map(|c| c.failure) {
            *failure_counts
                .entry(format!("{failure:?}"))
                .or_insert(0usize) += 1;
        }

        let (mut success_neighbors, mut success_constraints, mut success_edges) =
            summarize_hits(cells.iter(), |cell| cell.first_success.as_ref());
        let (mut match_neighbors, mut match_constraints, mut match_edges) =
            summarize_hits(cells.iter(), |cell| cell.first_final_match.as_ref());
        cells.sort_by_key(|c| std::cmp::Reverse(c.neighbors_processed));
        let top: Vec<String> = cells
            .iter()
            .take(5)
            .map(|c| {
                format!(
                    "{}:{}:{}:edges={}{}{}",
                    c.generator,
                    c.neighbors_processed,
                    if c.ok { "ok" } else { "err" },
                    c.final_edges,
                    c.first_success
                        .as_ref()
                        .map(|h| format!(":first={}c{}", h.neighbors_processed, h.edges))
                        .unwrap_or_else(String::new),
                    c.first_final_match
                        .as_ref()
                        .map(|h| format!(":match={}c{}", h.neighbors_processed, h.edges))
                        .unwrap_or_else(String::new),
                )
            })
            .collect();

        eprintln!(
            "EARLYPROBE {name}: cells={} ok={} err={} exhausted={} fallback_all_constraints={} \
             first_success={} first_final_match={} total_successes={} failures={:?}",
            points.len(),
            ok,
            err,
            exhausted,
            all_constraints,
            first_success_count,
            first_match_count,
            total_successes,
            failure_counts
        );
        eprintln!(
            "EARLYPROBE {name}: first_success_neighbors {}",
            summarize_usize(&mut success_neighbors)
        );
        eprintln!(
            "EARLYPROBE {name}: first_success_constraints {}",
            summarize_usize(&mut success_constraints)
        );
        eprintln!(
            "EARLYPROBE {name}: first_success_edges {}",
            summarize_usize(&mut success_edges)
        );
        eprintln!(
            "EARLYPROBE {name}: first_final_match_neighbors {}",
            summarize_usize(&mut match_neighbors)
        );
        eprintln!(
            "EARLYPROBE {name}: first_final_match_constraints {}",
            summarize_usize(&mut match_constraints)
        );
        eprintln!(
            "EARLYPROBE {name}: first_final_match_edges {} top={}",
            summarize_usize(&mut match_edges),
            top.join(",")
        );
    }
}

#[test]
#[ignore = "diagnostic: targeted large-N hemisphere cells for exhaustion/fallback scaling"]
fn probe_large_hemisphere_target_cells() {
    let n = env_usize("VORONOI_MESH_PROBE_N", 100_000);
    let points = hemisphere_points(n);
    let targets = parse_target_indices(n);
    let grid = CubeMapGrid::new(&points, crate::policy::knn_grid_resolution(points.len()));

    eprintln!(
        "LARGEPROBE hemisphere: n={} targets={:?}",
        points.len(),
        targets
    );
    for generator_idx in targets {
        let cell = probe_early_extraction_cell(&points, &grid, generator_idx);
        eprintln!(
            "LARGEPROBE hemisphere cell={}: ok={} failure={:?} neighbors={} final_edges={} \
             exhausted={} fallback_all_constraints={} successes={} first_success={} first_match={}",
            cell.generator,
            cell.ok,
            cell.failure,
            cell.neighbors_processed,
            cell.final_edges,
            cell.knn_exhausted,
            cell.fallback_all_constraints,
            cell.successes,
            format_hit(cell.first_success.as_ref()),
            format_hit(cell.first_final_match.as_ref()),
        );
    }
}

fn format_hit(hit: Option<&EarlyExtractHit>) -> String {
    hit.map(|hit| {
        format!(
            "neighbors:{} constraints:{} edges:{}",
            hit.neighbors_processed, hit.accepted_constraints, hit.edges
        )
    })
    .unwrap_or_else(|| "none".to_string())
}

#[test]
fn projection_invalid_stays_distinct_from_exhausted_unbounded() {
    assert_eq!(
        classify_terminal_failure(false, Some(CellFailure::ProjectionInvalid), true),
        Some(CellFailure::ProjectionInvalid)
    );
    assert_eq!(
        classify_terminal_failure(false, None, true),
        Some(CellFailure::UnboundedAfterExhaustion)
    );
}

#[test]
fn too_many_vertices_is_a_structured_failure() {
    assert_eq!(
        classify_terminal_failure(true, Some(CellFailure::TooManyVertices), false),
        Some(CellFailure::TooManyVertices)
    );
}

#[test]
fn bounded_nonfailed_cell_has_no_terminal_failure() {
    assert_eq!(classify_terminal_failure(true, None, true), None);
    assert_eq!(classify_terminal_failure(true, None, false), None);
}

#[test]
#[ignore = "diagnostic: per-cell neighbor counts before bounded success or unbounded exhaustion"]
fn probe_unbounded_exhaustion_neighbor_counts() {
    let mut cases = vec![
        ("fib_100", fibonacci_points(100)),
        ("fib_500", fibonacci_points(500)),
        ("great_circle_50", great_circle_points(50, 0.0)),
        ("great_circle_jitter_50", great_circle_points(50, 0.01)),
        ("hemisphere_100", hemisphere_points(100)),
        ("hemisphere_500", hemisphere_points(500)),
        ("latitude_ring_32", pole_with_latitude_ring(32, 0.5)),
        ("latitude_ring_64", pole_with_latitude_ring(64, 0.5)),
    ];
    if std::env::var_os("VORONOI_MESH_PROBE_LARGE").is_some() {
        cases.extend([
            ("fib_2k", fibonacci_points(2_000)),
            ("great_circle_200", great_circle_points(200, 0.0)),
            ("great_circle_jitter_200", great_circle_points(200, 0.01)),
            ("hemisphere_2k", hemisphere_points(2_000)),
            ("latitude_ring_256", pole_with_latitude_ring(256, 0.5)),
        ]);
    }

    for (name, points) in cases {
        let grid = CubeMapGrid::new(&points, crate::policy::knn_grid_resolution(points.len()));
        let mut cells: Vec<ProbeCell> = (0..points.len())
            .map(|generator_idx| probe_cell(&points, &grid, generator_idx))
            .collect();
        let mut ok_neighbors: Vec<usize> = cells
            .iter()
            .filter(|c| c.ok)
            .map(|c| c.neighbors_processed)
            .collect();
        let mut ok_edges: Vec<usize> = cells
            .iter()
            .filter(|c| c.ok)
            .map(|c| c.final_edges)
            .collect();
        let mut fail_neighbors: Vec<usize> = cells
            .iter()
            .filter(|c| !c.ok)
            .map(|c| c.neighbors_processed)
            .collect();
        let exhausted = cells.iter().filter(|c| c.knn_exhausted).count();
        let bounded_failures = cells.iter().filter(|c| !c.ok && c.bounded).count();
        let fallback_projection: usize = cells.iter().map(|c| c.fallback_projection).sum();
        let fallback_polygon_cap: usize = cells.iter().map(|c| c.fallback_polygon_cap).sum();
        let fallback_all_constraints: usize =
            cells.iter().map(|c| c.fallback_all_constraints).sum();
        let spherical_extract_ok = cells
            .iter()
            .filter(|c| !c.ok && c.spherical_extract_vertices.is_some())
            .count();
        let mut spherical_extract_vertices: Vec<usize> = cells
            .iter()
            .filter_map(|c| c.spherical_extract_vertices)
            .collect();
        let mut spherical_extract_edges: Vec<usize> = cells
            .iter()
            .filter_map(|c| c.spherical_extract_edges)
            .collect();
        let mut failure_counts = std::collections::BTreeMap::new();
        for failure in cells.iter().filter_map(|c| c.failure) {
            *failure_counts
                .entry(format!("{failure:?}"))
                .or_insert(0usize) += 1;
        }
        cells.sort_by_key(|c| std::cmp::Reverse(c.neighbors_processed));
        let top: Vec<String> = cells
            .iter()
            .take(5)
            .map(|c| {
                format!(
                    "{}:{}:{}{}",
                    c.generator,
                    c.neighbors_processed,
                    if c.ok { "ok" } else { "err" },
                    c.failure
                        .map(|f| format!(":{f:?}"))
                        .unwrap_or_else(String::new)
                )
            })
            .collect();
        eprintln!(
            "CELLPROBE {name}: cells={} ok={} err={} exhausted={} bounded_failures={} \
             fallback_projection={} fallback_polygon_cap={} fallback_all_constraints={} \
             spherical_extract_ok={} failures={:?}",
            points.len(),
            ok_neighbors.len(),
            fail_neighbors.len(),
            exhausted,
            bounded_failures,
            fallback_projection,
            fallback_polygon_cap,
            fallback_all_constraints,
            spherical_extract_ok,
            failure_counts
        );
        eprintln!(
            "CELLPROBE {name}: ok_neighbors {}",
            summarize_usize(&mut ok_neighbors)
        );
        eprintln!(
            "CELLPROBE {name}: ok_edges {}",
            summarize_usize(&mut ok_edges)
        );
        eprintln!(
            "CELLPROBE {name}: fail_neighbors {} top={}",
            summarize_usize(&mut fail_neighbors),
            top.join(",")
        );
        eprintln!(
            "CELLPROBE {name}: spherical_extract_vertices {}",
            summarize_usize(&mut spherical_extract_vertices)
        );
        eprintln!(
            "CELLPROBE {name}: spherical_extract_edges {}",
            summarize_usize(&mut spherical_extract_edges)
        );
    }
}

#[test]
#[ignore = "diagnostic: projection fallback incidence on cap-only and anchored-cap cases"]
fn probe_projection_fallback_cases() {
    let cases = [
        ("cap_only_100_r0.1", cap_fibonacci_points(100, 0.1)),
        ("cap_only_100_r0.5", cap_fibonacci_points(100, 0.5)),
        ("cap_only_100_r1.0", cap_fibonacci_points(100, 1.0)),
        ("cap_only_100_r1.5", cap_fibonacci_points(100, 1.5)),
        ("cap_only_500_r0.1", cap_fibonacci_points(500, 0.1)),
        ("cap_only_500_r1.0", cap_fibonacci_points(500, 1.0)),
        ("cap_only_500_r1.5", cap_fibonacci_points(500, 1.5)),
        ("cap_antipode_100_r0.1", cap_with_antipode(100, 0.1)),
        ("cap_antipode_500_r0.1", cap_with_antipode(500, 0.1)),
        ("cap_antipode_500_r1.0", cap_with_antipode(500, 1.0)),
        ("cap_octa_500_r0.1", cap_with_octahedral_anchors(500, 0.1)),
        ("cap_octa_500_r1.0", cap_with_octahedral_anchors(500, 1.0)),
    ];

    for (name, points) in cases {
        let grid = CubeMapGrid::new(&points, crate::policy::knn_grid_resolution(points.len()));
        let mut cells: Vec<ProbeCell> = (0..points.len())
            .map(|generator_idx| probe_cell(&points, &grid, generator_idx))
            .collect();
        let ok = cells.iter().filter(|c| c.ok).count();
        let err = cells.len() - ok;
        let exhausted = cells.iter().filter(|c| c.knn_exhausted).count();
        let fallback_projection: usize = cells.iter().map(|c| c.fallback_projection).sum();
        let fallback_polygon_cap: usize = cells.iter().map(|c| c.fallback_polygon_cap).sum();
        let fallback_all_constraints: usize =
            cells.iter().map(|c| c.fallback_all_constraints).sum();
        let mut neighbors: Vec<usize> = cells.iter().map(|c| c.neighbors_processed).collect();
        let mut edges: Vec<usize> = cells
            .iter()
            .filter(|c| c.ok)
            .map(|c| c.final_edges)
            .collect();
        let mut failure_counts = std::collections::BTreeMap::new();
        for failure in cells.iter().filter_map(|c| c.failure) {
            *failure_counts
                .entry(format!("{failure:?}"))
                .or_insert(0usize) += 1;
        }
        cells.sort_by_key(|c| std::cmp::Reverse(c.neighbors_processed));
        let top: Vec<String> = cells
            .iter()
            .take(5)
            .map(|c| {
                format!(
                    "{}:{}:{}:edges={}{}",
                    c.generator,
                    c.neighbors_processed,
                    if c.ok { "ok" } else { "err" },
                    c.final_edges,
                    c.failure
                        .map(|failure| format!(":{failure:?}"))
                        .unwrap_or_else(String::new)
                )
            })
            .collect();

        eprintln!(
            "PROJPROBE {name}: cells={} ok={} err={} exhausted={} fallback_projection={} \
             fallback_polygon_cap={} fallback_all_constraints={} failures={:?}",
            points.len(),
            ok,
            err,
            exhausted,
            fallback_projection,
            fallback_polygon_cap,
            fallback_all_constraints,
            failure_counts
        );
        eprintln!(
            "PROJPROBE {name}: neighbors {}",
            summarize_usize(&mut neighbors)
        );
        eprintln!(
            "PROJPROBE {name}: ok_edges {} top={}",
            summarize_usize(&mut edges),
            top.join(",")
        );
    }
}

#[test]
fn direct_cursor_builds_normal_cell() {
    let points = octahedron_points();
    let grid = CubeMapGrid::new(&points, 4);
    let policy = TerminationConfig::default().packed_policy(points.len());
    let mut ctx = CellBuildContext::new(&grid, policy);
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);

    let stats = build_cell_into(
        &mut ctx,
        CellBuildRequest {
            points: &points,
            grid: &grid,
            generator_idx: 0,
            directed_ctx,
            packed: None,
            incoming_checks: &[],
        },
    )
    .expect("cell build should succeed");

    assert!(ctx.output_buffer().vertices.len() >= 3);
    assert!(!stats.knn_exhausted || !stats.did_packed);
}

fn canonical_test_point(p: glam::DVec3) -> Vec3 {
    let unit = p.normalize();
    let input = Vec3::new(unit.x as f32, unit.y as f32, unit.z as f32);
    let canonical = input.as_dvec3().normalize();
    Vec3::new(canonical.x as f32, canonical.y as f32, canonical.z as f32)
}

fn audit_uniform_points(seed: u64, n: usize) -> Vec<Vec3> {
    let mut state = seed;
    let mut sample = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) * (2.0 / ((1_u64 << 53) as f64)) - 1.0
    };
    (0..n)
        .map(|_| loop {
            let p = glam::DVec3::new(sample(), sample(), sample());
            if p.length_squared() > 1.0e-12 {
                break canonical_test_point(p);
            }
        })
        .collect()
}

fn audit_clustered_points(seed: u64, n: usize) -> Vec<Vec3> {
    let mut points = audit_uniform_points(seed ^ 0xd1b5_4a32_d192_ed03, n);
    let center = glam::DVec3::new(0.31, 0.52, 0.79).normalize();
    let mut state = seed;
    let mut sample = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) * (2.0 / ((1_u64 << 53) as f64)) - 1.0
    };
    for point in points.iter_mut().take(n / 3) {
        let jitter = glam::DVec3::new(sample(), sample(), sample()) * 0.025;
        *point = canonical_test_point(center + jitter);
    }
    points
}

fn assert_all_omitted_constraints_are_unchanged(
    ctx: &CellBuildContext,
    points: &[Vec3],
    generator_idx: usize,
) -> usize {
    assert!(ctx.builder.is_bounded());
    assert!(
        !ctx.builder.is_fallback(),
        "oracle fixture unexpectedly fell back"
    );
    let accepted: std::collections::HashSet<usize> = ctx.builder.neighbor_indices_iter().collect();
    let mut replayed = 0usize;
    for (neighbor_idx, &neighbor) in points.iter().enumerate() {
        if neighbor_idx == generator_idx || accepted.contains(&neighbor_idx) {
            continue;
        }
        replayed += 1;
        assert!(
            ctx.builder.candidate_would_be_unchanged(neighbor),
            "termination omitted a cutting constraint: generator={generator_idx}, \
             neighbor={neighbor_idx}, accepted={}",
            accepted.len()
        );
    }
    replayed
}

#[test]
fn shell_termination_survives_all_omitted_constraints() {
    let cases = [
        audit_uniform_points(0x9e37_79b9_7f4a_7c15, 96),
        audit_clustered_points(0xa076_1d64_78bd_642f, 120),
    ];
    let mut cells = 0usize;
    let mut replayed = 0usize;
    let mut terminated = 0usize;

    for points in cases {
        let grid = CubeMapGrid::new(&points, 8);
        let policy = TerminationConfig::default().packed_policy(points.len());
        let slot_map = vec![0_u32; points.len()];
        for generator_idx in 0..points.len() {
            let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &slot_map, 24, (1 << 24) - 1);
            let mut ctx = CellBuildContext::new(&grid, policy);
            let stats = build_cell_into(
                &mut ctx,
                CellBuildRequest {
                    points: &points,
                    grid: &grid,
                    generator_idx,
                    directed_ctx,
                    packed: None,
                    incoming_checks: &[],
                },
            )
            .expect("shell oracle cell should build");
            cells += 1;
            replayed += assert_all_omitted_constraints_are_unchanged(&ctx, &points, generator_idx);
            terminated +=
                usize::from(stats.termination_checkpoint == Some(TerminationCheckpoint::Shell));
        }
    }

    assert_eq!(
        terminated, cells,
        "every oracle cell should terminate by certificate"
    );
    assert!(replayed > cells, "oracle must replay omitted constraints");
}

#[test]
#[allow(clippy::default_constructed_unit_structs)]
fn packed_termination_checkpoints_survive_all_omitted_constraints() {
    let mut checkpoints = std::collections::HashSet::new();
    let mut replayed = 0usize;

    for seed in 0..64_u64 {
        let points = audit_clustered_points(0xe703_7ed1_a0b4_28db ^ seed, 288);
        let grid = CubeMapGrid::new(&points, 8);
        let (cell, start, end) = (0..grid.cell_offsets().len() - 1)
            .map(|cell| {
                let start = grid.cell_offsets()[cell] as usize;
                let end = grid.cell_offsets()[cell + 1] as usize;
                (cell, start, end)
            })
            .max_by_key(|&(_, start, end)| end - start)
            .expect("grid has cells");
        assert!(
            end - start >= 16,
            "packed fixture did not form a dense group"
        );

        let queries: Vec<u32> = (start..end).map(|slot| slot as u32).collect();
        let mut slot_map = vec![1_u32 << 24; points.len()];
        for (local, slot) in (start..end).enumerate() {
            slot_map[slot] = local as u32;
        }
        let layout = PackedSlotLayout::new(&slot_map, 24, (1 << 24) - 1);
        let group =
            PackedGroupInput::new(cell, 0, grid.cell_offsets()[cell], queries.len(), 0, layout);
        let mut packed_scratch = PackedKnnCellScratch::new();
        let mut timings = PackedKnnTimings::default();
        let PreparedPackedGroupStatus::Ready(mut prepared) =
            packed_scratch.prepare_group_directed(&grid, group, &mut timings)
        else {
            panic!("packed oracle group unexpectedly chose the slow path");
        };

        // The first query local has no earlier same-bin center points, so its
        // directed eligible set is the complete generator set.
        let query_slot = queries[0];
        let generator_idx = grid.point_indices()[query_slot as usize] as usize;
        let directed_ctx = DirectedEligibility::from_layout(0, 0, layout);
        let policy = PackedNeighborPolicy::for_point_count(points.len());
        let packed = crate::cube_grid::PackedQuery::new(&mut prepared, &mut timings, 0, policy);
        let mut ctx = CellBuildContext::new(&grid, policy);
        let stats = build_cell_into(
            &mut ctx,
            CellBuildRequest {
                points: &points,
                grid: &grid,
                generator_idx,
                directed_ctx,
                packed: Some(packed),
                incoming_checks: &[],
            },
        )
        .expect("packed oracle cell should build");
        assert!(stats.did_packed);
        replayed += assert_all_omitted_constraints_are_unchanged(&ctx, &points, generator_idx);
        if let Some(checkpoint) = stats.termination_checkpoint {
            checkpoints.insert(checkpoint);
        }
        if checkpoints.contains(&TerminationCheckpoint::PackedPreBatch)
            && checkpoints.contains(&TerminationCheckpoint::PackedMidBatch)
            && checkpoints.contains(&TerminationCheckpoint::PackedPostBatch)
        {
            break;
        }
    }

    assert!(
        replayed > 0,
        "packed oracle must replay omitted constraints"
    );
    for expected in [
        TerminationCheckpoint::PackedPreBatch,
        TerminationCheckpoint::PackedMidBatch,
        TerminationCheckpoint::PackedPostBatch,
    ] {
        assert!(
            checkpoints.contains(&expected),
            "packed corpus did not exercise {expected:?}; saw {checkpoints:?}"
        );
    }
}

#[test]
fn exhausted_chart_replays_discarded_horizon_constraints_spherically() {
    let delta = 4.0e-7f32;
    let mut points = vec![Vec3::Z];
    for i in 0..3 {
        let angle = std::f32::consts::TAU * i as f32 / 3.0;
        points.push(Vec3::new(delta * angle.cos(), delta * angle.sin(), -1.0).normalize());
    }

    let grid = CubeMapGrid::new(&points, 4);
    let policy = TerminationConfig::default().packed_policy(points.len());
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);
    let mut ctx = CellBuildContext::new(&grid, policy);

    let stats = build_cell_into(
        &mut ctx,
        CellBuildRequest {
            points: &points,
            grid: &grid,
            generator_idx: 0,
            directed_ctx,
            packed: None,
            incoming_checks: &[],
        },
    )
    .expect("unrestricted spherical replay should recover the horizon cell");

    assert!(stats.knn_exhausted);
    assert_eq!(stats.fallback_all_constraints, 1);
    assert!(ctx.builder.is_fallback());
    assert_eq!(ctx.output_buffer.vertices.len(), 3);

    let mut edge_neighbors = ctx.output_buffer.edge_neighbor_globals.clone();
    edge_neighbors.sort_unstable();
    assert_eq!(edge_neighbors, [1, 2, 3]);

    let generator = points[0].as_dvec3().normalize();
    for &(_, vertex) in &ctx.output_buffer.vertices {
        let vertex = vertex.as_dvec3().normalize();
        for &neighbor in &points[1..] {
            let neighbor = neighbor.as_dvec3().normalize();
            assert!(
                generator.dot(vertex) >= neighbor.dot(vertex) - 2.0e-6,
                "recovered vertex violates a replayed spherical constraint"
            );
        }
    }
}

#[test]
fn projection_invalid_detail_includes_replay_payload_summary() {
    let g = Vec3::new(0.0, 0.0, 1.0);
    let mut builder = Topo2DBuilder::new(17, g);

    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

    builder
        .clip_with_slot_edgecheck_policy(11, 21, h1)
        .expect("edgecheck clip should apply");
    builder
        .clip_with_slot_policy(12, 22, h2)
        .expect("normal clip should apply");
    builder
        .clip_with_slot_policy(13, 23, h3)
        .expect("normal clip should apply");

    let detail = fallback_detail(
        &builder,
        CellFailure::ProjectionInvalid,
        Topo2DBuilder::fallback_request_for_failure(CellFailure::ProjectionInvalid),
    )
    .expect("projection invalid should produce fallback detail");

    assert!(detail.contains("ProjectionLimit"));
    assert!(detail.contains("replay_constraints=3"));
    assert!(detail.contains("replay_generator_idx=17"));
}

#[test]
fn forced_handoff_mid_build_still_finishes_the_cell() {
    let points = octahedron_points();
    let grid = CubeMapGrid::new(&points, 4);
    let policy = TerminationConfig::default().packed_policy(points.len());
    let fake_slot_map = vec![0u32; points.len()];
    let directed_ctx = DirectedEligibility::new(u8::MAX, 0, &fake_slot_map, 0, 0);
    let mut ctx = CellBuildContext::new(&grid, policy);
    ctx.force_fallback_after_neighbors_processed = Some(2);

    let stats = build_cell_into(
        &mut ctx,
        CellBuildRequest {
            points: &points,
            grid: &grid,
            generator_idx: 0,
            directed_ctx,
            packed: None,
            incoming_checks: &[],
        },
    )
    .expect("cell build should succeed even after forced mid-build fallback");

    assert!(
        ctx.builder.is_fallback(),
        "builder should have handed off to fallback"
    );
    assert!(
        ctx.builder.is_bounded(),
        "fallback-built cell should be bounded"
    );
    assert_eq!(ctx.builder.failure(), None);
    assert!(
        ctx.output_buffer().vertices.len() >= 3,
        "fallback-built cell should extract vertices",
    );
    let mut stored_positions = Vec::new();
    for &(_, position) in &ctx.output_buffer().vertices {
        let bits = (
            position.x.to_bits(),
            position.y.to_bits(),
            position.z.to_bits(),
        );
        if !stored_positions.contains(&bits) {
            stored_positions.push(bits);
        }
    }
    assert!(
        stored_positions.len() >= 3,
        "forced spherical handoff must retain three exact stored positions"
    );
    assert!(ctx.builder.accepted_constraint_count() >= 3);
    assert!(!stats.knn_exhausted || !stats.did_packed);
}
