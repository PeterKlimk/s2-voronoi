use super::failure::classify_terminal_failure;
use super::fallback_detail;
use super::{
    build_cell_into, clip_seed_neighbors, consume_stream, finish_cell, BuildCounters, BuildTrace,
    CellBuildContext, CellBuildRequest, StreamPhase,
};
use crate::cube_grid::{CubeMapGrid, DirectedEligibility, DirectedNeighborStream};
use crate::knn_clipping::cell_build::CellFailure;
use crate::knn_clipping::topo2d::builder::{sort3_u32, FallbackConstraint, TangentBasis};
use crate::knn_clipping::topo2d::Topo2DBuilder;
use crate::knn_clipping::TerminationConfig;
use glam::{DVec3, Vec3};

fn octahedron_points() -> Vec<Vec3> {
    vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
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
    spherical_extract_vertices: Option<usize>,
    spherical_extract_edges: Option<usize>,
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

    clip_seed_neighbors(&mut ctx, points, pos_slots, &[], &mut trace, &mut counters);

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

    let result = finish_cell(&mut ctx, points, generator_idx, &trace, &mut counters);
    let spherical_extract = if result.is_err() {
        let constraints = ctx.builder.accepted_spherical_constraints(points);
        extract_from_spherical_constraints(generator_idx, points[generator_idx], &constraints)
            .map(|cell| (cell.vertices.len(), cell.edge_neighbors.len()))
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
        spherical_extract_vertices: spherical_extract.map(|(vertices, _)| vertices),
        spherical_extract_edges: spherical_extract.map(|(_, edges)| edges),
    }
}

#[derive(Debug, Clone)]
struct ExtractedSphericalCell {
    vertices: Vec<([u32; 3], Vec3)>,
    edge_neighbors: Vec<u32>,
}

#[derive(Clone, Copy)]
struct SphericalCandidate {
    position: Vec3,
    plane_a: usize,
    plane_b: usize,
}

fn extract_from_spherical_constraints(
    generator_idx: usize,
    generator: Vec3,
    constraints: &[FallbackConstraint],
) -> Option<ExtractedSphericalCell> {
    if constraints.len() < 3 {
        return None;
    }
    let mut candidates = Vec::new();
    for a in 0..constraints.len() {
        for b in a + 1..constraints.len() {
            let cross = constraints[a].normal.cross(constraints[b].normal);
            let len2 = cross.length_squared();
            if !len2.is_finite() || len2 <= 1e-24 {
                continue;
            }
            let inv_len = len2.sqrt().recip();
            for sign in [1.0, -1.0] {
                let dir = cross * (sign * inv_len);
                if constraints.iter().all(|constraint| {
                    constraint.normal.dot(dir) >= -crate::tolerances::FALLBACK_PLANE_TOL
                }) {
                    let p = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
                    push_spherical_candidate(
                        &mut candidates,
                        SphericalCandidate {
                            position: p,
                            plane_a: a,
                            plane_b: b,
                        },
                    );
                }
            }
        }
    }
    if candidates.len() < 3 {
        return None;
    }

    let basis = TangentBasis::new(DVec3::new(
        generator.x as f64,
        generator.y as f64,
        generator.z as f64,
    ));
    candidates.sort_by(|a, b| {
        let pa = DVec3::new(
            a.position.x as f64,
            a.position.y as f64,
            a.position.z as f64,
        );
        let pb = DVec3::new(
            b.position.x as f64,
            b.position.y as f64,
            b.position.z as f64,
        );
        let aa = pa.dot(basis.t2).atan2(pa.dot(basis.t1));
        let ab = pb.dot(basis.t2).atan2(pb.dot(basis.t1));
        aa.total_cmp(&ab)
    });

    let gen_idx = generator_idx as u32;
    let mut vertices = Vec::with_capacity(candidates.len());
    let mut edge_neighbors = Vec::with_capacity(candidates.len());
    for i in 0..candidates.len() {
        let c = candidates[i];
        let key = sort3_u32(
            gen_idx,
            constraints[c.plane_a].neighbor_idx as u32,
            constraints[c.plane_b].neighbor_idx as u32,
        );
        vertices.push((key, c.position));

        let next = candidates[(i + 1) % candidates.len()];
        let shared = [c.plane_a, c.plane_b]
            .into_iter()
            .find(|&plane| plane == next.plane_a || plane == next.plane_b)?;
        edge_neighbors.push(constraints[shared].neighbor_idx as u32);
    }

    Some(ExtractedSphericalCell {
        vertices,
        edge_neighbors,
    })
}

fn push_spherical_candidate(out: &mut Vec<SphericalCandidate>, candidate: SphericalCandidate) {
    const DEDUP_LEN2: f32 = 1e-12;
    if out
        .iter()
        .any(|existing| (existing.position - candidate.position).length_squared() <= DEDUP_LEN2)
    {
        return;
    }
    out.push(candidate);
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
    if std::env::var_os("S2_PROBE_LARGE").is_some() {
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
             fallback_projection={} fallback_polygon_cap={} spherical_extract_ok={} failures={:?}",
            points.len(),
            ok_neighbors.len(),
            fail_neighbors.len(),
            exhausted,
            bounded_failures,
            fallback_projection,
            fallback_polygon_cap,
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
            seed_neighbors: &[],
        },
    )
    .expect("cell build should succeed");

    assert!(ctx.output_buffer().vertices.len() >= 3);
    assert!(!stats.knn_exhausted || !stats.did_packed);
}

#[test]
fn projection_invalid_detail_includes_replay_payload_summary() {
    let g = Vec3::new(0.0, 0.0, 1.0);
    let mut builder = Topo2DBuilder::new(17, g);

    let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
    let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
    let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

    builder
        .clip_with_slot_edgecheck_policy(11, 21, h1, 0.125)
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
            seed_neighbors: &[],
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
    assert!(ctx.builder.accepted_constraint_count() >= 3);
    assert!(!stats.knn_exhausted || !stats.did_packed);
}
