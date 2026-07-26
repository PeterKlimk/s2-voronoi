use super::*;
use crate::test_support::{effective_arrays, effective_generators, fib_sphere};
use glam::Vec3;

#[test]
fn strict_issue_messages_are_stable() {
    use StrictValidationIssue as Issue;

    let cases = [
        (
            Issue::GeneratorCellCountMismatch,
            "generator/cell count mismatch",
        ),
        (Issue::OffSphereVertex, "off-sphere vertex"),
        (Issue::WeldMap, "weld map"),
        (Issue::InvalidCellSpan, "invalid cell span"),
        (Issue::InvalidVertexReference, "invalid vertex reference"),
        (Issue::DuplicateVertexInCell, "duplicate vertex in cell"),
        (Issue::DegenerateCell, "degenerate cell"),
        (Issue::DuplicateCell, "duplicate cell"),
        (Issue::LowIncidenceVertex, "low-incidence vertex"),
        (
            Issue::UnpairedOverusedOrMisorientedEdge,
            "unpaired, overused, or misoriented edge",
        ),
        (Issue::AntipodalEdge, "antipodal edge"),
        (Issue::DisconnectedSubdivision, "disconnected subdivision"),
        (Issue::BadEulerCharacteristic, "bad euler characteristic"),
    ];

    for (issue, expected) in cases {
        assert_eq!(issue.message(), expected);
    }
}

/// A single triangular cell: its three edges are each used once, and the
/// sphere has no boundary, so all three are unpaired interior edges —
/// not strictly valid, with every vertex index in range.
fn invalid_diagram() -> SphericalVoronoi {
    // One generator => one cell (num_cells mirrors generator count).
    SphericalVoronoi::from_raw_parts(
        vec![Vec3::new(0.0, 0.0, 1.0)],
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        vec![crate::diagram::VoronoiCell::new(0, 3)],
        vec![0, 1, 2],
        None,
    )
}

#[test]
fn report_detects_two_position_cycle_without_an_adjacent_zero_edge() {
    let diagram = SphericalVoronoi::from_raw_parts(
        vec![Vec3::new(0.0, 0.0, 1.0)],
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![crate::diagram::VoronoiCell::new(0, 4)],
        vec![0, 1, 2, 3],
        None,
    );
    let report = validate(&diagram);
    assert_eq!(report.zero_length_edges, 0);
    assert_eq!(report.cells_with_fewer_than_three_stored_positions, 1);
    assert!(report
        .representation_notes()
        .iter()
        .any(|note| note.contains("fewer than three stored positions")));
}

#[test]
fn verify_gate_errors_only_when_enabled_and_invalid() {
    const CHILD_ENV: &str = "VORONOI_MESH_VERIFY_GATE_CHILD";
    let diagram = invalid_diagram();
    assert!(!validate(&diagram).is_strictly_valid());

    if let Ok(case) = std::env::var(CHILD_ENV) {
        let result = verify_sphere_if_enabled(&diagram);
        match case.as_str() {
            "disabled" => assert!(
                result.is_ok(),
                "disabled gate must not error even on an invalid diagram"
            ),
            "enabled" => match result.expect_err("enabled gate must reject invalid output") {
                crate::VoronoiError::ComputationFailed(msg) => {
                    assert!(msg.contains("VORONOI_MESH_VERIFY"), "message: {msg}");
                }
                other => panic!("unexpected error variant: {other:?}"),
            },
            other => panic!("unknown child case {other}"),
        }
        return;
    }

    let run_child = |case: &str, enabled: bool| {
        let mut command =
            std::process::Command::new(std::env::current_exe().expect("unit-test executable path"));
        command
            .arg("--exact")
            .arg("validation::verify_gate_tests::verify_gate_errors_only_when_enabled_and_invalid")
            .arg("--nocapture")
            .env(CHILD_ENV, case);
        if enabled {
            command.env("VORONOI_MESH_VERIFY", "1");
        } else {
            command.env_remove("VORONOI_MESH_VERIFY");
        }
        let output = command.output().expect("run isolated verification case");
        assert!(
            output.status.success(),
            "verification child {case} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    };

    for (case, enabled) in [("disabled", false), ("enabled", true)] {
        run_child(case, enabled);
    }
}

fn cells_from_cycles(cycles: &[Vec<u32>]) -> (Vec<crate::diagram::VoronoiCell>, Vec<u32>) {
    let mut cells = Vec::with_capacity(cycles.len());
    let mut indices = Vec::new();
    for cycle in cycles {
        let start = u32::try_from(indices.len()).expect("test cell-index start fits u32");
        let count = u16::try_from(cycle.len()).expect("test cell cycle fits u16");
        indices.extend_from_slice(cycle);
        cells.push(crate::diagram::VoronoiCell::new(start, count));
    }
    (cells, indices)
}

fn assert_strict_reason(
    generators: &[Vec3],
    vertices: &[Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
    expected: &'static str,
) {
    assert_eq!(generators.len(), cells.len(), "test fixture cardinality");
    let diagram = diagram_from_effective(generators, vertices, cells, cell_indices);
    assert_eq!(verify_sphere_fast(&diagram), Err(expected));
    assert_eq!(
        verify_sphere_effective_strict(
            generators,
            vertices,
            LiveCellLayout::new(cells, cell_indices),
        ),
        Err(expected)
    );
}

fn diagram_from_effective(
    generators: &[Vec3],
    vertices: &[Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
) -> SphericalVoronoi {
    assert_eq!(generators.len(), cells.len(), "test fixture cardinality");
    SphericalVoronoi::from_raw_parts(
        generators.to_vec(),
        vertices.to_vec(),
        cells.to_vec(),
        cell_indices.to_vec(),
        None,
    )
}

/// The slice validator must reach the SAME verdict (and first error) as the
/// canonical `verify_sphere_fast` it stands in for at the local-rebuild gate.
fn assert_agree(d: &SphericalVoronoi) {
    let (v, c, ci) = effective_arrays(d);
    let generators = effective_generators(d);
    let fast = verify_sphere_fast(d);
    let eff = verify_sphere_effective_strict(&generators, &v, LiveCellLayout::new(&c, &ci));
    assert_eq!(fast, eff, "fast={fast:?} effective={eff:?}");
}

#[test]
fn effective_strict_matches_fast() {
    // Valid: a real computed diagram (no coincident points => no weld map).
    let good = crate::compute(&fib_sphere(64)).expect("compute");
    assert!(
        verify_sphere_fast(&good).is_ok(),
        "compute output must be valid"
    );
    assert_agree(&good);

    // Invalid: three unpaired interior edges + degree-1 vertices.
    assert_agree(&invalid_diagram());

    // Invalid: a vertex repeated within one cell.
    assert_agree(&SphericalVoronoi::from_raw_parts(
        vec![Vec3::new(0.0, 0.0, 1.0)],
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        vec![crate::diagram::VoronoiCell::new(0, 4)],
        vec![0, 1, 0, 2],
        None,
    ));

    // Invalid: two cells with identical boundaries (duplicate cell) — pins
    // the sort-based duplicate-signature pass against the incremental
    // hashset the sequential reference uses.
    let (v, mut c, mut ci) = effective_arrays(&good);
    let first = c[0];
    ci.extend_from_within(first.vertex_start()..first.vertex_start() + first.vertex_count());
    c.push(crate::diagram::VoronoiCell::new(
        (ci.len() - first.vertex_count()) as u32,
        first.vertex_count() as u16,
    ));
    let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
    let dup_cell = SphericalVoronoi::from_raw_parts(generators, v, c, ci, None);
    assert!(verify_sphere_fast(&dup_cell).is_err());
    assert_agree(&dup_cell);

    // Raw effective arrays can still be rejected before they become a
    // diagram. A checked SpherePoint-backed SphericalVoronoi cannot safely
    // contain these coordinates, so there is intentionally no
    // verify_sphere_fast comparison for these cases.
    let (mut v, c, ci) = effective_arrays(&good);
    v[0] *= 2.0;
    let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
    assert_eq!(
        verify_sphere_effective_strict(&generators, &v, LiveCellLayout::new(&c, &ci)),
        Err("off-sphere vertex")
    );

    // Invalid: non-finite coordinates must not exploit NaN comparison
    // semantics in the effective-array gate.
    for coordinate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (mut v, c, ci) = effective_arrays(&good);
        v[0].x = coordinate;
        let generators = vec![Vec3::new(0.0, 0.0, 1.0); c.len()];
        assert_eq!(
            verify_sphere_effective_strict(&generators, &v, LiveCellLayout::new(&c, &ci)),
            Err("off-sphere vertex")
        );
    }
}

#[test]
fn weld_map_policy_matches_between_fast_gate_and_report() {
    let good = crate::compute(&fib_sphere(64)).expect("compute");
    let generators = effective_generators(&good);
    let (vertices, cells, indices) = effective_arrays(&good);
    let first = cells[0];
    let second = cells[1];
    assert_ne!(
        &indices[first.vertex_start()..first.vertex_start() + first.vertex_count()],
        &indices[second.vertex_start()..second.vertex_start() + second.vertex_count()],
        "fixture cells must differ",
    );

    let mut weld_map: Vec<u32> = (0..cells.len() as u32).collect();
    weld_map[1] = 0;
    let corrupt =
        SphericalVoronoi::from_raw_parts(generators, vertices, cells, indices, Some(weld_map));

    assert_eq!(verify_sphere_fast(&corrupt), Err("weld map"));
    let report = validate_impl(&corrupt);
    assert_eq!(report.welded_twin_cells, 1);
    assert_eq!(report.weld_map_issues, 1);
    assert!(!report.is_strictly_valid());
}

#[test]
fn strict_negative_controls_pin_shared_reasons() {
    let good = crate::compute(&fib_sphere(64)).expect("compute");
    let base_generators = effective_generators(&good);
    let (base_vertices, base_cells, base_indices) = effective_arrays(&good);

    let low_incidence = invalid_diagram();
    let low_incidence_generators = effective_generators(&low_incidence);
    let (vertices, cells, indices) = effective_arrays(&low_incidence);
    assert_strict_reason(
        &low_incidence_generators,
        &vertices,
        &cells,
        &indices,
        "low-incidence vertex",
    );

    // Invalid vertex id: both representations can carry a structurally
    // valid live span whose contents reference past the vertex buffer.
    let mut indices = base_indices.clone();
    indices[base_cells[0].vertex_start()] = base_vertices.len() as u32;
    assert_strict_reason(
        &base_generators,
        &base_vertices,
        &base_cells,
        &indices,
        "invalid vertex reference",
    );

    // Isolate degeneracy before global edge/incidence checks by shortening
    // one live span to two distinct ids.
    let mut cells = base_cells.clone();
    cells[0] = crate::diagram::VoronoiCell::new(cells[0].vertex_start() as u32, 2);
    assert_strict_reason(
        &base_generators,
        &base_vertices,
        &cells,
        &base_indices,
        "degenerate cell",
    );

    let mut indices = base_indices.clone();
    let repeated = base_cells
        .iter()
        .find(|cell| cell.vertex_count() >= 4)
        .copied()
        .expect("baseline cell with at least four vertices");
    let start = repeated.vertex_start();
    indices[start + 2] = indices[start];
    assert_strict_reason(
        &base_generators,
        &base_vertices,
        &base_cells,
        &indices,
        "duplicate vertex in cell",
    );

    let mut generators = base_generators.clone();
    let mut cells = base_cells.clone();
    let mut indices = base_indices.clone();
    let source = base_cells[0];
    let span = &base_indices[source.vertex_start()..source.vertex_start() + source.vertex_count()];
    let start = indices.len();
    indices.extend_from_slice(span);
    cells.push(crate::diagram::VoronoiCell::new(
        start as u32,
        span.len() as u16,
    ));
    generators.push(Vec3::Z);
    assert_strict_reason(
        &generators,
        &base_vertices,
        &cells,
        &indices,
        "duplicate cell",
    );

    // Reversing one otherwise-valid cell preserves ids, signatures, and
    // incidence while making its shared edges use the same direction.
    let mut indices = base_indices.clone();
    let start = base_cells[0].vertex_start();
    let end = start + base_cells[0].vertex_count();
    indices[start..end].reverse();
    assert_strict_reason(
        &base_generators,
        &base_vertices,
        &base_cells,
        &indices,
        "unpaired, overused, or misoriented edge",
    );

    // Add a distinct triangular face using one already-paired edge. No
    // referenced incidence falls, so grouped-edge classification is the
    // first failing global fact.
    let mut generators = base_generators.clone();
    let mut cells = base_cells.clone();
    let mut indices = base_indices.clone();
    let first = &base_indices
        [base_cells[0].vertex_start()..base_cells[0].vertex_start() + base_cells[0].vertex_count()];
    let a = first[0];
    let b = first[1];
    let x = (0..base_vertices.len() as u32)
        .find(|v| *v != a && *v != b && !first.contains(v))
        .expect("unrelated existing vertex");
    let new_start = indices.len();
    indices.extend_from_slice(&[a, b, x]);
    cells.push(crate::diagram::VoronoiCell::new(new_start as u32, 3));
    generators.push(Vec3::Z);
    assert_strict_reason(
        &generators,
        &base_vertices,
        &cells,
        &indices,
        "unpaired, overused, or misoriented edge",
    );

    // Geometry-only corruption leaves every combinatorial fact intact.
    let mut vertices = base_vertices.clone();
    let first = &base_indices
        [base_cells[0].vertex_start()..base_cells[0].vertex_start() + base_cells[0].vertex_count()];
    vertices[first[1] as usize] = -vertices[first[0] as usize];
    assert_strict_reason(
        &base_generators,
        &vertices,
        &base_cells,
        &base_indices,
        "antipodal edge",
    );

    // Two disjoint valid spheres retain local validity and Euler=4; the
    // connectivity check deliberately precedes Euler.
    let mut generators = base_generators.clone();
    generators.extend_from_slice(&base_generators);
    let mut vertices = base_vertices.clone();
    let vertex_offset = vertices.len() as u32;
    vertices.extend_from_slice(&base_vertices);
    let mut cells = base_cells.clone();
    let mut indices = base_indices.clone();
    for cell in &base_cells {
        let span = &base_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
        let start = indices.len();
        indices.extend(span.iter().map(|v| *v + vertex_offset));
        cells.push(crate::diagram::VoronoiCell::new(
            start as u32,
            span.len() as u16,
        ));
    }
    assert_strict_reason(
        &generators,
        &vertices,
        &cells,
        &indices,
        "disconnected subdivision",
    );

    // A 3x3 periodic quadrangulation is connected, closed, oriented, and
    // has degree-four vertices, but V-E+F = 9-18+9 = 0.
    let torus_points: Vec<Vec3> = (0..9)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / 9.0;
            Vec3::new(angle.cos(), angle.sin(), 0.5).normalize()
        })
        .collect();
    let mut cycles = Vec::with_capacity(9);
    let vertex = |x: usize, y: usize| ((y % 3) * 3 + (x % 3)) as u32;
    for y in 0..3 {
        for x in 0..3 {
            cycles.push(vec![
                vertex(x, y),
                vertex(x + 1, y),
                vertex(x + 1, y + 1),
                vertex(x, y + 1),
            ]);
        }
    }
    let (cells, indices) = cells_from_cycles(&cycles);
    assert_strict_reason(
        &torus_points,
        &torus_points,
        &cells,
        &indices,
        "bad euler characteristic",
    );
}

#[test]
fn effective_strict_pins_raw_structure_reasons() {
    let good = crate::compute(&fib_sphere(64)).expect("compute");
    let generators = effective_generators(&good);
    let (vertices, mut cells, indices) = effective_arrays(&good);

    assert_eq!(
        verify_sphere_effective_strict(
            &generators[..generators.len() - 1],
            &vertices,
            LiveCellLayout::new(&cells, &indices),
        ),
        Err("generator/cell count mismatch")
    );

    cells[0] = crate::diagram::VoronoiCell::new(indices.len() as u32, 1);
    assert_eq!(
        verify_sphere_effective_strict(
            &generators,
            &vertices,
            LiveCellLayout::new(&cells, &indices),
        ),
        Err("invalid cell span")
    );
}

#[test]
fn strict_self_loop_reason_is_dominated_by_cell_checks() {
    // Exhaustive small cycles prove the ordering invariant: an adjacent
    // repeated id is either a duplicate id, or the one-vertex wraparound
    // case is already degenerate.
    for len in 1..=4 {
        for mut code in 0..3usize.pow(len as u32) {
            let mut cycle = vec![0u32; len];
            for value in &mut cycle {
                *value = (code % 3) as u32;
                code /= 3;
            }
            let has_self_loop = (0..len).any(|i| cycle[i] == cycle[(i + 1) % cycle.len()]);
            if !has_self_loop {
                continue;
            }
            let mut distinct = cycle.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(distinct.len() < cycle.len() || distinct.len() < 3);
        }
    }

    let generators = [Vec3::Z];
    let vertices = [Vec3::X, Vec3::Y, Vec3::Z];
    for (cycle, expected) in [
        (vec![0], "degenerate cell"),
        (vec![0, 0], "duplicate vertex in cell"),
        (vec![0, 1, 1], "duplicate vertex in cell"),
    ] {
        let (cells, indices) = cells_from_cycles(&[cycle]);
        assert_strict_reason(&generators, &vertices, &cells, &indices, expected);
    }

    let (cells, indices) = cells_from_cycles(&[vec![0, 1, 1]]);
    let report = validate_impl(&diagram_from_effective(
        &generators,
        &vertices,
        &cells,
        &indices,
    ));
    assert_eq!(report.self_loop_edges, 1);
}

#[test]
fn report_pins_edge_use_classifications() {
    let boundary = validate_impl(&invalid_diagram());
    assert_eq!(boundary.boundary_edges, 3);
    assert_eq!(boundary.overused_edges, 0);
    assert_eq!(boundary.same_direction_edge_pairs, 0);

    let good = crate::compute(&fib_sphere(64)).expect("compute");
    let generators = effective_generators(&good);
    let (vertices, cells, base_indices) = effective_arrays(&good);

    let mut reversed_indices = base_indices.clone();
    let start = cells[0].vertex_start();
    let end = start + cells[0].vertex_count();
    reversed_indices[start..end].reverse();
    let reversed = validate_impl(&diagram_from_effective(
        &generators,
        &vertices,
        &cells,
        &reversed_indices,
    ));
    assert_eq!(reversed.boundary_edges, 0);
    assert_eq!(reversed.overused_edges, 0);
    assert!(reversed.same_direction_edge_pairs > 0);

    let mut duplicate_generators = generators.clone();
    duplicate_generators.push(Vec3::Z);
    let mut duplicate_cells = cells.clone();
    let mut duplicate_indices = base_indices.clone();
    let source = cells[0];
    let span = &base_indices[source.vertex_start()..source.vertex_start() + source.vertex_count()];
    let duplicate_start = duplicate_indices.len();
    duplicate_indices.extend_from_slice(span);
    duplicate_cells.push(crate::diagram::VoronoiCell::new(
        duplicate_start as u32,
        span.len() as u16,
    ));
    let overused = validate_impl(&diagram_from_effective(
        &duplicate_generators,
        &vertices,
        &duplicate_cells,
        &duplicate_indices,
    ));
    assert_eq!(overused.boundary_edges, 0);
    assert_eq!(overused.overused_edges, span.len());
    assert_eq!(overused.same_direction_edge_pairs, 0);
}

#[test]
fn edge_use_classifier_distinguishes_all_outcomes() {
    assert_eq!(classify_edge_uses(0, false), EdgeUseClass::Boundary);
    assert_eq!(classify_edge_uses(1, false), EdgeUseClass::Boundary);
    assert_eq!(classify_edge_uses(2, true), EdgeUseClass::Paired);
    assert_eq!(classify_edge_uses(2, false), EdgeUseClass::SameDirection);
    assert_eq!(classify_edge_uses(3, false), EdgeUseClass::Overused);
    assert_eq!(classify_edge_uses(3, true), EdgeUseClass::Overused);
}
