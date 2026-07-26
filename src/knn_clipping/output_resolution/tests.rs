use super::*;

#[test]
fn bounded_components_do_not_collapse_a_long_epsilon_chain() {
    let vertices = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.4, 0.0, 1.0),
        Vec3::new(0.8, 0.0, 1.0),
        Vec3::new(1.2, 0.0, 1.0),
    ];
    let mut components = BoundedComponents::new(&[0, 1, 2, 3]);
    assert!(components.try_union(0, 1, &vertices, 0.75).unwrap());
    assert!(!components.try_union(1, 2, &vertices, 0.75).unwrap());
    assert!(components.try_union(2, 3, &vertices, 0.75).unwrap());
    assert_eq!(components.find(0), components.find(1));
    assert_eq!(components.find(2), components.find(3));
    assert_ne!(components.find(0), components.find(2));
}

#[derive(Clone)]
struct ResolutionFixture {
    generators: Vec<Vec3>,
    vertices: Vec<Vec3>,
    cells: Vec<VoronoiCell>,
    indices: Vec<u32>,
}

#[derive(Clone, Copy)]
struct ExpectedResolution {
    edges: usize,
    components: usize,
    contracted_edges: usize,
    preserved_components: usize,
}

struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 7;
        self.0 ^= self.0 >> 9;
        self.0 ^= self.0 << 8;
        self.0
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.next() as usize % (i + 1));
        }
    }
}

fn unit(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z).normalize()
}

fn cells_from_cycles(cycles: &[&[u32]]) -> (Vec<VoronoiCell>, Vec<u32>) {
    let mut cells = Vec::new();
    let mut indices = Vec::new();
    for cycle in cycles {
        cells.push(VoronoiCell::new(indices.len() as u32, cycle.len() as u16));
        indices.extend_from_slice(cycle);
    }
    (cells, indices)
}

fn owned_cells_from_cycles(cycles: &[Vec<u32>]) -> (Vec<VoronoiCell>, Vec<u32>) {
    let mut cells = Vec::with_capacity(cycles.len());
    let mut indices = Vec::new();
    for cycle in cycles {
        cells.push(VoronoiCell::new(indices.len() as u32, cycle.len() as u16));
        indices.extend_from_slice(cycle);
    }
    (cells, indices)
}

fn live_cycles(cells: &[VoronoiCell], indices: &[u32]) -> Vec<Vec<u32>> {
    cells
        .iter()
        .map(|cell| {
            let start = cell.vertex_start();
            indices[start..start + cell.vertex_count()].to_vec()
        })
        .collect()
}

/// Closed n-gonal prism with selected lower-ring edges made exact-zero.
/// The cell orientation is globally coherent: lower ring forward, upper
/// ring backward, and each side opposite to both adjacent rings.
fn prism_fixture(n: usize, zero_edges: &[(usize, usize)]) -> ResolutionFixture {
    assert!(n >= 4);
    let tau = std::f32::consts::TAU;
    let mut vertices = Vec::with_capacity(2 * n);
    // Keep every synthetic edge far from the antipodal-policy boundary;
    // these fixtures exercise quotient topology, not near-pi geometry.
    for ring_z in [1.5, 2.5] {
        for i in 0..n {
            let angle = tau * i as f32 / n as f32;
            vertices.push(unit(angle.cos(), angle.sin(), ring_z));
        }
    }

    // Tiny fixture-local union-find. Exact coordinate replacement then
    // creates the requested path/forest/cycle component.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            x = parent[x];
        }
        x
    }
    for &(a, b) in zero_edges {
        assert!(a < n && b < n);
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        parent[rb] = ra;
    }
    for i in 0..n {
        let root = find(&mut parent, i);
        vertices[i] = vertices[root];
    }

    let mut cycles = Vec::with_capacity(n + 2);
    cycles.push((0..n as u32).collect());
    cycles.push((n as u32..(2 * n) as u32).rev().collect());
    for i in 0..n {
        let j = (i + 1) % n;
        cycles.push(vec![j as u32, i as u32, (n + i) as u32, (n + j) as u32]);
    }
    let (cells, indices) = owned_cells_from_cycles(&cycles);
    let generators = (0..cells.len())
        .map(|i| {
            let angle = tau * (i as f32 + 0.375) / cells.len() as f32;
            unit(angle.cos(), angle.sin(), 0.2 + 0.01 * i as f32)
        })
        .collect();
    ResolutionFixture {
        generators,
        vertices,
        cells,
        indices,
    }
}

fn permute_fixture(mut fixture: ResolutionFixture, seed: u64) -> ResolutionFixture {
    let mut rng = TestRng::new(seed);
    let mut new_for_old: Vec<usize> = (0..fixture.vertices.len()).collect();
    rng.shuffle(&mut new_for_old);
    let mut vertices = vec![Vec3::ZERO; fixture.vertices.len()];
    for (old, &new) in new_for_old.iter().enumerate() {
        vertices[new] = fixture.vertices[old];
    }

    let reverse_all = rng.next() & 1 != 0;
    let old_cycles = live_cycles(&fixture.cells, &fixture.indices);
    let mut faces: Vec<(Vec<u32>, Vec3)> = old_cycles
        .into_iter()
        .zip(fixture.generators)
        .map(|(cycle, generator)| {
            let mut cycle: Vec<u32> = cycle
                .into_iter()
                .map(|old| new_for_old[old as usize] as u32)
                .collect();
            if reverse_all {
                cycle.reverse();
            }
            let len = cycle.len();
            cycle.rotate_left(rng.next() as usize % len);
            (cycle, generator)
        })
        .collect();
    rng.shuffle(&mut faces);
    let (cycles, generators): (Vec<_>, Vec<_>) = faces.into_iter().unzip();
    let (cells, indices) = owned_cells_from_cycles(&cycles);
    fixture.vertices = vertices;
    fixture.generators = generators;
    fixture.cells = cells;
    fixture.indices = indices;
    fixture
}

fn incident_candidate_cells(
    candidates: &[(u32, u32)],
    cells: &[VoronoiCell],
    indices: &[u32],
) -> Vec<usize> {
    let mut endpoints = FxHashSet::default();
    for &(a, b) in candidates {
        endpoints.insert(a);
        endpoints.insert(b);
    }
    cells
        .iter()
        .enumerate()
        .filter_map(|(cell_idx, cell)| {
            let start = cell.vertex_start();
            indices[start..start + cell.vertex_count()]
                .iter()
                .any(|vertex| endpoints.contains(vertex))
                .then_some(cell_idx)
        })
        .collect()
}

fn assert_localized_matches_exhaustive(
    fixture: ResolutionFixture,
    expected: ExpectedResolution,
    context: &str,
) {
    let candidates = collect_zero_edges(&fixture.vertices, &fixture.cells, &fixture.indices)
        .expect("fixture edge discovery");
    let candidate_cells = incident_candidate_cells(&candidates, &fixture.cells, &fixture.indices);

    let mut exhaustive_cells = fixture.cells.clone();
    let mut exhaustive_indices = fixture.indices.clone();
    let exhaustive = canonicalize_exact_zero_edges(
        &fixture.vertices,
        &mut exhaustive_cells,
        &mut exhaustive_indices,
        None,
        None,
    )
    .unwrap_or_else(|error| panic!("{context}: exhaustive canonicalization failed: {error}"));

    let mut localized_cells = fixture.cells;
    let mut localized_indices = fixture.indices;
    let localized = canonicalize_exact_zero_edges(
        &fixture.vertices,
        &mut localized_cells,
        &mut localized_indices,
        Some(candidates),
        Some(candidate_cells),
    )
    .unwrap_or_else(|error| panic!("{context}: localized canonicalization failed: {error}"));

    assert_eq!(localized, exhaustive, "{context}: report mismatch");
    assert_eq!(
        live_cycles(&localized_cells, &localized_indices),
        live_cycles(&exhaustive_cells, &exhaustive_indices),
        "{context}: quotient mismatch"
    );
    assert_eq!(
        exhaustive.report.exact_zero_edges_detected, expected.edges,
        "{context}"
    );
    assert_eq!(
        exhaustive.report.exact_zero_components_detected, expected.components,
        "{context}"
    );
    assert_eq!(
        exhaustive.report.exact_zero_edges_contracted, expected.contracted_edges,
        "{context}"
    );
    assert_eq!(
        exhaustive.report.cell_killing_components_preserved, expected.preserved_components,
        "{context}"
    );

    let diagram = crate::SphericalVoronoi::from_raw_parts(
        fixture.generators,
        fixture.vertices,
        exhaustive_cells,
        exhaustive_indices,
        None,
    );
    let validation = crate::validation::validate(&diagram);
    assert!(
        validation.is_strictly_valid(),
        "{context}: terminal fixture failed strict validation: {}",
        validation.headline()
    );
}

#[test]
fn contracts_non_cell_killing_cube_edge() {
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
    let generators = vec![
        unit(0.0, 0.0, -1.0),
        unit(0.0, 0.0, 1.0),
        unit(0.0, -1.0, 0.0),
        unit(0.0, 1.0, 0.0),
        unit(-1.0, 0.0, 0.0),
        unit(1.0, 0.0, 0.0),
    ];
    let (mut cells, mut indices) = cells_from_cycles(&[
        &[0, 3, 2, 1],
        &[4, 5, 6, 7],
        &[0, 1, 5, 4],
        &[3, 7, 6, 2],
        &[0, 4, 7, 3],
        &[1, 2, 6, 5],
    ]);

    let before = crate::SphericalVoronoi::from_raw_parts(
        generators.clone(),
        vertices.clone(),
        cells.clone(),
        indices.clone(),
        None,
    );
    let before_validation = crate::validation::validate(&before);
    assert!(before_validation.is_strictly_valid());
    assert_eq!(before_validation.zero_length_edges, 1);

    let report =
        canonicalize_exact_zero_edges(&vertices, &mut cells, &mut indices, None, None).unwrap();

    assert_eq!(report.report.exact_zero_edges_detected, 1);
    assert_eq!(report.report.exact_zero_edges_contracted, 1);
    assert_eq!(report.report.exact_zero_components_contracted, 1);
    assert_eq!(report.report.exact_zero_edges_remaining, 0);
    assert!(report.cell_killing_generators.is_empty());
    assert_eq!(cells[0].vertex_count(), 3);
    assert_eq!(cells[2].vertex_count(), 3);
    assert!(crate::validation::verify_sphere_effective_strict(
        &generators,
        &vertices,
        crate::cell_layout::LiveCellLayout::new(&cells, &indices),
    )
    .is_ok());

    let after = crate::SphericalVoronoi::from_raw_parts(generators, vertices, cells, indices, None);
    assert_eq!(crate::validation::validate(&after).zero_length_edges, 0);
}

#[test]
fn preserve_declines_cell_killing_tetrahedron_edge() {
    let mut vertices = vec![
        unit(1.0, 1.0, 1.0),
        unit(1.0, -1.0, -1.0),
        unit(-1.0, 1.0, -1.0),
        unit(-1.0, -1.0, 1.0),
    ];
    vertices[1] = vertices[0];
    let (mut cells, mut indices) =
        cells_from_cycles(&[&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[1, 2, 3]]);
    let original = indices.clone();

    let report =
        canonicalize_exact_zero_edges(&vertices, &mut cells, &mut indices, None, None).unwrap();

    assert_eq!(report.report.exact_zero_edges_detected, 1);
    assert_eq!(report.report.exact_zero_edges_contracted, 0);
    assert_eq!(report.report.cell_killing_components_preserved, 1);
    assert_eq!(report.report.exact_zero_edges_remaining, 1);
    assert_eq!(report.cell_killing_generators, [0, 1]);
    assert_eq!(indices, original);
    assert!(cells.iter().all(|cell| cell.vertex_count() == 3));
}

#[test]
fn elision_rejects_whole_mesh_collapse() {
    let mut vertices = vec![
        unit(1.0, 1.0, 1.0),
        unit(1.0, -1.0, -1.0),
        unit(-1.0, 1.0, -1.0),
        unit(-1.0, -1.0, 1.0),
    ];
    vertices[1] = vertices[0];
    let generators = vec![Vec3::Z; 4];
    let (cells, indices) = cells_from_cycles(&[&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[1, 2, 3]]);

    let error = elide_exact_zero_cells_for_mesh(&generators, &vertices, &cells, &indices)
        .expect_err("a quotient that consumes the entire sphere must fail");
    assert!(error.to_string().contains("cell elision"));
}

#[test]
fn elision_vertex_link_check_rejects_two_spheres_pinched_at_one_vertex() {
    let vertices = vec![
        unit(1.0, 1.0, 1.0),
        unit(1.0, -1.0, -1.0),
        unit(-1.0, 1.0, -1.0),
        unit(-1.0, -1.0, 1.0),
        unit(1.0, -1.0, 1.0),
        unit(-1.0, 1.0, 1.0),
        unit(-1.0, -1.0, -1.0),
    ];
    let (cells, indices) = cells_from_cycles(&[
        &[0, 2, 1],
        &[0, 1, 3],
        &[0, 3, 2],
        &[1, 2, 3],
        &[0, 5, 4],
        &[0, 4, 6],
        &[0, 6, 5],
        &[4, 5, 6],
    ]);
    let diagram = crate::SphericalVoronoi::from_raw_parts(
        vec![Vec3::Z; cells.len()],
        vertices,
        cells,
        indices,
        None,
    );
    assert!(!elision_links_are_single_cycles(&diagram));
}

#[test]
fn degree_two_suppression_requires_opposite_owner_rotations() {
    let mut cycles = [Some(vec![0, 1, 2]), Some(vec![0, 1, 2])];
    let error = suppress_elision_degree_two_vertices(&mut cycles, 3)
        .expect_err("same-direction owner rotations must not be stitched");
    assert!(error.to_string().contains("owner rotations disagree"));
}

#[test]
fn localized_discovery_matches_exhaustive_for_component_families_and_permutations() {
    let maximal_tree: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 1)).collect();
    let safe_shared_cell = vec![(0, 1), (3, 4), (6, 7)];
    let killing_shared_cell = vec![(0, 1), (2, 3)];
    let killing_cycle: Vec<(usize, usize)> = (0..8).map(|i| (i, (i + 1) % 8)).collect();
    let cases = [
        (
            "maximal-safe-tree",
            prism_fixture(8, &maximal_tree),
            ExpectedResolution {
                edges: 5,
                components: 1,
                contracted_edges: 5,
                preserved_components: 0,
            },
        ),
        (
            "safe-shared-cell-components",
            prism_fixture(8, &safe_shared_cell),
            ExpectedResolution {
                edges: 3,
                components: 3,
                contracted_edges: 3,
                preserved_components: 0,
            },
        ),
        (
            "jointly-cell-killing-components",
            prism_fixture(4, &killing_shared_cell),
            ExpectedResolution {
                edges: 2,
                components: 2,
                contracted_edges: 0,
                preserved_components: 2,
            },
        ),
        (
            "cell-killing-cycle",
            prism_fixture(8, &killing_cycle),
            ExpectedResolution {
                edges: 8,
                components: 1,
                contracted_edges: 0,
                preserved_components: 1,
            },
        ),
    ];

    for (name, fixture, expected) in cases {
        for seed in 0..24 {
            let context = format!("{name} permutation seed {seed}");
            assert_localized_matches_exhaustive(
                permute_fixture(fixture.clone(), seed),
                expected,
                &context,
            );
        }
    }
}

#[test]
fn localized_discovery_matches_exhaustive_for_randomized_prism_forests() {
    const N: usize = 12;
    for seed in 0..64u64 {
        let mut rng = TestRng::new(seed);
        let mut candidate_edges: Vec<(usize, usize)> = (0..N - 1).map(|i| (i, i + 1)).collect();
        rng.shuffle(&mut candidate_edges);
        // At most N-3 forest edges leaves at least three vertices in the
        // lower face, so every generated component is contractible.
        let count = 1 + rng.next() as usize % (N - 3);
        candidate_edges.truncate(count);
        let mut selected_slots: Vec<usize> = candidate_edges.iter().map(|&(a, _)| a).collect();
        selected_slots.sort_unstable();
        let components = 1 + selected_slots
            .windows(2)
            .filter(|pair| pair[1] != pair[0] + 1)
            .count();
        let fixture = permute_fixture(prism_fixture(N, &candidate_edges), seed ^ 0xa5a5_5a5a);
        let context = format!("randomized prism forest seed {seed}");
        assert_localized_matches_exhaustive(
            fixture,
            ExpectedResolution {
                edges: count,
                components,
                contracted_edges: count,
                preserved_components: 0,
            },
            &context,
        );
    }
}
