use super::*;
use glam::Vec3;
use std::collections::BTreeSet;

fn edge_record(a: u32, b: u32) -> EdgeRecord {
    EdgeRecord {
        key: (((b as u64) << 32) | a as u64).into(),
    }
}

#[test]
fn localized_residual_scan_enforces_multiplicity_and_orientation() {
    let keys = vec![[0, 1, 2]; 3];

    let cells = vec![VoronoiCell::new(0, 3), VoronoiCell::new(3, 3)];
    let opposite = vec![0, 1, 2, 2, 1, 0];
    assert!(scan_unpaired_interior(
        LiveCellLayout::new(&cells, &opposite),
        VertexKeys::Flat(&keys),
        &[0, 1],
    )
    .expect("valid paired scan")
    .is_empty());

    let same_direction = vec![0, 1, 2, 0, 1, 2];
    assert_eq!(
        scan_unpaired_interior(
            LiveCellLayout::new(&cells, &same_direction),
            VertexKeys::Flat(&keys),
            &[0, 1],
        )
        .expect("same-direction scan")
        .len(),
        3,
        "every shared edge is misoriented"
    );

    let cells = vec![
        VoronoiCell::new(0, 3),
        VoronoiCell::new(3, 3),
        VoronoiCell::new(6, 3),
    ];
    let overused = vec![0, 1, 2, 2, 1, 0, 0, 1, 2];
    assert_eq!(
        scan_unpaired_interior(
            LiveCellLayout::new(&cells, &overused),
            VertexKeys::Flat(&keys),
            &[0, 1, 2],
        )
        .expect("overused scan")
        .len(),
        3,
        "every shared edge has a third use"
    );

    let self_loop = vec![0, 1, 2, 0, 0, 2];
    let bad = scan_unpaired_interior(
        LiveCellLayout::new(&cells[..2], &self_loop),
        VertexKeys::Flat(&keys),
        &[0, 1],
    )
    .expect("self-loop scan");
    assert!(bad.iter().any(|&(a, b, _)| a == b));
}

/// Pre-reuse collector retained as an independent test oracle.
fn edge_segments_allocating_baseline(
    cell_idx: u32,
    neighbor: u32,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
) -> Vec<(u32, u32)> {
    let slice = cell_vertex_slice(cell_idx, cells, cell_indices).expect("valid cell span");
    let mut out = Vec::new();
    for i in 0..slice.len() {
        let vi = slice[i];
        let vj = slice[(i + 1) % slice.len()];
        let ki = vertex_keys.get(vi).expect("valid vertex key");
        let kj = vertex_keys.get(vj).expect("valid vertex key");
        if shared_neighbor(cell_idx, ki, kj) == Some(neighbor) {
            out.push((vi, vj));
        }
    }
    out
}

#[test]
fn reused_segment_buffers_match_allocating_baseline_across_records_and_rounds() {
    let vertex_keys: Vec<VertexKey> = vec![
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 1, 4],
        [0, 1, 5],
        [0, 2, 5],
    ];
    let mut cells = vec![VoronoiCell::new(0, 6)];
    cells.extend((1..6).map(|_| VoronoiCell::new(6, 0)));
    let cell_indices = vec![0, 1, 2, 3, 4, 5];
    let records = [edge_record(0, 1), edge_record(0, 2), edge_record(0, 4)];
    let keys = VertexKeys::Flat(&vertex_keys);
    let mut seg_a = Vec::new();
    let mut seg_b = Vec::new();

    for round in 0..2 {
        for (record_idx, record) in records.iter().enumerate() {
            let (a, b) = unpack_edge(record.key.as_u64());
            let expected_a = edge_segments_allocating_baseline(a, b, &cells, &cell_indices, keys);
            let expected_b = edge_segments_allocating_baseline(b, a, &cells, &cell_indices, keys);
            let layout = LiveCellLayout::new(&cells, &cell_indices);
            edge_segments_for_neighbor_into(a, b, layout, keys, &mut seg_a)
                .expect("reused A collector");
            edge_segments_for_neighbor_into(b, a, layout, keys, &mut seg_b)
                .expect("reused B collector");
            assert_eq!(seg_a, expected_a, "round {round}, record {record_idx}, A");
            assert_eq!(seg_b, expected_b, "round {round}, record {record_idx}, B");
            if round == 0 && record_idx == 0 {
                assert_eq!(seg_a, [(0, 1), (3, 4)]);
            }
        }
        // Model a productive reconciliation round shrinking the live span. The
        // second pass must clear stale segments while retaining capacity.
        cells[0] = VoronoiCell::new(0, 5);
    }
}

/// Per-cell vertex-id sequences — the representation-independent view
/// shared by both apply backends (rebuild compacts the index buffer,
/// in-place leaves stale tail slots; the sequences must be identical).
fn cell_sequences(cells: &[VoronoiCell], cell_indices: &[u32]) -> Vec<Vec<u32>> {
    cells
        .iter()
        .enumerate()
        .map(|(i, _)| {
            cell_vertex_slice(i as u32, cells, cell_indices)
                .expect("valid span")
                .to_vec()
        })
        .collect()
}

/// Run both backends on clones of the input and assert they produce the
/// same per-cell sequences; returns the in-place result.
#[allow(clippy::type_complexity)]
fn run_both_backends(
    records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
) -> (bool, Vec<VoronoiCell>, Vec<u32>, Vec<VoronoiCell>, Vec<u32>) {
    let (mut cells_r, mut idx_r) = (cells.to_vec(), cell_indices.to_vec());
    let residual_r = reconcile_edge_mismatches(
        records,
        vertices,
        &mut cells_r,
        &mut idx_r,
        VertexKeys::Flat(vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        ReconcileOptions::with_apply(ReconcileApply::Rebuild),
    )
    .expect("rebuild reconciliation should succeed");

    let (mut cells_p, mut idx_p) = (cells.to_vec(), cell_indices.to_vec());
    let residual_p = reconcile_edge_mismatches(
        records,
        vertices,
        &mut cells_p,
        &mut idx_p,
        VertexKeys::Flat(vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        ReconcileOptions::with_apply(ReconcileApply::InPlace),
    )
    .expect("in-place reconciliation should succeed");

    assert_eq!(
        residual_r, residual_p,
        "backends disagree on post-reconciliation residuals"
    );
    assert_eq!(
        cell_sequences(&cells_r, &idx_r),
        cell_sequences(&cells_p, &idx_p),
        "backends disagree on per-cell vertex sequences"
    );
    let changed = cell_sequences(&cells_p, &idx_p) != cell_sequences(cells, cell_indices);
    (changed, cells_r, idx_r, cells_p, idx_p)
}

/// Normalized partition (each vertex -> its component's min member) so two
/// union-finds can be compared structurally.
fn partition(uf: &mut SparseUnionFind, n: u32) -> Vec<u32> {
    let mut root = vec![0u32; n as usize];
    for v in 0..n {
        root[v as usize] = uf.find(v);
    }
    // canonicalize: map each root to its smallest member
    let mut canon = std::collections::BTreeMap::<u32, u32>::new();
    for v in 0..n {
        let r = root[v as usize];
        canon.entry(r).and_modify(|m| *m = (*m).min(v)).or_insert(v);
    }
    (0..n).map(|v| canon[&root[v as usize]]).collect()
}

#[test]
fn diameter_gate_rejects_a_transitive_component_transactionally() {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.75 * eps, 0.0, 0.0),
        Vec3::new(1.5 * eps, 0.0, 0.0),
    ];
    let mut proposed = SparseUnionFind::new();
    assert!(proposed.union(0, 1));
    assert!(proposed.union(1, 2));

    let mut ledger = MergeLedger::default();
    let (accepted, merges, rejected, _) = bound_merge_components(
        &mut proposed,
        &vertices,
        &[],
        &[],
        VertexKeys::Flat(&[]),
        &mut ledger,
        eps,
    )
    .expect("diameter gate");

    assert_eq!(merges, 0, "no order-dependent prefix may be accepted");
    assert_eq!(rejected.len(), 1);
    assert!(accepted.touched_ids().is_empty());
    assert!(ledger.members.is_empty());
}

#[test]
fn diameter_gate_remembers_members_across_rounds() {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.75 * eps, 0.0, 0.0),
        Vec3::new(-0.5 * eps, 0.0, 0.0),
    ];
    let mut ledger = MergeLedger::default();

    let mut first = SparseUnionFind::new();
    assert!(first.union(0, 1));
    let (_, first_merges, first_rejected, _) = bound_merge_components(
        &mut first,
        &vertices,
        &[],
        &[],
        VertexKeys::Flat(&[]),
        &mut ledger,
        eps,
    )
    .expect("first-round diameter gate");
    assert_eq!(first_merges, 1);
    assert!(first_rejected.is_empty());

    // The surviving representative 0 is close to 2, but the hidden prior
    // member 1 is 1.25 eps away. A per-round-only gate would miss this.
    let mut second = SparseUnionFind::new();
    assert!(second.union(0, 2));
    let (accepted, second_merges, second_rejected, _) = bound_merge_components(
        &mut second,
        &vertices,
        &[],
        &[],
        VertexKeys::Flat(&[]),
        &mut ledger,
        eps,
    )
    .expect("second-round diameter gate");
    assert_eq!(second_merges, 0);
    assert_eq!(second_rejected.len(), 1);
    assert!(accepted.touched_ids().is_empty());
    assert_eq!(ledger.members.get(&0), Some(&vec![0, 1]));
}

fn sparse_face_safety_fixture() -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>, Vec<VertexKey>) {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.25 * eps, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let mut cells = vec![VoronoiCell::new(0, 0); 100];
    cells[50] = VoronoiCell::new(0, 3);
    let cell_indices = vec![0, 1, 2];
    let vertex_keys = vec![[50, 51, 52]; 3];
    (vertices, cells, cell_indices, vertex_keys)
}

#[test]
fn merge_safety_scans_only_key_owner_cover() {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let (vertices, cells, cell_indices, vertex_keys) = sparse_face_safety_fixture();
    let mut proposed = SparseUnionFind::new();
    assert!(proposed.union(0, 1));
    let (_, merges, rejected, stats) = bound_merge_components(
        &mut proposed,
        &vertices,
        &cells,
        &cell_indices,
        VertexKeys::Flat(&vertex_keys),
        &mut MergeLedger::default(),
        eps,
    )
    .expect("localized face-safety scan");

    assert_eq!(
        merges, 0,
        "the covered cell would collapse below a triangle"
    );
    assert_eq!(rejected.len(), 1);
    assert_eq!(stats.scanned_cells, 3);
    assert_eq!(stats.global_fallbacks, 0);
}

#[test]
fn merge_safety_missing_provenance_falls_back_globally() {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let (vertices, cells, cell_indices, _) = sparse_face_safety_fixture();
    let mut proposed = SparseUnionFind::new();
    assert!(proposed.union(0, 1));
    let (_, merges, rejected, stats) = bound_merge_components(
        &mut proposed,
        &vertices,
        &cells,
        &cell_indices,
        VertexKeys::Flat(&[]),
        &mut MergeLedger::default(),
        eps,
    )
    .expect("global face-safety fallback");

    assert_eq!(merges, 0);
    assert_eq!(rejected.len(), 1);
    assert_eq!(stats.scanned_cells, cells.len());
    assert_eq!(stats.global_fallbacks, 1);
}

#[test]
fn merge_safety_cover_expands_prior_round_members() {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.1 * eps, 0.0, 0.0),
        Vec3::new(0.2 * eps, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let vertex_keys = vec![[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]];
    let mut ledger = MergeLedger::default();

    let mut first = SparseUnionFind::new();
    assert!(first.union(0, 1));
    let (_, first_merges, first_rejected, _) = bound_merge_components(
        &mut first,
        &vertices,
        &[],
        &[],
        VertexKeys::Flat(&vertex_keys),
        &mut ledger,
        eps,
    )
    .expect("first-round merge");
    assert_eq!(first_merges, 1);
    assert!(first_rejected.is_empty());

    // Cell 3 originally referenced retired id 1. After applying the first
    // merge it references representative 0, whose own key does not name
    // cell 3. The ledger member 1 must keep cell 3 in the next cover.
    let mut cells = vec![VoronoiCell::new(0, 0); 6];
    cells[3] = VoronoiCell::new(0, 3);
    let cell_indices = vec![0, 2, 3];
    let mut second = SparseUnionFind::new();
    assert!(second.union(0, 2));
    let (_, second_merges, second_rejected, _) = bound_merge_components(
        &mut second,
        &vertices,
        &cells,
        &cell_indices,
        VertexKeys::Flat(&vertex_keys),
        &mut ledger,
        eps,
    )
    .expect("second-round merge safety");

    assert_eq!(second_merges, 0);
    assert_eq!(second_rejected.len(), 1);
    assert_eq!(ledger.members.get(&0), Some(&vec![0, 1]));
}

#[test]
fn rejected_chain_becomes_an_explicit_local_rebuild_seed() {
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.75 * eps, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.5 * eps, 0.0, 0.0),
        Vec3::new(10.0 * eps, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];
    let vertex_keys: Vec<VertexKey> = vec![
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 1, 4],
        [0, 1, 5],
        [0, 2, 5],
    ];
    let mut cells = vec![VoronoiCell::new(0, 6)];
    cells.extend((1..6).map(|_| VoronoiCell::new(6, 0)));
    let mut cell_indices = vec![0, 1, 2, 3, 4, 5];
    let records = [edge_record(0, 1)];
    let before = cell_sequences(&cells, &cell_indices);
    let mut state = ReconcileRunState::default();

    run_reconciliation_rounds(
        &records,
        &vertices,
        &mut cells,
        &mut cell_indices,
        VertexKeys::Flat(&vertex_keys),
        eps,
        ReconcileOptions::with_apply(ReconcileApply::InPlace),
        MergeMode::Primary,
        &mut state,
    )
    .expect("chain reconciliation");

    let result = state.into_result(Vec::new());
    assert_eq!(cell_sequences(&cells, &cell_indices), before);
    assert_eq!(result.local_rebuild_seed_pairs, [(0, 1)]);
    assert_eq!(result.merge_affected_cells, [0, 1, 2, 3, 4]);
}

/// The localized BFS dup-scan must union the same components as the global
/// O(V) scan — including a *chain*: corner [0,1,2] is triplicated (copies
/// in cells 0,1,2) and corner [2,3,4] is duplicated (copies in cells 2,3).
/// Only edge (0,1) is recorded, so cells 3,4 are reached purely through the
/// damaged-cell 1-ring expansion off cell 2.
#[test]
fn localized_dupscan_matches_global_with_chain() {
    // vertices 0,1,2 = copies of corner [0,1,2]; 3,4 = copies of [2,3,4].
    let vertex_keys: Vec<VertexKey> = vec![[0, 1, 2], [0, 1, 2], [0, 1, 2], [2, 3, 4], [2, 3, 4]];
    // cell c -> its corner vertex ids
    let cells = vec![
        VoronoiCell::new(0, 1), // cell 0: [v0]
        VoronoiCell::new(1, 1), // cell 1: [v1]
        VoronoiCell::new(2, 2), // cell 2: [v2, v3]
        VoronoiCell::new(4, 1), // cell 3: [v4]
        VoronoiCell::new(5, 0), // cell 4: (no owned corners in this fixture)
    ];
    let cell_indices = vec![0u32, 1, 2, 3, 4];
    let records = [edge_record(0, 1)];

    let mut uf_local = SparseUnionFind::new();
    let mut merged_local = 0usize;
    localized_dup_key_unions(
        &records,
        LiveCellLayout::new(&cells, &cell_indices),
        VertexKeys::Flat(&vertex_keys),
        &mut uf_local,
        &mut merged_local,
    )
    .expect("localized dup scan");

    let mut uf_global = SparseUnionFind::new();
    let mut merged_global = 0usize;
    global_dup_key_unions(
        VertexKeys::Flat(&vertex_keys),
        &mut uf_global,
        &mut merged_global,
    );

    assert_eq!(
        partition(&mut uf_local, 5),
        partition(&mut uf_global, 5),
        "localized BFS dup-scan must match the global scan's components (chain case)"
    );
    assert_eq!(merged_local, merged_global, "same number of merges");
    // Sanity: the two corners are distinct components, each fully merged.
    let p = partition(&mut uf_local, 5);
    assert_eq!(p[0], p[1], "corner [0,1,2] copies unioned");
    assert_eq!(p[1], p[2], "corner [0,1,2] third copy unioned via 1-ring");
    assert_eq!(p[3], p[4], "chained corner [2,3,4] copies unioned");
    assert_ne!(p[0], p[3], "distinct corners stay distinct");
}

#[test]
fn reconciliation_seeds_rebuild_for_cell_killing_one_sided_edge_collapse() {
    let vertices = vec![
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(5.0e-8, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    ];
    let vertex_keys = vec![
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 4],
        [1, 4, 5],
        [1, 2, 5],
    ];
    let cells = vec![VoronoiCell::new(0, 3), VoronoiCell::new(3, 3)];
    let cell_indices = vec![0, 1, 2, 3, 4, 5];

    let records = [edge_record(0, 1)];
    let mut reconciled_cells = cells.clone();
    let mut reconciled_indices = cell_indices.clone();
    let result = reconcile_edge_mismatches(
        &records,
        &vertices,
        &mut reconciled_cells,
        &mut reconciled_indices,
        VertexKeys::Flat(&vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        ReconcileOptions::with_apply(ReconcileApply::InPlace),
    )
    .expect("cell-killing one-sided edge should local_rebuild cleanly");

    assert_eq!(
        cell_sequences(&reconciled_cells, &reconciled_indices),
        cell_sequences(&cells, &cell_indices),
        "generator-preserving reconciliation must not collapse a cell"
    );
    assert!(
        !result.local_rebuild_seed_pairs.is_empty(),
        "cell-killing collapse must seed Hull3d"
    );
}

fn mismatched_shared_edge_fixture(
    dx: f32,
    dy: f32,
) -> (Vec<Vec3>, Vec<VertexKey>, Vec<VoronoiCell>, Vec<u32>) {
    let vertices = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0 + dx, dy, 0.0),
        Vec3::new(dy, 1.0 + dx, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    ];
    let vertex_keys = vec![
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 4],
        [0, 1, 5],
        [1, 4, 5],
    ];
    let cells = vec![VoronoiCell::new(0, 3), VoronoiCell::new(3, 3)];
    let cell_indices = vec![0, 1, 2, 3, 4, 5];
    (vertices, vertex_keys, cells, cell_indices)
}

fn one_shared_endpoint_fixture(dx: f32) -> (Vec<Vec3>, Vec<VertexKey>, Vec<VoronoiCell>, Vec<u32>) {
    let vertices = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(dx, 1.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    ];
    let vertex_keys = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 4], [1, 4, 5]];
    let cells = vec![VoronoiCell::new(0, 3), VoronoiCell::new(3, 3)];
    let cell_indices = vec![0, 1, 2, 0, 3, 4];
    (vertices, vertex_keys, cells, cell_indices)
}

#[test]
fn reconcile_bounds_one_shared_endpoint_inference() {
    let records = [edge_record(0, 1)];
    let eps = crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS;

    for (dx, expected_merges) in [(0.5 * eps, 1), (2.0 * eps, 0)] {
        let (vertices, vertex_keys, cells, cell_indices) = one_shared_endpoint_fixture(dx);
        let (_, merges) = collect_merges(
            &records,
            &vertices,
            &cells,
            &cell_indices,
            VertexKeys::Flat(&vertex_keys),
            eps,
            MergeMode::Primary,
            false,
            ReconcileOptions::default(),
        )
        .expect("one-shared-endpoint merge collection");
        assert_eq!(
            merges, expected_merges,
            "one-shared-endpoint inference at distance {dx}"
        );
    }
}

#[test]
fn reconcile_reconciles_epsilon_close_shared_edge_endpoints() {
    let (vertices, vertex_keys, cells, cell_indices) =
        mismatched_shared_edge_fixture(2.0e-7, 4.0e-8);
    let layout = LiveCellLayout::new(&cells, &cell_indices);

    let seg_a_before =
        edge_segments_for_neighbor(0, 1, layout, VertexKeys::Flat(&vertex_keys)).unwrap();
    let seg_b_before =
        edge_segments_for_neighbor(1, 0, layout, VertexKeys::Flat(&vertex_keys)).unwrap();
    assert_eq!(seg_a_before.len(), 1);
    assert_eq!(seg_b_before.len(), 1);
    let before_a = BTreeSet::from([seg_a_before[0].0, seg_a_before[0].1]);
    let before_b = BTreeSet::from([seg_b_before[0].0, seg_b_before[0].1]);
    assert_ne!(
        before_a, before_b,
        "fixture must start with mismatched shared-edge endpoint ids"
    );

    let records = [edge_record(0, 1)];
    let (changed, _, _, new_cells, new_indices) =
        run_both_backends(&records, &vertices, &cells, &cell_indices, &vertex_keys);
    assert!(
        changed,
        "expected mismatched shared-edge endpoints to be reconciled"
    );
    let seg_a = edge_segments_for_neighbor(
        0,
        1,
        LiveCellLayout::new(&new_cells, &new_indices),
        VertexKeys::Flat(&vertex_keys),
    )
    .unwrap();
    let seg_b = edge_segments_for_neighbor(
        1,
        0,
        LiveCellLayout::new(&new_cells, &new_indices),
        VertexKeys::Flat(&vertex_keys),
    )
    .unwrap();
    assert_eq!(seg_a.len(), 1, "cell 0 should still expose one shared edge");
    assert_eq!(seg_b.len(), 1, "cell 1 should still expose one shared edge");

    let set_a = BTreeSet::from([seg_a[0].0, seg_a[0].1]);
    let set_b = BTreeSet::from([seg_b[0].0, seg_b[0].1]);
    assert_eq!(
        set_a, set_b,
        "reconciled shared edge should use the same endpoint ids on both sides"
    );

    let mut footprint_cells = cells.clone();
    let mut footprint_indices = cell_indices.clone();
    let result = reconcile_edge_mismatches(
        &records,
        &vertices,
        &mut footprint_cells,
        &mut footprint_indices,
        VertexKeys::Flat(&vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        ReconcileOptions::with_apply(ReconcileApply::InPlace),
    )
    .expect("reconciliation should report its mutation footprint");
    assert!(!result.resolution_scan_cells.is_empty());
    assert!(result.resolution_scan_cells.contains(&0));
    assert!(result.resolution_scan_cells.contains(&1));
}

#[test]
fn reconcile_refuses_distant_shared_edge_endpoints() {
    let (vertices, vertex_keys, cells, cell_indices) =
        mismatched_shared_edge_fixture(1.0e-5, 2.0e-6);
    let records = [edge_record(0, 1)];

    let (changed, _, _, _, _) =
        run_both_backends(&records, &vertices, &cells, &cell_indices, &vertex_keys);
    assert!(!changed, "distant endpoint identities must not be merged");

    let mut reconciled_cells = cells.clone();
    let mut reconciled_indices = cell_indices.clone();
    let result = reconcile_edge_mismatches(
        &records,
        &vertices,
        &mut reconciled_cells,
        &mut reconciled_indices,
        VertexKeys::Flat(&vertex_keys),
        crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        ReconcileOptions::with_apply(ReconcileApply::InPlace),
    )
    .expect("distant mismatch should remain a controlled residual");
    assert!(
        !result.residual_pairs.is_empty(),
        "rejected endpoint pairing must remain visible to reconciliation/error handling"
    );
    assert!(
        result.resolution_scan_cells.is_empty(),
        "a rejected non-mutating proposal needs no resolution rescan"
    );
    assert_eq!(
        cell_sequences(&reconciled_cells, &reconciled_indices),
        cell_sequences(&cells, &cell_indices),
        "rejecting a distant pairing must not mutate cell boundaries"
    );
}
