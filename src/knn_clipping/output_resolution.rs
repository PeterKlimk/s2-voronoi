//! Final output-resolution canonicalization.
//!
//! This stage runs after reconciliation and optional Local3d repair. Its
//! baseline policy contracts distinct vertex ids whose stored f32 directions
//! are exactly equal, but only when the transaction preserves every effective
//! generator cell. Cell-killing components are retained for the default
//! `Preserve` policy and surfaced through telemetry.

use std::collections::BTreeMap;

use glam::Vec3;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::diagram::VoronoiCell;
use crate::OutputResolutionReport;

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<u32>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len as u32).collect(),
        }
    }

    fn find(&mut self, value: u32) -> u32 {
        let mut root = value;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        let mut current = value;
        while self.parent[current as usize] != current {
            let next = self.parent[current as usize];
            self.parent[current as usize] = root;
            current = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Lowest id is the deterministic surviving vertex.
        let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[drop as usize] = keep;
    }
}

#[derive(Debug)]
struct ZeroComponent {
    representative: u32,
    members: Vec<u32>,
    edge_count: usize,
}

#[inline]
fn same_stored_direction(a: Vec3, b: Vec3) -> bool {
    // `==` deliberately canonicalizes signed zero. Non-finite output vertices
    // are rejected before this stage can commit a transaction.
    a.x == b.x && a.y == b.y && a.z == b.z
}

fn state_error(message: impl Into<String>) -> crate::VoronoiError {
    crate::VoronoiError::ComputationFailed(message.into())
}

fn cell_span<'a>(
    cell_idx: usize,
    cells: &[VoronoiCell],
    cell_indices: &'a [u32],
) -> Result<&'a [u32], crate::VoronoiError> {
    let cell = cells.get(cell_idx).ok_or_else(|| {
        state_error(format!(
            "output resolution referenced out-of-range cell {cell_idx}"
        ))
    })?;
    let start = cell.vertex_start();
    let end = start
        .checked_add(cell.vertex_count())
        .ok_or_else(|| state_error("output-resolution cell span overflow"))?;
    cell_indices.get(start..end).ok_or_else(|| {
        state_error(format!(
            "output-resolution cell {cell_idx} span [{start}..{end}) exceeds index buffer len {}",
            cell_indices.len()
        ))
    })
}

/// Return each undirected edge whose distinct endpoint ids have exactly equal
/// stored directions. The two cell uses are deduplicated here.
fn collect_zero_edges(
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    let mut out = Vec::new();
    for cell_idx in 0..cells.len() {
        let span = cell_span(cell_idx, cells, cell_indices)?;
        if span.len() < 2 {
            continue;
        }
        for edge_idx in 0..span.len() {
            let a = span[edge_idx];
            let b = span[(edge_idx + 1) % span.len()];
            if a == b {
                continue;
            }
            let va = vertices.get(a as usize).copied().ok_or_else(|| {
                state_error(format!(
                    "output-resolution edge references out-of-range vertex {a}"
                ))
            })?;
            let vb = vertices.get(b as usize).copied().ok_or_else(|| {
                state_error(format!(
                    "output-resolution edge references out-of-range vertex {b}"
                ))
            })?;
            if same_stored_direction(va, vb) {
                out.push((a.min(b), a.max(b)));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn build_components(zero_edges: &[(u32, u32)]) -> (Vec<ZeroComponent>, FxHashMap<u32, usize>) {
    let mut member_ids = Vec::with_capacity(zero_edges.len() * 2);
    for &(a, b) in zero_edges {
        member_ids.extend([a, b]);
    }
    member_ids.sort_unstable();
    member_ids.dedup();
    let local_for_vertex: FxHashMap<u32, u32> = member_ids
        .iter()
        .enumerate()
        .map(|(local, &vertex)| (vertex, local as u32))
        .collect();
    let mut uf = DisjointSet::new(member_ids.len());
    for &(a, b) in zero_edges {
        uf.union(local_for_vertex[&a], local_for_vertex[&b]);
    }

    let mut members_by_root = BTreeMap::<u32, Vec<u32>>::new();
    for &(a, b) in zero_edges {
        let root = uf.find(local_for_vertex[&a]);
        members_by_root.entry(root).or_default().extend([a, b]);
    }
    for members in members_by_root.values_mut() {
        members.sort_unstable();
        members.dedup();
    }

    let mut edge_count_by_root = BTreeMap::<u32, usize>::new();
    for &(a, _) in zero_edges {
        *edge_count_by_root
            .entry(uf.find(local_for_vertex[&a]))
            .or_default() += 1;
    }

    let mut component_for_vertex = FxHashMap::default();
    let mut components = Vec::with_capacity(members_by_root.len());
    for (root, members) in members_by_root {
        let component_idx = components.len();
        for &member in &members {
            component_for_vertex.insert(member, component_idx);
        }
        components.push(ZeroComponent {
            representative: members[0],
            members,
            edge_count: edge_count_by_root[&root],
        });
    }
    (components, component_for_vertex)
}

/// Components which occur in one cell must be accepted or declined together:
/// otherwise a sequence of individually safe edits can jointly kill a cell,
/// and the result becomes order-dependent.
fn interaction_groups(
    components: &[ZeroComponent],
    component_for_vertex: &FxHashMap<u32, usize>,
    candidate_cells: &[usize],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Result<Vec<Vec<usize>>, crate::VoronoiError> {
    let mut uf = DisjointSet::new(components.len());
    for &cell_idx in candidate_cells {
        let span = cell_span(cell_idx, cells, cell_indices)?;
        let mut in_cell = Vec::new();
        for &vertex in span {
            let component = component_for_vertex
                .get(&vertex)
                .copied()
                .unwrap_or(usize::MAX);
            if component != usize::MAX && !in_cell.contains(&component) {
                in_cell.push(component);
            }
        }
        if let Some((&first, rest)) = in_cell.split_first() {
            for &other in rest {
                uf.union(first as u32, other as u32);
            }
        }
    }

    let mut by_root = BTreeMap::<u32, Vec<usize>>::new();
    for component in 0..components.len() {
        by_root
            .entry(uf.find(component as u32))
            .or_default()
            .push(component);
    }
    Ok(by_root.into_values().collect())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RewriteFailure {
    CellKilling,
    NonSimple,
}

fn rewrite_cycle(
    span: &[u32],
    mut replacement: impl FnMut(u32) -> u32,
) -> Result<Vec<u32>, RewriteFailure> {
    let mut rewritten = Vec::with_capacity(span.len());
    for &original in span {
        let mapped = replacement(original);
        if rewritten.last().copied() != Some(mapped) {
            rewritten.push(mapped);
        }
    }
    if rewritten.len() > 1 && rewritten[0] == *rewritten.last().unwrap() {
        rewritten.pop();
    }
    if rewritten.len() < 3 {
        return Err(RewriteFailure::CellKilling);
    }
    for i in 0..rewritten.len() {
        if rewritten[(i + 1)..].contains(&rewritten[i]) {
            return Err(RewriteFailure::NonSimple);
        }
    }
    Ok(rewritten)
}

#[derive(Default)]
struct EdgeUses {
    forward: usize,
    reverse: usize,
}

#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    u64::from(a.min(b)) | (u64::from(a.max(b)) << 32)
}

/// Local quotient certificate. Every changed edge is incident to a contracted
/// vertex, and every cell referencing such a vertex is in `affected_cells`.
/// This lets the production path prove edge agreement, incidence, duplicate
/// faces, and Euler preservation without sorting every edge in the diagram.
#[allow(clippy::too_many_arguments)]
fn verify_affected_quotient(
    vertices_before: &[u32],
    representatives: &[u32],
    removed_vertex_count: usize,
    old_cycles: &[(usize, Vec<u32>)],
    new_cells: &[VoronoiCell],
    new_indices: &[u32],
) -> Result<bool, crate::VoronoiError> {
    let contracted: FxHashSet<u32> = vertices_before.iter().copied().collect();
    let representatives: FxHashSet<u32> = representatives.iter().copied().collect();
    let mut old_edges = FxHashSet::default();
    let mut new_edges = FxHashMap::<u64, EdgeUses>::default();
    let mut rep_incidence = FxHashMap::<u32, usize>::default();
    let mut signatures = FxHashSet::<Vec<u32>>::default();

    for (cell_idx, old) in old_cycles {
        for edge_idx in 0..old.len() {
            let a = old[edge_idx];
            let b = old[(edge_idx + 1) % old.len()];
            if contracted.contains(&a) || contracted.contains(&b) {
                old_edges.insert(edge_key(a, b));
            }
        }

        let new = cell_span(*cell_idx, new_cells, new_indices)?;
        let mut signature = new.to_vec();
        signature.sort_unstable();
        if !signatures.insert(signature) {
            return Ok(false);
        }
        for &vertex in new {
            if representatives.contains(&vertex) {
                *rep_incidence.entry(vertex).or_default() += 1;
            }
        }
        for edge_idx in 0..new.len() {
            let a = new[edge_idx];
            let b = new[(edge_idx + 1) % new.len()];
            if a == b {
                return Ok(false);
            }
            if representatives.contains(&a) || representatives.contains(&b) {
                let uses = new_edges.entry(edge_key(a, b)).or_default();
                if a < b {
                    uses.forward += 1;
                } else {
                    uses.reverse += 1;
                }
            }
        }
    }

    if representatives
        .iter()
        .any(|representative| rep_incidence.get(representative).copied().unwrap_or(0) < 3)
    {
        return Ok(false);
    }
    if new_edges
        .values()
        .any(|uses| uses.forward != 1 || uses.reverse != 1)
    {
        return Ok(false);
    }
    if old_edges.len().checked_sub(new_edges.len()) != Some(removed_vertex_count) {
        return Ok(false);
    }

    Ok(true)
}

fn restore_cells(saved: &[(usize, Vec<u32>)], cells: &mut [VoronoiCell], cell_indices: &mut [u32]) {
    for (cell_idx, original) in saved {
        let start = cells[*cell_idx].vertex_start();
        cell_indices[start..start + original.len()].copy_from_slice(original);
        cells[*cell_idx] = VoronoiCell::new(start as u32, original.len() as u16);
    }
}

pub(super) fn collect_zero_edges_in_cells(
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    cell_ids: &[usize],
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    let mut out = Vec::new();
    for &cell_idx in cell_ids {
        let span = cell_span(cell_idx, cells, cell_indices)?;
        for edge_idx in 0..span.len() {
            let a = span[edge_idx];
            let b = span[(edge_idx + 1) % span.len()];
            if a == b {
                continue;
            }
            let va = vertex_pos_for_resolution(vertices, a)?;
            let vb = vertex_pos_for_resolution(vertices, b)?;
            if same_stored_direction(va, vb) {
                out.push((a.min(b), a.max(b)));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn vertex_pos_for_resolution(vertices: &[Vec3], vertex: u32) -> Result<Vec3, crate::VoronoiError> {
    vertices.get(vertex as usize).copied().ok_or_else(|| {
        state_error(format!(
            "output-resolution edge references out-of-range vertex {vertex}"
        ))
    })
}

/// Canonicalize exact stored-zero edges under the default generator-preserving
/// policy. Work beyond the initial edge scan is cold-path only.
pub(crate) fn canonicalize_exact_zero_edges(
    vertices: &[Vec3],
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    hinted_candidates: Option<Vec<(u32, u32)>>,
    localized_candidate_cells: Option<Vec<usize>>,
) -> Result<OutputResolutionReport, crate::VoronoiError> {
    let zero_edges = match hinted_candidates {
        Some(candidates) => candidates,
        None => collect_zero_edges(vertices, cells, cell_indices)?,
    };
    let mut report = OutputResolutionReport {
        exact_zero_edges_detected: zero_edges.len(),
        ..OutputResolutionReport::default()
    };
    if zero_edges.is_empty() {
        return Ok(report);
    }

    let (components, component_for_vertex) = build_components(&zero_edges);
    report.exact_zero_components_detected = components.len();
    let candidate_cells = localized_candidate_cells.unwrap_or_else(|| (0..cells.len()).collect());
    let groups = interaction_groups(
        &components,
        &component_for_vertex,
        &candidate_cells,
        cells,
        cell_indices,
    )?;

    let mut group_for_component = vec![usize::MAX; components.len()];
    for (group_idx, group) in groups.iter().enumerate() {
        for &component_idx in group {
            group_for_component[component_idx] = group_idx;
        }
    }

    // Groups are interaction-connected through cells, so one cell can touch
    // at most one group. Classify every group in one linear pass rather than
    // rebuilding and strictly validating the whole diagram per component.
    let mut group_failure = vec![None; groups.len()];
    let mut cells_by_group = vec![Vec::<usize>::new(); groups.len()];
    for &cell_idx in &candidate_cells {
        let span = cell_span(cell_idx, cells, cell_indices)?;
        let mut touched_group = None;
        let rewritten = rewrite_cycle(span, |vertex| {
            let component_idx = component_for_vertex
                .get(&vertex)
                .copied()
                .unwrap_or(usize::MAX);
            if component_idx == usize::MAX {
                return vertex;
            }
            let group_idx = group_for_component[component_idx];
            debug_assert!(touched_group.is_none_or(|seen| seen == group_idx));
            touched_group = Some(group_idx);
            components[component_idx].representative
        });
        if let Some(group_idx) = touched_group {
            cells_by_group[group_idx].push(cell_idx);
        }
        if let (Some(group_idx), Err(failure)) = (touched_group, rewritten) {
            group_failure[group_idx] = Some(match (group_failure[group_idx], failure) {
                (Some(RewriteFailure::CellKilling), _) | (_, RewriteFailure::CellKilling) => {
                    RewriteFailure::CellKilling
                }
                _ => RewriteFailure::NonSimple,
            });
        }
    }

    let mut replacements = FxHashMap::default();
    let mut accepted_components = 0usize;
    let mut accepted_edges = 0usize;
    let mut accepted_vertices = Vec::new();
    let mut accepted_representatives = Vec::new();
    let mut affected_cells = Vec::new();
    let mut removed_vertex_count = 0usize;
    for (group_idx, group) in groups.iter().enumerate() {
        match group_failure[group_idx] {
            Some(RewriteFailure::CellKilling) => {
                report.cell_killing_components_preserved += group.len();
            }
            Some(RewriteFailure::NonSimple) => {
                report.topology_rejected_components += group.len();
            }
            None => {
                accepted_components += group.len();
                affected_cells.extend_from_slice(&cells_by_group[group_idx]);
                for &component_idx in group {
                    let component = &components[component_idx];
                    accepted_edges += component.edge_count;
                    removed_vertex_count += component.members.len() - 1;
                    accepted_vertices.extend_from_slice(&component.members);
                    accepted_representatives.push(component.representative);
                    for &member in &component.members {
                        replacements.insert(member, component.representative);
                    }
                }
            }
        }
    }

    if !replacements.is_empty() {
        affected_cells.sort_unstable();
        affected_cells.dedup();
        let mut saved = Vec::with_capacity(affected_cells.len());
        for &cell_idx in &affected_cells {
            let original = cell_span(cell_idx, cells, cell_indices)?.to_vec();
            let rewritten = rewrite_cycle(&original, |vertex| {
                replacements.get(&vertex).copied().unwrap_or(vertex)
            })
            .map_err(|_| {
                state_error("preclassified exact-zero transaction changed classification")
            })?;
            let start = cells[cell_idx].vertex_start();
            cell_indices[start..start + rewritten.len()].copy_from_slice(&rewritten);
            cells[cell_idx] = VoronoiCell::new(start as u32, rewritten.len() as u16);
            saved.push((cell_idx, original));
        }

        let local_ok = verify_affected_quotient(
            &accepted_vertices,
            &accepted_representatives,
            removed_vertex_count,
            &saved,
            cells,
            cell_indices,
        )?;
        if !local_ok {
            restore_cells(&saved, cells, cell_indices);
            report.topology_rejected_components += accepted_components;
        } else {
            report.exact_zero_components_contracted = accepted_components;
            report.exact_zero_edges_contracted = accepted_edges;

            let local_remaining =
                collect_zero_edges_in_cells(vertices, cells, cell_indices, &affected_cells)?;
            let newly_exposed = local_remaining
                .iter()
                .filter(|edge| zero_edges.binary_search(edge).is_err())
                .count();
            report.exact_zero_edges_remaining = zero_edges.len() - accepted_edges + newly_exposed;
        }
    }

    if replacements.is_empty() || report.exact_zero_edges_contracted == 0 {
        report.exact_zero_edges_remaining = zero_edges.len();
    }
    if std::env::var_os("VORONOI_MESH_RESOLUTION_KV").is_some() {
        eprintln!(
            "OUTPUT_RESOLUTION_KV detected_edges={} detected_components={} contracted_edges={} contracted_components={} preserved_cell_killing={} rejected_topology={} remaining_edges={}",
            report.exact_zero_edges_detected,
            report.exact_zero_components_detected,
            report.exact_zero_edges_contracted,
            report.exact_zero_components_contracted,
            report.cell_killing_components_preserved,
            report.topology_rejected_components,
            report.exact_zero_edges_remaining,
        );
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let candidate_cells =
            incident_candidate_cells(&candidates, &fixture.cells, &fixture.indices);

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
            exhaustive.exact_zero_edges_detected, expected.edges,
            "{context}"
        );
        assert_eq!(
            exhaustive.exact_zero_components_detected, expected.components,
            "{context}"
        );
        assert_eq!(
            exhaustive.exact_zero_edges_contracted, expected.contracted_edges,
            "{context}"
        );
        assert_eq!(
            exhaustive.cell_killing_components_preserved, expected.preserved_components,
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

        assert_eq!(report.exact_zero_edges_detected, 1);
        assert_eq!(report.exact_zero_edges_contracted, 1);
        assert_eq!(report.exact_zero_components_contracted, 1);
        assert_eq!(report.exact_zero_edges_remaining, 0);
        assert_eq!(cells[0].vertex_count(), 3);
        assert_eq!(cells[2].vertex_count(), 3);
        assert!(crate::validation::verify_sphere_effective_strict(
            &generators,
            &vertices,
            &cells,
            &indices
        )
        .is_ok());

        let after =
            crate::SphericalVoronoi::from_raw_parts(generators, vertices, cells, indices, None);
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

        assert_eq!(report.exact_zero_edges_detected, 1);
        assert_eq!(report.exact_zero_edges_contracted, 0);
        assert_eq!(report.cell_killing_components_preserved, 1);
        assert_eq!(report.exact_zero_edges_remaining, 1);
        assert_eq!(indices, original);
        assert!(cells.iter().all(|cell| cell.vertex_count() == 3));
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
}
