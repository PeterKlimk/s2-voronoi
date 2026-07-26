//! Final output-resolution canonicalization.
//!
//! This stage runs after reconciliation and optional Hull3d rebuilding. Its
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

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct CanonicalizationOutcome {
    pub report: OutputResolutionReport,
    /// Effective generator cells that would become unrepresentable if their
    /// interacting exact-zero transaction were committed.
    pub cell_killing_generators: Vec<usize>,
    /// Cells rewritten by a committed exact transaction.
    pub changed_cells: Vec<usize>,
    /// Stable ids identified by the committed exact transaction.
    contracted_vertices: Vec<u32>,
    /// Representatives retained by the committed exact transaction.
    retained_representatives: Vec<u32>,
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

#[inline]
fn stored_chord_squared(a: Vec3, b: Vec3) -> f64 {
    let dx = f64::from(a.x) - f64::from(b.x);
    let dy = f64::from(a.y) - f64::from(b.y);
    let dz = f64::from(a.z) - f64::from(b.z);
    dx * dx + dy * dy + dz * dz
}

pub(super) fn collect_positive_edges_in_cells(
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    cell_ids: &[usize],
    threshold_squared: f64,
) -> Result<Vec<(f64, u32, u32)>, crate::VoronoiError> {
    let mut out = Vec::new();
    for &cell_idx in cell_ids {
        let span = cell_span(cell_idx, cells, cell_indices)?;
        for edge_idx in 0..span.len() {
            let a = span[edge_idx];
            let b = span[(edge_idx + 1) % span.len()];
            if a == b {
                continue;
            }
            let distance_squared = stored_chord_squared(
                vertex_pos_for_resolution(vertices, a)?,
                vertex_pos_for_resolution(vertices, b)?,
            );
            if distance_squared > 0.0 && distance_squared <= threshold_squared {
                out.push((distance_squared, a.min(b), a.max(b)));
            }
        }
    }
    out.sort_unstable_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    out.dedup_by(|left, right| left.1 == right.1 && left.2 == right.2);
    Ok(out)
}

#[derive(Debug, Default)]
pub(crate) struct PositiveResolutionReport {
    pub(crate) hinted_cells: usize,
    pub(crate) confirmed_candidates: usize,
    pub(crate) attempted_contractions: usize,
    pub(crate) accepted_contractions: usize,
    pub(crate) displacement_declines: usize,
    pub(crate) cell_declined_components: usize,
    pub(crate) topology_declined_components: usize,
    pub(crate) newly_exposed_positive_edges: usize,
    pub(crate) max_representative_displacement_bound: f64,
    pub(crate) changed_cells: Vec<usize>,
}

struct BoundedComponents {
    parent: Vec<u32>,
    representative: Vec<u32>,
    members: Vec<usize>,
    radius: Vec<f64>,
}

impl BoundedComponents {
    fn new(vertices: &[u32]) -> Self {
        Self {
            parent: (0..vertices.len() as u32).collect(),
            representative: vertices.to_vec(),
            members: vec![1; vertices.len()],
            radius: vec![0.0; vertices.len()],
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

    fn try_union(
        &mut self,
        a: u32,
        b: u32,
        vertices: &[Vec3],
        threshold: f64,
    ) -> Result<bool, crate::VoronoiError> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return Ok(false);
        }
        let ia = ra as usize;
        let ib = rb as usize;
        let rep_a = self.representative[ia];
        let rep_b = self.representative[ib];
        let distance = stored_chord_squared(
            vertex_pos_for_resolution(vertices, rep_a)?,
            vertex_pos_for_resolution(vertices, rep_b)?,
        )
        .sqrt();
        let bound_a = self.radius[ia].max(distance + self.radius[ib]);
        let bound_b = self.radius[ib].max(distance + self.radius[ia]);
        let admissible_a = bound_a <= threshold;
        let admissible_b = bound_b <= threshold;
        if !admissible_a && !admissible_b {
            return Ok(false);
        }
        let retain_a = match (admissible_a, admissible_b) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => bound_a
                .total_cmp(&bound_b)
                .then_with(|| self.members[ib].cmp(&self.members[ia]))
                .then_with(|| rep_a.cmp(&rep_b))
                .is_le(),
            (false, false) => unreachable!(),
        };
        let (keep, drop, bound) = if retain_a {
            (ia, ib, bound_a)
        } else {
            (ib, ia, bound_b)
        };
        self.parent[drop] = keep as u32;
        self.members[keep] += self.members[drop];
        self.radius[keep] = bound;
        Ok(true)
    }
}

fn complete_links_are_single_cycles(
    vertices: &FxHashSet<u32>,
    cell_ids: &[usize],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Result<bool, crate::VoronoiError> {
    let mut links = FxHashMap::<u32, Vec<(u32, u32)>>::default();
    for &cell_idx in cell_ids {
        let cycle = cell_span(cell_idx, cells, cell_indices)?;
        for position in 0..cycle.len() {
            let vertex = cycle[position];
            if vertices.contains(&vertex) {
                links.entry(vertex).or_default().push((
                    cycle[(position + cycle.len() - 1) % cycle.len()],
                    cycle[(position + 1) % cycle.len()],
                ));
            }
        }
    }
    for edges in links.values() {
        let mut next_for = FxHashMap::<u32, u32>::default();
        let mut incoming = FxHashSet::<u32>::default();
        for &(from, to) in edges {
            if from == to || next_for.insert(from, to).is_some() || !incoming.insert(to) {
                return Ok(false);
            }
        }
        if next_for.len() != incoming.len()
            || next_for.keys().any(|vertex| !incoming.contains(vertex))
        {
            return Ok(false);
        }
        let Some(&start) = next_for.keys().next() else {
            continue;
        };
        let mut current = start;
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(current) {
                if current != start {
                    return Ok(false);
                }
                break;
            }
            let Some(&next) = next_for.get(&current) else {
                return Ok(false);
            };
            current = next;
        }
        if visited.len() != next_for.len() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Apply one deterministic, cell-preserving positive-resolution batch.
/// `candidates` is the terminal source set, `component_cells` contains every
/// use of a selected endpoint, and `certificate_cells` additionally contains
/// the complete stars of vertices in rewritten cells.
#[allow(clippy::too_many_arguments)]
pub(super) fn simplify_positive_edges(
    vertices: &[Vec3],
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    candidates: Vec<(f64, u32, u32)>,
    component_cells: &[usize],
    certificate_cells: &[usize],
    threshold: f64,
    hinted_cells: usize,
) -> Result<PositiveResolutionReport, crate::VoronoiError> {
    let threshold_squared = threshold * threshold;
    let mut report = PositiveResolutionReport {
        hinted_cells,
        confirmed_candidates: candidates.len(),
        attempted_contractions: candidates.len(),
        ..PositiveResolutionReport::default()
    };
    if candidates.is_empty() {
        return Ok(report);
    }

    let mut endpoint_ids = Vec::with_capacity(candidates.len() * 2);
    for &(_, a, b) in &candidates {
        endpoint_ids.extend([a, b]);
    }
    endpoint_ids.sort_unstable();
    endpoint_ids.dedup();
    let local_for_vertex: FxHashMap<u32, u32> = endpoint_ids
        .iter()
        .enumerate()
        .map(|(local, &vertex)| (vertex, local as u32))
        .collect();
    let mut bounded = BoundedComponents::new(&endpoint_ids);
    let mut accepted_unions = 0usize;
    for &(_, a, b) in &candidates {
        if bounded.try_union(
            local_for_vertex[&a],
            local_for_vertex[&b],
            vertices,
            threshold,
        )? {
            accepted_unions += 1;
        } else if bounded.find(local_for_vertex[&a]) != bounded.find(local_for_vertex[&b]) {
            report.displacement_declines += 1;
        }
    }

    let mut members_by_root = BTreeMap::<u32, Vec<u32>>::new();
    for &vertex in &endpoint_ids {
        let root = bounded.find(local_for_vertex[&vertex]);
        members_by_root.entry(root).or_default().push(vertex);
    }
    let mut component_for_vertex = FxHashMap::default();
    let mut components = Vec::new();
    let mut component_bounds = Vec::new();
    for (root, members) in members_by_root {
        if members.len() < 2 {
            continue;
        }
        let component_idx = components.len();
        for &member in &members {
            component_for_vertex.insert(member, component_idx);
        }
        component_bounds.push(bounded.radius[root as usize]);
        components.push(ZeroComponent {
            representative: bounded.representative[root as usize],
            edge_count: members.len() - 1,
            members,
        });
    }
    debug_assert_eq!(
        components
            .iter()
            .map(|component| component.edge_count)
            .sum::<usize>(),
        accepted_unions
    );
    if components.is_empty() {
        return Ok(report);
    }

    let groups = interaction_groups(
        &components,
        &component_for_vertex,
        component_cells,
        cells,
        cell_indices,
    )?;
    let mut group_for_component = vec![usize::MAX; components.len()];
    for (group_idx, group) in groups.iter().enumerate() {
        for &component_idx in group {
            group_for_component[component_idx] = group_idx;
        }
    }
    let mut group_failure = vec![None; groups.len()];
    let mut cells_by_group = vec![Vec::<usize>::new(); groups.len()];
    for &cell_idx in component_cells {
        let span = cell_span(cell_idx, cells, cell_indices)?;
        let mut touched_group = None;
        let rewritten = rewrite_cycle(span, |vertex| {
            let Some(&component_idx) = component_for_vertex.get(&vertex) else {
                return vertex;
            };
            let group_idx = group_for_component[component_idx];
            debug_assert!(touched_group.is_none_or(|seen| seen == group_idx));
            touched_group = Some(group_idx);
            components[component_idx].representative
        });
        if let Some(group_idx) = touched_group {
            cells_by_group[group_idx].push(cell_idx);
            if let Err(failure) = rewritten {
                group_failure[group_idx] = Some(match (group_failure[group_idx], failure) {
                    (Some(RewriteFailure::CellKilling), _) | (_, RewriteFailure::CellKilling) => {
                        RewriteFailure::CellKilling
                    }
                    _ => RewriteFailure::NonSimple,
                });
            }
        }
    }

    let mut replacements = FxHashMap::default();
    let mut affected_cells = Vec::new();
    let mut accepted_vertices = Vec::new();
    let mut accepted_representatives = Vec::new();
    let mut removed_vertex_count = 0usize;
    let mut accepted_components = 0usize;
    for (group_idx, group) in groups.iter().enumerate() {
        if let Some(failure) = group_failure[group_idx] {
            match failure {
                RewriteFailure::CellKilling => report.cell_declined_components += group.len(),
                RewriteFailure::NonSimple => report.topology_declined_components += group.len(),
            }
            continue;
        }
        accepted_components += group.len();
        affected_cells.extend_from_slice(&cells_by_group[group_idx]);
        for &component_idx in group {
            let component = &components[component_idx];
            report.max_representative_displacement_bound = report
                .max_representative_displacement_bound
                .max(component_bounds[component_idx]);
            removed_vertex_count += component.members.len() - 1;
            accepted_vertices.extend_from_slice(&component.members);
            accepted_representatives.push(component.representative);
            for &member in &component.members {
                replacements.insert(member, component.representative);
            }
        }
    }
    if replacements.is_empty() {
        return Ok(report);
    }
    let positive_removed_vertex_count = removed_vertex_count;
    affected_cells.sort_unstable();
    affected_cells.dedup();

    let mut saved = Vec::with_capacity(certificate_cells.len());
    for &cell_idx in certificate_cells {
        saved.push((cell_idx, cell_span(cell_idx, cells, cell_indices)?.to_vec()));
    }
    for &cell_idx in &affected_cells {
        let original = cell_span(cell_idx, cells, cell_indices)?.to_vec();
        let rewritten = rewrite_cycle(&original, |vertex| {
            replacements.get(&vertex).copied().unwrap_or(vertex)
        })
        .map_err(|_| state_error("preclassified positive transaction changed classification"))?;
        let start = cells[cell_idx].vertex_start();
        cell_indices[start..start + rewritten.len()].copy_from_slice(&rewritten);
        cells[cell_idx] = VoronoiCell::new(start as u32, rewritten.len() as u16);
    }

    let induced_exact =
        collect_zero_edges_in_cells(vertices, cells, cell_indices, &affected_cells)?;
    if !induced_exact.is_empty() {
        let exact = canonicalize_exact_zero_edges(
            vertices,
            cells,
            cell_indices,
            Some(induced_exact),
            Some(certificate_cells.to_vec()),
        )?;
        if exact.report.exact_zero_edges_remaining != 0 {
            restore_cells(&saved, cells, cell_indices);
            report.topology_declined_components += accepted_components;
            report.max_representative_displacement_bound = 0.0;
            return Ok(report);
        }
        removed_vertex_count += exact
            .contracted_vertices
            .len()
            .saturating_sub(exact.retained_representatives.len());
        accepted_vertices.extend(exact.contracted_vertices);
        accepted_representatives.extend(exact.retained_representatives);
        affected_cells.extend(exact.changed_cells);
        affected_cells.sort_unstable();
        affected_cells.dedup();
    }

    let mut affected_vertices = FxHashSet::default();
    for &cell_idx in &affected_cells {
        affected_vertices.extend(cell_span(cell_idx, cells, cell_indices)?.iter().copied());
    }
    let quotient_ok = verify_affected_quotient(
        &accepted_vertices,
        &accepted_representatives,
        removed_vertex_count,
        &saved,
        cells,
        cell_indices,
    )? && complete_links_are_single_cycles(
        &affected_vertices,
        certificate_cells,
        cells,
        cell_indices,
    )?;
    if !quotient_ok {
        restore_cells(&saved, cells, cell_indices);
        report.topology_declined_components += accepted_components;
        report.max_representative_displacement_bound = 0.0;
        return Ok(report);
    }

    let source_edges: FxHashSet<(u32, u32)> = candidates.iter().map(|&(_, a, b)| (a, b)).collect();
    let remaining = collect_positive_edges_in_cells(
        vertices,
        cells,
        cell_indices,
        &affected_cells,
        threshold_squared,
    )?;
    report.newly_exposed_positive_edges = remaining
        .iter()
        .filter(|&&(_, a, b)| !source_edges.contains(&(a, b)))
        .count();
    report.accepted_contractions = positive_removed_vertex_count;
    report.changed_cells = affected_cells;
    Ok(report)
}

/// Canonicalize exact stored-zero edges under the default generator-preserving
/// policy. Work beyond the initial edge scan is cold-path only.
pub(super) fn canonicalize_exact_zero_edges(
    vertices: &[Vec3],
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    hinted_candidates: Option<Vec<(u32, u32)>>,
    localized_candidate_cells: Option<Vec<usize>>,
) -> Result<CanonicalizationOutcome, crate::VoronoiError> {
    let zero_edges = match hinted_candidates {
        Some(candidates) => candidates,
        None => collect_zero_edges(vertices, cells, cell_indices)?,
    };
    let mut report = OutputResolutionReport {
        exact_zero_edges_detected: zero_edges.len(),
        ..OutputResolutionReport::default()
    };
    if zero_edges.is_empty() {
        return Ok(CanonicalizationOutcome {
            report,
            cell_killing_generators: Vec::new(),
            changed_cells: Vec::new(),
            contracted_vertices: Vec::new(),
            retained_representatives: Vec::new(),
        });
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
    let mut cell_killing_generators = Vec::new();
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
            if failure == RewriteFailure::CellKilling {
                cell_killing_generators.push(cell_idx);
            }
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

    let mut changed_cells = Vec::new();
    let mut committed_vertices = Vec::new();
    let mut committed_representatives = Vec::new();
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
            changed_cells = affected_cells.clone();
            committed_vertices = accepted_vertices;
            committed_representatives = accepted_representatives;

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
    cell_killing_generators.sort_unstable();
    cell_killing_generators.dedup();
    Ok(CanonicalizationOutcome {
        report,
        cell_killing_generators,
        changed_cells,
        contracted_vertices: committed_vertices,
        retained_representatives: committed_representatives,
    })
}

/// Cold global transaction used by explicit cell-mesh elision.
#[derive(Debug)]
pub(crate) struct EffectiveCellElision {
    pub diagram: crate::SphericalVoronoi,
    pub effective_to_cell: Vec<Option<u32>>,
    pub cell_to_effective: Vec<u32>,
    pub zero_edges_before: usize,
    pub zero_components_before: usize,
    pub effective_cells_elided: usize,
    pub degree_two_vertices_suppressed: usize,
    pub max_suppression_cross_track_radians: f64,
}

fn rewrite_cycle_for_elision(
    span: &[u32],
    replacements: &FxHashMap<u32, u32>,
) -> Result<Vec<u32>, crate::VoronoiError> {
    let mut rewritten = Vec::with_capacity(span.len());
    for &original in span {
        let mapped = replacements.get(&original).copied().unwrap_or(original);
        if rewritten.last().copied() != Some(mapped) {
            rewritten.push(mapped);
        }
    }
    if rewritten.len() > 1 && rewritten[0] == *rewritten.last().unwrap() {
        rewritten.pop();
    }
    if rewritten.len() >= 3 {
        for i in 0..rewritten.len() {
            if rewritten[(i + 1)..].contains(&rewritten[i]) {
                return Err(state_error(
                    "cell elision produced a non-simple surviving cell",
                ));
            }
        }
    }
    Ok(rewritten)
}

/// In an oriented closed 2-manifold, the link of every live vertex is one
/// directed cycle. Edge pairing alone does not reject two closed fans pinched
/// together at one vertex, so elision checks this explicitly.
fn elision_links_are_single_cycles(diagram: &crate::SphericalVoronoi) -> bool {
    let mut link_edges = vec![Vec::<(u32, u32)>::new(); diagram.num_vertices()];
    for cell in diagram.iter_cells() {
        let cycle = cell.vertex_indices;
        for i in 0..cycle.len() {
            let vertex = cycle[i] as usize;
            let prev = cycle[(i + cycle.len() - 1) % cycle.len()];
            let next = cycle[(i + 1) % cycle.len()];
            if vertex >= link_edges.len() || prev == next {
                return false;
            }
            link_edges[vertex].push((prev, next));
        }
    }

    for edges in link_edges.into_iter().filter(|edges| !edges.is_empty()) {
        let mut next_for = FxHashMap::<u32, u32>::default();
        let mut incoming = FxHashSet::<u32>::default();
        for (from, to) in &edges {
            if next_for.insert(*from, *to).is_some() || !incoming.insert(*to) {
                return false;
            }
        }
        if next_for.len() != incoming.len()
            || next_for.keys().any(|vertex| !incoming.contains(vertex))
        {
            return false;
        }

        let start = edges[0].0;
        let mut current = start;
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(current) {
                if current != start {
                    return false;
                }
                break;
            }
            let Some(&next) = next_for.get(&current) else {
                return false;
            };
            current = next;
        }
        if visited.len() != next_for.len() {
            return false;
        }
    }
    true
}

/// Removing a face can leave an ordinary boundary vertex incident to only the
/// same two surviving faces. Those faces then share two consecutive edges;
/// suppressing the degree-two subdivision point merges them into one edge and
/// preserves `V - E + F`. The two owner rotations must agree exactly.
fn suppress_elision_degree_two_vertices(
    cycles: &mut [Option<Vec<u32>>],
    vertex_count: usize,
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    let mut suppressed = Vec::new();
    loop {
        let mut incident = vec![Vec::<usize>::new(); vertex_count];
        for (cell, cycle) in cycles.iter().enumerate() {
            let Some(cycle) = cycle else { continue };
            for &vertex in cycle {
                let Some(owners) = incident.get_mut(vertex as usize) else {
                    return Err(state_error(
                        "cell elision cycle references an out-of-range vertex",
                    ));
                };
                owners.push(cell);
            }
        }

        if incident
            .iter()
            .any(|owners| !owners.is_empty() && owners.len() == 1)
        {
            return Err(state_error(
                "cell elision produced a degree-one boundary vertex",
            ));
        }
        let Some((vertex, owners)) = incident
            .iter()
            .enumerate()
            .find(|(_, owners)| owners.len() == 2)
        else {
            break;
        };
        let owners = [owners[0], owners[1]];
        let mut rotations = [(0u32, 0u32, 0usize); 2];
        for (slot, &cell) in owners.iter().enumerate() {
            let cycle = cycles[cell].as_ref().unwrap();
            let position = cycle
                .iter()
                .position(|&candidate| candidate as usize == vertex)
                .ok_or_else(|| state_error("cell elision lost vertex incidence"))?;
            rotations[slot] = (
                cycle[(position + cycle.len() - 1) % cycle.len()],
                cycle[(position + 1) % cycle.len()],
                position,
            );
        }
        if rotations[0].0 != rotations[1].1 || rotations[0].1 != rotations[1].0 {
            return Err(state_error(
                "cell-elision degree-two suppression owner rotations disagree",
            ));
        }

        for (slot, &cell) in owners.iter().enumerate().rev() {
            let cycle = cycles[cell].as_mut().unwrap();
            cycle.remove(rotations[slot].2);
            if cycle.len() < 3 {
                cycles[cell] = None;
            }
        }
        suppressed.push((vertex as u32, rotations[0].0, rotations[0].1));
    }
    Ok(suppressed)
}

pub(crate) fn elide_exact_zero_cells_for_mesh(
    generators: &[Vec3],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Result<EffectiveCellElision, crate::VoronoiError> {
    if generators.len() != cells.len() {
        return Err(state_error("cell-elision generator/cell count mismatch"));
    }
    let zero_edges = collect_zero_edges(vertices, cells, cell_indices)?;
    let (components, _) = build_components(&zero_edges);
    let mut replacements = FxHashMap::default();
    for component in &components {
        for &member in &component.members {
            replacements.insert(member, component.representative);
        }
    }

    let mut rewritten_cycles = Vec::with_capacity(cells.len());
    for effective in 0..cells.len() {
        let rewritten =
            rewrite_cycle_for_elision(cell_span(effective, cells, cell_indices)?, &replacements)?;
        if rewritten.len() < 3 {
            rewritten_cycles.push(None);
            continue;
        }
        rewritten_cycles.push(Some(rewritten));
    }
    let suppressed_vertices =
        suppress_elision_degree_two_vertices(&mut rewritten_cycles, vertices.len())?;
    let mut max_suppression_cross_track_radians = 0.0f64;
    for &(vertex, start, end) in &suppressed_vertices {
        let point = vertices[vertex as usize].as_dvec3();
        let start = vertices[start as usize].as_dvec3();
        let end = vertices[end as usize].as_dvec3();
        let normal = start.cross(end);
        let normal_len = normal.length();
        if !normal_len.is_finite() || normal_len == 0.0 {
            return Err(state_error(
                "cell-elision degree-two suppression has an undefined replacement arc",
            ));
        }
        let cross_track = (point.dot(normal / normal_len).abs().min(1.0)).asin();
        max_suppression_cross_track_radians = max_suppression_cross_track_radians.max(cross_track);
    }

    let mut effective_to_cell = vec![None; cells.len()];
    let mut cell_to_effective = Vec::with_capacity(cells.len());
    let mut final_generators = Vec::with_capacity(generators.len());
    let mut final_cells = Vec::with_capacity(cells.len());
    let mut final_indices = Vec::with_capacity(cell_indices.len());
    for (effective, cycle) in rewritten_cycles.into_iter().enumerate() {
        let Some(cycle) = cycle else { continue };
        let final_cell = final_cells.len() as u32;
        effective_to_cell[effective] = Some(final_cell);
        cell_to_effective.push(effective as u32);
        final_generators.push(generators[effective]);
        final_cells.push(VoronoiCell::new(
            final_indices.len() as u32,
            cycle.len() as u16,
        ));
        final_indices.extend(cycle);
    }

    let effective_cells_elided = cells.len() - final_cells.len();
    if !zero_edges.is_empty() && effective_cells_elided == 0 {
        return Err(state_error(
            "exact-zero cell quotient did not eliminate a cell",
        ));
    }
    let mut diagram = crate::SphericalVoronoi::from_raw_parts(
        final_generators,
        vertices.to_vec(),
        final_cells,
        final_indices,
        None,
    );
    diagram.compact_vertices();

    if !elision_links_are_single_cycles(&diagram) {
        return Err(state_error(
            "cell elision produced a disconnected vertex link",
        ));
    }
    let validation = crate::validation::validate(&diagram);
    if !validation.is_strictly_valid() {
        return Err(state_error(format!(
            "cell elision failed strict validation: {}",
            validation.headline()
        )));
    }
    if validation.zero_length_edges != 0 {
        return Err(state_error(format!(
            "cell elision retained {} exact-zero edges",
            validation.zero_length_edges
        )));
    }

    Ok(EffectiveCellElision {
        diagram,
        effective_to_cell,
        cell_to_effective,
        zero_edges_before: zero_edges.len(),
        zero_components_before: components.len(),
        effective_cells_elided,
        degree_two_vertices_suppressed: suppressed_vertices.len(),
        max_suppression_cross_track_radians,
    })
}

#[cfg(test)]
mod tests;
