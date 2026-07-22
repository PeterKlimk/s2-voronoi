//! Cold positive-threshold simplification for explicit cell-mesh conversion.

use std::collections::BTreeMap;

use glam::Vec3;
use rustc_hash::{FxHashMap, FxHashSet};

type EdgeKey = (u32, u32);
type OptionalCycles = Vec<Option<Vec<u32>>>;
type StableLedger = Vec<Vec<u32>>;
type ExactClosureResult = (OptionalCycles, StableLedger, usize, usize);
type CompactedOutput = (Vec<Vec3>, Vec<Vec<u32>>, Vec<Option<u32>>, Vec<u32>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CellPolicy {
    Preserve,
    Error,
    Elide,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub diameter_pair_comparisons: u64,
    pub cell_index_visits: u64,
    pub provenance_member_checks: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WorkStats {
    pub diameter_pair_comparisons: u64,
    pub cell_index_visits: u64,
    pub provenance_member_checks: u64,
    pub candidate_high_water: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Preparation,
    SourcePreflight,
    Exact,
    Positive,
    Suppression,
    FinalCertification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailureKind {
    UnsupportedStoredDegeneracy,
    UnresolvedExactGroup,
    CellEliminationRequired,
    UnsafeQuotient,
    IllConditionedReplacementArc,
    PositiveSuppressionDeviation,
    DiameterLimit,
    CellIndexLimit,
    ProvenanceLimit,
    CounterOverflow,
    Validation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuppressionCause {
    Exact,
    Positive,
}

#[derive(Debug)]
pub(crate) struct Failure {
    pub kind: FailureKind,
    pub phase: Phase,
    pub message: String,
    pub work: WorkStats,
    pub affected_effective_cells: Vec<usize>,
}

#[derive(Debug, Default)]
pub(crate) struct ResultStats {
    pub round_attempts: usize,
    pub productive_rounds: usize,
    pub exact_candidate_occurrences: u64,
    pub positive_candidate_occurrences: u64,
    pub later_round_candidate_occurrences: u64,
    pub committed_transactions: usize,
    pub exact_components_committed: usize,
    pub positive_components_committed: usize,
    pub positive_components_declined_diameter: u64,
    pub positive_groups_declined_cell: u64,
    pub positive_groups_declined_topology: u64,
    pub final_positive_edges: usize,
    pub final_exact_edges: usize,
    pub vertices_retired: usize,
    pub max_component_members: usize,
    pub max_component_diameter: f64,
    pub max_representative_displacement: f64,
    pub effective_cells_elided: usize,
    pub exact_suppression_members: usize,
    pub max_exact_suppression_cross_track_radians: f64,
    pub positive_suppression_members: usize,
    pub max_positive_suppression_unit_arc_chord: f64,
}

#[derive(Debug)]
pub(crate) struct Outcome {
    pub vertices: Vec<Vec3>,
    pub cycles: Vec<Vec<u32>>,
    pub effective_to_cell: Vec<Option<u32>>,
    pub cell_to_effective: Vec<u32>,
    pub stats: ResultStats,
    pub work: WorkStats,
}

#[derive(Debug)]
struct WorkTracker {
    limits: Limits,
    stats: WorkStats,
}

impl WorkTracker {
    fn new(limits: Limits) -> Self {
        Self {
            limits,
            stats: WorkStats::default(),
        }
    }

    fn charge_cell_indices(&mut self, count: usize, phase: Phase) -> Result<(), Failure> {
        self.charge_many(
            count as u64,
            self.limits.cell_index_visits,
            phase,
            FailureKind::CellIndexLimit,
            |stats| &mut stats.cell_index_visits,
        )
    }

    fn charge_diameter_pair(&mut self, phase: Phase) -> Result<(), Failure> {
        self.charge_many(
            1,
            self.limits.diameter_pair_comparisons,
            phase,
            FailureKind::DiameterLimit,
            |stats| &mut stats.diameter_pair_comparisons,
        )
    }

    fn charge_provenance_member(&mut self, phase: Phase) -> Result<(), Failure> {
        self.charge_many(
            1,
            self.limits.provenance_member_checks,
            phase,
            FailureKind::ProvenanceLimit,
            |stats| &mut stats.provenance_member_checks,
        )
    }

    fn charge_many(
        &mut self,
        count: u64,
        limit: u64,
        phase: Phase,
        limit_kind: FailureKind,
        field: impl Fn(&mut WorkStats) -> &mut u64,
    ) -> Result<(), Failure> {
        let current = *field(&mut self.stats);
        let Some(next) = current.checked_add(count) else {
            return Err(self.failure(
                FailureKind::CounterOverflow,
                phase,
                "simplification work counter overflow",
            ));
        };
        if next > limit {
            return Err(self.failure(limit_kind, phase, "simplification work limit exceeded"));
        }
        *field(&mut self.stats) = next;
        Ok(())
    }

    fn note_candidates(&mut self, count: usize) {
        self.stats.candidate_high_water = self.stats.candidate_high_water.max(count as u64);
    }

    fn failure(&self, kind: FailureKind, phase: Phase, message: impl Into<String>) -> Failure {
        Failure {
            kind,
            phase,
            message: message.into(),
            work: self.stats,
            affected_effective_cells: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EdgeUse {
    cell: usize,
    from: u32,
    to: u32,
}

#[derive(Debug, Clone, Copy)]
struct EdgeRecord {
    first: EdgeUse,
    second: Option<EdgeUse>,
}

#[derive(Debug)]
struct Snapshot {
    edges: FxHashMap<EdgeKey, EdgeRecord>,
    incidence: Vec<Vec<usize>>,
    live_vertices: Vec<bool>,
}

fn edge_key(a: u32, b: u32) -> EdgeKey {
    (a.min(b), a.max(b))
}

fn same_stored_position(a: Vec3, b: Vec3) -> bool {
    a.x == b.x && a.y == b.y && a.z == b.z
}

fn exactly_antipodal(a: Vec3, b: Vec3) -> bool {
    a.x == -b.x && a.y == -b.y && a.z == -b.z
}

fn stored_distance_squared(a: Vec3, b: Vec3) -> f64 {
    let delta = a.as_dvec3() - b.as_dvec3();
    delta.length_squared()
}

fn build_snapshot(
    cycles: &[Option<Vec<u32>>],
    vertex_count: usize,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<Snapshot, Failure> {
    let mut edges = FxHashMap::<EdgeKey, EdgeRecord>::default();
    let mut incidence = vec![Vec::new(); vertex_count];
    let mut live_vertices = vec![false; vertex_count];

    for (cell, cycle) in cycles.iter().enumerate() {
        let Some(cycle) = cycle else { continue };
        for (offset, &from) in cycle.iter().enumerate() {
            work.charge_cell_indices(1, phase)?;
            let to = cycle[(offset + 1) % cycle.len()];
            if from as usize >= vertex_count || to as usize >= vertex_count {
                return Err(work.failure(
                    FailureKind::UnsafeQuotient,
                    phase,
                    "cell cycle references an out-of-range vertex",
                ));
            }
            live_vertices[from as usize] = true;
            incidence[from as usize].push(cell);
            let edge_use = EdgeUse { cell, from, to };
            match edges.entry(edge_key(from, to)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(EdgeRecord {
                        first: edge_use,
                        second: None,
                    });
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let record = entry.get_mut();
                    if record.second.is_some() {
                        return Err(work.failure(
                            FailureKind::UnsafeQuotient,
                            phase,
                            "live edge has more than two cell uses",
                        ));
                    }
                    record.second = Some(edge_use);
                }
            }
        }
    }

    for record in edges.values() {
        let Some(second) = record.second else {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "live edge has only one cell use",
            ));
        };
        if record.first.from != second.to || record.first.to != second.from {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "paired edge uses have the same orientation",
            ));
        }
    }

    Ok(Snapshot {
        edges,
        incidence,
        live_vertices,
    })
}

fn preflight_stored_positions(
    vertices: &[Vec3],
    cycles: &[Option<Vec<u32>>],
    work: &mut WorkTracker,
) -> Result<(), Failure> {
    for (cell, cycle) in cycles.iter().enumerate() {
        let Some(cycle) = cycle else { continue };
        let mut first = None;
        let mut second = None;
        let mut has_three = false;
        let mut has_zero_edge = false;
        for (offset, &vertex) in cycle.iter().enumerate() {
            work.charge_cell_indices(1, Phase::Preparation)?;
            let point = vertices[vertex as usize];
            match (first, second) {
                (None, _) => first = Some(point),
                (Some(a), None) if !same_stored_position(a, point) => second = Some(point),
                (Some(a), Some(b))
                    if !same_stored_position(a, point) && !same_stored_position(b, point) =>
                {
                    has_three = true;
                }
                _ => {}
            }
            let next = vertices[cycle[(offset + 1) % cycle.len()] as usize];
            has_zero_edge |= same_stored_position(point, next);
        }
        if !has_three && !has_zero_edge {
            let mut failure = work.failure(
                FailureKind::UnsupportedStoredDegeneracy,
                Phase::SourcePreflight,
                "cell has fewer than three exact stored positions without an exact-zero edge",
            );
            failure.affected_effective_cells.push(cell);
            return Err(failure);
        }
    }
    Ok(())
}

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

    fn union_min(&mut self, a: u32, b: u32) {
        let a = self.find(a);
        let b = self.find(b);
        if a == b {
            return;
        }
        let (keep, drop) = if a < b { (a, b) } else { (b, a) };
        self.parent[drop as usize] = keep;
    }
}

#[derive(Debug, Clone)]
struct Component {
    representative: u32,
    members: Vec<u32>,
    source_members: Vec<u32>,
    diameter: f64,
    displacement: f64,
}

fn candidate_keys(
    snapshot: &Snapshot,
    vertices: &[Vec3],
    threshold_squared: Option<f64>,
) -> Vec<EdgeKey> {
    let mut keys = Vec::new();
    for &(a, b) in snapshot.edges.keys() {
        let is_candidate = match threshold_squared {
            None => same_stored_position(vertices[a as usize], vertices[b as usize]),
            Some(threshold_squared) => {
                !same_stored_position(vertices[a as usize], vertices[b as usize])
                    && stored_distance_squared(vertices[a as usize], vertices[b as usize])
                        <= threshold_squared
            }
        };
        if is_candidate {
            keys.push((a, b));
        }
    }
    keys.sort_unstable();
    keys
}

fn build_components(
    candidates: &[EdgeKey],
    vertex_count: usize,
    ledger: &[Vec<u32>],
) -> Vec<Component> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut uf = DisjointSet::new(vertex_count);
    for &(a, b) in candidates {
        uf.union_min(a, b);
    }
    let mut members = BTreeMap::<u32, Vec<u32>>::new();
    for &(a, b) in candidates {
        let root = uf.find(a);
        members.entry(root).or_default().extend([a, b]);
    }

    let mut components = Vec::with_capacity(members.len());
    for (root, mut members) in members {
        members.sort_unstable();
        members.dedup();
        let representative = members[0];
        debug_assert_eq!(root, representative);
        let mut source_members = Vec::new();
        for &member in &members {
            source_members.extend_from_slice(&ledger[member as usize]);
        }
        source_members.sort_unstable();
        source_members.dedup();
        components.push(Component {
            representative,
            members,
            source_members,
            diameter: 0.0,
            displacement: 0.0,
        });
    }
    components
}

fn certify_component_diameter(
    component: &mut Component,
    vertices: &[Vec3],
    threshold_squared: f64,
    work: &mut WorkTracker,
) -> Result<bool, Failure> {
    let mut max_squared = 0.0f64;
    for (offset, &a) in component.source_members.iter().enumerate() {
        for &b in &component.source_members[(offset + 1)..] {
            work.charge_diameter_pair(Phase::Positive)?;
            let distance_squared =
                stored_distance_squared(vertices[a as usize], vertices[b as usize]);
            if distance_squared > threshold_squared {
                return Ok(false);
            }
            max_squared = max_squared.max(distance_squared);
        }
    }
    component.diameter = max_squared.sqrt();
    let representative = vertices[component.representative as usize];
    component.displacement = component
        .source_members
        .iter()
        .map(|&member| stored_distance_squared(vertices[member as usize], representative).sqrt())
        .fold(0.0, f64::max);
    Ok(true)
}

fn interaction_groups(
    components: &[Component],
    cycles: &[Option<Vec<u32>>],
    vertex_count: usize,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<Vec<Vec<usize>>, Failure> {
    if components.is_empty() {
        return Ok(Vec::new());
    }
    let mut component_for_vertex = vec![usize::MAX; vertex_count];
    for (component, data) in components.iter().enumerate() {
        for &member in &data.members {
            component_for_vertex[member as usize] = component;
        }
    }
    let mut uf = DisjointSet::new(components.len());
    let mut marked = vec![false; components.len()];
    let mut touched = Vec::new();
    for cycle in cycles.iter().flatten() {
        touched.clear();
        for &vertex in cycle {
            work.charge_cell_indices(1, phase)?;
            let component = component_for_vertex[vertex as usize];
            if component != usize::MAX && !marked[component] {
                marked[component] = true;
                touched.push(component);
            }
        }
        if let Some((&first, rest)) = touched.split_first() {
            for &other in rest {
                uf.union_min(first as u32, other as u32);
            }
        }
        for &component in &touched {
            marked[component] = false;
        }
    }
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for component in 0..components.len() {
        grouped
            .entry(uf.find(component as u32))
            .or_default()
            .push(component);
    }
    let mut groups: Vec<Vec<usize>> = grouped.into_values().collect();
    groups.sort_by_key(|group| {
        group
            .iter()
            .map(|&component| components[component].representative)
            .min()
            .unwrap()
    });
    Ok(groups)
}

#[derive(Debug)]
enum RewriteFailure {
    Work(Failure),
    CellKilling(Vec<usize>),
    NonSimple,
}

fn rewrite_all_cycles(
    cycles: &[Option<Vec<u32>>],
    replacements: &[u32],
    allow_cell_elision: bool,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<Vec<Option<Vec<u32>>>, RewriteFailure> {
    let mut rewritten_cycles = Vec::with_capacity(cycles.len());
    let mut killed = Vec::new();
    let mut seen = vec![false; replacements.len()];
    let mut touched = Vec::new();
    for (cell, cycle) in cycles.iter().enumerate() {
        let Some(cycle) = cycle else {
            rewritten_cycles.push(None);
            continue;
        };
        let mut rewritten = Vec::with_capacity(cycle.len());
        for &vertex in cycle {
            work.charge_cell_indices(1, phase)
                .map_err(RewriteFailure::Work)?;
            let mapped = replacements[vertex as usize];
            if rewritten.last().copied() != Some(mapped) {
                rewritten.push(mapped);
            }
        }
        if rewritten.len() > 1 && rewritten[0] == *rewritten.last().unwrap() {
            rewritten.pop();
        }
        if rewritten.len() < 3 {
            killed.push(cell);
            rewritten_cycles.push(None);
            continue;
        }
        touched.clear();
        for &vertex in &rewritten {
            if seen[vertex as usize] {
                for &touched_vertex in &touched {
                    seen[touched_vertex as usize] = false;
                }
                return Err(RewriteFailure::NonSimple);
            }
            seen[vertex as usize] = true;
            touched.push(vertex);
        }
        for &vertex in &touched {
            seen[vertex as usize] = false;
        }
        rewritten_cycles.push(Some(rewritten));
    }
    if !allow_cell_elision && !killed.is_empty() {
        return Err(RewriteFailure::CellKilling(killed));
    }
    Ok(rewritten_cycles)
}

#[derive(Debug)]
struct AffectedCover {
    cells: Vec<bool>,
    vertices: Vec<bool>,
    edges: FxHashSet<EdgeKey>,
}

fn transaction_affected_cover(
    before: &[Option<Vec<u32>>],
    after: &[Option<Vec<u32>>],
    vertex_count: usize,
) -> AffectedCover {
    debug_assert_eq!(before.len(), after.len());
    let mut cells = vec![false; before.len()];
    let mut vertices = vec![false; vertex_count];
    let mut edges = FxHashSet::default();
    for cell in 0..before.len() {
        if before[cell] == after[cell] {
            continue;
        }
        cells[cell] = true;
        for cycle in [before[cell].as_deref(), after[cell].as_deref()]
            .into_iter()
            .flatten()
        {
            for (offset, &vertex) in cycle.iter().enumerate() {
                vertices[vertex as usize] = true;
                let next = cycle[(offset + 1) % cycle.len()];
                edges.insert(edge_key(vertex, next));
            }
        }
    }
    AffectedCover {
        cells,
        vertices,
        edges,
    }
}

fn changed_ledger_vertices(before: &[Vec<u32>], after: &[Vec<u32>]) -> Vec<bool> {
    debug_assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after)
        .map(|(before, after)| before != after)
        .collect()
}

fn is_work_failure(kind: FailureKind) -> bool {
    matches!(
        kind,
        FailureKind::DiameterLimit
            | FailureKind::CellIndexLimit
            | FailureKind::ProvenanceLimit
            | FailureKind::CounterOverflow
    )
}

fn face_has_three_stored_positions(vertices: &[Vec3], cycle: &[u32]) -> bool {
    let mut first = None;
    let mut second = None;
    for &vertex in cycle {
        let point = vertices[vertex as usize];
        match (first, second) {
            (None, _) => first = Some(point),
            (Some(a), None) if !same_stored_position(a, point) => second = Some(point),
            (Some(a), Some(b))
                if !same_stored_position(a, point) && !same_stored_position(b, point) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn face_has_exact_edge(vertices: &[Vec3], cycle: &[u32]) -> bool {
    cycle.iter().enumerate().any(|(offset, &vertex)| {
        let next = cycle[(offset + 1) % cycle.len()];
        same_stored_position(vertices[vertex as usize], vertices[next as usize])
    })
}

struct TopologyCertification<'a> {
    require_no_exact_edges: bool,
    affected_cells: Option<&'a [bool]>,
    pending: Option<&'a [Option<SuppressionCause>]>,
}

fn certify_topology(
    vertices: &[Vec3],
    cycles: &[Option<Vec<u32>>],
    snapshot: &Snapshot,
    certification: TopologyCertification<'_>,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<(), Failure> {
    let TopologyCertification {
        require_no_exact_edges,
        affected_cells,
        pending,
    } = certification;
    let mut live_cells = Vec::new();
    let mut signatures = FxHashSet::<Vec<u32>>::default();
    let mut link_edges = vec![Vec::<(u32, u32)>::new(); vertices.len()];

    for (cell, cycle) in cycles.iter().enumerate() {
        let Some(cycle) = cycle else { continue };
        live_cells.push(cell);
        if cycle.len() < 3 {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "live face has fewer than three vertices",
            ));
        }
        for (offset, &vertex) in cycle.iter().enumerate() {
            work.charge_cell_indices(1, phase)?;
            let previous = cycle[(offset + cycle.len() - 1) % cycle.len()];
            let next = cycle[(offset + 1) % cycle.len()];
            if previous == next {
                return Err(work.failure(
                    FailureKind::UnsafeQuotient,
                    phase,
                    "quotient produced a collapsed vertex link edge",
                ));
            }
            link_edges[vertex as usize].push((previous, next));
        }
        let affected = affected_cells
            .and_then(|affected| affected.get(cell))
            .copied()
            .unwrap_or(false);
        if !face_has_three_stored_positions(vertices, cycle)
            && (require_no_exact_edges || affected || !face_has_exact_edge(vertices, cycle))
        {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "live face has fewer than three exact stored positions",
            ));
        }
        let mut signature = cycle.clone();
        signature.sort_unstable();
        if !signatures.insert(signature) {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "quotient produced duplicate live faces",
            ));
        }
    }

    if live_cells.is_empty() {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "quotient removed every cell",
        ));
    }
    for (vertex, owners) in snapshot.incidence.iter().enumerate() {
        if !snapshot.live_vertices[vertex] {
            continue;
        }
        let allowed_pending = owners.len() == 2
            && pending
                .and_then(|pending| pending.get(vertex))
                .copied()
                .flatten()
                .is_some();
        if owners.len() < 3 && !allowed_pending {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "quotient produced a vertex with incidence below three",
            ));
        }
    }

    for (vertex, links) in link_edges.iter().enumerate() {
        if !snapshot.live_vertices[vertex] {
            continue;
        }
        let mut next_for = FxHashMap::<u32, u32>::default();
        let mut incoming = FxHashSet::<u32>::default();
        for &(from, to) in links {
            if next_for.insert(from, to).is_some() || !incoming.insert(to) {
                return Err(work.failure(
                    FailureKind::UnsafeQuotient,
                    phase,
                    "quotient produced a branched vertex link",
                ));
            }
        }
        if next_for.len() != incoming.len()
            || next_for.keys().any(|neighbor| !incoming.contains(neighbor))
        {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "quotient produced an open vertex link",
            ));
        }
        let start = links[0].0;
        let mut current = start;
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(current) {
                if current != start {
                    return Err(work.failure(
                        FailureKind::UnsafeQuotient,
                        phase,
                        "quotient produced a non-cyclic vertex link",
                    ));
                }
                break;
            }
            let Some(&next) = next_for.get(&current) else {
                return Err(work.failure(
                    FailureKind::UnsafeQuotient,
                    phase,
                    "quotient produced an incomplete vertex link",
                ));
            };
            current = next;
        }
        if visited.len() != next_for.len() {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "quotient pinched two vertex-link cycles together",
            ));
        }
    }

    let live_vertex_count = snapshot.live_vertices.iter().filter(|&&live| live).count();
    let euler = live_vertex_count as i64 - snapshot.edges.len() as i64 + live_cells.len() as i64;
    if euler != 2 {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            format!("quotient has Euler characteristic {euler}, expected 2"),
        ));
    }

    let mut adjacency = vec![Vec::<usize>::new(); cycles.len()];
    for (&(a, b), record) in &snapshot.edges {
        if exactly_antipodal(vertices[a as usize], vertices[b as usize]) {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "quotient produced an exactly antipodal edge",
            ));
        }
        let affected_edge = affected_cells.is_some_and(|affected| {
            affected.get(record.first.cell).copied().unwrap_or(false)
                || record
                    .second
                    .and_then(|second| affected.get(second.cell))
                    .copied()
                    .unwrap_or(false)
        });
        if (require_no_exact_edges || affected_edge)
            && same_stored_position(vertices[a as usize], vertices[b as usize])
        {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "transaction retained an induced exact-zero edge",
            ));
        }
        let second = record.second.unwrap();
        adjacency[record.first.cell].push(second.cell);
        adjacency[second.cell].push(record.first.cell);
    }
    let start = live_cells[0];
    let mut stack = vec![start];
    let mut seen = vec![false; cycles.len()];
    seen[start] = true;
    while let Some(cell) = stack.pop() {
        for &neighbor in &adjacency[cell] {
            if !seen[neighbor] {
                seen[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    if live_cells.iter().any(|&cell| !seen[cell]) {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "quotient disconnected live face adjacency",
        ));
    }
    Ok(())
}

fn merge_component_ledgers(ledger: &mut [Vec<u32>], components: &[Component], group: &[usize]) {
    for &component_index in group {
        let component = &components[component_index];
        let representative = component.representative as usize;
        ledger[representative] = component.source_members.clone();
        for &member in &component.members {
            if member as usize != representative {
                ledger[member as usize].clear();
            }
        }
    }
}

fn replacements_for_group(
    vertex_count: usize,
    components: &[Component],
    group: &[usize],
) -> Vec<u32> {
    let mut replacements: Vec<u32> = (0..vertex_count as u32).collect();
    for &component_index in group {
        let component = &components[component_index];
        for &member in &component.members {
            replacements[member as usize] = component.representative;
        }
    }
    replacements
}

fn selected_candidate_keys(
    candidates: &[EdgeKey],
    components: &[Component],
    group: &[usize],
) -> FxHashSet<EdgeKey> {
    let mut component_for_vertex = FxHashMap::<u32, usize>::default();
    for &component in group {
        for &member in &components[component].members {
            component_for_vertex.insert(member, component);
        }
    }
    candidates
        .iter()
        .copied()
        .filter(|(a, b)| {
            matches!(
                (component_for_vertex.get(a), component_for_vertex.get(b)),
                (Some(a_component), Some(b_component)) if a_component == b_component
            )
        })
        .collect()
}

#[derive(Debug)]
enum TransactionResult {
    Commit {
        cycles: Vec<Option<Vec<u32>>>,
        ledger: Vec<Vec<u32>>,
        retired_vertices: usize,
        exact_components_closed: usize,
    },
    DeclineCell,
    DeclineTopology,
}

fn apply_exact_closure(
    vertices: &[Vec3],
    mut cycles: Vec<Option<Vec<u32>>>,
    mut ledger: Vec<Vec<u32>>,
    deferred_exact: &FxHashSet<EdgeKey>,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<ExactClosureResult, RewriteFailure> {
    let mut retired = 0usize;
    let mut component_count = 0usize;
    loop {
        let snapshot =
            build_snapshot(&cycles, vertices.len(), work, phase).map_err(RewriteFailure::Work)?;
        let candidates: Vec<EdgeKey> = candidate_keys(&snapshot, vertices, None)
            .into_iter()
            .filter(|key| !deferred_exact.contains(key))
            .collect();
        if candidates.is_empty() {
            return Ok((cycles, ledger, retired, component_count));
        }
        let components = build_components(&candidates, vertices.len(), &ledger);
        component_count += components.len();
        let all: Vec<usize> = (0..components.len()).collect();
        let replacements = replacements_for_group(vertices.len(), &components, &all);
        cycles = rewrite_all_cycles(&cycles, &replacements, false, work, phase)?;
        merge_component_ledgers(&mut ledger, &components, &all);
        retired += components
            .iter()
            .map(|component| component.members.len() - 1)
            .sum::<usize>();
    }
}

struct TransactionRequest<'a> {
    vertices: &'a [Vec3],
    cycles: &'a [Option<Vec<u32>>],
    ledger: &'a [Vec<u32>],
    components: &'a [Component],
    group: &'a [usize],
    deferred_exact: &'a FxHashSet<EdgeKey>,
    positive: bool,
    policy: CellPolicy,
}

fn attempt_non_elide_transaction(
    request: TransactionRequest<'_>,
    work: &mut WorkTracker,
) -> Result<TransactionResult, Failure> {
    let phase = if request.positive {
        Phase::Positive
    } else {
        Phase::Exact
    };
    let replacements =
        replacements_for_group(request.vertices.len(), request.components, request.group);
    let mut ledger = request.ledger.to_vec();
    let cycles = match rewrite_all_cycles(request.cycles, &replacements, false, work, phase) {
        Ok(cycles) => cycles,
        Err(RewriteFailure::Work(failure)) => return Err(failure),
        Err(RewriteFailure::CellKilling(cells)) => {
            if request.positive {
                if request.policy == CellPolicy::Error {
                    let mut failure = work.failure(
                        FailureKind::CellEliminationRequired,
                        phase,
                        "positive contraction would eliminate a cell",
                    );
                    failure.affected_effective_cells = cells;
                    return Err(failure);
                }
                return Ok(TransactionResult::DeclineCell);
            }
            let mut failure = work.failure(
                FailureKind::UnresolvedExactGroup,
                phase,
                "exact-zero contraction would eliminate a cell",
            );
            failure.affected_effective_cells = cells;
            return Err(failure);
        }
        Err(RewriteFailure::NonSimple) => {
            if request.positive {
                return Ok(TransactionResult::DeclineTopology);
            }
            return Err(work.failure(
                FailureKind::UnresolvedExactGroup,
                phase,
                "exact-zero contraction would produce a non-simple face",
            ));
        }
    };
    merge_component_ledgers(&mut ledger, request.components, request.group);
    let initially_retired = request
        .group
        .iter()
        .map(|&component| request.components[component].members.len() - 1)
        .sum::<usize>();
    let (cycles, ledger, closure_retired, exact_components_closed) = match apply_exact_closure(
        request.vertices,
        cycles,
        ledger,
        request.deferred_exact,
        work,
        phase,
    ) {
        Ok(result) => result,
        Err(RewriteFailure::Work(failure)) => return Err(failure),
        Err(RewriteFailure::CellKilling(cells)) if request.positive => {
            if request.policy == CellPolicy::Error {
                let mut failure = work.failure(
                    FailureKind::CellEliminationRequired,
                    phase,
                    "positive contraction's induced exact closure would eliminate a cell",
                );
                failure.affected_effective_cells = cells;
                return Err(failure);
            }
            return Ok(TransactionResult::DeclineCell);
        }
        Err(RewriteFailure::NonSimple) if request.positive => {
            return Ok(TransactionResult::DeclineTopology);
        }
        Err(RewriteFailure::CellKilling(cells)) => {
            let mut failure = work.failure(
                FailureKind::UnresolvedExactGroup,
                phase,
                "induced exact-zero closure would eliminate a cell",
            );
            failure.affected_effective_cells = cells;
            return Err(failure);
        }
        Err(RewriteFailure::NonSimple) => {
            return Err(work.failure(
                FailureKind::UnresolvedExactGroup,
                phase,
                "induced exact-zero closure would produce a non-simple face",
            ));
        }
    };
    let affected_cover =
        transaction_affected_cover(request.cycles, &cycles, request.vertices.len());
    let snapshot = build_snapshot(&cycles, request.vertices.len(), work, phase)?;
    match certify_topology(
        request.vertices,
        &cycles,
        &snapshot,
        TopologyCertification {
            require_no_exact_edges: request.positive,
            affected_cells: Some(&affected_cover.cells),
            pending: None,
        },
        work,
        phase,
    ) {
        Ok(()) => {}
        Err(failure) if request.positive && !is_work_failure(failure.kind) => {
            return Ok(TransactionResult::DeclineTopology);
        }
        Err(failure) if !request.positive && !is_work_failure(failure.kind) => {
            return Err(Failure {
                kind: FailureKind::UnresolvedExactGroup,
                ..failure
            });
        }
        Err(failure) => return Err(failure),
    }
    Ok(TransactionResult::Commit {
        cycles,
        ledger,
        retired_vertices: initially_retired + closure_retired,
        exact_components_closed,
    })
}

const ENDPOINT_CROSS_SQ_FLOOR: f64 = 1.0e-24;
const PROJECTION_SQ_FLOOR: f64 = 1.0e-24;
const ARC_MEMBERSHIP_TAU: f64 = 64.0 * f64::EPSILON;

#[derive(Debug, Clone, Copy)]
enum BagNode {
    Leaf(u32),
    Meld(usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BagRoot {
    node: usize,
    positive: bool,
}

#[derive(Debug)]
struct ProvenanceState {
    nodes: Vec<BagNode>,
    edge_roots: FxHashMap<EdgeKey, BagRoot>,
    sinks: Vec<Option<BagRoot>>,
    expected: Vec<bool>,
    exact_cross_track: Vec<Option<f64>>,
}

#[derive(Debug, Clone)]
struct ProvenanceOwners {
    edge_roots: FxHashMap<EdgeKey, BagRoot>,
    sinks: Vec<Option<BagRoot>>,
}

#[derive(Debug)]
struct AffectedProvenanceOwners {
    edges: Vec<EdgeKey>,
    sinks: Vec<u32>,
}

struct BagMembers<'a> {
    nodes: &'a [BagNode],
    stack: Vec<usize>,
}

impl Iterator for BagMembers<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match self.nodes[node] {
                BagNode::Leaf(source) => return Some(source),
                BagNode::Meld(a, b) => {
                    self.stack.push(b);
                    self.stack.push(a);
                }
            }
        }
        None
    }
}

impl ProvenanceState {
    fn new(vertex_count: usize) -> Self {
        Self {
            nodes: Vec::new(),
            edge_roots: FxHashMap::default(),
            sinks: vec![None; vertex_count],
            expected: vec![false; vertex_count],
            exact_cross_track: vec![None; vertex_count],
        }
    }

    fn leaf(&mut self, source: u32, positive: bool) -> BagRoot {
        let node = self.nodes.len();
        self.nodes.push(BagNode::Leaf(source));
        BagRoot { node, positive }
    }

    fn meld(&mut self, a: BagRoot, b: BagRoot, force_positive: bool) -> BagRoot {
        let node = self.nodes.len();
        self.nodes.push(BagNode::Meld(a.node, b.node));
        BagRoot {
            node,
            positive: force_positive || a.positive || b.positive,
        }
    }

    fn insert_root(
        &mut self,
        destination: &mut Option<BagRoot>,
        root: BagRoot,
        force_positive: bool,
    ) {
        let mut root = BagRoot {
            positive: root.positive || force_positive,
            ..root
        };
        if let Some(existing) = destination.take() {
            root = self.meld(existing, root, force_positive);
        }
        *destination = Some(root);
    }

    fn owner_snapshot(&self) -> ProvenanceOwners {
        ProvenanceOwners {
            edge_roots: self.edge_roots.clone(),
            sinks: self.sinks.clone(),
        }
    }

    fn members(&self, root: BagRoot) -> BagMembers<'_> {
        BagMembers {
            nodes: &self.nodes,
            stack: vec![root.node],
        }
    }
}

fn affected_provenance_owners(
    before: &ProvenanceOwners,
    after: &ProvenanceState,
    cover: &AffectedCover,
) -> AffectedProvenanceOwners {
    let mut edges: Vec<EdgeKey> = after
        .edge_roots
        .iter()
        .filter_map(|(&key, &root)| {
            (before.edge_roots.get(&key).copied() != Some(root) || cover.edges.contains(&key))
                .then_some(key)
        })
        .collect();
    edges.sort_unstable();
    let sinks = after
        .sinks
        .iter()
        .enumerate()
        .filter_map(|(representative, &root)| {
            let root = root?;
            (before.sinks.get(representative).copied().flatten() != Some(root)
                || cover.vertices.get(representative).copied().unwrap_or(false))
            .then_some(representative as u32)
        })
        .collect();
    AffectedProvenanceOwners { edges, sinks }
}

fn normalized(point: Vec3) -> Option<glam::DVec3> {
    let point = point.as_dvec3();
    let len_sq = point.length_squared();
    (len_sq.is_finite() && len_sq > 0.0).then(|| point / len_sq.sqrt())
}

fn angular_distance(a: glam::DVec3, b: glam::DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b).clamp(-1.0, 1.0))
}

fn positive_angle(normal: glam::DVec3, a: glam::DVec3, b: glam::DVec3) -> f64 {
    let angle = normal.dot(a.cross(b)).atan2(a.dot(b).clamp(-1.0, 1.0));
    if angle == 0.0 {
        0.0
    } else if angle < 0.0 {
        angle + core::f64::consts::TAU
    } else {
        angle
    }
}

fn point_to_minor_arc_chord(point: Vec3, start: Vec3, end: Vec3) -> Option<f64> {
    if same_stored_position(start, end) || exactly_antipodal(start, end) {
        return None;
    }
    let point = normalized(point)?;
    let start = normalized(start)?;
    let end = normalized(end)?;
    let normal_raw = start.cross(end);
    let cross_sq = normal_raw.length_squared();
    if !cross_sq.is_finite() || cross_sq <= ENDPOINT_CROSS_SQ_FLOOR {
        return None;
    }
    let cross_len = cross_sq.sqrt();
    let normal = normal_raw / cross_len;
    let theta = cross_len.atan2(start.dot(end).clamp(-1.0, 1.0));
    let projected_raw = point - normal * normal.dot(point);
    let projection_sq = projected_raw.length_squared();
    let mut selected = None::<(glam::DVec3, f64)>;
    if projection_sq.is_finite() && projection_sq > PROJECTION_SQ_FLOOR {
        let projected = projected_raw / projection_sq.sqrt();
        for candidate in [projected, -projected] {
            let from_start = positive_angle(normal, start, candidate);
            let to_end = positive_angle(normal, candidate, end);
            let on_arc = from_start <= theta + ARC_MEMBERSHIP_TAU
                && to_end <= theta + ARC_MEMBERSHIP_TAU
                && ((from_start + to_end) - theta).abs() <= ARC_MEMBERSHIP_TAU;
            if on_arc
                && selected
                    .as_ref()
                    .is_none_or(|(_, previous)| from_start < *previous)
            {
                selected = Some((candidate, from_start));
            }
        }
    }
    let nearest = selected.map(|(candidate, _)| candidate).unwrap_or_else(|| {
        let to_start = angular_distance(point, start);
        let to_end = angular_distance(point, end);
        if to_start <= to_end {
            start
        } else {
            end
        }
    });
    let delta = angular_distance(point, nearest);
    Some(2.0 * (delta * 0.5).sin())
}

fn point_to_representative_chord(point: Vec3, representative: Vec3) -> Option<f64> {
    let point = normalized(point)?;
    let representative = normalized(representative)?;
    let delta = angular_distance(point, representative);
    Some(2.0 * (delta * 0.5).sin())
}

fn exact_cross_track_radians(point: Vec3, start: Vec3, end: Vec3) -> Option<f64> {
    let point = normalized(point)?;
    let start = normalized(start)?;
    let end = normalized(end)?;
    let normal = start.cross(end);
    let len = normal.length();
    (len.is_finite() && len > 0.0).then(|| point.dot(normal / len).abs().min(1.0).asin())
}

fn owner_for_source(ledger: &[Vec<u32>], vertex_count: usize) -> Vec<Option<u32>> {
    let mut owner = vec![None; vertex_count];
    for (representative, members) in ledger.iter().enumerate() {
        for &member in members {
            owner[member as usize] = Some(representative as u32);
        }
    }
    owner
}

fn put_edge_root(
    provenance: &mut ProvenanceState,
    key: EdgeKey,
    root: BagRoot,
    force_positive: bool,
) {
    let mut destination = provenance.edge_roots.remove(&key);
    provenance.insert_root(&mut destination, root, force_positive);
    if let Some(root) = destination {
        provenance.edge_roots.insert(key, root);
    }
}

fn put_sink_root(
    provenance: &mut ProvenanceState,
    representative: u32,
    root: BagRoot,
    force_positive: bool,
) {
    let mut destination = provenance.sinks[representative as usize].take();
    provenance.insert_root(&mut destination, root, force_positive);
    provenance.sinks[representative as usize] = destination;
}

fn reconcile_provenance(
    provenance: &mut ProvenanceState,
    ledger: &[Vec<u32>],
    live_edges: &FxHashMap<EdgeKey, EdgeRecord>,
    live_vertices: &[bool],
    positive_origin: bool,
    changed_ledger_vertices: &[bool],
) -> Result<(), &'static str> {
    let owners = owner_for_source(ledger, provenance.sinks.len());
    let old_edges = std::mem::take(&mut provenance.edge_roots);
    for ((old_a, old_b), root) in old_edges {
        let positive_endpoint_change = positive_origin
            && (changed_ledger_vertices
                .get(old_a as usize)
                .copied()
                .unwrap_or(false)
                || changed_ledger_vertices
                    .get(old_b as usize)
                    .copied()
                    .unwrap_or(false));
        let a = owners[old_a as usize]
            .filter(|&owner| live_vertices.get(owner as usize).copied() == Some(true));
        let b = owners[old_b as usize]
            .filter(|&owner| live_vertices.get(owner as usize).copied() == Some(true));
        let (a, b) = match (a, b) {
            (None, None) => return Err("carrying edge lost both live endpoint owners"),
            (Some(owner), None) | (None, Some(owner)) => {
                put_sink_root(
                    provenance,
                    owner,
                    root,
                    positive_origin || positive_endpoint_change,
                );
                continue;
            }
            (Some(a), Some(b)) => (a, b),
        };
        if a == b {
            put_sink_root(
                provenance,
                a,
                root,
                positive_origin || positive_endpoint_change,
            );
            continue;
        }
        let key = edge_key(a, b);
        let changed = key != edge_key(old_a, old_b);
        if live_edges.contains_key(&key) {
            put_edge_root(
                provenance,
                key,
                root,
                positive_endpoint_change || (positive_origin && changed),
            );
        } else {
            put_sink_root(provenance, a.min(b), root, positive_origin);
        }
    }
    let old_sinks = std::mem::take(&mut provenance.sinks);
    provenance.sinks = vec![None; old_sinks.len()];
    for (old, root) in old_sinks.into_iter().enumerate() {
        let Some(root) = root else { continue };
        let representative = owners[old].ok_or("vertex sink lost its representative owner")?;
        if live_vertices.get(representative as usize).copied() != Some(true) {
            return Err("vertex sink lost its live representative owner");
        }
        put_sink_root(
            provenance,
            representative,
            root,
            positive_origin
                && (representative as usize != old
                    || changed_ledger_vertices.get(old).copied().unwrap_or(false)),
        );
    }
    Ok(())
}

fn certify_positive_provenance(
    provenance: &ProvenanceState,
    vertices: &[Vec3],
    threshold: f64,
    affected: Option<&AffectedProvenanceOwners>,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<f64, Failure> {
    let mut maximum = 0.0f64;
    let edge_keys: Vec<EdgeKey> = match affected {
        Some(affected) => affected.edges.clone(),
        None => {
            let mut keys: Vec<_> = provenance.edge_roots.keys().copied().collect();
            keys.sort_unstable();
            keys
        }
    };
    for (a, b) in edge_keys {
        let Some(&root) = provenance.edge_roots.get(&(a, b)) else {
            continue;
        };
        if !root.positive {
            continue;
        }
        for member in provenance.members(root) {
            work.charge_provenance_member(phase)?;
            let Some(chord) = point_to_minor_arc_chord(
                vertices[member as usize],
                vertices[a as usize],
                vertices[b as usize],
            ) else {
                return Err(work.failure(
                    FailureKind::IllConditionedReplacementArc,
                    phase,
                    "positive suppression owner edge is ill-conditioned",
                ));
            };
            if chord > threshold {
                return Err(work.failure(
                    FailureKind::PositiveSuppressionDeviation,
                    phase,
                    "positive suppression member exceeds its owner-arc threshold",
                ));
            }
            maximum = maximum.max(chord);
        }
    }
    let sink_ids: Vec<u32> = match affected {
        Some(affected) => affected.sinks.clone(),
        None => provenance
            .sinks
            .iter()
            .enumerate()
            .filter_map(|(representative, root)| root.is_some().then_some(representative as u32))
            .collect(),
    };
    for representative in sink_ids {
        let Some(root) = provenance.sinks[representative as usize] else {
            continue;
        };
        if !root.positive {
            continue;
        }
        for member in provenance.members(root) {
            work.charge_provenance_member(phase)?;
            let Some(chord) = point_to_representative_chord(
                vertices[member as usize],
                vertices[representative as usize],
            ) else {
                return Err(work.failure(
                    FailureKind::IllConditionedReplacementArc,
                    phase,
                    "positive suppression sink is ill-conditioned",
                ));
            };
            if chord > threshold {
                return Err(work.failure(
                    FailureKind::PositiveSuppressionDeviation,
                    phase,
                    "positive suppression member exceeds its representative threshold",
                ));
            }
            maximum = maximum.max(chord);
        }
    }
    Ok(maximum)
}

#[derive(Debug)]
struct ElideState {
    cycles: Vec<Option<Vec<u32>>>,
    ledger: Vec<Vec<u32>>,
    pending: Vec<Option<SuppressionCause>>,
    provenance: ProvenanceState,
}

fn refresh_pending(
    pending: &mut [Option<SuppressionCause>],
    snapshot: &Snapshot,
    affected_vertices: &[bool],
    transaction_cause: SuppressionCause,
) {
    for (vertex, pending_cause) in pending.iter_mut().enumerate() {
        if !affected_vertices.get(vertex).copied().unwrap_or(false) {
            continue;
        }
        if !snapshot.live_vertices[vertex] || snapshot.incidence[vertex].len() != 2 {
            *pending_cause = None;
            continue;
        }
        *pending_cause = Some(match (*pending_cause, transaction_cause) {
            (Some(SuppressionCause::Positive), _) | (_, SuppressionCause::Positive) => {
                SuppressionCause::Positive
            }
            _ => SuppressionCause::Exact,
        });
    }
}

fn apply_exact_closure_elide(
    vertices: &[Vec3],
    cycles: &mut Vec<Option<Vec<u32>>>,
    ledger: &mut [Vec<u32>],
    deferred_exact: &FxHashSet<EdgeKey>,
    work: &mut WorkTracker,
    phase: Phase,
) -> Result<(usize, usize), RewriteFailure> {
    let mut retired = 0usize;
    let mut component_count = 0usize;
    loop {
        let snapshot =
            build_snapshot(cycles, vertices.len(), work, phase).map_err(RewriteFailure::Work)?;
        let candidates: Vec<EdgeKey> = candidate_keys(&snapshot, vertices, None)
            .into_iter()
            .filter(|key| !deferred_exact.contains(key))
            .collect();
        if candidates.is_empty() {
            return Ok((retired, component_count));
        }
        let components = build_components(&candidates, vertices.len(), ledger);
        component_count += components.len();
        let all: Vec<usize> = (0..components.len()).collect();
        let replacements = replacements_for_group(vertices.len(), &components, &all);
        *cycles = rewrite_all_cycles(cycles, &replacements, true, work, phase)?;
        merge_component_ledgers(ledger, &components, &all);
        retired += components
            .iter()
            .map(|component| component.members.len() - 1)
            .sum::<usize>();
    }
}

struct ElideTransactionRequest<'a> {
    vertices: &'a [Vec3],
    components: &'a [Component],
    group: &'a [usize],
    deferred_exact: &'a FxHashSet<EdgeKey>,
    positive: bool,
    threshold: f64,
}

fn attempt_elide_transaction(
    request: ElideTransactionRequest<'_>,
    state: &mut ElideState,
    work: &mut WorkTracker,
) -> Result<TransactionResultElide, Failure> {
    let phase = if request.positive {
        Phase::Positive
    } else {
        Phase::Exact
    };
    // Topology and component membership are ordinary transaction scratch.
    // Provenance bags remain in the persistent arena: the only non-committing
    // outcome below is detected before provenance mutation, so bag nodes and
    // member leaves are never copied merely to support rollback.
    let mut scratch_cycles = state.cycles.clone();
    let mut scratch_ledger = state.ledger.clone();
    let mut scratch_pending = state.pending.clone();
    let provenance_before = state.provenance.owner_snapshot();
    let replacements =
        replacements_for_group(request.vertices.len(), request.components, request.group);
    scratch_cycles = match rewrite_all_cycles(&scratch_cycles, &replacements, true, work, phase) {
        Ok(cycles) => cycles,
        Err(RewriteFailure::Work(failure)) => return Err(failure),
        Err(RewriteFailure::NonSimple) => {
            if request.positive {
                return Ok(TransactionResultElide::DeclineTopology);
            }
            return Err(work.failure(
                FailureKind::UnresolvedExactGroup,
                phase,
                "exact Elide quotient would produce a non-simple face",
            ));
        }
        Err(RewriteFailure::CellKilling(_)) => unreachable!("Elide permits cell removal"),
    };
    merge_component_ledgers(&mut scratch_ledger, request.components, request.group);
    let initial_retired = request
        .group
        .iter()
        .map(|&component| request.components[component].members.len() - 1)
        .sum::<usize>();
    let (closure_retired, exact_components_closed) = match apply_exact_closure_elide(
        request.vertices,
        &mut scratch_cycles,
        &mut scratch_ledger,
        request.deferred_exact,
        work,
        phase,
    ) {
        Ok(retired) => retired,
        Err(RewriteFailure::Work(failure)) => return Err(failure),
        Err(RewriteFailure::NonSimple) => {
            return Err(work.failure(
                FailureKind::UnsafeQuotient,
                phase,
                "combined Elide exact closure produced a non-simple face",
            ));
        }
        Err(RewriteFailure::CellKilling(_)) => unreachable!("Elide permits cell removal"),
    };
    let affected_cover =
        transaction_affected_cover(&state.cycles, &scratch_cycles, request.vertices.len());
    let changed_ledger = changed_ledger_vertices(&state.ledger, &scratch_ledger);
    let snapshot = build_snapshot(&scratch_cycles, request.vertices.len(), work, phase)?;
    refresh_pending(
        &mut scratch_pending,
        &snapshot,
        &affected_cover.vertices,
        if request.positive {
            SuppressionCause::Positive
        } else {
            SuppressionCause::Exact
        },
    );
    reconcile_provenance(
        &mut state.provenance,
        &scratch_ledger,
        &snapshot.edges,
        &snapshot.live_vertices,
        request.positive,
        &changed_ledger,
    )
    .map_err(|message| work.failure(FailureKind::UnsafeQuotient, phase, message))?;
    let affected_provenance =
        affected_provenance_owners(&provenance_before, &state.provenance, &affected_cover);
    certify_topology(
        request.vertices,
        &scratch_cycles,
        &snapshot,
        TopologyCertification {
            require_no_exact_edges: request.positive,
            affected_cells: Some(&affected_cover.cells),
            pending: Some(&scratch_pending),
        },
        work,
        phase,
    )?;
    let max_positive_deviation = certify_positive_provenance(
        &state.provenance,
        request.vertices,
        request.threshold,
        Some(&affected_provenance),
        work,
        phase,
    )?;
    state.cycles = scratch_cycles;
    state.ledger = scratch_ledger;
    state.pending = scratch_pending;
    Ok(TransactionResultElide::Commit {
        retired_vertices: initial_retired + closure_retired,
        exact_components_closed,
        max_positive_deviation,
    })
}

#[derive(Debug)]
enum TransactionResultElide {
    Commit {
        retired_vertices: usize,
        exact_components_closed: usize,
        max_positive_deviation: f64,
    },
    DeclineTopology,
}

fn suppress_one_pending(
    vertices: &[Vec3],
    state: &mut ElideState,
    vertex: u32,
    threshold: f64,
    work: &mut WorkTracker,
) -> Result<(SuppressionCause, f64, f64, usize, usize), Failure> {
    let phase = Phase::Suppression;
    let cause = state.pending[vertex as usize]
        .ok_or_else(|| work.failure(FailureKind::UnsafeQuotient, phase, "vertex is not pending"))?;
    let snapshot = build_snapshot(&state.cycles, vertices.len(), work, phase)?;
    let owners = &snapshot.incidence[vertex as usize];
    if owners.len() != 2 {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "pending suppression vertex does not have two owners",
        ));
    }
    let mut rotations = [(0u32, 0u32, 0usize); 2];
    for (slot, &cell) in owners.iter().enumerate() {
        let cycle = state.cycles[cell].as_ref().unwrap();
        let offset = cycle
            .iter()
            .position(|&candidate| candidate == vertex)
            .ok_or_else(|| {
                work.failure(
                    FailureKind::UnsafeQuotient,
                    phase,
                    "pending suppression lost owner incidence",
                )
            })?;
        rotations[slot] = (
            cycle[(offset + cycle.len() - 1) % cycle.len()],
            cycle[(offset + 1) % cycle.len()],
            offset,
        );
    }
    if rotations[0].0 != rotations[1].1 || rotations[0].1 != rotations[1].0 {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "degree-two suppression owner rotations disagree",
        ));
    }
    let start = rotations[0].0;
    let end = rotations[0].1;
    if start == end {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "degree-two suppression has one repeated endpoint id",
        ));
    }
    let coincident_endpoints =
        same_stored_position(vertices[start as usize], vertices[end as usize]);
    if !coincident_endpoints && exactly_antipodal(vertices[start as usize], vertices[end as usize])
    {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "degree-two suppression would create an antipodal edge",
        ));
    }
    let exact_cross_track = if coincident_endpoints {
        // There is no replacement arc to measure: after suppression the new
        // exact-zero edge is immediately consumed by mandatory closure.
        0.0
    } else {
        exact_cross_track_radians(
            vertices[vertex as usize],
            vertices[start as usize],
            vertices[end as usize],
        )
        .ok_or_else(|| {
            work.failure(
                FailureKind::IllConditionedReplacementArc,
                phase,
                "degree-two suppression replacement arc is undefined",
            )
        })?
    };

    let mut scratch_cycles = state.cycles.clone();
    let mut scratch_ledger = state.ledger.clone();
    let mut scratch_pending = state.pending.clone();
    let provenance_before = state.provenance.owner_snapshot();
    let consumed_keys = [
        edge_key(vertex, start),
        edge_key(vertex, end),
        edge_key(start, end),
    ];
    let positive_origin = cause == SuppressionCause::Positive
        || consumed_keys.iter().any(|key| {
            state
                .provenance
                .edge_roots
                .get(key)
                .is_some_and(|root| root.positive)
        })
        || state.provenance.sinks[vertex as usize].is_some_and(|root| root.positive);
    let transaction_cause = if positive_origin {
        SuppressionCause::Positive
    } else {
        SuppressionCause::Exact
    };
    let mut root = state.provenance.leaf(vertex, positive_origin);
    for key in consumed_keys {
        if let Some(existing) = state.provenance.edge_roots.remove(&key) {
            root = state.provenance.meld(root, existing, positive_origin);
        }
    }
    if let Some(existing) = state.provenance.sinks[vertex as usize].take() {
        root = state.provenance.meld(root, existing, positive_origin);
    }
    state.provenance.expected[vertex as usize] = true;
    // Acceptance-time exact telemetry is intentionally retained even if an
    // already-tainted carrying root upgrades this member to positive for its
    // current/final geometric certificate.
    if cause == SuppressionCause::Exact {
        state.provenance.exact_cross_track[vertex as usize] = Some(exact_cross_track);
    }
    for (slot, &cell) in owners.iter().enumerate().rev() {
        let cycle = scratch_cycles[cell].as_mut().unwrap();
        cycle.remove(rotations[slot].2);
        if cycle.len() < 3 {
            scratch_cycles[cell] = None;
        }
    }
    scratch_ledger[vertex as usize].clear();
    scratch_pending[vertex as usize] = None;
    let replacement_snapshot = build_snapshot(&scratch_cycles, vertices.len(), work, phase)?;
    let replacement_key = edge_key(start, end);
    if replacement_snapshot.edges.contains_key(&replacement_key) {
        put_edge_root(
            &mut state.provenance,
            replacement_key,
            root,
            positive_origin,
        );
    } else {
        let sink = [start, end]
            .into_iter()
            .filter(|&candidate| replacement_snapshot.live_vertices[candidate as usize])
            .min()
            .ok_or_else(|| {
                work.failure(
                    FailureKind::UnsafeQuotient,
                    phase,
                    "suppression replacement lost both live endpoint owners",
                )
            })?;
        put_sink_root(&mut state.provenance, sink, root, positive_origin);
    }
    let deferred = FxHashSet::default();
    let (closure_retired, exact_components_closed) = apply_exact_closure_elide(
        vertices,
        &mut scratch_cycles,
        &mut scratch_ledger,
        &deferred,
        work,
        phase,
    )
    .map_err(|failure| match failure {
        RewriteFailure::Work(failure) => failure,
        RewriteFailure::CellKilling(_) | RewriteFailure::NonSimple => work.failure(
            FailureKind::UnsafeQuotient,
            phase,
            "suppression-induced exact closure failed",
        ),
    })?;
    let affected_cover = transaction_affected_cover(&state.cycles, &scratch_cycles, vertices.len());
    let changed_ledger = changed_ledger_vertices(&state.ledger, &scratch_ledger);
    let final_snapshot = build_snapshot(&scratch_cycles, vertices.len(), work, phase)?;
    refresh_pending(
        &mut scratch_pending,
        &final_snapshot,
        &affected_cover.vertices,
        transaction_cause,
    );
    reconcile_provenance(
        &mut state.provenance,
        &scratch_ledger,
        &final_snapshot.edges,
        &final_snapshot.live_vertices,
        positive_origin,
        &changed_ledger,
    )
    .map_err(|message| work.failure(FailureKind::UnsafeQuotient, phase, message))?;
    let affected_provenance =
        affected_provenance_owners(&provenance_before, &state.provenance, &affected_cover);
    certify_topology(
        vertices,
        &scratch_cycles,
        &final_snapshot,
        TopologyCertification {
            require_no_exact_edges: true,
            affected_cells: Some(&affected_cover.cells),
            pending: Some(&scratch_pending),
        },
        work,
        phase,
    )?;
    let positive_deviation = certify_positive_provenance(
        &state.provenance,
        vertices,
        threshold,
        Some(&affected_provenance),
        work,
        phase,
    )?;
    state.cycles = scratch_cycles;
    state.ledger = scratch_ledger;
    state.pending = scratch_pending;
    Ok((
        transaction_cause,
        exact_cross_track,
        positive_deviation,
        closure_retired,
        exact_components_closed,
    ))
}

fn audit_provenance(provenance: &ProvenanceState) -> Result<(usize, usize, f64), &'static str> {
    let mut node_state = vec![0u8; provenance.nodes.len()];
    let mut seen = vec![false; provenance.expected.len()];
    let mut exact_count = 0usize;
    let mut positive_count = 0usize;
    let mut max_exact_cross_track = 0.0f64;

    let roots = provenance
        .edge_roots
        .values()
        .copied()
        .chain(provenance.sinks.iter().flatten().copied());
    for root in roots {
        let mut stack = vec![(root.node, false)];
        while let Some((node, exiting)) = stack.pop() {
            if exiting {
                node_state[node] = 2;
                continue;
            }
            match node_state[node] {
                1 => return Err("suppression provenance bag contains a cycle"),
                2 => return Err("suppression provenance bag has shared ownership"),
                _ => {}
            }
            node_state[node] = 1;
            stack.push((node, true));
            match provenance.nodes[node] {
                BagNode::Leaf(source) => {
                    let source = source as usize;
                    if source >= seen.len() || !provenance.expected[source] || seen[source] {
                        return Err("suppression provenance has an unexpected or duplicate member");
                    }
                    seen[source] = true;
                    if let Some(cross_track) = provenance.exact_cross_track[source] {
                        exact_count += 1;
                        max_exact_cross_track = max_exact_cross_track.max(cross_track);
                    }
                    if root.positive {
                        positive_count += 1;
                    }
                }
                BagNode::Meld(a, b) => {
                    stack.push((b, false));
                    stack.push((a, false));
                }
            }
        }
    }
    if provenance
        .expected
        .iter()
        .zip(&seen)
        .any(|(&expected, &seen)| expected != seen)
    {
        return Err("suppression provenance is missing an expected member");
    }
    Ok((exact_count, positive_count, max_exact_cross_track))
}

fn simplify_elide(
    vertices: &[Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
    threshold: f64,
    limits: Limits,
) -> Result<Outcome, Failure> {
    let threshold_squared = threshold * threshold;
    let mut cycles = Vec::with_capacity(cells.len());
    for cell in cells {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        let Some(cycle) = cell_indices.get(start..end) else {
            return Err(Failure {
                kind: FailureKind::Validation,
                phase: Phase::Preparation,
                message: "source cell span exceeds the index buffer".into(),
                work: WorkStats::default(),
                affected_effective_cells: Vec::new(),
            });
        };
        cycles.push(Some(cycle.to_vec()));
    }
    let mut work = WorkTracker::new(limits);
    preflight_stored_positions(vertices, &cycles, &mut work)?;
    let ledger: Vec<Vec<u32>> = (0..vertices.len() as u32)
        .map(|vertex| vec![vertex])
        .collect();
    let mut state = ElideState {
        cycles,
        ledger,
        pending: vec![None; vertices.len()],
        provenance: ProvenanceState::new(vertices.len()),
    };
    let mut stats = ResultStats::default();

    'fixed_point: loop {
        stats.round_attempts += 1;
        let snapshot = build_snapshot(&state.cycles, vertices.len(), &mut work, Phase::Exact)?;
        let exact_candidates = candidate_keys(&snapshot, vertices, None);
        work.note_candidates(exact_candidates.len());
        add_occurrences(
            &mut stats.exact_candidate_occurrences,
            exact_candidates.len(),
            &work,
            Phase::Exact,
        )?;
        if stats.productive_rounds > 0 {
            add_occurrences(
                &mut stats.later_round_candidate_occurrences,
                exact_candidates.len(),
                &work,
                Phase::Exact,
            )?;
        }
        if !exact_candidates.is_empty() {
            let components = build_components(&exact_candidates, vertices.len(), &state.ledger);
            stats.max_component_members = stats.max_component_members.max(
                components
                    .iter()
                    .map(|component| component.source_members.len())
                    .max()
                    .unwrap_or(0),
            );
            let groups = interaction_groups(
                &components,
                &state.cycles,
                vertices.len(),
                &mut work,
                Phase::Exact,
            )?;
            let group = &groups[0];
            let selected = selected_candidate_keys(&exact_candidates, &components, group);
            let deferred: FxHashSet<EdgeKey> = exact_candidates
                .iter()
                .copied()
                .filter(|key| !selected.contains(key))
                .collect();
            match attempt_elide_transaction(
                ElideTransactionRequest {
                    vertices,
                    components: &components,
                    group,
                    deferred_exact: &deferred,
                    positive: false,
                    threshold,
                },
                &mut state,
                &mut work,
            )? {
                TransactionResultElide::Commit {
                    retired_vertices,
                    exact_components_closed,
                    max_positive_deviation,
                } => {
                    stats.productive_rounds += 1;
                    stats.committed_transactions += 1;
                    stats.exact_components_committed += group.len() + exact_components_closed;
                    stats.vertices_retired += retired_vertices;
                    stats.max_positive_suppression_unit_arc_chord = stats
                        .max_positive_suppression_unit_arc_chord
                        .max(max_positive_deviation);
                    continue 'fixed_point;
                }
                TransactionResultElide::DeclineTopology => {
                    unreachable!("mandatory exact Elide transaction cannot be declined")
                }
            }
        }

        let positive_candidates = candidate_keys(&snapshot, vertices, Some(threshold_squared));
        work.note_candidates(positive_candidates.len());
        add_occurrences(
            &mut stats.positive_candidate_occurrences,
            positive_candidates.len(),
            &work,
            Phase::Positive,
        )?;
        if stats.productive_rounds > 0 {
            add_occurrences(
                &mut stats.later_round_candidate_occurrences,
                positive_candidates.len(),
                &work,
                Phase::Positive,
            )?;
        }
        let mut committed_positive = false;
        if !positive_candidates.is_empty() {
            let mut eligible = Vec::new();
            for mut component in
                build_components(&positive_candidates, vertices.len(), &state.ledger)
            {
                stats.max_component_members = stats
                    .max_component_members
                    .max(component.source_members.len());
                if certify_component_diameter(
                    &mut component,
                    vertices,
                    threshold_squared,
                    &mut work,
                )? {
                    eligible.push(component);
                } else {
                    add_occurrences(
                        &mut stats.positive_components_declined_diameter,
                        1,
                        &work,
                        Phase::Positive,
                    )?;
                }
            }
            if !eligible.is_empty() {
                let groups = interaction_groups(
                    &eligible,
                    &state.cycles,
                    vertices.len(),
                    &mut work,
                    Phase::Positive,
                )?;
                let deferred = FxHashSet::default();
                for group in &groups {
                    match attempt_elide_transaction(
                        ElideTransactionRequest {
                            vertices,
                            components: &eligible,
                            group,
                            deferred_exact: &deferred,
                            positive: true,
                            threshold,
                        },
                        &mut state,
                        &mut work,
                    )? {
                        TransactionResultElide::Commit {
                            retired_vertices,
                            exact_components_closed,
                            max_positive_deviation,
                        } => {
                            for &component in group {
                                stats.max_component_diameter = stats
                                    .max_component_diameter
                                    .max(eligible[component].diameter);
                                stats.max_representative_displacement = stats
                                    .max_representative_displacement
                                    .max(eligible[component].displacement);
                            }
                            stats.productive_rounds += 1;
                            stats.committed_transactions += 1;
                            stats.positive_components_committed += group.len();
                            stats.exact_components_committed += exact_components_closed;
                            stats.vertices_retired += retired_vertices;
                            stats.max_positive_suppression_unit_arc_chord = stats
                                .max_positive_suppression_unit_arc_chord
                                .max(max_positive_deviation);
                            committed_positive = true;
                            break;
                        }
                        TransactionResultElide::DeclineTopology => {
                            add_occurrences(
                                &mut stats.positive_groups_declined_topology,
                                1,
                                &work,
                                Phase::Positive,
                            )?;
                        }
                    }
                }
            }
        }
        if committed_positive {
            continue 'fixed_point;
        }

        if let Some(vertex) = state
            .pending
            .iter()
            .enumerate()
            .find_map(|(vertex, cause)| cause.is_some().then_some(vertex as u32))
        {
            let (
                cause,
                exact_cross_track,
                positive_deviation,
                closure_retired,
                exact_components_closed,
            ) = suppress_one_pending(vertices, &mut state, vertex, threshold, &mut work)?;
            stats.productive_rounds += 1;
            stats.committed_transactions += 1;
            stats.vertices_retired += 1 + closure_retired;
            stats.exact_components_committed += exact_components_closed;
            if cause == SuppressionCause::Exact {
                stats.exact_suppression_members += 1;
                stats.max_exact_suppression_cross_track_radians = stats
                    .max_exact_suppression_cross_track_radians
                    .max(exact_cross_track);
            }
            stats.max_positive_suppression_unit_arc_chord = stats
                .max_positive_suppression_unit_arc_chord
                .max(positive_deviation);
            continue 'fixed_point;
        }
        break 'fixed_point;
    }

    let final_snapshot = build_snapshot(
        &state.cycles,
        vertices.len(),
        &mut work,
        Phase::FinalCertification,
    )?;
    stats.final_exact_edges = candidate_keys(&final_snapshot, vertices, None).len();
    stats.final_positive_edges =
        candidate_keys(&final_snapshot, vertices, Some(threshold_squared)).len();
    certify_topology(
        vertices,
        &state.cycles,
        &final_snapshot,
        TopologyCertification {
            require_no_exact_edges: true,
            affected_cells: None,
            pending: Some(&state.pending),
        },
        &mut work,
        Phase::FinalCertification,
    )?;
    if state.pending.iter().any(Option::is_some) {
        return Err(work.failure(
            FailureKind::UnsafeQuotient,
            Phase::FinalCertification,
            "fixed point retained a pending degree-two subdivision",
        ));
    }
    let final_positive_deviation = certify_positive_provenance(
        &state.provenance,
        vertices,
        threshold,
        None,
        &mut work,
        Phase::FinalCertification,
    )?;
    let (exact_members, positive_members, max_exact_cross_track) =
        audit_provenance(&state.provenance).map_err(|message| {
            work.failure(
                FailureKind::UnsafeQuotient,
                Phase::FinalCertification,
                message,
            )
        })?;
    stats.exact_suppression_members = exact_members;
    stats.positive_suppression_members = positive_members;
    stats.max_exact_suppression_cross_track_radians = max_exact_cross_track;
    stats.max_positive_suppression_unit_arc_chord = stats
        .max_positive_suppression_unit_arc_chord
        .max(final_positive_deviation);
    stats.effective_cells_elided = state.cycles.iter().filter(|cycle| cycle.is_none()).count();
    let (vertices, cycles, effective_to_cell, cell_to_effective) =
        compact_output(vertices, state.cycles);
    Ok(Outcome {
        vertices,
        cycles,
        effective_to_cell,
        cell_to_effective,
        stats,
        work: work.stats,
    })
}

fn add_occurrences(
    field: &mut u64,
    count: usize,
    work: &WorkTracker,
    phase: Phase,
) -> Result<(), Failure> {
    *field = field.checked_add(count as u64).ok_or_else(|| {
        work.failure(
            FailureKind::CounterOverflow,
            phase,
            "simplification result occurrence counter overflow",
        )
    })?;
    Ok(())
}

fn compact_output(vertices: &[Vec3], cycles: OptionalCycles) -> CompactedOutput {
    let mut used = vec![false; vertices.len()];
    for cycle in cycles.iter().flatten() {
        for &vertex in cycle {
            used[vertex as usize] = true;
        }
    }
    let mut old_to_new = vec![u32::MAX; vertices.len()];
    let mut final_vertices = Vec::with_capacity(used.iter().filter(|&&used| used).count());
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            old_to_new[old] = final_vertices.len() as u32;
            final_vertices.push(vertices[old]);
        }
    }
    let mut effective_to_cell = vec![None; cycles.len()];
    let mut cell_to_effective = Vec::new();
    let mut final_cycles = Vec::new();
    for (effective, cycle) in cycles.into_iter().enumerate() {
        let Some(mut cycle) = cycle else { continue };
        for vertex in &mut cycle {
            *vertex = old_to_new[*vertex as usize];
        }
        effective_to_cell[effective] = Some(final_cycles.len() as u32);
        cell_to_effective.push(effective as u32);
        final_cycles.push(cycle);
    }
    (
        final_vertices,
        final_cycles,
        effective_to_cell,
        cell_to_effective,
    )
}

pub(crate) fn simplify(
    vertices: &[Vec3],
    cells: &[crate::diagram::VoronoiCell],
    cell_indices: &[u32],
    threshold: f64,
    policy: CellPolicy,
    limits: Limits,
) -> Result<Outcome, Failure> {
    if policy == CellPolicy::Elide {
        return simplify_elide(vertices, cells, cell_indices, threshold, limits);
    }
    let threshold_squared = threshold * threshold;
    let mut cycles = Vec::with_capacity(cells.len());
    for cell in cells {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        let Some(cycle) = cell_indices.get(start..end) else {
            return Err(Failure {
                kind: FailureKind::Validation,
                phase: Phase::Preparation,
                message: "source cell span exceeds the index buffer".into(),
                work: WorkStats::default(),
                affected_effective_cells: Vec::new(),
            });
        };
        cycles.push(Some(cycle.to_vec()));
    }
    let mut work = WorkTracker::new(limits);
    preflight_stored_positions(vertices, &cycles, &mut work)?;
    let mut ledger: Vec<Vec<u32>> = (0..vertices.len() as u32)
        .map(|vertex| vec![vertex])
        .collect();
    let mut stats = ResultStats::default();

    'fixed_point: loop {
        stats.round_attempts += 1;
        let snapshot = build_snapshot(&cycles, vertices.len(), &mut work, Phase::Exact)?;
        let exact_candidates = candidate_keys(&snapshot, vertices, None);
        work.note_candidates(exact_candidates.len());
        add_occurrences(
            &mut stats.exact_candidate_occurrences,
            exact_candidates.len(),
            &work,
            Phase::Exact,
        )?;
        if stats.productive_rounds > 0 {
            add_occurrences(
                &mut stats.later_round_candidate_occurrences,
                exact_candidates.len(),
                &work,
                Phase::Exact,
            )?;
        }
        if !exact_candidates.is_empty() {
            let components = build_components(&exact_candidates, vertices.len(), &ledger);
            stats.max_component_members = stats.max_component_members.max(
                components
                    .iter()
                    .map(|component| component.source_members.len())
                    .max()
                    .unwrap_or(0),
            );
            let groups = interaction_groups(
                &components,
                &cycles,
                vertices.len(),
                &mut work,
                Phase::Exact,
            )?;
            let group = &groups[0];
            let selected = selected_candidate_keys(&exact_candidates, &components, group);
            let deferred_exact: FxHashSet<EdgeKey> = exact_candidates
                .iter()
                .copied()
                .filter(|key| !selected.contains(key))
                .collect();
            match attempt_non_elide_transaction(
                TransactionRequest {
                    vertices,
                    cycles: &cycles,
                    ledger: &ledger,
                    components: &components,
                    group,
                    deferred_exact: &deferred_exact,
                    positive: false,
                    policy,
                },
                &mut work,
            )? {
                TransactionResult::Commit {
                    cycles: committed_cycles,
                    ledger: committed_ledger,
                    retired_vertices,
                    exact_components_closed,
                } => {
                    cycles = committed_cycles;
                    ledger = committed_ledger;
                    stats.productive_rounds += 1;
                    stats.committed_transactions += 1;
                    stats.exact_components_committed += group.len() + exact_components_closed;
                    stats.vertices_retired += retired_vertices;
                    continue 'fixed_point;
                }
                TransactionResult::DeclineCell | TransactionResult::DeclineTopology => {
                    unreachable!("mandatory exact transaction cannot be declined")
                }
            }
        }

        let positive_candidates = candidate_keys(&snapshot, vertices, Some(threshold_squared));
        work.note_candidates(positive_candidates.len());
        add_occurrences(
            &mut stats.positive_candidate_occurrences,
            positive_candidates.len(),
            &work,
            Phase::Positive,
        )?;
        if stats.productive_rounds > 0 {
            add_occurrences(
                &mut stats.later_round_candidate_occurrences,
                positive_candidates.len(),
                &work,
                Phase::Positive,
            )?;
        }
        if positive_candidates.is_empty() {
            break 'fixed_point;
        }

        let mut eligible = Vec::new();
        for mut component in build_components(&positive_candidates, vertices.len(), &ledger) {
            stats.max_component_members = stats
                .max_component_members
                .max(component.source_members.len());
            if certify_component_diameter(&mut component, vertices, threshold_squared, &mut work)? {
                eligible.push(component);
            } else {
                add_occurrences(
                    &mut stats.positive_components_declined_diameter,
                    1,
                    &work,
                    Phase::Positive,
                )?;
            }
        }
        if eligible.is_empty() {
            break 'fixed_point;
        }
        let groups = interaction_groups(
            &eligible,
            &cycles,
            vertices.len(),
            &mut work,
            Phase::Positive,
        )?;
        let deferred_exact = FxHashSet::default();
        let mut committed = false;
        for group in &groups {
            match attempt_non_elide_transaction(
                TransactionRequest {
                    vertices,
                    cycles: &cycles,
                    ledger: &ledger,
                    components: &eligible,
                    group,
                    deferred_exact: &deferred_exact,
                    positive: true,
                    policy,
                },
                &mut work,
            )? {
                TransactionResult::Commit {
                    cycles: committed_cycles,
                    ledger: committed_ledger,
                    retired_vertices,
                    exact_components_closed,
                } => {
                    for &component in group {
                        stats.max_component_diameter = stats
                            .max_component_diameter
                            .max(eligible[component].diameter);
                        stats.max_representative_displacement = stats
                            .max_representative_displacement
                            .max(eligible[component].displacement);
                    }
                    cycles = committed_cycles;
                    ledger = committed_ledger;
                    stats.productive_rounds += 1;
                    stats.committed_transactions += 1;
                    stats.positive_components_committed += group.len();
                    stats.exact_components_committed += exact_components_closed;
                    stats.vertices_retired += retired_vertices;
                    committed = true;
                    break;
                }
                TransactionResult::DeclineCell => {
                    add_occurrences(
                        &mut stats.positive_groups_declined_cell,
                        1,
                        &work,
                        Phase::Positive,
                    )?;
                }
                TransactionResult::DeclineTopology => {
                    add_occurrences(
                        &mut stats.positive_groups_declined_topology,
                        1,
                        &work,
                        Phase::Positive,
                    )?;
                }
            }
        }
        if committed {
            continue 'fixed_point;
        }
        break 'fixed_point;
    }

    let final_snapshot = build_snapshot(
        &cycles,
        vertices.len(),
        &mut work,
        Phase::FinalCertification,
    )?;
    stats.final_exact_edges = candidate_keys(&final_snapshot, vertices, None).len();
    stats.final_positive_edges =
        candidate_keys(&final_snapshot, vertices, Some(threshold_squared)).len();
    certify_topology(
        vertices,
        &cycles,
        &final_snapshot,
        TopologyCertification {
            require_no_exact_edges: true,
            affected_cells: None,
            pending: None,
        },
        &mut work,
        Phase::FinalCertification,
    )?;
    let (vertices, cycles, effective_to_cell, cell_to_effective) = compact_output(vertices, cycles);
    Ok(Outcome {
        vertices,
        cycles,
        effective_to_cell,
        cell_to_effective,
        stats,
        work: work.stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagram::VoronoiCell;

    fn cube_fixture() -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
        let mut vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0).normalize(),
            Vec3::new(1.0, -1.0, -1.0).normalize(),
            Vec3::new(1.0, 1.0, -1.0).normalize(),
            Vec3::new(-1.0, 1.0, -1.0).normalize(),
            Vec3::new(-1.0, -1.0, 1.0).normalize(),
            Vec3::new(1.0, -1.0, 1.0).normalize(),
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-1.0, 1.0, 1.0).normalize(),
        ];
        vertices[0] = (vertices[1] + Vec3::new(-1.0e-3, 2.0e-4, 3.0e-4)).normalize();
        let cycles: [&[u32]; 6] = [
            &[0, 3, 2, 1],
            &[4, 5, 6, 7],
            &[0, 1, 5, 4],
            &[1, 2, 6, 5],
            &[2, 3, 7, 6],
            &[3, 0, 4, 7],
        ];
        let mut cells = Vec::new();
        let mut indices = Vec::new();
        for cycle in cycles {
            cells.push(VoronoiCell::new(indices.len() as u32, cycle.len() as u16));
            indices.extend_from_slice(cycle);
        }
        (vertices, cells, indices)
    }

    fn generous_limits() -> Limits {
        Limits {
            diameter_pair_comparisons: 1_000_000,
            cell_index_visits: 1_000_000,
            provenance_member_checks: 1_000_000,
        }
    }

    fn tetrahedron_fixture() -> (Vec<Vec3>, Vec<VoronoiCell>, Vec<u32>) {
        let mut vertices = vec![
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-1.0, -1.0, 1.0).normalize(),
            Vec3::new(-1.0, 1.0, -1.0).normalize(),
            Vec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        vertices[0] = (vertices[1] + Vec3::new(1.0e-3, 2.0e-4, -3.0e-4)).normalize();
        let cycles: [&[u32]; 4] = [&[0, 1, 2], &[0, 3, 1], &[0, 2, 3], &[1, 3, 2]];
        let mut cells = Vec::new();
        let mut indices = Vec::new();
        for cycle in cycles {
            cells.push(VoronoiCell::new(indices.len() as u32, cycle.len() as u16));
            indices.extend_from_slice(cycle);
        }
        (vertices, cells, indices)
    }

    #[test]
    fn preserve_collapses_one_isolated_short_cube_edge() {
        let (vertices, cells, indices) = cube_fixture();
        let threshold = stored_distance_squared(vertices[0], vertices[1]).sqrt() * 1.01;
        let outcome = simplify(
            &vertices,
            &cells,
            &indices,
            threshold,
            CellPolicy::Preserve,
            generous_limits(),
        )
        .unwrap();

        assert_eq!(outcome.vertices.len(), 7);
        assert_eq!(outcome.cycles.len(), 6);
        assert_eq!(outcome.stats.positive_components_committed, 1);
        assert_eq!(outcome.stats.vertices_retired, 1);
        assert_eq!(outcome.stats.final_exact_edges, 0);
        assert_eq!(outcome.stats.final_positive_edges, 0);
    }

    #[test]
    fn preserve_declines_a_component_over_the_requested_diameter() {
        let (mut vertices, cells, indices) = cube_fixture();
        vertices[2] = (vertices[1] + Vec3::new(1.0e-3, 2.0e-4, -3.0e-4)).normalize();
        let edge_01 = stored_distance_squared(vertices[0], vertices[1]).sqrt();
        let edge_12 = stored_distance_squared(vertices[1], vertices[2]).sqrt();
        let diameter = stored_distance_squared(vertices[0], vertices[2]).sqrt();
        let threshold = edge_01.max(edge_12) * 1.01;
        assert!(diameter > threshold);

        let outcome = simplify(
            &vertices,
            &cells,
            &indices,
            threshold,
            CellPolicy::Preserve,
            generous_limits(),
        )
        .unwrap();

        assert_eq!(outcome.vertices.len(), 8);
        assert_eq!(outcome.stats.positive_components_committed, 0);
        assert_eq!(outcome.stats.positive_components_declined_diameter, 1);
        assert!(outcome.stats.final_positive_edges >= 2);
    }

    #[test]
    fn error_stops_at_the_first_cell_killing_positive_group() {
        let (vertices, cells, indices) = tetrahedron_fixture();
        let threshold = stored_distance_squared(vertices[0], vertices[1]).sqrt() * 1.01;
        let failure = simplify(
            &vertices,
            &cells,
            &indices,
            threshold,
            CellPolicy::Error,
            generous_limits(),
        )
        .unwrap_err();

        assert_eq!(failure.kind, FailureKind::CellEliminationRequired);
        assert_eq!(failure.phase, Phase::Positive);
        assert_eq!(failure.affected_effective_cells, vec![0, 1]);
    }

    #[test]
    fn work_limits_charge_atomically_and_overflow_wins() {
        let mut work = WorkTracker::new(Limits {
            diameter_pair_comparisons: 2,
            cell_index_visits: 3,
            provenance_member_checks: 4,
        });
        work.charge_cell_indices(3, Phase::Preparation).unwrap();
        let failure = work.charge_cell_indices(1, Phase::Preparation).unwrap_err();
        assert_eq!(failure.kind, FailureKind::CellIndexLimit);
        assert_eq!(failure.work.cell_index_visits, 3);

        let mut overflow = WorkTracker::new(Limits {
            diameter_pair_comparisons: 0,
            cell_index_visits: u64::MAX,
            provenance_member_checks: 0,
        });
        overflow.stats.cell_index_visits = u64::MAX;
        let failure = overflow
            .charge_cell_indices(1, Phase::FinalCertification)
            .unwrap_err();
        assert_eq!(failure.kind, FailureKind::CounterOverflow);
        assert_eq!(failure.work.cell_index_visits, u64::MAX);
    }

    #[test]
    fn minor_arc_metric_handles_projection_endpoint_and_conditioning_cases() {
        let start = Vec3::X;
        let end = Vec3::Y;
        let midpoint = (start + end).normalize();
        assert_eq!(point_to_minor_arc_chord(midpoint, start, end), Some(0.0));

        // The point is normal to the supporting great-circle plane, so the
        // projection is singular and the deterministic endpoint fallback is
        // used. Both endpoints tie at a quarter turn.
        let fallback = point_to_minor_arc_chord(Vec3::Z, start, end).unwrap();
        assert!((fallback - 2.0f64.sqrt()).abs() < 1.0e-15);

        let below_floor = Vec3::new(1.0, 1.0e-13, 0.0).normalize();
        assert_eq!(point_to_minor_arc_chord(start, start, below_floor), None);
        let above_floor = Vec3::new(1.0, 2.0e-12, 0.0).normalize();
        assert_eq!(
            point_to_minor_arc_chord(start, start, above_floor),
            Some(0.0)
        );
    }

    #[test]
    fn skewed_provenance_bags_are_audited_iteratively() {
        const MEMBERS: usize = 10_000;
        let mut provenance = ProvenanceState::new(MEMBERS + 2);
        let mut root = provenance.leaf(0, true);
        provenance.expected[0] = true;
        for member in 1..MEMBERS as u32 {
            let leaf = provenance.leaf(member, true);
            provenance.expected[member as usize] = true;
            root = provenance.meld(root, leaf, true);
        }
        provenance
            .edge_roots
            .insert((MEMBERS as u32, MEMBERS as u32 + 1), root);

        let (exact, positive, max_cross_track) = audit_provenance(&provenance).unwrap();
        assert_eq!(exact, 0);
        assert_eq!(positive, MEMBERS);
        assert_eq!(max_cross_track, 0.0);
    }

    #[test]
    fn preflight_rejects_non_edge_driven_two_position_faces_after_charging_entries() {
        let vertices = vec![Vec3::X, Vec3::Y, Vec3::X, Vec3::Y];
        let cycles = vec![Some(vec![0, 1, 2, 3])];
        let mut work = WorkTracker::new(generous_limits());
        let failure = preflight_stored_positions(&vertices, &cycles, &mut work).unwrap_err();
        assert_eq!(failure.kind, FailureKind::UnsupportedStoredDegeneracy);
        assert_eq!(failure.phase, Phase::SourcePreflight);
        assert_eq!(failure.affected_effective_cells, vec![0]);
        assert_eq!(failure.work.cell_index_visits, 4);

        let mut limited = WorkTracker::new(Limits {
            cell_index_visits: 3,
            ..generous_limits()
        });
        let failure = preflight_stored_positions(&vertices, &cycles, &mut limited).unwrap_err();
        assert_eq!(failure.kind, FailureKind::CellIndexLimit);
        assert_eq!(failure.phase, Phase::Preparation);
        assert_eq!(failure.work.cell_index_visits, 3);
        assert!(failure.affected_effective_cells.is_empty());
    }

    #[test]
    fn stored_chord_candidate_boundary_is_inclusive() {
        let vertices = vec![Vec3::X, Vec3::new(1.0, 1.0e-3, 0.0).normalize()];
        let key = (0, 1);
        let snapshot = Snapshot {
            edges: FxHashMap::from_iter([(
                key,
                EdgeRecord {
                    first: EdgeUse {
                        cell: 0,
                        from: 0,
                        to: 1,
                    },
                    second: Some(EdgeUse {
                        cell: 1,
                        from: 1,
                        to: 0,
                    }),
                },
            )]),
            incidence: vec![vec![0, 1], vec![0, 1]],
            live_vertices: vec![true, true],
        };
        let distance_squared = stored_distance_squared(vertices[0], vertices[1]);
        assert!(candidate_keys(
            &snapshot,
            &vertices,
            Some(f64::from_bits(distance_squared.to_bits() - 1))
        )
        .is_empty());
        assert_eq!(
            candidate_keys(&snapshot, &vertices, Some(distance_squared)),
            vec![key]
        );
        assert_eq!(
            candidate_keys(
                &snapshot,
                &vertices,
                Some(f64::from_bits(distance_squared.to_bits() + 1))
            ),
            vec![key]
        );
    }

    #[test]
    fn diameter_and_provenance_limits_fail_before_the_governed_check() {
        let vertices = vec![Vec3::X, Vec3::Y];
        let mut component = Component {
            representative: 0,
            members: vec![0, 1],
            source_members: vec![0, 1],
            diameter: 0.0,
            displacement: 0.0,
        };
        let mut work = WorkTracker::new(Limits {
            diameter_pair_comparisons: 0,
            ..generous_limits()
        });
        let failure =
            certify_component_diameter(&mut component, &vertices, 4.0, &mut work).unwrap_err();
        assert_eq!(failure.kind, FailureKind::DiameterLimit);
        assert_eq!(failure.work.diameter_pair_comparisons, 0);

        let mut provenance = ProvenanceState::new(2);
        let root = provenance.leaf(0, true);
        provenance.expected[0] = true;
        provenance.edge_roots.insert((0, 1), root);
        let mut work = WorkTracker::new(Limits {
            provenance_member_checks: 0,
            ..generous_limits()
        });
        let failure = certify_positive_provenance(
            &provenance,
            &vertices,
            2.0,
            None,
            &mut work,
            Phase::FinalCertification,
        )
        .unwrap_err();
        assert_eq!(failure.kind, FailureKind::ProvenanceLimit);
        assert_eq!(failure.work.provenance_member_checks, 0);

        work.limits.provenance_member_checks = 1;
        assert_eq!(
            certify_positive_provenance(
                &provenance,
                &vertices,
                2.0,
                None,
                &mut work,
                Phase::FinalCertification,
            )
            .unwrap(),
            0.0
        );
        assert_eq!(work.stats.provenance_member_checks, 1);
    }

    #[test]
    fn positive_pending_refresh_only_upgrades_affected_vertices() {
        let snapshot = Snapshot {
            edges: FxHashMap::default(),
            incidence: vec![vec![0, 1], vec![0, 1], vec![0, 1]],
            live_vertices: vec![true; 3],
        };
        let mut pending = vec![
            Some(SuppressionCause::Exact),
            None,
            Some(SuppressionCause::Exact),
        ];
        refresh_pending(
            &mut pending,
            &snapshot,
            &[false, true, true],
            SuppressionCause::Positive,
        );

        assert_eq!(pending[0], Some(SuppressionCause::Exact));
        assert_eq!(pending[1], Some(SuppressionCause::Positive));
        assert_eq!(pending[2], Some(SuppressionCause::Positive));
    }

    #[test]
    fn current_provenance_certificate_only_checks_affected_owners() {
        let vertices = vec![Vec3::X, Vec3::Y, Vec3::X, Vec3::Y, Vec3::Z];
        let mut provenance = ProvenanceState::new(vertices.len());
        let affected_root = provenance.leaf(0, true);
        let unrelated_root = provenance.leaf(4, true);
        provenance.edge_roots.insert((0, 1), affected_root);
        provenance.edge_roots.insert((2, 3), unrelated_root);
        let affected = AffectedProvenanceOwners {
            edges: vec![(0, 1)],
            sinks: Vec::new(),
        };
        let mut work = WorkTracker::new(Limits {
            provenance_member_checks: 1,
            ..generous_limits()
        });

        assert_eq!(
            certify_positive_provenance(
                &provenance,
                &vertices,
                0.1,
                Some(&affected),
                &mut work,
                Phase::Positive,
            )
            .unwrap(),
            0.0
        );
        assert_eq!(work.stats.provenance_member_checks, 1);

        let mut global_work = WorkTracker::new(generous_limits());
        let failure = certify_positive_provenance(
            &provenance,
            &vertices,
            0.1,
            None,
            &mut global_work,
            Phase::FinalCertification,
        )
        .unwrap_err();
        assert_eq!(failure.kind, FailureKind::PositiveSuppressionDeviation);
    }

    #[test]
    fn positive_endpoint_contraction_upgrades_unchanged_edge_and_sink_owners() {
        let mut provenance = ProvenanceState::new(3);
        let edge_root = provenance.leaf(0, false);
        let sink_root = provenance.leaf(1, false);
        provenance.edge_roots.insert((0, 2), edge_root);
        provenance.sinks[0] = Some(sink_root);
        let ledger = vec![vec![0, 1], Vec::new(), vec![2]];
        let live_edges = FxHashMap::from_iter([(
            (0, 2),
            EdgeRecord {
                first: EdgeUse {
                    cell: 0,
                    from: 0,
                    to: 2,
                },
                second: Some(EdgeUse {
                    cell: 1,
                    from: 2,
                    to: 0,
                }),
            },
        )]);

        reconcile_provenance(
            &mut provenance,
            &ledger,
            &live_edges,
            &[true, false, true],
            true,
            &[true, true, false],
        )
        .unwrap();

        assert!(provenance.edge_roots[&(0, 2)].positive);
        assert!(provenance.sinks[0].unwrap().positive);
    }

    #[test]
    fn exact_transaction_rejects_an_affected_two_position_face() {
        let (mut vertices, _, _) = cube_fixture();
        vertices[0] = Vec3::X;
        vertices[1] = Vec3::Y;
        vertices[2] = Vec3::X;
        vertices[3] = Vec3::Y;
        let cycles = vec![
            Some(vec![0, 3, 2, 1]),
            Some(vec![4, 5, 6, 7]),
            Some(vec![0, 1, 5, 4]),
            Some(vec![1, 2, 6, 5]),
            Some(vec![2, 3, 7, 6]),
            Some(vec![3, 0, 4, 7]),
        ];
        let mut work = WorkTracker::new(generous_limits());
        let snapshot = build_snapshot(&cycles, vertices.len(), &mut work, Phase::Exact).unwrap();
        let mut affected_cells = vec![false; cycles.len()];
        affected_cells[0] = true;

        let failure = certify_topology(
            &vertices,
            &cycles,
            &snapshot,
            TopologyCertification {
                require_no_exact_edges: false,
                affected_cells: Some(&affected_cells),
                pending: None,
            },
            &mut work,
            Phase::Exact,
        )
        .unwrap_err();
        assert_eq!(failure.kind, FailureKind::UnsafeQuotient);
        assert!(failure.message.contains("exact stored positions"));
    }

    #[test]
    fn entry_semantics_precede_later_cell_index_charges() {
        let mut work = WorkTracker::new(Limits {
            cell_index_visits: 1,
            ..generous_limits()
        });
        let failure =
            build_snapshot(&[Some(vec![3, 0, 1])], 3, &mut work, Phase::Preparation).unwrap_err();

        assert_eq!(failure.kind, FailureKind::UnsafeQuotient);
        assert_eq!(failure.work.cell_index_visits, 1);
        assert!(failure.message.contains("out-of-range"));
    }

    #[test]
    fn committed_contraction_exposes_a_fresh_short_edge_next_round() {
        let (mut vertices, cells, indices) = cube_fixture();
        vertices[0] = Vec3::X;
        vertices[1] = Vec3::new(1.0, 1.0e-3, 0.0).normalize();
        vertices[2] = Vec3::new(1.0, -1.0e-3, 0.0).normalize();
        let threshold = 1.1e-3;
        assert!(stored_distance_squared(vertices[0], vertices[1]).sqrt() < threshold);
        assert!(stored_distance_squared(vertices[0], vertices[2]).sqrt() < threshold);
        assert!(stored_distance_squared(vertices[1], vertices[2]).sqrt() > threshold);

        let outcome = simplify(
            &vertices,
            &cells,
            &indices,
            threshold,
            CellPolicy::Preserve,
            generous_limits(),
        )
        .unwrap();
        assert_eq!(outcome.stats.positive_components_committed, 1);
        assert!(outcome.stats.later_round_candidate_occurrences > 0);
        assert!(outcome.stats.positive_components_declined_diameter > 0);
        assert!(outcome.stats.final_positive_edges > 0);
    }

    #[test]
    fn live_edge_table_rejects_unpaired_and_same_direction_uses() {
        let mut work = WorkTracker::new(generous_limits());
        let failure =
            build_snapshot(&[Some(vec![0, 1, 2])], 3, &mut work, Phase::Preparation).unwrap_err();
        assert_eq!(failure.kind, FailureKind::UnsafeQuotient);
        assert!(failure.message.contains("only one cell use"));

        let mut work = WorkTracker::new(generous_limits());
        let failure = build_snapshot(
            &[Some(vec![0, 1, 2]), Some(vec![0, 1, 2])],
            3,
            &mut work,
            Phase::Preparation,
        )
        .unwrap_err();
        assert_eq!(failure.kind, FailureKind::UnsafeQuotient);
        assert!(failure.message.contains("same orientation"));
    }
}
