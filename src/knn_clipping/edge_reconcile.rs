//! Narrow shared-edge reconciliation helpers for post-processing.
//!
//! This pass is intentionally limited. It is not a generic recovery layer for arbitrary
//! topology failures; it only reconciles unresolved shared-edge mismatches that survive
//! live dedup.
//!
//! The two supported anomaly classes are:
//! - one-sided epsilon edges, where one polygon emits a tiny boundary edge and the other side
//!   collapses it away
//! - shared-edge endpoint identity mismatches, typically from near-degenerate vertex ownership
//!   choices where adjacent polygons pick different generator triplets for the same corner

mod telemetry;

use glam::Vec3;

use crate::cell_layout::{CellSpanError, LiveCellLayout};
use crate::diagram::VoronoiCell;
use crate::live_dedup::VertexKey;
use crate::live_dedup::{EdgeKey, EdgeRecord, ShardedVertexKeys};

pub(crate) use telemetry::emit_primary_reconcile_telemetry;

/// Read-only view of vertex keys passed to reconciliation. `Flat` backs the
/// unit tests (and any caller holding a contiguous array); `Sharded` is the
/// production path, looking keys up per-shard without a global concatenation.
/// `Copy` so it threads through the reconciliation helpers by value.
#[derive(Clone, Copy)]
pub(crate) enum VertexKeys<'a> {
    // Used by the unit tests (and any caller holding a contiguous array).
    #[cfg(test)]
    Flat(&'a [VertexKey]),
    Sharded(&'a ShardedVertexKeys),
}

impl VertexKeys<'_> {
    #[inline]
    fn get(&self, vid: u32) -> Option<VertexKey> {
        match self {
            #[cfg(test)]
            VertexKeys::Flat(s) => s.get(vid as usize).copied(),
            VertexKeys::Sharded(s) => s.get(vid),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            #[cfg(test)]
            VertexKeys::Flat(s) => s.len(),
            VertexKeys::Sharded(s) => s.len(),
        }
    }

    /// Visit every `(vid, key)` in global slot order. Only the global-scan
    /// escape path and the debug oracle need this; the localized BFS does not.
    fn for_each(&self, f: impl FnMut(u32, VertexKey)) {
        match self {
            #[cfg(test)]
            VertexKeys::Flat(s) => {
                let mut f = f;
                for (i, &k) in s.iter().enumerate() {
                    f(i as u32, k);
                }
            }
            VertexKeys::Sharded(s) => s.for_each(f),
        }
    }
}

fn reconcile_state_error(message: impl Into<String>) -> crate::VoronoiError {
    crate::VoronoiError::ComputationFailed(message.into())
}

/// Error for post-reconciliation residuals on the plain compute paths: a non-empty
/// residual list means the output is provably not a valid subdivision (some
/// interior edge stays unpaired, overused, or misoriented), and those paths have no report channel to
/// surface it — so they fail loud rather than return a known-invalid
/// diagram. `pairs` are the offending cell/generator pairs (capped in the
/// message). Never constructed on clean runs (the list is empty).
pub(crate) fn residual_error(pairs: &[(u32, u32)]) -> crate::VoronoiError {
    let shown: Vec<String> = pairs
        .iter()
        .take(8)
        .map(|&(a, b)| format!("({a},{b})"))
        .collect();
    let more = if pairs.len() > 8 {
        format!(" (+{} more)", pairs.len() - 8)
    } else {
        String::new()
    };
    crate::VoronoiError::ComputationFailed(format!(
        "edge reconciliation left {} bad interior edge(s) (unpaired, overused, or \
         misoriented) — output is not a valid \
         subdivision: {}{more}. Use compute_with_report to inspect, or report this input.",
        pairs.len(),
        shown.join(" ")
    ))
}

/// Error used when reconciliation requested a Hull3d local rebuild and the
/// configured rebuild path did not accept a replacement.
pub(crate) fn reconciliation_rejection_error(pairs: &[(u32, u32)]) -> crate::VoronoiError {
    let shown: Vec<String> = pairs
        .iter()
        .take(8)
        .map(|&(a, b)| format!("({a},{b})"))
        .collect();
    let more = if pairs.len() > 8 {
        format!(" and {} more", pairs.len() - 8)
    } else {
        String::new()
    };
    crate::VoronoiError::ComputationFailed(format!(
        "reconciliation found component(s) requiring Hull3d near cell pair(s) {}{} and Hull3d did not accept a replacement",
        shown.join(", "),
        more,
    ))
}

#[inline]
pub(crate) fn unpack_edge(key: u64) -> (u32, u32) {
    (key as u32, (key >> 32) as u32)
}

fn key_contains(key: VertexKey, value: u32) -> bool {
    key[0] == value || key[1] == value || key[2] == value
}

pub(crate) fn shared_neighbor(cell_idx: u32, a: VertexKey, b: VertexKey) -> Option<u32> {
    if !key_contains(a, cell_idx) || !key_contains(b, cell_idx) {
        return None;
    }
    a.iter()
        .find(|&&candidate| candidate != cell_idx && key_contains(b, candidate))
        .copied()
}

fn cell_vertex_slice<'a>(
    cell_idx: u32,
    cells: &[VoronoiCell],
    cell_indices: &'a [u32],
) -> Result<&'a [u32], crate::VoronoiError> {
    cell_vertex_slice_from_layout(cell_idx, LiveCellLayout::new(cells, cell_indices))
}

fn cell_vertex_slice_from_layout<'a>(
    cell_idx: u32,
    layout: LiveCellLayout<'_, 'a>,
) -> Result<&'a [u32], crate::VoronoiError> {
    match layout.checked_span(cell_idx as usize) {
        Ok(span) => Ok(span),
        Err(CellSpanError::CellOutOfBounds { cell, cell_count }) => {
            Err(reconcile_state_error(format!(
                "edge reconciliation referenced out-of-range cell {cell} (cells={cell_count})"
            )))
        }
        Err(CellSpanError::SpanEndOverflow { cell, start, count }) => {
            Err(reconcile_state_error(format!(
                "edge reconciliation cell {cell} span start {start} + count {count} overflows usize"
            )))
        }
        Err(CellSpanError::SpanOutOfBounds {
            cell,
            start,
            end,
            index_count,
        }) => Err(reconcile_state_error(format!(
            "edge reconciliation cell {cell} span [{start}..{end}) exceeds cell index buffer len {index_count}"
        ))),
    }
}

#[cfg(test)]
pub(crate) fn edge_segments_for_neighbor(
    cell_idx: u32,
    neighbor: u32,
    layout: LiveCellLayout<'_, '_>,
    vertex_keys: VertexKeys<'_>,
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    let mut out = Vec::new();
    edge_segments_for_neighbor_into(cell_idx, neighbor, layout, vertex_keys, &mut out)?;
    Ok(out)
}

fn edge_segments_for_neighbor_into(
    cell_idx: u32,
    neighbor: u32,
    layout: LiveCellLayout<'_, '_>,
    vertex_keys: VertexKeys<'_>,
    out: &mut Vec<(u32, u32)>,
) -> Result<(), crate::VoronoiError> {
    out.clear();
    let slice = cell_vertex_slice_from_layout(cell_idx, layout)?;
    let n = slice.len();
    if n < 2 {
        return Ok(());
    }

    for i in 0..n {
        let vi = slice[i];
        let vj = slice[(i + 1) % n];
        let ki = vertex_keys.get(vi).ok_or_else(|| {
            reconcile_state_error(format!(
                "edge reconciliation vertex id {} out of range for vertex_keys len {}",
                vi,
                vertex_keys.len()
            ))
        })?;
        let kj = vertex_keys.get(vj).ok_or_else(|| {
            reconcile_state_error(format!(
                "edge reconciliation vertex id {} out of range for vertex_keys len {}",
                vj,
                vertex_keys.len()
            ))
        })?;
        if shared_neighbor(cell_idx, ki, kj) == Some(neighbor) {
            out.push((vi, vj));
        }
    }
    Ok(())
}

fn dist_sq(a: Vec3, b: Vec3) -> f32 {
    (a - b).length_squared()
}

fn dist_sq_f64(a: Vec3, b: Vec3) -> f64 {
    let dx = f64::from(a.x) - f64::from(b.x);
    let dy = f64::from(a.y) - f64::from(b.y);
    let dz = f64::from(a.z) - f64::from(b.z);
    dx * dx + dy * dy + dz * dz
}

fn vertex_pos(vertices: &[Vec3], vertex_id: u32) -> Result<Vec3, crate::VoronoiError> {
    vertices.get(vertex_id as usize).copied().ok_or_else(|| {
        reconcile_state_error(format!(
            "edge reconciliation vertex id {} out of range for vertex buffer len {}",
            vertex_id,
            vertices.len()
        ))
    })
}

use super::union_find::SparseUnionFind;

/// Rebuilt cell table and index buffer after reconciliation.
pub(crate) type ReconciledCells = (Vec<VoronoiCell>, Vec<u32>);

/// Outcome of [`reconcile_edge_mismatches`].
///
/// `merge_affected_cells` exists for the reconciliation's localized residual scan:
/// identity merges remap vertex references in place, so a cell in this set can
/// reference a surviving vertex whose key triple does not name it — the one
/// production violation of the key-ownership invariant ("a vertex keyed
/// `(a, b, c)` is referenced only by cells `a`, `b`, `c`"). Consumers relying
/// on that invariant to localize must treat these cells as always in scope.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ReconcileResult {
    /// Surviving bad interior edges (unpaired, overused, or misoriented), as
    /// owning cell pairs for the caller's report / reconciliation trigger.
    pub residual_pairs: Vec<(u32, u32)>,
    /// Cell pairs whose proposed tolerance component exceeded the configured
    /// diameter. These are explicit Hull3d seeds even when the unmodified
    /// output happens not to expose an unpaired edge.
    pub local_rebuild_seed_pairs: Vec<(u32, u32)>,
    /// Cells whose spans were rewritten by identity merges (sorted, deduped):
    /// the union of key triples over every vertex id that entered a merge.
    pub merge_affected_cells: Vec<u32>,
    /// Complete local footprint whose final cycles must be rescanned for exact
    /// stored-zero edges after reconciliation. Empty when no span changed.
    /// Includes record-owner cells for collinear drops and key-owner cells for
    /// accepted identity merges.
    pub resolution_scan_cells: Vec<u32>,
    /// Number of cell cycles examined by the merge-safety face check across
    /// all reconciliation rounds. Exposed to timing telemetry so localized
    /// coverage can be compared with the full diagram size.
    pub merge_safety_scan_cells: usize,
    /// Number of rounds whose merge-safety cover could not be certified from
    /// vertex provenance and therefore used the global cell scan.
    pub merge_safety_global_fallbacks: usize,
}

/// Original vertex ids represented by a surviving id after accepted
/// reconciliation rounds. Keeping this ledger across rounds prevents a later
/// short link from extending an earlier component beyond the epsilon-diameter
/// policy after the earlier members have disappeared from the cell spans.
#[derive(Default)]
struct MergeLedger {
    members: rustc_hash::FxHashMap<u32, Vec<u32>>,
}

struct RejectedMergeComponent {
    current_ids: Vec<u32>,
    member_ids: Vec<u32>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct MergeSafetyStats {
    scanned_cells: usize,
    global_fallbacks: usize,
}

/// Mutable bookkeeping shared by the primary and synthesized-backstop
/// reconciliation passes. Constructed only after the empty-record fast path.
#[derive(Default)]
struct ReconcileRunState {
    merge_ledger: MergeLedger,
    local_rebuild_seed_pairs: Vec<(u32, u32)>,
    merge_affected_cells: Vec<u32>,
    resolution_scan_cells: Vec<u32>,
    merge_safety: MergeSafetyStats,
}

impl ReconcileRunState {
    fn record_changed_cells(&mut self, candidate_cells: &[u32]) {
        self.resolution_scan_cells
            .extend_from_slice(candidate_cells);
    }

    fn into_result(mut self, residual_pairs: Vec<(u32, u32)>) -> ReconcileResult {
        self.local_rebuild_seed_pairs.sort_unstable();
        self.local_rebuild_seed_pairs.dedup();
        self.merge_affected_cells.sort_unstable();
        self.merge_affected_cells.dedup();
        if !self.resolution_scan_cells.is_empty() {
            self.resolution_scan_cells
                .extend_from_slice(&self.merge_affected_cells);
        }
        self.resolution_scan_cells.sort_unstable();
        self.resolution_scan_cells.dedup();
        ReconcileResult {
            residual_pairs,
            local_rebuild_seed_pairs: self.local_rebuild_seed_pairs,
            merge_affected_cells: self.merge_affected_cells,
            resolution_scan_cells: self.resolution_scan_cells,
            merge_safety_scan_cells: self.merge_safety.scanned_cells,
            merge_safety_global_fallbacks: self.merge_safety.global_fallbacks,
        }
    }
}

impl MergeLedger {
    fn expanded_members(&self, current_ids: &[u32]) -> Vec<u32> {
        let mut expanded = Vec::new();
        for &id in current_ids {
            if let Some(members) = self.members.get(&id) {
                expanded.extend_from_slice(members);
            } else {
                expanded.push(id);
            }
        }
        expanded.sort_unstable();
        expanded.dedup();
        expanded
    }

    fn commit(&mut self, representative: u32, current_ids: &[u32], expanded: Vec<u32>) {
        for &id in current_ids {
            self.members.remove(&id);
        }
        self.members.insert(representative, expanded);
    }
}

/// How reconciliation merges are applied to the cell arrays.
///
/// `InPlace` is the production default: only cells naming a merged vertex
/// are touched (found via the vertex-key triplets), spans shrink in place,
/// and the index buffer keeps stale tail slots (never read — cells are
/// `(start, count)` spans). Its work is O(defects), not O(diagram). See
/// `docs/performance.md#source-pinned-performance-decisions`.
///
/// `Rebuild` is the full-rewrite differential
/// oracle: the two backends must produce identical per-cell vertex
/// sequences (pinned by the unit tests below and the full-pipeline
/// differential in tests/edge_reconciliation.rs via `VORONOI_MESH_RECONCILE_REBUILD`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ReconcileApply {
    InPlace,
    Rebuild,
}

/// Immutable diagnostic/oracle choices for one defect-bearing reconciliation.
/// Production constructs this only after confirming that mismatch records exist;
/// explicit constructors keep unit differentials independent of process state.
#[derive(Clone, Copy)]
pub(crate) struct ReconcileOptions {
    apply: ReconcileApply,
    force_global_dupscan: bool,
    emit_telemetry: bool,
}

impl ReconcileOptions {
    pub(crate) const fn with_apply(apply: ReconcileApply) -> Self {
        Self {
            apply,
            force_global_dupscan: false,
            emit_telemetry: false,
        }
    }

    /// Snapshot all reconciliation environment knobs once per defect-bearing
    /// computation. Keep each knob's historical value semantics exact.
    pub(crate) fn read_from_env() -> Self {
        Self {
            apply: match std::env::var("VORONOI_MESH_RECONCILE_REBUILD") {
                Ok(v) if v == "1" => ReconcileApply::Rebuild,
                _ => ReconcileApply::InPlace,
            },
            force_global_dupscan: matches!(
                std::env::var("VORONOI_MESH_RECONCILE_GLOBAL_DUPSCAN"),
                Ok(v) if v == "1"
            ),
            emit_telemetry: std::env::var_os("VORONOI_MESH_RECONCILE_TELEMETRY").is_some(),
        }
    }

    pub(crate) const fn emit_telemetry(self) -> bool {
        self.emit_telemetry
    }
}

impl Default for ReconcileOptions {
    fn default() -> Self {
        Self::with_apply(ReconcileApply::InPlace)
    }
}

/// Hard cap on reconciliation rounds; each productive round strictly shrinks some
/// cell span, so termination is structural — the cap is a backstop.
const MAX_RECONCILIATION_ROUNDS: usize = 8;

/// How a reconciliation pass interprets its records when pairing endpoints.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MergeMode {
    /// Bookkeeping-driven records (live-dedup detection): identity and
    /// epsilon-bounded nearest-endpoint pairing for 1-1 segment mismatches.
    Primary,
    /// Output-invariant backstop records (synthesized from unpaired
    /// interior edges): eps-bounded proximity unions only — never
    /// force-merge distant vertices on synthesized evidence.
    ProximityOnly,
}

/// Reconcile unresolved shared-edge mismatches by merging vertex
/// identities, patching `cells` / `cell_indices` via the chosen backend.
///
/// Runs the bookkeeping-driven reconciliation to a fixpoint (merges can expose
/// newly pairable states), then checks the output invariant directly:
/// every interior edge must be used by exactly two cells. Unpaired
/// findings synthesize an eps-bounded backstop pass (the owning cell pair
/// is recovered from the endpoint keys' shared generators); whatever
/// survives is returned as cell pairs for the caller's report rather than
/// force-merged. Returns an empty vec on clean runs (no records) without
/// touching anything — the scans are paid only on defect runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconcile_edge_mismatches(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: VertexKeys<'_>,
    // Degenerate-length threshold in spherical chord units.
    degenerate_len_eps: f32,
    options: ReconcileOptions,
) -> Result<ReconcileResult, crate::VoronoiError> {
    if edge_records.is_empty() {
        // Production fast path: with no detected mismatch there is nothing to
        // reconciliation, and the O(total cell indices) output-invariant scan is
        // skipped — avoiding it on clean runs is the whole point of this
        // early return. Soundness rests on a detection-completeness claim:
        // every bad interior edge produces >= 1 detection record, so an
        // empty record set implies a clean output. That follows from the
        // coverage contract (docs/architecture.md "stitching invariant"): a
        // one-sided edge is either cross-bin (its overflow is a singleton or
        // a mismatch => record) or same-bin (a forwarded check goes unconsumed
        // => record); a same-bin later cell cannot be the lone owner. Rather
        // than trust that argument silently, debug builds run the scan anyway
        // and assert it is clean — turning detection-completeness into a
        // continuously-checked invariant at ZERO release cost. If this ever
        // fires, a defect escaped detection and the early return is unsafe for
        // that input class (revisit the contract, not just this assert).
        #[cfg(debug_assertions)]
        {
            let unpaired = scan_unpaired_interior_global(LiveCellLayout::new(cells, cell_indices))?;
            assert!(
                unpaired.is_empty(),
                "edge-reconcile early-return invariant violated: {} bad interior \
                 edge(s) with ZERO detection records — a defect escaped detection \
                 (see docs/architecture.md stitching invariant)",
                unpaired.len()
            );
        }
        return Ok(ReconcileResult::default());
    }

    // Defect-bearing checked builds audit the cell/index pairing once before
    // reconciliation's readers and mutators rely on it. The clean fast path
    // above and every release build remain untouched.
    #[cfg(debug_assertions)]
    LiveCellLayout::new(cells, cell_indices).debug_assert_valid();

    reconcile_recorded_mismatches(
        edge_records,
        vertices,
        cells,
        cell_indices,
        vertex_keys,
        degenerate_len_eps,
        options,
    )
}

#[allow(clippy::too_many_arguments)]
fn reconcile_recorded_mismatches(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: VertexKeys<'_>,
    degenerate_len_eps: f32,
    options: ReconcileOptions,
) -> Result<ReconcileResult, crate::VoronoiError> {
    debug_assert!(!edge_records.is_empty());

    let mut state = ReconcileRunState::default();
    let primary_candidates = affected_cells_from_records(edge_records);
    let primary_changed = run_reconciliation_rounds(
        edge_records,
        vertices,
        cells,
        cell_indices,
        vertex_keys,
        degenerate_len_eps,
        options,
        MergeMode::Primary,
        &mut state,
    )?;
    if primary_changed {
        state.record_changed_cells(&primary_candidates);
    }

    let unpaired = scan_unpaired_interior(
        LiveCellLayout::new(cells, cell_indices),
        vertex_keys,
        &primary_candidates,
    )?;
    if unpaired.is_empty() {
        return Ok(state.into_result(Vec::new()));
    }
    let synth = synthesize_backstop_records(&unpaired, vertex_keys, cells.len());
    if !synth.is_empty() {
        let synth_changed = run_reconciliation_rounds(
            &synth,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            degenerate_len_eps,
            options,
            MergeMode::ProximityOnly,
            &mut state,
        )?;
        if synth_changed {
            state.record_changed_cells(&affected_cells_from_records(&synth));
        }
    }
    // Residual scan covers both passes' touched regions.
    let mut residual_candidates = primary_candidates;
    residual_candidates.extend(affected_cells_from_records(&synth));
    residual_candidates.sort_unstable();
    residual_candidates.dedup();
    let residual = scan_unpaired_interior(
        LiveCellLayout::new(cells, cell_indices),
        vertex_keys,
        &residual_candidates,
    )?;
    Ok(state.into_result(
        residual
            .iter()
            .map(|&(va, vb, owner)| cell_pair_for_unpaired(va, vb, owner, vertex_keys))
            .collect(),
    ))
}

/// If `key` has exactly two distinct generators (one doubled), return the
/// single (non-doubled) one — the cell that owns the spurious collinear
/// vertex. `None` for a proper triple point or a fully-degenerate key.
#[inline]
fn degenerate_single(key: VertexKey) -> Option<u32> {
    let [a, b, c] = key;
    if a == b && b == c {
        None
    } else if a == b {
        Some(c)
    } else if a == c {
        Some(b)
    } else if b == c {
        Some(a)
    } else {
        None
    }
}

/// Cells named (as the two edge endpoints) by the detection records — the only
/// cells a reconciliation round can legitimately need to touch. Sorted + deduped.
fn affected_cells_from_records(edge_records: &[EdgeRecord]) -> Vec<u32> {
    let mut cells = Vec::with_capacity(edge_records.len() * 2);
    for record in edge_records {
        let (a, b) = unpack_edge(record.key.as_u64());
        cells.push(a);
        cells.push(b);
    }
    cells.sort_unstable();
    cells.dedup();
    cells
}

/// Debug-only: assert the localized `drop_degenerate_collinear_vertices` cannot
/// miss a defect. Every cell that owns a droppable degenerate vertex must be in
/// `candidate_cells`; otherwise a degenerate (= unpaired-edge) defect exists
/// that no detection record names — a detection-completeness contract violation
/// (see docs/architecture.md "stitching invariant"), making localization unsafe.
/// O(total edges) but debug-only, so it costs nothing in release.
#[cfg(debug_assertions)]
fn assert_candidate_covers_droppable(
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    candidate_cells: &[u32],
) {
    for (ci, cell) in cells.iter().enumerate() {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        if end > cell_indices.len() {
            continue;
        }
        let owns_droppable = cell_indices[start..end]
            .iter()
            .any(|&v| vertex_keys.get(v).and_then(degenerate_single) == Some(ci as u32));
        debug_assert!(
            !owns_droppable || candidate_cells.binary_search(&(ci as u32)).is_ok(),
            "edge-reconcile localization gap: cell {ci} owns a droppable degenerate \
             vertex but no edge_record names it (detection-completeness contract \
             violated; see docs/architecture.md stitching invariant)"
        );
    }
}

/// Drop spurious collinear vertices (degenerate keys with a repeated
/// generator) from the cells that own them. Such a vertex lies on a single
/// bisector — both its incident edges in that cell go to the same neighbor
/// — so it is not a Voronoi triple point, and removing it merges the two
/// collinear segments into the real edge (exact). Returns whether anything
/// was dropped.
///
/// Touches only `candidate_cells` (the cells named by the detection records),
/// not the whole vertex set: by the detection-completeness contract, every
/// droppable degenerate vertex's owner cell is an endpoint of some unresolved
/// edge, so the records' cells cover them all. This keeps a reconciliation round
/// O(defect size) instead of O(total vertices) per round. Debug builds assert
/// the coverage via `assert_candidate_covers_droppable`; scale evidence is in
/// `docs/performance.md#source-pinned-performance-decisions`.
fn drop_degenerate_collinear_vertices(
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    vertex_keys: VertexKeys<'_>,
    candidate_cells: &[u32],
) -> bool {
    let mut changed = false;
    for &c in candidate_cells {
        if (c as usize) >= cells.len() {
            continue;
        }
        let cell = cells[c as usize];
        let start = cell.vertex_start();
        let count = cell.vertex_count();
        let end = start + count;
        if end > cell_indices.len() {
            continue;
        }
        let span = &cell_indices[start..end];
        // Compute the kept chain in a scratch buffer first; only write back
        // if we will actually commit (never partially mutate the span).
        let kept: Vec<u32> = span
            .iter()
            .copied()
            .filter(|&v| vertex_keys.get(v).and_then(degenerate_single) != Some(c))
            .collect();
        // Guard: never collapse a cell below a triangle.
        if kept.len() != count && kept.len() >= 3 {
            cell_indices[start..start + kept.len()].copy_from_slice(&kept);
            cells[c as usize] = VoronoiCell::new(start as u32, kept.len() as u16);
            changed = true;
        }
    }
    changed
}

/// Drive collect+apply to a fixpoint (capped). The duplicate-key backstop
/// scan runs only in the first Primary round — its unions are idempotent
/// once applied, and re-counting them would defeat convergence detection.
#[allow(clippy::too_many_arguments)]
fn run_reconciliation_rounds(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: VertexKeys<'_>,
    degenerate_len_eps: f32,
    options: ReconcileOptions,
    mode: MergeMode,
    state: &mut ReconcileRunState,
) -> Result<bool, crate::VoronoiError> {
    let mut any = false;
    // The only cells a round can need to touch are those named by the records.
    // Computed once; reconciliation rounds only remove vertices, so this set is a valid
    // (shrinking) cover for every round, not just the first.
    let candidate_cells = affected_cells_from_records(edge_records);
    #[cfg(debug_assertions)]
    assert_candidate_covers_droppable(cells, cell_indices, vertex_keys, &candidate_cells);
    for round in 0..MAX_RECONCILIATION_ROUNDS {
        // Drop spurious collinear (degenerate-key) vertices first: a vertex
        // whose key has only two distinct generators is not a triple point,
        // it lies on a single bisector (both incident edges go to the same
        // neighbor) — removing it merges the two collinear segments into the
        // real edge and is exact. One cell can carry such a point where its
        // neighbor sees a straight edge, which is precisely an unpaired-edge
        // defect; this heals it with no cross-cell rewrite.
        let dropped =
            drop_degenerate_collinear_vertices(cells, cell_indices, vertex_keys, &candidate_cells);
        let scan_dup_keys = mode == MergeMode::Primary && round == 0;
        let (mut proposed, _) = collect_merges(
            edge_records,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            degenerate_len_eps,
            mode,
            scan_dup_keys,
            options,
        )?;
        let (mut uf, merged, rejected_components, round_merge_safety) = bound_merge_components(
            &mut proposed,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            &mut state.merge_ledger,
            degenerate_len_eps,
        )?;
        state.merge_safety.scanned_cells += round_merge_safety.scanned_cells;
        state.merge_safety.global_fallbacks += round_merge_safety.global_fallbacks;
        if !rejected_components.is_empty() {
            record_rejected_component_seeds(
                &rejected_components,
                edge_records,
                cells,
                cell_indices,
                vertex_keys,
                &mut state.local_rebuild_seed_pairs,
                &mut state.merge_affected_cells,
            )?;
        }
        let merged_changed = if merged == 0 {
            false
        } else {
            // Record the cells this apply may rewrite: the key-triple union
            // over every id that entered the union-find (the same coverage
            // set `apply_merges_in_place` derives). Lenient on missing keys —
            // the rebuild backend tolerates synthetic fixtures without them.
            for v in uf.touched_ids() {
                if let Some(key) = vertex_keys.get(v) {
                    state
                        .merge_affected_cells
                        .extend(key.iter().copied().filter(|&g| (g as usize) < cells.len()));
                }
            }
            match options.apply {
                ReconcileApply::Rebuild => {
                    let (new_cells, new_indices) =
                        apply_merges_rebuild(&mut uf, cells, cell_indices)?;
                    let changed = cell_spans_differ(
                        LiveCellLayout::new(cells, cell_indices),
                        LiveCellLayout::new(&new_cells, &new_indices),
                    )?;
                    *cells = new_cells;
                    *cell_indices = new_indices;
                    changed
                }
                ReconcileApply::InPlace => {
                    apply_merges_in_place(&mut uf, cells, cell_indices, vertex_keys)?
                }
            }
        };
        any |= dropped || merged_changed;
        // Converged when a round neither dropped a degenerate vertex nor
        // applied a merge. Each productive round strictly shrinks some span,
        // so this terminates well within the cap.
        if !dropped && !merged_changed {
            break;
        }
    }
    Ok(any)
}

/// Semantic per-cell sequence comparison (the rebuild backend compacts the
/// index buffer, so raw buffer equality would spin the fixpoint loop).
fn cell_spans_differ(
    old_layout: LiveCellLayout<'_, '_>,
    new_layout: LiveCellLayout<'_, '_>,
) -> Result<bool, crate::VoronoiError> {
    if old_layout.cell_count() != new_layout.cell_count() {
        return Ok(true);
    }
    for ci in 0..old_layout.cell_count() {
        let o = cell_vertex_slice_from_layout(ci as u32, old_layout)?;
        let n = cell_vertex_slice_from_layout(ci as u32, new_layout)?;
        if o != n {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Output-invariant scan: every non-boundary undirected edge must have exactly
/// two uses in opposite directions. Returns one sorted
/// `(vertex_a, vertex_b, owning_cell)` record per bad edge.
///
/// Localized to the reconciliation's touched region: reconciliation modifies only the
/// cells named by the detection records (`candidate_cells`) and the vertices
/// they share, so only those cells and their 1-ring can be incident to a
/// post-reconciliation unpaired edge. We build the edge-use map over that region, then
/// partner-verify each locally-single use against the true neighbor cell's span
/// (recovered from the endpoint keys) to reject edges whose real partner merely
/// lies outside the scanned region. This makes the scan O(defect) instead of
/// O(total edges). See `docs/performance.md#source-pinned-performance-decisions`.
///
/// Debug builds assert the localized result is identical to the global scan, so
/// any gap in the locality argument is caught immediately at zero release cost.
pub(crate) fn scan_unpaired_interior(
    layout: LiveCellLayout<'_, '_>,
    vertex_keys: VertexKeys<'_>,
    candidate_cells: &[u32],
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    let out = scan_unpaired_interior_localized(layout, vertex_keys, candidate_cells)?;
    #[cfg(debug_assertions)]
    {
        let global = scan_unpaired_interior_global(layout)?;
        // Both are sorted; compare directly.
        debug_assert_eq!(
            out, global,
            "edge-reconcile localized unpaired-scan disagrees with the global scan \
             (locality argument violated; see docs/architecture.md stitching invariant)"
        );
    }
    Ok(out)
}

fn scan_unpaired_interior_localized(
    layout: LiveCellLayout<'_, '_>,
    vertex_keys: VertexKeys<'_>,
    candidate_cells: &[u32],
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    use rustc_hash::FxHashMap as HashMap;
    // Scan region = candidate cells + their 1-ring (the cells named by the
    // generators in their vertices' keys).
    let mut region: Vec<u32> = Vec::new();
    for &c in candidate_cells {
        if (c as usize) >= layout.cell_count() {
            continue;
        }
        region.push(c);
        let span = cell_vertex_slice_from_layout(c, layout)?;
        for &v in span {
            if let Some(key) = vertex_keys.get(v) {
                for g in key {
                    if (g as usize) < layout.cell_count() {
                        region.push(g);
                    }
                }
            }
        }
    }
    region.sort_unstable();
    region.dedup();

    // value = (use count, lower->higher count, first owner)
    let mut uses: HashMap<(u32, u32), (u32, u32, u32)> = HashMap::default();
    for &ci in &region {
        let span = cell_vertex_slice_from_layout(ci, layout)?;
        let n = span.len();
        // Degenerate (< 3 vertex) cells have no well-formed edge cycle;
        // validation reports them separately.
        if n < 3 {
            continue;
        }
        for k in 0..n {
            let a = span[k];
            let b = span[if k + 1 == n { 0 } else { k + 1 }];
            let key = (a.min(b), a.max(b));
            let use_ = uses.entry(key).or_insert((0, 0, ci));
            use_.0 += 1;
            use_.1 += u32::from(a < b);
        }
    }

    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    for ((a, b), (count, forward_count, owner)) in uses {
        let mut total_count = count;
        let mut total_forward = forward_count;
        if a != b && count == 1 {
            // A single use within the localized region may have its real
            // partner just outside it. Recover that cell from the endpoint
            // keys, then include every occurrence and direction from its span.
            if let (Some(ka), Some(kb)) = (vertex_keys.get(a), vertex_keys.get(b)) {
                if let Some((g1, g2)) = key_common_pair(ka, kb) {
                    let partner = if g1 == owner {
                        Some(g2)
                    } else if g2 == owner {
                        Some(g1)
                    } else {
                        None
                    };
                    if let Some(partner) = partner.filter(|&p| p != owner) {
                        let (partner_count, partner_forward) =
                            cell_edge_uses(partner, a, b, layout)?;
                        total_count += partner_count;
                        total_forward += partner_forward;
                    }
                }
            }
        }
        if a == b || total_count != 2 || total_forward != 1 {
            out.push((a, b, owner));
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// `(use count, lower->higher count)` for edge `(a,b)` in one cell.
fn cell_edge_uses(
    cell_id: u32,
    a: u32,
    b: u32,
    layout: LiveCellLayout<'_, '_>,
) -> Result<(u32, u32), crate::VoronoiError> {
    if (cell_id as usize) >= layout.cell_count() {
        return Ok((0, 0));
    }
    let span = cell_vertex_slice_from_layout(cell_id, layout)?;
    let n = span.len();
    if n < 3 {
        return Ok((0, 0));
    }
    let mut count = 0u32;
    let mut forward = 0u32;
    for k in 0..n {
        let x = span[k];
        let y = span[if k + 1 == n { 0 } else { k + 1 }];
        if (x == a && y == b) || (x == b && y == a) {
            count += 1;
            forward += u32::from(x < y);
        }
    }
    Ok((count, forward))
}

/// Global O(total edges) reference scan — the debug differential for the
/// localized `scan_unpaired_interior`, and the whole-diagram check behind the
/// empty-records early return. Debug-only; the production path is localized.
#[cfg(debug_assertions)]
fn scan_unpaired_interior_global(
    layout: LiveCellLayout<'_, '_>,
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    use rustc_hash::FxHashMap as HashMap;
    let mut uses: HashMap<(u32, u32), (u32, u32, u32)> = HashMap::default();
    for ci in 0..layout.cell_count() {
        let span = cell_vertex_slice_from_layout(ci as u32, layout)?;
        let n = span.len();
        if n < 3 {
            continue;
        }
        for k in 0..n {
            let a = span[k];
            let b = span[if k + 1 == n { 0 } else { k + 1 }];
            let key = (a.min(b), a.max(b));
            let use_ = uses.entry(key).or_insert((0, 0, ci as u32));
            use_.0 += 1;
            use_.1 += u32::from(a < b);
        }
    }
    let mut out: Vec<(u32, u32, u32)> = uses
        .into_iter()
        .filter(|&((a, b), (count, forward, _))| a == b || count != 2 || forward != 1)
        .map(|((a, b), (_, _, owner))| (a, b, owner))
        .collect();
    out.sort_unstable();
    Ok(out)
}

/// The two generators shared by both endpoint keys — for a well-formed
/// edge these are exactly the owning cell pair.
fn key_common_pair(k1: VertexKey, k2: VertexKey) -> Option<(u32, u32)> {
    let mut common = [0u32; 3];
    let mut n = 0;
    for &g in &k1 {
        if key_contains(k2, g) && n < 3 {
            common[n] = g;
            n += 1;
        }
    }
    if n == 2 {
        Some((common[0].min(common[1]), common[0].max(common[1])))
    } else {
        None
    }
}

/// Synthesize reconciliation records from unpaired interior edges: the owning cell
/// pair recovered from the endpoint keys' shared generators, deduplicated.
fn synthesize_backstop_records(
    unpaired: &[(u32, u32, u32)],
    vertex_keys: VertexKeys<'_>,
    num_cells: usize,
) -> Vec<EdgeRecord> {
    let mut keys: Vec<u64> = unpaired
        .iter()
        .filter_map(|&(va, vb, _)| {
            let k1 = vertex_keys.get(va)?;
            let k2 = vertex_keys.get(vb)?;
            let (a, b) = key_common_pair(k1, k2)?;
            // In production every key member has a cell; tolerate synthetic
            // fixtures whose keys name nonexistent generators (mirrors the
            // out-of-range tolerance in apply_merges_in_place).
            if (a as usize) >= num_cells || (b as usize) >= num_cells {
                return None;
            }
            Some((a as u64) | ((b as u64) << 32))
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .map(|k| EdgeRecord {
            key: EdgeKey::from(k),
        })
        .collect()
}

/// Report identity for a residual unpaired edge: the endpoint keys' shared
/// generator pair when well-formed, else the owning cell twice.
fn cell_pair_for_unpaired(va: u32, vb: u32, owner: u32, vertex_keys: VertexKeys<'_>) -> (u32, u32) {
    match (vertex_keys.get(va), vertex_keys.get(vb)) {
        (Some(k1), Some(k2)) => key_common_pair(k1, k2).unwrap_or((owner, owner)),
        _ => (owner, owner),
    }
}

/// Walk the unresolved edge records and collect vertex-identity merges into
/// a union-find. Both apply backends consume the exact same merge set.
/// Union every pair of segment-endpoint vertices, across and within the
/// two sides, that lie within the degenerate length scale. Local to one
/// defective edge, so the quadratic pairing is over a handful of ids.
fn proximity_union_segments(
    seg_a: &[(u32, u32)],
    seg_b: &[(u32, u32)],
    vertices: &[Vec3],
    degenerate_len_eps_sq: f32,
    uf: &mut SparseUnionFind,
    merged: &mut usize,
) -> Result<(), crate::VoronoiError> {
    let mut ids: Vec<u32> = Vec::with_capacity((seg_a.len() + seg_b.len()) * 2);
    for &(v0, v1) in seg_a.iter().chain(seg_b.iter()) {
        ids.push(v0);
        ids.push(v1);
    }
    ids.sort_unstable();
    ids.dedup();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let d = dist_sq(vertex_pos(vertices, ids[i])?, vertex_pos(vertices, ids[j])?);
            if d <= degenerate_len_eps_sq && uf.union(ids[i], ids[j]) {
                *merged += 1;
            }
        }
    }
    Ok(())
}

/// Union all same-key vertex duplicates by a single O(V) pass over every key.
/// First-seen (lowest id, since iteration is sequential) is the representative.
fn global_dup_key_unions(
    vertex_keys: VertexKeys<'_>,
    uf: &mut SparseUnionFind,
    merged: &mut usize,
) {
    let mut first_by_key: rustc_hash::FxHashMap<VertexKey, u32> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(vertex_keys.len(), Default::default());
    vertex_keys.for_each(|i, key| match first_by_key.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(i);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
            if uf.union(*e.get(), i) {
                *merged += 1;
            }
        }
    });
}

/// Localized same-key duplicate union: a BFS over only the defect-affected
/// region instead of the O(V) global scan.
///
/// A same-key duplicate is, by the keyed-identity model, a re-emitted copy of
/// one abstract corner `[a,b,c]` (geometrically coincident — the same
/// circumcenter); the copies are split among the three cells `a`, `b`, `c` that
/// meet there. Duplicates are created by a defective edge and propagate only
/// through corners that share a cell, so every duplicated corner is connected,
/// via a path of shared cells, back to a cell named by a detection record.
///
/// The BFS therefore:
/// - seeds with the record cells (both endpoints of every edge record);
/// - scans each cell's corner vertices into a *small* `first_by_key` map keyed
///   by the local region (not all V);
/// - on a real collision (a different id with the same key — `other != v`),
///   unions the copies and marks the cell **damaged**;
/// - when a cell is damaged, enqueues its full 1-ring (every other generator
///   named in its corners' keys), so the next link of a duplicate chain is
///   reached. Self-references (the same id seen from another of its owner
///   cells) never expand, which keeps non-defective regions out of the scan.
///
/// Bounded by (duplicate cluster + its 1-ring + seed cells) = O(defect region).
/// A `#[cfg(debug_assertions)]` oracle pins the result equal to the global scan.
fn localized_dup_key_unions(
    edge_records: &[EdgeRecord],
    layout: LiveCellLayout<'_, '_>,
    vertex_keys: VertexKeys<'_>,
    uf: &mut SparseUnionFind,
    merged: &mut usize,
) -> Result<(), crate::VoronoiError> {
    use rustc_hash::{FxHashMap, FxHashSet};

    let mut first_by_key: FxHashMap<VertexKey, u32> = FxHashMap::default();
    let mut scanned: FxHashSet<u32> = FxHashSet::default();
    let mut worklist: Vec<u32> = affected_cells_from_records(edge_records);

    while let Some(cell) = worklist.pop() {
        if cell as usize >= layout.cell_count() || !scanned.insert(cell) {
            continue;
        }
        let slice = cell_vertex_slice_from_layout(cell, layout)?;
        let mut damaged = false;
        for &v in slice {
            let Some(key) = vertex_keys.get(v) else {
                continue;
            };
            match first_by_key.entry(key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(v);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    let other = *e.get();
                    // `other == v` is the same corner vertex seen again from
                    // another of its owner cells — not a duplicate, no union.
                    if other != v {
                        if uf.union(other, v) {
                            *merged += 1;
                        }
                        damaged = true;
                    }
                }
            }
        }
        if damaged {
            // Expand to this damaged cell's 1-ring: every other generator that
            // shares a corner with it may hold a further copy / chain link.
            for &v in slice {
                if let Some(key) = vertex_keys.get(v) {
                    for &g in key.iter() {
                        if g != cell && !scanned.contains(&g) {
                            worklist.push(g);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Debug oracle: assert the localized BFS unioned exactly the same same-key
/// duplicates as the global scan would — i.e. every pair the global scan would
/// merge is already connected in `uf`. Catches any gap in the connectivity
/// contract on every defect-bearing test input at zero release cost.
#[cfg(debug_assertions)]
fn assert_localized_dupscan_complete(vertex_keys: VertexKeys<'_>, uf: &mut SparseUnionFind) {
    let mut first_by_key: rustc_hash::FxHashMap<VertexKey, u32> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(vertex_keys.len(), Default::default());
    vertex_keys.for_each(|i, key| match first_by_key.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(i);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
            let other = *e.get();
            debug_assert_eq!(
                uf.find(other),
                uf.find(i),
                "edge-reconcile localized dup-scan gap: vertices {other} and {i} share a \
                 key but the localized BFS did not union them — the duplicate-connectivity \
                 contract is violated (set VORONOI_MESH_RECONCILE_GLOBAL_DUPSCAN=1 to fall back)"
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn collect_merges(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    degenerate_len_eps: f32,
    mode: MergeMode,
    scan_dup_keys: bool,
    options: ReconcileOptions,
) -> Result<(SparseUnionFind, usize), crate::VoronoiError> {
    // Sparse: only the handful of vertices named by defective edges ever
    // enter the structure, so clean and near-clean runs skip the O(V) init
    // a dense UnionFind would pay. Representative choice is identical to
    // the dense version (see SparseUnionFind docs), so output is unchanged.
    let mut uf = SparseUnionFind::new();
    let mut merged = 0usize;
    let degenerate_len_eps_sq: f32 = degenerate_len_eps * degenerate_len_eps;
    let layout = LiveCellLayout::new(cells, cell_indices);

    // Identity backstop: the keyed-identity model admits exactly one vertex
    // per key, but index propagation fails across a defective edge (the
    // mismatched endpoint's index is not forwarded), so a later cell can
    // re-create an already-emitted key — duplicate ids for one abstract
    // vertex. Downstream, cross-bin cells reached through two such edges
    // reference different copies, producing unpaired edges whose thirds
    // fully agree (no per-edge record names them). Same-key duplicates ARE
    // the same vertex by model definition: union them all up front. Gated
    // on defect runs, so clean runs never pay the O(V) scan.
    if scan_dup_keys {
        if options.force_global_dupscan {
            global_dup_key_unions(vertex_keys, &mut uf, &mut merged);
        } else {
            localized_dup_key_unions(edge_records, layout, vertex_keys, &mut uf, &mut merged)?;
            // Debug oracle: the localized BFS must union exactly the same
            // same-key duplicates as the O(V) global scan. Costs nothing in
            // release; catches any gap in the connectivity contract immediately.
            #[cfg(debug_assertions)]
            assert_localized_dupscan_complete(vertex_keys, &mut uf);
        }
    }

    // Reuse exactly two segment buffers across every record in this reconciliation
    // round. Irregular edges may expose arbitrarily many segments, so the
    // buffers retain their full contents and grow as needed rather than using
    // a fixed-size/capped representation.
    let mut seg_a = Vec::new();
    let mut seg_b = Vec::new();
    for record in edge_records {
        let (a, b) = unpack_edge(record.key.as_u64());
        edge_segments_for_neighbor_into(a, b, layout, vertex_keys, &mut seg_a)?;
        edge_segments_for_neighbor_into(b, a, layout, vertex_keys, &mut seg_b)?;
        if mode == MergeMode::ProximityOnly {
            proximity_union_segments(
                &seg_a,
                &seg_b,
                vertices,
                degenerate_len_eps_sq,
                &mut uf,
                &mut merged,
            )?;
            continue;
        }
        if seg_a.len() != 1 || seg_b.len() != 1 {
            // Irregular topology (sliver chains, overlapping defects): union
            // every pair of segment-endpoint vertices — across and within
            // the two sides — that lie within the degenerate length scale.
            // Position-based and local to the defective edge, so it stays
            // O(defect size); it collapses duplicate-position vertices with
            // distinct keys (an exact-tie corner committed under two
            // attributions) and sliver chains the per-segment logic cannot
            // pair up.
            proximity_union_segments(
                &seg_a,
                &seg_b,
                vertices,
                degenerate_len_eps_sq,
                &mut uf,
                &mut merged,
            )?;

            // Special-case: one-sided, zero-length boundary edge.
            //
            // This shows up when a cell's topology contains an epsilon edge (often from a
            // near-degenerate configuration). One cell still emits the tiny edge, but the other
            // side effectively collapses it away, so we can't find a matching segment.
            //
            // If we detect an essentially zero-length edge on the emitting side, collapse it
            // (and, if possible, merge it onto an exactly coincident vertex in the neighbor cell).
            if (seg_a.len() == 1 && seg_b.is_empty()) || (seg_b.len() == 1 && seg_a.is_empty()) {
                let (_emit_cell, other_cell, emit_seg) = if seg_a.len() == 1 {
                    (a, b, seg_a[0])
                } else {
                    (b, a, seg_b[0])
                };
                let (v0, v1) = emit_seg;
                let len_sq = dist_sq(vertex_pos(vertices, v0)?, vertex_pos(vertices, v1)?);
                if len_sq <= degenerate_len_eps_sq {
                    if uf.union(v0, v1) {
                        merged += 1;
                    }

                    // If the neighbor cell contains an exactly coincident vertex, merge onto it
                    // to improve global consistency across cells.
                    let other_cell = other_cell as usize;
                    if other_cell < cells.len() {
                        let slice = cell_vertex_slice(other_cell as u32, cells, cell_indices)?;
                        for &vi in [v0, v1].iter() {
                            let vi_pos = vertex_pos(vertices, vi)?;
                            let mut best: Option<(u32, f32)> = None;
                            for &vj in slice {
                                let d = dist_sq(vi_pos, vertex_pos(vertices, vj)?);
                                best = Some(match best {
                                    None => (vj, d),
                                    Some((best_vj, best_d)) => {
                                        if d < best_d {
                                            (vj, d)
                                        } else {
                                            (best_vj, best_d)
                                        }
                                    }
                                });
                            }
                            if let Some((vj, best_d)) = best {
                                if best_d <= degenerate_len_eps_sq && uf.union(vi, vj) {
                                    merged += 1;
                                }
                            }
                        }
                    }
                }
            }
            continue;
        }
        let (a0, a1) = seg_a[0];
        let (b0, b1) = seg_b[0];

        let share_a0 = a0 == b0 || a0 == b1;
        let share_a1 = a1 == b0 || a1 == b1;
        if share_a0 && share_a1 {
            continue;
        }
        if share_a0 || share_a1 {
            let (keep_a, keep_b) = if a0 == b0 {
                (a1, b1)
            } else if a0 == b1 {
                (a1, b0)
            } else if a1 == b0 {
                (a0, b1)
            } else {
                (a0, b0)
            };
            let distance_sq = dist_sq(vertex_pos(vertices, keep_a)?, vertex_pos(vertices, keep_b)?);
            if distance_sq <= degenerate_len_eps_sq && uf.union(keep_a, keep_b) {
                merged += 1;
            }
            continue;
        }

        let d00a = dist_sq(vertex_pos(vertices, a0)?, vertex_pos(vertices, b0)?);
        let d00b = dist_sq(vertex_pos(vertices, a1)?, vertex_pos(vertices, b1)?);
        let d01a = dist_sq(vertex_pos(vertices, a0)?, vertex_pos(vertices, b1)?);
        let d01b = dist_sq(vertex_pos(vertices, a1)?, vertex_pos(vertices, b0)?);
        let d00 = d00a + d00b;
        let d01 = d01a + d01b;
        if d00 <= d01 {
            if d00a <= degenerate_len_eps_sq && d00b <= degenerate_len_eps_sq {
                if uf.union(a0, b0) {
                    merged += 1;
                }
                if uf.union(a1, b1) {
                    merged += 1;
                }
            }
        } else if d01a <= degenerate_len_eps_sq && d01b <= degenerate_len_eps_sq {
            if uf.union(a0, b1) {
                merged += 1;
            }
            if uf.union(a1, b0) {
                merged += 1;
            }
        }
    }

    Ok((uf, merged))
}

/// Convert the threshold-graph proposals from one round into accepted merge
/// components. A component is accepted only when every pair of original
/// members represented across all prior rounds is within `eps`; otherwise the
/// entire component is rejected transactionally and handed to Hull3d by the
/// caller. Distances are accumulated in f64 over the stored f32 coordinates so
/// the policy is a defensible bound rather than another f32 rounding layer.
struct MergeCandidate {
    representative: u32,
    current_ids: Vec<u32>,
    expanded: Vec<u32>,
}

/// Return the complete cell cover for the proposed components, or `None` when
/// provenance is incomplete and the caller must scan globally.
///
/// Initially, a vertex id can be referenced only by the three generator cells
/// in its key. After an accepted merge, the surviving representative can also
/// be referenced by cells from the retired ids' keys. `expanded` is the
/// persistent ledger of precisely those original ids, so the union of their
/// key triples remains a complete cover across reconciliation rounds and apply modes.
fn merge_safety_cell_cover(
    candidates: &[MergeCandidate],
    vertex_keys: VertexKeys<'_>,
    cell_count: usize,
) -> Option<Vec<u32>> {
    if cell_count == 0 {
        return Some(Vec::new());
    }

    let mut cover = Vec::new();
    for candidate in candidates {
        for &id in &candidate.expanded {
            let key = vertex_keys.get(id)?;
            for owner in key {
                if owner as usize >= cell_count {
                    return None;
                }
                cover.push(owner);
            }
        }
    }
    cover.sort_unstable();
    cover.dedup();
    Some(cover)
}

/// Simulate all proposed components jointly over either a certified local
/// cover or the full diagram. Cells must be visited in ascending order to
/// retain the existing transaction semantics: once a component is rejected,
/// later faces are evaluated as if that component will not be applied.
fn reject_face_unsafe_components(
    candidates: &[MergeCandidate],
    candidate_for_id: &rustc_hash::FxHashMap<u32, usize>,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    cover: Option<&[u32]>,
) -> Result<Vec<bool>, crate::VoronoiError> {
    let mut rejected = vec![false; candidates.len()];
    let mut normalized = Vec::new();
    let mut touched_candidates = Vec::new();

    let mut visit = |cell_idx: u32| -> Result<(), crate::VoronoiError> {
        let span = cell_vertex_slice(cell_idx, cells, cell_indices)?;
        normalized.clear();
        normalized.reserve(span.len());
        touched_candidates.clear();
        for &id in span {
            let mapped = match candidate_for_id.get(&id).copied() {
                Some(idx) if !rejected[idx] => {
                    if !touched_candidates.contains(&idx) {
                        touched_candidates.push(idx);
                    }
                    candidates[idx].representative
                }
                _ => id,
            };
            if normalized.last().copied() != Some(mapped) {
                normalized.push(mapped);
            }
        }
        if normalized.len() > 1 && normalized[0] == *normalized.last().unwrap() {
            normalized.pop();
        }
        let has_duplicate =
            (0..normalized.len()).any(|i| normalized[(i + 1)..].contains(&normalized[i]));
        if normalized.len() < 3 || has_duplicate {
            for &idx in &touched_candidates {
                rejected[idx] = true;
            }
        }
        Ok(())
    };

    match cover {
        Some(cell_ids) => {
            for &cell_idx in cell_ids {
                visit(cell_idx)?;
            }
        }
        None => {
            for cell_idx in 0..cells.len() {
                visit(cell_idx as u32)?;
            }
        }
    }

    Ok(rejected)
}

fn bound_merge_components(
    proposed: &mut SparseUnionFind,
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    ledger: &mut MergeLedger,
    eps: f32,
) -> Result<
    (
        SparseUnionFind,
        usize,
        Vec<RejectedMergeComponent>,
        MergeSafetyStats,
    ),
    crate::VoronoiError,
> {
    let touched = proposed.touched_ids();
    let mut groups = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for id in touched {
        groups.entry(proposed.find(id)).or_default().push(id);
    }

    let eps_sq = f64::from(eps) * f64::from(eps);
    let mut rejected_components = Vec::new();
    let mut candidates = Vec::new();

    for (representative, current_ids) in groups {
        let expanded = ledger.expanded_members(&current_ids);
        let mut within_diameter = true;
        'pairs: for i in 0..expanded.len() {
            let a = vertex_pos(vertices, expanded[i])?;
            for &b_id in &expanded[(i + 1)..] {
                let b = vertex_pos(vertices, b_id)?;
                if dist_sq_f64(a, b) > eps_sq {
                    within_diameter = false;
                    break 'pairs;
                }
            }
        }

        if !within_diameter {
            rejected_components.push(RejectedMergeComponent {
                current_ids,
                member_ids: expanded,
            });
            continue;
        }

        candidates.push(MergeCandidate {
            representative,
            current_ids,
            expanded,
        });
    }

    // Reconciliation may collapse a diameter-bounded triangulation diagonal while
    // reconciling an observed topology defect. This is load-bearing for exact
    // degree-4+ grids, where Hull3d is not a scalable substitute. The edit
    // must still preserve every cell and avoid a non-simple face.
    let mut candidate_for_id = rustc_hash::FxHashMap::<u32, usize>::default();
    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        for &id in &candidate.current_ids {
            candidate_for_id.insert(id, candidate_idx);
        }
    }

    // Consider all components together. Multiple individually safe merges in
    // one face can jointly erase or fold it; decline every component touching
    // such a face under Preserve. The expanded provenance ledger gives a
    // complete local cover. Missing or invalid provenance falls back to the
    // previous global scan.
    let cover = merge_safety_cell_cover(&candidates, vertex_keys, cells.len());
    let rejected = reject_face_unsafe_components(
        &candidates,
        &candidate_for_id,
        cells,
        cell_indices,
        cover.as_deref(),
    )?;
    let merge_safety = MergeSafetyStats {
        scanned_cells: cover.as_ref().map_or(cells.len(), Vec::len),
        global_fallbacks: usize::from(cover.is_none()),
    };

    // Continuously audit the provenance proof in checked/debug builds. The
    // oracle starts from the same candidate state and visits every cell in the
    // original order; any missed reference or joint-face interaction is a
    // localization bug, not an acceptable output difference.
    #[cfg(debug_assertions)]
    if cover.is_some() {
        let global_rejected = reject_face_unsafe_components(
            &candidates,
            &candidate_for_id,
            cells,
            cell_indices,
            None,
        )?;
        debug_assert_eq!(
            rejected, global_rejected,
            "localized merge-safety cover disagreed with global oracle"
        );
    }

    let mut accepted = SparseUnionFind::new();
    let mut accepted_merges = 0usize;
    for (candidate, rejected) in candidates.into_iter().zip(rejected) {
        if rejected {
            rejected_components.push(RejectedMergeComponent {
                current_ids: candidate.current_ids,
                member_ids: candidate.expanded,
            });
            continue;
        }

        for &id in &candidate.current_ids {
            if id != candidate.representative && accepted.union(candidate.representative, id) {
                accepted_merges += 1;
            }
        }
        ledger.commit(
            candidate.representative,
            &candidate.current_ids,
            candidate.expanded,
        );
    }

    Ok((accepted, accepted_merges, rejected_components, merge_safety))
}

#[allow(clippy::too_many_arguments)]
fn record_rejected_component_seeds(
    rejected: &[RejectedMergeComponent],
    edge_records: &[EdgeRecord],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    local_rebuild_seed_pairs: &mut Vec<(u32, u32)>,
    merge_affected_cells: &mut Vec<u32>,
) -> Result<(), crate::VoronoiError> {
    use rustc_hash::FxHashSet;

    let mut seg_a = Vec::new();
    let mut seg_b = Vec::new();
    let layout = LiveCellLayout::new(cells, cell_indices);
    for component in rejected {
        let current_ids: FxHashSet<u32> = component.current_ids.iter().copied().collect();
        for &id in &component.member_ids {
            if let Some(key) = vertex_keys.get(id) {
                merge_affected_cells
                    .extend(key.iter().copied().filter(|&g| (g as usize) < cells.len()));
            }
        }

        let seeds_before = local_rebuild_seed_pairs.len();
        for record in edge_records {
            let (a, b) = unpack_edge(record.key.as_u64());
            edge_segments_for_neighbor_into(a, b, layout, vertex_keys, &mut seg_a)?;
            edge_segments_for_neighbor_into(b, a, layout, vertex_keys, &mut seg_b)?;
            let touches_rejected = seg_a
                .iter()
                .chain(&seg_b)
                .any(|&(v0, v1)| current_ids.contains(&v0) || current_ids.contains(&v1));
            if touches_rejected {
                local_rebuild_seed_pairs.push((a.min(b), a.max(b)));
            }
        }

        // Same-key duplicate proposals can be discovered through the localized
        // identity scan without appearing on the recorded edge's current
        // segment. Seed those components from their own generator keys.
        if local_rebuild_seed_pairs.len() == seeds_before {
            for &id in &component.member_ids {
                let Some(key) = vertex_keys.get(id) else {
                    continue;
                };
                for i in 0..key.len() {
                    for j in (i + 1)..key.len() {
                        let (a, b) = (key[i].min(key[j]), key[i].max(key[j]));
                        if a != b && (b as usize) < cells.len() {
                            local_rebuild_seed_pairs.push((a, b));
                        }
                    }
                }
            }
        }
        if local_rebuild_seed_pairs.len() == seeds_before {
            if let Some(record) = edge_records.first() {
                let (a, b) = unpack_edge(record.key.as_u64());
                local_rebuild_seed_pairs.push((a.min(b), a.max(b)));
            }
        }
    }

    Ok(())
}

/// Original full-rewrite apply: rebuild every cell span into fresh compacted
/// arrays. O(diagram); retained as the differential oracle for `InPlace`.
fn apply_merges_rebuild(
    uf: &mut SparseUnionFind,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Result<ReconciledCells, crate::VoronoiError> {
    let mut new_cells: Vec<VoronoiCell> = Vec::with_capacity(cells.len());
    let mut new_indices: Vec<u32> = Vec::with_capacity(cell_indices.len());

    for (cell_idx, cell) in cells.iter().enumerate() {
        let base = new_indices.len();
        let mut seen: Vec<u32> = Vec::with_capacity(cell.vertex_count());
        for &vi in cell_vertex_slice(cell_idx as u32, cells, cell_indices)? {
            let rep = uf.find(vi);
            if !seen.contains(&rep) {
                seen.push(rep);
                new_indices.push(rep);
            }
        }
        let count = new_indices.len() - base;
        let count_u16 = u16::try_from(count).map_err(|_| {
            crate::VoronoiError::RepresentationLimit(
                "reconciled cell vertex count exceeds u16 capacity".to_string(),
            )
        })?;
        let start_u32 = u32::try_from(base).map_err(|_| {
            crate::VoronoiError::RepresentationLimit(
                "reconciled cell index buffer exceeds u32 capacity".to_string(),
            )
        })?;
        new_cells.push(VoronoiCell::new(start_u32, count_u16));
    }

    Ok((new_cells, new_indices))
}

/// O(defects) apply: patch only the cells that can reference a merged
/// vertex, in place. A vertex keyed `(A, B, T)` appears only in the
/// boundaries of cells A, B and T, so the union of key triplets over every
/// id that entered the union-find covers all referencing cells. Each
/// affected span is rewritten in place (ids replaced by representatives,
/// duplicates dropped keeping first occurrence — the same per-cell sequence
/// the rebuild produces) and its count shrunk; stale tail slots in the
/// index buffer are never read.
fn apply_merges_in_place(
    uf: &mut SparseUnionFind,
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    vertex_keys: VertexKeys<'_>,
) -> Result<bool, crate::VoronoiError> {
    let mut changed = false;
    let mut affected: Vec<u32> = Vec::new();
    for v in uf.touched_ids() {
        let key = vertex_keys.get(v).ok_or_else(|| {
            reconcile_state_error(format!(
                "edge reconciliation merged vertex id {} out of range for vertex_keys len {}",
                v,
                vertex_keys.len()
            ))
        })?;
        affected.extend_from_slice(&key);
    }
    affected.sort_unstable();
    affected.dedup();
    // In production every triplet member is a generator index and thus has
    // a cell; tolerate out-of-range members (synthetic test fixtures) — the
    // debug scan below still verifies no reference was missed.
    affected.retain(|&c| (c as usize) < cells.len());

    for &cell_idx in &affected {
        let cell_idx_usize = cell_idx as usize;
        let cell = cells[cell_idx_usize];
        let start = cell.vertex_start();
        let count = cell.vertex_count();
        let end = start + count;
        if end > cell_indices.len() {
            return Err(reconcile_state_error(format!(
                "edge reconciliation cell {cell_idx_usize} span [{start}..{end}) exceeds cell \
                 index buffer len {}",
                cell_indices.len()
            )));
        }
        let span = &mut cell_indices[start..end];
        // In-place rewrite: w trails r, so reads are never clobbered; kept
        // slots still get their representative written (id may change
        // without any duplicate forming).
        let mut w = 0usize;
        for r in 0..count {
            let orig = span[r];
            let rep = uf.find(orig);
            if rep != orig {
                changed = true;
            }
            if !span[..w].contains(&rep) {
                span[w] = rep;
                w += 1;
            } else {
                changed = true;
            }
        }
        if w != count {
            cells[cell_idx_usize] = VoronoiCell::new(start as u32, w as u16);
        }
    }

    // The triplet-coverage argument above is a construction invariant, not
    // a local check — verify it exhaustively in debug builds: no cell may
    // still reference a merged-away id.
    #[cfg(debug_assertions)]
    for (ci, cell) in cells.iter().enumerate() {
        let span = &cell_indices[cell.vertex_start()..cell.vertex_start() + cell.vertex_count()];
        for &vi in span {
            debug_assert_eq!(
                uf.find(vi),
                vi,
                "cell {ci} still references non-representative vertex {vi} after in-place reconciliation"
            );
        }
    }

    Ok(changed)
}

#[cfg(test)]
mod tests;
