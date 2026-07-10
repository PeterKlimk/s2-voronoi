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

use super::live_dedup::{EdgeKey, EdgeRecord, ShardedVertexKeys};
use crate::diagram::VoronoiCell;
use crate::knn_clipping::cell_build::VertexKey;

/// Read-only view of vertex keys passed to reconciliation. `Flat` backs the
/// unit tests (and any caller holding a contiguous array); `Sharded` is the
/// production path, looking keys up per-shard without a global concatenation.
/// `Copy` so it threads through the repair helpers by value.
#[derive(Clone, Copy)]
pub(crate) enum VertexKeys<'a> {
    // Used by the unit tests (and any caller holding a contiguous array).
    #[cfg_attr(not(test), allow(dead_code))]
    Flat(&'a [VertexKey]),
    Sharded(&'a ShardedVertexKeys),
}

impl VertexKeys<'_> {
    #[inline]
    fn get(&self, vid: u32) -> Option<VertexKey> {
        match self {
            VertexKeys::Flat(s) => s.get(vid as usize).copied(),
            VertexKeys::Sharded(s) => s.get(vid),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            VertexKeys::Flat(s) => s.len(),
            VertexKeys::Sharded(s) => s.len(),
        }
    }

    /// Visit every `(vid, key)` in global slot order. Only the global-scan
    /// escape path and the debug oracle need this; the localized BFS does not.
    fn for_each(&self, mut f: impl FnMut(u32, VertexKey)) {
        match self {
            VertexKeys::Flat(s) => {
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

/// Error for post-repair residuals on the plain compute paths: a non-empty
/// residual list means the output is provably not a valid subdivision (some
/// interior edge stays unpaired), and those paths have no report channel to
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
        "edge reconciliation left {} unpaired interior edge(s) — output is not a valid \
         subdivision: {}{more}. Use compute_with_report to inspect, or report this input.",
        pairs.len(),
        shown.join(" ")
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
    let cell_idx_usize = cell_idx as usize;
    if cell_idx_usize >= cells.len() {
        return Err(reconcile_state_error(format!(
            "edge reconciliation referenced out-of-range cell {} (cells={})",
            cell_idx_usize,
            cells.len()
        )));
    }
    let cell = &cells[cell_idx_usize];
    let start = cell.vertex_start();
    let end = start + cell.vertex_count();
    if end > cell_indices.len() {
        return Err(reconcile_state_error(format!(
            "edge reconciliation cell {} span [{}..{}) exceeds cell index buffer len {}",
            cell_idx_usize,
            start,
            end,
            cell_indices.len()
        )));
    }
    Ok(&cell_indices[start..end])
}

pub(crate) fn edge_segments_for_neighbor(
    cell_idx: u32,
    neighbor: u32,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    let slice = cell_vertex_slice(cell_idx, cells, cell_indices)?;
    let n = slice.len();
    if n < 2 {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
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
    Ok(out)
}

fn dist_sq<P: crate::knn_clipping::live_dedup::VertexPosition>(a: P, b: P) -> f32 {
    a.dist_sq(b)
}

fn vertex_pos<P: crate::knn_clipping::live_dedup::VertexPosition>(
    vertices: &[P],
    vertex_id: u32,
) -> Result<P, crate::VoronoiError> {
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

/// Outcome of [`reconcile_unresolved_edges`].
///
/// `merge_affected_cells` exists for the repair's localized residual scan:
/// identity merges remap vertex references in place, so a cell in this set can
/// reference a surviving vertex whose key triple does not name it — the one
/// production violation of the key-ownership invariant ("a vertex keyed
/// `(a, b, c)` is referenced only by cells `a`, `b`, `c`"). Consumers relying
/// on that invariant to localize must treat these cells as always in scope.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ReconcileResult {
    /// Surviving unpaired interior edges, as owning cell pairs for the
    /// caller's report / repair trigger.
    pub residual_pairs: Vec<(u32, u32)>,
    /// Cells whose spans were rewritten by identity merges (sorted, deduped):
    /// the union of key triples over every vertex id that entered a merge.
    pub merge_affected_cells: Vec<u32>,
}

/// How reconciliation merges are applied to the cell arrays.
///
/// `InPlace` is the production default: only cells naming a merged vertex
/// are touched (found via the vertex-key triplets), spans shrink in place,
/// and the index buffer keeps stale tail slots (never read — cells are
/// `(start, count)` spans). O(defects) instead of O(diagram); measured
/// ~382ms saved at 2M single-threaded on a defect-bearing run.
///
/// `Rebuild` is the original full rewrite, retained as the differential
/// oracle: the two backends must produce identical per-cell vertex
/// sequences (pinned by the unit tests below and the full-pipeline
/// differential in tests/edge_repair_net.rs via `VORONOI_MESH_EDGE_REPAIR_REBUILD`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RepairApply {
    InPlace,
    Rebuild,
}

/// Production apply mode: in-place unless `VORONOI_MESH_EDGE_REPAIR_REBUILD=1`
/// selects the rebuild oracle (diagnostic / differential-testing knob,
/// read once per compute on the cold path).
pub(crate) fn repair_apply_from_env() -> RepairApply {
    match std::env::var("VORONOI_MESH_EDGE_REPAIR_REBUILD") {
        Ok(v) if v == "1" => RepairApply::Rebuild,
        _ => RepairApply::InPlace,
    }
}

/// Hard cap on repair rounds; each productive round strictly shrinks some
/// cell span, so termination is structural — the cap is a backstop.
const MAX_REPAIR_ROUNDS: usize = 8;

/// How a repair pass interprets its records when pairing endpoints.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MergeMode {
    /// Bookkeeping-driven records (live-dedup detection): full pairing
    /// semantics, including the forced nearest-endpoint pairing for
    /// 1-1 segment mismatches.
    Primary,
    /// Output-invariant backstop records (synthesized from unpaired
    /// interior edges): eps-bounded proximity unions only — never
    /// force-merge distant vertices on synthesized evidence.
    ProximityOnly,
}

/// Reconcile unresolved shared-edge mismatches by merging vertex
/// identities, patching `cells` / `cell_indices` via the chosen backend.
///
/// Runs the bookkeeping-driven repair to a fixpoint (merges can expose
/// newly pairable states), then checks the output invariant directly:
/// every interior edge must be used by exactly two cells. Unpaired
/// findings synthesize an eps-bounded backstop pass (the owning cell pair
/// is recovered from the endpoint keys' shared generators); whatever
/// survives is returned as cell pairs for the caller's report rather than
/// force-merged. Returns an empty vec on clean runs (no records) without
/// touching anything — the scans are paid only on defect runs.
#[allow(clippy::too_many_arguments)] // geometry-parameterized repair seam
pub(crate) fn reconcile_unresolved_edges<P: crate::knn_clipping::live_dedup::VertexPosition>(
    edge_records: &[EdgeRecord],
    vertices: &[P],
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: VertexKeys<'_>,
    // Degenerate-length threshold in the caller's coordinate space (chord
    // units on the sphere, normalized rect units on the plane); each
    // geometry owns and justifies its constant.
    degenerate_len_eps: f32,
    apply: RepairApply,
    // Geometry's boundary classification: true when a single-use edge
    // between these vertex ids is legitimate (plane rect walls). The
    // sphere and the periodic plane have no boundary.
    is_boundary_edge: impl Fn(u32, u32) -> bool,
) -> Result<ReconcileResult, crate::VoronoiError> {
    if edge_records.is_empty() {
        // Production fast path: with no detected mismatch there is nothing to
        // repair, and the O(total cell indices) output-invariant scan is
        // skipped — avoiding it on clean runs is the whole point of this
        // early return. Soundness rests on a detection-completeness claim:
        // every unpaired interior edge produces >= 1 detection record, so an
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
            let unpaired = scan_unpaired_interior_global(cells, cell_indices, &is_boundary_edge)?;
            assert!(
                unpaired.is_empty(),
                "edge-reconcile early-return invariant violated: {} unpaired interior \
                 edge(s) with ZERO detection records — a defect escaped detection \
                 (see docs/architecture.md stitching invariant)",
                unpaired.len()
            );
        }
        return Ok(ReconcileResult::default());
    }
    let mut merge_affected_cells: Vec<u32> = Vec::new();
    let done = |residual_pairs: Vec<(u32, u32)>, mut affected: Vec<u32>| {
        affected.sort_unstable();
        affected.dedup();
        ReconcileResult {
            residual_pairs,
            merge_affected_cells: affected,
        }
    };
    let primary_candidates = affected_cells_from_records(edge_records);
    run_repair_rounds(
        edge_records,
        vertices,
        cells,
        cell_indices,
        vertex_keys,
        degenerate_len_eps,
        apply,
        MergeMode::Primary,
        &mut merge_affected_cells,
    )?;

    let unpaired = scan_unpaired_interior(
        cells,
        cell_indices,
        vertex_keys,
        &primary_candidates,
        &is_boundary_edge,
    )?;
    if unpaired.is_empty() {
        return Ok(done(Vec::new(), merge_affected_cells));
    }
    let synth = synthesize_backstop_records(&unpaired, vertex_keys, cells.len());
    if !synth.is_empty() {
        run_repair_rounds(
            &synth,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            degenerate_len_eps,
            apply,
            MergeMode::ProximityOnly,
            &mut merge_affected_cells,
        )?;
    }
    // Residual scan covers both passes' touched regions.
    let mut residual_candidates = primary_candidates;
    residual_candidates.extend(affected_cells_from_records(&synth));
    residual_candidates.sort_unstable();
    residual_candidates.dedup();
    let residual = scan_unpaired_interior(
        cells,
        cell_indices,
        vertex_keys,
        &residual_candidates,
        &is_boundary_edge,
    )?;
    Ok(done(
        residual
            .iter()
            .map(|&(va, vb, owner)| cell_pair_for_unpaired(va, vb, owner, vertex_keys))
            .collect(),
        merge_affected_cells,
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
/// cells a repair round can legitimately need to touch. Sorted + deduped.
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
/// edge, so the records' cells cover them all. This keeps a repair round
/// O(defect size) instead of O(total vertices) per round — the latter made a
/// 3-defect run cost seconds at 2.5M. Debug
/// builds assert the coverage via `assert_candidate_covers_droppable`.
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
fn run_repair_rounds<P: crate::knn_clipping::live_dedup::VertexPosition>(
    edge_records: &[EdgeRecord],
    vertices: &[P],
    cells: &mut Vec<VoronoiCell>,
    cell_indices: &mut Vec<u32>,
    vertex_keys: VertexKeys<'_>,
    degenerate_len_eps: f32,
    apply: RepairApply,
    mode: MergeMode,
    // Accumulates the cells whose spans a merge apply may rewrite (see
    // `ReconcileResult::merge_affected_cells`); the caller sorts/dedups.
    merge_affected_cells: &mut Vec<u32>,
) -> Result<bool, crate::VoronoiError> {
    let mut any = false;
    // The only cells a round can need to touch are those named by the records.
    // Computed once; repair rounds only remove vertices, so this set is a valid
    // (shrinking) cover for every round, not just the first.
    let candidate_cells = affected_cells_from_records(edge_records);
    #[cfg(debug_assertions)]
    assert_candidate_covers_droppable(cells, cell_indices, vertex_keys, &candidate_cells);
    for round in 0..MAX_REPAIR_ROUNDS {
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
        let (mut uf, merged) = collect_merges(
            edge_records,
            vertices,
            cells,
            cell_indices,
            vertex_keys,
            degenerate_len_eps,
            mode,
            scan_dup_keys,
        )?;
        let merged_changed = if merged == 0 {
            false
        } else {
            // Record the cells this apply may rewrite: the key-triple union
            // over every id that entered the union-find (the same coverage
            // set `apply_merges_in_place` derives). Lenient on missing keys —
            // the rebuild backend tolerates synthetic fixtures without them.
            for v in uf.touched_ids() {
                if let Some(key) = vertex_keys.get(v) {
                    merge_affected_cells
                        .extend(key.iter().copied().filter(|&g| (g as usize) < cells.len()));
                }
            }
            match apply {
                RepairApply::Rebuild => {
                    let (new_cells, new_indices) =
                        apply_merges_rebuild(&mut uf, cells, cell_indices)?;
                    let changed = cell_spans_differ(cells, cell_indices, &new_cells, &new_indices)?;
                    *cells = new_cells;
                    *cell_indices = new_indices;
                    changed
                }
                RepairApply::InPlace => {
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
    old_cells: &[VoronoiCell],
    old_indices: &[u32],
    new_cells: &[VoronoiCell],
    new_indices: &[u32],
) -> Result<bool, crate::VoronoiError> {
    if old_cells.len() != new_cells.len() {
        return Ok(true);
    }
    for ci in 0..old_cells.len() {
        let o = cell_vertex_slice(ci as u32, old_cells, old_indices)?;
        let n = cell_vertex_slice(ci as u32, new_cells, new_indices)?;
        if o != n {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Output-invariant scan: interior undirected edges used by exactly one cell
/// (and not a legitimate boundary). Returns (vertex_a, vertex_b, owning cell),
/// sorted.
///
/// Localized to the repair's touched region: reconciliation modifies only the
/// cells named by the detection records (`candidate_cells`) and the vertices
/// they share, so only those cells and their 1-ring can be incident to a
/// post-repair unpaired edge. We build the edge-use map over that region, then
/// partner-verify each singleton against the true neighbor cell's span
/// (recovered from the endpoint keys) to reject edges whose real partner merely
/// lies outside the scanned region. This makes the scan O(defect) instead of
/// O(total edges) — the global scan cost ~17 s on a 2.5M run with only 3
/// defects.
///
/// Debug builds assert the localized result is identical to the global scan, so
/// any gap in the locality argument is caught immediately at zero release cost.
pub(crate) fn scan_unpaired_interior(
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    candidate_cells: &[u32],
    is_boundary_edge: &impl Fn(u32, u32) -> bool,
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    let out = scan_unpaired_interior_localized(
        cells,
        cell_indices,
        vertex_keys,
        candidate_cells,
        is_boundary_edge,
    )?;
    #[cfg(debug_assertions)]
    {
        let global = scan_unpaired_interior_global(cells, cell_indices, is_boundary_edge)?;
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
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    candidate_cells: &[u32],
    is_boundary_edge: &impl Fn(u32, u32) -> bool,
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    use rustc_hash::FxHashMap as HashMap;
    // Scan region = candidate cells + their 1-ring (the cells named by the
    // generators in their vertices' keys).
    let mut region: Vec<u32> = Vec::new();
    for &c in candidate_cells {
        if (c as usize) >= cells.len() {
            continue;
        }
        region.push(c);
        let span = cell_vertex_slice(c, cells, cell_indices)?;
        for &v in span {
            if let Some(key) = vertex_keys.get(v) {
                for g in key {
                    if (g as usize) < cells.len() {
                        region.push(g);
                    }
                }
            }
        }
    }
    region.sort_unstable();
    region.dedup();

    let mut uses: HashMap<(u32, u32), (u32, u32)> = HashMap::default();
    for &ci in &region {
        let span = cell_vertex_slice(ci, cells, cell_indices)?;
        let n = span.len();
        // Degenerate (< 3 vertex) cells have no well-formed edge cycle;
        // validation reports them separately.
        if n < 3 {
            continue;
        }
        for k in 0..n {
            let a = span[k];
            let b = span[if k + 1 == n { 0 } else { k + 1 }];
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            uses.entry(key).or_insert((0, ci)).0 += 1;
        }
    }

    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    for ((a, b), (count, owner)) in uses {
        if count != 1 || is_boundary_edge(a, b) {
            continue;
        }
        // Partner-verify: a singleton within the scanned region is genuinely
        // unpaired only if the edge's *other* cell does not carry it. Recover
        // the true cell pair from the endpoint keys; if the partner (the cell
        // of the pair that is not `owner`) carries this edge, it was paired all
        // along and simply lay outside the scanned region.
        if let (Some(ka), Some(kb)) = (vertex_keys.get(a), vertex_keys.get(b)) {
            if let Some((g1, g2)) = key_common_pair(ka, kb) {
                let partner = if g1 == owner { g2 } else { g1 };
                if partner != owner && cell_has_edge(partner, a, b, cells, cell_indices)? {
                    continue;
                }
            }
        }
        out.push((a, b, owner));
    }
    out.sort_unstable();
    Ok(out)
}

/// Whether `cell_id`'s boundary cycle contains the undirected edge (a, b).
fn cell_has_edge(
    cell_id: u32,
    a: u32,
    b: u32,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
) -> Result<bool, crate::VoronoiError> {
    if (cell_id as usize) >= cells.len() {
        return Ok(false);
    }
    let span = cell_vertex_slice(cell_id, cells, cell_indices)?;
    let n = span.len();
    if n < 3 {
        return Ok(false);
    }
    for k in 0..n {
        let x = span[k];
        let y = span[if k + 1 == n { 0 } else { k + 1 }];
        if (x == a && y == b) || (x == b && y == a) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Global O(total edges) reference scan — the debug differential for the
/// localized `scan_unpaired_interior`, and the whole-diagram check behind the
/// empty-records early return. Debug-only; the production path is localized.
#[cfg(debug_assertions)]
fn scan_unpaired_interior_global(
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    is_boundary_edge: &impl Fn(u32, u32) -> bool,
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    use rustc_hash::FxHashMap as HashMap;
    let mut uses: HashMap<(u32, u32), (u32, u32)> = HashMap::default();
    for ci in 0..cells.len() {
        let span = cell_vertex_slice(ci as u32, cells, cell_indices)?;
        let n = span.len();
        if n < 3 {
            continue;
        }
        for k in 0..n {
            let a = span[k];
            let b = span[if k + 1 == n { 0 } else { k + 1 }];
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            uses.entry(key).or_insert((0, ci as u32)).0 += 1;
        }
    }
    let mut out: Vec<(u32, u32, u32)> = uses
        .into_iter()
        .filter(|&((a, b), (count, _))| count == 1 && !is_boundary_edge(a, b))
        .map(|((a, b), (_, owner))| (a, b, owner))
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

/// Synthesize repair records from unpaired interior edges: the owning cell
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
fn proximity_union_segments<P: crate::knn_clipping::live_dedup::VertexPosition>(
    seg_a: &[(u32, u32)],
    seg_b: &[(u32, u32)],
    vertices: &[P],
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

/// Escape hatch: force the O(V) global same-key duplicate scan instead of the
/// localized BFS. Diagnostic / differential safety valve, read once on the cold
/// defect path.
fn dupscan_force_global() -> bool {
    matches!(std::env::var("VORONOI_MESH_EDGE_REPAIR_GLOBAL_DUPSCAN"), Ok(v) if v == "1")
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
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    uf: &mut SparseUnionFind,
    merged: &mut usize,
) -> Result<(), crate::VoronoiError> {
    use rustc_hash::{FxHashMap, FxHashSet};

    let mut first_by_key: FxHashMap<VertexKey, u32> = FxHashMap::default();
    let mut scanned: FxHashSet<u32> = FxHashSet::default();
    let mut worklist: Vec<u32> = affected_cells_from_records(edge_records);

    while let Some(cell) = worklist.pop() {
        if cell as usize >= cells.len() || !scanned.insert(cell) {
            continue;
        }
        let slice = cell_vertex_slice(cell, cells, cell_indices)?;
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
                 contract is violated (set VORONOI_MESH_EDGE_REPAIR_GLOBAL_DUPSCAN=1 to fall back)"
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn collect_merges<P: crate::knn_clipping::live_dedup::VertexPosition>(
    edge_records: &[EdgeRecord],
    vertices: &[P],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: VertexKeys<'_>,
    degenerate_len_eps: f32,
    mode: MergeMode,
    scan_dup_keys: bool,
) -> Result<(SparseUnionFind, usize), crate::VoronoiError> {
    // Sparse: only the handful of vertices named by defective edges ever
    // enter the structure, so clean and near-clean runs skip the O(V) init
    // a dense UnionFind would pay. Representative choice is identical to
    // the dense version (see SparseUnionFind docs), so output is unchanged.
    let mut uf = SparseUnionFind::new();
    let mut merged = 0usize;
    let degenerate_len_eps_sq: f32 = degenerate_len_eps * degenerate_len_eps;

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
        if dupscan_force_global() {
            global_dup_key_unions(vertex_keys, &mut uf, &mut merged);
        } else {
            localized_dup_key_unions(
                edge_records,
                cells,
                cell_indices,
                vertex_keys,
                &mut uf,
                &mut merged,
            )?;
            // Debug oracle: the localized BFS must union exactly the same
            // same-key duplicates as the O(V) global scan. Costs nothing in
            // release; catches any gap in the connectivity contract immediately.
            #[cfg(debug_assertions)]
            assert_localized_dupscan_complete(vertex_keys, &mut uf);
        }
    }

    for record in edge_records {
        let (a, b) = unpack_edge(record.key.as_u64());
        let seg_a = edge_segments_for_neighbor(a, b, cells, cell_indices, vertex_keys)?;
        let seg_b = edge_segments_for_neighbor(b, a, cells, cell_indices, vertex_keys)?;
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
            if uf.union(keep_a, keep_b) {
                merged += 1;
            }
            continue;
        }

        let d00 = dist_sq(vertex_pos(vertices, a0)?, vertex_pos(vertices, b0)?)
            + dist_sq(vertex_pos(vertices, a1)?, vertex_pos(vertices, b1)?);
        let d01 = dist_sq(vertex_pos(vertices, a0)?, vertex_pos(vertices, b1)?)
            + dist_sq(vertex_pos(vertices, a1)?, vertex_pos(vertices, b0)?);
        if d00 <= d01 {
            if uf.union(a0, b0) {
                merged += 1;
            }
            if uf.union(a1, b1) {
                merged += 1;
            }
        } else {
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
                "cell {ci} still references non-representative vertex {vi} after in-place repair"
            );
        }
    }

    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use std::collections::BTreeSet;

    fn edge_record(a: u32, b: u32) -> EdgeRecord {
        EdgeRecord {
            key: (((b as u64) << 32) | a as u64).into(),
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
        let residual_r = reconcile_unresolved_edges(
            records,
            vertices,
            &mut cells_r,
            &mut idx_r,
            VertexKeys::Flat(vertex_keys),
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
            RepairApply::Rebuild,
            |_, _| false,
        )
        .expect("rebuild reconciliation should succeed");

        let (mut cells_p, mut idx_p) = (cells.to_vec(), cell_indices.to_vec());
        let residual_p = reconcile_unresolved_edges(
            records,
            vertices,
            &mut cells_p,
            &mut idx_p,
            VertexKeys::Flat(vertex_keys),
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
            RepairApply::InPlace,
            |_, _| false,
        )
        .expect("in-place reconciliation should succeed");

        assert_eq!(
            residual_r, residual_p,
            "backends disagree on post-repair residuals"
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

    /// The localized BFS dup-scan must union the same components as the global
    /// O(V) scan — including a *chain*: corner [0,1,2] is triplicated (copies
    /// in cells 0,1,2) and corner [2,3,4] is duplicated (copies in cells 2,3).
    /// Only edge (0,1) is recorded, so cells 3,4 are reached purely through the
    /// damaged-cell 1-ring expansion off cell 2.
    #[test]
    fn localized_dupscan_matches_global_with_chain() {
        // vertices 0,1,2 = copies of corner [0,1,2]; 3,4 = copies of [2,3,4].
        let vertex_keys: Vec<VertexKey> =
            vec![[0, 1, 2], [0, 1, 2], [0, 1, 2], [2, 3, 4], [2, 3, 4]];
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
            &cells,
            &cell_indices,
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
    fn repair_collapses_one_sided_epsilon_edge() {
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

        let (changed, cells_rebuild, idx_rebuild, cells_in_place, _) = run_both_backends(
            &[edge_record(0, 1)],
            &vertices,
            &cells,
            &cell_indices,
            &vertex_keys,
        );
        assert!(changed, "expected one-sided epsilon edge to be reconciled");
        assert_eq!(
            cells_in_place[0].vertex_count(),
            2,
            "epsilon edge should collapse"
        );
        assert_eq!(
            idx_rebuild.len(),
            5,
            "rebuild should compact away the merged per-cell index"
        );
        assert_eq!(cells_rebuild[0].vertex_count(), 2);
    }

    #[test]
    fn repair_reconciles_mismatched_shared_edge_endpoints() {
        let vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0 + 1.0e-5, 2.0e-6, 0.0),
            Vec3::new(2.0e-6, 1.0 + 1.0e-5, 0.0),
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

        let seg_a_before =
            edge_segments_for_neighbor(0, 1, &cells, &cell_indices, VertexKeys::Flat(&vertex_keys))
                .unwrap();
        let seg_b_before =
            edge_segments_for_neighbor(1, 0, &cells, &cell_indices, VertexKeys::Flat(&vertex_keys))
                .unwrap();
        assert_eq!(seg_a_before.len(), 1);
        assert_eq!(seg_b_before.len(), 1);
        let before_a = BTreeSet::from([seg_a_before[0].0, seg_a_before[0].1]);
        let before_b = BTreeSet::from([seg_b_before[0].0, seg_b_before[0].1]);
        assert_ne!(
            before_a, before_b,
            "fixture must start with mismatched shared-edge endpoint ids"
        );

        let (changed, _, _, new_cells, new_indices) = run_both_backends(
            &[edge_record(0, 1)],
            &vertices,
            &cells,
            &cell_indices,
            &vertex_keys,
        );
        assert!(
            changed,
            "expected mismatched shared-edge endpoints to be reconciled"
        );
        let seg_a = edge_segments_for_neighbor(
            0,
            1,
            &new_cells,
            &new_indices,
            VertexKeys::Flat(&vertex_keys),
        )
        .unwrap();
        let seg_b = edge_segments_for_neighbor(
            1,
            0,
            &new_cells,
            &new_indices,
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
    }
}
