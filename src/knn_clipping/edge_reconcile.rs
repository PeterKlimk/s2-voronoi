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

use super::live_dedup::{EdgeKey, EdgeRecord};
use crate::diagram::VoronoiCell;
use crate::knn_clipping::cell_build::VertexKey;

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
    vertex_keys: &[VertexKey],
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
        let ki = *vertex_keys.get(vi as usize).ok_or_else(|| {
            reconcile_state_error(format!(
                "edge reconciliation vertex id {} out of range for vertex_keys len {}",
                vi,
                vertex_keys.len()
            ))
        })?;
        let kj = *vertex_keys.get(vj as usize).ok_or_else(|| {
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
/// differential in tests/edge_repair_net.rs via `S2_EDGE_REPAIR_REBUILD`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RepairApply {
    InPlace,
    Rebuild,
}

/// Production apply mode: in-place unless `S2_EDGE_REPAIR_REBUILD=1`
/// selects the rebuild oracle (diagnostic / differential-testing knob,
/// read once per compute on the cold path).
pub(crate) fn repair_apply_from_env() -> RepairApply {
    match std::env::var("S2_EDGE_REPAIR_REBUILD") {
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
    vertex_keys: &[VertexKey],
    // Degenerate-length threshold in the caller's coordinate space (chord
    // units on the sphere, normalized rect units on the plane); each
    // geometry owns and justifies its constant.
    degenerate_len_eps: f32,
    apply: RepairApply,
    // Geometry's boundary classification: true when a single-use edge
    // between these vertex ids is legitimate (plane rect walls). The
    // sphere and the periodic plane have no boundary.
    is_boundary_edge: impl Fn(u32, u32) -> bool,
) -> Result<Vec<(u32, u32)>, crate::VoronoiError> {
    if edge_records.is_empty() {
        return Ok(Vec::new());
    }
    run_repair_rounds(
        edge_records,
        vertices,
        cells,
        cell_indices,
        vertex_keys,
        degenerate_len_eps,
        apply,
        MergeMode::Primary,
    )?;

    let unpaired = scan_unpaired_interior(cells, cell_indices, &is_boundary_edge)?;
    if unpaired.is_empty() {
        return Ok(Vec::new());
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
        )?;
    }
    let residual = scan_unpaired_interior(cells, cell_indices, &is_boundary_edge)?;
    Ok(residual
        .iter()
        .map(|&(va, vb, owner)| cell_pair_for_unpaired(va, vb, owner, vertex_keys))
        .collect())
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

/// Drop spurious collinear vertices (degenerate keys with a repeated
/// generator) from the cells that own them. Such a vertex lies on a single
/// bisector — both its incident edges in that cell go to the same neighbor
/// — so it is not a Voronoi triple point, and removing it merges the two
/// collinear segments into the real edge (exact). Returns whether anything
/// was dropped. Touches only cells that own a degenerate vertex.
fn drop_degenerate_collinear_vertices(
    cells: &mut [VoronoiCell],
    cell_indices: &mut [u32],
    vertex_keys: &[VertexKey],
) -> bool {
    let mut affected: Vec<u32> = Vec::new();
    for key in vertex_keys {
        if let Some(single) = degenerate_single(*key) {
            affected.push(single);
        }
    }
    if affected.is_empty() {
        return false;
    }
    affected.sort_unstable();
    affected.dedup();
    affected.retain(|&c| (c as usize) < cells.len());

    let mut changed = false;
    for &c in &affected {
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
            .filter(|&v| {
                vertex_keys
                    .get(v as usize)
                    .and_then(|&k| degenerate_single(k))
                    != Some(c)
            })
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
    vertex_keys: &[VertexKey],
    degenerate_len_eps: f32,
    apply: RepairApply,
    mode: MergeMode,
) -> Result<bool, crate::VoronoiError> {
    let mut any = false;
    for round in 0..MAX_REPAIR_ROUNDS {
        // Drop spurious collinear (degenerate-key) vertices first: a vertex
        // whose key has only two distinct generators is not a triple point,
        // it lies on a single bisector (both incident edges go to the same
        // neighbor) — removing it merges the two collinear segments into the
        // real edge and is exact. One cell can carry such a point where its
        // neighbor sees a straight edge, which is precisely an unpaired-edge
        // defect; this heals it with no cross-cell rewrite.
        let dropped = drop_degenerate_collinear_vertices(cells, cell_indices, vertex_keys);
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

/// Output-invariant scan: undirected edges over all cell boundaries used
/// exactly once that are not legitimate boundary edges. Returns
/// (vertex_a, vertex_b, owning cell), sorted. O(total cell indices).
fn scan_unpaired_interior(
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    is_boundary_edge: &impl Fn(u32, u32) -> bool,
) -> Result<Vec<(u32, u32, u32)>, crate::VoronoiError> {
    use std::collections::HashMap;
    let mut uses: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for ci in 0..cells.len() {
        let span = cell_vertex_slice(ci as u32, cells, cell_indices)?;
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
    vertex_keys: &[VertexKey],
    num_cells: usize,
) -> Vec<EdgeRecord> {
    let mut keys: Vec<u64> = unpaired
        .iter()
        .filter_map(|&(va, vb, _)| {
            let k1 = *vertex_keys.get(va as usize)?;
            let k2 = *vertex_keys.get(vb as usize)?;
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
fn cell_pair_for_unpaired(va: u32, vb: u32, owner: u32, vertex_keys: &[VertexKey]) -> (u32, u32) {
    match (vertex_keys.get(va as usize), vertex_keys.get(vb as usize)) {
        (Some(&k1), Some(&k2)) => key_common_pair(k1, k2).unwrap_or((owner, owner)),
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

#[allow(clippy::too_many_arguments)]
fn collect_merges<P: crate::knn_clipping::live_dedup::VertexPosition>(
    edge_records: &[EdgeRecord],
    vertices: &[P],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
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
        let mut first_by_key: std::collections::HashMap<VertexKey, u32> =
            std::collections::HashMap::with_capacity(vertex_keys.len());
        for (i, key) in vertex_keys.iter().enumerate() {
            match first_by_key.entry(*key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(i as u32);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    if uf.union(*e.get(), i as u32) {
                        merged += 1;
                    }
                }
            }
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
    vertex_keys: &[VertexKey],
) -> Result<bool, crate::VoronoiError> {
    let mut changed = false;
    let mut affected: Vec<u32> = Vec::new();
    for v in uf.touched_ids() {
        let key = *vertex_keys.get(v as usize).ok_or_else(|| {
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
            vertex_keys,
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
            vertex_keys,
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
            edge_segments_for_neighbor(0, 1, &cells, &cell_indices, &vertex_keys).unwrap();
        let seg_b_before =
            edge_segments_for_neighbor(1, 0, &cells, &cell_indices, &vertex_keys).unwrap();
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
        let seg_a =
            edge_segments_for_neighbor(0, 1, &new_cells, &new_indices, &vertex_keys).unwrap();
        let seg_b =
            edge_segments_for_neighbor(1, 0, &new_cells, &new_indices, &vertex_keys).unwrap();
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
