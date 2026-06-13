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

use super::live_dedup::EdgeRecord;
use crate::diagram::VoronoiCell;
use crate::knn_clipping::cell_build::VertexKey;

fn reconcile_state_error(message: impl Into<String>) -> crate::VoronoiError {
    crate::VoronoiError::ComputationFailed(message.into())
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

/// Reconcile unresolved shared-edge mismatches by merging vertex identities,
/// patching `cells` / `cell_indices` via the chosen backend. Returns whether
/// anything merged (false leaves the arrays untouched).
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
) -> Result<bool, crate::VoronoiError> {
    let (mut uf, merged) = collect_merges(
        edge_records,
        vertices,
        cells,
        cell_indices,
        vertex_keys,
        degenerate_len_eps,
    )?;
    if merged == 0 {
        return Ok(false);
    }
    match apply {
        RepairApply::Rebuild => {
            let (new_cells, new_indices) = apply_merges_rebuild(&mut uf, cells, cell_indices)?;
            *cells = new_cells;
            *cell_indices = new_indices;
        }
        RepairApply::InPlace => {
            apply_merges_in_place(&mut uf, cells, cell_indices, vertex_keys)?;
        }
    }
    Ok(true)
}

/// Walk the unresolved edge records and collect vertex-identity merges into
/// a union-find. Both apply backends consume the exact same merge set.
fn collect_merges<P: crate::knn_clipping::live_dedup::VertexPosition>(
    edge_records: &[EdgeRecord],
    vertices: &[P],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
    degenerate_len_eps: f32,
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
    if !edge_records.is_empty() {
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
        if seg_a.len() != 1 || seg_b.len() != 1 {
            // Irregular topology (sliver chains, overlapping defects): union
            // every pair of segment-endpoint vertices — across and within
            // the two sides — that lie within the degenerate length scale.
            // Position-based and local to the defective edge, so it stays
            // O(defect size); it collapses duplicate-position vertices with
            // distinct keys (an exact-tie corner committed under two
            // attributions) and sliver chains the per-segment logic cannot
            // pair up.
            let mut endpoint_ids: Vec<u32> = Vec::new();
            for &(v0, v1) in seg_a.iter().chain(seg_b.iter()) {
                endpoint_ids.push(v0);
                endpoint_ids.push(v1);
            }
            endpoint_ids.sort_unstable();
            endpoint_ids.dedup();
            for i in 0..endpoint_ids.len() {
                for j in (i + 1)..endpoint_ids.len() {
                    let d = dist_sq(
                        vertex_pos(vertices, endpoint_ids[i])?,
                        vertex_pos(vertices, endpoint_ids[j])?,
                    );
                    if d <= degenerate_len_eps_sq && uf.union(endpoint_ids[i], endpoint_ids[j]) {
                        merged += 1;
                    }
                }
            }

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
) -> Result<(), crate::VoronoiError> {
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
            let rep = uf.find(span[r]);
            if !span[..w].contains(&rep) {
                span[w] = rep;
                w += 1;
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

    Ok(())
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
        let changed_r = reconcile_unresolved_edges(
            records,
            vertices,
            &mut cells_r,
            &mut idx_r,
            vertex_keys,
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
            RepairApply::Rebuild,
        )
        .expect("rebuild reconciliation should succeed");

        let (mut cells_p, mut idx_p) = (cells.to_vec(), cell_indices.to_vec());
        let changed_p = reconcile_unresolved_edges(
            records,
            vertices,
            &mut cells_p,
            &mut idx_p,
            vertex_keys,
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
            RepairApply::InPlace,
        )
        .expect("in-place reconciliation should succeed");

        assert_eq!(
            changed_r, changed_p,
            "backends disagree on whether to merge"
        );
        assert_eq!(
            cell_sequences(&cells_r, &idx_r),
            cell_sequences(&cells_p, &idx_p),
            "backends disagree on per-cell vertex sequences"
        );
        (changed_p, cells_r, idx_r, cells_p, idx_p)
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
