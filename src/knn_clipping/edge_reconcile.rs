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

pub(crate) fn reconcile_unresolved_edges<P: crate::knn_clipping::live_dedup::VertexPosition>(
    edge_records: &[EdgeRecord],
    vertices: &[P],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
    // Degenerate-length threshold in the caller's coordinate space (chord
    // units on the sphere, normalized rect units on the plane); each
    // geometry owns and justifies its constant.
    degenerate_len_eps: f32,
) -> Result<Option<ReconciledCells>, crate::VoronoiError> {
    // Sparse: only the handful of vertices named by defective edges ever
    // enter the structure, so clean and near-clean runs skip the O(V) init
    // a dense UnionFind would pay. Representative choice is identical to
    // the dense version (see SparseUnionFind docs), so output is unchanged.
    let mut uf = SparseUnionFind::new();
    let mut merged = 0usize;
    let degenerate_len_eps_sq: f32 = degenerate_len_eps * degenerate_len_eps;

    for record in edge_records {
        let (a, b) = unpack_edge(record.key.as_u64());
        let seg_a = edge_segments_for_neighbor(a, b, cells, cell_indices, vertex_keys)?;
        let seg_b = edge_segments_for_neighbor(b, a, cells, cell_indices, vertex_keys)?;
        if seg_a.len() != 1 || seg_b.len() != 1 {
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

    if merged == 0 {
        return Ok(None);
    }

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

    Ok(Some((new_cells, new_indices)))
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

        let repaired = reconcile_unresolved_edges(
            &[edge_record(0, 1)],
            &vertices,
            &cells,
            &cell_indices,
            &vertex_keys,
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        )
        .expect("reconciliation should succeed without capacity overflow")
        .expect("expected one-sided epsilon edge to be reconciled");

        let (new_cells, new_indices) = repaired;
        assert_eq!(
            new_cells[0].vertex_count(),
            2,
            "epsilon edge should collapse"
        );
        assert_eq!(
            new_indices.len(),
            5,
            "collapsing the epsilon edge should remove one per-cell index"
        );
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

        let repaired = reconcile_unresolved_edges(
            &[edge_record(0, 1)],
            &vertices,
            &cells,
            &cell_indices,
            &vertex_keys,
            crate::tolerances::RECONCILE_DEGENERATE_LEN_EPS,
        )
        .expect("reconciliation should succeed without capacity overflow")
        .expect("expected mismatched shared-edge endpoints to be reconciled");

        let (new_cells, new_indices) = repaired;
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
