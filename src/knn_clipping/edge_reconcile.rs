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

use glam::Vec3;

use super::live_dedup::EdgeRecord;
use crate::knn_clipping::cell_builder::VertexKey;
use crate::VoronoiCell;

#[inline]
pub(super) fn unpack_edge(key: u64) -> (u32, u32) {
    (key as u32, (key >> 32) as u32)
}

fn key_contains(key: VertexKey, value: u32) -> bool {
    key[0] == value || key[1] == value || key[2] == value
}

pub(super) fn shared_neighbor(cell_idx: u32, a: VertexKey, b: VertexKey) -> Option<u32> {
    if !key_contains(a, cell_idx) || !key_contains(b, cell_idx) {
        return None;
    }
    a.iter()
        .find(|&&candidate| candidate != cell_idx && key_contains(b, candidate))
        .copied()
}

pub(super) fn edge_segments_for_neighbor(
    cell_idx: u32,
    neighbor: u32,
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
) -> Vec<(u32, u32)> {
    let cell_idx_usize = cell_idx as usize;
    if cell_idx_usize >= cells.len() {
        return Vec::new();
    }
    let cell = &cells[cell_idx_usize];
    let start = cell.vertex_start();
    let end = start + cell.vertex_count();
    let slice = &cell_indices[start..end];
    let n = slice.len();
    if n < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..n {
        let vi = slice[i];
        let vj = slice[(i + 1) % n];
        let ki = vertex_keys[vi as usize];
        let kj = vertex_keys[vj as usize];
        if shared_neighbor(cell_idx, ki, kj) == Some(neighbor) {
            out.push((vi, vj));
        }
    }
    out
}

fn dist_sq(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    d.length_squared()
}

use super::union_find::UnionFind;

pub(super) fn reconcile_unresolved_edges(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
) -> Option<(Vec<VoronoiCell>, Vec<u32>)> {
    let mut uf = UnionFind::new(vertices.len());
    let mut merged = 0usize;
    const DEGENERATE_LEN_EPS: f32 = 1e-6;
    const DEGENERATE_LEN_EPS_SQ: f32 = DEGENERATE_LEN_EPS * DEGENERATE_LEN_EPS;

    for record in edge_records {
        let (a, b) = unpack_edge(record.key.as_u64());
        let seg_a = edge_segments_for_neighbor(a, b, cells, cell_indices, vertex_keys);
        let seg_b = edge_segments_for_neighbor(b, a, cells, cell_indices, vertex_keys);
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
                let v0_usize = v0 as usize;
                let v1_usize = v1 as usize;
                if v0_usize < vertices.len() && v1_usize < vertices.len() {
                    let len_sq = dist_sq(vertices[v0_usize], vertices[v1_usize]);
                    if len_sq <= DEGENERATE_LEN_EPS_SQ {
                        if uf.union(v0, v1) {
                            merged += 1;
                        }

                        // If the neighbor cell contains an exactly coincident vertex, merge onto it
                        // to improve global consistency across cells.
                        let other_cell = other_cell as usize;
                        if other_cell < cells.len() {
                            let cell = &cells[other_cell];
                            let start = cell.vertex_start();
                            let end = start + cell.vertex_count();
                            let slice = &cell_indices[start..end];
                            for &vi in [v0, v1].iter() {
                                let vi_usize = vi as usize;
                                if vi_usize >= vertices.len() {
                                    continue;
                                }
                                let mut best: Option<(u32, f32)> = None;
                                for &vj in slice {
                                    let vj_usize = vj as usize;
                                    if vj_usize >= vertices.len() {
                                        continue;
                                    }
                                    let d = dist_sq(vertices[vi_usize], vertices[vj_usize]);
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
                                    if best_d <= DEGENERATE_LEN_EPS_SQ && uf.union(vi, vj) {
                                        merged += 1;
                                    }
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

        let d00 = dist_sq(vertices[a0 as usize], vertices[b0 as usize])
            + dist_sq(vertices[a1 as usize], vertices[b1 as usize]);
        let d01 = dist_sq(vertices[a0 as usize], vertices[b1 as usize])
            + dist_sq(vertices[a1 as usize], vertices[b0 as usize]);
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
        return None;
    }

    let mut new_cells: Vec<VoronoiCell> = Vec::with_capacity(cells.len());
    let mut new_indices: Vec<u32> = Vec::with_capacity(cell_indices.len());

    for cell in cells {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        let base = new_indices.len();
        let mut seen: Vec<u32> = Vec::with_capacity(cell.vertex_count());
        for &vi in &cell_indices[start..end] {
            let rep = uf.find(vi);
            if !seen.contains(&rep) {
                seen.push(rep);
                new_indices.push(rep);
            }
        }
        let count = new_indices.len() - base;
        let count_u16 = u16::try_from(count).expect("cell vertex count exceeds u16 capacity");
        let start_u32 = u32::try_from(base).expect("cell index buffer exceeds u32 capacity");
        new_cells.push(VoronoiCell::new(start_u32, count_u16));
    }

    Some((new_cells, new_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        )
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

        let seg_a_before = edge_segments_for_neighbor(0, 1, &cells, &cell_indices, &vertex_keys);
        let seg_b_before = edge_segments_for_neighbor(1, 0, &cells, &cell_indices, &vertex_keys);
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
        )
        .expect("expected mismatched shared-edge endpoints to be reconciled");

        let (new_cells, new_indices) = repaired;
        let seg_a = edge_segments_for_neighbor(0, 1, &new_cells, &new_indices, &vertex_keys);
        let seg_b = edge_segments_for_neighbor(1, 0, &new_cells, &new_indices, &vertex_keys);
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
