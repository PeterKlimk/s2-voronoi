//! Edge repair helpers for post-processing.

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
    for &candidate in &a {
        if candidate != cell_idx && key_contains(b, candidate) {
            return Some(candidate);
        }
    }
    None
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

pub(super) fn segment_len_stats(
    segments: &[(u32, u32)],
    vertices: &[Vec3],
) -> (Option<f32>, Option<f32>) {
    let mut min_len_sq: Option<f32> = None;
    let mut max_len_sq: Option<f32> = None;
    for &(a, b) in segments {
        let a = a as usize;
        let b = b as usize;
        if a >= vertices.len() || b >= vertices.len() {
            continue;
        }
        let len_sq = dist_sq(vertices[a], vertices[b]);
        min_len_sq = Some(min_len_sq.map_or(len_sq, |v| v.min(len_sq)));
        max_len_sq = Some(max_len_sq.map_or(len_sq, |v| v.max(len_sq)));
    }
    (min_len_sq.map(|v| v.sqrt()), max_len_sq.map(|v| v.sqrt()))
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i as u32);
        }
        Self {
            parent,
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: u32) -> u32 {
        let idx = x as usize;
        let p = self.parent[idx];
        if p != x {
            let root = self.find(p);
            self.parent[idx] = root;
        }
        self.parent[idx]
    }

    fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let ra_idx = ra as usize;
        let rb_idx = rb as usize;
        if self.rank[ra_idx] < self.rank[rb_idx] {
            self.parent[ra_idx] = rb;
        } else if self.rank[ra_idx] > self.rank[rb_idx] {
            self.parent[rb_idx] = ra;
        } else {
            self.parent[rb_idx] = ra;
            self.rank[ra_idx] = self.rank[ra_idx].saturating_add(1);
        }
        true
    }
}

pub(super) fn repair_bad_edges(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &[VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
) -> Option<(Vec<VoronoiCell>, Vec<u32>)> {
    let log = super::log_enabled();

    #[derive(Debug)]
    struct SkipDebug {
        a: u32,
        b: u32,
        seg_a_len: usize,
        seg_b_len: usize,
        seg_a_len_min: Option<f32>,
        seg_a_len_max: Option<f32>,
        seg_b_len_min: Option<f32>,
        seg_b_len_max: Option<f32>,
        min_cross_cell_len: Option<f32>,
    }

    fn min_cross_cell_len(
        a: u32,
        b: u32,
        cells: &[VoronoiCell],
        cell_indices: &[u32],
        vertices: &[Vec3],
    ) -> Option<f32> {
        let a = a as usize;
        let b = b as usize;
        if a >= cells.len() || b >= cells.len() {
            return None;
        }
        let a_cell = &cells[a];
        let b_cell = &cells[b];
        let a_start = a_cell.vertex_start();
        let a_end = a_start + a_cell.vertex_count();
        let b_start = b_cell.vertex_start();
        let b_end = b_start + b_cell.vertex_count();
        let a_slice = &cell_indices[a_start..a_end];
        let b_slice = &cell_indices[b_start..b_end];
        if a_slice.is_empty() || b_slice.is_empty() {
            return None;
        }
        let mut best_sq: Option<f32> = None;
        for &ai in a_slice {
            let ai = ai as usize;
            if ai >= vertices.len() {
                continue;
            }
            for &bi in b_slice {
                let bi = bi as usize;
                if bi >= vertices.len() {
                    continue;
                }
                let d = dist_sq(vertices[ai], vertices[bi]);
                best_sq = Some(best_sq.map_or(d, |v| v.min(d)));
            }
        }
        best_sq.map(|v| v.sqrt())
    }

    let mut uf = UnionFind::new(vertices.len());
    let mut merged = 0usize;
    let mut skipped = 0usize;
    let mut already = 0usize;
    let mut skip_debug: Vec<SkipDebug> = Vec::new();
    const MAX_SKIP_DEBUG: usize = 16;
    let mut degenerate_fixed = 0usize;
    let mut degenerate_unions = 0usize;
    const DEGENERATE_LEN_EPS: f32 = 1e-6;
    const DEGENERATE_LEN_EPS_SQ: f32 = DEGENERATE_LEN_EPS * DEGENERATE_LEN_EPS;

    if log {
        eprintln!("edge repair: input_edges={}", edge_records.len());
    }

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
            let mut handled_degenerate = false;
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
                        handled_degenerate = true;
                        degenerate_fixed += 1;
                        if uf.union(v0, v1) {
                            merged += 1;
                            degenerate_unions += 1;
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
                                        degenerate_unions += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if handled_degenerate {
                continue;
            }

            skipped += 1;
            if log && skip_debug.len() < MAX_SKIP_DEBUG {
                let (seg_a_len_min, seg_a_len_max) = segment_len_stats(&seg_a, vertices);
                let (seg_b_len_min, seg_b_len_max) = segment_len_stats(&seg_b, vertices);
                skip_debug.push(SkipDebug {
                    a,
                    b,
                    seg_a_len: seg_a.len(),
                    seg_b_len: seg_b.len(),
                    seg_a_len_min,
                    seg_a_len_max,
                    seg_b_len_min,
                    seg_b_len_max,
                    min_cross_cell_len: min_cross_cell_len(a, b, cells, cell_indices, vertices),
                });
            }
            continue;
        }
        let (a0, a1) = seg_a[0];
        let (b0, b1) = seg_b[0];

        let share_a0 = a0 == b0 || a0 == b1;
        let share_a1 = a1 == b0 || a1 == b1;
        if share_a0 && share_a1 {
            already += 1;
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
            if !seen.iter().any(|&x| x == rep) {
                seen.push(rep);
                new_indices.push(rep);
            }
        }
        let count = new_indices.len() - base;
        let count_u16 = u16::try_from(count).expect("cell vertex count exceeds u16 capacity");
        let start_u32 = u32::try_from(base).expect("cell index buffer exceeds u32 capacity");
        new_cells.push(VoronoiCell::new(start_u32, count_u16));
    }

    if log {
        eprintln!(
            "edge repair: merged={} skipped={} already_aligned={}",
            merged, skipped, already
        );
        if degenerate_fixed > 0 {
            eprintln!(
                "  degenerate edge repair: fixed_edges={} unions={}",
                degenerate_fixed, degenerate_unions
            );
        }
        if !skip_debug.is_empty() {
            eprintln!(
                "  skipped edge length debug (showing {}):",
                skip_debug.len()
            );
            for dbg in &skip_debug {
                eprintln!(
                    "    edge=({},{}) seg_a={} seg_b={} seg_a_len=[{:?},{:?}] seg_b_len=[{:?},{:?}] min_cross_cell_len={:?}",
                    dbg.a,
                    dbg.b,
                    dbg.seg_a_len,
                    dbg.seg_b_len,
                    dbg.seg_a_len_min,
                    dbg.seg_a_len_max,
                    dbg.seg_b_len_min,
                    dbg.seg_b_len_max,
                    dbg.min_cross_cell_len
                );
            }
        }
    }

    Some((new_cells, new_indices))
}
