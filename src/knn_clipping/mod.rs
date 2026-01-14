//! GPU-friendly spherical Voronoi computation via half-space (great circle) clipping.
//!
//! This module implements a "meshless" approach where each Voronoi cell is computed
//! independently from its k nearest neighbors. This enables massive parallelism on GPU.

macro_rules! maybe_par_into_iter {
    ($v:expr) => {{
        #[cfg(feature = "parallel")]
        {
            $v.into_par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            $v.into_iter()
        }
    }};
}

mod cell_builder;
mod constants;
mod knn;
mod live_dedup;
mod timing;
mod topo2d;

use glam::Vec3;
use std::sync::OnceLock;

// Re-exports (internal use)
pub use cell_builder::VertexKey;
pub use knn::CubeMapGridKnn;
use live_dedup::EdgeRecord;

fn log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("S2V_LOG")
            .ok()
            .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

#[derive(Debug, Clone, Copy)]
pub struct TerminationConfig {
    /// Enables adaptive k-NN + early termination checks.
    ///
    /// When disabled, the builder will still run the initial k-NN pass but will not
    /// attempt to terminate early; the k-NN schedule still runs to ensure
    /// correctness (so this should generally remain enabled for performance).
    pub check_start: usize,
    pub check_step: usize,
}

// Keep the k-NN schedule and the default termination cadence in one place.
pub(super) const KNN_RESUME_K: usize = 18;
pub(super) const KNN_RESTART_MAX: usize = 48;
pub(super) const KNN_RESTART_KS: [usize; 2] = [24, KNN_RESTART_MAX];

/// Target points per cell for the cube-map KNN grid.
/// Lower = more cells, faster scans, more heap overhead.
/// Higher = fewer cells, longer scans, less overhead.
pub(super) const KNN_GRID_TARGET_DENSITY: f64 = 16.0;

// Default termination cadence:
// - start near the end of the initial k pass
// - then check roughly twice per initial-k window
const DEFAULT_TERMINATION_CHECK_START: usize = 8;
const DEFAULT_TERMINATION_CHECK_STEP: usize = 1;

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            check_start: DEFAULT_TERMINATION_CHECK_START,
            check_step: DEFAULT_TERMINATION_CHECK_STEP,
        }
    }
}

impl TerminationConfig {
    #[inline]
    pub fn should_check(&self, neighbors_processed: usize) -> bool {
        self.check_step > 0
            && neighbors_processed >= self.check_start
            && (neighbors_processed - self.check_start) % self.check_step == 0
    }
}

#[derive(Debug, Clone, Copy)]
struct LowIncidenceVertex {
    index: u32,
    degree: u8,
    key: VertexKey,
}

fn collect_low_incidence_vertices(
    cell_indices: &[u32],
    num_vertices: usize,
    vertex_keys: &[VertexKey],
) -> Vec<LowIncidenceVertex> {
    debug_assert_eq!(
        vertex_keys.len(),
        num_vertices,
        "vertex keys out of sync with vertex positions"
    );

    let mut degree: Vec<u8> = vec![0; num_vertices];
    for &vi in cell_indices {
        let idx = vi as usize;
        if idx < num_vertices && degree[idx] < 3 {
            degree[idx] += 1;
        }
    }

    let mut out = Vec::new();
    for (i, &d) in degree.iter().enumerate() {
        if d < 3 {
            out.push(LowIncidenceVertex {
                index: i as u32,
                degree: d,
                key: vertex_keys[i],
            });
        }
    }
    out
}

#[inline]
fn unpack_edge(key: u64) -> (u32, u32) {
    (key as u32, (key >> 32) as u32)
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

fn key_contains(key: VertexKey, value: u32) -> bool {
    key[0] == value || key[1] == value || key[2] == value
}

fn shared_neighbor(cell_idx: u32, a: VertexKey, b: VertexKey) -> Option<u32> {
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

fn build_vertex_cells(
    cells: &[crate::VoronoiCell],
    cell_indices: &[u32],
    num_vertices: usize,
) -> Vec<Vec<u32>> {
    let mut vertex_cells: Vec<Vec<u32>> = vec![Vec::new(); num_vertices];
    for (cell_idx, cell) in cells.iter().enumerate() {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        for &vi in &cell_indices[start..end] {
            let vi_usize = vi as usize;
            if vi_usize < num_vertices {
                vertex_cells[vi_usize].push(cell_idx as u32);
            }
        }
    }
    vertex_cells
}

fn edge_segments_for_neighbor(
    cell_idx: u32,
    neighbor: u32,
    cells: &[crate::VoronoiCell],
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

fn debug_low_incidence_edges(
    low_vertices: &[LowIncidenceVertex],
    vertex_cells: &[Vec<u32>],
    cells: &[crate::VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
) {
    for v in low_vertices {
        let vi = v.index as usize;
        if vi >= vertex_cells.len() {
            continue;
        }
        let cells_with_vertex = &vertex_cells[vi];
        eprintln!(
            "  vi={} degree={} key=[{}, {}, {}] cells={:?}",
            v.index, v.degree, v.key[0], v.key[1], v.key[2], cells_with_vertex
        );

        for &cell_idx in cells_with_vertex {
            let cell_idx_usize = cell_idx as usize;
            if cell_idx_usize >= cells.len() {
                continue;
            }
            let cell = &cells[cell_idx_usize];
            let start = cell.vertex_start();
            let end = start + cell.vertex_count();
            let slice = &cell_indices[start..end];
            let n = slice.len();
            if n < 2 {
                continue;
            }

            let pos = slice.iter().position(|&x| x == v.index);
            let Some(pos) = pos else { continue };
            let prev = slice[(pos + n - 1) % n];
            let next = slice[(pos + 1) % n];

            let key_v = vertex_keys[v.index as usize];
            let key_prev = vertex_keys[prev as usize];
            let key_next = vertex_keys[next as usize];

            for (label, other_vi, other_key) in [("prev", prev, key_prev), ("next", next, key_next)]
            {
                let neighbor = shared_neighbor(cell_idx, key_v, other_key);
                match neighbor {
                    Some(b) => {
                        let b_segments = edge_segments_for_neighbor(
                            b,
                            cell_idx,
                            cells,
                            cell_indices,
                            vertex_keys,
                        );
                        eprintln!(
                            "    cell={} edge {} -> neighbor={} local vi={} key=[{}, {}, {}] vj={} key=[{}, {}, {}] other_cell_segments={}",
                            cell_idx,
                            label,
                            b,
                            v.index,
                            key_v[0],
                            key_v[1],
                            key_v[2],
                            other_vi,
                            other_key[0],
                            other_key[1],
                            other_key[2],
                            b_segments.len()
                        );
                        if let Some((b0, b1)) = b_segments.get(0) {
                            let kb0 = vertex_keys[*b0 as usize];
                            let kb1 = vertex_keys[*b1 as usize];
                            eprintln!(
                                "      other_cell_edge vi={} key=[{}, {}, {}], vj={} key=[{}, {}, {}]",
                                b0,
                                kb0[0],
                                kb0[1],
                                kb0[2],
                                b1,
                                kb1[0],
                                kb1[1],
                                kb1[2]
                            );
                        }
                    }
                    None => {
                        eprintln!("    cell={} edge {} -> neighbor=NONE", cell_idx, label);
                    }
                }
            }
        }
    }
}

fn dist_sq(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    d.length_squared()
}

fn segment_len_stats(segments: &[(u32, u32)], vertices: &[Vec3]) -> (Option<f32>, Option<f32>) {
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

fn debug_low_degree_vertices(
    label: &str,
    generators: &[Vec3],
    knn: Option<&CubeMapGridKnn<'_>>,
    vertices: &[Vec3],
    vertex_keys: &[VertexKey],
    cells: &[crate::VoronoiCell],
    cell_indices: &[u32],
    max_print: usize,
) {
    if max_print == 0 {
        return;
    }

    let low_vertices = collect_low_incidence_vertices(cell_indices, vertices.len(), vertex_keys);
    if low_vertices.is_empty() {
        return;
    }

    let include_d0 = std::env::var("S2V_DEBUG_LOW_DEGREE_INCLUDE_D0")
        .ok()
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let knn_k: usize = std::env::var("S2V_DEBUG_LOW_DEGREE_KNN_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // We care primarily about d1/d2; d0 is expected when edge repair unions vertices
    // but doesn't compact the vertex array.
    let mut interesting: Vec<LowIncidenceVertex> = low_vertices
        .iter()
        .copied()
        .filter(|v| v.degree == 1 || v.degree == 2 || (include_d0 && v.degree == 0))
        .collect();
    if interesting.is_empty() {
        return;
    }

    // Print d1/d2 first, then d0.
    interesting.sort_by_key(|v| (v.degree == 0, v.degree, v.index));

    let d0 = low_vertices.iter().filter(|v| v.degree == 0).count();
    let d1 = low_vertices.iter().filter(|v| v.degree == 1).count();
    let d2 = low_vertices.iter().filter(|v| v.degree == 2).count();
    eprintln!(
        "  low-degree vertex debug ({}): total={} (d0={}, d1={}, d2={})",
        label,
        low_vertices.len(),
        d0,
        d1,
        d2
    );

    let vertex_cells = build_vertex_cells(cells, cell_indices, vertices.len());
    let knn_k = knn_k
        .min(KNN_RESTART_MAX)
        .min(generators.len().saturating_sub(1));
    let mut knn_scratch = (knn_k > 0).then(|| knn.map(|k| k.make_scratch())).flatten();
    let mut knn_out: Vec<usize> = Vec::new();

    fn knn_contains_with_rank(
        knn: &CubeMapGridKnn<'_>,
        generators: &[Vec3],
        src: u32,
        dst: u32,
        k: usize,
        scratch: &mut crate::cube_grid::CubeMapGridScratch,
        out: &mut Vec<usize>,
    ) -> Option<(bool, Option<usize>, f32, f32)> {
        if k == 0 {
            return None;
        }
        let src_usize = src as usize;
        let dst_usize = dst as usize;
        if src_usize >= generators.len() || dst_usize >= generators.len() {
            return None;
        }
        out.clear();
        knn.knn_into(generators[src_usize], src_usize, k, scratch, out);
        let rank = out.iter().position(|&x| x == dst_usize);
        let dot = generators[src_usize].dot(generators[dst_usize]);
        let min_dot = out
            .iter()
            .map(|&j| generators[src_usize].dot(generators[j]))
            .fold(1.0_f32, |acc, v| acc.min(v));
        Some((rank.is_some(), rank, dot, min_dot))
    }

    let mut printed = 0usize;
    for v in &interesting {
        if v.degree == 0 && !include_d0 {
            continue;
        }

        let cells_with_vertex = &vertex_cells[v.index as usize];
        let mut missing: [u32; 3] = [u32::MAX; 3];
        let mut missing_len = 0usize;
        for &g in &v.key {
            if !cells_with_vertex.iter().any(|&c| c == g) {
                if missing_len < 3 {
                    missing[missing_len] = g;
                }
                missing_len += 1;
            }
        }

        let [a, b, c] = v.key;
        let gen_dist = |i: u32, j: u32| -> Option<f32> {
            let i = i as usize;
            let j = j as usize;
            if i >= generators.len() || j >= generators.len() {
                return None;
            }
            Some((generators[i] - generators[j]).length())
        };
        let dab = gen_dist(a, b);
        let dac = gen_dist(a, c);
        let dbc = gen_dist(b, c);

        eprintln!(
            "  vi={} degree={} pos=({:.6},{:.6},{:.6}) key=[{}, {}, {}] gen_dists=[{:?},{:?},{:?}] cells={:?} missing={:?}",
            v.index,
            v.degree,
            vertices[v.index as usize].x,
            vertices[v.index as usize].y,
            vertices[v.index as usize].z,
            a,
            b,
            c,
            dab,
            dac,
            dbc,
            cells_with_vertex,
            &missing[..missing_len.min(3)]
        );

        if knn_k > 0 {
            let Some(knn) = knn else {
                eprintln!("    knn debug requested but knn unavailable");
                continue;
            };
            let Some(scratch) = knn_scratch.as_mut() else {
                eprintln!("    knn debug requested but scratch unavailable");
                continue;
            };
            let k = knn_k;

            let mut check = |src: u32, dst: u32| {
                let Some((contains, rank, dot, min_dot)) =
                    knn_contains_with_rank(knn, generators, src, dst, k, scratch, &mut knn_out)
                else {
                    return;
                };
                eprintln!(
                    "    knn(k={}): {} contains {}? {} (rank={:?}, dot={:.6}, min_dot_in_knn={:.6})",
                    k, src, dst, contains, rank, dot, min_dot
                );
            };
            // Check directed kNN membership for each missing generator vs the other two.
            for &m in &missing[..missing_len.min(3)] {
                if m == u32::MAX {
                    continue;
                }
                for &other in &[a, b, c] {
                    if other == m {
                        continue;
                    }
                    check(m, other);
                }
            }
        }

        for (x, y, label) in [(a, b, "ab"), (a, c, "ac"), (b, c, "bc")] {
            let seg_xy = edge_segments_for_neighbor(x, y, cells, cell_indices, vertex_keys);
            let seg_yx = edge_segments_for_neighbor(y, x, cells, cell_indices, vertex_keys);
            let (xy_min, xy_max) = segment_len_stats(&seg_xy, vertices);
            let (yx_min, yx_max) = segment_len_stats(&seg_yx, vertices);
            eprintln!(
                "    edge {}: {}->{} segs={} lens=[{:?},{:?}] | {}->{} segs={} lens=[{:?},{:?}]",
                label,
                x,
                y,
                seg_xy.len(),
                xy_min,
                xy_max,
                y,
                x,
                seg_yx.len(),
                yx_min,
                yx_max
            );

            if knn_k > 0 {
                let one_sided = (seg_xy.is_empty() && !seg_yx.is_empty())
                    || (!seg_xy.is_empty() && seg_yx.is_empty());
                if one_sided {
                    if let Some((contains_xy, rank_xy, dot_xy, min_dot_xy)) = knn.and_then(|knn| {
                        knn_scratch.as_mut().and_then(|scratch| {
                            knn_contains_with_rank(
                                knn,
                                generators,
                                x,
                                y,
                                knn_k,
                                scratch,
                                &mut knn_out,
                            )
                        })
                    }) {
                        eprintln!(
                            "      knn(one-sided): {}->{} contains? {} (rank={:?}, dot={:.6}, min_dot_in_knn={:.6})",
                            x, y, contains_xy, rank_xy, dot_xy, min_dot_xy
                        );
                    }
                    if let Some((contains_yx, rank_yx, dot_yx, min_dot_yx)) = knn.and_then(|knn| {
                        knn_scratch.as_mut().and_then(|scratch| {
                            knn_contains_with_rank(
                                knn,
                                generators,
                                y,
                                x,
                                knn_k,
                                scratch,
                                &mut knn_out,
                            )
                        })
                    }) {
                        eprintln!(
                            "      knn(one-sided): {}->{} contains? {} (rank={:?}, dot={:.6}, min_dot_in_knn={:.6})",
                            y, x, contains_yx, rank_yx, dot_yx, min_dot_yx
                        );
                    }
                }
            }
        }

        if v.degree == 1 || v.degree == 2 {
            debug_low_incidence_edges(
                std::slice::from_ref(v),
                &vertex_cells,
                cells,
                cell_indices,
                vertex_keys,
            );
        }

        printed += 1;
        if printed >= max_print {
            break;
        }
    }
}

fn repair_bad_edges(
    edge_records: &[EdgeRecord],
    vertices: &[Vec3],
    cells: &[crate::VoronoiCell],
    cell_indices: &[u32],
    vertex_keys: &[VertexKey],
) -> Option<(Vec<crate::VoronoiCell>, Vec<u32>)> {
    let log = log_enabled();

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
        cells: &[crate::VoronoiCell],
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

    let mut new_cells: Vec<crate::VoronoiCell> = Vec::with_capacity(cells.len());
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
        new_cells.push(crate::VoronoiCell::new(start_u32, count_u16));
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

/// Result of merging coincident generators before Voronoi computation.
pub struct MergeResult {
    /// Points to use for Voronoi (representatives only, or all if no merges).
    pub effective_points: Vec<Vec3>,
    /// Maps original point index -> representative index in effective_points.
    /// If no merges occurred, this is just identity (0, 1, 2, ...).
    pub original_to_effective: Vec<usize>,
    /// Number of points that were merged (removed).
    pub num_merged: usize,
}

/// Find and merge coincident (near-identical) generators.
/// Uses strict radius-based merging to ensure no remaining pair is within threshold.
/// Returns effective points (representatives) and a mapping from original to effective indices.
///
/// NOTE: A potentially more efficient approach for the future would be to detect close
/// neighbors during cell construction and "borrow" the cell from the close neighbor
/// when a cell dies. This would avoid the preprocessing pass but requires more complex
/// recovery logic in the cell builder.
pub fn merge_close_points(points: &[Vec3], threshold: f32) -> MergeResult {
    let n = points.len();
    if n == 0 {
        return MergeResult {
            effective_points: Vec::new(),
            original_to_effective: Vec::new(),
            num_merged: 0,
        };
    }

    if threshold <= 0.0 {
        return MergeResult {
            effective_points: points.to_vec(),
            original_to_effective: (0..n).collect(),
            num_merged: 0,
        };
    }

    let threshold_sq = threshold * threshold;

    // Use a cube-map grid to restrict candidate pairs to small spatial bins.
    //
    // This avoids the very high constant factors of hashing ~N distinct 3D grid keys
    // (especially when `threshold` is tiny) while still checking all near pairs
    // by scanning the 3×3 neighborhood of each cell.
    //
    // Target fewer points per cell than the KNN grid to keep pairwise checks cheap.
    const PREPROCESS_TARGET_POINTS_PER_CELL: f64 = 12.0;
    let target = PREPROCESS_TARGET_POINTS_PER_CELL.max(1.0);
    let res = ((n as f64 / (6.0 * target)).sqrt() as usize).max(4);
    let grid = crate::cube_grid::CubeMapGrid::new(points, res);
    let num_cells = 6 * res * res;

    struct MergeDsu {
        parent: Vec<u32>,
    }

    impl MergeDsu {
        fn new(n: usize) -> Self {
            let mut parent = Vec::with_capacity(n);
            for i in 0..n {
                parent.push(i as u32);
            }
            Self { parent }
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

        // Order-dependent merge: keep the smaller representative as the parent.
        fn union_keep_min(&mut self, a: u32, b: u32) -> bool {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra == rb {
                return false;
            }
            let (min, max) = if ra <= rb { (ra, rb) } else { (rb, ra) };
            self.parent[max as usize] = min;
            true
        }
    }

    let mut dsu = MergeDsu::new(n);

    for cell in 0..num_cells {
        let a_points = grid.cell_points(cell);
        if a_points.len() >= 2 {
            for i in 0..a_points.len() {
                let ai = a_points[i] as usize;
                for j in (i + 1)..a_points.len() {
                    let aj = a_points[j] as usize;
                    let dist_sq = (points[ai] - points[aj]).length_squared();
                    if dist_sq < threshold_sq {
                        let _ = dsu.union_keep_min(ai as u32, aj as u32);
                    }
                }
            }
        }

        let neighbors = grid.cell_neighbors(cell);
        for &nb_u32 in neighbors {
            if nb_u32 == u32::MAX {
                continue;
            }
            let nb = nb_u32 as usize;
            if nb <= cell {
                continue;
            }
            let b_points = grid.cell_points(nb);
            if a_points.is_empty() || b_points.is_empty() {
                continue;
            }
            for &ai_u32 in a_points {
                let ai = ai_u32 as usize;
                for &bj_u32 in b_points {
                    let bj = bj_u32 as usize;
                    let dist_sq = (points[ai] - points[bj]).length_squared();
                    if dist_sq < threshold_sq {
                        let _ = dsu.union_keep_min(ai as u32, bj as u32);
                    }
                }
            }
        }
    }

    // Count how many unique representatives we have
    let mut rep_to_effective: Vec<Option<usize>> = vec![None; n];
    let mut effective_points = Vec::new();
    let mut original_to_effective = vec![0usize; n];

    for i in 0..n {
        let rep = dsu.find(i as u32) as usize;
        if rep_to_effective[rep].is_none() {
            rep_to_effective[rep] = Some(effective_points.len());
            effective_points.push(points[rep]);
        }
        original_to_effective[i] = rep_to_effective[rep].unwrap();
    }

    let num_merged = n - effective_points.len();

    MergeResult {
        effective_points,
        original_to_effective,
        num_merged,
    }
}

fn compute_voronoi_gpu_style_core(
    points: &[Vec3],
    termination: TerminationConfig,
    skip_preprocess: bool,
) -> crate::SphericalVoronoi {
    use timing::{Timer, TimingBuilder};

    let mut tb = TimingBuilder::new();

    // Preprocessing: merge close points
    let t = Timer::start();
    let (effective_points, merge_result) = if skip_preprocess {
        (points.to_vec(), None)
    } else {
        let threshold = std::env::var("S2V_PREPROCESS_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or_else(|| constants::merge_threshold_for_density(points.len()));
        let result = merge_close_points(points, threshold);
        let pts = if result.num_merged > 0 {
            result.effective_points.clone()
        } else {
            points.to_vec()
        };
        (pts, Some(result))
    };
    tb.set_preprocess(t.elapsed());
    let needs_remap = merge_result.as_ref().map_or(false, |r| r.num_merged > 0);
    if let Some(r) = &merge_result {
        if r.num_merged > 0 {
            if log_enabled() {
                eprintln!(
                    "preprocess: merged {} close generators ({} -> {})",
                    r.num_merged,
                    points.len(),
                    r.effective_points.len()
                );
            }
        }
    }

    // Build KNN on effective points (this is the timed grid build)
    let t = Timer::start();
    let knn = CubeMapGridKnn::new(&effective_points);
    tb.set_knn_build(t.elapsed());
    #[cfg(feature = "timing")]
    tb.set_knn_build_sub(knn.grid_build_timings().clone());

    // Build cells using sharded live dedup
    let t = Timer::start();
    let sharded = live_dedup::build_cells_sharded_live_dedup(&effective_points, &knn, termination);
    tb.set_cell_construction(t.elapsed(), sharded.cell_sub.clone().into_sub_phases());

    let t = Timer::start();
    let (all_vertices, all_vertex_keys, bad_edges, eff_cells, eff_cell_indices, dedup_sub) =
        live_dedup::assemble_sharded_live_dedup(sharded);
    tb.set_dedup(t.elapsed(), dedup_sub);

    let debug_pre: usize = std::env::var("S2V_DEBUG_LOW_DEGREE_PRE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    debug_low_degree_vertices(
        "pre-repair",
        &effective_points,
        Some(&knn),
        &all_vertices,
        &all_vertex_keys,
        &eff_cells,
        &eff_cell_indices,
        debug_pre,
    );

    if log_enabled() && !bad_edges.is_empty() {
        eprintln!("edge checks (live): bad_edges={}", bad_edges.len());
        for record in &bad_edges {
            let (a, b) = unpack_edge(record.key.as_u64());
            eprintln!("  edge=({},{}) reason={:?}", a, b, record.reason);
        }
    }

    let repair_edges_storage: Vec<EdgeRecord> = bad_edges
        .iter()
        .map(|b| EdgeRecord { key: b.key })
        .collect();
    if log_enabled() {
        eprintln!(
            "edge repair: using live bad edges (count={})",
            repair_edges_storage.len()
        );
    }

    let t = Timer::start();
    let (eff_cells, eff_cell_indices) = if let Some((cells, indices)) = repair_bad_edges(
        &repair_edges_storage,
        &all_vertices,
        &eff_cells,
        &eff_cell_indices,
        &all_vertex_keys,
    ) {
        (cells, indices)
    } else {
        (eff_cells, eff_cell_indices)
    };
    tb.set_edge_repair(t.elapsed());

    let debug_post: usize = std::env::var("S2V_DEBUG_LOW_DEGREE_POST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    debug_low_degree_vertices(
        "post-repair",
        &effective_points,
        Some(&knn),
        &all_vertices,
        &all_vertex_keys,
        &eff_cells,
        &eff_cell_indices,
        debug_post,
    );

    // Remap cells back to original point indices if we merged
    let t = Timer::start();
    let (cells, cell_indices) = if needs_remap {
        use crate::VoronoiCell;
        let merge_result = merge_result.as_ref().unwrap();

        // Each original point maps to an effective point's cell
        let mut new_cells = Vec::with_capacity(points.len());
        let mut new_cell_indices: Vec<u32> = Vec::new();

        for orig_idx in 0..points.len() {
            let eff_idx = merge_result.original_to_effective[orig_idx];
            let eff_cell = &eff_cells[eff_idx];

            let start = u32::try_from(new_cell_indices.len())
                .expect("cell index buffer exceeds u32 capacity");
            let eff_start = eff_cell.vertex_start();
            let eff_end = eff_start + eff_cell.vertex_count();
            new_cell_indices.extend_from_slice(&eff_cell_indices[eff_start..eff_end]);

            let count_u16 =
                u16::try_from(eff_cell.vertex_count()).expect("cell vertex count exceeds u16");
            new_cells.push(VoronoiCell::new(start, count_u16));
        }
        (new_cells, new_cell_indices)
    } else {
        (eff_cells, eff_cell_indices)
    };

    let voronoi =
        crate::SphericalVoronoi::from_raw_parts(points.to_vec(), all_vertices, cells, cell_indices);
    tb.set_assemble(t.elapsed());

    // Report timing if feature enabled
    let timings = tb.finish();
    timings.report(points.len());

    voronoi
}

/// Compute spherical Voronoi diagram using the GPU-style algorithm.
///
/// Uses adaptive k-NN (12→24→48→full) with early termination.
///
/// Timing output is controlled by the `timing` feature flag:
/// ```bash
/// cargo run --release --features timing
/// ```
pub fn compute_voronoi_gpu_style(points: &[Vec3]) -> crate::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, false)
}

/// Compute spherical Voronoi with custom termination config (for benchmarks).
pub fn compute_voronoi_gpu_style_with_termination(
    points: &[Vec3],
    termination: TerminationConfig,
) -> crate::SphericalVoronoi {
    compute_voronoi_gpu_style_core(points, termination, false)
}

/// Compute spherical Voronoi WITHOUT preprocessing (merge close points).
/// For benchmarking only - assumes points are already well-spaced.
pub fn compute_voronoi_gpu_style_no_preprocess(points: &[Vec3]) -> crate::SphericalVoronoi {
    let termination = TerminationConfig::default();
    compute_voronoi_gpu_style_core(points, termination, true)
}

// NOTE: benchmark_voronoi function was removed during crate extraction.
// It compared knn_clipping vs qhull backends and belongs in hex3's benchmarks.
