//! Debug logging helpers for low-degree vertices.

use glam::Vec3;

use super::edge_repair::{edge_segments_for_neighbor, segment_len_stats, shared_neighbor};
use super::CubeMapGridKnn;
use super::KNN_RESTART_MAX;
use crate::knn_clipping::cell_builder::VertexKey;
use crate::VoronoiCell;

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

fn build_vertex_cells(
    cells: &[VoronoiCell],
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

fn debug_low_incidence_edges(
    low_vertices: &[LowIncidenceVertex],
    vertex_cells: &[Vec<u32>],
    cells: &[VoronoiCell],
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

pub(super) fn debug_low_degree_vertices(
    label: &str,
    generators: &[Vec3],
    knn: Option<&CubeMapGridKnn<'_>>,
    vertices: &[Vec3],
    vertex_keys: &[VertexKey],
    cells: &[VoronoiCell],
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
