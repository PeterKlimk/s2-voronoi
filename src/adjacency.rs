//! Cell adjacency (Voronoi neighbors / Delaunay edges).
//!
//! Built on demand from a computed diagram by pairing shared boundary edges.
//! The neighbor storage mirrors the diagram's cell layout, so entry `k` of a
//! cell's neighbor list is the cell across boundary edge
//! `(vertex_indices[k], vertex_indices[(k + 1) % len])`, and welded twins
//! alias their canonical cell's adjacency exactly like they alias its
//! boundary.

use crate::SphericalVoronoi;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Sentinel for an edge with no paired cell (unpaired or overused edge).
///
/// Strictly valid diagrams (see [`crate::validation::validate`]) contain
/// none; it can appear when adjacency is built from a degraded diagram.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Per-cell neighbor lists, aligned with cell boundary edges.
///
/// Neighbor entries are **canonical** cell indices: edges belong to canonical
/// cells, so a welded twin never appears as a neighbor (its canonical cell
/// does), and querying a twin returns its canonical cell's list.
#[derive(Debug, Clone)]
pub struct CellAdjacency {
    cells: Vec<(u32, u16)>,
    neighbors: Vec<u32>,
}

impl CellAdjacency {
    /// Neighbor cells of `cell`, one per boundary edge.
    ///
    /// Entry `k` is the cell across the edge from boundary vertex `k` to
    /// vertex `k + 1` (cyclic), or [`NO_NEIGHBOR`] for a defective edge.
    #[inline]
    pub fn neighbors_of(&self, cell: usize) -> &[u32] {
        let (start, len) = self.cells[cell];
        if len == 0 {
            return &[];
        }
        &self.neighbors[start as usize..start as usize + len as usize]
    }

    /// Number of cells (same as the source diagram).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// True when every edge has a paired neighbor (no [`NO_NEIGHBOR`]
    /// entries). Always true for strictly valid diagrams.
    pub fn is_complete(&self) -> bool {
        self.neighbors.iter().all(|&n| n != NO_NEIGHBOR)
    }
}

impl SphericalVoronoi {
    /// Build the cell adjacency (Voronoi neighbor graph / Delaunay edges).
    ///
    /// One pass over all boundary edges plus a sort: O(E log E), independent
    /// of the main computation. The adjacency of cell `i` is aligned with its
    /// boundary: `adjacency.neighbors_of(i)[k]` lies across the edge
    /// `(cell(i).vertex_indices[k], cell(i).vertex_indices[k + 1])` (cyclic).
    ///
    /// The undirected neighbor pairs `(i, neighbors_of(i)[k])` over canonical
    /// cells are exactly the Delaunay edges of the generator set.
    pub fn build_adjacency(&self) -> CellAdjacency {
        let num_cells = self.num_cells();

        // The adjacency mirrors the diagram's cell layout (including weld
        // aliasing), so twins share their canonical cell's neighbor slice.
        let mut cells: Vec<(u32, u16)> = Vec::with_capacity(num_cells);
        let mut neighbors_len = 0usize;
        for i in 0..num_cells {
            let view = self.cell(i);
            let start = self.cell_start(i);
            cells.push((start, view.len() as u16));
            if self.canonical_cell_index(i) == i {
                neighbors_len = neighbors_len.max(start as usize + view.len());
            }
        }
        let mut neighbors = vec![NO_NEIGHBOR; neighbors_len];

        // Directed edge records: (undirected vertex-pair key, cell, edge pos),
        // canonical cells only. Equal keys pair up after sorting.
        let mut records: Vec<(u64, u64)> = Vec::new();
        for i in 0..num_cells {
            if self.canonical_cell_index(i) != i {
                continue;
            }
            let view = self.cell(i);
            let len = view.len();
            if len < 2 {
                continue;
            }
            for k in 0..len {
                let a = view.vertex_indices[k];
                let b = view.vertex_indices[(k + 1) % len];
                if a == b {
                    continue;
                }
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let key = ((lo as u64) << 32) | hi as u64;
                records.push((key, ((i as u64) << 16) | k as u64));
            }
        }

        #[cfg(feature = "parallel")]
        records.par_sort_unstable();
        #[cfg(not(feature = "parallel"))]
        records.sort_unstable();

        let mut run_start = 0usize;
        while run_start < records.len() {
            let key = records[run_start].0;
            let mut run_end = run_start + 1;
            while run_end < records.len() && records[run_end].0 == key {
                run_end += 1;
            }
            // Exactly two uses = a properly shared edge; anything else
            // (boundary or overused edge in a degraded diagram) stays
            // NO_NEIGHBOR.
            if run_end - run_start == 2 {
                let (cell_a, pos_a) = unpack(records[run_start].1);
                let (cell_b, pos_b) = unpack(records[run_start + 1].1);
                let (start_a, _) = cells[cell_a];
                let (start_b, _) = cells[cell_b];
                neighbors[start_a as usize + pos_a] = cell_b as u32;
                neighbors[start_b as usize + pos_b] = cell_a as u32;
            }
            run_start = run_end;
        }

        CellAdjacency { cells, neighbors }
    }
}

#[inline]
fn unpack(payload: u64) -> (usize, usize) {
    ((payload >> 16) as usize, (payload & 0xffff) as usize)
}
