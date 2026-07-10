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

/// Sentinel for an edge with no paired cell.
///
/// On the sphere, strictly valid diagrams (see
/// [`crate::validation::validate`]) contain none. On the plane, edges on the
/// domain rectangle's boundary legitimately have no neighbor and always
/// carry this sentinel; interior edges of a strictly valid planar diagram
/// never do.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Per-cell neighbor lists, aligned with cell boundary edges.
///
/// Neighbor entries are **canonical** cell indices: edges belong to canonical
/// cells, so a welded twin never appears as a neighbor (its canonical cell
/// does), and querying a twin returns its canonical cell's list.
///
/// With the `serde` feature, deserialization is CHECKED (spans must lie
/// inside the neighbor buffer); malformed input fails with an error instead
/// of constructing an adjacency that panics later.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "CellAdjacencyWire"))]
pub struct CellAdjacency {
    cells: Vec<(u32, u16)>,
    neighbors: Vec<u32>,
}

/// Wire mirror for checked deserialization; field names must match
/// [`CellAdjacency`] (the derived `Serialize` defines the format).
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct CellAdjacencyWire {
    cells: Vec<(u32, u16)>,
    neighbors: Vec<u32>,
}

#[cfg(feature = "serde")]
impl TryFrom<CellAdjacencyWire> for CellAdjacency {
    type Error = String;

    fn try_from(w: CellAdjacencyWire) -> Result<Self, String> {
        for (i, &(start, len)) in w.cells.iter().enumerate() {
            if len != 0 && start as usize + len as usize > w.neighbors.len() {
                return Err(format!(
                    "adjacency cell {i} span [{start}..+{len}) exceeds neighbor buffer len {}",
                    w.neighbors.len()
                ));
            }
        }
        Ok(CellAdjacency {
            cells: w.cells,
            neighbors: w.neighbors,
        })
    }
}

impl CellAdjacency {
    /// Neighbor cells of `cell`, one per boundary edge.
    ///
    /// Entry `k` is the cell across the edge from boundary vertex `k` to
    /// vertex `k + 1` (cyclic), or [`NO_NEIGHBOR`] for a defective edge.
    ///
    /// # Panics
    ///
    /// Panics if `cell >= self.num_cells()`. Use [`Self::get_neighbors_of`]
    /// for checked access to user-supplied indices.
    #[inline]
    #[track_caller]
    pub fn neighbors_of(&self, cell: usize) -> &[u32] {
        let (start, len) = self.cells[cell];
        if len == 0 {
            return &[];
        }
        &self.neighbors[start as usize..start as usize + len as usize]
    }

    /// Checked form of [`Self::neighbors_of`]: `None` when `cell` is out of
    /// bounds.
    #[inline]
    pub fn get_neighbors_of(&self, cell: usize) -> Option<&[u32]> {
        (cell < self.cells.len()).then(|| self.neighbors_of(cell))
    }

    /// Number of cells (same as the source diagram).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// True when every edge has a paired neighbor (no [`NO_NEIGHBOR`]
    /// entries). Always true for strictly valid spherical diagrams; planar
    /// diagrams with boundary-touching cells are never complete (rect
    /// boundary edges have no neighbor by construction).
    pub fn is_complete(&self) -> bool {
        self.neighbors.iter().all(|&n| n != NO_NEIGHBOR)
    }
}

/// Shared adjacency construction over raw cell layout: ranges into the flat
/// index buffer plus the weld-canonical mapping. Purely combinatorial, so
/// the spherical and planar diagrams use the same core.
fn build_adjacency_from_parts(
    num_cells: usize,
    cell_range: impl Fn(usize) -> (u32, u16),
    cell_indices: &[u32],
    canonical: impl Fn(usize) -> usize,
) -> CellAdjacency {
    // The adjacency mirrors the diagram's cell layout (including weld
    // aliasing), so twins share their canonical cell's neighbor slice.
    let mut cells: Vec<(u32, u16)> = Vec::with_capacity(num_cells);
    let mut neighbors_len = 0usize;
    for i in 0..num_cells {
        let (start, len) = cell_range(i);
        cells.push((start, len));
        if canonical(i) == i {
            neighbors_len = neighbors_len.max(start as usize + len as usize);
        }
    }
    let mut neighbors = vec![NO_NEIGHBOR; neighbors_len];

    // Directed edge records: (undirected vertex-pair key, cell, edge pos),
    // canonical cells only. Equal keys pair up after sorting.
    let mut records: Vec<(u64, u64)> = Vec::new();
    for i in 0..num_cells {
        if canonical(i) != i {
            continue;
        }
        let (start, len) = cell_range(i);
        let len = len as usize;
        if len < 2 {
            continue;
        }
        let indices = &cell_indices[start as usize..start as usize + len];
        for k in 0..len {
            let a = indices[k];
            let b = indices[(k + 1) % len];
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
        // Exactly two uses = a properly shared edge; anything else (a
        // boundary edge on the plane, or a defective edge) stays NO_NEIGHBOR.
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
        build_adjacency_from_parts(
            self.num_cells(),
            |i| (self.cell_start(i), self.cell(i).len() as u16),
            self.cell_indices_raw(),
            |i| self.canonical_cell_index(i),
        )
    }
}

#[inline]
fn unpack(payload: u64) -> (usize, usize) {
    ((payload >> 16) as usize, (payload & 0xffff) as usize)
}
