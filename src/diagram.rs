//! Spherical Voronoi diagram storage and access.

use crate::UnitVec3;
use glam::Vec3;

/// A spherical Voronoi diagram on the unit sphere.
///
/// The diagram consists of:
/// - Generator points (input points, one per cell)
/// - Vertices (intersection points of cell boundaries)
/// - Cells (regions closest to each generator)
///
/// Each cell is represented as a list of vertex indices forming a spherical polygon.
#[derive(Debug, Clone)]
pub struct SphericalVoronoi {
    /// Generator points (input), one per cell.
    pub generators: Vec<UnitVec3>,

    /// Voronoi vertices (shared between cells).
    pub vertices: Vec<UnitVec3>,

    /// Per-cell data: (start_index, vertex_count) into cell_indices.
    cells: Vec<CellData>,

    /// Flattened vertex indices for all cells.
    cell_indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct CellData {
    start: u32,
    len: u16,
}

impl SphericalVoronoi {
    /// Create an empty diagram with generators but no computed cells.
    ///
    /// This is a small test helper for constructing diagrams without running the backend.
    #[cfg(test)]
    pub(crate) fn empty(generators: Vec<UnitVec3>) -> Self {
        let n = generators.len();
        Self {
            generators,
            vertices: Vec::new(),
            cells: vec![CellData { start: 0, len: 0 }; n],
            cell_indices: Vec::new(),
        }
    }

    /// Create a diagram from raw parts.
    ///
    /// This is used by the computation backends to construct the final diagram.
    pub fn from_parts(
        generators: Vec<UnitVec3>,
        vertices: Vec<UnitVec3>,
        cells: Vec<VoronoiCell>,
        cell_indices: Vec<u32>,
    ) -> Self {
        Self {
            generators,
            vertices,
            cells: cells
                .into_iter()
                .map(|c| CellData {
                    start: c.vertex_start,
                    len: c.vertex_count,
                })
                .collect(),
            cell_indices,
        }
    }

    /// Create a diagram from raw parts using VoronoiCell and Vec3.
    ///
    /// This is the internal constructor used by knn_clipping backend.
    pub(crate) fn from_raw_parts(
        generators: Vec<Vec3>,
        vertices: Vec<Vec3>,
        cells: Vec<VoronoiCell>,
        cell_indices: Vec<u32>,
    ) -> Self {
        Self {
            generators: generators
                .into_iter()
                .map(|v| UnitVec3::new(v.x, v.y, v.z))
                .collect(),
            vertices: vertices
                .into_iter()
                .map(|v| UnitVec3::new(v.x, v.y, v.z))
                .collect(),
            cells: cells
                .into_iter()
                .map(|c| CellData {
                    start: c.vertex_start,
                    len: c.vertex_count,
                })
                .collect(),
            cell_indices,
        }
    }

    /// Number of cells (same as number of generators).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.generators.len()
    }

    /// Number of vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get a view of a specific cell.
    #[inline]
    pub fn cell(&self, index: usize) -> CellView<'_> {
        let data = &self.cells[index];
        let start = data.start as usize;
        let end = start + data.len as usize;
        CellView {
            generator_index: index,
            vertex_indices: &self.cell_indices[start..end],
        }
    }

    /// Iterate over all cells.
    pub fn iter_cells(&self) -> impl Iterator<Item = CellView<'_>> {
        (0..self.num_cells()).map(move |i| self.cell(i))
    }

    /// Get the generator (center point) of a cell.
    #[inline]
    pub fn generator(&self, index: usize) -> UnitVec3 {
        self.generators[index]
    }

    /// Get a vertex by index.
    #[inline]
    pub fn vertex(&self, index: usize) -> UnitVec3 {
        self.vertices[index]
    }
}

/// A view into a single Voronoi cell.
#[derive(Debug, Clone, Copy)]
pub struct CellView<'a> {
    /// Index of the generator point for this cell.
    pub generator_index: usize,

    /// Indices of vertices forming the cell boundary (in order).
    pub vertex_indices: &'a [u32],
}

impl<'a> CellView<'a> {
    /// Number of vertices in this cell.
    #[inline]
    pub fn len(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Returns true if this cell has no vertices (degenerate).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertex_indices.is_empty()
    }
}

/// Internal cell storage (used by backends during construction).
#[derive(Debug, Clone, Copy)]
pub struct VoronoiCell {
    vertex_start: u32,
    vertex_count: u16,
}

impl VoronoiCell {
    /// Create a new VoronoiCell.
    #[inline]
    pub fn new(vertex_start: u32, vertex_count: u16) -> Self {
        Self {
            vertex_start,
            vertex_count,
        }
    }

    /// Start index into the flat cell_indices buffer.
    #[inline]
    pub fn vertex_start(&self) -> usize {
        self.vertex_start as usize
    }

    /// Number of vertices for this cell.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_diagram() {
        let generators = vec![UnitVec3::new(1.0, 0.0, 0.0), UnitVec3::new(0.0, 1.0, 0.0)];
        let diagram = SphericalVoronoi::empty(generators);

        assert_eq!(diagram.num_cells(), 2);
        assert_eq!(diagram.num_vertices(), 0);
        assert!(diagram.cell(0).is_empty());
        assert!(diagram.cell(1).is_empty());
    }

    #[test]
    fn test_from_parts() {
        let generators = vec![UnitVec3::new(1.0, 0.0, 0.0), UnitVec3::new(-1.0, 0.0, 0.0)];
        let vertices = vec![
            UnitVec3::new(0.0, 1.0, 0.0),
            UnitVec3::new(0.0, -1.0, 0.0),
            UnitVec3::new(0.0, 0.0, 1.0),
            UnitVec3::new(0.0, 0.0, -1.0),
        ];
        // Cell 0 uses vertices 0,2,1,3; Cell 1 uses same but different order
        let cells = vec![VoronoiCell::new(0, 4), VoronoiCell::new(4, 4)];
        let cell_indices = vec![0, 2, 1, 3, 0, 3, 1, 2];

        let diagram = SphericalVoronoi::from_parts(generators, vertices, cells, cell_indices);

        assert_eq!(diagram.num_cells(), 2);
        assert_eq!(diagram.num_vertices(), 4);
        assert_eq!(diagram.cell(0).len(), 4);
        assert_eq!(diagram.cell(1).len(), 4);
    }
}
