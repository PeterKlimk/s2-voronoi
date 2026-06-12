//! Planar Voronoi diagram storage and access.

use crate::diagram::VoronoiCell;
use bytemuck::{Pod, Zeroable};

/// A point in the plane.
///
/// Small `#[repr(C)]` representation with a stable layout, mirroring
/// [`crate::UnitVec3`] for the spherical API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanePoint {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl PlanePoint {
    /// Create a new point.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl From<[f32; 2]> for PlanePoint {
    #[inline]
    fn from([x, y]: [f32; 2]) -> Self {
        Self::new(x, y)
    }
}

impl From<PlanePoint> for [f32; 2] {
    #[inline]
    fn from(p: PlanePoint) -> Self {
        [p.x, p.y]
    }
}

/// Trait for types usable as planar input points (zero-copy input from
/// various math libraries), mirroring [`crate::UnitVec3Like`].
pub trait PlanePointLike {
    /// X coordinate.
    fn x(&self) -> f32;
    /// Y coordinate.
    fn y(&self) -> f32;
}

impl PlanePointLike for PlanePoint {
    #[inline]
    fn x(&self) -> f32 {
        self.x
    }
    #[inline]
    fn y(&self) -> f32 {
        self.y
    }
}

impl PlanePointLike for [f32; 2] {
    #[inline]
    fn x(&self) -> f32 {
        self[0]
    }
    #[inline]
    fn y(&self) -> f32 {
        self[1]
    }
}

impl PlanePointLike for (f32, f32) {
    #[inline]
    fn x(&self) -> f32 {
        self.0
    }
    #[inline]
    fn y(&self) -> f32 {
        self.1
    }
}

#[cfg(feature = "glam")]
impl PlanePointLike for glam::Vec2 {
    #[inline]
    fn x(&self) -> f32 {
        self.x
    }
    #[inline]
    fn y(&self) -> f32 {
        self.y
    }
}

/// An axis-aligned bounding rectangle: the planar Voronoi domain.
///
/// All input points must lie inside (or on the boundary of) the rect; hull
/// cells are clipped to it. See `compute_plane`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlaneRect {
    /// Minimum corner (inclusive).
    pub min: PlanePoint,
    /// Maximum corner (inclusive).
    pub max: PlanePoint,
}

impl PlaneRect {
    /// Create a rect from its min/max corners.
    #[inline]
    pub const fn new(min: PlanePoint, max: PlanePoint) -> Self {
        Self { min, max }
    }

    /// The unit square `[0, 1] x [0, 1]`.
    #[inline]
    pub const fn unit() -> Self {
        Self::new(PlanePoint::new(0.0, 0.0), PlanePoint::new(1.0, 1.0))
    }

    /// Rect width (`max.x - min.x`).
    #[inline]
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Rect height (`max.y - min.y`).
    #[inline]
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// True when `p` lies inside the rect (boundary inclusive).
    #[inline]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.min.x && x <= self.max.x && y >= self.min.y && y <= self.max.y
    }
}

/// A planar Voronoi diagram over a bounded rectangle.
///
/// The diagram is a strict subdivision of the rect: every cell is a convex
/// polygon, hull cells are clipped to the rect boundary, and cell areas sum
/// to the rect area. Mirrors [`crate::SphericalVoronoi`]'s storage and
/// access patterns.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanarVoronoi {
    /// Generator points (input), one per cell.
    generators: Vec<PlanePoint>,

    /// Voronoi vertices (shared between cells), in rect coordinates.
    vertices: Vec<PlanePoint>,

    /// Per-cell data: (start_index, vertex_count) into cell_indices.
    cells: Vec<CellData>,

    /// Flattened vertex indices for all cells.
    cell_indices: Vec<u32>,

    /// Canonical cell index per cell when generators within the planar weld
    /// radius (~1e-6 of the longer rect side; always including exact
    /// duplicates) were welded; `None` when every generator owns its own
    /// cell. Welded twins alias their canonical cell's boundary storage.
    weld_map: Option<Vec<u32>>,

    /// The domain rectangle the diagram subdivides.
    rect: PlaneRect,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CellData {
    start: u32,
    len: u16,
}

impl PlanarVoronoi {
    pub(crate) fn from_raw_parts(
        generators: Vec<PlanePoint>,
        vertices: Vec<PlanePoint>,
        cells: Vec<VoronoiCell>,
        cell_indices: Vec<u32>,
        weld_map: Option<Vec<u32>>,
        rect: PlaneRect,
    ) -> Self {
        debug_assert!(weld_map.as_ref().is_none_or(|m| m.len() == cells.len()));
        Self {
            generators,
            vertices,
            cells: cells
                .into_iter()
                .map(|c| CellData {
                    start: c.vertex_start() as u32,
                    len: c.vertex_count() as u16,
                })
                .collect(),
            cell_indices,
            weld_map,
            rect,
        }
    }

    /// Number of cells (same as number of input generators).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.generators.len()
    }

    /// Number of shared Voronoi vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Borrow all generator points.
    #[inline]
    pub fn generators(&self) -> &[PlanePoint] {
        &self.generators
    }

    /// Borrow all Voronoi vertices.
    #[inline]
    pub fn vertices(&self) -> &[PlanePoint] {
        &self.vertices
    }

    /// Get one generator point.
    #[inline]
    pub fn generator(&self, i: usize) -> PlanePoint {
        self.generators[i]
    }

    /// Get one Voronoi vertex.
    #[inline]
    pub fn vertex(&self, i: usize) -> PlanePoint {
        self.vertices[i]
    }

    /// Vertex indices of cell `i`, in counterclockwise polygon order.
    #[inline]
    pub fn cell(&self, i: usize) -> &[u32] {
        let c = self.cells[i];
        &self.cell_indices[c.start as usize..c.start as usize + c.len as usize]
    }

    /// Iterate over all cells as vertex-index slices.
    pub fn iter_cells(&self) -> impl Iterator<Item = &[u32]> + '_ {
        (0..self.num_cells()).map(move |i| self.cell(i))
    }

    /// Canonical cell index per generator when generators within the planar
    /// weld radius (always including exact duplicates) were welded
    /// (`weld_map()[i] == i` for canonical cells), `None` when no welds
    /// occurred.
    #[inline]
    pub fn weld_map(&self) -> Option<&[u32]> {
        self.weld_map.as_deref()
    }

    /// The domain rectangle this diagram subdivides.
    #[inline]
    pub fn rect(&self) -> PlaneRect {
        self.rect
    }
}
