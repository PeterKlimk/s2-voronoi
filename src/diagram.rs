//! Spherical Voronoi diagram storage and access.

use crate::SpherePoint;
use glam::Vec3;

/// A spherical Voronoi diagram on the unit sphere.
///
/// The diagram consists of:
/// - Generator points (input points, one per cell)
/// - Vertices (intersection points of cell boundaries)
/// - Cells (regions closest to each generator)
///
/// Each cell is represented as a list of vertex indices forming a spherical polygon.
///
/// With the `serde` feature, deserialization is CHECKED: the wire data must
/// satisfy the structural invariants the accessors rely on (a nonempty site
/// set, cell spans inside the index buffer, live vertex indices in range, and
/// welded twins aliasing their canonical cell span), and malformed input fails
/// with a descriptive error instead of constructing a diagram that panics
/// later. Semantic validity (edge pairing, Euler
/// characteristic, …) is NOT re-checked on deserialization; run
/// [`crate::validation::validate`] if the data's provenance is untrusted. The
/// serialized form mirrors the storage layout and may change in minor
/// releases pre-1.0.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "wire::SphericalVoronoiWire"))]
pub struct SphericalVoronoi {
    /// Generator points (input), one per cell.
    generators: Vec<SpherePoint>,

    /// Voronoi vertices (shared between cells).
    vertices: Vec<SpherePoint>,

    /// Per-cell data: (start_index, vertex_count) into cell_indices.
    cells: Vec<CellData>,

    /// Flattened vertex indices for all cells.
    cell_indices: Vec<u32>,

    /// Canonical cell index per cell when generators were welded, `None` when
    /// every generator owns its own cell. Welded twins alias their canonical
    /// cell's boundary storage; `weld_map[i] == i` for canonical cells.
    weld_map: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CellData {
    start: u32,
    len: u16,
}

impl SphericalVoronoi {
    /// Create an empty diagram with generators but no computed cells.
    ///
    /// This is a small test helper for constructing diagrams without running the backend.
    #[cfg(test)]
    pub(crate) fn empty(generators: Vec<SpherePoint>) -> Self {
        let n = generators.len();
        Self {
            generators,
            vertices: Vec::new(),
            cells: vec![CellData { start: 0, len: 0 }; n],
            cell_indices: Vec::new(),
            weld_map: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_cells_and_indices(
        generators: Vec<SpherePoint>,
        vertices: Vec<SpherePoint>,
        cell_starts: Vec<u32>,
        cell_counts: Vec<u16>,
        cell_indices: Vec<u32>,
    ) -> Self {
        debug_assert_eq!(cell_starts.len(), cell_counts.len());
        Self {
            generators,
            vertices,
            cells: cell_starts
                .into_iter()
                .zip(cell_counts)
                .map(|(start, len)| CellData { start, len })
                .collect(),
            cell_indices,
            weld_map: None,
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
        weld_map: Option<Vec<u32>>,
    ) -> Self {
        debug_assert!(weld_map.as_ref().is_none_or(|m| m.len() == cells.len()));
        #[cfg(feature = "profiling")]
        {
            for &generator in &generators {
                crate::point_audit::record_vec3(
                    crate::point_audit::PointProducer::FinalGenerator,
                    generator,
                );
            }
            for &vertex in &vertices {
                crate::point_audit::record_vec3(
                    crate::point_audit::PointProducer::FinalVertex,
                    vertex,
                );
            }
        }
        // SAFETY: all backend producers use the audited f64-normalize-then-
        // round storage rule before assembly. Reconciliation only selects or
        // reuses those certified positions.
        let generators = unsafe { crate::types::sphere_points_from_vec3(generators) };
        // SAFETY: same producer invariant as generators; Local3d minted
        // vertices use the same canonical storage rule.
        let vertices = unsafe { crate::types::sphere_points_from_vec3(vertices) };
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
            weld_map,
        }
    }

    /// Number of cells (same as number of generators).
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.generators.len()
    }

    /// Borrow all generator points.
    #[inline]
    pub fn generators(&self) -> &[SpherePoint] {
        &self.generators
    }

    /// Borrow generator coordinates as tightly packed xyz triples.
    #[inline]
    pub fn generators_xyz(&self) -> &[[f32; 3]] {
        crate::types::sphere_points_as_xyz(&self.generators)
    }

    /// Number of vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Borrow all Voronoi vertices.
    #[inline]
    pub fn vertices(&self) -> &[SpherePoint] {
        &self.vertices
    }

    /// Borrow Voronoi vertex coordinates as tightly packed xyz triples.
    #[inline]
    pub fn vertices_xyz(&self) -> &[[f32; 3]] {
        crate::types::sphere_points_as_xyz(&self.vertices)
    }

    /// Start offset of a cell's range in the shared index buffer (used by
    /// adjacency construction to mirror the cell layout).
    #[inline]
    pub(crate) fn cell_start(&self, index: usize) -> u32 {
        self.cells[index].start
    }

    #[inline]
    pub(crate) fn cell_indices_raw(&self) -> &[u32] {
        &self.cell_indices
    }

    /// Get a view of a specific cell.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()`. Use [`Self::get_cell`] for
    /// checked access to user-supplied indices.
    #[inline]
    #[track_caller]
    pub fn cell(&self, index: usize) -> CellView<'_> {
        let data = &self.cells[index];
        let start = data.start as usize;
        let end = start + data.len as usize;
        CellView {
            generator_index: index,
            vertex_indices: &self.cell_indices[start..end],
        }
    }

    /// Checked form of [`Self::cell`]: `None` when `index` is out of bounds.
    #[inline]
    pub fn get_cell(&self, index: usize) -> Option<CellView<'_>> {
        (index < self.num_cells()).then(|| self.cell(index))
    }

    /// Iterate over all cells.
    pub fn iter_cells(&self) -> impl Iterator<Item = CellView<'_>> {
        (0..self.num_cells()).map(move |i| self.cell(i))
    }

    /// Get the generator (center point) of a cell.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()`. Use [`Self::get_generator`] for
    /// checked access to user-supplied indices.
    #[inline]
    #[track_caller]
    pub fn generator(&self, index: usize) -> SpherePoint {
        self.generators[index]
    }

    /// Checked form of [`Self::generator`]: `None` when `index` is out of
    /// bounds.
    #[inline]
    pub fn get_generator(&self, index: usize) -> Option<SpherePoint> {
        self.generators.get(index).copied()
    }

    /// Get a vertex by index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_vertices()`. Use [`Self::get_vertex`] for
    /// checked access to user-supplied indices.
    #[inline]
    #[track_caller]
    pub fn vertex(&self, index: usize) -> SpherePoint {
        self.vertices[index]
    }

    /// Checked form of [`Self::vertex`]: `None` when `index` is out of bounds.
    #[inline]
    pub fn get_vertex(&self, index: usize) -> Option<SpherePoint> {
        self.vertices.get(index).copied()
    }

    /// Canonical cell index for a cell.
    ///
    /// Generators welded together during preprocessing share one cell; the
    /// canonical index is the smallest input index in the weld class, and the
    /// welded cells alias the canonical cell's boundary. For non-welded cells
    /// (the overwhelmingly common case) this is the identity.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_cells()` (historically the no-weld case
    /// silently returned the invalid index back). Use
    /// [`Self::get_canonical_cell_index`] for checked access.
    #[inline]
    #[track_caller]
    pub fn canonical_cell_index(&self, index: usize) -> usize {
        self.get_canonical_cell_index(index).unwrap_or_else(|| {
            panic!(
                "cell index {index} out of bounds (num_cells {})",
                self.num_cells()
            )
        })
    }

    /// Checked form of [`Self::canonical_cell_index`]: `None` when `index` is
    /// out of bounds.
    #[inline]
    pub fn get_canonical_cell_index(&self, index: usize) -> Option<usize> {
        if index >= self.num_cells() {
            return None;
        }
        Some(match &self.weld_map {
            Some(map) => map[index] as usize,
            None => index,
        })
    }

    /// Canonical-cell mapping when generators were welded, `None` otherwise.
    ///
    /// When present, `weld_map()[i]` is the canonical cell index for cell `i`
    /// (see [`Self::canonical_cell_index`]).
    #[inline]
    pub fn weld_map(&self) -> Option<&[u32]> {
        self.weld_map.as_deref()
    }

    /// Number of cells that are welded twins of another (canonical) cell.
    pub fn welded_twin_count(&self) -> usize {
        match &self.weld_map {
            Some(map) => map
                .iter()
                .enumerate()
                .filter(|&(i, &c)| c as usize != i)
                .count(),
            None => 0,
        }
    }

    /// Remove vertices that no cell references and compact the index storage.
    ///
    /// Edge repair may leave a handful of unreferenced vertices behind rather
    /// than paying this pass on every computation (see the orphan-vertices
    /// representation note in `docs/correctness.md`). Call this when
    /// a dense vertex array matters (serialization, GPU upload). Vertex
    /// indices are remapped; welded twins keep aliasing their canonical
    /// cell's boundary. Returns the number of vertices removed.
    pub fn compact_vertices(&mut self) -> usize {
        let num_vertices = self.vertices.len();
        let mut used = vec![false; num_vertices];
        for i in 0..self.cells.len() {
            if self.canonical_cell_index(i) != i {
                continue;
            }
            for &vi in self.cell(i).vertex_indices {
                if (vi as usize) < num_vertices {
                    used[vi as usize] = true;
                }
            }
        }

        let removed = used.iter().filter(|&&u| !u).count();
        if removed == 0 {
            return 0;
        }

        let mut old_to_new = vec![u32::MAX; num_vertices];
        let mut new_vertices = Vec::with_capacity(num_vertices - removed);
        for (old, &is_used) in used.iter().enumerate() {
            if is_used {
                old_to_new[old] = new_vertices.len() as u32;
                new_vertices.push(self.vertices[old]);
            }
        }

        // Rebuild cells and the index buffer in one pass. Canonical cells
        // always precede their twins (the canonical index is the smallest in
        // the weld class), so twins can reuse the rebuilt CellData.
        let mut new_cells: Vec<CellData> = Vec::with_capacity(self.cells.len());
        let mut new_indices: Vec<u32> = Vec::with_capacity(self.cell_indices.len());
        for i in 0..self.cells.len() {
            let canonical = self.canonical_cell_index(i);
            if canonical != i {
                debug_assert!(canonical < i);
                let alias = new_cells[canonical];
                new_cells.push(alias);
                continue;
            }
            let start = new_indices.len() as u32;
            let data = &self.cells[i];
            let range = data.start as usize..data.start as usize + data.len as usize;
            new_indices.extend(
                self.cell_indices[range]
                    .iter()
                    .map(|&vi| old_to_new.get(vi as usize).copied().unwrap_or(u32::MAX)),
            );
            new_cells.push(CellData {
                start,
                len: data.len,
            });
        }

        self.vertices = new_vertices;
        self.cells = new_cells;
        self.cell_indices = new_indices;
        removed
    }
}

/// A view into a single Voronoi cell.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
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
pub(crate) struct VoronoiCell {
    vertex_start: u32,
    vertex_count: u16,
}

impl VoronoiCell {
    /// Create a new VoronoiCell.
    #[inline]
    pub(crate) fn new(vertex_start: u32, vertex_count: u16) -> Self {
        Self {
            vertex_start,
            vertex_count,
        }
    }

    /// Start index into the flat cell_indices buffer.
    #[inline]
    pub(crate) fn vertex_start(&self) -> usize {
        self.vertex_start as usize
    }

    /// Number of vertices for this cell.
    #[inline]
    pub(crate) fn vertex_count(&self) -> usize {
        self.vertex_count as usize
    }
}

/// Checked deserialization support: the wire mirror of the storage layout
/// plus the structural validation that makes `Deserialize` safe on untrusted
/// bytes. Field names must stay in lockstep with [`SphericalVoronoi`] — the
/// derived `Serialize` on the real struct defines the format.
#[cfg(feature = "serde")]
mod wire {
    use super::{CellData, SpherePoint, SphericalVoronoi};

    #[derive(serde::Deserialize)]
    pub struct SphericalVoronoiWire {
        generators: Vec<SpherePoint>,
        vertices: Vec<SpherePoint>,
        cells: Vec<CellData>,
        cell_indices: Vec<u32>,
        weld_map: Option<Vec<u32>>,
    }

    impl TryFrom<SphericalVoronoiWire> for SphericalVoronoi {
        type Error = String;

        fn try_from(w: SphericalVoronoiWire) -> Result<Self, String> {
            if w.generators.is_empty() {
                return Err("diagram must contain at least one generator".to_owned());
            }
            if w.cells.len() != w.generators.len() {
                return Err(format!(
                    "cell count {} does not match generator count {}",
                    w.cells.len(),
                    w.generators.len()
                ));
            }
            for (i, c) in w.cells.iter().enumerate() {
                let end = c.start.checked_add(c.len as u32).ok_or_else(|| {
                    format!(
                        "cell {i} span start {} + len {} overflows u32",
                        c.start, c.len
                    )
                })?;
                let start = c.start as usize;
                let end = end as usize;
                if end > w.cell_indices.len() {
                    return Err(format!(
                        "cell {i} span [{start}..{end}) exceeds index buffer len {}",
                        w.cell_indices.len()
                    ));
                }
                // Only LIVE spans are validated: the in-place edge repair can
                // legitimately leave stale never-read slots in the buffer tail.
                if let Some(&vi) = w.cell_indices[start..end]
                    .iter()
                    .find(|&&vi| vi as usize >= w.vertices.len())
                {
                    return Err(format!(
                        "cell {i} references vertex {vi} out of bounds (num_vertices {})",
                        w.vertices.len()
                    ));
                }
            }
            if let Some(map) = &w.weld_map {
                if map.len() != w.cells.len() {
                    return Err(format!(
                        "weld map len {} does not match cell count {}",
                        map.len(),
                        w.cells.len()
                    ));
                }
                for (i, &c) in map.iter().enumerate() {
                    let c = c as usize;
                    // Canonical = smallest index in the weld class and a
                    // fixpoint of the map — the invariants compact_vertices
                    // and canonical-cell aliasing rely on.
                    if c > i || map[c] as usize != c {
                        return Err(format!(
                            "weld map entry {i} -> {c} is not a canonical (self-mapped, \
                             preceding) cell index"
                        ));
                    }
                    if w.cells[i] != w.cells[c] {
                        return Err(format!(
                            "welded cell {i} does not alias canonical cell {c}'s span"
                        ));
                    }
                }
            }
            Ok(SphericalVoronoi {
                generators: w.generators,
                vertices: w.vertices,
                cells: w.cells,
                cell_indices: w.cell_indices,
                weld_map: w.weld_map,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, y: f32, z: f32) -> SpherePoint {
        SpherePoint::try_from_xyz([x, y, z]).unwrap()
    }

    #[test]
    fn test_empty_diagram() {
        let generators = vec![point(1.0, 0.0, 0.0), point(0.0, 1.0, 0.0)];
        let diagram = SphericalVoronoi::empty(generators);

        assert_eq!(diagram.num_cells(), 2);
        assert_eq!(diagram.num_vertices(), 0);
        assert!(diagram.cell(0).is_empty());
        assert!(diagram.cell(1).is_empty());
    }

    #[test]
    fn test_compact_vertices_removes_orphans_and_remaps() {
        let generators = vec![point(1.0, 0.0, 0.0), point(-1.0, 0.0, 0.0)];
        // Vertex 2 is an orphan; 0, 1, 3, 4 are referenced.
        let vertices = vec![
            point(0.0, 1.0, 0.0),
            point(0.0, -1.0, 0.0),
            point(0.577, 0.577, 0.577),
            point(0.0, 0.0, 1.0),
            point(0.0, 0.0, -1.0),
        ];
        let mut diagram = SphericalVoronoi::from_cells_and_indices(
            generators,
            vertices,
            vec![0, 4],
            vec![4, 4],
            vec![0, 3, 1, 4, 0, 4, 1, 3],
        );

        let removed = diagram.compact_vertices();
        assert_eq!(removed, 1);
        assert_eq!(diagram.num_vertices(), 4);
        // Indices above the orphan shift down by one; the rest are unchanged.
        assert_eq!(diagram.cell(0).vertex_indices, &[0, 2, 1, 3]);
        assert_eq!(diagram.cell(1).vertex_indices, &[0, 3, 1, 2]);
        assert_eq!(diagram.vertex(2), point(0.0, 0.0, 1.0));
        // Second call is a no-op.
        assert_eq!(diagram.compact_vertices(), 0);
    }

    #[test]
    fn test_compact_vertices_preserves_weld_aliasing() {
        let generators = vec![
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 1e-7, 0.0).normalize(),
            glam::Vec3::new(-1.0, 0.0, 0.0),
        ];
        let vertices = vec![
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::new(0.3, 0.3, 0.9).normalize(), // orphan
            glam::Vec3::new(0.0, -1.0, 0.0),
            glam::Vec3::new(0.0, 0.0, 1.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
        ];
        // Cell 1 is a welded twin of cell 0.
        let cells = vec![
            VoronoiCell::new(0, 4),
            VoronoiCell::new(0, 4),
            VoronoiCell::new(4, 4),
        ];
        let cell_indices = vec![0, 3, 2, 4, 0, 4, 2, 3];
        let mut diagram = SphericalVoronoi::from_raw_parts(
            generators,
            vertices,
            cells,
            cell_indices,
            Some(vec![0, 0, 2]),
        );

        assert_eq!(diagram.compact_vertices(), 1);
        assert_eq!(diagram.num_vertices(), 4);
        assert_eq!(diagram.canonical_cell_index(1), 0);
        assert_eq!(
            diagram.cell(1).vertex_indices,
            diagram.cell(0).vertex_indices,
            "welded twin must still alias its canonical cell after compaction"
        );
        assert_eq!(diagram.cell(0).vertex_indices, &[0, 2, 1, 3]);
        assert_eq!(diagram.welded_twin_count(), 1);
    }

    #[test]
    fn test_from_parts() {
        let generators = vec![point(1.0, 0.0, 0.0), point(-1.0, 0.0, 0.0)];
        let vertices = vec![
            point(0.0, 1.0, 0.0),
            point(0.0, -1.0, 0.0),
            point(0.0, 0.0, 1.0),
            point(0.0, 0.0, -1.0),
        ];
        // Cell 0 uses vertices 0,2,1,3; Cell 1 uses same but different order
        let cell_starts = vec![0, 4];
        let cell_counts = vec![4, 4];
        let cell_indices = vec![0, 2, 1, 3, 0, 3, 1, 2];

        let diagram = SphericalVoronoi::from_cells_and_indices(
            generators,
            vertices,
            cell_starts,
            cell_counts,
            cell_indices,
        );

        assert_eq!(diagram.num_cells(), 2);
        assert_eq!(diagram.num_vertices(), 4);
        assert_eq!(diagram.cell(0).len(), 4);
        assert_eq!(diagram.cell(1).len(), 4);
    }

    #[test]
    fn raw_parts_transfer_point_allocations_without_copying() {
        let generators = vec![
            crate::types::canonical_vec3_from_dvec3(glam::DVec3::new(1.0, 2.0, 3.0)),
            crate::types::canonical_vec3_from_dvec3(glam::DVec3::new(-1.0, -2.0, -3.0)),
        ];
        let vertices = vec![
            crate::types::canonical_vec3_from_dvec3(glam::DVec3::new(3.0, -2.0, 1.0)),
            crate::types::canonical_vec3_from_dvec3(glam::DVec3::new(-3.0, 2.0, -1.0)),
        ];
        let generator_pointer = generators.as_ptr().cast::<u8>();
        let vertex_pointer = vertices.as_ptr().cast::<u8>();
        let generator_bits: Vec<[u32; 3]> = generators
            .iter()
            .map(|point| point.to_array().map(f32::to_bits))
            .collect();
        let vertex_bits: Vec<[u32; 3]> = vertices
            .iter()
            .map(|point| point.to_array().map(f32::to_bits))
            .collect();

        let diagram = SphericalVoronoi::from_raw_parts(
            generators,
            vertices,
            vec![VoronoiCell::new(0, 0), VoronoiCell::new(0, 0)],
            Vec::new(),
            None,
        );

        assert_eq!(
            diagram.generators().as_ptr().cast::<u8>(),
            generator_pointer
        );
        assert_eq!(diagram.vertices().as_ptr().cast::<u8>(), vertex_pointer);
        assert_eq!(
            diagram.generators_xyz().as_ptr().cast::<u8>(),
            generator_pointer
        );
        assert_eq!(diagram.vertices_xyz().as_ptr().cast::<u8>(), vertex_pointer);
        assert_eq!(
            diagram
                .generators_xyz()
                .iter()
                .map(|xyz| xyz.map(f32::to_bits))
                .collect::<Vec<_>>(),
            generator_bits
        );
        assert_eq!(
            diagram
                .vertices_xyz()
                .iter()
                .map(|xyz| xyz.map(f32::to_bits))
                .collect::<Vec<_>>(),
            vertex_bits
        );
    }
}
