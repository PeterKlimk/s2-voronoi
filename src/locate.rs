//! Point location: map a query point to the Voronoi cell that contains it
//! (equivalently, find its nearest generator).
//!
//! Locators are built on demand from a computed diagram (one grid build
//! over the generators) and answer repeated queries in near-constant time
//! via the same shell-expansion frontiers the construction pipeline uses:
//! collect ring candidates nearest-first until the ring certificate proves
//! no unseen generator can beat the best found. Queries take `&mut self`
//! (each locator carries its own scratch); clone the locator for use from
//! multiple threads.
//!
//! Returned indices are **canonical** cell indices: for welded inputs the
//! twin's canonical cell is returned (a query can never be strictly closer
//! to a twin than to its canonical, since welded generators coincide within
//! the weld radius).

use glam::{Vec2, Vec3};

use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch, DirectedEligibility};
use crate::packed_layout::PackedSlotLayout;
use crate::plane_grid::periodic::{PeriodicGrid, PeriodicGridScratch, PeriodicShellFrontier};
use crate::plane_grid::{PlaneGrid, PlaneGridScratch};
use crate::{PlanarVoronoi, PlaneTopology, SphericalVoronoi, UnitVec3Like};

const LOCATE_LOCAL_SHIFT: u32 = 24;
const LOCATE_LOCAL_MASK: u32 = (1u32 << LOCATE_LOCAL_SHIFT) - 1;

/// All-emit eligibility: every slot is bin 0, the query claims bin 1, so
/// every cell is "cross-bin" and the directed rules never filter.
fn all_emit_map(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// Point locator for a [`SphericalVoronoi`] diagram.
///
/// Build with [`SphericalVoronoi::build_locator`]; query with
/// [`Self::locate`].
pub struct SphereLocator {
    grid: CubeMapGrid,
    slot_gen_map: Vec<u32>,
    canonical: Option<Vec<u32>>,
    scratch: CubeMapGridScratch,
}

impl SphericalVoronoi {
    /// Build a point locator over this diagram's generators (one spatial
    /// grid build; queries are then near-constant time).
    pub fn build_locator(&self) -> SphereLocator {
        let generators: Vec<Vec3> = self
            .generators()
            .iter()
            .map(|g| Vec3::new(g.x, g.y, g.z))
            .collect();
        let n = generators.len();
        let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&generators, res);
        let scratch = grid.make_scratch();
        SphereLocator {
            grid,
            slot_gen_map: all_emit_map(n),
            canonical: self.weld_map().map(|m| m.to_vec()),
            scratch,
        }
    }
}

impl SphereLocator {
    /// Cell containing the query direction (nearest generator by angle).
    ///
    /// The query is assumed unit-normalized, like the diagram's inputs.
    /// Returns the canonical cell index.
    pub fn locate<P: UnitVec3Like>(&mut self, p: &P) -> usize {
        let query = Vec3::new(p.x(), p.y(), p.z());
        let layout =
            PackedSlotLayout::new(&self.slot_gen_map, LOCATE_LOCAL_SHIFT, LOCATE_LOCAL_MASK);
        let ctx = DirectedEligibility::from_layout(1, 0, layout);
        let n = self.slot_gen_map.len();
        let mut frontier = self.grid.shell_frontier(query, n, &mut self.scratch, ctx);
        let mut batch: Vec<u32> = Vec::new();
        let mut best: Option<(f32, u32)> = None;
        while let Some(layer) = frontier.frontier(&mut batch) {
            // Layers are sorted nearest-first (descending dot).
            let candidate = (layer.first_dot, batch[0]);
            if best.is_none_or(|(dot, _)| candidate.0 > dot) {
                best = Some(candidate);
            }
            if best.is_some_and(|(dot, _)| dot >= layer.unseen_bound) {
                break;
            }
            frontier.advance();
        }
        let (_, slot) = best.expect("locator requires a non-empty diagram");
        let idx = self.grid.point_indices()[slot as usize] as usize;
        match &self.canonical {
            Some(map) => map[idx] as usize,
            None => idx,
        }
    }
}

enum PlaneLocatorGrid {
    Bounded(PlaneGrid, PlaneGridScratch),
    Periodic(PeriodicGrid, PeriodicGridScratch),
}

/// Point locator for a [`PlanarVoronoi`] diagram (bounded or periodic).
///
/// Build with [`PlanarVoronoi::build_locator`]; query with
/// [`Self::locate`].
pub struct PlaneLocator {
    grid: PlaneLocatorGrid,
    slot_gen_map: Vec<u32>,
    canonical: Option<Vec<u32>>,
    /// Rect-to-normalized transform (mirrors the compute pipeline's).
    min: Vec2,
    scale: f32,
    /// Normalized domain extents.
    domain: Vec2,
}

impl PlanarVoronoi {
    /// Build a point locator over this diagram's generators (one spatial
    /// grid build; queries are then near-constant time). Works for both
    /// bounded and periodic diagrams; periodic queries wrap.
    pub fn build_locator(&self) -> PlaneLocator {
        let rect = self.rect();
        let extent = rect.width().max(rect.height());
        let scale = 1.0 / extent;
        let domain = Vec2::new(rect.width() * scale, rect.height() * scale);
        let min = Vec2::new(rect.min.x, rect.min.y);
        let n = self.num_cells();
        let res = ((n as f64 / 16.0).sqrt() as usize).clamp(1, 4096);

        let normalized: Vec<Vec2> = self
            .generators()
            .iter()
            .map(|g| {
                let p = (Vec2::new(g.x, g.y) - min) * scale;
                match self.topology() {
                    PlaneTopology::Bounded => p,
                    PlaneTopology::Periodic => Vec2::new(
                        p.x.rem_euclid(domain.x).min(next_below(domain.x)),
                        p.y.rem_euclid(domain.y).min(next_below(domain.y)),
                    ),
                }
            })
            .collect();

        let grid = match self.topology() {
            PlaneTopology::Bounded => {
                let g = PlaneGrid::new(&normalized, res);
                let s = g.make_scratch();
                PlaneLocatorGrid::Bounded(g, s)
            }
            PlaneTopology::Periodic => {
                let g = PeriodicGrid::new(&normalized, res, domain.x, domain.y);
                let s = g.make_scratch();
                PlaneLocatorGrid::Periodic(g, s)
            }
        };

        PlaneLocator {
            grid,
            slot_gen_map: all_emit_map(n),
            canonical: self.weld_map().map(|m| m.to_vec()),
            min,
            scale,
            domain,
        }
    }
}

#[inline]
fn next_below(p: f32) -> f32 {
    f32::from_bits(p.to_bits() - 1)
}

impl PlaneLocator {
    /// Cell containing the query point (nearest generator).
    ///
    /// Bounded diagrams accept any finite query (points outside the rect
    /// locate to the nearest generator all the same); periodic queries are
    /// wrapped into the domain. Returns the canonical cell index.
    pub fn locate(&mut self, x: f32, y: f32) -> usize {
        let q = (Vec2::new(x, y) - self.min) * self.scale;
        let layout =
            PackedSlotLayout::new(&self.slot_gen_map, LOCATE_LOCAL_SHIFT, LOCATE_LOCAL_MASK);
        let ctx = DirectedEligibility::from_layout(1, 0, layout);
        let n = self.slot_gen_map.len();

        let mut batch: Vec<u32> = Vec::new();
        let mut dists: Vec<f32> = Vec::new();
        let mut best: Option<(f32, u32)> = None;

        let slot = match &mut self.grid {
            PlaneLocatorGrid::Bounded(grid, scratch) => {
                let mut frontier =
                    crate::plane_grid::PlaneShellFrontier::new(grid, q, n, scratch, ctx);
                while let Some(layer) = frontier.frontier(&mut batch, &mut dists) {
                    let candidate = (dists[0], batch[0]);
                    if best.is_none_or(|(d, _)| candidate.0 < d) {
                        best = Some(candidate);
                    }
                    if best.is_some_and(|(d, _)| d <= layer.unseen_bound) {
                        break;
                    }
                    frontier.advance();
                }
                let (_, slot) = best.expect("locator requires a non-empty diagram");
                slot
            }
            PlaneLocatorGrid::Periodic(grid, scratch) => {
                let qw = Vec2::new(
                    q.x.rem_euclid(self.domain.x).min(next_below(self.domain.x)),
                    q.y.rem_euclid(self.domain.y).min(next_below(self.domain.y)),
                );
                let mut frontier = PeriodicShellFrontier::new(grid, qw, n, scratch, ctx);
                while let Some(layer) = frontier.frontier(&mut batch, &mut dists) {
                    let candidate = (dists[0], batch[0]);
                    if best.is_none_or(|(d, _)| candidate.0 < d) {
                        best = Some(candidate);
                    }
                    if best.is_some_and(|(d, _)| d <= layer.unseen_bound) {
                        break;
                    }
                    frontier.advance();
                }
                let (_, slot) = best.expect("locator requires a non-empty diagram");
                slot
            }
        };

        let idx = match &self.grid {
            PlaneLocatorGrid::Bounded(grid, _) => grid.point_indices()[slot as usize] as usize,
            PlaneLocatorGrid::Periodic(grid, _) => grid.point_indices()[slot as usize] as usize,
        };
        match &self.canonical {
            Some(map) => map[idx] as usize,
            None => idx,
        }
    }
}
