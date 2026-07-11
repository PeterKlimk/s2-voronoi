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

use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cube_grid::{CubeMapGrid, CubeMapGridScratch, DirectedEligibility};
use crate::packed_layout::PackedSlotLayout;
use crate::{SphericalVoronoi, UnitVec3Like};

const LOCATE_LOCAL_SHIFT: u32 = 24;
const LOCATE_LOCAL_MASK: u32 = (1u32 << LOCATE_LOCAL_SHIFT) - 1;

/// All-emit eligibility: every slot is bin 0, the query claims bin 1, so
/// every cell is "cross-bin" and the directed rules never filter.
fn all_emit_map(n: usize) -> Vec<u32> {
    (0..n).map(all_emit_slot).collect()
}

#[inline]
const fn all_emit_slot(_slot: usize) -> u32 {
    0
}

/// Run `f` over every query with a per-worker scratch from `mk` (parallel
/// with the `parallel` feature, plain loop otherwise).
fn map_with_scratch<P, S, MK, F>(queries: &[P], mk: MK, f: F) -> Vec<usize>
where
    P: Sync,
    S: Send,
    MK: Fn() -> S + Sync + Send,
    F: Fn(&mut S, &P) -> usize + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        queries
            .par_iter()
            .map_init(&mk, |scratch, p| f(scratch, p))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = mk();
        queries.iter().map(|p| f(&mut scratch, p)).collect()
    }
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
        sphere_locate_core(
            &self.grid,
            &self.slot_gen_map,
            self.canonical.as_deref(),
            &mut self.scratch,
            query,
        )
    }

    /// Locate every query, in input order (parallel with the `parallel`
    /// feature — each worker gets its own scratch, so this takes `&self`).
    pub fn locate_many<P: UnitVec3Like + Sync>(&self, queries: &[P]) -> Vec<usize> {
        map_with_scratch(
            queries,
            || self.grid.make_scratch(),
            |scratch, p| {
                sphere_locate_core(
                    &self.grid,
                    &self.slot_gen_map,
                    self.canonical.as_deref(),
                    scratch,
                    Vec3::new(p.x(), p.y(), p.z()),
                )
            },
        )
    }
}

fn sphere_locate_core(
    grid: &CubeMapGrid,
    slot_gen_map: &[u32],
    canonical: Option<&[u32]>,
    scratch: &mut CubeMapGridScratch,
    query: Vec3,
) -> usize {
    let layout = PackedSlotLayout::new(slot_gen_map, LOCATE_LOCAL_SHIFT, LOCATE_LOCAL_MASK);
    let ctx = DirectedEligibility::from_layout(1, 0, layout);
    let n = slot_gen_map.len();
    let mut frontier = grid.shell_frontier(query, n, scratch, ctx);
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
    let idx = grid.point_indices()[slot as usize] as usize;
    match canonical {
        Some(map) => map[idx] as usize,
        None => idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_emit_layout_does_not_leak_slot_bits_into_bin() {
        let boundary = 1usize << LOCATE_LOCAL_SHIFT;
        for slot in [boundary - 1, boundary, boundary + 1] {
            assert_eq!(all_emit_slot(slot), 0);
        }

        let map = all_emit_map(3);
        let layout = PackedSlotLayout::new(&map, LOCATE_LOCAL_SHIFT, LOCATE_LOCAL_MASK);
        let eligibility = DirectedEligibility::from_layout(1, 0, layout);

        for slot in 0..map.len() {
            assert_eq!(layout.bin_local(slot as u32), (0, 0));
            assert!(eligibility.allows_center_slot(slot as u32));
        }
    }
}
