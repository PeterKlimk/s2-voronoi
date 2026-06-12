//! Periodic (toroidal) cell builder: minimum-image bisector clipping.
//!
//! The torus sibling of [`super::builder::PlaneCellBuilder`]. Differences:
//!
//! - No rect walls: the polygon seeds with the gnomonic builder's
//!   bounding-reference triangle and becomes bounded only once constraints
//!   close it (like the sphere). Termination additionally requires
//!   boundedness.
//! - Neighbors enter in chart coordinates as **minimum-image**
//!   displacements via [`wrap_half`], which is bit-exactly antisymmetric
//!   (strict comparisons; IEEE negation is exact), so the two cells of a
//!   shared edge construct the identical bisector line. The exact-tie case
//!   `|d| == p/2` is the one non-antisymmetric input; its bisector lies at
//!   distance `p/4` from both generators, which the half-period guard
//!   excludes from ever touching a valid cell.
//! - The **half-period guard**: at extraction, the cell's chart radius must
//!   satisfy `max_r2 < (min_period / 4)^2`. Below that, every non-minimal
//!   image's bisector provably misses the cell, so clipping with nearest
//!   images only is exact. Violations (too few generators for the domain)
//!   fail with `CellFailure::UnboundedAfterExhaustion` or the dedicated
//!   guard failure, surfaced as `UnsupportedGeometry`.
//!
//! Vertices are extracted at canonical wrapped positions in `[0, p)`; the
//! unwrap convention (each vertex within half a period of its generator) is
//! the public `cell_polygon` helper's job.
//!
//! The guard is also what keeps the image-agnostic vertex keys sound: on a
//! torus a generator triple can circumscribe multiple distinct points via
//! different image combinations, but below the guard radius only the
//! nearest image's bisector can touch a cell, so each neighbor contributes
//! at most one edge per cell and `[g, a, b]` identifies at most one vertex.

// Only builder-level tests consume this module so far; the periodic
// driver / compute_plane_periodic pipeline lands next and removes this.
#![allow(dead_code)]

use glam::{DVec2, Vec2};

use crate::fp;
use crate::knn_clipping::cell_build::{CellFailure, CellOutputBuffer};
use crate::knn_clipping::topo2d::builder::sort3_u32;
use crate::knn_clipping::topo2d::clippers::{clip_convex, clip_convex_edgecheck};
use crate::knn_clipping::topo2d::types::{ClipResult, HalfPlane, PolyBuffer};
use crate::tolerances::PLANE_TERMINATION_GUARD;

/// Signed minimum-image displacement in `(-p/2, p/2]`, bit-exactly
/// antisymmetric for `|d| != p/2` (strict comparisons; see module docs).
#[inline(always)]
pub(crate) fn wrap_half(d: f32, p: f32) -> f32 {
    let half = 0.5 * p;
    if d > half {
        d - p
    } else if d < -half {
        d + p
    } else {
        d
    }
}

/// Incremental periodic Voronoi cell builder (minimum-image clipping).
pub(crate) struct PeriodicCellBuilder {
    pub(crate) generator_idx: usize,
    generator: DVec2,
    /// Domain periods (normalized; longer side = 1).
    px: f32,
    py: f32,
    /// Squared half-period guard radius: `(min(px, py) / 4)^2`.
    guard_r2: f64,

    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,
    neighbor_slots: Vec<u32>,

    poly_a: PolyBuffer,
    poly_b: PolyBuffer,
    use_a: bool,

    failed: Option<CellFailure>,
    term_threshold_cache: f64,
    term_cache_valid: bool,
}

impl PeriodicCellBuilder {
    pub(crate) fn new(generator_idx: usize, generator: Vec2, px: f32, py: f32) -> Self {
        let min_p = px.min(py) as f64;
        let guard_r = min_p / 4.0;
        let mut builder = Self {
            generator_idx,
            generator: DVec2::new(generator.x as f64, generator.y as f64),
            px,
            py,
            guard_r2: guard_r * guard_r,
            half_planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            neighbor_slots: Vec::with_capacity(32),
            poly_a: PolyBuffer::new(),
            poly_b: PolyBuffer::new(),
            use_a: true,
            failed: None,
            term_threshold_cache: 0.0,
            term_cache_valid: false,
        };
        builder.seed();
        builder
    }

    pub(crate) fn reset(&mut self, generator_idx: usize, generator: Vec2) {
        self.generator_idx = generator_idx;
        self.generator = DVec2::new(generator.x as f64, generator.y as f64);
        self.half_planes.clear();
        self.neighbor_indices.clear();
        self.neighbor_slots.clear();
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
        self.term_cache_valid = false;
        self.seed();
    }

    /// Seed with the unbounded reference triangle (sentinel planes); the
    /// cell becomes bounded when bisectors close it.
    fn seed(&mut self) {
        self.poly_a.init_bounding(1e6);
    }

    #[inline]
    fn current_poly(&self) -> &PolyBuffer {
        if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        }
    }

    #[inline]
    pub(crate) fn is_bounded(&self) -> bool {
        !self.current_poly().has_bounding_ref()
    }

    #[inline]
    pub(crate) fn vertex_count(&self) -> usize {
        self.current_poly().len
    }

    /// Bisector to the nearest image of `neighbor`, in generator-centered
    /// chart coordinates.
    #[inline]
    fn bisector_coefficients(&self, neighbor: Vec2) -> (f64, f64, f64) {
        // Wrap in f32 (bit-exact antisymmetry between the two cells of an
        // edge), then lift to f64 for the clip math.
        let qu = wrap_half(neighbor.x - self.generator.x as f32, self.px) as f64;
        let qv = wrap_half(neighbor.y - self.generator.y as f32, self.py) as f64;
        let c = 0.5 * fp::fma_f64(qu, qu, qv * qv);
        (-qu, -qv, c)
    }

    pub(crate) fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec2,
    ) -> Result<(), CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }
        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);
        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a)
        };
        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot)
            .map(|_| ())
    }

    /// Clip with the opposite side's epsilon (edge-check seeds).
    pub(crate) fn clip_with_slot_edgecheck(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec2,
        hp_eps: f32,
    ) -> Result<(), CellFailure> {
        if !hp_eps.is_finite() || hp_eps <= 0.0 {
            return self.clip_with_slot(neighbor_idx, neighbor_slot, neighbor);
        }
        if let Some(f) = self.failed {
            return Err(f);
        }
        let (a, b, c) = self.bisector_coefficients(neighbor);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized_with_eps(a, b, c, plane_idx, hp_eps as f64);
        let clip_result = if self.use_a {
            clip_convex_edgecheck(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex_edgecheck(&self.poly_b, &hp, &mut self.poly_a)
        };
        self.commit_clip(clip_result, hp, neighbor_idx, neighbor_slot)
            .map(|_| ())
    }

    fn commit_clip(
        &mut self,
        clip_result: ClipResult,
        hp: HalfPlane,
        neighbor_idx: usize,
        neighbor_slot: u32,
    ) -> Result<ClipResult, CellFailure> {
        match clip_result {
            ClipResult::TooManyVertices => {
                self.failed = Some(CellFailure::TooManyVertices);
                return Err(CellFailure::TooManyVertices);
            }
            ClipResult::Changed => {
                self.half_planes.push(hp);
                self.neighbor_indices.push(neighbor_idx);
                self.neighbor_slots.push(neighbor_slot);
                self.use_a = !self.use_a;
                self.term_cache_valid = false;
            }
            ClipResult::Unchanged => {}
        }
        if self.current_poly().len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }
        Ok(clip_result)
    }

    /// Sound termination on the torus: requires a bounded polygon, the
    /// Euclidean threshold `d2 > 4 * max_r2`, and rejects cells that exceed
    /// the half-period guard (those cannot be certified by nearest-image
    /// clipping at all and must fail rather than terminate).
    pub(crate) fn can_terminate(&mut self, min_unseen_dist_sq_bound: f32) -> bool {
        if self.failed.is_some() || !self.is_bounded() || self.vertex_count() < 3 {
            return false;
        }
        if !self.term_cache_valid {
            let max_r2 = self.current_poly().max_r2;
            self.term_threshold_cache = 4.0 * max_r2 * (1.0 + PLANE_TERMINATION_GUARD);
            self.term_cache_valid = true;
        }
        (min_unseen_dist_sq_bound as f64) > self.term_threshold_cache
    }

    /// Extract vertices (key + canonical wrapped position) and per-edge
    /// neighbor records.
    ///
    /// Enforces the half-period guard: a cell whose chart radius reaches
    /// `min_period / 4` cannot be certified by nearest-image clipping
    /// (a non-minimal image's bisector could touch it), so extraction
    /// fails rather than emit a possibly-wrong cell.
    pub(crate) fn to_vertex_data(
        &self,
        buffer: &mut CellOutputBuffer<Vec2>,
    ) -> Result<(), CellFailure> {
        let poly = self.current_poly();
        if self.failed.is_some() || poly.len < 3 || !self.is_bounded() {
            return Err(CellFailure::UnboundedAfterExhaustion);
        }
        if poly.max_r2 >= self.guard_r2 {
            return Err(CellFailure::UnboundedAfterExhaustion);
        }

        buffer.clear();
        buffer.vertices.reserve(poly.len);
        buffer.edge_neighbor_globals.reserve(poly.len);
        buffer.edge_neighbor_slots.reserve(poly.len);
        buffer.edge_neighbor_eps.reserve(poly.len);

        let gen_idx = self.generator_idx as u32;
        let plane_count = self.half_planes.len();
        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            if !u.is_finite() || !v.is_finite() {
                return Err(CellFailure::NoValidSeed);
            }
            let (plane_a, plane_b) = poly.vertex_planes[i];
            let edge_plane = poly.edge_planes[i];
            if plane_a >= plane_count || plane_b >= plane_count || edge_plane >= plane_count {
                return Err(CellFailure::NoValidSeed);
            }

            // Canonical wrapped position: the unwrap convention recovers the
            // generator-local image because the guard bounds the cell within
            // a half period.
            let pos = Vec2::new(
                ((u + self.generator.x) as f32).rem_euclid(self.px),
                ((v + self.generator.y) as f32).rem_euclid(self.py),
            );
            let n1 = self.neighbor_indices[plane_a] as u32;
            let n2 = self.neighbor_indices[plane_b] as u32;
            buffer.vertices.push((sort3_u32(gen_idx, n1, n2), pos));

            buffer
                .edge_neighbor_globals
                .push(self.neighbor_indices[edge_plane] as u32);
            buffer
                .edge_neighbor_slots
                .push(self.neighbor_slots[edge_plane]);
            buffer
                .edge_neighbor_eps
                .push(self.half_planes[edge_plane].eps as f32);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane_grid::periodic::min_image_dist_sq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn uniform(n: usize, seed: u64, px: f32, py: f32) -> Vec<Vec2> {
        let mut r = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| Vec2::new(r.gen_range(0.0..px), r.gen_range(0.0..py)))
            .collect()
    }

    fn brute_cell(
        points: &[Vec2],
        gi: usize,
        px: f32,
        py: f32,
    ) -> Result<CellOutputBuffer<Vec2>, CellFailure> {
        let mut b = PeriodicCellBuilder::new(gi, points[gi], px, py);
        for (j, &p) in points.iter().enumerate() {
            if j != gi {
                b.clip_with_slot(j, j as u32, p)?;
            }
        }
        let mut buf = CellOutputBuffer::default();
        b.to_vertex_data(&mut buf)?;
        Ok(buf)
    }

    fn cell_area_unwrapped(buf: &CellOutputBuffer<Vec2>, g: Vec2, px: f32, py: f32) -> f64 {
        // Unwrap each vertex to within half a period of the generator
        // (the documented client-side convention), then shoelace.
        let unwrap = |v: Vec2| -> (f64, f64) {
            (
                (g.x + wrap_half(v.x - g.x, px)) as f64,
                (g.y + wrap_half(v.y - g.y, py)) as f64,
            )
        };
        let n = buf.vertices.len();
        let mut acc = 0.0f64;
        for k in 0..n {
            let (ax, ay) = unwrap(buf.vertices[k].1);
            let (bx, by) = unwrap(buf.vertices[(k + 1) % n].1);
            acc += ax * by - bx * ay;
        }
        (0.5 * acc).abs()
    }

    #[test]
    fn periodic_cells_partition_the_torus() {
        for &(px, py, n, seed) in &[(1.0f32, 1.0f32, 250usize, 5u64), (1.0, 0.4, 300, 7)] {
            let points = uniform(n, seed, px, py);
            let mut total = 0.0f64;
            let mut edge_counts: std::collections::HashMap<(u32, u32), usize> = Default::default();
            for gi in 0..n {
                let buf = brute_cell(&points, gi, px, py)
                    .unwrap_or_else(|f| panic!("cell {gi} failed: {f:?}"));
                total += cell_area_unwrapped(&buf, points[gi], px, py);
                for &nb in &buf.edge_neighbor_globals {
                    let key = (gi.min(nb as usize) as u32, gi.max(nb as usize) as u32);
                    *edge_counts.entry(key).or_insert(0) += 1;
                }
            }
            let expected = px as f64 * py as f64;
            assert!(
                (total - expected).abs() < 1e-4 * expected,
                "areas sum to {total}, expected {expected} (px={px}, py={py})"
            );
            // Torus: EVERY edge is interior — exactly two uses, no exceptions.
            for (&(a, b), &count) in &edge_counts {
                assert_eq!(count, 2, "edge ({a},{b}) used {count} times");
            }
        }
    }

    #[test]
    fn periodic_seam_cells_wrap_correctly() {
        // A generator at the corner: its cell spans all four wrapped
        // quadrants; neighbors across every seam must clip it.
        let mut points = uniform(200, 11, 1.0, 1.0);
        points.push(Vec2::new(0.001, 0.001));
        let gi = points.len() - 1;
        let buf = brute_cell(&points, gi, 1.0, 1.0).unwrap();
        assert!(buf.vertices.len() >= 3);
        // All unwrapped vertices lie within the guard radius of the corner
        // generator (min-image, not raw distance).
        for &(_, v) in &buf.vertices {
            let d = min_image_dist_sq(v, points[gi], 1.0, 1.0);
            assert!(d < 0.25 * 0.25, "vertex {v:?} too far from seam generator");
        }
    }

    #[test]
    fn periodic_guard_rejects_underpopulated_torus() {
        // 3 generators on the unit torus: cells must exceed the half-period
        // guard and extraction must fail (no silent wrong answers).
        let points = vec![
            Vec2::new(0.1, 0.1),
            Vec2::new(0.6, 0.2),
            Vec2::new(0.3, 0.7),
        ];
        let mut failures = 0;
        for gi in 0..points.len() {
            if brute_cell(&points, gi, 1.0, 1.0).is_err() {
                failures += 1;
            }
        }
        assert!(
            failures > 0,
            "underpopulated torus must fail the half-period guard"
        );
    }

    #[test]
    fn periodic_wrap_half_antisymmetry() {
        // Bit-exact antisymmetry away from the |d| == p/2 tie.
        let p = 1.0f32;
        for &d in &[0.0f32, 0.1, 0.49999, 0.500001, 0.7, 0.999, -0.3, -0.51] {
            let w = wrap_half(d, p);
            let wn = wrap_half(-d, p);
            assert_eq!(w.to_bits(), (-wn).to_bits(), "asymmetric at d={d}");
        }
    }
}
