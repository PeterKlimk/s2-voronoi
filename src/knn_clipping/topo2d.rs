//! 2D topology builder using gnomonic projection.
//!
//! Projects spherical half-space constraints to 2D lines in the generator's tangent plane,
//! performs half-plane intersection to determine the active constraint set and cyclic vertex
//! order, then computes 3D vertex positions from plane pairs.
//!
//! This avoids O(p³) triplet seeding entirely - half-plane intersection is O(p·v) where
//! v is the vertex count (typically 6-12).
//!
//! Optimized for convex polygons: exploits single entry/exit property, uses fixed arrays
//! for the polygon buffer, double-buffer swap pattern.

use glam::{DVec3, Vec3};

use super::cell_builder::{CellFailure, VertexData, VertexKey};

const EPS_INSIDE: f64 = 1e-12;

/// Maximum vertices in the polygon buffer.
/// This is intentionally generous because we start with a bounding triangle.
/// In practice, Voronoi cells rarely exceed 20 vertices.
const MAX_POLY_VERTICES: usize = 64;

/// A 2D half-plane constraint: a*u + b*v + c >= 0.
///
/// Note: This is intentionally *not* normalized. Clipping and intersection are scale-invariant,
/// and we use a scale-aware epsilon for inside/outside classification.
#[derive(Debug, Clone, Copy)]
struct HalfPlane {
    a: f64,
    b: f64,
    c: f64,
    plane_idx: usize,
    eps: f64,
}

impl HalfPlane {
    #[inline]
    fn new_unnormalized(a: f64, b: f64, c: f64, plane_idx: usize) -> Self {
        let norm = (a * a + b * b).sqrt();
        let eps = EPS_INSIDE * norm;
        HalfPlane {
            a,
            b,
            c,
            plane_idx,
            eps,
        }
    }

    #[inline]
    fn signed_dist(&self, u: f64, v: f64) -> f64 {
        self.a.mul_add(u, self.b.mul_add(v, self.c))
    }
}

/// Fixed-size polygon buffer for clipping.
#[derive(Clone)]
struct PolyBuffer {
    // Hot scalars first - keep in same cache line
    len: usize,
    max_r2: f64,
    has_bounding_ref: bool,
    // Cold arrays after
    vertices: [(f64, f64); MAX_POLY_VERTICES],
    vertex_planes: [(usize, usize); MAX_POLY_VERTICES],
    edge_planes: [usize; MAX_POLY_VERTICES],
}

impl PolyBuffer {
    #[inline]
    fn new() -> Self {
        Self {
            len: 0,
            max_r2: 0.0,
            has_bounding_ref: false,
            vertices: [(0.0, 0.0); MAX_POLY_VERTICES],
            vertex_planes: [(0, 0); MAX_POLY_VERTICES],
            edge_planes: [0; MAX_POLY_VERTICES],
        }
    }

    fn init_bounding(&mut self, bound: f64) {
        self.vertices[0] = (0.0, bound);
        self.vertices[1] = (-bound * 0.866, -bound * 0.5);
        self.vertices[2] = (bound * 0.866, -bound * 0.5);
        self.vertex_planes[0] = (usize::MAX, usize::MAX);
        self.vertex_planes[1] = (usize::MAX, usize::MAX);
        self.vertex_planes[2] = (usize::MAX, usize::MAX);
        self.edge_planes[0] = usize::MAX;
        self.edge_planes[1] = usize::MAX;
        self.edge_planes[2] = usize::MAX;
        self.len = 3;
        self.max_r2 = bound * bound;
        self.has_bounding_ref = true;
    }

    #[inline]
    fn clear(&mut self) {
        self.len = 0;
        self.max_r2 = 0.0;
        self.has_bounding_ref = false;
    }

    /// Pure write - no accumulator updates. Caller tracks max_r2/has_bounding_ref.
    #[inline]
    fn push_raw(&mut self, v: (f64, f64), vp: (usize, usize), ep: usize) {
        let i = self.len;
        debug_assert!(i < MAX_POLY_VERTICES);
        // SAFETY: debug_assert guarantees i < MAX_POLY_VERTICES
        unsafe {
            *self.vertices.get_unchecked_mut(i) = v;
            *self.vertex_planes.get_unchecked_mut(i) = vp;
            *self.edge_planes.get_unchecked_mut(i) = ep;
        }
        self.len = i + 1;
    }

    /// Get minimum cos across all vertices (for termination).
    /// Computes lazily from 2D gnomonic coords: cos(θ) = 1 / sqrt(1 + u² + v²)
    #[inline]
    fn min_cos(&self) -> f64 {
        if self.len == 0 {
            return 1.0;
        }
        1.0 / (1.0 + self.max_r2).sqrt()
    }

    /// Check if polygon still references bounding triangle.
    #[inline]
    fn has_bounding_ref(&self) -> bool {
        self.has_bounding_ref
    }
}

/// Orthonormal tangent basis for gnomonic projection.
pub struct TangentBasis {
    pub t1: DVec3,
    pub t2: DVec3,
    pub g: DVec3,
}

impl TangentBasis {
    pub fn new(g: DVec3) -> Self {
        let arbitrary = if g.x.abs() <= g.y.abs() && g.x.abs() <= g.z.abs() {
            DVec3::X
        } else if g.y.abs() <= g.z.abs() {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let t1 = g.cross(arbitrary).normalize();
        let t2 = g.cross(t1);
        TangentBasis { t1, t2, g }
    }

    #[inline]
    pub fn plane_to_line(&self, n: DVec3) -> (f64, f64, f64) {
        (n.dot(self.t1), n.dot(self.t2), n.dot(self.g))
    }
}

/// Bookkeeping for an edge transition (entry or exit point).
#[derive(Clone, Copy)]
struct Transition {
    idx: usize,  // Edge start index (usize::MAX = no transition found)
    next: usize, // Edge end index
    d0: f64,     // Signed distance at idx
    d1: f64,     // Signed distance at next
}

impl Default for Transition {
    #[inline]
    fn default() -> Self {
        Self {
            idx: usize::MAX,
            next: 0,
            d0: 0.0,
            d1: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClipResult {
    /// Polygon unchanged (all vertices inside). Note: `out` is NOT written.
    Unchanged,
    Changed,
    TooManyVertices,
}

/// Clip a convex polygon by a half-plane (standalone function to avoid borrow conflicts).
///
/// # Returns
/// - `Unchanged` if all vertices are inside the half-plane. **`out` is not modified.**
/// - `Changed` if the polygon was clipped (result written to `out`)
/// - `TooManyVertices` if the result would exceed `MAX_POLY_VERTICES`
#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;
    if n < 3 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    // Pass 1: classify vertices and track entry/exit transitions
    let neg_eps = -hp.eps;
    let mut inside_count = 0usize;
    let mut entry = Transition::default();
    let mut exit = Transition::default();

    let last_idx = n - 1;
    let (lu, lv) = poly.vertices[last_idx];
    let (fu, fv) = poly.vertices[0];
    let last_d = hp.signed_dist(lu, lv);
    let first_d = hp.signed_dist(fu, fv);
    let first_in = first_d >= neg_eps;
    let last_in = last_d >= neg_eps;

    inside_count += first_in as usize;

    // Interior edges: 0->1 through (n-3)->(n-2)
    let mut prev_d = first_d;
    let mut prev_in = first_in;
    for i in 1..last_idx {
        let (u, v) = poly.vertices[i];
        let curr_d = hp.signed_dist(u, v);
        let curr_in = curr_d >= neg_eps;
        inside_count += curr_in as usize;

        if prev_in != curr_in {
            if curr_in {
                entry = Transition {
                    idx: i - 1,
                    next: i,
                    d0: prev_d,
                    d1: curr_d,
                };
            } else {
                exit = Transition {
                    idx: i - 1,
                    next: i,
                    d0: prev_d,
                    d1: curr_d,
                };
            }
        }

        prev_d = curr_d;
        prev_in = curr_in;
    }

    // Edge (n-2) -> (n-1): use precomputed last_d/last_in
    inside_count += last_in as usize;
    if prev_in != last_in {
        if last_in {
            entry = Transition {
                idx: last_idx - 1,
                next: last_idx,
                d0: prev_d,
                d1: last_d,
            };
        } else {
            exit = Transition {
                idx: last_idx - 1,
                next: last_idx,
                d0: prev_d,
                d1: last_d,
            };
        }
    }

    // Wrap-around edge: (n-1) -> 0
    if last_in != first_in {
        if first_in {
            entry = Transition {
                idx: last_idx,
                next: 0,
                d0: last_d,
                d1: first_d,
            };
        } else {
            exit = Transition {
                idx: last_idx,
                next: 0,
                d0: last_d,
                d1: first_d,
            };
        }
    }

    // All inside: unchanged (out is NOT written)
    if inside_count == n {
        return ClipResult::Unchanged;
    }

    // All outside: empty result
    if inside_count == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    debug_assert!(entry.idx != usize::MAX && exit.idx != usize::MAX);
    if inside_count + 2 > MAX_POLY_VERTICES {
        return ClipResult::TooManyVertices;
    }

    // Compute intersection points
    let (eu0, ev0) = poly.vertices[entry.idx];
    let (eu1, ev1) = poly.vertices[entry.next];
    let t_entry = entry.d0 / (entry.d0 - entry.d1);
    let entry_pt = (eu0 + t_entry * (eu1 - eu0), ev0 + t_entry * (ev1 - ev0));
    let entry_edge_plane = poly.edge_planes[entry.idx];

    let (xu0, xv0) = poly.vertices[exit.idx];
    let (xu1, xv1) = poly.vertices[exit.next];
    let t_exit = exit.d0 / (exit.d0 - exit.d1);
    let exit_pt = (xu0 + t_exit * (xu1 - xu0), xv0 + t_exit * (xv1 - xv0));
    let exit_edge_plane = poly.edge_planes[exit.idx];

    // Pass 2: build output polygon
    if poly.has_bounding_ref {
        build_output::<true>(
            poly,
            out,
            n,
            entry_pt,
            entry_edge_plane,
            entry.next,
            exit_pt,
            exit_edge_plane,
            exit.idx,
            hp.plane_idx,
        );
    } else {
        build_output::<false>(
            poly,
            out,
            n,
            entry_pt,
            entry_edge_plane,
            entry.next,
            exit_pt,
            exit_edge_plane,
            exit.idx,
            hp.plane_idx,
        );
    }

    // The exit point's edge_planes was set to hp.plane_idx, but the vertex
    // *before* exit (last surviving input vertex) should keep exit_edge_plane
    // since its outgoing edge leads to the exit intersection point.
    if out.len >= 2 {
        out.edge_planes[out.len - 2] = exit_edge_plane;
    }

    ClipResult::Changed
}

/// Build the output polygon from entry to exit.
///
/// `TRACK_BOUNDING`: if true, tracks whether bounding refs survive (slow path).
/// If false, assumes already bounded and skips sentinel checks (fast path).
#[inline(always)]
fn build_output<const TRACK_BOUNDING: bool>(
    poly: &PolyBuffer,
    out: &mut PolyBuffer,
    n: usize,
    entry_pt: (f64, f64),
    entry_edge_plane: usize,
    entry_next: usize,
    exit_pt: (f64, f64),
    exit_edge_plane: usize,
    exit_idx: usize,
    hp_plane_idx: usize,
) {
    out.len = 0;

    #[inline(always)]
    fn r2_of(v: (f64, f64)) -> f64 {
        v.0 * v.0 + v.1 * v.1
    }

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;

    // Push helper that tracks max_r2, and conditionally tracks bounding refs
    macro_rules! push {
        ($v:expr, $vp:expr, $ep:expr) => {{
            let v = $v;
            let vp = $vp;
            out.push_raw(v, vp, $ep);
            let r2 = r2_of(v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if TRACK_BOUNDING {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection point
    push!(entry_pt, (entry_edge_plane, hp_plane_idx), entry_edge_plane);

    // Surviving input vertices from entry_next to exit_idx (cyclic)
    if entry_next <= exit_idx {
        for i in entry_next..=exit_idx {
            push!(poly.vertices[i], poly.vertex_planes[i], poly.edge_planes[i]);
        }
    } else {
        for i in entry_next..n {
            push!(poly.vertices[i], poly.vertex_planes[i], poly.edge_planes[i]);
        }
        for i in 0..=exit_idx {
            push!(poly.vertices[i], poly.vertex_planes[i], poly.edge_planes[i]);
        }
    }

    // Exit intersection point
    push!(exit_pt, (exit_edge_plane, hp_plane_idx), hp_plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if TRACK_BOUNDING { has_bounding } else { false };
}

/// Incremental 2D topology builder for spherical Voronoi cells.
///
/// Simpler algorithm that avoids O(p³) triplet seeding.
pub struct Topo2DBuilder {
    generator_idx: usize,
    generator: DVec3,
    basis: TangentBasis,

    // Half-planes and neighbor data (grow as needed)
    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,

    // Current polygon (double-buffered with fixed arrays)
    poly_a: PolyBuffer,
    poly_b: PolyBuffer,
    use_a: bool,

    // State
    failed: Option<CellFailure>,
    term_sin_pad: f64,
    term_cos_pad: f64,
}

impl Topo2DBuilder {
    /// Create a new builder for the given generator.
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        let angle_pad = 8.0 * f32::EPSILON as f64;
        let (term_sin_pad, term_cos_pad) = angle_pad.sin_cos();
        let gen64 =
            DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        let basis = TangentBasis::new(gen64);

        let mut poly_a = PolyBuffer::new();
        poly_a.init_bounding(1e6);

        Self {
            generator_idx,
            generator: gen64,
            basis,
            half_planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            poly_a,
            poly_b: PolyBuffer::new(),
            use_a: true,
            failed: None,
            term_sin_pad,
            term_cos_pad,
        }
    }

    /// Reset the builder for a new cell.
    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        let gen64 =
            DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        self.generator_idx = generator_idx;
        self.generator = gen64;
        self.basis = TangentBasis::new(gen64);
        self.half_planes.clear();
        self.neighbor_indices.clear();
        self.poly_a.init_bounding(1e6);
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
    }

    /// Add a neighbor and clip the cell.
    ///
    /// The caller is responsible for filtering duplicates and near-coincident neighbors.
    /// This method just clips—it doesn't check if the neighbor was already added.
    #[cfg_attr(feature = "profiling", inline(never))]
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) -> Result<(), CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }

        let n64 = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64).normalize();

        // Create half-plane from bisector.
        let normal_unnorm = self.generator - n64;
        let (a, b, c) = self.basis.plane_to_line(normal_unnorm);
        let plane_idx = self.half_planes.len();
        let hp = HalfPlane::new_unnormalized(a, b, c, plane_idx);

        // Clip polygon first, then only store if plane actually clipped.
        // If P ⊆ hp (unchanged), then P' = P ∩ other ⊆ P ⊆ hp for all future clips,
        // so redundant planes stay redundant forever.
        let clip_result = if self.use_a {
            clip_convex(&self.poly_a, &hp, &mut self.poly_b)
        } else {
            clip_convex(&self.poly_b, &hp, &mut self.poly_a)
        };

        match clip_result {
            ClipResult::TooManyVertices => {
                self.failed = Some(CellFailure::TooManyVertices);
                return Err(CellFailure::TooManyVertices);
            }
            ClipResult::Changed => {
                // Only store planes that actually contribute to the cell
                self.half_planes.push(hp);
                self.neighbor_indices.push(neighbor_idx);
                self.use_a = !self.use_a;
            }
            ClipResult::Unchanged => {} // Redundant plane - skip storage
        }

        // Check if clipped away
        let poly = self.current_poly();
        if poly.len < 3 {
            self.failed = Some(CellFailure::ClippedAway);
            return Err(CellFailure::ClippedAway);
        }

        Ok(())
    }

    #[inline]
    fn current_poly(&self) -> &PolyBuffer {
        if self.use_a {
            &self.poly_a
        } else {
            &self.poly_b
        }
    }

    /// Check if the cell is bounded (no bounding triangle references).
    #[inline]
    pub fn is_bounded(&self) -> bool {
        !self.current_poly().has_bounding_ref()
    }

    /// Check if the cell has failed.
    #[inline]
    pub fn is_failed(&self) -> bool {
        self.failed.is_some()
    }

    /// Get failure reason.
    #[inline]
    pub fn failure(&self) -> Option<CellFailure> {
        self.failed
    }

    /// Get current vertex count.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.current_poly().len
    }

    /// Check if neighbor already added.
    #[inline]
    pub fn has_neighbor(&self, neighbor_idx: usize) -> bool {
        self.neighbor_indices.contains(&neighbor_idx)
    }

    /// Iterate over neighbor indices.
    #[inline]
    pub fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices.iter().copied()
    }

    /// Check if we can terminate early.
    pub fn can_terminate(&mut self, max_unseen_dot_bound: f32) -> bool {
        if !self.is_bounded() || self.vertex_count() < 3 {
            return false;
        }

        let min_cos = self.current_poly().min_cos();
        // min_cos > 1.0 means degenerate vertex (sentinel value 2.0) - unsafe to terminate
        if min_cos <= 0.0 || min_cos > 1.0 {
            return false;
        }

        // Same bound as the legacy termination logic: 2 * max_vertex_angle
        let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
        let cos_theta_pad = min_cos * self.term_cos_pad - sin_theta * self.term_sin_pad;
        let cos_2max = 2.0 * cos_theta_pad * cos_theta_pad - 1.0;
        let threshold = cos_2max - 3.0 * f32::EPSILON as f64;

        (max_unseen_dot_bound as f64) < threshold
    }

    /// Convert to vertex data with simple triplet keys.
    /// Each vertex key is (cell, neighbor_a, neighbor_b) sorted.
    #[allow(dead_code)]
    pub fn to_vertex_data(&self) -> Result<Vec<VertexData>, CellFailure> {
        let mut out = Vec::new();
        self.to_vertex_data_into(&mut out)?;
        Ok(out)
    }

    /// Convert to vertex data, writing into provided buffer.
    #[allow(dead_code)]
    pub fn to_vertex_data_into(&self, out: &mut Vec<VertexData>) -> Result<(), CellFailure> {
        self.to_vertex_data_impl(out, None)
    }

    /// Convert to vertex data, also returning the edge neighbor for each vertex->next edge.
    pub fn to_vertex_data_with_edge_neighbors_into(
        &self,
        out: &mut Vec<VertexData>,
        edge_neighbors: &mut Vec<u32>,
    ) -> Result<(), CellFailure> {
        self.to_vertex_data_impl(out, Some(edge_neighbors))
    }

    #[cfg_attr(feature = "profiling", inline(never))]
    fn to_vertex_data_impl(
        &self,
        out: &mut Vec<VertexData>,
        mut edge_neighbors: Option<&mut Vec<u32>>,
    ) -> Result<(), CellFailure> {
        if !self.is_bounded() {
            return Err(CellFailure::NoValidSeed);
        }

        let poly = self.current_poly();
        if poly.len < 3 {
            return Err(CellFailure::NoValidSeed);
        }

        out.clear();
        out.reserve(poly.len);
        if let Some(edge_neighbors) = edge_neighbors.as_deref_mut() {
            edge_neighbors.clear();
            edge_neighbors.reserve(poly.len);
        }

        let gen_idx = self.generator_idx as u32;
        for i in 0..poly.len {
            let (u, v) = poly.vertices[i];
            let (plane_a, plane_b) = poly.vertex_planes[i];

            // Compute 3D vertex position from gnomonic (u, v).
            let dir = self.basis.g + self.basis.t1 * u + self.basis.t2 * v;
            let dir_len = dir.length();
            if dir_len < 1e-14 {
                return Err(CellFailure::NoValidSeed);
            }
            let v_pos = dir / dir_len;
            let pos = Vec3::new(v_pos.x as f32, v_pos.y as f32, v_pos.z as f32);

            // Build triplet key: (cell, neighbor_a, neighbor_b) sorted
            let def_a = self.neighbor_indices[plane_a] as u32;
            let def_b = self.neighbor_indices[plane_b] as u32;

            let mut a = gen_idx;
            let mut b = def_a;
            let mut c = def_b;
            sort3(&mut a, &mut b, &mut c);
            let key: VertexKey = [a, b, c];

            out.push((key, pos));
            if let Some(edge_neighbors) = edge_neighbors.as_deref_mut() {
                let edge_plane = poly.edge_planes[i];
                let neighbor = if edge_plane == usize::MAX {
                    u32::MAX
                } else {
                    self.neighbor_indices[edge_plane] as u32
                };
                edge_neighbors.push(neighbor);
            }
        }

        Ok(())
    }

    /// Count active planes (planes that define vertices).
    pub fn count_active_planes(&self) -> (usize, usize) {
        let poly = self.current_poly();
        let mut active = vec![false; self.half_planes.len()];

        for i in 0..poly.len {
            let (pa, pb) = poly.vertex_planes[i];
            if pa < active.len() {
                active[pa] = true;
            }
            if pb < active.len() {
                active[pb] = true;
            }
        }

        let active_count = active.iter().filter(|&&x| x).count();
        (active_count, self.half_planes.len())
    }
}

#[inline]
fn sort3(a: &mut u32, b: &mut u32, c: &mut u32) {
    if *a > *b {
        std::mem::swap(a, b);
    }
    if *b > *c {
        std::mem::swap(b, c);
    }
    if *a > *b {
        std::mem::swap(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tangent_basis() {
        let g = DVec3::new(0.0, 0.0, 1.0);
        let basis = TangentBasis::new(g);

        assert!((basis.t1.dot(basis.t2)).abs() < 1e-10);
        assert!((basis.t1.dot(basis.g)).abs() < 1e-10);
        assert!((basis.t2.dot(basis.g)).abs() < 1e-10);
        assert!((basis.t1.length() - 1.0).abs() < 1e-10);
        assert!((basis.t2.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_incremental_triangle() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
        let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
        let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

        assert!(!builder.is_bounded());

        builder.clip(1, h1).unwrap();
        assert!(!builder.is_bounded());

        builder.clip(2, h2).unwrap();
        assert!(!builder.is_bounded());

        builder.clip(3, h3).unwrap();
        assert!(builder.is_bounded());

        // Verify vertex count
        assert!(builder.vertex_count() >= 3);
    }

    #[test]
    fn test_incremental_square() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
        let h2 = Vec3::new(0.0, 1.0, 0.5).normalize();
        let h3 = Vec3::new(-1.0, 0.0, 0.5).normalize();
        let h4 = Vec3::new(0.0, -1.0, 0.5).normalize();

        builder.clip(1, h1).unwrap();
        builder.clip(2, h2).unwrap();
        builder.clip(3, h3).unwrap();
        builder.clip(4, h4).unwrap();

        assert!(builder.is_bounded());
        assert_eq!(builder.vertex_count(), 4);
    }

    #[test]
    fn test_early_termination_check() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        // Very close neighbors
        let h1 = Vec3::new(0.1, 0.0, 0.99).normalize();
        let h2 = Vec3::new(-0.05, 0.087, 0.99).normalize();
        let h3 = Vec3::new(-0.05, -0.087, 0.99).normalize();

        builder.clip(1, h1).unwrap();
        builder.clip(2, h2).unwrap();
        builder.clip(3, h3).unwrap();

        assert!(builder.is_bounded());

        // With a very far next neighbor, should be able to terminate
        let far_dot = 0.5f32; // ~60 degrees away
        let can_term = builder.can_terminate(far_dot);
        // Cell is very small, next neighbor is far, should terminate
        assert!(can_term);
    }

    #[test]
    fn test_to_vertex_data() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
        let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
        let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

        builder.clip(1, h1).unwrap();
        builder.clip(2, h2).unwrap();
        builder.clip(3, h3).unwrap();

        let vertices = builder.to_vertex_data().unwrap();

        assert_eq!(vertices.len(), 3);
        for (key, pos) in &vertices {
            // Position should be on unit sphere
            let len = pos.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "vertex not on sphere: len={}",
                len
            );

            // Key should be a sorted triplet
            let [a, b, c] = key;
            assert!(a < b && b < c, "triplet not sorted");
        }
    }
}
