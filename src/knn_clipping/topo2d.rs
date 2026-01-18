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
    fn new_unnormalized(a: f64, b: f64, c: f64, plane_idx: usize) -> Self {
        // f32 sqrt is ~2x faster; eps precision doesn't need f64
        let n2: f64 = a.mul_add(a, b * b);
        let norm = (n2 as f32).sqrt() as f64;
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
    us: [f64; MAX_POLY_VERTICES],
    vs: [f64; MAX_POLY_VERTICES],
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
            us: [0.0; MAX_POLY_VERTICES],
            vs: [0.0; MAX_POLY_VERTICES],
            vertex_planes: [(0, 0); MAX_POLY_VERTICES],
            edge_planes: [0; MAX_POLY_VERTICES],
        }
    }

    fn init_bounding(&mut self, bound: f64) {
        self.us[0] = 0.0;
        self.vs[0] = bound;
        self.us[1] = -bound * 0.866;
        self.vs[1] = -bound * 0.5;
        self.us[2] = bound * 0.866;
        self.vs[2] = -bound * 0.5;
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
    fn push_raw(&mut self, u: f64, v: f64, vp: (usize, usize), ep: usize) {
        let i = self.len;
        debug_assert!(i < MAX_POLY_VERTICES);
        // SAFETY: debug_assert guarantees i < MAX_POLY_VERTICES
        unsafe {
            *self.us.get_unchecked_mut(i) = u;
            *self.vs.get_unchecked_mut(i) = v;
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
    /// First tangent vector in the generator's tangent plane.
    pub t1: DVec3,
    /// Second tangent vector in the generator's tangent plane.
    pub t2: DVec3,
    /// The generator unit vector itself (normal to the tangent plane).
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

fn clip_convex(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;
    if n < 3 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    // Pass 1: classify vertices using bitset (autovectorized).
    let neg_eps = -hp.eps;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    // We use a u64 mask. If MAX_POLY_VERTICES > 64, this needs u128 or [u64; N].
    // Current MAX is 64, so u64 is sufficient.
    debug_assert!(n <= 64, "Polygon too large for u64 bitmask");

    let mut mask = 0u64;
    let full_mask: u64 = if n == 64 { !0u64 } else { (1u64 << n) - 1 };

    // SAFETY: i < n <= MAX_POLY_VERTICES
    unsafe {
        for i in 0..n {
            let u = *us.add(i);
            let v = *vs.add(i);
            let d = hp.signed_dist(u, v);

            // typically compiles to branchless setcc + shift/or
            mask |= (u64::from(d >= neg_eps)) << i;
        }
    }

    // Fast path: All outside
    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    // Fast path: All inside
    if mask == full_mask {
        return ClipResult::Unchanged;
    }

    // Only now do we need the count
    let inside_count = mask.count_ones() as usize;

    if inside_count + 2 > MAX_POLY_VERTICES {
        return ClipResult::TooManyVertices;
    }

    // Mixed case: Find transitions.
    // Edge i connects vertex (i-1) to i.
    // Transition exists if mask[i-1] != mask[i].
    // We compute this by XORing mask with its rotation.
    // rotated: bit i gets bit i-1.
    // Since n might be < 64, we simulate rotation on the sub-range 0..n.

    let prev_mask = (mask << 1) | (mask >> (n - 1));
    let trans_mask = (mask ^ prev_mask) & ((1u64 << n) - 1);

    debug_assert_eq!(
        trans_mask.count_ones(),
        2,
        "Convex polygon should have exactly 2 transitions"
    );

    // Extract transition indices
    let idx1 = trans_mask.trailing_zeros() as usize;

    // Clear the first bit to find the second
    let trans_mask_rem = trans_mask & !(1 << idx1);
    let idx2 = trans_mask_rem.trailing_zeros() as usize;

    // Identify entry and exit
    // If mask[k] is 1 (inside) and trans_mask[k] is 1, it means mask[k-1] was 0 (outside).
    // So edge k-1 -> k enters the polygon. -> Entry next = k.
    // If mask[k] is 0 (outside) and trans_mask[k] is 1, it means mask[k-1] was 1 (inside).
    // So edge k-1 -> k exits the polygon. -> Exit next = k.

    let (entry_next, exit_next) = if (mask & (1 << idx1)) != 0 {
        (idx1, idx2)
    } else {
        (idx2, idx1)
    };

    // Calculate transition data only for the two intersection points
    // We need d0 (dist at i-1) and d1 (dist at i) for the intersection formula.
    // entry.next = entry_next. idx = entry_next - 1 (modulo n).

    let calc_transition = |next_idx: usize| -> (usize, usize, f64, f64) {
        let prev_idx = if next_idx == 0 { n - 1 } else { next_idx - 1 };
        let d0 = hp.signed_dist(poly.us[prev_idx], poly.vs[prev_idx]);
        let d1 = hp.signed_dist(poly.us[next_idx], poly.vs[next_idx]);
        (prev_idx, next_idx, d0, d1)
    };

    let (entry_idx, entry_next_idx, entry_d0, entry_d1) = calc_transition(entry_next);
    let (exit_idx, exit_next_idx, exit_d0, exit_d1) = calc_transition(exit_next);

    // Compute intersection points
    let t_entry = entry_d0 / (entry_d0 - entry_d1);
    let entry_u = t_entry.mul_add(
        poly.us[entry_next_idx] - poly.us[entry_idx],
        poly.us[entry_idx],
    );
    let entry_v = t_entry.mul_add(
        poly.vs[entry_next_idx] - poly.vs[entry_idx],
        poly.vs[entry_idx],
    );
    let entry_pt = (entry_u, entry_v);
    let entry_edge_plane = poly.edge_planes[entry_idx];

    let t_exit = exit_d0 / (exit_d0 - exit_d1);
    let exit_u = t_exit.mul_add(
        poly.us[exit_next_idx] - poly.us[exit_idx],
        poly.us[exit_idx],
    );
    let exit_v = t_exit.mul_add(
        poly.vs[exit_next_idx] - poly.vs[exit_idx],
        poly.vs[exit_idx],
    );
    let exit_pt = (exit_u, exit_v);
    let exit_edge_plane = poly.edge_planes[exit_idx];

    // Pass 2: build output polygon
    if poly.has_bounding_ref {
        build_output::<true>(
            poly,
            out,
            n,
            entry_pt,
            entry_edge_plane,
            entry_next_idx,
            exit_pt,
            exit_edge_plane,
            exit_idx,
            hp.plane_idx,
        );
    } else {
        build_output::<false>(
            poly,
            out,
            n,
            entry_pt,
            entry_edge_plane,
            entry_next_idx,
            exit_pt,
            exit_edge_plane,
            exit_idx,
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
        v.0.mul_add(v.0, v.1 * v.1)
    }

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;

    // Push helper that tracks max_r2, and conditionally tracks bounding refs
    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            let r2 = r2_of((u, v));
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if TRACK_BOUNDING {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection point
    push!(
        entry_pt.0,
        entry_pt.1,
        (entry_edge_plane, hp_plane_idx),
        entry_edge_plane
    );

    // Surviving input vertices from entry_next to exit_idx (cyclic)
    if entry_next <= exit_idx {
        for i in entry_next..=exit_idx {
            push!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }
    } else {
        for i in entry_next..n {
            push!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }
        for i in 0..=exit_idx {
            push!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }
    }

    // Exit intersection point
    push!(
        exit_pt.0,
        exit_pt.1,
        (exit_edge_plane, hp_plane_idx),
        hp_plane_idx
    );

    out.max_r2 = max_r2;
    out.has_bounding_ref = if TRACK_BOUNDING { has_bounding } else { false };
}

/// Incremental 2D topology builder for spherical Voronoi cells.
///
/// Uses gnomonic projection to avoid expensive 3D triplet seeding.
pub struct Topo2DBuilder {
    generator_idx: usize,
    generator: DVec3,
    basis: TangentBasis,

    // Half-planes and neighbor data (grow as needed)
    half_planes: Vec<HalfPlane>,
    neighbor_indices: Vec<usize>,
    /// Slot indices for each neighbor (SOA index, u32::MAX if not from packed_knn).
    neighbor_slots: Vec<u32>,

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
            neighbor_slots: Vec::with_capacity(32),
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
        self.neighbor_slots.clear();
        self.poly_a.init_bounding(1e6);
        self.poly_b.clear();
        self.use_a = true;
        self.failed = None;
    }

    /// Add a neighbor and clip the cell.
    ///
    /// The caller is responsible for filtering duplicates and near-coincident neighbors.
    /// This method just clips—it doesn't check if the neighbor was already added.

    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) -> Result<(), CellFailure> {
        self.clip_with_slot(neighbor_idx, u32::MAX, neighbor)
    }

    /// Add a neighbor and clip the cell, also storing the slot index.
    ///
    /// The slot is the SOA index from packed_knn. Use u32::MAX if not from packed_knn.
    pub fn clip_with_slot(
        &mut self,
        neighbor_idx: usize,
        neighbor_slot: u32,
        neighbor: Vec3,
    ) -> Result<(), CellFailure> {
        if let Some(f) = self.failed {
            return Err(f);
        }

        debug_assert!(
            (neighbor.length_squared() - 1.0).abs() < 1e-5,
            "neighbor not unit-normalized: |N|² = {}",
            neighbor.length_squared()
        );

        let n_raw = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64);

        // Bisector direction is G - unit(N). Instead of computing |N| via sqrt,
        // use first-order Taylor: |N| = √(1+ε) ≈ 1 + ε/2 where ε = |N|² - 1.
        //
        // G*(1 + ε/2) - N = G*scale - N, where scale = 1 + 0.5*(s-1) = 0.5*(s+1)
        //
        // For f32-normalized inputs, ε ≈ O(1e-7), so Taylor error is O(ε²) ≈ 1e-15.
        // If ε is large, the "neighbor was normalized" invariant is broken upstream.
        let len_sq = n_raw.length_squared();
        let scale = len_sq.mul_add(0.5, 0.5);

        let g = self.generator;
        let normal_unnorm = DVec3::new(
            g.x.mul_add(scale, -n_raw.x),
            g.y.mul_add(scale, -n_raw.y),
            g.z.mul_add(scale, -n_raw.z),
        );

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
                self.neighbor_slots.push(neighbor_slot);
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
        let cos_theta_pad = min_cos.mul_add(self.term_cos_pad, -sin_theta * self.term_sin_pad);
        let cos_2max = (2.0 * cos_theta_pad).mul_add(cos_theta_pad, -1.0);
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
        self.to_vertex_data_impl(out, None, None)
    }

    /// Convert to vertex data, also returning the edge neighbor for each vertex->next edge.

    /// Convert to vertex data, returning both global neighbor indices and slots.
    pub fn to_vertex_data_full(
        &self,
        out: &mut Vec<VertexData>,
        edge_neighbors: &mut Vec<u32>,
        edge_neighbor_slots: &mut Vec<u32>,
    ) -> Result<(), CellFailure> {
        self.to_vertex_data_impl(out, Some(edge_neighbors), Some(edge_neighbor_slots))
    }

    fn to_vertex_data_impl(
        &self,
        out: &mut Vec<VertexData>,
        mut edge_neighbors: Option<&mut Vec<u32>>,
        mut edge_neighbor_slots: Option<&mut Vec<u32>>,
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
        if let Some(edge_neighbor_slots) = edge_neighbor_slots.as_deref_mut() {
            edge_neighbor_slots.clear();
            edge_neighbor_slots.reserve(poly.len);
        }

        let gen_idx = self.generator_idx as u32;
        for i in 0..poly.len {
            let u = poly.us[i];
            let v = poly.vs[i];
            let (plane_a, plane_b) = poly.vertex_planes[i];

            // Compute 3D vertex position from gnomonic (u, v).
            let dir = DVec3::new(
                u.mul_add(self.basis.t1.x, v.mul_add(self.basis.t2.x, self.basis.g.x)),
                u.mul_add(self.basis.t1.y, v.mul_add(self.basis.t2.y, self.basis.g.y)),
                u.mul_add(self.basis.t1.z, v.mul_add(self.basis.t2.z, self.basis.g.z)),
            );
            // Check len² before sqrt to decouple branch from sqrt latency.
            // (1e-28 corresponds to r ≈ 1e14, i.e., ~90° from generator).
            let len2 = dir.length_squared();
            if len2 < 1e-28 {
                return Err(CellFailure::NoValidSeed);
            }
            // Use scalar recip + packed mul instead of packed div for lower latency.
            let inv_len = len2.sqrt().recip();
            let v_pos = dir * inv_len;
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
            if let Some(edge_neighbor_slots) = edge_neighbor_slots.as_deref_mut() {
                let edge_plane = poly.edge_planes[i];
                let slot = if edge_plane == usize::MAX {
                    u32::MAX
                } else {
                    self.neighbor_slots[edge_plane]
                };
                edge_neighbor_slots.push(slot);
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
