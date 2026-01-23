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
use std::simd::StdFloat;

use super::cell_builder::{CellFailure, VertexData, VertexKey};

const EPS_INSIDE: f64 = 1e-12;

#[cfg(feature = "timing")]
pub(super) struct ClipConvexStatsSnapshot {
    pub calls: u64,
    pub early_unchanged_hits: u64,
    pub early_unchanged_hits_bounded: u64,
    // Buckets are indexed by n, for n in 0..=16. Larger n are accumulated in *_gt_16.
    pub calls_by_n: [u64; 17],
    pub hits_by_n: [u64; 17],
    pub calls_gt_16: u64,
    pub hits_gt_16: u64,
}

#[cfg(feature = "timing")]
mod clip_convex_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    static CALLS: AtomicU64 = AtomicU64::new(0);
    static EARLY_UNCHANGED_HITS: AtomicU64 = AtomicU64::new(0);
    static EARLY_UNCHANGED_HITS_BOUNDED: AtomicU64 = AtomicU64::new(0);

    static CALLS_GT_16: AtomicU64 = AtomicU64::new(0);
    static HITS_GT_16: AtomicU64 = AtomicU64::new(0);

    static CALLS_BY_N: [AtomicU64; 17] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    static HITS_BY_N: [AtomicU64; 17] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    #[inline(always)]
    pub(super) fn record_call(n: usize) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        if n <= 16 {
            CALLS_BY_N[n].fetch_add(1, Ordering::Relaxed);
        } else {
            CALLS_GT_16.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub(super) fn record_early_unchanged(n: usize, bounded: bool) {
        EARLY_UNCHANGED_HITS.fetch_add(1, Ordering::Relaxed);
        if bounded {
            EARLY_UNCHANGED_HITS_BOUNDED.fetch_add(1, Ordering::Relaxed);
        }
        if n <= 16 {
            HITS_BY_N[n].fetch_add(1, Ordering::Relaxed);
        } else {
            HITS_GT_16.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn take() -> super::ClipConvexStatsSnapshot {
        let calls = CALLS.swap(0, Ordering::Relaxed);
        let early_unchanged_hits = EARLY_UNCHANGED_HITS.swap(0, Ordering::Relaxed);
        let early_unchanged_hits_bounded = EARLY_UNCHANGED_HITS_BOUNDED.swap(0, Ordering::Relaxed);

        let mut calls_by_n = [0u64; 17];
        let mut hits_by_n = [0u64; 17];
        for i in 0..=16 {
            calls_by_n[i] = CALLS_BY_N[i].swap(0, Ordering::Relaxed);
            hits_by_n[i] = HITS_BY_N[i].swap(0, Ordering::Relaxed);
        }
        let calls_gt_16 = CALLS_GT_16.swap(0, Ordering::Relaxed);
        let hits_gt_16 = HITS_GT_16.swap(0, Ordering::Relaxed);

        super::ClipConvexStatsSnapshot {
            calls,
            early_unchanged_hits,
            early_unchanged_hits_bounded,
            calls_by_n,
            hits_by_n,
            calls_gt_16,
            hits_gt_16,
        }
    }
}

#[cfg(feature = "timing")]
pub(super) fn take_clip_convex_stats() -> ClipConvexStatsSnapshot {
    clip_convex_stats::take()
}

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
    ab2: f64,
    plane_idx: usize,
    eps: f64,
}

impl HalfPlane {
    fn new_unnormalized(a: f64, b: f64, c: f64, plane_idx: usize) -> Self {
        // f32 sqrt is ~2x faster; eps precision doesn't need f64
        let ab2: f64 = a.mul_add(a, b * b);
        let norm = (ab2 as f32).sqrt() as f64;
        let eps = EPS_INSIDE * norm;

        HalfPlane {
            a,
            b,
            c,
            ab2,
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

#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
fn clip_convex_small<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // Pass 1: classify vertices and compute signed distances once.
    //
    // This is the original "small" baseline: a compact bitmask for the inside set plus a tiny
    // amount of bit-twiddling to find the 2 transition edges in the mixed case.
    //
    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let mut inside_mask: u8 = 0;

    // SAFETY: i < N <= 8 <= MAX_POLY_VERTICES
    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            inside_mask |= ((d >= neg_eps) as u8) << i;
        }
    }

    let full_mask: u8 = ((1u16 << N) - 1) as u8;
    if inside_mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if inside_mask == full_mask {
        return ClipResult::Unchanged;
    }

    // Mixed case: find the two transition indices using bit tricks (like the u64 path).
    let prev_mask: u8 = ((inside_mask << 1) | (inside_mask >> (N - 1))) & full_mask;
    let trans_mask: u8 = (inside_mask ^ prev_mask) & full_mask;

    debug_assert_eq!(
        trans_mask.count_ones(),
        2,
        "Convex polygon should have exactly 2 transitions"
    );

    let idx1 = trans_mask.trailing_zeros() as usize;
    let trans_mask_rem = trans_mask & !(1u8 << idx1);
    let idx2 = trans_mask_rem.trailing_zeros() as usize;

    let (entry_next, exit_next) = if (inside_mask & (1u8 << idx1)) != 0 {
        (idx1, idx2)
    } else {
        (idx2, idx1)
    };

    let entry_idx = if entry_next == 0 {
        N - 1
    } else {
        entry_next - 1
    };
    let exit_idx = if exit_next == 0 { N - 1 } else { exit_next - 1 };

    // Build output directly into `out` (only in the mixed case).
    out.len = 0;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection: edge (entry_idx -> entry_next) crosses out->in.
    // SAFETY: entry_idx and entry_next are in 0..N; dists[0..N] were written in Pass 1.
    let (d_entry, d_entry_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    push!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    // Surviving input vertices from entry_next to exit_idx (cyclic), inclusive.
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
        for i in entry_next..N {
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

    // Exit intersection: edge (exit_idx -> exit_next) crosses in->out.
    // SAFETY: exit_idx and exit_next are in 0..N; dists[0..N] were written in Pass 1.
    let (d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

// specialized "hot path" for Quads
#[inline(always)]
fn clip_convex_quad(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    debug_assert_eq!(poly.len, 4);

    let neg_eps = -hp.eps;

    // Pass 1: compute signed distances and an inside bitmask.
    //
    // Note: building `mask` directly avoids storing/loading a separate `[bool; 4]`.
    let mut dists = [0.0f64; 4];
    let mut mask: u8 = 0;
    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    unsafe {
        for i in 0..4 {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists[i] = d;
            mask |= ((d >= neg_eps) as u8) << i;
        }
    }

    // Fast paths
    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }
    if mask == 0xF {
        return ClipResult::Unchanged;
    }

    // Output via jump table.
    out.len = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    // Helper to compute intersection parameter along segment `in -> out`.
    #[inline(always)]
    fn get_t(d_in: f64, d_out: f64) -> f64 {
        d_in / (d_in - d_out)
    }

    // Macro to push a vertex by index
    macro_rules! push_v {
        ($idx:expr) => {{
            let i = $idx;
            let u = poly.us[i];
            let v = poly.vs[i];
            let vp = poly.vertex_planes[i];
            out.push_raw(u, v, vp, poly.edge_planes[i]);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection: writes `edge_plane = poly.edge_planes[edge_idx]`.
    macro_rules! push_entry {
        ($in_idx:expr, $out_idx:expr, $ep_idx:expr) => {{
            let d_in = dists[$in_idx];
            let d_out = dists[$out_idx];
            let t = get_t(d_in, d_out);

            let u = t.mul_add(poly.us[$out_idx] - poly.us[$in_idx], poly.us[$in_idx]);
            let v = t.mul_add(poly.vs[$out_idx] - poly.vs[$in_idx], poly.vs[$in_idx]);
            let edge_plane = poly.edge_planes[$ep_idx];
            let vp = (edge_plane, hp.plane_idx);

            out.push_raw(u, v, vp, edge_plane);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= edge_plane == usize::MAX;
            }
        }};
    }

    // Exit intersection: writes `edge_plane = hp.plane_idx`.
    macro_rules! push_exit {
        ($in_idx:expr, $out_idx:expr, $ep_idx:expr) => {{
            let d_in = dists[$in_idx];
            let d_out = dists[$out_idx];
            let t = get_t(d_in, d_out);

            let u = t.mul_add(poly.us[$out_idx] - poly.us[$in_idx], poly.us[$in_idx]);
            let v = t.mul_add(poly.vs[$out_idx] - poly.vs[$in_idx], poly.vs[$in_idx]);
            let edge_plane = poly.edge_planes[$ep_idx];
            let vp = (edge_plane, hp.plane_idx);

            out.push_raw(u, v, vp, hp.plane_idx);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= edge_plane == usize::MAX;
            }
        }};
    }

    #[cold]
    fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
        clip_convex_small_bool::<4>(poly, hp, out)
    }

    // THE JUMP TABLE
    // This tells the compiler the exact control flow graph.
    // It allows "Instruction Level Parallelism" (ILP).
    // e.g. In case 0b0011, the CPU can divide d0/(d0-d3) AND d1/(d1-d2) at the same time.
    match mask {
        // 1 Vertex Inside
        0b0001 => {
            // V0 in
            push_entry!(0, 3, 3); // Entry on edge 3->0
            push_v!(0);
            push_exit!(0, 1, 0); // Exit on edge 0->1
        }
        0b0010 => {
            // V1 in
            push_entry!(1, 0, 0);
            push_v!(1);
            push_exit!(1, 2, 1);
        }
        0b0100 => {
            // V2 in
            push_entry!(2, 1, 1);
            push_v!(2);
            push_exit!(2, 3, 2);
        }
        0b1000 => {
            // V3 in
            push_entry!(3, 2, 2);
            push_v!(3);
            push_exit!(3, 0, 3);
        }

        // 2 Vertices Inside (Sequential)
        0b0011 => {
            // V0, V1 in
            push_entry!(0, 3, 3); // Entry from 3->0
            push_v!(0);
            push_v!(1);
            push_exit!(1, 2, 1); // Exit to 1->2
        }
        0b0110 => {
            // V1, V2 in
            push_entry!(1, 0, 0);
            push_v!(1);
            push_v!(2);
            push_exit!(2, 3, 2);
        }
        0b1100 => {
            // V2, V3 in
            push_entry!(2, 1, 1);
            push_v!(2);
            push_v!(3);
            push_exit!(3, 0, 3);
        }
        0b1001 => {
            // V3, V0 in
            push_entry!(3, 2, 2);
            push_v!(3);
            push_v!(0);
            push_exit!(0, 1, 0);
        }

        // 3 Vertices Inside
        0b0111 => {
            // V0, V1, V2 in (V3 out)
            push_entry!(0, 3, 3);
            push_v!(0);
            push_v!(1);
            push_v!(2);
            push_exit!(2, 3, 2);
        }
        0b1110 => {
            // V1, V2, V3 in (V0 out)
            push_entry!(1, 0, 0);
            push_v!(1);
            push_v!(2);
            push_v!(3);
            push_exit!(3, 0, 3);
        }
        0b1101 => {
            // V0, V2, V3 in (V1 out)
            push_entry!(2, 1, 1);
            push_v!(2);
            push_v!(3);
            push_v!(0);
            push_exit!(0, 1, 0);
        }
        0b1011 => {
            // V0, V1, V3 in (V2 out)
            push_entry!(3, 2, 2);
            push_v!(3);
            push_v!(0);
            push_v!(1);
            push_exit!(1, 2, 1);
        }

        // Non-convex / Disconnected cases
        // (e.g. 0101 - V0 and V2 in, V1 and V3 out)
        // Mathematically impossible for a convex polygon clipped by a line.
        _ => return fallback(poly, hp, out),
    }

    // Finalize
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

struct ClipCtx<'a> {
    poly: &'a PolyBuffer,
    hp: &'a HalfPlane,
    out: &'a mut PolyBuffer,
    max_r2: f64,
    has_bounding: bool,
    track_bounding: bool,
}

impl<'a> ClipCtx<'a> {
    #[inline(always)]
    fn new(poly: &'a PolyBuffer, hp: &'a HalfPlane, out: &'a mut PolyBuffer) -> Self {
        out.len = 0;
        Self {
            poly,
            hp,
            out,
            max_r2: 0.0,
            has_bounding: false,
            track_bounding: poly.has_bounding_ref,
        }
    }

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    #[inline(always)]
    fn get_t(d_in: f64, d_out: f64) -> f64 {
        d_in / (d_in - d_out)
    }

    #[inline(always)]
    fn push_v<const IDX: usize>(&mut self) {
        let u = self.poly.us[IDX];
        let v = self.poly.vs[IDX];
        let vp = self.poly.vertex_planes[IDX];
        self.out.push_raw(u, v, vp, self.poly.edge_planes[IDX]);
        let r2 = Self::r2_of(u, v);
        if r2 > self.max_r2 {
            self.max_r2 = r2;
        }
        if self.track_bounding {
            self.has_bounding |= vp.0 == usize::MAX;
        }
    }

    #[inline(always)]
    fn push_entry<const LANES: usize, const IN: usize, const OUT: usize, const EDGE: usize>(
        &mut self,
        dists: &[f64; LANES],
    ) {
        let d_in = dists[IN];
        let d_out = dists[OUT];
        let t = Self::get_t(d_in, d_out);

        let u = t.mul_add(
            self.poly.us[OUT] - self.poly.us[IN],
            self.poly.us[IN],
        );
        let v = t.mul_add(
            self.poly.vs[OUT] - self.poly.vs[IN],
            self.poly.vs[IN],
        );
        let edge_plane = self.poly.edge_planes[EDGE];
        let vp = (edge_plane, self.hp.plane_idx);

        self.out.push_raw(u, v, vp, edge_plane);
        let r2 = Self::r2_of(u, v);
        if r2 > self.max_r2 {
            self.max_r2 = r2;
        }
        if self.track_bounding {
            self.has_bounding |= edge_plane == usize::MAX;
        }
    }

    #[inline(always)]
    fn push_exit<const LANES: usize, const IN: usize, const OUT: usize, const EDGE: usize>(
        &mut self,
        dists: &[f64; LANES],
    ) {
        let d_in = dists[IN];
        let d_out = dists[OUT];
        let t = Self::get_t(d_in, d_out);

        let u = t.mul_add(
            self.poly.us[OUT] - self.poly.us[IN],
            self.poly.us[IN],
        );
        let v = t.mul_add(
            self.poly.vs[OUT] - self.poly.vs[IN],
            self.poly.vs[IN],
        );
        let edge_plane = self.poly.edge_planes[EDGE];
        let vp = (edge_plane, self.hp.plane_idx);

        self.out.push_raw(u, v, vp, self.hp.plane_idx);
        let r2 = Self::r2_of(u, v);
        if r2 > self.max_r2 {
            self.max_r2 = r2;
        }
        if self.track_bounding {
            self.has_bounding |= edge_plane == usize::MAX;
        }
    }

    #[inline(always)]
    fn finish(&mut self) {
        self.out.max_r2 = self.max_r2;
        self.out.has_bounding_ref = if self.track_bounding {
            self.has_bounding
        } else {
            false
        };
    }
}

mod clip_jump_tables {
    use super::*;

    pub(super) mod tables {
        include!("clip_tables/table_n3.rs");
        include!("clip_tables/table_n4.rs");
        include!("clip_tables/table_n5.rs");
        include!("clip_tables/table_n6.rs");
        include!("clip_tables/table_n7.rs");
        include!("clip_tables/table_n8.rs");
    }

    #[inline(always)]
    pub(super) fn run_n3(mask: u8, dists: &[f64; 4], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<3>(poly, hp, out)
        }
        include!("clip_tables/ngon_n3.rs");
        ctx.finish();
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_table_n3(
        mask: u8,
        dists: &[f64; 4],
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        debug_assert_eq!(poly.len, 3);
        debug_assert!(mask != 0 && mask != 0b111);

        let (start, len) = tables::TABLE_N3[mask as usize];
        if len == 0 {
            return clip_convex_small_bool::<3>(poly, hp, out);
        }

        // Layout:
        // - entry intersection: prev(start) -> start
        // - len vertices starting at start
        // - exit intersection: last -> next(last)
        let n = 3usize;
        let start = start as usize;
        let len = len as usize;

        let prev = if start == 0 { n - 1 } else { start - 1 };
        let last = (start + len - 1) % n;
        let next = (last + 1) % n;

        out.len = 0;
        let mut max_r2 = 0.0f64;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        #[inline(always)]
        fn r2_of(u: f64, v: f64) -> f64 {
            u.mul_add(u, v * v)
        }

        #[inline(always)]
        fn get_t(d_in: f64, d_out: f64) -> f64 {
            d_in / (d_in - d_out)
        }

        macro_rules! push_raw_track {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                let u = $u;
                let v = $v;
                let vp = $vp;
                out.push_raw(u, v, vp, $ep);
                let r2 = r2_of(u, v);
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }};
        }

        // Entry
        // Match the jump-table entry parameterization: in=start (inside), out=prev (outside).
        let t_entry = get_t(dists[start], dists[prev]);
        let entry_u = t_entry.mul_add(poly.us[prev] - poly.us[start], poly.us[start]);
        let entry_v = t_entry.mul_add(poly.vs[prev] - poly.vs[start], poly.vs[start]);
        let entry_ep = poly.edge_planes[prev];
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        // Vertices
        for k in 0..len {
            let i = (start + k) % n;
            push_raw_track!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }

        // Exit
        let t_exit = get_t(dists[last], dists[next]);
        let exit_u = t_exit.mul_add(poly.us[next] - poly.us[last], poly.us[last]);
        let exit_v = t_exit.mul_add(poly.vs[next] - poly.vs[last], poly.vs[last]);
        let exit_ep = poly.edge_planes[last];
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_n4(mask: u8, dists: &[f64; 4], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<4>(poly, hp, out)
        }
        include!("clip_tables/ngon_n4.rs");
        ctx.finish();
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_table_n4(
        mask: u8,
        dists: &[f64; 4],
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        debug_assert_eq!(poly.len, 4);
        debug_assert!(mask != 0 && mask != 0b1111);

        let (start, len) = tables::TABLE_N4[mask as usize];
        if len == 0 {
            return clip_convex_small_bool::<4>(poly, hp, out);
        }

        let n = 4usize;
        let start = start as usize;
        let len = len as usize;

        let prev = if start == 0 { n - 1 } else { start - 1 };
        let last = (start + len - 1) % n;
        let next = (last + 1) % n;

        out.len = 0;
        let mut max_r2 = 0.0f64;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        #[inline(always)]
        fn r2_of(u: f64, v: f64) -> f64 {
            u.mul_add(u, v * v)
        }

        #[inline(always)]
        fn get_t(d_in: f64, d_out: f64) -> f64 {
            d_in / (d_in - d_out)
        }

        macro_rules! push_raw_track {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                let u = $u;
                let v = $v;
                let vp = $vp;
                out.push_raw(u, v, vp, $ep);
                let r2 = r2_of(u, v);
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }};
        }

        // Entry
        // Match the jump-table entry parameterization: in=start (inside), out=prev (outside).
        let t_entry = get_t(dists[start], dists[prev]);
        let entry_u = t_entry.mul_add(poly.us[prev] - poly.us[start], poly.us[start]);
        let entry_v = t_entry.mul_add(poly.vs[prev] - poly.vs[start], poly.vs[start]);
        let entry_ep = poly.edge_planes[prev];
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        // Vertices
        for k in 0..len {
            let i = (start + k) % n;
            push_raw_track!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }

        // Exit
        let t_exit = get_t(dists[last], dists[next]);
        let exit_u = t_exit.mul_add(poly.us[next] - poly.us[last], poly.us[last]);
        let exit_v = t_exit.mul_add(poly.vs[next] - poly.vs[last], poly.vs[last]);
        let exit_ep = poly.edge_planes[last];
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_n5(mask: u8, dists: &[f64; 8], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<5>(poly, hp, out)
        }
        include!("clip_tables/ngon_n5.rs");
        ctx.finish();
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_table_n5(
        mask: u8,
        dists: &[f64; 8],
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        debug_assert_eq!(poly.len, 5);
        debug_assert!(mask != 0 && mask != 0b1_1111);

        let (start, len) = tables::TABLE_N5[mask as usize];
        if len == 0 {
            return clip_convex_small_bool::<5>(poly, hp, out);
        }

        let n = 5usize;
        let start = start as usize;
        let len = len as usize;

        let prev = if start == 0 { n - 1 } else { start - 1 };
        let last = (start + len - 1) % n;
        let next = (last + 1) % n;

        out.len = 0;
        let mut max_r2 = 0.0f64;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        #[inline(always)]
        fn r2_of(u: f64, v: f64) -> f64 {
            u.mul_add(u, v * v)
        }

        #[inline(always)]
        fn get_t(d_in: f64, d_out: f64) -> f64 {
            d_in / (d_in - d_out)
        }

        macro_rules! push_raw_track {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                let u = $u;
                let v = $v;
                let vp = $vp;
                out.push_raw(u, v, vp, $ep);
                let r2 = r2_of(u, v);
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }};
        }

        // Match the jump-table entry parameterization: in=start (inside), out=prev (outside).
        let t_entry = get_t(dists[start], dists[prev]);
        let entry_u = t_entry.mul_add(poly.us[prev] - poly.us[start], poly.us[start]);
        let entry_v = t_entry.mul_add(poly.vs[prev] - poly.vs[start], poly.vs[start]);
        let entry_ep = poly.edge_planes[prev];
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        for k in 0..len {
            let i = (start + k) % n;
            push_raw_track!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }

        let t_exit = get_t(dists[last], dists[next]);
        let exit_u = t_exit.mul_add(poly.us[next] - poly.us[last], poly.us[last]);
        let exit_v = t_exit.mul_add(poly.vs[next] - poly.vs[last], poly.vs[last]);
        let exit_ep = poly.edge_planes[last];
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_n6(mask: u8, dists: &[f64; 8], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<6>(poly, hp, out)
        }
        include!("clip_tables/ngon_n6.rs");
        ctx.finish();
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_table_n6(
        mask: u8,
        dists: &[f64; 8],
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        debug_assert_eq!(poly.len, 6);
        debug_assert!(mask != 0 && mask != 0b11_1111);

        let (start, len) = tables::TABLE_N6[mask as usize];
        if len == 0 {
            return clip_convex_small_bool::<6>(poly, hp, out);
        }

        let n = 6usize;
        let start = start as usize;
        let len = len as usize;

        let prev = if start == 0 { n - 1 } else { start - 1 };
        let last = (start + len - 1) % n;
        let next = (last + 1) % n;

        out.len = 0;
        let mut max_r2 = 0.0f64;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        #[inline(always)]
        fn r2_of(u: f64, v: f64) -> f64 {
            u.mul_add(u, v * v)
        }

        #[inline(always)]
        fn get_t(d_in: f64, d_out: f64) -> f64 {
            d_in / (d_in - d_out)
        }

        macro_rules! push_raw_track {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                let u = $u;
                let v = $v;
                let vp = $vp;
                out.push_raw(u, v, vp, $ep);
                let r2 = r2_of(u, v);
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }};
        }

        // Match the jump-table entry parameterization: in=start (inside), out=prev (outside).
        let t_entry = get_t(dists[start], dists[prev]);
        let entry_u = t_entry.mul_add(poly.us[prev] - poly.us[start], poly.us[start]);
        let entry_v = t_entry.mul_add(poly.vs[prev] - poly.vs[start], poly.vs[start]);
        let entry_ep = poly.edge_planes[prev];
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        for k in 0..len {
            let i = (start + k) % n;
            push_raw_track!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }

        let t_exit = get_t(dists[last], dists[next]);
        let exit_u = t_exit.mul_add(poly.us[next] - poly.us[last], poly.us[last]);
        let exit_v = t_exit.mul_add(poly.vs[next] - poly.vs[last], poly.vs[last]);
        let exit_ep = poly.edge_planes[last];
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_n7(mask: u8, dists: &[f64; 8], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<7>(poly, hp, out)
        }
        include!("clip_tables/ngon_n7.rs");
        ctx.finish();
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_table_n7(
        mask: u8,
        dists: &[f64; 8],
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        debug_assert_eq!(poly.len, 7);
        debug_assert!(mask != 0 && mask != 0b111_1111);

        let (start, len) = tables::TABLE_N7[mask as usize];
        if len == 0 {
            return clip_convex_small_bool::<7>(poly, hp, out);
        }

        let n = 7usize;
        let start = start as usize;
        let len = len as usize;

        let prev = if start == 0 { n - 1 } else { start - 1 };
        let last = (start + len - 1) % n;
        let next = (last + 1) % n;

        out.len = 0;
        let mut max_r2 = 0.0f64;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        #[inline(always)]
        fn r2_of(u: f64, v: f64) -> f64 {
            u.mul_add(u, v * v)
        }

        #[inline(always)]
        fn get_t(d_in: f64, d_out: f64) -> f64 {
            d_in / (d_in - d_out)
        }

        macro_rules! push_raw_track {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                let u = $u;
                let v = $v;
                let vp = $vp;
                out.push_raw(u, v, vp, $ep);
                let r2 = r2_of(u, v);
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }};
        }

        // Match the jump-table entry parameterization: in=start (inside), out=prev (outside).
        let t_entry = get_t(dists[start], dists[prev]);
        let entry_u = t_entry.mul_add(poly.us[prev] - poly.us[start], poly.us[start]);
        let entry_v = t_entry.mul_add(poly.vs[prev] - poly.vs[start], poly.vs[start]);
        let entry_ep = poly.edge_planes[prev];
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        for k in 0..len {
            let i = (start + k) % n;
            push_raw_track!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }

        let t_exit = get_t(dists[last], dists[next]);
        let exit_u = t_exit.mul_add(poly.us[next] - poly.us[last], poly.us[last]);
        let exit_v = t_exit.mul_add(poly.vs[next] - poly.vs[last], poly.vs[last]);
        let exit_ep = poly.edge_planes[last];
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_n8(mask: u8, dists: &[f64; 8], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<8>(poly, hp, out)
        }
        include!("clip_tables/ngon_n8.rs");
        ctx.finish();
        ClipResult::Changed
    }

    #[inline(always)]
    pub(super) fn run_table_n8(
        mask: u8,
        dists: &[f64; 8],
        poly: &PolyBuffer,
        hp: &HalfPlane,
        out: &mut PolyBuffer,
    ) -> ClipResult {
        debug_assert_eq!(poly.len, 8);
        debug_assert!(mask != 0 && mask != 0xFF);

        let (start, len) = tables::TABLE_N8[mask as usize];
        if len == 0 {
            return clip_convex_small_bool::<8>(poly, hp, out);
        }

        let n = 8usize;
        let start = start as usize;
        let len = len as usize;

        let prev = if start == 0 { n - 1 } else { start - 1 };
        let last = (start + len - 1) % n;
        let next = (last + 1) % n;

        out.len = 0;
        let mut max_r2 = 0.0f64;
        let mut has_bounding = false;
        let track_bounding = poly.has_bounding_ref;

        #[inline(always)]
        fn r2_of(u: f64, v: f64) -> f64 {
            u.mul_add(u, v * v)
        }

        #[inline(always)]
        fn get_t(d_in: f64, d_out: f64) -> f64 {
            d_in / (d_in - d_out)
        }

        macro_rules! push_raw_track {
            ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                let u = $u;
                let v = $v;
                let vp = $vp;
                out.push_raw(u, v, vp, $ep);
                let r2 = r2_of(u, v);
                if r2 > max_r2 {
                    max_r2 = r2;
                }
                if track_bounding {
                    has_bounding |= vp.0 == usize::MAX;
                }
            }};
        }

        // Match the jump-table entry parameterization: in=start (inside), out=prev (outside).
        let t_entry = get_t(dists[start], dists[prev]);
        let entry_u = t_entry.mul_add(poly.us[prev] - poly.us[start], poly.us[start]);
        let entry_v = t_entry.mul_add(poly.vs[prev] - poly.vs[start], poly.vs[start]);
        let entry_ep = poly.edge_planes[prev];
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        for k in 0..len {
            let i = (start + k) % n;
            push_raw_track!(
                poly.us[i],
                poly.vs[i],
                poly.vertex_planes[i],
                poly.edge_planes[i]
            );
        }

        let t_exit = get_t(dists[last], dists[next]);
        let exit_u = t_exit.mul_add(poly.us[next] - poly.us[last], poly.us[last]);
        let exit_v = t_exit.mul_add(poly.vs[next] - poly.vs[last], poly.vs[last]);
        let exit_ep = poly.edge_planes[last];
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

        out.max_r2 = max_r2;
        out.has_bounding_ref = if track_bounding { has_bounding } else { false };
        ClipResult::Changed
    }
}

#[inline(always)]
fn run_jump_table_4<const N: usize>(mask: u8, dists: &[f64; 4], ctx: &mut ClipCtx<'_>) -> ClipResult {
    if N == 3 {
        clip_jump_tables::run_n3(mask, dists, ctx)
    } else if N == 4 {
        clip_jump_tables::run_n4(mask, dists, ctx)
    } else {
        unreachable!("run_jump_table_4 only supports N=3 or N=4")
    }
}

#[inline(always)]
fn run_jump_table_8<const N: usize>(mask: u8, dists: &[f64; 8], ctx: &mut ClipCtx<'_>) -> ClipResult {
    if N == 5 {
        clip_jump_tables::run_n5(mask, dists, ctx)
    } else if N == 6 {
        clip_jump_tables::run_n6(mask, dists, ctx)
    } else if N == 7 {
        clip_jump_tables::run_n7(mask, dists, ctx)
    } else if N == 8 {
        clip_jump_tables::run_n8(mask, dists, ctx)
    } else {
        unreachable!("run_jump_table_8 only supports N=5..=8")
    }
}

#[inline(always)]
#[cfg(any(test, feature = "microbench"))]
fn clip_convex_table_ngon<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);

    let neg_eps = -hp.eps;
    let full_mask_val: u8 = ((1u16 << N) - 1) as u8;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    if N == 3 || N == 4 {
        let mut dists = [0.0f64; 4];
        let mut mask: u8 = 0;
        unsafe {
            for i in 0..N {
                let d = hp.signed_dist(*us.add(i), *vs.add(i));
                *dists.get_unchecked_mut(i) = d;
                mask |= ((d >= neg_eps) as u8) << i;
            }
        }

        if mask == 0 {
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            return ClipResult::Changed;
        }
        if mask == full_mask_val {
            return ClipResult::Unchanged;
        }

        match N {
            3 => clip_jump_tables::run_table_n3(mask, &dists, poly, hp, out),
            4 => clip_jump_tables::run_table_n4(mask, &dists, poly, hp, out),
            _ => unreachable!(),
        }
    } else {
        let mut dists = [0.0f64; 8];
        let mut mask: u8 = 0;
        unsafe {
            for i in 0..N {
                let d = hp.signed_dist(*us.add(i), *vs.add(i));
                *dists.get_unchecked_mut(i) = d;
                mask |= ((d >= neg_eps) as u8) << i;
            }
        }

        if mask == 0 {
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            return ClipResult::Changed;
        }
        if mask == full_mask_val {
            return ClipResult::Unchanged;
        }

        match N {
            5 => clip_jump_tables::run_table_n5(mask, &dists, poly, hp, out),
            6 => clip_jump_tables::run_table_n6(mask, &dists, poly, hp, out),
            7 => clip_jump_tables::run_table_n7(mask, &dists, poly, hp, out),
            8 => clip_jump_tables::run_table_n8(mask, &dists, poly, hp, out),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
#[cfg(any(test, feature = "microbench"))]
fn clip_convex_match_ngon<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);

    let neg_eps = -hp.eps;
    let full_mask_val: u8 = ((1u16 << N) - 1) as u8;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    if N == 3 || N == 4 {
        let mut dists = [0.0f64; 4];
        let mut mask: u8 = 0;
        unsafe {
            for i in 0..N {
                let d = hp.signed_dist(*us.add(i), *vs.add(i));
                *dists.get_unchecked_mut(i) = d;
                mask |= ((d >= neg_eps) as u8) << i;
            }
        }
        if mask == 0 {
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            return ClipResult::Changed;
        }
        if mask == full_mask_val {
            return ClipResult::Unchanged;
        }
        let mut ctx = ClipCtx::new(poly, hp, out);
        run_jump_table_4::<N>(mask, &dists, &mut ctx)
    } else {
        let mut dists = [0.0f64; 8];
        let mut mask: u8 = 0;
        unsafe {
            for i in 0..N {
                let d = hp.signed_dist(*us.add(i), *vs.add(i));
                *dists.get_unchecked_mut(i) = d;
                mask |= ((d >= neg_eps) as u8) << i;
            }
        }
        if mask == 0 {
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            return ClipResult::Changed;
        }
        if mask == full_mask_val {
            return ClipResult::Unchanged;
        }
        let mut ctx = ClipCtx::new(poly, hp, out);
        run_jump_table_8::<N>(mask, &dists, &mut ctx)
    }
}

#[inline(always)]
fn clip_convex_simd_4lane<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert!(N == 3 || N == 4);
    debug_assert_eq!(poly.len, N);

    use std::simd::prelude::*;
    use std::simd::f64x4;

    let us_simd = f64x4::from_slice(&poly.us[0..4]);
    let vs_simd = f64x4::from_slice(&poly.vs[0..4]);

    let a_simd = f64x4::splat(hp.a);
    let b_simd = f64x4::splat(hp.b);
    let c_simd = f64x4::splat(hp.c);
    let neg_eps_simd = f64x4::splat(-hp.eps);

    let dists_vec = a_simd.mul_add(us_simd, b_simd.mul_add(vs_simd, c_simd));
    let mask_simd = dists_vec.simd_ge(neg_eps_simd);
    let full_simd_mask = mask_simd.to_bitmask() as u8;
    let full_mask_val: u8 = ((1u16 << N) - 1) as u8;
    let mask = full_simd_mask & full_mask_val;

    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }
    if mask == full_mask_val {
        return ClipResult::Unchanged;
    }

    let dists_arr = dists_vec.to_array();
    let mut ctx = ClipCtx::new(poly, hp, out);
    run_jump_table_4::<N>(mask, &dists_arr, &mut ctx)
}

#[inline(always)]
fn clip_convex_simd_8lane<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert!((5..=8).contains(&N));
    debug_assert_eq!(poly.len, N);

    use std::simd::prelude::*;
    use std::simd::f64x8;

    let us_simd = f64x8::from_slice(&poly.us[0..8]);
    let vs_simd = f64x8::from_slice(&poly.vs[0..8]);

    let a_simd = f64x8::splat(hp.a);
    let b_simd = f64x8::splat(hp.b);
    let c_simd = f64x8::splat(hp.c);
    let neg_eps_simd = f64x8::splat(-hp.eps);

    let dists_vec = a_simd.mul_add(us_simd, b_simd.mul_add(vs_simd, c_simd));
    let mask_simd = dists_vec.simd_ge(neg_eps_simd);
    let full_simd_mask = mask_simd.to_bitmask() as u8;
    let full_mask_val: u8 = ((1u16 << N) - 1) as u8;
    let mask = full_simd_mask & full_mask_val;

    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }
    if mask == full_mask_val {
        return ClipResult::Unchanged;
    }

    let dists_arr = dists_vec.to_array();
    let mut ctx = ClipCtx::new(poly, hp, out);
    run_jump_table_8::<N>(mask, &dists_arr, &mut ctx)
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_simd_tri(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    clip_convex_simd_4lane::<3>(poly, hp, out)
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_simd_quad(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    clip_convex_simd_4lane::<4>(poly, hp, out)
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_simd_pent(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    clip_convex_simd_8lane::<5>(poly, hp, out)
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_simd_hex(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    clip_convex_simd_8lane::<6>(poly, hp, out)
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_simd_hept(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    clip_convex_simd_8lane::<7>(poly, hp, out)
}

#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_simd_oct(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    clip_convex_simd_8lane::<8>(poly, hp, out)
}

#[inline(always)]
#[cfg(any(test, feature = "microbench"))]
fn clip_convex_simd<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    match N {
        3 => clip_convex_simd_tri(poly, hp, out),
        4 => clip_convex_simd_quad(poly, hp, out),
        5 => clip_convex_simd_pent(poly, hp, out),
        6 => clip_convex_simd_hex(poly, hp, out),
        7 => clip_convex_simd_hept(poly, hp, out),
        8 => clip_convex_simd_oct(poly, hp, out),
        _ => unreachable!("clip_convex_simd only supports N=3..=8"),
    }
}

/// Bool array variant: uses `[bool; N]` instead of bitmask for classification.
/// Simpler logic, no bit manipulation - test if it's competitive for small N.
#[cfg_attr(feature = "profiling", inline(never))]
fn clip_convex_small_bool<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    // Pass 1: classify vertices, track any_inside/all_inside in one pass
    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    // SAFETY: i < N <= 8 <= MAX_POLY_VERTICES
    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            let is_inside = d >= neg_eps;
            inside[i] = is_inside;
            any_inside |= is_inside;
            all_inside &= is_inside;
        }
    }

    // Fast path: all outside
    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    // Fast path: all inside
    if all_inside {
        return ClipResult::Unchanged;
    }

    // Mixed case: find the two transition indices (outside -> inside and inside -> outside)
    // Single pass: find both transitions at once
    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        // Early exit once both transitions are found
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    // Build output
    out.len = 0;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection: edge (entry_idx -> entry_next) crosses out->in
    let (d_entry, d_entry_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    push!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    // Surviving input vertices from entry_next to exit_idx (cyclic), inclusive
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
        for i in entry_next..N {
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

    // Exit intersection: edge (exit_idx -> exit_next) crosses in->out
    let (d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

fn clip_convex_bitmask(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;
    debug_assert!(n <= 64, "Polygon too large for u64 bitmask");

    // Pass 1: classify vertices using bitset
    let neg_eps = -hp.eps;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    // We want an uninitialized stack array without paying for a zero-fill.
    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; MAX_POLY_VERTICES] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let mut mask = 0u64;
    let full_mask: u64 = if n == 64 { !0u64 } else { (1u64 << n) - 1 };

    // SAFETY: i < n <= MAX_POLY_VERTICES
    unsafe {
        let mut bit = 1u64;
        for i in 0..n {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists.get_unchecked_mut(i).write(d);
            mask |= (0u64.wrapping_sub((d >= neg_eps) as u64)) & bit;
            bit <<= 1;
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

    let us = &poly.us[..n];
    let vs = &poly.vs[..n];
    let edge_planes = &poly.edge_planes[..n];

    // Calculate transition data only for the two intersection points
    // We need d0 (dist at i-1) and d1 (dist at i) for the intersection formula.
    // entry.next = entry_next. idx = entry_next - 1 (modulo n).

    let calc_transition = |next_idx: usize| -> (usize, usize, f64, f64) {
        let prev_idx = if next_idx == 0 { n - 1 } else { next_idx - 1 };
        // SAFETY: prev_idx and next_idx are in 0..n; dists[0..n] were written in Pass 1.
        unsafe {
            let d0 = dists.get_unchecked(prev_idx).assume_init_read();
            let d1 = dists.get_unchecked(next_idx).assume_init_read();
            (prev_idx, next_idx, d0, d1)
        }
    };

    let (entry_idx, entry_next_idx, entry_d0, entry_d1) = calc_transition(entry_next);
    let (exit_idx, exit_next_idx, exit_d0, exit_d1) = calc_transition(exit_next);

    // Compute intersection points
    let t_entry = entry_d0 / (entry_d0 - entry_d1);
    // SAFETY: entry_idx and entry_next_idx are in 0..n.
    let (entry_u0, entry_u1, entry_v0, entry_v1, entry_edge_plane) = unsafe {
        (
            *us.get_unchecked(entry_idx),
            *us.get_unchecked(entry_next_idx),
            *vs.get_unchecked(entry_idx),
            *vs.get_unchecked(entry_next_idx),
            *edge_planes.get_unchecked(entry_idx),
        )
    };
    let entry_u = t_entry.mul_add(entry_u1 - entry_u0, entry_u0);
    let entry_v = t_entry.mul_add(entry_v1 - entry_v0, entry_v0);
    let entry_pt = (entry_u, entry_v);

    let t_exit = exit_d0 / (exit_d0 - exit_d1);
    // SAFETY: exit_idx and exit_next_idx are in 0..n.
    let (exit_u0, exit_u1, exit_v0, exit_v1, exit_edge_plane) = unsafe {
        (
            *us.get_unchecked(exit_idx),
            *us.get_unchecked(exit_next_idx),
            *vs.get_unchecked(exit_idx),
            *vs.get_unchecked(exit_next_idx),
            *edge_planes.get_unchecked(exit_idx),
        )
    };
    let exit_u = t_exit.mul_add(exit_u1 - exit_u0, exit_u0);
    let exit_v = t_exit.mul_add(exit_v1 - exit_v0, exit_v0);
    let exit_pt = (exit_u, exit_v);

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

    ClipResult::Changed
}

/// Clip a convex polygon by a half-plane (standalone function to avoid borrow conflicts).
///
/// # Returns
/// - `Unchanged` if all vertices are inside the half-plane. **`out` is not modified.**
/// - `Changed` if the polygon was clipped (result written to `out`)
/// - `TooManyVertices` if the result would exceed `MAX_POLY_VERTICES`
fn clip_convex(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;
    #[cfg(feature = "timing")]
    clip_convex_stats::record_call(n);

    debug_assert!(n >= 3, "clip_convex expects poly.len >= 3, got {}", n);

    // O(1) redundant-plane early-out:
    // If the entire polygon lies within the disk u²+v² <= max_r2, and that disk is fully inside
    // the half-plane, then the half-plane cannot clip the polygon.
    //
    // For linear form d(u,v) = a*u + b*v + c, min over ||(u,v)|| <= r is c - ||(a,b)|| * r.
    // We classify inside with d >= -eps, so require: c - ||(a,b)||*r >= -eps.
    // Let t = c + eps; require t >= ||(a,b)|| * r. Avoid sqrt by squaring.
    let max_r2 = poly.max_r2;
    if !poly.has_bounding_ref && max_r2 > 0.0 {
        let t = hp.c + hp.eps;
        if t >= 0.0 {
            if t * t >= hp.ab2 * max_r2 {
                debug_assert!(
                    (0..n).all(|i| hp.signed_dist(poly.us[i], poly.vs[i]) >= -hp.eps),
                    "clip_convex early-out returned Unchanged, but a vertex was outside"
                );
                #[cfg(feature = "timing")]
                clip_convex_stats::record_early_unchanged(n, !poly.has_bounding_ref);
                return ClipResult::Unchanged;
            }
        }
    }
    match n {
        3 => clip_convex_simd_tri(poly, hp, out),
        4 => clip_convex_simd_quad(poly, hp, out),
        5 => clip_convex_simd_pent(poly, hp, out),
        6 => clip_convex_simd_hex(poly, hp, out),
        7 => clip_convex_simd_hept(poly, hp, out),
        8 => clip_convex_simd_oct(poly, hp, out),
        _ => clip_convex_bitmask(poly, hp, out),
    }
}

/// Build the output polygon from entry to exit.
///
/// `TRACK_BOUNDING`: if true, tracks whether bounding refs survive (slow path).
/// If false, assumes already bounded and skips sentinel checks (fast path).
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
    let us = &poly.us[..n];
    let vs = &poly.vs[..n];
    let vertex_planes = &poly.vertex_planes[..n];
    let edge_planes = &poly.edge_planes[..n];
    if entry_next <= exit_idx {
        let us_r = &us[entry_next..(exit_idx + 1)];
        let vs_r = &vs[entry_next..(exit_idx + 1)];
        let vps_r = &vertex_planes[entry_next..(exit_idx + 1)];
        let eps_r = &edge_planes[entry_next..(exit_idx + 1)];
        for (((&u, &v), &vp), &ep) in us_r.iter().zip(vs_r).zip(vps_r).zip(eps_r) {
            push!(u, v, vp, ep);
        }
    } else {
        let us_r = &us[entry_next..n];
        let vs_r = &vs[entry_next..n];
        let vps_r = &vertex_planes[entry_next..n];
        let eps_r = &edge_planes[entry_next..n];
        for (((&u, &v), &vp), &ep) in us_r.iter().zip(vs_r).zip(vps_r).zip(eps_r) {
            push!(u, v, vp, ep);
        }

        let us_r = &us[0..(exit_idx + 1)];
        let vs_r = &vs[0..(exit_idx + 1)];
        let vps_r = &vertex_planes[0..(exit_idx + 1)];
        let eps_r = &edge_planes[0..(exit_idx + 1)];
        for (((&u, &v), &vp), &ep) in us_r.iter().zip(vs_r).zip(vps_r).zip(eps_r) {
            push!(u, v, vp, ep);
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
    #[allow(dead_code)]
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

#[cfg(feature = "microbench")]
pub(crate) fn run_clip_convex_microbench() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    let target_ms: u64 = std::env::var("S2_VORONOI_BENCH_TARGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200)
        .clamp(10, 10_000);
    let samples: usize = std::env::var("S2_VORONOI_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(9)
        .clamp(3, 101);

    fn median(mut xs: Vec<f64>) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    fn make_regular_poly<const N: usize>(radius: f64) -> PolyBuffer {
        let mut p = PolyBuffer::new();
        p.len = N;
        p.max_r2 = 0.0;
        p.has_bounding_ref = false;
        for i in 0..N {
            let theta = std::f64::consts::TAU * (i as f64) / (N as f64);
            let (s, c) = theta.sin_cos();
            let u = radius * c;
            let v = radius * s;
            p.us[i] = u;
            p.vs[i] = v;
            p.vertex_planes[i] = (i, (i + 1) % N);
            p.edge_planes[i] = i;
            p.max_r2 = p.max_r2.max(u * u + v * v);
        }
        p
    }

    fn calibrate(target: Duration, mut f: impl FnMut(u64)) -> u64 {
        let mut iters = 1u64;
        loop {
            let t0 = Instant::now();
            f(iters);
            let dt = t0.elapsed();
            if dt >= target {
                return iters;
            }
            iters = iters.saturating_mul(2);
            if iters == 0 {
                return u64::MAX;
            }
        }
    }

    fn bench_ns_per_call(
        label: &str,
        target: Duration,
        samples: usize,
        calls_per_iter: u64,
        mut f: impl FnMut(u64),
    ) -> (f64, f64) {
        let mut runs = Vec::with_capacity(samples);

        // Warmup.
        f(10_000);

        let iters = calibrate(target, |n| f(n));
        for _ in 0..samples {
            let t0 = Instant::now();
            f(iters);
            let dt = t0.elapsed();
            let total_calls = (iters as f64) * (calls_per_iter as f64);
            let ns = dt.as_secs_f64() * 1e9 / total_calls;
            runs.push(ns);
        }

        let med = median(runs.clone());
        let min = runs
            .into_iter()
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        eprintln!("{label:<32} median {med:>10.2} ns/call  min {min:>10.2}");
        (med, min)
    }

    fn run_for<const N: usize>(target: Duration, samples: usize) {
        let poly = make_regular_poly::<N>(1.0);

        // Mixed: alternate a small set of planes, ensuring work stays in the "changed" regime.
        let hps = [
            HalfPlane::new_unnormalized(1.0, 0.0, -0.2, 1_000_000),
            HalfPlane::new_unnormalized(-1.0, 0.0, 0.2, 1_000_001),
            HalfPlane::new_unnormalized(0.0, 1.0, -0.2, 1_000_002),
            HalfPlane::new_unnormalized(0.0, -1.0, 0.2, 1_000_003),
        ];
        // Unchanged: vary planes to avoid the optimizer hoisting/eliminating work.
        // All of these contain the unit-radius regular polygon comfortably.
        let hps_unchanged = [
            HalfPlane::new_unnormalized(1.0, 0.0, 2.0, 1_000_004),
            HalfPlane::new_unnormalized(-1.0, 0.0, 2.0, 1_000_005),
            HalfPlane::new_unnormalized(0.0, 1.0, 2.0, 1_000_006),
            HalfPlane::new_unnormalized(0.0, -1.0, 2.0, 1_000_007),
        ];

        // Pre-allocate output buffers.
        let mut out_a = PolyBuffer::new();
        let mut out_b = PolyBuffer::new();
        let mut out_c = PolyBuffer::new();
        let mut out_d = PolyBuffer::new();
        let mut out_s = PolyBuffer::new();
        let mut out_q = PolyBuffer::new();

        // Sanity: ensure the intended regimes.
        assert!(matches!(
            clip_convex_small::<N>(&poly, &hps[0], &mut out_a),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_bitmask(&poly, &hps[0], &mut out_b),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps[0], &mut out_c),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_match_ngon::<N>(&poly, &hps[0], &mut out_d),
            ClipResult::Changed
        ));
        assert!(matches!(
            clip_convex_simd::<N>(&poly, &hps[0], &mut out_s),
            ClipResult::Changed
        ));
        if N == 4 {
            assert!(matches!(
                clip_convex_quad(&poly, &hps[0], &mut out_q),
                ClipResult::Changed
            ));
        }
        out_a.len = 13;
        out_a.us[0] = 123.0;
        out_b.len = 13;
        out_b.us[0] = 123.0;
        out_c.len = 13;
        out_c.us[0] = 123.0;
        out_d.len = 13;
        out_d.us[0] = 123.0;
        out_s.len = 13;
        out_s.us[0] = 123.0;
        out_q.len = 13;
        out_q.us[0] = 123.0;
        assert!(matches!(
            clip_convex_small::<N>(&poly, &hps_unchanged[0], &mut out_a),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_bitmask(&poly, &hps_unchanged[0], &mut out_b),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_small_bool::<N>(&poly, &hps_unchanged[0], &mut out_c),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_match_ngon::<N>(&poly, &hps_unchanged[0], &mut out_d),
            ClipResult::Unchanged
        ));
        assert!(matches!(
            clip_convex_simd::<N>(&poly, &hps_unchanged[0], &mut out_s),
            ClipResult::Unchanged
        ));
        if N == 4 {
            assert!(matches!(
                clip_convex_quad(&poly, &hps_unchanged[0], &mut out_q),
                ClipResult::Unchanged
            ));
        }

        eprintln!("\nclip_convex microbench (N={N})");

        if N == 4 {
            bench_ns_per_call("quad mixed", target, samples, 1, |iters| {
                let poly = black_box(&poly);
                let hps = black_box(&hps);
                let out = black_box(&mut out_q);
                for i in 0..iters {
                    let hp = &hps[(i as usize) & 3];
                    let r = clip_convex_quad(poly, hp, out);
                    black_box(r);
                }
            });
        }

        let (small_mixed, _) = bench_ns_per_call("small mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps);
            let out = black_box(&mut out_a);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_small::<N>(poly, hp, out);
                black_box(r);
            }
        });
        let (mask_mixed, _) = bench_ns_per_call("bitmask mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps);
            let out = black_box(&mut out_b);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_bitmask(poly, hp, out);
                black_box(r);
            }
        });
        let (bool_mixed, _) = bench_ns_per_call("bool mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps);
            let out = black_box(&mut out_c);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("match_ngon mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps);
            let out = black_box(&mut out_d);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_match_ngon::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("table_ngon mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps);
            let out = black_box(&mut out_d);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_table_ngon::<N>(poly, hp, out);
                black_box(r);
            }
        });
        let (simd_mixed, _) = bench_ns_per_call("simd mixed", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps);
            let out = black_box(&mut out_s);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_simd::<N>(poly, hp, out);
                black_box(r);
            }
        });

        if N == 4 {
            bench_ns_per_call("quad unchanged", target, samples, 1, |iters| {
                let poly = black_box(&poly);
                let hps = black_box(&hps_unchanged);
                let out = black_box(&mut out_q);
                for i in 0..iters {
                    let hp = &hps[(i as usize) & 3];
                    let r = clip_convex_quad(poly, hp, out);
                    black_box(r);
                }
            });
        }

        let (small_unch, _) = bench_ns_per_call("small unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps_unchanged);
            let out = black_box(&mut out_a);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_small::<N>(poly, hp, out);
                black_box(r);
            }
        });
        let (mask_unch, _) = bench_ns_per_call("bitmask unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps_unchanged);
            let out = black_box(&mut out_b);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_bitmask(poly, hp, out);
                black_box(r);
            }
        });
        let (bool_unch, _) = bench_ns_per_call("bool unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps_unchanged);
            let out = black_box(&mut out_c);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_small_bool::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("match_ngon unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps_unchanged);
            let out = black_box(&mut out_d);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_match_ngon::<N>(poly, hp, out);
                black_box(r);
            }
        });
        bench_ns_per_call("table_ngon unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps_unchanged);
            let out = black_box(&mut out_d);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_table_ngon::<N>(poly, hp, out);
                black_box(r);
            }
        });
        let (simd_unch, _) = bench_ns_per_call("simd unchanged", target, samples, 1, |iters| {
            let poly = black_box(&poly);
            let hps = black_box(&hps_unchanged);
            let out = black_box(&mut out_s);
            for i in 0..iters {
                let hp = &hps[(i as usize) & 3];
                let r = clip_convex_simd::<N>(poly, hp, out);
                black_box(r);
            }
        });
    }

    let target = Duration::from_millis(target_ms);
    run_for::<3>(target, samples);
    run_for::<4>(target, samples);
    run_for::<5>(target, samples);
    run_for::<6>(target, samples);
    run_for::<7>(target, samples);
    run_for::<8>(target, samples);
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

    #[derive(Clone)]
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        #[inline(always)]
        fn next_u64(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        #[inline(always)]
        fn next_f64(&mut self) -> f64 {
            // Uniform in [0,1)
            let bits = self.next_u64() >> 11; // 53 bits
            (bits as f64) * (1.0 / ((1u64 << 53) as f64))
        }

        #[inline(always)]
        fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next_f64()
        }

        #[inline(always)]
        fn range_usize(&mut self, hi: usize) -> usize {
            debug_assert!(hi > 0);
            (self.next_u64() as usize) % hi
        }
    }

    fn clip_convex_simd_main(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
        let n = poly.len;
        debug_assert!(n >= 3, "clip_convex expects poly.len >= 3, got {}", n);

        let max_r2 = poly.max_r2;
        if !poly.has_bounding_ref && max_r2 > 0.0 {
            let t = hp.c + hp.eps;
            if t >= 0.0 && t * t >= hp.ab2 * max_r2 {
                return ClipResult::Unchanged;
            }
        }

        match n {
            3 => clip_convex_simd_tri(poly, hp, out),
            4 => clip_convex_simd_quad(poly, hp, out),
            5 => clip_convex_simd_pent(poly, hp, out),
            6 => clip_convex_simd_hex(poly, hp, out),
            7 => clip_convex_simd_hept(poly, hp, out),
            8 => clip_convex_simd_oct(poly, hp, out),
            _ => clip_convex_bitmask(poly, hp, out),
        }
    }

    fn fill_sentinel(out: &mut PolyBuffer) {
        out.len = 7;
        out.max_r2 = 123.0;
        out.has_bounding_ref = true;
        for i in 0..out.len {
            out.us[i] = f64::NAN;
            out.vs[i] = f64::NAN;
            out.vertex_planes[i] = (999, 999);
            out.edge_planes[i] = 999;
        }
    }

    fn approx(a: f64, b: f64) -> bool {
        let d = (a - b).abs();
        d <= 1e-10 || d <= 1e-10 * a.abs().max(b.abs())
    }

    fn approx_allow_nan(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || approx(a, b)
    }

    fn poly_eq_cyclic(a: &PolyBuffer, b: &PolyBuffer) -> bool {
        if a.len != b.len {
            return false;
        }
        if a.len == 0 {
            return approx(a.max_r2, b.max_r2) && a.has_bounding_ref == b.has_bounding_ref;
        }
        if !approx(a.max_r2, b.max_r2) || a.has_bounding_ref != b.has_bounding_ref {
            return false;
        }
        let n = a.len;

        'rot: for off in 0..n {
            for i in 0..n {
                let j = (i + off) % n;
                if !approx(a.us[i], b.us[j])
                    || !approx(a.vs[i], b.vs[j])
                    || a.vertex_planes[i] != b.vertex_planes[j]
                    || a.edge_planes[i] != b.edge_planes[j]
                {
                    continue 'rot;
                }
            }
            return true;
        }

        false
    }

    fn poly_eq_cyclic_allow_nan(a: &PolyBuffer, b: &PolyBuffer) -> bool {
        if a.len != b.len {
            return false;
        }
        if a.len == 0 {
            return approx_allow_nan(a.max_r2, b.max_r2) && a.has_bounding_ref == b.has_bounding_ref;
        }
        if !approx_allow_nan(a.max_r2, b.max_r2) || a.has_bounding_ref != b.has_bounding_ref {
            return false;
        }
        let n = a.len;

        'rot: for off in 0..n {
            for i in 0..n {
                let j = (i + off) % n;
                if !approx_allow_nan(a.us[i], b.us[j])
                    || !approx_allow_nan(a.vs[i], b.vs[j])
                    || a.vertex_planes[i] != b.vertex_planes[j]
                    || a.edge_planes[i] != b.edge_planes[j]
                {
                    continue 'rot;
                }
            }
            return true;
        }

        false
    }

    fn make_poly<const N: usize>(us: [f64; N], vs: [f64; N]) -> PolyBuffer {
        let mut p = PolyBuffer::new();
        p.len = N;
        p.max_r2 = 0.0;
        p.has_bounding_ref = false;
        for i in 0..N {
            p.us[i] = us[i];
            p.vs[i] = vs[i];
            p.vertex_planes[i] = (i, (i + 1) % N);
            p.edge_planes[i] = i;
            p.max_r2 = p.max_r2.max(us[i] * us[i] + vs[i] * vs[i]);
        }
        p
    }

    fn make_regular_poly<const N: usize>(radius: f64) -> PolyBuffer {
        let mut p = PolyBuffer::new();
        p.len = N;
        p.max_r2 = 0.0;
        p.has_bounding_ref = false;
        for i in 0..N {
            let theta = std::f64::consts::TAU * (i as f64) / (N as f64);
            let (s, c) = theta.sin_cos();
            let u = radius * c;
            let v = radius * s;
            p.us[i] = u;
            p.vs[i] = v;
            p.vertex_planes[i] = (i, (i + 1) % N);
            p.edge_planes[i] = i;
            p.max_r2 = p.max_r2.max(u * u + v * v);
        }
        p
    }

    fn random_half_plane_targeting_vertex(
        rng: &mut XorShift64,
        poly: &PolyBuffer,
        plane_idx: usize,
    ) -> HalfPlane {
        let theta = rng.range_f64(0.0, core::f64::consts::TAU);
        let scale = rng.range_f64(0.25, 4.0);
        let a = scale * theta.cos();
        let b = scale * theta.sin();

        // Choose a vertex and place the boundary near it, within a few eps.
        let i = rng.range_usize(poly.len.max(1));
        let ui = poly.us[i];
        let vi = poly.vs[i];
        let ab2 = a.mul_add(a, b * b);
        let norm = (ab2 as f32).sqrt() as f64;
        let eps = EPS_INSIDE * norm;
        let delta = rng.range_f64(-4.0 * eps, 4.0 * eps);
        let c = -(a.mul_add(ui, b * vi)) + delta;

        HalfPlane::new_unnormalized(a, b, c, plane_idx)
    }

    fn random_half_plane_generic(rng: &mut XorShift64, plane_idx: usize) -> HalfPlane {
        let theta = rng.range_f64(0.0, core::f64::consts::TAU);
        let scale = rng.range_f64(0.25, 4.0);
        let a = scale * theta.cos();
        let b = scale * theta.sin();
        let c = rng.range_f64(-0.25, 0.25);
        HalfPlane::new_unnormalized(a, b, c, plane_idx)
    }

    #[test]
    fn test_simd_to_bitmask_lane_order() {
        use std::simd::prelude::*;
        use std::simd::f64x8;

        let v = f64x8::from_array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]);
        let m = v.simd_ge(f64x8::splat(0.0));
        assert_eq!(m.to_bitmask(), 0b1111_0000);
    }

    #[test]
    fn test_clip_convex_simd_matches_scalar_random_sequences() {
        let mut rng = XorShift64::new(0xC0FFEE_1234_5678);

        for case in 0..200usize {
            let mut poly_scalar = PolyBuffer::new();
            poly_scalar.init_bounding(1e6);
            let mut poly_simd = poly_scalar.clone();

            let mut out_scalar = PolyBuffer::new();
            let mut out_simd = PolyBuffer::new();

            for step in 0..64usize {
                let hp = random_half_plane_targeting_vertex(&mut rng, &poly_scalar, step);

                fill_sentinel(&mut out_scalar);
                fill_sentinel(&mut out_simd);

                let r_scalar = clip_convex(&poly_scalar, &hp, &mut out_scalar);
                let r_simd = clip_convex_simd_main(&poly_simd, &hp, &mut out_simd);
                assert_eq!(
                    r_scalar, r_simd,
                    "case={case} step={step} n={}",
                    poly_scalar.len
                );

                match r_scalar {
                    ClipResult::Unchanged => {
                        // `out` must not be modified.
                        assert!(out_scalar.us[0].is_nan());
                        assert!(out_simd.us[0].is_nan());
                    }
                    ClipResult::TooManyVertices => unreachable!("not expected in this test"),
                    ClipResult::Changed => {
                        assert!(
                            poly_eq_cyclic(&out_scalar, &out_simd),
                            "case={case} step={step} polys diverged (n={})",
                            poly_scalar.len
                        );

                        poly_scalar = out_scalar.clone();
                        poly_simd = out_simd.clone();

                        if poly_scalar.len < 3 {
                            break;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_clip_convex_table_ngon_matches_scalar_random_sequences() {
        let mut rng = XorShift64::new(0xBADA55_1234_5678);

        for case in 0..200usize {
            let n = 3 + (case % 6);
            let poly = match n {
                3 => make_regular_poly::<3>(1.0),
                4 => make_regular_poly::<4>(1.0),
                5 => make_regular_poly::<5>(1.0),
                6 => make_regular_poly::<6>(1.0),
                7 => make_regular_poly::<7>(1.0),
                8 => make_regular_poly::<8>(1.0),
                _ => unreachable!(),
            };

            let mut poly_match = poly.clone();
            let mut poly_table = poly.clone();
            let mut out_match = PolyBuffer::new();
            let mut out_table = PolyBuffer::new();

            for step in 0..64usize {
                let hp = random_half_plane_generic(&mut rng, step);
                fill_sentinel(&mut out_match);
                fill_sentinel(&mut out_table);

                let r_match = match n {
                    3 => clip_convex_match_ngon::<3>(&poly_match, &hp, &mut out_match),
                    4 => clip_convex_match_ngon::<4>(&poly_match, &hp, &mut out_match),
                    5 => clip_convex_match_ngon::<5>(&poly_match, &hp, &mut out_match),
                    6 => clip_convex_match_ngon::<6>(&poly_match, &hp, &mut out_match),
                    7 => clip_convex_match_ngon::<7>(&poly_match, &hp, &mut out_match),
                    8 => clip_convex_match_ngon::<8>(&poly_match, &hp, &mut out_match),
                    _ => unreachable!(),
                };
                let r_table = match n {
                    3 => clip_convex_table_ngon::<3>(&poly_table, &hp, &mut out_table),
                    4 => clip_convex_table_ngon::<4>(&poly_table, &hp, &mut out_table),
                    5 => clip_convex_table_ngon::<5>(&poly_table, &hp, &mut out_table),
                    6 => clip_convex_table_ngon::<6>(&poly_table, &hp, &mut out_table),
                    7 => clip_convex_table_ngon::<7>(&poly_table, &hp, &mut out_table),
                    8 => clip_convex_table_ngon::<8>(&poly_table, &hp, &mut out_table),
                    _ => unreachable!(),
                };

                assert_eq!(matches!(r_table, ClipResult::TooManyVertices), false);
                assert_eq!(
                    matches!(r_match, ClipResult::Unchanged),
                    matches!(r_table, ClipResult::Unchanged)
                );

                if matches!(r_match, ClipResult::Changed) {
                    assert!(
                        poly_eq_cyclic_allow_nan(&out_match, &out_table),
                        "table_ngon mismatch: n={n} case={case} step={step}"
                    );
                    poly_match = out_match.clone();
                    poly_table = out_table.clone();
                }
                if poly_match.len < 3 {
                    break;
                }
            }
        }
    }

    #[test]
    fn test_clip_convex_small_matches_bitmask_square() {
        let poly = make_poly::<4>([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]);
        let hp = HalfPlane::new_unnormalized(1.0, 0.0, -0.5, 7);

        // Mixed case
        let mut out_small = PolyBuffer::new();
        let mut out_mask = PolyBuffer::new();
        assert_eq!(
            clip_convex_small::<4>(&poly, &hp, &mut out_small),
            ClipResult::Changed
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp, &mut out_mask),
            ClipResult::Changed
        );
        assert!(poly_eq_cyclic(&out_small, &out_mask));

        // All inside should not touch `out`
        let hp_inside = HalfPlane::new_unnormalized(1.0, 0.0, 10.0, 7);
        fill_sentinel(&mut out_small);
        fill_sentinel(&mut out_mask);
        assert_eq!(
            clip_convex_small::<4>(&poly, &hp_inside, &mut out_small),
            ClipResult::Unchanged
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp_inside, &mut out_mask),
            ClipResult::Unchanged
        );
        assert_eq!(out_small.len, 7);
        assert_eq!(out_mask.len, 7);

        // All outside
        let hp_outside = HalfPlane::new_unnormalized(1.0, 0.0, -10.0, 7);
        assert_eq!(
            clip_convex_small::<4>(&poly, &hp_outside, &mut out_small),
            ClipResult::Changed
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp_outside, &mut out_mask),
            ClipResult::Changed
        );
        assert_eq!(out_small.len, 0);
        assert_eq!(out_mask.len, 0);
    }

    #[test]
    fn test_clip_convex_small_matches_bitmask_bounding_triangle() {
        let mut poly = PolyBuffer::new();
        poly.init_bounding(10.0);
        let hp = HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 12); // u >= 0

        let mut out_small = PolyBuffer::new();
        let mut out_mask = PolyBuffer::new();
        assert_eq!(
            clip_convex_small::<3>(&poly, &hp, &mut out_small),
            ClipResult::Changed
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp, &mut out_mask),
            ClipResult::Changed
        );
        assert!(poly_eq_cyclic(&out_small, &out_mask));
    }

    #[test]
    fn test_clip_convex_bool_matches_bitmask_square() {
        let poly = make_poly::<4>([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]);
        let hp = HalfPlane::new_unnormalized(1.0, 0.0, -0.5, 7);

        // Mixed case
        let mut out_bool = PolyBuffer::new();
        let mut out_mask = PolyBuffer::new();
        assert_eq!(
            clip_convex_small_bool::<4>(&poly, &hp, &mut out_bool),
            ClipResult::Changed
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp, &mut out_mask),
            ClipResult::Changed
        );
        assert!(poly_eq_cyclic(&out_bool, &out_mask));

        // All inside should not touch `out`
        let hp_inside = HalfPlane::new_unnormalized(1.0, 0.0, 10.0, 7);
        fill_sentinel(&mut out_bool);
        fill_sentinel(&mut out_mask);
        assert_eq!(
            clip_convex_small_bool::<4>(&poly, &hp_inside, &mut out_bool),
            ClipResult::Unchanged
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp_inside, &mut out_mask),
            ClipResult::Unchanged
        );
        assert_eq!(out_bool.len, 7);
        assert_eq!(out_mask.len, 7);

        // All outside
        let hp_outside = HalfPlane::new_unnormalized(1.0, 0.0, -10.0, 7);
        assert_eq!(
            clip_convex_small_bool::<4>(&poly, &hp_outside, &mut out_bool),
            ClipResult::Changed
        );
        assert_eq!(
            clip_convex_bitmask(&poly, &hp_outside, &mut out_mask),
            ClipResult::Changed
        );
        assert_eq!(out_bool.len, 0);
        assert_eq!(out_mask.len, 0);
    }

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
