use crate::fp;

use crate::tolerances::CLIP_EPS_INSIDE;
pub const MAX_POLY_VERTICES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipResult {
    /// Polygon unchanged (all vertices inside). Note: `out` is NOT written.
    Unchanged,
    Changed,
    TooManyVertices,
}

/// A 2D half-plane constraint: a*u + b*v + c >= 0.
#[derive(Debug, Clone, Copy)]
pub struct HalfPlane {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub ab2: f64,
    pub plane_idx: usize,
    pub eps: f64,
}

impl HalfPlane {
    /// Spherical-backend constructor: applies the `CLIP_EPS_INSIDE` base
    /// slack (strict rule; probe-overridable under `p5_shadow`).
    pub fn new_unnormalized(a: f64, b: f64, c: f64, plane_idx: usize) -> Self {
        #[cfg(feature = "p5_shadow")]
        let base = crate::knn_clipping::p5_shadow::clip_eps_override().unwrap_or(CLIP_EPS_INSIDE);
        #[cfg(not(feature = "p5_shadow"))]
        let base = CLIP_EPS_INSIDE;
        Self::new_unnormalized_base_eps(a, b, c, plane_idx, base)
    }

    /// Construct an unnormalized half-plane with an explicit base
    /// inside-slack (`eps = base_eps * |n|`). The planar backends pass
    /// `PLANE_CLIP_EPS_INSIDE` (they keep the agreement bias; see
    /// tolerances.rs).
    pub fn new_unnormalized_base_eps(
        a: f64,
        b: f64,
        c: f64,
        plane_idx: usize,
        base_eps: f64,
    ) -> Self {
        let ab2: f64 = fp::fma_f64(a, a, b * b);
        let norm = (ab2 as f32).sqrt() as f64;
        HalfPlane {
            a,
            b,
            c,
            ab2,
            plane_idx,
            eps: base_eps * norm,
        }
    }

    /// Construct an unnormalized half-plane while supplying an external epsilon.
    ///
    /// This is intended for edgecheck-derived seed constraints where we want to reuse an
    /// epsilon computed on the opposite side of the same undirected edge, avoiding an extra sqrt.
    pub fn new_unnormalized_with_eps(a: f64, b: f64, c: f64, plane_idx: usize, eps: f64) -> Self {
        let ab2: f64 = fp::fma_f64(a, a, b * b);
        HalfPlane {
            a,
            b,
            c,
            ab2,
            plane_idx,
            eps,
        }
    }

    // The production clippers now evaluate distances 8 wide via
    // fp::signed_dists_mask8 (lane math identical to this formula); the
    // scalar form remains the reference for tests and microbenches.
    #[cfg_attr(not(any(test, feature = "microbench")), allow(dead_code))]
    #[inline]
    pub fn signed_dist(&self, u: f64, v: f64) -> f64 {
        fp::fma_f64(self.a, u, fp::fma_f64(self.b, v, self.c))
    }
}

/// Fixed-size polygon buffer for clipping.
#[derive(Clone)]
pub struct PolyBuffer {
    pub len: usize,
    pub max_r2: f64,
    pub has_bounding_ref: bool,
    pub us: [f64; MAX_POLY_VERTICES],
    pub vs: [f64; MAX_POLY_VERTICES],
    pub vertex_planes: [(usize, usize); MAX_POLY_VERTICES],
    pub edge_planes: [usize; MAX_POLY_VERTICES],
}

impl PolyBuffer {
    #[inline]
    pub fn new() -> Self {
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

    pub fn init_bounding(&mut self, bound: f64) {
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
    pub fn clear(&mut self) {
        self.len = 0;
        self.max_r2 = 0.0;
        self.has_bounding_ref = false;
    }

    #[inline]
    pub fn push_raw(&mut self, u: f64, v: f64, vp: (usize, usize), ep: usize) {
        let i = self.len;
        debug_assert!(i < MAX_POLY_VERTICES);
        unsafe {
            *self.us.get_unchecked_mut(i) = u;
            *self.vs.get_unchecked_mut(i) = v;
            *self.vertex_planes.get_unchecked_mut(i) = vp;
            *self.edge_planes.get_unchecked_mut(i) = ep;
        }
        self.len = i + 1;
    }

    #[inline]
    pub fn min_cos(&self) -> f64 {
        if self.len == 0 {
            return 1.0;
        }
        1.0 / (1.0 + self.max_r2).sqrt()
    }

    #[inline]
    pub fn has_bounding_ref(&self) -> bool {
        self.has_bounding_ref
    }
}
