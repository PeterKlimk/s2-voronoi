use crate::fp;

pub const MAX_POLY_VERTICES: usize = 24;
pub type PlaneId = u32;
pub const INVALID_PLANE_ID: PlaneId = PlaneId::MAX;

#[inline]
pub(crate) fn plane_id(idx: usize) -> PlaneId {
    debug_assert!(PlaneId::try_from(idx).is_ok(), "plane index exceeds u32");
    idx as PlaneId
}

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
    pub plane_idx: PlaneId,
}

impl HalfPlane {
    /// Construct the strict half-plane `a*u + b*v + c >= 0`.
    pub fn new_unnormalized(a: f64, b: f64, c: f64, plane_idx: usize) -> Self {
        let ab2: f64 = fp::fma_f64(a, a, b * b);
        HalfPlane {
            a,
            b,
            c,
            ab2,
            plane_idx: plane_id(plane_idx),
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
    pub vertex_planes: [(PlaneId, PlaneId); MAX_POLY_VERTICES],
    pub edge_planes: [PlaneId; MAX_POLY_VERTICES],
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
        self.vertex_planes[0] = (INVALID_PLANE_ID, INVALID_PLANE_ID);
        self.vertex_planes[1] = (INVALID_PLANE_ID, INVALID_PLANE_ID);
        self.vertex_planes[2] = (INVALID_PLANE_ID, INVALID_PLANE_ID);
        self.edge_planes[0] = INVALID_PLANE_ID;
        self.edge_planes[1] = INVALID_PLANE_ID;
        self.edge_planes[2] = INVALID_PLANE_ID;
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
    pub fn push_raw(&mut self, u: f64, v: f64, vp: (PlaneId, PlaneId), ep: PlaneId) {
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
    pub fn has_bounding_ref(&self) -> bool {
        self.has_bounding_ref
    }
}
