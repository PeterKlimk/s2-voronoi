//! Floating-point helpers and the 8-lane SIMD backend seam.
//!
//! All explicit SIMD in the crate goes through [`PointChunk8`] / [`Dots8`],
//! so the backend is swappable for benchmarking and portability:
//!
//! - default: the `wide` crate (explicit SIMD on stable Rust)
//! - `simd_scalar`: plain arrays relying on autovectorization (debugging /
//!   comparison; a few percent slower on generic targets)
//!
//! All backends are lane-wise only (no horizontal reductions), so results
//! are bit-identical across them (see tests/backend_fingerprint.rs).
//!
//! History: the original backend was nightly `portable_simd`. Benchmarks on
//! the reference Ryzen 3600 (2026-06) showed `wide` at parity (within ~1-2%,
//! winning some runs), so the nightly requirement was dropped. The portable
//! backend can be recovered from git history if `std::simd` stabilizes.

/// An `f32` ordered via `total_cmp` (NaN ordered consistently, not rejected).
///
/// Shared by the spatial-grid frontiers for sorted (distance, slot) emission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl OrdF32 {
    #[inline]
    pub(crate) fn new(v: f32) -> Self {
        OrdF32(v)
    }

    #[inline]
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

#[inline(always)]
pub(crate) fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "fma"))]
    {
        a * b + c
    }
}

#[inline(always)]
pub(crate) fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "fma"))]
    {
        a * b + c
    }
}

#[inline(always)]
pub(crate) fn dot3_f32(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    // Match the baseline left-associative order:
    //   (ax*bx + ay*by) + az*bz
    let ab = fma_f32(ax, bx, ay * by);
    fma_f32(az, bz, ab)
}

/// Eight candidate points in SoA form, loaded once and dotted against many
/// queries (the load hoisting is load-bearing in the packed kNN hot loops).
pub(crate) struct PointChunk8 {
    x: backend::V,
    y: backend::V,
    z: backend::V,
}

/// Eight dot products; mask extraction is cheap and `to_array` is deferred
/// until a mask is known to be non-empty.
pub(crate) struct Dots8(backend::V);

impl PointChunk8 {
    /// Load the first 8 elements of each slice (caller guarantees length).
    #[inline(always)]
    pub(crate) fn from_slices(xs: &[f32], ys: &[f32], zs: &[f32]) -> Self {
        Self {
            x: backend::load_slice(xs),
            y: backend::load_slice(ys),
            z: backend::load_slice(zs),
        }
    }

    #[inline(always)]
    pub(crate) fn from_arrays(x: [f32; 8], y: [f32; 8], z: [f32; 8]) -> Self {
        Self {
            x: backend::load_array(x),
            y: backend::load_array(y),
            z: backend::load_array(z),
        }
    }

    /// Lane-wise dot products against a broadcast query point.
    #[inline(always)]
    pub(crate) fn dots(&self, qx: f32, qy: f32, qz: f32) -> Dots8 {
        Dots8(backend::dot3(self.x, self.y, self.z, qx, qy, qz))
    }
}

impl Dots8 {
    /// Bitmask of lanes with dot > threshold (lane i -> bit i).
    #[inline(always)]
    pub(crate) fn mask_gt(&self, threshold: f32) -> u32 {
        backend::mask_gt(self.0, threshold)
    }

    #[inline(always)]
    pub(crate) fn to_array(&self) -> [f32; 8] {
        backend::to_array(self.0)
    }
}

/// Eight planar points (x, y lanes) for the planar packed-kNN hot loops;
/// the 2D sibling of [`PointChunk8`].
pub(crate) struct PlaneChunk8 {
    x: backend::V,
    y: backend::V,
}

/// Eight squared distances; planar semantics are "smaller is closer", so
/// the mask is `mask_lt`.
pub(crate) struct Dists8(backend::V);

impl PlaneChunk8 {
    /// Load the first 8 elements of each slice (caller guarantees length).
    #[inline(always)]
    pub(crate) fn from_slices(xs: &[f32], ys: &[f32]) -> Self {
        Self {
            x: backend::load_slice(xs),
            y: backend::load_slice(ys),
        }
    }

    #[inline(always)]
    pub(crate) fn from_arrays(x: [f32; 8], y: [f32; 8]) -> Self {
        Self {
            x: backend::load_array(x),
            y: backend::load_array(y),
        }
    }

    /// Lane-wise squared Euclidean distances to a broadcast query point.
    #[inline(always)]
    pub(crate) fn dist_sqs(&self, qx: f32, qy: f32) -> Dists8 {
        Dists8(backend::dist_sq2(self.x, self.y, qx, qy))
    }

    /// Lane-wise minimum-image squared distances on a torus.
    #[inline(always)]
    pub(crate) fn dist_sqs_wrapped(&self, qx: f32, qy: f32, px: f32, py: f32) -> Dists8 {
        Dists8(backend::dist_sq2_wrapped(self.x, self.y, qx, qy, px, py))
    }
}

impl Dists8 {
    /// Bitmask of lanes with dist_sq < threshold (lane i -> bit i).
    #[inline(always)]
    pub(crate) fn mask_lt(&self, threshold: f32) -> u32 {
        backend::mask_lt(self.0, threshold)
    }

    #[inline(always)]
    pub(crate) fn to_array(&self) -> [f32; 8] {
        backend::to_array(self.0)
    }
}

/// Signed distances of the first 8 polygon vertices to a half-plane
/// (`fma_f64(a, u, fma_f64(b, v, c))`, lane-exact with the scalar formula)
/// plus the inside bitmask (`d >= neg_eps`, lane i -> bit i).
///
/// Callers may pass slices longer than the live vertex count; lanes past it
/// compute on stale-but-finite data and must be masked off by the caller.
#[inline(always)]
pub(crate) fn signed_dists_mask8(
    a: f64,
    b: f64,
    c: f64,
    us: &[f64],
    vs: &[f64],
    neg_eps: f64,
) -> ([f64; 8], u32) {
    backend::signed_dists_mask8(a, b, c, us, vs, neg_eps)
}

#[cfg(not(feature = "simd_scalar"))]
mod backend {
    use wide::{f32x8, f64x4, CmpGe, CmpGt, CmpLt};

    pub(super) type V = f32x8;

    #[inline(always)]
    pub(super) fn load_slice(s: &[f32]) -> V {
        let arr: [f32; 8] = s[..8].try_into().unwrap();
        f32x8::from(arr)
    }

    #[inline(always)]
    pub(super) fn load_array(a: [f32; 8]) -> V {
        f32x8::from(a)
    }

    #[inline(always)]
    pub(super) fn dot3(x: V, y: V, z: V, qx: f32, qy: f32, qz: f32) -> V {
        let qx = f32x8::splat(qx);
        let qy = f32x8::splat(qy);
        let qz = f32x8::splat(qz);
        #[cfg(feature = "fma")]
        {
            let ab = x.mul_add(qx, y * qy);
            z.mul_add(qz, ab)
        }
        #[cfg(not(feature = "fma"))]
        {
            (x * qx + y * qy) + z * qz
        }
    }

    #[inline(always)]
    pub(super) fn mask_gt(v: V, threshold: f32) -> u32 {
        v.cmp_gt(f32x8::splat(threshold)).move_mask() as u32 & 0xff
    }

    #[inline(always)]
    pub(super) fn dist_sq2(x: V, y: V, qx: f32, qy: f32) -> V {
        let dx = x - f32x8::splat(qx);
        let dy = y - f32x8::splat(qy);
        #[cfg(feature = "fma")]
        {
            dx.mul_add(dx, dy * dy)
        }
        #[cfg(not(feature = "fma"))]
        {
            dx * dx + dy * dy
        }
    }

    /// Minimum-image squared distances on a torus with periods `(px, py)`.
    /// Lane math mirrors `plane_grid::periodic::wrap_abs`: `|d|.min(p - |d|)`
    /// per axis (coordinates and query are both inside `[0, p)`).
    #[inline(always)]
    pub(super) fn dist_sq2_wrapped(x: V, y: V, qx: f32, qy: f32, px: f32, py: f32) -> V {
        let dx = (x - f32x8::splat(qx)).abs();
        let dx = dx.min(f32x8::splat(px) - dx);
        let dy = (y - f32x8::splat(qy)).abs();
        let dy = dy.min(f32x8::splat(py) - dy);
        #[cfg(feature = "fma")]
        {
            dx.mul_add(dx, dy * dy)
        }
        #[cfg(not(feature = "fma"))]
        {
            dx * dx + dy * dy
        }
    }

    #[inline(always)]
    pub(super) fn mask_lt(v: V, threshold: f32) -> u32 {
        v.cmp_lt(f32x8::splat(threshold)).move_mask() as u32 & 0xff
    }

    #[inline(always)]
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v.to_array()
    }

    #[inline(always)]
    fn dists4(a: f64, b: f64, c: f64, u: f64x4, v: f64x4) -> f64x4 {
        #[cfg(feature = "fma")]
        {
            u.mul_add(f64x4::splat(a), v.mul_add(f64x4::splat(b), f64x4::splat(c)))
        }
        #[cfg(not(feature = "fma"))]
        {
            f64x4::splat(a) * u + (f64x4::splat(b) * v + f64x4::splat(c))
        }
    }

    #[inline(always)]
    pub(super) fn signed_dists_mask8(
        a: f64,
        b: f64,
        c: f64,
        us: &[f64],
        vs: &[f64],
        neg_eps: f64,
    ) -> ([f64; 8], u32) {
        let u_lo = f64x4::from(<[f64; 4]>::try_from(&us[..4]).unwrap());
        let u_hi = f64x4::from(<[f64; 4]>::try_from(&us[4..8]).unwrap());
        let v_lo = f64x4::from(<[f64; 4]>::try_from(&vs[..4]).unwrap());
        let v_hi = f64x4::from(<[f64; 4]>::try_from(&vs[4..8]).unwrap());
        let d_lo = dists4(a, b, c, u_lo, v_lo);
        let d_hi = dists4(a, b, c, u_hi, v_hi);
        let eps4 = f64x4::splat(neg_eps);
        let mask = (d_lo.cmp_ge(eps4).move_mask() as u32 & 0xf)
            | ((d_hi.cmp_ge(eps4).move_mask() as u32 & 0xf) << 4);
        let lo = d_lo.to_array();
        let hi = d_hi.to_array();
        (
            [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]],
            mask,
        )
    }
}

#[cfg(feature = "simd_scalar")]
mod backend {
    pub(super) type V = [f32; 8];

    #[inline(always)]
    pub(super) fn load_slice(s: &[f32]) -> V {
        s[..8].try_into().unwrap()
    }

    #[inline(always)]
    pub(super) fn load_array(a: [f32; 8]) -> V {
        a
    }

    #[inline(always)]
    pub(super) fn dot3(x: V, y: V, z: V, qx: f32, qy: f32, qz: f32) -> V {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = super::fma_f32(z[i], qz, super::fma_f32(x[i], qx, y[i] * qy));
        }
        out
    }

    #[inline(always)]
    pub(super) fn mask_gt(v: V, threshold: f32) -> u32 {
        let mut bits = 0u32;
        for (i, &d) in v.iter().enumerate() {
            bits |= u32::from(d > threshold) << i;
        }
        bits
    }

    #[inline(always)]
    pub(super) fn dist_sq2(x: V, y: V, qx: f32, qy: f32) -> V {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            let dx = x[i] - qx;
            let dy = y[i] - qy;
            out[i] = super::fma_f32(dx, dx, dy * dy);
        }
        out
    }

    /// Scalar twin of the wide backend's minimum-image distances (identical
    /// lane math).
    #[inline(always)]
    pub(super) fn dist_sq2_wrapped(x: V, y: V, qx: f32, qy: f32, px: f32, py: f32) -> V {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            let dx = (x[i] - qx).abs();
            let dx = dx.min(px - dx);
            let dy = (y[i] - qy).abs();
            let dy = dy.min(py - dy);
            out[i] = super::fma_f32(dx, dx, dy * dy);
        }
        out
    }

    #[inline(always)]
    pub(super) fn mask_lt(v: V, threshold: f32) -> u32 {
        let mut bits = 0u32;
        for (i, &d) in v.iter().enumerate() {
            bits |= u32::from(d < threshold) << i;
        }
        bits
    }

    #[inline(always)]
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v
    }

    #[inline(always)]
    pub(super) fn signed_dists_mask8(
        a: f64,
        b: f64,
        c: f64,
        us: &[f64],
        vs: &[f64],
        neg_eps: f64,
    ) -> ([f64; 8], u32) {
        let mut dists = [0.0f64; 8];
        let mut mask = 0u32;
        for i in 0..8 {
            let d = super::fma_f64(a, us[i], super::fma_f64(b, vs[i], c));
            dists[i] = d;
            mask |= u32::from(d >= neg_eps) << i;
        }
        (dists, mask)
    }
}
