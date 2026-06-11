//! Floating-point helpers and the 8-lane SIMD backend seam.
//!
//! All explicit SIMD in the crate goes through [`PointChunk8`] / [`Dots8`],
//! so the backend is swappable for benchmarking and portability:
//!
//! - default: `std::simd` (portable_simd, requires nightly)
//! - `simd_wide`: the `wide` crate (stable Rust, same instructions)
//! - `simd_scalar`: plain arrays relying on autovectorization (stable)
//!
//! All backends are lane-wise only (no horizontal reductions), so results
//! are bit-identical across them.

#[cfg(all(feature = "simd_wide", feature = "simd_scalar"))]
compile_error!("features `simd_wide` and `simd_scalar` are mutually exclusive");

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

#[cfg(not(any(feature = "simd_wide", feature = "simd_scalar")))]
mod backend {
    use std::simd::cmp::SimdPartialOrd;
    use std::simd::f32x8;
    #[cfg(feature = "fma")]
    use std::simd::StdFloat;

    pub(super) type V = f32x8;

    #[inline(always)]
    pub(super) fn load_slice(s: &[f32]) -> V {
        f32x8::from_slice(s)
    }

    #[inline(always)]
    pub(super) fn load_array(a: [f32; 8]) -> V {
        f32x8::from_array(a)
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
        v.simd_gt(f32x8::splat(threshold)).to_bitmask() as u32
    }

    #[inline(always)]
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v.to_array()
    }
}

#[cfg(feature = "simd_wide")]
mod backend {
    use wide::{f32x8, CmpGt};

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
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v.to_array()
    }
}

#[cfg(all(feature = "simd_scalar", not(feature = "simd_wide")))]
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
    pub(super) fn to_array(v: V) -> [f32; 8] {
        v
    }
}
