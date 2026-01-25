//! Floating-point helpers (feature-gated).

use std::simd::f32x8;
#[cfg(feature = "fma")]
use std::simd::StdFloat;

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

#[inline(always)]
pub(crate) fn dot3_f32x8(
    ax: f32x8,
    ay: f32x8,
    az: f32x8,
    bx: f32x8,
    by: f32x8,
    bz: f32x8,
) -> f32x8 {
    #[cfg(feature = "fma")]
    {
        let ab = ax.mul_add(bx, ay * by);
        az.mul_add(bz, ab)
    }
    #[cfg(not(feature = "fma"))]
    {
        (ax * bx + ay * by) + az * bz
    }
}
