use super::types::{ClipResult, HalfPlane, PolyBuffer};

mod bitmask;
mod output;
mod small;

use bitmask::clip_bitmask;
use small::{clip_small_ptr, clip_small_ptr_d};

/// Clip a convex polygon by a half-plane.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
    let n = poly.len;

    debug_assert!(n >= 3, "clip_convex expects poly.len >= 3, got {}", n);

    let has_bounding_ref = poly.has_bounding_ref;
    // Early-unchanged is only relevant for bounded polys. Empirically, it is very rare for small N
    // (e.g. triangles), so we skip the check below a small-N threshold to save overhead.
    const EARLY_UNCHANGED_MIN_N: usize = 5;
    if !has_bounding_ref && n >= EARLY_UNCHANGED_MIN_N {
        let max_r2 = poly.max_r2;
        if max_r2 > 0.0 {
            let t = hp.c + hp.eps;
            if t >= 0.0 && t * t >= hp.ab2 * max_r2 {
                return ClipResult::Unchanged;
            }
        }
    }

    // Dispatch strategy (x86_64):
    // - N=4,8: `% N` is cheap (bitmask), so the plain ptr variant tends to win.
    // - N=3,5,6,7: avoid `% N` in the hot loop via the ptr_d variant.
    // We `match` on N first, then branch on bounded/unbounded per arm. This can improve branch
    // prediction when `has_bounding_ref` correlates with N, while still allowing the inner push
    // path to be specialized via `const TRACK_BOUNDING`.
    match n {
        3 => {
            if has_bounding_ref {
                clip_small_ptr_d::<3, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<3, false>(poly, hp, out)
            }
        }
        4 => {
            if has_bounding_ref {
                clip_small_ptr::<4, true>(poly, hp, out)
            } else {
                clip_small_ptr::<4, false>(poly, hp, out)
            }
        }
        5 => {
            if has_bounding_ref {
                clip_small_ptr_d::<5, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<5, false>(poly, hp, out)
            }
        }
        6 => {
            if has_bounding_ref {
                clip_small_ptr_d::<6, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<6, false>(poly, hp, out)
            }
        }
        7 => {
            if has_bounding_ref {
                clip_small_ptr_d::<7, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<7, false>(poly, hp, out)
            }
        }
        8 => {
            if has_bounding_ref {
                clip_small_ptr::<8, true>(poly, hp, out)
            } else {
                clip_small_ptr::<8, false>(poly, hp, out)
            }
        }
        _ => clip_bitmask(poly, hp, out),
    }
}

/// Clip a convex polygon by a half-plane, skipping the early-unchanged bounding check.
///
/// This is intended for edgecheck-derived seed constraints where we expect the half-plane to be
/// active and want to avoid extra branchy prechecks.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex_edgecheck(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    let n = poly.len;
    debug_assert!(n >= 3, "clip_convex expects poly.len >= 3, got {}", n);

    let has_bounding_ref = poly.has_bounding_ref;
    match n {
        3 => {
            if has_bounding_ref {
                clip_small_ptr_d::<3, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<3, false>(poly, hp, out)
            }
        }
        4 => {
            if has_bounding_ref {
                clip_small_ptr::<4, true>(poly, hp, out)
            } else {
                clip_small_ptr::<4, false>(poly, hp, out)
            }
        }
        5 => {
            if has_bounding_ref {
                clip_small_ptr_d::<5, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<5, false>(poly, hp, out)
            }
        }
        6 => {
            if has_bounding_ref {
                clip_small_ptr_d::<6, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<6, false>(poly, hp, out)
            }
        }
        7 => {
            if has_bounding_ref {
                clip_small_ptr_d::<7, true>(poly, hp, out)
            } else {
                clip_small_ptr_d::<7, false>(poly, hp, out)
            }
        }
        8 => {
            if has_bounding_ref {
                clip_small_ptr::<8, true>(poly, hp, out)
            } else {
                clip_small_ptr::<8, false>(poly, hp, out)
            }
        }
        _ => clip_bitmask(poly, hp, out),
    }
}


#[cfg(any(test, feature = "microbench"))]
#[allow(unused_imports)]
pub(crate) use small::clip_convex_small_bool;
