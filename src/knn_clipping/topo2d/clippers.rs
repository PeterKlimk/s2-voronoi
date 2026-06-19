use super::types::{ClipResult, HalfPlane, PolyBuffer, INVALID_PLANE_ID};

mod bitmask;
mod output;
mod small;

use bitmask::clip_bitmask;
use small::{clip_small_ptr, clip_small_ptr_d};

/// Context for canonical escalation of near-margin clip decisions
/// (P5 stage 2, docs/p5-consistency-design.md): the raw f32 positions every
/// cell shares, so the exact in-circle predicate answers the same question
/// with the same bits everywhere. `disabled()` (empty positions) keeps all
/// decisions local — used by the planar builders (their canonical predicate
/// is the 2D incircle; stage 2c) and synthetic test polygons.
#[derive(Clone, Copy)]
pub(crate) struct EscalationCtx<'a> {
    pub generator_raw: glam::Vec3,
    pub neighbor_raw: glam::Vec3,
    pub neighbor_positions: &'a [glam::Vec3],
}

impl EscalationCtx<'static> {
    pub(crate) fn disabled() -> Self {
        Self {
            generator_raw: glam::Vec3::ZERO,
            neighbor_raw: glam::Vec3::ZERO,
            neighbor_positions: &[],
        }
    }
}

/// Replace near-margin local classifications with the exact canonical
/// in-circle answer. Cold: fires for ~1 in 30k decisions at the 1e-8
/// filter. Lanes with synthetic plane attribution (bounding triangle,
/// out-of-range test fixtures) and exact ties (canonical == 0; SoS is
/// stage 2b) keep their local classification.
#[cold]
#[inline(never)]
fn escalate_mask(
    mut mask: u64,
    dists: &[f64],
    n: usize,
    filter_eps: f64,
    poly: &PolyBuffer,
    esc: &EscalationCtx<'_>,
) -> u64 {
    for (i, &d) in dists.iter().enumerate().take(n) {
        // NaN-safe: only finite sub-band margins escalate.
        if d.abs() >= filter_eps || d.is_nan() {
            continue;
        }
        let (pa, pb) = poly.vertex_planes[i];
        if pa == INVALID_PLANE_ID || pb == INVALID_PLANE_ID {
            continue;
        }
        let (Some(&a), Some(&b)) = (
            esc.neighbor_positions.get(pa as usize),
            esc.neighbor_positions.get(pb as usize),
        ) else {
            continue;
        };
        let sign = crate::knn_clipping::canonical::in_circle_sphere_sign(
            esc.generator_raw,
            a,
            b,
            esc.neighbor_raw,
        );
        if sign == 0 {
            continue;
        }
        // sign < 0: the opposing generator is outside the vertex's
        // circumcircle — the vertex is kept.
        if sign < 0 {
            mask |= 1u64 << i;
        } else {
            mask &= !(1u64 << i);
        }
    }
    mask
}

/// Near-margin pre-screen (hot: one abs+compare per live lane) plus the
/// cold canonical fixup. Returns the possibly-corrected mask.
#[inline(always)]
fn maybe_escalate(
    mask: u64,
    dists: &[f64],
    n: usize,
    hp: &HalfPlane,
    poly: &PolyBuffer,
    esc: &EscalationCtx<'_>,
) -> u64 {
    #[cfg(feature = "p5_shadow")]
    let factor = crate::knn_clipping::p5_shadow::escalation_factor_override()
        .unwrap_or(crate::tolerances::CLIP_ESCALATION_FACTOR);
    #[cfg(not(feature = "p5_shadow"))]
    let factor = crate::tolerances::CLIP_ESCALATION_FACTOR;
    // Production: `factor` is the compile-time const `CLIP_ESCALATION_FACTOR`,
    // which is 0.0, so `filter_eps` is 0.0 and the near-margin band is empty —
    // the scan below can never set `near`, and `escalate_mask` never runs.
    // The compiler can't prove `hp.eps * 0.0 == 0.0` (IEEE: inf/nan would give
    // nan), so without this const-foldable guard it still emits the per-lane
    // abs+compare scan on every clip. Behavior is unchanged: the guarded path
    // returns exactly the same `mask` the scan would. The `p5_shadow` build
    // keeps the runtime path (its `factor` can be overridden nonzero).
    #[cfg(not(feature = "p5_shadow"))]
    if factor == 0.0 {
        return mask;
    }
    let filter_eps = hp.eps * factor;
    let mut near = false;
    for &d in &dists[..n] {
        near |= d.abs() < filter_eps;
    }
    if near {
        escalate_mask(mask, dists, n, filter_eps, poly, esc)
    } else {
        mask
    }
}

/// Intersection parameter with flipped-decision guards: after canonical
/// escalation a transition's distances may not bracket zero (the vertex
/// sits on the line to working precision), so clamp into the segment and
/// fall back to the segment start on a degenerate 0/0. Identity for the
/// ordinary bracketed case.
#[inline(always)]
fn lerp_t(d0: f64, d1: f64) -> f64 {
    let t = d0 / (d0 - d1);
    if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Dispatch clipping to the best specialization for the current polygon size.
///
/// This is the shared match table used by both `clip_convex` and `clip_convex_edgecheck`.
///
/// - N=4,8: `% N` is cheap (bitmask), so the plain ptr variant tends to win.
/// - N=3,5,6,7: avoid `% N` in the hot loop via the ptr_d variant.
///
/// We `match` on N first, then branch on bounded/unbounded per arm.
#[inline(always)]
fn dispatch_clip(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
    esc: &EscalationCtx<'_>,
) -> ClipResult {
    let n = poly.len;
    let has_bounding_ref = poly.has_bounding_ref;
    match n {
        3 => {
            if has_bounding_ref {
                clip_small_ptr_d::<3, true>(poly, hp, out, esc)
            } else {
                clip_small_ptr_d::<3, false>(poly, hp, out, esc)
            }
        }
        4 => {
            if has_bounding_ref {
                clip_small_ptr::<4, true>(poly, hp, out, esc)
            } else {
                clip_small_ptr::<4, false>(poly, hp, out, esc)
            }
        }
        5 => {
            if has_bounding_ref {
                clip_small_ptr_d::<5, true>(poly, hp, out, esc)
            } else {
                clip_small_ptr_d::<5, false>(poly, hp, out, esc)
            }
        }
        6 => {
            if has_bounding_ref {
                clip_small_ptr_d::<6, true>(poly, hp, out, esc)
            } else {
                clip_small_ptr_d::<6, false>(poly, hp, out, esc)
            }
        }
        7 => {
            if has_bounding_ref {
                clip_small_ptr_d::<7, true>(poly, hp, out, esc)
            } else {
                clip_small_ptr_d::<7, false>(poly, hp, out, esc)
            }
        }
        8 => {
            if has_bounding_ref {
                clip_small_ptr::<8, true>(poly, hp, out, esc)
            } else {
                clip_small_ptr::<8, false>(poly, hp, out, esc)
            }
        }
        _ => clip_bitmask(poly, hp, out, esc),
    }
}

/// Clip a convex polygon by a half-plane.
#[cfg_attr(feature = "profiling", inline(never))]
pub(crate) fn clip_convex(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
    esc: &EscalationCtx<'_>,
) -> ClipResult {
    let n = poly.len;

    debug_assert!(n >= 3, "clip_convex expects poly.len >= 3, got {}", n);

    let has_bounding_ref = poly.has_bounding_ref;
    // Early-unchanged is only relevant for bounded polys. Empirically, it is very rare for small N
    // (e.g. triangles), so we skip the check below a small-N threshold to save overhead.
    const EARLY_UNCHANGED_MIN_N: usize = 5;
    if !has_bounding_ref && n >= EARLY_UNCHANGED_MIN_N {
        let max_r2 = poly.max_r2;
        if max_r2 > 0.0 {
            // Clearance must exceed the escalation band so an early
            // Unchanged cannot skip a lane the canonical evaluator should
            // see (no-op at the production factor of 0, where this is the
            // legacy bound minus the eps slack — conservative direction).
            // `CLIP_ESCALATION_FACTOR` is the compile-time const 0.0 in
            // production, so `t` is just `hp.c`. The compiler can't fold the
            // `hp.eps * 0.0` away (IEEE: an inf/nan `eps` would yield nan), so
            // without this const guard it emits a dead mul+sub on every bounded
            // early-unchanged check. Same const-fold trick as `maybe_escalate`.
            let t = if crate::tolerances::CLIP_ESCALATION_FACTOR == 0.0 {
                hp.c
            } else {
                hp.c - hp.eps * crate::tolerances::CLIP_ESCALATION_FACTOR
            };
            if t >= 0.0 && t * t >= hp.ab2 * max_r2 {
                return ClipResult::Unchanged;
            }
        }
    }

    dispatch_clip(poly, hp, out, esc)
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
    esc: &EscalationCtx<'_>,
) -> ClipResult {
    debug_assert!(
        poly.len >= 3,
        "clip_convex expects poly.len >= 3, got {}",
        poly.len
    );

    dispatch_clip(poly, hp, out, esc)
}

#[cfg(any(test, feature = "microbench"))]
#[allow(unused_imports)]
pub(crate) use small::clip_convex_small_bool;
