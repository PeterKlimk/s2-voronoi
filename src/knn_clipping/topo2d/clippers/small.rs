/// Find the first inside-entry and inside-exit transitions of a cyclic mask.
///
/// Callers guarantee a *mixed* mask (the all-inside / none-inside cases are
/// handled before calling), and any mixed mask on a cyclic boundary contains
/// at least one `false -> true` and one `true -> false` transition, so both
/// scans always succeed; the `unreachable!` is a bug trap, not an input
/// state. If the epsilon-scale inside test produces multiple surviving arcs
/// (numerically non-convex polygon), the first arc is kept; the edge-check /
/// reconciliation layer owns any resulting cross-cell disagreement.
// Retained for the baseline bool clipper (microbench comparisons); the hot
// clippers use the branchless bitmask form.
#[cfg(any(test, feature = "microbench"))]
#[inline(always)]
fn find_entry_exit_transitions<const N: usize>(
    inside: &[bool; N],
) -> ((usize, usize), (usize, usize)) {
    let mut entry_idx = None;
    let mut exit_idx = None;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if entry_idx.is_none() && !inside[prev] && inside[i] {
            entry_idx = Some((prev, i));
        } else if exit_idx.is_none() && inside[prev] && !inside[i] {
            exit_idx = Some((prev, i));
        }
        if entry_idx.is_some() && exit_idx.is_some() {
            break;
        }
    }

    match (entry_idx, exit_idx) {
        (Some(entry), Some(exit)) => (entry, exit),
        _ => unreachable!(
            "mixed cyclic mask must contain both transitions (N={}, inside={:?})",
            N, inside
        ),
    }
}

/// Baseline small-N clipper for microbenchmark comparisons.
#[cfg(any(test, feature = "microbench"))]
#[allow(dead_code)]
#[allow(clippy::needless_range_loop)] // index drives 3 parallel outputs + pointer reads
pub(crate) fn clip_convex_small_bool<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= super::super::types::MAX_POLY_VERTICES);

    let neg_eps = -hp.eps;

    // SAFETY: `[MaybeUninit<f64>; N]` is valid in an uninitialized state.
    let mut dists: [core::mem::MaybeUninit<f64>; N] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

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

    if !any_inside {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if all_inside {
        return ClipResult::Unchanged;
    }

    let ((entry_idx, entry_next), (exit_idx, exit_next)) = find_entry_exit_transitions(&inside);

    out.len = 0;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        fp::fma_f64(u, u, v * v)
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
                has_bounding |= vp.0 == super::super::types::INVALID_PLANE_ID;
            }
        }};
    }

    let (d_entry, d_entry_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = fp::fma_f64(
        t_entry,
        poly.us[entry_next] - poly.us[entry_idx],
        poly.us[entry_idx],
    );
    let entry_v = fp::fma_f64(
        t_entry,
        poly.vs[entry_next] - poly.vs[entry_idx],
        poly.vs[entry_idx],
    );
    let entry_ep = poly.edge_planes[entry_idx];
    push!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    let mut i = entry_next;
    loop {
        push!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % N;
    }

    let (d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = fp::fma_f64(
        t_exit,
        poly.us[exit_next] - poly.us[exit_idx],
        poly.us[exit_idx],
    );
    let exit_v = fp::fma_f64(
        t_exit,
        poly.vs[exit_next] - poly.vs[exit_idx],
        poly.vs[exit_idx],
    );
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Initialized signed distances for one small-polygon SIMD evaluation.
///
/// The four-lane variant deliberately has no padded upper lanes: escalation
/// receives only this initialized slice, and every transition index is below
/// the const-generic live-lane count.
enum SmallDists {
    Four([f64; 4]),
    Eight([f64; 8]),
}

impl SmallDists {
    #[inline(always)]
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Four(dists) => dists,
            Self::Eight(dists) => dists,
        }
    }
}

#[inline(always)]
fn eval_small_dists<const N: usize>(poly: &PolyBuffer, hp: &HalfPlane) -> (SmallDists, u32) {
    let neg_eps = -hp.eps;
    let (us_chunks, _) = poly.us.as_chunks::<8>();
    let (vs_chunks, _) = poly.vs.as_chunks::<8>();
    if N <= 4 {
        let (dists, mask) =
            fp::signed_dists_mask4(hp.a, hp.b, hp.c, &us_chunks[0], &vs_chunks[0], neg_eps);
        (SmallDists::Four(dists), mask)
    } else {
        let (dists, mask) =
            fp::signed_dists_mask8(hp.a, hp.b, hp.c, &us_chunks[0], &vs_chunks[0], neg_eps);
        (SmallDists::Eight(dists), mask)
    }
}

/// Small-N clipper using modulo iteration (for N=4,8 where `% N` is a bitmask).
#[inline(always)]
#[allow(clippy::needless_range_loop)] // index drives 3 parallel outputs + pointer reads
pub(super) fn clip_small_ptr<const N: usize, const TRACK_BOUNDING: bool>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
    esc: &super::EscalationCtx<'_>,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    // For N <= 4 (triangles/quads) only the low four lanes are live, so one
    // f64x4 eval suffices; N in 5..=8 needs all eight. `N` is a const generic,
    // so this branch resolves at compile time. Lanes past N read
    // stale-but-finite data and are masked down to the live N bits; lane math
    // is bit-identical to the scalar signed_dist formula.
    let full: u32 = (1u32 << N) - 1;
    let (dists, inside_bits) = eval_small_dists::<N>(poly, hp);
    let dists = dists.as_slice();
    let mask = inside_bits & full;
    // P5 escalation: near-margin lanes are re-decided by the exact
    // canonical predicate (one abs+compare per live lane on the hot path).
    let mask = super::maybe_escalate(mask as u64, dists, N, hp, poly, esc) as u32;

    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if mask == full {
        return ClipResult::Unchanged;
    }

    // First entry (prev outside -> inside) and first exit (prev inside ->
    // outside) transitions of the cyclic mask, branchlessly: rot[i] holds
    // inside[prev(i)]. A mixed cyclic mask always contains both.
    let rot = ((mask << 1) | (mask >> (N - 1))) & full;
    let entry_bits = mask & !rot;
    let exit_bits = rot & !mask;
    debug_assert!(entry_bits != 0 && exit_bits != 0);
    let entry_next = entry_bits.trailing_zeros() as usize;
    let exit_next = exit_bits.trailing_zeros() as usize;
    let entry_idx = if entry_next == 0 {
        N - 1
    } else {
        entry_next - 1
    };
    let exit_idx = if exit_next == 0 { N - 1 } else { exit_next - 1 };

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        fp::fma_f64(u, u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            *out.us.get_unchecked_mut(out_len) = u;
            *out.vs.get_unchecked_mut(out_len) = v;
            *out.vertex_planes.get_unchecked_mut(out_len) = vp;
            *out.edge_planes.get_unchecked_mut(out_len) = ep;
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if TRACK_BOUNDING {
                has_bounding |= vp.0 == super::super::types::INVALID_PLANE_ID;
            }
        }};
    }

    unsafe {
        let d_entry = *dists.get_unchecked(entry_idx);
        let d_entry_next = *dists.get_unchecked(entry_next);
        let d_exit = *dists.get_unchecked(exit_idx);
        let d_exit_next = *dists.get_unchecked(exit_next);
        let (t_entry, t_exit) =
            super::lerp_t_pair(d_entry, d_entry_next, d_exit, d_exit_next, hp.eps);
        let entry_u = fp::fma_f64(
            t_entry,
            *us.add(entry_next) - *us.add(entry_idx),
            *us.add(entry_idx),
        );
        let entry_v = fp::fma_f64(
            t_entry,
            *vs.add(entry_next) - *vs.add(entry_idx),
            *vs.add(entry_idx),
        );
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i = (i + 1) % N;
        }

        let exit_u = fp::fma_f64(
            t_exit,
            *us.add(exit_next) - *us.add(exit_idx),
            *us.add(exit_idx),
        );
        let exit_v = fp::fma_f64(
            t_exit,
            *vs.add(exit_next) - *vs.add(exit_idx),
            *vs.add(exit_idx),
        );
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if TRACK_BOUNDING { has_bounding } else { false };
    ClipResult::Changed
}

/// Small-N clipper using modulo-free iteration (for N=3,5,6,7 where `% N` is expensive).
#[inline(always)]
#[allow(clippy::needless_range_loop)] // index drives 3 parallel outputs + pointer reads
pub(super) fn clip_small_ptr_d<const N: usize, const TRACK_BOUNDING: bool>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
    esc: &super::EscalationCtx<'_>,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    // For N <= 4 (triangles/quads) only the low four lanes are live, so one
    // f64x4 eval suffices; N in 5..=8 needs all eight. `N` is a const generic,
    // so this branch resolves at compile time. Lanes past N read
    // stale-but-finite data and are masked down to the live N bits; lane math
    // is bit-identical to the scalar signed_dist formula.
    let full: u32 = (1u32 << N) - 1;
    let (dists, inside_bits) = eval_small_dists::<N>(poly, hp);
    let dists = dists.as_slice();
    let mask = inside_bits & full;
    // P5 escalation: near-margin lanes are re-decided by the exact
    // canonical predicate (one abs+compare per live lane on the hot path).
    let mask = super::maybe_escalate(mask as u64, dists, N, hp, poly, esc) as u32;

    if mask == 0 {
        out.len = 0;
        out.max_r2 = 0.0;
        out.has_bounding_ref = false;
        return ClipResult::Changed;
    }

    if mask == full {
        return ClipResult::Unchanged;
    }

    // First entry (prev outside -> inside) and first exit (prev inside ->
    // outside) transitions of the cyclic mask, branchlessly: rot[i] holds
    // inside[prev(i)]. A mixed cyclic mask always contains both.
    let rot = ((mask << 1) | (mask >> (N - 1))) & full;
    let entry_bits = mask & !rot;
    let exit_bits = rot & !mask;
    debug_assert!(entry_bits != 0 && exit_bits != 0);
    let entry_next = entry_bits.trailing_zeros() as usize;
    let exit_next = exit_bits.trailing_zeros() as usize;
    let entry_idx = if entry_next == 0 {
        N - 1
    } else {
        entry_next - 1
    };
    let exit_idx = if exit_next == 0 { N - 1 } else { exit_next - 1 };

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        fp::fma_f64(u, u, v * v)
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            *out.us.get_unchecked_mut(out_len) = u;
            *out.vs.get_unchecked_mut(out_len) = v;
            *out.vertex_planes.get_unchecked_mut(out_len) = vp;
            *out.edge_planes.get_unchecked_mut(out_len) = ep;
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if TRACK_BOUNDING {
                has_bounding |= vp.0 == super::super::types::INVALID_PLANE_ID;
            }
        }};
    }

    unsafe {
        // Read all four distances first, then issue both divisions for ILP
        let d_entry = *dists.get_unchecked(entry_idx);
        let d_entry_next = *dists.get_unchecked(entry_next);
        let d_exit = *dists.get_unchecked(exit_idx);
        let d_exit_next = *dists.get_unchecked(exit_next);

        let (t_entry, t_exit) =
            super::lerp_t_pair(d_entry, d_entry_next, d_exit, d_exit_next, hp.eps);

        let entry_u = fp::fma_f64(
            t_entry,
            *us.add(entry_next) - *us.add(entry_idx),
            *us.add(entry_idx),
        );
        let entry_v = fp::fma_f64(
            t_entry,
            *vs.add(entry_next) - *vs.add(entry_idx),
            *vs.add(entry_idx),
        );
        let entry_ep = *eps.add(entry_idx);
        push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        let mut i = entry_next;
        loop {
            push_idx!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            if i == exit_idx {
                break;
            }
            i += 1;
            if i == N {
                i = 0;
            }
        }

        let exit_u = fp::fma_f64(
            t_exit,
            *us.add(exit_next) - *us.add(exit_idx),
            *us.add(exit_idx),
        );
        let exit_v = fp::fma_f64(
            t_exit,
            *vs.add(exit_next) - *vs.add(exit_idx),
            *vs.add(exit_idx),
        );
        let exit_ep = *eps.add(exit_idx);
        push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if TRACK_BOUNDING { has_bounding } else { false };
    ClipResult::Changed
}
use super::super::types::{ClipResult, HalfPlane, PolyBuffer};
use crate::fp;
