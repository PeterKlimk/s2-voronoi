//! Experimental / microbench-only `clip_convex` variants.
//!
//! This module is compiled only for tests or when the `microbench` feature is enabled.
//! Keeping these out of `topo2d.rs` helps keep the main implementation readable.

#![allow(dead_code)]

use super::clippers::clip_convex_small_bool;
use super::types::{ClipResult, HalfPlane, PolyBuffer};
use std::simd::StdFloat;

/// Single-pass small-bool variant:
/// - no `inside[]` / `dists[]` arrays
/// - captures entry/exit transitions + distances while classifying
/// - uses modulo-free vertex copy (two linear ranges)
#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_small_bool_stream<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    // Seed with last vertex for the wrap-around edge (N-1 -> 0).
    let mut any_inside = false;
    let mut all_inside = true;

    let mut prev_idx = N - 1;
    let mut d_prev = unsafe { hp.signed_dist(*us.add(prev_idx), *vs.add(prev_idx)) };
    let mut prev_inside = d_prev >= neg_eps;
    any_inside |= prev_inside;
    all_inside &= prev_inside;

    // Captured transitions (prev -> cur).
    let mut entry: Option<(usize, usize, f64, f64)> = None;
    let mut exit: Option<(usize, usize, f64, f64)> = None;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            let inside = d >= neg_eps;

            any_inside |= inside;
            all_inside &= inside;

            if entry.is_none() && !prev_inside && inside {
                entry = Some((prev_idx, i, d_prev, d));
            } else if exit.is_none() && prev_inside && !inside {
                exit = Some((prev_idx, i, d_prev, d));
            }

            if entry.is_some() && exit.is_some() {
                // Mixed case is guaranteed once both transitions are found.
                break;
            }

            prev_idx = i;
            d_prev = d;
            prev_inside = inside;
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

    let (entry_idx, entry_next, d_entry, d_entry_next) =
        entry.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next, d_exit, d_exit_next) =
        exit.expect("convex polygon must have exit transition");

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

    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Like `clip_convex_small_bool_stream`, but uses raw pointers for input reads in the mixed case.
#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_small_bool_stream_ptr<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();
    let vps = poly.vertex_planes.as_ptr();
    let eps = poly.edge_planes.as_ptr();

    let mut any_inside = false;
    let mut all_inside = true;

    let mut prev_idx = N - 1;
    let mut d_prev = unsafe { hp.signed_dist(*us.add(prev_idx), *vs.add(prev_idx)) };
    let mut prev_inside = d_prev >= neg_eps;
    any_inside |= prev_inside;
    all_inside &= prev_inside;

    let mut entry: Option<(usize, usize, f64, f64)> = None;
    let mut exit: Option<(usize, usize, f64, f64)> = None;

    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            let inside = d >= neg_eps;

            any_inside |= inside;
            all_inside &= inside;

            if entry.is_none() && !prev_inside && inside {
                entry = Some((prev_idx, i, d_prev, d));
            } else if exit.is_none() && prev_inside && !inside {
                exit = Some((prev_idx, i, d_prev, d));
            }

            if entry.is_some() && exit.is_some() {
                break;
            }

            prev_idx = i;
            d_prev = d;
            prev_inside = inside;
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

    let (entry_idx, entry_next, d_entry, d_entry_next) =
        entry.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next, d_exit, d_exit_next) =
        exit.expect("convex polygon must have exit transition");

    out.len = 0;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

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

    unsafe {
        let t_entry = d_entry / (d_entry - d_entry_next);
        let entry_u = t_entry.mul_add(*us.add(entry_next) - *us.add(entry_idx), *us.add(entry_idx));
        let entry_v = t_entry.mul_add(*vs.add(entry_next) - *vs.add(entry_idx), *vs.add(entry_idx));
        let entry_ep = *eps.add(entry_idx);
        push_raw_track!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

        if entry_next <= exit_idx {
            for i in entry_next..=exit_idx {
                push_raw_track!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            }
        } else {
            for i in entry_next..N {
                push_raw_track!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            }
            for i in 0..=exit_idx {
                push_raw_track!(*us.add(i), *vs.add(i), *vps.add(i), *eps.add(i));
            }
        }

        let t_exit = d_exit / (d_exit - d_exit_next);
        let exit_u = t_exit.mul_add(*us.add(exit_next) - *us.add(exit_idx), *us.add(exit_idx));
        let exit_v = t_exit.mul_add(*vs.add(exit_next) - *vs.add(exit_idx), *vs.add(exit_idx));
        let exit_ep = *eps.add(exit_idx);
        push_raw_track!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);
    }

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_small_bool_split<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    if poly.has_bounding_ref {
        clip_convex_small_bool_unbounded::<N>(poly, hp, out)
    } else {
        clip_convex_small_bool_bounded::<N>(poly, hp, out)
    }
}

#[inline(always)]
#[cfg(any(test, feature = "microbench"))]
fn clip_convex_small_bool_bounded<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);
    debug_assert!(!poly.has_bounding_ref);

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

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    out.len = 0;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    let mut max_r2 = 0.0f64;

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
        }};
    }

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

    // Copy inside run entry_next..=exit_idx (cyclic)
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
        i += 1;
        if i == N {
            i = 0;
        }
    }

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
    out.has_bounding_ref = false;
    ClipResult::Changed
}

#[inline(always)]
#[cfg(any(test, feature = "microbench"))]
fn clip_convex_small_bool_unbounded<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);
    debug_assert!(poly.has_bounding_ref);

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

    // Mixed case: find the two transition indices
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

    let (entry_idx, entry_next) = entry_idx.expect("convex polygon must have entry transition");
    let (exit_idx, exit_next) = exit_idx.expect("convex polygon must have exit transition");

    out.len = 0;

    let mut has_bounding = false;

    macro_rules! push_track_bounding {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let vp = $vp;
            out.push_raw($u, $v, vp, $ep);
            has_bounding |= vp.0 == usize::MAX;
        }};
    }

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
    push_track_bounding!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    // Copy inside run entry_next..=exit_idx (cyclic)
    let mut i = entry_next;
    loop {
        push_track_bounding!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i += 1;
        if i == N {
            i = 0;
        }
    }

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
    push_track_bounding!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.has_bounding_ref = has_bounding;
    if has_bounding {
        out.max_r2 = 0.0;
    } else {
        // This clip produced the first bounded polygon. Compute max_r2 in one scan.
        let mut max_r2 = 0.0f64;
        for j in 0..out.len {
            let u = out.us[j];
            let v = out.vs[j];
            let r2 = u.mul_add(u, v * v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
        }
        out.max_r2 = max_r2;
    }

    ClipResult::Changed
}

#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_small<const N: usize>(
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

        let u = t.mul_add(self.poly.us[OUT] - self.poly.us[IN], self.poly.us[IN]);
        let v = t.mul_add(self.poly.vs[OUT] - self.poly.vs[IN], self.poly.vs[IN]);
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

        let u = t.mul_add(self.poly.us[OUT] - self.poly.us[IN], self.poly.us[IN]);
        let v = t.mul_add(self.poly.vs[OUT] - self.poly.vs[IN], self.poly.vs[IN]);
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
        include!("../clip_tables/table_n3.rs");
        include!("../clip_tables/table_n4.rs");
        include!("../clip_tables/table_n5.rs");
        include!("../clip_tables/table_n6.rs");
        include!("../clip_tables/table_n7.rs");
        include!("../clip_tables/table_n8.rs");
    }

    #[inline(always)]
    pub(super) fn run_n3(mask: u8, dists: &[f64; 4], ctx: &mut ClipCtx<'_>) -> ClipResult {
        #[cold]
        fn fallback(poly: &PolyBuffer, hp: &HalfPlane, out: &mut PolyBuffer) -> ClipResult {
            clip_convex_small_bool::<3>(poly, hp, out)
        }
        include!("../clip_tables/ngon_n3.rs");
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
        include!("../clip_tables/ngon_n4.rs");
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
        include!("../clip_tables/ngon_n5.rs");
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
        include!("../clip_tables/ngon_n6.rs");
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
        include!("../clip_tables/ngon_n7.rs");
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
        include!("../clip_tables/ngon_n8.rs");
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
fn run_jump_table_4<const N: usize>(
    mask: u8,
    dists: &[f64; 4],
    ctx: &mut ClipCtx<'_>,
) -> ClipResult {
    if N == 3 {
        clip_jump_tables::run_n3(mask, dists, ctx)
    } else if N == 4 {
        clip_jump_tables::run_n4(mask, dists, ctx)
    } else {
        unreachable!("run_jump_table_4 only supports N=3 or N=4")
    }
}

#[inline(always)]
fn run_jump_table_8<const N: usize>(
    mask: u8,
    dists: &[f64; 8],
    ctx: &mut ClipCtx<'_>,
) -> ClipResult {
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
pub(super) fn clip_convex_table_ngon<const N: usize>(
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
pub(super) fn clip_convex_match_ngon<const N: usize>(
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

    use std::simd::f64x4;
    use std::simd::prelude::*;

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

    use std::simd::f64x8;
    use std::simd::prelude::*;

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
pub(super) fn clip_convex_simd<const N: usize>(
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

/// Hoisted division variant: starts both divisions as early as possible to hide latency
/// behind the vertex copying loop.
#[cfg_attr(feature = "profiling", inline(never))]
pub(super) fn clip_convex_small_bool_b<const N: usize>(
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

    // HOISTED: Start both divisions immediately to hide latency.
    // The CPU can pipeline these while we do setup / copy loop.
    let (d_entry, d_entry_next, d_exit, d_exit_next) = unsafe {
        (
            dists.get_unchecked(entry_idx).assume_init_read(),
            dists.get_unchecked(entry_next).assume_init_read(),
            dists.get_unchecked(exit_idx).assume_init_read(),
            dists.get_unchecked(exit_next).assume_init_read(),
        )
    };
    let t_entry = d_entry / (d_entry - d_entry_next);
    let t_exit = d_exit / (d_exit - d_exit_next);

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

    // Entry intersection
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

    // Exit intersection
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Bool variant with proper division hoisting: precompute ALL edge interpolation parameters
/// before knowing which edges are transitions. This allows all divisions to be in flight
/// simultaneously while we scan for transitions.
#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_bool_c<const N: usize>(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, N);
    debug_assert!(N >= 3 && N <= 8);

    let neg_eps = -hp.eps;

    let us = poly.us.as_ptr();
    let vs = poly.vs.as_ptr();

    // Pass 1: compute distances and classify vertices
    let mut dists = [0.0f64; N];
    let mut inside = [false; N];
    let mut any_inside = false;
    let mut all_inside = true;

    // SAFETY: i < N <= 8 <= MAX_POLY_VERTICES
    unsafe {
        for i in 0..N {
            let d = hp.signed_dist(*us.add(i), *vs.add(i));
            dists[i] = d;
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

    // HOISTED: Precompute ALL N edge interpolation parameters.
    // Even though only 2 edges are transitions, computing all lets the CPU
    // pipeline all divisions in parallel while we search for transitions.
    let mut t_vals = [0.0f64; N];
    for i in 0..N {
        let next = if i + 1 == N { 0 } else { i + 1 };
        let d_i = dists[i];
        let d_next = dists[next];
        // t = d_i / (d_i - d_next)
        // This division is valid even if (d_i - d_next) == 0 (same side - won't be used)
        t_vals[i] = d_i / (d_i - d_next);
    }

    // Find the two transitions
    let mut entry_idx = 0usize;
    let mut entry_next = 0usize;
    let mut exit_idx = 0usize;
    let mut found_entry = false;

    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        if !inside[prev] && inside[i] {
            entry_idx = prev;
            entry_next = i;
            found_entry = true;
        } else if inside[prev] && !inside[i] {
            exit_idx = prev;
            if found_entry {
                break;
            }
        }
    }

    // Build output
    out.len = 0;

    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            // Use .max() for branchless cmov
            max_r2 = max_r2.max(r2_of(u, v));
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection (use precomputed t_vals[entry_idx])
    let t_entry = t_vals[entry_idx];
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

    // Exit intersection (use precomputed t_vals[exit_idx])
    let exit_next = if exit_idx + 1 == N { 0 } else { exit_idx + 1 };
    let t_exit = t_vals[exit_idx];
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
    ClipResult::Changed
}

/// Fully unrolled triangle clipping with match-based dispatch.
/// For N=3, there are only 6 possible inside/outside configurations (excluding all-in/all-out).
#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_tri_unrolled(
    poly: &PolyBuffer,
    hp: &HalfPlane,
    out: &mut PolyBuffer,
) -> ClipResult {
    debug_assert_eq!(poly.len, 3);

    let neg_eps = -hp.eps;

    let d0 = hp.signed_dist(poly.us[0], poly.vs[0]);
    let d1 = hp.signed_dist(poly.us[1], poly.vs[1]);
    let d2 = hp.signed_dist(poly.us[2], poly.vs[2]);

    let i0 = d0 >= neg_eps;
    let i1 = d1 >= neg_eps;
    let i2 = d2 >= neg_eps;

    // Encode as 3-bit pattern: (i2 << 2) | (i1 << 1) | i0
    let pattern = (i2 as u8) << 2 | (i1 as u8) << 1 | (i0 as u8);

    match pattern {
        0b111 => ClipResult::Unchanged, // All inside
        0b000 => {
            // All outside
            out.len = 0;
            out.max_r2 = 0.0;
            out.has_bounding_ref = false;
            ClipResult::Changed
        }
        _ => {
            // Mixed case - precompute all 3 interpolation parameters
            let t01 = d0 / (d0 - d1); // edge 0->1
            let t12 = d1 / (d1 - d2); // edge 1->2
            let t20 = d2 / (d2 - d0); // edge 2->0

            out.len = 0;
            let mut max_r2 = 0.0f64;
            let mut has_bounding = false;
            let track_bounding = poly.has_bounding_ref;

            #[inline(always)]
            fn interp(poly: &PolyBuffer, from: usize, to: usize, t: f64) -> (f64, f64) {
                let u = t.mul_add(poly.us[to] - poly.us[from], poly.us[from]);
                let v = t.mul_add(poly.vs[to] - poly.vs[from], poly.vs[from]);
                (u, v)
            }

            macro_rules! emit_vertex {
                ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
                    let u = $u;
                    let v = $v;
                    let vp = $vp;
                    out.push_raw(u, v, vp, $ep);
                    max_r2 = max_r2.max(u.mul_add(u, v * v));
                    if track_bounding {
                        has_bounding |= vp.0 == usize::MAX;
                    }
                }};
            }

            macro_rules! emit_input {
                ($i:expr) => {{
                    emit_vertex!(
                        poly.us[$i],
                        poly.vs[$i],
                        poly.vertex_planes[$i],
                        poly.edge_planes[$i]
                    );
                }};
            }

            let hp_idx = hp.plane_idx;

            match pattern {
                // One vertex inside (3 output vertices)
                0b001 => {
                    // Only v0 inside: emit interp(2,0), v0, interp(0,1)
                    let (eu, ev) = interp(poly, 2, 0, t20);
                    emit_vertex!(eu, ev, (poly.edge_planes[2], hp_idx), poly.edge_planes[2]);
                    emit_input!(0);
                    let (xu, xv) = interp(poly, 0, 1, t01);
                    emit_vertex!(xu, xv, (poly.edge_planes[0], hp_idx), hp_idx);
                }
                0b010 => {
                    // Only v1 inside: emit interp(0,1), v1, interp(1,2)
                    let (eu, ev) = interp(poly, 0, 1, t01);
                    emit_vertex!(eu, ev, (poly.edge_planes[0], hp_idx), poly.edge_planes[0]);
                    emit_input!(1);
                    let (xu, xv) = interp(poly, 1, 2, t12);
                    emit_vertex!(xu, xv, (poly.edge_planes[1], hp_idx), hp_idx);
                }
                0b100 => {
                    // Only v2 inside: emit interp(1,2), v2, interp(2,0)
                    let (eu, ev) = interp(poly, 1, 2, t12);
                    emit_vertex!(eu, ev, (poly.edge_planes[1], hp_idx), poly.edge_planes[1]);
                    emit_input!(2);
                    let (xu, xv) = interp(poly, 2, 0, t20);
                    emit_vertex!(xu, xv, (poly.edge_planes[2], hp_idx), hp_idx);
                }
                // Two vertices inside (4 output vertices)
                0b011 => {
                    // v0, v1 inside: emit interp(2,0), v0, v1, interp(1,2)
                    let (eu, ev) = interp(poly, 2, 0, t20);
                    emit_vertex!(eu, ev, (poly.edge_planes[2], hp_idx), poly.edge_planes[2]);
                    emit_input!(0);
                    emit_input!(1);
                    let (xu, xv) = interp(poly, 1, 2, t12);
                    emit_vertex!(xu, xv, (poly.edge_planes[1], hp_idx), hp_idx);
                }
                0b110 => {
                    // v1, v2 inside: emit interp(0,1), v1, v2, interp(2,0)
                    let (eu, ev) = interp(poly, 0, 1, t01);
                    emit_vertex!(eu, ev, (poly.edge_planes[0], hp_idx), poly.edge_planes[0]);
                    emit_input!(1);
                    emit_input!(2);
                    let (xu, xv) = interp(poly, 2, 0, t20);
                    emit_vertex!(xu, xv, (poly.edge_planes[2], hp_idx), hp_idx);
                }
                0b101 => {
                    // v0, v2 inside: emit interp(1,2), v2, v0, interp(0,1)
                    let (eu, ev) = interp(poly, 1, 2, t12);
                    emit_vertex!(eu, ev, (poly.edge_planes[1], hp_idx), poly.edge_planes[1]);
                    emit_input!(2);
                    emit_input!(0);
                    let (xu, xv) = interp(poly, 0, 1, t01);
                    emit_vertex!(xu, xv, (poly.edge_planes[0], hp_idx), hp_idx);
                }
                _ => unreachable!(),
            }

            out.max_r2 = max_r2;
            out.has_bounding_ref = if track_bounding { has_bounding } else { false };
            ClipResult::Changed
        }
    }
}

/// Bool variant using .max() for branchless max_r2 tracking.
#[cfg_attr(feature = "profiling", inline(never))]
#[cfg(any(test, feature = "microbench"))]
pub(super) fn clip_convex_bool_maxr2<const N: usize>(
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

    // Mixed case: find the two transition indices
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

    // Key difference: use .max() instead of if-branch for r2 tracking
    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            max_r2 = max_r2.max(r2_of(u, v));
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection
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

    // Surviving input vertices
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

    // Exit intersection
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
