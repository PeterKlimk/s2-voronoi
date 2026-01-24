//! Batch-oriented polygon clipping using SIMD (AVX-512 f64x8 or AVX2 2×f64x4).
//!
//! This module implements a batch clipping approach where multiple half-planes are processed
//! together using SIMD. The key idea is:
//!
//! 1. Classify all N vertices against all 8 half-planes in parallel using SIMD
//! 2. Filter out half-planes that are all-outside (discarded) or all-inside (unchanged)
//! 3. Clip against remaining half-planes, maintaining per-vertex classification masks
//! 4. Reclassify only new intersection vertices (not original vertices) after each clip
//!
//! This reduces redundant distance calculations and leverages SIMD parallelism.
//!
//! On AVX2 hardware, f64x8 operations are executed as two f64x4 operations.

use super::types::{ClipResult, HalfPlane, PolyBuffer, MAX_POLY_VERTICES};
use std::simd::f64x8;
use std::simd::cmp::SimdPartialOrd;

type VertexMasks = [u16; MAX_POLY_VERTICES];

/// Batch of 8 half-planes in SIMD-friendly layout.
#[derive(Clone, Copy)]
pub struct HpBatch8 {
    /// SIMD vectors of half-plane coefficients: [a0, a1, ..., a7], etc.
    pub a: f64x8,
    pub b: f64x8,
    pub c: f64x8,
    /// Precomputed `a^2 + b^2` per lane.
    pub ab2: f64x8,
    /// Per-lane epsilon values
    pub eps: f64x8,
    /// Scalar copies for cheap lane access (avoid repeated `to_array()` in hot paths).
    pub a_arr: [f64; 8],
    pub b_arr: [f64; 8],
    pub c_arr: [f64; 8],
    pub ab2_arr: [f64; 8],
    pub eps_arr: [f64; 8],
    /// Per-lane plane indices (stored as array for convenience)
    pub plane_idxs: [usize; 8],
    /// Number of active half-planes (1-8)
    pub len: u8,
}

impl HpBatch8 {
    /// Create a batch from up to 8 half-planes.
    ///
    /// # Safety
    /// Caller must ensure `hps.len() <= 8`.
    #[inline]
    pub unsafe fn from_slice_unchecked(hps: &[HalfPlane]) -> Self {
        debug_assert!(hps.len() <= 8);

        let mut a_arr = [0.0f64; 8];
        let mut b_arr = [0.0f64; 8];
        let mut c_arr = [0.0f64; 8];
        let mut ab2_arr = [0.0f64; 8];
        let mut eps_arr = [0.0f64; 8];
        let mut plane_idxs = [0usize; 8];

        for (i, hp) in hps.iter().enumerate() {
            a_arr[i] = hp.a;
            b_arr[i] = hp.b;
            c_arr[i] = hp.c;
            ab2_arr[i] = hp.ab2;
            eps_arr[i] = hp.eps;
            plane_idxs[i] = hp.plane_idx;
        }

        Self {
            a: f64x8::from_array(a_arr),
            b: f64x8::from_array(b_arr),
            c: f64x8::from_array(c_arr),
            ab2: f64x8::from_array(ab2_arr),
            eps: f64x8::from_array(eps_arr),
            a_arr,
            b_arr,
            c_arr,
            ab2_arr,
            eps_arr,
            plane_idxs,
            len: hps.len() as u8,
        }
    }

    /// Create a batch from exactly 8 half-planes.
    #[inline]
    pub fn from_array(hps: [HalfPlane; 8]) -> Self {
        unsafe { Self::from_slice_unchecked(&hps) }
    }

    /// Get the i-th half-plane.
    #[inline]
    pub fn get(&self, i: usize) -> HalfPlane {
        debug_assert!(i < 8);
        HalfPlane {
            a: self.a_arr[i],
            b: self.b_arr[i],
            c: self.c_arr[i],
            ab2: self.ab2_arr[i],
            plane_idx: self.plane_idxs[i],
            eps: self.eps_arr[i],
        }
    }

    /// Check if lane i is active.
    #[inline]
    pub fn is_lane_active(&self, lane_mask: u8, i: usize) -> bool {
        (lane_mask >> i) & 1 == 1
    }
}

/// Classify all vertices of a polygon against all half-planes in a batch.
///
/// Returns an array where each element is a bitmask of which half-planes the vertex is inside.
/// For example, if vertex 0 is inside half-planes 0 and 2, the result[0] = 0b00000101.
///
/// # Arguments
/// * `poly` - The polygon to classify
/// * `batch` - The batch of half-planes
///
/// # Returns
/// An array of `poly.len` u16 values, each containing an 8-bit mask.
#[inline]
pub fn classify_batch(poly: &PolyBuffer, batch: &HpBatch8) -> VertexMasks {
    let n = poly.len;
    debug_assert!(
        n <= MAX_POLY_VERTICES,
        "poly.len exceeds MAX_POLY_VERTICES"
    );

    let mut result = [0u16; MAX_POLY_VERTICES];

    let a = batch.a;
    let b = batch.b;
    let c = batch.c;
    let neg_eps = -batch.eps;

    for i in 0..n {
        let u = poly.us[i];
        let v = poly.vs[i];

        // SIMD: compute signed distances for all 8 half-planes
        // dist = a*u + b*v + c
        let u_vec = f64x8::splat(u);
        let v_vec = f64x8::splat(v);

        let dist = a * u_vec + b * v_vec + c;

        // Check if dist >= -eps for each lane
        let inside_mask = dist.simd_ge(neg_eps);
        let bits = inside_mask.to_bitmask() as u16;

        result[i] = bits;
    }

    result
}

#[inline]
fn classify_batch_with_lane_masks(poly: &PolyBuffer, batch: &HpBatch8) -> (VertexMasks, [u64; 8]) {
    let n = poly.len;
    debug_assert!(
        n <= MAX_POLY_VERTICES,
        "poly.len exceeds MAX_POLY_VERTICES"
    );
    debug_assert!(n <= 64, "lane bitmasks are stored in u64");

    let mut vertex_masks = [0u16; MAX_POLY_VERTICES];
    let mut lane_masks = [0u64; 8];

    let a = batch.a;
    let b = batch.b;
    let c = batch.c;
    let neg_eps = -batch.eps;

    for i in 0..n {
        let u_vec = f64x8::splat(poly.us[i]);
        let v_vec = f64x8::splat(poly.vs[i]);

        let dist = a * u_vec + b * v_vec + c;
        let inside_mask = dist.simd_ge(neg_eps);
        let bits = inside_mask.to_bitmask() as u16;

        vertex_masks[i] = bits;

        let bit = 1u64 << i;
        lane_masks[0] |= (((bits >> 0) & 1) as u64) * bit;
        lane_masks[1] |= (((bits >> 1) & 1) as u64) * bit;
        lane_masks[2] |= (((bits >> 2) & 1) as u64) * bit;
        lane_masks[3] |= (((bits >> 3) & 1) as u64) * bit;
        lane_masks[4] |= (((bits >> 4) & 1) as u64) * bit;
        lane_masks[5] |= (((bits >> 5) & 1) as u64) * bit;
        lane_masks[6] |= (((bits >> 6) & 1) as u64) * bit;
        lane_masks[7] |= (((bits >> 7) & 1) as u64) * bit;
    }

    (vertex_masks, lane_masks)
}

#[inline]
fn classify_lanes_only(poly: &PolyBuffer, batch: &HpBatch8) -> [u64; 8] {
    let n = poly.len;
    debug_assert!(n <= MAX_POLY_VERTICES);
    debug_assert!(n <= 64, "lane bitmasks are stored in u64");

    let mut lane_masks = [0u64; 8];

    let a = batch.a;
    let b = batch.b;
    let c = batch.c;
    let neg_eps = -batch.eps;

    for i in 0..n {
        let u_vec = f64x8::splat(poly.us[i]);
        let v_vec = f64x8::splat(poly.vs[i]);

        let dist = a * u_vec + b * v_vec + c;
        let inside_mask = dist.simd_ge(neg_eps);
        let bits = inside_mask.to_bitmask() as u8;

        lane_masks[0] |= ((bits & 1) as u64) << i;
        lane_masks[1] |= (((bits >> 1) & 1) as u64) << i;
        lane_masks[2] |= (((bits >> 2) & 1) as u64) << i;
        lane_masks[3] |= (((bits >> 3) & 1) as u64) << i;
        lane_masks[4] |= (((bits >> 4) & 1) as u64) << i;
        lane_masks[5] |= (((bits >> 5) & 1) as u64) << i;
        lane_masks[6] |= (((bits >> 6) & 1) as u64) << i;
        lane_masks[7] |= (((bits >> 7) & 1) as u64) << i;
    }

    lane_masks
}

/// Classify a single vertex against a subset of active half-planes.
///
/// Returns an 8-bit mask indicating which half-planes the vertex is inside.
///
/// # Arguments
/// * `u` - Vertex u coordinate
/// * `v` - Vertex v coordinate
/// * `batch` - The batch of half-planes
/// * `lane_mask` - Bitmask of which lanes are active (only these bits are set in result)
#[inline]
pub fn classify_one_vertex(u: f64, v: f64, batch: &HpBatch8, lane_mask: u8) -> u16 {
    let u_vec = f64x8::splat(u);
    let v_vec = f64x8::splat(v);

    let dist = batch.a * u_vec + batch.b * v_vec + batch.c;
    let neg_eps = -batch.eps;

    let inside_mask = dist.simd_ge(neg_eps);
    let mut bits = inside_mask.to_bitmask() as u16;

    // Mask out inactive lanes
    bits &= lane_mask as u16;

    bits
}

#[inline]
fn full_vertex_mask(n: usize) -> u64 {
    debug_assert!(n <= 64);
    if n == 64 { !0u64 } else { (1u64 << n) - 1 }
}

#[inline]
fn vertices_inside_lane_mask(vertex_masks: &VertexMasks, n: usize, lane: usize) -> u64 {
    debug_assert!(lane < 8);
    let mut m = 0u64;
    for i in 0..n {
        m |= (((vertex_masks[i] >> lane) & 1) as u64) << i;
    }
    m
}

/// Clip a polygon against a batch of half-planes.
///
/// # Arguments
/// * `poly` - Input polygon (>= 3 vertices)
/// * `batch` - Batch of 1-8 half-planes
/// * `out` - Output polygon buffer
///
/// # Returns
/// `ClipResult` indicating the outcome
#[cfg_attr(feature = "profiling", inline(never))]
pub fn clip_batch(poly: &PolyBuffer, batch: &HpBatch8, out: &mut PolyBuffer) -> ClipResult {
    let mut scratch = PolyBuffer::new();
    clip_batch_with_scratch(poly, batch, out, &mut scratch)
}

/// Clip a polygon against a batch of half-planes, reusing a caller-provided scratch buffer.
///
/// This avoids allocating/zeroing large `PolyBuffer`s on every call and is the intended
/// fast path for benchmarking/experimentation.
///
/// # Safety / Aliasing
/// `scratch` must not alias `out` or `poly`.
#[cfg_attr(feature = "profiling", inline(never))]
pub fn clip_batch_with_scratch(
    poly: &PolyBuffer,
    batch: &HpBatch8,
    out: &mut PolyBuffer,
    scratch: &mut PolyBuffer,
) -> ClipResult {
    let n = poly.len;
    debug_assert!(n >= 3 && n <= MAX_POLY_VERTICES);
    debug_assert!(n <= 64, "lane bitmasks are stored in u64");
    debug_assert!(
        !std::ptr::eq(out, scratch),
        "`out` and `scratch` must be distinct"
    );

    // Track which lanes are still active.
    debug_assert!(batch.len <= 8);
    let mut lane_mask: u8 = ((1u16 << batch.len) - 1) as u8;

    // Fast filter (disk bound): cheaply prove some lanes are all-inside (unchanged),
    // or (rarely) prove the polygon is entirely outside a lane (empty).
    //
    // Mirrors `clip_convex`'s early-unchanged condition for bounded polys. For unbounded
    // polys (has bounding refs), we avoid the unchanged early-out because the current
    // polygon may be an artificial bound of an unbounded region.
    let max_r2 = poly.max_r2;
    if max_r2 > 0.0 {
        let s = batch.c + batch.eps;
        let rhs = batch.ab2 * f64x8::splat(max_r2);
        let s2 = s * s;

        let lanes_ge = s2.simd_ge(rhs).to_bitmask() as u8;
        if lanes_ge != 0 {
            // Early-empty: s <= 0 and |s| >= sqrt(ab2*max_r2) => max_dist <= -eps.
            let lanes_le0 = s.simd_le(f64x8::splat(0.0)).to_bitmask() as u8;
            let empty_lanes = lanes_ge & lanes_le0 & lane_mask;
            if empty_lanes != 0 {
                out.clear();
                return ClipResult::Changed;
            }

            // Early-unchanged: s >= 0 and s^2 >= ab2*max_r2 (bounded polys only).
            if !poly.has_bounding_ref {
                let lanes_ge0 = s.simd_ge(f64x8::splat(0.0)).to_bitmask() as u8;
                let unchanged_lanes = lanes_ge & lanes_ge0 & lane_mask;
                lane_mask &= !unchanged_lanes;
                if lane_mask == 0 {
                    return ClipResult::Unchanged;
                }
            }
        }
    }

    // Initial classification: all vertices against all half-planes.
    // (We still compute all 8 lanes, but lane_mask controls which lanes matter.)
    if n <= 12 {
        // Small-N fast path: avoid heavy per-vertex mask bookkeeping; just re-classify
        // the current polygon against remaining lanes each time.
        let mut cur_slot: u8 = 0; // 0=input `poly`, 1=`out`, 2=`scratch`
        let mut any_changed = false;

        while lane_mask != 0 {
            let (cur_n, lane_vertex_masks) = match cur_slot {
                0 => (poly.len, classify_lanes_only(poly, batch)),
                1 => (out.len, classify_lanes_only(out, batch)),
                2 => (scratch.len, classify_lanes_only(scratch, batch)),
                _ => unreachable!(),
            };
            let full_mask = full_vertex_mask(cur_n);

            let mut did_mixed = false;
            for lane in 0..8 {
                let lane_bit = 1u8 << lane;
                if (lane_mask & lane_bit) == 0 {
                    continue;
                }

                let inside_mask = lane_vertex_masks[lane] & full_mask;
                if inside_mask == 0 {
                    out.clear();
                    return ClipResult::Changed;
                }
                if inside_mask == full_mask {
                    lane_mask &= !lane_bit;
                    continue;
                }

                // Mixed: clip and restart (masks are now stale).
                let hp = HalfPlane {
                    a: batch.a_arr[lane],
                    b: batch.b_arr[lane],
                    c: batch.c_arr[lane],
                    ab2: batch.ab2_arr[lane],
                    eps: batch.eps_arr[lane],
                    plane_idx: batch.plane_idxs[lane],
                };

                match cur_slot {
                    0 => {
                        clip_one_lane_mixed_no_masks(poly, &hp, inside_mask, out);
                        cur_slot = 1;
                        if out.len == 0 {
                            out.clear();
                            return ClipResult::Changed;
                        }
                    }
                    1 => {
                        clip_one_lane_mixed_no_masks(out, &hp, inside_mask, scratch);
                        cur_slot = 2;
                        if scratch.len == 0 {
                            out.clear();
                            return ClipResult::Changed;
                        }
                    }
                    2 => {
                        clip_one_lane_mixed_no_masks(scratch, &hp, inside_mask, out);
                        cur_slot = 1;
                        if out.len == 0 {
                            out.clear();
                            return ClipResult::Changed;
                        }
                    }
                    _ => unreachable!(),
                }

                any_changed = true;
                lane_mask &= !lane_bit;
                did_mixed = true;
                break;
            }

            if !did_mixed {
                break;
            }
        }

        if !any_changed {
            return ClipResult::Unchanged;
        }
        if cur_slot == 2 {
            std::mem::swap(out, scratch);
        }
        return ClipResult::Changed;
    }

    let (mut current_masks, mut lane_vertex_masks) = classify_batch_with_lane_masks(poly, batch);
    let mut next_masks: VertexMasks = [0u16; MAX_POLY_VERTICES];
    let mut next_lane_vertex_masks: [u64; 8] = [0u64; 8];

    // Ping-pong between `out` and `scratch` for intermediate polygons.
    // 0 = input `poly`, 1 = `out`, 2 = `scratch`.
    let mut cur_slot: u8 = 0;

    let mut any_changed = false;

    for lane in 0..8 {
        if (lane_mask & (1 << lane)) == 0 {
            continue;
        }

        let n = match cur_slot {
            0 => poly.len,
            1 => out.len,
            2 => scratch.len,
            _ => unreachable!(),
        };
        let full_mask = full_vertex_mask(n);
        let inside_mask = lane_vertex_masks[lane] & full_mask;

        if inside_mask == 0 {
            out.clear();
            return ClipResult::Changed;
        }

        if inside_mask == full_mask {
            lane_mask &= !(1 << lane);
            if lane_mask == 0 && !any_changed {
                return ClipResult::Unchanged;
            }
            continue;
        }

        let inside_count = inside_mask.count_ones() as usize;
        if inside_count + 2 > MAX_POLY_VERTICES {
            return ClipResult::TooManyVertices;
        }

        // Mixed case: clip and update masks by reclassifying only the two new vertices.
        let rem_lane_mask = lane_mask & !(1 << lane);
        let plane_idx = batch.plane_idxs[lane];

        let hp = HalfPlane {
            a: batch.a_arr[lane],
            b: batch.b_arr[lane],
            c: batch.c_arr[lane],
            eps: batch.eps_arr[lane],
            plane_idx,
            ab2: batch.ab2_arr[lane],
        };

        match cur_slot {
            0 => {
                clip_one_lane_mixed(
                    poly,
                    &current_masks,
                    batch,
                    &hp,
                    rem_lane_mask,
                    inside_mask,
                    out,
                    &mut next_masks,
                    &mut next_lane_vertex_masks,
                );
                cur_slot = 1;
            }
            1 => {
                clip_one_lane_mixed(
                    out,
                    &current_masks,
                    batch,
                    &hp,
                    rem_lane_mask,
                    inside_mask,
                    scratch,
                    &mut next_masks,
                    &mut next_lane_vertex_masks,
                );
                cur_slot = 2;
            }
            2 => {
                clip_one_lane_mixed(
                    scratch,
                    &current_masks,
                    batch,
                    &hp,
                    rem_lane_mask,
                    inside_mask,
                    out,
                    &mut next_masks,
                    &mut next_lane_vertex_masks,
                );
                cur_slot = 1;
            }
            _ => unreachable!(),
        }

        any_changed = true;
        lane_mask &= !(1 << lane);

        let new_len = match cur_slot {
            1 => out.len,
            2 => scratch.len,
            _ => unreachable!(),
        };
        if new_len == 0 {
            out.clear();
            return ClipResult::Changed;
        }

        // Post-clip lane filtering using the freshly-built lane masks.
        // This avoids scanning `current_masks` again for the remaining lanes.
        let new_full_mask = full_vertex_mask(new_len);
        for rem_lane in 0..8 {
            if (lane_mask & (1 << rem_lane)) == 0 {
                continue;
            }
            let m = next_lane_vertex_masks[rem_lane] & new_full_mask;
            if m == 0 {
                out.clear();
                return ClipResult::Changed;
            }
            if m == new_full_mask {
                lane_mask &= !(1 << rem_lane);
            }
        }

        std::mem::swap(&mut current_masks, &mut next_masks);
        lane_vertex_masks = next_lane_vertex_masks;

        if lane_mask == 0 {
            break;
        }
    }

    if !any_changed {
        return ClipResult::Unchanged;
    }

    if cur_slot == 2 {
        std::mem::swap(out, scratch);
    }
    ClipResult::Changed
}

#[inline]
fn clip_one_lane_mixed(
    poly: &PolyBuffer,
    poly_masks: &VertexMasks,
    batch: &HpBatch8,
    hp: &HalfPlane,
    rem_lane_mask: u8,
    inside_mask: u64,
    out: &mut PolyBuffer,
    out_masks: &mut VertexMasks,
    out_lane_masks: &mut [u64; 8],
) {
    out_lane_masks.fill(0);
    let n = poly.len;
    let full_mask = full_vertex_mask(n);

    debug_assert!(inside_mask != 0);
    debug_assert!(inside_mask != full_mask);

    let prev_mask = (inside_mask << 1) | (inside_mask >> (n - 1));
    let trans_mask = (inside_mask ^ prev_mask) & full_mask;

    // In the common convex case there are exactly 2 transitions.
    // If not, fall back to a simple scan (avoids panics on degenerate inputs).
    let (entry_next, exit_next) = if trans_mask.count_ones() == 2 {
        let idx1 = trans_mask.trailing_zeros() as usize;
        let trans_mask_rem = trans_mask & !(1u64 << idx1);
        let idx2 = trans_mask_rem.trailing_zeros() as usize;

        if (inside_mask & (1u64 << idx1)) != 0 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        }
    } else {
        let mut entry_next = None;
        let mut exit_next = None;
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let prev_inside = (inside_mask >> prev) & 1 != 0;
            let cur_inside = (inside_mask >> i) & 1 != 0;
            if entry_next.is_none() && !prev_inside && cur_inside {
                entry_next = Some(i);
            } else if exit_next.is_none() && prev_inside && !cur_inside {
                exit_next = Some(i);
            }
            if entry_next.is_some() && exit_next.is_some() {
                break;
            }
        }
        (entry_next.expect("entry transition"), exit_next.expect("exit transition"))
    };

    let entry_idx = if entry_next == 0 { n - 1 } else { entry_next - 1 };
    let exit_idx = if exit_next == 0 { n - 1 } else { exit_next - 1 };

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    #[inline(always)]
    fn signed_dist(hp: &HalfPlane, u: f64, v: f64) -> f64 {
        hp.a.mul_add(u, hp.b.mul_add(v, hp.c))
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr, $mask:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            let mask = $mask & (rem_lane_mask as u16);
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
                *out_masks.get_unchecked_mut(out_len) = mask;
            }
            let bit = 1u64 << out_len;
            if (mask & (1 << 0)) != 0 {
                out_lane_masks[0] |= bit;
            }
            if (mask & (1 << 1)) != 0 {
                out_lane_masks[1] |= bit;
            }
            if (mask & (1 << 2)) != 0 {
                out_lane_masks[2] |= bit;
            }
            if (mask & (1 << 3)) != 0 {
                out_lane_masks[3] |= bit;
            }
            if (mask & (1 << 4)) != 0 {
                out_lane_masks[4] |= bit;
            }
            if (mask & (1 << 5)) != 0 {
                out_lane_masks[5] |= bit;
            }
            if (mask & (1 << 6)) != 0 {
                out_lane_masks[6] |= bit;
            }
            if (mask & (1 << 7)) != 0 {
                out_lane_masks[7] |= bit;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection.
    let d_entry = signed_dist(hp, poly.us[entry_idx], poly.vs[entry_idx]);
    let d_entry_next = signed_dist(hp, poly.us[entry_next], poly.vs[entry_next]);
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    let entry_mask = if rem_lane_mask != 0 {
        classify_one_vertex(entry_u, entry_v, batch, rem_lane_mask)
    } else {
        0
    };
    push_idx!(
        entry_u,
        entry_v,
        (entry_ep, hp.plane_idx),
        entry_ep,
        entry_mask
    );

    // Walk original vertices between entry and exit.
    let mut i = entry_next;
    loop {
        push_idx!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i],
            poly_masks[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % n;
    }

    // Exit intersection.
    let d_exit = signed_dist(hp, poly.us[exit_idx], poly.vs[exit_idx]);
    let d_exit_next = signed_dist(hp, poly.us[exit_next], poly.vs[exit_next]);
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    let exit_mask = if rem_lane_mask != 0 {
        classify_one_vertex(exit_u, exit_v, batch, rem_lane_mask)
    } else {
        0
    };
    push_idx!(
        exit_u,
        exit_v,
        (exit_ep, hp.plane_idx),
        hp.plane_idx,
        exit_mask
    );

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
}

#[inline]
fn clip_one_lane_mixed_no_masks(poly: &PolyBuffer, hp: &HalfPlane, inside_mask: u64, out: &mut PolyBuffer) {
    let n = poly.len;
    let full_mask = full_vertex_mask(n);

    debug_assert!(inside_mask != 0);
    debug_assert!(inside_mask != full_mask);

    let prev_mask = (inside_mask << 1) | (inside_mask >> (n - 1));
    let trans_mask = (inside_mask ^ prev_mask) & full_mask;

    // In the common convex case there are exactly 2 transitions.
    // If not, fall back to a simple scan (avoids panics on degenerate inputs).
    let (entry_next, exit_next) = if trans_mask.count_ones() == 2 {
        let idx1 = trans_mask.trailing_zeros() as usize;
        let trans_mask_rem = trans_mask & !(1u64 << idx1);
        let idx2 = trans_mask_rem.trailing_zeros() as usize;

        if (inside_mask & (1u64 << idx1)) != 0 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        }
    } else {
        let mut entry_next = None;
        let mut exit_next = None;
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let prev_inside = (inside_mask >> prev) & 1 != 0;
            let cur_inside = (inside_mask >> i) & 1 != 0;
            if entry_next.is_none() && !prev_inside && cur_inside {
                entry_next = Some(i);
            } else if exit_next.is_none() && prev_inside && !cur_inside {
                exit_next = Some(i);
            }
            if entry_next.is_some() && exit_next.is_some() {
                break;
            }
        }
        (entry_next.expect("entry transition"), exit_next.expect("exit transition"))
    };

    let entry_idx = if entry_next == 0 { n - 1 } else { entry_next - 1 };
    let exit_idx = if exit_next == 0 { n - 1 } else { exit_next - 1 };

    #[inline(always)]
    fn r2_of(u: f64, v: f64) -> f64 {
        u.mul_add(u, v * v)
    }

    #[inline(always)]
    fn signed_dist(hp: &HalfPlane, u: f64, v: f64) -> f64 {
        hp.a.mul_add(u, hp.b.mul_add(v, hp.c))
    }

    let mut out_len: usize = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push_idx {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            let ep = $ep;
            unsafe {
                *out.us.get_unchecked_mut(out_len) = u;
                *out.vs.get_unchecked_mut(out_len) = v;
                *out.vertex_planes.get_unchecked_mut(out_len) = vp;
                *out.edge_planes.get_unchecked_mut(out_len) = ep;
            }
            out_len += 1;
            let r2 = r2_of(u, v);
            if r2 > max_r2 {
                max_r2 = r2;
            }
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    // Entry intersection.
    let d_entry = signed_dist(hp, poly.us[entry_idx], poly.vs[entry_idx]);
    let d_entry_next = signed_dist(hp, poly.us[entry_next], poly.vs[entry_next]);
    let t_entry = d_entry / (d_entry - d_entry_next);
    let entry_u = t_entry.mul_add(poly.us[entry_next] - poly.us[entry_idx], poly.us[entry_idx]);
    let entry_v = t_entry.mul_add(poly.vs[entry_next] - poly.vs[entry_idx], poly.vs[entry_idx]);
    let entry_ep = poly.edge_planes[entry_idx];
    push_idx!(entry_u, entry_v, (entry_ep, hp.plane_idx), entry_ep);

    // Walk original vertices between entry and exit.
    let mut i = entry_next;
    loop {
        push_idx!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % n;
    }

    // Exit intersection.
    let d_exit = signed_dist(hp, poly.us[exit_idx], poly.vs[exit_idx]);
    let d_exit_next = signed_dist(hp, poly.us[exit_next], poly.vs[exit_next]);
    let t_exit = d_exit / (d_exit - d_exit_next);
    let exit_u = t_exit.mul_add(poly.us[exit_next] - poly.us[exit_idx], poly.us[exit_idx]);
    let exit_v = t_exit.mul_add(poly.vs[exit_next] - poly.vs[exit_idx], poly.vs[exit_idx]);
    let exit_ep = poly.edge_planes[exit_idx];
    push_idx!(exit_u, exit_v, (exit_ep, hp.plane_idx), hp.plane_idx);

    out.len = out_len;
    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::clip_convex;

    #[test]
    fn test_hp_batch8_creation() {
        let hps = [
            HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0),
            HalfPlane::new_unnormalized(0.0, 1.0, 0.0, 1),
            HalfPlane::new_unnormalized(-1.0, 0.0, 0.0, 2),
            HalfPlane::new_unnormalized(0.0, -1.0, 0.0, 3),
            HalfPlane::new_unnormalized(1.0, 1.0, 0.0, 4),
            HalfPlane::new_unnormalized(-1.0, 1.0, 0.0, 5),
            HalfPlane::new_unnormalized(1.0, -1.0, 0.0, 6),
            HalfPlane::new_unnormalized(-1.0, -1.0, 0.0, 7),
        ];

        let batch = HpBatch8::from_array(hps);

        assert_eq!(batch.len, 8);
        assert_eq!(batch.plane_idxs, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_classify_batch() {
        // Create a simple square polygon centered at origin
        let mut poly = PolyBuffer::new();
        poly.len = 4;
        poly.us[0] = 0.5;
        poly.vs[0] = 0.5;
        poly.us[1] = -0.5;
        poly.vs[1] = 0.5;
        poly.us[2] = -0.5;
        poly.vs[2] = -0.5;
        poly.us[3] = 0.5;
        poly.vs[3] = -0.5;

        // Half-planes that divide the space
        let hps = [
            HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 0), // x >= 0
            HalfPlane::new_unnormalized(0.0, 1.0, 0.0, 1), // y >= 0
            HalfPlane::new_unnormalized(-1.0, 0.0, 0.0, 2), // -x >= 0 (x <= 0)
            HalfPlane::new_unnormalized(0.0, -1.0, 0.0, 3), // -y >= 0 (y <= 0)
            HalfPlane::new_unnormalized(1.0, 1.0, 0.0, 4),
            HalfPlane::new_unnormalized(-1.0, 1.0, 0.0, 5),
            HalfPlane::new_unnormalized(1.0, -1.0, 0.0, 6),
            HalfPlane::new_unnormalized(-1.0, -1.0, 0.0, 7),
        ];

        let batch = HpBatch8::from_array(hps);
        let masks = classify_batch(&poly, &batch);

        // Vertex 0: (0.5, 0.5) is inside x>=0 and y>=0, outside x<=0 and y<=0.
        assert_eq!(masks[0] & 1, 1); // x >= 0
        assert_eq!(masks[0] & 2, 2); // y >= 0
        assert_eq!(masks[0] & 4, 0); // x <= 0
        assert_eq!(masks[0] & 8, 0); // y <= 0
    }

    #[test]
    fn test_clip_batch_matches_sequential() {
        let mut poly = PolyBuffer::new();
        poly.len = 4;
        poly.us[0] = 0.5;
        poly.vs[0] = 0.5;
        poly.us[1] = -0.5;
        poly.vs[1] = 0.5;
        poly.us[2] = -0.5;
        poly.vs[2] = -0.5;
        poly.us[3] = 0.5;
        poly.vs[3] = -0.5;

        let hps = [
            HalfPlane::new_unnormalized(1.0, 0.0, 0.0, 10), // x >= 0
            HalfPlane::new_unnormalized(0.0, 1.0, 0.0, 11), // y >= 0
            HalfPlane::new_unnormalized(-1.0, 0.0, 0.5, 12), // x <= 0.5
            HalfPlane::new_unnormalized(0.0, -1.0, 0.5, 13), // y <= 0.5
            HalfPlane::new_unnormalized(1.0, 1.0, 0.0, 14),
            HalfPlane::new_unnormalized(-1.0, 1.0, 0.5, 15),
            HalfPlane::new_unnormalized(1.0, -1.0, 0.5, 16),
            HalfPlane::new_unnormalized(-1.0, -1.0, 0.5, 17),
        ];

        // Sequential application.
        let mut cur = poly.clone();
        let mut tmp = PolyBuffer::new();
        for hp in hps {
            match clip_convex(&cur, &hp, &mut tmp) {
                ClipResult::Unchanged => {}
                ClipResult::Changed => std::mem::swap(&mut cur, &mut tmp),
                ClipResult::TooManyVertices => panic!("unexpected TooManyVertices in test"),
            }
            if cur.len == 0 {
                break;
            }
        }

        // Batch application.
        let batch = HpBatch8::from_array(hps);
        let mut out = PolyBuffer::new();
        let mut scratch = PolyBuffer::new();
        let r = clip_batch_with_scratch(&poly, &batch, &mut out, &mut scratch);

        match r {
            ClipResult::Unchanged => {
                // Contract: `out` is not written. Ensure sequential also made no changes.
                assert_eq!(cur.len, poly.len);
                for i in 0..cur.len {
                    let du = (cur.us[i] - poly.us[i]).abs();
                    let dv = (cur.vs[i] - poly.vs[i]).abs();
                    assert!(
                        du < 1e-12,
                        "sequential u mismatch at {i}: {} vs {}",
                        cur.us[i],
                        poly.us[i]
                    );
                    assert!(
                        dv < 1e-12,
                        "sequential v mismatch at {i}: {} vs {}",
                        cur.vs[i],
                        poly.vs[i]
                    );
                }
            }
            ClipResult::Changed => {
                if cur.len == 0 {
                    assert_eq!(out.len, 0);
                    return;
                }

                assert_eq!(out.len, cur.len);
                for i in 0..cur.len {
                    let du = (out.us[i] - cur.us[i]).abs();
                    let dv = (out.vs[i] - cur.vs[i]).abs();
                    assert!(
                        du < 1e-12,
                        "u mismatch at {i}: {} vs {}",
                        out.us[i],
                        cur.us[i]
                    );
                    assert!(
                        dv < 1e-12,
                        "v mismatch at {i}: {} vs {}",
                        out.vs[i],
                        cur.vs[i]
                    );
                }
            }
            ClipResult::TooManyVertices => panic!("unexpected TooManyVertices in test"),
        }
    }
}
