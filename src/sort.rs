//! Sorting utilities and building blocks.
//!
//! This module provides low-level sorting primitives for building custom
//! sorting algorithms, particularly optimized for small value ranges (0-32).

use std::hint::select_unpredictable;
use std::mem::MaybeUninit;
use std::ptr;

use crate::generated::sort_nets::{sort16_tail_out, sort8_net};
use crate::generated::sort_nets::{sort16_tail_out_12_4, sort24_tail_out, sort24_tail_out_20_4};

#[inline(always)]
fn cswap_unpredictable_u64(v: &mut [u64], i: usize, j: usize) {
    debug_assert!(i != j);
    debug_assert!(i < v.len());
    debug_assert!(j < v.len());

    // Use raw pointers to avoid bounds checks in this hot path.
    unsafe {
        let pi = v.as_mut_ptr().add(i);
        let pj = v.as_mut_ptr().add(j);
        let va = *pi;
        let vb = *pj;
        let cond = va <= vb;
        *pi = select_unpredictable(cond, va, vb);
        *pj = select_unpredictable(cond, vb, va);
    }
}

#[inline(always)]
unsafe fn merge_up_u64(
    mut left_src: *const u64,
    mut right_src: *const u64,
    mut dst: *mut u64,
) -> (*const u64, *const u64, *mut u64) {
    let left_val = ptr::read(left_src);
    let right_val = ptr::read(right_src);
    let is_l = left_val <= right_val;
    let val = select_unpredictable(is_l, left_val, right_val);
    ptr::write(dst, val);
    right_src = right_src.add((!is_l) as usize);
    left_src = left_src.add(is_l as usize);
    dst = dst.add(1);
    (left_src, right_src, dst)
}

#[inline(always)]
unsafe fn merge_down_u64(
    mut left_src: *const u64,
    mut right_src: *const u64,
    mut dst: *mut u64,
) -> (*const u64, *const u64, *mut u64) {
    let left_val = ptr::read(left_src);
    let right_val = ptr::read(right_src);
    let is_l = left_val <= right_val;
    let val = select_unpredictable(is_l, right_val, left_val);
    ptr::write(dst, val);
    right_src = right_src.wrapping_sub(is_l as usize);
    left_src = left_src.wrapping_sub((!is_l) as usize);
    dst = dst.sub(1);
    (left_src, right_src, dst)
}

#[inline(always)]
unsafe fn bidirectional_merge_u64(v: *const u64, len: usize, dst: *mut u64) {
    debug_assert!(len >= 2);

    let len_div_2 = len / 2;
    debug_assert!(len_div_2 != 0);

    let mut left = v;
    let mut right = v.add(len_div_2);
    let mut out = dst;

    let mut left_rev = v.add(len_div_2 - 1);
    let mut right_rev = v.add(len - 1);
    let mut out_rev = dst.add(len - 1);

    for _ in 0..len_div_2 {
        (left, right, out) = merge_up_u64(left, right, out);
        (left_rev, right_rev, out_rev) = merge_down_u64(left_rev, right_rev, out_rev);
    }

    if (len & 1) != 0 {
        let left_end = left_rev.wrapping_add(1);
        let left_nonempty = left < left_end;
        let last_src = if left_nonempty { left } else { right };
        ptr::copy_nonoverlapping(last_src, out, 1);
    }
}

#[cfg(test)]
const SENTINEL: u64 = u64::MAX;

/// Sort a small slice (N <= 35) using sorting networks + insertion sort.
///
/// Strategy (where `rem = N & 7`):
/// - `N < 8`: fallback to `sort_unstable`
/// - `rem == 0`: sort with the lower network
/// - `rem <= 2`: sort with the lower network, then insertion-insert the suffix
/// - `rem == 3`: pad up to the higher network (when possible)
/// - `rem >= 4`: pad to the higher network by placing `u64::MAX` sentinels in registers
///
/// Requirements:
/// - `N <= 35` (larger sizes fall back to `sort_unstable`)
/// - `u64::MAX` must not appear in the input (used as the sentinel for padding)
// Public here only so the `microbench` feature can re-export it from the crate
// root; the normal library surface does not expose the private `sort` module.
#[allow(unreachable_pub)]
pub fn sort_small(v: &mut [u64]) {
    let n = v.len();
    if n < 8 {
        v.sort_unstable();
        return;
    }

    if n > 35 {
        v.sort_unstable();
        return;
    }

    let rem = n & 7; // 0..=7
    let down = n & !7;

    // Note: `u64::MAX` is our padding sentinel; callers must avoid it.
    // (Checked in tests; not asserted here to keep this hot path lean.)

    unsafe {
        let base = v.as_mut_ptr();

        // Use the 4-tail-reg hybrids for all 16/24 sorting to reduce live registers/spills.
        //
        // Key detail: for pad-up, the tail starts at 12 (for 16) and 20 (for 24), so `tail_len`
        // must be computed relative to those cut points to avoid out-of-bounds reads.
        if rem == 0 {
            match n {
                8 => sort8_in_place(base),
                16 => sort16_tail_out_12_4(base, base.add(12), 4),
                24 => sort24_tail_out_20_4(base, base.add(20), 4),
                32 => sort32_maybe_padded(base, 32),
                _ => unreachable!("unexpected N (must be 8,16,24,32): {n}"),
            }
            return;
        }

        if rem <= 3 {
            // Canonical schedule tweak: insertion of 3 values is noticeably slower than either
            // insertion of 1/2 or just padding up to the next network. Prefer padding for rem=3
            // whenever we have a suitable pad-up implementation.
            if rem == 3 {
                match down {
                    // 9..=11: pad to 16 (tail starts at 8).
                    8 => {
                        sort16_tail_out(base, base.add(8), 3);
                        return;
                    }
                    // 17..=19: pad to 24 (tail starts at 16).
                    16 => {
                        sort24_tail_out(base, base.add(16), 3);
                        return;
                    }
                    // 25..=27: pad to 32 (two padded-to-16 runs + merge).
                    24 => {
                        sort32_maybe_padded(base, n);
                        return;
                    }
                    // 35 = 32 + 3: no 40-network; fall back to insertion.
                    32 => {}
                    _ => unreachable!("unexpected down (must be 8,16,24,32): {down}"),
                }
            }

            match down {
                8 => sort8_in_place(base),
                16 => sort16_tail_out_12_4(base, base.add(12), 4),
                24 => sort24_tail_out_20_4(base, base.add(20), 4),
                32 => sort32_maybe_padded(base, 32),
                _ => unreachable!("unexpected down (must be 8,16,24,32): {down}"),
            }
            insert_suffix(v, down, rem);
            return;
        }

        // rem >= 4: pad up to the higher network by placing `u64::MAX` sentinels in tail regs.
        let up = down + 8;
        match up {
            16 => sort16_tail_out_12_4(base, base.add(12), n - 12), // n in 12..=15 => 0..=3
            24 => sort24_tail_out_20_4(base, base.add(20), n - 20), // n in 20..=23 => 0..=3
            32 => sort32_maybe_padded(base, n),                     // n in 28..=31
            _ => unreachable!("unexpected up (must be 16,24,32): {up}"),
        }
    }
}

/// Insert `rem` suffix elements into the sorted network result.
#[inline(always)]
unsafe fn insert_suffix(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));
    // Assumes v[..base] is sorted ascending.
    // After each iteration, v[..base+i+1] remains sorted.

    let p = v.as_mut_ptr();

    // rem=1: classic insertion is best (fast path hits often).
    if rem == 1 {
        let idx = base;
        debug_assert!(idx > 0);

        let tail = p.add(idx);
        let mut sift = tail.sub(1);

        if *tail >= *sift {
            return;
        }

        let tmp = *tail;
        loop {
            ptr::copy_nonoverlapping(sift, sift.add(1), 1);

            if sift == p {
                *p = tmp;
                return;
            }

            sift = sift.sub(1);
            if tmp >= *sift {
                *sift.add(1) = tmp;
                return;
            }
        }
    }

    // Sort the suffix first (branchless), so insertion happens in ascending order.
    // This reduces redundant shifting, especially for rem=3.
    if rem >= 2 {
        let suffix = std::slice::from_raw_parts_mut(p.add(base), rem);
        cswap_unpredictable_u64(suffix, 0, 1);
        if rem == 3 {
            cswap_unpredictable_u64(suffix, 1, 2);
            cswap_unpredictable_u64(suffix, 0, 1);
        }
    }

    // rem=2/3: merge the tiny sorted suffix into the sorted prefix from the back.
    // This moves each prefix element at most once (vs shifting the prefix up to `rem` times).
    merge_sorted_suffix_back(p, base, rem);
}

#[inline(always)]
unsafe fn merge_sorted_suffix_back(p: *mut u64, base: usize, rem: usize) {
    debug_assert!(base > 0);
    debug_assert!((2..=3).contains(&rem));

    // Load suffix into registers first (merge writes into the suffix area).
    let r0 = *p.add(base);
    let r1 = *p.add(base + 1);
    let mut r2 = 0u64;
    if rem == 3 {
        r2 = *p.add(base + 2);
    }

    let mut right_idx: isize = rem as isize - 1;
    let mut left_idx: isize = base as isize - 1;
    let mut out: isize = (base + rem - 1) as isize;

    while right_idx >= 0 {
        let rv = match right_idx {
            0 => r0,
            1 => r1,
            2 => r2,
            _ => unreachable!(),
        };

        if left_idx >= 0 {
            let lv = *p.add(left_idx as usize);
            if lv > rv {
                *p.add(out as usize) = lv;
                left_idx -= 1;
            } else {
                *p.add(out as usize) = rv;
                right_idx -= 1;
            }
        } else {
            *p.add(out as usize) = rv;
            right_idx -= 1;
        }

        out -= 1;
    }
}

// On AVX2, inlining this leaf twice makes the general dispatcher preserve two
// extra callee-saved registers on every sort. Portable codegen retains the
// inline form; see the measured boundary in docs/performance.md.
#[cfg_attr(target_feature = "avx2", inline(never))]
#[cfg_attr(not(target_feature = "avx2"), inline(always))]
unsafe fn sort8_in_place(base: *mut u64) {
    let out = sort8_net(
        *base.add(0),
        *base.add(1),
        *base.add(2),
        *base.add(3),
        *base.add(4),
        *base.add(5),
        *base.add(6),
        *base.add(7),
    );
    ptr::copy_nonoverlapping(out.as_ptr(), base, 8);
}

/// Sort N padded to 32, where `n` is in 28..=32.
///
/// For `n < 32`, padding is done by placing `SENTINEL` values in registers (via a
/// temporary upper half), never by writing sentinels into `v` beyond `n`.
unsafe fn sort32_maybe_padded(base: *mut u64, n: usize) {
    debug_assert!((16..=32).contains(&n));

    // Sort both halves (len/2 and len-len/2) using sort16_tail_out padded to 16.
    //
    // This lines up with std's smallsort shape (sort two runs, then one merge) while
    // avoiding a dedicated sort32 network and avoiding sorting padded sentinels.
    let mid = n / 2; // 8..=16
    debug_assert!((8..=16).contains(&mid));
    let left_len = mid;
    let right_len = n - mid; // 8..=16
    debug_assert!((8..=16).contains(&right_len));

    sort16_tail_out(base, base.add(8), left_len - 8);
    sort16_tail_out(base.add(mid), base.add(mid + 8), right_len - 8);

    // Merge (bidirectional) into a stack scratch buffer and copy back.
    //
    // `bidirectional_merge_u64` always writes exactly `n` outputs, so we can avoid zeroing the
    // whole 32-element buffer by using `MaybeUninit`.
    let mut tmp = MaybeUninit::<[u64; 32]>::uninit();
    let tmp_ptr = tmp.as_mut_ptr() as *mut u64;
    bidirectional_merge_u64(base, n, tmp_ptr);
    copy_back_u64_16_to_32(base, tmp_ptr, n);
}

#[inline(always)]
unsafe fn copy_back_u64_16_to_32(dst: *mut u64, src: *const u64, n: usize) {
    debug_assert!((16..=32).contains(&n));
    // Prefer constant-sized copies to encourage inlining vs a call to `memcpy`.
    match n {
        16 => ptr::copy_nonoverlapping(src, dst, 16),
        17 => ptr::copy_nonoverlapping(src, dst, 17),
        18 => ptr::copy_nonoverlapping(src, dst, 18),
        19 => ptr::copy_nonoverlapping(src, dst, 19),
        20 => ptr::copy_nonoverlapping(src, dst, 20),
        21 => ptr::copy_nonoverlapping(src, dst, 21),
        22 => ptr::copy_nonoverlapping(src, dst, 22),
        23 => ptr::copy_nonoverlapping(src, dst, 23),
        24 => ptr::copy_nonoverlapping(src, dst, 24),
        25 => ptr::copy_nonoverlapping(src, dst, 25),
        26 => ptr::copy_nonoverlapping(src, dst, 26),
        27 => ptr::copy_nonoverlapping(src, dst, 27),
        28 => ptr::copy_nonoverlapping(src, dst, 28),
        29 => ptr::copy_nonoverlapping(src, dst, 29),
        30 => ptr::copy_nonoverlapping(src, dst, 30),
        31 => ptr::copy_nonoverlapping(src, dst, 31),
        32 => ptr::copy_nonoverlapping(src, dst, 32),
        _ => unreachable!("n must be 16..=32 (got {n})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_small_8() {
        let mut v = vec![8u64, 7, 6, 5, 4, 3, 2, 1];
        sort_small(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_sort_small_16() {
        let mut v = vec![16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        sort_small(&mut v);
        assert_eq!(
            v,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn test_sort_small_24() {
        let mut v = vec![
            24u64, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
            1,
        ];
        sort_small(&mut v);
        assert_eq!(
            v,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24
            ]
        );
    }

    #[test]
    fn test_sort_small_25() {
        // 25 = 24 + 1 (rem=1, round down)
        let mut v = vec![
            25u64, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
            2, 1,
        ];
        sort_small(&mut v);
        assert_eq!(
            v,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25
            ]
        );
    }

    #[test]
    fn test_sort_small_26() {
        // 26 = 24 + 2 (rem=2, round down)
        let mut v = vec![
            26u64, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
            4, 3, 2, 1,
        ];
        sort_small(&mut v);
        assert_eq!(
            v,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26
            ]
        );
    }

    #[test]
    fn test_sort_small_padding_to_16() {
        for n in 9..=15 {
            let mut v: Vec<u64> = (0..n as u64).map(|x| (x * 7919 + 123) % 10_000).collect();
            v.reverse();
            let mut expected = v.clone();
            expected.sort_unstable();
            sort_small(&mut v);
            assert_eq!(v, expected, "failed for n={n}");
        }
    }

    #[test]
    fn test_sort_small_padding_to_24() {
        for n in 17..=23 {
            let mut v: Vec<u64> = (0..n as u64)
                .map(|x| (x * 104729 + 17) % 1_000_000)
                .collect();
            v.reverse();
            let mut expected = v.clone();
            expected.sort_unstable();
            sort_small(&mut v);
            assert_eq!(v, expected, "failed for n={n}");
        }
    }

    #[test]
    fn test_sort_small_padding_to_32() {
        for n in 27..=31 {
            let mut v: Vec<u64> = (0..n as u64)
                .map(|x| (x * 31_557 + 7) % 1_000_000)
                .collect();
            v.reverse();
            let mut expected = v.clone();
            expected.sort_unstable();
            sort_small(&mut v);
            assert_eq!(v, expected, "failed for n={n}");
        }
    }

    #[test]
    fn test_sort_small_32_to_35() {
        for n in 32..=35 {
            let mut v: Vec<u64> = (0..n as u64)
                .map(|x| (x * 2654435761u64 + 101) % 1_000_000_000)
                .collect();
            v.reverse();
            let mut expected = v.clone();
            expected.sort_unstable();
            sort_small(&mut v);
            assert_eq!(v, expected, "failed for n={n}");
        }
    }

    #[test]
    fn test_sort_small_matches_std_for_small_n() {
        // Spot-check a bunch of sizes/seeds against std for confidence.
        for n in 0..=35 {
            for seed in 0..20u64 {
                let mut v: Vec<u64> = (0..n as u64)
                    .map(|i| (i.wrapping_mul(6364136223846793005).wrapping_add(seed)) % 1_000_000)
                    .collect();
                v.reverse();

                debug_assert!(v.iter().all(|&x| x != SENTINEL));

                let mut expected = v.clone();
                expected.sort_unstable();
                sort_small(&mut v);
                assert_eq!(v, expected, "failed for n={n} seed={seed}");
            }
        }
    }
}
