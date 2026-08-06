//! Sorting utilities and building blocks.
//!
//! This module provides low-level sorting primitives for building custom
//! sorting algorithms, particularly optimized for small value ranges (0-35).

use std::hint::select_unpredictable;
use std::mem::MaybeUninit;
use std::ptr;

use crate::generated::sort_nets::{sort16_tail_out, sort16_tail_out_12_4, sort8_net};

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

/// Sort a small slice (N <= 35) using sorting networks and short merges.
///
/// The 17..=24 range sorts a 16-key block and a 1..=8-key suffix, then
/// merges the two sorted runs backward using a fixed-size stack buffer.
/// Other supported sizes retain the 8/16 networks and the two-run 32 path.
///
/// Requirements:
/// - `N <= 35` (larger sizes fall back to `sort_unstable`)
/// - `u64::MAX` must not appear in inputs whose path pads to 16 or 32
// Public here only so the `microbench` feature can re-export it from the crate
// root; the normal library surface does not expose the private `sort` module.
#[allow(unreachable_pub)]
pub fn sort_small(v: &mut [u64]) {
    let n = v.len();
    if !(8..=35).contains(&n) {
        v.sort_unstable();
        return;
    }

    unsafe {
        let base = v.as_mut_ptr();

        match n {
            8 => sort8_in_place(base),
            9..=10 => {
                sort8_in_place(base);
                insert_suffix(v, 8, n - 8);
            }
            // Padding three values to 16 remains faster than insertion.
            11 => sort16_tail_out(base, base.add(8), 3),
            12..=15 => sort16_tail_out_12_4(base, base.add(12), n - 12),
            16 => sort16_tail_out_12_4(base, base.add(12), 4),
            17..=24 => {
                sort16_tail_out_12_4(base, base.add(12), 4);
                let suffix_len = n - 16;
                if suffix_len <= 3 {
                    insert_suffix(v, 16, suffix_len);
                } else {
                    sort_and_merge_suffix_back_8(base, 16, suffix_len);
                }
            }
            25..=32 => sort32_maybe_padded(base, n),
            33..=35 => {
                sort32_maybe_padded(base, 32);
                insert_suffix(v, 32, n - 32);
            }
            _ => unreachable!(),
        }
    }
}

/// Insert `rem` suffix elements into the sorted network result.
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

/// Sort a 4..=8 element suffix and merge it into a sorted prefix in place.
///
/// Padding the suffix in fixed storage keeps the random-key path branchless.
/// The saved copy also prevents the backward merge from overwriting unread keys.
unsafe fn sort_and_merge_suffix_back_8(p: *mut u64, base: usize, rem: usize) {
    debug_assert!(base > 0);
    debug_assert!((4..=8).contains(&rem));

    let mut right = [u64::MAX; 8];
    ptr::copy_nonoverlapping(p.add(base), right.as_mut_ptr(), rem);
    right = sort8_net(
        right[0], right[1], right[2], right[3], right[4], right[5], right[6], right[7],
    );

    let mut left_len = base;
    let mut right_len = rem;
    let mut out = base + rem;
    while right_len != 0 {
        if left_len == 0 {
            ptr::copy_nonoverlapping(right.as_ptr(), p, right_len);
            break;
        }
        let left_val = *p.add(left_len - 1);
        let right_val = right[right_len - 1];
        let take_left = left_val > right_val;
        out -= 1;
        *p.add(out) = select_unpredictable(take_left, left_val, right_val);
        left_len -= take_left as usize;
        right_len -= (!take_left) as usize;
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

/// Sort N padded to 32, where `n` is in 25..=32.
///
/// For `n < 32`, padding is done by placing `SENTINEL` values in registers (via a
/// temporary upper half), never by writing sentinels into `v` beyond `n`.
unsafe fn sort32_maybe_padded(base: *mut u64, n: usize) {
    debug_assert!((25..=32).contains(&n));

    // Sort both halves (len/2 and len-len/2) using sort16_tail_out padded to 16.
    //
    // This lines up with std's smallsort shape (sort two runs, then one merge) while
    // avoiding a dedicated sort32 network and avoiding sorting padded sentinels.
    let mid = n / 2; // 12..=16
    debug_assert!((12..=16).contains(&mid));
    let left_len = mid;
    let right_len = n - mid; // 13..=16
    debug_assert!((13..=16).contains(&right_len));

    sort16_tail_out(base, base.add(8), left_len - 8);
    sort16_tail_out(base.add(mid), base.add(mid + 8), right_len - 8);

    // Merge (bidirectional) into a stack scratch buffer and copy back.
    //
    // `bidirectional_merge_u64` always writes exactly `n` outputs, so we can avoid zeroing the
    // whole 32-element buffer by using `MaybeUninit`.
    let mut tmp = MaybeUninit::<[u64; 32]>::uninit();
    let tmp_ptr = tmp.as_mut_ptr() as *mut u64;
    bidirectional_merge_u64(base, n, tmp_ptr);
    ptr::copy_nonoverlapping(tmp_ptr, base, n);
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
    fn test_sort_small_block_merge_exhaustive_interleavings() {
        // Exhaust every possible rank interleaving of the sorted 16-key block
        // and sorted 1..=8-key suffix. Reverse both inputs so their sorting
        // paths are exercised too.
        for suffix_len in 1..=8 {
            let n = 16 + suffix_len;
            let limit = 1u32 << n;
            let mut mask = (1u32 << suffix_len) - 1;

            loop {
                let mut input = [0u64; 24];
                let mut left = 0;
                let mut right = 16;
                for rank in 0..n {
                    if (mask & (1u32 << rank)) != 0 {
                        input[right] = rank as u64;
                        right += 1;
                    } else {
                        input[left] = rank as u64;
                        left += 1;
                    }
                }
                input[..16].reverse();
                input[16..n].reverse();

                sort_small(&mut input[..n]);
                for (rank, &value) in input[..n].iter().enumerate() {
                    assert_eq!(value, rank as u64, "n={n} mask={mask:#x}");
                }

                // Gosper's hack: advance to the next mask with suffix_len bits.
                let low_bit = mask & mask.wrapping_neg();
                let ripple = mask + low_bit;
                let next = ripple | (((mask ^ ripple) >> 2) / low_bit);
                if next >= limit {
                    break;
                }
                mask = next;
            }
        }
    }

    #[test]
    fn test_sort_small_block_merge_random_duplicates_and_max() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for n in 17..=24 {
            for sample in 0..2_000 {
                let mut input = [0u64; 24];
                for (i, value) in input[..n].iter_mut().enumerate() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *value = match (sample + i) % 9 {
                        0 => 0,
                        1 => u64::MAX,
                        2..=4 => state & 7,
                        _ => state,
                    };
                }

                let mut expected = input;
                expected[..n].sort_unstable();
                sort_small(&mut input[..n]);
                assert_eq!(input[..n], expected[..n], "n={n} sample={sample}");
            }
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
