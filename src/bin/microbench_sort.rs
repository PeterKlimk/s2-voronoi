//! Microbenchmark for sorting strategies on small `u64` arrays.
//!
//! This is intentionally minimal: it compares `slice::sort_unstable()` against a pure
//! pointer-based sorting-network implementation (generated from `sorting_networks.json`).
//!
//! Run (example):
//!   cargo run --release --bin microbench_sort -- --sizes=18,20,22 --target-ms=500

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::ptr;
use std::time::Instant;

#[path = "../bin_support/sort_nets_microbench.rs"]
mod sort_nets_microbench;

#[path = "../bin_support/sort_nets_microbench_reg.rs"]
mod sort_nets_microbench_reg;

#[path = "../sort_nets.rs"]
mod sort_nets_weird;

#[derive(Debug, Clone)]
struct Config {
    sizes: Vec<usize>,
    cases: usize,
    warmup_iters: usize,
    iters: usize,
    repeats: usize,
    warmup_ms: Option<u64>,
    target_ms: Option<u64>,
    seed: u64,
    verify: bool,
    full: bool,
    sched: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sizes: (8..=32).collect(),
            cases: 2048,
            warmup_iters: 25_000,
            iters: 250_000,
            repeats: 15,
            warmup_ms: None,
            target_ms: None,
            seed: 42,
            verify: true,
            full: false,
            sched: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    median_ns: f64,
    min_ns: f64,
    max_ns: f64,
}

impl Stats {
    fn jitter_pct(self) -> f64 {
        if self.median_ns == 0.0 {
            0.0
        } else {
            100.0 * (self.max_ns - self.min_ns) / self.median_ns
        }
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = xs.len() / 2;
    if xs.len().is_multiple_of(2) {
        0.5 * (xs[mid - 1] + xs[mid])
    } else {
        xs[mid]
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    for arg in std::env::args().skip(1) {
        let mut split = arg.splitn(2, '=');
        let key = split.next().unwrap_or("");
        let val = split.next();

        match (key, val) {
            ("--sizes", Some(v)) => {
                cfg.sizes = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>().expect("invalid --sizes entry"))
                    .collect();
            }
            ("--cases", Some(v)) => cfg.cases = v.parse().expect("invalid --cases"),
            ("--warmup", Some(v)) => cfg.warmup_iters = v.parse().expect("invalid --warmup"),
            ("--iters", Some(v)) => cfg.iters = v.parse().expect("invalid --iters"),
            ("--repeats", Some(v)) => cfg.repeats = v.parse().expect("invalid --repeats"),
            ("--warmup-ms", Some(v)) => {
                cfg.warmup_ms = Some(v.parse().expect("invalid --warmup-ms"))
            }
            ("--target-ms", Some(v)) => {
                cfg.target_ms = Some(v.parse().expect("invalid --target-ms"))
            }
            ("--seed", Some(v)) => cfg.seed = v.parse().expect("invalid --seed"),
            ("--no-verify", None) => cfg.verify = false,
            ("--full", None) => cfg.full = true,
            ("--sched", None) => cfg.sched = true,
            ("--help", _) | ("-h", _) => {
                eprintln!(
                    "microbench_sort options:\n  \
--sizes=8,9,10.. (comma list)\n  \
--cases=N (default 2048)\n  \
--warmup=N (default 25000)\n  \
--iters=N (default 250000)\n  \
--repeats=N (default 15)\n  \
--warmup-ms=N (duration-based warmup)\n  \
--target-ms=N (duration-based measurement; overrides --iters)\n  \
--seed=N (default 42)\n  \
--no-verify (skip correctness checks)\n  \
--full (show all algorithms)\n  \
--sched (compare schedule variants)"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg: {arg} (use --help)");
                std::process::exit(2);
            }
        }
    }

    assert!(!cfg.sizes.is_empty(), "--sizes must be non-empty");
    assert!(cfg.cases > 0, "--cases must be > 0");
    assert!(cfg.iters > 0, "--iters must be > 0");
    assert!(cfg.repeats > 0, "--repeats must be > 0");
    if let Some(ms) = cfg.target_ms {
        assert!(ms > 0, "--target-ms must be > 0");
    }
    if let Some(ms) = cfg.warmup_ms {
        assert!(ms > 0, "--warmup-ms must be > 0");
    }
    if cfg.full && cfg.sched {
        eprintln!("choose one: --full or --sched");
        std::process::exit(2);
    }
    cfg
}

fn generate_cases(len: usize, cases: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = vec![0u64; len * cases];
    for x in &mut out {
        let mut v: u64 = rng.gen();
        // Several strategies use `u64::MAX` as a padding sentinel; avoid it.
        if v == u64::MAX {
            v = u64::MAX - 1;
        }
        *x = v;
    }
    out
}

type SortFn = fn(&mut [u64]);

fn benchmark_cases_in_place(
    cases_flat: &[u64],
    len: usize,
    f: SortFn,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> Stats {
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);

    let mut buf = vec![0u64; len];

    let warmup_iters = warmup.min(iters);
    let mut acc = 0u64;
    for i in 0..warmup_iters {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        (f)(&mut buf);
        acc ^= buf[0];
        acc ^= buf[len - 1];
    }
    black_box(acc);

    let mut samples = Vec::with_capacity(repeats);
    let mut min_ns = f64::INFINITY;
    let mut max_ns: f64 = 0.0;
    for r in 0..repeats {
        let start = Instant::now();
        let mut acc = r as u64;
        for i in 0..iters {
            let case_idx = (i + r) % num_cases;
            let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
            buf.copy_from_slice(src);
            (f)(&mut buf);
            acc ^= buf[0];
            acc ^= buf[len - 1];
        }
        black_box(acc);
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / iters as f64;
        min_ns = min_ns.min(ns);
        max_ns = max_ns.max(ns);
        samples.push(ns);
    }

    Stats {
        median_ns: median(samples),
        min_ns,
        max_ns,
    }
}

fn benchmark_cases_in_place_timed(
    cases_flat: &[u64],
    len: usize,
    f: SortFn,
    warmup_ms: u64,
    target_ms: u64,
    repeats: usize,
) -> Stats {
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);

    let mut buf = vec![0u64; len];
    let warmup_dur = std::time::Duration::from_millis(warmup_ms);
    let target_dur = std::time::Duration::from_millis(target_ms);

    let mut acc = 0u64;
    let start = Instant::now();
    let mut i = 0usize;
    while start.elapsed() < warmup_dur {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        (f)(&mut buf);
        acc ^= buf[0];
        acc ^= buf[len - 1];
        i += 1;
    }
    black_box(acc);

    let mut samples = Vec::with_capacity(repeats);
    let mut min_ns = f64::INFINITY;
    let mut max_ns: f64 = 0.0;
    for rep in 0..repeats {
        let start = Instant::now();
        let mut acc = rep as u64;
        let mut iters = 0usize;
        while start.elapsed() < target_dur {
            let case_idx = (iters + rep) % num_cases;
            let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
            buf.copy_from_slice(src);
            (f)(&mut buf);
            acc ^= buf[0];
            acc ^= buf[len - 1];
            iters += 1;
        }
        black_box(acc);
        if iters == 0 {
            iters = 1;
        }
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / iters as f64;
        min_ns = min_ns.min(ns);
        max_ns = max_ns.max(ns);
        samples.push(ns);
    }

    Stats {
        median_ns: median(samples),
        min_ns,
        max_ns,
    }
}

#[inline(never)]
fn std_sort(v: &mut [u64]) {
    v.sort_unstable();
}

#[inline(never)]
fn net_sort(v: &mut [u64]) {
    sort_nets_microbench::sort_net_or_std(v);
}

#[inline(never)]
fn net_sort_reg(v: &mut [u64]) {
    sort_nets_microbench_reg::sort_net_or_std(v);
}

#[inline(never)]
fn weird_tail_sort(v: &mut [u64]) {
    let n = v.len();
    if n < 2 {
        return;
    }
    if n < 8 {
        v.sort_unstable();
        return;
    }
    if n <= 16 {
        unsafe {
            let base = v.as_mut_ptr();
            if n >= 12 {
                sort_nets_weird::sort16_tail_out_12_4(base, base.add(12), n - 12);
            } else {
                sort_nets_weird::sort16_tail_out(base, base.add(8), n - 8);
            }
        }
        return;
    }
    if n <= 24 {
        unsafe {
            let base = v.as_mut_ptr();
            if n >= 20 {
                sort_nets_weird::sort24_tail_out_20_4(base, base.add(20), n - 20);
            } else {
                sort_nets_weird::sort24_tail_out(base, base.add(16), n - 16);
            }
        }
        return;
    }
    if n <= 32 {
        // Sort two runs (len/2 and len-len/2) using the 16-tail networks, then merge.
        let mid = n / 2; // 12..=16
        debug_assert!((12..=16).contains(&mid));
        debug_assert!((12..=16).contains(&(n - mid)));

        unsafe {
            let base = v.as_mut_ptr();

            // Left run
            let left = base;
            let left_len = mid;
            if left_len >= 12 {
                sort_nets_weird::sort16_tail_out_12_4(left, left.add(12), left_len - 12);
            } else {
                sort_nets_weird::sort16_tail_out(left, left.add(8), left_len - 8);
            }

            // Right run
            let right = base.add(mid);
            let right_len = n - mid;
            if right_len >= 12 {
                sort_nets_weird::sort16_tail_out_12_4(right, right.add(12), right_len - 12);
            } else {
                sort_nets_weird::sort16_tail_out(right, right.add(8), right_len - 8);
            }

            let mut tmp = [0u64; 32];
            bidirectional_merge_halves_u64(base, n, tmp.as_mut_ptr());
            ptr::copy_nonoverlapping(tmp.as_ptr(), base, n);
        }
        return;
    }

    v.sort_unstable();
}

#[inline(never)]
fn sort_small_current(v: &mut [u64]) {
    s2_voronoi::sort::sort_small(v);
}

#[inline(always)]
fn cswap_u64(a: &mut u64, b: &mut u64) {
    let va = *a;
    let vb = *b;
    let cond = va <= vb;
    *a = std::hint::select_unpredictable(cond, va, vb);
    *b = std::hint::select_unpredictable(cond, vb, va);
}

#[inline(always)]
unsafe fn merge_sorted_suffix_back_current(p: *mut u64, base: usize, rem: usize) {
    debug_assert!(base > 0);
    debug_assert!((2..=3).contains(&rem));

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

#[inline(always)]
unsafe fn upper_bound_u64(p: *const u64, len: usize, x: u64) -> usize {
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let v = ptr::read(p.add(mid));
        if v <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Merge a tiny sorted suffix into a sorted prefix by copying whole runs.
///
/// This is semantically equivalent to a stable merge where the prefix comes first for ties
/// (i.e. suffix values are inserted after equal prefix values).
#[inline(always)]
unsafe fn merge_sorted_suffix_back_blockcopy(p: *mut u64, base: usize, rem: usize) {
    debug_assert!(base > 0);
    debug_assert!((2..=3).contains(&rem));

    // Load suffix into registers first (merge writes into the suffix area).
    let s0 = ptr::read(p.add(base));
    let s1 = ptr::read(p.add(base + 1));
    let mut s2 = 0u64;
    if rem == 3 {
        s2 = ptr::read(p.add(base + 2));
    }

    let mut left_end = base;
    let mut out_end = base + rem;

    for k in (0..rem).rev() {
        let rv = match k {
            0 => s0,
            1 => s1,
            2 => s2,
            _ => unreachable!(),
        };

        // Find the first prefix element > rv (upper_bound), so we move the run of elements > rv
        // as a block, then place rv just before it.
        let pos = upper_bound_u64(p, left_end, rv);
        let move_count = left_end - pos;

        out_end -= move_count;
        if move_count != 0 {
            ptr::copy(p.add(pos), p.add(out_end), move_count);
        }

        out_end -= 1;
        ptr::write(p.add(out_end), rv);
        left_end = pos;
    }
}

#[inline(always)]
unsafe fn merge_sorted_suffix_back_branchless(p: *mut u64, base: usize, rem: usize) {
    debug_assert!(base > 0);
    debug_assert!((2..=3).contains(&rem));

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

        let left_nonempty = left_idx >= 0;
        let lv = if left_nonempty {
            *p.add(left_idx as usize)
        } else {
            u64::MIN
        };

        let take_left = left_nonempty && (lv > rv);
        let val = std::hint::select_unpredictable(take_left, lv, rv);
        *p.add(out as usize) = val;

        left_idx -= take_left as isize;
        right_idx -= (!take_left) as isize;
        out -= 1;
    }
}

#[inline(always)]
unsafe fn insert_suffix_with_merge(
    v: &mut [u64],
    base: usize,
    rem: usize,
    merge_sorted_suffix_back: unsafe fn(*mut u64, usize, usize),
) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();

    if rem >= 2 {
        // Sort the suffix first (branchless), so insertion happens in ascending order.
        let s0 = p.add(base);
        let s1 = p.add(base + 1);
        let mut a0 = *s0;
        let mut a1 = *s1;
        cswap_u64(&mut a0, &mut a1);
        *s0 = a0;
        *s1 = a1;
        if rem == 3 {
            let s2 = p.add(base + 2);
            let mut a2 = *s2;
            cswap_u64(&mut a1, &mut a2);
            cswap_u64(&mut a0, &mut a1);
            *s0 = a0;
            *s1 = a1;
            *s2 = a2;
        }
    }

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

    merge_sorted_suffix_back(p, base, rem);
}

#[inline(always)]
unsafe fn insert_suffix_binsearch(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((2..=3).contains(&rem));

    let p = v.as_mut_ptr();

    // Load suffix into registers first (insertion will overwrite v[base..]).
    let mut s0 = ptr::read(p.add(base));
    let mut s1 = ptr::read(p.add(base + 1));
    let mut s2 = 0u64;
    if rem == 3 {
        s2 = ptr::read(p.add(base + 2));
    }

    // Sort suffix ascending (branchless).
    cswap_u64(&mut s0, &mut s1);
    if rem == 3 {
        cswap_u64(&mut s1, &mut s2);
        cswap_u64(&mut s0, &mut s1);
    }

    let mut cur_len = base;

    // Insert in ascending order to keep the prefix sorted after each step.
    // Use `upper_bound` for stability w.r.t. equal keys (suffix after prefix).
    for k in 0..rem {
        let x = match k {
            0 => s0,
            1 => s1,
            2 => s2,
            _ => unreachable!(),
        };

        let pos = upper_bound_u64(p, cur_len, x);
        let move_count = cur_len - pos;
        if move_count != 0 {
            ptr::copy(p.add(pos), p.add(pos + 1), move_count);
        }
        ptr::write(p.add(pos), x);
        cur_len += 1;
    }
}

#[inline(always)]
unsafe fn insert_tail_u64(begin: *mut u64, tail: *mut u64) {
    debug_assert!(begin < tail);

    let mut sift = tail.sub(1);
    let tail_val = ptr::read(tail);
    if tail_val >= ptr::read(sift) {
        return;
    }

    let tmp = tail_val;
    loop {
        ptr::copy_nonoverlapping(sift, sift.add(1), 1);
        if sift == begin {
            ptr::write(begin, tmp);
            return;
        }
        sift = sift.sub(1);
        if tmp >= ptr::read(sift) {
            ptr::write(sift.add(1), tmp);
            return;
        }
    }
}

#[inline(always)]
unsafe fn insert_suffix_std_style(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));
    debug_assert!(base > 0);

    let p = v.as_mut_ptr();
    for i in base..(base + rem) {
        insert_tail_u64(p, p.add(i));
    }
}

#[inline(always)]
unsafe fn sort32_maybe_padded_local(base: *mut u64, n: usize) {
    debug_assert!((28..=32).contains(&n));
    let mid = n / 2; // 14..=16
    let left_len = mid;
    let right_len = n - mid; // 14..=16

    sort_nets_weird::sort16_tail_out(base, base.add(8), left_len - 8);
    sort_nets_weird::sort16_tail_out(base.add(mid), base.add(mid + 8), right_len - 8);

    let mut tmp = [0u64; 32];
    bidirectional_merge_halves_u64(base, n, tmp.as_mut_ptr());
    ptr::copy_nonoverlapping(tmp.as_ptr(), base, n);
}

#[inline(always)]
unsafe fn sort_down_multiple_of_8(base: *mut u64, n: usize) {
    match n {
        8 => sort_nets_microbench_reg::sort8_net(base),
        16 => sort_nets_weird::sort16_tail_out_12_4(base, base.add(12), 4),
        24 => sort_nets_weird::sort24_tail_out_20_4(base, base.add(20), 4),
        32 => sort32_maybe_padded_local(base, 32),
        _ => unreachable!("unexpected multiple-of-8 n: {n}"),
    }
}

#[inline(always)]
unsafe fn sort_pad_up_to_multiple_of_8(base: *mut u64, n: usize, up: usize) {
    debug_assert!(n < up);
    match up {
        16 => {
            // n in 9..=15; choose better specialized ABI for >=12.
            if n >= 12 {
                sort_nets_weird::sort16_tail_out_12_4(base, base.add(12), n - 12);
            } else {
                sort_nets_weird::sort16_tail_out(base, base.add(8), n - 8);
            }
        }
        24 => {
            // n in 17..=23; choose better specialized ABI for >=20.
            if n >= 20 {
                sort_nets_weird::sort24_tail_out_20_4(base, base.add(20), n - 20);
            } else {
                sort_nets_weird::sort24_tail_out(base, base.add(16), n - 16);
            }
        }
        32 => sort32_maybe_padded_local(base, n), // n in 28..=31
        _ => unreachable!("unexpected pad target: {up}"),
    }
}

/// Variant (1): for `rem <= 3`, prefer pad-up for 9..=11 and 17..=19 (avoid insert).
#[inline(never)]
fn sort_small_variant_pad_up(v: &mut [u64]) {
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

    unsafe {
        let base = v.as_mut_ptr();

        if rem == 0 {
            sort_down_multiple_of_8(base, n);
            return;
        }

        if rem <= 3 {
            // Only apply where we have a natural pad-up target implemented.
            // - 9..=11: 8+rem => pad to 16
            // - 17..=19: 16+rem => pad to 24
            //
            // Keep the `rem=1` insertion path: padding up for +1 is usually not worth it.
            if rem >= 2 && (down == 8 || down == 16) {
                sort_pad_up_to_multiple_of_8(base, n, down + 8);
                return;
            }

            sort_down_multiple_of_8(base, down);
            insert_suffix_with_merge(v, down, rem, merge_sorted_suffix_back_current);
            return;
        }

        sort_pad_up_to_multiple_of_8(base, n, down + 8);
    }
}

/// Schedule variant: keep insert/merge for rem=1/2, but pad-up for rem=3 (when possible).
///
/// This targets the "insert 3 is slow" hypothesis and fixes the `8+3` and `16+3` corners
/// (11 and 19) without changing behavior for `25/26` (insert into 24 still happens).
#[inline(never)]
fn sort_small_variant_pad_rem3(v: &mut [u64]) {
    let n = v.len();
    if n < 8 || n > 35 {
        v.sort_unstable();
        return;
    }

    let rem = n & 7;
    let down = n & !7;

    unsafe {
        let base = v.as_mut_ptr();

        if rem == 0 {
            sort_down_multiple_of_8(base, n);
            return;
        }

        if rem == 3 && down < 32 {
            sort_pad_up_to_multiple_of_8(base, n, down + 8);
            return;
        }

        if rem <= 3 {
            sort_down_multiple_of_8(base, down);
            insert_suffix_with_merge(v, down, rem, merge_sorted_suffix_back_current);
            return;
        }

        sort_pad_up_to_multiple_of_8(base, n, down + 8);
    }
}

/// Schedule variant: for 25..=31, always pad-up to 32 (avoid inserting into 24).
///
/// Everything else uses the pad-rem3 schedule above.
#[inline(never)]
fn sort_small_variant_pad_25_31_to_32(v: &mut [u64]) {
    let n = v.len();
    if n < 8 || n > 35 {
        v.sort_unstable();
        return;
    }

    unsafe {
        if (25..=31).contains(&n) {
            sort_pad_up_to_multiple_of_8(v.as_mut_ptr(), n, 32);
            return;
        }
    }

    sort_small_variant_pad_rem3(v);
}

/// Variant (5): keep the same structure, but use std-like tail insertion for rem=1/2/3.
#[inline(never)]
fn sort_small_variant_std_insert(v: &mut [u64]) {
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

    // Keep base sorting apples-to-apples with `sort_small_current` by using the library sorter
    // for the sorted prefix (and the whole slice for rem>=4). This isolates the insertion policy.
    if rem == 0 || rem >= 4 {
        sort_small_current(v);
        return;
    }

    sort_small_current(&mut v[..down]);
    unsafe { insert_suffix_std_style(v, down, rem) };
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
    let val = std::hint::select_unpredictable(is_l, left_val, right_val);
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
    let val = std::hint::select_unpredictable(is_l, right_val, left_val);
    ptr::write(dst, val);
    right_src = right_src.wrapping_sub(is_l as usize);
    left_src = left_src.wrapping_sub((!is_l) as usize);
    dst = dst.sub(1);
    (left_src, right_src, dst)
}

#[inline(always)]
unsafe fn bidirectional_merge_halves_u64(v: *const u64, len: usize, dst: *mut u64) {
    debug_assert!(len >= 2);
    debug_assert!(len <= 32);

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

#[inline(always)]
fn insert_one_sorted_prefix(buf: &mut [u64], sorted_len: usize, x: u64) {
    debug_assert!(sorted_len < buf.len());
    let mut i = sorted_len;
    while i > 0 && buf[i - 1] > x {
        buf[i] = buf[i - 1];
        i -= 1;
    }
    buf[i] = x;
}

#[inline(always)]
fn sort_run_up_to_16(v: &mut [u64]) {
    let n = v.len();
    if n < 2 {
        return;
    }
    debug_assert!(n <= 16);

    match n {
        2..=3 => {
            let mut scratch = [u64::MAX; 4];
            scratch[..n].copy_from_slice(v);
            unsafe { sort_nets_microbench_reg::sort4_net(scratch.as_mut_ptr()) };
            v.copy_from_slice(&scratch[..n]);
        }
        4 => unsafe {
            sort_nets_microbench_reg::sort4_net(v.as_mut_ptr());
        },
        5 => {
            unsafe { sort_nets_microbench_reg::sort4_net(v.as_mut_ptr()) };
            let x = v[4];
            insert_one_sorted_prefix(v, 4, x);
        }
        6..=7 => {
            let mut scratch = [u64::MAX; 8];
            scratch[..n].copy_from_slice(v);
            unsafe { sort_nets_microbench_reg::sort8_net(scratch.as_mut_ptr()) };
            v.copy_from_slice(&scratch[..n]);
        }
        8 => unsafe {
            sort_nets_microbench_reg::sort8_net(v.as_mut_ptr());
        },
        9 => {
            unsafe { sort_nets_microbench_reg::sort8_net(v.as_mut_ptr()) };
            let x = v[8];
            insert_one_sorted_prefix(v, 8, x);
        }
        10..=11 => {
            let mut scratch = [u64::MAX; 12];
            scratch[..n].copy_from_slice(v);
            unsafe { sort_nets_microbench_reg::sort12_net(scratch.as_mut_ptr()) };
            v.copy_from_slice(&scratch[..n]);
        }
        12 => unsafe {
            sort_nets_microbench_reg::sort12_net(v.as_mut_ptr());
        },
        13 => {
            unsafe { sort_nets_microbench_reg::sort12_net(v.as_mut_ptr()) };
            let x = v[12];
            insert_one_sorted_prefix(v, 12, x);
        }
        14..=15 => {
            let mut scratch = [u64::MAX; 16];
            scratch[..n].copy_from_slice(v);
            unsafe { sort_nets_microbench_reg::sort16_net(scratch.as_mut_ptr()) };
            v.copy_from_slice(&scratch[..n]);
        }
        16 => unsafe {
            sort_nets_microbench_reg::sort16_net(v.as_mut_ptr());
        },
        _ => unreachable!("n must be 2..=16 (got {n})"),
    }
}

/// Sorting scheme:
/// - For n <= 16: sort a single run using only 4/8/12/16 networks (+ insert-by-1 at 5/9/13).
/// - For 17..=32: sort two runs (len/2 and len-len/2), then merge.
fn half_scheme_sort(v: &mut [u64]) {
    let n = v.len();
    if n < 2 {
        return;
    }
    if n <= 16 {
        sort_run_up_to_16(v);
        return;
    }
    if n > 32 {
        v.sort_unstable();
        return;
    }

    let mid = n / 2; // 8..=16
    debug_assert!((8..=16).contains(&mid));
    debug_assert!((8..=16).contains(&(n - mid)));

    sort_run_up_to_16(&mut v[..mid]);
    sort_run_up_to_16(&mut v[mid..]);

    let mut tmp = [0u64; 32];
    unsafe { bidirectional_merge_halves_u64(v.as_ptr(), n, tmp.as_mut_ptr()) };
    v.copy_from_slice(&tmp[..n]);
}

fn verify_matches_std(cases_flat: &[u64], len: usize, f: SortFn) {
    let num_cases = cases_flat.len() / len;
    let mut buf = vec![0u64; len];
    let mut ref_buf = vec![0u64; len];
    for case_idx in 0..num_cases {
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        ref_buf.copy_from_slice(src);
        std_sort(&mut ref_buf);
        (f)(&mut buf);
        assert_eq!(buf, ref_buf, "mismatch at case {case_idx} (len={len})");
    }
}

fn main() {
    let cfg = parse_args();

    println!("Sort Microbench (u64)");
    if let Some(target_ms) = cfg.target_ms {
        let warmup_ms = cfg.warmup_ms.unwrap_or(200);
        println!(
            "sizes={:?} cases={} warmup_ms={} target_ms={} repeats={}",
            cfg.sizes, cfg.cases, warmup_ms, target_ms, cfg.repeats
        );
    } else {
        println!(
            "sizes={:?} cases={} warmup_iters={} iters={} repeats={}",
            cfg.sizes, cfg.cases, cfg.warmup_iters, cfg.iters, cfg.repeats
        );
    }
    println!();

    if cfg.sched {
        println!(
            "{:>4} {:>11} {:>8} {:>11} {:>8} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
            "N",
            "std med",
            "std jit",
            "sort_small",
            "jit",
            "pad_up",
            "jit",
            "pad_rem3",
            "jit",
            "pad_32",
            "jit",
        );
        println!("{:-<118}", "");
    } else if cfg.full {
        println!(
            "{:>4} {:>11} {:>8} {:>11} {:>8} {:>11} {:>8} {:>11} {:>8} {:>11} {:>8} {:>11} {:>8} {:>11} {:>8}",
            "N",
            "std med",
            "std jit",
            "net med",
            "net jit",
            "net_reg",
            "jit",
            "weird_tail",
            "jit",
            "sort_small",
            "jit",
            "pad_up",
            "jit",
            "std_ins",
            "jit",
        );
        println!("{:-<156}", "");
    } else {
        println!(
            "{:>4} {:>11} {:>8} {:>11} {:>8} {:>9}",
            "N", "std med", "std jit", "sort_small", "jit", "speedup"
        );
        println!("{:-<58}", "");
    }

    for &n in &cfg.sizes {
        let cases = generate_cases(
            n,
            cfg.cases,
            cfg.seed ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );

        if cfg.verify {
            verify_matches_std(&cases, n, sort_small_current);
            if cfg.sched {
                verify_matches_std(&cases, n, sort_small_variant_pad_up);
                verify_matches_std(&cases, n, sort_small_variant_pad_rem3);
                verify_matches_std(&cases, n, sort_small_variant_pad_25_31_to_32);
            }
            if cfg.full {
                verify_matches_std(&cases, n, net_sort);
                verify_matches_std(&cases, n, net_sort_reg);
                verify_matches_std(&cases, n, weird_tail_sort);
                verify_matches_std(&cases, n, sort_small_variant_pad_up);
                verify_matches_std(&cases, n, half_scheme_sort);
                verify_matches_std(&cases, n, sort_small_variant_std_insert);
            }
        }

        let (std_s, small_s) = if let Some(target_ms) = cfg.target_ms {
            let warmup_ms = cfg.warmup_ms.unwrap_or(200);
            (
                benchmark_cases_in_place_timed(
                    &cases,
                    n,
                    std_sort,
                    warmup_ms,
                    target_ms,
                    cfg.repeats,
                ),
                benchmark_cases_in_place_timed(
                    &cases,
                    n,
                    sort_small_current,
                    warmup_ms,
                    target_ms,
                    cfg.repeats,
                ),
            )
        } else {
            (
                benchmark_cases_in_place(
                    &cases,
                    n,
                    std_sort,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                ),
                benchmark_cases_in_place(
                    &cases,
                    n,
                    sort_small_current,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                ),
            )
        };

        if cfg.sched {
            let (pad_up_s, pad_rem3_s, pad32_s) = if let Some(target_ms) = cfg.target_ms {
                let warmup_ms = cfg.warmup_ms.unwrap_or(200);
                (
                    benchmark_cases_in_place_timed(
                        &cases,
                        n,
                        sort_small_variant_pad_up,
                        warmup_ms,
                        target_ms,
                        cfg.repeats,
                    ),
                    benchmark_cases_in_place_timed(
                        &cases,
                        n,
                        sort_small_variant_pad_rem3,
                        warmup_ms,
                        target_ms,
                        cfg.repeats,
                    ),
                    benchmark_cases_in_place_timed(
                        &cases,
                        n,
                        sort_small_variant_pad_25_31_to_32,
                        warmup_ms,
                        target_ms,
                        cfg.repeats,
                    ),
                )
            } else {
                (
                    benchmark_cases_in_place(
                        &cases,
                        n,
                        sort_small_variant_pad_up,
                        cfg.warmup_iters,
                        cfg.iters,
                        cfg.repeats,
                    ),
                    benchmark_cases_in_place(
                        &cases,
                        n,
                        sort_small_variant_pad_rem3,
                        cfg.warmup_iters,
                        cfg.iters,
                        cfg.repeats,
                    ),
                    benchmark_cases_in_place(
                        &cases,
                        n,
                        sort_small_variant_pad_25_31_to_32,
                        cfg.warmup_iters,
                        cfg.iters,
                        cfg.repeats,
                    ),
                )
            };

            println!(
                "{:>4} {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>10.1} {:>7.1}% {:>10.1} {:>7.1}% {:>10.1} {:>7.1}%",
                n,
                std_s.median_ns,
                std_s.jitter_pct(),
                small_s.median_ns,
                small_s.jitter_pct(),
                pad_up_s.median_ns,
                pad_up_s.jitter_pct(),
                pad_rem3_s.median_ns,
                pad_rem3_s.jitter_pct(),
                pad32_s.median_ns,
                pad32_s.jitter_pct(),
            );
        } else if cfg.full {
            let (net_s, net_reg_s, weird_s, pad_s, stdins_s) =
                if let Some(target_ms) = cfg.target_ms {
                    let warmup_ms = cfg.warmup_ms.unwrap_or(200);
                    (
                        benchmark_cases_in_place_timed(
                            &cases,
                            n,
                            net_sort,
                            warmup_ms,
                            target_ms,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place_timed(
                            &cases,
                            n,
                            net_sort_reg,
                            warmup_ms,
                            target_ms,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place_timed(
                            &cases,
                            n,
                            weird_tail_sort,
                            warmup_ms,
                            target_ms,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place_timed(
                            &cases,
                            n,
                            sort_small_variant_pad_up,
                            warmup_ms,
                            target_ms,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place_timed(
                            &cases,
                            n,
                            sort_small_variant_std_insert,
                            warmup_ms,
                            target_ms,
                            cfg.repeats,
                        ),
                    )
                } else {
                    (
                        benchmark_cases_in_place(
                            &cases,
                            n,
                            net_sort,
                            cfg.warmup_iters,
                            cfg.iters,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place(
                            &cases,
                            n,
                            net_sort_reg,
                            cfg.warmup_iters,
                            cfg.iters,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place(
                            &cases,
                            n,
                            weird_tail_sort,
                            cfg.warmup_iters,
                            cfg.iters,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place(
                            &cases,
                            n,
                            sort_small_variant_pad_up,
                            cfg.warmup_iters,
                            cfg.iters,
                            cfg.repeats,
                        ),
                        benchmark_cases_in_place(
                            &cases,
                            n,
                            sort_small_variant_std_insert,
                            cfg.warmup_iters,
                            cfg.iters,
                            cfg.repeats,
                        ),
                    )
                };

            println!(
                "{:>4} {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>11.1} {:>7.1}%",
                n,
                std_s.median_ns,
                std_s.jitter_pct(),
                net_s.median_ns,
                net_s.jitter_pct(),
                net_reg_s.median_ns,
                net_reg_s.jitter_pct(),
                weird_s.median_ns,
                weird_s.jitter_pct(),
                small_s.median_ns,
                small_s.jitter_pct(),
                pad_s.median_ns,
                pad_s.jitter_pct(),
                stdins_s.median_ns,
                stdins_s.jitter_pct(),
            );
        } else {
            // "speedup" is "how many times faster sort_small is than std" (higher is better).
            let speedup = std_s.median_ns / small_s.median_ns;
            println!(
                "{:>4} {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>8.2}x",
                n,
                std_s.median_ns,
                std_s.jitter_pct(),
                small_s.median_ns,
                small_s.jitter_pct(),
                speedup
            );
        }
    }
}
