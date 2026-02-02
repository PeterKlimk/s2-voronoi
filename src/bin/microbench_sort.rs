//! Microbenchmark for sorting algorithms.
//!
//! Compares sort_small (sorting networks) against std's sort_unstable for small N.

#![cfg_attr(test, feature(test))]

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use s2_voronoi::sort::{bidirectional_same_size_merge, merge_forward, sort_small};
use s2_voronoi::sort_nets::{sort16_tail_out, sort8_net};
#[path = "../bin_support/sort_nets_tail4.rs"]
mod sort_nets_tail4;
use std::hint::black_box;
use std::hint::select_unpredictable;
use std::ptr;
use std::time::Instant;

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
    breakdown: bool,
    sort16_abi: bool,
    insert_bench: bool,
    insert_bases: Vec<usize>,
    tailreg_bench: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sizes: (8..=32).collect(),
            cases: 1024,
            warmup_iters: 25_000,
            iters: 250_000,
            repeats: 21,
            warmup_ms: None,
            target_ms: None,
            seed: 42,
            breakdown: false,
            sort16_abi: false,
            insert_bench: false,
            insert_bases: vec![24],
            tailreg_bench: false,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args = std::env::args().skip(1);
    for arg in args {
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
            ("--insert-bases", Some(v)) => {
                cfg.insert_bases = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>().expect("invalid --insert-bases entry"))
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
            ("--breakdown", None) => cfg.breakdown = true,
            ("--sort16-abi", None) => cfg.sort16_abi = true,
            ("--insert-bench", None) => cfg.insert_bench = true,
            ("--tailreg-bench", None) => cfg.tailreg_bench = true,
            ("--help", _) | ("-h", _) => {
                eprintln!(
                    "microbench_sort options:\n  \
--sizes=8,9,10.. (comma list)\n  \
--insert-bench (run insertion-variant shootout)\n  \
--insert-bases=8,16,24,32 (comma list; default 24)\n  \
--tailreg-bench (compare 8 tail regs vs 4 tail regs in pad-up)\n  \
--cases=N (default 1024)\n  \
--warmup=N (default 25000)\n  \
--iters=N (default 250000)\n  \
--repeats=N (default 21)\n  \
--warmup-ms=N (duration-based warmup)\n  \
--target-ms=N (duration-based measurement; overrides --iters)\n  \
--seed=N (default 42)\n  \
--breakdown (split down+ins)\n  \
--sort16-abi (compare tail output ABIs)"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg: {arg} (use --help)");
                std::process::exit(2);
            }
        }
    }

    assert!(cfg.cases > 0, "--cases must be > 0");
    assert!(cfg.iters > 0, "--iters must be > 0");
    assert!(cfg.repeats > 0, "--repeats must be > 0");
    if let Some(ms) = cfg.target_ms {
        assert!(ms > 0, "--target-ms must be > 0");
    }
    if let Some(ms) = cfg.warmup_ms {
        assert!(ms > 0, "--warmup-ms must be > 0");
    }
    assert!(
        !cfg.insert_bases.is_empty(),
        "--insert-bases must be non-empty"
    );
    for &base in &cfg.insert_bases {
        assert!(
            [8usize, 16, 24, 32].contains(&base),
            "--insert-bases entries must be one of 8,16,24,32"
        );
    }
    cfg
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

/// Generate `cases` random u64 arrays, flattened.
fn generate_cases(len: usize, cases: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = vec![0u64; len * cases];
    for x in &mut out {
        *x = rng.gen();
    }
    out
}

// Replicates the "down+ins" insertion stage from `s2_voronoi::sort::sort_small`.
// Used only for microbench breakdowns.
#[inline(always)]
unsafe fn insert_suffix_like_sort_small(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!(rem <= 3);

    let p = v.as_mut_ptr();

    #[inline(always)]
    fn cswap_unpredictable_u64(v: &mut [u64], i: usize, j: usize) {
        debug_assert!(i != j);
        debug_assert!(i < v.len());
        debug_assert!(j < v.len());
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

    if rem >= 2 {
        let suffix = std::slice::from_raw_parts_mut(p.add(base), rem);
        cswap_unpredictable_u64(suffix, 0, 1);
        if rem == 3 {
            cswap_unpredictable_u64(suffix, 1, 2);
            cswap_unpredictable_u64(suffix, 0, 1);
        }
    }

    for idx in base..base + rem {
        let tail = p.add(idx);
        let mut sift = tail.sub(1);

        if *tail >= *sift {
            continue;
        }

        let tmp = *tail;
        loop {
            std::ptr::copy_nonoverlapping(sift, sift.add(1), 1);

            if sift == p {
                *p = tmp;
                break;
            }

            sift = sift.sub(1);
            if tmp >= *sift {
                *sift.add(1) = tmp;
                break;
            }
        }
    }
}

fn benchmark_cases_in_place<F>(
    cases_flat: &[u64],
    len: usize,
    mut f: F,
    warmup: usize,
    iters: usize,
    repeats: usize,
) -> Stats
where
    F: FnMut(&mut [u64]),
{
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);

    let mut buf = vec![0u64; len];

    // Warmup (helps reduce turbo/ICache effects).
    let warmup_iters = warmup.min(iters);
    let mut acc = 0u64;
    for i in 0..warmup_iters {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        f(&mut buf);
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
            f(&mut buf);
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

fn benchmark_cases_in_place_timed<F>(
    cases_flat: &[u64],
    len: usize,
    mut f: F,
    warmup_ms: u64,
    target_ms: u64,
    repeats: usize,
) -> Stats
where
    F: FnMut(&mut [u64]),
{
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);

    let mut buf = vec![0u64; len];

    let warmup_dur = std::time::Duration::from_millis(warmup_ms);
    let target_dur = std::time::Duration::from_millis(target_ms);

    // Duration-based warmup.
    let mut acc = 0u64;
    let start = Instant::now();
    let mut i = 0usize;
    while start.elapsed() < warmup_dur {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        f(&mut buf);
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
            f(&mut buf);
            acc ^= buf[0];
            acc ^= buf[len - 1];
            iters += 1;
        }
        black_box(acc);

        // Ensure we always have at least one iteration for the division.
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

/// Std sort_unstable
fn std_sort_unstable(v: &mut [u64]) {
    v.sort_unstable();
}

/// Sorting network based sort (for small N <= 35)
fn network_sort(v: &mut [u64]) {
    sort_small(v);
}

fn run_small_sort_benchmarks(cfg: &Config) {
    println!("Small Sorting Microbenchmarks (u64)");
    if let Some(target_ms) = cfg.target_ms {
        let warmup_ms = cfg.warmup_ms.unwrap_or(200);
        println!(
            "cases={} warmup_ms={} target_ms={} repeats={}",
            cfg.cases, warmup_ms, target_ms, cfg.repeats
        );
    } else {
        println!(
            "cases={} warmup={} iters={} repeats={}",
            cfg.cases, cfg.warmup_iters, cfg.iters, cfg.repeats
        );
    }
    if cfg.breakdown {
        println!("breakdown=on");
    }
    println!();

    println!(
        "{:>4} {:>11} {:>8} {:>11} {:>8} {:>9} {:>12}",
        "N", "std med", "std jit", "net med", "net jit", "speedup", "strategy"
    );
    println!("{:-<60}", "");

    for &n in &cfg.sizes {
        let cases = generate_cases(
            n,
            cfg.cases,
            cfg.seed ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );

        let (std_s, net_s) = if let Some(target_ms) = cfg.target_ms {
            let warmup_ms = cfg.warmup_ms.unwrap_or(200);
            (
                benchmark_cases_in_place_timed(
                    &cases,
                    n,
                    std_sort_unstable,
                    warmup_ms,
                    target_ms,
                    cfg.repeats,
                ),
                benchmark_cases_in_place_timed(
                    &cases,
                    n,
                    network_sort,
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
                    std_sort_unstable,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                ),
                benchmark_cases_in_place(
                    &cases,
                    n,
                    network_sort,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                ),
            )
        };
        let speedup = std_s.median_ns / net_s.median_ns;

        let rem = n & 7;
        let strategy = if n < 8 {
            "fallback"
        } else if rem == 0 {
            "net"
        } else if rem <= 3 {
            "down+ins"
        } else {
            "pad-up"
        };
        let marker = if speedup > 1.2 {
            "✓"
        } else if speedup < 0.9 {
            "✗"
        } else {
            "~"
        };

        println!(
            "{} {:>3} {:>11.1} {:>7.1}% {:>11.1} {:>7.1}% {:>8.2}x {:>12}",
            marker,
            n,
            std_s.median_ns,
            std_s.jitter_pct(),
            net_s.median_ns,
            net_s.jitter_pct(),
            speedup,
            strategy
        );

        if cfg.breakdown && strategy == "down+ins" {
            let down = n & !7;

            let mut prepared = cases.clone();
            for case_idx in 0..cfg.cases {
                let start = case_idx * n;
                sort_small(&mut prepared[start..start + down]);
            }

            let down_only = |buf: &mut [u64]| sort_small(&mut buf[..down]);
            let ins_only =
                |buf: &mut [u64]| unsafe { insert_suffix_like_sort_small(buf, down, rem) };

            let (down_s, ins_s) = if let Some(target_ms) = cfg.target_ms {
                let warmup_ms = cfg.warmup_ms.unwrap_or(200);
                (
                    benchmark_cases_in_place_timed(
                        &prepared,
                        n,
                        down_only,
                        warmup_ms,
                        target_ms,
                        cfg.repeats,
                    ),
                    benchmark_cases_in_place_timed(
                        &prepared,
                        n,
                        ins_only,
                        warmup_ms,
                        target_ms,
                        cfg.repeats,
                    ),
                )
            } else {
                (
                    benchmark_cases_in_place(
                        &prepared,
                        n,
                        down_only,
                        cfg.warmup_iters,
                        cfg.iters,
                        cfg.repeats,
                    ),
                    benchmark_cases_in_place(
                        &prepared,
                        n,
                        ins_only,
                        cfg.warmup_iters,
                        cfg.iters,
                        cfg.repeats,
                    ),
                )
            };

            println!(
                "      split: down {:>8.1}ns ({:>5.1}%)  ins {:>8.1}ns ({:>5.1}%)",
                down_s.median_ns,
                down_s.jitter_pct(),
                ins_s.median_ns,
                ins_s.jitter_pct()
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum InsertDist {
    Random,
    WorstCase,
    BestCase,
}

impl InsertDist {
    fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::WorstCase => "worst",
            Self::BestCase => "best",
        }
    }

    fn seed_tag(self) -> u64 {
        match self {
            Self::Random => 0,
            Self::WorstCase => 1,
            Self::BestCase => 2,
        }
    }
}

fn generate_insert_cases(
    base: usize,
    rem: usize,
    cases: usize,
    seed: u64,
    dist: InsertDist,
) -> Vec<u64> {
    let len = base + rem;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = vec![0u64; len * cases];

    // Keep `u64::MAX` out to avoid accidentally interacting with `sort_small`'s sentinel
    // assumptions if you ever swap preparation strategies.
    #[inline(always)]
    fn clamp(x: u64) -> u64 {
        x & (u64::MAX - 1)
    }

    for case_idx in 0..cases {
        let start = case_idx * len;
        let buf = &mut out[start..start + len];

        match dist {
            InsertDist::Random => {
                for x in buf.iter_mut() {
                    *x = clamp(rng.gen());
                }
            }
            InsertDist::WorstCase => {
                // Prefix values all have the high bit set; suffix values don't.
                // Ensures inserted elements tend to go to the very front.
                for x in &mut buf[..base] {
                    *x = clamp(rng.gen::<u64>() | (1u64 << 63));
                }
                for x in &mut buf[base..] {
                    *x = clamp(rng.gen::<u64>() & ((1u64 << 63) - 1));
                }
            }
            InsertDist::BestCase => {
                // Prefix values all have the high bit clear; suffix values all have it set.
                // Ensures inserted elements tend to stay at the very end (fast path).
                for x in &mut buf[..base] {
                    *x = clamp(rng.gen::<u64>() & ((1u64 << 63) - 1));
                }
                for x in &mut buf[base..] {
                    *x = clamp(rng.gen::<u64>() | (1u64 << 63));
                }
            }
        }
    }

    out
}

#[inline(always)]
unsafe fn insert_suffix_baseline(v: &mut [u64], base: usize, rem: usize) {
    insert_suffix_like_sort_small(v, base, rem);
}

#[inline(always)]
unsafe fn sort_suffix_unpredictable(v: *mut u64, rem: usize) {
    #[inline(always)]
    unsafe fn cswap_unpredictable_ptr(a: *mut u64, b: *mut u64) {
        let va = *a;
        let vb = *b;
        let cond = va <= vb;
        *a = select_unpredictable(cond, va, vb);
        *b = select_unpredictable(cond, vb, va);
    }

    if rem >= 2 {
        cswap_unpredictable_ptr(v.add(0), v.add(1));
        if rem == 3 {
            cswap_unpredictable_ptr(v.add(1), v.add(2));
            cswap_unpredictable_ptr(v.add(0), v.add(1));
        }
    }
}

/// Insert suffix using a single `ptr::copy` (memmove) per inserted element.
/// Keeps the same "sort suffix first" behavior as `sort_small`.
unsafe fn insert_suffix_memmove_linear(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();
    sort_suffix_unpredictable(p.add(base), rem);

    for idx in base..base + rem {
        let tmp = *p.add(idx);
        let mut pos = idx;

        // Find insertion position without moving anything.
        while pos > 0 {
            let prev = p.add(pos - 1);
            if tmp >= *prev {
                break;
            }
            pos -= 1;
        }

        if pos != idx {
            // Shift the block [pos, idx) right by 1 (overlapping, so use `ptr::copy`).
            ptr::copy(p.add(pos), p.add(pos + 1), idx - pos);
            *p.add(pos) = tmp;
        }
    }
}

/// Insert suffix using binary search (upper_bound) + one memmove per element.
unsafe fn insert_suffix_memmove_binary(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();
    sort_suffix_unpredictable(p.add(base), rem);

    for idx in base..base + rem {
        let tmp = *p.add(idx);

        // upper_bound in v[..idx] for `tmp` (first element > tmp)
        let mut left = 0usize;
        let mut right = idx;
        while left < right {
            let mid = left + ((right - left) >> 1);
            let midv = *p.add(mid);
            if tmp < midv {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        let pos = left;

        if pos != idx {
            ptr::copy(p.add(pos), p.add(pos + 1), idx - pos);
            *p.add(pos) = tmp;
        }
    }
}

/// Like `insert_suffix_memmove_binary`, but with the same fast-path as the baseline:
/// if the element is already >= its predecessor, skip all work.
unsafe fn insert_suffix_memmove_binary_fastpath(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();
    sort_suffix_unpredictable(p.add(base), rem);

    for idx in base..base + rem {
        let tmp = *p.add(idx);

        if tmp >= *p.add(idx - 1) {
            continue;
        }

        // upper_bound in v[..idx] for `tmp` (first element > tmp)
        let mut left = 0usize;
        let mut right = idx;
        while left < right {
            let mid = left + ((right - left) >> 1);
            let midv = *p.add(mid);
            if tmp < midv {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        let pos = left;

        debug_assert!(pos < idx);
        ptr::copy(p.add(pos), p.add(pos + 1), idx - pos);
        *p.add(pos) = tmp;
    }
}

/// Binary search + manual backward shift loop (avoids calling `memmove` for small copies).
unsafe fn insert_suffix_binary_shiftloop_fastpath(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();
    sort_suffix_unpredictable(p.add(base), rem);

    for idx in base..base + rem {
        let tmp = *p.add(idx);

        if tmp >= *p.add(idx - 1) {
            continue;
        }

        // upper_bound in v[..idx] for `tmp` (first element > tmp)
        let mut left = 0usize;
        let mut right = idx;
        while left < right {
            let mid = left + ((right - left) >> 1);
            let midv = *p.add(mid);
            if tmp < midv {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        let pos = left;

        debug_assert!(pos < idx);
        for k in (pos..idx).rev() {
            *p.add(k + 1) = *p.add(k);
        }
        *p.add(pos) = tmp;
    }
}

/// Same as `insert_suffix_memmove_linear`, but specialized to avoid the `for` loop overhead.
unsafe fn insert_suffix_memmove_linear_unrolled(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();
    sort_suffix_unpredictable(p.add(base), rem);

    #[inline(always)]
    unsafe fn insert_one(p: *mut u64, idx: usize) {
        let tmp = *p.add(idx);
        let mut pos = idx;
        while pos > 0 {
            let prev = p.add(pos - 1);
            if tmp >= *prev {
                break;
            }
            pos -= 1;
        }
        if pos != idx {
            ptr::copy(p.add(pos), p.add(pos + 1), idx - pos);
            *p.add(pos) = tmp;
        }
    }

    match rem {
        1 => insert_one(p, base),
        2 => {
            insert_one(p, base);
            insert_one(p, base + 1);
        }
        3 => {
            insert_one(p, base);
            insert_one(p, base + 1);
            insert_one(p, base + 2);
        }
        _ => unreachable!(),
    }
}

/// Insert suffix without the "sort suffix first" step (often loses, but good to measure).
unsafe fn insert_suffix_memmove_linear_no_suffix_sort(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();

    for idx in base..base + rem {
        let tmp = *p.add(idx);
        let mut pos = idx;
        while pos > 0 {
            let prev = p.add(pos - 1);
            if tmp >= *prev {
                break;
            }
            pos -= 1;
        }
        if pos != idx {
            ptr::copy(p.add(pos), p.add(pos + 1), idx - pos);
            *p.add(pos) = tmp;
        }
    }
}

/// Insert suffix by sorting it, then doing an in-place merge-from-the-back into `v[..base+rem]`.
unsafe fn insert_suffix_merge_back(v: &mut [u64], base: usize, rem: usize) {
    debug_assert!(base <= v.len());
    debug_assert!(base + rem <= v.len());
    debug_assert!((1..=3).contains(&rem));

    let p = v.as_mut_ptr();
    sort_suffix_unpredictable(p.add(base), rem);

    if rem == 1 {
        insert_suffix_like_sort_small(v, base, rem);
        return;
    }

    // Load suffix into regs first (merge writes into the suffix area).
    let r0 = *p.add(base);
    let r1 = *p.add(base + 1);
    let r2 = if rem == 3 { *p.add(base + 2) } else { 0u64 };

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

fn run_insert_variant_benchmarks(cfg: &Config) {
    let dists = [
        InsertDist::Random,
        InsertDist::WorstCase,
        InsertDist::BestCase,
    ];
    let variants: [(&str, unsafe fn(&mut [u64], usize, usize)); 7] = [
        ("memmove linear", insert_suffix_memmove_linear),
        ("memmove binary", insert_suffix_memmove_binary),
        ("memmove binary fp", insert_suffix_memmove_binary_fastpath),
        (
            "binary shiftloop fp",
            insert_suffix_binary_shiftloop_fastpath,
        ),
        ("merge-back", insert_suffix_merge_back),
        ("memmove linear unr", insert_suffix_memmove_linear_unrolled),
        (
            "memmove no suf sort",
            insert_suffix_memmove_linear_no_suffix_sort,
        ),
    ];

    println!();
    println!("Insert Suffix Microbenchmarks (u64; base=sorted prefix, rem=1..=3)");
    if let Some(target_ms) = cfg.target_ms {
        let warmup_ms = cfg.warmup_ms.unwrap_or(200);
        println!(
            "cases={} warmup_ms={} target_ms={} repeats={}",
            cfg.cases, warmup_ms, target_ms, cfg.repeats
        );
    } else {
        println!(
            "cases={} warmup={} iters={} repeats={}",
            cfg.cases, cfg.warmup_iters, cfg.iters, cfg.repeats
        );
    }
    println!();

    println!(
        "{:>5} {:>4} {:>6} {:>20} {:>11} {:>8} {:>9}",
        "base", "rem", "dist", "variant", "med (ns)", "jit", "vs base"
    );
    println!("{:-<77}", "");

    for &base in &cfg.insert_bases {
        for rem in 1..=3usize {
            let len = base + rem;
            for dist in dists {
                let seed = cfg.seed
                    ^ (base as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (rem as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ dist.seed_tag().wrapping_mul(0x94D0_49BB_1331_11EB);

                let mut prepared = generate_insert_cases(base, rem, cfg.cases, seed, dist);
                for case_idx in 0..cfg.cases {
                    let start = case_idx * len;
                    sort_small(&mut prepared[start..start + base]);
                }

                // Quick correctness check on one case per (base,rem,dist).
                let src0 = prepared[0..len].to_vec();
                let mut expected0 = src0.clone();
                expected0.sort_unstable();

                let mut check = src0.clone();
                unsafe { insert_suffix_baseline(&mut check, base, rem) };
                assert_eq!(
                    check,
                    expected0,
                    "baseline insert produced wrong result (base={base} rem={rem} dist={})",
                    dist.name()
                );
                for (name, f) in variants {
                    let mut check = src0.clone();
                    unsafe { f(&mut check, base, rem) };
                    assert_eq!(
                        check,
                        expected0,
                        "variant {name} produced wrong result (base={base} rem={rem} dist={})",
                        dist.name()
                    );
                }

                let baseline_stats = {
                    let f = |buf: &mut [u64]| unsafe { insert_suffix_baseline(buf, base, rem) };
                    if let Some(target_ms) = cfg.target_ms {
                        let warmup_ms = cfg.warmup_ms.unwrap_or(200);
                        benchmark_cases_in_place_timed(
                            &prepared,
                            len,
                            f,
                            warmup_ms,
                            target_ms,
                            cfg.repeats,
                        )
                    } else {
                        benchmark_cases_in_place(
                            &prepared,
                            len,
                            f,
                            cfg.warmup_iters,
                            cfg.iters,
                            cfg.repeats,
                        )
                    }
                };

                println!(
                    "{:>5} {:>4} {:>6} {:>20} {:>11.1} {:>7.1}% {:>8.2}x",
                    base,
                    rem,
                    dist.name(),
                    "baseline shift-loop",
                    baseline_stats.median_ns,
                    baseline_stats.jitter_pct(),
                    1.0
                );

                for (name, f) in variants {
                    let stats = {
                        let f = |buf: &mut [u64]| unsafe { f(buf, base, rem) };
                        if let Some(target_ms) = cfg.target_ms {
                            let warmup_ms = cfg.warmup_ms.unwrap_or(200);
                            benchmark_cases_in_place_timed(
                                &prepared,
                                len,
                                f,
                                warmup_ms,
                                target_ms,
                                cfg.repeats,
                            )
                        } else {
                            benchmark_cases_in_place(
                                &prepared,
                                len,
                                f,
                                cfg.warmup_iters,
                                cfg.iters,
                                cfg.repeats,
                            )
                        }
                    };

                    println!(
                        "{:>5} {:>4} {:>6} {:>20} {:>11.1} {:>7.1}% {:>8.2}x",
                        base,
                        rem,
                        dist.name(),
                        name,
                        stats.median_ns,
                        stats.jitter_pct(),
                        baseline_stats.median_ns / stats.median_ns
                    );
                }

                println!("{:-<77}", "");
            }
        }
    }
}

fn run_merge_benchmarks(cfg: &Config) {
    let len = 16;
    let cases_l = generate_cases(len, cfg.cases, cfg.seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let cases_r = generate_cases(len, cfg.cases, cfg.seed ^ 0x5A5A_5A5A_5A5A_5A5A);

    // Pre-sort inputs once so we only measure the merge.
    let mut left_sorted = cases_l;
    let mut right_sorted = cases_r;
    for i in 0..cfg.cases {
        left_sorted[i * len..(i + 1) * len].sort_unstable();
        right_sorted[i * len..(i + 1) * len].sort_unstable();
    }

    let num_cases = cfg.cases;
    let mut dst = vec![0u64; 2 * len];

    let warmup_iters = cfg.warmup_iters.min(cfg.iters);

    let mut bench_merge = |f: fn(&[u64], &[u64], &mut [u64])| -> Stats {
        if let Some(target_ms) = cfg.target_ms {
            let warmup_ms = cfg.warmup_ms.unwrap_or(200);
            let warmup_dur = std::time::Duration::from_millis(warmup_ms);
            let target_dur = std::time::Duration::from_millis(target_ms);

            // Warmup (duration-based).
            let mut acc = 0u64;
            let start = Instant::now();
            let mut i = 0usize;
            while start.elapsed() < warmup_dur {
                let case_idx = i % num_cases;
                let l = &left_sorted[case_idx * len..(case_idx + 1) * len];
                let r = &right_sorted[case_idx * len..(case_idx + 1) * len];
                f(l, r, &mut dst);
                acc ^= dst[0];
                acc ^= dst[2 * len - 1];
                i += 1;
            }
            black_box(acc);

            let mut samples = Vec::with_capacity(cfg.repeats);
            let mut min_ns = f64::INFINITY;
            let mut max_ns: f64 = 0.0;

            for rep in 0..cfg.repeats {
                let start = Instant::now();
                let mut acc = rep as u64;
                let mut iters = 0usize;
                while start.elapsed() < target_dur {
                    let case_idx = (iters + rep) % num_cases;
                    let l = &left_sorted[case_idx * len..(case_idx + 1) * len];
                    let r = &right_sorted[case_idx * len..(case_idx + 1) * len];
                    f(l, r, &mut dst);
                    acc ^= dst[0];
                    acc ^= dst[2 * len - 1];
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
        } else {
            // Warmup (iteration-based).
            let mut acc = 0u64;
            for i in 0..warmup_iters {
                let case_idx = i % num_cases;
                let l = &left_sorted[case_idx * len..(case_idx + 1) * len];
                let r = &right_sorted[case_idx * len..(case_idx + 1) * len];
                f(l, r, &mut dst);
                acc ^= dst[0];
                acc ^= dst[2 * len - 1];
            }
            black_box(acc);

            let mut samples = Vec::with_capacity(cfg.repeats);
            let mut min_ns = f64::INFINITY;
            let mut max_ns: f64 = 0.0;

            for rep in 0..cfg.repeats {
                let start = Instant::now();
                let mut acc = rep as u64;
                for i in 0..cfg.iters {
                    let case_idx = (i + rep) % num_cases;
                    let l = &left_sorted[case_idx * len..(case_idx + 1) * len];
                    let r = &right_sorted[case_idx * len..(case_idx + 1) * len];
                    f(l, r, &mut dst);
                    acc ^= dst[0];
                    acc ^= dst[2 * len - 1];
                }
                black_box(acc);
                let elapsed = start.elapsed();
                let ns = elapsed.as_nanos() as f64 / cfg.iters as f64;
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
    };

    fn merge_forward_u64(left: &[u64], right: &[u64], dst: &mut [u64]) {
        merge_forward(left, right, dst, &mut |a, b| a < b);
    }

    fn merge_bidir_u64(left: &[u64], right: &[u64], dst: &mut [u64]) {
        bidirectional_same_size_merge(left, right, dst, &mut |a, b| a < b);
    }

    println!();
    println!("Merge Microbenchmarks (len=16 + len=16 -> 32)");
    println!("{:>18} {:>11} {:>8}", "merge", "med (ns)", "jit");
    println!("{:-<38}", "");

    let fwd = bench_merge(merge_forward_u64);
    let bidir = bench_merge(merge_bidir_u64);
    println!(
        "{:>18} {:>11.1} {:>7.1}%",
        "forward",
        fwd.median_ns,
        fwd.jitter_pct()
    );
    println!(
        "{:>18} {:>11.1} {:>7.1}%",
        "bidirectional",
        bidir.median_ns,
        bidir.jitter_pct()
    );
    println!(
        "{:>18} {:>7.2}x",
        "speedup",
        fwd.median_ns / bidir.median_ns
    );

    println!();
    run_sort16_variant_benchmarks(cfg);

    if cfg.sort16_abi {
        println!();
        run_sort16_tail_abi_benchmarks(cfg);
    }
}

fn sort16_direct_net(v: &mut [u64]) {
    debug_assert_eq!(v.len(), 16);
    unsafe { sort16_tail_out(v.as_mut_ptr(), v.as_mut_ptr().add(8), 8) }
}

fn sort16_2x8_bidir(v: &mut [u64]) {
    debug_assert_eq!(v.len(), 16);

    unsafe {
        let base = v.as_mut_ptr();
        let a = sort8_net(
            *base.add(0),
            *base.add(1),
            *base.add(2),
            *base.add(3),
            *base.add(4),
            *base.add(5),
            *base.add(6),
            *base.add(7),
        );
        ptr::copy_nonoverlapping(a.as_ptr(), base.add(0), 8);

        let b = sort8_net(
            *base.add(8),
            *base.add(9),
            *base.add(10),
            *base.add(11),
            *base.add(12),
            *base.add(13),
            *base.add(14),
            *base.add(15),
        );
        ptr::copy_nonoverlapping(b.as_ptr(), base.add(8), 8);
    }

    let (l, r) = v.split_at(8);
    let mut tmp = [0u64; 16];
    bidirectional_same_size_merge(l, r, &mut tmp[..], &mut |a, b| a < b);
    v.copy_from_slice(&tmp);
}

fn run_sort16_variant_benchmarks(cfg: &Config) {
    let len = 16;
    let cases = generate_cases(len, cfg.cases, cfg.seed ^ 0xD1D1_D1D1_D1D1_D1D1);

    let bench = |f: fn(&mut [u64])| -> Stats {
        if let Some(target_ms) = cfg.target_ms {
            let warmup_ms = cfg.warmup_ms.unwrap_or(200);
            benchmark_cases_in_place_timed(&cases, len, f, warmup_ms, target_ms, cfg.repeats)
        } else {
            benchmark_cases_in_place(&cases, len, f, cfg.warmup_iters, cfg.iters, cfg.repeats)
        }
    };

    let direct = bench(sort16_direct_net);
    let two8 = bench(sort16_2x8_bidir);

    println!("Sort16 Variant Microbenchmarks (len=16)");
    println!("{:>22} {:>11} {:>8}", "variant", "med (ns)", "jit");
    println!("{:-<44}", "");
    println!(
        "{:>22} {:>11.1} {:>7.1}%",
        "sort16_tail_out",
        direct.median_ns,
        direct.jitter_pct()
    );
    println!(
        "{:>22} {:>11.1} {:>7.1}%",
        "sort8+sort8+bidir",
        two8.median_ns,
        two8.jitter_pct()
    );
    println!(
        "{:>22} {:>7.2}x",
        "speedup",
        two8.median_ns / direct.median_ns
    );
}

const SENTINEL: u64 = u64::MAX;

#[inline(always)]
unsafe fn cswap_reg_u64(a: &mut u64, b: &mut u64) {
    let va = *a;
    let vb = *b;
    let cond = va <= vb;
    *a = select_unpredictable(cond, va, vb);
    *b = select_unpredictable(cond, vb, va);
}

#[inline(always)]
unsafe fn cswap_reg_ptr_u64(reg: &mut u64, base: *mut u64, idx: usize) {
    let ptr = base.add(idx);
    let val = *ptr;
    let r = *reg;
    let cond = val <= r;
    *ptr = select_unpredictable(cond, val, r);
    *reg = select_unpredictable(cond, r, val);
}

#[inline(always)]
unsafe fn cswap_ptr_u64(base: *mut u64, i: usize, j: usize) {
    debug_assert!(j > i);
    let pi = base.add(i);
    let pj = base.add(j);
    let a = *pi;
    let b = *pj;
    let cond = a <= b;
    *pi = select_unpredictable(cond, a, b);
    *pj = select_unpredictable(cond, b, a);
}

macro_rules! sort16_net_body_u64 {
    ($base:expr, $r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident) => {{
        // Keep this comparator sequence in sync with `src/sort_nets.rs` `sort16_tail_out`.
        cswap_reg_ptr_u64(&mut $r5, $base, 0);
        cswap_reg_ptr_u64(&mut $r4, $base, 1);
        cswap_reg_ptr_u64(&mut $r7, $base, 2);
        cswap_reg_ptr_u64(&mut $r6, $base, 3);
        cswap_reg_ptr_u64(&mut $r0, $base, 4);
        cswap_ptr_u64($base, 5, 6);
        cswap_reg_ptr_u64(&mut $r3, $base, 7);
        cswap_reg_u64(&mut $r1, &mut $r2);
        cswap_ptr_u64($base, 0, 5);
        cswap_ptr_u64($base, 1, 7);
        cswap_reg_ptr_u64(&mut $r1, $base, 2);
        cswap_ptr_u64($base, 3, 4);
        cswap_reg_ptr_u64(&mut $r5, $base, 6);
        cswap_reg_u64(&mut $r0, &mut $r6);
        cswap_reg_u64(&mut $r2, &mut $r7);
        cswap_reg_u64(&mut $r3, &mut $r4);
        cswap_ptr_u64($base, 0, 1);
        cswap_ptr_u64($base, 2, 3);
        cswap_ptr_u64($base, 4, 5);
        cswap_reg_ptr_u64(&mut $r0, $base, 6);
        cswap_reg_ptr_u64(&mut $r1, $base, 7);
        cswap_reg_u64(&mut $r2, &mut $r3);
        cswap_reg_u64(&mut $r4, &mut $r5);
        cswap_reg_u64(&mut $r6, &mut $r7);
        cswap_ptr_u64($base, 0, 2);
        cswap_ptr_u64($base, 1, 3);
        cswap_reg_ptr_u64(&mut $r2, $base, 4);
        cswap_reg_ptr_u64(&mut $r3, $base, 5);
        cswap_ptr_u64($base, 6, 7);
        cswap_reg_u64(&mut $r0, &mut $r1);
        cswap_reg_u64(&mut $r4, &mut $r6);
        cswap_reg_u64(&mut $r5, &mut $r7);
        cswap_ptr_u64($base, 1, 2);
        cswap_reg_ptr_u64(&mut $r4, $base, 3);
        cswap_ptr_u64($base, 4, 6);
        cswap_ptr_u64($base, 5, 7);
        cswap_reg_u64(&mut $r0, &mut $r2);
        cswap_reg_u64(&mut $r1, &mut $r3);
        cswap_reg_u64(&mut $r5, &mut $r6);
        cswap_ptr_u64($base, 1, 4);
        cswap_ptr_u64($base, 2, 6);
        cswap_reg_ptr_u64(&mut $r0, $base, 5);
        cswap_reg_ptr_u64(&mut $r2, $base, 7);
        cswap_reg_u64(&mut $r1, &mut $r5);
        cswap_reg_u64(&mut $r3, &mut $r6);
        cswap_ptr_u64($base, 2, 4);
        cswap_ptr_u64($base, 3, 6);
        cswap_reg_u64(&mut $r1, &mut $r4);
        cswap_reg_u64(&mut $r3, &mut $r5);
        cswap_ptr_u64($base, 3, 5);
        cswap_reg_ptr_u64(&mut $r0, $base, 6);
        cswap_reg_ptr_u64(&mut $r1, $base, 7);
        cswap_reg_u64(&mut $r2, &mut $r4);
        cswap_ptr_u64($base, 3, 4);
        cswap_ptr_u64($base, 5, 6);
        cswap_reg_ptr_u64(&mut $r0, $base, 7);
        cswap_reg_u64(&mut $r1, &mut $r2);
        cswap_reg_u64(&mut $r3, &mut $r4);
        cswap_ptr_u64($base, 6, 7);
        cswap_reg_u64(&mut $r0, &mut $r1);
    }};
}

#[inline(always)]
unsafe fn store_tail_u64(
    out: *mut u64,
    tail_len: usize,
    r0: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    r4: u64,
    r5: u64,
    r6: u64,
    r7: u64,
) {
    debug_assert!(tail_len <= 8);
    match tail_len {
        0 => {}
        1 => *out.add(0) = r0,
        2 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
        }
        3 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
            *out.add(2) = r2;
        }
        4 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
            *out.add(2) = r2;
            *out.add(3) = r3;
        }
        5 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
            *out.add(2) = r2;
            *out.add(3) = r3;
            *out.add(4) = r4;
        }
        6 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
            *out.add(2) = r2;
            *out.add(3) = r3;
            *out.add(4) = r4;
            *out.add(5) = r5;
        }
        7 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
            *out.add(2) = r2;
            *out.add(3) = r3;
            *out.add(4) = r4;
            *out.add(5) = r5;
            *out.add(6) = r6;
        }
        8 => {
            *out.add(0) = r0;
            *out.add(1) = r1;
            *out.add(2) = r2;
            *out.add(3) = r3;
            *out.add(4) = r4;
            *out.add(5) = r5;
            *out.add(6) = r6;
            *out.add(7) = r7;
        }
        _ => unreachable!(),
    }
}

#[inline(never)]
fn sort16_outer_padded_ret_array(base: *mut u64, tail_len: usize) {
    unsafe {
        let mut regs = [SENTINEL; 8];
        ptr::copy_nonoverlapping(base.add(8), regs.as_mut_ptr(), tail_len);
        let [mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7] = regs;

        sort16_net_body_u64!(base, r0, r1, r2, r3, r4, r5, r6, r7);

        // Return-array style: materialize the full aggregate, then copy out the prefix.
        let out = [r0, r1, r2, r3, r4, r5, r6, r7];
        ptr::copy_nonoverlapping(out.as_ptr(), base.add(8), tail_len);
    }
}

#[inline(never)]
unsafe fn sort16_inner_padded_store_baseadd(base: *mut u64, tail_len: usize) {
    let mut regs = [SENTINEL; 8];
    ptr::copy_nonoverlapping(base.add(8), regs.as_mut_ptr(), tail_len);
    let [mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7] = regs;

    sort16_net_body_u64!(base, r0, r1, r2, r3, r4, r5, r6, r7);
    store_tail_u64(base.add(8), tail_len, r0, r1, r2, r3, r4, r5, r6, r7);
}

#[inline(never)]
unsafe fn sort16_inner_padded_store_out(base: *mut u64, out: *mut u64, tail_len: usize) {
    let mut regs = [SENTINEL; 8];
    ptr::copy_nonoverlapping(base.add(8), regs.as_mut_ptr(), tail_len);
    let [mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7] = regs;

    sort16_net_body_u64!(base, r0, r1, r2, r3, r4, r5, r6, r7);
    store_tail_u64(out, tail_len, r0, r1, r2, r3, r4, r5, r6, r7);
}

#[inline(never)]
fn sort16_inner_baseadd(base: *mut u64, tail_len: usize) {
    unsafe { sort16_inner_padded_store_baseadd(base, tail_len) }
}

#[inline(never)]
fn sort16_inner_out_param(base: *mut u64, tail_len: usize) {
    unsafe { sort16_inner_padded_store_out(base, base.add(8), tail_len) }
}

#[inline(never)]
unsafe fn sort16_inner_padded_return_tuple(
    base: *mut u64,
    tail_len: usize,
) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    let mut regs = [SENTINEL; 8];
    ptr::copy_nonoverlapping(base.add(8), regs.as_mut_ptr(), tail_len);
    let [mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7] = regs;

    sort16_net_body_u64!(base, r0, r1, r2, r3, r4, r5, r6, r7);
    (r0, r1, r2, r3, r4, r5, r6, r7)
}

#[inline(never)]
fn sort16_inner_ret_tuple(base: *mut u64, tail_len: usize) {
    unsafe {
        let (r0, r1, r2, r3, r4, r5, r6, r7) = sort16_inner_padded_return_tuple(base, tail_len);
        store_tail_u64(base.add(8), tail_len, r0, r1, r2, r3, r4, r5, r6, r7);
    }
}

fn benchmark_sort16_tail_abi_timed(
    cases_flat: &[u64],
    tail_len: usize,
    f: fn(*mut u64, usize),
    warmup_ms: u64,
    target_ms: u64,
    repeats: usize,
) -> Stats {
    let len = 16usize;
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);

    let mut buf = vec![0u64; len];
    let warmup_dur = std::time::Duration::from_millis(warmup_ms);
    let target_dur = std::time::Duration::from_millis(target_ms);

    let pick = 8 + tail_len - 1;

    // Warmup.
    let mut acc = 0u64;
    let start = Instant::now();
    let mut i = 0usize;
    while start.elapsed() < warmup_dur {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        f(buf.as_mut_ptr(), tail_len);
        acc ^= buf[0];
        acc ^= buf[7];
        acc ^= buf[pick];
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
            f(buf.as_mut_ptr(), tail_len);
            acc ^= buf[0];
            acc ^= buf[7];
            acc ^= buf[pick];
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

fn benchmark_sort16_tail_abi_iters(
    cases_flat: &[u64],
    tail_len: usize,
    f: fn(*mut u64, usize),
    warmup_iters: usize,
    iters: usize,
    repeats: usize,
) -> Stats {
    let len = 16usize;
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);

    let mut buf = vec![0u64; len];
    let pick = 8 + tail_len - 1;

    // Warmup.
    let warmup_iters = warmup_iters.min(iters);
    let mut acc = 0u64;
    for i in 0..warmup_iters {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        f(buf.as_mut_ptr(), tail_len);
        acc ^= buf[0];
        acc ^= buf[7];
        acc ^= buf[pick];
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
            f(buf.as_mut_ptr(), tail_len);
            acc ^= buf[0];
            acc ^= buf[7];
            acc ^= buf[pick];
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

fn run_sort16_tail_abi_benchmarks(cfg: &Config) {
    let cases = generate_cases(16, cfg.cases, cfg.seed ^ 0x1600_0000_0000_0001);

    let variants: [(&str, fn(*mut u64, usize)); 4] = [
        ("outer ret [u64;8]", sort16_outer_padded_ret_array),
        ("inner store base+8", sort16_inner_baseadd),
        ("inner store out ptr", sort16_inner_out_param),
        ("inner ret tuple", sort16_inner_ret_tuple),
    ];

    println!("Sort16 Tail ABI Microbenchmarks (len=16, tail_len=4..=8)");
    println!(
        "{:>6} {:>22} {:>11} {:>8}",
        "tail", "variant", "med (ns)", "jit"
    );
    println!("{:-<52}", "");

    for tail_len in 4..=8 {
        for (name, f) in variants {
            let s = if let Some(target_ms) = cfg.target_ms {
                let warmup_ms = cfg.warmup_ms.unwrap_or(200);
                benchmark_sort16_tail_abi_timed(
                    &cases,
                    tail_len,
                    f,
                    warmup_ms,
                    target_ms,
                    cfg.repeats,
                )
            } else {
                benchmark_sort16_tail_abi_iters(
                    &cases,
                    tail_len,
                    f,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                )
            };
            println!(
                "{:>6} {:>22} {:>11.1} {:>7.1}%",
                tail_len,
                name,
                s.median_ns,
                s.jitter_pct()
            );
        }
        println!("{:-<52}", "");
    }
}

fn run_tailreg_variant_benchmarks(cfg: &Config) {
    println!();
    println!("Pad-Up Tail-Reg Variant Microbenchmarks (u64)");
    if let Some(target_ms) = cfg.target_ms {
        let warmup_ms = cfg.warmup_ms.unwrap_or(200);
        println!(
            "cases={} warmup_ms={} target_ms={} repeats={}",
            cfg.cases, warmup_ms, target_ms, cfg.repeats
        );
    } else {
        println!(
            "cases={} warmup={} iters={} repeats={}",
            cfg.cases, cfg.warmup_iters, cfg.iters, cfg.repeats
        );
    }
    println!();
    println!(
        "{:>4} {:>8} {:>13} {:>8} {:>13} {:>8} {:>8}",
        "N", "tailregs", "med (ns)", "jit", "alt (ns)", "jit", "speedup"
    );
    println!("{:-<74}", "");

    fn bench<F: FnMut(&mut [u64])>(cfg: &Config, cases: &[u64], n: usize, f: F) -> Stats {
        if let Some(target_ms) = cfg.target_ms {
            let warmup_ms = cfg.warmup_ms.unwrap_or(200);
            benchmark_cases_in_place_timed(&cases, n, f, warmup_ms, target_ms, cfg.repeats)
        } else {
            benchmark_cases_in_place(&cases, n, f, cfg.warmup_iters, cfg.iters, cfg.repeats)
        }
    }

    for n in 12..=15 {
        let cases = generate_cases(n, cfg.cases, cfg.seed ^ (n as u64).wrapping_mul(0xC0FFEE));
        let base = 8usize;

        let tail8 = |v: &mut [u64]| unsafe {
            sort16_tail_out(v.as_mut_ptr(), v.as_mut_ptr().add(base), n - base);
        };
        let tail4 = |v: &mut [u64]| unsafe {
            sort_nets_tail4::sort16_tail4(v.as_mut_ptr(), v.as_mut_ptr().add(12), n - 12);
        };

        let s8 = bench(cfg, &cases, n, tail8);
        let s4 = bench(cfg, &cases, n, tail4);
        println!(
            "{:>4} {:>8} {:>13.1} {:>7.1}% {:>13.1} {:>7.1}% {:>7.2}x",
            n,
            "8->16",
            s8.median_ns,
            s8.jitter_pct(),
            s4.median_ns,
            s4.jitter_pct(),
            s8.median_ns / s4.median_ns
        );
    }

    println!("{:-<74}", "");

    for n in 20..=23 {
        let cases = generate_cases(n, cfg.cases, cfg.seed ^ (n as u64).wrapping_mul(0xDEADBEEF));
        let base = 16usize;

        let tail8 = |v: &mut [u64]| unsafe {
            s2_voronoi::sort_nets::sort24_tail_out(
                v.as_mut_ptr(),
                v.as_mut_ptr().add(base),
                n - base,
            );
        };
        let tail4 = |v: &mut [u64]| unsafe {
            sort_nets_tail4::sort24_tail4(v.as_mut_ptr(), v.as_mut_ptr().add(20), n - 20);
        };

        let s8 = bench(cfg, &cases, n, tail8);
        let s4 = bench(cfg, &cases, n, tail4);
        println!(
            "{:>4} {:>8} {:>13.1} {:>7.1}% {:>13.1} {:>7.1}% {:>7.2}x",
            n,
            "8->24",
            s8.median_ns,
            s8.jitter_pct(),
            s4.median_ns,
            s4.jitter_pct(),
            s8.median_ns / s4.median_ns
        );
    }
}

fn main() {
    let cfg = parse_args();
    run_small_sort_benchmarks(&cfg);
    run_merge_benchmarks(&cfg);
    if cfg.insert_bench {
        run_insert_variant_benchmarks(&cfg);
    }
    if cfg.tailreg_bench {
        run_tailreg_variant_benchmarks(&cfg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_sorts_correct() {
        for n in [8, 16, 24, 25, 26, 27, 28, 29, 30, 31, 32] {
            let data = generate_cases(n, 1, 42);

            let mut v1 = data.clone();
            std_sort_unstable(&mut v1);

            let mut v2 = data.clone();
            network_sort(&mut v2);

            assert_eq!(v1, v2, "sort failed for n={}", n);
        }
    }
}
