//! Microbenchmark for "unknown-k" neighbor selection strategies.
//!
//! We model a producer that can emit neighbors in sorted order in fixed-size buckets (default 8),
//! but doesn't know ahead of time how many neighbors the downstream clipper will actually consume.
//! A hidden variable (`need`) is sampled from a configurable distribution and determines when we
//! stop requesting buckets.
//!
//! This benchmarks three strategies:
//! - `select8`: Per-bucket `select_nth_unstable(k-1)` + sort prefix (baseline).
//! - `select16`: Preselect 16 once, then emit 8 + 8 (avoids re-select for common 9..16 needs).
//! - `pivot`: Pivot-first partitioning with overshoot: if the "< pivot" bucket is small enough,
//!   sort+emit it; otherwise repartition the left bucket.
//!
//! Notes:
//! - Keys are `u64`; "best" = smallest.
//! - We always fully sort what we emit (so the consumer can pop sequentially).
//! - We intentionally allow overshoot (emit more than immediately needed), since in the real
//!   pipeline we don't know the termination point.
//!
//! Run (example):
//!   cargo run --release --bin microbench_packed_select -- --lens=18,20,22 --target-ms=500
//!   cargo run --release --bin microbench_packed_select -- --algo=pivot --pivot-emit-max=16

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Config {
    lens: Vec<usize>,
    algo: AlgoKind,
    cases: usize,
    warmup_iters: usize,
    iters: usize,
    repeats: usize,
    warmup_ms: Option<u64>,
    target_ms: Option<u64>,
    seed: u64,
    bucket: usize,
    lookahead: usize,
    // Pivot-first knobs
    pivot_emit_max: usize,
    // Need distribution
    need_const: Option<usize>,
    need_weights: Vec<(usize, u32)>,
    show_counts: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lens: (14..=24).collect(),
            algo: AlgoKind::All,
            cases: 2048,
            warmup_iters: 25_000,
            iters: 250_000,
            repeats: 21,
            warmup_ms: None,
            target_ms: None,
            seed: 42,
            bucket: 8,
            lookahead: 0,
            pivot_emit_max: 16,
            need_const: None,
            need_weights: default_need_weights_neighbors_detail(),
            show_counts: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlgoKind {
    All,
    Select8,
    Select16,
    Pivot,
    PivotMin2,
    SortAll,
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

#[derive(Debug, Default, Clone, Copy)]
struct Counters {
    partitions: u64,
    selects: u64,
    sorts: u64,
    emitted: u64,
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

#[inline(always)]
fn ceil_to_bucket(x: usize, bucket: usize) -> usize {
    if bucket == 0 {
        return x;
    }
    let q = x / bucket;
    let r = x % bucket;
    if r == 0 {
        x
    } else {
        (q + 1) * bucket
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    for arg in std::env::args().skip(1) {
        let mut split = arg.splitn(2, '=');
        let key = split.next().unwrap_or("");
        let val = split.next();

        match (key, val) {
            ("--lens", Some(v)) => {
                cfg.lens = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>().expect("invalid --lens entry"))
                    .collect();
            }
            ("--algo", Some(v)) => {
                cfg.algo = match v {
                    "all" => AlgoKind::All,
                    "select8" => AlgoKind::Select8,
                    "select16" => AlgoKind::Select16,
                    "pivot" => AlgoKind::Pivot,
                    "pivot-min2" => AlgoKind::PivotMin2,
                    "sort-all" => AlgoKind::SortAll,
                    _ => panic!(
                        "invalid --algo (use all|select8|select16|pivot|pivot-min2|sort-all)"
                    ),
                };
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
            ("--bucket", Some(v)) => cfg.bucket = v.parse().expect("invalid --bucket"),
            ("--lookahead", Some(v)) => cfg.lookahead = v.parse().expect("invalid --lookahead"),
            ("--pivot-emit-max", Some(v)) => {
                cfg.pivot_emit_max = v.parse().expect("invalid --pivot-emit-max")
            }
            ("--need", Some(v)) => cfg.need_const = Some(v.parse().expect("invalid --need")),
            ("--need-weights", Some(v)) => cfg.need_weights = parse_need_weights(v),
            ("--show-counts", None) => cfg.show_counts = true,
            ("--help", _) | ("-h", _) => {
                eprintln!(
                    "microbench_packed_select options:\n  \
--algo=all|select8|select16|pivot|pivot-min2|sort-all (default all)\n  \
--lens=14,15,... (comma list; default 14..24)\n  \
--cases=N (default 2048)\n  \
--warmup=N (default 25000)\n  \
--iters=N (default 250000)\n  \
--repeats=N (default 21)\n  \
--warmup-ms=N (duration-based warmup)\n  \
--target-ms=N (duration-based measurement; overrides --iters)\n  \
--seed=N (default 42)\n  \
--bucket=N (default 8)\n  \
--lookahead=N (default 0; adds to sampled need)\n  \
--need=N (constant need; overrides distribution)\n  \
--need-weights=8:482,9:135,... (weights; overrides default)\n  \
--pivot-emit-max=N (default 16)\n  \
--show-counts (print avg partitions/selects/emitted per trial)"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg: {arg} (use --help)");
                std::process::exit(2);
            }
        }
    }

    assert!(!cfg.lens.is_empty(), "--lens must be non-empty");
    assert!(cfg.cases > 0, "--cases must be > 0");
    assert!(cfg.iters > 0, "--iters must be > 0");
    assert!(cfg.repeats > 0, "--repeats must be > 0");
    assert!(cfg.bucket > 0, "--bucket must be > 0");
    assert!(
        cfg.pivot_emit_max >= cfg.bucket,
        "--pivot-emit-max must be >= --bucket"
    );
    if let Some(ms) = cfg.target_ms {
        assert!(ms > 0, "--target-ms must be > 0");
    }
    if let Some(ms) = cfg.warmup_ms {
        assert!(ms > 0, "--warmup-ms must be > 0");
    }
    if cfg.need_const.is_none() {
        assert!(
            !cfg.need_weights.is_empty(),
            "--need-weights must be non-empty if --need is not set"
        );
    }
    cfg
}

fn parse_need_weights(s: &str) -> Vec<(usize, u32)> {
    let mut out = Vec::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let mut it = part.splitn(2, ':');
        let n = it
            .next()
            .unwrap_or("")
            .parse::<usize>()
            .expect("invalid --need-weights entry (n)");
        let w = it
            .next()
            .unwrap_or("")
            .parse::<u32>()
            .expect("invalid --need-weights entry (w)");
        if w > 0 {
            out.push((n, w));
        }
    }
    if out.is_empty() {
        panic!("--need-weights parsed to empty");
    }
    out
}

fn default_need_weights_neighbors_detail() -> Vec<(usize, u32)> {
    // From a representative run:
    //   neighbors_detail: 4=0.2% 5=0.8% 6=1.9% 7=3.1% 8=48.2% 9=13.5% 10=10.3%
    //                    11=7.5% 12=5.2% 13=3.5% 14=2.3% 15=1.4% 16=0.1%
    //                    17=1.3% 18=0.3% 19=0.2% 20=0.1%
    //
    // Store as integer weights (per 0.1%) to keep sampling cheap.
    vec![
        (4, 2),
        (5, 8),
        (6, 19),
        (7, 31),
        (8, 482),
        (9, 135),
        (10, 103),
        (11, 75),
        (12, 52),
        (13, 35),
        (14, 23),
        (15, 14),
        (16, 1),
        (17, 13),
        (18, 3),
        (19, 2),
        (20, 1),
    ]
}

fn generate_cases(len: usize, cases: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = vec![0u64; len * cases];
    for x in &mut out {
        *x = rng.gen();
    }
    out
}

fn generate_needs(cases: usize, len: usize, cfg: &Config, seed: u64) -> Vec<u16> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(cases);
    let dist = NeedDist::new(cfg.need_const, &cfg.need_weights);
    for _ in 0..cases {
        let mut need = dist.sample(&mut rng) + cfg.lookahead;
        need = need.min(len);
        out.push(need as u16);
    }
    out
}

struct NeedDist {
    cdf: Vec<(usize, u32)>,
    total: u32,
    constant: Option<usize>,
}

impl NeedDist {
    fn new(constant: Option<usize>, weights: &[(usize, u32)]) -> Self {
        if let Some(c) = constant {
            return Self {
                cdf: Vec::new(),
                total: 0,
                constant: Some(c),
            };
        }
        let mut cdf = Vec::with_capacity(weights.len());
        let mut sum = 0u32;
        for &(n, w) in weights {
            sum = sum.saturating_add(w);
            cdf.push((n, sum));
        }
        assert!(sum > 0);
        Self {
            cdf,
            total: sum,
            constant: None,
        }
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        if let Some(c) = self.constant {
            return c;
        }
        let r = rng.gen_range(0..self.total);
        // Linear scan is fine (<= ~32 entries).
        for &(n, acc) in &self.cdf {
            if r < acc {
                return n;
            }
        }
        self.cdf.last().map(|(n, _)| *n).unwrap_or(8)
    }
}

fn sort_prefix(v: &mut [u64], counters: &mut Option<&mut Counters>) {
    v.sort_unstable();
    if let Some(c) = counters.as_deref_mut() {
        c.sorts += 1;
    }
}

#[inline(always)]
fn partition_pivot_idx_lt(
    v: &mut [u64],
    pivot_idx: usize,
    counters: &mut Option<&mut Counters>,
) -> usize {
    debug_assert!(!v.is_empty());
    let n = v.len();
    if n == 1 {
        return 0;
    }
    debug_assert!(pivot_idx < n);
    if let Some(c) = counters.as_deref_mut() {
        c.partitions += 1;
    }

    let pivot = v[pivot_idx];
    v.swap(pivot_idx, n - 1);
    let mut store = 0usize;
    for i in 0..(n - 1) {
        if v[i] < pivot {
            v.swap(i, store);
            store += 1;
        }
    }
    v.swap(store, n - 1);
    store
}

#[inline(always)]
fn choose_pivot_idx_first(v: &[u64]) -> usize {
    let _ = v;
    0
}

#[inline(always)]
fn choose_pivot_idx_min2(v: &[u64]) -> usize {
    if v.len() < 2 {
        return 0;
    }
    if v[0] <= v[1] {
        0
    } else {
        1
    }
}

fn pivot_find_prefix_len(
    v: &mut [u64],
    min_emit: usize,
    max_emit: usize,
    counters: &mut Option<&mut Counters>,
    choose_pivot_idx: fn(&[u64]) -> usize,
) -> usize {
    debug_assert!(min_emit > 0);
    debug_assert!(max_emit >= min_emit);
    debug_assert!(max_emit <= v.len());

    if v.len() <= max_emit {
        // Emit all: no need to partition.
        return v.len();
    }

    let pivot_idx = partition_pivot_idx_lt(v, choose_pivot_idx(v), counters);
    let left_len = pivot_idx;

    // If the "< pivot" bucket is large, recurse into it.
    if left_len >= max_emit {
        return pivot_find_prefix_len(
            &mut v[..left_len],
            min_emit,
            max_emit,
            counters,
            choose_pivot_idx,
        );
    }

    // If "< pivot" is already enough and small-ish, emit it (overshoot allowed).
    if left_len >= min_emit {
        return left_len;
    }

    // Need to include the pivot itself, then fill from the right side.
    let taken = left_len + 1;
    if taken >= min_emit {
        return taken;
    }

    let extra = pivot_find_prefix_len(
        &mut v[taken..],
        min_emit - taken,
        max_emit - taken,
        counters,
        choose_pivot_idx,
    );
    taken + extra
}

fn run_select8(
    v: &mut [u64],
    need: usize,
    bucket: usize,
    counters: &mut Option<&mut Counters>,
) -> u64 {
    let mut produced = 0usize;
    let mut start = 0usize;
    let mut acc = 0u64;

    while produced < need && start < v.len() {
        let remaining = &mut v[start..];
        let emit = bucket.min(remaining.len());
        if emit == 0 {
            break;
        }

        if remaining.len() > emit {
            remaining.select_nth_unstable(emit - 1);
            if let Some(c) = counters.as_deref_mut() {
                c.selects += 1;
            }
        }

        sort_prefix(&mut remaining[..emit], counters);
        acc ^= remaining[0];
        acc ^= remaining[emit - 1];
        produced += emit;
        start += emit;
        if let Some(c) = counters.as_deref_mut() {
            c.emitted += emit as u64;
        }
    }
    acc
}

fn run_select16(
    v: &mut [u64],
    need: usize,
    bucket: usize,
    counters: &mut Option<&mut Counters>,
) -> u64 {
    let mut produced = 0usize;
    let mut start = 0usize;
    let mut cache_ready = 0usize;
    let mut acc = 0u64;

    while produced < need && start < v.len() {
        let remaining = &mut v[start..];
        if remaining.is_empty() {
            break;
        }

        if cache_ready > 0 {
            let emit = bucket.min(cache_ready).min(remaining.len());
            sort_prefix(&mut remaining[..emit], counters);
            acc ^= remaining[0];
            acc ^= remaining[emit - 1];
            produced += emit;
            start += emit;
            cache_ready -= emit;
            if let Some(c) = counters.as_deref_mut() {
                c.emitted += emit as u64;
            }
            continue;
        }

        let pref = (2 * bucket).min(remaining.len());
        let emit = bucket.min(pref);
        if emit == 0 {
            break;
        }

        if remaining.len() > pref {
            remaining.select_nth_unstable(pref - 1);
            if let Some(c) = counters.as_deref_mut() {
                c.selects += 1;
            }
        }

        if pref > emit {
            remaining[..pref].select_nth_unstable(emit - 1);
            if let Some(c) = counters.as_deref_mut() {
                c.selects += 1;
            }
        }

        sort_prefix(&mut remaining[..emit], counters);
        acc ^= remaining[0];
        acc ^= remaining[emit - 1];
        produced += emit;
        start += emit;
        cache_ready = pref - emit;
        if let Some(c) = counters.as_deref_mut() {
            c.emitted += emit as u64;
        }
    }
    acc
}

fn run_pivot(
    v: &mut [u64],
    need: usize,
    bucket: usize,
    emit_max: usize,
    counters: &mut Option<&mut Counters>,
    choose_pivot_idx: fn(&[u64]) -> usize,
) -> u64 {
    let mut produced = 0usize;
    let mut start = 0usize;
    let mut acc = 0u64;

    while produced < need && start < v.len() {
        let remaining = &mut v[start..];
        if remaining.is_empty() {
            break;
        }
        if remaining.len() <= bucket {
            sort_prefix(remaining, counters);
            acc ^= remaining[0];
            acc ^= remaining[remaining.len() - 1];
            if let Some(c) = counters.as_deref_mut() {
                c.emitted += remaining.len() as u64;
            }
            break;
        }

        let max_emit = emit_max.min(remaining.len());
        let emit = pivot_find_prefix_len(remaining, bucket, max_emit, counters, choose_pivot_idx)
            .min(max_emit);
        sort_prefix(&mut remaining[..emit], counters);
        acc ^= remaining[0];
        acc ^= remaining[emit - 1];
        produced += emit;
        start += emit;
        if let Some(c) = counters.as_deref_mut() {
            c.emitted += emit as u64;
        }
    }

    acc
}

fn run_sort_all(
    v: &mut [u64],
    need: usize,
    bucket: usize,
    counters: &mut Option<&mut Counters>,
) -> u64 {
    if v.is_empty() || need == 0 {
        return 0;
    }
    sort_prefix(v, counters);
    let produced = ceil_to_bucket(need.min(v.len()), bucket).min(v.len());
    if let Some(c) = counters.as_deref_mut() {
        c.emitted += produced as u64;
    }
    let mut acc = 0u64;
    acc ^= v[0];
    acc ^= v[produced - 1];
    acc
}

fn benchmark_cases<F>(
    cases_flat: &[u64],
    needs: &[u16],
    len: usize,
    mut f: F,
    warmup_iters: usize,
    iters: usize,
    repeats: usize,
) -> Stats
where
    F: FnMut(&mut [u64], usize) -> u64,
{
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);
    assert_eq!(needs.len(), num_cases);

    let mut buf = vec![0u64; len];

    // Warmup.
    let warmup_iters = warmup_iters.min(iters);
    let mut acc = 0u64;
    for i in 0..warmup_iters {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        let need = needs[case_idx] as usize;
        acc ^= f(&mut buf, need);
    }
    black_box(acc);

    let mut samples = Vec::with_capacity(repeats);
    let mut min_ns = f64::INFINITY;
    let mut max_ns: f64 = 0.0;

    for r in 0..repeats {
        let start_t = Instant::now();
        let mut acc = r as u64;
        for i in 0..iters {
            let case_idx = (i + r) % num_cases;
            let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
            buf.copy_from_slice(src);
            let need = needs[case_idx] as usize;
            acc ^= f(&mut buf, need);
        }
        black_box(acc);
        let elapsed = start_t.elapsed();
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

fn benchmark_cases_timed<F>(
    cases_flat: &[u64],
    needs: &[u16],
    len: usize,
    mut f: F,
    warmup_ms: u64,
    target_ms: u64,
    repeats: usize,
) -> Stats
where
    F: FnMut(&mut [u64], usize) -> u64,
{
    let num_cases = cases_flat.len() / len;
    assert!(num_cases > 0);
    assert_eq!(cases_flat.len(), num_cases * len);
    assert_eq!(needs.len(), num_cases);

    let mut buf = vec![0u64; len];

    let warmup_dur = std::time::Duration::from_millis(warmup_ms);
    let target_dur = std::time::Duration::from_millis(target_ms);

    // Duration warmup.
    let mut acc = 0u64;
    let start = Instant::now();
    let mut i = 0usize;
    while start.elapsed() < warmup_dur {
        let case_idx = i % num_cases;
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        let need = needs[case_idx] as usize;
        acc ^= f(&mut buf, need);
        i += 1;
    }
    black_box(acc);

    let mut samples = Vec::with_capacity(repeats);
    let mut min_ns = f64::INFINITY;
    let mut max_ns: f64 = 0.0;

    for rep in 0..repeats {
        let start_t = Instant::now();
        let mut acc = rep as u64;
        let mut iters = 0usize;
        while start_t.elapsed() < target_dur {
            let case_idx = (iters + rep) % num_cases;
            let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
            buf.copy_from_slice(src);
            let need = needs[case_idx] as usize;
            acc ^= f(&mut buf, need);
            iters += 1;
        }
        black_box(acc);

        if iters == 0 {
            iters = 1;
        }
        let elapsed = start_t.elapsed();
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

fn simulate_counts_for_algo<F>(cases_flat: &[u64], needs: &[u16], len: usize, mut f: F) -> Counters
where
    F: FnMut(&mut [u64], usize, &mut Counters) -> u64,
{
    let num_cases = cases_flat.len() / len;
    let mut buf = vec![0u64; len];
    let mut counters = Counters::default();
    let mut acc = 0u64;
    for case_idx in 0..num_cases {
        let src = &cases_flat[case_idx * len..(case_idx + 1) * len];
        buf.copy_from_slice(src);
        let need = needs[case_idx] as usize;
        acc ^= f(&mut buf, need, &mut counters);
    }
    black_box(acc);
    counters
}

fn main() {
    let cfg = parse_args();

    println!("Packed-Select Microbench (u64 keys)");
    println!(
        "lens={:?} bucket={} pivot_emit_max={} cases={} repeats={}",
        cfg.lens, cfg.bucket, cfg.pivot_emit_max, cfg.cases, cfg.repeats
    );
    if let Some(n) = cfg.need_const {
        println!("need=const({}) lookahead={}", n, cfg.lookahead);
    } else {
        let total: u32 = cfg.need_weights.iter().map(|(_, w)| *w).sum();
        println!(
            "need=dist(entries={}, total_weight={}) lookahead={}",
            cfg.need_weights.len(),
            total,
            cfg.lookahead
        );
    }
    if let Some(target_ms) = cfg.target_ms {
        let warmup_ms = cfg.warmup_ms.unwrap_or(200);
        println!("warmup_ms={} target_ms={}", warmup_ms, target_ms);
    } else {
        println!(
            "warmup_iters={} iters={} repeats={}",
            cfg.warmup_iters, cfg.iters, cfg.repeats
        );
    }
    println!();

    println!(
        "{:>4} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7} {:>11} {:>7} {:>9} {:>7}",
        "N",
        "select8",
        "jit",
        "select16",
        "jit",
        "pivot",
        "jit",
        "pivot_min2",
        "jit",
        "sort_all",
        "jit"
    );
    println!("{:-<80}", "");

    for &len in &cfg.lens {
        let seed = cfg.seed ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let cases = generate_cases(len, cfg.cases, seed);
        let needs = generate_needs(cfg.cases, len, &cfg, seed ^ 0xD1B5_4A32_19E3_7C11);

        let do_timed = cfg.target_ms.is_some();
        let (warmup_ms, target_ms) = (cfg.warmup_ms.unwrap_or(200), cfg.target_ms.unwrap_or(0));

        let mut s8 = Stats {
            median_ns: 0.0,
            min_ns: 0.0,
            max_ns: 0.0,
        };
        let mut s16 = s8;
        let mut sp = s8;
        let mut sp2 = s8;
        let mut sa = s8;

        let mut printed_any = false;

        if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::Select8 {
            let bucket = cfg.bucket;
            let mut none: Option<&mut Counters> = None;
            let fun = |buf: &mut [u64], need: usize| run_select8(buf, need, bucket, &mut none);
            s8 = if do_timed {
                benchmark_cases_timed(&cases, &needs, len, fun, warmup_ms, target_ms, cfg.repeats)
            } else {
                benchmark_cases(
                    &cases,
                    &needs,
                    len,
                    fun,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                )
            };
            printed_any = true;
        }
        if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::Select16 {
            let bucket = cfg.bucket;
            let mut none: Option<&mut Counters> = None;
            let fun = |buf: &mut [u64], need: usize| run_select16(buf, need, bucket, &mut none);
            s16 = if do_timed {
                benchmark_cases_timed(&cases, &needs, len, fun, warmup_ms, target_ms, cfg.repeats)
            } else {
                benchmark_cases(
                    &cases,
                    &needs,
                    len,
                    fun,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                )
            };
            printed_any = true;
        }
        if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::Pivot {
            let bucket = cfg.bucket;
            let emit_max = cfg.pivot_emit_max;
            let mut none: Option<&mut Counters> = None;
            let fun = |buf: &mut [u64], need: usize| {
                run_pivot(
                    buf,
                    need,
                    bucket,
                    emit_max,
                    &mut none,
                    choose_pivot_idx_first,
                )
            };
            sp = if do_timed {
                benchmark_cases_timed(&cases, &needs, len, fun, warmup_ms, target_ms, cfg.repeats)
            } else {
                benchmark_cases(
                    &cases,
                    &needs,
                    len,
                    fun,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                )
            };
            printed_any = true;
        }
        if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::PivotMin2 {
            let bucket = cfg.bucket;
            let emit_max = cfg.pivot_emit_max;
            let mut none: Option<&mut Counters> = None;
            let fun = |buf: &mut [u64], need: usize| {
                run_pivot(
                    buf,
                    need,
                    bucket,
                    emit_max,
                    &mut none,
                    choose_pivot_idx_min2,
                )
            };
            sp2 = if do_timed {
                benchmark_cases_timed(&cases, &needs, len, fun, warmup_ms, target_ms, cfg.repeats)
            } else {
                benchmark_cases(
                    &cases,
                    &needs,
                    len,
                    fun,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                )
            };
            printed_any = true;
        }
        if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::SortAll {
            let bucket = cfg.bucket;
            let mut none: Option<&mut Counters> = None;
            let fun = |buf: &mut [u64], need: usize| run_sort_all(buf, need, bucket, &mut none);
            sa = if do_timed {
                benchmark_cases_timed(&cases, &needs, len, fun, warmup_ms, target_ms, cfg.repeats)
            } else {
                benchmark_cases(
                    &cases,
                    &needs,
                    len,
                    fun,
                    cfg.warmup_iters,
                    cfg.iters,
                    cfg.repeats,
                )
            };
            printed_any = true;
        }

        if !printed_any {
            continue;
        }

        println!(
            "{:>4} {:>9.1} {:>6.1}% {:>9.1} {:>6.1}% {:>9.1} {:>6.1}% {:>11.1} {:>6.1}% {:>9.1} {:>6.1}%",
            len,
            s8.median_ns,
            s8.jitter_pct(),
            s16.median_ns,
            s16.jitter_pct(),
            sp.median_ns,
            sp.jitter_pct(),
            sp2.median_ns,
            sp2.jitter_pct(),
            sa.median_ns,
            sa.jitter_pct()
        );

        if cfg.show_counts {
            let denom = cfg.cases as f64;
            if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::Select8 {
                let c = simulate_counts_for_algo(&cases, &needs, len, |buf, need, ctr| {
                    let mut opt: Option<&mut Counters> = Some(ctr);
                    run_select8(buf, need, cfg.bucket, &mut opt)
                });
                println!(
                    "      counts select8: selects={:.2} sorts={:.2} emitted={:.2}",
                    c.selects as f64 / denom,
                    c.sorts as f64 / denom,
                    c.emitted as f64 / denom
                );
            }
            if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::Select16 {
                let c = simulate_counts_for_algo(&cases, &needs, len, |buf, need, ctr| {
                    let mut opt: Option<&mut Counters> = Some(ctr);
                    run_select16(buf, need, cfg.bucket, &mut opt)
                });
                println!(
                    "      counts select16: selects={:.2} sorts={:.2} emitted={:.2}",
                    c.selects as f64 / denom,
                    c.sorts as f64 / denom,
                    c.emitted as f64 / denom
                );
            }
            if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::Pivot {
                let c = simulate_counts_for_algo(&cases, &needs, len, |buf, need, ctr| {
                    let mut opt: Option<&mut Counters> = Some(ctr);
                    run_pivot(
                        buf,
                        need,
                        cfg.bucket,
                        cfg.pivot_emit_max,
                        &mut opt,
                        choose_pivot_idx_first,
                    )
                });
                println!(
                    "      counts pivot: partitions={:.2} sorts={:.2} emitted={:.2}",
                    c.partitions as f64 / denom,
                    c.sorts as f64 / denom,
                    c.emitted as f64 / denom
                );
            }
            if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::PivotMin2 {
                let c = simulate_counts_for_algo(&cases, &needs, len, |buf, need, ctr| {
                    let mut opt: Option<&mut Counters> = Some(ctr);
                    run_pivot(
                        buf,
                        need,
                        cfg.bucket,
                        cfg.pivot_emit_max,
                        &mut opt,
                        choose_pivot_idx_min2,
                    )
                });
                println!(
                    "      counts pivot_min2: partitions={:.2} sorts={:.2} emitted={:.2}",
                    c.partitions as f64 / denom,
                    c.sorts as f64 / denom,
                    c.emitted as f64 / denom
                );
            }
            if cfg.algo == AlgoKind::All || cfg.algo == AlgoKind::SortAll {
                let c = simulate_counts_for_algo(&cases, &needs, len, |buf, need, ctr| {
                    let mut opt: Option<&mut Counters> = Some(ctr);
                    run_sort_all(buf, need, cfg.bucket, &mut opt)
                });
                println!(
                    "      counts sort_all: sorts={:.2} emitted={:.2}",
                    c.sorts as f64 / denom,
                    c.emitted as f64 / denom
                );
            }
        }
    }
}
