//! Microbenchmark: small-N sorting networks vs a branchless cmov insertion sort.
//!
//! Isolates the packed-kNN candidate sort. The networks (production `sort_small`)
//! are compared against a plain branchless insertion sort and `sort_unstable`.
//!
//! Designed for a noisy box: a fixed input pool is processed for a fixed number of
//! rounds, so `instructions:u` is deterministic and is the primary signal. Run each
//! impl under `perf stat` interleaved and compare instructions first, cycles second.
//!
//! Result (Ryzen 3600, random keys, N=8..32): the networks beat `sort_unstable` by
//! 11-37% instructions and 43-78% cycles. The cycle gap is the point — the keys are
//! essentially random, so a comparison sort's data-dependent branches mispredict
//! ~half the time, and that penalty dominates; the network is straight-line cmov and
//! pays none of it. A plain branchless insertion sort is NOT the cheap win it looks
//! like: it is O(n^2) cmov swaps, so at N=24..32 it runs 2-3x more instructions than
//! the networks and is slower than even `sort_unstable`. The networks are justified.
//!
//!   cargo build --release --features microbench --bin microbench_sort
//!   B=target/release/microbench_sort
//!   for N in 8 12 16 24 32; do
//!     for impl in net ins std; do
//!       taskset -c 2 perf stat -r 7 -e instructions,cycles $B $impl $N 2>&1 | ...
//!     done
//!   done
//!
//! Args: <net|ins|std> <N> [rounds]. Prints a checksum (must match across impls for
//! the same N) and wall time.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::hint::{black_box, select_unpredictable};
use std::time::Instant;

/// Branchless insertion sort: each element is sifted toward the front by adjacent
/// cmov compare-swaps, fixed trip count (no data-dependent branch). O(n^2) swaps.
#[inline]
fn ins_sort(v: &mut [u64]) {
    let n = v.len();
    for i in 1..n {
        let mut j = i;
        while j > 0 {
            // SAFETY: j in 1..=i < n, so j and j-1 are in bounds.
            unsafe {
                let pa = v.as_mut_ptr().add(j - 1);
                let pb = v.as_mut_ptr().add(j);
                let a = *pa;
                let b = *pb;
                let le = a <= b;
                *pa = select_unpredictable(le, a, b);
                *pb = select_unpredictable(le, b, a);
            }
            j -= 1;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("net");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4000);

    // A pool of independent N-length arrays, laid out flat. Keys mimic the real
    // packed key: (desc << 32) | idx, with desc != 0xFFFFFFFF so no key is u64::MAX
    // (the networks' padding sentinel). 4096 arrays ~= a few hundred KB at N=32.
    let pool_arrays = 4096usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0x5232_u64.wrapping_add(n as u64));
    let mut master: Vec<u64> = Vec::with_capacity(pool_arrays * n);
    for _ in 0..pool_arrays * n {
        let desc = rng.gen_range(0u32..0xFFFF_FFFFu32) as u64;
        let idx = rng.gen::<u32>() as u64;
        master.push((desc << 32) | idx);
    }

    let mut work = master.clone();
    let mut checksum = 0u64;

    let t = Instant::now();
    for _ in 0..rounds {
        // Restore the unsorted pool (common-mode memcpy across all impls), then sort
        // each segment with the selected implementation.
        work.copy_from_slice(&master);
        for seg in work.chunks_mut(n) {
            match which {
                "net" => s2_voronoi::bench_sort_small(seg),
                "ins" => ins_sort(seg),
                "std" => seg.sort_unstable(),
                other => panic!("unknown impl {other:?} (use net|ins|std)"),
            }
            // Fold the smallest element in to defeat dead-code elimination.
            checksum = checksum.wrapping_add(black_box(seg[0]));
        }
    }
    let dt = t.elapsed();

    println!(
        "impl={which} N={n} rounds={rounds} arrays={pool_arrays} \
         sorts={} checksum={checksum:016x} {:.3}s",
        rounds * pool_arrays,
        dt.as_secs_f64()
    );
}
