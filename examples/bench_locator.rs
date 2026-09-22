//! Deterministic benchmark of canonical point-location queries.
//!
//! Arguments: GENERATORS QUERIES REPEATS [uniform|fib|clustered].
//! Defaults: 100000 250000 4 uniform. Queries are uniformly distributed;
//! `clustered` puts 80% of generators in a cap around +Z.
//!
//! Build and collect single-worker counters on an available CPU:
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --example bench_locator
//! RAYON_NUM_THREADS=1 perf stat -e '{instructions:u,branches:u,cycles:u}' -- \
//!   taskset -c 4 target/release/examples/bench_locator 100000 250000 4 uniform
//! ```
//! Reported `ns_per_query` covers only repeated `locate_point` calls and checksum
//! accumulation. Process-wide perf counters also include point/query generation,
//! diagram construction, and locator construction. Compare matching checksums and
//! alternate baseline/candidate runs; use equal-length executable paths.

use std::hint::black_box;
use std::time::Instant;
use voronoi_mesh::{compute, SpherePoint};

fn random_point(state: &mut u64) -> SpherePoint {
    let mut next = || {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f64 / (1u64 << 24) as f64
    };
    let z = 2.0 * next() - 1.0;
    let theta = std::f64::consts::TAU * next();
    let r = (1.0 - z * z).sqrt();
    SpherePoint::try_from_xyz([(r * theta.cos()) as f32, (r * theta.sin()) as f32, z as f32])
        .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let nq: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(250_000);
    let repeats: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(4);
    let dist = args.get(4).map(String::as_str).unwrap_or("uniform");
    assert!(
        args.len() <= 5,
        "expected GENERATORS QUERIES REPEATS DISTRIBUTION"
    );
    assert!(n > 0 && nq > 0 && repeats > 0, "counts must be positive");
    let total_queries = nq.checked_mul(repeats).expect("query count overflow");
    match dist {
        "uniform" | "fib" | "clustered" => {}
        _ => panic!("unknown distribution {dist:?}; expected uniform, fib, or clustered"),
    }
    let clustered_count = n - n.div_ceil(5);
    let mut state = 12345;
    let points: Vec<_> = (0..n)
        .map(|i| match dist {
            "fib" => {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let theta = i as f64 * std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
                let r = (1.0 - z * z).sqrt();
                SpherePoint::try_from_xyz([
                    (r * theta.cos()) as f32,
                    (r * theta.sin()) as f32,
                    z as f32,
                ])
                .unwrap()
            }
            "clustered" if i < clustered_count => {
                let p = random_point(&mut state).to_array();
                SpherePoint::try_from_xyz([0.1 * p[0], 0.1 * p[1], 1.0 + 0.1 * p[2]]).unwrap()
            }
            "uniform" | "clustered" => random_point(&mut state),
            _ => unreachable!("distribution checked above"),
        })
        .collect();
    let diagram = compute(&points).unwrap();
    let mut locator = diagram.build_locator();
    state = 98765;
    let queries: Vec<_> = (0..nq).map(|_| random_point(&mut state)).collect();
    let mut checksum = 0u64;
    let t = Instant::now();
    for _ in 0..repeats {
        for &q in &queries {
            checksum = checksum
                .rotate_left(7)
                .wrapping_add(black_box(locator.locate_point(black_box(q))) as u64);
        }
    }
    println!("sites={n} queries={nq} repeats={repeats} dist={dist} ns_per_query={:.3} checksum={checksum:016x}", t.elapsed().as_nanos() as f64 / total_queries as f64);
}
