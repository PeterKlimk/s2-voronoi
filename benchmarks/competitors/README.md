# Spherical Voronoi competitor harness

This harness asks whether local kNN-driven clipping pays relative to global
convex-hull and incremental spherical-Delaunay construction.

All programs read the same headerless binary input: packed little-endian
`f32` triples `(x, y, z)`. Input generation and loading are outside the timed
region. Every result is one machine-readable `RESULT key=value...` line.
The generator deterministically resamples the vanishingly rare sites that
collide after conversion to packed `f32`; the repair count is printed when a
dataset is created. The campaign runner also rejects stale cached inputs with
duplicate triples.

The three initial backends are deliberately different algorithm families:

- `bench_compare`: this crate's complete deduplicated spherical diagram.
- `bench_cgal_sphere`: CGAL incremental Delaunay triangulation on the sphere.
- `bench_qhull_sphere`: Qhull 3D convex hull, dual to spherical Delaunay.
- `bench-stripack-sphere`: classical incremental spherical Delaunay via STRIPACK.

`construct_ms` is the native construction call. `materialize_ms` traverses the
result, constructs spherical dual points where needed, counts incidences, and
computes a checksum. CGAL and Qhull retain triangulations rather than the exact
shared-cell representation produced by this crate, so neither timing alone is
an output-equivalent headline. Report both construction-only and total results.

## Build and smoke test

```bash
cmake -S benchmarks/competitors -B target/competitors/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build target/competitors/build

RUSTFLAGS="-C target-cpu=native" cargo build --release --features tools \
  --bin bench_compare --target-dir target/competitors/rust
RUSTFLAGS="-C target-cpu=native" cargo build --release \
  --manifest-path benchmarks/competitors/stripack-runner/Cargo.toml \
  --target-dir target/competitors/stripack

python3 benchmarks/competitors/generate_points.py \
  target/competitors/data/fib-10k.f32 10k --dist fib

target/competitors/rust/release/bench_compare \
  target/competitors/data/fib-10k.f32 --repeat 3
target/competitors/build/bench_cgal_sphere \
  target/competitors/data/fib-10k.f32 --repeat 3
target/competitors/build/bench_qhull_sphere \
  target/competitors/data/fib-10k.f32 --repeat 3
target/competitors/stripack/release/bench-stripack-sphere \
  target/competitors/data/fib-10k.f32 --repeat 3
```

The STRIPACK wrapper requires a `f64` unit norm within `1e-10`, which ordinary
packed `f32` vectors cannot satisfy. Its input adapter therefore promotes each
shared `f32` triple and renormalizes it in `f64` before timing. No point is
reordered or otherwise changed. STRIPACK is documented as expected
`O(n log n)` for randomly ordered inputs but potentially `O(n²)` for ordered
latitude inputs. Treat ordered Fibonacci results as an ordering stress case;
use the uniform dataset for its expected-complexity comparison.

The wrapper's `voronoi_cells()` extraction is itself extremely expensive: in
the initial 100k uniform smoke test, construction took about 149 ms while
triangle/circumcenter extraction took 7.64 s. The campaign therefore defaults
to `stripack-construct`, which reports triangulation construction plus cheap
structural counts. Select backend `stripack` explicitly for full extraction at
small sizes. This distinction must remain visible in reported results.

For controlled runs, place `taskset`, `/usr/bin/time -v`, and `perf stat`
outside the benchmark command. Do not change the machine's known-reliable CPU
governor. Fibonacci and uniform datasets should be tested separately.

The campaign runner automates warmups, rotated backend order, affinity, `perf`
counters, peak RSS, and raw CSV capture:

```bash
# Controlled single-thread algorithm comparison.
python3 benchmarks/competitors/run_campaign.py \
  --dist fib --threads 1 --cpus 0 --rounds 7

# Repeat separately for random data.
python3 benchmarks/competitors/run_campaign.py \
  --dist uniform --threads 1 --cpus 0 --rounds 7

# Best-available comparison: competitors remain serial; only s2 uses 16 threads.
python3 benchmarks/competitors/run_campaign.py \
  --dist fib --threads 16 --cpus 0-15 --rounds 7
```

Use `--sizes` to extend a run. The conservative defaults are `10k 100k`;
establish time and memory behavior before adding `500k`, `1m`, `2.5m`, or `5m`.
Raw results go under `target/competitors/results/` and are never committed.

Alternative native/static builds can be selected without replacing the
baseline artifacts using `--qhull-bin PATH` and `--stripack-bin PATH`.

Summarize medians, bootstrap 95% confidence intervals, and within-round paired
ratios without altering the raw CSV:

```bash
python3 benchmarks/competitors/analyze_campaign.py \
  target/competitors/results/fib-t1.csv --metric construct_ms
python3 benchmarks/competitors/analyze_campaign.py \
  target/competitors/results/fib-t1.csv --metric total_ms
```

## August 2026 native Linux result

A seven-round rotated campaign on the 16-core Ryzen host compared the current
native binaries at 500k, 1M, and 2.5M sites. CPUs 0--15 selected one logical CPU
per physical core. The 2.5M medians summarize the large-size result:

| input | backend | workers | construct | materialized total | peak RSS |
|---|---|---:|---:|---:|---:|
| Fibonacci | voronoi-mesh | 1 | 1,483 ms | 1,514 ms | 611 MiB |
| Fibonacci | voronoi-mesh | 16 | 210 ms | 243 ms | 654 MiB |
| Fibonacci | CGAL | 1 | 1,623 ms | 1,838 ms | 568 MiB |
| uniform | voronoi-mesh | 1 | 2,066 ms | 2,097 ms | 633 MiB |
| uniform | voronoi-mesh | 16 | 269 ms | 332 ms | 667 MiB |
| uniform | CGAL | 1 | 1,665 ms | 1,902 ms | 568 MiB |

At 2.5M, `voronoi-mesh` construction scaled by 7.05x on Fibonacci and 7.69x
on uniform from one to 16 physical cores. Against serial CGAL, its 16-worker
construction was 7.67x faster by paired geomean on Fibonacci (95% bootstrap
CI 7.56--7.77x) and 6.23x faster on uniform (6.16--6.28x). For materialized
total time the corresponding advantages were 7.54x and 5.75x.

The single-worker comparison is the important qualification. Clipping was
about 9% faster than CGAL construction on Fibonacci at 2.5M, but about 24%
slower on random uniform input. Once each retained structure was traversed and
dual points were constructed where needed, the uniform deficit narrowed to
about 10%. CGAL still retains a triangulation rather than the exact shared-cell
mesh produced by this crate, so these are useful algorithm-family and scaling
comparisons, not output-equivalent claims.

The initial 2.5M uniform fixture contained one pair of bit-identical sites after
conversion to `f32`. With preprocessing deliberately disabled, that correctly
failed as a duplicate-generator input. The generator and cache check now
guarantee unique packed triples; exactly one site was deterministically
resampled for this campaign.
