# Spherical Voronoi competitor harness

This harness asks whether local kNN-driven clipping pays relative to global
convex-hull and incremental spherical-Delaunay construction.

All programs read the same headerless binary input: packed little-endian
`f32` triples `(x, y, z)`. Input generation and loading are outside the timed
region. Every result is one machine-readable `RESULT key=value...` line.

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
