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

The backends deliberately span different algorithm families:

- `bench_compare`: this crate's complete deduplicated spherical diagram.
- `bench_cgal_sphere`: CGAL incremental Delaunay triangulation on the sphere.
- `bench_qhull_sphere`: Qhull 3D convex hull, dual to spherical Delaunay.
- `bench-stripack-sphere`: classical incremental spherical Delaunay via STRIPACK.
- `bench_vortex_sphere`: Vortex's kNN-driven spherical clipping, the closest
  algorithm-family comparison.

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

# Pins Vortex commit 3d59c66, applies the documented headless/libMeshb-v8
# compatibility patch, and builds a native adapter with capacity for 16 workers.
benchmarks/competitors/build_vortex.sh

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
OMP_NUM_THREADS=16 target/competitors/vortex-make-t16/bin/bench_vortex_sphere \
  target/competitors/data/fib-10k.f32 --threads 16 --neighbors 50 --repeat 3
```

Vortex's normal sphere CLI Morton-orders sites and uses its sphere quadtree, so
the adapter does the same. Reordering, quadtree construction, clipping, and
mandatory per-cell property calculation are all inside `construct_ms`; binary
input loading and f32-to-f64 promotion are outside it. `vortex-construct`
disables optional mesh/facet storage. Backend `vortex` additionally selects
`--full`, which stores Vortex's per-cell polygon mesh. That representation
duplicates boundary vertices per cell and includes the input sites, rather than
assembling the shared-vertex cell mesh returned by this crate, so report it as
a separate materialization boundary rather than an output-equivalent total.

Upstream fixes its clipping worker capacity at compile time and otherwise uses
the host's logical CPU count. The adapter therefore supports the two controlled
configurations needed for the initial gate: `--threads 1` disables parallel
clipping, while `--threads 16` uses the pinned 16-worker build. Set
`OMP_NUM_THREADS` to the same value for Vortex's quadtree loops. Intermediate
thread counts require separately compiled capacities and are deliberately
rejected instead of silently oversubscribing the affinity mask.

The pinned Vortex checkout currently needs `vortex.patch`: its upstream CMake
tracks libMeshb's moving default branch but still names the removed v7 source
and header, and its Ninja helper targets contain unescaped Make syntax. The
patch disables unrelated visualization, updates only the libMeshb filenames,
adds the adapter target, permits the fixed 16-worker capacity, and applies
native release code generation. The build uses Unix Makefiles and enables
whole-program LTO; the crate's own release profile does not use LTO. No Vortex
geometry or neighbor-search source is changed.

Vortex clips against a fixed initial candidate budget. The adapter preserves
its upstream default of 50 with `--neighbors 50`; the campaign runner exposes
the same choice as `--vortex-neighbors`. A lower budget is only a valid tuning
result when `failures=0`, because Vortex reports cells whose candidate set was
insufficient. Tune against every reported distribution and size, and retain the
failure count with the timing data.

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

### Current physical-core scaling curve

After disabling the motherboard's default PBO overclock, a fresh seven-round
rotated-order run measured the current native build at commit `2774e04`. Each
point used the first `T` physical cores from CPU 0 through CPU `T - 1`; SMT
siblings 16--31 were excluded. Inputs were the same cached unique 2.5M-site
fixtures used by the competitor campaign, with preprocessing disabled and file
loading outside the timed construction region.

| workers | Fibonacci construct | speedup (95% paired bootstrap CI) | uniform construct | speedup (95% paired bootstrap CI) |
|---:|---:|---:|---:|---:|
| 1 | 1,473.8 ms | 1.00x | 2,060.4 ms | 1.00x |
| 2 | 787.5 ms | 1.87x (1.86--1.88) | 1,071.9 ms | 1.92x (1.91--1.94) |
| 4 | 450.4 ms | 3.28x (3.26--3.32) | 596.3 ms | 3.46x (3.44--3.48) |
| 8 | 270.5 ms | 5.45x (5.43--5.48) | 382.0 ms | 5.41x (5.33--5.47) |
| 12 | 237.0 ms | 6.26x (6.00--6.41) | 315.3 ms | 6.50x (5.96--7.14) |
| 16 | 210.8 ms | 6.99x (6.87--7.14) | 267.5 ms | 7.73x (7.66--7.87) |

The uniform 12-worker samples were unusually variable (290.8--349.9 ms), but
the 16-worker samples tightened again (260.0--280.8 ms). This does not indicate
a terminal memory-bandwidth plateau: both distributions continue improving
from 8 to 16 physical cores, although efficiency naturally falls as serial
materialization and memory traffic become larger fractions of elapsed time.

Using the retained seven-round serial-CGAL medians as reference, current
16-worker construction is 7.70x faster on Fibonacci and 6.22x faster on
uniform. The qualification above still applies: CGAL retains a spherical
triangulation, whereas this crate constructs the shared Voronoi-cell mesh.
Raw measurements are retained in
`target/competitors/results/s2-current-2.5m-scaling.raw`.

## Vortex same-family comparison

The pinned Vortex adapter was first gated at 100k and 500k sites using its
upstream-default 50-neighbor candidate budget and without LTO. These runs
compare this crate's complete shared diagram against `vortex-construct`, which
includes Vortex's Morton ordering, sphere-quadtree construction, clipping, and
mandatory per-cell properties but deliberately omits its optional duplicated
polygon mesh. Consequently, a win for this crate is conservative with respect
to completed output work, though the representations and numerical policies
remain different.

| sites | input | workers | voronoi-mesh | Vortex construct-only | Vortex / voronoi-mesh paired geomean |
|---:|---|---:|---:|---:|---:|
| 100k | Fibonacci | 1 | 59.6 ms | 562.2 ms | 9.43x (95% CI 9.36--9.48) |
| 100k | uniform | 1 | 81.7 ms | 737.0 ms | 9.02x (8.95--9.09) |
| 100k | Fibonacci | 16 | 15.9 ms | 98.3 ms | 6.01x (5.52--6.49) |
| 100k | uniform | 16 | 20.5 ms | 105.2 ms | 5.20x (4.96--5.42) |
| 500k | Fibonacci | 1 | 295.0 ms | 3,047.6 ms | 10.34x (10.32--10.35) |
| 500k | uniform | 1 | 398.4 ms | 3,910.4 ms | 9.79x (9.73--9.84) |
| 500k | Fibonacci | 16 | 49.6 ms | 368.8 ms | 7.43x (6.84--8.02) |
| 500k | uniform | 16 | 60.7 ms | 431.5 ms | 7.07x (6.85--7.29) |

All Vortex runs reported zero incomplete cells. At 500k, Vortex retired about
30.0B/38.0B instructions on Fibonacci/uniform versus this crate's 3.4--3.6B/
3.9--4.1B across the thread-count variants. The roughly 8.8--9.8x work ratio
is close to the serial time ratio, while Vortex's stronger small-size parallel
scaling narrows the wall-time gap at 16 workers. The result therefore points
primarily to candidate-search and per-cell work efficiency, not simply thread
scheduling or language/runtime overhead.

Median peak RSS at 500k was 162.6/162.7 MiB for serial Vortex versus
127.8/131.2 MiB for serial `voronoi-mesh`; at 16 workers it was 161.5/161.6 MiB
versus 145.7/148.1 MiB. The raw CSV files are retained under
`target/competitors/results/vortex-{fib,uniform}-{100k,500k}-t{1,16}.csv`.

### Tuned fairness audit

A follow-up audit varied Vortex's candidate budget. At 500k uniform sites, 33
neighbors left two incomplete cells, while 34 was the smallest tested budget
with no failures; Fibonacci completed at still smaller budgets. We therefore
used 34 for both inputs rather than selecting a distribution-specific value.
We also enabled whole-program LTO for Vortex, which improved its serial result
by 3.53% on both distributions in seven rotated pairs. This is deliberately
favorable to the competitor: `voronoi-mesh` remained at its normal non-LTO
release configuration.

| sites | input | workers | voronoi-mesh | tuned Vortex construct-only | Vortex / voronoi-mesh paired geomean |
|---:|---|---:|---:|---:|---:|
| 500k | Fibonacci | 1 | 294.9 ms | 2,602.4 ms | 8.77x (95% CI 8.68--8.83) |
| 500k | uniform | 1 | 398.7 ms | 3,494.6 ms | 8.80x (8.76--8.85) |
| 500k | Fibonacci | 16 | 51.2 ms | 323.6 ms | 6.55x (6.25--6.83) |
| 500k | uniform | 16 | 62.8 ms | 397.0 ms | 6.03x (5.62--6.43) |

All tuned runs reported zero failures. Median retired instructions were
27.0B/35.0B for serial Vortex versus 3.41B/3.87B for this crate on
Fibonacci/uniform; the 16-worker counts were essentially unchanged. Median
peak RSS was 131.5/131.5 MiB for serial Vortex versus 127.9/131.3 MiB for this
crate. At 16 workers, Vortex used 130.4/130.3 MiB versus 146.1/148.2 MiB for
this crate. The tuned raw CSV files are retained under
`target/competitors/results/vortex-tuned-lto-{fib,uniform}-500k-t{1,16}.csv`.
