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
- `bench_fade2d_sphere`: Fade2D planar Delaunay over a stereographic reduction,
  with spherical topology recovered from the planar faces and outer hull.

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

# Optional proprietary Fade2D adapter. Download the official release separately,
# retain it below ignored target/, and configure only the Fade target:
cmake -S benchmarks/competitors -B target/competitors/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFADE2D_ROOT="$PWD/target/competitors/fade/fadeRelease_v2.17.3"
cmake --build target/competitors/build --target bench_fade2d_sphere

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
target/competitors/build/bench_fade2d_sphere \
  target/competitors/data/fib-10k.f32 --threads 1 --repeat 3
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

The Fade2D adapter is optional because the vendor archive and license are not
redistributable project dependencies. `FADE2D_ROOT` must name an extracted
official release containing `include_fade2d/Fade_2D.h`; CMake selects the
matching Linux library and does not build the adapter otherwise. The 2.17.3
student license permits non-commercial scientific research, caps 2D
triangulations at one million points, and rejects larger calls; commercial use
or a larger evaluation requires another license. Neither the archive nor
library is committed.

Backend `fade2d-stereo` renormalizes packed inputs in `f64`, removes one pole,
stereographically projects the remaining sites, calls Fade2D, then closes the
spherical triangulation with one face per planar convex-hull edge. Projection
and triangulation comprise `construct_ms`; face traversal, hull recovery,
spherical circumcenters, counts, and checksums comprise `materialize_ms`.
`fade2d-stereo-fast` runs the same path with Fade Fast Mode enabled.

Two Fade-only controls are not spherical competitor results. `fade2d-native`
triangulates normalized `(x,y)` coordinates directly to measure the
projection-conditioning cost. `fade2d-raster` and `fade2d-raster-fast` consume
the generator's exact integer `planar-grid` coordinates through `--raw-plane`;
they isolate the predicate-heavy raster case without spherical normalization
or projection.

Regular mode retains Fade's documented multiple-precision fallback when
required. Fast Mode disables that fallback and is intended primarily to avoid
frequent expensive predicates on grid-aligned raster data. The main Fibonacci
and continuous-uniform campaign inputs are not raster grids.

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

Alternative binaries can be selected without replacing baseline artifacts
using `--qhull-bin PATH`, `--stripack-bin PATH`, `--vortex-bin PATH`, and
`--fade2d-bin PATH`.

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

### Five- and ten-million local scaling

A separate seven-round campaign extended this crate alone to 5M and 10M sites.
It used the same native binary, physical-core affinity, packed distinct inputs,
and `perf`/RSS wrapper as the competitor runs. The primary path disabled
preprocessing because the generated inputs are already distinct; the production
weld pass was measured separately rather than silently included.

| sites | input | workers | construct | 1-to-T speedup | peak RSS |
|---:|---|---:|---:|---:|---:|
| 5M | Fibonacci | 1 | 3,015.8 ms | 1.00x | 1,122.7 MiB |
| 5M | Fibonacci | 4 | 1,061.2 ms | 2.84x | 1,123.4 MiB |
| 5M | Fibonacci | 8 | 683.5 ms | 4.41x | 1,123.9 MiB |
| 5M | Fibonacci | 16 | 470.8 ms | 6.35x (95% CI 6.09--6.62) | 1,150.5 MiB |
| 5M | uniform | 1 | 4,496.1 ms | 1.00x | 1,154.6 MiB |
| 5M | uniform | 4 | 1,452.8 ms | 3.09x | 1,139.7 MiB |
| 5M | uniform | 8 | 893.1 ms | 5.03x | 1,152.9 MiB |
| 5M | uniform | 16 | 656.0 ms | 6.89x (6.70--7.07) | 1,167.5 MiB |
| 10M | Fibonacci | 1 | 6,491.7 ms | 1.00x | 2,209.6 MiB |
| 10M | Fibonacci | 4 | 2,095.8 ms | 3.10x | 2,229.4 MiB |
| 10M | Fibonacci | 8 | 1,399.7 ms | 4.64x | 2,239.0 MiB |
| 10M | Fibonacci | 16 | 974.1 ms | 6.65x (6.47--6.87) | 2,274.5 MiB |
| 10M | uniform | 1 | 9,286.2 ms | 1.00x | 2,213.6 MiB |
| 10M | uniform | 4 | 3,022.8 ms | 3.07x | 2,281.9 MiB |
| 10M | uniform | 8 | 1,987.3 ms | 4.67x | 2,308.6 MiB |
| 10M | uniform | 16 | 1,374.4 ms | 6.50x (5.76--7.04) | 2,311.0 MiB |

From 5M to 10M, serial construction grew 2.15x on Fibonacci and 2.07x on
uniform; 16-worker construction grew 2.07x and 2.10x. Peak RSS remained close
to linear at about 220--245 bytes per generator. The output counts remained
complete in every run.

The weld pass found no mergeable sites in these fixtures. Its cost was
distribution-sensitive: effectively neutral on Fibonacci, but at 10M uniform
it raised 16-worker construction from 1,374.4 to 2,050.0 ms and peak RSS from
2,311.0 to 2,462.7 MiB. This is expected optional correctness work, not a
construction scaling defect. Raw files are
`target/competitors/results/local-scale-{fib,uniform}-{5m,10m}-t{1,4,8,16}.csv`.

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

The same tuned configuration was extended to 1M and 2.5M after both
distributions passed a 34-neighbor gate with zero incomplete cells:

| sites | input | workers | voronoi-mesh | tuned Vortex construct-only | Vortex / voronoi-mesh paired geomean |
|---:|---|---:|---:|---:|---:|
| 1M | Fibonacci | 1 | 643.7 ms | 6,864.3 ms | 10.49x (95% CI 10.29--10.69) |
| 1M | uniform | 1 | 871.5 ms | 8,698.3 ms | 10.02x (9.90--10.17) |
| 1M | Fibonacci | 16 | 102.8 ms | 919.1 ms | 8.62x (8.24--9.05) |
| 1M | uniform | 16 | 142.6 ms | 1,060.6 ms | 7.09x (5.91--8.06) |
| 2.5M | Fibonacci | 1 | 1,605.9 ms | 14,706.3 ms | 9.03x (8.65--9.27) |
| 2.5M | uniform | 1 | 2,276.5 ms | 19,604.3 ms | 8.61x (8.31--9.02) |
| 2.5M | Fibonacci | 16 | 238.5 ms | 1,905.7 ms | 8.00x (7.87--8.11) |
| 2.5M | uniform | 16 | 302.2 ms | 2,288.6 ms | 7.52x (7.36--7.64) |

All extended runs reported zero failures. At 2.5M, Vortex retired 149.4B/190.2B
instructions on Fibonacci/uniform versus 18.9B/21.5B here in serial; parallel
instruction counts were essentially unchanged. Peak RSS converged near
625--627 MiB for Vortex and 592--626 MiB for this crate. Raw files are
`target/competitors/results/vortex-tuned-lto-{fib,uniform}-{1m,2.5m}-t{1,16}.csv`.

## Fade2D stereographic comparison

The optional Fade2D 2.17.3 adapter was measured in seven-round rotated pairs on
CPU 0 for serial runs and physical CPUs 0--15 for parallel runs. Inputs were the
same packed, distinct `f32` Fibonacci and seeded-uniform sites used by the
campaign. Each spherical reconstruction produced exactly `2n-4` Delaunay faces
and `3(2n-4)` incidences with every generator degree at least three.

Construction medians:

| sites | input | workers | voronoi-mesh | Fade2D stereo | Fade / voronoi-mesh paired geomean |
|---:|---|---:|---:|---:|---:|
| 100k | Fibonacci | 1 | 64.6 ms | 31.8 ms | 0.49x (95% CI 0.48--0.50) |
| 500k | Fibonacci | 1 | 319.9 ms | 197.9 ms | 0.62x (0.61--0.63) |
| 1M | Fibonacci | 1 | 652.7 ms | 514.5 ms | 0.79x (0.74--0.85) |
| 100k | uniform | 1 | 86.0 ms | 36.5 ms | 0.46x (0.42--0.53) |
| 500k | uniform | 1 | 444.4 ms | 220.9 ms | 0.48x (0.45--0.51) |
| 1M | uniform | 1 | 877.9 ms | 659.8 ms | 0.75x (0.73--0.76) |
| 100k | Fibonacci | 16 | 16.7 ms | 32.7 ms | 2.02x (1.84--2.21) |
| 500k | Fibonacci | 16 | 56.2 ms | 82.6 ms | 1.45x (1.39--1.50) |
| 1M | Fibonacci | 16 | 98.0 ms | 148.1 ms | 1.52x (1.47--1.58) |
| 100k | uniform | 16 | 19.5 ms | 33.9 ms | 1.74x (1.69--1.79) |
| 500k | uniform | 16 | 67.3 ms | 82.2 ms | 1.22x (1.12--1.34) |
| 1M | uniform | 16 | 116.7 ms | 176.8 ms | 1.51x (1.43--1.59) |

One million sites is the largest comparison permitted by the student library.
A 5M uniform gate was attempted, but Fade rejected insertion with its license
exception before a timed sample could be recorded. Extending this scaling curve
to 5M or 10M therefore requires a commercial or extended evaluation license;
partitioning the input would change the algorithm and is not a valid workaround.

Serial Fade construction is faster at every measured size, but the gap narrows
with size because stereographic coordinates have a heavy tail near the removed
pole. On 1M uniform sites, the native planar control took 352.1 ms versus
659.8 ms for stereographic construction. This is a conditioning cost, not the
projection loop itself: its median was about 15 ms. At 16 workers this crate is
faster at every measured size. Fade's serial-to-16-worker construction speedup
at 1M was 3.47x on Fibonacci and 3.73x on uniform, versus 6.66x and 7.52x here.

### Predicate-mode audit

Fade's regular mode and Fast Mode were compared in seven rotated rounds at
100k, 500k, and 1M sites. The sphere-compatible set covered Fibonacci, uniform,
a nominal stereographic grid, and the same grid with `0.1%`-of-spacing jitter.
Those grid points were inverse-projected, rounded to packed `f32`, normalized,
and projected again, so they are near-grid spherical inputs rather than an
exact planar raster.

At 1M and one worker, Fast/regular construction paired-geomean ratios were
0.964 on Fibonacci (95% CI 0.945--0.982), 0.991 on uniform (0.983--0.999),
0.959 on the nominal grid (0.937--0.982), and 0.980 on the jittered grid
(0.963--1.003). At 16 workers every interval included parity. Fast Mode reduced
retired instructions by 2.8% on serial Fibonacci, 2.4% on serial uniform, 5.7%
on the nominal grid, and 3.9% on the jittered grid at 1M. Robust predicates
therefore impose measurable but modest work on these sphere-compatible inputs.

A separate Fade-only control fed exact integer `(x,y)` raster coordinates
directly to Fade, bypassing spherical normalization and projection. This is not
a spherical competitor result; it isolates the predicate-heavy case named by
Fade's Fast Mode documentation. At 1M, regular/Fast construction medians were
1,011.5/278.9 ms on one worker, a Fast/regular paired ratio of 0.275
(0.273--0.278), and 296.5/159.9 ms on 16 workers, ratio 0.542
(0.520--0.569). Serial retired instructions fell from 15.90B to 3.46B; parallel
instructions fell from 16.16B to 5.19B. Fast Mode is therefore a major raster
optimization, but it does not explain ordinary spherical-input performance.

Regular and Fast Mode produced identical, stable topology fingerprints for all
serial cases and for parallel Fibonacci, uniform, and jittered-grid cases.
Every tested 1M regular/Fast raster triangulation passed Fade's separate
multiprecision `checkValidity(true)`. Exactly cocircular parallel grids had
run-dependent valid Delaunay fingerprints in both modes, including within
regular mode: scheduling selects among legal grid diagonals. That is degeneracy
non-uniqueness, not evidence that Fast Mode returned an invalid triangulation.
Raw files are retained under `target/competitors/results/` with names matching
`fade-mode-<distribution>-t<workers>.csv`.

The closest completed-output comparison is total time, though the retained
representations remain different: Fade stores the planar triangulation while
this crate stores a shared Voronoi-cell mesh. At 1M serial, total medians were
656.5 versus 665.0 ms on Fibonacci and 809.7 versus 891.2 ms on uniform
(Fade versus this crate). At 16 workers, this crate was 2.90x and 2.47x faster
by paired-geomean total-time ratio. Fade's hull closure matched this crate's
unordered Delaunay face set at 10k for both inputs. At 1M, both had `2n-4`
faces, but topology fingerprints differed: this crate emitted eight fewer
incidences on Fibonacci and ten fewer on uniform. Its certified shared mesh can
contain non-triangular Delaunay duals in numerically degenerate local
configurations.

Raw CSV files are retained under
`target/competitors/results/fade-{fib,uniform}-t{1,16}.csv`; the native planar
control is `fade-native-uniform-t1.csv`.
