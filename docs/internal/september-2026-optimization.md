# September 2026 optimization experiments

Date: 2026-09-22. The initial pass retained nearest-only point location and
edge-derived clipping provenance. The construction follow-up below adds shared
constraint lookup, compact constraint records, and deferred vertex-key sorting,
with Fibonacci and uniform construction as the primary performance targets.

## Measurement setup

The baseline is `062b962` **plus the working-tree changes present at the start of
this session**, not the clean commit. Frozen source, original diff, binaries, raw
CSVs, and test logs are in `/tmp/voronoi-opt-sep22`. Those pre-existing changes were
preserved. Baseline and candidates used the same `Cargo.lock` and Rust 1.97.1.

Hardware: Ryzen 9 5900XT, 16 physical cores / 32 logical processors. Native builds
used `RUSTFLAGS="-C target-cpu=native"`; portable controls used empty `RUSTFLAGS`.
All builds used release optimization. Counter groups were
`{instructions:u,branches:u,cycles:u}`, with no multiplexing observed. Runs
alternated baseline/candidate order. Construction used seed 12345 and disabled
preprocessing unless specified. Locator inputs are deterministic in the retained
example. Telemetry was used separately from production counter measurements.

Three identical-binary pairs on each of 500k Fibonacci and uniform inputs showed
less than one part per million instruction/branch variation. Cycles varied more,
and compilation and other host activity caused much larger variation during the
campaign. Instruction reductions are structural evidence, not implied equivalent
wall-time improvements. In particular, the small construction cycle changes are
unresolved and need a quiet timing run.

## Nearest-only point location

Previously `nearest_unrestricted_slot` consumed the general sorted shell-prefix
stream while using only each prefix's best candidate. It now takes the minimum
packed key of each complete gathered layer and tests the existing next-layer
bound. This deletes sorting, partitioning, and copying of unused candidates, plus
the separate locator batch buffer. The shared layer schedule, candidate dot
arithmetic, and geometric bounds are unchanged. The tools-only quality sampler
uses the same specialized lookup.

Ascending packed-key order preserves dot ordering (including signed zero) and
ascending-slot ties within a layer. Strict `>` between layer winners preserves
previous-layer preference at equal dots. The full-layer minimum makes the former
bound on the unconsumed part of that same layer unnecessary.

Final isolated locator comparison: three alternating pairs, 100k generators,
250k fixed uniform canonical queries repeated four times, one worker pinned to
CPU 4. The whole-process counters include unchanged point/query generation,
diagram construction, and locator setup. The timed loop covers `locate_point`
and checksum accumulation. Every output checksum matched.

| Generator distribution | Instructions | Branches | Cycles | Query-loop median, baseline → candidate |
|---|---:|---:|---:|---:|
| Uniform | -42.005% | -36.454% | -46.928% | 1214 → 600 ns/query |
| Fibonacci | -42.223% | -37.776% | -46.863% | 1256 → 630 ns/query |

These large cycle/time effects survived concurrent host work; they are measured
results for this workload, not a universal 2x guarantee. Five-pair controls also
improved whole-process instructions: 16-site uniform -1.292%, 16-site clustered
-1.480%, 10k uniform -43.216%, and 10k clustered -32.256%. Clustered inputs put 80%
of sites near +Z and use globally uniform queries. Sparse inputs have little
sorting to remove.

The integrated patch, rechecked with identical copies of the polished retained
harness, preserved the instruction result: uniform -42.061%, Fibonacci -42.271%
(three alternating pairs each). Cycles fell 42.487%/42.579%, and query-loop time
fell 45.126%/45.088%; all six pairs favored the candidate. The timing variation
between this run and the isolated run illustrates the host noise, while retired
work is stable. All checksums matched. Raw data: `locator-integrated.csv`.

Reproduce with identical copies of the retained harness on baseline and candidate:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --example bench_locator
RAYON_NUM_THREADS=1 perf stat -e '{instructions:u,branches:u,cycles:u}' -- \
  taskset -c 4 target/release/examples/bench_locator 100000 250000 4 uniform
```

Use `fib` or `clustered` for the final argument; rotate run order and check matching
checksums. Raw locator results: `locator-final.csv`, `locator-controls.csv`, and
`locator-perf.csv` in the artifact directory.

## Edge-derived clipping provenance

A polygon already stores the plane of each outgoing edge. Its vertex plane pair
is therefore the unordered pair of preceding/outgoing edge labels. Previously
both representations were copied on every changed clip. Release builds now keep
only edge labels and reconstruct the pair during extraction, eliminating the
8-byte pair load/store for each copied survivor and pair stores at intersections.
The two fixed polygon buffers together shrink by 384 bytes.

The invariant is topological: entry intersection, contiguous old boundary arc,
exit intersection. It does not require distinct coordinates, positive edge
lengths, or distinct plane IDs. Canonical vertex-key sorting removes pair-order
differences. Spherical fallback polygons retain their separate representation.
Checked/test builds independently maintain the original pair stream and assert
agreement after every changed clip and when extracting a pair. Bounding-plane
tracking now reads outgoing edge labels.

Integrated final native construction counters, default bin policy:

| Workload | Workers | Paired runs | Instructions | Branches |
|---|---:|---:|---:|---:|
| 1M Fibonacci, two builds/process | 1 | 7 | -0.940% | -0.933% |
| 1M uniform, two builds/process | 1 | 7 | -1.036% | -0.890% |
| 1M Fibonacci, three builds/process | 16 | 4 | -0.983% | -0.992% |
| 1M uniform, three builds/process | 16 | 4 | -1.033% | -0.890% |

Every pair reduced both counters. Single-worker runs used CPU 2; 16-worker runs
used physical CPUs 0–15. Raw files: `final-native.csv` and
`final-16threads.csv`. Small cycle effects remain unresolved under host contention.

The isolated clipping change also reduced instructions/branches in all three
pairs of each 100k control: clustered -0.445%/-0.293%, mega -0.213%/-0.109%,
cubed -0.994%/-0.946%, great-circle -0.720%/-0.978%. Portable 500k controls reduced
instructions 1.148%/1.217% on Fibonacci/uniform and branches 0.697%/0.626%.
Raw files: `clipping-regimes.csv` and `clipping-generic.csv`.

An integrated 100k gate found clustered instructions/branches down
0.339%/0.602% and cubed down 0.799%/0.743%, but mega instructions up 0.326%
despite 0.057% fewer branches. The combined source changes also moved the directed
shell frontier and packed interior-threshold helper out of their former callers.
Two bounded boundary probes did not fix it: moving nearest-only lookup into a
child module preserved the relevant machine code, while ordinary `#[inline]` on
the shell frontier increased mega instructions to +0.552% versus baseline. Neither
was retained. The 0.326% mega instruction cost is an explicit tradeoff of the
integrated result, accepted alongside the much larger locator saving and ordinary
construction reductions. No dense-case throughput win is claimed.

## Initial rejected experiments

- **Deferred vertex-key sorting (decision superseded by the follow-up below):**
  sorting only unresolved corners removed
  0.65–0.71% of native ordinary-input instructions, but increased 100k mega
  instructions by 0.453% and branches by 0.069% in every pair. Moving the helper
  to its proper module was counter-neutral. The change also altered unrelated
  compiler inlining around shell traversal, which represents a large fraction
  of mega work; exact attribution remains unresolved. The initial decision was
  to preserve canonical extraction keys. Candidate and evidence remain in `dedup-candidate`,
  `dedup-regimes.csv`, and `dedup-results.md` in the artifact directory.
- **Skip identical reference overrides:** at 96 bins, saved only 0.070–0.076%
  instructions on 500k uniform/Fibonacci; cycles were mixed. Not retained in
  this pass. `patch-candidate` and `patch-96.csv` preserve the experiment.
- **Dense weld sweep:** existing occupancy refinement limited the observed
  100k mega maximum occupancy to 326; preprocessing consumed only about 0.4%
  of that diagnostic run. No implementation was justified by this opportunity
  check. Larger or more concentrated inputs would need a fresh census.

## Validation

- Full portable `cargo test --release`: 412 passed, zero failed, 23 existing
  ignored tests (including doctests).
- Integrated native checked library and correctness, adversarial, high-degree,
  locator, edge-reconciliation, and small-N suites: 337 passed, zero failed.
- Exact 100k baseline/candidate diagram fingerprints matched at one and six
  workers with six bins: representation `0991e1df6f60d5de`, semantic topology
  `961e56d915d09a4e` (199,996 vertices / 100,000 cells).
- Locator differential tests retain the old sorted traversal as an oracle across
  ordinary, concentrated, sparse, and empty grids, multiple resolutions,
  repeated sites, and axis/antipodal queries. All ten no-default-feature public
  locator tests passed on the isolated final locator change.
- `cargo clippy --all-targets --features tools,microbench,serde,glam` and
  `cargo fmt --check` passed.
- The tools-enabled release library suite passed all 277 active tests, including
  the quality sampler that also uses nearest-only lookup.
- The integrated scalar, no-default-feature release library/correctness/locator
  suites passed 291 active tests. Release checks with
  `tools,microbench,serde,glam` passed as well.

## Construction follow-up: Fibonacci and uniform priority

The follow-up baseline is the entire retained initial patch, including the locator
change. Frozen source, binaries, CSVs, and logs are in `/tmp/voronoi-opt-round2`.
Dense-cap results are recorded as secondary tradeoffs rather than a veto on small
ordinary-workload improvements. Every number in this section is an additional
change relative to that follow-up baseline unless labeled cumulative.

Three source changes are retained:

1. **Reuse each boundary constraint during extraction.** The outgoing edge supplies
   the current corner's second neighbor, the next corner's first neighbor, and its
   own forwarding slot. Seed the closing edge once, carry the preceding neighbor,
   and reuse the current lookup. This removes repeated constraint lookups, cyclic
   predecessor work, and the unreachable bounding-edge output branch. Checked
   builds still verify the independently maintained plane-pair oracle. Vertex
   coordinates and normalization arithmetic are unchanged.
2. **Keep accepted constraint IDs at their grid width.** Store `neighbor_idx` as
   `u32`, alongside the existing `u32` slot, shrinking each constraint from 16 to
   8 bytes on 64-bit targets. Grid construction already rejects point counts
   outside u32 capacity, and production neighbors come from u32 point records.
   A debug assertion records that invariant at clip commit; cold replay widens
   only for indexing. An initial per-clip checked conversion added branches, so
   the retained implementation relies on the existing checked producer boundary.
3. **Canonicalize only unresolved native corner keys.** AVX2 emission resolves
   about two-thirds of corners through incoming edge checks before it needs an
   ownership key. Gnomonic extraction now leaves these triples unordered and
   emission sorts only unresolved ones, before ownership decisions or persistent
   storage. Endpoint membership/XOR is order-independent. Generic code retains
   its established sorted-extraction, owner-first emission order. Separate
   `VertexAttribution`/`VertexKey` aliases document these contracts; the existing
   sorting helper now lives with output keys rather than tangent-basis geometry.

Isolated native 1M, one-worker counter gates (two builds/process):

| Candidate | Pairs | Fibonacci instructions | Uniform instructions | Fibonacci branches | Uniform branches |
|---|---:|---:|---:|---:|---:|
| Deferred key sorting | 5 | -0.992% | -0.902% | approximately unchanged | approximately unchanged |
| Shared constraint lookup | 5 | -1.232% | -1.120% | -3.220% | -2.849% |
| Compact records | 3 | -0.775% | -0.709% | approximately unchanged | approximately unchanged |
| Lookup plus sorting | 5 | -2.129% | -1.936% | -3.220% | -2.849% |
| All three | 7 | -2.430% | -2.214% | -3.221% | -2.849% |

The gains overlap: once extraction reuses constraints, compact records have fewer
lookups left to accelerate. Do not add the isolated percentages. Raw CSVs are
`dedup.csv`, `extract.csv`, `compact-bounded.csv`, `combined.csv`, and `all.csv`.

Integrated native controls, all with the default bin policy:

| Workload | Workers | Pairs | Fibonacci instructions / branches | Uniform instructions / branches |
|---|---:|---:|---:|---:|
| 1M, three builds/process | 16 | 4 | -2.401% / -3.159% | -2.180% / -2.805% |
| 4M, two builds/process | 16 | 3 | -2.416% / -3.207% | -2.170% / -2.789% |
| 1M, preprocessing enabled, two builds/process | 1 | 3 | -2.352% / -3.087% | -2.149% / -2.743% |

Every pair reduced both counters. Secondary 100k controls also reduced
instructions/branches in all three pairs: clustered -0.382%/-0.538%, mega
-0.202%/-0.186%, cubed -2.111%/-2.613%. Raw files: `final-16threads.csv`,
`final-4m.csv`, `final-preprocess.csv`, and `final-secondary.csv`.

Portable codegen also improves: three paired 1M/one-worker runs reduced
instructions by 1.426% on Fibonacci and 1.296% on uniform, and branches by
3.073%/2.733%. This retains sorted extraction, so only the shared lookup and
compact-record changes apply. Raw file: `final-generic.csv`. Baseline and final
portable binaries were rebuilt in separate target directories after detecting
and discarding an initial shared-target comparison that reused one executable.

Cycles are still host-noise limited. For example, the seven-pair 1M Fibonacci
all-three run favored the candidate in six pairs, but one +9.34% outlier made
the geometric mean +0.78%; uniform was -0.93%, favorable in five pairs. The
4M multi-worker uniform set had a +30.6% outlier. These are not evidence of
a stable elapsed-time speedup or regression. A quiet construction timing session
is still needed for a small throughput claim.

**Cumulative structural result:** comparing median retired counts for the same
1M/two-build workload with the frozen pre-session baseline gives Fibonacci
-3.348% instructions / -4.124% branches and uniform -3.227% / -3.713%. These
cross-pass comparisons are suitable for the stable work counters, not cycles.

Follow-up validation:

- Full portable release suite: 414 passed, zero failed, 23 existing ignored tests.
- Final native checked library/correctness/fingerprint suites: 288 passed; the
  earlier combined candidate also passed 357 checked tests including adversarial,
  high-degree, edge reconciliation, API, and small-N coverage.
- Final native tools-enabled release library/API/correctness/fingerprint suites:
  322 passed. Scalar/no-default-feature library/correctness/locator: 293 passed.
- Explicit native 100k fingerprints still match the original representation and
  semantic hashes above at both one and six workers, with six bins.
- New tests cover every permutation of endpoint attribution triples and large
  unsigned generator IDs with distinct forwarding slots across the closing edge.
- Clippy with `tools,microbench,serde,glam`, formatting, and diff whitespace checks
  passed. The locator implementation is committed as `b5a1f18` and the construction
  changes as `ccf82f3`; pre-existing working-tree edits remain separate.
