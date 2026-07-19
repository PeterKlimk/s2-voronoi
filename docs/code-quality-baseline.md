# Code-quality baseline and lifecycle rename map

**Status:** Milestone 0 baseline captured; QUAL-001A vocabulary migration validated

**Date:** 2026-07-17

This record pins the starting evidence and exact first migration for
[`QUAL-001`](code-quality-plan.md). It is deliberately counter-oriented because the measurement
host is a busy shared machine. A quiet wall-clock campaign is reserved for a repeatable signal that
cannot be classified through semantic checks, retired work, memory, or code size.

## Baseline identity

- **Measured revision:** `ec8c491e239e8184dd2de62238360383c3345af4`
- **Last production-code revision:** `fde41b6` (`ec8c491` differs only by the two cleanup-plan
  documentation commits)
- **Toolchain:** `rustc 1.97.0 (2d8144b78 2026-07-07)`, LLVM 22.1.6,
  `cargo 1.97.0 (c980f4866 2026-06-30)`
- **MSRV contract:** Rust 1.88; this baseline is not an MSRV validation run
- **Host:** WSL2 Linux 6.6.87.2, AMD Ryzen 5 3600, 6 physical / 12 logical CPUs
- **Benchmark build:** release, `tools`, `-C target-cpu=native`
- **Artifact:** `/tmp/bench_compare/bench_0`, build id
  `61f788139e9bc2899d660b3ce7f2356730f3f1d1`
- **Artifact size:** 2,178,079 text bytes; 55,784 data bytes; 2,998,928 file bytes

The benchmark artifact is temporary. The revision, build inputs, commands, and summary below are
the durable baseline; candidate comparisons rebuild both the candidate and its immediate parent.

## Semantic baseline

The ignored `backend_fingerprint` test uses a fixed 100k random-sphere input (seed 7). Its
representation fingerprint includes stored coordinates and vertex ids; its semantic fingerprint
canonicalizes cell cycles through incident-generator identities.

| Backend / execution | Representation | Semantic topology | Vertices | Cells |
|---|---:|---:|---:|---:|
| default, 1 thread, 6 bins | `0991e1df6f60d5de` | `961e56d915d09a4e` | 199,996 | 100,000 |
| default, 6 threads, 96 bins | `0e65ca5dbe8fe07c` | `961e56d915d09a4e` | 199,996 | 100,000 |
| scalar SIMD seam, 1 thread, 6 bins | `0991e1df6f60d5de` | `961e56d915d09a4e` | 199,996 | 100,000 |
| hardware FMA, native, 1 thread, 6 bins | `30bde3bc634ebb50` | `961e56d915d09a4e` | 199,996 | 100,000 |

Commands:

```bash
RAYON_NUM_THREADS=1 VORONOI_MESH_BIN_COUNT=6 \
  cargo test --release --test backend_fingerprint backend_fingerprint -- --ignored --nocapture

RAYON_NUM_THREADS=6 VORONOI_MESH_BIN_COUNT=96 \
  cargo test --release --test backend_fingerprint backend_fingerprint -- --ignored --nocapture

RAYON_NUM_THREADS=1 VORONOI_MESH_BIN_COUNT=6 \
  cargo test --release --features simd_scalar \
  --test backend_fingerprint backend_fingerprint -- --ignored --nocapture

RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 VORONOI_MESH_BIN_COUNT=6 \
  cargo test --release --features fma \
  --test backend_fingerprint backend_fingerprint -- --ignored --nocapture
```

The defect-bearing baseline also passed:

```text
edge_repair_net: 5 passed, 5 ignored
escalate_local:   4 passed, 1 ignored
```

These are historical test target names. QUAL-001A renames them according to the map below.

## Counter baseline

The current `scripts/bench_perf.sh` harness was used with five measured samples after a first-round
warm-up. All runs used 500k points, seed 12345, preprocessing disabled, and the default local
rebuild policy. Single-thread runs used `RAYON_NUM_THREADS=1` and CPU 0; default-parallel runs were
unpinned so the Rayon pool could use the host.

### Median retired work

The percentage in parentheses is `(max - min) / median` across the five samples. Counts are for the
whole benchmark process, including deterministic point generation.

| Regime | Distribution | Instructions | Branches | Branch misses | Cache misses |
|---|---|---:|---:|---:|---:|
| 1 thread, CPU 0 | Fibonacci | 3,419,644,674 (0.001%) | 377,984,367 (0.002%) | 12,669,130 (7.50%) | 8,189,538 (44.99%) |
| 1 thread, CPU 0 | uniform | 3,734,723,609 (0.001%) | 431,888,065 (0.001%) | 20,398,270 (2.24%) | 13,753,682 (24.14%) |
| default parallel | Fibonacci | 3,492,881,656 (0.179%) | 392,165,697 (0.309%) | 13,908,227 (3.28%) | 9,289,750 (18.52%) |
| default parallel | uniform | 3,822,378,633 (0.325%) | 447,918,186 (0.566%) | 22,685,717 (2.20%) | 18,418,672 (27.74%) |

### Noisy counters and memory

| Regime | Distribution | Cycles median | Cycle range | Task-clock median | Task-clock range | Max RSS sample |
|---|---|---:|---:|---:|---:|---:|
| 1 thread, CPU 0 | Fibonacci | 2,014,477,151 | 17.37% | 768 ms | 21.28% | 137,448 KiB |
| 1 thread, CPU 0 | uniform | 2,500,593,827 | 11.33% | 813 ms | 17.93% | 134,560 KiB |
| default parallel | Fibonacci | 2,514,102,135 | 1.68% | 1,211 ms | 26.08% | 149,096 KiB |
| default parallel | uniform | 3,117,391,998 | 3.48% | 1,503 ms | 32.28% | 149,268 KiB |

The perf samples reported zero context switches and zero CPU migrations. Under WSL2 those software
counters are not useful contamination filters, so the comparison must rely on paired rotation and
the stability of retired instructions/branches. Cache and task-clock variation confirms that they
are attribution aids, not first-line acceptance metrics on this host.

Commands:

```bash
./scripts/bench_build.sh HEAD
./scripts/bench_perf.sh -r 5 -s 500k -d fib \
  --csv /tmp/code_quality_perf_st_fib.csv
./scripts/bench_perf.sh -r 5 -s 500k -d uniform \
  --csv /tmp/code_quality_perf_st_uniform.csv
./scripts/bench_perf.sh -r 5 -s 500k -d fib --multi --no-pin \
  --csv /tmp/code_quality_perf_mt_fib.csv
./scripts/bench_perf.sh -r 5 -s 500k -d uniform --multi --no-pin \
  --csv /tmp/code_quality_perf_mt_uniform.csv
```

## Comparison rule for QUAL-001A

The rename is expected to be behavior- and hot-work-neutral. Compare the candidate against its
immediate parent with both artifacts in the same rotated `bench_perf.sh` run.

- Semantic fingerprints and defect-suite outcomes must match the contract above.
- Single-thread instructions and branches are the primary codegen sentinel. Any repeatable paired
  movement above 0.1% requires inspection; exact or near-exact equality is expected.
- Default-parallel instructions/branches use a 1% decision band because this baseline shows up to
  0.57% natural spread and parallel work scheduling is not representation-deterministic.
- Peak RSS and binary text size must not move repeatably by more than 1% without an explained
  tool/API-surface cause.
- Cycles, cache events, task-clock, and elapsed time are advisory on this machine. Request a quiet
  run when they show a repeatable adverse direction above 3% that stable retired-work, RSS, and
  code-size evidence cannot explain.
- A new ordinary-path allocation, branch, or environment lookup is independently sufficient to
  reject or redesign the change.

These are gates for the naming migration, not universal thresholds for later hot-path extraction.
Each later workstream must set its own immediate-parent rule from its affected regimes.

## QUAL-001A validation result

The coordinated vocabulary migration was validated on 2026-07-17 against immediate parent
`3bf5050`. It changed names and documentation, removed the unread
`VORONOI_MESH_RECLIP_REPAIR` knob, and did not add compatibility aliases.

- All four backend fingerprints matched the semantic and representation values above exactly.
- `cargo test --release`, `cargo test --profile checked`, the no-default-feature build, and the
  `serde,glam` build passed. Both default and native all-feature clippy runs passed with warnings
  denied.
- The renamed defect suites retained their outcomes: `edge_reconciliation` passed 5 tests with 5
  ignored, and `local_rebuild` passed 4 tests with 1 ignored.
- The probe-feature target exposed two inherited active tests whose historical mega fixture now
  resolves before defect-driven rebuilding. The same tests fail at `3bf5050`; they are now marked
  as diagnostics with that reason, leaving the feature target compiling cleanly with 14 ignored
  probes.
- Paired single-thread candidate/parent medians were effectively identical: instructions moved
  `+0.000250%` (Fibonacci) and `-0.000190%` (uniform); branches moved `+0.000053%` and
  `-0.000769%` respectively.
- Default-parallel medians stayed within the declared scheduling band: instructions moved
  `+0.231277%` and `+0.237042%`; branches moved `+0.412397%` and `+0.404437%`. Paired means were
  closer to zero than the medians.
- Binary text grew by 112 bytes (`+0.0051%`); data size was unchanged. One-shot peak-RSS samples
  ranged from `-1.11%` to `+0.12%`, with no adverse sample above 1%.
- Cycle medians ranged from `-1.41%` to `+0.15%`. Task-clock was substantially noisier, but neither
  had a corroborating retired-work signal, so the conditional quiet run was not warranted.

## QUAL-001F validation result

Current-architecture hygiene was completed on 2026-07-17 as three attributable changes after
QUAL-001A:

1. unused compatibility re-exports and the empty `TerminationConfig` were removed;
2. the unconsumed `VertexPosition` / `Vec2` seam was specialized to the crate's spherical `Vec3`
   backend, including reconciliation's always-false boundary policy; and
3. default, all-feature, and all-target `unreachable_pub` audits restricted 216 internal
   visibility spellings, refreshed module ownership documentation, and made generated
   sorting-network visibility reproducible from the generator.

The doc-hidden root surfaces for `tools`, `profiling`, `microbench`, and
`local_rebuild_probe`, plus experimental report diagnostics, were retained: each has a current
repository binary, integration test, or defect fixture. QUAL-001H owns the decision to reorganize
those diagnostics rather than silently removing them in a visibility pass.

Validation evidence:

- `cargo fmt --check`, the sorting-network generator check, default and native all-feature clippy
  with warnings denied, default/all-feature `-D unreachable-pub`, and all-target/all-feature check
  passed.
- The full release and checked suites passed. The no-default and `serde,glam` matrices passed with
  one harness thread; the `local_rebuild_probe` target compiled with its 14 manual probes ignored.
- An initial validation orchestration accidentally overlapped long-running Cargo commands on the
  busy host; the affected targets passed alone and the complete matrices passed serialized. That
  run did not establish process-environment interference. QUAL-001H subsequently audited the
  actual within-process mutation boundaries directly.
- Single-thread and six-thread/96-bin fingerprints remained exactly
  `0991e1df6f60d5de` and `0e65ca5dbe8fe07c`, with semantic topology
  `961e56d915d09a4e`. The FMA representation fingerprint remained
  `30bde3bc634ebb50` after the spherical specialization.
- Compatibility removal versus `82e2b4e` was counter-neutral: single-thread instruction/branch
  medians stayed within `0.00003%`, default-parallel retired work stayed within `0.03%`, and
  loadable binary sections were byte-identical.
- The spherical specialization versus `e37962c` moved instruction medians by
  `+0.01297%` to `+0.01441%` across Fibonacci/uniform and single/default-parallel cells; branch
  medians stayed between `-0.00072%` and `+0.00053%`. Cycle medians ranged from `-0.61%` to
  `+0.10%` without a corroborating adverse retired-work signal. Text shrank 280 bytes; total
  text/data/BSS size shrank 8 bytes.
- The final visibility/documentation candidate and immediate parent `83b9392` had identical
  `.text`, `.rodata`, and `.data` hashes and identical text/data/BSS sizes. ELF build IDs differed
  because Rust metadata/symbol visibility changed; loadable program content did not.

No quiet wall-clock run was warranted: the only stable counter movement was a negligible
`~0.014%` instruction increase after deleting an unused generic seam, while cycles and task clock
remained uncorrelated shared-host noise.

## QUAL-001H environment-isolation result

The first diagnostics/test-layout slice was validated on 2026-07-18 against immediate parent
`a777508`.

- [`environment-knobs.md`](environment-knobs.md) classifies every library, tool, campaign, and
  manual-probe variable found in Rust and repository scripts. It records reader ownership,
  snapshot cadence, current writers, and the integration-test process boundary.
- The stale `VORONOI_MESH_PLANE_GRID_DENSITY` documentation was removed; no planar backend or
  reader exists in the repository.
- Active integration tests now share one panic-safe helper that serializes within each target,
  preserves exact pre-existing `OsString` values, restores in reverse order during unwinding, and
  recovers a poisoned mutex. A direct regression exercises panic restoration and subsequent reuse.
- The verification-gate unit test exercises enabled and disabled behavior in filtered child-test
  processes. It tests the real environment parser and error mapping without mutating the shared
  library-unit-test process.
- Default release, checked, no-default-feature, and `serde,glam` suites passed under their ordinary
  parallel test harnesses. The local-rebuild probe target compiled with all 14 manual probes
  ignored. Default and native all-feature clippy passed with warnings denied.
- The optimized benchmark binary was byte-for-byte identical to `a777508`, including equal
  text/data/BSS sizes. No performance sampling or quiet wall-clock run was necessary.

At this boundary, ignored `local_rebuild_probe` cases still owned manual environment mutation and a
forced-rebuild switch. The following QUAL-001H slice removed both.

## QUAL-001H manual-probe isolation result

The second diagnostics/test-layout slice was validated on 2026-07-18 against immediate parent
`50e419c`.

- Every A0 snapshot consumer now goes through the shared `stash_fast_triples` helper and an
  explicit `with_a0_fast_capture` scope. The state and captured payload are both thread-local; the
  scope composes when nested and restores its prior state during panic unwinding.
- The former process-global forced-rebuild switch was redundant: every repository consumer set it
  only around A0 capture, while the A0 branch returns before rebuild-mode selection. The switch,
  setter, reader, atomic storage, and A0 environment lookup were removed without replacement.
- Cargo now declares `local_rebuild_probe` as an all-ignored manual test target requiring the
  internal feature. The target retains the same name and all 14 named probes.
- A focused release regression covered nesting, thread isolation, and panic restoration. The
  manual target listed all 14 cases, and a reduced 1,000-point A0 exact-reference probe passed.
- Default release, checked, no-default-feature, and `serde,glam` suites passed under their ordinary
  parallel harnesses. All-target/all-feature Clippy passed with warnings denied.
- The 1-thread/6-bin and 6-thread/96-bin representation fingerprints remained exactly
  `0991e1df6f60d5de` and `0e65ca5dbe8fe07c`; both retained semantic topology
  `961e56d915d09a4e` with 199,996 vertices and 100,000 cells.
- Parent and candidate optimized benchmark builds used the same stable toolchain and dependency
  lock. Their `.text` and `.rodata` sections were byte-identical, and both reported 2,179,216 text,
  55,840 data, and 4,096 BSS bytes. Full files differ through build metadata, and `.data.rel.ro`
  differs only in Rust panic-location line numbers shifted by the deleted source lines. No
  performance-counter sampling or quiet wall-clock run was necessary.

## QUAL-001H manual-campaign target result

The third diagnostics/test-layout slice was validated on 2026-07-18 against immediate parent
`b99d59d`.

- Cargo now excludes the wholly ignored `coincidence_probes` and `robustness_campaign` targets
  unless the internal `manual_probes` feature is selected. The tools-dependent fidelity campaign
  explicitly requires `tools` rather than compiling as an empty default target.
- Source-level reproduction commands and the robustness campaign driver select the required
  feature. Target names, test names, environment inputs, and per-case process isolation remain
  unchanged.
- Mixed active/manual targets were deliberately retained: their isolated ignored cases reuse the
  surrounding fixture setup, and splitting them would increase duplication without improving
  state isolation.
- The planned `quality` surface decision was already present before QUAL-001: `quality.rs` is
  doc-hidden and `tools`-gated, with current consumers in `bench_voronoi` and the fidelity campaign.
- Cargo target listings reported the expected 5 coincidence probes, 4 robustness cases, and 1
  fidelity case behind their declared features. One release-mode case from each target passed,
  including the environment-driven campaign paths; both campaign scripts passed shell syntax
  validation.
- The ordinary release suite passed without compiling those three targets. All-target/all-feature
  Clippy passed with warnings denied.
- The optimized benchmark retained byte-identical `.text` and `.rodata` sections against the saved
  immediate-parent artifact, with equal 2,179,216 text, 55,840 data, and 4,096 BSS bytes. No
  performance-counter sampling or quiet wall-clock run was necessary.

## QUAL-001H local-rebuild cold-options result

The fourth diagnostics/test-layout slice was validated on 2026-07-18 against immediate parent
`7a60bc2`.

- `LocalRebuildDiagnostics` snapshots the debug flag and feature-only global-Delaunay selector once
  per actual rebuild attempt. The grow loop and commit gate receive the captured debug value rather
  than rereading process state.
- The snapshot is constructed after mode and trigger checks. Disabled configurations, ordinary
  no-defect computations, and A0 capture return before either diagnostic lookup. Previously the
  debug variable was read on every enabled computation and three times during an attempt.
- The complete production local-rebuild target passed. The probe-only target compiled with all 14
  named manual cases, and a clean fixture run with the debug variable present confirmed that it no
  longer emits a false rebuild-trigger diagnostic.
- `cargo clippy --all-targets --all-features -- -D warnings`, the complete release and checked
  suites, the no-default-features release suite, and the `serde,glam` release suite all passed.
- The 100k semantic fingerprint remained `961e56d915d09a4e` in both the 1-thread/6-bin and
  6-thread/96-bin checks, with the expected representation fingerprints `0991e1df6f60d5de` and
  `0e65ca5dbe8fe07c`, 199,996 vertices, and 100,000 cells.
- With matched native release builds, the candidate had 2,177,651 text, 55,784 data, and 1,611 BSS
  bytes versus 2,177,911 text, 55,784 data, and 1,339 BSS bytes for the parent: text fell 260 bytes,
  data was unchanged, and the total allocation increased 12 bytes because BSS grew 272 bytes.
- Seven interleaved, CPU-pinned 500k Fibonacci runs retired a mean 3,420,125,021 instructions for
  the candidate and 3,420,130,510 for the parent, a neutral -5,489 (-0.00016%) candidate delta.
  Every measured run had zero context switches and CPU migrations. There was no adverse counter
  signal warranting a quiet wall-clock run.

## QUAL-001H reconciliation cold-options result

The fifth diagnostics/test-layout slice was validated on 2026-07-18 against immediate parent
`76942be`.

- `ReconcileOptions` snapshots telemetry, apply-backend selection, and the global duplicate-scan
  fallback once per defect-bearing computation. The immutable record is passed through telemetry,
  primary/backstop rounds, and duplicate collection; explicit test options remain independent of
  process state.
- A zero mismatch-record computation constructs only the default value and performs no
  reconciliation environment lookup. Previously the apply-backend variable was read on every
  computation, while telemetry and the duplicate-scan selector read process state inside their
  respective stage helpers.
- The record preserves each variable's exact historical value semantics and combined settings:
  telemetry analysis still honors the captured global-scan override when both are enabled.
- All 13 reconciliation unit tests and the complete deterministic defect suite passed. Targeted
  runs also passed with the global duplicate scan forced, telemetry enabled, and both flags enabled
  together; the existing differential continued to cover both apply backends.
- `cargo clippy --all-targets --all-features -- -D warnings`, the complete release and checked
  suites, the no-default-features release suite, and the `serde,glam` release suite all passed.
- The 100k semantic fingerprint remained `961e56d915d09a4e` in both the 1-thread/6-bin and
  6-thread/96-bin checks, with representation fingerprints `0991e1df6f60d5de` and
  `0e65ca5dbe8fe07c`, 199,996 vertices, and 100,000 cells.
- Matched native release builds reported 2,177,671 text, 55,784 data, and 1,579 BSS bytes for the
  candidate versus 2,177,651 text, 55,784 data, and 1,611 BSS bytes for the parent. Text grew 20
  bytes, data was unchanged, BSS fell 32 bytes, and the total footprint fell 12 bytes.
- Seven interleaved, CPU-pinned 500k Fibonacci runs retired a mean 3,420,119,124 instructions for
  the candidate and 3,420,122,919 for the parent, a neutral -3,795 (-0.00011%) candidate delta.
  Five of seven pairs favored the candidate; every run had zero context switches and CPU
  migrations. There was no adverse counter signal warranting a quiet wall-clock run.

## QUAL-001H singleton-diagnostics result

The sixth and closing diagnostics/test-layout slice was validated on 2026-07-18 against immediate
parent `bd2f37a`.

- Live assembly now checks for a non-empty mismatch set before reading its origin diagnostic.
  `VORONOI_MESH_EDGE_MISMATCH_ORIGINS` replaces the stale internal
  `VORONOI_MESH_UNPAIRED_ORIGINS` name; its value semantics and defect-bearing output are
  unchanged. `ComputeReport` remains the zero-event evidence, so clean runs no longer emit an
  all-zero origin line.
- Output-resolution telemetry was audited but deliberately left unchanged: its no-zero-edge early
  return already precedes the `VORONOI_MESH_RESOLUTION_KV` lookup, while a known exact-zero fixture
  still emitted the complete structured resolution result.
- A clean 10k benchmark emitted no origin line with the renamed knob present. The deterministic
  in-bin defect fixture emitted the expected total of three mismatches, split into two thirds
  mismatches and one unconsumed check.
- `cargo clippy --all-targets --all-features -- -D warnings`, the complete release and checked
  suites, the no-default-features release suite, and the `serde,glam` release suite all passed.
- The 100k semantic fingerprint remained `961e56d915d09a4e` in both the 1-thread/6-bin and
  6-thread/96-bin checks, with representation fingerprints `0991e1df6f60d5de` and
  `0e65ca5dbe8fe07c`, 199,996 vertices, and 100,000 cells.
- Matched native release builds reported 2,177,695 text, 55,784 data, and 1,563 BSS bytes for the
  candidate versus 2,177,671 text, 55,784 data, and 1,579 BSS bytes for the parent. Text grew 24
  bytes, data was unchanged, BSS fell 16 bytes, and the total footprint grew 8 bytes.
- Seven interleaved, CPU-pinned 500k Fibonacci runs retired a mean 3,420,123,521 instructions for
  the candidate and 3,420,134,408 for the parent, a neutral/favorable -10,887 (-0.00032%) candidate
  delta. Six of seven pairs favored the candidate; every run had zero context switches and CPU
  migrations. There was no adverse counter signal warranting a quiet wall-clock run.

## QUAL-001E dense-band policy result

The first numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`5998cb9`.

- The packed dense-cell gather's raw `1e-3` inflation is now the named
  `DENSE_BAND_RADIUS_INFLATION` policy in `policy.rs`. Its documentation records that the value is
  a dimensionless `f32` fraction, that it expands the gather chord radius, and that false positives
  add work without changing the strict dot-space coverage boundary.
- The value, arithmetic expression, comparison directions, and control flow are unchanged. This is
  a name-and-ownership change only; it does not tune the dense-band algorithm.
- `cargo fmt`, both default and all-feature Clippy with warnings denied, the complete release and
  checked suites, the no-default-features release suite, and the `serde,glam` release suite passed.
- The release `tools` benchmark before and after the change had identical SHA-256
  `295d983048d512272dbd019e2a162da572050df7594f3929d06cae9711a571ed`. Exact artifact identity
  supersedes counter and semantic-fingerprint comparison for this slice; there was no reason to
  request a quiet wall-clock run.
- The initial remaining-literal review found that equal `1e-24` fallback spellings currently serve
  two roles: rejecting squared cross products that are too small to normalize and deduplicating
  f64 fallback vertices by squared distance. They require separate semantic constants, not one
  mechanically shared name; preserving their equal values is a separate fact from their ownership.

## QUAL-001E fallback threshold result

The second numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`8f08126`.

- Seven raw fallback `1e-24` comparisons and the clip-local dedup constant are now represented by
  two authoritative `f64` tolerances: `FALLBACK_INTERSECTION_CROSS_LEN2_FLOOR` rejects non-finite
  or `<=` squared cross norms before normalization, while `FALLBACK_VERTEX_DEDUP_LEN2` collapses
  fallback unit directions at `<=` squared chord distance.
- Both values remain exactly `1e-24`. Their shared bit pattern is deliberately not encoded as a
  shared semantic constant: intersection conditioning and vertex identity may require independent
  future analysis and tuning.
- The constant documentation records units and comparison directions. No hierarchy assertion was
  added because the two values have no load-bearing ordering relationship.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical section sizes and total footprint. `.text`
  (`a20c69a6…f0de8`), `.rodata` (`f8b29268…7761`), and exception tables were byte-identical. The
  whole-file hash changed only with the build id, symbol/string metadata, and 13 one-byte line
  fields in 24-byte source-location records in `.data.rel.ro`; executable code and numeric data did
  not change. There was no counter signal requiring a quiet wall-clock run.

## QUAL-001E unit-distinct `1e-12` result

The third numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`38b2057`.

- Four raw production `1e-12` spellings are now three authoritative constants:
  `FALLBACK_EDGE_ARC_ANGLE_PAD: f64` is radians added to an inclusive arc-extent comparison;
  `GNOMONIC_METRIC_R2_RELATIVE_PAD: f64` is a dimensionless fraction used in `bound * (1 + pad)`;
  and `LOCAL_REBUILD_STEREOGRAPHIC_DENOMINATOR_FLOOR: f32` clamps the dimensionless
  `1 - dot(point, pole)` divisor with `max` in both the production projected rebuild and its
  feature-only global oracle.
- All three retain the exact `1e-12` value and each use retains its prior expression, type, and
  comparison direction. No hierarchy assertion was added because angular slack, relative metric
  inflation, and a divisor floor have no meaningful ordering relationship.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical section sizes and total footprint. `.text`
  (`a20c69a6…f0de8`), `.rodata` (`f8b29268…7761`), and exception tables were byte-identical. The
  whole-file difference was confined to build/symbol metadata and 41 changed bytes in
  source-location records; executable code and numeric data did not change. There was no reason
  for a counter or quiet wall-clock run.

## QUAL-001E owner-arc registry result

The fourth numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`1d189d7`.

- The owner-conditioned spherical-arc thresholds moved from module-local constants to
  `tolerances.rs` as `OWNER_ARC_PLANE_SIN_TOL` and `OWNER_ARC_EXACT_PI_SIN_TOL`.
- The plane residual remains a dimensionless `f64` sine/dot tolerance of `2e-6`; an arc is rejected
  when its maximum endpoint residual is `>` the value, so equality remains valid. The exact-pi
  threshold remains a dimensionless `f64` cross-length sine of `1e-12`; an opposite-facing arc is
  classified as ambiguous when the sine is `<=` the value.
- The constants are deliberately independent of the fallback plane and arc-angle tolerances. Their
  current values and numerical ordering do not create a shared hierarchy.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical section sizes and total footprint. All inspected
  loadable sections, including `.text`, `.rodata`, `.data.rel.ro`, and exception tables, were
  byte-identical. After removing symbols and the build-id note, both complete artifacts had SHA-256
  `a9c01ba20bbe32194ce765864c12fa9087c77a965038c57f4fbe908c8d0c56c8`. No counter or quiet
  wall-clock run was warranted.

## QUAL-001E weld wall-guard result

The fifth numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`15cdc94`.

- The two raw weld candidate-grid `1e-6` guards moved to `tolerances.rs` as independent constants.
  `GRID_WELD_WALL_ABS_PAD: f32` remains an absolute, dimensionless plane-dot/chord-scale reserve;
  the grid-integrated path still scans an adjacent cell when
  `abs(plane_dot) < threshold + GRID_WELD_WALL_ABS_PAD`.
- `STANDALONE_WELD_WALL_RELATIVE_PAD: f64` remains a dimensionless relative inflation; the
  standalone preprocessing path still forms its quantized wall pad as
  `threshold * (1 + STANDALONE_WELD_WALL_RELATIVE_PAD)` and uses the same strict `<` wall-distance
  comparison. The equal constant values do not establish a shared unit, expression, or hierarchy.
- Both guards can only admit extra candidate-cell scans. The final computed-f32 weld predicate
  remains the strict `distance_squared < radius_squared`, so this slice changes neither its weld
  radius nor its equality boundary.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical section sizes and total footprint. `.text`
  (`a20c69a6…f0de8`), `.rodata` (`f8b29268…7761`), and exception tables
  (`1771916b…bcbb`) were byte-identical. The whole-file difference was confined to build/symbol
  metadata and 75 changed source-location bytes in `.data.rel.ro`; executable code and numeric data
  did not change. There was no counter signal requiring a quiet wall-clock run.

## QUAL-001E coplanar compatibility-policy result

The sixth numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`5863b63`.

- The near-great-circle compatibility classifier's raw plane-residual thresholds moved to
  `tolerances.rs` as independent `f64` values. `NEAR_GREAT_CIRCLE_MAX_PLANE_SIN_TOL` remains
  `2.0e-6`, and `NEAR_GREAT_CIRCLE_RMS_PLANE_SIN_TOL` remains `5.0e-7`. Both are dimensionless
  sine/dot residual bounds, and the classifier still rejects when either measured residual is
  strictly `>` its bound; equality remains accepted.
- Their numerical ordering does not establish a derived fraction or load-bearing hierarchy. The
  maximum and RMS tests constrain different aggregates over the same point residuals.
- The raw `1.0e-2f64` realized joggle became `COPLANAR_PERTURBATION_SCALE` in `policy.rs`. It remains
  a dimensionless normal-offset coefficient multiplied by the same stable signed value before
  point renormalization. It is intentionally output-changing robust-mode policy, not a
  coplanarity-classification tolerance. The local `scale` binding was retained so the iterator
  closure keeps its original capture and optimized shape.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical section sizes and total footprint. `.text`
  (`a20c69a6…f0de8`), `.rodata` (`f8b29268…7761`), exception tables (`1771916b…bcbb`), and unwind
  sections were byte-identical. The whole-file difference was confined to build/symbol metadata
  and 14 changed source-location bytes in `.data.rel.ro`; executable code and numeric data did not
  change. No counter or quiet wall-clock run was warranted.

## QUAL-001E projected-Delaunay sizing result

The seventh numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`2e49a1a`.

- The local projected-Delaunay path's raw `1e-9` chart-span floor moved to `tolerances.rs` as
  `LOCAL_REBUILD_DELAUNAY_SPAN_FLOOR: f64`. Stereographic chart coordinates are dimensionless; the
  maximum measured axis span remains clamped with `max` to this value before sizing the synthetic
  construction envelope. It remains a nonzero sizing guard, not a point-acceptance classifier.
- The raw `1000.0` super-triangle multiplier moved independently to `policy.rs` as
  `LOCAL_REBUILD_SUPER_TRIANGLE_SCALE: f64`. The same span is still multiplied by the same
  dimensionless expansion before the three synthetic vertices are formed. Its value and role are
  not derived from the minimum-span floor.
- No coordinate expression, predicate, insertion order, robust predicate input, or downstream
  triangle filtering changed.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical section sizes and total footprint. `.text`
  (`a20c69a6…f0de8`), `.rodata` (`f8b29268…7761`), exception tables (`1771916b…bcbb`), and unwind
  sections were byte-identical. The whole-file difference was confined to build/symbol metadata
  and 40 changed source-location bytes in `.data.rel.ro`; executable code and numeric data did not
  change. No counter or quiet wall-clock run was warranted.

## QUAL-001E centroid degeneracy-floor result

The eighth numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`ef9f22c`.

- The two raw `f64::EPSILON` centroid comparisons moved to `tolerances.rs` as independent values.
  `CENTROID_EDGE_CROSS_LEN_FLOOR` remains a dimensionless unit-endpoint cross/sine magnitude; an
  edge is still skipped when `cross_len <=` the floor, avoiding division by a degenerate cross
  length.
- `CENTROID_INTEGRAL_LEN_FLOOR` remains the final accumulated-vector magnitude guard; the cell still
  returns its generator when `integral.length() <=` the floor rather than normalizing a degenerate
  direction. Equality retains the fallback behavior in both comparisons.
- The constants both remain exactly `f64::EPSILON`, but their equal machine-floor values do not
  couple per-edge omission to whole-cell fallback or establish a shared tuning hierarchy.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- Matched release `tools` benchmarks had identical sizes and byte-identical loadable sections,
  including `.text`, `.rodata`, `.data.rel.ro`, exception tables, and unwind data. After stripping
  symbols and the build-id note, both complete artifacts had SHA-256
  `802417baf66cc5394d41803f3478134478e1aaae57c3843bebe1180e2f2ae495`. No counter or quiet
  wall-clock run was warranted.

## QUAL-001E point-envelope diagnostic result

The ninth numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`2db0296`.

- The profiling-only point-envelope absolute-error bands are now named local constants:
  `ABS_ERROR_1E_MINUS_6_BOUND`, `ABS_ERROR_1E_MINUS_5_BOUND`, and
  `ABS_ERROR_1E_MINUS_4_BOUND`. The f32 epsilon band base is likewise the explicit local
  `F32_EPSILON_BOUND`. These values remain diagnostic bucket boundaries and do not participate in
  normalization, geometry, validation, or acceptance policy.
- Every bucket still counts with a strict `error > bound` comparison. The four epsilon-relative
  multipliers and three absolute values are unchanged.
- The profiling summary fields changed from ambiguous `over_1e6` / `over_1e5` / `over_1e4` to
  exponent-aware `over_1e_minus_6` / `over_1e_minus_5` / `over_1e_minus_4`. The benchmark's emitted
  keys changed in parallel from `gt_1e6` etc. to `gt_1e_minus_6` etc. This intentionally breaks the
  doc-hidden profiling surface while no external users exist; no compatibility aliases remain.
- Matched deterministic 1k profiling runs produced identical per-producer counts, maxima, rule
  comparisons, topology hash `f36e65e7876fa06a`, and coordinate hash `62c6f747b95ed029`; only the
  three corrected key names and noisy timing fields differed.
- The non-profiling release `tools` artifact remained byte-identical after stripping symbols and the
  build-id note, with SHA-256
  `802417baf66cc5394d41803f3478134478e1aaae57c3843bebe1180e2f2ae495`. The profiling artifact's
  total footprint remained 2,264,940 bytes; with its longer diagnostic labels and resulting
  alignment, `size` reported 32 more text bytes and 32 fewer BSS bytes. Production builds were
  unaffected, so no counter or quiet wall-clock run was warranted.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, the `serde,glam` release suite, and an
  explicit `tools,profiling` release suite passed.

## QUAL-001E gnomonic initialization-policy result

The tenth numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`82c9ab1`.

- The gnomonic tangent-basis builder's raw south-pole branch boundary moved to `policy.rs` as
  `GNOMONIC_TANGENT_BASIS_SOUTH_POLE_SWITCH_Z: f64`. The alternate basis is still selected only
  when `g.z < -0.999_999_9`; equality still uses the general `1 + z` formula.
- Both raw `init_bounding(1e6)` calls now use the independent
  `GNOMONIC_INITIAL_BOUNDING_EXTENT: f64` construction policy. New and reset builders still begin
  with the same synthetic square before clipping; no projection-limit or cell-acceptance boundary
  changed.
- The raw debug assertion band is now the module-local `f32`
  `DEBUG_NEIGHBOR_NORM_SQUARED_ERROR_LIMIT`. It remains exactly `1e-5` with the same strict `<`
  comparison and is explicitly diagnostic rather than a production tolerance.
- The final inventory also confirmed that quality and reconciliation histogram ranges are already
  named local diagnostic boundaries, and exact coefficients such as halves, double-angle factors,
  and unit clamps should remain inline. Still-raw production policy remains in the `0.9` reference-
  axis switches and the locator's distinct target density; QUAL-001E therefore remains active.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- The matched release `tools` artifacts were completely byte-identical, both with SHA-256
  `f15123985e07e8a880813669dcdc3a12c2488f0f66bd4be688717b04118172ef`. No counter or quiet
  wall-clock run was warranted.

## QUAL-001E reference-axis policy result

The eleventh numerical/policy-constant slice was validated on 2026-07-18 against immediate parent
`fcd12c4`.

- The repeated helper-axis component boundary moved to `policy.rs` as separately typed
  `REFERENCE_AXIS_COMPONENT_SWITCH_F32` and `REFERENCE_AXIS_COMPONENT_SWITCH_F64` construction
  policies. The `f64` value serves the Delaunay dual and near-great-circle coverage paths; the
  `f32` value serves projected local rebuilding.
- Every site retains the exact value `0.9`, strict `<` comparison, and X-on-true/Y-on-false choice;
  equality therefore still selects Y. Keeping both types avoids casts or type widening in promoted
  geometry.
- Tool helpers and the feature-only global-Delaunay A/B probe retain local literals. They are not
  production policy consumers and were deliberately excluded from the shared policy surface.
- `cargo fmt`, both default and native all-feature Clippy with warnings denied, the complete release
  and checked suites, the no-default-features release suite, and the `serde,glam` release suite
  passed.
- The matched release `tools` artifacts had identical sizes (`2,183,020` text, `55,536` data, `592`
  BSS) and byte-identical `.text`, `.rodata`, exception-table, and unwind sections. The whole-file
  difference was confined to build/symbol metadata and 18 changed source-location bytes in
  `.data.rel.ro`; executable code and numeric data did not change. No counter or quiet wall-clock
  run was warranted.

## QUAL-001E final grid-policy result

The twelfth and final numerical/policy-constant slice was validated on 2026-07-18 against immediate
parent `197e539`.

- The locator's raw target density moved to `policy.rs` as
  `LOCATOR_GRID_TARGET_DENSITY: f64`. Its value remains exactly `16.0`, and the resolution formula,
  truncating cast, and minimum resolution remain unchanged. It is explicitly independent of the
  tuned kNN construction density and its environment override.
- The tools-only low-degree neighbor diagnostic now names its module-local `f32` spatial-hash cell
  side as `LOW_DEGREE_NEIGHBOR_GRID_CELL_SIZE`. The exact `1e-4` value, reciprocal calculation,
  neighboring-bin scan, and separate `LOW_DEGREE_DUPLICATE_EPS` classification threshold are
  unchanged.
- A closing inventory of non-test source classified the remaining nontrivial literals as registry
  tolerances, named policy or diagnostic constants, exact formula coefficients, or deliberately
  local feature/tool values. No unclassified production policy remains, closing QUAL-001E.
- `cargo fmt`, both default and all-feature Clippy with warnings denied, the complete release and
  checked suites, the no-default-features release suite, and the `serde,glam` release suite passed.
  The explicit `tools,profiling` release suite also passed for the diagnostic consumer.
- The matched release `tools` artifacts had identical sizes (`2,183,020` text, `55,536` data, `592`
  BSS) and byte-identical `.text`, `.rodata`, exception-table, and unwind sections. The whole-file
  difference was confined to build/symbol metadata and 23 changed source-location bytes in
  `.data.rel.ro`; executable code and numeric data did not change. No counter or quiet wall-clock
  run was warranted.

## QUAL-001I architecture vocabulary result

The first durable-documentation slice was validated on 2026-07-19 against immediate parent
`d04b085`.

- `architecture.md` now defines the execution-ordered stages from input adaptation through
  original-index remapping, with an explicit contract and primary owner for each. Assembly, edge
  reconciliation, local-rebuild acceptance, and output resolution are separate terms; validation
  and derived views are consumers rather than hidden repair stages.
- The module map now assigns construction/query/performance policy to `policy.rs` and distinguishes
  the `tools`-only quality surface from the `profiling`-only point audit. The `cube_grid` header no
  longer advertises removed kNN/range methods, and the `live_dedup` header describes current
  sharded ownership and assembly rather than a versioned design sketch.
- All-feature Clippy with warnings denied, all-feature doc tests, and the release API/correctness
  suites passed.
- The matched release `tools` artifacts had identical sizes (`2,183,020` text, `55,536` data, `592`
  BSS) and byte-identical `.text`, `.rodata`, exception-table, and unwind sections. Only 28
  source-location bytes in `.data.rel.ro` moved; executable code and numeric data did not change,
  so no counter or quiet wall-clock run was warranted.

## QUAL-001I source-comment result

The second durable-documentation slice was validated on 2026-07-19 against immediate parent
`1c7d3aa`.

- Host-specific codegen, grid-policy, reconciliation, and local-rebuild measurements moved from
  production comments into `performance.md#source-pinned-performance-decisions`. Source comments
  retain the local invariant and link to that record instead of embedding mutable timing history.
- Comparative `old`/`legacy` wording was replaced with the actual alternative contract: all-pairs
  probes, full-rewrite or eager-map oracles, per-builder normalization, second-pass demotion, and
  unit-generator formulas. Remaining `current` uses refer to live runtime state; correctness
  tolerance evidence and intentional fixture/probe names remain explicit.
- All-feature Clippy with warnings denied, all-feature doc tests, and the release API/correctness
  suites passed.
- The matched release `tools` artifacts had identical sizes (`2,183,020` text, `55,536` data, `592`
  BSS) and byte-identical `.text`, `.rodata`, exception-table, and unwind sections. Only 74
  source-location bytes in `.data.rel.ro` moved; executable code and numeric data did not change,
  so no counter or quiet wall-clock run was warranted.

## QUAL-001I current-guidance result

The final durable-documentation slice was validated on 2026-07-19 against immediate parent
`9f93333`.

- Cargo metadata confirmed 11 explicit features, 18 default integration-test targets, and four
  feature-gated test targets. `AGENTS.md` now matches those sets and the live module tree; README
  distinguishes the three semver-covered features from internal repository hooks.
- The compiled-code environment inventory contains every `VORONOI_*` name present in Rust source
  and tests. Its scope now excludes shell-only orchestration and external build/allocator contracts,
  and it records the two private child/sentinel variables used only by tests.
- All-feature Clippy with warnings denied, all-feature doc tests, and the release API/correctness
  suites passed.
- The matched release `tools` artifacts had identical sizes (`2,183,020` text, `55,536` data, `592`
  BSS), build IDs, and byte-identical stripped files. Executable, read-only, relocation-backed, and
  unwind sections were also byte-identical, so no counter or quiet wall-clock run was warranted.

## QUAL-001G first typed-identity result

The first typed-identity slice was validated on 2026-07-19 against immediate parent `ddb1fee`.

- A transparent `CellId(u32)` now guards `WorkingDiagram::splice_generator`; conversion occurs at
  the production mutation call site and focused unit fixture, while the overlay's maps, sets,
  boundaries, and vertex ids retain their existing raw representations. Unit coverage pins the
  wrapper's value, size, and alignment.
- A broader reconciliation-to-pipeline seed-pair owner was measured and reverted. Seven
  interleaved 500k single-threaded Fibonacci pairs showed repeatable +0.1602% instructions and
  +1.6619% branches despite near-neutral size.
- The accepted splice-local candidate changed release size from `2,183,020` to `2,183,028` text
  bytes, left data at `55,536`, and reduced BSS from `592` to `576`. Across the same seven counter
  pairs, mean candidate/parent ratios were `0.999999342` instructions and `1.000000817` branches,
  with no directional signal. Wall clock was intentionally ignored on the busy host.
- All-target/all-feature Clippy with warnings denied, the complete release suite, and compilation
  of the feature-gated local-rebuild probe target passed.

## QUAL-001G vertex-lookup result

The second typed-identity slice was validated on 2026-07-19 against immediate parent `8126d07`.

- A transparent `VertexId(u32)` now guards `WorkingDiagram::vpos` and `WorkingDiagram::vkey`.
  Explicit construction occurs at lookup sites; vectors, maps, sets, sorted records, probe/public
  data, and `vid_for` output retain raw `u32` representations.
- Unit coverage pins both overlay identity wrappers' values, sizes, and alignments. The matched
  release `tools` artifacts have identical sizes (`2,183,028` text, `55,536` data, `576` BSS).
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `0.999999422` instructions and `0.999998669` branches, with pair ranges
  `0.999995618..=1.000002622` and `0.999993350..=1.000000549`; there was no directional signal.
  Wall clock was intentionally ignored on the busy host.
- All-target/all-feature Clippy with warnings denied, the complete release suite, and compilation
  of the feature-gated local-rebuild probe target passed.

## QUAL-001G owner/creation result

The third typed-identity slice was validated on 2026-07-19 against immediate parent `933f312`.

- `WorkingDiagram::vid_for` now returns `VertexId`, and `WorkingDiagram::owners` accepts it. New
  and cached ids therefore remain typed through creation, key/position lookup, and owner lookup;
  conversion back to `u32` occurs only when the splice path stores its boundary vector. Existing
  raw vectors, maps, sets, sorted records, and probe/public representations are unchanged.
- The release `tools` artifact changed from `2,183,028` to `2,183,064` text bytes, retained
  `55,536` data bytes, and changed from `576` to `544` BSS bytes. At section granularity, `.text`
  added 32 bytes and `.eh_frame` added four, while relocation padding fell by 32 bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.000003070` instructions and `0.999998483` branches, with pair ranges
  `1.000000984..=1.000004778` and `0.999989737..=1.000001551`; there was no directional signal.
  Wall clock was intentionally ignored on the busy host.
- Formatting, all-target/all-feature Clippy with warnings denied, the complete release suite, and
  compilation of the feature-gated local-rebuild probe target passed.
- The local overlay's `VertexId` boundary now has a natural endpoint. Extending the wrapper through
  raw traversal collections would add conversion syntax without making an operation's contract
  clearer, so further adoption requires a distinct identity boundary rather than mechanical spread.

## QUAL-001B first live-layout reader result

The first live-layout slice was validated on 2026-07-19 against immediate parent `e038336`.

- A private `LiveCellLayout` now pairs `VoronoiCell` records with their backing index buffer. It
  offers record-based live-span access plus checked lookup whose typed errors distinguish an
  invalid cell id from a live span beyond the buffer. Independent lifetimes correctly express that
  returned spans borrow only the index buffer.
- Scalar and parallel topology summaries now use record-based access, while reconciliation's
  existing shared reader delegates checked access to the view. Unit tests pin stale-tail exclusion
  and both malformed-layout outcomes. Storage, compaction, and mutation behavior are unchanged.
- The accepted accessor preserves the old explicit check sequence: cell bound, span end bound,
  then normal slicing. An initial `slice.get(start..end)` form was reverted after seven interleaved
  500k single-threaded Fibonacci pairs showed repeatable +0.1337% instructions and +1.6620%
  branches.
- The accepted release `tools` artifact changed from `2,183,064` to `2,183,140` text bytes, from
  `55,536` to `55,512` data bytes, and from `544` to `504` BSS bytes, for 12 bytes more overall.
  Across seven counter pairs, mean candidate/parent ratios were `0.999737702` instructions and
  `1.000000400` branches, with pair ranges `0.999736172..=0.999739930` and
  `0.999997418..=1.000004913`. Wall clock was intentionally ignored on the busy host.
- Formatting, all-target/all-feature Clippy with warnings denied, the complete release and checked
  suites, and the no-default-features release suite passed.

## QUAL-001B threaded segment-reader result

The second live-layout slice was validated on 2026-07-19 against immediate parent `d2467c8`.

- The shared-edge segment reader and its reuse-buffer form now take one `LiveCellLayout` instead of
  independent cell/index slices. Primary merge collection, rejected-component seed discovery,
  optional reconciliation telemetry, and focused cross-module tests construct the pairing once per
  read operation. A test that deliberately shrinks a cell between rounds reconstructs the view
  after each mutation, making its borrow boundary explicit.
- The release executable file changed from `2,999,072` to `2,999,024` bytes. Section accounting
  moved 464 bytes into `.text` and 392 bytes out of unwind data; a 4 KiB virtual alignment shift
  does not increase the file.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `0.999999033` instructions and `0.999999956` branches, with pair ranges
  `0.999994779..=1.000001060` and `0.999998867..=1.000000665`; there was no directional signal.
  Wall clock was intentionally ignored on the busy host.
- Formatting, all-target/all-feature Clippy with warnings denied, the complete release suite, and
  the checked suite passed.

## QUAL-001B rejected semantic-comparison experiment

The next candidate was measured on 2026-07-19 against immediate parent `d925745` and reverted.

- `cell_spans_differ` was changed from four independently pairable slices to two
  `LiveCellLayout` values. The executable became eight bytes smaller and aggregate text accounting
  fell by 252 bytes, so code size was not the rejection reason.
- The default form produced mean candidate/parent ratios of `1.001597135` instructions and
  `1.016621779` branches across seven interleaved 500k single-threaded Fibonacci pairs. Every pair
  regressed. Marking the cold comparison never-inline produced `1.001595237` and `1.016619152`;
  forcing it always-inline produced `1.001597995` and `1.016619828`.
- The invariant result across all three compiler shapes shows that this signature change perturbs
  clean-path optimization outside its nominally cold work. The raw semantic-comparison signature
  remains in place; retry only after a material surrounding codegen or compiler change.

## QUAL-001B localized duplicate-reader result

The third accepted live-layout slice was validated on 2026-07-19 against immediate parent
`33d5888`.

- `localized_dup_key_unions` now takes one `LiveCellLayout`. `collect_merges` constructs the view
  once and reuses it for the defect-only duplicate-key BFS and the shared-edge segment scan,
  preventing those readers from observing differently paired cell/index slices.
- The focused localized-versus-global duplicate-scan oracle passed. Aggregate release section
  sizes were identical (`2,183,212` text, `55,512` data, `4,520` BSS), while the executable file
  changed from `2,999,024` to `2,998,992` bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.000001526` instructions and `1.000002087` branches, with pair ranges
  `0.999998548..=1.000004487` and `0.999997928..=1.000011972`; there was no directional signal.
  Wall clock was intentionally ignored on the busy host.
- Formatting, all-target/all-feature Clippy with warnings denied, the complete release suite, and
  the checked suite passed.

## QUAL-001B rejected unpaired-reader experiment

The next reader-family candidate was measured on 2026-07-19 against immediate parent `7f45956`
and reverted.

- The whole-family form passed one `LiveCellLayout` through the unpaired-scan entry, localized
  region scan, partner-cell edge count, and debug global oracle. Seven interleaved 500k
  single-threaded Fibonacci pairs produced mean candidate/parent ratios of `1.001598315`
  instructions and `1.016619163` branches; every pair regressed.
- A split form restored the raw outer ABI and constructed the view inside the entry, leaving only
  the localized scan, partner lookup, and debug oracle typed. It repeated the same signal:
  `1.001599651` instructions and `1.016624977` branches.
- Code size was not deciding evidence: the whole form added 360 executable bytes and the split form
  added 16, while aggregate mapped accounting was flat or smaller. Both implementations were
  reverted, and the rebuilt source is identical to the parent.

## QUAL-001B checked structural-audit result

The checked-build invariant slice was validated on 2026-07-19 against immediate parent `47d2e02`.

- `LiveCellLayout::debug_assert_valid` checks that the cell count and backing index-buffer length
  fit their u32-backed representations, then verifies every record's live span is contained in the
  buffer. Unit coverage includes a valid stale-tail layout and a malformed out-of-bounds span.
- Reconciliation invokes the audit only after its empty-record early return. Clean checked runs
  retain their existing fast path; defect-bearing checked runs validate the pairing once before
  readers or mutators rely on it. The method and call are both absent when debug assertions are
  disabled.
- The release `tools` artifact retained identical aggregate accounting (`2,183,212` text, `55,512`
  data, `4,520` BSS). `.text`, `.rodata`, `.eh_frame`, and `.gcc_except_table` were byte-identical,
  and executable symbol addresses were unchanged. The file grew 40 bytes solely through changed
  compiler-symbol/source-location and build metadata; no runtime counter comparison was warranted.
- The focused checked test passed. Formatting, all-target/all-feature Clippy with warnings denied,
  and the complete release and checked suites passed.

## QUAL-001B rejected mutable-layout experiment

The first mutation-owner candidate was measured on 2026-07-19 against immediate parent `51669ba`
and reverted.

- A private `LiveCellLayoutMut` paired mutable cell records with their backing index buffer. Its
  `rewrite_and_shrink` operation wrote a shorter cycle into the existing prefix, updated the cell
  count, and deliberately preserved the stale tail. The defect-only collinear-drop path used it
  without changing its outer signature or malformed-span behavior; a focused unit test pinned both
  live-cycle and stale-tail results.
- The helper was fully inlined, but the release artifact reproduced the earlier optimizer-cliff
  fingerprint: aggregate text fell from `2,183,212` to `2,182,960` bytes, BSS fell from `4,520` to
  `680` bytes, and the executable became 48 bytes smaller.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.001598698` instructions and `1.016618637` branches, with pair ranges
  `1.001596324..=1.001600426` and `1.016613236..=1.016624389`. Every pair regressed; all samples
  recorded zero context switches and CPU migrations.
- The implementation was reverted. Rebuilding the restored source reproduced the initially
  captured parent artifact hash, confirming that no production change remains.

## QUAL-001C validation-oracle expansion

The pre-extraction validation oracle was expanded on 2026-07-19 against immediate parent `9dd46db`.

- Exact fast-diagram/effective-array reasons are now pinned for low incidence, invalid vertex ids,
  degeneracy, duplicate vertex ids, duplicate cell signatures, grouped edge-use failures,
  owner-conditioned antipodal edges, disconnected subdivisions, and bad Euler characteristic.
- A connected, closed, oriented 3x3 toroidal quadrangulation with degree-four vertices isolates the
  Euler reason (`V-E+F = 0`) from the earlier connectivity check. Separate effective-only fixtures
  pin generator/cell cardinality and invalid live-span failures.
- An exhaustive enumeration of small cycles proves that fail-fast self-loop classification is
  dominated by duplicate-id or degeneracy checks; representative fixtures pin the observable
  earlier reasons. Accumulating-report fixtures independently pin boundary, overused, and
  same-direction edge counters.
- All additions are test-only. The complete release `tools` artifact, including its SHA-256 hash
  and file size, is byte-identical to the parent, so no counter comparison is warranted.

## QUAL-001C typed edge-use classification

The first shared validation fact was accepted on 2026-07-19 against immediate parent `367dc4e`.

- Private `EdgeUseClass` and `classify_edge_uses` now define paired, boundary, overused, and
  same-direction outcomes once. The two fail-fast gates map all non-paired outcomes to the existing
  `"unpaired, overused, or misoriented edge"` reason; the accumulating report maps them to its
  existing separate counters.
- The release `tools` artifact changed from `2,183,212` to `2,183,224` text bytes, retained `55,512`
  data bytes, changed from `4,520` to `4,504` BSS bytes, and grew from `2,999,032` to `2,999,048`
  file bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `0.999995974` instructions and `0.999993778` branches, with pair ranges
  `0.999991449..=1.000002457` and `0.999971992..=1.000009634`. There was no directional regression;
  every sample recorded zero context switches and CPU migrations.

## QUAL-001C typed strict-reason experiment

The proposed fail-fast reason enum was rejected on 2026-07-19 against immediate parent `2813e0e`.

- A private `StrictValidationIssue` represented every reason returned by the two fail-fast
  validators. Its exact-text mapping preserved existing logging, tests, and diagnostics; the
  effective parallel scan retained its `(cell, check_rank)` first-failure ordering. The accumulating
  report was unchanged.
- The release `tools` artifact changed from `2,183,224` to `2,183,324` text bytes, from `55,512` to
  `55,632` data bytes, from `4,504` to `4,304` BSS bytes, and from `2,999,048` to `3,000,384` file
  bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.001865670` instructions and `1.016621585` branches, with pair ranges
  `1.001860072..=1.001870619` and `1.016612677..=1.016631824`. Every pair regressed; every sample
  recorded zero context switches and CPU migrations.
- The implementation was reverted. Static fail-fast strings remain the measured Pareto choice
  until surrounding codegen changes enough to justify retesting.

## QUAL-001C dominated self-loop branches

The dead fail-fast branches were removed on 2026-07-19 against immediate parent `6099b9f`.

- Both strict validators already reject every self-loop cycle during the earlier duplicate-id or
  degeneracy checks, as pinned by exhaustive small-cycle coverage. Their unreachable
  `"self-loop edge"` branches and the now-unused effective-scan rank were deleted. The accumulating
  report still counts self-loops, with a direct regression assertion.
- The release `tools` artifact changed from `2,183,224` to `2,183,164` text bytes, retained `55,512`
  data bytes, changed from `4,504` to `472` BSS bytes, and shrank from `2,999,048` to `2,998,984`
  file bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `0.999998353` instructions and `0.999999721` branches, with pair ranges
  `0.999994632..=1.000003450` and `0.999990385..=1.000012341`. There was no directional regression;
  every sample recorded zero context switches and CPU migrations.

## QUAL-001C weld-policy oracle

The weld-specific policy boundary was pinned on 2026-07-19 against immediate parent `b730470`.

- A deliberately corrupt alias maps one cell to a canonical cell with a different boundary. The
  fast validator must return the exact `"weld map"` reason; the accumulating report must record one
  welded twin, one weld-map issue, and a non-strict verdict.
- The addition is test-only. The complete release `tools` artifact retained SHA-256
  `8613a4c080929a18d960e93da2212f18d0be8b2c6c415cf0979d9d1e641eb946` and file size `2,998,984`
  bytes exactly, so no counter comparison was warranted.

## QUAL-001C shared weld-predicate experiment

The proposed weld-alias predicate was rejected on 2026-07-19 against immediate parent `2db1ffc`.

- One inline helper owned the canonical-target and identical-boundary checks duplicated by the fast
  and accumulating validators. Both callers retained their existing traversal, twin count, and
  fail-fast versus accumulating behavior; the weld-policy oracle passed.
- The release `tools` artifact changed from `2,183,164` to `2,183,216` text bytes, from `55,512` to
  `55,464` data bytes, from `472` to `488` BSS bytes, and from `2,998,984` to `2,999,264` file bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.001603680` instructions and `1.016618430` branches, with pair ranges
  `1.001600829..=1.001607581` and `1.016615165..=1.016621164`. Every pair regressed; every sample
  recorded zero context switches and CPU migrations.
- The helper was reverted. Rebuilding restored parent SHA-256
  `8613a4c080929a18d960e93da2212f18d0be8b2c6c415cf0979d9d1e641eb946`; the duplicated local
  expression remains the measured Pareto choice.

## QUAL-001A local-rebuild status enum

The first lifecycle-state migration was accepted on 2026-07-19 against immediate parent `520ff78`.

- Public non-exhaustive `LocalRebuildStatus` distinguishes `NotTriggered`, `Disabled`, `Rejected`,
  `Accepted`, and the doc-hidden diagnostic-capture path. The same status flows through the
  internal outcome. Low-incidence and Euler defect facts remain separate.
- `LocalRebuildReport` now stores the status and derives `attempted()`/`accepted()` from it. All
  repository consumers migrated atomically; the machine-readable `local_rebuild_attempted` and
  `local_rebuild_accepted` field names and boolean values are unchanged. The impossible
  false-attempted/true-accepted state is no longer representable.
- The release `tools` artifact changed from `2,183,164` to `2,183,376` text bytes, retained `55,512`
  data bytes, changed from `472` to `4,360` BSS bytes, and grew from `2,998,984` to `2,999,192` file
  bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.000001952` instructions and `0.999998724` branches, with pair ranges
  `0.999994359..=1.000005231` and `0.999993544..=1.000004873`. There was no directional regression;
  every sample recorded zero context switches and CPU migrations.

## QUAL-001A resolution discovery mode

The second lifecycle-state migration was accepted on 2026-07-19 against immediate parent
`1faedea`.

- Private `ResolutionDiscoveryMode` has exactly `CertifiedHint` and
  `ExhaustiveDriftFallback` states, replacing the exact-inverse `certified_hint` and
  `drift_fallback` booleans. Exact-zero candidate discovery branches directly on the mode.
- Timing now stores only the fallback bit. Human-readable mode output and the machine-readable
  `resolution_certified_hint` and `resolution_fallback_drift` fields are derived with their exact
  existing names and values.
- The release `tools` artifact changed from `2,183,376` to `2,183,392` text bytes, retained
  `55,512` data bytes, changed from `4,360` to `4,344` BSS bytes, and retained both aggregate size
  `2,243,248` and file size `2,999,192` bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.000000548` instructions and `1.000004166` branches, with pair ranges
  `0.999995682..=1.000005031` and `0.999995948..=1.000011936`. There was no directional regression;
  every sample recorded zero context switches and CPU migrations.

## QUAL-001A effective-input ownership

The third lifecycle-state migration was accepted on 2026-07-19 against immediate parent
`a7595c2`.

- Private `EffectiveInput` has exactly identity and merged states. The identity state borrows the
  canonicalized original points; the merged state owns the complete `MergeResult`, including its
  representative points and original-to-effective map.
- `PipelineState` no longer contains independently optional effective points and merge metadata.
  The preparation phase returns a named `PreparedPointsAndGrid` record instead of an ambiguous
  four-element tuple. Preprocess report counts derive from the effective-input owner.
- A direct test pins disabled, weld-with-no-merge, and actual-merge preparation. The complete
  release, checked, no-default-feature, and all-feature Clippy gates passed, including existing API
  coverage for effective diagrams, standalone large-threshold welding, and final remapping.
- The matched release `tools` artifact changed from `2,181,431` to `2,180,775` text bytes, retained
  `55,456` data bytes, changed from `2,259` to `2,915` BSS bytes, retained aggregate size
  `2,239,146`, and shrank from `2,995,584` to `2,994,920` file bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `0.999998159` instructions and `0.999998587` branches, with pair ranges
  `0.999993028..=1.000001173` and `0.999987382..=1.000003031`. There was no directional regression;
  every sample recorded zero context switches and CPU migrations.

## QUAL-001A lifecycle rename map

The migration is intentionally breaking and atomic across the compiling repository. No deprecated
aliases are added. Geometry, control flow, numerical constants, report contents, and state shapes
remain unchanged; the invalid-state cleanup recorded in QUAL-001A follows as a separate commit.

### Public API and report surface

| Current | Replacement |
|---|---|
| `RepairMode` | `LocalRebuildMode` |
| `RepairMode::Disabled` | `LocalRebuildMode::Disabled` |
| `RepairMode::Local3d` | `LocalRebuildMode::Hull3d` |
| `RepairMode::LocalProjected` | `LocalRebuildMode::ProjectedDelaunay` |
| `VoronoiConfig::repair_mode` | `VoronoiConfig::local_rebuild_mode` |
| `VoronoiConfig::with_repair_mode` | `VoronoiConfig::with_local_rebuild_mode` |
| `RepairReport` | `LocalRebuildReport` |
| `ComputeReport::repair` | `ComputeReport::local_rebuild` |
| `pre_repair_edge_mismatch_count` | `assembly_edge_mismatch_count` |
| `pre_repair_edge_mismatches` | `assembly_edge_mismatches` |
| `post_repair_unpaired_edges` | `residual_unpaired_edges` |
| `post_repair_escalation_pairs` | `residual_reconciliation_pairs` |
| `unresolved_edge_pairs` | `reconciliation_edge_records` |
| `has_post_repair_residuals` | `has_output_residuals` |
| `UnresolvedEdgeOrigin` | `EdgeMismatchOrigin` |
| `UnresolvedEdgeOrigin::PostRepairUnpaired` | `EdgeMismatchOrigin::PostReconciliationUnpaired` |

`assembly_edge_mismatch_*` names facts detected by live dedup/assembly before post-assembly
reconciliation. `reconciliation_edge_records` remains the diagnostic aggregate that includes
initial facts and any synthesized post-reconciliation backstop records. The residual fields name
facts about the returned output rather than implying that one ambiguous repair stage ran.

The affected configuration and report types do not derive serde. The serde audit found only point,
diagram, and cell-mesh wire types; their field names and formats are outside this migration.

### Reconciliation internals

| Current | Replacement |
|---|---|
| `UnresolvedEdgeMismatch` | `EdgeMismatch` |
| `unresolved_edges` variables | `edge_mismatches` |
| `reconcile_unresolved_edges` | `reconcile_edge_mismatches` |
| `RepairApply` | `ReconcileApply` |
| `repair_apply_from_env` | `ReconcileOptions::read_from_env` |
| `MAX_REPAIR_ROUNDS` | `MAX_RECONCILIATION_ROUNDS` |
| `run_repair_rounds` | `run_reconciliation_rounds` |
| reconciliation-local `repaired_*` variables | `reconciled_*` |
| `escalation_pairs` / `reconciliation_escalations` | `local_rebuild_seed_pairs` |
| `escalation_error` | `reconciliation_rejection_error` |

The existing `edge_reconcile.rs`, `ReconcileResult`, `reconcile_edges`, `residual_pairs`, and
`merge_affected_cells` names already describe their stage and remain.

### Local-rebuild internals

| Current | Replacement |
|---|---|
| `knn_clipping/escalate.rs` / module `escalate` | `knn_clipping/local_rebuild.rs` / `local_rebuild` |
| `RepairOutcome` | `LocalRebuildOutcome` |
| `RepairResult` | `LocalRebuildResult` |
| `maybe_repair_effective` | `maybe_rebuild_effective` |
| `RepairVertex` | `RebuildVertex` |
| `RepairFan` | `RebuildFan` |
| `repair_grow_loop` | `run_rebuild_growth` |
| `repair_local_hull` | `rebuild_with_local_hull` |
| `repair_local_exact` | `rebuild_with_projected_delaunay` |
| `repair_delaunator` | `rebuild_with_global_delaunay` |
| `EscalationStats` | `LocalRebuildStats` |
| `ESCALATE_GATHER_K` | `LOCAL_REBUILD_GATHER_K` |
| `ESCALATE_MAX_ROUNDS` | `LOCAL_REBUILD_MAX_ROUNDS` |
| `escalation_enabled` | `local_rebuild_probe_forced` |
| `set_escalation_enabled` | `set_local_rebuild_forced` |
| local-rebuild `repair_*` variables | corresponding `rebuild_*` names |
| `resolution_repair_scan_cells` | `resolution_rebuild_scan_cells` |

This commit does not convert `LocalRebuildOutcome { attempted, accepted, ... }` into the planned
state enum. Keeping the rename and state-model change separate preserves attribution and gives the
later commit a clear behavioral diff.

### Features, probes, environment, and tools

| Current | Replacement / action |
|---|---|
| Cargo feature and root module `escalate_probe` | `local_rebuild_probe` |
| `VORONOI_MESH_RECLIP_REPAIR` | remove; no production reader exists |
| `VORONOI_MESH_REPAIR_MODE` | `VORONOI_MESH_LOCAL_REBUILD_MODE` |
| mode value `local3d` | `hull3d` |
| mode value `projected` | `projected-delaunay` |
| `VORONOI_MESH_EDGE_REPAIR_REBUILD` | `VORONOI_MESH_RECONCILE_REBUILD` |
| `VORONOI_MESH_EDGE_REPAIR_GLOBAL_DUPSCAN` | `VORONOI_MESH_RECONCILE_GLOBAL_DUPSCAN` |
| `VORONOI_MESH_ESCALATE_DEBUG` | `VORONOI_MESH_LOCAL_REBUILD_DEBUG` |
| `VORONOI_MESH_ESCALATE_DELAUNATOR` | `VORONOI_MESH_LOCAL_REBUILD_GLOBAL_DELAUNAY` |
| `VORONOI_MESH_ESCALATE_PROBE_A0` | `VORONOI_MESH_LOCAL_REBUILD_PROBE_A0` |
| `VORONOI_MESH_ESCALATE_DIST` | `VORONOI_MESH_LOCAL_REBUILD_DIST` |
| `VORONOI_MESH_ESCALATE_N` | `VORONOI_MESH_LOCAL_REBUILD_N` |
| `VORONOI_MESH_ESCALATE_SEED` | `VORONOI_MESH_LOCAL_REBUILD_SEED` |
| `VORONOI_MESH_ESCALATE_K` | `VORONOI_MESH_LOCAL_REBUILD_K` |
| benchmark option `--no-repair` | `--no-local-rebuild` |
| KV fields `repair_attempted` / `repair_accepted` | `local_rebuild_attempted` / `local_rebuild_accepted` |

Campaign scripts and parsers migrate in the same commit. None of these probe/diagnostic names gets
a fallback reader for the old spelling.

### Test targets

| Current | Replacement |
|---|---|
| `tests/edge_repair_net.rs` | `tests/edge_reconciliation.rs` |
| `tests/escalate_local.rs` | `tests/local_rebuild.rs` |
| `tests/escalate.rs` | `tests/local_rebuild_probe.rs` |
| `tests/reclip_repair.rs` | `tests/local_rebuild_contract.rs` |

Test function names and assertions follow the same stage-specific vocabulary. The no-op
`VORONOI_MESH_RECLIP_REPAIR` setup is deleted from `local_rebuild_contract`; serialization remains
only for environment variables that the test actually changes.

### Documentation boundary

Update current contract and operational material in `README.md`, `docs/architecture.md`,
`docs/correctness.md`, `docs/performance.md`, the crate docs, scripts, and active work-log text.
Use **reconciliation** for identity/cycle cleanup and **local rebuild** for the Hull3d or projected
Delaunay replacement transaction.

Do not mechanically rewrite closed audit records, historical benchmark narratives, or rejected
experiment descriptions when `repair` is part of the recorded historical name. Add a short mapping
only where a current command, identifier, or link would otherwise become unusable.

## Atomic migration boundary and gates

The first implementation commit may span public API, production internals, tests, tools, scripts,
and current docs because splitting it would require temporary aliases or a non-compiling tree. It
must not include state-enum redesign, tolerance movement, algorithm changes, or phase extraction.

Minimum validation:

```bash
cargo fmt
cargo clippy --all-targets
RUSTFLAGS="-C target-cpu=native" cargo clippy --all-targets --all-features
cargo test --release
cargo test --profile checked
cargo test --release --no-default-features
cargo test --release --features serde,glam
cargo test --release --features local_rebuild_probe --test local_rebuild_probe
```

Then rerun the four semantic fingerprint commands and the two renamed defect-bearing targets. Build
the candidate and immediate parent together and run the four counter cells above interleaved. A
quiet wall-clock run is required only by the comparison rule's unexplained repeatable signal.
