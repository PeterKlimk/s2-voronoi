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

## QUAL-001B semantic-comparison experiment and layout retest

The first form was measured on 2026-07-19 against immediate parent `d925745` and reverted.

- `cell_spans_differ` was changed from four independently pairable slices to two
  `LiveCellLayout` values. The executable became eight bytes smaller and aggregate text accounting
  fell by 252 bytes, so code size was not the rejection reason.
- The default form produced mean candidate/parent ratios of `1.001597135` instructions and
  `1.016621779` branches across seven interleaved 500k single-threaded Fibonacci pairs. Every pair
  regressed. Marking the cold comparison never-inline produced `1.001595237` and `1.016619152`;
  forcing it always-inline produced `1.001597995` and `1.016619828`.
- The invariant result across all three inline shapes showed that this signature change perturbed
  clean-path optimization outside its nominally cold work. The raw semantic-comparison signature
  remained in place pending a material surrounding-codegen or compiler change.

The candidate was re-audited on 2026-07-20 after reconciliation orchestration and the surrounding
pipeline had changed materially. The two-layout signature is now retained.

- The ordinary benchmark cannot execute the changed comparison: `cell_spans_differ` is reachable
  only through `ReconcileApply::Rebuild`, while the default benchmark uses the in-place backend.
  Any ordinary-run counter movement therefore comes from compiler partition/layout effects rather
  than work performed by the new abstraction.
- The default multi-codegen-unit artifact reproduced the family fingerprint at a changed scale.
  Seven-pair candidate/parent instruction and branch means were `1.000996118` / `1.013599487` on
  500k Fibonacci, `1.000966835` / `1.011605457` on 500k uniform seed 12345,
  `1.000151321` / `1.001864842` on 100k clustered seed 1, and `1.000075869` /
  `1.000731124` on 100k mega seed 1. Cycles were noise-dominated and directionally favorable in
  all four matrices; every sample recorded zero context switches and CPU migrations.
- Rebuilding the same source pair with `-C codegen-units=1` removed the displacement completely.
  Candidate and parent had identical section and file sizes; `.text`, `.rodata`, exception, and
  unwind sections were byte-identical. Seven 500k Fibonacci pairs were neutral at
  `0.999999795` instructions and `0.999998433` branches. This isolates the default-build result to
  codegen partitioning rather than the typed comparison itself.
- Under the retained default build the candidate removes 444 text bytes, adds 448 BSS bytes,
  changes aggregate accounting by four bytes, and reduces file size by 168 bytes. The comparison
  now requires two coherent layouts at its boundary; exact rebuild/in-place reconciliation
  coverage remains the active-path semantic oracle.

This retest also changes how the repeated 2026-07-19 fingerprint should be read. The rejected
unpaired-reader, mutable-layout, strict-reason, weld-predicate, and effective-validation forms all
changed code that was inactive in their clean Fibonacci counter workload and produced the same
roughly `+0.16%` instructions / `+1.66%` branches (later `+0.129%` / `+1.36%`) artifact. Those are
evidence that the then-current default artifact moved to a less favorable optimizer/layout state,
not evidence that each abstraction performed that much extra work. Subsequent retests accepted the
unpaired-reader, strict-reason, and effective-validation boundaries. The low-value mutable helper
and weld predicate remain deferred; future inactive-path retests should include an alternate
codegen-partition control before attributing clean-path counters to the changed source.

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

## QUAL-001B unpaired-reader experiment and retest

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

The whole-family form was retested on 2026-07-20 against parent `add4409` after the semantic span
comparison and surrounding reconciliation orchestration changed. It is now retained.

- `scan_unpaired_interior`, its localized region scan, partner-cell lookup, and checked-only global
  oracle now receive one `LiveCellLayout`. Production constructs a fresh view after each possible
  reconciliation mutation, so the paired borrow cannot outlive the arrays it describes.
- Seven-pair candidate/parent instruction and branch means were `0.999998856` / `0.999999338` on
  500k Fibonacci, `0.999999804` / `1.000000368` on 500k uniform seed 12345,
  `0.999998424` / `0.999997622` on 100k clustered seed 1, and `0.999995289` /
  `0.999999346` on 100k mega seed 1. Cycles were noise-dominated, and every sample recorded zero
  context switches and CPU migrations.
- The release artifact added 168 text bytes, removed 160 BSS bytes, grew aggregate accounting by
  eight bytes, and grew the file by 120 bytes. Focused release and checked reconciliation oracles
  passed.

The prior optimizer cliff is absent in the current compiler/surrounding-code shape, so this reader
family no longer needs the alternate-codegen control used for the semantic comparison.

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

## QUAL-001C typed strict-reason experiment and retest

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

The typed taxonomy was retested on 2026-07-20 against parent `4ec759c` and is now retained.

- `StrictValidationIssue` owns all 13 fail-fast identities and their exact messages. Both strict
  validators select variants rather than spelling messages locally; the effective parallel scan
  carries a typed issue through lexicographic `(cell, check_rank)` selection. A direct mapping test
  pins every string. The accumulating report remains independent.
- Default-build seven-pair instruction/branch means were `0.998721841` / `1.001286980` on 500k
  Fibonacci, `0.998828952` / `1.001126394` on 500k uniform seed 12345, `0.999821539` /
  `1.000186691` on 100k clustered seed 1, and `1.000135946` / `1.000075151` on 100k mega seed 1.
  Ordinary regimes retire about 0.12% fewer instructions in exchange for 0.11%--0.13% more
  branches; mega movement is about one basis point. Cycles were unresolved and every sample had
  zero switches/migrations.
- A one-codegen-unit Fibonacci control was neutral at `0.999997298` instructions and
  `0.999995896` branches. The retained default artifact removes 11,376 text and 1,192 BSS bytes,
  adds 280 data bytes, shrinks aggregate accounting by 12,288 bytes, and reduces file size by 7,480
  bytes. The cross-regime/default split is therefore another codegen-layout tradeoff rather than
  work performed by validation on the ordinary path.

The substantial code-size and ordinary instruction reductions, neutral controlled build, exact
typed contract, and absence of a resolved cycle loss justify retaining the enum. The tiny mega
counter increase remains below the practical noise threshold; no forced-inline attribute is kept.

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

- Public non-exhaustive `LocalRebuildStatus` initially distinguished `NotTriggered`, `Disabled`,
  `Rejected`, `Accepted`, and a doc-hidden diagnostic-capture path. Low-incidence and Euler defect
  facts remained separate.
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

A 2026-07-20 API-boundary follow-up removed diagnostic interception from the public enum.

- Private `LocalRebuildExecution` now distinguishes `Completed(LocalRebuildStatus)` from the
  feature-gated A0 capture. The public status has only the four ordinary outcomes; a probe callback
  that completes report construction maps to `NotTriggered` at that final boundary while the
  existing side channel retains the captured state.
- A feature-gated integration-path unit test pins both the capture and the public conversion. The
  default `tools` artifact retains identical text, data, BSS, aggregate, and file sizes. Seven 500k
  Fibonacci pairs were neutral at `0.999997426` instructions and `0.999994791` branches; all
  samples recorded zero context switches and CPU migrations.

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

## QUAL-001A effective-geometry ownership

The fourth lifecycle-state migration was accepted on 2026-07-19 against immediate parent
`c39f83f`.

- Private `EffectiveGeometry` now owns effective-space vertex positions, cell span records, and
  their live index buffer from assembly through final remapping. `PipelineState` no longer carries
  those arrays as independently replaceable fields.
- Reconciliation mutates the geometry owner and returns only `ReconcileResult`; the former
  `ReconciledWithResiduals` tuple is removed. Local rebuilding retains the exact append, strict
  validation, truncate-on-rejection, and cell-array swap-on-acceptance sequence.
- Assembly vertex keys remain separately borrowed partial provenance. Their ownership comment now
  records reconciliation, local-rebuild, and output-resolution consumers plus conservative
  fallback for rebuild-minted ids beyond the assembly store.
- The complete release, checked, no-default-feature, and all-feature Clippy gates passed. Focused
  API, reconciliation, local-rebuild, output-resolution, and effective-mesh elision suites also
  passed.
- The matched release `tools` artifact changed from `2,180,775` to `2,179,291` text bytes, retained
  `55,456` data bytes, changed from `2,915` to `4,403` BSS bytes, changed aggregate size from
  `2,239,146` to `2,239,150`, and shrank from `2,994,920` to `2,993,880` file bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.000000038` instructions and `0.999998878` branches, with pair ranges
  `0.999987021..=1.000009329` and `0.999983860..=1.000013754`. There was no directional regression;
  every sample recorded zero context switches and CPU migrations.

## QUAL-001B local-rebuild overlay layout

The fourth accepted live-layout slice was validated on 2026-07-20 against immediate parent
`72a681d`.

- `WorkingDiagram` now owns one `LiveCellLayout` instead of independently pairable base cell and
  index references. Its renamed `from_reconciled` constructor accepts the already-paired view and
  audits it only in checked builds.
- Base-boundary lookup retains the trusted direct cell-index and slice expression. Cell count,
  residual-scan capacity, and unspliced materialization also read through the view. Override
  selection, minted vertex storage, flattening order, and returned arrays are unchanged.
- Focused coverage includes stale backing slots, base-span reads, override substitution, and final
  flattening. The production Hull3d/projected rebuild, rebuild-contract, reconciliation,
  output-resolution, API, and correctness suites passed, followed by the complete release,
  checked, no-default-feature, and all-feature Clippy gates.
- The matched release `tools` artifact changed from `2,179,291` to `2,179,227` text bytes, from
  `55,456` to `55,408` data bytes, and from `4,403` to `4,499` BSS bytes. Aggregate accounting fell
  from `2,239,150` to `2,239,134`, and file size fell from `2,993,880` to `2,993,832` bytes.
- Across seven interleaved 500k single-threaded Fibonacci counter pairs, mean candidate/parent
  ratios were `1.000000228` instructions and `1.000001841` branches, with pair ranges
  `0.999997202..=1.000004276` and `0.999996770..=1.000010283`. Every sample recorded zero context
  switches and CPU migrations; wall clock was treated as advisory on the shared host.
- No active-overlay counter result is claimed. The deterministic 100k mega fixtures at fraction
  `0.8`, seeds 1, 2, and 15, are now strict-valid with local rebuilding disabled, and a debug run
  of that test did not trigger the overlay. Direct overlay tests and the production rebuild suites
  retain semantic coverage until a natural accepted-splice workload is available.

## QUAL-001B effective-validation layout and retest

The effective-validation migration was measured on 2026-07-20 against immediate parent
`7c70983` and fully reverted.

- The full form passed `LiveCellLayout` from the local-rebuild candidate transaction through
  `verify_sphere_effective_strict` and its parallel cell scan. It preserved exact error ordering,
  messages, malformed-span rejection, capacity, and transaction behavior; the complete release,
  checked, no-default-feature, and all-feature Clippy gates passed.
- Seven 500k single-threaded Fibonacci pairs all regressed. Mean candidate/parent ratios were
  `1.001290682` instructions and `1.013601256` branches, with ranges
  `1.001285718..=1.001295949` and `1.013587754..=1.013614888`.
- Restoring the raw four-slice verifier ABI while retaining one layout only inside the private scan
  did not change the result: `1.001293695` instructions and `1.013603930` branches, ranges
  `1.001289474..=1.001299176` and `1.013599255..=1.013611676`.
- Restoring the validator completely and retaining only overflow-safe `checked_span` hardening also
  reproduced it: `1.001294912` instructions and `1.013604706` branches, ranges
  `1.001284480..=1.001299841` and `1.013583502..=1.013616981`.
- The isolated hardening produced the familiar optimizer-cliff artifact fingerprint: text fell 364
  bytes, BSS fell 3,744 bytes, aggregate accounting fell 4,108 bytes, and the executable file grew
  368 bytes. Every sample in all three experiments recorded zero context switches and CPU
  migrations; wall clock was ignored on the busy host.
- All production changes were reverted. The effective gate retains its raw portable checked-add
  expression, and `LiveCellLayout::checked_span` remains limited to current internal valid-layout
  callers. Retry only after a material compiler or surrounding-codegen change.

The full boundary was retested on 2026-07-20 against parent `b8bd22e` and is now retained.

- The verifier receives generators, candidate vertices, and one `LiveCellLayout`; only the
  cell/index pair is coupled. Parallel scan ordering, static reasons, and transaction timing are
  unchanged. Checked span-end addition and a typed overflow error preserve the former portable raw
  rejection contract.
- Seven-pair instruction/branch means were `1.000296110` / `0.999999107` on 500k Fibonacci,
  `1.000310428` / `1.000001768` on 500k uniform seed 12345, `1.000176884` / `0.999996530` on 100k
  clustered seed 1, and `1.000119413` / `0.999999036` on 100k mega seed 1. Cycles had no adverse
  signal; every sample had zero switches/migrations.
- A one-codegen-unit control was neutral at `1.000002647` instructions and `1.000002939` branches
  on Fibonacci, identifying the small default instruction movement as codegen partitioning. The
  default artifact removes 44 text and 24 data bytes, adds 72 BSS bytes and four aggregate bytes,
  and grows the file by 832 bytes.

The historical optimizer cliff is absent, and the stronger transaction boundary plus portable
overflow behavior justify retaining the current form.

## QUAL-001B assembly-handoff closure

The final live-layout stage was inventoried on 2026-07-20 and closed without a production change.

- Assembly constructs a freshly compacted, generator-ordered layout: checked prefixes partition an
  exactly sized index buffer, direct scatter initializes every slot, and sparse overrides finish
  before any live-window read or return.
- `run_core_pipeline` is the sole production consumer of `AssemblyResult`. It immediately moves
  vertices, cells, and indices into the accepted `EffectiveGeometry` owner, with no intervening
  branch, failure, mutation, or independent use.
- An owned cell-layout field in `AssemblyResult` would be unpacked immediately or propagated into
  the whole post-assembly pipeline. The first form protects no meaningful lifetime; the second
  duplicates the geometry owner and reopens codegen-sensitive mutable signatures.
- A read-only layout inside assembly cannot cover scatter/override mutation and would only replace
  exact-zero hint discovery's one trusted local span expression.
- No candidate offered a maintainability gain proportional to its hot assembly ABI/codegen risk,
  so no runtime measurement was warranted. Retry only if a second consumer or natural shared owner
  creates a real ownership lifetime.

## QUAL-001D reconciliation run state

The first reconciliation-orchestration extraction was accepted on 2026-07-20 against immediate
parent `37b0f65`.

- Private `ReconcileRunState` owns the merge ledger, rejected-component rebuild seeds,
  merge-affected cells, mutation scan cells, and merge-safety counters shared across the primary
  and synthesized-backstop fixpoints.
- `run_reconciliation_rounds` receives one state reference instead of four independently pairable
  mutable accumulators, and consuming finalization replaces the local result-building closure.
- The empty-record release return and checked detection-completeness oracle remain before state
  construction. Primary/backstop order, options, allocations, raw cell-layout signatures, numeric
  policy, apply backends, and `ReconcileResult` are unchanged.
- Complete release, checked, no-default-feature, and all-feature Clippy gates passed, including the
  real-defect reconciliation net and in-place/full-rebuild differential.
- The artifact removed 544 text bytes and 3,552 BSS bytes, retained data size, reduced aggregate
  accounting by 4,096 bytes, and reduced file size by 616 bytes.
- Seven 500k single-threaded Fibonacci pairs were neutral: mean candidate/parent ratios were
  `1.000003220` instructions and `1.000004729` branches. Seven approximately 100k `cubed` pairs
  were also neutral at `1.000010264` and `1.000007825`; five approximately 500k `cubed`
  confirmation pairs measured `1.000003216` and `1.000004527`.
- Every counter sample recorded zero context switches and CPU migrations. Wall clock was ignored
  on the busy host.

## QUAL-001D reconciliation defect-body helper

The second reconciliation-orchestration extraction was accepted on 2026-07-20 against immediate
parent `971c378`.

- `reconcile_edge_mismatches` retains the empty-record return, checked detection-completeness
  oracle, and checked defect-layout audit. Only the nonempty-record program moved to the private
  `reconcile_recorded_mismatches` helper with the same seven explicit inputs.
- Primary/backstop ordering, options, allocations, mutation and error timing, raw layout
  signatures, numeric policy, apply backends, and `ReconcileResult` are unchanged. Complete
  release, checked, no-default-feature, and all-feature Clippy gates passed.
- LLVM inlined the source boundary. `reconcile_edge_mismatches` remained `0xc6a` bytes and
  `run_reconciliation_rounds` remained `0x2d10` bytes; no standalone helper symbol was emitted.
  Text grew by 8 bytes, data was unchanged, BSS shrank by 16 bytes, aggregate section accounting
  shrank by 8 bytes, and file size grew by 24 bytes.
- Seven 500k single-threaded Fibonacci pairs were neutral: mean candidate/parent ratios were
  `0.999998589` instructions and `0.999999242` branches. Seven approximately 100k `cubed` pairs
  were also neutral at `1.000010280` and `1.000020864`.
- Every counter sample recorded zero context switches and CPU migrations. The conditional 500k
  `cubed` confirmation was unnecessary because both the active counters and relevant symbol sizes
  were unambiguous. Wall clock was ignored on the busy host.

## QUAL-001D local-rebuild candidate transaction

The first local-rebuild extraction was accepted on 2026-07-20 against immediate parent `2026037`.

- Private `LocalRebuildCandidate` owns the minted positions, complete replacement cell/index
  arrays, and sorted override footprint after consuming `WorkingDiagram`. Candidate construction
  retains cycle-start canonicalization; consuming commit retains append, whole-diagram strict
  validation, diagnostics, truncate-on-rejection, and paired replacement.
- A0 capture, disabled/no-trigger returns, defect normalization, diagnostic environment reads,
  grid scratch, overlay growth, oracle selection, and zero-splice rejection remain in
  `maybe_rebuild_effective` with their original ordering.
- Direct tests pin accepted position/array/footprint installation and rejected exact-length
  rollback with the original cell/index allocations untouched. Complete release, checked,
  no-default-feature, and all-target/all-feature Clippy gates passed.
- LLVM emitted no standalone candidate or `maybe_rebuild_effective` body. `run_core_pipeline` grew
  by 384 bytes. GNU aggregate accounting added 1,308 text bytes and 2,784 bytes of alignment
  padding reported as BSS, data was unchanged, actual `.bss` stayed at 291 bytes, and file size
  grew by 992 bytes.
- Seven 500k single-threaded Fibonacci pairs were neutral at `0.999999187` instructions and
  `0.999999792` branches. Seven deterministic productive-rejection pairs were also neutral at
  `0.999991525` and `1.000000202`; every pair verified the two-round, seven-splice, full-candidate,
  strict-rejection fingerprint.
- Every counter sample recorded zero context switches and CPU migrations. No inline attribute or
  quiet wall-clock confirmation was justified.

## QUAL-001D assembly exact-zero hint confirmation

The first live-assembly extraction was accepted on 2026-07-20 against immediate parent `62b7851`.

- Private `ConfirmedZeroEdgeHints` owns the exact stored-zero candidate vector and the pre-scan
  hint-cell count. `confirm_exact_zero_edge_hints` owns only the final read-only gather, cell-cycle
  scan, normalized-pair insertion, sort, and dedup after sparse patching.
- The timer stays outside and around the helper. Mutable shard repair, global materialization,
  generator-/shard-order unsafe scatter, sparse overrides, construction hints, and output policy
  are unchanged. A direct duplicate-discovery regression and complete release, checked,
  no-default-feature, and all-target/all-feature Clippy gates passed.
- LLVM fully inlined the helper. `assemble_sharded_live_dedup` shrank from `0x2fee` to `0x2fbc`
  bytes; aggregate `.text` shrank by 48 bytes, data and actual `.bss` were unchanged, and file size
  shrank by 72 bytes.
- Seven interleaved single-thread pairs per gate were neutral. Candidate/parent instruction and
  branch means were `1.00000131` / `0.99999811` for default-bin Fibonacci, `0.99999901` /
  `0.99999471` for default-bin uniform seed 12345, `0.99999881` / `0.99999586` for 96-bin
  Fibonacci, `1.00000183` / `0.99999503` for 96-bin uniform seed 12345, and `1.00000211` /
  `1.00000062` for clustered seed 1.
- Every counter sample recorded zero context switches and CPU migrations. No forced inline
  attribute, cache attribution, or quiet wall-clock confirmation was justified.

## QUAL-001D packed directed range extraction

The first packed-preparation extraction was accepted on 2026-07-20 against source baseline
`eb56662` after one compact variant was rejected.

- `collect_directed_ranges` owned center-plus-neighbor ordering, same-bin classification, and the
  hard/aggregate work gates. A four-field summary returned center bounds and eligible/all-ring
  counts. Direct classification, packed brute-force, complete release, checked,
  no-default-feature, and all-target/all-feature Clippy gates passed.
- LLVM fully inlined the default helper and kept the main driver closure at `0x1199` bytes. The
  compact form shrank `.text` by 512 bytes and file size by 720 bytes, with unchanged data and
  actual `.bss`. It improved instructions on 500k Fibonacci and default/high-bin uniform, but added
  a repeatable 0.1397% on 100k clustered and 0.0127% on 100k mega. Branches improved in every
  regime. Forced inlining reproduced the same result.
- The retained form restores the original later center-range read. Seven-pair means are
  `1.000102288` / `0.999942805` instructions/branches on 500k Fibonacci, `1.000099301` /
  `0.999959540` on 500k uniform, `1.000091140` / `0.999953536` on 96-bin uniform,
  `1.000010707` / `0.999988071` on 100k clustered, and `1.000004870` / `0.999983614` on 100k mega.
  It adds 64 text bytes and 88 file bytes; data, actual `.bss`, and the `0x1199` driver closure are
  unchanged. All samples for both shapes recorded zero switches/migrations.
- The approximately one-basis-point ordinary instruction displacement is accepted as practical
  neutrality: branches fall slightly, dense/rebuilt-grid regimes are neutral, and the helper gives
  one named, independently tested classification and budget boundary. The retained rebuilt artifact
  is byte-identical to the measured source-shaped artifact.

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
