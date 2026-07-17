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
