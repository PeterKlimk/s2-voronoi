# Code quality and maintainability plan

**Status:** accepted direction; staged implementation not started

**Date:** 2026-07-17

This plan records the readability, maintainability, and code-quality pass requested after the
July 2026 optimization work. The objective is to move the implementation toward the Pareto
frontier: make future correctness and performance work easier to reason about without giving back
established throughput, memory behavior, or numerical guarantees.

This is not a general rewrite and it is not a request to make deliberately specialized code look
like ordinary application Rust. The SIMD kernels, SoA storage, direct scatters, explicit inlining,
and defect-local algorithms exist for measured reasons. Cleanup is successful only when it
clarifies ownership and invariants while preserving those reasons.

[`work-log.md`](work-log.md) remains the authoritative queue. This document owns the detailed
scope, ordering, and acceptance gates for `QUAL-001`.

## Compatibility posture

As of 2026-07-17, the crate has no external users. This cleanup should use that pre-adoption window
to establish the public API and vocabulary that future users should inherit, rather than preserve
misleading names through aliases or deprecation layers.

Repository tests, tools, examples, feature combinations, and any checked-in serialized fixtures
remain real consumers and must migrate coherently. A public rename or removal is not permission to
change geometry, error conditions, numerical policy, or performance behavior. If an internal
persisted-data dependency or unrecorded workspace consumer is discovered, record it and make an
explicit schema/compatibility decision before proceeding.

## Outcome sought

After this program:

- the public API, reports, diagnostics, and internal implementation use distinct names and explicit
  state transitions for construction, reconciliation, local rebuilding, validation, and output
  resolution;
- live cell-cycle access is mediated by one internal abstraction that makes stale backing-buffer
  tails impossible to consume accidentally;
- strict validation has one shared invariant vocabulary and shared classification primitives,
  while retaining allocation-free/fail-fast and detailed-report traversal strategies;
- large phase-oriented functions are split at their existing semantic boundaries without changing
  the generated hot loops;
- numerical tolerances and performance policy constants each have one authoritative registry;
- raw integer identities are typed at ambiguity-prone cold seams without widening hot storage;
- obsolete compatibility paths are removed rather than prolonged for hypothetical consumers, and
  test knobs and module descriptions are either removed or given a current owner; and
- active regression tests are separated from ignored research probes and process-global diagnostic
  controls are explicit.

## Non-goals

- No behavioral or algorithmic API redesign hidden inside a naming cleanup. Breaking renames and
  removals are allowed; changes to what the computation means remain separate decisions.
- No change to the supported geometric, welding, reconciliation, local-rebuild, or
  output-resolution contract.
- No tolerance-value tuning combined with a naming or relocation change.
- No conversion of working SoA data to AoS for aesthetic reasons.
- No removal of unsafe code whose measured replacement is slower. Unsafe boundaries should become
  smaller and easier to audit, not disappear at any cost.
- No generic framework for hypothetical geometry backends unless at least two current backends use
  and test it.
- No bundled cleanup commit spanning multiple hot phases. Attribution is part of correctness for
  performance work.

## Baseline assessment

The codebase is technically strong. Architectural rationale, numerical comments, adversarial
coverage, differential oracles, and local unsafe justifications are all substantially better than
typical performance-oriented code. Clippy across all targets/features, formatting, and the focused
release API/correctness/validation suites were clean at the start of this plan.

The principal risk is change amplification rather than an identified output defect. The
non-generated implementation is roughly forty thousand Rust lines. Several production functions
have grown into phase-oriented mini-programs, multiple validators encode overlapping versions of
the same contract, and repair-related state is represented through correlated raw arrays,
`Option`s, booleans, and `u32` pairs. The code generally explains each local decision, but the type
system and module boundaries do not yet carry enough of the global argument.

## Findings inventory

### F1 — Repair vocabulary describes multiple mechanisms

`repair` currently refers to two materially different operations:

1. post-assembly edge reconciliation, including identity merges and collinear drops; and
2. Local3d/LocalProjected neighborhood rebuilding followed by a whole-diagram acceptance gate.

The overlap appears in `RepairMode`, `RepairApply`, `run_repair_rounds`,
`post_repair_unpaired`, and escalation terminology. Some values named `post_repair` are produced
after reconciliation but before optional local rebuilding. This makes it unnecessarily difficult
to tell which topology a diagnostic describes.

The intended vocabulary is:

- **construction** — per-cell clipping and live dedup emission;
- **assembly** — global vertex/cell materialization;
- **reconciliation** — defect-local identity and cell-cycle cleanup;
- **local rebuild** — Local3d or LocalProjected replacement of a defect neighborhood;
- **acceptance gate** — whole-effective-diagram strict validation of a proposed rebuild; and
- **output resolution** — terminal exact stored-zero canonicalization/elision policy.

Public and internal names should converge on this glossary in one coordinated migration. Report
fields, error/diagnostic names, tests, examples, and documentation should not preserve ambiguous
historical terminology solely for compatibility. Do not add deprecated aliases unless a concrete
current repository consumer makes a short migration bridge worthwhile.

### F2 — Live cell storage has a load-bearing implicit invariant

Reconciliation can shrink `VoronoiCell::vertex_count` in place without compacting the shared
`cell_indices` buffer. Stale tail entries are legal storage but are not live topology. A consumer
that scans the raw buffer instead of each cell's live window observes phantom incidences; this has
already caused a large no-op acceptance-gate cost and is documented in `summarize_topology`.

The representation is valid and efficient, but passing bare `cells` and `cell_indices` arguments
through many helpers makes correct access a convention. One internal live-layout abstraction
should own that convention.

### F3 — Strict validity is implemented along overlapping paths

`validation.rs` currently contains:

- a fail-fast verifier for the optional ordinary-output verification gate;
- a parallel effective-array verifier used by the local-rebuild acceptance gate; and
- a detailed public report builder.

Their cost models legitimately differ, but their cell, incidence, edge-pairing, connectivity,
antipodal-arc, and Euler classifications overlap. Differential tests detect some divergence after
the fact; they do not prevent a future invariant change from requiring several synchronized edits.

The target is shared facts and classification primitives, not one forced traversal. The detailed
report must remain diagnostic-rich, and the fast gate must not allocate report maps or strings on
success.

### F4 — Several functions contain too many semantic phases

The clearest hotspots are:

- `PackedKnnCellScratch::prepare_group_directed` — range discovery, security thresholds, selection
  policy, dense-band handling, SIMD center scan, SIMD ring scan, and prepared-output construction;
- `assemble_sharded_live_dedup` — bookkeeping collection, overflow resolution, deferred patching,
  shard finalization, vertex materialization, cell prefixing/scatter, incidence reduction,
  overrides, exact-zero hints, and timing assembly;
- `reconcile_unresolved_edges` / `run_repair_rounds` — primary and backstop evidence, fixpoint
  rounds, merge safety, application, residual scans, and escalation seeding; and
- `WorkingDiagram` / the grow loop — oracle gathering, overlay mutation, winding reconciliation,
  local/global residual scans, and materialization.

Comments already label most phases. Extracting them can improve reviewability, but the packed and
assembly paths require codegen and benchmark gates because ordinary function boundaries or wider
state objects can alter inlining, alias analysis, and cache behavior.

### F5 — Correlated fields permit impossible states

Examples include:

- `effective_points: Option<_>` and `merge_result: Option<_>`, which describe one preprocessing
  choice but can theoretically disagree;
- `RepairOutcome { attempted, accepted, ... }`, where acceptance implies an attempt;
- `ResolutionDiscoveryDecision { certified_hint, drift_fallback }`, whose booleans are inverses;
  and
- multiple raw vectors in `PipelineState` whose meaning changes after a local rebuild is accepted.

Cold orchestration should use enums or phase-owned records so invalid combinations cannot be
constructed. This is not a request to replace compact hot-path flags where their representation is
measured.

### F6 — The tolerance registry is not fully authoritative

`tolerances.rs` correctly separates numerical slack from `policy.rs`, but production literals
remain outside both registries. Examples include repeated fallback `1e-24` squared-length tests,
fallback angular comparisons, weld wall-proximity guards, dense-band inflation, and coplanar
classification thresholds.

Some are tolerances, some are conditioning floors, and some are performance-policy margins. The
cleanup must classify them rather than moving every number mechanically. The first pass only names
and relocates existing values; any value change is a separate numerical task with the full
correctness campaign.

### F7 — Architectural fossils obscure current ownership

The audit found:

- internal compatibility re-exports left after `live_dedup`, timing, and cell-output types moved;
- an empty `TerminationConfig` that only forwards to `PackedNeighborPolicy`;
- a `VertexPosition` abstraction and `Vec2` implementation documented as serving a planar sibling,
  although no planar driver exists in this repository;
- module maps that omit current top-level modules; and
- internal public/re-exported diagnostic surfaces retained mainly for old paths or probes.

These may represent a future workspace direction, but future intent should not masquerade as a
current dependency. With no external users, removal is the default; retention requires an active
repository consumer, owner, and test.

### F8 — Raw integer identities remain ambiguous at cold seams

The implementation already has useful `BinId`, local-id, and edge-key wrappers, but reconciliation
and repair frequently pass `u32`, `[u32; 3]`, and `(u32, u32)` for generator ids, cell ids, vertex
ids, and cell pairs. Functions such as residual classification convert vertex endpoints into owner
pairs, making accidental role confusion easy during review.

Zero-cost `repr(transparent)` newtypes should be introduced gradually at orchestration and repair
boundaries. Packed arrays and SIMD-facing streams should remain raw where wrapping them harms
layout or codegen.

### F9 — Diagnostics and probes leak into production control flow and test organization

Runtime environment lookups are spread across compute, reconciliation, escalation, assembly,
validation, output resolution, timing, binning, and policy. Some are supported operational knobs;
others select differential or probe-only behavior. The latter make production flow harder to read
and complicate process-global test isolation.

Specific stale or mixed-purpose areas include:

- `tests/reclip_repair.rs` sets `VORONOI_MESH_RECLIP_REPAIR`, which has no reader in `src`; the test
  exercises current default local repair under retired Tier-2 terminology;
- `tests/escalate.rs` mixes a small active suite with a larger collection of ignored research
  probes and direct environment mutation;
- environment restoration is hand-written and is not panic-safe in several helpers; and
- `quality.rs` is a large doc-hidden diagnostic module compiled without a dedicated quality/tools
  gate, a choice that should be made explicitly rather than inherited.

The target is one diagnostic-options snapshot at appropriate cold entry points, feature-gated
probe code, panic-safe test guards, and a clear separation between CI regressions and manual
campaign tools. Supported user knobs must retain their documented behavior.

### F10 — Local comments mix durable invariants with historical measurements

Many comments correctly preserve why a non-obvious optimization exists. Some also embed exact
historical timings, old alternatives, or retired-path terminology. Those details are valuable but
age differently from the invariant itself.

Source comments should retain:

- safety preconditions;
- semantic invariants;
- the reason a superficially simpler form is invalid; and
- the benchmark gate that must be rerun.

Detailed experiment results and commit-era timings should live in `performance.md`,
`memory-layout-ideas.md`, or an appropriate decision record, with a short source link. Module maps
and diagnostic-knob documentation should be refreshed as part of the same hygiene pass.

## Workstreams

### QUAL-001A — Vocabulary and lifecycle model

**Risk:** low to medium

**Hot-path impact expected:** none

1. Add the stage glossary above to `docs/architecture.md` and inventory the public/internal names
   that map to each stage.
2. Rename public and internal functions, types, report fields, diagnostics, and applicable error
   variants that currently use generic or misleading `repair` wording.
3. Update repository consumers—tests, examples, tools, documentation, feature combinations, and
   checked-in serialized fixtures—in the same migration, without compatibility aliases by default.
4. Replace inverse/correlated cold state with enums, for example:
   - identity versus merged effective input;
   - certified-local versus global-fallback resolution discovery; and
   - not-needed, disabled, rejected, or accepted local-rebuild outcomes.
5. Separate defect facts (`low_incidence`, Euler failure, residual pairs) from the action outcome.

Acceptance requires equal geometric results and report contents after applying the declared field
mapping, plus no new branching or allocation in per-cell construction. Public names may break;
behavioral categories may not change as part of this workstream.

### QUAL-001B — Live cell-layout abstraction

**Risk:** medium

**Hot-path impact expected:** none on cell construction; possible cold-path/assembly impact

Introduce a small internal view/owner around cells and their backing index buffer. The abstraction
should provide:

- checked and unchecked-by-invariant live-span access in one place;
- iteration over live cell cycles and live half-edges;
- in-place cycle rewrite/shrink operations that preserve stale-tail semantics explicitly;
- semantic comparison independent of backing-buffer compaction;
- live incidence/topology summary helpers; and
- debug validation of span bounds and representation limits.

Migration order:

1. topology summary and reconciliation readers;
2. reconciliation mutation backends;
3. local rebuild overlay/materialization;
4. effective validation; and
5. assembly handoff.

Do not change `SphericalVoronoi` storage or force compaction. A successful version makes raw buffer
iteration difficult outside the defining module while producing identical final bytes.

### QUAL-001C — Validation fact engine

**Risk:** medium to high because it protects the return contract

**Hot-path impact expected:** none unless optional verification/reporting is requested

1. Define shared cell and edge issue classifications, including stable internal reason enums rather
   than repeated string literals.
2. Share cell-signature, edge-group, owner-conditioned antipodal, incidence, connectivity, and
   Euler primitives where their semantics are identical.
3. Retain three consumers:
   - fail-fast ordinary-diagram verification;
   - effective-array acceptance verification; and
   - full public diagnostic accumulation.
4. Make welded-twin handling an explicit input policy rather than an implicit fork in otherwise
   duplicated loops.
5. Expand the differential suite so every negative-control category is fed to every applicable
   consumer and compared by semantic reason, not only boolean verdict.

The fast verifier must remain success-path allocation-conscious. Consolidation is rejected if it
requires building `ValidationReport`, hash maps, diagnostic strings, or stored-position telemetry
for an ordinary success gate that did not previously need them.

### QUAL-001D — Phase extraction

**Risk:** varies; reconcile/escalate medium, assembly/packed high

**Hot-path impact expected:** must be proven neutral

Apply one phase extraction at a time:

1. reconciliation orchestration and round state;
2. local-rebuild oracle, overlay, residual scan, and commit preparation;
3. live assembly bookkeeping, vertex materialization, cell layout, sparse patching, and hint
   collection; and
4. packed preparation range setup, threshold selection, center scan, ring scan, and finalization.

Use narrow phase records or borrowed views instead of long argument lists. For SoA data, prefer a
zero-sized/narrow borrowed view with inline accessors rather than changing storage. Preserve
`inline(always)`/`inline(never)` decisions until measurements justify changing them.

Each numbered item is a separate benchmarked change. Packed center and ring loops should not be
abstracted behind dynamic dispatch, iterator trait objects, heap-owned phase results, or a generic
framework. If extraction changes codegen adversely, keep the executable loop flattened and extract
only setup, classification, and invariant-bearing state.

### QUAL-001E — Numerical and policy constant audit

**Risk:** low for naming, high for value changes

**Hot-path impact expected:** none

Classify every production floating-point literal outside tests/benchmarks as one of:

- exact mathematical value;
- numerical tolerance/conditioning floor (`tolerances.rs`);
- performance heuristic (`policy.rs`);
- diagnostic bucket boundary; or
- intentionally local test/probe value.

Name repeated values such as fallback intersection/dedup length floors once. Record units and
comparison direction (`<`, `<=`, squared versus unsquared, dot versus chord). Add constant
hierarchy tests where ordering is load-bearing. Do not alter a bit pattern in the same commit as
the audit relocation.

### QUAL-001F — Current architecture hygiene

**Risk:** low

**Hot-path impact expected:** none, verified by build/codegen checks where aliases disappear

1. Remove public and internal compatibility re-exports that have no current repository consumer,
   and update remaining call sites to current module ownership.
2. Replace or remove empty `TerminationConfig`; direct use of `PackedNeighborPolicy` is the current
   likely endpoint.
3. Decide the planar abstraction explicitly:
   - retain it only with a current consumer/test and a documented repository boundary; or
   - specialize the current crate to `Vec3` and reintroduce generality when a second backend lands.
4. Audit doc-hidden exports and internal `pub` visibility.
5. Refresh the AGENTS and architecture module maps.
6. Keep generated sorting networks and their boundary clearly identified.

### QUAL-001G — Typed identity boundaries

**Risk:** medium

**Hot-path impact expected:** none at first; later adoption must be measured

Start with reconciliation and local rebuilding, where roles are most ambiguous and work is cold.
Candidate types are `GeneratorId`, `CellId`, `VertexId`, `SlotId`, and `CellPair`. Reuse existing
`EdgeKey`, `BinId`, and local-id wrappers rather than creating parallel vocabularies.

Conversions should be explicit at storage boundaries, checked where data may be malformed, and
free in optimized code. Avoid a highly generic index trait or converting every loop variable. Move
into packed/assembly hot arrays only if assembly output and benchmark counters remain neutral.

### QUAL-001H — Diagnostics, probes, and test layout

**Risk:** low to medium

**Hot-path impact expected:** neutral or improved

1. Inventory environment knobs by category: supported operational, internal diagnostic,
   differential oracle, manual benchmark, or obsolete.
2. Read related cold knobs once into a diagnostic/options record and pass explicit values to the
   relevant stage.
3. Keep clean-path guarantees such as avoiding environment lookups when no defect record exists.
4. Rename `reclip_repair` coverage around the current local-rebuild contract and remove the unused
   environment setting.
5. Move ignored escalation/campaign probes out of mixed active suites into clearly named manual
   probe targets or tools.
6. Use panic-safe scoped environment guards and serialize mutations at the appropriate process
   boundary.
7. Decide whether `quality` remains an always-built doc-hidden API or becomes an internal feature;
   document and test the chosen surface.

### QUAL-001I — Durable documentation

**Risk:** low

**Hot-path impact expected:** none

1. Update `docs/architecture.md` with the stage glossary and current module ownership.
2. Move detailed historical timing claims out of source when a durable decision record already
   exists, leaving the invariant and a link.
3. Audit comments containing `old`, `legacy`, `current`, `only repair pass`, or retired feature
   names for accuracy.
4. Keep `AGENTS.md`, README feature descriptions, environment knobs, and test maps aligned with the
   actual tree.
5. Record rejected cleanup attempts with measurements so future passes do not repeat them.

## Execution order

### Milestone 0 — Pin the baseline

- Record the starting commit, supported feature matrix, binary sizes, and representative benchmark
  commands.
- Capture semantic fingerprints across thread/bin counts, default/scalar SIMD, and FMA using the
  existing WORK-002 campaign.
- Select defect-bearing reconciliation and accepted/rejected local-rebuild fixtures.
- Predeclare the performance equivalence/noise rule on the measurement host.

Captured 2026-07-17 in [`code-quality-baseline.md`](code-quality-baseline.md), including the exact
QUAL-001A public/internal rename map. On the busy shared WSL2 host, single-thread instructions and
branches are the primary sentinel; quiet wall-clock work is conditional on an unexplained,
repeatable adverse counter signal.

### Milestone 1 — Low-risk semantic cleanup

1. QUAL-001A coordinated public/internal vocabulary and lifecycle migration.
2. QUAL-001F compatibility-surface removal, empty wrappers, planar decision, and module maps.
3. QUAL-001H stale test knob and probe organization.
4. QUAL-001E constant inventory and name-only relocation.
5. QUAL-001I durable comment/documentation updates.

These changes should be small, independently reviewable commits except where a public/internal
rename must be atomic to keep the repository compiling. They establish the vocabulary used by
later structural work without carrying a deprecated parallel API.

### Milestone 2 — Make invariants structural

1. QUAL-001G typed ids at cold reconciliation/rebuild boundaries.
2. QUAL-001B live cell-layout abstraction and migration through cold consumers.
3. QUAL-001C shared validation facts and differential coverage.
4. Complete QUAL-001A state enums once phase ownership is clear.

The existing whole-diagram validation and differential oracles remain in place throughout this
milestone; they are removed or simplified only after the replacement has independent negative
controls.

### Milestone 3 — Split cold phase programs

1. Reconciliation orchestration.
2. Local-rebuild oracle/overlay/residual modules.
3. Report/output-resolution orchestration if the earlier state model exposes a useful seam.

Run defect-scale performance checks even though these paths are cold: cubed and defect-bearing
large inputs can make reconciliation scale with `n`.

### Milestone 4 — Split hot phase programs

1. Live assembly.
2. Packed group preparation.

These are last because the present flattened forms encode recent optimizations and compiler-shape
knowledge. Extract one phase, inspect optimized code where necessary, run the full performance
gate, and either retain or revert that extraction before attempting the next.

## Acceptance gates

### Per-change minimum

Every implementation change runs:

```bash
cargo fmt
cargo clippy --all-targets
RUSTFLAGS="-C target-cpu=native" cargo clippy --all-targets --all-features
cargo test --release
cargo test --profile checked
```

Feature-sensitive changes also check supported no-default, serde, and glam combinations. Pure
documentation commits need formatting/link review and relevant command validation rather than an
unrelated full campaign.

### Semantic gate

- Public names, module paths, report fields, diagnostic knobs, and serialized field names may
  break where QUAL-001 explicitly records the replacement. No deprecation or alias period is
  required without a concrete current consumer.
- Public behavior, geometric results, error conditions, and numerical policy remain unchanged by
  those migrations. Any semantic API change is a separate decision and commit.
- Before changing a serialized representation, search the repository for persisted fixtures and
  consumers and record the schema decision; absence of external users does not justify silently
  invalidating known internal data.
- Ordinary refactors retain exact output bytes where the current implementation is deterministic.
- Where bin count, FMA, or a deliberately degenerate policy permits representation movement,
  semantic-topology fingerprints and strict/intrinsic geometry verdicts remain unchanged.
- Reconciliation in-place and rebuild oracles continue to agree.
- Accepted local rebuilds pass the whole effective-diagram gate; rejected proposals restore the
  original arrays exactly.
- Output-resolution discovery/canonicalization reports and mutation footprints remain equivalent.
- WORK-002 negative controls remain active after every validation, reconciliation, or local-rebuild
  change.

### Performance gate

Performance-sensitive cleanup is compared against its immediate parent, not against a distant
release. Use optimized builds, interleaved runs, pinned thread/bin settings, and the existing
scripts. At minimum:

- one thread: Fibonacci and uniform;
- physical-core/default parallelism: Fibonacci and uniform;
- packed/query changes: clustered or bimodal plus mega/dense/great-circle as applicable;
- assembly changes: ordered Fibonacci and scrambled uniform, with default and high bin counts;
- reconciliation changes: cubed and the deterministic edge-repair fixture; and
- local-rebuild changes: an accepted splice, a rejected/no-op attempt, and a large defect-bearing
  case.

Measure wall time/cycles as outcomes and retired instructions, branches, cache behavior, peak RSS,
binary text size, and phase timings for attribution. A change fails the Pareto gate when an adverse
movement is repeatable outside the predeclared noise band. A readability benefit is not a license
for a known throughput or memory regression; instead adjust the seam, preserve flattening with
inlining, or keep only the cold-path portion of the abstraction.

Do not average away regime-specific losses. A hybrid is acceptable when it keeps the existing fast
representation for the affected regime and improves structure elsewhere without adding an
ordinary-path branch.

## Commit and review policy

- One logical invariant or phase extraction per commit.
- A repository-wide public/internal naming migration may be one atomic commit when splitting it
  would require temporary aliases or leave the tree uncompilable.
- No numerical value changes in structural commits.
- No simultaneous assembly and packed-query refactor.
- Include the exact validation and benchmark commands in each hot-path commit message or work-log
  entry.
- Keep a rejected-experiment note when compiler/codegen effects defeat a plausible cleanup.
- After each milestone, update this document and `work-log.md`; do not let completed checklist
  prose become a second active queue.

## Completion criteria

`QUAL-001` is complete when every finding F1–F10 is either resolved or explicitly retained with a
current rationale, owner, and test; all accepted changes pass their semantic and performance gates;
the future-facing public surface and the internal lifecycle vocabulary agree; the current
module/test/knob documentation agrees with the tree; and there is no known repeatable regression
against the pinned baseline.
