# Triage and Work Log

**Status:** active

**Last reorganized:** 2026-07-21

This is the authoritative list of unfinished correctness, robustness, and design work. Historical
investigations stay in [`audit-triage.md`](audit-triage.md); design rationale stays in the linked
policy documents. Larger uncommitted possibilities are collected separately in
[`algorithmic-performance-ideas.md`](algorithmic-performance-ideas.md) and
[`feature-api-wishlist.md`](feature-api-wishlist.md). An unchecked item elsewhere should either be
moved here or treated as stale.

## Current state

- The July 2026 correctness and safety audit is closed. AUD-001 through AUD-017 have no open
  correctness or policy finding under the documented production contract.
- The production promise is a construction-certified, edge-agreeing, Euler-valid spherical mesh
  or a defined error. It is not exact combinatorial equality with one ideal normalized-site model.
- Exact stored-zero edges are detected after final topology mutation and safely contracted when doing so does
  not remove an effective generator cell.
- Every current post-assembly mutator reports a complete changed-cell footprint for terminal
  exact-zero scanning; only globally uncertified representative drift forces global discovery.
- Full validation now counts canonical cells with fewer than three exact stored positions; generic
  `SphericalCellMesh` validation rejects them. This is intentionally validation/report-path work,
  not an ordinary construction pass.
- Qhull and the legacy P5 shadow path have been retired.
- Full strict validation remains available for testing and campaigns without burdening the default
  fast path.

There is no active P0/P1 correctness defect. Construction-certificate differential maintenance
remains ongoing. The weld-radius/whole-cell bound attempt is complete: it identified a metric
margin gap, added explicit validation telemetry, and retained the default weld as empirical
prevention rather than an overclaimed proof.

## Triage vocabulary

**Status**

- **Ready** — scoped enough to implement without another design decision.
- **Decision** — implementation should wait for explicit policy agreement.
- **Ongoing** — retained campaign or maintenance practice, not a finite feature task.
- **Backburner** — recorded idea with no current commitment.
- **Blocked** — depends on a concrete unresolved prerequisite.

**Priority**

- **P0** — memory safety or a successful ordinary computation returning materially corrupted data.
- **P1** — release-blocking safety or correctness contract work.
- **P2** — worthwhile robustness, policy, or capability work.
- **P3** — optional diagnostics, research, or workload-specific optimization.

## Queue summary

| ID | Priority | Status | Next gate |
|---|---:|---|---|
| WORK-002 | P2 | Ongoing | Exercise after construction/reconciliation/rebuild changes |
| POINT-001 | P2 | Completed | Checked `SpherePoint`, audited storage rule, and packed views landed |
| POINT-002 | P2 | Completed | Closure-based zero-intermediate ingest landed without direct-path regression |
| POINT-003 | P1 | Completed | Locator query validation and normalized ranking/certification landed |
| WELD-001 | P2 | Completed | Metric proof gap classified; validation telemetry retained |
| QUAL-001 | P2 | Completed | Reopen only for a new invariant/consumer or changed codegen evidence |
| RES-002 | P2 | Completed | Post-review affected-state and resource contracts landed |
| PERF-001 | P3 | Backburner | Obtain motivating workload and crossover data |
| PERF-002 | P3 | Completed | Reopen only for a materially different kernel mechanism or workload |
| PERF-003 | P3 | Completed | Reopen only for a separately designed multi-query resident kernel |
| RESEARCH-001 | P3 | Backburner | Expand the production combinatorics contract |
| RESEARCH-002 | P3 | Backburner | Justify diagnostic cost and conditioning policy |
| RESEARCH-003 | P3 | Backburner | First choose a compatible exact-zero/SoS model |
| RESEARCH-004 | P3 | Backburner | Commit to full f64 representation and search bounds |

## Completed work

### POINT-003 — Locator query contract

- **Priority:** P1
- **Status:** Completed 2026-07-17
- **Decision:** interpret every finite, nonzero unit-space query radially, normalize it with the
  same f64-to-f32 rule as checked points, and make invalid single and batch queries explicit.
- **Result:** `locate` returns `SphereQueryError`; `locate_many` returns the lowest indexed
  `IndexedSphereQueryError`. A deterministic regression proves that the old raw-dot versus
  normalized-bound mismatch could return a non-nearest generator for scaled input. World-space
  locators retain their existing projection errors and reuse the checked-point path. Public
  `locate_point` and `locate_many_points` methods provide that same infallible path for reusable
  canonical queries, avoiding the measured 2.4--5.1% raw-query normalization cost and the
  12-byte/query batch buffer. Raw batch records no longer need to be `Sync`.
  Details are recorded in
  [`point-api-plan.md`](point-api-plan.md#stage-3--design-locator-validation-as-a-correctness-change).

### POINT-002 — Closure-based point ingest

- **Priority:** P2
- **Status:** Completed 2026-07-17
- **Decision:** retain the simple `UnitVec3Like` direct API and add `_by` variants for adapting
  foreign records and math types without making `SphericalVoronoi` generic or requiring callers to
  allocate an intermediate coordinate vector.
- **Result:** `compute_by`, `compute_with_by`, and `compute_with_report_by` share one backend
  collector with the direct path. The extractor runs once per accepted point, too-short inputs fail
  before extraction, validation preserves original indices, and neither records nor extractors
  need to be `Sync` while adaptation remains serial. Exact-output API tests, release
  checks, Linux counters/Cachegrind, and native Mac interleaved timing passed. Detailed performance
  evidence is recorded in [`point-api-plan.md`](point-api-plan.md#stage-2--add-closure-based-ingest-independently).

### POINT-001 — Checked spherical point boundary

- **Priority:** P2
- **Status:** Completed 2026-07-17
- **Decision:** implement private-field packed `SpherePoint` with a finite
  `2 * f32::EPSILON` squared-norm envelope. Raw arrays and closure ingest remain the
  interoperability boundary; glam remains internal arithmetic.
- **Numerical rule:** replace the existing ordinary/fallback f32 normalization at its current
  producer seam with promoted-f64-normalize-then-round. Generators, centroids, embedding
  projection, and Hull3d minting already use that order.
- **Evidence:** the producer audit, candidate topology/fidelity campaign, full release suite,
  Cachegrind/Linux counters, and native Mac timing are recorded in
  [`point-api-plan.md`](point-api-plan.md#stage-0-findings-and-decision).
- **Implementation scope:** migrate public producers/consumers and checked serde together; add the
  audited final allocation ownership transfer and packed xyz views; preserve safe raw import APIs.
  Locator query semantics remain a separate correctness/API stage.
- **Result:** all stored/public directions now use the finite `2 * f32::EPSILON` envelope and
  checked private-field type. Exact packed views and backend allocation ownership reuse are tested;
  checked serde rejects invalid stored values. Full default and `serde,glam` release suites,
  no-default-feature checks, doctests, and focused Valgrind Memcheck passed. Clippy completed with
  the repository's existing warnings. The Mac and Linux performance evidence is recorded in
  [`point-api-plan.md`](point-api-plan.md#stage-1--apply-the-representation-decision-and-remove-the-final-point-conversion).

## Ongoing work

### WELD-001 — Sufficient-welding whole-cell geometry bound

- **Priority:** P2
- **Status:** Completed 2026-07-15
- **Goal:** determine whether the default weld separation provably prevents an effective generator
  cell from collapsing at f32 output resolution.
- **Starting theorem:** minimum geodesic site separation `alpha` gives every exact Voronoi cell a
  contained spherical cap of radius `alpha / 2`.
- **Required composition:** bound input canonicalization/normalization, f64 clipping, gnomonic and
  spherical fallback conversion, reconciliation/local-rebuild displacement, and final f32 storage against
  that cap floor. Separate whole-cell survival from cocircular non-cell-killing zero edges, which
  welding cannot prevent.
- **Finding:** the default recomputed squared weld threshold is
  `1.8189892951256392e-12` (`0x2bffffff`, one f32 ULP below `2^-39`). A conservative raw canonical
  chord floor is `1.3486989111824338e-6`; accounting for canonical norm slack gives a normalized-site
  chord floor of `1.1102803320808713e-6` and ideal inradius `5.551401660404641e-7`. The more
  conservative unequal-norm gnomonic/chord-cost model yields about `4.3593087648967823e-7`.
- **Why the pure metric proof does not close:** reconciliation's `1e-6` component diameter is
  larger than those inradius margins. It cannot simply be charged as arbitrary coordinate
  displacement—reconciliation changes IDs, retains positions, and transactionally rejects fewer
  than three IDs—but the current structural gates do not prove three final coordinate classes.
  Hull3d likewise passes whole-diagram strict topology validation without a coordinate-separation
  or Hausdorff certificate. A naïve displacement composition would require a weld chord above
  roughly `2.0e-6` nominal, `2.2384e-6` with normalization slack, or `2.4768e-6` under the
  conservative off-shell model, before other construction/storage errors.
- **Implemented classification:** full validation counts cells with fewer than three exact stored
  coordinate classes, including an alternating two-position cycle with no adjacent zero edge.
  Generic `SphericalCellMesh` validation rejects that geometry. `Preserve` still treats it as
  representation telemetry rather than a Voronoi-topology defect.
- **Evidence:** active threshold-adjacent tests cover axis, cube-face seam, cube corner, f32
  half/quarter exponent-boundary orientations, two rotations, and near-cocircular stress. The
  manually run extended campaign covered 256 rotated threshold-adjacent cases. Direct final
  coordinate scans found no collapse and agreed with full validation. Forced spherical handoff,
  reconciliation/local-rebuild/fallback output, and welding-disabled positive controls are pinned separately.
- **Rejected fast-path certificate:** a construction-local x-separation certificate plus sparse
  terminal scans was implemented and measured. Ten-round interleaved 500k counters added about
  0.75–0.80% retired instructions and 0.41–0.45% branches. It changed no current policy outcome:
  `Preserve` takes no action, Error/Elide operate on exact-zero components, report-bearing compute
  already runs full validation, and plain compute discarded the count. The always-on certificate
  was therefore removed; a future broader Error/Elide policy may pay for classification only when
  selected. After removal, eight-round comparisons against `11b49a4` returned instruction ratios
  `1.000000` for both Fibonacci and uniform, with branch ratios `1.000001` and `1.000003`.
- **Conclusion:** do not promote the current weld radius to a standalone mathematical guarantee
  that output geometry cannot collapse. The production contract instead combines strong weld
  evidence, explicit validation telemetry, and Preserve/Error/Elide policy surfaces for observed
  exact-zero components.

### WORK-002 — Construction-certificate differential maintenance

- **Priority:** P2
- **Status:** Ongoing
- **Scope:**
  - keep the exhaustive uniform small-`n` geometry campaign active;
  - after reconciliation or Hull3d edits, check owner equality and edge-bisector residuals as well
    as topology;
  - retain negative controls for reversed faces, duplicate faces/references, moved vertices, and
    disconnected unions of closed complexes;
  - compare semantic topology across thread counts, bin counts, default/scalar SIMD, and FMA; and
  - keep welding and deterministic perturbation in separate expected-policy buckets.
- **Acceptance:** every supported successful result passes the relevant strict and intrinsic
  geometry checks; Hull3d rebuilding remains a valid success rather than a failure count.
- **Latest maintenance (2026-07-15):**
  - the extended uniform small-`n` campaign passed 62,464 intrinsic assessments; worst ownership,
    vertex cross-track, and edge cross-track errors were respectively `1.507e-7`, `7.918e-8`, and
    `1.833e-16` radians;
  - clustered 1M seed 1 remained strict-valid across 1/6 threads and 6/96 bins, with the same three
    assembly defects, four-cell reconciliation footprint, 18 hinted versus 17 terminal zero
    edges, zero ownership mismatches in 733 samples, and `5.875e-8 rad` maximum sampled edge
    cross-track error;
  - the edge-reconciliation suite passed all five active tests, including exact output agreement between
    the production in-place reconciler and full-rebuild oracle; and
  - a vertex-id-independent semantic-topology fingerprint agreed across 1/6 threads, 6/96 bins,
    default SIMD, scalar SIMD, and hardware FMA. Default/scalar representations were byte-identical
    at matching bin counts; bin-count and FMA representation changes left topology unchanged.

## Decision-gated output policy

These tasks are optional extensions. The Hex3 exact-zero incident is already resolved under the
default `Preserve` behavior.

### RES-001 — Public cell-killing outcomes

- **Priority:** P2
- **Status:** Completed 2026-07-15
- **Goal:** expose `Error` and `Elide` behavior when satisfying output resolution would remove an
  effective generator cell. `Error` and explicit cell-mesh `Elide` are implemented.
- **Implemented:** `CellKillingPolicy::{Preserve, Error}` applies equally to plain, report-bearing,
  and embedded computations. `CellEliminationRequired` reports original input indices, expanding
  affected preprocessing weld classes, after all safe exact-zero contractions.
- **Elision transaction:** the implemented direction is an explicit postprocess returning a distinct
  spherical cell mesh with `input -> Option<cell>` and `cell -> canonical input` mappings. A
  test-only global transaction elides the two cells in the 18-site fixture, then necessarily
  suppresses two degree-two boundary vertices under opposite owner-rotation checks. The final
  16-face quotient has no zero edges, single-cycle vertex links, complete adjacency, and strict
  validation; its maximum forced-merge cross-track deviation is `1.861e-8 rad`. The welded fixture
  maps original inputs `[1, 10, 18]` to `None`. Pinched links, disagreeing rotations, and whole-mesh
  collapse are rejected.
- **Public surface:** a consuming `ComputeOutput::into_elided_cell_mesh` conversion owns
  the cold global transaction and returns a distinct `SphericalCellMesh`, compact provenance
  mappings, the original `ComputeReport`, and an elision report. The first surface exposes geometry,
  ordered cell cycles, source attribution, combinatorial adjacency, compaction, and generic S2-mesh
  validation. Locator, Delaunay, Lloyd, and conditioned area/centroid APIs remain Voronoi-only until
  separate semantics are justified. Unsafe conversion is all-or-error through a distinct
  `CellElisionError`; there is no implicit Preserve fallback. Embedded parity is a thin unit-mesh
  wrapper without locator/Lloyd claims. Rejected conversion retains the original successful output,
  and mesh vertex storage is dense and immutable.
- **Deferred extension:** decide the generic spherical-arc contract before adding cell-mesh
  area/centroid methods.
- **Invariant:** `Preserve` remains the default and never silently removes an effective generator.
- **Regression foundation:** an end-to-end 18-site fixture disables preprocessing welding while
  retaining distinct f32 generators. It returns a strict-valid mesh with three preserved
  cell-killing exact-zero components, including a triangle whose zero edge cannot be contracted
  without deleting its generator cell. The fixture is stable across default SIMD, scalar SIMD,
  and hardware FMA and is shared groundwork for `Preserve`, `Error`, and `Elide` tests.
- **Reference:** [`output-resolution-policy.md`](output-resolution-policy.md).

### RES-002 — Optional positive-threshold edge simplification

- **Priority:** P2
- **Status:** Completed 2026-07-23 after post-implementation review fixes
- **Dependencies:** RES-001 (completed)
- **Goal:** let graphical or physics consumers explicitly remove represented nonzero slivers.
- **Landed implementation:** explicit consuming unit and embedded conversions; inclusive stored-chord
  discovery; all-pairs diameter-bounded minimum-id components; exact-first deterministic fixed
  point; transactional complete-mesh topology certification; `Preserve`, first-failure `Error`,
  and provenance-preserving `Elide`; checked work budgets; and distinct success/failure reports
  which retain the untouched source on error.
- **Post-review fixes:** pending-suppression cause upgrades now follow the transaction's affected
  cover; exact transactions certify stored-position cardinality and induced exact edges on affected
  faces; current provenance checks visit only affected owners and stream bag members under the work
  budget; charged vertex-id validation precedes adjacency lookahead; affected endpoint stars retain
  current positive-owner recertification; and mixed exact/positive Elide, unchanged-edge/sink cause
  upgrades, and a positive cell-killing triangular-prism suppression pin the corrected behavior.
- **Pinned policy:** default work limits are 100,000,000 diameter comparisons, 100,000,000
  cell-index visits, and 100,000,000 provenance checks. Positive suppression uses `1e-24`
  squared endpoint-cross/projection floors and `64 * f64::EPSILON` arc-membership tolerance.
- **Performance acceptance:** ordinary 500k Fibonacci single-threaded construction is gated at no
  more than +0.5% retired instructions and +1.0% branches versus revision `ee8e73b`; seven
  interleaved `perf stat` rounds measured +0.127% and +0.366%, respectively. The dedicated
  `bench_simplify` probe (10k Fibonacci, single-threaded) measured 42.9 ms candidate-free
  (`239,952` cell-index visits), 297 ms with 36 candidate occurrences / 8 commits, and 2.28 s with
  3,403 occurrences / 82 commits. A 100k candidate-heavy probe stopped recoverably at the default
  100,000,000-visit ceiling after 10.86 s rather than running unbounded. These are cold conversion
  characterizations, not ordinary-path budgets. Shared exact-elision focused and full release
  suites remain unchanged and passing.
- **Contract:** the result is a valid spherical cell complex after explicit simplification, not the
  exact Voronoi diagram of the original generators.
- **References:** [`output-resolution-policy.md`](output-resolution-policy.md) and
  [`positive-edge-simplification-plan.md`](positive-edge-simplification-plan.md).

## Performance robustness

### PERF-001 — Total-query-work circuit breaker

- **Priority:** P3
- **Status:** Backburner; scale-relative work telemetry implemented 2026-07-16
- **Motivation:** Perturbed great-circle inputs can become gnomonically bounded yet process nearly
  every generator for some cells. The existing exhaustion replay is correct but does not detect
  this successful high-work regime.
- **Candidate direction:** a progress-aware total-work budget followed by unrestricted spherical,
  Hull3d, or global-hull rebuilding.
- **Measurement available:** timing builds report total candidate-work and no-geometric-progress
  tail quantiles plus counts at 4x/16x/64x each run's median. Batched exhaustion-recovery cells are
  reported as exclusions from the latter. `bench_voronoi --dist great-circle` provides a directly
  successful high-work case; `mega` distinguishes a small extreme tail from broad scale-dependent
  work.
- **Before implementation:** measure the actual cold-replay crossover, avoid a fixed candidate
  count such as 128, and prove that the handoff cannot turn a valid success into a failure.
- **Reference:** AUD-015 in [`audit-triage.md`](audit-triage.md).

### PERF-002 — Post-review kernel hypotheses

- **Priority:** P3
- **Status:** Completed 2026-07-21; post-review shortlist exhausted
- **Closed experiments:** fixed smaller packed prefixes, heap-maintained compact overflow with
  exact rebuild, order-preserving shell-cell clip bypass, center-only high-threshold prediction,
  the final one-vector high-threshold refinement, seed-first micro-batched preparation, and a
  same-grid-cell regional local hull were rejected. The first three were decided by
  retired-instruction and branch counters; the latter four failed timing-only workload gates. The
  temporary rejected probes were removed; earlier generally useful census/oracle code remains.
  Full branch and counter history is in
  [`kernel-optimization-experiment-log.md`](kernel-optimization-experiment-log.md#pass-closeout).
- **Threshold result:** exact center overshoot plus one whole-ring sample removed the ordinary and
  negative density-contrast false positives, but its apparent 100k clustered margin collapsed at
  500k and on splittable input. Reopen only if the distribution statistic becomes free or changing
  the threshold no longer requires center selection and lazy-tail reconstruction.
- **Regional result:** exact same-cell overlap did not repay even the optimistic pair-count floor
  of the current naive local hull in any measured group. Reopen regional construction only with a
  subquadratic algorithm, reusable triangulation, or a materially tighter certified candidate set;
  duplicated-attempt counts alone do not justify guard/replay/stitching work.

### PERF-003 — Group-shared shell traversal

- **Priority:** P3
- **Status:** Completed 2026-07-21; sequential shared schedule implemented
- **Evidence:** same-cell shell queries repeat 93.5--99.1% of group-local grid-cell visits on 100k
  uniform, clustered, mega, and great-circle inputs and 500k clustered. At least 95% of positive
  resident/query work occurs at active width 16 or greater. Fibonacci performs no shell work. The
  temporary oracle was removed; full counters and caveats are in the
  [`kernel optimization experiment log`](kernel-optimization-experiment-log.md#group-shared-shell-traversal-census).
- **Result:** `CubeMapGridScratch` retains the discovered BFS cell order and layer offsets while
  consecutive queries share a start cell. Queries still compute their own cap bounds, resident
  dots, key ordering, clipping, and edge forwarding; later queries extend the schedule only when
  necessary. Focused tests prove reused batches and bound bits match fresh traversal.
- **Counters:** three native single-threaded pairs reduced instructions/branches by 0.043%/0.129%
  on Fibonacci, 0.066%/0.172% on uniform, 0.392%/0.893% on clustered, 3.048%/5.271% on mega,
  0.655%/1.921% on great-circle, and 0.455%/1.014% on 500k clustered. All structural pairs agreed;
  cycles were noisy except for a consistent 3.06% mega improvement.
- **Memory:** the retained union added about 1.2 MiB peak RSS on single-threaded 100k mega and
  5.6 MiB with default threading; great-circle and 500k clustered changes were below 0.3 MiB.
  This is accepted rather than cap away the strongest reuse regime.
- **All-core guardrail:** on 12 cores, instructions and branches remained favorable throughout the
  matrix, including -2.148%/-4.201% at 500k mega. That case added 26.2 MiB peak RSS and 3.18% cache
  references with neutral cache misses. Aggregate cycles improved 2.67%, but pair noise was large;
  a quiet repeat may refine throughput evidence but is not required to retain the optimization.
- **Higher-ceiling follow-up:** a resident-by-query tiled dot/key kernel is separate work. Do not
  infer its saving from the optimistic position-load ratio. First specify the block dependency
  model and account for lost same-block forwarded seeds, speculative rows, query termination
  masks, sorting/storage traffic, and great-circle trace divergence.

The remaining code-specific performance experiments are maintained separately in the open queue in
[`performance.md`](performance.md) and the memory backlog in
[`memory-layout-ideas.md`](memory-layout-ideas.md). Larger research hypotheses live in
[`algorithmic-performance-ideas.md`](algorithmic-performance-ideas.md). They are not correctness
tasks and are not duplicated here.

## Code quality and maintainability

### QUAL-001 — Performance-preserving cleanup

- **Priority:** P2
- **Status:** Completed 2026-07-20 at measured selective boundaries
- **Goal:** reduce change amplification and make pipeline invariants structural without giving back
  established throughput, memory behavior, or numerical/correctness guarantees.
- **Compatibility posture:** there are no external users as of 2026-07-17. Use this window for
  coordinated breaking public/internal renames and removal of obsolete shims; do not preserve
  misleading surfaces for hypothetical consumers. Repository consumers and known serialized data
  still require an explicit migration.
- **First milestone:** pin the semantic/performance baseline, establish distinct construction /
  assembly / reconciliation / local-rebuild / acceptance / output-resolution vocabulary across
  the public API and implementation, then remove stale compatibility aliases, obsolete test
  terminology, and inaccurate module maps.
- **Baseline:** counter-oriented Milestone 0 evidence and the exact atomic lifecycle rename map are
  pinned in [`code-quality-baseline.md`](code-quality-baseline.md). The shared host's wall clock is
  advisory; single-thread retired instructions/branches are the primary first-change sentinel.
- **Post-closeout layout audit:** the formerly rejected semantic old/new cell-span comparison now
  takes two coherent `LiveCellLayout` values. Its ordinary benchmark path is inactive, a
  one-codegen-unit control produced byte-identical executable code and neutral counters, and the
  default multi-regime matrices showed no cycle regression. The repeated historical clean-path
  fingerprint is now classified as codegen-partition/layout movement; other candidates sharing it
  remain deferred until their value and current outcome are tested independently.
- **QUAL-001A result:** public, internal, feature, environment, CLI, report, test, and current-doc
  terminology now distinguishes assembly mismatches, reconciliation, residual output facts, and
  local rebuilding. The migration was breaking and alias-free; the unread reclip knob was removed.
  Semantic fingerprints matched exactly, single-thread retired work was effectively identical,
  multi-thread retired work remained within its declared noise band, and code size/RSS gates
  passed. Detailed measurements are in
  [`code-quality-baseline.md`](code-quality-baseline.md#qual-001a-validation-result).
- **QUAL-001F result:** unused compatibility re-exports and the empty `TerminationConfig` were
  removed. With no second backend in the repository, live dedup and reconciliation now own
  spherical `Vec3` positions directly. The compiler visibility audit restricted 216 unreachable
  internal spellings across default, all-feature, and test builds, including generator-owned
  sorting-network exports, to crate scope. Feature-gated diagnostics and experimental report
  fields remain because repository probes/tools and defect fixtures consume them; their long-term
  organization belongs to QUAL-001H. Module maps now describe the actual tree.
- **QUAL-001H environment result:** every environment variable now has a recorded category, reader
  boundary, and current writer in [`environment-knobs.md`](environment-knobs.md). Active
  integration-test mutation uses one exact-restore, panic-safe scoped guard; the verification unit
  test uses isolated child processes. Ordinary parallel feature matrices passed, and the optimized
  production binary was byte-identical to its parent. The stale planar grid-density name was
  removed from current documentation.
- **QUAL-001H local-rebuild probe result:** the all-ignored target is now explicitly feature-gated
  in Cargo and retains its 14 named workloads. A0 capture uses one nested, thread-local, panic-safe
  scope. The redundant process-global forced-rebuild switch and A0 environment reader were removed.
- **QUAL-001H campaign result:** wholly ignored coincidence and robustness targets require the
  internal `manual_probes` feature; the fidelity campaign explicitly requires `tools`. Mixed suites
  retain isolated ignored cases beside their shared active fixtures. The planned `quality` decision
  was already implemented: its doc-hidden surface has been `tools`-gated since `c620fe4`.
- **QUAL-001H local-rebuild options result:** debug output and the feature-only global-Delaunay
  selector are snapshotted once per actual rebuild attempt and passed explicitly through the grow
  loop. Disabled and no-trigger computations return before either lookup.
- **QUAL-001H reconciliation options result:** telemetry, apply-backend selection, and the global
  duplicate-scan fallback are snapshotted together only after mismatch records exist, then passed
  explicitly through telemetry and reconciliation rounds. Clean computations perform no
  reconciliation environment lookup.
- **QUAL-001H singleton result:** the stale internal `VORONOI_MESH_UNPAIRED_ORIGINS` name became
  `VORONOI_MESH_EDGE_MISMATCH_ORIGINS`, and its lookup/output now occurs only when assembly has a
  mismatch. Output-resolution telemetry was retained unchanged because it already returns before
  its lookup when no exact-zero edge exists. This closes QUAL-001H.
- **QUAL-001E policy result:** the raw dense-band `1e-3` gather inflation is now the named,
  dimensionless `f32` policy `DENSE_BAND_RADIUS_INFLATION` in `policy.rs`. Its optimized benchmark
  artifact is byte-identical to the parent, and all required feature/test matrices pass.
- **QUAL-001E fallback result:** seven raw `1e-24` comparisons are now separately owned as the f64
  squared-cross conditioning floor `FALLBACK_INTERSECTION_CROSS_LEN2_FLOOR` and squared-chord
  identity threshold `FALLBACK_VERTEX_DEDUP_LEN2`. Both exact values and all `<=` comparisons are
  unchanged. Optimized executable sections are byte-identical; only source-line metadata moved.
- **QUAL-001E `1e-12` result:** four raw sites are now three independent tolerances for fallback
  edge-arc radians, gnomonic metric-relative inflation, and the local-rebuild stereographic `f32`
  denominator floor. Their equal value does not imply a hierarchy or shared semantics. Optimized
  executable sections remain byte-identical; only source-line metadata moved.
- **QUAL-001E owner-arc result:** owner-plane residual and exact-pi sine thresholds now live in
  `tolerances.rs` with their `f64` units and distinct `>` / `<=` rejection boundaries documented.
  Their stripped optimized binaries are byte-identical to the parent.
- **QUAL-001E weld-wall result:** the cube-grid weld's additive `f32` wall pad and preprocessing
  weld's relative `f64` wall inflation are now separately named in `tolerances.rs`. Their values,
  expressions, strict comparison boundaries, and final computed-f32 weld predicate are unchanged.
  Optimized executable sections and numeric data are byte-identical; only source metadata moved.
- **QUAL-001E coplanar-policy result:** the near-great-circle maximum and RMS plane-residual bounds
  now live as independent `f64` tolerances, while the realized perturbation amplitude is named
  separately as robust-mode policy. Values, strict `>` rejection boundaries, arithmetic, and the
  local closure capture are unchanged. Optimized executable sections and numeric data are
  byte-identical; only source metadata moved.
- **QUAL-001E projected-Delaunay result:** the local rebuild's `f64` minimum chart-span guard and
  super-triangle expansion factor now live separately as conditioning tolerance and construction
  policy. Values, `max`/multiplication expressions, synthetic geometry, and robust predicate inputs
  were unchanged in that experiment. The projected diagnostic oracle and these constants were
  subsequently removed in the final pre-API-lock cleanup; Hull3d is now the sole rebuild model.
- **QUAL-001E centroid result:** the per-edge cross-length skip and final accumulated-integral
  fallback now use independently named `f64::EPSILON` floors. Both `<=` boundaries and their
  respective skip/generator outcomes are unchanged. Stripped optimized binaries are identical.
- **QUAL-001E point-envelope result:** profiling-only epsilon/absolute-error bands are named local
  diagnostic boundaries, and ambiguous `1e6` field/output vocabulary now spells negative exponents
  explicitly. Thresholds, strict `>` comparisons, counts, maxima, and semantic hashes are
  unchanged. The non-profiling stripped artifact is identical.
- **QUAL-001E gnomonic-initialization result:** the south-pole tangent-basis branch and duplicated
  initial chart bound now have separate construction-policy names; the neighbor norm check is a
  named local debug diagnostic. Values, branch equality, initial geometry, and assertion direction
  are unchanged. The complete optimized artifact is byte-identical.
- **QUAL-001E reference-axis result:** production helper-axis selection now uses separately typed
  `f32` and `f64` policy constants. Every exact `0.9` value, strict `<` comparison, X/Y choice, and
  operand type is unchanged; tool helpers and the feature-only A/B probe remain local. Optimized
  executable sections and numeric data are byte-identical; only source metadata moved.
- **QUAL-001E final grid-policy result:** the locator's exact `16.0` target density is now a central
  production policy distinct from the kNN construction grid, while the low-degree diagnostic's
  exact `1e-4` hash-grid cell size is named locally and remains separate from its duplicate
  threshold. Formulas and types are unchanged; optimized executable sections and numeric data are
  byte-identical, with only source metadata movement.
- **Final-inventory result:** exact mathematical coefficients remain inline; tolerance and policy
  registries own production boundaries; and quality, reconciliation, point-envelope, feature-probe,
  and tool values remain named or deliberately local diagnostics. No unclassified production
  floating-point policy remains, closing QUAL-001E.
- **QUAL-001I architecture result:** `architecture.md` now defines the execution-ordered stage
  vocabulary and primary owners from input adaptation through original-index remapping. It keeps
  validation/derived views outside hidden repair terminology, distinguishes the `tools` and
  `profiling` diagnostic owners, and corrects stale `cube_grid` and `live_dedup` module headers.
  Optimized executable sections and numeric data are byte-identical to the parent; only source
  metadata moved.
- **QUAL-001I source-comment result:** production comments now state local invariants and link to a
  consolidated performance-decision record instead of carrying host-specific timing anecdotes.
  Comparative wording names the actual superseded mechanism; remaining runtime-state uses,
  correctness tolerance evidence, and intentional fixture/probe names were retained. Optimized
  executable sections and numeric data are byte-identical; only source metadata moved.
- **QUAL-001I current-guidance result:** the root agent guide now identifies its actual scope and
  matches the live module, feature, and test-target sets. README distinguishes the three supported
  Cargo features from internal hooks. The environment inventory now states its compiled-code scope,
  records both private test-only variables, and distinguishes the verification fast path from its
  strict fallback. Existing rejected-experiment records close the final durable-documentation item.
- **QUAL-001G first identity result:** a transparent `CellId` now protects the local-rebuild
  overlay's splice mutation from accidental vertex/generator id substitution while raw storage and
  neighboring algorithms remain unchanged. A broader typed rebuild-seed owner was rejected after
  seven counter pairs showed +0.1602% instructions and +1.6619% branches on the clean path. The
  splice-local form was structurally neutral in the same matrix and reduced total size accounting
  by eight bytes.
- **QUAL-001G vertex-lookup result:** a transparent `VertexId` now guards the local overlay's
  position and key accessors. Raw collection/storage element types are unchanged, release section
  sizes are identical, and seven counter pairs were neutral (mean -0.000058% instructions and
  -0.000133% branches).
- **QUAL-001G owner/creation result:** `vid_for` now returns `VertexId` and `owners` consumes it,
  keeping the identity typed through creation, key/position access, and owner lookup. Raw `u32`
  remains at collection and stored-boundary edges. The release artifact added 32 `.text` bytes and
  four unwind bytes, offset by 32 fewer padding bytes; seven counter pairs were neutral (mean
  +0.000307% instructions and -0.000152% branches).
- **QUAL-001G boundary decision:** the local overlay's `VertexId` seam is complete. The remaining
  candidates either stay raw throughout traversal/storage or recreate the rejected cross-phase
  pair owner, so mechanical wrapper expansion stops here.
- **QUAL-001B first reader result:** `LiveCellLayout` now owns live-span lookup for topology and
  reconciliation readers, with typed invalid-cell/invalid-span failures and unit coverage for
  stale tails. Seven counter pairs improved instructions by 0.02623% with neutral branches
  (+0.000040%); the release artifact grew 12 bytes overall. A `slice.get(range)` accessor was
  rejected after repeatable +0.1337% instructions and +1.6620% branches.
- **QUAL-001B segment-reader result:** the shared-edge segment family now accepts one
  `LiveCellLayout` through primary reconciliation, rejected-component seeding, optional telemetry,
  and tests. Deliberate between-round mutation rebuilds the view at that boundary. Seven counter
  pairs were neutral (mean -0.000097% instructions and -0.000004% branches), and the release
  executable file is 48 bytes smaller.
- **QUAL-001B semantic-comparison result:** converting the old/new live-span comparison from four
  slices to two layouts was initially reverted. Default, never-inline, and always-inline forms all
  produced the same repeatable clean-path counter displacement (about +0.1597% instructions and
  +1.6620% branches), despite an eight-byte-smaller executable. The 2026-07-20 retest established
  that this diagnostic-rebuild function is inactive in the ordinary benchmark and that a
  one-codegen-unit control makes candidate/parent executable code byte-identical. With no measured
  cycle regression across four default-build regimes, the typed signature is now retained and the
  historical signal is classified as codegen-partition/layout movement.
- **QUAL-001B duplicate-reader result:** the defect-only localized duplicate-key BFS now consumes
  the same `LiveCellLayout` that merge collection passes to segment readers. Its localized/global
  oracle remains unchanged. Seven counter pairs were neutral (mean +0.000153% instructions and
  +0.000209% branches), aggregate section sizes were identical, and the executable is 32 bytes
  smaller.
- **QUAL-001B unpaired-reader decision:** migrating the localized unpaired-edge scan, partner lookup,
  and debug global oracle was initially reverted. The whole-family form produced +0.1598%
  instructions and +1.6619% branches; retaining the raw outer ABI and typing only internals
  produced +0.1600% and +1.6625%. The 2026-07-20 whole-family retest was neutral across Fibonacci,
  uniform, clustered, and mega, so the layout boundary is now retained through the full reader
  family.
- **QUAL-001B structural-audit result:** `LiveCellLayout::debug_assert_valid` now checks u32-backed
  cell/index capacity and every declared live span. Reconciliation invokes it only after the clean
  no-record fast path, so only defect-bearing checked builds pay for the scan. The release `.text`,
  `.rodata`, unwind sections, aggregate section sizes, and symbol addresses are identical to parent;
  only source-location/build metadata changed, so counter runs were unnecessary.
- **QUAL-001B mutation-owner decision:** a `LiveCellLayoutMut::rewrite_and_shrink` helper was tested
  on the defect-only collinear-drop mutation, with the outer reconciliation signature unchanged and
  direct coverage for stale-tail preservation. Despite full inlining, it reproduced the known
  optimizer cliff in all seven pairs: +0.15987% instructions and +1.66186% branches. The source was
  reverted; keep this local mutation flattened until surrounding codegen changes materially.
- **QUAL-001C inventory result:** the fast diagram gate, raw effective-array gate, and accumulating
  public report share the strict contract but intentionally differ in structural-input checks,
  welded-face policy, stopping behavior, and diagnostics. Several low-level facts are already
  shared; the remaining duplication is primarily traversal policy. The inventory also identifies
  incomplete exact-reason differential coverage and a fail-fast self-loop reason dominated by
  earlier duplicate/degenerate checks.
- **QUAL-001C oracle result:** test-only fixtures now pin literal shared reasons for low incidence,
  invalid references, degeneracy, duplicate vertices/cells, grouped-edge failures, antipodal arcs,
  connectivity, and Euler. Effective-only cardinality/span reasons, exhaustive self-loop dominance,
  and the report's boundary/overused/same-direction counters are also pinned. A connected closed
  torus fixture isolates Euler from the earlier connectivity check. The release artifact is
  byte-identical to parent.
- **QUAL-001C edge-class result:** `EdgeUseClass` now centrally distinguishes paired, boundary,
  overused, and same-direction groups. Both strict gates retain their combined error string and the
  report retains separate counters. The artifact added 12 text bytes, removed 16 BSS bytes, and
  grew 16 file bytes. Seven counter pairs were neutral (mean `0.999995974` instructions and
  `0.999993778` branches); all samples had zero context switches and migrations.
- **QUAL-001C typed-reason result:** a private `StrictValidationIssue` enum and exact-text mapping
  initially reproduced the optimizer cliff and was reverted. The 2026-07-20 retest is retained:
  both fail-fast validators now select one of 13 typed reasons, the effective scan carries it
  through `(cell, check_rank)` ordering, and direct coverage pins every message. Default ordinary
  instructions fall about 0.12% while branches rise about 0.12%; clustered is closer to neutral,
  mega moves by roughly one basis point, cycles show no resolved loss, and a one-codegen-unit
  control is neutral. The default artifact shrinks by 12 KiB aggregate.
- **QUAL-001C dead-branch result:** the fail-fast self-loop returns proven dominated by earlier
  duplicate/degenerate checks are removed. A direct assertion retains the accumulating report's
  self-loop telemetry. The artifact removed 60 text bytes, 4,032 BSS bytes, and 64 file bytes;
  seven counter pairs were neutral (mean `0.999998353` instructions and `0.999999721` branches).
- **QUAL-001C weld-oracle result:** a corrupt alias fixture now pins the fast gate's exact
  `"weld map"` reason and the accumulating report's matching welded-twin/issue counts. The change is
  test-only and the complete release artifact is byte-identical to parent.
- **QUAL-001C weld-predicate decision:** an inline shared predicate preserved the fast gate's early
  return and the report's accumulating count, but reproduced the optimizer cliff in all seven
  pairs: +0.16037% instructions and +1.66184% branches, with zero context switches and migrations.
  The helper was reverted and the restored artifact matches parent exactly.
- **QUAL-001C boundary:** differential reason coverage and the profitable shared facts are complete.
  Remaining scan duplication expresses different traversal policy or has failed the counter gate;
  stop mechanical validation extraction here.
- **QUAL-001A state inventory:** internal and public local-rebuild outcomes duplicate
  `attempted`/`accepted` booleans, admit the impossible false/true combination, and conflate
  ordinary no-trigger, disabled-policy, and diagnostic-capture paths. Low-incidence and Euler
  signals are independent defect facts, not action states. The exact consumers and migration are
  recorded in [`lifecycle-state-inventory.md`](lifecycle-state-inventory.md).
- **QUAL-001A local-rebuild state result:** `LocalRebuildStatus` represents the four public
  not-triggered, disabled, rejected, and accepted outcomes. Public report booleans were removed
  atomically; derived methods preserve the existing KV values. Private `LocalRebuildExecution`
  now owns the feature-gated diagnostic interception, so repository probe control cannot become a
  public status. The original enum migration added 212 text bytes, 3,888 BSS bytes, and 208 file
  bytes; seven counter pairs were neutral (mean `1.000001952` instructions and `0.999998724`
  branches). The seam-removal follow-up leaves every default artifact size identical and is neutral
  across seven pairs (`0.999997426` instructions, `0.999994791` branches); its feature-gated test
  pins capture plus public conversion.
- **QUAL-001A resolution state result:** `ResolutionDiscoveryMode` replaces the exact-inverse
  certified-hint/drift-fallback booleans, and timing stores only the fallback bit while deriving both
  existing KV values. The artifact exchanged 16 BSS bytes for 16 text bytes with identical total
  and file size. Seven counter pairs were neutral (mean `1.000000548` instructions and
  `1.000004166` branches).
- **QUAL-001A effective-input inventory:** production permits only identity or actually merged
  input, but currently splits the representative points out of `MergeResult` and stores both halves
  as independent `Option`s. The selected boundary is one identity/merged enum that retains the
  complete merge result; a named preparation record will replace the ambiguous return tuple. The
  identity state continues to borrow original points without retained duplicate storage. Exact
  consumers and the migration gate are recorded in
  [`lifecycle-state-inventory.md`](lifecycle-state-inventory.md).
- **QUAL-001A effective-input result:** `EffectiveInput` now owns either identity state or the
  complete `MergeResult`; representative points are no longer moved out and paired with independent
  optional metadata. `PreparedPointsAndGrid` is a named record, and a direct contract test pins
  disabled, no-merge, and actual-merge outcomes. The matched release artifact kept aggregate size
  at `2,239,146` bytes, moved 656 bytes from text to BSS, and reduced file size by 664 bytes. Seven
  counter pairs were neutral (mean `0.999998159` instructions and `0.999998587` branches), with
  zero context switches and migrations.
- **QUAL-001A effective-geometry inventory:** positions, cell spans, and the live index buffer form
  one coherent mutable diagram across reconciliation, local rebuilding, output resolution, report
  cloning, and remapping. Assembly vertex keys are intentionally partial provenance after minted
  rebuild vertices and remain outside the geometry owner; residual records and mutation footprints
  also have independent lifetimes. The selected boundary is one private `EffectiveGeometry` record
  that preserves accepted rebuilding's append-only base-position reuse. Exact consumers and the
  migration gate are recorded in
  [`lifecycle-state-inventory.md`](lifecycle-state-inventory.md).
- **QUAL-001A effective-geometry result:** private `EffectiveGeometry` now owns positions, cell
  spans, and the live index buffer from assembly through remapping. Reconciliation returns only its
  diagnostics, local rebuilding mutates the owner without changing append/rollback behavior, and
  assembly provenance remains separately named and intentionally partial. The artifact removed
  1,484 text bytes, added 1,488 BSS bytes, retained data size, grew aggregate accounting by four
  bytes, and reduced file size by 1,040 bytes. Seven counter pairs were neutral (mean
  `1.000000038` instructions and `0.999998878` branches), with zero context switches and
  migrations. This completes the planned QUAL-001A cold-state modeling slice.
- **QUAL-001B overlay reassessment:** `WorkingDiagram` borrows reconciled cell records and their
  backing index buffer independently even though every base-boundary read and materialization needs
  the pair. The selected slice stores one `LiveCellLayout`, adds only direct-index span and backing
  length accessors, and renames the constructor to describe its post-reconciliation input. The
  previously rejected mutable-rewrite signature remains a raw measured fallback; the semantic
  comparison and unpaired scan were accepted by later layout retests. Exact consumers and the
  migration gate are recorded in
  [`live-cell-layout-inventory.md`](live-cell-layout-inventory.md).
- **QUAL-001B overlay result:** `WorkingDiagram` now stores one `LiveCellLayout`; its
  `from_reconciled` boundary requires the caller to pair cell records with their backing indices.
  Base traversal, residual capacity, and flattening use direct accessors while override and minted
  storage remain unchanged. Release/checked/no-default/Clippy gates passed. The artifact shrank 16
  aggregate bytes and 48 file bytes; seven 500k single-thread Fibonacci pairs were neutral (mean
  `1.000000228` instructions, `1.000001841` branches), with zero switches/migrations. The 100k mega
  seeds 1, 2, and 15 are now valid with rebuilding disabled, and a debug run of that test did not
  trigger the overlay, so no active-overlay counter result is claimed.
- **QUAL-001B effective-validation inventory:** the gate has one production caller and one private
  parallel cell-scan family. Only candidate cells and their backing indices are one representation;
  generators retain an independent cardinality check, and temporarily appended vertices have a
  distinct transaction lifetime. The selected migration carries `LiveCellLayout` through that
  scan without sharing traversal or failure types with the other validators. Its prerequisite is
  an explicit checked-span end-overflow error so 32-bit malformed input retains the gate's current
  checked-add behavior. The boundary, rejected shapes, and gate are recorded in
  [`effective-validation-layout-inventory.md`](effective-validation-layout-inventory.md).
- **QUAL-001B effective-validation result:** the full caller-to-scan layout originally displaced clean
  Fibonacci counters in all seven pairs (`1.001290682` instructions, `1.013601256` branches).
  Restoring the raw outer ABI and typing only the private scan reproduced the result
  (`1.001293695`, `1.013603930`). Restoring the validator completely and retaining only
  overflow-safe checked-span hardening also reproduced it (`1.001294912`, `1.013604706`) with the
  familiar BSS-collapse fingerprint, so all production changes were reverted. The 2026-07-20
  retest after surrounding codegen changes retained the full boundary: default instructions added
  only `0.012%..=0.031%` across four regimes with neutral branches and no adverse cycle signal,
  while a one-codegen-unit control was neutral. The effective gate now owns one portable checked
  cell/index layout through its private parallel scan.
- **QUAL-001B assembly-handoff decision:** assembly creates a stronger freshly compacted layout and
  has one production consumer, which immediately moves its vectors into `EffectiveGeometry`.
  Nesting an owned layout in `AssemblyResult` would add only a wrapper/unwrapper move; carrying it
  farther would duplicate the accepted geometry owner and reopen measured mutation signatures.
  The stage is closed without a production candidate or benchmark. Retry only if a second consumer
  or a natural shared owner appears; details are in
  [`assembly-handoff-layout-inventory.md`](assembly-handoff-layout-inventory.md).
- **QUAL-001D reconciliation state result:** `ReconcileRunState` now owns the merge ledger,
  rejected-component rebuild seeds, merge-affected cells, mutation scan cells, and merge-safety
  counters shared across primary and synthesized backstop rounds. It replaces the finalizing
  closure and four independently threaded accumulators while leaving the empty-record return,
  flattened pass sequence, allocations, option boundary, and measured raw layout signatures
  unchanged. Complete validation passed. Seven 500k Fibonacci pairs were neutral
  (`1.000003220` instructions, `1.000004729` branches); active 100k `cubed` pairs were neutral
  (`1.000010264`, `1.000007825`), as were five 500k `cubed` confirmations (`1.000003216`,
  `1.000004527`). All samples had zero switches/migrations. The artifact removed 544 text bytes,
  3,552 BSS bytes, and 616 file bytes. Details are in
  [`reconciliation-orchestration-inventory.md`](reconciliation-orchestration-inventory.md).
- **QUAL-001D reconciliation helper result:** `reconcile_edge_mismatches` retains the empty-record
  gate and checked layout audit, then delegates the recorded-mismatch program to a private helper
  with the same seven explicit arguments. LLVM inlined the boundary and retained the relevant
  function sizes. Complete validation passed. Seven 500k Fibonacci pairs were neutral
  (`0.999998589` instructions, `0.999999242` branches), as were seven active 100k `cubed` pairs
  (`1.000010280`, `1.000020864`); every sample had zero switches/migrations. The artifact added 8
  text bytes and 24 file bytes, retained data size, and removed 16 BSS bytes. Details are in
  [`reconciliation-defect-body-inventory.md`](reconciliation-defect-body-inventory.md).
- **QUAL-001D local-rebuild transaction result:** `LocalRebuildCandidate` now owns minted
  positions, complete replacement arrays, and the sorted mutation footprint after consuming the
  overlay. Its consuming method owns append, whole-diagram validation, diagnostics, rollback, and
  paired commit; all trigger and growth policy remains in `maybe_rebuild_effective`. Direct
  accepted/rejected transaction tests and complete validation passed. LLVM emitted no standalone
  boundary. Seven 500k Fibonacci pairs were neutral (`0.999999187` instructions,
  `0.999999792` branches), as were seven verified seed-224 productive-rejection pairs
  (`0.999991525`, `1.000000202`); all samples had zero switches/migrations. Actual `.bss` was
  unchanged, while the artifact added 1,308 text bytes and 992 file bytes. Details are in
  [`local-rebuild-transaction-inventory.md`](local-rebuild-transaction-inventory.md); the fixture
  search remains in [`local-rebuild-trigger-hunt.md`](local-rebuild-trigger-hunt.md).
- **QUAL-001D assembly hint-confirmation result:** private `ConfirmedZeroEdgeHints` and
  `confirm_exact_zero_edge_hints` now own only the read-only final post-patch exact-zero scan and
  its correlated candidate/count result. Mutable repair, materialization, unsafe scatter, sparse
  patching, timing, and output policy remain unchanged. The direct evidence regression and complete
  validation gates passed. LLVM fully inlined the helper, shrank the assembly body by 50 bytes and
  aggregate text by 48 bytes, retained data and actual `.bss`, and reduced file size by 72 bytes.
  Seven counter pairs for each default/high-bin Fibonacci and uniform gate plus clustered seed 1
  were neutral; instruction means ranged from `0.99999881` to `1.00000211`, branch means from
  `0.99999471` to `1.00000062`, and every sample had zero switches/migrations. Details are in
  [`assembly-phase-inventory.md`](assembly-phase-inventory.md).
- **QUAL-001D packed-range result:** `collect_directed_ranges` and its four-field summary now own
  center-plus-neighbor ordering, same-bin eligibility, and hard/aggregate work gates. The compact
  form was rejected after adding 0.1397% instructions on clustered input. The retained form keeps
  the original later center-range read: ordinary Fibonacci and default/high-bin uniform add about
  0.009–0.010% instructions while removing about 0.004–0.006% branches; clustered and mega are
  effectively neutral. It adds 64 text bytes and 88 file bytes, with unchanged data, actual `.bss`,
  and driver-closure size. Direct invariants and every complete validation gate pass; all counter
  samples had zero switches/migrations. This is accepted as practical neutrality for a useful
  tested phase boundary. Details are in
  [`packed-preparation-inventory.md`](packed-preparation-inventory.md).
- **Closeout:** all nine workstreams are completed or closed at explicit measured boundaries.
  `CellId`/`VertexId` cover the remaining ambiguity-prone cold overlay seams; other raw identities
  are storage/traversal values or recreate the rejected rebuild-pair owner. Selective cell-layout
  and validation sharing stops at the accepted readers/classifications, while measured
  optimizer-sensitive expressions remain local. Reconciliation, local-rebuild transaction,
  assembly hint confirmation, and packed range setup provide the justified phase boundaries;
  remaining hot programs stay flattened.
- **Reopen condition:** a new consumer or ownership invariant, a correctness defect, or a material
  compiler/profile change that invalidates the recorded codegen evidence. Function length or an
  old unchecked milestone item alone is not sufficient.
- **Gate:** every production refactor preserves semantic fingerprints and passes the affected
  workload's interleaved performance gate against its immediate parent. Readability alone does not
  justify a repeatable throughput, memory, or code-size regression.
- **Reference:** [`code-quality-plan.md`](code-quality-plan.md).

## Research backburner

### RESEARCH-001 — Unified exact normalized-site combinatorics

- **Priority:** P3
- **Status:** Backburner
- **Scope if revived:** choose one normalized site model, add filtered exact clipping signs, derive
  compatible kNN termination bounds, share one exact-zero/SoS policy, handle non-simplicial
  (>3-generator) vertices, and prevent tolerance-based reconciliation from becoming an authority over exact
  combinatorics.
- **Gate:** first measure exact-filter activation and fast-path cost. This is an optional add-on,
  not the core graphical contract.

### RESEARCH-002 — Certified exhaustive ownership diagnostic

- **Priority:** P3
- **Status:** Backburner
- **Goal:** independently search normalized-site ownership without claiming that the production
  combinatorics use that same model.
- **Gate:** justify its cost and define conditioning/degeneracy buckets before treating differences
  as failures.

### RESEARCH-003 — Independent certified reference

- **Priority:** P3
- **Status:** Backburner
- **Goal:** select or build a comparison implementation with certified-adaptive predicates and an
  explicit exact-zero/SoS policy compatible with RESEARCH-001.
- **Rule:** a reference result is not deciding evidence merely because it comes from another
  library. Qhull is explicitly not a correctness oracle.

### RESEARCH-004 — f64 input and output

- **Priority:** P3
- **Status:** Backburner
- **Scope:** parallel f64 site/output types, f64 clipping/reconciliation/local-rebuild/validation/measures, a sound f64
  search certificate, and an exact-duplicate policy. An f64 API must not silently round through the
  existing f32 representation.
- **Reference:** [`../ROADMAP.md`](../ROADMAP.md).

## Suggested order

1. Continue WORK-002 whenever construction, reconciliation, or Hull3d changes.
2. Execute QUAL-001 in staged, independently benchmarked milestones.
3. Decide RES-002 only when positive-threshold mesh conditioning is wanted.
4. Revisit PERF-001 only with a motivating workload and crossover measurements.
5. Keep RESEARCH-001 through RESEARCH-004 parked unless the project contract expands.

## Closed and retired work

### WORK-001 — Output-resolution certificate soak and component hardening

- **Priority:** P2
- **Status:** Completed 2026-07-15
- **Differential fixtures:** an oriented prism family covers a maximal safe zero-edge tree,
  multiple safe components sharing a cell, multiple individually safe components that jointly
  kill a cell, and a cell-killing cycle. Twenty-four vertex/cell/cycle permutations per family
  plus 64 deterministic randomized forest assemblies produced identical localized and exhaustive
  reports and quotients; every terminal synthetic diagram validated strictly.
- **Production soak:** 29 strict-valid timing-enabled cases covered eight ordinary and
  density-contrast distributions at 50k plus focused 1M clustered and 100k mega cases. All 29 used
  certified discovery with no drift fallback. The runs visited 235,681 hint cells, rechecked 72
  construction candidates, and detected 71 final exact-zero edges.
- **Mutation evidence:** clustered 1M seed 1 reported four reconciliation scan cells and changed 18
  construction candidates into 17 actual terminal edges, directly exercising the stale-hint
  recheck. No accepted Hull3d rebuild occurred naturally in the bounded soak; its complete splice
  footprint remains pinned by a direct regression and stays observable in ongoing WORK-002 runs.
- **Boundary coverage:** signed zero, exact threshold, and adjacent-f32 cases remain pinned.

### WORK-003 — Post-construction zero-edge invalidation

- **Priority:** P2
- **Status:** Completed 2026-07-15
- **Resolution:** reconciliation reports the exact local cover for accepted merges and collinear
  drops; accepted Hull3d rebuilding reports every spliced cell. Terminal discovery rescans those final cycles
  and rechecks construction-candidate neighborhoods before canonicalization.
- **Locality rule:** a cycle rewrite can only create an edge in a rewritten cycle. A mutator that
  changes an existing vertex position must report every incident cell. Representative-drift or
  missing provenance retains the whole-diagram fallback.
- **Regression:** an unhinted zero edge attributed to a post-construction mutation produces the
  same report and quotient as exhaustive discovery; direct tests pin reconciliation and Hull3d
  footprint reporting.
- **Fast path:** 10-round, 500k, single-thread perf counters versus `43a125a` measured
  instruction/branch ratios of `1.00126`/`1.00172` for Fibonacci and `0.99943`/`1.00174` for
  uniform. The mutation footprints were empty in ordinary runs; release inlining at the existing
  per-generator phase seams is pinned to prevent unrelated cold-path codegen perturbation.

- AUD-001 through AUD-017: closed; see [`audit-triage.md`](audit-triage.md).
- Exact stored-zero baseline and discovery certificate: implemented and reviewed.
- Near-pi owner-plane validation and geometry consumers: implemented.
- Actual-exhaustion spherical reconstruction: implemented; fixed-budget early handoff rejected for
  now.
- Qhull comparison backend and P5 shadow audit plumbing: removed.
- Rejected optimization experiments and their measurements remain in
  [`performance.md`](performance.md); do not reopen them without a materially different design or
  workload.
