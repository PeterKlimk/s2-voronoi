# Triage and Work Log

**Status:** active

**Last reorganized:** 2026-07-15

This is the authoritative list of unfinished correctness, robustness, and design work. Historical
investigations stay in [`audit-triage.md`](audit-triage.md); design rationale stays in the linked
policy documents. An unchecked item elsewhere should either be moved here or treated as stale.

## Current state

- The July 2026 correctness and safety audit is closed. AUD-001 through AUD-017 have no open
  correctness or policy finding under the documented production contract.
- The production promise is a construction-certified, edge-agreeing, Euler-valid spherical mesh
  or a defined error. It is not exact combinatorial equality with one ideal normalized-site model.
- Exact stored-zero edges are detected after final repair and safely contracted when doing so does
  not remove an effective generator cell.
- Every current post-assembly mutator reports a complete changed-cell footprint for terminal
  exact-zero scanning; only globally uncertified representative drift forces global discovery.
- Full validation and `OutputResolutionReport` now count canonical cells with fewer than three
  exact stored positions. A construction-local separation certificate plus exact scans of its
  sparse risk set and every mutator footprint make that classification complete without a global
  ordinary-path scan.
- Qhull and the legacy P5 shadow path have been retired.
- Full strict validation remains available for testing and campaigns without burdening the default
  fast path.

There is no active P0/P1 correctness defect. Construction-certificate differential maintenance
remains ongoing. The weld-radius/whole-cell bound attempt is complete: it identified a metric
margin gap and replaced the missing global proof with a direct finite-representation certificate.

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
| WORK-002 | P2 | Ongoing | Exercise after construction/repair changes |
| WELD-001 | P2 | Completed | Metric proof gap classified; direct certificate implemented |
| RES-002 | P2 | Decision | Choose positive-threshold units and certificates |
| PERF-001 | P3 | Backburner | Obtain motivating workload and crossover data |
| RESEARCH-001 | P3 | Backburner | Expand the production combinatorics contract |
| RESEARCH-002 | P3 | Backburner | Justify diagnostic cost and conditioning policy |
| RESEARCH-003 | P3 | Backburner | First choose a compatible exact-zero/SoS model |
| RESEARCH-004 | P3 | Backburner | Commit to full f64 representation and search bounds |

## Ongoing work

### WELD-001 — Sufficient-welding whole-cell geometry bound

- **Priority:** P2
- **Status:** Completed 2026-07-15
- **Goal:** determine whether the default weld separation provably prevents an effective generator
  cell from collapsing at f32 output resolution.
- **Starting theorem:** minimum geodesic site separation `alpha` gives every exact Voronoi cell a
  contained spherical cap of radius `alpha / 2`.
- **Required composition:** bound input canonicalization/normalization, f64 clipping, gnomonic and
  spherical fallback conversion, reconciliation/repair displacement, and final f32 storage against
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
  Local3d likewise passes whole-diagram strict topology validation without a coordinate-separation
  or Hausdorff certificate. A naïve displacement composition would require a weld chord above
  roughly `2.0e-6` nominal, `2.2384e-6` with normalization slack, or `2.4768e-6` under the
  conservative off-shell model, before other construction/storage errors.
- **Implemented closure:** each hot constructed cell proves three final stored-position classes
  when its local x coordinates contain three successively separated classes beyond the existing
  `2r + 1 ULP` representative-drift bound. Only uncertified cells are retained for an exact final
  coordinate-class scan. Reconciliation and Local3d footprints are always scanned; representative
  drift invalidation scans the whole diagram. Full validation independently counts the same
  condition. `Preserve` still treats it as representation telemetry rather than a topology defect.
- **Evidence:** active threshold-adjacent tests cover axis, cube-face seam, cube corner, f32
  half/quarter exponent-boundary orientations, two rotations, and near-cocircular stress. The
  manually run extended campaign covered 256 rotated threshold-adjacent cases. Direct final
  coordinate scans found no collapse and agreed with localized telemetry. Forced spherical handoff,
  mutation-footprint, and welding-disabled positive controls are pinned separately.
- **Fast-path cost:** the stored-position min/max work shares the existing exact-zero hint pass and
  adds one short pass over each extracted cell. Ten-round interleaved single-thread native counters
  at 500k sites versus `11b49a4` measured instruction/branch ratios `1.00795`/`1.00450` for
  Fibonacci and `1.00746`/`1.00409` for uniform seed 12345. Cycle ratios were `1.00460` and
  `0.97096` respectively and are treated as noisy; the stable cost estimate is about 0.75–0.80%
  retired instructions.
- **Conclusion:** do not promote the current weld radius to a standalone mathematical guarantee
  that output geometry cannot collapse. The production contract instead combines strong weld
  evidence with a direct complete finite-representation classification and explicit
  Preserve/Error/Elide policy surfaces.

### WORK-002 — Construction-certificate differential maintenance

- **Priority:** P2
- **Status:** Ongoing
- **Scope:**
  - keep the exhaustive uniform small-`n` geometry campaign active;
  - after reconciliation or Local3d edits, check owner equality and edge-bisector residuals as well
    as topology;
  - retain negative controls for reversed faces, duplicate faces/references, moved vertices, and
    disconnected unions of closed complexes;
  - compare semantic topology across thread counts, bin counts, default/scalar SIMD, and FMA; and
  - keep welding and deterministic perturbation in separate expected-policy buckets.
- **Acceptance:** every supported successful result passes the relevant strict and intrinsic
  geometry checks; Local3d escalation remains a valid success rather than a failure count.
- **Latest maintenance (2026-07-15):**
  - the extended uniform small-`n` campaign passed 62,464 intrinsic assessments; worst ownership,
    vertex cross-track, and edge cross-track errors were respectively `1.507e-7`, `7.918e-8`, and
    `1.833e-16` radians;
  - clustered 1M seed 1 remained strict-valid across 1/6 threads and 6/96 bins, with the same three
    pre-repair defects, four-cell reconciliation footprint, 18 hinted versus 17 terminal zero
    edges, zero ownership mismatches in 733 samples, and `5.875e-8 rad` maximum sampled edge
    cross-track error;
  - the repair-net suite passed all five active tests, including exact output agreement between
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
- **Status:** Blocked
- **Dependencies:** RES-001
- **Goal:** let graphical or physics consumers explicitly remove represented nonzero slivers.
- **Decisions required:**
  - canonical threshold units (squared chord internally, with or without an angular convenience
    API);
  - positive-threshold option/report fields on the distinct cell-mesh conversion;
  - whether pre-storage f64 collision telemetry is useful; and
  - the component-diameter and geometric-deviation certificate.
- **Contract:** the result is a valid spherical cell complex after explicit simplification, not the
  exact Voronoi diagram of the original generators.
- **Reference:** [`output-resolution-policy.md`](output-resolution-policy.md).

## Performance robustness

### PERF-001 — Total-query-work circuit breaker

- **Priority:** P3
- **Status:** Backburner
- **Motivation:** Perturbed great-circle inputs can become gnomonically bounded yet process nearly
  every generator for some cells. The existing exhaustion replay is correct but does not detect
  this successful high-work regime.
- **Candidate direction:** a progress-aware total-work budget followed by unrestricted spherical,
  Local3d, or global-hull escalation.
- **Before implementation:** measure the actual cold-replay crossover, avoid a fixed candidate
  count such as 128, and prove that the handoff cannot turn a valid success into a failure.
- **Reference:** AUD-015 in [`audit-triage.md`](audit-triage.md).

The remaining code-specific performance experiments are maintained separately in the open queue in
[`performance.md`](performance.md) and the memory backlog in
[`memory-layout-ideas.md`](memory-layout-ideas.md). They are not correctness tasks and are not
duplicated here.

## Research backburner

### RESEARCH-001 — Unified exact normalized-site combinatorics

- **Priority:** P3
- **Status:** Backburner
- **Scope if revived:** choose one normalized site model, add filtered exact clipping signs, derive
  compatible kNN termination bounds, share one exact-zero/SoS policy, handle non-simplicial
  (>3-generator) vertices, and prevent tolerance repair from becoming an authority over exact
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
- **Scope:** parallel f64 site/output types, f64 clipping/repair/validation/measures, a sound f64
  search certificate, and an exact-duplicate policy. An f64 API must not silently round through the
  existing f32 representation.
- **Reference:** [`../ROADMAP.md`](../ROADMAP.md).

## Suggested order

1. Continue WORK-002 whenever construction, reconciliation, or Local3d changes.
2. Decide RES-002 only when positive-threshold mesh conditioning is wanted.
3. Revisit PERF-001 only with a motivating workload and crossover measurements.
4. Keep RESEARCH-001 through RESEARCH-004 parked unless the project contract expands.

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
- **Repair evidence:** clustered 1M seed 1 reported four reconciliation scan cells and changed 18
  construction candidates into 17 actual terminal edges, directly exercising the stale-hint
  recheck. No accepted Local3d repair occurred naturally in the bounded soak; its complete splice
  footprint remains pinned by a direct regression and stays observable in ongoing WORK-002 runs.
- **Boundary coverage:** signed zero, exact threshold, and adjacent-f32 cases remain pinned.

### WORK-003 — Post-construction zero-edge invalidation

- **Priority:** P2
- **Status:** Completed 2026-07-15
- **Resolution:** reconciliation reports the exact local cover for accepted merges and collinear
  drops; accepted Local3d reports every spliced cell. Terminal discovery rescans those final cycles
  and rechecks construction-candidate neighborhoods before canonicalization.
- **Locality rule:** a cycle rewrite can only create an edge in a rewritten cycle. A mutator that
  changes an existing vertex position must report every incident cell. Representative-drift or
  missing provenance retains the whole-diagram fallback.
- **Regression:** an unhinted zero edge attributed to a post-construction mutation produces the
  same report and quotient as exhaustive discovery; direct tests pin reconciliation and Local3d
  footprint reporting.
- **Fast path:** 10-round, 500k, single-thread perf counters versus `43a125a` measured
  instruction/branch ratios of `1.00126`/`1.00172` for Fibonacci and `0.99943`/`1.00174` for
  uniform. The repair footprints were empty in ordinary runs; release inlining at the existing
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
