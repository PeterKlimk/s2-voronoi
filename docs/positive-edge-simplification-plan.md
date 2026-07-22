# RES-002 replacement plan: sparse positive-edge resolution

**Status:** reviewed redesign approved for implementation; current implementation remains only as
a comparison oracle

**Date:** 2026-07-23

## Decision

Replace the post-computation positive-simplification engine with a construction-aware extension of
exact-zero output resolution.

The caller supplies epsilon before construction. The existing per-cell resolution scan records a
conservative positive candidate hint. After assembly, repair, and exact-zero resolution, the
pipeline confirms candidates only in hinted or mutated cells. It then selects deterministic bounded
components, rewrites the affected cells as one batch, and validates the final mesh once.

The replacement is cell-preserving. A positive contraction which would remove a generator cell or
fail a local manifold check is declined. Positive `Error` and `Elide`, degree-two suppression, and
post-computation thresholds are outside this plan.

## Goal

Given a positive unit-sphere chord threshold before construction:

- find every final live edge at or below the threshold without a normal whole-mesh edge scan;
- contract safe positive edges using the sparse ownership model of exact-zero resolution;
- preserve every effective generator cell;
- bound every retired source vertex's displacement to its retained representative by epsilon;
- consider every positive edge in the terminal pre-simplification mesh once;
- return a dense, strictly valid `SphericalCellMesh`; and
- leave ordinary `compute` behavior and performance unchanged.

The result is an approximate spherical cell complex, not a Voronoi diagram of the source sites.

## Non-goals

- Choosing epsilon after construction.
- Positive cell elision or positive cell-killing errors.
- General polyline/degree-two arc simplification.
- Optimal decimation, optimized vertex positions, or adaptive thresholds.
- Retrying subsets to maximize contractions after a combined batch fails certification.
- Global Hausdorff, area, locator, Lloyd, or Delaunay guarantees.
- Parallel contractions in the first implementation.
- Public work-budget controls.
- A fixed point over positive edges exposed by positive contraction.

Ordinary exact-zero canonicalization and explicit exact-zero cell elision retain their current
contracts.

## API direction

Use a dedicated computation entry point whose options are available to the backend from the start:

```rust,ignore
let options = CellSimplificationOptions::from_chord_length(epsilon)?;
let output = compute_simplified_with(points, VoronoiConfig::default(), options)?;
```

The exact names should follow the existing default/configured, closure-ingest, and embedded naming
patterns. The successful output contains the mesh, ordinary `ComputeReport`, and a small
simplification report.

`CellSimplificationOptions` initially contains only a finite chord threshold in `(0, 2]`.

Remove the current positive-only surface rather than adapting it:

- `ComputeOutput::into_simplified_cell_mesh`;
- `SimplificationCellPolicy`;
- `CellSimplificationLimits` and public work counters;
- recoverable post-computation simplification errors; and
- positive suppression/Elide telemetry.

This is an intentional pre-release API correction. Ordinary computation and exact-elision APIs do
not change.

## Semantics

### Candidate metric

Promote stored f32 coordinates to f64, then compare squared chord distance inclusively with
`epsilon * epsilon`. Exact stored equality remains a separate first phase.

### Representative and displacement

A contraction retains an existing live vertex; it does not average or create a position. Each
provisional component stores its representative, member count, and a conservative upper bound `r`
on stored-chord displacement from every source member to that representative.

For components represented by `a` and `b`, let `d` be their stored chord. Retaining `a` gives:

```text
r_to_a = max(r_a, d + r_b)
```

and retaining `b` gives the symmetric bound. These follow directly from the Euclidean triangle
inequality over promoted stored coordinates. Accept a union when either bound is at most epsilon;
choose the smaller admissible bound, then break ties by larger member count and lower stable id.

This constant-work conservative test avoids both source-member scans and the current all-pairs
diameter cost while providing the directly useful guarantee:

```text
stored_chord(source_position, final_representative) <= epsilon
```

### Long chains

Positive proximity is not transitive. Do not form one component from an entire connected chain.
Order live candidates by:

1. stored chord squared;
2. lower stable endpoint id; then
3. higher stable endpoint id.

Use this order to build bounded provisional unions. A long chain therefore becomes several bounded
clusters connected by coarser edges instead of being collapsed or declined wholesale.

The operation does not recursively simplify positive edges exposed by its own contractions. Those
edges were not candidates in the terminal source mesh. Exact-zero edges exposed by a contraction
remain mandatory representation closure.

## Algorithm

### 1. Construction hint

The exact-zero path already scans every extracted cell edge and records a cell when its
x-separation could become exact zero after representative selection. For a positive request, use
the same scan with a conservatively rounded bound:

```text
hot_x_threshold = epsilon + 2 * representative_x_drift_bound
```

Final chord `<= epsilon` implies final x-separation `<= epsilon`, so the existing drift proof makes
this a complete necessary hint. Record cell ids, not pre-dedup edge ids.

Ordinary computation keeps its current specialization: no positive branch, allocation, or retained
metadata.

### 2. Final candidate confirmation

Run positive confirmation after deduplication, reconciliation, optional local rebuild, and
exact-zero canonicalization. Scan the union of:

- construction-hinted cells;
- reconciliation-changed cells;
- local-rebuild-changed cells; and
- exact-resolution-changed cells.

The exact resolver must expose its changed-cell footprint. If representative drift exceeds the
existing certificate or ownership data are incomplete, perform one exhaustive final scan.

Confirm full stored chord distance using final positions, canonicalize endpoint ids, sort by the
defined candidate order, and deduplicate the two cell uses.

If cell-preserving exact resolution leaves an exact-zero edge or a face with fewer than three
stored positions, return an explicit unrepresentable-cell-mesh error before positive work. This is
an expected source outcome, not an internal validation failure.

If no positive candidate remains, skip construction of the contraction engine and proceed directly
to cell-mesh materialization and validation.

### 3. Deterministic batch selection

Retain stable vertex ids until final compaction. Build sparse provisional union state only for
confirmed candidate endpoints.

Visit the sorted candidates once. For each edge:

1. resolve its provisional roots and skip a self-edge;
2. compute the two triangle-inequality radius bounds;
3. select the deterministic admissible representative; and
4. union the clusters when a bound passes, otherwise record a displacement decline.

This produces deterministic, radius-bounded components without constructing the whole
threshold-connected graph or retrying rejected edges.

Use the existing vertex keys to collect every cell incident to a selected component. Transitively
group selected components which co-occur in a cell, as in exact resolution. Classify every group in
one pass over the candidate-cell cover. Cell-killing or non-simple groups are declined as a whole;
accepted groups contribute replacements to one combined affected-cell rewrite.

Collect the complete stars needed by the link certificate before mutation. If a rebuilt vertex has
no retained key provenance, build one exhaustive incidence view for this batch rather than
maintaining a second dynamic topology representation or falling back per component.

### 4. Batched rewrite and local certificate

Extend `output_resolution::verify_affected_quotient`; do not use the current global positive
certificate. The tentative batch must prove that:

- every affected face retains at least three ids and three stored positions;
- no affected face repeats an id or duplicates another affected face;
- no cell is removed;
- every changed edge has two opposite-oriented cell uses;
- affected live vertices retain incidence at least three;
- every affected vertex's complete link is one unbranched cycle;
- the local Euler delta matches the retired vertex and changed edges; and
- no exact-zero or exactly antipodal edge remains from the transaction.

After tentative rewriting, rescan the affected cells for induced exact-zero edges. Resolve those
edges on the same scratch state using a narrowly extracted exact affected-cycle helper. If closure
or the combined local certificate fails, roll back the entire tentative positive batch without
retrying subsets. Do not attempt to invoke the existing whole-operation exact canonicalizer as a
nested transaction.

Scratch contains saved affected spans and local incidence/union state. Edge contraction only
shrinks cycles, so no whole-mesh cycle or membership clone is needed.

The source bounds the batch structurally: confirmed candidates are no more numerous than live
edges, sorted union selection is `O(K log K)` with constant-work merge bounds, and candidate cells
plus complete stars are classified once. The only unconditional whole-mesh work is final
validation. This is why the replacement needs no public work budget.

Identifying points in a connected complex cannot disconnect it; with no face deletion, complete
manifold links and the local Euler check remove the need for a separate global connectivity
traversal. Final strict validation remains the independent backstop.

### 5. Affected rescan and finalization

After publishing the accepted batch:

1. rescan the affected cells once to count newly exposed positive edges;
2. compact live vertices once;
3. rebuild dense cell spans and original-input mappings once; and
4. run strict `SphericalCellMesh` validation once.

Unsafe candidates are normal declines. An unexpected final validation failure returns an internal
computation error; no partial mesh is published.

## Report

Keep the public report limited to:

- requested threshold and promoted squared comparison value;
- hinted and confirmed candidate counts;
- attempted and accepted contractions;
- displacement-, cell-, and topology-declined counts;
- positive edges newly exposed by the accepted batch;
- vertices removed;
- maximum accepted representative-displacement bound; and
- final validation.

Detailed selection and affected-cover counters belong to tools/timing instrumentation.

## Reuse and deletion

Reuse or narrowly generalize:

- the hot x-separation hint and drift fallback;
- reconciliation/rebuild mutation footprints;
- stored-distance helpers;
- exact-zero closure;
- vertex-key/incidence affected-cell discovery;
- shrinking-span rewrite and rollback;
- the affected quotient certificate; and
- cell-mesh mapping, compaction, and validation.

Delete after the replacement passes correctness and performance gates:

- repeated whole-mesh snapshots and certification;
- whole-cycle and whole-ledger proposal clones;
- all-pairs threshold-component diameter checks;
- interaction-group global fixed-point rescans;
- dynamic candidate queues, endpoint generations, and mutable whole-mesh incidence;
- provenance bags, sinks, cause taint, and arc certification;
- positive degree-two suppression and cell elision;
- public simplification budgets/phase errors; and
- the post-computation conversion API.

Do not introduce a generic mesh-editing framework. A shared helper must have the same ownership and
invariant in exact and positive resolution.

## Performance gates

The current engine is the comparison baseline: 10k Fibonacci measured 42.9 ms candidate-free,
297 ms for 36 candidate occurrences / 8 commits, and 2.28 s for 3,403 occurrences / 82 commits. A
100k candidate-heavy run hit the 100,000,000 cell-visit limit after 10.86 s.

Acceptance requires:

- no material ordinary-compute counter regression;
- no full final-edge scan or contraction-engine construction on the certified candidate-free path;
- one candidate sort/selection pass and one affected-cell classification/rewrite pass;
- no work proportional to `accepted_components * total_cell_indices`;
- material improvement on the recorded sparse and heavy 10k fixtures;
- the 100k heavy fixture completing or declining locally without the former global-work failure;
  and
- no whole-mesh clone per proposal.

Record release wall time and structural counters for candidate-free, sparse, long-chain, and heavy
inputs. Set absolute timing gates from the first minimal prototype on the same reference machine,
not by extrapolation.

## Required tests

- Sparse hint discovery equals exhaustive final discovery, including threshold boundaries and
  drift fallback.
- Repair, rebuild, and exact-resolution mutation cells cannot hide candidates.
- Candidate-free input bypasses the contraction engine.
- An isolated positive edge contracts without removing a cell.
- Cell-killing, non-simple, duplicate-face, low-incidence, pinched-link, misoriented-edge, and
  antipodal proposals decline without mutation.
- A long chain partitions into deterministic radius-bounded clusters.
- Cumulative displacement passes below/at epsilon and declines above it.
- Positive edges exposed by the batch are reported but not recursively simplified.
- Induced exact closure commits or rolls back with the tentative positive batch.
- Local decisions agree with exhaustive scratch apply plus strict validation in property fixtures.
- Original-input weld provenance remains coherent.
- Repeated runs, thread counts, SIMD backends, FMA, supported features, checked tests, and the full
  release suite retain their applicable contracts.

The current engine may generate fixtures and negative cases; agreement with its all-or-nothing
component semantics is not required.

## Implementation sequence

1. Preserve current benchmarks and fixtures as the comparison baseline.
2. Add the internal preconstruction threshold, generalized hot hint, exact changed-cell footprint,
   and sparse-vs-exhaustive discovery tests without changing public behavior.
3. Implement deterministic bounded-component selection, one affected-cell batch, and the
   strengthened affected-link certificate behind an internal entry point.
4. Pass the performance gates before adding public API.
5. Replace the unit-sphere and embedded public surfaces and documentation.
6. Delete the old global engine and obsolete API/tests/telemetry.
7. Run formatting, all-target clippy, focused checked/feature suites, full release tests, and
   interleaved performance comparisons.

Each stage is one coherent commit. Review must first ask whether a mechanism is necessary for the
stated goal; implementation nits do not justify expanding scope.

## Completion criteria

RES-002 is complete when epsilon is supplied before construction, sparse discovery matches the
exhaustive oracle, one deterministic batch partitions long chains into displacement-bounded
clusters, every accepted edit is cell-preserving and locally certified, newly exposed positive
edges are reported without recursion, the final mesh validates strictly, ordinary computation does
not regress, the heavy benchmarks materially improve, and the old global engine is deleted.

## Deferred extensions

- Positive cell elision or cell-killing `Error`.
- Independent degree-two arc simplification.
- Optimized/new representative positions.
- Adaptive thresholds or parallel contractions.
- Post-computation threshold indexes.
- Stronger global geometric error metrics.
