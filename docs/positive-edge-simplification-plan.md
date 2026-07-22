# Positive-threshold edge simplification plan

**Status:** revision 3; policy approval and Stage 0 calibration required before production work

**Date:** 2026-07-22

This plan defines RES-002: an explicit, consumer-selected positive edge threshold over a completed
Voronoi computation. The result is a simplified `SphericalCellMesh`, not a Voronoi diagram of the
original or surviving generators. The governing policy remains
[`output-resolution-policy.md`](output-resolution-policy.md).

Revision 2 incorporated an independent review of the first plan. Its main correction was that
the positive simplifier should extend exact-zero output resolution, not extract reconciliation's
full merge classifier. Reconciliation contains useful distance and diameter precedents, but its
representative choice, component interaction, rejection order, provenance, and recovery semantics
are intentionally different.

Revision 3 makes exact stored-zero resolution a mandatory phase before optional positive
simplification, retains source vertex ids until one final compaction, expands affected-link checks
to complete vertex stars, and defines the remaining representation and resource boundaries.

## Goal and non-goals

The goal is an opt-in consuming conversion which:

- considers every live final edge at or below a validated positive chord threshold;
- repeatedly simplifies until no further admissible threshold contraction exists;
- prevents a chain of short edges from collapsing a feature wider than the threshold;
- applies deterministic, transactional topology checks;
- preserves, errors on, or explicitly elides cell-killing components as requested;
- returns a dense, strictly valid abstract S2 cell complex with original-input provenance; and
- reports accepted, declined, and remaining work plus bounded vertex displacement.

This is not:

- input-site preprocessing or welding;
- another reconciliation tolerance or a replacement for mismatch repair;
- an implicit stage in `compute`, `compute_with`, or `compute_with_report`;
- a promise that the result remains a Voronoi diagram;
- a global minimum-feature-size optimizer;
- a certified Hausdorff approximation; or
- a global proof that nonincident shorter great-circle arcs do not cross.

The existing exact-zero construction canonicalization and `into_elided_cell_mesh` behavior remain
unchanged and acquire no positive-threshold work.

## Correct ownership and reuse boundaries

### Output resolution is the primary base

`knn_clipping/output_resolution.rs` already owns the semantics closest to RES-002:

- minimum-id deterministic representatives;
- maximal connected edge components;
- transitive grouping of components which interact through a face;
- cyclic face rewriting and consecutive-duplicate removal;
- cell-killing and non-simple classification;
- a local quotient certificate with rollback;
- exact-zero residual discovery;
- global cell elision and degree-two suppression; and
- final compaction and strict validation.

The positive engine should generalize these operations behind threshold-parameterized internal
helpers while leaving the ordinary exact-zero call site exact-equality-specific.

The existing local quotient certificate is necessary but not sufficient unchanged. Positive
contractions move geometry and exercise arbitrary manifold edge contractions, so the generalized
certificate must also verify all affected vertex links remain single cycles. Whole-result strict
validation remains the terminal backstop. An unexpected terminal validation failure rejects the
conversion rather than being guessed back to one component.

### Reconciliation supplies precedents, not transaction semantics

`knn_clipping/edge_reconcile.rs` may share only helpers proven to have identical contracts:

- promoted-f64 stored-position squared distance;
- the early-exit all-pairs component-diameter threshold predicate; and
- possibly the face-cycle normalization predicate.

Do not share or change:

- reconciliation's union-by-rank representative selection;
- its ascending-face, partial component-rejection order;
- its persistent multi-round merge ledger;
- vertex-key-derived localization;
- mismatch-record candidate discovery;
- Hull3d seed generation; or
- repair telemetry.

If extracting a small helper perturbs ordinary-path codegen or makes ownership less clear, retain a
small duplicated predicate with differential tests. Source sharing is not itself a performance or
correctness goal.

### Cell-mesh conversion owns public preparation and provenance

`cell_mesh.rs` already owns:

- validation of the source `ComputeOutput`;
- selection and copying of the preferred effective diagram;
- composition of preprocessing weld classes back to original inputs;
- recoverable failures which retain the original successful output; and
- embedded unit-sphere wrapping.

Exact elision and positive simplification should share source preparation, provenance composition,
mesh finalization, and error ownership. They need not share one public report type.

## Policy decisions for approval

These are the recommended decisions. Stage 0 records approval in the governing policy and work
log before code or public API work begins.

### Cold explicit placement

Positive simplification runs only through an explicit consuming cell-mesh conversion. The first
implementation globally scans the final cell-index stream. It does not widen the construction hot
hint or thread a threshold through `VoronoiConfig`.

This preserves zero ordinary-path cost and supports a threshold chosen after computation. A sparse
construction hint remains a later evidence-gated optimization.

### Chord threshold and inclusive comparison

- The public unit is Euclidean chord length between stored spherical directions.
- The accepted value is finite, strictly positive, and at most `2.0`.
- Stored f32 coordinates are promoted to f64 before subtraction and accumulation.
- Candidate discovery compares `distance_squared <= threshold_squared`.
- The report records both requested chord length and the exact squared f64 comparison value.
- `acos(clamp(dot))` is never used for classification.

An angular constructor may convert radians to chord length later. Exact stored zero remains a
separate named operation, not a nominal threshold value.

### Fixed-point semantics

The operation runs to a deterministic fixed point. One pass is insufficient because moving an
endpoint to a surviving representative can make an adjacent edge short or exactly zero.

Each round:

1. resolves exact stored-zero components to closure under the selected cell policy;
2. discovers positive, nonzero live edges within the requested threshold;
3. forms and classifies positive components and face-interaction groups;
4. commits every admissible positive group;
5. rebuilds the incidence view over the rewritten live cycles; and
6. repeats if at least one vertex or cell was removed.

Exact zero has precedence. Positive bridges cannot attach a mandatory exact-zero equivalence class
to an over-diameter positive chain and veto its contraction. The exact phase first forms minimum-id
zero classes independently of the positive threshold. Once every zero obligation is resolved, the
positive phase treats each surviving zero representative as one vertex. A zero edge exposed by a
tentative positive contraction triggers nested exact-zero closure before that positive group is
committed. Under Preserve the optional positive group is rolled back and declined if its induced
zero obligation cannot be resolved without cell loss; under Error it produces the defined
cell-elimination error; under Elide it participates in the temporary global quotient. The next
round's exact phase is a backstop, not permission to commit an unresolved zero obligation.

The engine retains the source effective diagram's vertex ids, position array, and member ids across
all rounds. Retired vertices and elided faces become inactive but are not renumbered. Compaction and
full strict validation occur once, after the final no-progress probe. Intermediate certificates
permit a newly exposed zero edge only inside the tentative positive-plus-exact closure transaction;
they never commit one or permit a newly created antipodal edge.

The process terminates because every committed round strictly decreases the number of live
vertices or cells. Previously declined groups are reconsidered after progress because neighboring
contractions can change their classification. A no-progress round is the fixed point.

The final report distinguishes:

- threshold edges remaining because their components were declined;
- candidate-edge occurrences seen after the first round; and
- exact-zero edges remaining, which must agree with final mesh validation.

### Deterministic diameter-bounded components

Exact-zero and positive candidate edges are collected separately. Within each phase, edges are
canonicalized as `(min_id, max_id)`, sorted, and deduplicated. Union-find is deterministic and each
component's representative is its lowest live vertex id.

A threshold-connected component is admissible only when every pair of its current original member
positions is within the configured chord threshold. A component which contains a short-edge chain
but exceeds that diameter is declined whole. It is not greedily partitioned.

Increasing the threshold is therefore not guaranteed to produce a strict superset of contractions:
a new bridge can join two previously admissible components into one over-diameter component. This
non-monotonic behavior is documented and covered by a regression fixture.

"Original members" means stable vertex ids present in the source effective diagram. A persistent
member ledger follows representatives across exact and positive fixed-point phases so later unions
cannot hide an earlier component's full diameter. Because ids are not compacted during iteration,
the ledger never needs a renumbering translation.

### Complete affected-cell incidence

The public conversion does not retain reconciliation's vertex-key provenance. It builds a
temporary vertex-to-live-cell incidence table from the current cell cycles. Every face incident to
any member of a proposed component enters its rewrite and quotient cover.

Contracting a member also changes link edges at neighboring vertices. After tentative rewriting,
the certificate collects every vertex appearing in an old or new affected face, then uses the
incidence table to add that vertex's complete live star. Link-cycle and low-incidence checks run on
those complete stars, not merely on faces which mention a component member. A global intermediate
topology check remains a debug/test oracle.

The table is rebuilt after each committed round initially. Incremental maintenance is deferred
until profiling shows that it matters. The cold path favors a simple exhaustive certificate over a
second provenance invariant.

### Cell policies and failure semantics

The conversion uses a cell-mesh-specific policy enum:

- `Preserve`: commit admissible interaction groups only when every effective cell survives.
  Positive-length cell-killing and topology-unsafe groups are declined and reported. If any exact
  stored-zero obligation cannot be resolved for any reason, the distinct strict cell-mesh
  conversion returns a recoverable unrepresentable-source error; it cannot honestly return a
  strict `SphericalCellMesh` containing that edge.
- `Error`: simulate the complete fixed-point request without exposing partial output. If any
  otherwise admissible requested group would kill a cell, or any exact-zero obligation cannot be
  resolved, return a recoverable error containing the untouched source, affected original input
  indices, and a failure report. Every affected effective generator expands through preprocessing
  weld classes to all original input members. If no such condition occurs, return the same mesh
  `Preserve` would produce.
- `Elide`: permit cell-killing groups and apply the global face quotient. Diameter-rejected and
  locally topology-unsafe positive groups remain declined. Exact-zero obligations are never
  declined for positive-diameter reasons; the exact phase either contracts/elides them or rejects
  the conversion. If the combined quotient, degree-two suppression, or terminal validation fails,
  reject the entire conversion and return the untouched source plus a failure report. The first
  implementation does not retry subsets after a failed global quotient.

Safe work mentioned by an `Error` report is simulated work, not a mutation retained in the
returned source. This removes the previous ambiguity around "perform safe contractions first."

Exact-zero and positive candidates use consecutive phases, not one threshold component. Within
each phase, components which interact through a face remain one transaction. Exact-zero precedence
is deterministic and independent of positive candidate discovery order.

### Geometry and type contract

The first version promises exactly what is checked:

- connected, oriented, closed abstract S2 topology;
- dense, finite, on-sphere vertex storage;
- simple stored face cycles and single-cycle vertex links;
- paired, oppositely oriented, nonzero, non-antipodal edges; and
- coherent source-input provenance.

It does not certify nonincident great-circle arc crossings, spherical polygon overlap, geometric
face orientation, or a global embedding/Hausdorff bound. Before exposing RES-002, public
`SphericalCellMesh` and validator documentation must use this precise abstract-complex language
rather than unqualified "cell decomposition" wording.

Selecting an existing representative moves every retired member by no more than the certified
component diameter. This is a vertex-displacement bound, not a boundary Hausdorff bound.

Every edge created by a tentative rewrite is checked over promoted-f64 stored coordinates. A newly
created exact-zero edge enters the nested mandatory exact closure before the positive group can
commit. A newly created exactly antipodal edge rejects the responsible positive group; under a
combined Elide quotient it rejects the whole conversion because the first version does not retry
subsets. Final success contains neither class.

Degree-two suppression requires both:

- the existing exact opposite-owner rotation check; and
- a promoted-f64 point-to-minor-arc deviation check, not distance to the supporting great circle.

For endpoints `a`, `b` and removed vertex `v`, resolve the unique minor arc from `a` to `b` using
`atan2(length(cross), dot)` angles and the oriented `a x b` plane. Project `v` onto that plane. If
the conditioned projection lies within the oriented minor arc, its spherical distance is the
cross-track distance; otherwise the nearest point is the nearer endpoint. Convert that distance to
chord length and require it to be `<=` the requested simplification chord threshold. This single
point-to-arc metric covers both cross-track and outside-the-arc displacement.

Coincident, exactly antipodal, non-finite, or insufficiently conditioned replacement endpoints do
not define an acceptable replacement arc. Stage 0 pins a named squared-cross conditioning floor
with immediately adjacent numerical tests. Failure of the owner-rotation, conditioning, or
point-to-arc threshold check rejects the entire Elide conversion; the first version does not retry
without the originating cell-elision group.

### Resource policy

Fixed-point rescans and exact all-pairs diameter checks require an explicit deterministic work
budget. The public options contain a `CellSimplificationLimits` value covering at least:

- maximum candidate edges retained in one phase;
- maximum total diameter pair comparisons; and
- maximum total live cell-index visits across rounds.

The structural progress bound is the initial number of live vertices plus cells, so a separately
configurable round cap is unnecessary. Reports still count attempted and productive rounds. A
round attempt includes both phases; the terminal no-progress attempt is counted as attempted but
not productive. Work spent on its proof is charged to the candidate and cell-index budgets.

A live cell-index visit is charged once for every entry in a cell slice scanned by incidence
construction, discovery, face classification, star certification, or a global oracle. Repeated
arithmetic over data already copied into a temporary cycle is not charged again. Diameter pair
comparisons are charged before evaluation. Candidate retention is checked before push/reserve.

Exceeding a limit returns a recoverable resource-limit error with the untouched source and consumed
work counters. It never silently changes to representative-radius approximation, skips candidates,
or accepts an uncertified component.

Incidence and the persistent ledger require `O(I + V)` auxiliary storage, bounded by the source
mesh itself. All cold-path buffers use checked sizes and `try_reserve`; configured candidate-limit
exhaustion and allocation failure are distinct recoverable error kinds. The plan does not promise
an arbitrary byte-perfect memory ceiling.

Stage 0 must select and document default numeric limits, the arc-conditioning floor, and numeric
performance gates using private instrumentation or a throwaway prototype on targeted
candidate-heavy workloads. Callers may raise work limits explicitly. An unlimited convenience mode
is not provided initially. No production API is committed until those values are pinned.

## Proposed API shape

Names remain provisional until Stage 0 approval:

```rust,ignore
#[non_exhaustive]
pub struct CellSimplificationOptions {
    // Validated chord threshold, cell policy, and work limits.
}

#[non_exhaustive]
pub enum SimplificationCellPolicy {
    Preserve,
    Error,
    Elide,
}

#[non_exhaustive]
pub struct CellSimplificationLimits {
    // Private fields with documented defaults and checked builders.
}

impl CellSimplificationOptions {
    pub fn from_chord_length(chord: f32) -> Result<Self, ThresholdError>;
    pub fn with_cell_policy(self, policy: SimplificationCellPolicy) -> Self;
    pub fn with_limits(self, limits: CellSimplificationLimits) -> Self;
}

impl ComputeOutput {
    pub fn into_simplified_cell_mesh(
        self,
        options: CellSimplificationOptions,
    ) -> Result<SimplifiedCellMeshOutput, CellSimplificationError>;
}
```

`SimplifiedCellMeshOutput` contains the mesh, original `ComputeReport`, and successful
`CellSimplificationReport`. `CellSimplificationError` owns the original `ComputeOutput`, a stable
error kind, and a failure report populated through the last completed classification step.

Keep `into_elided_cell_mesh()` as the exact stored-zero convenience operation. Do not redefine it
as a floating threshold of zero.

## Report definitions

The success and failure reports share a diagnostic payload. Every count names its unit explicitly.
At minimum record:

- requested chord threshold and squared f64 comparison threshold;
- round attempts and productive rounds;
- per-phase candidate-edge occurrences, summed after per-phase deduplication;
- later-round candidate-edge occurrences, without claiming cross-round identity;
- accepted zero-component and positive-component occurrences;
- positive component occurrences declined for excess diameter;
- positive interaction-group occurrences declined for cell preservation, topology, representation,
  or arc reasons;
- final remaining positive threshold edges and the exact-zero subset;
- effective cells and original source inputs elided;
- source vertices retired and final stored vertices compacted away;
- maximum component member count;
- maximum accepted component diameter and representative displacement;
- maximum accepted degree-two point-to-minor-arc chord deviation;
- candidate high-water mark, diameter pair comparisons, and charged live cell-index visits; and
- final `CellMeshValidationReport` on success.

Components and interaction groups are occurrence counts: reconsidering a declined group in a later
round increments the corresponding count again. Edges are unique only within one phase attempt.
Stable source ids make per-round diagnostics reproducible, but the public report does not promise
cross-round geometric identity. Exact-zero remaining is explicitly a subset of final threshold
edges remaining and is zero on every successful strict output; a nonzero value can appear only in
a failure report. Diameter and displacement maxima cover accepted components only.

## Transaction algorithm

The initial implementation uses simple cold-path data structures:

1. Prepare a private mutable effective cell mesh, stable source-id member ledger, live flags, and
   original-input provenance from the source.
2. Build vertex-to-live-cell incidence from all current face cycles.
3. Collect, sort, and deduplicate exact stored-zero edges. Resolve their minimum-id components to
   closure before positive discovery. Preserve/Error fail recoverably on an unresolved zero
   obligation; Elide applies the exact global quotient without compacting ids.
4. Scan every unique live edge and collect positive, nonzero edges within the inclusive squared
   chord threshold, stopping with a resource error at the candidate limit.
5. Build minimum-id positive components while expanding each representative through the persistent
   source-member ledger.
6. Run the all-pairs source-member diameter predicate. Charge every comparison before evaluation;
   exceeding the work budget errors, while a proven over-diameter component is declined. Maintain
   the exact maximum over the complete scan of each accepted component for report telemetry;
   rejected components may exit on their first violating pair and do not claim an exact diameter.
7. Build transitive face-interaction groups over the complete rewrite incidence cover.
8. Simulate each group over every affected face and classify cell-killing and non-simple cycles.
9. Tentatively rewrite the complete cover. Extend through incidence to the complete stars of every
   old/new affected-face vertex, then certify unique faces, paired opposite edge uses, Euler delta,
   all live vertex incidence, and single-cycle affected links. Scan every newly created edge: queue
   exact-zero edges into a nested mandatory closure and reject newly antipodal edges. The positive
   group commits only with that closure; Preserve rolls it back if closure would kill a cell, Error
   returns its defined failure, and Elide includes it in the temporary quotient. Roll back any
   other failed group.
10. Under Elide, apply locally admissible groups to one temporary global quotient, remove killed
    faces, and suppress only degree-two vertices passing owner rotation, arc conditioning, and
    point-to-minor-arc deviation. Do not compact ids or run terminal strict validation yet. Any
    combined quotient failure rejects the conversion rather than retrying subsets.
11. After any committed progress, rebuild incidence and begin the next round over stable ids.
12. A no-progress attempt performs the terminal discovery needed to prove the fixed point and
    records all remaining threshold edges. It counts as attempted, not productive.
13. Compact once, compose provenance, run full strict validation once, and return the distinct mesh
    output.

Interaction groups are processed in deterministic order by their minimum source member. Preserve
may decline one group and accept a disjoint later group. Groups which share any face are already one
transaction and cannot be partially accepted.

The engine mutates only a private conversion buffer. Any error can therefore return the untouched
source without reverse-applying partial work. An intermediate debug/test oracle runs the generic
topology checks over all live faces while permitting zero edges only inside a tentative nested
closure; final validation alone requires the complete strict representation contract.

## Implementation stages

### Stage 0 — Approve policy and pin limits

Before implementation:

1. approve cold placement, chord units, exact-zero precedence, fixed-point semantics, whole-chain
   rejection, stable ids, complete-star incidence, cell-policy failure behavior, and abstract
   geometry wording;
2. use a private instrumented harness or throwaway prototype to benchmark candidate-heavy
   components and degree-two arc conditioning, then select numeric default work limits and the
   named arc-conditioning floor;
3. define stable report count names and error kinds;
4. update `output-resolution-policy.md` and the RES-002 work-log status; and
5. capture ordinary-compute and exact-elision release baselines with numeric acceptance gates.

Experimental code used for calibration is permitted and discarded or isolated. No production
module refactor or public API work starts while these decisions remain unapproved.

### Stage 1 — Refactor cell-mesh preparation only

Extract shared private helpers from `cell_mesh.rs` for:

- preferred-diagram materialization;
- effective/original-input provenance construction;
- dense cell-mesh finalization; and
- recoverable error ownership.

Keep behavior and public exact-elision types unchanged. Require exact-elision output parity,
focused tests, and no material performance regression.

### Stage 2 — Implement the Preserve fixed-point engine

Generalize output-resolution internals for cold positive use:

- global inclusive edge discovery;
- temporary complete incidence;
- mandatory exact-zero closure before positive discovery;
- minimum-id components and persistent source-member ledger;
- exact diameter and work accounting;
- transitive face-interaction groups;
- complete face simulation and complete affected-star expansion;
- generalized quotient/link, low-incidence, zero, and antipodal checks;
- rollback and fixed-point rescans; and
- complete Preserve reporting.

All work operates on private conversion buffers. Terminal validation failure is a recoverable
conversion error and a test/certificate defect, not a normal declined-component outcome.
Preserve also returns the defined unrepresentable-source error when a declined cell-killing or
otherwise unsafe source component leaves an exact stored-zero edge, because strict cell-mesh
validation cannot accept it.

### Stage 3 — Add Error and Elide

Add the simulation report and affected-input mapping for `Error`. Then adapt the existing exact
global quotient for positive `Elide`, including fixed-point rounds, combined exact/positive
phases, stable ids through every round, point-to-minor-arc-certified degree-two suppression,
all-or-error global validation, one final compaction, and successful provenance composition.

Require positive-versus-exact parity when a fixture exposes only exact-zero candidates and a
separate positive-bridge fixture proving exact-zero precedence.

### Stage 4 — Embedded parity, contract wording, and campaigns

Add the consuming `EmbeddedComputeOutput` wrapper over the unit implementation. Update README,
correctness, architecture, public cell-mesh/validator docs, output-resolution policy, environment
inventory if needed, and the work log.

Run the full correctness, determinism, feature, and performance matrices below before closing
RES-002.

### Stage 5 — Optimize only from conversion profiles

Measure discovery, incidence rebuild, diameter checks, transaction simulation, quotient work,
compaction, and validation separately. Optimize the dominant conversion phase only.

A construction-time sparse hint is eligible only if the global discovery scan is material. It must
extend the documented `t + 2r` necessary-condition proof, confirm full final chord distance, include
every post-assembly mutation footprint, and fall back globally on drift or incomplete provenance.

## Complexity and resource model

Let `I` be the total number of live cell-index entries, `E` the number of unique live edges, `K`
the number of candidate edges in a round, and `m_c` the retained source-member count of component
`c`.

Per round:

- incidence construction and edge discovery are expected `O(I)`;
- candidate sorting is `O(K log K)`;
- sparse component building is expected `O(K alpha(K))`;
- exact diameter work is `O(sum_c m_c^2)` and explicitly budgeted;
- face simulation and quotient checks are proportional to the complete affected-cell index cover;
  and
- validation includes expected hash-table work plus `O(sum_f d_f log d_f)` face-signature sorting.

Fixed-point work multiplies scans by the number of productive rounds. Structural progress bounds
those rounds by the initial live vertices plus cells; the configured candidate, pair-comparison,
and cell-index-visit limits provide the practical cutoff. Documentation must not summarize the
general conversion as unconditionally `O(V + E + F)`.

Do not replace all-pairs diameter with distance from the representative alone; that permits twice
the requested component diameter.

## Test and campaign matrix

### Threshold, component, and fixed-point tests

- distances immediately below, equal to, and above the threshold;
- invalid zero, negative, NaN, infinite, and greater-than-diameter thresholds;
- smallest positive f32 threshold accepted with no ordinary positive candidate;
- minimum-id representative under pair, edge, face, and vertex-id permutations;
- transitive over-diameter chain rejected whole;
- non-monotonic bridge case documented and pinned;
- positive bridge to an exact-zero class cannot veto the mandatory zero phase;
- contraction which exposes a new short edge in the next round;
- contraction which exposes a new exact-zero edge and cannot commit before nested closure;
- no-progress fixed point with remaining declined candidates;
- persistent member ledger preventing cross-round diameter escape;
- candidate, pair-comparison, and cell-index-visit limits exactly met and exceeded by one unit;
- the terminal no-progress probe is charged and counted as attempted but not productive; and
- checked buffer-allocation failure returns the untouched source.

### Quotient and policy tests

- one safe contraction and several disjoint safe interaction groups;
- multiple components interacting through one face;
- complete endpoint incidence when additional faces do not own the candidate edge;
- complete neighboring-vertex stars when link-changing faces do not contain a merged member;
- non-simple face, duplicate face, incidence/orientation, Euler, connectivity, and pinched-link
  rejection;
- low-incidence degree-one and degree-two live vertices rejected independently of link cyclicity;
- newly created exact-zero edge resolved in the positive transaction and newly antipodal edge
  rejected;
- localized quotient classification compared with exhaustive tentative apply and validation;
- triangular and multi-edge cell killing under Preserve, Error, and Elide;
- Preserve declines a positive cell-killing edge, errors recoverably on an unresolvable source
  exact-zero edge, and rolls back a positive group whose induced zero closure would kill a cell;
- Error reports a positive group's induced cell-killing zero closure and expands every affected
  preprocessing weld class;
- Error returns byte-equivalent source data plus affected original inputs and failure report;
- Elide global quotient failure returns the untouched source;
- agreeing and disagreeing degree-two owner rotations;
- point-to-minor-arc deviation below, equal to, and above the threshold, including a
  zero-cross-track point outside the minor arc;
- replacement-arc cross conditioning immediately below, equal to, and above its named floor;
- whole-mesh collapse and invalid replacement arc rejection;
- exact-zero closure followed by positive components interacting through the same face; and
- exact-only positive engine parity with `into_elided_cell_mesh`.

### End-to-end and feature tests

- clean short-nonzero-edge diagram with no reconciliation trigger;
- existing 18-site exact-zero fixture with bracketing positive thresholds;
- preprocessing-weld classes mapped coherently when their effective cell is elided;
- candidate-free identity conversion;
- invalid-source recovery;
- embedded success and error-recovery parity;
- strict final cell-mesh validation on every success;
- identical-source simplification agreement across thread counts and repeated runs;
- cross-bin/SIMD/FMA semantic agreement only for fixtures whose candidate distances have a proved
  margin from the threshold, with fixed-source inclusive-boundary behavior tested separately; and
- default, `--no-default-features`, `serde`, `glam`, and applicable combined feature builds.

The geometric tests establish only the documented abstract-complex, displacement, and
degree-two-arc contracts. A crossing fixture must demonstrate the public disclaimer unless a
future global embedding validator is added.

### Performance gates

Stage 0 records exact commands, workload seeds/sizes, thread counts, run counts, counter metrics,
and peak-memory method. Final acceptance requires:

- no new ordinary-compute runtime work and no statistically/materially adverse clean-path counter
  shift versus the immediate parent;
- no material exact-elision regression from shared preparation/finalization;
- separate conversion timings for candidate-free, sparse-candidate, and candidate-heavy inputs;
- report counters agreeing with instrumented phase totals; and
- deterministic candidate/work-limit behavior and checked allocation recovery.

Use the repository's established interleaved release benchmarking rules; replace qualitative
"material" wording with numeric thresholds in the Stage 0 record before implementation begins.

## Completion criteria

RES-002 is complete when:

1. Stage 0 decisions and numeric resource/performance gates are approved and recorded;
2. positive simplification is available only through an explicit consuming cell-mesh conversion;
3. mandatory exact-zero closure precedes positive discovery in every fixed-point round;
4. stable source ids and membership are retained until one final compaction;
5. every currently live edge is reconsidered until a documented no-progress fixed point;
6. accepted positive components use deterministic minimum-id representatives and certified
   all-pairs diameter;
7. complete rewrite incidence, affected vertex stars, and transitive face interaction make
   transactions independent of discovery order;
8. every accepted Preserve contraction passes quotient, link, incidence, zero, and antipodal
   checks;
9. Preserve, Error, and Elide implement their documented recoverable outcomes;
10. degree-two suppression has owner-rotation, conditioning, and point-to-minor-arc certification;
11. every success passes one final strict validation with coherent original-input provenance;
12. reports define accepted, declined, later-round, remaining, elided, displacement, and resource
    units precisely;
13. ordinary computation and exact elision retain their correctness and performance contracts; and
14. public documentation states the approximate non-Voronoi, abstract-complex geometry limit.

## Deferred extensions

- an angular convenience API beyond validated chord conversion;
- pre-storage f64 collision telemetry;
- adaptive or per-region thresholds;
- greedy or optimal partitioning of over-diameter chains;
- a global nonincident spherical-arc crossing certificate;
- global Hausdorff, area, or physics-quality bounds;
- locator, Delaunay, Lloyd, area, or centroid semantics for simplified cells; and
- threshold-aware construction hints before conversion profiling justifies them.
