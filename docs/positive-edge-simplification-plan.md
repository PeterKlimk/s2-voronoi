# Positive-threshold edge simplification plan

**Status:** revision 2; policy approval and resource limits required before implementation

**Date:** 2026-07-22

This plan defines RES-002: an explicit, consumer-selected positive edge threshold over a completed
Voronoi computation. The result is a simplified `SphericalCellMesh`, not a Voronoi diagram of the
original or surviving generators. The governing policy remains
[`output-resolution-policy.md`](output-resolution-policy.md).

This revision incorporates an independent review of the first plan. Its main correction is that
the positive simplifier should extend exact-zero output resolution, not extract reconciliation's
full merge classifier. Reconciliation contains useful distance and diameter precedents, but its
representative choice, component interaction, rejection order, provenance, and recovery semantics
are intentionally different.

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
- exact all-pairs component-diameter calculation; and
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

1. discovers all currently live threshold edges;
2. forms and classifies components and face-interaction groups;
3. commits every admissible group for the selected cell policy;
4. rebuilds the incidence view over the rewritten live cycles; and
5. repeats if at least one vertex or cell was removed.

The process terminates because every committed round strictly decreases the number of live
vertices or cells. Previously declined groups are reconsidered after progress because neighboring
contractions can change their classification. A no-progress round is the fixed point.

The final report distinguishes:

- threshold edges remaining because their components were declined;
- newly exposed edges seen after the first round; and
- exact-zero edges remaining, which must agree with final mesh validation.

### Deterministic diameter-bounded components

Candidate edges are canonicalized as `(min_id, max_id)`, sorted, and deduplicated. Union-find is
deterministic and each component's representative is its lowest live vertex id.

A threshold-connected component is admissible only when every pair of its current original member
positions is within the configured chord threshold. A component which contains a short-edge chain
but exceeds that diameter is declined whole. It is not greedily partitioned.

Increasing the threshold is therefore not guaranteed to produce a strict superset of contractions:
a new bridge can join two previously admissible components into one over-diameter component. This
non-monotonic behavior is documented and covered by a regression fixture.

For the one-shot cell-mesh operation, "original members" means the vertices present in the source
effective diagram. A persistent member ledger follows representatives across fixed-point rounds so
later unions cannot hide an earlier component's full diameter.

### Complete affected-cell incidence

The public conversion does not retain reconciliation's vertex-key provenance. It builds a
temporary vertex-to-live-cell incidence table from the current cell cycles. Every face incident to
any member of a proposed component enters its complete classification and quotient cover.

The table is rebuilt after each committed round initially. Incremental maintenance is deferred
until profiling shows that it matters. The cold path favors a simple exhaustive certificate over a
second provenance invariant.

### Cell policies and failure semantics

The conversion uses a cell-mesh-specific policy enum:

- `Preserve`: commit admissible interaction groups only when every effective cell survives.
  Positive-length cell-killing and topology-unsafe groups are declined and reported. If an exact
  stored-zero edge remains because its component cannot be contracted without killing a cell, the
  distinct strict cell-mesh conversion returns a recoverable unrepresentable-source error; it
  cannot honestly return a strict `SphericalCellMesh` containing that edge.
- `Error`: simulate the complete fixed-point request without exposing partial output. If any
  otherwise admissible requested group would kill a cell, return a recoverable error containing
  the untouched source, affected original input indices, and a failure report. If no cell is
  killed, return the same mesh `Preserve` would produce.
- `Elide`: permit cell-killing groups and apply the global face quotient. Diameter-rejected and
  locally topology-unsafe groups remain declined. If the combined quotient, degree-two
  suppression, or terminal validation fails, reject the entire conversion and return the untouched
  source plus a failure report. The first implementation does not retry subsets after a failed
  global quotient.

Safe work mentioned by an `Error` report is simulated work, not a mutation retained in the
returned source. This removes the previous ambiguity around "perform safe contractions first."

Exact-zero and positive candidates participate in the same per-round transaction. Processing
order must not make a zero component safe or unsafe relative to a positive component.

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

Degree-two suppression requires both:

- the existing exact opposite-owner rotation check; and
- a shorter-arc containment check showing that the removed vertex lies on, or within the reported
  tolerance of, the replacement minor arc rather than merely its great circle.

The report keeps cross-track and along-track/containment telemetry separate. A zero cross-track
residual alone is not accepted as an arc-deviation certificate.

### Resource policy

Fixed-point rescans and exact all-pairs diameter checks require an explicit deterministic work
budget. The public options contain a `CellSimplificationLimits` value covering at least:

- maximum fixed-point rounds;
- maximum total diameter pair comparisons; and
- maximum total live cell-index visits across rounds.

Exceeding a limit returns a recoverable resource-limit error with the untouched source and consumed
work counters. It never silently changes to representative-radius approximation, skips candidates,
or accepts an uncertified component.

Stage 0 must select and document default numeric limits using targeted candidate-heavy workloads.
Callers may raise limits explicitly. An unlimited convenience mode is not provided initially.

This numeric calibration is the only intentionally unresolved implementation value in this plan;
implementation cannot pass Stage 0 until it is pinned.

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
- fixed-point rounds attempted and committed;
- sum of per-round unique live threshold edges detected;
- unique threshold edges first exposed after round one;
- threshold-connected components and face-interaction groups considered;
- components declined for excess diameter;
- interaction groups declined for cell preservation;
- interaction groups declined by face or quotient topology checks;
- per-round candidate edges removed by representative contraction;
- additional live edges removed by cycle normalization;
- edges removed by degree-two suppression;
- threshold and exact-zero edges remaining at the fixed point;
- effective cells and original source inputs elided;
- live vertices compacted away;
- maximum component member count;
- maximum certified component diameter;
- maximum representative displacement;
- maximum degree-two cross-track residual;
- maximum degree-two along-track residual or containment margin;
- diameter pair comparisons and live cell-index visits consumed; and
- final `CellMeshValidationReport` on success.

Counts are accumulated without double-counting the same live edge identity within one round.
Because vertex ids may be compacted or retired between rounds, the report does not claim that a
cross-round edge count identifies one persistent geometric feature.

## Transaction algorithm

The initial implementation uses simple cold-path data structures:

1. Prepare a private mutable effective cell mesh and original-input provenance from the source.
2. Build vertex-to-live-cell incidence from all current face cycles.
3. Scan every unique live edge and collect those within the inclusive squared chord threshold.
4. Build minimum-id threshold components while expanding each representative through the
   persistent source-member ledger.
5. Reject components whose exact all-pairs source-member diameter exceeds the threshold or work
   budget.
6. Build transitive face-interaction groups over the complete incidence cover.
7. Simulate each group over every affected face and classify cell-killing and non-simple cycles.
8. For Preserve/Error-safe contractions, tentatively rewrite the complete cover and run the
   generalized local quotient certificate: unique faces, paired opposite edge uses, Euler delta,
   representative incidence, and single-cycle affected vertex links. Roll back a failed group.
9. Under Elide, apply locally admissible groups to one temporary global quotient, remove killed
   faces, suppress certified degree-two vertices, compact, and strictly validate. A failure rejects
   the entire conversion.
10. After any committed progress, rebuild incidence and begin the next round.
11. On a no-progress round, rescan and report every remaining threshold and exact-zero edge.
12. Compact final storage, compose provenance, run terminal validation, and return the distinct
    mesh output.

Interaction groups are processed in deterministic order by their minimum source member. Preserve
may decline one group and accept a disjoint later group. Groups which share any face are already one
transaction and cannot be partially accepted.

The engine mutates only a private conversion buffer. Any error can therefore return the untouched
source without reverse-applying partial work.

## Implementation stages

### Stage 0 — Approve policy and pin limits

Before implementation:

1. approve cold placement, chord units, fixed-point semantics, whole-chain rejection, complete
   incidence, cell-policy failure behavior, and abstract geometry wording;
2. benchmark synthetic candidate-heavy components and select numeric default work limits;
3. define stable report count names and error kinds;
4. update `output-resolution-policy.md` and the RES-002 work-log status; and
5. capture ordinary-compute and exact-elision release baselines.

No public API work starts while these decisions remain unapproved.

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
- minimum-id components and persistent source-member ledger;
- exact diameter and work accounting;
- transitive face-interaction groups;
- complete face simulation;
- generalized quotient and affected-link checks;
- rollback and fixed-point rescans; and
- complete Preserve reporting.

All work operates on private conversion buffers. Terminal validation failure is a recoverable
conversion error and a test/certificate defect, not a normal declined-component outcome.
Preserve also returns the defined unrepresentable-source error when a declined cell-killing
component leaves an exact stored-zero edge, because strict cell-mesh validation cannot accept it.

### Stage 3 — Add Error and Elide

Add the simulation report and affected-input mapping for `Error`. Then adapt the existing exact
global quotient for positive `Elide`, including fixed-point rounds, combined exact/positive
candidates, shorter-arc-certified degree-two suppression, all-or-error global validation, and
successful provenance composition.

Require positive-versus-exact parity when a fixture exposes only exact-zero candidates.

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

Fixed-point work multiplies scans by the number of committed rounds and is bounded by configured
round and cell-index-visit limits. Documentation must not summarize the general conversion as
unconditionally `O(V + E + F)`.

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
- contraction which exposes a new short edge in the next round;
- contraction which exposes a new exact-zero edge;
- no-progress fixed point with remaining declined candidates;
- persistent member ledger preventing cross-round diameter escape; and
- each work limit exactly met and exceeded by one unit.

### Quotient and policy tests

- one safe contraction and several disjoint safe interaction groups;
- multiple components interacting through one face;
- complete endpoint incidence when additional faces do not own the candidate edge;
- non-simple face, duplicate face, incidence/orientation, Euler, connectivity, and pinched-link
  rejection;
- localized quotient classification compared with exhaustive tentative apply and validation;
- triangular and multi-edge cell killing under Preserve, Error, and Elide;
- Preserve declines a positive cell-killing edge but errors recoverably on a cell-killing exact
  stored-zero edge;
- Error returns byte-equivalent source data plus affected original inputs and failure report;
- Elide global quotient failure returns the untouched source;
- agreeing and disagreeing degree-two owner rotations;
- shorter-arc containment, including a zero-cross-track point outside the minor arc;
- whole-mesh collapse and invalid replacement arc rejection;
- exact-zero and positive components in the same transaction; and
- exact-only positive engine parity with `into_elided_cell_mesh`.

### End-to-end and feature tests

- clean short-nonzero-edge diagram with no reconciliation trigger;
- existing 18-site exact-zero fixture with bracketing positive thresholds;
- preprocessing-weld classes mapped coherently when their effective cell is elided;
- candidate-free identity conversion;
- invalid-source recovery;
- embedded success and error-recovery parity;
- strict final cell-mesh validation on every success;
- semantic topology across thread counts, bin counts, default/scalar SIMD, and FMA; and
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
- deterministic resource-limit behavior before excessive memory or pair work occurs.

Use the repository's established interleaved release benchmarking rules; replace qualitative
"material" wording with numeric thresholds in the Stage 0 record before implementation begins.

## Completion criteria

RES-002 is complete when:

1. Stage 0 decisions and numeric resource/performance gates are approved and recorded;
2. positive simplification is available only through an explicit consuming cell-mesh conversion;
3. every currently live edge is reconsidered until a documented no-progress fixed point;
4. accepted components use deterministic minimum-id representatives and certified all-pairs
   diameter;
5. complete endpoint incidence and transitive face interaction make transactions independent of
   discovery order;
6. every accepted Preserve contraction passes the generalized quotient/link certificate;
7. Preserve, Error, and Elide implement their documented recoverable outcomes;
8. degree-two suppression has both owner-rotation and shorter-arc certification;
9. every success passes strict terminal validation with coherent original-input provenance;
10. reports define accepted, declined, newly exposed, remaining, elided, displacement, and resource
    units precisely;
11. ordinary computation and exact elision retain their correctness and performance contracts; and
12. public documentation states the approximate non-Voronoi, abstract-complex geometry limit.

## Deferred extensions

- an angular convenience API beyond validated chord conversion;
- pre-storage f64 collision telemetry;
- adaptive or per-region thresholds;
- greedy or optimal partitioning of over-diameter chains;
- a global nonincident spherical-arc crossing certificate;
- global Hausdorff, area, or physics-quality bounds;
- locator, Delaunay, Lloyd, area, or centroid semantics for simplified cells; and
- threshold-aware construction hints before conversion profiling justifies them.
