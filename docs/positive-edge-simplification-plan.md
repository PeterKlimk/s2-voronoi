# Positive-threshold edge simplification plan

**Status:** revision 9 implementation reopened after post-implementation review

**Date:** 2026-07-23

This plan defines RES-002: an explicit, consumer-selected positive edge threshold over a completed
Voronoi computation. The result is a simplified `SphericalCellMesh`, not a Voronoi diagram of the
original or surviving generators. The governing policy remains
[`output-resolution-policy.md`](output-resolution-policy.md).

Revisions 1 through 8 established the core safety model: cold postprocessing, mandatory exact-zero
closure, stable-id fixed-point transactions, complete topology certificates, diameter-bounded
components, and cause-aware suppression provenance. Git history retains the review-by-review
evolution; this document states only the current design.

Revision 9 audits that accumulated design for accidental complexity. It keeps the safety model but
uses one unique live-edge table instead of a second candidate-interning set, linear face-local mark
arrays instead of per-face sorting, deterministic first-failure Error semantics instead of
exhaustive simulation, stable source ids instead of rollback-sensitive dense member ordinals, and
success-only result counters instead of a fallible multi-field publication delta. It also requires
iterative bag traversal and clarifies that work-counter overflow precedes only the work it charges.

Post-implementation review reopened RES-002 on 2026-07-23. The implementation must still restrict
pending-suppression cause upgrades and current provenance recertification to affected state, enforce
stored-position cardinality for affected faces in exact transactions, preserve per-entry semantic
failure precedence while charging cell-index work, and add mixed exact/positive Elide coverage.

Implementation pins the public threshold and report shape described below. The checked default
budgets are 100,000,000 diameter pair comparisons, 100,000,000 cell-index visits, and
100,000,000 provenance member checks. The executable minor-arc predicate uses finite squared-cross
and projection floors of `1e-24` and `tau = 64 * f64::EPSILON`. These constants are part of the
documented policy surface; changes require adjacent-boundary tests and a new performance/resource
review. Ordinary computation remains the numeric performance control because the simplifier is
reachable only through an explicit consuming conversion. The accepted 500k Fibonacci,
single-threaded ordinary-path gate is at most +0.5% retired instructions and +1.0% branches against
the pre-implementation revision, using seven interleaved `perf stat` rounds. Cold conversion is
reported separately with candidate-free, sparse-candidate, and candidate-heavy work counts; its
first release values are characterization baselines rather than a promise of linear fixed-point
runtime.

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

## Complexity audit and scope guardrail

The design is larger than "raise the exact-zero threshold" for five unavoidable reasons:

- contraction can expose new short or exact-zero edges, so discovery must reach a fixed point;
- a short-edge chain needs a component-diameter check or it can erase a feature wider than the
  requested threshold;
- several components can interact through one face, so topology classification is transactional;
- a general positive contraction can pinch links or disconnect the complex even when rewritten face
  cycles look valid; and
- Elide can create degree-two boundary subdivisions whose later suppression changes an arc, so the
  requested geometric bound must follow those source vertices to their final owner.

Those are correctness or advertised-bound requirements. Revision 9 removes machinery which served
only richer diagnostics or one anticipated implementation: exhaustive failure collection, a second
candidate-key set, per-face ordered sets, fallible publication counters, and dense suppression
ordinals. The remaining bags, sinks, and current/final arc checks are localized to the Elide-capable
suppression stage; they are not on ordinary computation or the initial Preserve engine.

Stage 2 is therefore an explicit scope checkpoint, not merely an implementation milestone. Measure
the Preserve engine before committing Stage 3. If Error/Elide demand disproportionate complexity or
do not have a concrete consumer, they may remain deferred without redesigning the Preserve core.
Do not pull Stage-3 provenance machinery forward merely to make the first patch look general.

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

### Source representation preflight

Voronoi validation treats a face with fewer than three exact stored positions as representation
telemetry when its abstract topology remains valid. Generic `SphericalCellMesh` validation rejects
that face. The edge-driven RES-002 transaction handles adjacent exact-zero edges, but it does not
define a quotient for an alternating two-position cycle with no adjacent zero edge.

After materializing the preferred effective diagram, scan every live face for at least three exact
stored positions. A deficient face with no edge-driven exact closure returns a recoverable
`UnsupportedStoredDegeneracy` error for Preserve, Error, and Elide, with every affected effective
generator expanded to its original preprocessing weld class. Generalizing Elide to that geometry
is separate work; the simplifier must not reach terminal validation and misclassify it as an
unexpected internal defect.

Preflight visits cells and their index entries in ascending order, charging each entry before
examination. It returns at the first deficient face with that effective generator's complete
preprocessing weld class. If the budget expires before that face is classified, the resource-limit
error wins and `failure_phase` is `Preparation`; a classified failing face records
`SourcePreflight`. The Error contract does not require scanning later faces for more diagnostics.

## Policy decisions for approval

These are the recommended decisions. Stage 0 records approval in the governing policy and work
log before code or public API work begins.

### Cold explicit placement

Positive simplification runs only through an explicit consuming cell-mesh conversion. The first
implementation globally scans the final cell-index stream. It does not widen the construction hot
hint or thread a threshold through `VoronoiConfig`.

This preserves zero ordinary-path cost and supports a threshold chosen after computation. A sparse
construction hint remains a later evidence-gated optimization.

### Chord threshold and two explicit metrics

- The public unit is chord length on the unit sphere.
- The accepted value is finite, strictly positive, and at most `2.0`.
- Candidate discovery and component diameter use **stored chord**: subtract the raw stored f32
  coordinates after promotion to f64, accumulate squared distance, and compare
  `stored_distance_squared <= threshold_squared`.
- Positive-caused degree-two suppression uses **unit-arc chord deviation**: normalize endpoints and
  suppressed positions in f64, find the nearest point on the replacement minor arc, convert that
  angular distance to chord length, and compare it to the same requested scalar threshold.
- The report names both metrics and records the requested threshold plus the exact squared f64
  stored-chord comparison value.
- `acos(clamp(dot))` is never used for classification.

An angular constructor may convert radians to chord length later. Exact stored zero remains a
separate named operation, not a nominal threshold value. `SpherePoint` storage is only within a
norm envelope, so stored chord and unit-arc chord can differ at an inclusive boundary; Stage 0 pins
both formulas with adjacent-value tests rather than treating them as numerically interchangeable.

### Fixed-point semantics

The operation runs to a deterministic fixed point. One pass is insufficient because moving an
endpoint to a surviving representative can make an adjacent edge short or exactly zero.

Each round:

1. resolves exact stored-zero components to closure under the selected cell policy;
2. discovers positive, nonzero live edges within the requested threshold;
3. forms and classifies positive components and face-interaction groups;
4. commits the first admissible positive group in deterministic order, then abandons the remaining
   snapshot and restarts exact discovery/incidence;
5. under Elide, suppresses pending degree-two vertices to closure while recording cause and member
   provenance;
6. sends every suppression-created replacement edge back through exact/positive/antipodal
   discovery; and
7. repeats if at least one vertex, cell, or pending subdivision was removed.

Exact zero has precedence. Positive bridges cannot attach a mandatory exact-zero equivalence class
to an over-diameter positive chain and veto its contraction. The exact phase first forms minimum-id
zero classes independently of the positive threshold. Once every zero obligation is resolved, the
positive phase treats each surviving zero representative as one vertex. A zero edge exposed by a
tentative positive contraction triggers nested exact-zero closure before that positive group is
committed. Under Preserve the optional positive group is rolled back and declined if its induced
zero obligation cannot be resolved without cell loss; under Error it produces the defined
cell-elimination error; under Elide it participates in the temporary global quotient. The next
round's exact phase is a backstop, not permission to commit an unresolved zero obligation.

Cell elision may leave degree-two subdivision vertices. They remain explicit during exact and
positive contraction phases and are marked pending rather than immediately suppressed. The
intermediate topology certificate permits those marked degree-two vertices, but never an unmarked
low-incidence vertex or degree-one vertex. The suppression phase then removes pending subdivisions
deterministically. A replacement edge which is exact zero, positive within threshold, or antipodal
cannot bypass the normal policy: it restarts the corresponding discovery/transaction work before
the outer fixed point can finish.

The engine retains the source effective diagram's vertex ids, position array, and member ids across
all rounds. Retired vertices and elided faces become inactive but are not renumbered. Compaction and
full strict validation occur once, after the final no-progress probe. Intermediate certificates
permit a newly exposed zero edge only inside the tentative positive-plus-exact closure transaction
or while a suppression replacement is being routed immediately back into closure. They never
finish a productive round with one or permit a newly created antipodal edge.

The process terminates because every committed round strictly decreases the number of live
vertices or cells. Previously declined groups are reconsidered after progress because neighboring
contractions can change their classification. A no-progress round is the fixed point.

Candidate components, interaction groups, diameter results, and incidence covers belong to one
immutable discovery snapshot of the persistent working buffer. **Commit** means publishing one
complete certified transaction to that buffer. Any such topology-changing commit—including a
top-level exact quotient together with its induced exact closure, a positive group together with
all induced exact closure and face elision, or one suppression together with any induced
closure—invalidates the outer snapshot. The engine immediately abandons all remaining precomputed
work, rebuilds incidence, and restarts at exact discovery. It never attempts dynamic patching of
stale groups.

Nested work has a separate transaction-scratch level. A candidate transaction copies or journals
the complete affected state, expanding that scratch cover as induced exact closure reaches more
faces. It performs every induced exact quotient, face elision, and rediscovery on scratch and
rebuilds scratch incidence whenever an inner mutation invalidates it. Those inner mutations are
never called commits, never become separately visible in the persistent buffer, and remain
rollbackable as one unit under every policy. Preserve may discard and decline the transaction;
Error records the defined first failure; an Elide failure rejects the conversion while the
source remains untouched. Only after the entire positive-plus-induced-exact or
suppression-plus-induced-closure result passes its certificate is it published as one commit and
counted as one productive round. Work counters remain cumulative when scratch is discarded. Result
counters belong only to a successful conversion; failure reports do not pretend that private-buffer
transactions were published to the caller.

The final report distinguishes:

- threshold edges remaining because their components were declined;
- candidate-edge occurrences seen after the first round; and
- exact-zero edges remaining, which must agree with final mesh validation.

### Deterministic diameter-bounded components

Build one unique live-edge table with the incidence snapshot. Scan directed face uses in ascending
cell/cycle order and insert by canonical `(min_id, max_id)` key, retaining the two owner faces and
orientations; reject a duplicate same-direction use, a third use, or a key left with only one use.
An `FxHashMap` is sufficient because no semantic order comes from its iteration. The table supplies
face adjacency and edge pairing. Each phase filters matching keys into a candidate vector and sorts
that vector before component construction; it does not build a second interning set or retain
duplicate candidate uses. Union-find consumes sorted keys and each component's representative is
its lowest live vertex id.

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
connectivity check over live face adjacency runs before every commit. The mutation must therefore
leave one connected component as well as satisfying local quotient/link checks. A fuller global
intermediate topology check remains a debug/test oracle; connectivity itself is a required
production certificate and consumes the public cell-index budget.

Every old or new affected live face must also retain at least three distinct exact stored-position
triples after cycle normalization and induced exact closure. This is the same representation
predicate used by source preflight, not merely a three-vertex-id check. A deficient face with an
adjacent exact-zero edge keeps the scratch transaction in mandatory closure; a deficient face with
no such edge is an unsupported created degeneracy and cannot publish. For an optional positive
group, Preserve and Error decline and report the representation-unsafe occurrence. A mandatory
exact transaction returns `UnresolvedExactGroup`. Under Elide, failure inside the combined quotient
or suppression transaction rejects the conversion and returns the untouched source because the
first version does not retry subsets. Terminal validation remains a backstop, not the first place
this condition is discovered.

Face-local classification must not inherit the current exact helper's repeated `Vec::contains`
patterns. Reusable boolean mark arrays keyed by stable vertex id and dense component index, with a
face-local touched-id list used to clear only set entries, make cycle simplicity and component
interaction linear in face degree. Stored-position cardinality needs only
remember the first two distinct componentwise-equal f32 triples and stop when it finds a third;
source validation already excludes non-finite coordinates and f32 equality already equates signed
zero. These checks are deterministic `O(d)` for a face of degree `d`, with `O(V + K)` reusable marks
rather than per-face ordered sets. Canonical face signatures are still sorted when checking for
duplicate faces; that separate global check remains `O(sum_f d_f log d_f)`.

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
- `Error`: process the same deterministic sequence as Preserve on the private conversion buffer,
  but stop at the first otherwise admissible transaction which would kill a cell or the first
  unresolved exact-zero obligation. Return a recoverable error containing the untouched source,
  the failing transaction's affected original input indices, and work consumed through that
  failure. Every affected effective generator expands through preprocessing weld classes to all
  original input members. The first version deliberately does not simulate later independent
  failures merely to make the diagnostic exhaustive. If no such condition occurs, return the same
  mesh `Preserve` would produce. Positive groups which are topology- or representation-unsafe
  rather than otherwise admissible cell-killing requests are declined just as under Preserve.
- `Elide`: permit cell-killing groups and apply the global face quotient. Diameter-rejected and
  locally topology-unsafe positive groups remain declined. Exact-zero obligations are never
  declined for positive-diameter reasons; the exact phase either contracts/elides them or rejects
  the conversion. If the combined quotient, degree-two suppression, or terminal validation fails,
  reject the entire conversion and return the untouched source plus a failure report. The first
  implementation does not retry subsets after a failed global quotient.

Safe work preceding an `Error` remains confined to the private buffer and is not retained in the
returned source. If an exact group fails, positive work does not run because there is no valid
zero-free base. Deterministic first failure is the complete Error contract; exhaustive discovery of
all independent failures is unnecessary work and is not promised.

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

Every degree-two suppression first requires the existing exact opposite-owner rotation check, then
tentatively constructs the replacement endpoint pair before asking for arc geometry:

1. equal stored endpoints route the suppression plus replacement edge into mandatory exact closure;
2. exactly antipodal stored endpoints reject the Elide conversion; and
3. only a surviving nonzero, non-antipodal pair must define the replacement great circle and, when
   positive-caused, pass the unit-arc certificate.

Equality and exact antipodality are stored-coordinate predicates evaluated in that order before
normalization. Equality means all raw f32 components compare equal. Exact antipodality means every
raw component of one endpoint compares equal to the negated corresponding component of the other,
matching generic cell-mesh validation; it is not a normalized near-pi test.

Suppression records stable source-member provenance, but geometric path order is not part of the
contract. Each suppressed vertex contributes one immutable leaf containing its existing stable
source vertex id; that id also indexes the retained source-position and suppression-telemetry
arrays. A live undirected edge key `(lo, hi)` or live representative sink owns a move-only bag root
over those leaves. Bag melding, edge rekeying, and edge-to-sink transfer consume their input roots
and create or move a root in `O(1)` without traversing members. Scratch uses journaled roots or an
arena checkpoint so rollback never copies a bag and discarded scratch nodes are reclaimed together.

Each bag carries a lazy positive-cause taint. Exact rekeying retains its current taint. A positive
rekey, positive representative contraction, or suppression combining any positive-caused input sets
the result root's taint without visiting descendants. Taint is inherited while flattening. A source
member must be reachable from exactly one live edge or sink root; sharing a consumed root or finding
a duplicate member during final flattening is an invariant failure.

Cause attribution belongs to the outer scratch transaction, not whichever inner rewrite physically
removes the last edge use. A top-level positive group has `positive_origin = true`. A suppression
transaction has positive origin when its pending marker is positive or any consumed root is already
tainted. A top-level mandatory exact transaction has false origin, but retains any pre-existing root
taint. Every nested exact closure inherits its outer transaction's origin and can never downgrade it.
Any root created, rekeyed, merged, sunk, or orphaned by a positive-origin transaction is therefore
tainted even when the immediate mutation was an induced exact quotient.

This representation deliberately supports edge-key collisions: two geometrically distinct source
strands need not concatenate into one path because final certification depends only on each member,
its inherited cause, and its final owner. If contraction maps several carrying edges to the same
live key, meld their roots. No public or intermediate semantic depends on member order. Final
certification flattens each root once and sorts its member ids once for deterministic diagnostics.

Every transfer into an already occupied live edge or representative sink melds the source and
destination roots and consumes both old handles. When any contraction retires a vertex which owns a
sink, move that sink root to the surviving representative and meld it with any destination sink.
An exact contraction retains the combined taint; its endpoints have equal stored coordinates, so it
does not itself add positive cause. A positive contraction forces positive taint. These rules apply
to every retired component member, not only the chosen representative's incident edges.

Degree-two suppression consumes more than its incident edges. After endpoint classification, form
one provisional replacement bag by melding the two incident edge roots, the suppressed vertex's
existing sink root if present, its new immutable suppression-member record, and any root already on
the replacement edge. Exact suppression retains inherited taint; positive-origin suppression taints
the provisional root. If the nonzero replacement edge survives, it owns that root. If equality,
closure, face deletion, or endpoint collapse prevents that edge from surviving, the affected-root
reconciliation below moves the same root to the resulting deterministic sink. Antipodal or otherwise
invalid replacement geometry rejects scratch before ownership publishes. Thus suppressing a sink
representative never leaves its old sink attached to a retired vertex.

After every scratch topology mutation, reconcile every root whose old key, endpoint, or incident
face occurs in the expanding affected cover against the scratch live-edge set after endpoint
representative mapping. Unaffected roots retain their proven live keys and are not rescanned:

1. if the mapped endpoints are distinct and their canonical edge remains live, rekey or meld onto
   that live edge;
2. if both map to one live representative, move the root to that representative's sink; and
3. if the mapped endpoints remain distinct but the carrying edge has no live use because face
   elision or suppression removed it, move the root to the lower stable-id endpoint's sink.

If neither endpoint resolves to a live representative, reject the transaction as whole-mesh
collapse. Positive origin taints any transferred root; an exact-origin disappearance retains its
prior cause and threshold-independent acceptance telemetry.

After all affected-root reconciliation and before publication, geometrically certify every affected
positive-tainted root against its current owner. For a live edge root, charge and apply the complete
point-to-current-minor-arc predicate to every member. For a sink root, charge and apply the
point-to-current-representative predicate. This includes roots which were created, first tainted,
rekeyed to different endpoints, melded with another root, moved with a representative, or transferred
because their edge disappeared. Final fixed-point recertification remains mandatory. A current
owner-bound failure declines an optional positive group under Preserve/Error and rejects a mandatory
exact or combined Elide transaction. Untainted exact-only roots retain threshold-independent
telemetry and do not acquire a positive bound. Every ownership/cause change occurs only in scratch,
so failure rolls it back with topology. Before publication, every affected root must have exactly one
live edge or sink owner and no root may remain keyed to an absent edge.

Member leaves, bag nodes, live roots, and sinks require `O(S + E + V)` auxiliary storage, where `S`
is the number of suppressed source vertices and `S <= V_source`. Productive suppression creates one
leaf once. Ownership/cause operations never traverse prior members. A required current or final
geometric certificate may traverse a root, charging every member-distance check under the resource
policy below. Every traversal uses an explicit work stack; recursive descent is forbidden because
constant-time melding may create a bag of depth `S`. Final ownership flattening is `O(S)`, with at
most `O(S log S)` deterministic member-id sorting across owners.

Final ownership audit uses stable source ids directly rather than allocating rollback-sensitive
dense ordinals. A persistent `expected` bitmap of length `V_source` marks each vertex whose
suppression transaction committed; a separate `seen` bitmap and bag-node visitation states drive the
iterative traversal of every live edge and sink root. Reject a visiting node seen again as a cycle,
a completed node reached again as shared ownership, an unexpected or already-seen source id as
duplication, and any expected-but-unseen id as missing ownership. This audit does not depend on
incidental arena reachability, and failed scratch creates neither an expected bit nor an ordinal
gap.

Suppression has two causes:

- **Exact-caused:** the pending degree-two vertex arose solely from mandatory baseline exact-zero
  contraction/elision before optional positive work touched its incident quotient. Preserve the
  existing exact-elision acceptance rule: owner rotations must agree and the replacement great
  circle must be finite and defined. The positive threshold does not gate this suppression. Keep
  its existing acceptance-time cross-track value as telemetry, and require exact-only
  topology/provenance parity with `into_elided_cell_mesh` on the shared domain where baseline exact
  elision succeeds without suppression-created edge rediscovery.
- **Positive-caused:** the pending vertex arose from a positive contraction, a positively elided
  cell, or a bag carrying positive taint. Its final replacement arc
  must satisfy the unit-arc chord deviation bound below. Cause composes conservatively: if a
  suppression combines any positive-caused input, its replacement root is tainted; a later positive
  contraction of either endpoint taints the carried edge root; and a later positive contraction of
  a sink representative taints that sink root.

For final endpoints `a`, `b` and every positive-tainted suppressed source position `v`, promote each
stored vector `x` to f64, compute `len_sq = x dot x`, require finite `len_sq > 0.0`, and normalize as
`x / sqrt(len_sq)` in that order. Compute
`n_raw = a x b` and `cross_sq = n_raw dot n_raw`. Before division, require finite
`cross_sq > endpoint_cross_sq_floor`; equality or a smaller value is
`IllConditionedReplacementArc`. Only then compute `cross_len = sqrt(cross_sq)`,
`n = n_raw / cross_len`, and
`theta = atan2(cross_len, clamp(a dot b, -1, 1))` in that order. The earlier raw-coordinate equality
and antipodality predicates still run first.

Project with `q_raw = v - n * (n dot v)` and compute `projection_sq = q_raw dot q_raw`. If finite
`projection_sq > projection_sq_floor`, normalize it and consider `q` followed by `-q`. Equality or a
smaller value, and non-finite values, skip projection candidates and use the endpoint fallback
below. This includes the singular case where `v` is parallel to the arc-plane normal and every
point on the supporting great circle is equally distant.

For candidate `p`, compute
`raw_phi(x,y) = atan2(n dot cross(x,y), clamp(x dot y, -1, 1))`. Using
`core::f64::consts::{PI, TAU}`, map it into `[0, TAU)` by first canonicalizing either signed zero to
`+0.0`, then adding `TAU` exactly once only when the result is strictly negative. A finite candidate
lies on the closed minor arc exactly when all three inclusive tests hold for the named nonnegative
f64 tolerance `tau`:

- `phi(a,p) <= theta + tau`;
- `phi(p,b) <= theta + tau`; and
- `abs((phi(a,p) + phi(p,b)) - theta) <= tau`.

Stage 0 pins finite `0 <= tau < PI`; because conditioned `theta` is strictly below `PI`, this keeps
`theta + tau < TAU`. No comparison silently clamps an angle. If both projected candidates pass
because of tolerance, choose the one with smaller `phi(a,p)`, then `q` before `-q` on an exact tie.
If no projection lies on the arc, compute each endpoint distance as
`atan2(length(v x endpoint), clamp(v dot endpoint, -1, 1))`; choose the smaller distance, with `a`
winning an exact tie. No fallback uses `acos`.

For the selected nearest point `p`, compute
`delta = atan2(length(v x p), clamp(v dot p, -1, 1))` and unit-arc chord
`chord = 2.0 * sin(delta * 0.5)` in that order. Require `chord <= requested_threshold` without an
additional acceptance pad. Stage 0 pins both conditioning floors and `tau` with immediately adjacent
tests.

For a positive-tainted vertex-sink member `v` and final representative `r`, apply the same ordered
f64 normalization to both stored directions, compute
`delta = atan2(length(v x r), clamp(v dot r, -1, 1))`, then
`chord = 2.0 * sin(delta * 0.5)`, and require `chord <= requested_threshold`. This is the
degenerate-edge counterpart of final minor-arc recertification.

Coincident, exactly antipodal, non-finite, or insufficiently conditioned final endpoints do not
define an acceptable positive replacement arc. Coincident endpoints have already transferred their
bag root to exact closure/vertex provenance before this check. Stage 0 pins named squared-cross and
projection-conditioning floors as finite nonnegative values below `1.0`, plus every
normalization/orientation boundary with adjacent-value tests. After the outer fixed point,
recertify every positive-tainted edge-root member against its final live-edge endpoints and every
positive-tainted sink member against its final representative;
compaction subsequently renumbers those endpoints without moving them. Failure rejects the entire
Elide conversion; the first version does not retry without the originating cell-elision group.

The successful report distinguishes threshold-independent exact suppression telemetry from the
maximum final unit-arc chord deviation of finally positive-tainted suppression. It does not combine
them into one claimed positive-threshold bound.

### Resource policy

Fixed-point rescans and exact all-pairs diameter checks require an explicit deterministic work
budget. The public options contain a `CellSimplificationLimits` value covering:

- maximum total diameter pair comparisons;
- maximum total live cell-index visits across rounds; and
- maximum total provenance member-distance checks across current and final certification.

Candidate retention needs no separate public limit. The unique candidates are filtered from the
live-edge table, so `K <= E <= I/2`; their storage is already bounded by the source-derived
incidence snapshot. This removes a duplicate set/vector structure and its artificial partial-update
case without weakening the memory bound. Candidate counts and their high-water mark remain report
telemetry.

The structural progress bound is the initial number of live vertices plus cells, so a separately
configurable round cap is unnecessary. Reports still count attempted and productive rounds. A
productive round ends immediately after its one topology-changing commit and may therefore stop in
any phase. The terminal no-progress round traverses all three phases, is counted as attempted but
not productive, and charges its proof to the cell-index budget.

A live cell-index visit is charged once for every entry in a cell slice scanned by incidence
construction, source preflight, discovery, face classification, star certification, or a
production global fallback. Debug/test shadow oracles maintain private counters and never consume
public budgets or change a public outcome. Linear touched-mark and stored-cardinality work over an
already charged temporary cycle is not charged as more cell visits; quadratic membership scans are
forbidden. Duplicate-face signature sorting is bounded by the same charged incidence cover and the
source-derived storage bound. The first implementation therefore does not add a face-comparison
limit. Diameter pair comparisons are charged before evaluation.

Nested exact closure, connectivity checks, suppression-triggered rediscovery, and the terminal
probe charge the same pair and cell-index counters as their top-level equivalents.
Any nested provenance geometry check also charges the provenance-distance counter. Restarting after
a commit resets temporary candidate/group storage but never resets cumulative work budgets.

A provenance member-distance check is one point-to-current-arc or point-to-current-representative
classification for a suppressed member. Charge it before evaluating the metric. This counter covers
every pre-publication affected positive-tainted edge and sink check plus every final
positive-tainted edge/sink recertification. Lazy root melding and taint do not consume it because
they inspect no member.

Exceeding a limit returns a recoverable resource-limit error with the untouched source and consumed
work counters. It never silently changes to representative-radius approximation, skips candidates,
or accepts an uncertified component.

All cumulative work counters use checked `u64` arithmetic. Charge before performing the governed
work. Arithmetic overflow returns `CounterOverflow` before that work and before configured-limit
comparison; otherwise the first increment which would exceed a configured limit returns that
resource-limit kind. This precedence applies only to the operation being charged: an earlier
semantic failure wins because no later work charge is attempted. Limits stop immediately even in
the exact phase. Otherwise the first failure in deterministic algorithm, cell-id, edge-key, and
member-id order wins. A failure report contains the successfully consumed count before the rejected
increment and never wraps or saturates.

Successful-result counts are structurally bounded: committed transactions strictly reduce live
vertices or cells, retired vertices and suppression members are source-bounded, and elided cells are
face-bounded. Repeated candidate/decline occurrences are bounded by already charged cell-index work.
Maintain result bookkeeping in source-sized integers or derive it from checked work counters, then
convert losslessly to public `u64` fields when constructing a successful report. It is not another
work budget or error-precedence surface.

Incidence and the persistent component ledger require `O(I + V)` auxiliary storage. Suppression
bags add `O(S + E + V)` live storage with `S <= V_source`; root melding, rekeying, lazy taint, and
sink transfer are `O(1)` and do not traverse members. Each such handle operation corresponds to a
rewritten/certified edge already covered by charged cell-index work. Final flattening visits each
member once and deterministic per-owner sorting totals at most `O(S log S)`, bounded by the source
vertex count like final compaction. Any repeated geometric traversal is instead bounded by the
provenance member-distance limit. The live-edge table bounds candidate retention by source
incidence. RES-002 follows the crate's existing allocator behavior; it does not add a
crate-wide fallible-allocation seam or promise recovery from allocator exhaustion.

Stage 0 must select and document the three default numeric work limits, the endpoint-cross and
projection-conditioning floors, `tau`, checked counter/error kinds, and numeric performance gates
using private instrumentation or a throwaway prototype on targeted candidate-heavy workloads.
Callers may raise work limits explicitly. An unlimited convenience mode is not provided initially.
No production API is committed until those values are pinned.

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
error kind, and the intentionally narrow failure report defined below.

Keep `into_elided_cell_mesh()` as the exact stored-zero convenience operation. Do not redefine it
as a floating threshold of zero.

## Report definitions

Success and failure have deliberately different reports. A successful result owns result telemetry;
a failure report owns only the request, deterministic failure phase/kind, work consumed, and the
failing transaction's affected inputs when applicable. This avoids hypothetical result counters for
mutations which never reached the caller. Every count names its unit explicitly.

At minimum the successful report records:

- requested chord threshold and squared f64 comparison threshold;
- round attempts and productive rounds;
- per-phase unique candidate-edge occurrences filtered from the live-edge table;
- later-round candidate-edge occurrences, without claiming cross-round identity;
- committed transactions, excluding their inner scratch mutations;
- exact-zero and positive component occurrences contained in those committed transactions;
- positive component occurrences declined for excess diameter;
- positive interaction-group occurrences declined for cell preservation, topology, representation,
  or arc reasons;
- final remaining edges at or below threshold, split into positive-nonzero and exact-zero counts;
- effective cells and original source inputs elided;
- live source vertices retired by this conversion;
- total source-buffer vertex entries absent after final compaction, including pre-existing orphans;
- maximum component member count;
- maximum accepted component diameter and representative displacement;
- initially exact suppression occurrences and maximum acceptance-time cross-track telemetry;
- finally positive-tainted suppressed-member count and maximum final unit-arc chord deviation;
- candidate high-water mark, diameter pair comparisons, charged live cell-index visits, and charged
  provenance member-distance checks; and
- final `CellMeshValidationReport` on success.

Components and interaction groups are occurrence counts: reconsidering a declined group in a later
round increments the corresponding count again. Edges are unique only within one phase attempt.
Stable source ids make per-round diagnostics reproducible, but the public report does not promise
cross-round geometric identity. `remaining_edges_at_or_below_threshold` is the sum of its
positive-nonzero and exact-zero fields; exact-zero remaining is zero on every successful strict
output. Diameter and displacement maxima cover committed components only. Commit bookkeeping is
updated with the already-certified scratch topology as one infallible private-buffer operation.
Failed, resource-limited, or declined scratch changes none of the successful-result fields. Inner
scratch mutations are never counted as commits.

Suppression telemetry uses two intentionally non-disjoint populations. An initially exact
suppression always contributes to exact acceptance-time telemetry. If later lazy positive taint
reaches that member, it also contributes to the final positive-tainted member count and final bound.
This preserves baseline exact telemetry without allowing a later positive move to evade
recertification.

A failure report records `failure_phase`, the three successfully consumed work counters, the
candidate high-water reached before failure, the stable error kind, and affected original inputs
known at the failure. It does not expose final-edge, final-provenance, result-mutation, or
validation fields. An incomplete preflight reports `Preparation`; a completed failing scan reports
`SourcePreflight`. A resource
failure never performs extra discovery merely to enrich diagnostics beyond its budget.

## Transaction algorithm

The initial implementation uses simple cold-path data structures:

1. Prepare a private mutable effective cell mesh, stable source-id member ledger, live flags, and
   original-input provenance from the source.
2. Run stored-position preflight in ascending cell order, charging each index visit before
   examination. Return at the first face with fewer than three exact stored positions and no
   edge-driven exact closure, expanding that generator's weld class. A limit hit before classifying
   the face returns the resource error instead.
3. Build vertex-to-live-cell incidence plus the unique live-edge table from all current face cycles;
   verify paired opposite uses and derive live face adjacency from the table.
4. Filter and sort canonical exact stored-zero keys from the table. Process exact groups in
   deterministic order. Preserve or Error stops at the first unresolved exact group; there is no
   exhaustive simulation pass. If closure is possible, open transaction scratch for the first group
   and resolve every exact edge it induces, including any Elide face removal. Publish that complete
   closure as one commit, discard the outer snapshot, and restart at step 3. Exact commits use the same
   complete-star, incidence, stored-position-cardinality, representation, current-provenance, and
   production-connectivity certificate as positive commits. Continue until the exact phase is empty.
5. Filter canonical positive, nonzero keys within the inclusive squared chord threshold from the
   same already-unique edge table and sort them.
6. Build minimum-id positive components while expanding each representative through the persistent
   source-member ledger.
7. Run the all-pairs source-member diameter predicate. Charge every comparison before evaluation;
   exceeding the work budget errors, while a proven over-diameter component is declined. Maintain
   the exact maximum over the complete scan of each accepted component for report telemetry;
   rejected components may exit on their first violating pair and do not claim an exact diameter.
8. Build transitive face-interaction groups over the complete rewrite incidence cover using reusable
   component marks and touched-id clearing, never repeated linear membership scans.
9. Simulate groups in deterministic order over every affected face until one can commit. Classify
   cell-killing and non-simple cycles with the required linear mark/cardinality algorithms; declined
   groups do not invalidate the snapshot. Error stops at the first otherwise admissible cell-killing
   group.
10. Copy or journal the complete affected state into transaction scratch and tentatively rewrite
   its complete cover. Scan every newly created edge: exact-zero edges enter nested mandatory
   closure and antipodal edges reject. Each inner scratch mutation expands the cover and rebuilds
   invalidated scratch discovery/incidence, but nothing publishes yet. Once induced exact closure is
   empty, extend through scratch incidence to the complete stars of every old/new affected-face
   vertex, then certify unique faces, at least three distinct exact stored positions in every
   affected live face, paired opposite edge uses, Euler delta, every unmarked live vertex has
   incidence at least three, every marked pending subdivision has incidence exactly two, all
   affected links are single cycles, and live face adjacency remains connected. The positive group
   commits atomically only with all induced closure; Preserve rolls the whole scratch transaction
   back if closure would kill a cell, Error records its defined failure, and Elide includes it in the
   scratch quotient. Move and meld sinks owned by every retired vertex, then reconcile every affected
   bag root against the scratch live-edge set: meld occupied destinations, sink a collapsed edge at
   its live representative, and sink an edge deleted with distinct endpoints at the lower-id live
   endpoint. Apply transaction-origin taint. Charge and certify every affected positive-tainted edge
   root against its current minor arc and every affected positive-tainted sink against its current
   representative, then certify exactly one live owner for every affected root. Publish the scratch
   topology, ledger, roots, expected-source-id bits, and infallible result bookkeeping together;
   abandon the outer snapshot and restart at step 3.
11. Under Elide, a group may remove killed faces in transaction scratch and mark resulting
   degree-two subdivisions with exact/positive cause. Any combined quotient failure rejects the
   conversion rather than retrying subsets. Face removal publishes only as part of the single
   combined group commit and therefore shares its restart at step 3.
12. When exact and positive discovery make no commit, select the lowest-id pending degree-two
   vertex and open transaction scratch. Verify opposite owner rotation, tentatively construct its
   replacement endpoint pair, and classify the raw stored pair before resolving an arc: equality
   enters nested exact closure, exact componentwise antipodality rejects, and only a surviving
   nonzero pair must define a great circle. Meld both incident move-only edge roots, the suppressed
   vertex's existing sink root, the new stable-id leaf, and any pre-existing replacement-key root into
   one provisional replacement bag without visiting descendants. Apply suppression-origin taint.
   Rebuild scratch incidence, move/meld any other retired-vertex sinks, and reconcile every affected
   root with the live-edge set after each induced closure mutation.
   Before publishing, apply the same complete-star, unique-face, stored-position-cardinality,
   paired-edge, Euler, incidence/pending, link-cycle, newly-zero/antipodal, current edge/sink geometry,
   root-ownership, and global-connectivity production certificate used by quotient transactions.
   Commit the suppression, bag roots, stable expected-source-id bit, result bookkeeping, and all
   induced closure as one persistent mutation. Abandon the outer snapshot and restart at step 3. The
   new edge is thereby reconsidered by exact and positive discovery before another suppression.
13. A no-progress attempt with no pending subdivision performs the terminal discovery needed to
    prove the fixed point and records all remaining threshold edges. It counts as attempted, not
    productive.
14. Flatten every move-only root once with an explicit work stack, bag-node visitation states, and
    stable source-id expected/seen bitmaps. Reject cycles, shared nodes, unexpected/duplicate source
    ids, or any missing bit; sort source
    member ids per owner for deterministic diagnostics; resolve current endpoints for every edge root
    and the current representative for every sink; then apply threshold-independent exact
    suppression telemetry, recertify positive-tainted edge members against their final minor arc, and
    recertify positive-tainted sink members against their final representative. Any failure rejects
    the conversion.
15. Compact once, compose provenance, run full strict validation once, and return the distinct mesh
    output.

Interaction groups are processed in deterministic order by their minimum source member. Preserve
may decline groups until it finds one that commits. Groups which share any face are already one
transaction and cannot be partially accepted. A commit always restarts discovery; no later group
from the old snapshot is examined.

The engine mutates only a private conversion buffer. Any error can therefore return the untouched
source without reverse-applying partial work. An intermediate debug/test oracle runs the generic
topology checks over all live faces while permitting zero edges only inside a tentative nested
closure; final validation alone requires the complete strict representation contract.

## Implementation stages

### Stage 0 — Approve policy and pin limits

Before implementation:

1. approve cold placement, chord units, exact-zero precedence, fixed-point semantics, whole-chain
   rejection, two-level scratch/persistent transaction atomicity, restart-after-commit snapshots,
   stable ids, deferred cause-aware suppression, move-only lazy-taint provenance bags, deterministic
   sinks for deleted/collapsed edges and suppressed representatives, charged current edge/sink
   certification, complete-star/connectivity/stored-position certification, source-degeneracy
   preflight accounting, cell-policy failure behavior, success/failure report separation, and
   abstract geometry wording;
2. use a private instrumented harness or throwaway prototype to benchmark candidate-heavy
   components, high-degree linear face checks and signature sorting, degree-two arc conditioning,
   and suppression-heavy bag rekey/sink/flatten work, then pin stored-chord/unit-arc formulas,
   numeric default work limits,
   named endpoint-cross and projection-conditioning floors, `tau`, and checked counter
   types/overflow precedence;
3. define stable success/failure report fields, work-counter charge points, and error kinds;
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
- one paired live-edge table reused for adjacency and unique candidate discovery;
- unsupported stored-degeneracy preflight;
- temporary complete incidence;
- mandatory exact-zero closure before positive discovery;
- two-level scratch/persistent transaction atomicity and restart-after-every-commit snapshot
  ownership;
- minimum-id components and persistent source-member ledger;
- exact diameter and work accounting;
- transitive face-interaction groups;
- complete face simulation and complete affected-star expansion;
- deterministic linear face-local interaction/simplicity/cardinality checks;
- generalized quotient/link, low-incidence, connectivity, stored-position-cardinality, zero, and
  antipodal checks;
- rollback and fixed-point rescans; and
- complete Preserve reporting.

All work operates on private conversion buffers. Terminal validation failure is a recoverable
conversion error and a test/certificate defect, not a normal declined-component outcome.
Preserve also returns the defined unrepresentable-source error when a declined cell-killing or
otherwise unsafe source component leaves an exact stored-zero edge, because strict cell-mesh
validation cannot accept it.

### Stage 3 — Add Error and Elide

Add the deterministic first-failure report and affected-input mapping for `Error`. Then adapt the
existing exact global quotient for positive `Elide`, including fixed-point rounds, combined
exact/positive phases, stable ids through every round, pending-subdivision marking, cause-aware
suppression provenance,
replacement-edge rediscovery, move-only lazy-taint bag melding, deleted/collapsed-edge and
suppressed-representative sink transfer, charged current and final edge/sink certification,
stable-source-id bitmap ownership audit, iterative bag traversal, all-or-error global validation,
one final compaction, and successful provenance composition.

Require positive-versus-exact parity on exact-only fixtures where baseline elision needs no
suppression-created edge rediscovery, plus a separate positive-bridge fixture proving exact-zero
precedence. On an exact-only fixture where baseline elision rejects a suppression-created zero
edge, pin the new simplifier's stronger closure outcome separately rather than claiming parity.

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
the number of distinct canonical candidate edges in a phase attempt, `S` the number of suppressed
source vertices, and `m_c` the retained source-member count of component `c`.

Per round:

- incidence and the paired live-edge table are expected `O(I)`;
- exact/positive candidate filtering is `O(E)` and sorting is `O(K log K)`, with `O(K)` phase
  storage and no duplicate uses;
- sparse component building is expected `O(K alpha(K))`;
- exact diameter work is `O(sum_c m_c^2)` and explicitly budgeted;
- face interaction, simplicity, and stored-cardinality checks are `O(sum_f d_f)` over affected
  faces; duplicate-face signature sorting remains `O(sum_f d_f log d_f)`;
- provenance root melding, rekeying, lazy taint, and sink transfer are `O(1)` per rewritten carrying
  edge without member traversal;
- current and final provenance geometry work is proportional to the explicitly budgeted number of
  member-distance checks; and
- validation includes expected hash-table work plus `O(sum_f d_f log d_f)` face-signature sorting.

Across the successful conversion, each suppressed source vertex creates one stable-id leaf, so
`S <= V_source`. Iterative final provenance ownership flattening is `O(S)`, and deterministic
per-owner member sorting is at most `O(S log S)`. Failed scratch transactions may allocate
temporary root nodes, but journal or arena rollback reclaims them without visiting member records;
their handle operations correspond to already charged rewrite/certificate work. Repeated point
checks do visit members and
are governed by their own cumulative limit.

Fixed-point work multiplies scans by the number of productive rounds. Structural progress bounds
those rounds by the initial live vertices plus cells; the configured pair-comparison,
cell-index-visit, and provenance-distance limits provide the practical cutoff. Documentation must
not summarize the general conversion as unconditionally `O(V + E + F)`.

Do not replace all-pairs diameter with distance from the representative alone; that permits twice
the requested component diameter.

## Test and campaign matrix

### Threshold, component, and fixed-point tests

- distances immediately below, equal to, and above the threshold;
- stored-chord and normalized unit-arc chord fixtures which straddle the same inclusive scalar
  threshold because of the `SpherePoint` norm envelope;
- invalid zero, negative, NaN, infinite, and greater-than-diameter thresholds;
- smallest positive f32 threshold accepted with no ordinary positive candidate;
- minimum-id representative under pair, edge, face, and vertex-id permutations;
- transitive over-diameter chain rejected whole;
- non-monotonic bridge case documented and pinned;
- positive bridge to an exact-zero class cannot veto the mandatory zero phase;
- contraction which exposes a new short edge in the next round;
- contraction which exposes a new exact-zero edge and cannot commit before nested closure;
- a positive transaction with multiple induced scratch exact mutations either publishes once after
  full certification or rolls back completely, while retaining charged work counters;
- a topology-changing first group invalidates a later precomputed group and forces fresh
  incidence, diameter, and interaction discovery;
- no-progress fixed point with remaining declined candidates;
- persistent member ledger preventing cross-round diameter escape;
- pair-comparison, cell-index-visit, and provenance-member-distance limits exactly met and exceeded
  by one unit;
- the edge table accepts exactly two opposite directed uses per live key, rejects malformed
  pairing, and supplies one canonical candidate occurrence to both phases;
- high-degree interaction, cycle-simplicity, and stored-position checks scale linearly rather than
  using the current helper's quadratic repeated-membership pattern;
- each work-counter overflow returns before its governed operation, without wrapping or saturating,
  while an earlier semantic failure wins over any unattempted later charge;
- preflight cell-index limit exactly met and exceeded, with deterministic resource-versus-
  degeneracy error precedence and `failure_phase`;
- successful result counts include only fully certified committed transactions; failure reports
  contain no provisional result counters;
- the terminal no-progress probe is charged and counted as attempted but not productive; and
- debug/test shadow-oracle scans do not change public work counters or limit outcomes.

### Quotient and policy tests

- one safe contraction and several disjoint safe interaction groups;
- multiple components interacting through one face;
- complete endpoint incidence when additional faces do not own the candidate edge;
- complete neighboring-vertex stars when link-changing faces do not contain a merged member;
- non-simple face, duplicate face, incidence/orientation, Euler, connectivity, and pinched-link
  rejection;
- a rewrite or suppression which creates an alternating two-position face with no zero edge fails
  the production stored-position-cardinality certificate under each cell policy;
- a locally valid quotient which disconnects live face adjacency is rejected before commit;
- low-incidence degree-one and degree-two live vertices rejected independently of link cyclicity;
- newly created exact-zero edge resolved in the positive transaction and newly antipodal edge
  rejected;
- localized quotient classification compared with exhaustive tentative apply and validation;
- triangular and multi-edge cell killing under Preserve, Error, and Elide;
- Preserve declines a positive cell-killing edge, errors recoverably on an unresolvable source
  exact-zero edge, and rolls back a positive group whose induced zero closure would kill a cell;
- Error reports a positive group's induced cell-killing zero closure and expands every affected
  preprocessing weld class;
- Error returns the first failing exact group in deterministic order and does not simulate later
  exact or positive work on an invalid zero base;
- Error returns byte-equivalent source data plus affected original inputs and failure report;
- Elide global quotient failure returns the untouched source;
- agreeing and disagreeing degree-two owner rotations;
- point-to-minor-arc deviation below, equal to, and above the threshold, including a
  zero-cross-track point outside the minor arc;
- replacement-arc cross conditioning immediately below, equal to, and above its named floor;
- projection conditioning immediately below, equal to, and above its named floor, including a
  source position parallel to the arc-plane normal and deterministic endpoint fallback;
- arc membership immediately inside, equal to, and outside every inclusive `tau` boundary, with
  wrapped negative angles, signed zero, endpoint candidates, and both projections passing;
- exact raw componentwise antipodality is rejected before normalization, with nearby normalized
  near-pi endpoints proceeding to the ordinary conditioning rule;
- exact-caused suppression succeeds independently of a positive threshold smaller than its
  cross-track telemetry and matches exact elision on their shared success domain;
- later positive endpoint contraction upgrades carried exact suppression provenance to positive
  and certifies every member against the current and final arc;
- a tainted edge root which is created, receives a melded root, or changes endpoint pair fails or
  passes transactionally at the current point-to-arc threshold under each cell policy;
- suppression-created exact-zero, positive-nonzero, antipodal, and undefined replacement edges
  respectively restart exact closure, restart positive discovery, or reject;
- coincident suppression endpoints route to exact closure before any great-circle conditioning
  check;
- exact and positive contraction of a suppression-carrying edge transfer every bag member to the
  surviving vertex sink, with positive cause upgraded and finally bounded;
- positive contraction of a representative carrying only an exact-caused vertex sink upgrades and
  finally bounds those sink members without requiring a live carrying edge;
- exact contraction moves every retired member's sink to the surviving representative, preserves
  taint, and melds an occupied destination; positive contraction additionally taints and currently
  certifies the merged sink;
- exact- and positive-origin suppression of a vertex which already owns a sink consumes that sink
  into the provisional replacement bag, including replacement-edge survival, equality closure,
  deleted-edge fallback, destination collision, rollback, current certification, and final
  certification;
- contraction which rekeys multiple carrying edges onto one key melds every bag
  deterministically and certifies every member against the final edge;
- exact and positive face deletion which removes every use of a carrying edge while both endpoints
  survive transfers its bag to the lower-id sink, with rollback, lazy taint, current bound, and final
  recertification pinned;
- face deletion performed by induced exact closure inside a positive-origin transaction remains
  positive-caused, while top-level exact closure does not taint an untainted root;
- repeated rekey, bag melding, sink merging, and cause upgrade do not traverse existing member
  leaves; required current/final geometry traversals charge every member, use an explicit work
  stack even for a maximally skewed meld chain, and final ownership flatten uses node states plus
  stable source-id expected/seen bitmaps to detect cycles, shared roots, unexpected/duplicate ids,
  and missing members;
- a suppression transaction passes the same complete-star, paired-edge, Euler,
  stored-position-cardinality, incidence, link-cycle, current edge/sink geometry, ownership, and
  connectivity certificate as a quotient transaction;
- whole-mesh collapse and invalid replacement arc rejection;
- exact-zero closure followed by positive components interacting through the same face; and
- exact-only topology/provenance parity with `into_elided_cell_mesh` at multiple positive
  thresholds on fixtures requiring no replacement-edge rediscovery, including below the known
  suppression residual;
- exact-only suppression-created zero fixture pins the new closure result separately from current
  baseline exact-elision rejection.

### End-to-end and feature tests

- clean short-nonzero-edge diagram with no reconciliation trigger;
- alternating two-position face with no adjacent zero edge returns
  `UnsupportedStoredDegeneracy` and weld-expanded affected inputs;
- existing 18-site exact-zero fixture with bracketing positive thresholds;
- preprocessing-weld classes mapped coherently when their effective cell is elided;
- candidate-free identity conversion;
- invalid-source recovery;
- embedded success and error-recovery parity;
- strict final cell-mesh validation on every success;
- preflight, resource, exact-phase, positive-policy, recertification, and terminal-validation
  failures expose only the specified request/failure/work data and no success-only result fields;
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
- synthetic high-degree faces confirm the mandated linear mark/cardinality scaling and absence of
  quadratic face-membership helpers;
- suppression-heavy scaling confirms ownership/cause operations do not traverse existing bag
  members, geometry traversals match their charged counter, a skewed bag is traversed iteratively,
  and one-time flatten/sort is bounded;
- report counters agreeing with instrumented phase totals; and
- deterministic edge-table/candidate-order/work-limit behavior, with debug shadow oracles excluded.

Use the repository's established interleaved release benchmarking rules; replace qualitative
"material" wording with numeric thresholds in the Stage 0 record before implementation begins.

## Completion criteria

RES-002 is complete when:

1. Stage 0 decisions and numeric resource/performance gates are approved and recorded;
2. positive simplification is available only through an explicit consuming cell-mesh conversion;
3. unsupported non-edge-driven stored degeneracy is rejected during source preflight;
4. mandatory exact-zero closure precedes positive discovery in every fixed-point round;
5. stable source ids and membership are retained until one final compaction;
6. inner scratch mutations remain rollbackable, while each complete certified transaction publishes
   as one topology-changing commit which invalidates its outer discovery snapshot and restarts exact
   discovery;
7. every contraction- or suppression-created edge is reconsidered until a documented no-progress,
   no-pending-subdivision fixed point;
8. accepted positive components use deterministic minimum-id representatives and certified
   all-pairs diameter;
9. one paired live-edge table plus sorted candidate keys supplies adjacency and duplicate-free
   deterministic discovery, and
   deterministic linear face-local checks avoid quadratic membership scans;
10. complete rewrite incidence, affected vertex stars, connectivity, stored-position cardinality,
    and transitive face interaction make transactions independent of discovery order;
11. every published transaction passes quotient, link, incidence, connectivity, representation,
    current positive-tainted edge/sink geometry, root-ownership, zero, and antipodal checks;
12. Preserve, Error, and Elide implement their documented recoverable outcomes;
13. exact-caused suppression is threshold-independent and preserves exact-elision parity on the
    shared baseline success domain, while positive-caused suppression retains move-only lazy-taint
    provenance and passes final point-to-minor-arc certification;
14. colliding carrying edges and occupied sinks meld roots without member traversal; collapsed or
    deleted edges and suppressed sink representatives transfer every member to a deterministic,
    cause-upgraded, currently and finally recertified owner;
15. endpoint-cross, projection, orientation, signed-zero, tolerance, endpoint-fallback, and chord
    boundaries implement the pinned executable unit-arc formula;
16. every success passes one final strict validation with coherent original-input provenance;
17. success reports cover stored-chord and unit-arc metrics, exact and positive suppression,
    accepted/declined occurrences, final remaining edge classes, elision, displacement, and work
    units, while failure reports contain only justified request/failure/work diagnostics;
18. provenance storage/final sorting has the documented source-size bound, iterative
    stable-source-id bitmap ownership audit, and transaction-origin cause attribution, while
    repeated work is governed by pair-comparison, cell-index, and provenance-member-distance limits;
19. ordinary computation and exact elision retain their correctness and performance contracts; and
20. public documentation states the approximate non-Voronoi, abstract-complex geometry limit.

## Deferred extensions

- an angular convenience API beyond validated chord conversion;
- pre-storage f64 collision telemetry;
- adaptive or per-region thresholds;
- greedy or optimal partitioning of over-diameter chains;
- a global nonincident spherical-arc crossing certificate;
- global Hausdorff, area, or physics-quality bounds;
- locator, Delaunay, Lloyd, area, or centroid semantics for simplified cells; and
- threshold-aware construction hints before conversion profiling justifies them.
