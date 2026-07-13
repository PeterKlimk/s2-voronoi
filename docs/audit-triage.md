# Correctness and Safety Audit Triage

This document tracks findings from the July 2026 source audit of the spherical Voronoi pipeline.
It is a triage document, not a statement that every suspected issue is reachable from the public
API. Each item distinguishes confirmed behavior from proof gaps and contract decisions.

The audit covered:

- input canonicalization and the mathematical site set used by each backend path;
- cube-grid and packed-kNN unseen bounds;
- radius-of-security termination;
- gnomonic and spherical-fallback clipping;
- welding, live deduplication, reconciliation, and Local3d repair;
- geometric fidelity beyond abstract topology;
- the plain-`compute` validity gate;
- deserialization and unsafe parallel assembly.

The initial source audit was read-only. Resolutions implemented afterward are tracked inline below.

## Triage vocabulary

**Priority**

- **P0** — memory safety, or a successful ordinary computation returning materially corrupted
  geometry.
- **P1** — correctness contract violation, reachable bound/predicate defect, or public panic on
  accepted data.
- **P2** — dormant/feature-specific defect, proof gap, or contract mismatch that should be resolved
  after the P0/P1 items.
- **P3** — hardening, documentation precision, or additional validation coverage.

**Confidence**

- **Confirmed** — reproduced or directly falsified with a reachable construction.
- **Algorithmically confirmed** — the local operation is wrong for a constructible internal state;
  an end-to-end public-input failure has not necessarily been found.
- **Under-justified** — no counterexample was found, but the claimed guarantee does not follow from
  the checks or error analysis currently present.
- **Policy decision** — current behavior is implementable and internally intentional, but the
  promised mathematical semantics need to be chosen explicitly.

## Summary

| ID | Priority | Confidence | Finding | Immediate action |
|---|---:|---|---|---|
| AUD-001 | P0 | Resolved | Parallel vectors exposed uninitialized elements with premature `set_len` | Keep length zero through parallel initialization |
| AUD-002 | P0 | Resolved | Five ordinary sites produced a strictly valid but materially non-Voronoi diagram | Epsilon-gate inferred endpoint pairing; escalate rejected mismatches |
| AUD-003 | P1 | Resolved | Packed pre-/mid-batch termination dropped part of `unseen_bound` | Preserve `max(candidate_dot, unseen_bound)` |
| AUD-004 | P1 | Algorithmically confirmed | Fallback may classify an active constraint `Unchanged` and discard it | Retain tolerant constraints or make tolerance semantics explicit |
| AUD-005 | P2 | Algorithmically confirmed | Nonzero-epsilon membership and intersection use different boundaries | Interpolate at `d = -eps` or remove unsupported epsilon path |
| AUD-006 | P1 | Resolved | Accepted welded-twin serde payload could panic in adjacency lookup | Reject twins that do not alias canonical spans |
| AUD-007 | P2 | Resolved | Accepted empty serde diagram could panic on first locator query | Reject empty serialized diagrams |
| AUD-008 | P1 | Under-justified | Plain `compute` cheap checks are not equivalent to strict validation | Close detection gaps or run an equivalent fast strict gate |
| AUD-009 | P1 | Resolved | Reconciliation merges were not bounded to an epsilon-diameter feature | Transactionally diameter-gate components; escalate rejected chains |
| AUD-010 | P2 | Policy decision | Fast, fallback, repair, and exact-predicate paths do not share one exact site model or SoS policy | Choose and document the intended model; align paths |
| AUD-011 | P2 | Under-justified | Termination mathematics is sound, but its complete floating error envelope is empirical | Add a derived error budget and threshold-adjacent tests |
| AUD-012 | P3 | Policy decision | Welding is a strict computed-f32 threshold graph with transitive classes | Pin equality and transitive-chain semantics in docs/tests |
| AUD-013 | P2 | Policy decision | Qhull is not a robust correctness oracle and should be retired | Replace it with a robust reference and remove oracle-like uses |

## Confirmed defects

### AUD-001 — Premature `Vec::set_len` in parallel construction

- **Priority:** P0
- **Class:** memory safety
- **Confidence:** Confirmed unsafe-precondition violation
- **Status:** Resolved; parallel initialization retains vector length zero until all disjoint
  writes have joined.

Before the fix, two parallel paths set a vector's length before the newly exposed elements had been
initialized:

- cube-grid scatter in [`src/cube_grid/build.rs`](../src/cube_grid/build.rs#L164-L217);
- parallel vertex concatenation in
  [`src/live_dedup/assemble.rs`](../src/live_dedup/assemble.rs#L222-L251).

The prefix ranges appear disjoint and exhaustive on successful completion. That establishes the
intended absence of data races, but it does not satisfy `Vec::set_len`'s requirement that every
element below the new length is already initialized. `Copy` and no-drop element types do not relax
that requirement.

Other assembly scatters keep length zero while writing spare capacity and set the length only after
the parallel join; those do not share this finding.

**Implemented resolution**

Both affected scatters now write into reserved spare capacity while the vectors retain length zero.
They publish the initialized length only after the parallel join returns successfully. A panic or
unwind before that point therefore cannot expose or drop uninitialized elements.

**Acceptance criteria**

- No vector has a logical length covering uninitialized elements.
- Panic/unwind before join cannot expose or drop uninitialized `T` values.
- Existing prefix-range/disjointness assertions remain checked.
- Release tests pass with default parallelism and `--no-default-features`.
- Run Miri on any factored serial scatter logic; run an address-sanitized parallel stress build if
  supported by the toolchain.

### AUD-002 — Strictly valid but materially non-Voronoi five-site output

- **Priority:** P0
- **Class:** geometric correctness
- **Confidence:** Confirmed public-input counterexample
- **Status:** Resolved; the default path now epsilon-gates inferred 1x1 endpoint correspondence
  and routes rejected mismatches to Local3d/error.

The following ordinary random fixture is `random_sphere_points(5, 2)`:

```text
0 ( 0.5804965, -0.53599244, -0.61297300)
1 (-0.9531088, -0.29932830,  0.044567585)
2 ( 0.08629143,-0.34277410, -0.93544626)
3 ( 0.5260911, -0.65801470, -0.53874373)
4 ( 0.13459253, 0.75951900, -0.63640857)
```

Before the fix, default `compute_with_report` reported strict validity and four
`CrossBinThirdsMismatch` records, with no Local3d repair attempt. The returned geometry measures:

```text
max incident-site dot spread       0.3309645024
max shared-edge bisector residual  0.7347528929
absolute area-sum error            4.58e-7
```

The error is far outside f32 rounding or a plausible near-cocircular ambiguity band. Topology,
positive winding, area sum, and the qhull comparison triangle set do not expose it. Qhull agreement
is included only as a diagnostic observation, not as correctness evidence; see AUD-013. The
realized shared vertex/edge geometry is wrong.

The leading cause was the unconditional primary mismatch pairing in
[`src/knn_clipping/edge_reconcile.rs`](../src/knn_clipping/edge_reconcile.rs#L1163-L1205):

- with one shared endpoint, the other endpoints were unioned without a distance bound;
- with no shared endpoint, the cheaper pairing was chosen and both pairs were unioned without an
  absolute bound.

`RepairMode::Disabled` returns the same corrupted geometry, confirming that the destructive edit
occurs in reconciliation before optional Local3d repair.

**Implemented resolution**

- The five-site fixture is a normal geometric regression.
- Primary inferred endpoint unions require every paired endpoint to lie within
  `RECONCILE_DEGENERATE_LEN_EPS`; otherwise the mismatch remains visible to Local3d/error.
- Tests distinguish epsilon-close endpoint mismatch acceptance from distant mismatch rejection,
  including both repair-enabled and fail-loud behavior.
- Preserve mismatch origin through reconciliation so malformed-key/duplicate-side evidence need
  not receive the same policy as an ordinary epsilon-scale disagreement (remaining refinement).
- Consider a cheap post-reconciliation incident/bisector residual gate for edited components.

The gate-enabled campaign covered 79 isolated cases: 41 uniform cases through 4.5M points and 38
grid/mega/cocircular/cube/bimodal/Fibonacci cases. Across 17,479 mismatch records and 12,472
inferred pairings, every campaign pairing was within epsilon; the largest pairing/component
diameter was `1.712e-7`. The bad fixture's inferred distances were `1.935e-2` to `2.179e-1`.
Rejected fixture pairings repair in one Local3d round; repair-disabled computation fails loudly.

**Acceptance criteria**

- The fixture remains strictly valid.
- Incident-site and shared-edge residuals return to a derived f32-aware tolerance.
- The fix does not merely suppress the mismatch record or return the existing corrupted diagram.
- Existing large repaired fixtures remain strictly valid and geometrically close.
- Tests cover both one-shared-endpoint and disjoint one-vs-one mismatches.

### AUD-003 — Packed frontier drops the padded unseen bound

- **Priority:** P1
- **Class:** conservative-bound correctness
- **Confidence:** Confirmed reachable contract violation; no end-to-end omitted cutter yet
- **Status:** Resolved; every exact packed checkpoint now preserves both the batch-remainder and
  post-batch unseen bounds.

Packed post-batch production correctly included its geometric coverage floor and
`GRID_DOT_BOUND_PAD`. Pre-batch consumption instead assumed that `first_dot` dominated the packed
unseen set, unlike the shell path, which took a maximum. See
[`src/knn_clipping/cell_build/run/frontier.rs`](../src/knn_clipping/cell_build/run/frontier.rs#L49-L61).
The analogous assumption appeared in the packed mid-batch logic in
[`src/knn_clipping/cell_build/run.rs`](../src/knn_clipping/cell_build/run.rs#L615-L638).

A deterministic reachable dense-cell fixture produced:

```text
first_dot                     0.99999970
best un-emitted eligible dot  0.99999976
unseen_bound                  1.00000012
```

Fixture parameters: 640 normalized points around normalized `(0.31, 0.52, 0.79)`, uniform cube
jitter `1e-3`, ChaCha8 seed 101, grid resolution 8, query slot 81.

The termination guards masked this particular one-ulp ordering violation, so it did not reproduce
an omitted cutting constraint. The bound contract is nevertheless false.

**Implemented resolution**

The shared `complete_exact_bound` helper takes the conservative maximum for every source and is
used by both frontier and mid-batch termination:

```rust
batch.first_dot.max(batch.unseen_bound)
next_dot.max(batch.unseen_bound)
```

**Acceptance criteria**

- At every exact-batch frontier and mid-batch checkpoint, brute-force maximum unseen raw dot is no
  greater than the bound passed to `can_terminate`.
- Include the deterministic fixture above.
- Exercise Chunk0, tail, dense-band takeover, shell prefixes, scalar SIMD fallback, and FMA mode.

### AUD-004 — Fallback can discard a genuinely active constraint

- **Priority:** P1
- **Class:** local clipping correctness
- **Confidence:** Algorithmically confirmed

Fallback membership keeps points with normalized plane distance at least `-1e-12`, using
`FALLBACK_PLANE_TOL` from [`src/tolerances.rs`](../src/tolerances.rs#L117-L130). After clipping,
`push_constraint` may declare the polygon `Unchanged` when vertex count, edge labels, and a coarse
positional equivalence test agree; it then pops the constraint in
[`src/knn_clipping/topo2d/builder/clip.rs`](../src/knn_clipping/topo2d/builder/clip.rs#L395-L431).

For a plane `x >= 0`, a finite spherical polygon with one vertex at `x = -0.5e-12` is exactly cut by
the constraint but is wholly retained by the tolerance. The polygon remains bit-identical, is
classified `Unchanged`, and the active constraint is permanently forgotten.

Fallback never performs radius-of-security termination, which limits the immediate consequence,
but losing an accepted constraint makes final replay/extraction incomplete relative to exact
half-space clipping.

**Proposed resolution**

Choose one explicit policy:

- retain every constraint that is exact-active even when tolerance keeps all current vertices;
- define the fallback as solving tolerant halfspaces and retain enough metadata to enforce that
  tolerant problem consistently; or
- distinguish geometrically inactive, tolerance-only active, and exactly active outcomes.

**Acceptance criteria**

- Direct tests at `-tol`, `next_up(-tol)`, and `next_down(-tol)`.
- A tolerance-kept but exact-active constraint cannot silently disappear.
- Handoff/replay and final extraction enforce the same constraint set.

### AUD-005 — Nonzero-epsilon transition uses the wrong boundary

- **Priority:** P2
- **Class:** local clipping correctness; currently mostly dormant
- **Confidence:** Algorithmically confirmed

For nonzero half-plane epsilon, membership uses `d >= -eps`, but segment interpolation uses
`t = d0 / (d0 - d1)`, which targets `d = 0`. The correct interpolation for the selected tolerant
boundary is `(d0 + eps) / (d0 - d1)`.

For example, with `eps = 1`, `d0 = -0.5`, and `d1 = -2`, the intended tolerant-boundary
intersection is `t = 1/3`; the current formula produces `-1/3` and clamps to the segment start.
The emitted point is then attributed to the new plane despite not lying on the selected boundary.

Production ordinary spherical constraints currently use zero epsilon, so the strict path is not
affected. The issue applies to positive-epsilon edge-check/replay or diagnostic configurations.

**Acceptance criteria**

- Decide whether positive epsilon is supported.
- If supported, classification, interpolation, metadata, and replay use the same boundary.
- If unsupported, prevent positive epsilon from reaching the clipper and remove misleading paths.

### AUD-006 — Welded-twin serde payload can panic adjacency lookup

- **Priority:** P1
- **Class:** public robustness/panic safety
- **Confidence:** Confirmed
- **Status:** Resolved; checked deserialization rejects a welded twin whose `CellData` differs from
  its canonical cell.

Before the fix, checked deserialization validated live spans and weld-map canonicality but did not
require a welded twin's cell span to equal its canonical cell's span. See
[`src/diagram.rs`](../src/diagram.rs#L427-L489).

A payload with two generators, valid duplicated cell spans, and `weld_map = [0, 0]` was accepted.
`build_adjacency` sized neighbor storage from canonical spans but preserved the twin's different
span; `neighbors_of(1)` then sliced past the allocation in
[`src/adjacency.rs`](../src/adjacency.rs#L85-L90) and panics.

**Implemented resolution**

Reject any payload for which a twin's `CellData` differs from its canonical cell. Valid welded
round-trips are exercised through adjacency construction for every restored cell.

**Acceptance criteria**

- The minimal mismatched-span payload is rejected or safely canonicalized.
- Every accepted diagram is safe for `build_adjacency` and `neighbors_of` on every cell.

### AUD-007 — Empty serde diagram can panic locator query

- **Priority:** P2
- **Class:** public robustness/panic safety
- **Confidence:** Confirmed
- **Status:** Resolved; checked deserialization rejects an empty generator set.

An empty diagram with empty generators, vertices, cells, and indices passed checked
deserialization. `build_locator` accepted it, while the first query panicked at
[`src/locate.rs`](../src/locate.rs#L122-L124).

**Implemented resolution**

Reject empty serialized diagrams. This preserves the infallible locator API while closing the only
accepted representation with no nearest generator.

## Guarantee and policy gaps

### AUD-008 — Plain `compute` checks are weaker than strict validation

- **Priority:** P1
- **Class:** output-contract completeness
- **Confidence:** Under-justified; cheap-signal equivalence is synthetically falsified

Plain `compute` does not run full strict validation unless `VORONOI_MESH_VERIFY=1`. It rejects
post-repair singleton edges and, in most repair modes, low-incidence vertices.

The release reconciliation scan records edges used exactly once in
[`src/knn_clipping/edge_reconcile.rs`](../src/knn_clipping/edge_reconcile.rs#L621-L733).
Strict validation additionally rejects overused and same-direction pairs. Those defects can have no
singleton signal. Repair's stronger scan is only entered when seeded by a singleton or low-incidence
defect.

Additional gaps:

- `RepairMode::Disabled` returns before computing the low-incidence signal in
  [`src/knn_clipping/compute.rs`](../src/knn_clipping/compute.rs#L454-L491).
- The cheap incidence scan counts raw occurrences, while strict validation counts distinct cell
  incidence; duplicate references can mask a low-incidence vertex.
- Duplicate cells/vertices, self-loops, antipodal edges, connectivity, and Euler characteristic are
  not checked by the cheap gate.
- Some localization-completeness claims are debug assertions only.

No natural public input returning a silent-invalid topology was found in this audit. Accepted
Local3d repair is safe because it is committed only after a full strict effective-diagram gate.

**Proposed resolution**

Either derive and enforce construction invariants that exclude every omitted defect class, or run a
fast verdict-equivalent strict verifier before returning success. At minimum, repair-disabled mode
must retain the low-incidence gate.

**Acceptance criteria**

- Fault-injection tests cover same-direction, overused, duplicate-cell, duplicate-vertex,
  disconnected/Euler, self-loop, antipodal, and weld-map defects.
- For each mutation, the plain return gate and strict verifier agree.
- Run the differential for default and `RepairMode::Disabled` configurations.

### AUD-009 — Reconciliation was not bounded to epsilon-diameter features

- **Priority:** P1
- **Class:** repair semantics
- **Confidence:** Confirmed implementation behavior; policy mismatch
- **Status:** Resolved; positional components retain membership across rounds and must satisfy a
  full f64-measured epsilon-diameter bound.

Before the fix, ordinary endpoint proximity used inclusive `distance_squared <= epsilon_squared`, then closed those
pairs transitively with a disjoint-set union. A chain at `0`, `0.75 eps`, `1.5 eps`, and so on
became one identity even though the component diameter exceeded epsilon. Primary one-vs-one
mismatches were more permissive still: endpoint unions had no absolute distance bound.

The implemented operation was therefore a threshold-graph quotient plus unconditional
record-directed unions, not strictly an “epsilon-scale feature collapse.”

**Implemented resolution**

- Primary record-directed endpoint proposals are individually epsilon-bounded.
- Every round first collects proposals without mutating cells, expands them through a sparse ledger
  of all members retained from earlier rounds, and accepts a component only if every member pair is
  within epsilon. The diameter comparison is accumulated in f64 over stored f32 coordinates.
- An over-diameter component is rejected as a whole, so iteration order cannot select an arbitrary
  safe prefix. Its cell pairs explicitly seed Local3d even if no unpaired edge remains visible.
- Tests cover same-round chains, a cross-round hidden-member chain, transactional rejection, and
  escalation-seed propagation.

A post-change 40-case campaign through 500k points covered uniform, cocircular, cube, bimodal,
Fibonacci, and mega distributions: all cases succeeded, validated strictly, and left zero repair
residuals. A separate 99,846-site structured high-degree grid carried 3,612 mismatch records and
also completed strictly valid with no surviving residual or no-chain escalation.

AUD-002 supplies a concrete reason not to leave the primary path unbounded.

### AUD-010 — No single exact site model or unified SoS policy

- **Priority:** P2
- **Class:** mathematical contract
- **Confidence:** Policy decision backed by confirmed path differences

Let `C_i` be the once-rounded canonical f32 site promoted exactly to f64, and let
`U_i = normalize(C_i)`.

Current paths use different models:

- the fast gnomonic path deliberately retains `C_i` norm differences and builds unequal-norm chord
  bisectors in
  [`src/knn_clipping/topo2d/builder/projection.rs`](../src/knn_clipping/topo2d/builder/projection.rs#L122-L218);
- spherical fallback renormalizes generators and neighbors to `U_i` in the same module;
- Local3d uses robust predicates after f64 renormalization in
  [`src/knn_clipping/local_hull.rs`](../src/knn_clipping/local_hull.rs#L1-L22);
- production exact canonical escalation is disabled;
- exact-zero Local3d faces are resolved by insertion order rather than a shared symbolic policy;
- the great-circle path is an explicit realized `1e-2` joggle, not symbolic SoS;
- welding is a quotient/deletion policy, not a bounded perturbation of every labeled site.

Consequently, the full pipeline is not presently the exact Voronoi diagram of one common site set,
nor is a single global backward-error witness established.

**Decision required**

Choose the desired promise:

1. **Structural only:** valid subdivision or error, with geometry best-effort.
2. **Forced-sign correctness:** for a chosen canonical site model, every predicate outside a
   derived ambiguity band matches the exact result; exact ties follow one declared SoS policy.
3. **Global backward stability:** there exists one jointly perturbed site set inside a stated bound
   whose exact diagram is returned.

The audit recommends option 2 as a useful near-term target. Option 3 is substantially stronger and
would require a global realizability argument, not independent per-predicate tolerance.

### AUD-011 — Termination error envelope is empirical

- **Priority:** P2
- **Class:** numerical proof gap
- **Confidence:** Under-justified, not falsified for production defaults

Assuming its input `B` is a true upper bound on every unseen raw f32 dot, the radius-of-security
derivation is mathematically sound:

- strict `B < threshold` is the correct equality policy;
- unequal canonical norms add a non-negative radial term and do not invalidate the non-cut
  condition;
- the sign-dependent norm endpoint for positive/negative double-angle cosine is correct;
- the chart Gram/Gershgorin correction has the correct conservative direction.

The remaining gap is a complete forward-error derivation for:

- canonical norm endpoints;
- rounded polygon `max_r2`;
- chart Gram/stretches;
- double-angle threshold arithmetic;
- raw f32 dot evaluation; and
- the final signed-distance evaluation.

Current constants are explicitly empirical in [`src/tolerances.rs`](../src/tolerances.rs#L1-L15).

**Acceptance criteria**

- A written error budget showing each reserve and its conservative direction.
- Builder-level tests select candidates at `next_up`/`next_down` of the cached threshold.
- Cover both signs of `cos(2 theta)`, canonical norm endpoints, elongated polar charts, scalar SIMD,
  and FMA.

Probe-only negative termination-pad overrides can intentionally violate the certificate and should
remain outside the production correctness claim or be range-validated.

### AUD-012 — Welding equality and transitive-class semantics

- **Priority:** P3
- **Class:** documented policy precision
- **Confidence:** Confirmed implementation behavior

Welding compares computed f32 squared distance with computed f32 squared radius using strict `<`.
Exact computed equality is not welded. Classes are connected components of the strict-threshold
pair graph, and the minimum original index is the representative.

Therefore:

- “closer than” matches the implementation better than “within”;
- real-distance classification can move either way due to f32 rounding;
- a transitive chain can have endpoints much farther apart than the threshold;
- the representative displacement of a remote chain endpoint is not bounded by one threshold.

**Acceptance criteria**

- Pin `next_down`, equality, and `next_up` behavior around both radius and squared radius.
- Differentially compare grid and standalone detectors against the exact implemented f32 oracle.
- Document whether transitive threshold-graph quotient semantics are intentional.

### AUD-013 — Retire qhull as a correctness reference

- **Priority:** P2
- **Class:** reference implementation and feature policy
- **Confidence:** Policy decision informed by prior robustness failures

The `qhull` backend is useful for ordinary-case comparison, but it is not a source of truth for this
audit. Qhull is not robust in the extreme, nearly degenerate, or exactly degenerate regimes where
the production backend is most difficult to validate. Agreement cannot prove correctness, and a
disagreement cannot be attributed to the production backend without an independent robust witness.

This distinction matters because a non-robust reference can make a correct implementation look
wrong, bless a shared or coincident failure, or conceal that the compared programs solve different
degeneracy policies. It has produced misleading conclusions in prior work on this crate.

The current repository already describes qhull as a comparison/testing backend rather than the
primary production path. The remaining risk is treating its results as an oracle in correctness
tests, audit harnesses, or issue triage. It is also exposed as a supported Cargo feature and public
API, which gives the diagnostic backend more permanence than its trust level warrants.

**Proposed resolution**

1. Stop using qhull agreement as acceptance evidence. Existing comparisons should be labelled
   diagnostic and limited to well-conditioned fixtures.
2. Select a robust replacement capable of evaluating the canonical spherical problem with exact or
   adaptive predicates over the normalized f32 inputs.
3. Require an explicit, deterministic policy for exact coplanar/cocircular predicates. A robust
   sign without compatible SoS semantics is not a complete reference.
4. Compare combinatorial identities—hull facets, Delaunay triples, adjacency, and robust predicate
   signs—rather than relying primarily on floating vertex coordinates.
5. Retire the public qhull feature/API, or move it to clearly unsupported diagnostic tooling until
   it can be removed without misleading users.

**Replacement acceptance criteria**

- Exact or certified-adaptive predicate signs for the canonical site representation.
- Explicit handling of exact zero predicates and the crate's selected SoS policy.
- No silent topology production after a failed/uncertain predicate.
- A small-N independent brute-force reference to cross-check the replacement itself.
- Adversarial coverage for near-cocircular groups, exact cocircular cliques, great circles,
  near-coincident sites, cube seams, and large dynamic-range predicate margins.
- Differential results classify ambiguity and reference failure instead of assuming one side is
  correct.

Replacement library selection is intentionally left open. It should be evaluated on predicate and
degeneracy guarantees, not merely on API convenience or ordinary random fixtures.

## Recommended implementation sequence

1. **AUD-001:** resolved — removed unsafe-precondition violations.
2. **AUD-002/AUD-009:** resolved — added the five-site regression and bounded/escalated
   reconciliation edits.
3. **AUD-003:** resolved — preserved the complete packed unseen bound.
4. **AUD-004/AUD-005:** make fallback and epsilon clipping internally consistent.
5. **AUD-006/AUD-007:** resolved — tightened accepted serialized representations.
6. **AUD-008:** make the plain success gate match the advertised strict-validity contract.
7. **AUD-013:** remove qhull's oracle role and choose a robust reference strategy.
8. **AUD-010/AUD-011/AUD-012:** settle the mathematical policy and derive the remaining numerical
   envelope.

## Cross-cutting test backlog

- Add exhaustive small-`n` geometry tests. Existing quality fixtures around 100 sites missed the
  five-site reconciliation failure.
- After any reconciliation or Local3d edit, assess incident-site equality and shared-edge bisector
  residuals in addition to strict topology.
- Add negative-control diagrams: rotate vertices without generators, reverse one face, duplicate a
  face, duplicate a vertex reference, and join two closed sphere complexes.
- Compare semantic topology across thread counts, bin counts, default SIMD, scalar SIMD, and FMA.
- Keep welding/joggle cases in separate expected-policy buckets rather than treating them as ordinary
  geometric errors.
- Treat qhull as an ordinary-case diagnostic only, never as the deciding oracle.
- Use exact or certified-robust references only after excluding or explicitly resolving exact
  degeneracies under the same SoS policy.

## Audit validation performed

The audit ran targeted release checks including:

```bash
cargo test --release --features qhull quality::tests -- --nocapture
cargo test --release --features qhull \
  --test correctness --test delaunay --test weird_geometry -- --nocapture
cargo test --release --test edge_repair_net \
  net_in_bin_detection_and_repair -- --nocapture
cargo test --release --test escalate_local \
  local_escalation_makes_mega_strictly_valid -- --nocapture
```

All of those targeted existing tests passed. The qhull-enabled results are recorded only as
diagnostic comparisons, not correctness certification. Temporary read-only audit harnesses
reproduced AUD-002, AUD-003, AUD-006, and AUD-007; those harnesses were not retained in the
repository.

Post-resolution validation for AUD-006/AUD-007 included the complete release suite with serde
enabled. Retained regressions reject empty diagrams and mismatched welded-twin spans, while a valid
welded round-trip builds and queries adjacency for every restored cell.
