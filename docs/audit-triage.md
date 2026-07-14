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
| AUD-004 | P1 | Resolved | Fallback could classify an active constraint `Unchanged` and discard it | Retain nominally active constraints through tolerant no-ops |
| AUD-005 | P2 | Resolved | Nonzero-epsilon membership and intersection used different boundaries | Removed unsupported positive-epsilon clipping |
| AUD-006 | P1 | Resolved | Accepted welded-twin serde payload could panic in adjacency lookup | Reject twins that do not alias canonical spans |
| AUD-007 | P2 | Resolved | Accepted empty serde diagram could panic on first locator query | Reject empty serialized diagrams |
| AUD-008 | P1 | Resolved | Plain `compute` documentation exceeded its fast-path certificate | Edge-agreement construction certificate + fused Euler; strict validation optional/testing |
| AUD-009 | P1 | Resolved | Reconciliation merges were not bounded to an epsilon-diameter feature | Transactionally diameter-gate components; escalate rejected chains |
| AUD-010 | P2 | Resolved | Fast, fallback, repair, and exact-predicate paths do not share one exact site model or SoS policy | Structural production contract chosen; unified exact combinatorics deferred as an optional add-on |
| AUD-011 | P2 | Active; downstream guard derived conditionally, shell defect fixed, packed/grid lemmas open | The three-epsilon downstream reserve closes under stated IEEE/libm assumptions; packed frontier containment remains under-justified | Derive the remaining grid containment/association budget and decide the libm platform premise |
| AUD-012 | P3 | Resolved | Welding is intentionally a strict computed-f32 threshold graph with transitive classes | Retain boundary and detector-oracle tests |
| AUD-013 | P2 | Resolved | Qhull was not a robust correctness oracle | Removed feature, dependency, public API, comparisons, and oracle-like tooling |
| AUD-014 | P1 | Resolved | Local3d lost the hull-face sign and could mint the antipodal Voronoi vertex | Carry the oriented support normal through sorted-triple repair fans |
| AUD-015 | P1 | Resolved for correctness; performance policy deferred | Finite-chart no-ops were not spherical redundancy certificates | Replay unrestricted spherical constraints only after genuine chart exhaustion |
| AUD-016 | P1 | Resolved | Near-semicircle Voronoi edges conflicted with the strict representation contract | Use owner-plane conditioning; reserve SoS for exact-pi degeneracy |

## Audit closure state

There are no open P0 or P1 correctness findings in this audit. One P2 proof item remains active:

- **AUD-011 grid frontier:** derive complete cell containment and forward-map/wall association for
  the packed ring-2 certificate, including `GRID_PLANE_PAD`, `GRID_SIN_EPS`, the dense-band chord
  certificate, and the final four-epsilon export pad. The shell's demonstrated antipodal endpoint
  failure is fixed; no packed counterexample is known.
- **AUD-011 platform premise:** decide whether conventional IEEE round-to-nearest behavior plus a
  one-ulp-or-better `f64::sin_cos`/`sqrt` implementation is an accepted platform assumption. Rust
  does not itself specify a useful worst-case transcendental error bound.

The following are recorded backburner ideas, not blockers for the selected production contract:

- unified exact normalized-site combinatorics and a shared exact-zero/SoS model;
- a certified exhaustive normalized-site ownership diagnostic;
- an early total-query-work circuit breaker for pathological but successful constructions; and
- selection of an external certified-robust comparison implementation.

The cross-cutting campaigns below are continuing regression policy, not unfinished audit findings.

### Fast-path accounting at audit close

Ordinary release binaries were compared with pinned, single-threaded `perf stat` runs and
preprocessing disabled. Against pre-resolution `main` (`554aec0`), the audit branch (`4619d22`)
retired fewer instructions and branches in every paired run: `-1.655%/-1.543%` on 500k Fibonacci,
`-1.668%/-1.526%` on 500k uniform, and `-0.637%/-0.436%` on 100k clustered. Cycles remained the
secondary signal because machine noise and cache outliers were substantial.

The final near-antipodal shell fix was also isolated over 15 paired 500k uniform runs. It added
`0.0255%` retired instructions, removed `0.1025%` branches, and had unresolved cycles. Its common
state cost is one f64 inverse norm per grid cell (about 0.33 bytes per input point at ordinary
occupancy); query normalization and f64 cap evaluation occur only after shell takeover. Full strict
validation remains optional and was not included in these production-path measurements.

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
- **Status:** Resolved; the fallback tolerance is an uncertainty band around the nominal great
  circle, and a tolerant no-op no longer erases a nominally active constraint.

Fallback membership keeps points with normalized plane distance at least `-1e-12`, using
`FALLBACK_PLANE_TOL` from [`src/tolerances.rs`](../src/tolerances.rs#L117-L130). After clipping,
`push_constraint` previously declared the polygon `Unchanged` when vertex count, edge labels, and a
coarse positional equivalence test agreed; it then popped the constraint in
[`src/knn_clipping/topo2d/builder/clip.rs`](../src/knn_clipping/topo2d/builder/clip.rs#L395-L431).

For a plane `x >= 0`, a finite spherical polygon with one vertex at `x = -0.5e-12` is exactly cut by
the constraint but is wholly retained by the tolerance. The polygon remains bit-identical, is
classified `Unchanged`, and the active constraint is permanently forgotten.

Fallback never performs radius-of-security termination, which limits the immediate consequence,
but losing an accepted constraint makes final replay/extraction incomplete relative to exact
half-space clipping.

**Implemented resolution**

Fallback clipping now distinguishes three internal outcomes:

- `Redundant`: every current vertex satisfies the nominal `d >= 0` halfspace, so the constraint may
  be discarded for the convex polygon;
- `RetainedUnchanged`: tolerant classification leaves the working polygon unchanged, but at least
  one nominal margin is negative, so the constraint remains available to extraction and
  all-constraints reconstruction;
- `Changed`: the constraint changes the working polygon and is retained normally.

The returned `ClipResult` still reports whether the polygon buffer moved; the private disposition
separately records whether an unchanged constraint was retained.

**Acceptance criteria**

- Direct tests pin `-tol`, `next_up(-tol)`, and `next_down(-tol)` membership.
- A tolerance-kept but nominally active constraint is retained.
- A genuinely nominally redundant constraint is still discarded, preserving cold-path scan costs.

### AUD-005 — Nonzero-epsilon transition uses the wrong boundary

- **Priority:** P2
- **Class:** local clipping correctness; currently mostly dormant
- **Confidence:** Algorithmically confirmed
- **Status:** Resolved by retiring the legacy `p5_shadow` experiment and removing positive
  gnomonic epsilon from the clipper and edge-check transport.

Before removal, nonzero half-plane epsilon membership used `d >= -eps`, but segment interpolation used
`t = d0 / (d0 - d1)`, which targets `d = 0`. The correct interpolation for the selected tolerant
boundary is `(d0 + eps) / (d0 - d1)`.

For example, with `eps = 1`, `d0 = -0.5`, and `d1 = -2`, the intended tolerant-boundary
intersection is `t = 1/3`; the current formula produces `-1/3` and clamps to the segment start.
The emitted point is then attributed to the new plane despite not lying on the selected boundary.

Production ordinary spherical constraints already used zero epsilon, so removing the diagnostic
override and its transported metadata does not change the selected strict halfspace model.

**Implemented resolution**

- Gnomonic clipping structurally implements only `d >= 0` with intersections at `d = 0`.
- The legacy shadow module, feature, public diagnostic API, overrides, and probes were removed.
- Edge-check seeds no longer carry or replay half-plane epsilon metadata.
- Input canonicalization and Local3d's robust predicates remain production behavior and were not
  removed with the legacy experiment.

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
- **Confidence:** Construction certificate audited and stage-tested; full strict validation remains independent
- **Status:** Resolved for the chosen production contract; strict validation is optional/testing

Plain `compute` does not run full strict validation unless `VORONOI_MESH_VERIFY=1`. The original
documentation nevertheless promised that every return satisfied that stronger verifier. The
chosen production contract is narrower and graphical: exact shared-edge agreement, no surviving
known repair/low-incidence defect, safe representation, and spherical Euler characteristic. Full
connectivity and diagnostic validation remain optional and continuously exercised in testing.

Construction already performs the edge work needed for a certificate. In-bin shared edges forward
one check to the later cell; cross-bin shared edges emit one record per side into an equal-key run.
Endpoint thirds encode directed endpoint identity. The audit found one concrete multiplicity hole:
an in-bin cell could consume the same incoming check twice without recording a defect when both
endpoint comparisons happened to match.

Stronger-validator properties not adopted as unconditional production checks include connectivity,
duplicate whole faces, arbitrary nonconsecutive duplicate vertices, and antipodal geometry. They
remain useful diagnostics. No natural public input returning a silent-invalid topology was found.
Accepted Local3d repair remains safe because it is committed only after a full strict effective-
diagram gate.

**Implemented hardening and evidence**

- Low-incidence detection now runs before the repair-mode decision. `RepairMode::Disabled` retains
  the signal and plain `compute` fails rather than returning the known-invalid diagram. Default
  repair modes already paid for this scan; disabled mode now pays the same O(live cell indices)
  safety cost.
- The plain return decision is a pure test seam over accepted repair, residual-edge,
  over-diameter-escalation, low-incidence, and Euler signals.
- Repeated in-bin consumption now emits `InBinDuplicateSide`. Tests drive the real collect/resolve
  stage, including the >64-check spill tracker. Cross-bin equal-key runs already reject singleton,
  same-side, and 3+-record multiplicity. Endpoint agreement is pinned to reverse orientation.
- Reconciliation is a mutation boundary, so its defect-only localized post-pass now checks every
  touched edge for exactly two opposite uses, including self-loops, overuse, and same-direction
  pairs. Its debug oracle performs the same exact whole-diagram scan. The clean path still performs
  no global edge scan.
- The existing live-incidence traversal now also returns referenced `V` and live half-edge count
  `H`. With construction-certified edge agreement, `E = H/2`; plain return rejects odd `H` or
  `V - E + F != 2` without another traversal or allocation.
- A signal-free post-assembly fault matrix starts from a strictly-valid computed diagram and covers
  same-direction pairs, overused edges, duplicate cells, duplicate vertices, self-loops,
  disconnected/Euler defects, antipodal edges, and corrupt weld aliases. Duplicate-vertex and
  self-loop mutations trigger low incidence; overuse, duplicate-face, and disconnected-double-
  sphere mutations fail the fused Euler summary. Arbitrary post-assembly same-direction,
  antipodal, and corrupt-weld mutations still demonstrate properties supplied by construction
  rather than that scalar summary. The matrix does not claim those mutations are naturally emitted.
- `VORONOI_MESH_VERIFY=1` remains the full strict-validation testing-period gate. It is not enabled
  unconditionally because its global edge-record allocation and sort are material fast-path work.

**Acceptance criteria**

- [x] Fault-injection tests cover same-direction, overused, duplicate-cell, duplicate-vertex,
  disconnected/Euler, self-loop, antipodal, and weld-map defects.
- [x] Stage tests cover in-bin and cross-bin multiplicity plus reverse-orientation agreement.
- [x] Post-reconciliation localized and global-oracle scans use the exact same edge-agreement rule.
- [x] `RepairMode::Disabled` retains low-incidence and Euler return signals.
- [x] Full strict validation remains available as the independent testing-period differential.

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
- **Status:** Resolved at the contract level; exact unified combinatorics is backburnered

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

Three possible promises were considered:

1. **Structural only:** valid subdivision or error, with geometry best-effort.
2. **Forced-sign correctness:** for a chosen canonical site model, every predicate outside a
   derived ambiguity band matches the exact result; exact ties follow one declared SoS policy.
3. **Global backward stability:** there exists one jointly perturbed site set inside a stated bound
   whose exact diagram is returned.

**Decision**

The project chooses option 1 for its production contract: construction-certified edge agreement,
Euler validity, and measured geometric fidelity. Exact combinatorics for one unified `U_i` model is
not a core aim and is deferred as a possible future add-on. This matches the public correctness
documentation; it does not weaken the existing fail-loud topology policy.

The deferred add-on would require more than normalizing the fast-path bisectors. A credible design
would need filtered exact clipping signs, a re-derived normalized-site kNN termination certificate,
one exact-zero/SoS policy, exact handling of non-simplicial (>3-generator) vertices, and removal of
tolerance merging as an authority that can change exact combinatorics. Local3d and plane provenance
provide useful infrastructure, but repair alone cannot catch a topologically consistent wrong edge
flip. A future prototype should measure the filter hit rate and fast-path overhead before this is
reconsidered.

AUD-011 may use `U_i = normalize(f64(C_i))` as a **diagnostic reference for measuring geometric
error** without implying that the returned combinatorics are the exact diagram of that site set.

### AUD-011 — Numerical and geometric error envelopes are empirical

- **Priority:** P2
- **Class:** numerical proof and fidelity-measurement gap
- **Confidence:** Fidelity baseline established; downstream guard conditionally derived; frontier
  construction still has unproved containment/association lemmas

Assuming its input `B` is a true upper bound on every unseen raw f32 dot, the radius-of-security
derivation is mathematically sound:

- strict `B < threshold` is the correct equality policy;
- unequal canonical norms add a non-negative radial term and do not invalidate the non-cut
  condition;
- the sign-dependent norm endpoint for positive/negative double-angle cosine is correct;
- the chart Gram/Gershgorin correction has the correct conservative direction.

The downstream derivation now covers canonical norm endpoints, rounded polygon `max_r2`, chart
Gram/stretches, raw f32 dot evaluation, and the final signed-distance relationship. It is
conditional on conventional IEEE round-to-nearest behavior and a one-ulp-or-better implementation
of `f64::sin_cos`/`sqrt`; Rust does not specify a useful worst-case transcendental error bound. The
remaining grid-side proof obligations are complete cell-cap containment, forward-map/wall
association, the packed interior-plane envelope, and the dense-band chord certificate.

The frontier audit separately confirmed that shell and packed checkpoints pass the maximum of the
known batch remainder and post-batch unseen certificate. It found no reversed inequality or
`<`/`<=` defect: equality can denote a tangent cutter, so termination must remain strict. Release
contract tests passed for the default SIMD backend, scalar SIMD, and FMA arithmetic modes.

An exact zero-tolerance frontier oracle passed all twelve retained neighbor-contract scenarios under
the default, `simd_scalar`, and FMA configurations, including cube seams/corners, dense bands, and
production bin layouts. A later analytic audit nevertheless found a shell-cap counterexample those
finite scenarios missed. With an exactly represented `+X` cell center and a canonical query
proportional to `(-1, 2^-13, 0)`, the raw center dot rounds to `-1`, so the old formula derived
`sin(d) = 0`. A cell-corner point in the transverse direction exceeds the exported bound by about
`2.4e-5`, versus a four-epsilon pad of about `4.8e-7`. The lost term is `O(sin(r)*sqrt(epsilon))`,
so increasing a fixed `O(epsilon)` pad is not a sound repair.

The shell frontier now normalizes the promoted-f32 query when it builds a cold-path shell batch and
evaluates each discovered cell's center direction and spherical-cap expression in f64. This
preserves the transverse component at the near-antipodal endpoint; the focused construction is
retained as a regression. The packed cap helper only sees geographically nearby ring-2 cells when
its classification assumptions hold, so the same antipodal construction is not a demonstrated
packed failure. Its containment and wall-association proof remains open.

One repair-only implementation mismatch was found during that audit. Local3d's grid gather scored
candidates with `glam::Vec3::dot`, while the shell bound is certified for the crate's canonical
raw-f32 dot operation. Different association or FMA behavior could therefore round a candidate
across the unseen bound. The gather now uses `fp::dot3_f32`, matching the certificate it consumes.
This did not falsify the main termination theorem, but it closes an avoidable arithmetic seam in
the accepted escalation path.

Current constants are explicitly empirical in [`src/tolerances.rs`](../src/tolerances.rs#L1-L15).
That proof obligation is distinct from output fidelity: a sound termination certificate can return
a mesh whose f32 vertex positions and near-degenerate combinatorics still differ slightly from the
diagnostic normalized-site problem.

The existing internal quality report already samples ownership and measures incident-site and
shared-edge dot residuals. It should be extended or supplemented to quantify fidelity against the
diagnostic `U_i` sites in geometrically meaningful units:

- canonicalization displacement from the supplied f32 direction to `U_i`;
- vertex unit-norm error;
- incident-site equidistance residual at every or sampled Voronoi vertex;
- shared-edge cross-track angular error at endpoints and interior samples, normalizing raw dot
  residual by the bisector normal magnitude;
- ownership violation margin for samples inside each cell; and
- separate distributions for clean, reconciled, Local3d-repaired, welded, fallback, and explicitly
  perturbed outputs.

One unconditional worst-case angular bound is unlikely to be informative near coincident or
near-cocircular inputs: positional sensitivity diverges as the defining bisector/vertex system
becomes ill-conditioned. Results should therefore include conditioning buckets (at minimum site
separation or bisector-normal magnitude) and report percentiles plus maxima rather than hiding the
unstable regime in one aggregate.

**Acceptance criteria**

- A written error budget showing each reserve and its conservative direction.
- Builder-level tests select candidates at `next_up`/`next_down` of the cached threshold.
- Cover both signs of `cos(2 theta)`, canonical norm endpoints, elongated polar charts, scalar SIMD,
  and FMA.
- Define the diagnostic `U_i` computations and angular residual formulas precisely in f64.
- Run the fidelity measurements across uniform, Fibonacci, clustered, bimodal, cube/seam,
  cocircular, mega, high-degree, welded, and repaired fixtures, with conditioning buckets.
- Establish empirical baseline percentiles/maxima and retain focused regressions for any outlier;
  do not turn empirical thresholds into a mandatory production gate without separate cost and
  false-positive analysis.

Probe-only negative termination-pad overrides can intentionally violate the certificate and should
remain outside the production correctness claim or be range-validated.

**Implemented termination boundary coverage**

Builder tests now exercise the production cached threshold at the adjacent f32 values around it,
including literal equality, repeated cache use, cache invalidation after a changed clip, positive
and negative `cos(2 theta)`, and both sides of its zero transition. Deterministically searched
once-rounded canonical vectors cover observed norms below and above one and pin the
sign-dependent endpoint formula. SIMD/scalar dot-mask parity and packed frontier composition tests
also pin equality and adjacent-value behavior.

These tests establish the implemented comparison policy and arithmetic consistency; they do not
replace the missing grid containment/association proof. The downstream radius-of-security proof
does close under the explicit IEEE/library assumptions above, and the near-antipodal shell-cap
counterexample is now fixed and retained. AUD-011 remains active for the narrower grid-side lemmas
and for deciding whether the ordinary math-library assumption is an acceptable documented
platform premise.

**2026-07-14 adversarial falsification campaign**

Temporary, repository-clean probes attacked the production certificate with the actual clipping
predicate rather than a second copy of the threshold formula:

- 14,592 successfully built shell-stream cells replayed every omitted generator through the final
  Topo2D polygon: 2,107,441 omitted-generator clips, all `Unchanged`;
- candidates scanned across the cached boundary and perturbed by component f32 ulps exercised tiny
  cells, both signs of `cos(2 theta)`, both sides of its zero transition, elongated polar charts,
  and near-tangent cancellation: 375,706 candidates accepted by `can_terminate`, none cut; and
- 3,072 packed-driver cells from uniform and 75%-clustered inputs replayed all omitted generators,
  with no changed clip and no build failure.

The shell and threshold-adjacent campaigns passed under default SIMD, `simd_scalar`, and FMA. The
packed campaign exercised the production driver end-to-end, but did not separately count pre-,
mid-, and post-batch termination classes. The corpus is deterministic and finite, ran on x86_64,
and did not exhaust all representable f32 directions or canonical norm extrema. It therefore gives
strong empirical evidence, not a formal proof.

For a three-term dot, the standard weighted dot-product analysis gives `gamma_3`, not `gamma_5`,
even though the unfused expression contains three multiplications and two additions. Both the
ordinary association and the one-product-plus-two-FMA path are bounded by approximately
`1.500001 epsilon * sum(abs(g_i*h_i))`. The canonical norm envelope keeps that sum below
`(1+epsilon)^2`, leaving approximately `1.5 epsilon` of the three-epsilon downstream guard for the
f64 threshold and clipping-sign arithmetic. The chart's `1e-12` Gram inflation and eight-epsilon
angular pad dominate their derived f64 errors; the remaining tangent-case signed-distance margin
is about `0.75 epsilon`, versus a conservative evaluation error below `0.01 epsilon` at the chart
limit.

Permanent coverage now retains a reduced all-omitted-generator replay oracle for shell and packed
construction, a real-cutter adjacent-threshold test, and test-only checkpoint labels proving that
pre-, mid-, and post-batch packed termination are each exercised. The retained corpus passes under
default SIMD, `simd_scalar`, and FMA and adds no production instrumentation or runtime branch.

The shell fix changes the ill-conditioned expression rather than merely outward-rounding stored
sine/cosine independently, which can move the cap expression in the wrong direction for some
signs. It is confined to shell-frontier cell discovery: candidate dot ranking and the common packed
path are unchanged. Query normalization and the f64 cap evaluation occur only after shell takeover;
the promoted-center inverse norm is precomputed once per grid cell. At the usual occupancy the
table costs eight bytes per grid cell, about 0.33 bytes per input point. On the 200k uniform timing
fixture only 18 cells entered takeover (32 shell batches). A pinned 50-round paired comparison found
cell construction neutral at one-percent resolution: the new version's inferred change was about
`-0.7%`, with a `[-1.8%, +0.5%]` interval. An 80-round 500k comparison likewise could not resolve
the one-time grid-build change: about `-0.3%`, with a `[-2.7%, +2.1%]` interval. The downstream
derivation provides no justification for changing the three-epsilon guard.

**Implemented fidelity measurements**

Let `C_i` be the returned once-rounded canonical f32 generator promoted exactly to f64, and define
the diagnostic site `U_i = C_i / ||C_i||`. The diagnostic report now measures:

- input/canonical-site angular displacement with `atan2(||A x U_i||, A . U_i)` after normalizing
  the supplied direction `A` in f64;
- stored-vertex norm error `abs(||V|| - 1)` before diagnostic normalization;
- vertex and shared-edge cross-track angular error
  `asin(min(1, abs(P . (U_a - U_b)) / ||U_a - U_b||))`, where `P` is a normalized stored vertex
  or a normalized chord sample along the edge; and
- sampled cell ownership at one generator-biased interior point per sampled cell edge.

Cross-track values are bucketed by normalized-site chord distance at `2e-6`, `1e-5`, `1e-4`, and
`1e-3`, with maxima and percentiles retained in stable key/value output. The ownership locator uses
the production raw-f32 cube grid to propose the nearest site, then evaluates a proposed mismatch
against `U_i` in f64. It is therefore a fast candidate-assisted diagnostic, not a certified
exhaustive nearest-site oracle for the normalized-site problem; zero mismatches must be read with
that limitation.

[`scripts/fidelity_campaign.sh`](../scripts/fidelity_campaign.sh) runs each case in a fresh release
test process and keeps the full record in a line-oriented ledger. It requires strict validation and
no post-repair residual before emitting a result. Live progress is deliberately compact so the
conditioning records do not make a healthy run appear hung.

**2026-07-14 baseline**

The retained campaign ran 32/32 cases successfully: three seeds each of uniform, Fibonacci,
clustered, bimodal, cube/seam, near-cocircular, mega, hemisphere, cap, and welded input at about
100k generators, plus a 99,846-site cubed-sphere grid and a 1,000-site exact great-circle policy
fixture. It sampled 196,596 ownership points, 573,048 incident-site vertex pairs, and 963,920 edge
points; canonicalization covered all 3,100,846 supplied sites.

| Measurement | Campaign maximum | Largest per-case p99 |
|---|---:|---:|
| Canonicalization angle, ordinary/unperturbed | `4.864e-8 rad` | `4.051e-8 rad` |
| Vertex norm error | `1.518e-7` | `1.138e-7` |
| Vertex cross-track angle | `8.246e-8 rad` | `5.033e-8 rad` |
| Edge cross-track angle | `7.846e-8 rad` | `4.578e-8 rad` |
| Candidate-assisted ownership violations | `0 / 196,596` | n/a |

The exact great-circle fixture is intentionally separate: its deterministic realized perturbation
produced a `9.993e-3 rad` maximum input/output displacement, while its post-policy vertex and edge
cross-track errors remained in the same f32-scale band as ordinary cases. This is policy movement,
not accumulated clipping error.

Default welding left no measured edges below `2e-6` site chord. The next buckets contained 315
samples (`2e-6..1e-5`, maximum `3.815e-8 rad`), 30,460 samples (`1e-5..1e-4`, maximum
`5.361e-8 rad`), 345,925 samples (`1e-4..1e-3`, maximum `6.920e-8 rad`), and 587,220 samples above
`1e-3` (maximum `7.846e-8 rad`). This campaign found no close-site amplification; the closest
surviving bucket was better, not worse, than the aggregate. That is empirical evidence under the
default quotient policy, not a conditioning-independent worst-case proof.

Five runs exercised pre-repair mismatch handling (3,630 records, of which the structured grid
contributed 3,612). Reconciliation alone closed the clustered, cube, and grid cases. Two hemisphere
runs escalated to Local3d and both were accepted. The first campaign attempt exposed AUD-014 before
this clean baseline was recorded. Sixteen runs welded at least one pair, for 121 total merged sites.

**Acceptance status**

- [x] Written downstream forward-error budget for the three-epsilon guard under the stated
  IEEE/libm assumptions.
- [ ] Written grid-frontier containment/association budget for the internal pads and final
  four-epsilon export reserve.
- [ ] Accept and document the ordinary libm premise, or replace it with a controlled implementation
  whose error bound can be stated.
- [x] Threshold-neighbor tests across sign, norm endpoints, polar charts, SIMD/scalar, and FMA.
- [x] Precise diagnostic `U_i` angular formulas and conditioning buckets.
- [x] Retained multi-distribution campaign with strict-valid/no-residual prerequisites.
- [x] Empirical maxima and per-case p99 baseline, separated from explicit perturbation policy.

Deferred optional add-on: a certified exhaustive normalized-site ownership search, if that stronger
diagnostic is later worth its cost.

### AUD-012 — Welding equality and transitive-class semantics

- **Priority:** P3
- **Class:** documented policy precision
- **Confidence:** Confirmed implementation behavior
- **Status:** Resolved; the threshold-graph quotient is intentional and pinned in public docs/tests

Welding compares computed f32 squared distance with computed f32 squared radius using strict `<`.
Exact computed equality is not welded. Classes are connected components of the strict-threshold
pair graph, and the minimum original index is the representative.

Therefore:

- “closer than” matches the implementation better than “within”;
- real-distance classification can move either way due to f32 rounding;
- a transitive chain can have endpoints much farther apart than the threshold;
- the representative displacement of a remote chain endpoint is not bounded by one threshold.

**Acceptance criteria**

- [x] Pin `next_down`, equality, and `next_up` behavior around both radius and squared radius.
- [x] Differentially compare grid and standalone detectors against the exact implemented f32
  oracle.
- [x] Document that transitive threshold-graph quotient semantics are intentional.

**Decision and implemented resolution**

The existing policy is retained. Welding and reconciliation serve different purposes:
reconciliation edits realized output geometry and therefore diameter-gates positional components,
whereas preprocessing defines the effective input by quotienting every detected sub-threshold
pair. Splitting a welding component to impose a diameter bound would either leave a detected
near-coincident pair in separate classes or require an additional arbitrary partition policy,
both of which work against preprocessing's fail-avoidance role.

Both detectors now share one inlined strict squared-distance predicate. Tests pin adjacent radius
and squared-radius values, compare the grid-integrated and standalone pair sets with the exact
computed-f32 brute-force oracle, and retain a long-chain regression proving transitivity, unbounded
endpoint span, and lowest-index representation. The public `PreprocessMode` docs and correctness
contract state that this is a quotient policy rather than a bounded perturbation claim. These
changes add no production validation pass or fast-path work beyond the already-inlined comparison.

### AUD-013 — Retire qhull as a correctness reference

- **Priority:** P2
- **Class:** reference implementation and feature policy
- **Confidence:** Policy decision informed by prior robustness failures
- **Status:** Resolved; qhull removed from the crate and tooling

The former `qhull` backend was useful for ordinary-case comparison, but it was not a source of
truth for this audit. Qhull is not robust in the extreme, nearly degenerate, or exactly degenerate
regimes where the production backend is most difficult to validate. Agreement cannot prove
correctness, and a disagreement cannot be attributed to the production backend without an
independent robust witness.

This distinction matters because a non-robust reference can make a correct implementation look
wrong, bless a shared or coincident failure, or conceal that the compared programs solve different
degeneracy policies. It has produced misleading conclusions in prior work on this crate.

Before resolution, the repository described qhull as a comparison/testing backend rather than the
primary production path, but still exposed it as a supported Cargo feature and public API. That
gave the diagnostic backend more permanence than its trust level warranted and left a continuing
risk that tests or issue triage would treat it as an oracle.

**Implemented resolution**

- Removed the optional dependency, Cargo feature, implementation module, and public re-export.
- Removed qhull-only API and structure-comparison tests plus the cell-count comparison helper.
- Preserved `bench_voronoi --validate` as an intrinsic diagnostic: it now reruns the configured
  production path and reports strict subdivision validation and sampled geometric quality.
- Removed qhull from the supported feature and contributor documentation.

No external replacement was selected as part of retirement. A future reference must first match
the canonical site model and exact-zero/SoS decision from AUD-010; until then, intrinsic validation
and focused mathematical checks remain the acceptance evidence.

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

### AUD-014 — Local3d discarded the hull-face circumcenter sign

- **Priority:** P1
- **Class:** repair geometry / fail-loud robustness
- **Confidence:** Confirmed on public hemisphere inputs
- **Status:** Resolved; Local3d carries the oriented hull support normal through vertex minting

Local3d correctly built one exact-predicate normalized 3D hull, but reduced its faces to sorted
generator triples before splicing. A triple identifies which three sites meet but not which of its
two antipodal circumcenters is the hull face's Voronoi vertex. The later minting path reconstructed
the position by forcing the normal onto the same side as the generators.

That sign rule only works when the origin lies inside the local hull. For an upper-hemisphere local
gather, lower/rim hull faces have outward support normals whose dot with all three generators is
negative. The old rule flipped those vertices to the opposite pole. The repaired combinatorics
closed every residual edge, but the manufactured pole-to-pole edges made the whole-diagram strict
gate reject the repair, so default computation still failed. The defect reproduced at 1,000 and
100,000 hemisphere points with seed 1.

The repair fan now carries an optional oracle-selected mint position alongside the unchanged sorted
triple key. Local3d supplies the face's oriented outward support normal. Live triple-keyed rim
vertices still take precedence, preserving the existing splice agreement rule; only genuinely new
vertices use the position hint. Projected/probe oracles retain their previous deterministic
triple-derived fallback.

Unit tests prove that an origin-outside upper-hemisphere hull returns supporting outward normals
and that minting preserves the selected antipodal sign. An end-to-end 1,000-site regression requires
the default Local3d repair to be attempted, accepted, residual-free, and strictly valid. In the
post-fix AUD-011 campaign, the former 100,000-site failure repaired all seven detected defects and
had `6.866e-8 rad` maximum sampled edge cross-track error.

### AUD-015 — Finite-chart no-op constraints are not spherical redundancy certificates

- **Priority:** P1
- **Class:** chart horizon / recovery correctness and pathological performance
- **Confidence:** Confirmed mathematically and in the clipping/fallback data flow
- **Status:** Resolved for correctness at actual exhaustion; early circuit breakers deferred

The gnomonic builder clips a finite synthetic triangle `B` against real bisector half-planes. If
`R` is the unbounded real-plane intersection, the stored polygon is `P = B ∩ R`. An unchanged clip
proves only that `P` lies inside the candidate half-plane. While a synthetic edge remains, that does
not prove that `R`, or the full spherical cell, lies inside it.

For generator `g = (0,0,1)` and neighbor `h = (sin θ,0,cos θ)`, the pulled-back bisector is
`u <= tan(θ/2)`. As `θ` approaches pi, its boundary moves arbitrarily far toward the chart horizon
and can miss the finite synthetic triangle while remaining a real spherical boundary. At exact
antipodality the constraint is a no-op throughout the open chart but defines the chart horizon on
the sphere.

The gnomonic implementation returns immediately on `Unchanged`, without retaining the
neighbor. Fallback conversion and accepted-constraint exhaustion reconstruction both derive from
the retained, chart-changing constraint vector, so neither can recover such a plane. Losing every
synthetic reference is sufficient in exact convex geometry to make earlier no-ops retrospectively
safe: the final real intersection is then contained inside `B` and inside every earlier finite-proxy
no-op. Synthetic provenance propagation and zero-length degeneracies remain a focused regression
target for that certificate.

#### Recovery cost experiment

Release probes counted pre-bounded no-ops and cells that were still synthetic after candidate-work
budgets. Single-threaded production-stream measurements produced:

| Distribution | Size | Pre-bounded no-ops | Cells synthetic at 128 | Existing exhaustion recovery |
|---|---:|---:|---:|---:|
| uniform, three seeds | 500k | 80,096–80,885 | 0 | 0 |
| clustered, three seeds | 100k | 362,905–427,796 | 726–850 | 0 |
| bimodal, three seeds | 100k | 23,207–23,746 | 15–19 | 0 |
| hemisphere, three seeds | 50k | 270,890–294,851 | 29–38 | 5 |
| perturbed latitude ring, three seeds | 5k | 509,206–536,109 | 114–122 | 114–122 |

On representative seed 42, work performed after crossing 128 candidates while still synthetic was
zero for 100k uniform, 307,138 candidates for 100k clustered, 6,552 for 100k bimodal, 1,337,330 for
50k hemisphere, and 491,001 for the 5k ring. Those last two are 23% and 31% of their entire current
candidate work. This is the work an early handoff has an opportunity to avoid; the actual saving
depends on how soon the unrestricted spherical rebuild certifies and must be measured once that
path exists.

Lazy retention of every pre-bounded no-op was also A/B tested in one identical binary. On 200k
uniform sites it stored 32,048 ids and added 341,287 retired instructions (`0.0102%`); elapsed cycles
were below the measurement noise. The writes themselves are cheap. However, about 10% of normal
uniform cells performed at least one ultimately unnecessary store, and uncapped horizon cases grew
a per-worker vector toward `O(n)`. Once an early work budget is present, pure recalculation repeats
at most roughly the budget-sized prefix per handed-off cell, while retaining changed real
constraints. At a budget of 128, that idealized duplicate prefix is at most 1.6% of current candidate
work in every measured regime (a batch-aligned implementation may overshoot by one batch). This
bounds duplicated clipping of the prefix, not the cold query's total traversal or certification
cost.

The cost measurements originally motivated a possible early handoff. We are deliberately not
shipping that policy yet: candidate count is not a geometric failure certificate, and the 128
candidate experiment produced avoidable handoffs in clustered, bimodal, and hemisphere regimes.
The implemented correctness policy is therefore:

1. Keep the normal directed gnomonic path and its already-retained changed real constraints.
2. If the directed stream actually exhausts while synthetic references remain, discard the finite
   polygon and replay an unrestricted shell stream. Form the initial spherical polygon only from
   real constraints, then clip it against every remaining generator before extraction.
3. Never return a reconstruction made only from the gnomonic builder's accepted constraints: that
   set omits the finite-proxy no-ops whose global relevance caused this issue.
4. Do not retain pre-bounded no-op ids or switch at a fixed candidate budget on the fast path.
   Reconsider a progress-aware circuit breaker only after measuring the implemented cold replay's
   actual crossover against continued gnomonic clipping.

The retained near-antipodal-tripod regression makes all three real constraints miss the finite
gnomonic proxy, exhausts the directed stream, and verifies that unrestricted spherical replay
returns the three correct owning edges and satisfies every replayed constraint. Deterministic
hemisphere probes through 2,000 sites recovered four origin-outside cells per case; the complete
500-site and 1,000-site public hemisphere diagrams remained strict-valid. The recovery is cold: it
adds no candidate recording, allocation, or threshold branch to successful fast-path cells.

There is a separate nonlocal regime that a synthetic-edge budget does not detect. Perturbed
great-circle inputs became gnomonically bounded but processed 204,561 candidates at 1k sites,
542,165 at 2k, and 2,735,916 at 5k; the maximum per-cell count was approximately `n`. This requires
a total-query-work circuit breaker and Local3d/global-hull escalation if that performance regime is
later optimized; spherical chart recovery is not the relevant mechanism. The current inputs still
construct successfully and any assembled defects remain eligible for the existing Local3d repair.
No valid input is known to reach a pre-assembly cell-build error after the current recovery and
degeneracy policies, so a pre-assembly repair seam is not an active correctness requirement. It
would become one only if such a valid-input failure were reproduced.

### AUD-016 — Near-semicircle Voronoi edges conflict with the strict representation contract

- **Priority:** P1
- **Class:** structural return gate / large-cell representation policy
- **Confidence:** Reproduced by the intrinsic small-`n` campaign
- **Status:** Resolved

A valid, non-welded four-generator fixture with all sites in a common hemisphere returns a
tetrahedral cell complex whose true large-cell boundaries contain edges very close to a
semicircle. Two stored edge endpoint pairs fall within `ANTIPODAL_DOT_EPS`; the old strict validator
rejected them and excluded them from its edge count (`Euler=3`, two antipodal-edge invariants), even
though the ordinary production return gate accepted the diagram. The old chord-normalized
intrinsic sampler also reported about `3.3e-5` radians of artificial ownership error because its
interpolation is ill-conditioned near a semicircle.

This is not necessarily a wrong Local3d or clipping result. When the generator hull does not
contain the origin, legitimate spherical Voronoi edges can approach pi in length; changing hull or
clipping algorithms cannot remove that intrinsic regime. The selected structural contract is:

1. every ordinary edge is the shorter-than-pi arc on the bisector plane of its two owning
   generators;
2. in the near-pi conditioning band, recover that plane from the owners instead of the endpoint
   cross product, then reject only an exactly antipodal or owner-inconsistent representation; and
3. exact-pi degeneracies use the explicit SoS policy, because an endpoint pair alone cannot identify
   the two distinct semicircles of a lune.

The strict and repair gates now pair the two halfedges before applying owner-plane classification.
The detailed validator retains these edges in its incidence and Euler accounting. A focused
four-site regression is strict-valid with `Euler=2`, and the owner-plane intrinsic sampler allows
the exhaustive uniform `n=4..32`, seeds `0..63` campaign to run normally. The tolerance remains a
conditioning trigger, not an acceptance relaxation.

Exact affine-circle failures now take the generalized `PerturbCoplanar` cold retry. It selects a
stable plane in linear time, certifies every canonical f32 generator with exact `orient3d == 0`,
and only then applies the deterministic off-plane perturbation. A one-f32-ulp negative control pins
that this is not a tolerance classification. The prior conservative tolerance classifier remains
only for nominal full great circles whose canonical rounding prevents an exact certificate. Strict
mode still returns the original clean error.

The remaining geometric consumers now share the same owner-plane arc primitive. Quality sampling
uses Rodrigues interpolation on the owning bisector; centroid integration uses the stabilized
oriented plane and angle; and area uses a generator-centered triangle fan that splits a near-pi
boundary at its conditioned midpoint. On the four-site fixture, summed-area error fell from about
`1.1e-2` to `3.8e-5` steradians and sampled edge residuals are about `3e-8`.

No neighbor ids were added to the stored diagram. Individual measures first run their ordinary
degree-local pass; only an actually observed near-pi edge triggers a sparse whole-mesh owner scan.
`lloyd_step` batches every affected key into one scan and recomputes only those canonical cells.
Measured serial scans over 600k, 3M, and 6M halfedges took 2.6ms, 22.6ms, and 66.9ms respectively,
with memory proportional only to the rare requested keys. This avoids the full adjacency build's
measured 21.5ms, 124ms, and 271ms plus roughly 130–170MB transient storage at one million sites.
The spherical fallback clipper likewise uses its already-retained edge constraint plane for
near-pi intersections instead of reconstructing the plane and branch from an unstable endpoint
cross/midpoint.

On a one-million-site uniform release probe, computing every area took 319ms versus 243ms for the
old vertex-zero fan (+31%, reflecting the robust generator fan's two additional triangles per
ordinary six-edge cell). A full `lloyd_step` took 328ms versus 369ms for the old per-cell centroid
loop (-11%). These are on-demand measure costs; diagram construction is unchanged.

## Recommended implementation sequence

1. **AUD-001:** resolved — removed unsafe-precondition violations.
2. **AUD-002/AUD-009:** resolved — added the five-site regression and bounded/escalated
   reconciliation edits.
3. **AUD-003:** resolved — preserved the complete packed unseen bound.
4. **AUD-004/AUD-005:** resolved — retained tolerance-only-active fallback constraints and removed
   unsupported positive-epsilon clipping.
5. **AUD-006/AUD-007:** resolved — tightened accepted serialized representations.
6. **AUD-008:** resolved — construction certifies edge agreement, the incidence pass supplies a
   fused Euler gate, and full strict validation remains optional/testing.
7. **AUD-013:** resolved — removed qhull's oracle role and public/backend surface; robust reference
   selection remains gated on AUD-010's canonical-model decision.
8. **AUD-010:** resolved — structural production contract selected; unified exact combinatorics is
   a deferred optional add-on.
9. **AUD-014:** resolved — preserved Local3d's oriented hull-face circumcenter through minting.
10. **AUD-012:** resolved — retained and pinned strict computed-f32 transitive welding semantics.
11. **AUD-011:** derive the packed/grid containment and association lemmas, decide the explicit
    libm premise, and retain the measured fidelity baseline and boundary coverage.
12. **AUD-015:** resolved for correctness — unrestricted spherical replay now occurs only after
    genuine chart exhaustion; early performance handoff remains deferred.
13. **AUD-016:** resolved — owner-plane validation, exact-pi SoS, conditioned measures/diagnostics,
    and retained-plane spherical fallback intersections are covered by focused regressions.

## Ongoing cross-cutting regression policy

- Keep the exhaustive uniform small-`n` geometry campaign active. Existing quality fixtures around
  100 sites missed both its four-site large-cell case and the five-site reconciliation failure.
- After any reconciliation or Local3d edit, assess incident-site equality and shared-edge bisector
  residuals in addition to strict topology.
- Add negative-control diagrams: rotate vertices without generators, reverse one face, duplicate a
  face, duplicate a vertex reference, and join two closed sphere complexes.
- Compare semantic topology across thread counts, bin counts, default SIMD, scalar SIMD, and FMA.
- Keep welding/joggle cases in separate expected-policy buckets rather than treating them as ordinary
  geometric errors.
- Do not introduce an external comparison as deciding evidence without certified predicates and a
  compatible exact-zero/SoS policy.
- Use exact or certified-robust references only after excluding or explicitly resolving exact
  degeneracies under the same SoS policy.

## Audit validation performed

Before qhull retirement, the audit ran targeted release checks including (these historical commands
are no longer runnable on the current feature surface):

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

Post-resolution validation for AUD-004/AUD-005 also included the complete release suite with serde
enabled plus all-target compilation of the timing, microbench, and escalation-probe features.
Boundary regressions pin fallback classification on both sides of `-FALLBACK_PLANE_TOL` and the
retained-versus-redundant constraint dispositions.

### Post-AUD-008 strict-validation soak (2026-07-14)

The strengthened edge-agreement certificate and fused Euler gate were exercised without qhull in
an 85-case, one-process-per-case release campaign with `VORONOI_MESH_VERIFY=1`. The matrix covered
50 uniform cases from 100k through 3M sites (ten seeds per size), five seeds each of cocircular,
cube-clustered, bimodal, and Fibonacci inputs, and 15 mega density-contrast cases through 1M sites.

- All 85 default-repair cases succeeded, passed full strict validation, and left zero
  post-repair and no-chain residuals.
- Fourteen cases produced 90 construction mismatch records: 71 `InBinThirdsMismatch` and 19
  `InBinUnconsumedCheck`. Reconciliation cleared every record before return.
- Representative defect-bearing reruns recorded `repair.attempted = false`; these cases were
  resolved by bounded reconciliation without paying for Local3d. Local3d remains an acceptable
  escalation, but this campaign did not need it.
- A focused 1M-site clustered cap produced seven mismatch records and returned strict-valid. A
  99,846-site cubed-sphere grid exercised 3,612 in-bin and cross-bin mismatch records and also
  returned strict-valid with no residuals.
- A smaller `RepairMode::Disabled` differential used the plain API on 2M uniform, 500k mega, 200k
  clustered, and 99,846-site grid inputs. All four returned `Ok` and independently passed strict
  validation. The retained harness permits a clean error in disabled mode but rejects any invalid
  success.

No strict-validation or Euler-only failure appeared. This is empirical evidence that the cheap
certificate and strict validator agree over the sampled envelope, not a proof that an Euler-only
failure is unreachable. `tests/robustness_campaign.rs` now records Local3d attempted/accepted
telemetry and retains the disabled-mode plain-API differential for future campaigns.
