# Canonical Certification and Exact-Topology Direction (2026-06-21)

This note records the design direction that came out of the Tier-2 repair
discussion after reading `reclip-tier2-state-2026-06.md`,
`reclip-local-hull-design.md`, and `p5-consistency-design.md`.

The short version: stitching/overlay repair can probably be made to construct a
valid graph, but it would still be a graph-validity strategy over an approximate
diagram. If the goal is to compare honestly with exact/topology-driven Voronoi
implementations, the cleaner direction is to make the topology canonical first
and use repair only as a bug trap or rare exact fallback.

## Current position

The old Tier-2 statement "valid-or-clean-error is enough" is not the target we
want anymore. It is correct as a safety contract, but it is unsatisfying as the
algorithmic endpoint: a production Voronoi builder should very rarely fail to
construct a graph when the input is valid and the local neighbor set is
adequate.

At the same time, the "just stitch it" intuition remains mostly right under a
weaker contract:

- keep the boundary between the approximate regime and repaired regime fixed,
- connect a new contested edge to an existing vertex when it lands there,
- connect it to an existing edge by splitting that edge when it lands there,
- delete remaining degree-2 boundary vertices, and
- validate the final graph.

That overlay/collapse construction should be able to produce a valid
near-Voronoi graph for many cases where pin-by-key failed. The objection is not
graph validity. The objection is that the repaired seam may no longer be the
dual of a Delaunay triangulation, so APIs such as `delaunay_triangles()` become
hard to specify honestly at repaired seams.

Therefore the new bias is: accept a performance cost if needed, and move toward
canonical predicate topology rather than accumulating more graph-repair
machinery.

## What we are relaxing

The existing clipper is already not an exact Voronoi/Delaunay implementation in
the mathematical sense. It uses chart-local floating-point arithmetic and does
not use exact predicates in the hot path. For graphics and physics use cases,
epsilon-scale topology differences are often below the resolution users can act
on, especially when outputs are consumed as `f32`.

So the contract does not have to be "the exact symbolic Voronoi diagram of the
unperturbed real input" in every mode. The practical target is:

- no silent invalid graphs,
- near-Voronoi geometry on ordinary inputs,
- exact or simulation-of-simplicity-consistent topology when the diagram is
  near-degenerate enough that local floating-point topology is unreliable, and
- an implementation whose public claims are comparable to serious
  predicate-driven triangulation libraries.

This is stricter than "nearly Voronoi and valid graph", but it can still avoid
exact arithmetic on every hot operation.

## Important failed experiment

The first experimental attempt at `canonical_topology` replaced local
Sutherland-Hodgman inside/outside masks with exact canonical in-circle signs for
real vertices.

That failed for a structural reason: the exact sign oracle and the approximate
polygon representation were not coherent.

Observed failures included bounded local triangles where all three existing
approximate vertices were outside according to the exact predicate. The true
cell was not necessarily empty; rather, the approximate polygon had evolved
under chart-local decisions, lerped intersections, and sometimes earlier
rounding choices that did not match the exact predicate system. Applying exact
answers late to that approximate polygon can erase the representation instead
of repairing it.

Conclusion: exact predicates should not be dropped into the current clipper as
authoritative per-lane sign replacements unless the whole polygon construction
is also expressed in the same canonical system. Otherwise mixed local/exact
topology manufactures contradictions at the boundary between regimes.

## Two viable families

### 1. Exact/canonical topology as the primary clipper

This is the cleanest correctness story: every combinatorial decision is made by
a canonical predicate over generator ids and coordinates. On the sphere, the
core predicate for a candidate vertex `(g, a, b)` clipped by `h` is the sign of
the oriented 4-point determinant:

```text
in_circle_sphere(g, a, b; h) == orient3d(g, a, b, h)
```

With canonical row ordering, exact signs, and a simulation-of-simplicity tie
policy, all cells ask equivalent questions and get coherent answers. In the
limit, this is the route that could let us delete edge repair, stitching, and
convex-hull seam recovery as correctness mechanisms.

The cost is architectural: the clipper cannot freely use chart-local signed
distances as topology if those signs can disagree with canonical signs. It
either has to construct topology from canonical predicates directly, or it has
to rebuild locally when canonical certification says the approximate topology
is not trustworthy.

### 2. Fast clip plus canonical certification and exact fallback

This is the currently more attractive route.

Run the existing fast clipper unchanged as the common-case producer. During or
after clipping, certify that its topology is compatible with the canonical
predicate system. If certification passes, accept the fast result. If it fails,
rebuild the affected cell or small region using an exact/canonical method.

This avoids mixing sign systems inside one polygon update. The approximate
clipper either produces a certified topology, or its output is treated only as a
hint for a slower exact path.

The fallback can be one of:

- per-cell exact halfspace/arrangement rebuild over the delivered neighbor set,
- local spherical Delaunay/convex hull over the delivered candidate set,
- small-region exact rebuild for a connected suspicious component, or
- as a temporary bridge, graph overlay/collapse with strict validation.

The first three preserve the stronger Delaunay/Voronoi story. Overlay/collapse
is still useful as a graph-construction fallback, but it carries the public API
caveat above.

## Making "suspicious" concrete

The suspicious flag should not be a heuristic like "distance is small" unless
it is explicitly only a performance hint. The robust version is a filtered
canonical predicate certificate.

For every real clip decision that the fast clipper commits:

```text
question: sign orient3d(g, a, b, h)
local:    chart-local signed-distance classification
cert:     canonical filtered predicate result
```

The certificate has three outcomes:

- `CertifiedInside`,
- `CertifiedOutside`,
- `Uncertain`.

A cell or edge becomes suspicious if:

- the canonical sign is uncertain,
- the local sign disagrees with a certified canonical sign,
- a canonical exact tie is reached and the current representation cannot express
  the required high-degree or SoS topology,
- a tiny edge / duplicate vertex / degree-2 remnant appears whose defining
  quadruple is uncertified, or
- a final edge pairing depends on two cells that used different uncertified
  question sets.

The filter can be implemented with an adaptive determinant evaluation:

1. Compute a fast f64 determinant and a conservative error bound.
2. If `abs(det) > error_bound`, the sign is certified.
3. Otherwise mark suspicious or call the exact expansion predicate.

In production, the cheap path should usually stop at step 2. During development
or for a "correct topology" mode, step 3 can be used aggressively to measure the
real fallback rate.

The critical distinction from the rejected margin gate in
`p5-consistency-design.md` is that this gate is a function of the canonical
question itself, not of a displaced chart-local margin. It must be keyed by the
generator tuple, not by the local polygon's current numeric distance.

## Missing vertices and coherent wrong topology

Edge checks catch many one-sided failures: if one cell creates an edge or corner
and its neighbor does not, assembly sees the mismatch.

The harder case is coherent wrong topology: all involved cells agree that a
near-degenerate vertex or tiny edge does not exist. In that case ordinary
edge-pair checks may pass because there is no disagreement to observe.

A hot suspicious flag based only on vertices that already exist cannot
provably catch every missing vertex. If the relevant candidate triple never
forms in the approximate polygon, there may be no committed decision to audit.

To catch that class, certification needs one additional closure step:

- Given an accurate incoming neighbor set and accurate termination criteria,
  audit the accepted cell topology against the canonical local Delaunay/Voronoi
  topology of that candidate set, or
- ensure the hot clipper asks every canonical question that could create a
  vertex/edge within the certified margin.

The first option is simpler to reason about. For each suspicious cell or
suspicious connected component, build the exact local topology over the
delivered candidate set and compare it to the fast topology. This catches
coherent missing features because the audit is not limited to features the fast
clipper happened to construct.

Conditioning still helps decide when to audit:

- small certified determinant margins on any local quadruple,
- very short accepted edges,
- near-duplicate vertices under different keys,
- high-degree candidate clusters,
- local polygon vertices whose defining planes are ill-conditioned, and
- disagreement between local and canonical signs on any formed vertex.

But conditioning alone is not a proof unless it ranges over all candidate
questions that could affect the cell. The robust proof boundary is the exact
local audit.

## Expected cost

If the current builder is largely memory-bound, a cheap f64 determinant filter
may cost less than the raw operation count suggests, provided it reuses
generator coordinates already touched by the clipper and does not add broad
random gathers.

The real costs are likely:

- extra coordinate loads for the other two generators defining a vertex,
- branchiness from adaptive predicate fallback,
- loss of SIMD density if certification runs interleaved with hot clipping,
- larger candidate sets if termination is padded to support certification, and
- exact/local rebuilds on pathological clustered or near-cocircular inputs.

The upside is also real:

- less post-assembly repair work,
- fewer edgecheck/stitch paths on normal inputs,
- simpler correctness story,
- better public comparability with Delaunator-style predicate-driven libraries,
- and possibly faster total time on difficult cases if exact fallback replaces
  repeated repair/validation churn.

A plausible implementation strategy is therefore two-tiered:

1. Hot path: current clipping plus canonical filtered certification.
2. Cold path: exact local rebuild for suspicious cells/components.

This accepts a small common-case tax to avoid a large permanent repair stack.

## Proposed branch direction

The current experimental branch should pivot from "make exact signs
authoritative inside the current clipper" to "certify the current clipper and
fall back cleanly".

Suggested staged work:

1. Remove behavior-changing canonical mask substitution from the hot clipper.
2. Keep or rename the feature as a certification/audit feature.
3. Record per-cell suspicion reasons:
   - uncertain canonical determinant,
   - local/canonical sign disagreement,
   - exact tie requiring high-degree/SoS handling,
   - suspicious tiny edge or duplicate vertex,
   - audit mismatch.
4. Add a development-only exact audit over the delivered neighbor set for
   suspicious cells.
5. Measure:
   - suspicion rate on uniform, clustered, mega, and known adversarial seeds,
   - exact fallback rate,
   - wall-time cost with and without exact fallback,
   - whether edgecheck/reconcile defects disappear before repair.
6. Only after the audit is trusted, decide whether to delete or demote:
   - edge repair,
   - stitch repair,
   - local hull seam repair,
   - convex-hull repair experiments.

The intermediate success criterion is not "no suspicious cells". It is:

- all accepted fast cells are certified,
- all uncertified cells are rebuilt or cleanly reported,
- no known case silently accepts invalid or incoherent topology, and
- the common-case cost is tolerable.

## Open questions

1. What is the cheapest exact local rebuild that preserves the public
   Voronoi/Delaunay contract?
2. Can the existing `local_hull.rs` core be reused as the cold audit/rebuild for
   suspicious components without inheriting the old seam-stitch problem?
3. How large must the delivered candidate set be for a per-cell exact audit to
   prove no missing vertex, assuming the existing termination criteria are
   accurate?
4. Should the final production mode always rebuild suspicious cells, or should
   it expose a faster "nearly Voronoi, graph-valid" mode?
5. Can enough of the canonical determinant filter be vectorized or batched to
   make the common-case tax close to invisible on memory-bound runs?

