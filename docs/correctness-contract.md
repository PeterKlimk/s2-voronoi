# Correctness Contract ("Essentially Voronoi")

This document defines what the crate promises about its output, what it explicitly does not
promise, the outcome classes a caller can observe, and the evidence behind the boundary between
the two. `validation.rs` enforces the hard guarantee; this document is the conceptual companion to
it.

Some of this contract is implemented today; some is the agreed target. Each section notes status.

## The two-tier guarantee

No floating-point Voronoi implementation can promise "the mathematically exact diagram" — that
requires exact arithmetic. Instead of pretending otherwise, this crate splits the promise in two.

### 1. Topological guarantee (hard, machine-checked)

For inputs inside the supported envelope, the output is a **strictly valid subdivision of the
sphere**:

- Euler characteristic V − E + F = 2
- every edge appears in exactly two cells with opposite orientation
- no self-loops, no overused edges, no invalid references
- one connected component

This is the guarantee downstream users actually depend on. Game, simulation, and meshing code
breaks on a non-manifold graph, not on a vertex that is 1e-7 away from its true position.
`validation::validate` is the enforcement mechanism, and contract tests assert
`is_strictly_valid()` — not loosened approximations of it.

Status: holds today for non-degenerate inputs at all tested scales (fuzz-verified at 2M–4M points
across 15 seeds with preprocessing disabled), modulo the orphan-vertex representation note below.

### 2. Geometric guarantee (soft, quantified, policy-driven)

Vertex positions and edge geometry are accurate to floating-point working precision (inputs are
f32; the clipping pipeline computes in f64 internally). Features near the resolution floor —
epsilon-length edges, near-coincident vertices arising from near-cocircular generators — may be
kept, collapsed, or canonicalized. **Which of those happens is a policy choice, not a correctness
bug**, provided the topological guarantee holds.

`quality::assess` measures geometric fidelity (ownership margins, vertex/edge residuals); it is
deliberately separate from strict validity.

Status: behavior exists; the policy knob (keep / collapse / canonicalize epsilon features) is not
yet exposed in `VoronoiConfig`. The internal tolerances are empirical. That is acceptable — the
contract is enforced by the validator and the adversarial corpus, not by derived constants — but
the tolerances should live in one documented module (see roadmap).

## Default spherical contract

For finite normalized spherical inputs within representation capacity, the default configuration
is intended to return a strictly valid subdivision. The default robustness policies are part of
that contract:

- `PreprocessMode::Weld` merges subresolution coincident generators and remaps welded twins onto
  canonical cells.
- `DegenerateMode::PerturbGreatCircle` deterministically perturbs exact full-great-circle rank-2
  inputs into a nearby full-dimensional problem.
- `RepairMode::Local3d` rebuilds rare residual topology-defect neighborhoods and accepts only a
  strictly valid whole diagram.

The opt-out modes (`PreprocessMode::Disabled`, `DegenerateMode::Strict`, and
`RepairMode::Disabled`) exist for diagnostics, benchmarking, and callers that prefer clean errors
or raw fast-path behavior over robust default recovery.

## Outcome classes

A caller can observe four distinct outcome classes.

### 1. Supported success

The computation returns a diagram within the currently supported model:

- `compute` / `compute_with` / `compute_with_report` return `Ok(...)`
- default spherical output is a strictly valid S2 subdivision
- when preprocessing merges occur, the **effective** diagram actually solved by the backend is
  strictly valid, and welded twins in the returned diagram alias canonical cells
- `compute_with_report` records any pre-repair mismatch diagnostics separately from post-repair
  residuals

### 2. Supported failure

The computation returns a defined `Err(...)` for an input class or limit the backend recognizes
explicitly. These should fail cleanly without panic.

- `VoronoiError::UnsupportedGeometry` — proven geometric/model limits, not generic failure.
  Current example: a cell reaches the generator hemisphere boundary, so the gnomonic model is no
  longer valid.
- `VoronoiError::RepresentationLimit` — concrete storage/layout/indexing limits. Current examples:
  packed `(bin, local)` layout overflow; global assembled/remapped/reconciled index-buffer
  overflow; assembled/deferred-fallback vertex-id overflow in the `u32`-backed diagram model;
  total generator count exceeding the backend's `u32`-backed identifier model.
- `VoronoiError::DegenerateInput` — sub-weld coincidence that explains a `ClippedAway` micro-cell;
  names the coincident generator indices and points at the fix (enable welding, or merge).
- `VoronoiError::ComputationFailed` — terminal failure states understood operationally but not yet
  promoted to a narrower public class. Current examples: unbounded-after-exhaustion that is
  neither recoverable by all-constraints fallback nor handled by the rank-2 perturbation policy;
  **post-repair edge residual** (surviving unpaired interior edge(s) after reconciliation and
  local repair — the sphere has no boundary, so an unpaired interior edge means the graph is not a
  valid subdivision, and `compute` errors rather than ship it). No such default residual is
  currently pinned as a known input class; dense cap cases are a repair frontier and validate
  under the default config.

### 3. Preprocessing-altered solved problem

When preprocessing merges near-coincident generators, the backend solves a different effective
generator set and remaps the result back to the original input count. This is not the same
contract as a strict Voronoi solve over the original input set.

- `compute` returns the remapped/original-count diagram.
- `compute_with_report` also exposes `effective_diagram`, `returned_validation`,
  `effective_validation`, `preferred_diagram()`, `preferred_validation()`.
- If merges did not occur, returned and effective views coincide. If merges occurred, the
  effective view is the one the backend actually solved; the returned remapped diagram is a useful
  original-input convenience view but is not the primary correctness view, and strict validity of
  the remapped diagram does not guarantee the same thing as strict validity of the effective
  problem.

This is an intentional robustness tradeoff, not a hidden invariant.

### 4. Invariant failure / bug

Some internal states are still treated as bugs rather than supported input outcomes and may panic:
`ClippedAway` with no sub-weld coincidence to explain it; `NoValidSeed`; extraction
metadata/reconstruction invariant failures inside `topo2d::builder`; internal stream-state
contradictions; internal clipper assumptions that should hold if the algorithm's invariants are
respected. These are not part of the supported public failure contract.

## Supported geometry envelope

### Inputs expected to succeed

- unit-length finite input points on `S²`
- non-degenerate point sets large enough to form a Voronoi subdivision
- many practical near-degenerate configurations: small-jitter great-circle-like inputs;
  upper-hemisphere inputs whose large cells require cold all-constraints extraction; clustered-cap
  inputs with reasonable anchoring; many near-cocircular and edge-reconciliation cases

This is reflected in the always-on contract tests in `tests/adversarial.rs`. The named
weird-geometry fixture library in `tests/weird_geometry.rs` collects the current success/failure
boundary for rank-deficient, large-cell, high-degree, and welded-subresolution cases.

### Inputs expected to fail cleanly

The backend returns a defined `Err(...)` for:

- cells that cannot form a bounded generator-centered gnomonic polygon after exhausting the
  neighbor stream and cannot be reconstructed as a full-dimensional spherical cell from all
  accepted constraints. The pinned pure-great-circle fixture still fails this way in strict mode.
  This is a rank-deficient input class, not the ordinary large-cell case (upper-hemisphere /
  large-cell inputs are handled by the cold all-constraints spherical fallback).
- extreme scale/layout cases exceeding current packed/indexing capacity
- cells whose intermediate clip polygon exceeds the fixed clipping vertex budget
  (`MAX_POLY_VERTICES` is currently 24): a cell genuinely bordering more than ~22 others before
  fallback/recovery can contain it. This is a representation limit, not a numerical failure.
- some terminal construction failures that are recognized but not yet classified more precisely

Pure great-circle / coplanar cases are the clearest example of default robust degenerate handling
rather than exact lower-dimensional output (see "Default great-circle perturbation" below).

### The dense near-cocircular regime

The crate stores Voronoi vertices as **f32**, and `validation::validate` checks that a cell's
vertices are distinct. A Voronoi vertex is the circumcenter of its generators — a *construction*
(a division), not a predicate — and for the thin Delaunay triangles produced by a dense
near-cocircular cluster the circumcenter is ill-conditioned, so adjacent Voronoi vertices can land
closer than f32 can resolve.

This is now primarily a **repair frontier**, not a default failure class. The dominant residual is
cross-cell chart rounding rather than circumcenter representability: once 3D references
f64-renormalize directions before applying exact predicates, fast gnomonic output matches a CGAL
exact hull on tested uniform (100k/500k/1M) and mega (12k) inputs, and normalized local 3D repair
returns strict-valid on the remaining near-degenerate clusters. A benchmark-style 50k cap at
radius 0.05 is a pinned raw fast-path defect with `RepairMode::Disabled` that default local 3D
repair resolves to strict-valid; a 2026-06 sweep of 1M clustered-cap, bimodal, and near-cocircular
cases also returned strict-valid under the default config.

Average point spacing does **not** predict failure, so there is deliberately no exposed density
threshold; use `PreprocessMode::MergeWithin` to weld sub-resolution generators when the intended
model should treat them as coincident. The repair backstop is described in
`docs/repair-design.md`.

### Inputs still treated as probes rather than contract rows

Some stress inputs remain exploratory rather than pinned as stable success/failure-oracle cases:
high-volume fuzz, coincidence margin maps, large-cap scaling probes, fallback-incidence probes,
and external exact-reference comparisons. They live in ignored diagnostics and scheduled stress
runs. The named rows in `tests/weird_geometry.rs` are the current contract boundary; ignored tests
are probes, not disabled requirements waiting to be enabled.

## Coincident generators and the weld radius

### The resolvability map (empirical, 2026-06)

Probed with preprocessing disabled, 20k-point bases, defects measured by strict validation
(margin-mapping probes in `tests/coincidence_probes.rs`; the asserting contract tests derived
from them live in `tests/adversarial.rs`):

| configuration | breaks at (min pairwise chord) | valid from |
|---|---|---|
| isolated pairs, any ulp direction, any position | ~1e-12 and exact bit-equality | ~1e-10 |
| scattered clusters (k = 3, 5, 9), random positions | ≤ ~1.5e-8 | ~3e-8 |
| scattered clusters at cube-map seam centers | ≤ ~1.9e-8 | ~4e-8 |
| axis-aligned ulp pairs on seams (exact shared zero components) | up to ~1.2e-7 | — |
| axis-aligned pairs on the equator seam, separation swept | — | ≥ 2.7e-7 |

Key observations:

- **Pairs are nearly indestructible.** The f64 internal pipeline plus combinatorial vertex
  identity (vertices keyed by sorted generator-index triples) resolves pairs two orders of
  magnitude below f32 ulp scale.
- **Clusters are the binding constraint.** Three or more mutually-near generators fail around
  1.5–3e-8; the enclosed micro-cell gets clipped away.
- **Alignment is worse than proximity.** Pairs at exactly symmetric positions (cube corners, axis
  poles, 45° points — equivalently, chart-basis branch boundaries and cube-map seams) fail at
  separations ~4x larger than any scattered configuration. The identical configuration under an
  arbitrary rotation is strictly valid, so the failure is positional, not geometric.
- **There is exactly one hard failure mode.** Every non-recoverable failure observed is
  `ClippedAway` of an enclosed micro-cell. The fragility is concentrated in one code path.

### The weld policy

Generators within **WELD_RADIUS = ~1e-6 chord distance** are welded (treated as one generator)
before cell construction. Rationale:

- worst observed failure across all adversarial constructions: ~1.2e-7 → ~8x safety margin
- for regular/well-spaced data (grids, Lloyd-relaxed sets), spacing reaches 1e-6 only around
  ~10^13 points — the weld is invisible for any realistic regular workload
- for uniform random data, pairs within 1e-6 appear by birthday statistics around the
  low-millions point count; welding them is desirable (they are coincident at f32 scale) and the
  weld must therefore have correct output semantics, not just correct geometry

The weld is **required for graph validity** (because of the cluster and aligned regimes), not
input hygiene. Inputs known to respect the radius can disable it.

Welded generators' output semantics: welded input indices must reference the same cell in a way
the validator understands (shared cell, plus a weld report). Materializing duplicate copies of
the cell is what previously made large fuzz runs validate as INVALID — that representation is the
bug, not the welding.

Status: **implemented.** `PreprocessMode::Weld` (the default) welds at the coincident-distance
radius via a parallel quantized-key pass; welded twins alias their canonical cell's storage and
are exposed through `SphericalVoronoi::weld_map` / `canonical_cell_index`. `validate()` accounts
the subdivision over canonical cells and checks weld-map consistency as an invariant. The 2-4M
fuzz sweeps assert strict validity under the default config.

### Input validation

Non-finite components (NaN/inf) are rejected by an O(n) input check with an index-bearing
`VoronoiError::InvalidInput`. Unit-length is assumed, not enforced, and stays that way for
performance; the contract documents it. Status: **implemented.**

## The planar contract

`compute_plane` makes the same two-tier promise over the rectangle: a strictly valid subdivision
(every interior edge in exactly two cells with opposite orientations, single-use edges only on
the rect boundary, disk topology `V - E + (F + 1) = 2`), enforced by `validation::validate_plane`
and asserted by the planar fuzz and coincidence suites.

### The planar resolvability map (empirical, 2026-06)

Probed with the raw pipeline (no radius weld), normalized units (longer rect side = 1):

- **Pairs resolve at any distinct-f32 separation** — at generic positions, straddling grid-cell
  walls (the plane's only classification branches), and at rect corners. The sphere's worst
  failure class (chart-basis divergence between near-coincident twins at seams) has no planar
  analog: the plane has one global chart, and a shared bisector is built from exactly negated
  f64 differences on the two sides. A degenerate (zero) bisector is provably impossible for
  distinct normalized f32 coordinates — no f64 underflow exists for any f32 difference.
- **Clusters (k >= 3) below ~1 ulp of unit scale break topology** (degenerate cells, overused
  edges, broken Euler): invalid at min-separation 3e-8, valid from 6e-8 in every probed
  configuration. Individual bisectors stay well-formed; their mutual intersections do not.
- The subnormal-separation regime near the origin (distinct points ~1e-40 apart) fails the same
  way.

### The planar weld policy

`PLANE_WELD_DIST = 1e-6` normalized: generators within it weld to one cell (lowest original index
canonical, exposed via `PlanarVoronoi::weld_map()`). That is ~30x margin over the worst observed
failure (the sphere ships ~8x). Features below this scale exist only for >1e12 uniform points.

Detection is grid-integrated: the kNN spatial grid doubles as the weld detector (a pair can only
exist within one cell or across a radius-thin wall band), so duplicate-free inputs pay a
read-only scan and the detection grid is reused for the computation. One measured fact worth
knowing: uniform random f32 data naturally contains sub-radius pairs at production scales
(birthday effect — ~3 welds at 1M points, ~15 at 2M), so welded outputs are normal, not
exceptional, at millions of points.

The permanent probe suite is `tests/plane_coincidence_probes.rs` (ulp pairs/clusters at generic,
wall, corner, and subnormal positions, plus a randomized sweep in scheduled CI).

## The repair-backed graph contract

The active spherical strategy is:

```text
fast gnomonic clip
  -> assemble and validate
  -> if residual topology defects are detected, rebuild the implicated closure
     with normalized local 3D repair
  -> accept only a strictly valid whole diagram, otherwise return a loud error
```

This is deliberately weaker than a global exact-predicate construction, and that is the point. The
fast path is nearly always the normalized S2 graph in observed uniform and practical workloads. In
the rare near-degenerate regimes that expose fast-path disagreements, normalized local 3D repair
has produced the normalized truth graph in every observed repaired case, and no "coherently agreed
upon but wrong" repaired local topology has been observed. The hard public guarantee is therefore
**valid subdivision or error**; exact graph equality in adversarial tie regimes remains an
empirical repair property, not a symbolic promise.

The reference graph, when audited, is the convex hull / Delaunay graph of f64-renormalized S2
directions. Raw f32 radius drift is not part of the spherical contract.

`ComputeReport` separates repair diagnostics from output residuals: `pre_repair_edge_mismatches`
records live-dedup disagreements that exercised reconciliation/repair, while
`post_repair_unpaired_edges` records residual output-invariant failures. A strict-valid returned
diagram may legitimately have pre-repair mismatches; it must not have post-repair residuals.

Welding and degenerate perturbation are separate policies. Welding resolves subresolution
coincidence by solving an effective generator set and remapping welded twins onto canonical cells;
it does not make lower-dimensional inputs full-dimensional. Rank-deficient great-circle inputs use
the default degenerate policy below.

## Default great-circle perturbation

Pure great-circle inputs are rank-deficient: the exact diagram is a lower-dimensional lune
decomposition with two antipodal high-degree Voronoi vertices. That output class is not robust
under f32 input noise or normalization, where the same input usually becomes a nearby
full-dimensional diagram with tiny pole fans.

The default policy therefore returns a robust nearby full-dimensional diagram:
`DegenerateMode::PerturbGreatCircle` first lets the ordinary pipeline run; only after a failure
does it classify full great-circle rank-2 inputs and retry once with a deterministic off-plane
joggle. The returned diagram is explicitly the nearby perturbed solved problem, and
`ComputeReport::degenerate.perturbation_applied` records that fact.

This mode is deterministic and validation-gated, but it is not symbolic SoS and does not claim to
return the exact lower-dimensional Voronoi diagram. Its purpose is operational totality for a
fragile input class whose exact output is not robust under f32 rounding or normalization. Use
`DegenerateMode::Strict` when that input class should instead return the ordinary clean error.

## The residual failure: classified, not masked

The only hard failure mode at the coincidence boundary is `ClippedAway` of a micro-cell enclosed
by sub-weld-radius neighbors — reachable only with welding disabled or an undersized custom
radius. An earlier draft of this contract proposed emitting a degenerate cell and continuing;
that is **unsound**: by the time the micro-cell is clipped away, its neighbors have already been
clipped against its bisectors, so their boundaries carry edges that would pair against the
missing cell, producing unpaired edges. A late weld has the same problem.

The implemented resolution classifies instead of masking: on `ClippedAway`, the backend scans for
generators within the weld radius of the failing one and returns `VoronoiError::DegenerateInput`
naming the coincident generator indices and pointing at the fix (enable welding, or merge the
points). A `ClippedAway` with no coincident generators remains `ComputationFailed` — that case is
a bug, not an input class. Status: **implemented.**

## Orphan vertices (representation note)

The vertex array may contain entries referenced by no cell. These are deliberate leftovers from
edge repair (compaction was skipped to protect the latency budget); they appear in small numbers
(single digits at multi-million counts), do not affect the subdivision topology, and the Euler
check already ignores them.

Policy: strict validation classifies unreferenced vertices as a representation note
(`ValidationReport::representation_notes`), not an invalidity, and an opt-in
`SphericalVoronoi::compact_vertices()` removes them (one rebuild pass; call it when a dense
vertex array matters, e.g. serialization or GPU upload). A future refinement could make
compaction cost proportional to repair count by tracking indices orphaned during edge repair.
Status: **implemented** (reclassification + opt-in compaction).

## Validation semantics

Strict validation and fidelity are intentionally separate.

- `validation::validate` checks strict subdivision validity plus exact invariants of the produced
  object.
- `quality` tracks fidelity/quality metrics and is not part of strict validity.

Consequences:

- With preprocessing disabled, strict validation of the returned diagram is the primary success
  check.
- With preprocessing merges, strict validation of the **effective** diagram is usually the right
  correctness view for the solved problem.
- When using `compute_with_report`, `output.preferred_diagram()` and `report.preferred_validation()`
  are the intended "check this first" entry points after a merged solve.

The intended failure-taxonomy split is conservative: `UnsupportedGeometry` only for proven
unsupported geometric/model boundaries; `RepresentationLimit` only for concrete
implementation/storage/layout limits; `ComputationFailed` for known terminal failure states still
broader than those two; panic only for internal contradictions still believed to indicate a bug.
The crate should not broaden `Result` classifications unless it can state the failure reason
honestly and specifically.

## Paths to a stronger guarantee

These are optional hardening directions; none block release:

1. **Tolerance audit / forward error analysis.** The consistency-critical predicates are few:
   point-vs-half-plane sign in f64 chart coordinates, projection validity, extraction degeneracy.
   Bounding the computed-distance error and verifying the clip epsilon dominates it converts the
   central tolerances from empirical to derived-with-safety-factor.
2. **Repair completeness audit.** Continue broad CGAL/local-hull sweeps over repaired cases. The
   specific thing to hunt is a repaired diagram that is strictly valid but disagrees with
   normalized 3D truth. We have not observed one; finding one would change the contract.
3. **Exact oracle at small N.** A rational-arithmetic (or robust-predicate) reference
   implementation for N ≤ ~100, used as a combinatorial test oracle for boundary-region fuzzing
   (separations in [1e-8, 1e-6]).
4. **Canonical/exact topology research.** A full exact-by-construction fast path remains a
   research direction, but it is no longer the active product target. The failed mixed exact/local
   predicate experiments showed that exact signs cannot simply be dropped into the current
   approximate polygon evolution.

The most likely near-term refinements are: tightening which adversarial families are pinned as
expected success vs expected failure; an early trigger for the all-constraints fallback so large
upper-hemisphere cells need not wait for full neighbor-stream exhaustion; fallback strategies for
remaining unsupported geometry/workspace limits rather than only failing cleanly; and tightening
the remaining bug-only extraction/reconciliation invariants without broadening the public error
taxonomy prematurely.

## What this contract is not

- Not a promise of exact geometry. Users needing certified positions need exact arithmetic and a
  different performance class.
- Not a promise about inputs outside the envelope: sub-weld coincidence (welded), non-finite
  values (rejected), pure great-circle/coplanar sets in strict mode (defined failure; default mode
  perturbs full great-circle rank-2 input).
- Not stable vertex ordering or index assignment across versions.
