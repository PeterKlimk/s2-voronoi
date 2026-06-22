# Correctness Contract ("Essentially Voronoi")

This document defines what the crate promises about its output, what it explicitly does not
promise, and the evidence behind the boundary between the two. It is the conceptual companion to
`docs/supported-envelope.md` (which classifies outcomes) and `validation.rs` (which enforces the
hard guarantee).

Some of this contract is implemented today; some is the agreed target. Each section notes status.

## The two-tier guarantee

No floating-point Voronoi implementation can promise "the mathematically exact diagram" — that
requires exact arithmetic. Instead of pretending otherwise, this crate splits the promise in two:

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
performance; the contract documents it.

Status: **implemented.**

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

## The residual failure: classified, not masked

The only hard failure mode at the coincidence boundary is `ClippedAway` of a micro-cell enclosed
by sub-weld-radius neighbors — reachable only with welding disabled or an undersized custom
radius. An earlier draft of this contract proposed emitting a degenerate cell and continuing;
that is **unsound**: by the time the micro-cell is clipped away, its neighbors have already been
clipped against its bisectors, so their boundaries carry edges that would pair against the
missing cell, producing unpaired edges. A late weld has the same problem.

The implemented resolution classifies instead of masking: on `ClippedAway`, the backend scans for
generators within the weld radius of the failing one and returns
`VoronoiError::DegenerateInput` naming the coincident generator indices and pointing at the fix
(enable welding, or merge the points). A `ClippedAway` with no coincident generators remains
`ComputationFailed` — that case is a bug, not an input class.

Status: **implemented.**

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

## Current repair-backed graph contract

The active spherical strategy is:

```text
fast gnomonic clip
  -> assemble and validate
  -> if residual topology defects are detected, rebuild the implicated closure
     with normalized local 3D repair
  -> accept only a strictly valid whole diagram, otherwise return a loud error
```

This is deliberately weaker than a global exact-predicate construction, and that
is the point. The fast path is nearly always the normalized S2 graph in observed
uniform and practical workloads. In the rare near-degenerate regimes that expose
fast-path disagreements, normalized local 3D repair has produced the normalized
truth graph in every observed repaired case, and no "coherently agreed upon but
wrong" repaired local topology has been observed. The hard public guarantee is
therefore **valid subdivision or error**; exact graph equality in adversarial tie
regimes remains an empirical repair property, not a symbolic promise.

The reference graph, when audited, is the convex hull / Delaunay graph of
f64-renormalized S2 directions. Raw f32 radius drift is not part of the spherical
contract.

`ComputeReport` separates repair diagnostics from output residuals:
`pre_repair_edge_mismatches` records live-dedup disagreements that exercised
reconciliation/repair, while `post_repair_unpaired_edges` records residual
output-invariant failures. A strict-valid returned diagram may legitimately have
pre-repair mismatches; it must not have post-repair residuals.

Welding and degenerate perturbation are separate policies. Welding resolves
subresolution coincidence by solving an effective generator set and remapping
welded twins onto canonical cells. It does not make lower-dimensional inputs
full-dimensional. Rank-deficient great-circle inputs require the explicit
degenerate policy below when the caller wants an approximate diagram instead of
a clean error.

## Opt-in great-circle perturbation

Pure great-circle inputs are rank-deficient: the exact diagram is a lower-dimensional
lune decomposition with two antipodal high-degree Voronoi vertices. That output
class is not robust under f32 input noise or normalization, where the same input
usually becomes a nearby full-dimensional diagram with tiny pole fans.

The default policy therefore remains strict: fail cleanly if the ordinary backend
cannot produce a valid full-dimensional diagram. `DegenerateMode::PerturbGreatCircle`
is an opt-in robust mode. It first lets the ordinary pipeline run; only after a
failure does it classify full great-circle rank-2 inputs and retry once with a
deterministic off-plane joggle. The returned diagram is explicitly the nearby
perturbed solved problem, and `ComputeReport::degenerate.perturbation_applied`
records that fact.

This mode is deterministic and validation-gated, but it is not symbolic SoS and
does not claim to return the exact lower-dimensional Voronoi diagram. Its
purpose is operational totality for a fragile input class whose exact output is
not robust under f32 rounding or normalization.

## Paths to a stronger guarantee

These are optional hardening directions; none block release:

1. **Tolerance audit / forward error analysis.** The consistency-critical predicates are few:
   point-vs-half-plane sign in f64 chart coordinates, projection validity, extraction degeneracy.
   Bounding the computed-distance error and verifying the clip epsilon dominates it converts the
   central tolerances from empirical to derived-with-safety-factor.
2. **Repair completeness audit.** Continue broad CGAL/local-hull sweeps over
   repaired cases. The specific thing to hunt is a repaired diagram that is
   strictly valid but disagrees with normalized 3D truth. We have not observed
   one; finding one would change the contract.
3. **Exact oracle at small N.** A rational-arithmetic (or robust-predicate) reference
   implementation for N ≤ ~100, used as a combinatorial test oracle for boundary-region fuzzing
   (separations in [1e-8, 1e-6]).
4. **Canonical/exact topology research.** A full exact-by-construction fast path
   remains a research direction, but it is no longer the active product target.
   The failed mixed exact/local predicate experiments showed that exact signs
   cannot simply be dropped into the current approximate polygon evolution.

## What this contract is not

- Not a promise of exact geometry. Users needing certified positions need exact arithmetic and a
  different performance class.
- Not a promise about inputs outside the envelope: sub-weld coincidence (welded), non-finite
  values (rejected), pure great-circle/coplanar sets in strict mode (defined failure or opt-in
  perturbation; see `docs/supported-envelope.md`).
- Not stable vertex ordering or index assignment across versions.
