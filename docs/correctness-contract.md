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
(`tests/tmp_ulp_regimes.rs` probes; to be adopted into `tests/adversarial.rs`):

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

## Paths to a stronger guarantee

In increasing order of strength; none block release, all are roadmap items:

1. **Tolerance audit / forward error analysis.** The consistency-critical predicates are few:
   point-vs-half-plane sign in f64 chart coordinates, projection validity, extraction degeneracy.
   Bounding the computed-distance error and verifying the clip epsilon dominates it converts the
   central tolerances from empirical to derived-with-safety-factor.
2. **Consistency by construction (canonical predicates).** Today the same combinatorial fact
   ("does edge (a,b) exist?") is decided twice, in two different per-generator charts, and
   validity depends on the two epsilon-laden answers agreeing — the seam regime is exactly this
   dependency biting (twins at `TangentBasis` branch boundaries select different charts). Making
   every shared decision canonical — evaluated once, in a frame chosen by sorted generator index,
   inherited bit-for-bit by both cells — makes graph validity independent of floating-point
   accuracy. One then proves determinism (a code-structure property) instead of error bounds.
   This would also root-fix the seam regime and shrink what edge reconciliation has to repair.
3. **Exact oracle at small N.** A rational-arithmetic (or robust-predicate) reference
   implementation for N ≤ ~100, used as a combinatorial test oracle for boundary-region fuzzing
   (separations in [1e-8, 1e-6]).

## What this contract is not

- Not a promise of exact geometry. Users needing certified positions need exact arithmetic and a
  different performance class.
- Not a promise about inputs outside the envelope: sub-weld coincidence (welded), non-finite
  values (rejected), pure great-circle/coplanar sets and >90° cells (defined failure; see
  `docs/supported-envelope.md`).
- Not stable vertex ordering or index assignment across versions.
