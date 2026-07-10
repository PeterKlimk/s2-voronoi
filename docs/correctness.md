# Correctness

What the crate guarantees about its output, what it doesn't, and how it handles inputs at the edge
of what f32 can represent.

## The guarantee

For finite, unit-normalized input within representation capacity, `compute` either returns a
defined error or a **strictly valid subdivision of the sphere**:

- Euler characteristic `V - E + F = 2`
- every edge is shared by exactly two cells, with opposite orientation
- no self-loops, no overused edges, no dangling references
- one connected component

This is the property downstream code actually depends on — adjacency queries, meshing, rendering
with shared vertex buffers all break on a non-manifold graph, not on a vertex that is 1e-7 off its
true position. `validation::validate` checks it, and the test suite asserts it via fuzzing at
2M-4M points across many seeds.

Geometry is a separate, softer matter. Vertex positions are accurate to floating-point working
precision — inputs are f32, the clipping pipeline runs in f64, vertices are stored as f32. No f32
implementation can return mathematically exact positions, and this one does not claim to. Features
near the resolution floor (epsilon-length edges, vertices from near-cocircular generators) may be
kept or collapsed; which one happens is a policy choice, not a correctness bug, as long as the
graph stays valid.

## Coincident generators (welding)

Generators closer than a fixed radius (~1.4e-6 chord) are **welded** into one cell before
construction; the radius is derived from f32 rounding with a measured safety margin (~8x over the
worst adversarial construction). Welded inputs share a canonical cell, exposed through
`weld_map()`.

Welding is required for graph validity, not just input hygiene: three or more mutually-near
generators otherwise enclose a micro-cell that gets clipped to nothing, leaving the surrounding
cells with unpaired edges. It is on by default (`PreprocessMode::Weld`). Uniform random f32 data
contains sub-radius pairs by birthday statistics around the low millions of points, so welded
output is normal at scale, not exceptional. Callers who certify their own separation can disable
it (`PreprocessMode::Disabled`); `MergeWithin(r)` sets an explicit radius.

## Degenerate input

Pure great-circle (coplanar) input is rank-deficient: its exact diagram is a lower-dimensional
shape that is not stable under f32 rounding. By default (`DegenerateMode::PerturbGreatCircle`) the
pipeline runs normally, and only on failure detects this class and retries once with a
deterministic off-plane perturbation, returning the nearby full-dimensional diagram and recording
it in the report. `DegenerateMode::Strict` returns a clean error instead.

## Repair

The fast clipper is nearly always graph-correct. In rare near-degenerate regions, two cells can
disagree on an epsilon-scale combinatorial decision and leave a topology defect. `RepairMode::Local3d`
(the default) rebuilds the implicated neighborhood as one normalized local 3D hull and accepts the
result only if the whole diagram then validates; otherwise the computation fails loudly rather
than return a non-manifold graph. The guarantee is **valid subdivision or error** — exact graph
equality in adversarial tie regimes is an empirical property of the repair, not a symbolic
promise.

## Outcomes

A call resolves to one of:

- **Success** — a valid diagram. With welding, the *effective* diagram the backend solved is the
  authoritative one; `compute_with_report` exposes both the effective and the remapped views.
- **Defined error** — `UnsupportedGeometry` (a proven model limit, e.g. a cell reaching the
  generator hemisphere boundary), `RepresentationLimit` (storage/index capacity), `DegenerateInput`
  (sub-weld coincidence, naming the offending generators), or `ComputationFailed` (a terminal
  state not yet given a narrower class, including a surviving post-repair unpaired edge). These
  fail cleanly, without panic.
- **Panic** — reserved for internal invariant violations that indicate a bug, not an input class.

## Not promised

- Exact vertex positions. Use exact arithmetic and a different performance class if you need them.
- Anything about input outside the envelope beyond a clean error: non-finite values are rejected;
  sub-weld coincidence is welded; rank-deficient input is perturbed (or errors in strict mode).
- Stable vertex ordering or index assignment across versions.

Unit-length input is assumed, not enforced (it is canonicalized once at entry, so output
generators may differ from the raw input by ~1 ulp). Non-finite components are rejected with an
index-bearing `InvalidInput`.
