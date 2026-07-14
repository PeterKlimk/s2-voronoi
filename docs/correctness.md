# Correctness

What the crate guarantees about its output, what it doesn't, and how it handles inputs at the edge
of what f32 can represent.

## The guarantee

For finite, unit-normalized input within representation capacity, `compute` either returns a
defined error or a **construction-certified, edge-agreeing spherical mesh**:

- every edge is shared by exactly two cells, with opposite orientation
- no detected low-incidence or post-reconciliation edge defect survives
- Euler characteristic `V - E + F = 2`
- representation invariants required by the safe accessors hold

Edge agreement is certified by the construction bookkeeping rather than by rebuilding and sorting
all output edges. Within a bin, every forwarded check must be consumed exactly once and endpoint
identity must match in reverse order. Across bins, sorted equal-key runs must contain exactly one
record from each side. Any recorded defect enters reconciliation; its post-pass checks exact
multiplicity and orientation over the touched region. The existing incidence pass also supplies
`V` and the live half-edge count, making the Euler check effectively free.

This is the production contract graphical consumers depend on: cracks, overused boundaries, and
orientation disagreement fail loudly or repair. `validation::validate` is the stronger diagnostic
contract. It additionally checks connectivity, duplicate faces/vertices, antipodal edges, and
other global representation properties. `compute_with_report` runs it unconditionally;
`VORONOI_MESH_VERIFY=1` enables it as a return gate for plain `compute`. Robustness campaigns and
tests use that gate, but ordinary production does not pay for its second global edge sort.

Geometry is a separate, softer matter. Vertex positions are accurate to floating-point working
precision — inputs are canonicalized f32 directions, clipping uses f64, and vertices are stored as
f32. “The exact Voronoi diagram” is ambiguous without first choosing whether the mathematical sites
are the incoming f32 vectors, their canonicalized rounded vectors, or ideal renormalizations of
those vectors, plus one policy for exact predicate ties. The current pipeline does not claim exact
positions, exact combinatorics for one unified site model, or a global backward-error witness.

The numerical certificate assumes the conventional behavior of supported floating-point targets:
IEEE-754 binary32/binary64 round-to-nearest arithmetic without flushing relevant subnormal values,
correctly rounded basic operations, and `f64::sqrt`/`sin_cos` accurate to within one ulp. Rust does
not specify a useful worst-case transcendental error bound, so this is an explicit platform premise
rather than a language-level theorem. The grid, clipping, and unseen-neighbor error budgets are
derived under that premise and retain outward reserves for the supported default SIMD,
`simd_scalar`, and hardware-FMA paths.

Features near the resolution floor (epsilon-length edges, vertices from near-cocircular generators)
may be kept or collapsed; which one happens is a policy choice, not a correctness bug, as long as
the graph stays coherent. Reconciliation collapses a positional equivalence component only when its full
diameter over stored vertices is at most `RECONCILE_DEGENERATE_LEN_EPS`; pairwise chains cannot
extend that bound. Rejected components escalate to Local3d under the default repair policy.

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

Precisely, welding forms a graph whose edges satisfy
`computed_f32_distance_squared < computed_f32_radius_squared`. Exact computed equality is not an
edge. Weld classes are the graph's transitive connected components, represented by their lowest
original input index. This is an explicit quotient policy, not a promise that every class member
lies within one radius of its representative: a chain of individually sub-radius pairs can have
endpoints much farther apart. The report and `weld_map()` expose the resulting quotient.

## Degenerate input

Affinely coplanar spherical input is rank-deficient: all generators lie on one great or small
circle, and its exact endpoint-pair diagram is either lower-dimensional or contains ambiguous
exact-pi lune edges. By default (`DegenerateMode::PerturbCoplanar`) the pipeline runs normally and,
only on failure, certifies exact affine coplanarity over the canonical f32 generators and retries
once with a deterministic off-plane perturbation. A conservative compatibility classifier also
retains support for nominal full great circles whose f32 canonicalization prevents an exact
certificate. The returned diagram is a nearby full-dimensional one and the report records that
perturbation. `DegenerateMode::Strict` returns a clean error instead.

## Repair

The fast clipper is nearly always edge-agreeing. In rare near-degenerate regions, two cells can
disagree on an epsilon-scale combinatorial decision and leave a topology defect. `RepairMode::Local3d`
(the default) rebuilds the implicated neighborhood as one normalized local 3D hull and accepts the
result only if the whole diagram then validates; otherwise the computation fails loudly rather
than return a known-bad graph. The guarantee is **edge-agreeing mesh or error**; exact graph
equality with one ideal Voronoi construction in adversarial tie regimes is not a symbolic promise.

## Near-semicircle derived geometry

Legitimate Voronoi edges may approach pi when the generators lie in a common hemisphere. Endpoint
cross products and normalized chord interpolation are ill-conditioned there. Strict validation,
area, centroid, Lloyd targets, and optional quality sampling recover the supporting bisector plane
from the edge's two owning generators. Exact-pi endpoint pairs remain unrepresentable without an
arc identity and therefore take the explicit coplanar perturbation policy described above.

The diagram does not retain neighbor ids solely for these rare measures. Ordinary calls stay
degree-local; observing a near-pi edge triggers a cold sparse scan for only the requested owner
pairs, and `lloyd_step` batches all affected cells into one such scan.

## Outcomes

A call resolves to one of:

- **Success** — an edge-agreeing, Euler-valid diagram. With welding, the *effective* diagram the backend solved is the
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

## Embedded spheres

`compute_on_sphere` extends the coordinate contract, not the geometric model. A validated
`SphereEmbedding` supplies a finite f64 center and positive radius. Each finite world input other
than the center is interpreted only by its ray from that center and normalized in f64 before the
ordinary f32 unit-sphere computation. Consequently:

- off-shell inputs are radially projected rather than rejected;
- `generator_world` reconstructs the backend's canonicalized shell generator, not the original
  world input;
- topology and reports describe the recovered f32 directions, so embeddings that lose direction
  information during world-coordinate rounding cannot promise bit-identical results;
- core areas remain steradians, while embedded areas multiply them by `radius²` and may be
  positive infinity if the mathematical physical area exceeds finite f64 range;
- `MergeWithin` and the default weld radius retain dimensionless unit-chord semantics;
- the world centroid is the on-shell spherical/Lloyd target, not an unconstrained Euclidean center
  of mass.

World-input conversion failures use the original slice index in `InvalidInput`. Once conversion
succeeds, backend errors and report diagnostics retain the same original-versus-effective index
domains documented by `compute_with_report`; embedding does not remap those diagnostics.

The embedding constructor ensures shell coordinates are finite, but a very small radius can still
round away when added to a much larger center. Using f64 transformations cannot recover detail
already lost in source coordinates. Ellipsoids, sphere fitting, weighted sites, and unequal radial
site distances are outside this API's contract.
