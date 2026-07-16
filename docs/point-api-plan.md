# Point representation and interoperability plan

**Status:** Stage 0 audited; Option B selected for implementation.

This plan records the public-point decision around the measured output-materialization
optimization. It compares whether a checked semantic `SpherePoint` earns its additional API and
validation machinery over a simpler packed coordinate type, records the independent initial
preference for Option A, and closes the numerical gate in favor of Option B after producer-level
measurement. The decision is separate from the established choice to retain
`glam::Vec3`/`DVec3` as internal arithmetic types.

## Existing facts and constraints

- The public unit-sphere backend consumes and stores f32 coordinates. At entry it lifts finite
  in-band coordinates to f64, normalizes once, and rounds to canonical f32 generator directions.
- Candidate search and its certificates use canonical f32 sites; clipping, projection, predicates,
  and repair geometry use f64 where their error model requires it. Extracted vertices are rounded
  and normalized into f32 before live dedup and reconciliation.
- A stored direction is therefore not mathematically unit length. Canonical generators use an
  `f32::EPSILON` squared-norm envelope, while public vertex validation currently permits a wider
  `1e-4` envelope. Different vertex producers also normalize through different f32/f64 paths, so
  there is not yet one proven invariant shared by generators, vertices, centroids, and repairs.
- `UnitVec3` currently has public fields and infallible raw constructors and derives `Pod`,
  `Zeroable`, and optionally serde. It is a compact coordinate container and documentation marker,
  not an enforced unit-vector invariant.
- A downstream crate can implement `UnitVec3Like` for its own local record type, but cannot
  implement it for another foreign crate's vector type. Consequently the trait supports custom
  records but is not a version-independent bridge among existing math ecosystems.
- `UnitVec3` is returned or stored by `SphericalVoronoi`, `SphericalCellMesh`, centroid/Lloyd
  measures, embedding projections, and quality APIs. `SphereLocator` accepts generic
  `UnitVec3Like` queries and currently assumes rather than establishes unit normalization.
- The output API relies on contiguous generator and vertex slices. The diagram should remain
  non-generic.
- The null-write audit found a 4.5--5.5% native Mac ceiling for global assembly destinations. A
  separate final conversion currently maps backend `Vec3` generator/vertex allocations element by
  element into public `UnitVec3` allocations. At 2M generators and roughly 4M vertices, that point
  conversion alone reads and writes approximately 72 MB (68.7 MiB) each and temporarily owns both
  buffers. The null-write percentage is a broader assembly ceiling, not a forecast for this final
  conversion alone.
- A full f64 input/output representation is a separate research project. The fast path does not
  retain one authoritative f64 vertex per final key, and f64 output would change dedup,
  reconciliation, validation, measures, search bounds, and output-resolution policy.

## Decision criteria

The selected design should:

1. Accept arbitrary user point types without requiring an orphan-rule-impossible trait impl.
2. Preserve one owned, compact, contiguous f32 diagram representation.
3. Make normalization/error behavior explicit and consistent across construction and location.
4. Permit zero-copy bulk export as `&[[f32; 3]]`.
5. Permit an O(1) ownership transfer from final internal `Vec3` buffers without exposing a safe way
   to manufacture values that violate any promised semantic invariant.
6. Keep serde validation and wire shape deliberate rather than accidentally derived from storage.
7. Avoid measurable input-path regressions and capture a useful portion of the measured output
   traffic opportunity.
8. Avoid committing the entire diagram and algorithm surface to a user's math-library type.

## Representation options

### Option A — retain a simple public POD point

Keep `UnitVec3` (or rename it to a less invariant-heavy name such as `Point3f`) as a public
`#[repr(C)]` three-f32 value. Public fields, infallible `From<[f32; 3]>`, `Pod`, and `Zeroable`
remain valid because the type promises representation, not normalization. Computation and point
location validate/canonicalize raw values at their boundaries.

Add closure ingest and packed array views while retaining `UnitVec3Like` initially:

```rust
pub fn compute_xyz(points: &[[f32; 3]]) -> Result<SphericalVoronoi, VoronoiError>;
pub fn compute_by<T>(points: &[T], xyz: impl Fn(&T) -> [f32; 3] + Sync)
    -> Result<SphericalVoronoi, VoronoiError>;

impl SphericalVoronoi {
    pub fn vertices_xyz(&self) -> &[[f32; 3]];
    pub fn generators_xyz(&self) -> &[[f32; 3]];
}
```

Advantages:

- Smallest migration and simplest serde/bytemuck story.
- Honest about the existing soft invariant and useful for malformed-input diagnostics.
- O(1) internal ownership conversion can use a standard checked allocation cast.

Costs:

- The type name cannot honestly certify a sphere point.
- APIs returning it do not distinguish validated output from arbitrary coordinates.
- Every consumer must remember which boundaries require canonicalization.

### Option B — introduce a checked semantic `SpherePoint`

Use a private-field packed type for finite f32 directions inside the documented canonical norm
envelope:

```rust
#[repr(transparent)]
pub struct SpherePoint([f32; 3]);

impl SpherePoint {
    pub fn try_from_xyz(xyz: [f32; 3]) -> Result<Self, SpherePointError>;
    pub fn as_array(&self) -> &[f32; 3];
    pub fn to_array(self) -> [f32; 3];
    pub fn x(self) -> f32;
    pub fn y(self) -> f32;
    pub fn z(self) -> f32;
}
```

Raw input remains arrays/closures; users do not have to construct `SpherePoint` merely to call
`compute`. A future `try_from_xyz` would need an explicitly selected normalization contract and
must reject non-finite or directionless input. It cannot simply claim to reuse today's unit-backend
ingest: that path only normalizes finite values with squared length in `[0.25, 4]`, while the
world-embedding path has the scale-safe normalization policy. Crate-generated points would use a
crate-private constructor that accepts already-certified values. Deserialization must validate
exact stored values rather than silently normalizing wire data.

Do **not** implement public `Pod`, `Zeroable`, `AnyBitPattern`, infallible
`From<[f32; 3]>`, or another safe arbitrary-bit-pattern conversion. Those facilities would allow
safe construction of zero, NaN, or off-sphere values and make the semantic invariant false.
Zero-copy array views and the final allocation ownership transfer should instead be encapsulated in
small layout-asserted internal routines. The latter may require one audited unsafe boundary because
the type system cannot express that an internal `Vec3` buffer has already been geometrically
certified.

Advantages:

- Outputs, centroids, Lloyd sites, cell-mesh vertices, and canonical locator queries carry an
  explicit useful contract.
- Private construction provides one place to document the f32 norm envelope and reject invalid
  serde data.
- Raw input interoperability remains independent of the semantic output type.

Costs:

- More migration, custom serde, and negative invariant tests.
- Layout-safe public bytemuck construction must be withheld, so internal zero-copy transfer needs
  careful encapsulation.
- A near-unit invariant alone may not justify skipping entry canonicalization: canonical bit
  identity and mere norm-envelope membership are different promises.
- One type currently cannot state the strongest existing generator envelope while also accepting
  every validated vertex/centroid/repair producer. A `1e-4` common envelope would be too weak to
  justify canonical-query fast paths; the epsilon envelope would require a producer migration.
- Callers wanting a mutable/plain coordinate record must use arrays or their own type.

### Option C — use `[f32; 3]` as the only public point representation

Store `Vec<[f32; 3]>` directly and return arrays from all diagram, cell-mesh, measure, and embedding
APIs. Keep normalization entirely as an operation-level contract rather than a type-level one.

Advantages:

- Minimum type surface and maximum baseline interoperability.
- No layout cast is required at the public storage boundary if assembly writes arrays directly.
- Serde and FFI-oriented component access are straightforward.

Costs:

- Loses the domain distinction between world coordinates, arbitrary 3D vectors, generators, and
  spherical directions.
- Discoverability and coordinate methods are weaker.
- Users can accidentally feed unrelated triples into canonical-query APIs with no type signal.
- Moving from arrays to a semantic type later would be a broad breaking change.

### Rejected baseline — expose `glam::Vec3`

Making glam the stored/public type couples users to this crate's exact glam version, does not solve
interop with nalgebra or custom records, and makes a routine internal dependency part of the
diagram and serialization contract. Optional glam conversions can remain convenience features;
glam should not be the canonical boundary.

## Independent review and reopened decision gate

A conversation-blind review ranked the alternatives **A > B > C**. The decisive issue is that
`SpherePoint` would not currently establish one useful invariant across all of its proposed
producers. A weak common envelope would not remove checks or make raw locator queries safe; a
strong generator-grade envelope would require changing output, centroid, repair, serde, and query
contracts before the measured conversion optimization could land.

The review therefore recommended **Option A as the smallest safe next change**:

- Retain `UnitVec3` as an honest compact POD coordinate type for now. Its name documents intended
  use but does not certify normalization.
- Preserve existing public signatures while first removing the redundant final point-copy pass and
  adding packed xyz views.
- Add closure ingest later as the genuine interoperability improvement. An explicit `compute_xyz`
  is primarily discoverability because `[f32; 3]` already implements `UnitVec3Like`.
- Treat locator normalization/error behavior as a separate correctness/API decision. It is
  justified independently: candidate ranking and shell bounds must not consume differently scaled
  query models, but changing infallible single/batch APIs to report malformed queries needs its own
  design.
- Reconsider Option B only after a producer/invariant audit demonstrates one envelope that covers
  every returned point and materially simplifies downstream code.

Option C remains the simplicity control but discards useful domain naming without offering a
meaningful layout or interoperability advantage over A.

This simplicity ranking is not the final long-term choice. There are no users to preserve, and the
existing `1e-4` public vertex tolerance is documented as deliberately loose. Most identified
producers already normalize immediately before f32 storage: generators use f64 normalization,
fast extraction normalizes after f32 rounding, fallback and repair start from normalized f64
directions, and centroids normalize their f64 integral. A coherent numerical migration is
worthwhile if it replaces those variants with one storage rule, tightens the actual common
envelope, and makes a checked point useful to queries and consumers.

Consequently **Stage 0 below reopened A versus B**. Choose B if evidence shows that one shared
canonicalizer covers every legitimate producer without unacceptable topology, output-resolution,
or throughput movement and the resulting type removes meaningful checks or invalid states.
Otherwise retain A rather than creating a semantic wrapper around a weak invariant.

## Stage 0 findings and decision

The profiling-only audit attributes stored squared-norm error to canonical generators, ordinary
gnomonic extraction, spherical fallback extraction, newly minted Local3d vertices, final diagram
generators/vertices, centroids, embedding projection, and cell-mesh copies. It also evaluates both
rounding orders from the producer's original f64 direction rather than normalizing the already
stored f32 value a second time.

Observed current-rule envelopes:

- canonical generators and f64-produced centroids stayed below `1.01e-7` absolute squared-norm
  error;
- ordinary final vertices reached `3.06e-7` on uniform data and `3.31e-7` on the perturbed
  great-circle stress, while fallback extraction reached `2.29e-7` on mega seed 3;
- no finite observed value exceeded four `f32::EPSILON`, `1e-6`, or the historical `1e-4`
  validation tolerance; and
- cell-mesh positions/source sites are exact copies of final diagram vertices/generators.
  Embedding projection, centroid, and Local3d minting already normalize in f64 and round once.
  The active repair-net fixtures reused existing vertices and therefore did not mint a new repair
  position during this campaign; the mint path's f64-normalized support-normal construction and
  exact stored-position unit regression remain the supporting evidence for that cold producer.

Replacing, rather than layering over, the ordinary and fallback f32 normalizations with
promoted-f64-normalize-then-round produced one common rule. Across Fibonacci, uniform, clustered,
mega, perturbed great-circle, and cubed-sphere runs, every measured stored value stayed below
`1.04e-7` absolute squared-norm error. The public contract should nevertheless use the
conservative bound `2 * f32::EPSILON`: it covers component rounding and f64 normalization error
without claiming that empirical maxima are a proof or that stored f32 directions are exactly unit.

Correctness and representation effects:

- the full release suite passed under the candidate, including adversarial, repair-net,
  output-resolution, embedding, cell-elision, and weird-geometry suites;
- ordinary/random/clustered/great-circle/mega topology fingerprints were unchanged, although many
  coordinate bits intentionally changed;
- the deliberately cocircular cubed-sphere grid selected a different valid degenerate resolution.
  Both results were strictly valid with the same 997 pre-reconciliation defects; sampled maximum
  vertex norm error improved from `1.15e-7` to `4.13e-8` and maximum vertex/edge cross-track error
  from `6.43e-8` to `4.03e-8`. Orphan counts moved from 1,063 to 1,084, so this is an intentional
  representation/output-policy migration rather than a bit-preserving cleanup; and
- the candidate changed roughly 47--53% of final vertex coordinates on the sampled ordinary and
  dense corpora. Vertex triple keys still provide shared identity; exact coordinate and
  output-resolution movement must remain pinned during the real migration.

Performance effects after ensuring each mode performs exactly one normalization:

- Cachegrind at 50k reported 424.14M versus 426.55M instruction references (`+0.57%`) and 38.72M
  versus 39.02M conditional branches (`+0.78%`);
- Linux `perf stat` over 3 x 5 builds at 500k reported `+0.61%` retired instructions and `+0.72%`
  branches. WSL cycles and wall time were noisy and are not decision evidence; and
- on the native Mac, paired 20-run single-threaded 1M Fibonacci trials averaged 988.3 ms for the
  current rule and 988.0 ms for the promoted-f64 rule. The difference is noise; there is no
  observed native wall-time regression. Earlier layered measurements that paid both square roots
  are discarded.

**Decision: choose Option B.** The audit removes the independent review's main objection: all
crate-produced spherical directions can share a useful, tight, enforceable finite norm envelope,
and the unified rule has acceptable correctness and performance behavior. A private-field
`SpherePoint` materially prevents arbitrary safe construction and invalid serde states while raw
arrays/closures preserve import interoperability. The type does not make query certificates
magically exact: locator bounds must still account for the documented envelope or normalize under
one consistent model in Stage 3.

## Import and export surface

Common interoperability surface after the representation decision:

- Provide `compute_by(&[T], F)` so any user record or math type works without an integration
  feature or orphan-rule conflict for foreign types. The closure writes directly into the
  backend-owned input buffer; it must not materialize an intermediate coordinate vector.
- Document the existing direct `&[[f32; 3]]` support prominently; a named `compute_xyz` wrapper may
  be added only if it improves discoverability enough to repay API duplication.
- Provide zero-copy `vertices_xyz()` and `generators_xyz()` views.
- Provide ordinary iterated conversions for user-owned output types. A generic
  `SphericalVoronoi<P>` is out of scope.
- Do not add mint initially. It is a compatible non-breaking convenience if real demand appears.
- Defer f64 adapters. The current owned pipeline would normalize an f64 adapter before rounding and
  then normalize the resulting `Vec3` again at `run_core_pipeline`; avoiding that requires a
  trusted-canonical entry or relocation of canonicalization. Also do not promise identical bits
  between f32 and f64 sources except where the input coordinates are exactly equivalent.
- Design fallible locator query normalization separately, including indexed batch errors; do not
  silently change the current infallible query contract inside the output-layout optimization.

Tightly packed `[f32; 3]` is a deliberate semver layout commitment. It is useful for CPU memory
density, C-style component buffers, and three-float GPU vertex attributes, but it is not a promise
that `Vec`/slice itself is C ABI safe or that every GPU storage/uniform layout uses a 12-byte stride.

## Staged implementation plan

### Stage 0 — measure and test a unified stored-direction contract (complete)

- Instrument every public/stored point producer separately: canonical generators, ordinary fast
  extraction, fallback extraction, reconciliation/Local3d repair vertices, centroids/Lloyd sites,
  embedding projection, and cell-mesh source sites/vertices.
- Across ordinary, dense, welded, adversarial, repair, and cell-elision corpora, record maximum
  `abs(length_squared - 1)` and counts beyond 1/2/4/8 f32 epsilons and the current `1e-4` bound.
- Prototype one shared storage canonicalizer at the producers' existing normalization points. Do
  not add a second normalization pass. Compare promoted-f64-normalize-then-round with the current
  round-then-f32-normalize path and select one rule only if its rounding/order contract is explicit.
- Compare topology, vertex keys, exact stored-coordinate equalities, output-resolution decisions,
  repair/reconciliation outcomes, validation, and geometric error. Exact coordinate bits may move
  in this experiment, but every change must be attributed; a tighter norm is not automatically a
  better diagram if it destabilizes shared stored equality or certified output policy.
- Measure instructions, branches, cache behavior, native wall time, and peak RSS. Centralization
  should replace existing work rather than charge one extra square root per emitted vertex.
- Decide whether the resulting common envelope is strong enough for locator ranking/bounds and
  whether a checked type removes concrete validation or misuse paths.

Gate result:

- **Selected B:** one enforceable storage rule covers all legitimate producers, the migration has
  acceptable correctness/performance behavior, and the type materially strengthens construction,
  serde, or query contracts.

### Stage 1 — apply the representation decision and remove the final point conversion

**Completed 2026-07-17.** The implementation introduced private-field packed `SpherePoint`
storage across diagrams, explicit cell meshes, centroids/Lloyd output, and embedding projection;
kept `UnitVec3`/`UnitVec3Like` as unchecked raw-input adapters; added checked exact-value serde;
and exposed zero-copy packed xyz views. All legitimate producers use the Stage-0
f64-normalize-then-f32-round contract. The final `Vec<Vec3>` allocations transfer ownership across
one audited layout boundary, with compile-time size/alignment checks, exact-bit/pointer/capacity
tests, a compile-fail private-construction test, and focused Valgrind Memcheck coverage.

The allocation benchmark corrected one premise of this stage: the prior same-layout
`map(...).collect()` was already optimized into in-place collection by the tested Rust toolchain.
At 2M single-threaded Fibonacci, both versions reported `assemble = 0.0 ms`, while peak RSS was
effectively identical (baseline 539,800--539,828 KiB; candidate 539,620--539,952 KiB). The explicit
ownership transfer therefore guarantees and audits reuse but does not produce an additional
observable allocation win on this toolchain.

The remaining performance movement matches the already accepted Stage-0 numerical migration. At
500k, three Linux `perf stat` runs measured +0.38% retired instructions and +0.05% branches; cycles,
cache misses, and wall time remained noisy. Cachegrind at 50k measured +0.70% instruction references
and +0.26% branches. On the native Intel Mac, 20 interleaved single-threaded 1M Fibonacci rounds
were 1697.3 ms median before and 1702.3 ms after (+0.29%, classified `~same`; spreads 2.0% and
3.2%). This is consistent with the Stage-0 decision and shows no native wall-time regression.

- Under A, keep `UnitVec3`, `UnitVec3Like`, serde shape, and every existing public signature
  unchanged. Replace the two elementwise `Vec3 -> UnitVec3` collections in
  `SphericalVoronoi::from_raw_parts` with a checked allocation cast.
- Under B, migrate every public producer/consumer together to private-field `SpherePoint`, use the
  Stage-0 canonicalizer for all safe construction and checked deserialization, and keep public
  arbitrary-bit construction unavailable. Encapsulate the layout-compatible final allocation
  transfer behind a small audited internal boundary.
- In either branch, require exact size/alignment compatibility with target-specific `glam::Vec3`,
  retain a safe elementwise fallback where appropriate, and add zero-copy `vertices_xyz()` and
  `generators_xyz()` views with exact-bit tests.
- Treat cell metadata separately; do not transmute the two layout-unspecified structs in this
  stage.

### Stage 2 — add closure-based ingest independently

**Completed 2026-07-17.** Added `compute_by`, `compute_with_by`, and
`compute_with_report_by` over one internal `collect_points_by` adapter. The adapter allocates only
the backend's required `Vec<Vec3>` and invokes the extractor exactly once per accepted input point;
the existing `UnitVec3Like` path now uses the same adapter. Inputs shorter than four points fail
before extraction, and backend validation retains the original input index. Exact-output API tests
cover topology, coordinates, configuration, reports, and validation summaries for a local foreign
record type.

The existing direct path remained effectively codegen-neutral. At 500k single-threaded Fibonacci,
five Linux `perf stat` runs changed retired instructions from 3,424,971,022 to 3,424,966,720
(-0.00013%) and branches from 382,102,522 to 382,101,932 (-0.00015%); cycles and wall time were
within run noise. Cachegrind at 50k measured +0.00112% instruction references and +0.00179%
branches. On the native Intel Mac, 20 interleaved single-threaded 1M Fibonacci rounds measured
1694.6 ms before and 1692.7 ms after, classified `~same` (spreads 9.9% and 3.6%).

- Add one internal f32 `collect_points_by` routine and minimal `compute_by` equivalents without
  duplicating the computational pipeline.
- Retain `UnitVec3Like`; downstream local records already work, while `compute_by` primarily solves
  foreign math-type/version interop.
- Update examples to show direct arrays plus one custom/foreign-record closure.
- Defer f64 adapters until there is a single-normalization pipeline design.

### Stage 3 — design locator validation as a correctness change

**Completed 2026-07-17.** `SphereLocator::locate` is now fallible and normalizes every finite,
nonzero query in f64 before one f32 rounding. `locate_many` applies the same rule through a
canonical query buffer and returns `IndexedSphereQueryError` with the lowest invalid slice index;
zero and non-finite values have explicit `SphereQueryError` variants. Already projected
world-space queries use the internal checked-point path and do not pay a second normalization.

The motivating mismatch is retained as a regression fixture. With 700 deterministic generators,
scaling query `[-0.022981111, 0.5619437, 0.82685614]` by 16 previously multiplied raw candidate
dots while shell bounds remained normalized. The search stopped after the first occupied shell and
returned generator 211 (dot 0.9913483) instead of generator 576 (dot 0.9990854). The scaled and
unit forms now use the same canonical query and return the same brute-force nearest generator.
This is a correctness/API change: it adds one f64 normalization per raw unit-space query and an
owned canonical-query buffer for batches, and makes no throughput-win claim.

- Specify fallible single-query and indexed batch errors for zero, non-finite, and out-of-contract
  query directions.
- Normalize/rank queries under one model and test the current raw-dot versus normalized-bound
  mismatch explicitly.
- Do not couple this API change to the output conversion benchmark or claim it as a throughput win.

### Stage 4 — revisit deferred interoperability only with new evidence

- Reopen trait retirement, f64 ingest, mint, or an arrays-only public surface only in response to a
  concrete interoperability requirement.

## Validation and acceptance

Correctness/API checks:

- Exact output topology, report, and work-counter agreement with the current API on ordinary,
  welded, adversarial, repair, and cell-elision fixtures. Under A, coordinate bits must also agree;
  under B, every intentional bit change from the unified canonicalizer must be classified and the
  new output-resolution/geometry result independently accepted.
- Producer-attributed norm-envelope tests, including repair and centroid paths rather than only
  ordinary construction.
- Exact-bit packed-array views and current serde round trips for diagrams and cell meshes.
- Under B, checked serde must reject rather than normalize invalid stored values, and compile-fail
  tests should cover private fields/withheld arbitrary-bit construction where useful.
- Cross-feature checks for default, `serde`, `glam`, and no-default-feature builds.
- Miri or an equivalent focused memory-safety check for the internal allocation ownership cast,
  plus compile-time size/alignment assertions on every supported target family. Miri cannot prove
  geometric provenance or semantic normalization.

Performance checks:

- Linux retired instructions/branches and Cachegrind for the Stage-0 canonicalizer and final diagram
  construction, separately attributed.
- Native Mac interleaved 2M Fibonacci and uniform runs for production total time and the assemble
  phase. Do not use the null-write percentage as the expected gain.
- Peak RSS at 2M to verify that the source/destination point-allocation overlap was actually
  removed.
- In Stage 2, `compute_by` must not allocate an intermediate coordinate vector or regress the direct
  array path materially.

## Explicit non-goals

- Removing glam from internal arithmetic.
- Making the diagram generic over point or scalar type.
- Claiming exact unit length for stored f32 directions.
- Adding a superficial `vertices_xyz_f64()` that only widens stored f32 values.
- Implementing the full f64 representation tracked as RESEARCH-004.
- Redesigning global vertex/cell indexing or sharded construction in the same change.
