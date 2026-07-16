# Point representation and interoperability plan

**Status:** independently reviewed; Option A is the recommended first design.

This plan records the public-point decision around the measured output-materialization
optimization. It compares whether a checked semantic `SpherePoint` earns its additional API and
validation machinery over a simpler packed coordinate type, then stages the independently reviewed
Option A result. The decision is separate from the established choice to retain
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

## Independent review and decision

A conversation-blind review ranked the alternatives **A > B > C**. The decisive issue is that
`SpherePoint` would not currently establish one useful invariant across all of its proposed
producers. A weak common envelope would not remove checks or make raw locator queries safe; a
strong generator-grade envelope would require changing output, centroid, repair, serde, and query
contracts before the measured conversion optimization could land.

Proceed with **Option A**:

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

## Import and export surface

Under Option A:

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

### Stage 1 — remove only the redundant final point conversion

- Keep `UnitVec3`, `UnitVec3Like`, serde shape, and every public signature unchanged.
- Replace the two elementwise `Vec3 -> UnitVec3` collections in
  `SphericalVoronoi::from_raw_parts` with a checked internal allocation cast.
- Require exact size/alignment compatibility and retain a safe elementwise fallback for a supported
  target whose glam layout differs. The proof depends on both `UnitVec3` and the target-specific
  `glam::Vec3` layout.
- Add zero-copy `vertices_xyz()` and `generators_xyz()` views using the existing POD layouts.
- Treat cell metadata separately; do not transmute the two layout-unspecified structs in this
  stage.

### Stage 2 — add closure-based ingest independently

- Add one internal f32 `collect_points_by` routine and minimal `compute_by` equivalents without
  duplicating the computational pipeline.
- Retain `UnitVec3Like`; downstream local records already work, while `compute_by` primarily solves
  foreign math-type/version interop.
- Update examples to show direct arrays plus one custom/foreign-record closure.
- Defer f64 adapters until there is a single-normalization pipeline design.

### Stage 3 — design locator validation as a correctness change

- Specify fallible single-query and indexed batch errors for zero, non-finite, and out-of-contract
  query directions.
- Normalize/rank queries under one model and test the current raw-dot versus normalized-bound
  mismatch explicitly.
- Do not couple this API change to the output conversion benchmark or claim it as a throughput win.

### Stage 4 — reconsider representation only with new evidence

- Audit norm envelopes for every generator, vertex, repair, centroid, Lloyd, cell-mesh, embedding,
  serde, and locator producer.
- Reopen Option B only if one enforceable invariant removes meaningful downstream validation or
  prevents a demonstrated misuse class.
- Reopen trait retirement, f64 ingest, mint, or an arrays-only public surface only in response to a
  concrete interoperability requirement.

## Validation and acceptance

Correctness/API checks:

- Exact output topology, coordinate-bit, report, and work-counter agreement with the current API on
  ordinary, welded, adversarial, repair, and cell-elision fixtures.
- Exact-bit packed-array views and current serde round trips for diagrams and cell meshes.
- Cross-feature checks for default, `serde`, `glam`, and no-default-feature builds.
- Miri or an equivalent focused memory-safety check for the internal allocation ownership cast,
  plus compile-time size/alignment assertions on every supported target family. Miri cannot prove
  geometric provenance or semantic normalization.

Performance checks:

- Linux retired instructions/branches and Cachegrind for final diagram construction.
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
