# Roadmap

Plans past 0.1, roughly ordered. None of this is a compatibility promise; the
API is pre-1.0 and marked `#[non_exhaustive]` where evolution is expected.
Uncommitted caller-facing ideas remain in the
[feature and API wishlist](docs/feature-api-wishlist.md) until they are promoted here.

## 0.2 candidates

**f64 input and output (deferred TODO).** Inputs are currently rounded to f32
at entry, which sets the resolution floor (~1.4e-6 chord weld radius, ~6e-8
vertex quantization). The fallback and much of the clipping math are already
f64 and vertex identity is combinatorial, so this is substantial but not a
rewrite. Add parallel `UnitVec3d` / `SphericalVoronoi64` storage and f64 compute
entry points; carry f64 through clipping, repair, validation, measures, and
location; define an exact-duplicate policy and no-weld behavior for distinct
generators. The kNN/grid path needs its own sound f64 certificate: a coarse f32
mirror is acceptable only if bin coverage and unseen bounds conservatively
cover its quantization, because search termination does decide which
constraints reach geometry. Start scalar, differential-test against an
independent certified reference under the selected SoS policy and the f32
backend, establish multi-regime baselines, then decide whether a four-wide
packed f64 kNN path is worth maintaining. Do not ship an f64 API that silently
rounds inputs or stored vertices to f32.

**Weld as policy, not correctness (exact-duplicates-only mode).** The 0.1
fallback now lets a feasible `ClippedAway` cell replay its constraints in f64
and resume construction. The remaining limit is f32 output topology: sufficiently
tight multi-point clusters can still round distinct epsilon features together,
while exact bit-duplicates have no defined bisector at all. Keep default welding
for 0.1. Revisit exact-duplicates-only mode with f64 storage; gate it on the full
coincidence-probe corpus with welding off and strict verification on.

**Scaled-sphere convenience.** Voronoi structure on a sphere is purely
angular, so radius/center support is pre/post scaling: normalize inputs
(center subtraction in f64), scale vertices by R and areas by R² on output.
A thin wrapper plus a documented note that the weld radius is angular
(about 9 m on an Earth-sized sphere).

**f64 input acceptance** (`[f64; 3]`, `glam::DVec3` impls of the input
trait) may land earlier than the full f64 mode — it removes a conversion
paper cut even while storage stays f32.

## Later

- **Power (Laguerre) diagrams.** Weighted sites keep great-circle bisectors
  on the sphere (Sugihara), so the clipping core carries over; the hard
  parts are weight-aware termination certificates and the empty-cell
  contract.
- **Python bindings.** A pip-installable wrapper with scipy-convention
  output (`SphericalVoronoi`-style regions/vertices).
- **Examples.** A planet-generation example and a Bevy integration demo.
- **Planar and toroidal domains.** A bounded-rectangle and periodic (torus)
  backend over the same dedup engine exists on a parked branch and may
  return as a separate crate or feature once the sphere surface is stable.

## Non-goals (for now)

- Exact vertex positions (exact *combinatorics* are the contract; positions
  are floating-point best-effort).
- Ellipsoidal or geodesic-distance domains.
- GPU execution — the per-cell construction is GPU-friendly by heritage,
  but this crate targets CPUs; the stitched shared-vertex output is its
  reason to exist.
