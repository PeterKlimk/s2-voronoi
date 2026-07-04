# Roadmap

Plans past 0.1, roughly ordered. None of this is a compatibility promise; the
API is pre-1.0 and marked `#[non_exhaustive]` where evolution is expected.

## 0.2 candidates

**f64 input and output.** Inputs are currently rounded to f32 at entry, which
sets the resolution floor (~1.4e-6 chord weld radius, ~6e-8 vertex
quantization). The clipping math is already f64 and vertex identity is
combinatorial (generator triples), so a full-precision mode is a bounded
change, not a rewrite: keep the canonical generators in f64, keep the f32
copy as the spatial-search accelerator (the search layer only proposes
candidates — it never decides geometry), read f64 positions at the clip
gather, and store f64 vertices. Work items: a conservative pad on the
termination certificate to cover f32 mirror rounding, re-derived
weld/coincidence tolerances at the f64 floor, and an adversarial re-fuzz
(near-cocircular configurations get denser at f64 resolution).

**Weld as policy, not correctness (exact-duplicates-only mode).** Today
generators within a fixed radius are welded because sub-radius *clusters*
(k >= 3) can enclose a micro-cell that clips away entirely and breaks the
graph. But the failure is singular and loud, and the exact local repair
(default since 0.1) can in principle rebuild such neighborhoods with exact
predicates: let `ClippedAway` defer instead of abort, extend the repair
splice to insert a never-formed cell, and only exact bit-duplicates (which
have no defined bisector) still require merging. Output would contain
combinatorially-real epsilon cells with degenerate f32 geometry, so this is
an opt-in mode — the default weld stays, and welding/epsilon-collapse
becomes a post-pass policy choice rather than a correctness requirement.
Gate: the coincidence-probe corpus re-run with welding off and verification
on, asserting valid-or-error on every historical failure construction.

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
