# Fallback builder design note

This note defines the intended handoff between the current gnomonic cell builder and a future
non-hemispheric fallback builder for weird geometries.

## Goals

- Keep the current gnomonic path as the fast default for ordinary cells.
- Make unsupported-in-gnomonic cases explicit, structured, and local to the builder seam.
- Add a future fallback builder without forcing `cell_build/` to understand projection internals.

## Current state

`Topo2DBuilder` is now a wrapper around an internal builder implementation enum.

Today there is only one implementation:

- `GnomonicBuilder`

There is now also a minimal fallback stub behind the wrapper:

- `FallbackBuilder`

The stub does not implement alternate geometry yet. Its current purpose is to prove that wrapper
handoff can switch implementations, preserve replay state, and keep the rest of `cell_build/`
decoupled from gnomonic internals.

The wrapper also owns the first version of handoff policy:

- normal clip progress stays in the current builder
- `ProjectionInvalid` is classified as a fallback handoff request
- terminal builder failures such as `TooManyVertices` remain terminal

That policy is exposed internally through `BuilderClipOutcome` / `BuilderStepOutcome`, so the
future switch point is one explicit builder-owned decision instead of generic error bubbling.

The first handoff-transfer contract also exists now:

- fallback reconstructs accepted constraints from the current gnomonic accepted-plane state
- it uses accepted neighbor order, neighbor slots, and stored half-plane epsilon
- neighbor directions are recovered on demand from the current point set during handoff

## Handoff trigger

The first fallback trigger is:

- `ProjectionLimit`

That corresponds to the current `CellFailure::ProjectionInvalid` condition: the feasible region has
reached the generator hemisphere boundary, so the gnomonic model is no longer valid for the cell.

Non-goals for the first handoff trigger:

- neighbor-stream exhaustion
- clipped-away cells
- extraction invariant failures
- vertex-budget overflow

Those remain terminal outcomes unless a later design proves otherwise.

## Expected wrapper contract

The wrapper-level contract should stay conceptually like this:

1. Try clipping in the active builder implementation.
2. If the active implementation can continue, return normal clip progress.
3. If the active implementation proves it cannot represent the cell but another builder might,
   return a structured fallback request.
4. If the active implementation reaches a true terminal failure, return a terminal error.

That means `cell_build/` should react to one of three states:

- applied in current builder
- needs fallback handoff
- terminal failure

It should not need to know why gnomonic projection failed internally.

## State transfer

The fallback handoff should avoid sharing mutable clip state between implementations.

The safest first version is:

- keep the neighbor stream and attempted-neighbor tracking owned by `cell_build/`
- let the fallback builder reconstruct its own internal state
- replay already-accepted neighbor constraints in original neighbor order

Why replay first:

- the gnomonic clip polygon representation is implementation-specific
- extraction metadata is also implementation-specific
- replay is easier to verify than converting partially-built state across representations

Potential replay inputs:

- generator index and generator position
- accepted neighbor indices
- accepted neighbor slots
- optional edge-check epsilon when the accepted neighbor came from seeded edge checks

The current internal transfer is intentionally on-demand rather than persistently duplicated in the
steady-state gnomonic builder.

If replay cost becomes material, a later optimization can add a more compact transfer record.

## Failure taxonomy

The taxonomy should stay separated like this:

- `ProjectionInvalid`: builder-specific representational limit, eligible for fallback
- `UnboundedAfterExhaustion`: no bounded gnomonic polygon was ever formed. This is
  currently the terminal signal for the pinned pure-great-circle and upper-hemisphere
  fixtures (`tests/weird_geometry.rs::classify_weird_geometry_failures`). It is
  not handled by the existing handoff because the gnomonic polygon still contains
  bounding-box sentinel edges; a fallback for this class must reconstruct the cell
  directly from all accepted bisector constraints, not from a bounded gnomonic
  polygon.
- `TooManyVertices`: builder/resource failure; eligible for the bounded-polygon
  spherical fallback handoff.
- `ClippedAway`: terminal contradiction / invalid build path
- `NoValidSeed`: terminal extraction invariant failure

This separation matters because the public API should keep distinguishing:

- unsupported by the current geometry model
- exhausted without bounding
- internal contradiction or resource failure

## First fallback implementation

The first non-gnomonic fallback should optimize for correctness and containment, not speed.

Recommended constraints:

- internal-only behind `Topo2DBuilder`
- entered only from explicit wrapper handoff
- no attempt to share live clip state with `GnomonicBuilder`
- deterministic replay of accepted neighbors before consuming new neighbors

The current fallback covers bounded-polygon handoff/replay and spherical clipping
after `ProjectionInvalid` / `TooManyVertices` triggers. It does not cover
`UnboundedAfterExhaustion`; that requires an all-constraints spherical extraction
path that can start without a pre-existing bounded polygon.

## Open design questions

- What exact 2D or 3D representation should the fallback builder use?
- Do seeded edge-check clips need explicit replay metadata beyond `(neighbor_idx, slot, eps)`?
- Should repeated fallback-trigger conditions be cached to avoid reattempting gnomonic replay?
- When fallback succeeds, do we preserve the same extraction and live-dedup contracts unchanged?
- Should `UnboundedAfterExhaustion` route to a cold O(C³) all-pairs spherical
  halfspace-intersection extractor? This is the likely path for hemisphere cells:
  enumerate every pair of accepted bisector planes, keep the intersection
  directions satisfying all constraints, deduplicate, cyclically order around the
  generator, and emit the same `(generator, neighbor_a, neighbor_b)` vertex keys.

Until those are answered, the wrapper seam should remain small and explicit.

## Exhaustion probe notes

`UnboundedAfterExhaustion` is specifically "the neighbor stream ended while the
builder still had an unbounded gnomonic polygon." Exhaustion alone is not an
error: a cell can consume the whole stream, be bounded, extract vertices, and
return successfully.

The ignored unit probe
`probe_unbounded_exhaustion_neighbor_counts` prints direct single-cell build
statistics for representative cases:

```bash
cargo test --release probe_unbounded_exhaustion_neighbor_counts -- --ignored --nocapture
S2_PROBE_LARGE=1 cargo test --release probe_unbounded_exhaustion_neighbor_counts -- --ignored --nocapture
```

One run after the rank-2 perturbation work showed:

- `fib_100`: every cell succeeded; neighbors processed p50=9, max=12
- `fib_500`: every cell succeeded; p50=9, p99=39, max=45
- `great_circle_50`: every cell failed `UnboundedAfterExhaustion`; each processed
  49 neighbors
- `great_circle_jitter_50`: every cell succeeded but each processed 48 neighbors
- `hemisphere_100`: 96 cells succeeded, 4 failed; failing cells processed 99
  neighbors
- `hemisphere_500`: 496 cells succeeded, 4 failed; failing cells processed 499
  neighbors
- `latitude_ring_32` / `latitude_ring_64`: every cell succeeded, every cell
  exhausted the stream, and the two pole cells triggered polygon-cap fallback

With `S2_PROBE_LARGE=1`:

- `fib_2k`: every cell succeeded; neighbors processed p50=22, p99=28, max=29
- `great_circle_200`: every cell failed `UnboundedAfterExhaustion`; each processed
  199 neighbors
- `great_circle_jitter_200`: every cell succeeded; p50=170, p90=198, max=198
- `hemisphere_2k`: 1996 cells succeeded, 4 failed; failing cells processed 1999
  neighbors
- `latitude_ring_256`: every cell succeeded; nearly every cell exhausted the
  stream, and the two pole cells triggered polygon-cap fallback

Interpretation:

- ordinary full-sphere cells terminate very early
- rank-2 / near-rank-2 inputs are globally expensive because local termination
  cannot prove enough in the generator chart
- hemisphere failures are concentrated in a few cells, but those cells currently
  walk the whole stream before failing
- a future robust path should not wait for global exhaustion; it should enter the
  all-constraints spherical extractor once a cell remains unbounded after enough
  evidence that the generator-centered chart is the wrong model
