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
  still the terminal signal for pinned pure-great-circle fixtures in strict mode.
  Upper-hemisphere fixtures now route through a cold all-constraints spherical
  extractor after exhaustion. That path reconstructs the cell directly from all
  accepted bisector constraints, not from a bounded gnomonic polygon.
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
after `ProjectionInvalid` / `TooManyVertices` triggers. It also has a cold
`UnboundedAfterExhaustion` backstop that starts without a pre-existing bounded
polygon and extracts from all accepted constraints.

## Remaining design questions

- What exact 2D or 3D representation should the fallback builder use?
- Do seeded edge-check clips need explicit replay metadata beyond `(neighbor_idx, slot, eps)`?
- Should repeated fallback-trigger conditions be cached to avoid reattempting gnomonic replay?
- When fallback succeeds, do we preserve the same extraction and live-dedup contracts unchanged?
- When should the unbounded fallback trigger before full neighbor-stream
  exhaustion?

Until those are answered, the wrapper seam should remain small and explicit.

## All-constraints extractor

For `UnboundedAfterExhaustion`, production now routes the accepted bisector
constraints through a cold all-pairs spherical halfspace extractor: enumerate
every pair of accepted planes, keep the two intersection directions that satisfy
all accepted constraints, deduplicate, cyclically order around the generator,
and derive each vertex key as `(generator, neighbor_a, neighbor_b)`. Adjacent
vertices must share a constraint, producing the edge-neighbor cycle needed by
the normal output path.

That extractor reconstructs the upper-hemisphere exhaustion cells:

- `hemisphere_100`: 4 cells use all-constraints extraction and emit up to 7 edges
- `hemisphere_500`: 4 cells use all-constraints extraction and emit up to 9 edges
- `hemisphere_2k`: 4 cells use all-constraints extraction and emit up to 11 edges

It intentionally does not solve exact rank-2 great-circle inputs: those have no
full-dimensional spherical cell under the unperturbed constraints, so the
all-pairs extractor finds no 3+ vertex polygon. The default rank-2 perturbation
mode is the right route for that class; `DegenerateMode::Strict` preserves the
clean-error behavior for callers that want it.

## Early-trigger probe notes

The ignored unit probe `probe_early_all_constraints_trigger_points` checkpoints
unbounded cells during stream consumption. At each checkpoint it runs the cold
all-constraints extractor, records the first checkpoint that emits any cell, and
records the first checkpoint whose emitted vertex/edge-neighbor signature matches
the completed final cell.

```bash
cargo test --release probe_early_all_constraints_trigger_points -- --ignored --nocapture
S2_PROBE_LARGE=1 cargo test --release probe_early_all_constraints_trigger_points -- --ignored --nocapture
```

The key result is that "all-constraints extraction returned a polygon" is not a
safe production trigger by itself:

- `fib_2k`: 784 cells emitted an early all-constraints polygon while still
  unbounded, but none matched the final cell.
- `great_circle_jitter_200`: 43 cells emitted an early polygon, but none matched
  the final cell.
- `latitude_ring_256`: 256 cells emitted early polygons, but none matched the
  final cell.
- `hemisphere_2k`: 492 cells emitted early polygons; only the 4 eventual
  all-constraints fallback cells had a final match.

For the upper-hemisphere fallback cells, the first final-match checkpoint still
arrived late:

- `hemisphere_100`: 4 final matches; first-match neighbors min=66, max=99
- `hemisphere_500`: 4 final matches; min=336, max=464
- `hemisphere_2k`: 4 final matches; min=1349, max=1863

Interpretation: early all-constraints extraction is easy to make valid but hard
to make correct. A production early trigger needs an additional certificate that
later unseen constraints cannot cut the extracted cell, or a targeted query that
checks the extracted vertices/edges against the remaining spatial domain. Simple
thresholds based on "extractor succeeded", accepted-constraint count, or emitted
edge count are not enough.

### Large-N targeted probe

The all-cells early-trigger probe is useful at thousands of points, but the
million-point concern is concentrated in a few carrier cells. The ignored
`probe_large_hemisphere_target_cells` test builds a large upper-hemisphere input
and probes selected cells only. Configure it with:

```bash
S2_PROBE_N=1000000 cargo test --release probe_large_hemisphere_target_cells -- --ignored --nocapture
S2_PROBE_N=1000000 S2_PROBE_TARGETS=0,1,2,3,4 cargo test --release probe_large_hemisphere_target_cells -- --ignored --nocapture
```

One `S2_PROBE_N=1000000` run showed:

- cells `0..11` all processed `999999` neighbors
- cells `0..4` had final edges `16,16,17,17,17`
- cells `0..4` all emitted early polygons after only `20..25` neighbors, but
  those early polygons were not the final cell
- cells `0..3` first matched the final all-constraints signature only at
  exhaustion; cell `4` first matched at `505188` neighbors
- control cells at indices `250000`, `500000`, and `750000` processed `45..53`
  neighbors; the final cap-near cell `999999` processed `5659`

Interpretation: at million scale, the cold fallback itself is not the cost
center; the carrier cells still pay the full neighbor-stream proof. The final
cells remain small, so a useful optimization should certify the extracted
spherical cell against the remaining spatial frontier rather than continuing to
enumerate every neighbor.

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
- `hemisphere_100`: every cell succeeds; 4 cells exhaust the stream and take the
  all-constraints extractor after processing 99 neighbors
- `hemisphere_500`: every cell succeeds; 4 cells exhaust the stream and take the
  all-constraints extractor after processing 499 neighbors
- `latitude_ring_32` / `latitude_ring_64`: every cell succeeded, every cell
  exhausted the stream, and the two pole cells triggered polygon-cap fallback

With `S2_PROBE_LARGE=1`:

- `fib_2k`: every cell succeeded; neighbors processed p50=22, p99=28, max=29
- `great_circle_200`: every cell failed `UnboundedAfterExhaustion`; each processed
  199 neighbors
- `great_circle_jitter_200`: every cell succeeded; p50=170, p90=198, max=198
- `hemisphere_2k`: every cell succeeds; 5 cells exhaust the stream, and 4 of
  those take the all-constraints extractor after processing 1999 neighbors
- `latitude_ring_256`: every cell succeeded; nearly every cell exhausted the
  stream, and the two pole cells triggered polygon-cap fallback

Interpretation:

- ordinary full-sphere cells terminate very early
- rank-2 / near-rank-2 inputs are globally expensive because local termination
  cannot prove enough in the generator chart
- hemisphere exhaustion is concentrated in a few cells, and those cells
  currently walk the whole stream before the cold extractor succeeds
- a future robust path should not wait for global exhaustion; it should enter the
  all-constraints spherical extractor once a cell remains unbounded after enough
  evidence that the generator-centered chart is the wrong model

## Projection fallback probe notes

`ProjectionInvalid` is still structurally supported as a bounded-polygon handoff:
when a changed gnomonic polygon is already bounded but reaches the projection
radius limit, the builder enters the spherical fallback and replays accepted
constraints. The ignored `probe_projection_fallback_cases` test checks whether
natural cap-only and anchored-cap fixtures still exercise that path:

```bash
cargo test --release probe_projection_fallback_cases -- --ignored --nocapture
```

One sweep over cap-only Fibonacci inputs, cap-plus-antipode inputs, and
octahedron-anchored caps at radii `0.1`, `0.5`, `1.0`, and `1.5` radians found:

- all cases succeeded
- `fallback_projection=0` in every case
- cap-only cases used `fallback_all_constraints` on boundary cells instead
- antipode-anchored and octahedron-anchored cases avoided all-constraints fallback
  in the probed sizes, though some cells still processed almost the whole stream

Interpretation: the currently observable large-cell weird geometries do not
naturally hit the bounded `ProjectionInvalid` handoff. They either remain
unbounded until exhaustion, or they are bounded enough for ordinary extraction.
Projection fallback remains useful as a representation guard and forced-handoff
test path, but the practical robustness/performance issue is still
`UnboundedAfterExhaustion` and proof of irrelevance for unseen constraints.
