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

The first replay-transfer contract also exists now:

- `Topo2DBuilder::fallback_replay_plan()` returns accepted replay constraints in builder order
- each replay entry carries `neighbor_idx`, `neighbor_slot`, optional `hp_eps`, and the neighbor direction
- edgecheck-seeded constraints preserve their incoming epsilon for later deterministic replay

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

The current internal replay payload is intentionally limited to exactly that data.

If replay cost becomes material, a later optimization can add a more compact transfer record.

## Failure taxonomy

The taxonomy should stay separated like this:

- `ProjectionInvalid`: builder-specific representational limit, eligible for fallback
- `UnboundedAfterExhaustion`: no proof of unsupported geometry, not a fallback trigger by itself
- `TooManyVertices`: terminal builder/resource failure
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

The current stub only covers the handoff/replay portion of that plan. It still reports the same
terminal `ProjectionInvalid` outcome after handoff.

## Open design questions

- What exact 2D or 3D representation should the fallback builder use?
- Do seeded edge-check clips need explicit replay metadata beyond `(neighbor_idx, slot, eps)`?
- Should repeated fallback-trigger conditions be cached to avoid reattempting gnomonic replay?
- When fallback succeeds, do we preserve the same extraction and live-dedup contracts unchanged?

Until those are answered, the wrapper seam should remain small and explicit.
