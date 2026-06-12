# Supported Envelope

This document defines the current computation contract for the primary kNN + clipping backend.

It is intentionally narrower than "all spherical Voronoi diagrams". The implementation supports a
large practical subset of inputs, but it still has explicit geometric and representation limits.

For the conceptual contract (the hard-topological / soft-geometric split, the coincident-input
weld policy, and the measured margins behind the envelope boundaries), see
`docs/correctness-contract.md`.

## Outcome Classes

The backend currently has four distinct outcome classes.

### 1. Supported Success

The computation returns a diagram and the result is within the currently supported model.

Expected properties:

- `compute` / `compute_with` / `compute_with_report` return `Ok(...)`
- with preprocessing disabled, the returned diagram should be a strictly valid S2 subdivision
- with preprocessing enabled and no merges, the same should hold
- with preprocessing enabled and merges occurring, the **effective** diagram actually solved by the
  backend should be strictly valid

This is the normal successful contract.

### 2. Supported Failure

The computation returns a defined `Err(...)` for an input class or limit that the backend
recognizes explicitly.

Current public failure classes:

- `VoronoiError::UnsupportedGeometry`
  - used for proven geometric/model limits, not generic failure
  - current example: a cell reaches the generator hemisphere boundary, so the gnomonic model is no
    longer valid
- `VoronoiError::RepresentationLimit`
  - used for concrete storage/layout/indexing limits
  - current examples:
    - packed `(bin, local)` layout overflow
    - global assembled/remapped/reconciled index-buffer overflow
    - assembled/deferred-fallback vertex-id overflow in the `u32`-backed diagram model
    - total generator count exceeding the backend's `u32`-backed identifier model
- `VoronoiError::ComputationFailed`
  - used for terminal failure states that are understood operationally but are not yet promoted to
    a narrower public class
  - current examples:
    - unbounded-after-exhaustion
    - clipping vertex-budget exhaustion

These are considered part of the supported contract. They should fail cleanly without panic.

### 3. Preprocessing-Altered Solved Problem

When preprocessing merges near-coincident generators, the backend solves a different effective
generator set and may then remap the result back to the original input count.

This is not the same contract as a strict Voronoi solve over the original input set.

Current behavior:

- `compute` returns the remapped/original-count diagram
- `compute_with_report` also exposes:
  - `effective_diagram`
  - `returned_validation`
  - `effective_validation`
  - `preferred_diagram()`
  - `preferred_validation()`

Interpretation:

- if merges did not occur, returned and effective views coincide
- if merges occurred, the effective view is the one the backend actually solved
- the returned remapped diagram is still useful as an original-input convenience view, but it is
  not the primary correctness view for the solved problem once merges occur
- strict validity of the returned remapped diagram is not guaranteed to mean the same thing as
  strict validity of the effective solved problem

This is an intentional robustness tradeoff, not a hidden invariant.

### 4. Invariant Failure / Bug

Some internal states are still treated as bugs rather than supported input outcomes.

Examples:

- `ClippedAway` (only when no sub-weld coincidence explains it)
  - update (2026-06): this state is input-reachable with welding disabled — clusters of 3+
    generators below the weld radius can clip an enclosed micro-cell to empty. That
    coincidence-driven case is now classified as a supported failure
    (`VoronoiError::DegenerateInput` naming the coincident generators); a `ClippedAway` with no
    coincident generators remains the bug class (`ComputationFailed`). See
    `docs/correctness-contract.md`.
- `NoValidSeed`
- extraction metadata / reconstruction invariant failures inside `topo2d::builder`
- internal stream-state contradictions
- internal clipper assumptions that should hold if the algorithm's invariants are respected

These cases may still panic. They are not considered part of the supported public failure
contract.

## Current Supported Geometry Envelope

### Inputs expected to succeed

The backend is intended to support:

- unit-length finite input points on `S²`
- non-degenerate point sets large enough to form a Voronoi subdivision
- many practical near-degenerate configurations, including:
  - small-jitter great-circle-like inputs
  - clustered-cap inputs with reasonable anchoring
  - many near-cocircular and edge-reconciliation cases

This is reflected in the always-on contract tests in [`tests/adversarial.rs`](/home/pkzmbk/code/s2-voronoi/tests/adversarial.rs).

### Inputs expected to fail cleanly

The backend is expected to return a defined `Err(...)` for:

- cells that require the clipped feasible region to reach or exceed the generator hemisphere
  boundary in the current gnomonic model
- extreme scale/layout cases exceeding current packed/indexing capacity
- cells whose intermediate clip polygon exceeds the 64-vertex budget
  (`MAX_POLY_VERTICES`): a cell genuinely bordering more than ~62 others.
  First concretely reached (2026-06) by a 1M-point cap of angular radius
  0.05 with 6 anchor generators — each anchor's cell borders the entire cap
  rim. Returns the vertex-budget `ComputationFailed`; the bound is a
  representation choice, not a numerical failure
- some terminal construction failures that are recognized but not yet classified more precisely

Pure great-circle / coplanar cases are the clearest current example of supported failure rather
than supported success.

### Inputs still treated as outside the explicit contract

Some stress inputs are still exploratory rather than pinned as contract success/failure.

These remain in the adversarial suite as diagnostics or ignored stress runs rather than stable
regression-oracle cases.

## Validation Semantics

Strict validation and fidelity are intentionally separate.

- [`validation::validate`](/home/pkzmbk/code/s2-voronoi/src/validation.rs) checks strict
  subdivision validity plus exact invariants of the produced object
- [`quality`](/home/pkzmbk/code/s2-voronoi/src/quality.rs) tracks fidelity/quality metrics and is
  not part of strict validity

Important consequence:

- with preprocessing disabled, strict validation of the returned diagram is the primary success
  check
- with preprocessing merges, strict validation of the **effective** diagram is usually the right
  correctness view for the solved problem
- when using `compute_with_report`, `output.preferred_diagram()` and
  `report.preferred_validation()` are the intended "check this first" entry points after a merged
  solve

## Current Failure Taxonomy

The current intended split is:

- `UnsupportedGeometry`
  - only for proven unsupported geometric/model boundaries
- `RepresentationLimit`
  - only for concrete implementation/storage/layout limits
- `ComputationFailed`
  - for known terminal failure states that are still broader than the two classes above
- panic
  - only for internal contradictions still believed to indicate a bug

This split is intentionally conservative. The crate should not broaden `Result` classifications
unless it can state the failure reason honestly and specifically.

## Near-Term Follow-Ups

The most likely future refinements to this contract are:

- tightening which adversarial families are pinned as expected success vs expected failure
- clarifying whether `UnboundedAfterExhaustion` can be narrowed beyond `ComputationFailed`
- adding fallback strategies for currently unsupported geometry/workspace limits rather than only
  failing cleanly
- tightening the remaining bug-only extraction/reconciliation invariants without broadening the
  public error taxonomy prematurely
