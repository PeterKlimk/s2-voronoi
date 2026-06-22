# Engineering Findings

This document tracks broader codebase issues and technical debts as they are found during ongoing
work. It is not limited to publish/readiness blockers; it is meant to record serious correctness,
organization, and contract problems that should shape future refactors.

For release-oriented concerns, see `docs/publish-readiness-findings.md`.

## Current findings

### 1. Cells that extend beyond the generator hemisphere still need a complete fallback story

The current clipping path relies on a gnomonic projection centered at the generator. That
projection stops being valid once the cell extends beyond 90 degrees from the generator.

This is not a normal case for well-spread inputs, but it is a real algorithm boundary and needs an
explicit outcome rather than accidental failure. The explicit failure boundary now exists, and
`UnboundedAfterExhaustion` has a cold all-constraints spherical extractor for known
upper-hemisphere large-cell cases. The remaining gap is a complete fallback/model story for
projection-invalid cells and an early trigger that avoids walking the full neighbor stream before
taking the cold extractor.

Current evidence:

- `tests/adversarial.rs` pins upper-hemisphere cases as strict-success fixtures
- `tests/weird_geometry.rs` keeps pure rank-2 great-circle inputs as clean failures in strict mode

Current status:

- the builder now detects proven hemisphere/projection invalidity explicitly
- the public API returns `VoronoiError::UnsupportedGeometry` for that proven boundary
- unbounded-after-exhaustion hemisphere cells route through the cold all-constraints extractor
- fallback projection/model support for every projection-invalid case does not exist yet

Desired direction:

- keep the explicit failure boundary honest
- add an early trigger for the all-constraints extractor once a cell has enough evidence that the
  generator-centered chart is the wrong model
- route remaining projection-invalid cases to a fallback that can represent the cell, or keep
  returning a specific clean error

This should be treated as a correctness boundary, not as a best-effort corner case.

### 2. Public fallible API still does not fully match internal failure behavior

`compute` and `compute_with` return `Result`, but the backend still contains panic/expect paths for
unsupported or pathological inputs.

This overlaps the release-readiness doc, but it remains a core engineering issue because it blurs
the boundary between:

- invalid input
- unsupported geometry
- internal bug

Current status:

- several expected failure families now return real `Err`s:
  - `UnsupportedGeometry`
  - `RepresentationLimit`
  - `ComputationFailed`
- the remaining panic-only paths are much narrower and better diagnosed

Desired direction:

- convert expected unsupported/pathological outcomes into explicit internal failure states
- only reserve panics for invariant violations or true bugs

### 3. Validation semantics were too permissive for major semantic collapse

Historically, `validation::validate` could report output as effectively fine even when many cells
had collapsed into duplicates.

This is not just a documentation problem. It weakens the project's own ability to tell the
difference between:

- acceptable degraded output
- catastrophic semantic collapse

Current status:

- strict validation semantics were redesigned around subdivision/invariant failures
- duplicate-cell collapse is now a hard invalidity signal
- quality/fidelity work is now separated from strict validity

Desired direction:

- keep strict validation and quality reporting separate
- expand regression coverage around invalid-yet-nonpanic degraded outputs

### 4. Backend organization is better, but `knn_clipping` still carries too much phase coupling

The recent refactors improved the boundaries around:

- neighbor sourcing
- live dedup assembly
- edge reconciliation
- policy/heuristics

But `knn_clipping` is still a broad container, and some of the real phase boundaries are clearer in
practice than they are in the module structure.

Likely next organization target:

- make failure handling and phase ownership more explicit across:
  - neighbor sourcing
  - cell construction
  - live dedup / assembly
  - reconciliation
  - validation / error surfacing

### 5. Adversarial tests still only partially define the supported envelope

The adversarial corpus is useful, but several important cases are still:

- `#[ignore]`
- allowed to either succeed or fail
- framed as exploratory rather than as contract-defining regressions

Current status:

- many formerly observational adversarial cases are now explicit contract tests
- preprocess-aware report tests now pin the effective-vs-returned validation split
- a smaller set of ignored/diagnostic stress families remains on purpose

Desired direction:

- turn more adversarial cases into explicit contract tests
- especially around:
  - projection-invalid / >90 degree cells not covered by exhaustion fallback
  - near-degenerate shared-edge disagreement
  - panic-vs-error behavior

### 6. Density-based preprocessing can degrade tight clustered-cap inputs that otherwise validate

The default density-based preprocessing merges near-coincident generators before cell
construction. On tight clustered-cap fixtures, that merge step can turn an otherwise strictly valid
diagram into one with duplicate-cell collapse and related subdivision failures.

Historical evidence:

- `tests/validation.rs::test_validation_clustered_cap_tight_with_default_preprocessing_is_strictly_valid`
- `tests/validation.rs::test_validation_clustered_cap_tight_without_preprocess_is_strictly_valid`
- older revisions before the merge-policy reduction regressed this fixture by merging clustered points

Why this matters:

- the default "robust" mode is not just changing the solved problem abstractly; an overly
  aggressive threshold can materially worsen structural validity on some clustered inputs
- callers need observability and documentation around this tradeoff
- future preprocessing heuristics should be judged against strict validation, not just successful
  completion

Current status:

- the density-based merge fraction was reduced so this clustered-cap regression no longer triggers
  at low point counts
- preprocess-aware report validation still exists to surface effective-vs-returned validity when
  merges do happen

Desired direction:

- keep preprocessing effects observable
- revisit the density-based merge policy so it does not dominate small clustered regions so easily
- treat preprocessing as a separate policy surface with its own correctness tradeoffs, not as a
  transparent robustness improvement

Update (2026-06): finding #7 below establishes that the merge remap is the *sole* cause of strict
validation failures at 2–4M point counts, and the weld redesign in `docs/todo.md` P0 supersedes
the density-based policy entirely.

### 7. The large-count "bad edges" were the merge remap, not stitching (resolved interpretation)

The ignored 2–4M fuzz tests carried comments saying bad edges occur in that range. Re-running them
(2026-06) showed every validation failure is fully explained by the merge pre-filter's output
remapping: Euler exceeded 2 by exactly the duplicate-cell count on all 15 seed/scale combinations,
with overused edges ≈ one cell boundary per duplicate. With `PreprocessMode::Disabled`, the same
inputs validate strictly (modulo orphan vertices, finding #9).

Consequences:

- the intra/inter-bin correction system is clean at multi-million counts; no stitching defect
  exists in this regime
- the fuzz tests should be promoted to CI contract tests asserting strict validity once the weld
  rework (todo P0.1) lands; the stale comments should be deleted
- correctness issue, root cause in preprocessing remap semantics

### 8. Symmetric/seam positions are a real degenerate regime for near-coincident pairs

Ulp-scale pairs at exactly symmetric coordinates (axis poles, 45° points, cube corners — i.e.
cube-map seams and `TangentBasis` branch boundaries) produce catastrophic validation failures
(unpaired edges, multiple components), while the identical configuration under an arbitrary
rotation is strictly valid. Scattered clusters at the same seam centers are fine down to ~4e-8
separation, so the trigger is alignment, not proximity alone.

- suspected mechanism: the two twins select different tangent bases at the branch boundary and
  make divergent epsilon decisions in incompatible charts; unconfirmed
- practical relevance: axis-aligned and 45° points are common in real inputs (cube-sphere meshes,
  grids, hand-placed data)
- mitigations: the 1e-6 weld covers all constructions found so far (worst failure ~1.2e-7);
  the canonical-predicate refactor (todo P5) is the root fix
- full margin map in `docs/correctness-contract.md`; probes in `tests/coincidence_probes.rs`
- correctness issue (envelope boundary), currently fenced by the weld policy

### 9. Orphan vertices are an intentional representation choice that strict validation miscounts

Edge repair can drop the last reference to a vertex; compaction was deliberately skipped to
protect the latency budget. Result: single-digit unreferenced vertices at multi-million counts,
also reproducible at small counts with near-coincident pairs. They do not affect subdivision
topology (Euler already ignores them), but `validate` counts them as a subdivision issue — so the
crate's default output fails its own strict check on some seeds.

Resolved (2026-06): orphan vertices are now a representation note
(`ValidationReport::representation_notes`), not a subdivision issue, and
`SphericalVoronoi::compact_vertices()` removes them on demand. A possible refinement remains:
O(repairs) tracking during edge repair so compaction cost scales with defect count.

### 10. Single hard failure mode at the coincidence boundary: micro-cell `ClippedAway`

Every non-recoverable failure observed in the coincidence probes is `ClippedAway` of a cell
enclosed by sub-weld-radius neighbors — and any single-cell `ClippedAway` currently aborts the
entire computation. A NaN input component also surfaces through this path as a deep diagnostic
instead of an input-validation error.

Resolved (2026-06): non-finite inputs are rejected at compute entry with
`VoronoiError::InvalidInput { point_index, .. }`, and coincidence-driven `ClippedAway` is
classified as `DegenerateInput` naming the coincident generators (the degrade-and-continue
backstop idea proved unsound: neighbors already clipped against the missing cell's bisectors
would carry unpaired edges). Non-coincidence `ClippedAway` remains `ComputationFailed` (bug
class). See `docs/correctness-contract.md`.

## Working rules for new findings

When adding a new item here:

1. State the behavior concretely.
2. Record whether it is a correctness, organization, or contract issue.
3. Link the main source/test locations if they are already known.
4. State the desired end-state briefly.

This doc should stay high-signal. It is for serious items that should influence roadmap and
refactoring order, not for minor cleanup notes.

### 11. Grid resolution is tuned to one scenario

The cube-grid resolution is derived from a fixed target density (~16 points/cell from n, floor
res=4) that was tuned for large uniform inputs (the original 2.5M-point performance target) —
deliberately, and at the cost of large slowdowns on even mildly degenerate distributions:
clustered inputs produce mega-cells (degrading to O(k^2)-ish candidate work and, past 65k center
candidates, a bail to the slow path), and sparse/bimodal inputs pay chunking overhead tuned for
the wrong occupancy.

Desired direction (see todo P3.2): benchmark-derived target density across input sizes,
occupancy-feedback rebuild within a memory budget, and a shells-native big-cell path for
concentration beyond what global resolution can fix. Performance/robustness issue, root cause of
most "overfit to uniform" symptoms.

### 12. Packed-envelope reach shrinks with scale; takeover engagement grows sharply past ~2M

Measured 2026-06 (Ryzen 3600/WSL2, uniform, density 24): cells served by the takeover grow from
78 at 2M (0.004%) to 13,126 at 4M (0.33%) — a 168x jump for 2x points — driving mean
neighbors-before-termination from 8.45 to 12.25. Suspected cause: f32 threshold precision — at
4M, cell angular sizes (~1e-3 rad) push the security-threshold `1 - cos` values toward f32
epsilon, so more cells fail to prove safety within the packed stages. This is very likely why
the historical "bad edges" annotation named the 2-4M range specifically: that is where the
takeover and fallback paths started being exercised in volume.

Related: the shell takeover's per-layer certificate makes each takeover cell clip more than the
deleted cursor did (whole layers before the bound improves) — the granularity trade accepted at
the flip. Invisible at <=2M, ~5% of total clips at 4M uniform; re-measure at 8M+ and consider
f64 threshold computation or packed-stage extension if it grows.

Also measured: the density optimum is distribution-sensitive in the contrast direction — uniform
and clustered both peak at density 24, but bimodal (160x density contrast) monotonically prefers
lower densities (optimum <=12, with 24 costing ~5.7%). First quantified motivation for choosing
resolution from an occupancy-histogram percentile rather than the mean (pairs with the planned
big-cell path). Performance issue, scale/distribution envelope.

### 13. Edge-repair paths are now deterministically covered; defect detection is not bin-invariant

Context (2026-06): unresolved shared-edge mismatches occur at ~1-20 per multi-million uniform
run (none below ~2M; 2M seed 1 has one 3-defect site, seeds 2-4 have none), and the cross-bin
(Update, later in 2026-06: P5 stage-0 input canonicalization drove the natural 2M defect rate
to zero across ten seeds — the dominant defect source was the per-builder renormalization
asymmetry. The windowed net fixture remains the only known deterministic sphere defect source.)
detection/repair paths had never been deliberately exercised — a defect x bin-boundary
conjunction that uniform fuzzing essentially never samples, so a wrong repair could have hidden
indefinitely.

What was done:

- `ComputeReport::unresolved_edge_pairs` now reports each surviving mismatch with an
  `UnresolvedEdgeOrigin` naming the detection path (three in-bin cases, three overflow cases).
  Cold path, rare records; pure observability.
- `tests/edge_repair_net.rs` pins a real defect site (uniform 2M seed 1) via windowed
  extraction: a 10-mean-spacing cap plus sparse scaffold reproduces the defects at ~1.7k
  points. Scaffold size steers bin boundaries across the site (grid resolution is a function
  of n), yielding deterministic coverage of `InBinThirdsMismatch`, `InBinUnconsumedCheck`,
  `CrossBinSingleSided` (scaffold 280k), and `CrossBinThirdsMismatch` (scaffold 320k), each
  asserting strict post-repair validity.

Findings worth keeping:

- Synthetic degeneracy produces zero unresolved edges: exact quantized lattices, 1e-8
  cocircular rings, cube-vertex/great-circle/cap stress all resolve consistently. Exact ties
  resolve identically in both charts; defects need near-ties inside the narrow cross-chart
  rounding gap, found only by volume. Consequently the defect fixture is irreplaceable-by-
  construction and order/rotation sensitive (f32 re-rounding erases the gap).
- `InBinMissingCheck` (edge to earlier same-bin neighbor with no incoming check) appears
  unreachable by construction: such edges only enter a cell via replayed seeds, and a seed
  implies its check is present. The branch stays as a conservative repair route.
- Defect detection is NOT bin-invariant when a bin boundary cuts the defect site: same-bin
  pairs are clipped once and seed-forwarded while cross-bin pairs are clipped independently by
  both sides, so the epsilon disagreements legitimately differ (measured: 4 defects at bins=12
  vs 3 at bins=48, same input). The invariant that holds across layouts is strict validity
  after repair. This also means any future scheme treating the defect set as a canonical
  property of the input is unsound under the current two-chart evaluation; P5
  (consistency-by-construction) is what would make it canonical (and empty).
