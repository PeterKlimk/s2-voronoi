# Engineering Findings

This document tracks broader codebase issues and technical debts as they are found during ongoing
work. It is not limited to publish/readiness blockers; it is meant to record serious correctness,
organization, and contract problems that should shape future refactors.

For release-oriented concerns, see `docs/publish-readiness-findings.md`.

## Current findings

### 1. Cells that extend beyond the generator hemisphere still need a fallback story

The current clipping path relies on a gnomonic projection centered at the generator. That
projection stops being valid once the cell extends beyond 90 degrees from the generator.

This is not a normal case for well-spread inputs, but it is a real algorithm boundary and needs an
explicit outcome rather than accidental failure. The explicit failure boundary now exists; the
remaining gap is what the longer-term fallback/model story should be.

Current evidence:

- `tests/adversarial.rs` already documents the limitation in the great-circle and hemisphere cases
- `test_hemisphere_basic` and related ignored tests exercise this family of failures

Current status:

- the builder now detects proven hemisphere/projection invalidity explicitly
- the public API returns `VoronoiError::UnsupportedGeometry` for that proven boundary
- fallback projection/model support does not exist yet

Desired direction:

- keep the explicit failure boundary honest
- then either:
  - continue failing cleanly with a real error, or
  - route to a fallback that uses a projection/model that can represent the cell

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
  - hemisphere / >90 degree cells
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
- full margin map in `docs/correctness-contract.md`; probes in `tests/tmp_ulp_regimes.rs`
- correctness issue (envelope boundary), currently fenced by the weld policy

### 9. Orphan vertices are an intentional representation choice that strict validation miscounts

Edge repair can drop the last reference to a vertex; compaction was deliberately skipped to
protect the latency budget. Result: single-digit unreferenced vertices at multi-million counts,
also reproducible at small counts with near-coincident pairs. They do not affect subdivision
topology (Euler already ignores them), but `validate` counts them as a subdivision issue — so the
crate's default output fails its own strict check on some seeds.

Desired direction: reclassify as a representation note and/or add O(repairs) targeted compaction
(track orphaned indices during repair; patch only affected cells). Contract issue.

### 10. Single hard failure mode at the coincidence boundary: micro-cell `ClippedAway`

Every non-recoverable failure observed in the coincidence probes is `ClippedAway` of a cell
enclosed by sub-weld-radius neighbors — and any single-cell `ClippedAway` currently aborts the
entire computation. A NaN input component also surfaces through this path as a deep diagnostic
instead of an input-validation error.

Desired direction: O(n) finite-input check with index-bearing error; backstop that emits a
degenerate/welded cell plus report entry when the clipped-away cell's constraints are all
sub-weld-radius (todo P0.3/P0.4). Correctness/contract issue.

## Working rules for new findings

When adding a new item here:

1. State the behavior concretely.
2. Record whether it is a correctness, organization, or contract issue.
3. Link the main source/test locations if they are already known.
4. State the desired end-state briefly.

This doc should stay high-signal. It is for serious items that should influence roadmap and
refactoring order, not for minor cleanup notes.
