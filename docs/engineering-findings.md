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

## Working rules for new findings

When adding a new item here:

1. State the behavior concretely.
2. Record whether it is a correctness, organization, or contract issue.
3. Link the main source/test locations if they are already known.
4. State the desired end-state briefly.

This doc should stay high-signal. It is for serious items that should influence roadmap and
refactoring order, not for minor cleanup notes.
