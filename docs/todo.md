# TODO / Roadmap

This is the active working roadmap for `s2-voronoi`.

It is intentionally narrower than a vision statement and more actionable than
the engineering findings documents. If you want to know "what should we work on
next?", start here.

## Document Role

- This document supersedes the previous contents of `docs/todo.md`.
- This document is now the primary roadmap / prioritization entry point.
- `docs/engineering-findings.md` remains the place to record important
  correctness, contract, and organization problems.
- `docs/supported-envelope.md` remains the current public/backend contract.
- `docs/plan.md` is no longer the general roadmap; it is now a narrower design
  note for the packed `r=2` expansion work that has already largely landed.

## Guiding Goal

Build an open-source crate that is:

- high performance on practical S2 Voronoi workloads
- honest about its supported geometry and failure modes
- maintainable enough to keep optimizing without losing correctness
- robust in the sense of returning a valid graph or a clean error, rather than
  panicking or silently collapsing semantics

## Priority Order

## P0: Finish the non-panicking library contract

- Audit remaining panic-only paths in cell construction, topology extraction,
  reconciliation, and neighbor stream state.
- Convert user-reachable or stress-reachable terminal states into structured
  internal failures that map to `VoronoiError`.
- Keep panic reserved for true invariant bugs, not merely unsupported or
  pathological inputs.
- Add regression coverage for any panic that gets converted into an explicit
  error.

Why this is first:
An open-source geometry crate can survive a narrow supported envelope. It should
not routinely rely on panic as part of that envelope.

## P1: Keep the supported envelope explicit and test-backed

- Continue defining which adversarial families are supported success versus
  supported failure.
- Tighten cases that are still broad `ComputationFailed` buckets when repro
  evidence shows a sharper classification is possible.
- Keep hemisphere / >90 degree cells honest as an explicit algorithm boundary
  until a real fallback exists.
- Turn more adversarial cases into contract tests where the expected outcome is
  stable enough to pin.
- Add panic-vs-error coverage for the important edge families.

Why this is next:
The crate does not need to support every spherical Voronoi configuration yet,
but it does need a crisp and trustworthy contract.

## P2: Preserve strict validity as a hard gate

- Keep `validation::validate` focused on strict subdivision and exact invariant
  checks.
- Keep `quality` metrics separate from strict validity and do not blend the two
  back together.
- Add more regression coverage around invalid-but-nonpanic degraded outputs,
  especially duplicate-cell collapse and shared-edge disagreement.
- Treat "returned diagram is strictly valid" and "effective solved diagram is
  strictly valid" as distinct checks when preprocessing merges generators.

Why this matters:
Performance work is only useful if the project can still tell the difference
between acceptable degradation and a broken graph.

## P3: Keep preprocessing observable and policy-driven

- Keep preprocessing effects explicit in the public API and reports.
- Revisit the density-based merge policy carefully, especially for small tight
  clusters where merge behavior can dominate the solved problem.
- Judge preprocessing changes against strict validation outcomes, not just
  successful completion or speed.
- Consider whether the current default should remain "robust mode" long term or
  whether stricter and repaired entry points should eventually diverge.

Why this matters:
Preprocessing is useful, but it changes the solved problem. The crate should
continue treating that as a policy surface, not invisible magic.

## P4: Continue performance work through the policy layer

- Keep heuristic decisions centralized in `src/policy.rs`.
- Revisit packed `r=2` expansion activation heuristics with timing-driven
  comparisons, especially on `fibonacci+jitter` versus `--lloyd`.
- Add a small documented profiling baseline matrix for common benchmark
  distributions.
- Add policy-level tests that pin intended stage exposure and counter surfaces
  without trying to make CI enforce wall-clock numbers.
- Benchmark the frontier-certificate stream contract before doing more query
  path cleanup, so simplification decisions stay evidence-driven.

Why this matters:
The project already has the right optimization structure. The main risk now is
heuristic sprawl, not lack of low-level tuning ideas.

## P5: Reduce phase coupling in the backend

- Keep making phase ownership clearer across:
  - neighbor sourcing
  - cell construction
  - live dedup / assembly
  - edge reconciliation
  - error surfacing / validation
- Avoid abstraction churn that does not improve ownership or failure handling.
- Prefer refactors that reduce "who owns this transition?" ambiguity.

Why this matters:
The codebase is much healthier than before, but `knn_clipping` still carries
more cross-phase responsibility than is ideal for long-term maintenance.

## P6: Revisit the long-term geometry fallback story

- Decide whether cells extending beyond the generator hemisphere should remain a
  clean supported failure or eventually route to a different representation /
  projection model.
- If a fallback is explored, define success criteria up front:
  - preserve valid graph output
  - avoid contaminating the hot path for normal cases
  - keep error boundaries honest when the fallback also cannot represent a case

Why this is later:
This is important, but it is a larger algorithmic change than the current
hardening work. The crate can be useful before this is solved, provided the
failure contract stays explicit.

## Working Rules

- Do not trade away contract clarity for small benchmark wins.
- Do not add heuristics directly in hot-path cell processing when they belong in
  policy.
- Do not broaden the public promise beyond what tests and docs currently back.
- Prefer one coherent improvement with tests over multiple speculative
  micro-optimizations.

## Exit Criteria For "Solid Open-Source Crate" Status

The project is approaching the target state when all of the following are true:

- normal and adversarial supported cases are mostly non-panicking and
  contract-tested
- unsupported geometry and representation limits fail cleanly with explicit
  errors
- strict validation remains a reliable graph-validity gate
- preprocessing behavior is observable and documented
- performance policy changes are benchmarked against a small standard matrix
- the backend module boundaries are stable enough that further optimization does
  not require rediscovering hidden invariants
