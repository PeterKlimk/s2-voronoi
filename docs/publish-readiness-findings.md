# Publish Readiness Findings

This document captures the first review pass over `s2-voronoi`, focused on blockers and serious risks for open-source publication.

It is intentionally blunt and should be treated as a working hardening checklist, not a final verdict on the project.

## Summary

The project looks promising and worth continuing, but it is not yet publishable as a dependable general-purpose library.

The main issue is not raw performance or lack of interesting ideas. The main issue is that the library contract, validation story, and implementation behavior do not line up under stress.

## Confirmed Findings

### 1. Public `Result` contract does not match runtime behavior

`compute` and `compute_with` present computation as a fallible API:

- [`src/lib.rs`](/home/pkzmbk/code/s2-voronoi/src/lib.rs)
- [`src/error.rs`](/home/pkzmbk/code/s2-voronoi/src/error.rs)

In practice, only `InsufficientPoints` is returned today. The backend still panics for unsupported or extreme cases during cell construction:

- [`src/knn_clipping/live_dedup/build/process_cell.rs`](/home/pkzmbk/code/s2-voronoi/src/knn_clipping/live_dedup/build/process_cell.rs)

Historical note: this was originally confirmed by running the ignored hemisphere
adversarial case:

```bash
cargo test --release --test adversarial test_hemisphere_basic -- --ignored --nocapture
```

That run failed with multiple worker-thread panics instead of returning `Err`.
The upper-hemisphere path is no longer the reproducer: those fixtures now compute
and validate through the cold all-constraints fallback. The broader finding
remains relevant for any remaining backend panic/expect paths under unsupported
or extreme inputs.

Why this matters:

- Library consumers cannot reliably catch unsupported input classes.
- Panics are acceptable during research, but not as the default behavior of a published geometry crate.

### 2. Validation can report catastrophically degraded output as "Perfect"

`ValidationReport` tracks:

- `unique_cells`
- `duplicate_cells_count`

but `is_valid`, `is_perfect`, and `summary` do not treat heavy duplicate-cell collapse as a failure:

- [`src/validation.rs`](/home/pkzmbk/code/s2-voronoi/src/validation.rs)

This was confirmed by running:

```bash
cargo test --release --test adversarial test_clustered_cap_tight -- --nocapture
```

Observed output included:

```text
clustered_cap_tight: Perfect
  cells=100, vertices=26, total_cell_vertices=475, unique_cells=15, duplicates=85
```

Why this matters:

- The current validation layer can hide major semantic collapse.
- A release gate built on `validation::validate` is currently not trustworthy.

### 3. Default preprocessing silently changes the problem being solved

By default, the crate merges near-coincident points using a density-dependent threshold:

- [`src/knn_clipping/constants.rs`](/home/pkzmbk/code/s2-voronoi/src/knn_clipping/constants.rs)
- [`src/knn_clipping/preprocess.rs`](/home/pkzmbk/code/s2-voronoi/src/knn_clipping/preprocess.rs)
- [`src/knn_clipping/compute.rs`](/home/pkzmbk/code/s2-voronoi/src/knn_clipping/compute.rs)

Then it remaps the effective cell back onto all merged original generators.

Why this matters:

- This is a practical robustness technique, but it means the default behavior is already "best effort / repaired", not a strict Voronoi computation on the original input set.
- That may be fine, but it needs to be explicit in the API and docs.

### 4. Public API surface is wider than the abstraction is ready to support

The current API exposes internal storage more directly than is ideal:

- [`src/diagram.rs`](/home/pkzmbk/code/s2-voronoi/src/diagram.rs)

Examples:

- `SphericalVoronoi` exposes raw `generators` and `vertices`.
- `from_parts` is public.
- `VoronoiCell::new` allows callers to construct arbitrary layouts.

Why this matters:

- This locks in representation choices too early.
- It makes cleanup harder while the internals are still evolving.

### 5. Test coverage exists, but the most important edge cases are not yet protected

The project has more tests than the initial self-assessment suggested:

- [`tests/api.rs`](/home/pkzmbk/code/s2-voronoi/tests/api.rs)
- [`tests/correctness.rs`](/home/pkzmbk/code/s2-voronoi/tests/correctness.rs)
- [`tests/validation.rs`](/home/pkzmbk/code/s2-voronoi/tests/validation.rs)
- [`tests/adversarial.rs`](/home/pkzmbk/code/s2-voronoi/tests/adversarial.rs)

However:

- several high-value adversarial cases are `#[ignore]`
- some tests accept either success or failure
- some correctness tests are only loose sanity checks

Why this matters:

- The current suite is good for exploration and iteration.
- It is not yet strong enough to define the crate's real supported envelope.

### 6. Feature surface and safety story still feel internal

The crate exposes a fairly broad feature matrix:

- [`Cargo.toml`](/home/pkzmbk/code/s2-voronoi/Cargo.toml)

This mixes user-facing toggles with internal performance and tooling switches.

There are also several `unsafe` hot paths that appear performance-motivated, but do not yet have a polished public-facing safety explanation:

- [`src/sort.rs`](/home/pkzmbk/code/s2-voronoi/src/sort.rs)
- [`src/generated/sort_nets.rs`](/home/pkzmbk/code/s2-voronoi/src/generated/sort_nets.rs)
- [`src/knn_clipping/topo2d/clippers/small.rs`](/home/pkzmbk/code/s2-voronoi/src/knn_clipping/topo2d/clippers/small.rs)
- [`src/knn_clipping/live_dedup/assemble.rs`](/home/pkzmbk/code/s2-voronoi/src/knn_clipping/live_dedup/assemble.rs)

Why this matters:

- This is not necessarily bad engineering.
- It does make the crate feel more like an internal research/perf workspace than a polished public package.

## Recommended Hardening Order

1. Make `compute` non-panicking for unsupported/pathological inputs.
2. Tighten `validation` so semantic collapse is reported as failure.
3. Decide and document whether preprocessing is default behavior, opt-in repair, or a separate API.
4. Shrink the public API to the minimum stable surface.
5. Convert the adversarial corpus into real regression tests with explicit expected outcomes.
6. Add feature-matrix coverage for user-facing configurations.

## Overall Take

This does not look like a "start over" project.

The algorithmic core appears worth preserving. The main work ahead is to make the library honest about its current limits and robust in how it communicates failure.

## Follow-Up

Some of the highest-priority findings above have since been addressed in code and tests:

- the public `Result` contract now covers several explicit unsupported-geometry and
  representation-limit paths
- strict validation semantics were redesigned
- preprocessing is now explicit and observable
- the public diagram surface has been tightened
- `compute_with_report` now exposes the effective diagram and prefers effective validation when
  preprocessing changes the solved generator set
- many adversarial cases are now pinned as explicit success/failure contract tests instead of
  loose exploratory checks
- reconciliation, assembly, and extraction invariants now fail with clearer structure-limit or
  bug-only diagnostics rather than broad unchecked assumptions

The current supported-success / supported-failure / preprocessing / invariant-failure split is
documented in [`docs/supported-envelope.md`](/home/pkzmbk/code/s2-voronoi/docs/supported-envelope.md).
