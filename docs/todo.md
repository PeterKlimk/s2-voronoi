# TODO / Roadmap

This is the active working roadmap for `s2-voronoi`, updated after the 2026-06 release-readiness
evaluation. If you want to know "what should we work on next?", start here.

Related documents:

- `docs/correctness-contract.md`: the "essentially Voronoi" contract this roadmap implements,
  including the empirical coincidence/weld evidence.
- `docs/engineering-findings.md`: the findings log (correctness/contract/organization issues).
- `docs/supported-envelope.md`: outcome classification for the current backend.

## Guiding goal

A prestige-quality open-source crate that is the fastest spherical Voronoi implementation
available, with a machine-checked topological guarantee: **valid graph or clean error, at any
scale, for any input inside a precisely documented envelope.**

Headline claims to be able to make honestly at release:

- millions of cells per second on desktop CPUs (2.5M points < 500ms on a Ryzen 3600 class part)
- strict graph validity asserted by CI at multi-million point counts
- a documented coincidence policy with measured safety margins (not folklore epsilons)

## P0: Implement the correctness contract

The empirical groundwork is done (see correctness-contract.md); this is the implementation.

1. **Replace `MergeDensity` with a fixed-radius weld** (~1e-6 chord).
   - simple quantized spatial hash + tiny union-find, or reuse of the cube grid; O(n), parallel
   - fix welded-generator output semantics: welded indices share a cell (no materialized
     duplicate vertex lists — the current remap's duplicate cells are what fail validation)
   - validator understands weld classes; `compute_with_report` reports welds
   - keep `Disabled` (caller certifies separation) and `MergeWithin(r)` (caller policy)
2. **Orphan-vertex policy.**
   - reclassify unreferenced vertices as a representation note in strict validation, and/or
   - O(repairs) targeted compaction: record indices orphaned during edge repair; if non-empty at
     finalization, swap-remove and patch only affected cells (zero cost on clean runs)
3. **`ClippedAway` backstop.** A micro-cell clipped to empty by sub-weld-radius neighbors emits a
   degenerate/welded cell plus a report entry instead of aborting the whole computation.
4. **Input validation.** O(n) finite check with index-bearing error (NaN currently surfaces as a
   deep `ClippedAway` diagnostic).
5. **Promote the evidence to CI.**
   - un-ignore the 2M/3M/4M fuzz tests as scheduled CI asserting `STRICT_VALID` (they pass today
     with preprocessing disabled; after the weld rework they must pass with defaults)
   - adopt the coincidence probes (separation sweep, ulp clusters, seam/aligned pairs, rotated
     control) from `tests/tmp_ulp_regimes.rs` into `tests/adversarial.rs` as contract tests
   - delete the stale "bad edges" comments — the 2–4M failures were the merge remap, not stitching
6. **Tighten loose assertions** for supported inputs: Euler exactly 2, zero degenerate cells, zero
   duplicate-vertex cells (loose bounds remain only in explicitly-adversarial diagnostics).

## P1: Release engineering

1. **Stable-Rust build path.** Wrap the SIMD primitives (`fp.rs` dot/compare/bitmask) behind a
   thin abstraction with a scalar implementation; nightly + `simd` feature restores full speed.
   This unblocks crates.io.
2. **Feature consolidation.** User-meaningful features only (`parallel`, `simd`, `serde`,
   `glam`, `qhull`); internal/research flags (`microbench`, `simd_clip`, `fma`,
   `packed_knn_sort_small`, `profiling`) become doc-hidden or merge away.
3. **Finish the non-panicking contract** (carried from previous roadmap): remaining panic paths in
   clipper invariants (`clippers/small.rs`), fallback vertex sort (NaN-unsafe `partial_cmp`),
   stream-state contradictions → structured errors; panic only for true bugs.
4. Zero-warning builds, `rust-version` (MSRV) in Cargo.toml, `#[deny(missing_docs)]` on the
   public surface.

## P2: API completeness

In order of user value:

1. **Neighbor adjacency** (`cell_neighbors(i)`) — computed implicitly during clipping; the single
   most-requested Voronoi API.
2. **Cell areas / centroids** (spherical excess) — also the building block for the Lloyd
   relaxation demo, which is the crate's best showcase (interactive centroidal tessellation at
   millions of points).
3. **Delaunay dual access** — the adjacency is the Delaunay triangulation; exposing it doubles
   the addressable audience.
4. **`serde` feature** on `SphericalVoronoi`.
5. `compact()` (vertex compaction, see P0.2) as an explicit method.

Explicitly deferred: weighted Voronoi, f64 storage, no_std, dynamic insertion/deletion.

## P3: kNN / performance robustness

1. **Replace the heap cursor with generalized shell expansion.** The `DirectedNoKCursor`
   best-first walker buys exact distance ordering the clipper does not need (correctness needs
   coverage plus a sound unseen-dot bound, which Chebyshev shells provide from the precomputed
   cell bounds). Generalize the packed ring stages (r=1 tail, r=2 expand) to arbitrary r via BFS
   over the 3x3 adjacency with visited stamps; delete the dual-heap cursor and its duplicated
   filtering logic. The fallback path becomes a continuation of the fast path instead of a
   different algorithm. Benchmark the frontier-certificate contract before and after.
2. **Occupancy-driven grid resolution.** The build already histograms cell occupancy; if max
   occupancy exceeds a threshold, rebuild at higher resolution (build is O(n) and cheap). This
   removes the mega-cell degradation for clustered inputs and most of the "tuned for uniform"
   brittleness at the root.
3. **Derive policy constants from observed occupancy** (chunk sizes, termination cadence) at
   build time instead of fixed values tuned for ~16 points/cell uniform.
4. Keep all of this expressed through `src/policy.rs` (rule unchanged from previous roadmap).

## P4: Code quality (prestige pass)

1. **Centralize tolerances** in one module with a one-line empirical justification each
   (EPS_INSIDE, FALLBACK_PLANE_TOL, FALLBACK_DEDUP_DOT, DEGENERATE_LEN_EPS, MIN_PROJECTION_COS,
   the 1e-28 length checks, grid PAD/SIN_EPS). The margin data in correctness-contract.md is the
   evidence backing them.
2. **Split the god functions**: `build_cell_into` (~280 lines, cell_build/run.rs) and
   `collect_and_resolve_cell_edges` (~210 lines, live_dedup/edge_checks.rs); extract the
   edge-endpoint matching logic duplicated between the main and overflow paths.
3. **Document the stitching invariant.** The directed bin/local ordering that makes per-cell
   parallel construction compose into one consistent graph is the crate's most original idea and
   currently lives only in code structure. Write the argument down (likely in
   architecture.md / live_dedup.md).
4. Invariant comments on the remaining undocumented unsafe blocks (small clippers; the grid
   scatter is already documented).

## P5: Structural upgrade — consistency by construction

Make every shared combinatorial decision (edge existence, vertex existence) canonical: evaluated
once in a frame chosen by sorted generator index (or via world-space f64 predicates), inherited
bit-for-bit by both cells. Graph validity then no longer depends on epsilon agreement between two
charts; one proves determinism instead of error bounds.

Expected payoffs: root-fixes the seam/symmetric-position regime, shrinks edge reconciliation's
responsibilities, and upgrades the contract from "empirically safe with 8x margin" to "valid by
construction". This is the largest item on the list and should follow, not precede, release.

## P6: New geometries

The engine decomposes into geometry-specific layers (`cube_grid` indexing, `topo2d` chart +
bisectors) and a geometry-agnostic core (directed stitching, sharded dedup, reconciliation). A
`Geometry` abstraction needs: spatial index with conservative cell bounds, bisector constructor,
2D clipping chart, termination certificate.

- **Plane** is mostly simpler (no chart needed — clip directly; hemisphere machinery vanishes)
  with one new problem the sphere never has: unbounded cells need a boundary policy (bbox clip /
  marked-unbounded), which touches the stitching contract.
- Sequence: ship the sphere first; keep geometry-specific code quarantined behind the two module
  boundaries now; do the trait extraction when a second geometry is actually being built, so the
  abstraction is shaped by two instances rather than one.

## Working rules

(unchanged)

- Do not trade contract clarity for small benchmark wins.
- Heuristics live in policy, not in hot-path cell processing.
- Do not broaden the public promise beyond what tests and docs back.
- One coherent improvement with tests over multiple speculative micro-optimizations.

## Exit criteria for release

- P0 complete: fuzz + coincidence probes green in CI with default config, asserting strict
  validity.
- P1 complete: builds on stable (degraded perf), zero warnings, no input-reachable panics.
- README states the contract, the envelope, and the measured performance honestly.
- At least neighbor adjacency from P2 (areas/centroids strongly preferred — enables the Lloyd
  demo).
