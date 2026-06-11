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

1. ~~**Replace `MergeDensity` with a fixed-radius weld**~~ **Done.** `PreprocessMode::Weld`
   (default) welds at the coincident-distance radius via a parallel quantized-key sort pass
   (~110ms at 2M points); welded twins alias their canonical cell and are exposed via
   `SphericalVoronoi::weld_map`; the validator accounts canonical cells and checks weld-map
   consistency. `Disabled` and `MergeWithin(r)` retained.
2. ~~**Orphan-vertex policy.**~~ **Done.** Unreferenced vertices are a representation note
   (`ValidationReport::representation_notes`), and `SphericalVoronoi::compact_vertices()` removes
   them on demand. (Possible refinement: O(repairs) tracking during edge repair so compaction is
   proportional to defect count.)
3. ~~**`ClippedAway` backstop.**~~ **Done (as classification, not masking).** Emitting a
   degenerate cell turned out unsound — neighbors already clipped against the missing cell's
   bisectors would carry unpaired edges. Instead, a coincidence-driven `ClippedAway` (sub-weld
   neighbors present; only reachable via `Disabled`/undersized `MergeWithin`) returns an
   actionable `DegenerateInput` naming the coincident generators; without coincidence it stays
   `ComputationFailed` (bug class). See correctness-contract.md.
4. ~~**Input validation.**~~ **Done.** O(n) finite check at compute entry returns
   `VoronoiError::InvalidInput { point_index, .. }`.
5. ~~**Promote the evidence to CI.**~~ **Done** (modulo wiring scheduled CI when CI exists): the
   2M/3M/4M fuzz sweeps assert `STRICT_VALID` under the default config (`#[ignore]` for runtime
   only), weld contract tests cover exact duplicates, ulp clusters, and seam-aligned pairs,
   resolvable-regime tests cover above-weld pairs and the rotated-symmetric control with welding
   disabled, and the margin-mapping probes live on as a permanent diagnostic suite in
   `tests/coincidence_probes.rs` (run after numerics changes to detect boundary drift).
6. ~~**Tighten loose assertions**~~ **Done.** Euler exactly 2 (over referenced vertices), zero
   sub-3-vertex cells, zero duplicate boundary vertices in `tests/correctness.rs`.

**P0 is complete.** The correctness contract is implemented and enforced by asserting tests.

## P1: Release engineering

1. ~~**Stable-Rust build path.**~~ **Done — better than planned: nightly is gone entirely.** All
   explicit SIMD flows through the `fp.rs` backend seam (`PointChunk8`/`Dots8`); the default
   backend is the `wide` crate (stable Rust, explicit lanes), with `simd_scalar` as a
   debugging/comparison fallback. Reference-machine (Ryzen 3600, target-cpu=native) benchmarks
   showed wide at parity with the old portable_simd backend (within ~1-2%, winning some runs);
   all backends are bit-identical (tests/backend_fingerprint.rs). portable_simd was deleted
   (recoverable from git history if std::simd stabilizes). MSRV set to 1.88.
2. **Feature consolidation.** User-meaningful features only (`parallel`, `simd`, `serde`,
   `glam`, `qhull`); internal/research flags (`microbench`, `simd_clip`, `fma`,
   `packed_knn_sort_small`, `profiling`) become doc-hidden or merge away.
3. ~~**Finish the non-panicking contract**~~ **Done.** The clipper "invariant failure" panic was
   proven unreachable (mixed cyclic masks always carry both transitions) and converted to a
   documented `unreachable!`; the fallback angle sort uses `total_cmp` (the old
   `partial_cmp + unwrap_or` could trip std::sort's total-order check on NaN). Remaining panics
   are genuine bug traps per the supported-envelope contract.
4. Zero-warning builds **(done)** and `rust-version = "1.88"` **(done)**; remaining:
   `#[deny(missing_docs)]` on the public surface and the ~18 pre-existing clippy style lints
   (mostly too-many-arguments, tied to the P4 god-function splits).

## P2: API completeness

In order of user value:

1. ~~**Neighbor adjacency**~~ **Done.** `SphericalVoronoi::build_adjacency()` returns a
   `CellAdjacency` built by sort-based edge pairing (O(E log E), on demand); neighbor entry `k`
   of a cell sits across boundary edge `(v[k], v[k+1])`, welded twins alias their canonical
   cell's list, and the neighbor pairs double as Delaunay edges (covers most of item 3).
2. ~~**Cell areas / centroids**~~ **Done.** `cell_area(i)` (signed solid-angle fan, f64; sums
   to 4π over canonical cells, test-pinned) and `cell_centroid(i)` (exact boundary integral,
   orientation-robust) — Lloyd relaxation is a 3-line loop, with a convergence test as the demo.
3. **Delaunay dual access** — mostly covered by `build_adjacency()` (neighbor pairs are the
   Delaunay edges); an explicit triangle-list API remains if users ask for it.
4. ~~**`serde` feature**~~ **Done.** Optional derives on UnitVec3, SphericalVoronoi (weld map included), and CellAdjacency; round-trip test behind the feature.
5. ~~`compact()`~~ **Done** as `SphericalVoronoi::compact_vertices()` (landed with P0.2).

Explicitly deferred: weighted Voronoi, f64 storage, no_std, dynamic insertion/deletion.

## P3: kNN / performance robustness

Design principle (2026-06 discussion): the knot is not the measured fast path (chunk0 / tail /
expand-r2 — which already are Chebyshev shells r=0,1,2 with bespoke tuning, serving ~99.95% of
cells); it is that the fallback beyond r=2 is a *second algorithm* (the dual-heap
`DirectedNoKCursor`, ~0.05% of cells, a large share of the stack's conceptual weight). The end
state is one outward-marching shell frontier where the first three shells stay hand-tuned, the
rest are generated, the certificate is uniform (per-shell annulus bound), eligibility is decided
at cell level everywhere (a grid cell maps to one bin), and there is no second algorithm.

In implementation order:

1. **Standalone NN contract suite (before touching anything).** Test the neighbor-source layer
   in isolation from the Voronoi pipeline: emitted eligible set equals brute force, and every
   frontier certificate conservatively bounds every unseen eligible point. Explicit cube-grid
   edge/corner coverage: queries and points at the 8 corners (7-cell neighborhoods), the 12
   face-edge seams, face centers/poles, exact-seam coordinates (1/sqrt(3), 1/sqrt(2)), all
   points on one face (cross-face shell traversal), one point per face, antipodal pairs (maximal
   shell radius), near-wall points (classification robustness), tiny n, clusters, bimodal
   sparse+dense (forces deep shells). This suite is the strangler-pattern net: the shell rework
   must pass it unchanged. Note: emission *order* is deliberately not part of the contract (the
   current exact-order stream test pins the cursor's implementation and will be rewritten).
2. **Grid resolution policy.** Current resolution (target density 16 points/cell from n) is
   admittedly tuned to one scenario (see engineering-findings). Important constraint (2026-06):
   cells need *more* neighbors before terminating at higher densities, so the optimal
   points-per-cell is not a constant — it varies with point count and distribution. Replace
   with: (a) a target-density *curve* fit from a benchmark sweep across n (and sanity-checked on
   non-uniform distributions), not a single re-tuned constant; (b) occupancy feedback — the
   build already histograms occupancy; rebuild at higher resolution when max occupancy exceeds a
   bound, capped by a memory budget (total cells O(n)); (c) for concentration beyond what global
   resolution can fix within memory, a shells-native big-cell path (chunked selection within
   shell 0) instead of today's bail to the cursor (`SlowPath`). The termination-vs-density
   interaction also suggests recording neighbors-before-termination stats in timing runs so the
   curve has an explanatory model, not just fitted numbers.
3. **Shell generalization behind a policy switch.** Implement r >= 3 by BFS over the 3x3 cell
   adjacency with visited stamps (face/corner stitching machinery exists for r<=2), the same
   SIMD filter primitive, per-shell sorted emission, annulus certificate. Both paths live
   side by side; interleaved benchmarks on uniform + clustered + bimodal + the adversarial and
   fuzz corpus, plus a forced-deep-shell scenario. Then flip, delete the cursor, its scratch
   heaps, the stream splice state, and the exact-order test.
4. **Policy shrink.** After (2) normalizes occupancy, revisit the remaining constants (chunk
   sizes, termination cadence, count-model flags) — most were compensating for unbounded density
   variance and likely simplify or die. Keep what remains expressed through `src/policy.rs`.
5. Open questions parked from the discussion: whether `termination_max_k` survives (looks
   vestigial once certificates are uniform and occupancy is bounded), and per-shell vs per-cell
   certificate granularity (believed to tax only already-pathological cells; benchmark confirms).

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
