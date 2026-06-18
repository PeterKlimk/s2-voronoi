# Voronoi Performance Research Landscape (2026-06-18)

Deep-research sweep for algorithmic/perf ideas to improve the spherical+planar
half-space-clipping Voronoi library. Method: 6 search angles, 21 sources fetched,
87 claims extracted, 25 adversarially verified (22 confirmed, 3 killed via 2/3
refute votes). Each idea is annotated with how it differs from the parked/rejected
experiments in `docs/st-headroom-and-fade-comparison.md` and
`docs/optimization-ideas.md`.

Reminder of the system's own framing: the moat is MT scaling (independent cells);
ST is ~2.3x behind incremental Delaunay (Fade2D) by design; the named ST prize is
"directional/candidate-production certificates" — skip producing/examining
candidates provably unable to cut the cell, before evaluating them. Examine-and-reject
ratio is ~1.4 (fib) to ~2.4 (clustered) candidates per final edge.

---

## Standout ideas (genuinely new vs the ledger)

### 1. Point-in-cell / edge-straddling search — abandons the certificate entirely

**Source:** arXiv:2509.07175 — Xiao, Cao, Chen, "Efficient Computation of Voronoi
Diagrams Using Point-in-Cell Tests" (Sept 2025, IEEE TVCG submission).
Confidence: high (core property verified 3-0).

Instead of streaming candidates nearest-first and proving completeness via security
radius, it launches a clip for an edge **only when its two endpoints straddle the
cell** (one inside, one outside, decided by a point-in-cell test). Termination is
"all corner points test inside," not a distance threshold.

Mechanism: build bisector `B_ij`, intersect with current edge at `q`, run
point-in-cell test on `q`; if `q`'s nearest is some `v_k != i,j`, set `j <- k` and
shrink; if `q` tests inside, `B_ij` is the exact clip. Loop while any corner point
is outside.

Claimed structural property: **at most one clipping per final intersection edge;
only contributing clippings are ever executed.**

Why this differs from the ledger: the recurring failure mode for "reuse edge info"
ideas (endpoint/anchored seeds, prefix/seed reuse, distance-symmetry seeding) was
that they *move work earlier rather than removing it*. This does not reorder
candidate examination — it structurally eliminates the examine-and-reject of useless
candidates, the exact ~1.4-2.4 ratio behind the ~2.3x ST gap.

"Improved" variant (§6.1, verified 3-0): use `delta = ||nearest_neighbor - site||/2`
as an approximate cell radius to jump the probe point outward along the edge,
"remarkably" reducing point-in-cell test count. That nearest-neighbor distance is
already computed in the kNN stage — pairs with the "already-paid metadata" lever.

Catches / open questions:
- The point-in-cell tests are *themselves* nearest-neighbor queries. It does not
  escape kNN work; the real question is whether net examine ratio drops toward 1.0
  or the cost just relocates.
- "Best measured performance across distributions" claim was **KILLED 0-3** — treat
  efficiency as a *design property*, not benchmarked superiority.
- Not-yet-peer-reviewed preprint.
- Compatibility with the hybrid once-per-shard edge-forwarding regime is unverified.

Recommended first step: a **timing-only shadow probe** counting how many clips this
*would* launch per cell on fib/uniform/splittable/mega vs current examine-and-reject,
before any implementation. Directly answers convert-vs-relocate.

### 2. Directional geometric bound + best-first BVH culling

**Source:** arXiv:2605.06408 — Taveira et al., "Scalable GPU Construction of 3D
Voronoi and Power Diagrams" (May 2026). Confidence: high (verified 3-0).

A culling bound that depends on the candidate's **direction** relative to the
evolving cell, not just scalar distance. Defines a "directional radius" = max
distance to AABB corners in the octant containing the candidate (a conservative
upper bound on cell extent toward that candidate). If a region's lower-bound bisector
distance exceeds the directional radius, the whole region is discarded before any
candidate is evaluated. Combined with hierarchical best-first BVH traversal: a single
test at a high-level node can discard regions containing millions of sites
(per-node maxweight field enables subtree-level discard; verified 3-0).

Why this differs from the ledger: this is literally the named "directional/
candidate-production certificate" ST lever. The parked directional shell/cap
certificates (`agent/directional-certificates` f51523a, `agent/directional-cell-cap-gate-audit`
3eaf1c4) were ~neutral on uniform / win only in dense regimes, and were *shell-layer*
bounds. This is a *per-candidate, octant/AABB-corner direction-dependent* pre-cull —
structurally different.

Open question: is the win from direction-dependence itself, or from the BVH
multi-scale structure? That determines whether to port the bound onto the existing
ring-walk or to swap the cube-map/flat-grid index for a BVH.

Catches:
- GPU/3D-Euclidean. Needs reformulation onto the gnomonic tangent-plane +
  great-circle bisectors.
- BVH pointer-chasing may fight the memory-latency-bound profile (cf. Morton
  cell-order went neutral). Evaluate against existing band-prune ("punch 1").
- This paper is the closest current research peer; it confirms half-space clipping
  is the active SOTA baseline being pushed to 3D/power diagrams (verified 3-0).

---

## Secondary leads (domain-transfer, more speculative)

### kNN / locality

- **GROMACS M×N cluster-pairing** (Páll & Hess 2013, ScienceDirect S0010465513001975;
  verified 3-0). Process candidates in fixed tiles (4×4 = 16 interactions per 4 loads),
  eliminating in-register shuffle — the main SIMD bottleneck of per-element lists.
  Current packed stage is 8-wide per-element. **CAUTION:** the "cluster-size-to-SIMD-width
  alone = >=2x" sub-claim was **KILLED 0-3**; the data-reuse mechanism is real, the
  2x-alone framing is not.

- **Full-containment early-emit** (`neighboursPrune`/`neighboursStruct`,
  arXiv:2603.06771, Vinambres et al. 2026; verified 3-0). When a grid octant is
  provably entirely inside the query radius, emit all its points with zero per-point
  distance tests (could skip per-candidate checks in dense bins). Plus a range-based
  output that never materializes coordinates; SFC reorder cuts cache misses 25-75%,
  runtime up to 50%; up to 10x on large-radius search. NOTE: this is **point/index**
  SFC reordering + containment emit — distinct from the (neutral) Morton **cell**
  ordering already tried. Tempered by the library's observation that cache-miss cuts
  often do not convert to cycles.

- **AVX-512 widening of the packed stage** (SWIFT, Willis et al., arXiv:1804.06231;
  verified 3-0). Pseudo-Verlet lists: 2.24x (AVX 8-wide), 2.43x (AVX2), 4.07x
  (AVX-512 16-wide), but only after non-trivial algorithmic rework. **Hardware-gated:
  the reference Ryzen 3600 has no AVX-512.** Parking-lot item, not locally actionable.

### Parallel stitching / dedup

- **numRSN dependency-peeling** (P-Weld/B-Weld/F-Weld, Fathollahi, SIGGRAPH Asia 2023,
  ACM 10.1145/3610548.3618234; code github.com/nimaft97/parallel-vertex-clustering;
  verified 3-0). Turns a serial increasing-ID ordering dependency into a dependency
  graph parallelized by counter-peeling (`numRSN` = remaining smaller-ID proximity
  neighbors; finalize when it hits 0, then atomically decrement larger neighbors),
  bit-identical to serial. 7.7-10.9x best-case (3.86x on 14M-vertex). Relevance: a
  way to keep the hybrid regime's within-shard "clip-once, forward edge" ordering
  benefit while parallelizing it, instead of losing it across threads. Cross-domain
  (mesh reduction); inferential.

- **Sort-and-scan dedup** (Wald, "GPGPU-Parallel Re-indexing of Triangle Meshes",
  2021, arXiv:2109.09812; code github.com/ingowald/sampleCode-parallel-mesh-reindexing;
  verified 3-0). Hash-free dedup: sort -> adjacent-diff first-occurrence flag ->
  prefix-sum compact -> scatter, with the sorted generator-triple as the sort key.
  Caveat: library is already bandwidth-bound and saw cache-miss cuts not convert; a
  global sort may not beat sharded local hashing.

- **Border-vertex-only divide-and-conquer merge** (Funke/Sanders/Winkler, "Load-Balancing
  for Parallel Delaunay Triangulations", 2019, arXiv:1902.07554; verified 3-0).
  Re-triangulate only the small shard-boundary "border vertices" subset; data-sensitive
  partitioning nearly halves runtime on clustered/structured inputs. Maps onto
  density-aware shard cuts vs the current ~2x-threads bin count, and onto "only
  reconcile the shard-boundary subset." Caveat: Delaunay merge is a different
  combinatorial object than per-cell clip assembly.

---

## Refuted — do NOT act on as fact

- Basselin et al. (ScienceDirect S0097849323001073) offer **no new certificate** —
  just classical security radius (KILLED 1-2).
- SIMD cluster-sizing alone gives >=2x (KILLED 0-3).
- Point-in-cell method has "best measured performance across all distributions"
  (KILLED 0-3) — design property only, not benchmarked.

---

## Priority read

1. **Point-in-cell edge-straddling search (2509.07175)** — highest value: attacks the
   structural root of the ST gap, geometry-portable, no GPU dependency. Gate on a
   cheap timing-only shadow probe (convert vs relocate).
2. **Directional geometric bound (2605.06408)** — strongest certificate idea, but
   GPU->CPU port + direction-vs-BVH ambiguity to resolve first.
3. Everything else: incremental or hardware-gated (AVX-512 needs different silicon).

## Open questions carried from the sweep

- Can the directional/octant bound be reformulated on the gnomonic tangent plane to
  cull great-circle bisector candidates pre-evaluation, and does it beat the parked
  shell/cap certificate on *uniform* (where that was neutral)? Win from direction or
  from BVH?
- Does point-in-cell remove *net* work given its tests are themselves NN queries the
  library already pays in kNN — does it push ~1.4-2.4 toward ~1.0, and is it
  compatible with the hybrid once-per-shard edge-forwarding regime?
- Can numRSN peeling preserve the hybrid once-clip shared-edge advantage while
  parallelizing within-shard order, or does the DAG reintroduce cross-thread
  independent re-clipping?
- Given bandwidth-bound + Morton-neutral history, would global sort-and-scan dedup or
  SFC point reordering actually convert cache-miss cuts into cycle/wall wins?

## Source index

| url | quality | angle |
|---|---|---|
| arxiv.org/abs/2509.07175 | primary | point-in-cell Voronoi (TVCG sub.) |
| arxiv.org/abs/2605.06408 | primary | GPU 3D Voronoi/power, directional bound |
| arxiv.org/pdf/2603.06771 | primary | SFC + linear-octree neighbour search |
| arxiv.org/abs/1804.06231 | primary | SWIFT vectorized pseudo-Verlet |
| ScienceDirect S0010465513001975 | primary | GROMACS M×N cluster-pairing |
| ACM 10.1145/3610548.3618234 | primary | lock-free vertex clustering (P-Weld) |
| arxiv.org/abs/2109.09812 | primary | parallel mesh re-indexing (Wald) |
| arxiv.org/abs/1902.07554 | primary | load-balanced parallel Delaunay merge |
| ScienceDirect S0097849323001073 | primary | Basselin et al. (refuted certificate) |
