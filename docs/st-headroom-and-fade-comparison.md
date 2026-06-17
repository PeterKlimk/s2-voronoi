# Single-thread headroom & the Fade2D comparison

Status: investigation notes, 2026-06-17. Not a committed plan — a backlog of
algorithmic levers for the **single-threaded** planar/spherical clip path,
grounded in a head-to-head against Fade2D. Companion to
`docs/optimization-ideas.md` (micro/structural backlog) and
`docs/multi-regime-perf.md` (regime framing).

## Why this doc exists

We benchmarked `compute_plane` against **Fade2D v2.17.3** (Kornberger / Geom
Software), a best-in-class incremental-Delaunay library, on identical inputs
on the WSL2 / Ryzen 3600 box. The comparison split cleanly by threading model
and surfaced a structural picture of where our clip-based approach stands
relative to a Delaunay dual. This doc records the result and the resulting
algorithmic backlog.

### How this was measured (harness not retained)

The Fade SDK (~1 GB, license-restricted) and the throwaway C++ harness were
**not kept** — only the numbers below are. To reproduce:

- Get Fade2D from geom.at; link `lib_ubuntu24.04_x86_64/libfade2d.so` + system
  libgmp. `#include "License.h"` activates the free **student license**, with
  hard caps: **2D Delaunay 1e6 points, Voronoi diagram 1e5 points** (MeshGen
  5e4, etc.). Larger limits need an extended eval license (email
  bkorn@geom.at). The 100k Voronoi cap is why every head-to-head Voronoi number
  here stops at 100k.
- Feed both libraries **bit-identical** inputs (we dumped `bench_plane`'s
  generated points as little-endian f64 x,y pairs and read the same file from
  C++). Time Fade's three phases separately: `insert()` (Delaunay),
  `getVoronoiDiagram()` (lazy, ~0 ms), `getVoronoiCells()` (materialize
  comparable cell polygons). Use `setNumCPU(0)` for MT (commits all cores),
  `setNumCPU(1)` for ST.
- Fade is f64 coordinates + GMP **exact predicates** (robust); we are f32. The
  precision gap is considered minor since our math is f64 internally.

### Results (uniform, identical inputs, best-of-N)

Full Voronoi (insert + dualize + materialize cells), at the 100k cap and below:

| n    | compute_plane MT | Fade2D MT | compute_plane ST | Fade2D ST |
|------|------------------|-----------|------------------|-----------|
| 25k  | **10.2 ms**      | 30.5 ms   | 31.9 ms          | **11.6 ms** |
| 50k  | **15.7 ms**      | 37.8 ms   | 60.8 ms          | **24.2 ms** |
| 100k | **29.4 ms**      | 52.1 ms   | 118.6 ms         | **51.1 ms** |

Pure Delaunay `insert()` scaling (insert-only mode, up to the 1M 2D cap),
Fade commits all 12 logical cores:

| n    | insert ST | insert MT | MT speedup |
|------|-----------|-----------|------------|
| 100k | 46.1 ms   | 39.9 ms   | 1.16×      |
| 500k | 242.9 ms  | 108.0 ms  | 2.25×      |
| 1M   | 491.0 ms  | 203.3 ms  | 2.41×      |

### Reading of the result

- **ST: Fade wins ~2.3×.** Its sequential exact-predicate Delaunay core is
  excellent and does *more* work (robust arithmetic). Conceded.
- **MT at ≤100k: we win ~1.8×** — but partly because ≤100k is Fade's *worst*
  MT regime (1.16× scaling; thread overhead unamortized). The 100k Voronoi cap
  forces the comparison into exactly that regime.
- **MT scaling is the structural divide.** Fade's incremental Delaunay caps at
  ~2.4× on 6 physical cores (~40% efficiency) because insertion mutates a
  *shared* triangulation (point-location walks + flip cascades contend). Our
  per-cell half-space clipping is embarrassingly parallel and hits ~4×
  (~67%). **This is our one durable structural edge.**
- **Projected 1M MT (cannot measure past the cap):** Fade's insert *alone* at
  1M MT is 203 ms — already faster than our full `compute_plane` at 1M MT
  (~385 ms uniform). So our 100k MT win likely does **not** survive to 1M; Fade
  probably takes 1M MT too, by a smaller margin than its 2.3× ST lead. More
  cores move this back toward us (see "Many-core study").

## The structural model (what bounds ST)

Half-plane clipping is **O(n · k̄)**, k̄ ≈ neighbors *processed* per cell ≈
1.6× final degree (our `gather-reduction` measurements). Incremental Delaunay
is **O(n)** — O(1) amortized per cell. The constant favors us (f32, no GMP);
the operation *count* favors Delaunay.

Two redundancy sources, and how much each is already paid down:

1. **Shared-edge recompute (~2×).** Each Voronoi edge is geometrically
   determined twice — once per adjacent cell's clip. **Largely already
   collected:** within-bin edges are *handed off* (the bin is the serial unit;
   bins are the parallel unit), so this 2× tax survives only at **bin
   boundaries** (~√(n·B) edges vs O(n) total), which `edge_reconcile` cleans
   up. Do **not** chase the remaining boundary fraction by serializing across
   bins — that reintroduces the cross-bin dependency that caps MT scaling, i.e.
   it converges our design toward being a Delaunay dual and forfeits the moat.

2. **Examine-and-reject (~1.6×) — the real residual tax.** Even with perfect
   edge handoff, we must *examine* ~1.6× the contributing neighbors to *prove*
   cell completeness (the certificate). Each examined-but-rejected candidate
   costs a bisector eval + classify even when the clip itself is elided by
   handoff. Delaunay's flip cascade **never touches a non-edge**. This
   asymmetry — "we consider-and-reject to prove completeness; Delaunay only
   ever touches real edges" — is the structural core of the remaining ST gap.

**Honest ST ceiling for this algorithm class:** with edge-handoff already in,
realistic independence-preserving headroom is **~1.1–1.25×**, concentrated in
the completeness certificate, *not* the clipper. Reaching Fade's ST means
sharing edge computation = becoming Delaunay = losing the 4× MT scaling. The
2.3× ST gap and the 4×-vs-2.4× MT advantage are two readings of the same
independence property. **Strategic stance: don't chase ST parity; bank the
MT + sphere story, and harvest the small independence-safe ST wins below.**

## Investigation backlog

Tagged: **[safe]** = preserves per-cell independence (no MT cost);
**[boundary]** = touches the cross-bin seam; **[measure]** = prerequisite
instrumentation; **[strategic]** = positioning, not an ST-algorithm change.
Measurement rules per `optimization-ideas.md`: paired interleaved, ST for
stability, quiet box.

### Primary levers (the user's four)

1. **Completeness certificate — direction-aware** `[safe]`. Today we examine
   candidates out to ~1.6× degree by *radius*. A direction-aware certificate
   stops examining an angular sector once it is provably sealed, cutting
   examine-and-reject below 1.6× directly. This is *the* lever on the residual
   ST tax and helps MT (less work) too. Already flagged in the
   `gather-reduction` notes as "the only new normal-case lever." Hard,
   geometry-careful. **Gate on the examine-and-reject measurement (below).**

2. **Angular-sweep clipper** `[safe]`. Replace distance-ordered repeated
   Sutherland-Hodgman (O(k·c) per cell, re-walks the polygon each clip) with a
   single rotational sweep over the *certified* half-planes in angular order
   (O(k) after an O(k log k) angular sort). Each half-plane edits only a local
   boundary arc. Tension: angular order fights the distance-ordered
   termination, so two passes (certify by distance, then sort+sweep).
   **Demoted by the handoff finding** — within-bin handoff already elides ~half
   the fresh clips, shrinking the cost base this optimizes. Prototype behind a
   feature flag and A/B; expect modest (~1.1–1.3×) on the fresh-clip half.

3. **NN optimization** `[safe]`. The kNN gather that *produces* the candidate
   set is the second cost center (15–28% of cell construction). Delaunay
   amortizes point-location via spatial-sort/BRIO locality; we re-query the
   grid per cell. Our slot-order/AoS work attacks the cache side; the query
   itself is still per-cell. See "candidate-set reuse" below for the inter-cell
   angle.

4. **Handoff correctness** `[boundary]`. Verify within-bin candidate-edge
   handoff is firing for every shared edge it should (not silently falling
   through to recompute + reconcile). Audit the edge-check policy
   (`clip_with_slot_edgecheck_policy`, `live_dedup/edge_checks.rs`). Correctness
   first; any missed handoff is both a perf leak and reconcile load.

### Additional levers

5. **Candidate-set reuse between adjacent cells** `[safe]`. Process cells in
   spatial order and carry forward the overlapping kNN list — adjacent Voronoi
   cells share most neighbors. Inter-cell reuse of the gather, distinct from
   (3). Closest thing we have to BRIO's amortized point location. Note: Morton
   cell ordering was tried and is marginal (`morton-cell-order-marginal`); this
   is about reusing the *candidate set*, not just cache order.

6. **Cheap conservative reject before clip** `[safe]`. A support-distance /
   bounding-circle test so a bisector that provably cannot cut the current cell
   skips the full clip. Generalizes the N≤4 mask4 path upward. Trims the
   examine cost without a certificate redesign.

7. **Distance-symmetry certificate seeding** `[safe]`. bisector(p,q) distance
   is symmetric: a confirmed p↔q neighbor relationship seeds the *other* cell's
   certificate/candidate bound. Fuses handoff with the completeness certificate
   (1). Check whether handoff already conveys this implicitly before building.

8. **Low-degree specialized kernel** `[safe]`. Branch-light fast path for the
   dominant 5–7-edge cells (extends mask4's N≤4 specialization). Borders on
   micro; include only if profiling shows degree-dispatch branch cost.

9. **Better initial polygon seed** `[safe]`. Seed each cell from local grid
   structure so fewer far-neighbor clips are needed to converge. Speculative;
   interacts with (2) and (6).

10. **Cross-bin edge handoff** `[boundary]`. Close the residual boundary
    recompute (the only place the 2× tax survives). Reintroduces a cross-thread
    dependency — needs a lock-free claim or deferred scheme and a composition
    proof with `edge_reconcile`. Marginal (boundary is √-scaled); listed for
    completeness. See the deprioritized "Live within-bin edge repair" entry in
    `optimization-ideas.md` for the related proof obligations.

### Prerequisite measurement

11. **Instrument examine-and-reject ratio** `[measure]`. Candidates examined vs
    edges kept, per cell, per regime (the `timing` feature already tracks
    `neighbors_processed`; add edges-kept). Sizes the certificate prize: 1.6×
    means ~37% of examination cost is the theoretical max recovery; ~1.2× means
    the certificate lever is nearly dead and ST is at its floor for this class.
    **DONE (2026-06-17, initial counter probe):** timing KV now emits
    `final_edges_total`, `final_edges_max`, and `examine_per_edge =
    neighbors_total / final_edges_total` for sphere and plane drivers. Fixed
    seed, `RAYON_NUM_THREADS=1`, `--no-preprocess`, n=200k:

    | distribution | neighbors/cell | final edges/cell | examine/edge | note |
    |---|---:|---:|---:|---|
    | fib | 8.60 | 6.00 | 1.434 | quasi-regular lower bound; still nonzero headroom |
    | uniform | 9.89 | 6.00 | 1.648 | normal-case doc estimate confirmed |
    | gradient k=4 | 9.92 | 6.00 | 1.653 | sparse/graded case same ratio |
    | splittable | 14.32 | 6.00 | 2.387 | dense case adds extra examine tax |

    Hardware-counter spot check agreed on work scale: uniform 200k was
    ~2.60B instructions; splittable 200k was ~6.52B instructions. Verdict:
    the gate for lever (1) is satisfied. The direction-aware certificate is
    the right research prototype to design next; dense cases still need the
    packed/local-index work in `multi-regime-perf.md`, but they do not weaken
    the certificate case.

    **Prototype 0 (2026-06-17, known-batch directional shadow):** implemented a
    no-behavior-change timing probe for a conservative subset of the full
    direction-aware certificate. At exact-batch checks, if the existing scalar
    certificate would pass after the batch and every remaining known candidate
    in the batch is all-inside against the current polygon, the probe counts
    one shadow termination and the candidates that a real implementation would
    skip. It is a lower bound: it does not model directional bounds for unknown
    frontier regions and counts only the first hit per cell.

    | distribution | neighbors total | shadow-hit cells | shadow-saved candidates | saved / neighbors |
    |---|---:|---:|---:|---:|
    | fib | 1,720,255 | 62,809 | 526,061 | 30.6% |
    | uniform | 1,977,806 | 80,775 | 643,231 | 32.5% |
    | gradient k=4 | 1,983,723 | 80,946 | 645,520 | 32.5% |
    | splittable | 2,864,306 | 101,632 | 990,939 | 34.6% |

    Caveat: the naive shadow probe scans polygon vertices for every candidate
    tested and is intentionally too expensive to judge wall time. The signal is
    the custom counter: even this narrow known-batch certificate finds roughly a
    third of processed candidates skippable across regimes. Next design target:
    replace the O(batch · vertices) shadow test with a cheap sector/support
    envelope and add directional bounds for frontier regions.

    **Prototype 1 (2026-06-17, 64-sector support envelope):** added a
    timing-only conservative support cache over the current polygon. For a
    candidate half-plane direction, it uses the nearest cached sector plus a
    radius penalty; `support=true` should imply the exact all-vertices test is
    also true. The probe counts false-positive hits separately and saw zero in
    the fixed-seed 200k sweep below. Wall time is still not evidence here: the
    timing build computes both support and exact shadow checks so they can be
    compared.

    | distribution | exact shadow saved | support saved | support / exact | support false-positive hits |
    |---|---:|---:|---:|---:|
    | fib | 526,061 | 396,951 | 75.5% | 0 |
    | uniform | 643,231 | 518,214 | 80.6% | 0 |
    | gradient k=4 | 645,520 | 515,580 | 79.9% | 0 |
    | splittable | 990,939 | 839,818 | 84.8% | 0 |

    This is enough to justify a real implementation sketch: cache the support
    envelope on polygon changes, use it to skip the remainder of known exact
    batches when the beyond-batch scalar certificate passes, then separately
    design directional certificates for unknown frontier regions.

### Library-derived ideas parked during the scan

These came from comparing our design against mature geometry-library habits
(CGAL spatial sorting, Voro++-style single-cell construction, Qhull/local-hull
thinking, Triangle/CGAL predicate kernels, VoroTop topology analysis). They are
recorded only where they differ from the existing backlog above.

- **Locality variants beyond current slot order** `[safe, measure]`.
  Production already stores points in grid cell-major slot order, assigns
  within-bin locals in that order, and keys hot attempted-neighbor state by
  slot; a Morton cell-order probe was already marginal. The remaining library
  idea is narrower: try CGAL-like face-local Hilbert / median-Hilbert ordering
  or coarse-cell work scheduling as a measurement-only branch, with counters
  for cache misses, seed-handoff hits, packed group shape, and
  `examine_per_edge`. Low priority, but worth keeping because it is cheap and
  correctness-neutral.
- **Local Delaunay / lifted-hull oracle for pathological cells**
  `[safe fallback, oracle first]`. Instead of global Delaunay, gather a
  conservative local patch only when a cell's `neighbors_processed` or
  occupancy crosses a high threshold, then compute the local star/lower hull as
  a debug oracle or rare fallback. This is distinct from the clipping fallback
  ideas: it changes algorithm only for nasty cells and could validate
  direction-aware certificate prototypes.
- **Predicate-kernel packaging for P5** `[correctness architecture]`.
  P5 already says "canonical predicates"; the library-derived addition is
  packaging discipline. Build a tiny isolated filtered/exact predicate kernel
  with its own tests and perf counters before threading it through clipping, in
  the Triangle/CGAL style, instead of letting exactness grow organically inside
  the builder.
- **Topology-signature regression harness** `[test oracle]`. Borrow the
  VoroTop habit of classifying Voronoi cell topology. For fixed fixtures,
  record compact per-cell topology histograms or canonical graph signatures
  (degree distribution, vertex-degree patterns, maybe small-cell graph hashes).
  This catches "valid but combinatorially different" changes during certificate
  and predicate experiments; it is not a runtime feature.

### Strategic / positioning

12. **Many-core scaling study** `[strategic]`. Our structural edge is parallel
    efficiency (~4× vs Fade ~2.4× at 1M on 6 cores). On 32+ cores the gap
    widens and may flip the projected 1M MT verdict back to us. Quantify the
    crossover; it is the strongest honest claim we can make against a
    Delaunay-dual library.

13. **Fortune sweepline — planar-only escape hatch** `[strategic]`. O(n log n),
    beats clipping for planar ST on both asymptotics and constant. But
    inherently sequential (no MT) and does not generalize to the sphere — it
    abandons both reasons our design exists. Recorded as "considered,
    conditional": only if a planar-only single-threaded product becomes a goal.

## One-line summary

ST belongs to Delaunay for structural reasons (it touches only real edges; we
examine-and-reject to prove completeness). We already collected the big
edge-sharing win within bins without sacrificing parallelism. Remaining ST
headroom is small (~1.1–1.25×) and lives in a direction-aware completeness
certificate. Our durable advantage is MT scaling and native sphere support —
positioning, not single-core speed, is where we win.
