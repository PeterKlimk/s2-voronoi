# Local repair oracle — current thinking, questions, uncertainty (2026-06-22)

Working notes on the **dependency-free, local exact defect repair** (the
escalation engine). Companion to `docs/escalation-build-state-2026-06.md` (resume
anchor) and `docs/adaptive-canonical-clip-design-2026-06.md` (design). Authoritative
memory notes: `route-a-splice-diverges`, `fast-clip-is-projected-delaunay`.

Branch: `agent/canonical-predicate-topology`. Engine commit: `0e944dd`.

This doc deliberately records **what we believe, what we measured, what we only
inferred, and what is still open** — so we don't re-collapse distinct facts again
(we did exactly that once; see "The discrepancy" below).

---

## 1. What is built and proven

- **Engine** (`src/knn_clipping/escalate.rs`):
  - `local_delaunay_2d` — exact 2D Delaunay over a local gather via Bowyer–Watson,
    using `robust` exact `incircle`/`orient2d` (already a dep — **no `delaunator`,
    no new crate**).
  - `local_exact_incident` — builds **ONE** triangulation in a **single shared
    stereographic chart** over a 2-ring local gather; reads each closure
    generator's incident fan off it.
  - `repair_local_exact` — closure seeding (unpaired ∪ low-incidence), splice,
    grow-until-clean; behind the caller's whole-diagram **valid-or-revert** gate.
- **Wiring** (`compute.rs`): default (non-probe) path calls `repair_local_exact`.
  The `escalate_probe` build keeps `repair_delaunator` as an A/B oracle behind
  `S2_ESCALATE_DELAUNATOR`. `set_escalation_enabled` exported unconditionally
  (doc-hidden); **OFF by default**.
- **Proven** (`tests/escalate_local.rs`, default build, no probe / no delaunator):
  broad sweep — **25/33 inputs defective, ALL repaired to strictly valid**: mega
  100k s1–20, 300k/500k s1–3, 1m s1; clustered/bimodal already-valid stay valid.
  Convergence **matches the delaunator baseline exactly** (same rounds, same
  splice counts: s1=7, s2=15, s15=9). api 18 / correctness 12 pass; clippy clean.

So: **correctness milestone met** — local + crate-free, parity with the global
oracle scaffolding.

---

## 2. The core idea: the oracle must be PROJECTED, not raw-3D

In exact arithmetic **gnomonic ≡ stereographic ≡ raw-3D** — the same Delaunay
triangulation. They differ only at **near-cocircular 4-tuples**, where rounding
decides the diagonal. The asymmetry that matters for *repair*:

- The **fast clipper is projected** (gnomonic). At a near-cocircular vertex it
  picks the "projected" diagonal.
- A **projected oracle** (delaunator, or our single-chart `incircle`) picks the
  **same** diagonal as fast almost everywhere.
- **Raw `orient3d`** picks the **true-geometry** diagonal, which at a
  near-cocircular tuple can be the *opposite* of the projected one.

Two counts must be kept separate (we previously collapsed them):

| count | meaning | size | who must touch it |
|---|---|---|---|
| **defective cells** | fast is *internally inconsistent* (the unpaired edges) | ~tens | both oracles must fix |
| **diagonal-disagreement cells** | rebuild's diagonal ≠ fast's, *including valid cells* | projected: ≈ defects; raw: every near-cocircular tuple where true≠projected | raw rebuild disagrees with the fast rim on **all** of these |

Consequence: a **raw** rebuild disagrees with the fast rim at many *non-defective*
near-cocircular vertices, so the grow front never reaches a clean boundary and
spreads (observed **closure 5→48, no convergence** on mega s1). A **projected**
rebuild agrees with fast at exactly those self-consistent near-cocircular
vertices, so it only repairs the genuine handful (**5→7, 2 rounds**).

There are really **two independent consistency requirements** for a
non-cascading repair:

1. **Internal consistency** — spliced cells agree with *each other* at shared
   edges. Met by one shared triangulation (a pure-function-of-coords predicate
   evaluated once per geometric question).
2. **Rim consistency** — spliced cells agree with the *unspliced (fast)*
   neighbors at the repaired region's boundary, so the region stays bounded.
   Requires the oracle to match fast wherever fast is self-consistent → a
   **projected** metric.

Delaunator (global) and our local engine both satisfy (1) via one triangulation
and (2) via the projected chart.

---

## 3. THE DISCREPANCY (the open empirical question)

> "I thought only a small % of cells disagreed with the 3D hull?"

This is the honest accounting:

- **uniform**, fast Δ `local_hull` (raw 3D) = **4/12k ≈ 0.03%** — a real
  fast-vs-raw number, but uniform has very few near-cocircular configs.
- **mega**, fast Δ exact = **~tens of cells (0.01–0.02%)** — but this used
  **delaunator (PROJECTED)**, not raw.
- **mega**, fast Δ raw-3D — the only number we ever had was the **9%**, which we
  **discarded as a `local_hull` back-face artifact**.

So: **we never cleanly measured fast-vs-raw-3D on mega.** The claim "raw disagrees
with fast on *many* mega-cluster cells" is currently **inferred from the cascade
behavior**, not directly measured.

What the A/B *does* establish: holding the gather/candidate structure fixed and
flipping only the predicate (raw `in_circle_sphere_sign` → single-chart
`robust::incircle`) flips cascade → convergence. That isolates the **predicate
metric** as the driver of *that* cascade. It does **not by itself** quantify the
raw-vs-fast disagreement fraction in the dense cluster.

The reconciliation we believe (but have not fully verified): the 0.03% is a
uniform-dominated *global average*; inside a dense mega cluster the near-cocircular
density — and hence raw-vs-projected diagonal disagreement — is a much larger
*local* fraction. The cascade is consistent with that.

### Open measurement to settle it

Add a probe mirroring `a0_exact_reference_delaunator` but with a **per-cell raw-3D
reference** diagram, and report on mega specifically:

- `fast Δ raw` (cell-level disagreement count and fraction), vs `fast Δ delaunator`
  (projected) on the *same* input.
- Spatial distribution: is the raw disagreement concentrated in the dense cluster?
- Of the raw-vs-fast disagreements, how many are at **valid** (non-defective)
  cells (i.e. true ≠ projected but fast self-consistent)?

Expected if our model is right: `fast Δ raw` ≫ `fast Δ delaunator` on mega, with
the excess concentrated in the cluster and mostly at valid cells.

If that prediction *fails* (raw and projected disagree with fast by similar
amounts), then the 5→48 cascade was **not** primarily the metric — it would have
been an internal-consistency bug in the per-g raw code (incomplete-gather
over-admission), and the projected version "happened" to converge for another
reason. That alternative is currently **not ruled out by direct measurement**,
only made unlikely by the same-structure A/B.

---

## 4. Other open questions / uncertainty

- **On-by-default / config flag.** The engine is off by default behind a
  doc-hidden toggle. Decision pending: expose as `VoronoiConfig` option? Default
  on (it's gated valid-or-revert, so never worse)? Cost is cold-path only (fires
  only on defective builds), but unmeasured at scale.
- **Scale beyond 1m.** 1m s1 passes; 2m+ untested (the A0/perf notes mention 2m
  OOMs under `perf`, unrelated, but triangulation cost at large defect clusters
  is unprofiled). Bowyer–Watson here is O(n·triangles) on the *gather* (small),
  but the gather is rebuilt per grow round.
- **Gather sizing.** `ESCALATE_GATHER_K = 96`, `ring_k = gather_k.min(32)`,
  2-ring seed. These were tuned to pass, not swept. A pathological defect cluster
  larger than the gather would surface as a residual the gate reverts (safe but
  unrepaired) — untested whether that ever happens.
- **`PAIR_RING` is now dead.** The final code reads fans off the full
  triangulation, so the old per-g `PAIR_RING` candidate cap is gone. (Mentioned
  to avoid confusion with earlier intermediate versions.)
- **Exact-cocircular ties.** `incircle == 0` is treated as "not inside" (triangle
  kept). True 4-cocircular tuples are obscenely rare at f32; the gate catches any
  resulting issue. Not a designed-for case.
- **Single chart at the gather centroid.** Pole = antipode of the gather
  centroid. Fine because defective inputs are clustered (small angular extent). A
  defect cluster spanning a large angular region (≳ a hemisphere) could distort
  the stereographic chart — untested, probably irrelevant for real inputs.
- **Relationship to option A (clip-time exact).** Still optional, only for an
  exact-OUTPUT feature. Note A would want **raw** orient3d (hemisphere-safe, true
  topology) precisely *because* it's producing the canonical diagram, not pairing
  with the fast one — the opposite of the repair's projected requirement.

---

## 5. One-paragraph summary for a future reader

We have a working, dependency-free, local repair: rebuild the defect neighborhood
as one exact 2D Delaunay in a single shared stereographic chart and splice it
back, behind a valid-or-revert gate; it reaches strict validity on every mega
defect we have and matches the global delaunator oracle exactly. The non-obvious
lesson is that the repair oracle must be a **projected** metric (matching the
fast clipper), **not** raw 3D `orient3d` — even though raw is exact and internally
consistent, it picks the *true* near-cocircular diagonal, which disagrees with the
projected fast rim at many *valid* cells in a dense cluster and cascades. The one
thing we assert but have **not directly measured** is the *size* of that
raw-vs-fast disagreement on mega (we only have it for uniform, 0.03%, and a
discarded artifact number for mega); a per-cell raw-3D reference probe would
settle it.
