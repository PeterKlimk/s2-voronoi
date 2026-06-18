# Multi-regime performance — don't over-fit the uniform case

## North star

The engine is excellent on **uniform-ish** inputs (flat cube grid + SIMD
packed kNN, 3×3 almost always certifies) and degrades — sometimes **10×+** —
on others, dense/clustered most acutely (real example: hex3 at a 1153:1
density ratio).

The goal is **not** to flatten that into a one-size path. It is:

> **Keep best-in-class performance on the regular (≈uniform) distribution —
> that's the common case and the design target. Accept a *small* regression
> there if it buys 10–100× on another regime.**

A 1% uniform cost to turn a 10× dense cliff into a 1.2× slope is an excellent
trade. A 5% uniform cost to fix a rare regime is probably not. Uniform stays
the priority; other regimes must merely stop being *cliffs*.

**Corollary principle — no unbounded downside.** Every fast-path / optimization
must degrade to ~neutral in its *worst* regime, not catastrophically. An
"optimization" that can 10× *regress* is a liability regardless of its upside.
`expand_r2` is the cautionary example; it was removed after violating this.

## The regimes, and how we currently fare

| regime | examples | current behavior |
|--------|----------|------------------|
| **Uniform / quasi-uniform** | random sphere, fibonacci, Lloyd | **optimal** — flat grid + SIMD packed; 3×3 certifies |
| **Dense / clustered** | hex3 (1153:1), `mega`, `splittable`, `cap` | Punch 2 rebuild + Punch 1 dense center-cell band now handle the worst center-gather cliff; residual cost is certificate depth / shell takeover. |
| **Smooth gradient / sparse** | density-graded, sparse poles | measured mostly OK; grid density holds and shell takeover covers the sparse tail. |
| **Degenerate** | exact-cocircular, lattices, rect walls | correctness handled; plane perf via reconcile (the strict-plane lesson) |

The through-line: nearly every mechanism is tuned for the uniform regime. The
successful dense fixes are gated responses: global re-grid only when density
dominates scan work, and Punch 1 only on rebuilt residual dense cells.

## Work items (framed by the principle)

Each: problem → fix → status → priority → regime. Cross-refs at the bottom.

### 1. `expand_r2`: bound-or-remove — *unbounded downside, likely net-negative*
- **Problem**: the ring-2 packed expansion has a *post-hoc* cap
  (`PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY`) that bounds the candidate
  *list*, not the *scan* — on dense it scans the whole (also-dense) ring-2 band
  O(occ²), overflows, **clears it, and falls to the cursor anyway**. Maximal
  work for nothing (hex3: 8.6s→1.36s just disabling it). And it likely *rarely
  wins*: at the tuned density the 3×3 almost always certifies, so it fires
  mostly on sparse (small upside) and dense (catastrophe); it was
  enabled-by-default 27 min after being added, plausibly unvalidated.
- **Fix**: measure on/off on uniform (expect inert) + a genuinely sparse /
  gradient input (the *only* place it could earn its keep). If no real sparse
  win → **remove / default off** (also shrinks punch-1 to 2 hooks). If yes →
  **pre-gate by ring-2 size** (skip *before* gathering) so the downside is
  bounded ~neutral instead of 10×.
- **MEASURED (2026-06-14, `bench_run --converge`, on/off A/B at 500k)**:
  - uniform: **−1.7%** (CI [−5.3%, +2.0%], 43/69 rounds) → *neutral*, no
    meaningful win in the good regime.
  - splittable: **+645%** (7.45× slower, 0/12 rounds, unanimous).
  - mega: **>9× slower** — a *single* expand-on run hit 86 s of CPU vs the
    baseline's 9.5 s and hadn't finished (run killed; the point was made).
- **MEASURED (2026-06-15, the sparse gate — paired on/off, 500k ST):** the
  hypothesis was that gradient (sparse anti-pole, no dense-cap confound) is the
  one place expand_r2 could finally win. It does *fire* and divert cells off
  the shell takeover (k=2: shell 1.87%→0.09%, r2 catches 1.87%), but it is
  **3–4% SLOWER even here**: gradient k=2 ratio 1.041 (ON-faster 5/24), k=4
  1.029 (5/24). bimodal (cap + sparse bg) ON = 2.6× slower (the cap dominates).
  The scalar ring-2 band scan + per-fire setup costs *more* than letting those
  ~2% of cells fall to the shell. **So expand_r2 has NO winning regime in its
  current form** — uniform neutral, sparse −3-4%, dense catastrophic. The
  item-9 retrial bar is now concrete: a SIMD + resuming version must beat the
  shell takeover on the sparse divert by >4% to justify the stage existing.
  Verdict: **net-negative *as currently implemented*.** Even in the best
  regime it bought ~nothing (−1.7%, within noise); it only fires where it hurts.
- **Why it's slow (root cause, traced 2026-06-14)** — three independent flaws,
  none fundamental:
  1. **Scalar, not SIMD.** The band scan is a per-slot scalar `fp::dot3_f32`
     (`cold.rs:218`) over contiguous SoA (`cell_points_{x,y,z}`) — the *exact*
     shape Chunk0 vectorizes. It's scalar only because it was the "cold" path.
  2. **No reuse on failure.** On cap-overflow it does `keys.clear()`
     (`cold.rs:229`) — discards the whole scan — then the stream falls to the
     shell-cursor, which **re-walks from the home cell** (`shells.rs:4-7,57-59`)
     and re-scans the same neighborhood (consumer dedups). The covered cells +
     security bound are known, so resuming past them is possible but unbuilt.
  3. **Wrong thing bounded.** The cap limits the candidate *list*, not the
     *slots scanned* (`cold.rs:228`) — O(occ²) band scan happens regardless.
- **Decision (two-track):**
  - **Stop-gap (SHIPPED 2026-06-14):** lib default flipped
    `packed_knn_expand_r2 → false` (both `VoronoiConfig` and the internal
    `TerminationConfig`; the directed cursor is the correctness fallback so off
    is correctness-neutral — verified: lib+api+correctness+adversarial all
    green). Reversible via the knob; stops the measured 6.5–9× bleeding.
    **Does not** delete the stage.
  - **Real plan (retrial):** implement expand_r2 *properly* — SIMD band scan +
    clean handoff that **reuses** packed work into a **grouped, SIMD cursor**
    (see item 9) — and *then* re-measure whether expand_r2 justifies itself in
    **any** regime. If a well-built version still never wins, *then* remove it.
    The −1.7% uniform result suggests it may not, but that's a verdict to earn
    under a good implementation, not under the current scalar/no-reuse one.
- **REMOVED (2026-06-15).** Measured net-negative in EVERY sphere regime
  (uniform −1.7%, sparse/gradient −3-4%, dense 6.5-9×) and, on the **plane**,
  perf-NEUTRAL *and dead code* — both plane drivers hardcoded
  `for_point_count(.., false)`, so it never ran in production; forced on it
  validated strictly-valid but never won (uniform/clustered/grid/uniform-periodic
  all within ±1.5%, coin-flip sign tests). With no winning regime on either
  geometry and the shell takeover as the correctness fallback, the stage was
  deleted end-to-end (−1002 lines / 25 files; `cold.rs` gone; sphere diagram
  fingerprint bit-identical, full suite + clippy green). A future retrial
  (item 9: SIMD + resuming + grouped cursor) would re-introduce it from scratch
  with a concrete bar — beat the shell takeover on the sparse divert by >4%.
  **Priority: DONE (removed).** **Regime: dense + sparse (no winner, both
  geometries).**

### 2. `punch-2` — occupancy rebuild (global re-grid for *moderate* density)
- Re-grids the whole sphere finer when concentration is real (`Σocc²/n` over
  threshold). Fires on moderate clustering, neutral on uniform.
- **Limitation**: memory-capped (O(n) cells) → **insufficient for extreme**
  density; can't make cells small enough (hex3's old version *has* this and it
  fires, yet expand_r2 still dominates).
- **Status**: SHIPPED + quiet-box re-tuned (`Σocc²/n > 500`). **Priority: done
  / monitor.** **Regime: moderate dense.**

### 3. `punch-1` — per-dense-cell local index (*implemented dense fix*)
- **Problem**: a dense cell is **O(occ²)** to query regardless of SIMD (a
  constant factor); the rebuild's memory cap leaves residual dense cells.
- **Fix shipped on branch**: an axis-sort side index plus a certificate-safe
  center-cell band in `packed_knn` for rebuilt dense grids. The band is complete
  to its radius and the shell takeover backstops everything beyond it.
- **Synergy with punch-2**: rebuild handles *moderate* (cells reach ~target),
  punch-1 handles the *residual/extreme* the memory cap can't.
- **Measured outcome**: cap 25k went from about 106s to 6.2s; uniform was
  neutral; clustered/outlier regressions disappeared after gating the dense
  index on `grid_rebuilt`.
- **Status**: IMPLEMENTED on the Punch 1 axis-sort branch and present in code,
  still review/merge-gated because the kNN-completeness path is correctness
  critical. **Priority: review/productize.** **Regime: dense / extreme.**

### 4. Directed-cursor batching — recover same-cell correlation
- **Problem**: the cursor fallback is **per-query** — same-cell generators all
  scan the same (dense) cell independently, losing the SIMD + dedup the packed
  group gets for the 3×3. And like expand_r2, the cursor's ring scan is
  **scalar `fp::dot3_f32` over SoA** (`shells.rs:99`) — same vectorizable shape,
  also never SIMD'd.
- **Fix**: batch the cursor by cell (shared dot-matrix) **and** SIMD the ring
  scan. **Does NOT break the quadratic** (punch-1 does).
- **TRIED: SIMD the ring-scan dot alone — perf-NEUTRAL (2026-06-14).** Replaced
  `scan_cell`'s scalar `dot3_f32` with 8-wide `PointChunk8` (bit-identical dots,
  output unchanged, suites green). First A/B under the thin-LTO build read
  uniform **+6.5%** / splittable **+8.5% SLOWER** — but re-running the *same*
  change under **fat** LTO read **−0.9% / +0.5% (neutral)**. So the apparent
  thin regression was **layout luck, not the change** (a sharp demonstration of
  the LTO+1CGU layout-noise floor — see Measurement discipline). The honest
  verdict: the dot-SIMD is **perf-neutral**, which fits the mechanism —
  `scan_cell`'s cost is the *scalar candidate emit* (filter + push to `pending`),
  not the dot, so vectorizing only the dot neither helps nor hurts. Not shipped:
  neutral isn't worth the added complexity. The real dense lever is the
  **emit/heap pipeline or punch-1** (don't scan the whole ring), not the dot.
  The batched-cursor idea (shared work across a cell's queries) is untried.
- **Status**: dot-SIMD tried (neutral, not shipped); cell-batching still open.
  **Priority: LOW** (dot is not the bottleneck; batching partly subsumed by
  punch-1). **Regime: dense.**

### 5. Packed chunk-certificate — stop escalating wrongly on dense
- **Problem**: in a dense cell the chunk-based certificate can't close locally
  (loose bound with many near-equidistant points), so the query escalates to
  expand_r2 / the cursor **even though its neighbors are all in the home cell**.
- **Fix**: a tighter *local* certificate that closes within ring-0 when the
  k-th candidate is provably nearest — so dense queries never escalate.
- **Status**: still identified. Punch 1 removed the worst center-gather cliff,
  but cap-like inputs can still be dominated by genuine certificate depth /
  shell takeover. **Priority: MEDIUM-LOW.** **Regime: dense.**

### 6. Regime-aware dispatch (the meta-pattern)
- The engine should **detect the regime cheaply** (`Σocc²/n`; per-cell occ from
  the live `end - start`) and route: flat-SIMD for uniform, local index for
  dense, etc. The occupancy trigger is one instance — generalize the idea
  (e.g., it's also how a density-gate on expand_r2 or a batched-cursor decision
  would be made). **Status: framing** (informs 1/3/4). **Priority: framing.**

### 7. Sparse / gradient regime — ~~*measure it*~~ MEASURED (2026-06-15)
- We had focused on dense; the sparse end (gradient poles, low local density)
  was under-tested. **Done.** Regime map (500k, ST, expand_r2 OFF):

  | dist | nbrs/cell | shell% | tail% | grid_max_occ | rebuilt |
  |---|---|---|---|---|---|
  | fib | 8.28 | 0.00 | 2.14 | 41 | 0 |
  | uniform | 9.83 | 0.11 | 6.31 | 52 | 0 |
  | gradient k=2 | 9.96 | 1.87 | 6.02 | 140 | 0 |
  | gradient k=4 | 9.97 | 1.70 | 6.00 | 250 | 0 |
  | gradient k=8 | 12.94 | 0.87 | 5.31 | 441 | 0 |
  | bimodal | 12.56 | 19.32 | 5.20 | 216 | 1 |
  | outlier | 9.88 | 0.11 | 6.32 | 534 | 0 |

  **Findings:** (1) **nothing degrades on sparse** — neighbors-per-cell (the
  termination-depth proxy) stays ~8-13 across *every* regime, so the 24/cell
  grid density holds and the sparse end is handled gracefully (it just falls to
  the shell takeover: bimodal 19%, gradient 1-2%, others ~0). (2) **expand_r2
  loses even on the cleanest sparse regime** (gradient −3-4%; see item 1) — so
  it has no winning regime in its current form, and default-off is confirmed
  correct. **Status: DONE** (gate for item 1 resolved). **Regime: sparse.**

### 8. Alternate dense path (speculative)
- For *majority*-dense input, a dense cap is geometrically ~a planar patch; a
  fundamentally different local method (treat the cap as a local 2D problem)
  might beat re-gridding. Only if Punch 1 plus certificate work prove
  insufficient. **Priority:
  LOW / research.** **Regime: extreme dense.**

### 9. Unified candidate engine — the synthesis (where 1/4/5 converge)
The realization behind items 1, 4, 5: **every candidate scan in the kNN engine
is the same "dot over contiguous SoA" primitive** — Chunk0 vectorizes it; the
old cold paths (`expand_r2` band and shell-cursor ring) were scalar
`fp::dot3_f32`, purely by historical accident. The current design
also keeps **two separate candidate engines** (packed = SoA batch + dot
thresholds; cursor = cell-BFS frontier + visited stamps) bridged by the cheapest
possible glue ("let the consumer dedup" the re-emitted points), which throws away
packed work on every handoff.

The target architecture is **one** engine with four properties, each of which is
an item above:
- **SIMD everywhere** — one 8-wide ring-scan primitive over SoA, used by packed
  prep *and* the cursor (items 1, 4). Constant factor (~÷8); doesn't touch
  asymptotics. **Caveat (measured, item 4):** SIMD-ing the cursor's *dot alone*
  did NOT help — the dot isn't `scan_cell`'s bottleneck, the scalar emit is. So
  this facet only pays off if the *whole* candidate pipeline (emit/heap/merge)
  is vectorized together, not the dot in isolation. Lower-confidence after that
  result; punch-1 (avoid scanning the ring) is the surer dense lever.
- **Reusing / resume-from-bound** — don't `keys.clear()` and re-walk from ring 0
  on failure; carry the kept candidates + security bound + covered-cell set
  forward so the cursor *resumes* instead of restarting (item 1 flaw 2). Removes
  duplicate coverage.
- **Grouped** — batch a cell's queries against the shared ring they all scan
  (item 4). Amortizes per-query overhead and enables SIMD-across-queries.
- **Index-backed** — Punch 1 per dense cell so no center-cell gather goes
  O(occ²) (item 3). This part now exists for the packed center path.

In this design **`expand_r2` is not a special stage** — it's just "the r=2 layer
of a SIMD, grouped, resuming cursor." So the right experiment isn't "is expand_r2
worth it today" (measured: no) but **"once the cursor is SIMD + grouped +
resuming, does a dedicated r=2 batch still beat just letting the unified cursor
roll to ring 2?"** — re-trial expand_r2 across *all* regimes under that
implementation; remove it only if a good version still never wins.
- **Status**: framing updated after Punch 1. Remaining work is less "build the
  dense fix" and more "unify frontier/cursor contracts": resume/reuse handoff,
  grouped cursor if it still matters, and certificate-depth reduction for the
  cap tail. **Priority: MEDIUM.** **Regime: dense (with uniform-neutral as the
  constraint).**

## Canonical benchmarks (so we measure all regimes, not just uniform)

`bench_voronoi --dist` now spans the regimes: `uniform`, `fib`, `gradient`(k),
`outlier`, `splittable`, `mega`(frac), `clustered`, `bimodal`. **hex3 (1153:1)
is the canonical real-world dense workload** — record its shape as a fixture.
Per `docs/perf-profiling-plan.md`, any grid/query/clip change must be A/B'd
across **uniform + dense (splittable/mega) + a sparse/gradient cell**.

## Measurement discipline (the trap that motivates this doc)

Single-regime over-fitting is also a *measurement* trap: uniform-only benching
is **blind** to regime regressions. We shipped a **73% plane regression**
(strict-plane) and a **4–7× mis-tuned occupancy threshold** precisely by not
measuring the right regime. Rules:
- Every grid/query/clip change: A/B across uniform **and** the regime it could
  affect.
- When adding a fast-path, measure its **worst** regime, not just its target —
  enforce "no unbounded downside."
- **You don't need a quiet box.** `bench_run.sh --converge` (paired-interleaved
  rounds + per-round sign test) is decision-grade on a busy box for any effect
  above ~1–2%. Every effect this doc cares about — the regime regressions
  (13–73%) and the dense 10–100× wins — is far above that floor and converges in
  a handful of rounds. The only thing the busy box costs us is the exact
  *magnitude* of sub-2% micros (those come back UNRESOLVED; save them for a quiet
  box only if the number matters). So: measure as changes land, don't batch.
- **LTO + `codegen-units=1` raises the layout-noise floor.** Since we adopted
  thin-LTO/1-CGU, any source edit reshuffles the *whole* binary's layout, so the
  per-binary layout-luck floor for small-change A/Bs is now higher than the old
  ~1–2%. A small-change A/B that shows ±several% in a regime the change can't
  plausibly touch (e.g., a cursor-only edit moving *uniform*) is the tell — it's
  layout, not the change. **Demonstrated 2026-06-14:** the SIMD-cursor change
  (item 4) read **+6.5%/+8.5% slower under thin LTO** but **−0.9%/+0.5%
  (neutral) under fat LTO** — same source change, ~8 points of swing from the
  build profile alone. Isolate with a diff-disjoint control commit, rebuild
  under a second LTO setting (as here), or only trust effects large enough to
  dwarf layout (measure in the regime the change targets, at the extreme end).
- **Don't over-run `--converge` for a yes/no decision.** It runs each regime to
  full sign-test convergence (the 2026-06-14 SIMD-cursor probe took 46/78 rounds
  per regime — far more than needed to conclude "no win"). For "is this a win?"
  a low `--max-rounds` (~20–30) and/or a wider `--resolution` is enough; full
  convergence is only for pinning a precise *magnitude*. Also: once a couple of
  regimes clearly show no win, stop — don't wait for the remaining cells.

## Cross-references
- `docs/dense-cell-subindex-design.md` — punch-1 design summary
- `docs/punch1-center-cell-integration.md` — implemented center-cell band prune
- `docs/optimization-ideas.md` — compact performance ledger, occupancy re-calibration, rejected idea index
- `docs/perf-profiling-plan.md` — the measurement queue (where items 1/4/7 land)
- `docs/perf-testing-timeline.md` — per-commit classification
