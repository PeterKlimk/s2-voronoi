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
(`expand_r2` violates this today — see item 1.)

## The regimes, and how we currently fare

| regime | examples | current behavior |
|--------|----------|------------------|
| **Uniform / quasi-uniform** | random sphere, fibonacci, Lloyd | **optimal** — flat grid + SIMD packed; 3×3 certifies |
| **Dense / clustered** | hex3 (1153:1), `mega`, `splittable` | **O(occ²) per dense cell**; expand_r2 over-gather; per-query cursor; 10×+ losses |
| **Smooth gradient / sparse** | density-graded, sparse poles | mostly OK (one global grid handles it), but grid density + expand_r2 are *uniform-tuned*; **under-measured** |
| **Degenerate** | exact-cocircular, lattices, rect walls | correctness handled; plane perf via reconcile (the strict-plane lesson) |

The through-line: nearly every mechanism is tuned for the uniform regime, and
the *responses* to leaving it (expand_r2 escalation, global re-grid, per-query
cursor) were designed for mild departures and misbehave on extreme ones.

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
- **Status**: identified; measurement pending. **Priority: HIGH** (cheap,
  removes a sharp cliff, independent of everything else). **Regime: dense.**

### 2. `punch-2` — occupancy rebuild (global re-grid for *moderate* density)
- Re-grids the whole sphere finer when concentration is real (`Σocc²/n` over
  threshold). Fires on moderate clustering, neutral on uniform.
- **Limitation**: memory-capped (O(n) cells) → **insufficient for extreme**
  density; can't make cells small enough (hex3's old version *has* this and it
  fires, yet expand_r2 still dominates).
- **Status**: SHIPPED + quiet-box re-tuned (`Σocc²/n > 500`). **Priority: done
  / monitor.** **Regime: moderate dense.**

### 3. `punch-1` — per-dense-cell local index (*the real dense fix*)
- **Problem**: a dense cell is **O(occ²)** to query regardless of SIMD (a
  constant factor); the rebuild's memory cap leaves residual dense cells.
- **Fix**: a side index (axis-sort / kd-tree) per over-dense cell →
  **O(occ log occ)**. Hooks: `shells.rs::scan_cell`, `packed_knn` center-cell
  read, and — if expand_r2 survives — its ring-2 band gather (3 sites). The
  hard part is the producer/consumer integration (lazy best-first stream /
  radius), not the structure.
- **Synergy with punch-2**: rebuild handles *moderate* (cells reach ~target),
  punch-1 handles the *residual/extreme* the memory cap can't.
- **Status**: designed + scaffold (branch `agent/punch1-axissort`, structure
  increment done; integration unbuilt). **Priority: HIGH** (the prize; hex3
  justifies it). **Regime: dense / extreme.**

### 4. Directed-cursor batching — recover same-cell correlation
- **Problem**: the cursor fallback is **per-query** — same-cell generators all
  scan the same (dense) cell independently, losing the SIMD + dedup the packed
  group gets for the 3×3.
- **Fix**: batch the cursor by cell (shared dot-matrix). Recovers an ~8×
  constant; **does NOT break the quadratic** (punch-1 does).
- **Status**: identified. **Priority: MEDIUM** (constant-factor; cheaper than
  punch-1 but smaller; partly subsumed by punch-1). **Regime: dense.**

### 5. Packed chunk-certificate — stop escalating wrongly on dense
- **Problem**: in a dense cell the chunk-based certificate can't close locally
  (loose bound with many near-equidistant points), so the query escalates to
  expand_r2 / the cursor **even though its neighbors are all in the home cell**.
- **Fix**: a tighter *local* certificate that closes within ring-0 when the
  k-th candidate is provably nearest — so dense queries never escalate.
- **Status**: identified (deeper). **Priority: MEDIUM-LOW** (punch-1 may
  subsume; but a better certificate helps even without the index). **Regime:
  dense.**

### 6. Regime-aware dispatch (the meta-pattern)
- The engine should **detect the regime cheaply** (`Σocc²/n`; per-cell occ from
  the live `end - start`) and route: flat-SIMD for uniform, local index for
  dense, etc. The occupancy trigger is one instance — generalize the idea
  (e.g., it's also how a density-gate on expand_r2 or a batched-cursor decision
  would be made). **Status: framing** (informs 1/3/4). **Priority: framing.**

### 7. Sparse / gradient regime — *measure it*
- We've focused on dense; the sparse end (gradient poles, low local density) is
  **under-tested**. Grid density (24/cell) and expand_r2 are uniform-tuned;
  confirm nothing degrades, and it's the deciding test for whether expand_r2
  ever wins (item 1). **Status: gap.** **Priority: MEDIUM** (cheap; gates 1).
  **Regime: sparse.**

### 8. Alternate dense path (speculative)
- For *majority*-dense input, a dense cap is geometrically ~a planar patch; a
  fundamentally different local method (treat the cap as a local 2D problem)
  might beat re-gridding. Only if punch-1 proves insufficient. **Priority:
  LOW / research.** **Regime: extreme dense.**

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

## Cross-references
- `docs/dense-cell-subindex-design.md` — punch-1 detail (structure options, hooks, integration)
- `docs/optimization-ideas.md` — occupancy re-calibration, expand_r2 assessment, sub-index ledger entry
- `docs/perf-profiling-plan.md` — the measurement queue (where items 1/4/7 land)
- `docs/perf-testing-timeline.md` — per-commit classification
