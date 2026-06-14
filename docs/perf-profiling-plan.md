# Performance profiling plan (living doc)

**No longer gated on a quiet box.** `bench_run.sh --converge` makes paired A/B
**decision-grade on a busy box** for any effect above ~the resolution floor
(see Strategy) — so we measure changes *as they land* rather than batching for
a quiet run. The queue below is now just a record of what's been measured / is
pending, not a wait-list.

Companion: `docs/perf-testing-timeline.md` has the detailed per-commit
classification; `docs/multi-regime-perf.md` is the organizing frame.

## Strategy

- **Measure with `--converge`** (validated 2026-06-14 on a busy box, load 2–4):
  paired-interleaved rounds cancel slow drift; the per-round **sign test**
  decides direction (robust to the heavy-tailed sub-round bursts that inflate a
  parametric CI); it auto-stops when settled. A ~2% effect resolved in 57
  rounds at load 2–4; **big effects (10–100×, 13–73%) converge in a handful of
  rounds with tight magnitude too.**
- **Resolution floor**: direction is reliable down to ~1% even busy; *magnitude*
  stays imprecise for small effects on a busy box (CI ±~3%). For a precise
  *magnitude* on a sub-2% micro, a quiet box still helps — but that's the
  low-value tail. Everything in the multi-regime backlog is above the floor.
- First commit listed = baseline; others reported relative to it.

## Workload matrix (what to run)

| axis | values | why |
|------|--------|-----|
| sizes | 500k, 2M (+ 5M if RAM allows) | 2M is the decision-grade size; 500k for cross-check; 5M for scaling |
| dists | `uniform` always; `splittable` + `mega` for any grid/occupancy/index change; `gradient` for density-policy changes | uniform alone is blind to density-contrast regressions (occupancy work found 1.5–9× misses) |
| metric | `total`; sub-phases (`knn_build`, `cell_construction`, `dedup`, `edge_reconcile`, `preprocess`, `assemble`) via `--timing` when localizing | total for verdict; sub-phase to attribute |
| seeds | 1–3 | noise / shape variance |
| threads | ST default; **MT for the 500ms@2.5M release target** | ST is the stable comparison; MT is the headline number |

Run recipe (busy-box-OK via `--converge`; **list baseline first**):
```
./scripts/bench_build.sh <baseline> <candidate>      # or --chain N / explicit hashes
./scripts/bench_run.sh -s "500k 2m" -d uniform --converge --csv /tmp/uni.csv
./scripts/bench_run.sh -s "500k 1m" -d "uniform splittable mega" --converge --csv /tmp/occ.csv
# localize: add --timing to the build, -m <phase> to the run (e.g. -m dedup)
# MT target: --multi -s 2.5m -d uniform --converge
# tune: --resolution 0.01 (band), --max-rounds 160 (cap), --min-rounds 12
```

## Profiling queue (accumulating)

Each row is a commit (or pair) to A/B against its parent, with what to measure.
Status: `pending` until run, then record the verdict inline.

### New this session (uncharacterized) — primary

| commit | change | measure | expectation | status |
|--------|--------|---------|-------------|--------|
| `0def19e` | per-clip eps sqrt skip (both backends) | uniform 500k/2M, sphere+plane | tiny win <1% | **measured (bundled w/ `4ea01d9`)**: 500k uniform busy box, `--converge` → **UNRESOLVED** (best run +1.6% to +3.8%, sign 20–21/57–60 faster). Sub-2% bundle sits at the busy-box resolution floor — direction leans neutral/slightly-slower but magnitude unpinnable here. Quiet box only if we want the exact number. |
| `ff3528b` | plane strict clip rule | plane uniform 500k/2M | neutral (confirm; repair-rate could shift) | pending |
| `33b4962` | occupancy rebuild → Σocc²/n trigger | uniform (regression check) **+ splittable + mega** | uniform neutral; clustered much faster | pending |
| `c3d455b` | plane weld: compact grid in place vs rebuild | **bench_plane** `-m preprocess`, uniform 2M/5M (welds via birthday effect; or a welding dist) | preprocess win on welded runs; total ~neutral | pending — phase-localized, interim noisy signal OK |
| `4ea01d9` | bin assignment: fuse slot_gen_map into scatter (drop a pass) | uniform 2M sphere+plane, `-m dedup` (binning is in the dedup phase) | small win (one O(n) pass + indirection removed) | pending |

### Re-confirm on quiet box (benched only on noisy box)

| commit | change | measure | recorded (noisy) | status |
|--------|--------|---------|------------------|--------|
| `24b8df8` | micro-opt stack | uniform 500k/2M | −36ms@500k, −120ms cc@2M | pending |
| `73fb7f8` | sphere strict tie rule | uniform 500k/2M | perf-neutral | pending |
| `e659655` | stage-0 entry canonicalization | uniform 500k/2M, `-m preprocess` | +18ms@2M (a real cost to pin) | pending |
| `5dbfd5c` / grid-weld | weld redesign | `-m preprocess`, with welds | 378→45ms@2M | pending |

### Deferred experiments (not just A/B — need a build/measure)

| item | what | status |
|------|------|--------|
| expand_r2 on/off across regimes | A/B the runtime toggle via a 2nd build forcing it on | **DONE 2026-06-14**: uniform −1.7% (neutral), splittable +645% (7.45×), mega >9× (single run 86s vs 9.5s). Never a win, catastrophic dense. → flip lib default to off / density-gate. See multi-regime-perf.md item 1. |
| LTO + 1 CGU profile | thin-LTO + codegen-units=1 | **TRIED + REVERTED 2026-06-14** (was `f4c6426`, reverted). Showed uniform 500k −4.2% / 2M −2.7% vs no-LTO, but: (1) magnitude is layout-noise-confounded (the LTO+1CGU layout floor swung an unrelated neutral change ±8pts — see multi-regime-perf.md item 4 / measurement discipline), (2) never measured on dense/sparse regimes, (3) ~6× release build-time cost (8s→50s bench, 1m17s full test build). **Parked as an idea**: revisit only with a proper multi-regime sweep + diff-disjoint control to separate real LTO win from layout. Direction (LTO helps a few %) is likely real; exact figure is not trustworthy. |
| MT 500ms @ 2.5M | the release-facing number; `--multi -s 2.5m` | **MEASURED 2026-06-14**: clean seed ~2.4s MT (busy box), bottleneck `cell_construction` ~1.6s (parallel); knn 0.15s, dedup 0.24s. 500ms target ~5× out via the core compute. **BUT** the bench *default* seed (12345) @ 2.5m is corrupted by the edge_reconcile bug below (12–21s) — always measure with an explicit clean seed at 2.5m. |
| **edge_reconcile O(E) on any defect (HIGH)** | localize scans to affected cells | **FOUND + FIXED 2026-06-14**: at 2.5m, **3 edge_records → 6.7–21s** reconcile (clean seeds: 0 records → 0.004ms). Profiled: the dominant cost was the global `scan_unpaired_interior` (15M-entry SipHash map, ~17s), plus per-round `drop_degenerate_collinear_vertices` global sweep. **Fix**: both localized to `candidate_cells (∪ 1-ring)` with partner-verify, debug-asserted equal to the global scan. → edge_reconcile **~1.2s** (15× faster), output identical, valid; total MT 15s→2.7s. Residual ~1.2s = `collect_merges` `scan_dup_keys` (load-bearing global dup-key backstop, not localizable). See memory `edge-reconcile-global-scan`. |
| micro-opt catalog tail | implement + measure: top-pick-5 u8 bin scratch, fmodf purge, PolyBuffer u32 narrowing | not implemented |

### Future commits — append here

When a perf-relevant commit lands, add a row above (new-this-session table or a
fresh dated section). **Perf-relevant** = touches release hot-path code. **Skip**
(do not queue): docs, tests/scripts, `#[cfg(debug_assertions)]`-only, opt-in
(off-by-default) paths, and codegen-identical refactors (clippy collapses etc.) —
see the skip list in `docs/perf-testing-timeline.md`. Template:

```
| `<hash>` | <what changed> | <sizes/dists/metric> | <expectation> | pending |
```

## How to run (any box)

1. `bench_build.sh <baseline> <candidate>` (or `--chain N`) to stage binaries.
2. `bench_run.sh ... --converge` per regime (uniform / clustered). Sign test
   decides direction; it auto-stops when settled or at `--max-rounds`.
3. Record verdicts inline; move confirmed wins/neutrals to a "done" note and
   drop them from pending.
4. **If a row comes back UNRESOLVED**, it's a sub-resolution-floor effect (the
   busy box can't pin a <~2% magnitude). Options: accept "neutral, too small to
   matter," or save it for a quiet box if the exact number matters. Don't keep
   re-running it busy — that's the one thing `--converge` can't fix.
5. For a borderline win worth confirming, add a diff-disjoint control commit and
   converge that cell too, to measure the fixed layout-luck floor (~1–2%).
