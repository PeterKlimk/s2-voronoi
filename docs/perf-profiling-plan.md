# Performance profiling plan (living doc)

The box is noisy for the foreseeable future, so we **accumulate** perf-relevant
commits and do **one big quiet-box run** later rather than benching each change
as it lands. This doc is the standing plan: the workload matrix to run, and the
**queue** of commits to A/B when we run it. Append to the queue as perf-relevant
commits land; skip the rest.

Companion: `docs/perf-testing-timeline.md` has the detailed per-commit
classification and rationale for the initial (2026-06 strict-plane + occupancy)
batch. This doc is the forward-looking, append-as-we-go version.

## Strategy

- One paired quiet-box run covers the whole queue: `bench_build.sh --chain`
  over the queued commits, then the matrix below via `bench_run.sh --csv`.
- A bigger run later is always an option — nothing here expires. Don't
  half-measure on the noisy box; let the queue grow.
- Decision-grade protocol (dev-environment memory): 2M inputs, `-r 5+`,
  interleaved pairs in BOTH orders, min/median. Treat sub-1% as noise (layout
  luck alone is ±1-2% at 500k ST).

## Workload matrix (what to run)

| axis | values | why |
|------|--------|-----|
| sizes | 500k, 2M (+ 5M if RAM allows) | 2M is the decision-grade size; 500k for cross-check; 5M for scaling |
| dists | `uniform` always; `splittable` + `mega` for any grid/occupancy/index change; `gradient` for density-policy changes | uniform alone is blind to density-contrast regressions (occupancy work found 1.5–9× misses) |
| metric | `total`; sub-phases (`knn_build`, `cell_construction`, `dedup`, `edge_reconcile`, `preprocess`, `assemble`) via `--timing` when localizing | total for verdict; sub-phase to attribute |
| seeds | 1–3 | noise / shape variance |
| threads | ST default; **MT for the 500ms@2.5M release target** | ST is the stable comparison; MT is the headline number |

Run recipe:
```
./scripts/bench_build.sh --chain <N>            # or explicit hashes
./scripts/bench_run.sh -s "500k 2m" -d uniform --seeds "1 2 3" --csv /tmp/uni.csv
./scripts/bench_run.sh -s "500k 1m" -d "uniform splittable mega" --seeds "1 2" --csv /tmp/occ.csv
# localize a regression: add --timing to the build, -m <phase> to the run
# MT target: --multi -s 2.5m -d uniform
```

## Profiling queue (accumulating)

Each row is a commit (or pair) to A/B against its parent, with what to measure.
Status: `pending` until run, then record the verdict inline.

### New this session (uncharacterized) — primary

| commit | change | measure | expectation | status |
|--------|--------|---------|-------------|--------|
| `0def19e` | per-clip eps sqrt skip (both backends) | uniform 500k/2M, sphere+plane | tiny win <1% | pending |
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
| LTO + 1 CGU profile | build with thin-LTO + codegen-units=1, A/B whole-profile (was inconclusive on noisy box; a win would dwarf the micro-opts) | pending |
| MT 500ms @ 2.5M | the release-facing number; `--multi -s 2.5m`; not yet demonstrable (busy-box floor ~1.24s) | pending |
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

## When we run

1. Quiet the box (or accept it's a coarse pass and note it).
2. `bench_build.sh --chain` over the queue (or explicit hashes spanning it).
3. Run the matrix above, CSV per (uniform / clustered) sweep.
4. Record verdicts inline in the queue tables; move confirmed wins/neutrals to a
   "done" note and drop them from pending.
5. For anything borderline, add a diff-disjoint control commit and re-run that
   cell to measure the layout-luck floor.
