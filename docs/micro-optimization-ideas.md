# Micro-Optimization Ideas - Compact Backlog

Status: compacted from the 2026-06-12 43-agent read-only sweep.

The original file was a 1,500+ line raw catalog of 185 unbenchmarked ideas. It
was useful as a search dump, but too noisy for day-to-day planning. The measured
state now lives in:

- `docs/micro-optimization-matrix.md` - benchmark matrix and merge/reject notes
  for the first batch of ideas.
- `docs/optimization-ideas.md` - compact performance ledger for current work.

Use git history before this compaction if the full raw catalog is needed for
archaeology. Promote anything resurrected from that catalog back through the
measurement ledger only after a focused branch and benchmark pass.

## How To Use This File

- Treat this as a reminder list, not evidence.
- Do not retry rejected ideas without a new workload, a stronger mechanism, or
  a measurement flaw in the earlier run.
- Prefer `perf stat` counters (`instructions`, `cycles`, cache and branch
  events) plus paired/interleaved timing for sub-percent candidates.
- Keep changes on topic branches and record measured verdicts in
  `docs/optimization-ideas.md`.

## Measured Or Merged From The Sweep

The first sweep produced a stack of small wins plus several neutral/rejected
ideas. See `docs/micro-optimization-matrix.md` for the full campaign.

| idea | current state |
|---|---|
| Inline extraction checks / avoid duplicate debug projection | Merged as the clearest individual win. |
| Shell-frontier scratch reuse | Merged as part of the proven stack. |
| Packed frontier no-sentinel-fill | Merged as part of the proven stack. |
| Point-to-face reciprocal | Merged as part of the proven stack. |
| Packed tail invariant hoist | Merged as part of the proven stack. |
| Projection `max_r2` threshold compare | Merged on mechanistic prior after neutral timing. |
| Clip-batch slice-to-`batch.n` | Merged on mechanistic prior after neutral timing. |
| Directed-cell-mode short-circuit | Merged on mechanistic prior after neutral timing. |
| Packed query dot cache | Merged on mechanistic prior after neutral timing. |
| Fixed-array SIMD loaders / signed-distance refs | Merged on mechanistic prior after neutral timing. |
| Preprocess touched-reps | Merged as strictly less work on weld-heavy inputs. |
| Frontier buffer reuse + `OrdF32` key folding | Merged; also fixed the latent Eq/Ord disagreement around `-0.0`/NaN. |
| Fused `slot_gen_map` scatter | Done later as `4ea01d9`. |
| Binning cache/fuse, `cell_to_face` u32, periodic conditional wrap | Rejected or skipped; see matrix and ledger. |
| Thin LTO + one codegen unit | Tried and reverted; revisit only with a quiet-box multi-regime sweep and compile-time budget. |

## Remaining Sweep Ideas With Some Mechanical Plausibility

These are not recommendations. They are the small subset still worth remembering
because the mechanism is concrete and the prior is not obviously bad.

| idea | why it might matter | caution |
|---|---|---|
| SIMDize the packed center-pass scalar remainder | Replaces a small scalar dot-product tail with the existing padded SIMD pattern. | 50-80 line kernel rewrite; must earn its complexity with counters. |
| Narrow `PolyBuffer` plane indices from `usize` to `u32` | Could reduce partial-buffer footprint and cache pressure. | Invasive: touches clippers, extract/output, P5 shadow, and sentinels; f64 coordinate arrays dominate footprint. |
| Defer `signed_dists_mask8` distance materialization | All-inside/all-outside clips often need only the mask. | LLVM may already sink enough stores; verify assembly/counters first. |
| Edge-neighbor / dedup structure packing | Repeated small structs may be cache-sensitive in large-N assembly. | Shared-data layout changes are correctness-sensitive; needs targeted profiles. |
| Dense-regime local indexing variants | Directly attacks high-occupancy candidate scans. | Prefer the active dense-cell docs over resurrecting raw micro ideas. |

## Ideas To Leave Parked

| idea | reason |
|---|---|
| u8 `bin_for_cell` scratch | Smaller version of an already-rejected cache-cheap-arithmetic-in-memory idea. |
| Periodic `rem_euclid` purge | Remaining calls are setup/emit, not the hot minimum-image path; conditional wrapping has range risk. |
| Checked-in `target-cpu` config | Bad portability/default-build tradeoff; bench scripts can opt into native. |
| Wider catalog tail | Single-agent, unmeasured, low-prior ideas. Use git history only when looking for a specific mechanism. |

## Recovery Note

The pre-compaction raw sweep was intentionally removed from the working docs.
Recover it with:

```bash
git log -- docs/micro-optimization-ideas.md
git show <old-commit>:docs/micro-optimization-ideas.md
```
