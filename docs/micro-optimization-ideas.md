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

## Clipper Hot-Path Queue

Status: current working queue after parking the late-clip/proxy/certificate
experiments. The prior from those branches is that extra geometric rejection
state is hard to make cheaper than the clipper. The better target is the clip
kernel itself, especially changed/mixed clips and hot small-N cases.

Timing-only counters on `agent/clipper-hotpath-audit` (`2de4899`) record clip
class by source and polygon length:

- `clip_stream_full`, `clip_stream_radius`, `clip_stream_empty`,
  `clip_stream_mixed`, `clip_stream_too_many`
- `clip_edge_full`, `clip_edge_empty`, `clip_edge_mixed`, `clip_edge_too_many`
- `clip_stream_n3` through `clip_stream_n9p`, plus edge equivalents

Initial 100k single-thread samples showed:

- Empty clips were zero in the sampled fib/uniform/splittable regimes.
- Mixed stream clips were stable across regimes, about 611k-623k, with about
  2.8M output vertices total.
- Dense regimes mainly add unchanged/full/radius clips, not more mixed output.
- Edgecheck clips did not hit the counted convex clipper path in those samples;
  seed behavior should be audited separately before optimizing edgecheck-only
  clip branches.
- Hot N buckets are 3-6 for fib/uniform; splittable also has material 7/8/9+.

### Current Candidates

| idea | why it might matter | caution / next counter |
|---|---|---|
| Fast production `lerp_t` | Intersection is paid only on mixed changed clips, which are stable and real work. A fast path could use `d0 / (d0 - d1)` instead of `is_finite()` plus clamp. | First shadow/count how often transition `t` is non-finite or outside `[0, 1]`, especially under P5/escalation-sensitive cases. Keep guarded lerp for shadow/escalation if needed. |
| N/mask histogram | Needed before more specialization. Exact by-N full/empty/mixed/output counts decide whether N=4/5/6 or late unchanged clips deserve hand tuning. | Already available on `agent/clipper-hotpath-audit`; run the same counters on larger fib and dense cases before behavior changes. |
| Narrow `PolyBuffer` plane metadata to `u32` | Aligns with the memory-throughput thesis: `vertex_planes: [(usize, usize); 64]` and `edge_planes: [usize; 64]` are bulky hot polygon metadata. | Invasive and correctness-sensitive. Needs a branch with focused tests around extraction, sentinels, P5 shadow, and output assembly. |
| Avoid distance materialization on unchanged clips | A mask-only first pass could return early for `mask == full` without keeping distance arrays. Mixed clips would recompute only transition distances. | Risky because mixed clips are common in fib; recompute may lose. Only try after histogram says unchanged dominates the intended regime. |
| Edgecheck-specific assumptions | If seed/edgecheck clips are mostly mixed, an edgecheck-only path could drop full/empty branches. | Current counters saw no edge clipper calls in sampled runs, so first find the actual seed path and instrument it. |

Working ranking:

1. Use the N/mask histogram branch to confirm the hot shape on larger fib and
   dense inputs.
2. Try a fast `lerp_t` variant, guarded by a shadow counter for out-of-range or
   non-finite transition parameters.
3. Try `u32` plane metadata if the goal is memory traffic rather than arithmetic.
4. Leave mask-only unchanged as a maybe; it needs strong unchanged-by-N evidence.
5. Do not spend time on empty-output handling, more cached proxy state, or
   generic N>8 bitmask tuning unless a new profile contradicts the current
   counters.

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
