# S2 Voronoi: Remaining Investigation Handoff

Trimmed at `7d3c83b` after the deterministic correctness and resource-bound work was harvested.
This is a temporary handoff: delete it when the remaining findings are completed or transferred.

Packed-group shape assertions are defensive regression checks: cell-major bin construction already
guarantees complete cells with contiguous, layout-matched slots and locals.

## Performance experiments still worth measuring

These are experiments, not predicted wins. Use paired end-to-end counters and correctness
fingerprints. Start with retired instructions and branches, then paired cycles; use Cachegrind,
`llvm-mca`, or `cargo asm` for attribution. Run `bench_run.sh --converge` when counters are
inconclusive.

1. **Apply dense-band eligibility before the candidate cap.** Preserve the aggregate work budget
   and shell certificate.
2. **Batch shell takeover across same-cell queries.** Evaluate only as a whole-pipeline traversal
   and emission change.

## Do not broadly retry

- per-(ring cell, query) spherical-cap pruning — adjacent caps rarely prune; measured net loss;
- packed-to-shell attempted-slot filtering — low duplicate coverage and extra branching;
- scalar shell dot-only SIMD — measured 6.5-8.5% slower;
- lower grid target density — density 24 beat 16 by 4.8-7.1%;
- packed partial-selection rewrite — measured 7-14% loss at 2M;
- whole-ring packed bound skip — neutral or worse outside a narrow dense case;
- local optimization of packed radius-2 expansion — no winning regime; removed end-to-end.
- eager/adaptive local ring-tail batching — only 3.8-9.6% of queries requested tails across 100k
  fib/uniform/clustered/bimodal; productive lazy rescans were about 1.5% of 500k clustered runtime;
  batching useful requests requires a whole-pipeline traversal redesign.
- lazy recomputation of retained high-threshold `chunk0_keys` — despite 86.4% unused keys on 100k
  clustered, 75,653 later requests rebuilt 15.75M keys; measured +28.3% instructions, +65.1%
  branches, and +29.8% cycles. Keep the retained keys.

## Measurement tooling

- `scripts/bench_perf.sh`: pinned, interleaved hardware-counter comparison.
- `scripts/bench_run.sh --converge`: paired wall-time convergence.
- `VORONOI_MESH_TIMING_KV=1`: includes weld-pair and packed-key storage telemetry.

Hardware branch/cache misses are corroborating signals on the noisy reference host. Behaviorally
identical control builds showed instructions and branches stable to a few parts per million, cycles
at roughly a ±1.2% interval, and substantially noisier hardware miss counters.
