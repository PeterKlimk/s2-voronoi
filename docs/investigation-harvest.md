# S2 Voronoi: Remaining Investigation Handoff

Trimmed at `7d3c83b` after the deterministic correctness and resource-bound work was harvested.
This is a temporary handoff: delete it when the remaining findings are completed or transferred.

## Test and contract hardening

1. **Document the caller-owned frontier buffer protocol.** Exact frontier caching retains metadata,
   not slot contents. Document that the caller must preserve the buffer between repeated probes, or
   add misuse tests if that contract is intended to be detected.

Packed-group shape assertions are defensive regression checks: cell-major bin construction already
guarantees complete cells with contiguous, layout-matched slots and locals.

## Performance experiments still worth measuring

These are experiments, not predicted wins. Use paired end-to-end counters and correctness
fingerprints. Start with retired instructions and branches, then paired cycles; use Cachegrind,
`llvm-mca`, or `cargo asm` for attribution. Run `bench_run.sh --converge` when counters are
inconclusive.

1. **Defer dense-index construction until the retained grid is known.** Count discarded builds,
   instructions, cycles, allocations, and peak RSS on clustered workloads.
2. **Adaptive/batched ring-tail classification.** Measure tail possible/requested, empty rescans,
   repeated dot evaluations, and unused stored bytes. Avoid unconditional eager tails.
3. **Incremental shell-layer emission.** Measure layer size versus consumed prefix and mid-layer
   closure before designing a conservative next-dot bound.
4. **Bound the first packed chunk and materialize the remainder lazily.** Compare retained keys,
   later requests, recomputation, and peak bytes.
5. **Apply dense-band eligibility before the candidate cap.** Preserve the aggregate work budget
   and shell certificate.
6. **Batch shell takeover across same-cell queries.** Evaluate only as a whole-pipeline traversal
   and emission change.

## Do not broadly retry

- per-(ring cell, query) spherical-cap pruning — adjacent caps rarely prune; measured net loss;
- packed-to-shell attempted-slot filtering — low duplicate coverage and extra branching;
- scalar shell dot-only SIMD — measured 6.5-8.5% slower;
- lower grid target density — density 24 beat 16 by 4.8-7.1%;
- packed partial-selection rewrite — measured 7-14% loss at 2M;
- whole-ring packed bound skip — neutral or worse outside a narrow dense case;
- local optimization of packed radius-2 expansion — no winning regime; removed end-to-end.

## Measurement tooling

- `scripts/bench_perf.sh`: pinned, interleaved hardware-counter comparison.
- `scripts/bench_run.sh --converge`: paired wall-time convergence.
- `VORONOI_MESH_TIMING_KV=1`: includes weld-pair and packed-key storage telemetry.

Hardware branch/cache misses are corroborating signals on the noisy reference host. Behaviorally
identical control builds showed instructions and branches stable to a few parts per million, cycles
at roughly a ±1.2% interval, and substantially noisier hardware miss counters.
